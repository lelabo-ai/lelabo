from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image

from ..base import DataBundle, dataset_to_tensors, make_loader
from ..paths import dataset_dir
from ..registry import register_dataset
from ..splits import split_train_val
from ..transforms import AddRelativeNoise, Flatten


def _build_transform(
    *,
    augment: bool,
    add_noise: bool,
    flatten: bool,
    noise_sigma: float,
    out_size: int,
) -> transforms.Compose:
    tfms: List[Any] = []

    # (we do crop/resize inside the dataset; here only pixel-level transforms)
    if augment:
        tfms += [
            transforms.RandomHorizontalFlip(p=0.5),
        ]

    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    ]

    if add_noise and noise_sigma > 0.0:
        tfms.append(AddRelativeNoise(noise_sigma))

    if flatten:
        tfms.append(Flatten())

    return transforms.Compose(tfms)


def _read_ids(imagesets_main: Path, split: str) -> Optional[List[str]]:
    f = imagesets_main / f"{split}.txt"
    if f.exists():
        return [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return None


def _parse_boxes(xml_path: Path) -> List[Tuple[int, int, int, int]]:
    root = ET.parse(str(xml_path)).getroot()
    boxes: List[Tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", "0")))
        ymax = int(float(bnd.findtext("ymax", "0")))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-9)


class SSDDChips(Dataset):
    """
    SSDD -> binary classification chips:
      y=1 : crop around a ship bbox
      y=0 : random background crop with low IoU vs any ship bbox

    Works with your existing classification pipeline (cnn/bp).
    """

    def __init__(
        self,
        voc_root: str | Path,
        *,
        split: str = "train",            # train, test, test_inshore, test_offshore, ...
        out_size: int = 32,
        pad_frac: float = 0.10,
        neg_per_pos: float = 1.0,        # 1.0 => ~1 negative per positive
        max_neg_tries: int = 50,
        transform: Optional[transforms.Compose] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.voc_root = Path(voc_root)
        self.split = split
        self.out_size = int(out_size)
        self.pad_frac = float(pad_frac)
        self.neg_per_pos = float(neg_per_pos)
        self.max_neg_tries = int(max_neg_tries)
        self.transform = transform
        self.seed = int(seed)

        # Folder naming in your tree:
        # JPEGImages_train, Annotations_train, JPEGImages_test, Annotations_test, etc.
        img_dir = self.voc_root / f"JPEGImages_{split}"
        ann_dir = self.voc_root / f"Annotations_{split}"
        if not img_dir.exists():
            # some variants keep base folder (rare); fallback
            img_dir = self.voc_root / "JPEGImages"
        if not ann_dir.exists():
            ann_dir = self.voc_root / "Annotations"

        self.img_dir = img_dir
        self.ann_dir = ann_dir

        imagesets_main = self.voc_root / "ImageSets" / "Main"
        ids = _read_ids(imagesets_main, split)
        if ids is None:
            # fallback: derive from xmls
            ids = [p.stem for p in sorted(self.ann_dir.glob("*.xml"))]

        # Build positives list: (img_path, bbox)
        positives: List[Tuple[Path, Tuple[int, int, int, int]]] = []
        # Also keep per-image boxes to sample negatives later
        self._img_boxes: List[Tuple[Path, List[Tuple[int, int, int, int]]]] = []

        for sid in ids:
            xml_path = self.ann_dir / f"{sid}.xml"
            if not xml_path.exists():
                continue
            boxes = _parse_boxes(xml_path)
            # image extension varies; try common ones
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                p = self.img_dir / f"{sid}{ext}"
                if p.exists():
                    img_path = p
                    break
            if img_path is None:
                hits = list(self.img_dir.glob(f"{sid}.*"))
                if hits:
                    img_path = hits[0]
            if img_path is None:
                continue

            self._img_boxes.append((img_path, boxes))
            for b in boxes:
                positives.append((img_path, b))

        if len(positives) == 0:
            raise RuntimeError(
                f"No positive boxes found. Check paths:\n"
                f"  images: {self.img_dir}\n"
                f"  anns:   {self.ann_dir}\n"
                f"  split:  {split}"
            )

        self.positives = positives

        # Pre-generate a list of negatives as (img_path, bbox)
        # deterministic RNG:
        rng = random.Random(self.seed)
        n_neg = int(round(len(self.positives) * self.neg_per_pos))
        negatives: List[Tuple[Path, Tuple[int, int, int, int]]] = []

        # A helper to sample a negative crop from one image
        def sample_neg(img_w: int, img_h: int, boxes: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
            # choose a crop size roughly like average bbox size for that image (fallback fixed)
            if boxes:
                bw = max(8, int(sum((b[2]-b[0]) for b in boxes) / len(boxes)))
                bh = max(8, int(sum((b[3]-b[1]) for b in boxes) / len(boxes)))
            else:
                bw = bh = max(16, self.out_size)

            # allow some variability
            bw = int(bw * rng.uniform(0.8, 1.4))
            bh = int(bh * rng.uniform(0.8, 1.4))
            bw = max(8, min(bw, img_w))
            bh = max(8, min(bh, img_h))

            for _ in range(self.max_neg_tries):
                x1 = rng.randint(0, max(0, img_w - bw))
                y1 = rng.randint(0, max(0, img_h - bh))
                cand = (x1, y1, x1 + bw, y1 + bh)
                if all(_iou(cand, gt) < 0.10 for gt in boxes):
                    return cand
            return None

        # fill negatives
        img_idx = 0
        while len(negatives) < n_neg:
            img_path, boxes = self._img_boxes[img_idx % len(self._img_boxes)]
            img_idx += 1
            with Image.open(img_path) as im:
                w, h = im.size
            neg = sample_neg(w, h, boxes)
            if neg is not None:
                negatives.append((img_path, neg))

        self.negatives = negatives

    def __len__(self) -> int:
        return len(self.positives) + len(self.negatives)

    def _crop_resize(self, img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        w, h = img.size
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(round(self.pad_frac * bw))
        pad_y = int(round(self.pad_frac * bh))

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = img.crop((cx1, cy1, cx2, cy2))
        crop = crop.resize((self.out_size, self.out_size), resample=Image.BILINEAR)
        return crop

    def __getitem__(self, idx: int):
        if idx < len(self.positives):
            img_path, box = self.positives[idx]
            y = 1
        else:
            img_path, box = self.negatives[idx - len(self.positives)]
            y = 0

        img = Image.open(img_path).convert("RGB")
        chip = self._crop_resize(img, box)

        if self.transform is not None:
            x = self.transform(chip)
        else:
            x = transforms.ToTensor()(chip)

        return x, torch.tensor(y, dtype=torch.long)


def _make_ssdd_dataset(
    *,
    batch_size: int = 128,
    seed: int = 42,
    val_frac: float = 0.1,
    flatten: bool = False,
    input_noise_dataset: float = 0.0,
    noise_on_test: bool = False,
    augment: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    # SSDD-specific knobs
    split_train: str = "train",
    split_test: str = "test",
    out_size: int = 32,
    pad_frac: float = 0.10,
    neg_per_pos: float = 1.0,
    ssdd_variant: str = "PSeg_SSDD",  # BBox_SSDD, BBox_RBox_PSeg_SSDD, PSeg_SSDD, RBox_SSDD
    **_: object,
) -> DataBundle:
    # points to: .../.cache/data/ssdd/Official-SSDD-OPEN/BBox_SSDD/voc_style
    base = Path(dataset_dir("ssdd")) / "Official-SSDD-OPEN" / ssdd_variant / "voc_style"

    train_tfm = _build_transform(
        augment=augment,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        out_size=out_size,
    )
    val_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        out_size=out_size,
    )
    test_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0 and noise_on_test),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        out_size=out_size,
    )

    train_full = SSDDChips(
        base, split=split_train, out_size=out_size, pad_frac=pad_frac,
        neg_per_pos=neg_per_pos, transform=train_tfm, seed=seed
    )
    val_full = SSDDChips(
        base, split=split_train, out_size=out_size, pad_frac=pad_frac,
        neg_per_pos=neg_per_pos, transform=val_tfm, seed=seed
    )
    test_ds = SSDDChips(
        base, split=split_test, out_size=out_size, pad_frac=pad_frac,
        neg_per_pos=neg_per_pos, transform=test_tfm, seed=seed
    )

    n = len(train_full)
    tr_idx, va_idx = split_train_val(n, val_frac, seed)
    train_ds = Subset(train_full, tr_idx.tolist())
    val_ds = Subset(val_full, va_idx.tolist()) if va_idx.numel() > 0 else None

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    x_test, y_test = dataset_to_tensors(test_ds)

    if flatten:
        in_dim = int(x_test.shape[1])
        input_shape = None
    else:
        in_dim = None
        input_shape = (3, out_size, out_size)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=2,  # background vs ship
        in_dim=in_dim,
        input_shape=input_shape,
        x_test=x_test,
        y_test=y_test,
    )


@register_dataset("ssdd")
def make_ssdd_dataset(**kwargs) -> DataBundle:
    return _make_ssdd_dataset(**kwargs)
