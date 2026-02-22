from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from ..base import DataBundle, make_loader
from ..paths import dataset_dir
from ..registry import register_dataset
from ..splits import split_train_val
from ..transforms import AddRelativeNoise, Flatten


# -------------------------
# Utils
# -------------------------
def _find_first(existing: List[Path]) -> Optional[Path]:
    for p in existing:
        if p.exists():
            return p
    return None


def _load_json(p: Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _xywh_to_xyxy(b: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def _clip_box_xyxy(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Tuple[int, int, int, int]:
    x1i = int(max(0, min(w - 1, math.floor(x1))))
    y1i = int(max(0, min(h - 1, math.floor(y1))))
    x2i = int(max(0, min(w, math.ceil(x2))))
    y2i = int(max(0, min(h, math.ceil(y2))))
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return x1i, y1i, x2i, y2i


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
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


# -------------------------
# Transforms
# -------------------------
def _build_transform(
    *,
    augment: bool,
    add_noise: bool,
    flatten: bool,
    noise_sigma: float,
    chip_size: int,
) -> transforms.Compose:
    tfms: List[Any] = []

    # NOTE: augmentations here are chip-level (safe).
    if augment:
        tfms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
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


# -------------------------
# COCO index
# -------------------------
@dataclass(frozen=True)
class _ImageRec:
    file_name: str
    width: int
    height: int


class HRSIDCOCO:
    """Minimal COCO reader for HRSID: images + annotations (bbox)."""

    def __init__(self, ann_path: Path) -> None:
        data = _load_json(ann_path)

        # image_id -> record
        self.images: Dict[int, _ImageRec] = {}
        for im in data.get("images", []):
            self.images[int(im["id"])] = _ImageRec(
                file_name=str(im["file_name"]),
                width=int(im.get("width", 0)),
                height=int(im.get("height", 0)),
            )

        # image_id -> list of xyxy boxes
        self.boxes: Dict[int, List[Tuple[float, float, float, float]]] = {k: [] for k in self.images.keys()}
        for ann in data.get("annotations", []):
            img_id = int(ann["image_id"])
            if img_id not in self.images:
                continue
            bbox = ann.get("bbox", None)
            if bbox is None:
                continue
            x1, y1, x2, y2 = _xywh_to_xyxy([float(v) for v in bbox])
            # COCO sometimes includes tiny boxes; keep but clip later
            self.boxes[img_id].append((x1, y1, x2, y2))

        self.image_ids: List[int] = sorted(self.images.keys())


# -------------------------
# Classification chips dataset (image-level split safe)
# -------------------------
class HRSIDChips(Dataset):
    """
    Turns HRSID detection annotations into a binary classification dataset:
      y=1 ship chip (from GT bbox)
      y=0 background chip (hard negatives near ships by default)

    IMPORTANT: Provide image_ids from a split to avoid leakage.
    """

    def __init__(
        self,
        *,
        images_root: Path,
        coco: HRSIDCOCO,
        image_ids: List[int],
        transform: Optional[transforms.Compose],
        chip_size: int = 64,
        pad_frac: float = 0.20,
        neg_per_pos: float = 1.0,
        neg_mode: str = "hard_near",  # "hard_near" | "random"
        max_neg_tries: int = 80,
        iou_thr: float = 0.10,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.images_root = images_root
        self.coco = coco
        self.image_ids = list(image_ids)
        self.transform = transform
        self.chip_size = int(chip_size)
        self.pad_frac = float(pad_frac)
        self.neg_per_pos = float(neg_per_pos)
        self.neg_mode = str(neg_mode)
        self.max_neg_tries = int(max_neg_tries)
        self.iou_thr = float(iou_thr)
        self.rng = random.Random(int(seed))

        # Build positives (one per bbox)
        self.pos: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for img_id in self.image_ids:
            rec = self.coco.images[img_id]
            w, h = rec.width, rec.height
            # width/height might be missing in some dumps; fallback load image lazily when needed
            for b in self.coco.boxes.get(img_id, []):
                if w <= 0 or h <= 0:
                    # will clip later after opening image
                    self.pos.append((img_id, (int(b[0]), int(b[1]), int(b[2]), int(b[3]))))
                else:
                    x1, y1, x2, y2 = _clip_box_xyxy(b[0], b[1], b[2], b[3], w, h)
                    if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                        self.pos.append((img_id, (x1, y1, x2, y2)))

        if len(self.pos) == 0:
            raise RuntimeError("No positive boxes found in the provided split (check annotations & paths).")

        # Pre-generate negatives (keeps dataset deterministic and stable)
        n_neg = int(round(len(self.pos) * self.neg_per_pos))
        self.neg: List[Tuple[int, Tuple[int, int, int, int]]] = []
        self._generate_negatives(n_neg)

    def __len__(self) -> int:
        return len(self.pos) + len(self.neg)

    def _open_image(self, img_id: int) -> Image.Image:
        fn = self.coco.images[img_id].file_name
        p = self.images_root / fn
        if not p.exists():
            # sometimes file_name already includes "images/..."
            p2 = self.images_root / Path(fn).name
            if p2.exists():
                p = p2
            else:
                raise FileNotFoundError(f"Image not found: {p}")
        return Image.open(p).convert("RGB")

    def _ship_boxes_int(self, img_id: int, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        out: List[Tuple[int, int, int, int]] = []
        for b in self.coco.boxes.get(img_id, []):
            x1, y1, x2, y2 = _clip_box_xyxy(b[0], b[1], b[2], b[3], w, h)
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                out.append((x1, y1, x2, y2))
        return out

    def _chip_from_box(self, img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        w, h = img.size
        x1, y1, x2, y2 = box
        bw, bh = (x2 - x1), (y2 - y1)
        px = int(round(self.pad_frac * bw))
        py = int(round(self.pad_frac * bh))
        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(w, x2 + px)
        cy2 = min(h, y2 + py)
        crop = img.crop((cx1, cy1, cx2, cy2))
        return crop.resize((self.chip_size, self.chip_size), resample=Image.BILINEAR)

    def _sample_neg_box(
        self,
        *,
        w: int,
        h: int,
        ship_boxes: List[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        # Choose a crop size similar-ish to ships in this image; fallback fixed.
        if ship_boxes:
            avg_w = sum(b[2] - b[0] for b in ship_boxes) / len(ship_boxes)
            avg_h = sum(b[3] - b[1] for b in ship_boxes) / len(ship_boxes)
        else:
            avg_w = avg_h = float(self.chip_size)

        cw = int(max(12, min(w, avg_w * self.rng.uniform(0.9, 1.6))))
        ch = int(max(12, min(h, avg_h * self.rng.uniform(0.9, 1.6))))

        def ok(cand: Tuple[int, int, int, int]) -> bool:
            return all(_iou_xyxy(cand, gt) < self.iou_thr for gt in ship_boxes)

        if self.neg_mode == "random" or not ship_boxes:
            for _ in range(self.max_neg_tries):
                x1 = self.rng.randint(0, max(0, w - cw))
                y1 = self.rng.randint(0, max(0, h - ch))
                cand = (x1, y1, x1 + cw, y1 + ch)
                if ok(cand):
                    return cand
            return None

        # hard_near: sample around a random ship bbox neighborhood
        for _ in range(self.max_neg_tries):
            gt = self.rng.choice(ship_boxes)
            gx1, gy1, gx2, gy2 = gt
            gcx = (gx1 + gx2) // 2
            gcy = (gy1 + gy2) // 2

            # sample center near ship (within ~1–2 ship sizes)
            radx = int(max(16, (gx2 - gx1) * self.rng.uniform(1.0, 2.5)))
            rady = int(max(16, (gy2 - gy1) * self.rng.uniform(1.0, 2.5)))
            cx = int(min(w - 1, max(0, gcx + self.rng.randint(-radx, radx))))
            cy = int(min(h - 1, max(0, gcy + self.rng.randint(-rady, rady))))

            x1 = int(min(max(0, cx - cw // 2), max(0, w - cw)))
            y1 = int(min(max(0, cy - ch // 2), max(0, h - ch)))
            cand = (x1, y1, x1 + cw, y1 + ch)
            if ok(cand):
                return cand

        return None

    def _generate_negatives(self, n_neg: int) -> None:
        # Iterate image ids cyclically to spread negatives
        img_cursor = 0
        while len(self.neg) < n_neg:
            img_id = self.image_ids[img_cursor % len(self.image_ids)]
            img_cursor += 1

            img = self._open_image(img_id)
            w, h = img.size
            ship_boxes = self._ship_boxes_int(img_id, w, h)

            nb = self._sample_neg_box(w=w, h=h, ship_boxes=ship_boxes)
            if nb is not None:
                self.neg.append((img_id, nb))

    def __getitem__(self, idx: int):
        if idx < len(self.pos):
            img_id, box = self.pos[idx]
            y = 1
        else:
            img_id, box = self.neg[idx - len(self.pos)]
            y = 0

        img = self._open_image(img_id)
        chip = self._chip_from_box(img, box)

        x = self.transform(chip) if self.transform is not None else transforms.ToTensor()(chip)
        return x, torch.tensor(y, dtype=torch.long)


# -------------------------
# Builder
# -------------------------
def _make_hrsid_dataset(
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
    # HRSID-specific knobs
    chip_size: int = 64,
    pad_frac: float = 0.20,
    neg_per_pos: float = 1.0,
    neg_mode: str = "hard_near",   # "hard_near" is key to avoid trivial 100%
    # Optional domain split using inshore_offshore (if you want cross-domain test)
    domain: str = "all",           # "all" | "inshore" | "offshore"
    test_domain: Optional[str] = None,  # if set, overrides domain for test only
    **_: object,
) -> DataBundle:
    root = Path(dataset_dir("hrsid"))

    images_root = _find_first([
        root / "images",
        root / "Images",
        root / "JPEGImages",
    ])
    ann_root = _find_first([
        root / "annotations",
        root / "Annotations",
        root / "annotation",
    ])
    if images_root is None or ann_root is None:
        raise FileNotFoundError(f"HRSID expected folders under {root}: images/ and annotations/")

    # Pick an annotations file (COCO). We search for common names.
    ann_path = _find_first([
        ann_root / "train.json",
        ann_root / "instances_train.json",
        ann_root / "annotations.json",
        ann_root / "hrsid.json",
        ann_root / "HRSID.json",
    ])
    if ann_path is None:
        # fallback: first .json in annotations
        js = sorted(ann_root.glob("*.json"))
        if not js:
            raise FileNotFoundError(f"No COCO .json found in {ann_root}")
        ann_path = js[0]

    coco = HRSIDCOCO(ann_path)

    # Domain filtering (optional): expects lists in inshore_offshore/
    inout_dir = root / "inshore_offshore"
    inshore_ids: Optional[set[int]] = None
    offshore_ids: Optional[set[int]] = None
    if inout_dir.exists():
        # Try common filenames; if they don't exist, silently ignore.
        in_file = _find_first([inout_dir / "inshore.txt", inout_dir / "inshore_images.txt"])
        off_file = _find_first([inout_dir / "offshore.txt", inout_dir / "offshore_images.txt"])

        def read_ids(p: Optional[Path]) -> Optional[set[int]]:
            if p is None or not p.exists():
                return None
            s: set[int] = set()
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                # try interpret as image_id int; else match by filename
                if ln.isdigit():
                    s.add(int(ln))
                else:
                    # match by file_name suffix
                    for img_id, rec in coco.images.items():
                        if rec.file_name.endswith(ln) or Path(rec.file_name).name == ln:
                            s.add(img_id)
            return s

        inshore_ids = read_ids(in_file)
        offshore_ids = read_ids(off_file)

    def apply_domain(ids: List[int], dom: str) -> List[int]:
        dom = str(dom).lower()
        if dom == "all":
            return ids
        if dom == "inshore" and inshore_ids is not None:
            return [i for i in ids if i in inshore_ids]
        if dom == "offshore" and offshore_ids is not None:
            return [i for i in ids if i in offshore_ids]
        return ids  # fallback if files absent

    all_img_ids = coco.image_ids
    all_img_ids = apply_domain(all_img_ids, domain)

    # Image-level split (prevents leakage)
    n = len(all_img_ids)
    tr_idx, va_idx = split_train_val(n, val_frac, seed)

    train_img_ids = [all_img_ids[i] for i in tr_idx.tolist()]
    val_img_ids = [all_img_ids[i] for i in va_idx.tolist()] if va_idx.numel() > 0 else []

    # Test split: by default, use ALL images of (test_domain or domain) to keep data
    # (If you prefer a held-out test, set val_frac and then later use val as test.)
    td = test_domain if test_domain is not None else domain
    test_img_ids = apply_domain(coco.image_ids, td)

    train_tfm = _build_transform(
        augment=augment,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        chip_size=chip_size,
    )
    val_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        chip_size=chip_size,
    )
    test_tfm = _build_transform(
        augment=False,
        add_noise=(input_noise_dataset > 0.0 and noise_on_test),
        flatten=flatten,
        noise_sigma=input_noise_dataset,
        chip_size=chip_size,
    )

    train_full = HRSIDChips(
        images_root=images_root,
        coco=coco,
        image_ids=train_img_ids,
        transform=train_tfm,
        chip_size=chip_size,
        pad_frac=pad_frac,
        neg_per_pos=neg_per_pos,
        neg_mode=neg_mode,
        seed=seed,
    )
    val_ds = HRSIDChips(
        images_root=images_root,
        coco=coco,
        image_ids=val_img_ids,
        transform=val_tfm,
        chip_size=chip_size,
        pad_frac=pad_frac,
        neg_per_pos=neg_per_pos,
        neg_mode=neg_mode,
        seed=seed,
    ) if len(val_img_ids) > 0 else None

    test_ds = HRSIDChips(
        images_root=images_root,
        coco=coco,
        image_ids=test_img_ids,
        transform=test_tfm,
        chip_size=chip_size,
        pad_frac=pad_frac,
        neg_per_pos=neg_per_pos,
        neg_mode=neg_mode,
        seed=seed + 123,
    )

    train_loader = make_loader(
        train_full,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        seed_scope="hrsid.train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="hrsid.val",
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="hrsid.test",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if flatten:
        sample_x, _sample_y = test_ds[0]
        in_dim: Optional[int] = int(sample_x.numel())
        input_shape: Optional[Tuple[int, ...]] = None
    else:
        in_dim = None
        input_shape = (3, chip_size, chip_size)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=2,
        in_dim=in_dim,
        input_shape=input_shape,
        test_dataset=test_ds,
    )


@register_dataset("hrsid")
def make_hrsid_dataset(**kwargs) -> DataBundle:
    return _make_hrsid_dataset(**kwargs)
