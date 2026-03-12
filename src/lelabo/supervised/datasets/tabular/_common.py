from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....core.seed import derive_seed
from ..base import DataBundle, make_loader
from ..splits import split_train_val


def _apply_relative_noise_np(x: np.ndarray, sigma: float, *, seed: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    level = float(sigma)
    if level <= 0.0:
        return arr.copy()
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal(size=arr.shape)
    return arr + (level * np.abs(arr) * noise)


def build_tabular_classification_bundle(
    *,
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    batch_size: int,
    seed: int,
    val_frac: float,
    input_noise_dataset: float,
    noise_on_test: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataBundle:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset

    name = str(dataset_name).strip().lower()
    x_all = np.asarray(x, dtype=np.float64)
    y_all = np.asarray(y)

    test_split_seed = derive_seed(seed, name, "split", "test")
    val_split_seed = derive_seed(seed, name, "split", "val")

    x_trainval, x_test_raw, y_trainval, y_test_raw = train_test_split(
        x_all,
        y_all,
        test_size=0.2,
        random_state=int(test_split_seed),
        stratify=y_all,
    )

    train_idx, val_idx = split_train_val(len(x_trainval), val_frac, int(val_split_seed))
    train_idx_np = train_idx.cpu().numpy()
    val_idx_np = val_idx.cpu().numpy()

    x_train_raw = x_trainval[train_idx_np]
    y_train_raw = y_trainval[train_idx_np]
    x_val_raw = x_trainval[val_idx_np] if val_idx_np.size > 0 else None
    y_val_raw = y_trainval[val_idx_np] if val_idx_np.size > 0 else None

    noise_sigma = float(input_noise_dataset)
    if noise_sigma > 0.0:
        x_train_raw = _apply_relative_noise_np(
            x_train_raw,
            noise_sigma,
            seed=derive_seed(seed, name, "noise", "train"),
        )
        if x_val_raw is not None:
            x_val_raw = _apply_relative_noise_np(
                x_val_raw,
                noise_sigma,
                seed=derive_seed(seed, name, "noise", "val"),
            )
        if bool(noise_on_test):
            x_test_raw = _apply_relative_noise_np(
                x_test_raw,
                noise_sigma,
                seed=derive_seed(seed, name, "noise", "test"),
            )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw) if x_val_raw is not None else None
    x_test = scaler.transform(x_test_raw)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_raw, dtype=torch.long)
    x_val_t = torch.tensor(x_val, dtype=torch.float32) if x_val is not None else None
    y_val_t = torch.tensor(y_val_raw, dtype=torch.long) if y_val_raw is not None else None
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_raw, dtype=torch.long)

    train_ds = TensorDataset(x_train_t, y_train_t)
    val_ds = TensorDataset(x_val_t, y_val_t) if x_val_t is not None and y_val_t is not None else None
    test_ds = TensorDataset(x_test_t, y_test_t)

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        seed_scope=f"{name}.train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            seed_scope=f"{name}.val",
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if val_ds is not None
        else None
    )
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope=f"{name}.test",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=int(np.max(y_train_raw)) + 1,
        in_dim=int(x_train_t.shape[1]),
        input_shape=None,
        x_test=x_test_t,
        y_test=y_test_t,
        test_dataset=test_ds,
        meta={
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
    )
