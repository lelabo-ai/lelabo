from __future__ import annotations

import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from ..base import DataBundle, make_loader
from ..registry import register_dataset
from ..splits import split_train_val
from ..transforms import apply_relative_noise


@register_dataset("iris")
def make_iris_dataset(
    *,
    batch_size: int = 32,
    seed: int = 42,
    val_frac: float = 0.0,
    input_noise_dataset: float = 0.0,
    noise_on_test: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    **_: object,
) -> DataBundle:
    data = load_iris()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Xtrv, Xte, ytrv, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    Xtrv = torch.tensor(Xtrv, dtype=torch.float32)
    ytrv = torch.tensor(ytrv, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

    if input_noise_dataset > 0.0:
        Xtrv = apply_relative_noise(Xtrv, input_noise_dataset)
        if noise_on_test:
            Xte = apply_relative_noise(Xte, input_noise_dataset)

    n = Xtrv.size(0)
    tr_idx, va_idx = split_train_val(n, val_frac, seed)

    train_ds = TensorDataset(Xtrv[tr_idx], ytrv[tr_idx])
    val_ds = TensorDataset(Xtrv[va_idx], ytrv[va_idx]) if va_idx.numel() > 0 else None
    test_ds = TensorDataset(Xte, yte)

    train_loader = make_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        seed_scope="iris.train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="iris.val",
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="iris.test",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    num_classes = int(ytrv.max().item() + 1)
    in_dim = int(Xtrv.shape[1])

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        in_dim=in_dim,
        input_shape=None,
        x_test=Xte,
        y_test=yte,
    )
