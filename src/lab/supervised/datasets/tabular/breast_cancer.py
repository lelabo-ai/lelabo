from __future__ import annotations

import torch

from ..base import DataBundle, make_loader
from ..splits import split_train_val
from ..transforms import apply_relative_noise


def make_breast_cancer_dataset(
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
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import TensorDataset

    data = load_breast_cancer()
    X, y = data.data, data.target

    Xtrv, Xte, ytrv, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    if input_noise_dataset > 0.0:
        Xtrv_tensor = torch.tensor(Xtrv, dtype=torch.float32)
        Xtrv_tensor = apply_relative_noise(Xtrv_tensor, input_noise_dataset)
        Xtrv = Xtrv_tensor.numpy()

        if noise_on_test:
            Xte_tensor = torch.tensor(Xte, dtype=torch.float32)
            Xte_tensor = apply_relative_noise(Xte_tensor, input_noise_dataset)
            Xte = Xte_tensor.numpy()

    scaler = StandardScaler()
    Xtrv = scaler.fit_transform(Xtrv)
    Xte = scaler.transform(Xte)

    Xtrv = torch.tensor(Xtrv, dtype=torch.float32)
    ytrv = torch.tensor(ytrv, dtype=torch.long)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.long)

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
        seed_scope="breast_cancer.train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="breast_cancer.val",
        num_workers=num_workers,
        pin_memory=pin_memory,
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        seed_scope="breast_cancer.test",
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
