from __future__ import annotations

import torch


def split_train_val(n: int, val_frac: float, seed: int):
    """
    Returns train_idx, val_idx as torch.LongTensor.
    Deterministic split using a local generator.
    """
    val_frac = float(val_frac)
    if val_frac <= 0.0:
        return torch.arange(n), torch.empty(0, dtype=torch.long)

    val_size = int(round(n * val_frac))
    val_size = max(1, min(val_size, n - 1))

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    return train_idx, val_idx
