from __future__ import annotations

import multiprocessing as mp
import random
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lelabo.core.seed import make_dataloader_seeding


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: Optional[DataLoader]
    num_classes: Optional[int] = None
    in_dim: Optional[int] = None
    input_shape: Optional[Tuple[int, ...]] = None
    x_test: Optional[torch.Tensor] = None
    y_test: Optional[torch.Tensor] = None
    test_dataset: Optional[Dataset] = None
    meta: dict[str, Any] = field(default_factory=dict)


class TensorDatasetWithTransform(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        x = self.x[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[idx]


def dataset_to_tensors(ds: Dataset, max_items: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    limit = len(ds)
    if max_items is not None:
        if int(max_items) <= 0:
            raise ValueError(f"max_items must be positive when provided, got {max_items}")
        limit = min(limit, int(max_items))

    xs = []
    ys = []
    for i in range(limit):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys)


@lru_cache(maxsize=1)
def _multiprocess_workers_supported() -> bool:
    try:
        ctx = mp.get_context()
        lock = ctx.Lock()
        try:
            acquired = bool(lock.acquire(False))
            if acquired:
                lock.release()
        except Exception:
            return False
        return True
    except Exception:
        return False


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    seed_scope: str = "default",
    num_workers: int = 0,
    pin_memory: bool = True,
    **loader_kwargs: Any,
) -> DataLoader:
    requested_workers = max(0, int(num_workers))
    resolved_workers = requested_workers
    if requested_workers > 0 and not _multiprocess_workers_supported():
        warnings.warn(
            "Multiprocessing DataLoader workers are unavailable in this environment; "
            "falling back to num_workers=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_workers = 0

    g, worker_init_fn, loader_seed = make_dataloader_seeding(seed, scope=seed_scope)
    g = loader_kwargs.pop("generator", g)
    worker_init_fn = loader_kwargs.pop("worker_init_fn", worker_init_fn)
    if resolved_workers == 0:
        random.seed(loader_seed)
        np.random.seed(loader_seed % (2**32))
        torch.manual_seed(loader_seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=resolved_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        **loader_kwargs,
    )
