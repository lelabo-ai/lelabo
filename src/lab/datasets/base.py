from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset


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


def dataset_to_tensors(ds: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs, dim=0), torch.tensor(ys)


def make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    **loader_kwargs: Any,
) -> DataLoader:
    g = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
