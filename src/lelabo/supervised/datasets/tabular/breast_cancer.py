from __future__ import annotations

from ..base import DataBundle
from ._common import build_tabular_classification_bundle


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

    data = load_breast_cancer()
    return build_tabular_classification_bundle(
        x=data.data,
        y=data.target,
        dataset_name="breast_cancer",
        batch_size=batch_size,
        seed=seed,
        val_frac=val_frac,
        input_noise_dataset=input_noise_dataset,
        noise_on_test=noise_on_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
