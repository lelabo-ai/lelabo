from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
import torch

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))
seed_api = importlib.import_module("lelabo.core.seed")
splits_api = importlib.import_module("lelabo.supervised.datasets.splits")
tabular_common = importlib.import_module("lelabo.supervised.datasets.tabular._common")


def _build_synthetic_tabular_data() -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(120, dtype=np.float64).reshape(60, 2) + 1.0
    offset = np.concatenate(
        [
            np.full((20, 2), 0.0, dtype=np.float64),
            np.full((20, 2), 500.0, dtype=np.float64),
            np.full((20, 2), 1000.0, dtype=np.float64),
        ],
        axis=0,
    )
    x = base + offset
    y = np.array(([0] * 30) + ([1] * 30), dtype=np.int64)
    return x, y


def _expected_raw_splits(
    *,
    x: np.ndarray,
    y: np.ndarray,
    dataset_name: str,
    seed: int,
    val_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sklearn_model_selection = pytest.importorskip("sklearn.model_selection")
    train_test_split = sklearn_model_selection.train_test_split

    test_split_seed = seed_api.derive_seed(seed, dataset_name, "split", "test")
    val_split_seed = seed_api.derive_seed(seed, dataset_name, "split", "val")

    x_trainval, x_test, y_trainval, _ = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=int(test_split_seed),
        stratify=y,
    )
    train_idx, val_idx = splits_api.split_train_val(len(x_trainval), val_frac, int(val_split_seed))
    x_train = x_trainval[train_idx.cpu().numpy()]
    x_val = x_trainval[val_idx.cpu().numpy()]
    return x_train, x_val, x_test


def _dataset_tensors(bundle):
    train_x, _ = bundle.train_loader.dataset.tensors
    val_x, _ = bundle.val_loader.dataset.tensors
    test_x, _ = bundle.test_loader.dataset.tensors
    return train_x, val_x, test_x


def test_tabular_scaler_fits_on_train_split_only() -> None:
    pytest.importorskip("sklearn")

    x, y = _build_synthetic_tabular_data()
    seed = 17
    val_frac = 0.25
    bundle = tabular_common.build_tabular_classification_bundle(
        x=x,
        y=y,
        dataset_name="synthetic_tabular",
        batch_size=16,
        seed=seed,
        val_frac=val_frac,
        input_noise_dataset=0.0,
        noise_on_test=False,
        num_workers=0,
        pin_memory=False,
    )

    x_train_raw, x_val_raw, x_test_raw = _expected_raw_splits(
        x=x,
        y=y,
        dataset_name="synthetic_tabular",
        seed=seed,
        val_frac=val_frac,
    )

    scaler_mean = np.asarray(bundle.meta["scaler_mean"], dtype=np.float64)
    expected_train_mean = x_train_raw.mean(axis=0)
    leaked_mean = np.concatenate([x_train_raw, x_val_raw, x_test_raw], axis=0).mean(axis=0)

    assert np.allclose(scaler_mean, expected_train_mean)
    assert not np.allclose(scaler_mean, leaked_mean)


def test_tabular_noise_hits_test_only_when_enabled() -> None:
    pytest.importorskip("sklearn")

    x, y = _build_synthetic_tabular_data()
    common = dict(
        x=x,
        y=y,
        dataset_name="synthetic_tabular",
        batch_size=16,
        seed=29,
        val_frac=0.2,
        input_noise_dataset=0.15,
        num_workers=0,
        pin_memory=False,
    )

    no_test_noise = tabular_common.build_tabular_classification_bundle(
        **common,
        noise_on_test=False,
    )
    with_test_noise = tabular_common.build_tabular_classification_bundle(
        **common,
        noise_on_test=True,
    )

    train_a, val_a, test_a = _dataset_tensors(no_test_noise)
    train_b, val_b, test_b = _dataset_tensors(with_test_noise)

    assert torch.allclose(train_a, train_b)
    assert torch.allclose(val_a, val_b)
    assert not torch.allclose(test_a, test_b)


def test_tabular_preprocessing_is_reproducible_with_same_seed() -> None:
    pytest.importorskip("sklearn")

    x, y = _build_synthetic_tabular_data()
    common = dict(
        x=x,
        y=y,
        dataset_name="synthetic_tabular",
        batch_size=16,
        seed=41,
        val_frac=0.2,
        input_noise_dataset=0.2,
        noise_on_test=True,
        num_workers=0,
        pin_memory=False,
    )

    a = tabular_common.build_tabular_classification_bundle(**common)
    b = tabular_common.build_tabular_classification_bundle(**common)

    for left, right in zip(_dataset_tensors(a), _dataset_tensors(b), strict=True):
        assert torch.allclose(left, right)
