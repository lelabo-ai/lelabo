from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def _datasets_api():
    # Keep import order stable in this environment (torch first).
    importlib.import_module("torch")
    return importlib.import_module("lelabo.supervised.datasets")


def _require_heavy_datasets_enabled() -> None:
    if os.getenv("LELABO_DATASET_HEAVY", "0") != "1":
        pytest.skip("Set LELABO_DATASET_HEAVY=1 to run heavy real-dataset runtime tests.")


@pytest.mark.heavy
@pytest.mark.parametrize(
    ("dataset_name", "kwargs", "num_classes", "input_shape"),
    [
        (
            "mnist",
            {
                "batch_size": 32,
                "seed": 123,
                "val_frac": 0.1,
                "flatten": False,
                "num_workers": 0,
                "pin_memory": False,
                "train_max": 256,
                "val_max": 64,
                "test_max": 128,
            },
            10,
            (1, 28, 28),
        ),
        (
            "cifar10",
            {
                "batch_size": 32,
                "seed": 123,
                "val_frac": 0.1,
                "flatten": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            10,
            (3, 32, 32),
        ),
        (
            "cifar100",
            {
                "batch_size": 32,
                "seed": 123,
                "val_frac": 0.1,
                "flatten": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            100,
            (3, 32, 32),
        ),
    ],
)
def test_builtin_vision_datasets_runtime_heavy(
    dataset_name: str,
    kwargs: dict[str, object],
    num_classes: int,
    input_shape: tuple[int, ...],
) -> None:
    _require_heavy_datasets_enabled()
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    bundle = _datasets_api().get_dataset(dataset_name, **kwargs)

    assert bundle.train_loader is not None
    assert bundle.val_loader is not None
    assert bundle.test_loader is not None
    assert bundle.num_classes == int(num_classes)
    assert bundle.input_shape == tuple(input_shape)

    x, y = next(iter(bundle.train_loader))
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert x.ndim == 4
    assert tuple(x.shape[1:]) == tuple(input_shape)
    assert y.ndim == 1
    assert y.dtype == torch.long


@pytest.mark.heavy
@pytest.mark.filterwarnings(
    "ignore:'maxsplit' is passed as positional argument:DeprecationWarning:fsspec.utils"
)
@pytest.mark.filterwarnings(
    "ignore:co_lnotab is deprecated, use co_lines instead\\.:DeprecationWarning:datasets.utils._dill"
)
def test_glue_dataset_runtime_heavy() -> None:
    _require_heavy_datasets_enabled()
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets")
    pytest.importorskip("transformers")

    # Keep defaults explicit and deterministic for local reproducible heavy runs.
    try:
        bundle = _datasets_api().get_dataset(
            "glue",
            glue_task="sst2",
            hf_model="bert-base-uncased",
            batch_size=8,
            max_length=64,
            seed=123,
            num_workers=0,
            pin_memory=False,
        )
    except Exception as exc:
        msg = str(exc).lower()
        env_bound_markers = (
            "permission denied",
            "couldn't be found on the hugging face hub",
            "connection error",
            "timed out",
            "temporary failure in name resolution",
            "ssl",
        )
        if any(marker in msg for marker in env_bound_markers):
            pytest.skip(f"GLUE heavy test skipped due environment/cache/network constraints: {exc}")
        raise

    assert bundle.train_loader is not None
    assert bundle.val_loader is None
    assert bundle.test_loader is None
    assert bundle.num_classes is not None
    assert "val_loaders" in bundle.meta
    assert "validation" in bundle.meta["val_loaders"]

    train_batch = next(iter(bundle.train_loader))
    assert isinstance(train_batch, Mapping)
    assert {"input_ids", "attention_mask", "labels"}.issubset(train_batch.keys())
    assert torch.is_tensor(train_batch["labels"])

    val_loader = bundle.meta["val_loaders"]["validation"]
    val_batch = next(iter(val_loader))
    assert isinstance(val_batch, Mapping)
    assert {"input_ids", "attention_mask", "labels"}.issubset(val_batch.keys())
