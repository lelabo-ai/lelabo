from __future__ import annotations

import importlib
import sys

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def _datasets_api():
    # Keep import order stable in this environment (torch first).
    importlib.import_module("torch")
    return importlib.import_module("lelabo.supervised.datasets")


EXPECTED_BUILTIN_DATASETS = {
    "breast_cancer",
    "cifar10",
    "cifar100",
    "glue",
    "iris",
    "mnist",
}


def test_builtin_dataset_names_include_expected() -> None:
    names = set(_datasets_api().get_dataset_names())
    assert EXPECTED_BUILTIN_DATASETS.issubset(names)


@pytest.mark.parametrize(
    ("dataset_name", "module_name", "factory_name"),
    [
        ("iris", "lelabo.supervised.datasets.tabular.iris", "make_iris_dataset"),
        (
            "breast_cancer",
            "lelabo.supervised.datasets.tabular.breast_cancer",
            "make_breast_cancer_dataset",
        ),
        ("mnist", "lelabo.supervised.datasets.vision.mnist", "make_mnist_dataset"),
        ("cifar10", "lelabo.supervised.datasets.vision.cifar", "make_cifar10_dataset"),
        ("cifar100", "lelabo.supervised.datasets.vision.cifar", "make_cifar100_dataset"),
        ("glue", "lelabo.supervised.datasets.nlp.glue", "make_glue_dataset"),
    ],
)
def test_get_dataset_routes_to_expected_builtin_factory(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    module_name: str,
    factory_name: str,
) -> None:
    datasets_api = _datasets_api()
    # Ensure registry baseline is initialized before patching target factories.
    # Otherwise the first get_dataset() call may refresh/reload discovery and
    # override the monkeypatch.
    _ = datasets_api.get_dataset_names()
    target_module = importlib.import_module(module_name)
    calls: dict[str, object] = {}
    sentinel = object()

    def _fake_factory(**kwargs):
        calls["kwargs"] = dict(kwargs)
        return sentinel

    monkeypatch.setattr(target_module, factory_name, _fake_factory)

    out = datasets_api.get_dataset(
        dataset_name,
        batch_size=8,
        seed=7,
        smoke_marker="ok",
    )

    assert out is sentinel
    assert "kwargs" in calls
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs.get("batch_size") == 8
    assert kwargs.get("seed") == 7
    assert kwargs.get("smoke_marker") == "ok"
