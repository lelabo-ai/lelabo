from __future__ import annotations

import importlib
import sys
import types

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def _datasets_api():
    # Keep import order stable in this environment (torch first).
    importlib.import_module("torch")
    return importlib.import_module("lelabo.supervised.datasets")


def _first_supervised_batch(loader):
    batch = next(iter(loader))
    assert isinstance(batch, (tuple, list))
    assert len(batch) == 2
    return batch[0], batch[1]


@pytest.mark.parametrize("dataset_name", ["iris", "breast_cancer"])
def test_tabular_builtins_return_valid_bundle(dataset_name: str) -> None:
    torch = importlib.import_module("torch")
    pytest.importorskip("sklearn")

    bundle = _datasets_api().get_dataset(
        dataset_name,
        batch_size=16,
        seed=123,
        val_frac=0.2,
        num_workers=0,
        pin_memory=False,
    )

    assert bundle.train_loader is not None
    assert bundle.val_loader is not None
    assert bundle.test_loader is not None
    assert bundle.num_classes is not None and int(bundle.num_classes) >= 2
    assert bundle.in_dim is not None and int(bundle.in_dim) > 0
    assert bundle.input_shape is None

    x, y = _first_supervised_batch(bundle.train_loader)
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] <= 16


@pytest.mark.parametrize("dataset_name", ["iris", "breast_cancer"])
def test_tabular_builtins_are_reproducible_for_same_seed(dataset_name: str) -> None:
    torch = importlib.import_module("torch")
    pytest.importorskip("sklearn")
    datasets_api = _datasets_api()

    common_kwargs = {
        "batch_size": 16,
        "val_frac": 0.2,
        "num_workers": 0,
        "pin_memory": False,
    }

    a = datasets_api.get_dataset(dataset_name, seed=77, **common_kwargs)
    b = datasets_api.get_dataset(dataset_name, seed=77, **common_kwargs)
    c = datasets_api.get_dataset(dataset_name, seed=78, **common_kwargs)

    xa, ya = _first_supervised_batch(a.train_loader)
    xb, yb = _first_supervised_batch(b.train_loader)
    xc, yc = _first_supervised_batch(c.train_loader)

    assert torch.allclose(xa, xb)
    assert torch.equal(ya, yb)
    assert not (torch.allclose(xa, xc) and torch.equal(ya, yc))


def test_get_dataset_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        _datasets_api().get_dataset("does_not_exist")


def test_mnist_builtin_factory_contract_with_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")
    mnist_mod = importlib.import_module("lelabo.supervised.datasets.vision.mnist")

    class _FakeMNIST:
        def __init__(self, root, train, download, transform):
            self.transform = transform
            n = 48 if bool(train) else 24
            g = torch.Generator().manual_seed(100 if bool(train) else 200)
            self.x = torch.randn(n, 1, 28, 28, generator=g)
            self.y = torch.randint(0, 10, (n,), generator=g)

        def __len__(self):
            return int(self.y.size(0))

        def __getitem__(self, idx: int):
            x = self.x[int(idx)]
            if callable(self.transform):
                x = self.transform(x)
            return x, int(self.y[int(idx)].item())

    # Avoid importing real torchvision transforms in test.
    monkeypatch.setattr(
        mnist_mod,
        "_build_transform",
        lambda **kwargs: (lambda x: x.reshape(-1) if bool(kwargs.get("flatten")) else x),
    )

    fake_tv = types.ModuleType("torchvision")
    fake_tv_datasets = types.ModuleType("torchvision.datasets")
    setattr(fake_tv_datasets, "MNIST", _FakeMNIST)
    setattr(fake_tv, "datasets", fake_tv_datasets)
    monkeypatch.setitem(sys.modules, "torchvision", fake_tv)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", fake_tv_datasets)

    bundle = _datasets_api().get_dataset(
        "mnist",
        batch_size=8,
        seed=5,
        val_frac=0.2,
        num_workers=0,
        pin_memory=False,
        train_max=20,
        val_max=10,
        test_max=10,
    )

    assert bundle.num_classes == 10
    assert bundle.in_dim is None
    assert bundle.input_shape == (1, 28, 28)
    assert bundle.train_loader is not None
    assert bundle.val_loader is not None
    assert bundle.test_loader is not None

    x, y = _first_supervised_batch(bundle.train_loader)
    assert x.ndim == 4
    assert tuple(x.shape[1:]) == (1, 28, 28)
    assert y.ndim == 1


@pytest.mark.parametrize(
    ("dataset_name", "num_classes"),
    [
        ("cifar10", 10),
        ("cifar100", 100),
    ],
)
def test_cifar_builtin_factories_contract_with_fake_torchvision(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    num_classes: int,
) -> None:
    torch = importlib.import_module("torch")
    cifar_mod = importlib.import_module("lelabo.supervised.datasets.vision.cifar")

    def _make_fake_cifar(n_classes: int):
        class _FakeCIFAR:
            def __init__(self, root, train, download, transform):
                self.transform = transform
                n = 60 if bool(train) else 24
                g = torch.Generator().manual_seed(300 + int(n_classes) + (0 if bool(train) else 1))
                self.x = torch.randn(n, 3, 32, 32, generator=g)
                self.y = torch.randint(0, int(n_classes), (n,), generator=g)

            def __len__(self):
                return int(self.y.size(0))

            def __getitem__(self, idx: int):
                x = self.x[int(idx)]
                if callable(self.transform):
                    x = self.transform(x)
                return x, int(self.y[int(idx)].item())

        return _FakeCIFAR

    monkeypatch.setattr(
        cifar_mod,
        "_build_transform",
        lambda **kwargs: (lambda x: x.reshape(-1) if bool(kwargs.get("flatten")) else x),
    )

    fake_tv = types.ModuleType("torchvision")
    fake_tv_datasets = types.ModuleType("torchvision.datasets")
    setattr(fake_tv_datasets, "CIFAR10", _make_fake_cifar(10))
    setattr(fake_tv_datasets, "CIFAR100", _make_fake_cifar(100))
    setattr(fake_tv, "datasets", fake_tv_datasets)
    monkeypatch.setitem(sys.modules, "torchvision", fake_tv)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", fake_tv_datasets)

    bundle = _datasets_api().get_dataset(
        dataset_name,
        batch_size=8,
        seed=9,
        val_frac=0.2,
        flatten=False,
        augment=True,
        num_workers=0,
        pin_memory=False,
    )

    assert bundle.num_classes == int(num_classes)
    assert bundle.in_dim is None
    assert bundle.input_shape == (3, 32, 32)
    assert bundle.train_loader is not None
    assert bundle.val_loader is not None
    assert bundle.test_loader is not None

    x, y = _first_supervised_batch(bundle.train_loader)
    assert x.ndim == 4
    assert tuple(x.shape[1:]) == (3, 32, 32)
    assert y.ndim == 1
    assert int(y.max().item()) < int(num_classes)


def test_glue_builtin_factory_contract_with_fake_hf_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")

    class _LabelFeature:
        def __init__(self, num_classes: int):
            self.num_classes = int(num_classes)

    class _FakeSplit:
        def __init__(self, rows, *, num_classes: int = 2):
            self.rows = [dict(r) for r in rows]
            self.features = {"label": _LabelFeature(num_classes)}

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx: int):
            return dict(self.rows[int(idx)])

        def map(self, fn, batched: bool = False):
            if batched:
                keys = set().union(*[row.keys() for row in self.rows]) if self.rows else set()
                batch = {k: [row.get(k) for row in self.rows] for k in keys}
                updates = dict(fn(batch))
                out_rows = []
                for i, row in enumerate(self.rows):
                    merged = dict(row)
                    for k, values in updates.items():
                        if isinstance(values, list) and len(values) == len(self.rows):
                            merged[k] = values[i]
                        else:
                            merged[k] = values
                    out_rows.append(merged)
                out = _FakeSplit(out_rows)
                out.features = self.features
                return out

            out_rows = []
            for row in self.rows:
                merged = dict(row)
                updates = fn(dict(row))
                if isinstance(updates, dict):
                    merged.update(updates)
                out_rows.append(merged)
            out = _FakeSplit(out_rows)
            out.features = self.features
            return out

        def with_format(self, _fmt: str):
            return self

    class _FakeDatasetDict(dict):
        def map(self, fn, batched: bool = False):
            return _FakeDatasetDict({k: v.map(fn, batched=batched) for k, v in self.items()})

        def with_format(self, fmt: str):
            return _FakeDatasetDict({k: v.with_format(fmt) for k, v in self.items()})

    def _fake_load_dataset(name: str, task_name: str):
        assert name == "glue"
        assert task_name == "sst2"
        train_rows = [{"sentence": f"train-{i}", "label": i % 2} for i in range(12)]
        val_rows = [{"sentence": f"val-{i}", "label": i % 2} for i in range(6)]
        return _FakeDatasetDict(
            {
                "train": _FakeSplit(train_rows, num_classes=2),
                "validation": _FakeSplit(val_rows, num_classes=2),
            }
        )

    class _FakeTokenizer:
        def __call__(self, s1, s2=None, truncation=True, max_length=128):
            if isinstance(s1, list):
                n = len(s1)
                return {
                    "input_ids": [[1, 2, 3] for _ in range(n)],
                    "attention_mask": [[1, 1, 1] for _ in range(n)],
                }
            return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_model_name: str, use_fast: bool = True):
            return _FakeTokenizer()

    class _FakeDataCollatorWithPadding:
        def __init__(self, tokenizer, return_tensors: str = "pt"):
            self.tokenizer = tokenizer
            self.return_tensors = return_tensors

        def __call__(self, features):
            max_len = max(len(item["input_ids"]) for item in features)
            input_ids = []
            attention_masks = []
            for item in features:
                ids = list(item["input_ids"])
                mask = list(item["attention_mask"])
                pad = max_len - len(ids)
                input_ids.append(ids + [0] * pad)
                attention_masks.append(mask + [0] * pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

    fake_datasets_mod = types.ModuleType("datasets")
    setattr(fake_datasets_mod, "load_dataset", _fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_mod)

    fake_transformers_mod = types.ModuleType("transformers")
    setattr(fake_transformers_mod, "AutoTokenizer", _FakeAutoTokenizer)
    setattr(fake_transformers_mod, "DataCollatorWithPadding", _FakeDataCollatorWithPadding)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers_mod)

    bundle = _datasets_api().get_dataset(
        "glue",
        glue_task="sst2",
        hf_model="fake-model",
        batch_size=4,
        seed=12,
        num_workers=0,
        pin_memory=False,
    )

    assert bundle.train_loader is not None
    assert bundle.val_loader is None
    assert bundle.test_loader is None
    assert bundle.num_classes == 2
    assert "val_loaders" in bundle.meta
    assert "validation" in bundle.meta["val_loaders"]

    train_batch = next(iter(bundle.train_loader))
    assert isinstance(train_batch, dict)
    assert {"input_ids", "attention_mask", "labels"}.issubset(train_batch.keys())
    assert train_batch["labels"].dtype == torch.long

    val_loader = bundle.meta["val_loaders"]["validation"]
    val_batch = next(iter(val_loader))
    assert isinstance(val_batch, dict)
    assert {"input_ids", "attention_mask", "labels"}.issubset(val_batch.keys())
