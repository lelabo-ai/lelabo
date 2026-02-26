from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def _models_registry():
    # Keep import order stable in this environment (torch first).
    importlib.import_module("torch")
    return importlib.import_module("lelabo.models.registry")


def _default_args() -> Namespace:
    return Namespace(
        hidden=16,
        layers=2,
        hf_model="bert-base-uncased",
        hf_trust_remote_code=False,
    )


def _install_fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    nn = importlib.import_module("torch.nn")

    class _FakeWeights:
        IMAGENET1K_V1 = object()

    class _FakeBlock(nn.Module):
        def forward(self, x):
            return x

    class _FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.ModuleList([_FakeBlock()])
            self.layer2 = nn.ModuleList([_FakeBlock()])
            self.layer3 = nn.ModuleList([_FakeBlock()])
            self.layer4 = nn.ModuleList([_FakeBlock()])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(3, 1000)

        def forward(self, x):
            for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
                for block in layer:
                    x = block(x)
            x = self.gap(x).flatten(1)
            return self.fc(x)

    def _make_fake_resnet(*, weights=None):
        _ = weights
        return _FakeBackbone()

    tv_mod = types.ModuleType("torchvision")
    tv_models_mod = types.ModuleType("torchvision.models")
    setattr(tv_models_mod, "ResNet18_Weights", _FakeWeights)
    setattr(tv_models_mod, "ResNet34_Weights", _FakeWeights)
    setattr(tv_models_mod, "ResNet50_Weights", _FakeWeights)
    setattr(tv_models_mod, "resnet18", _make_fake_resnet)
    setattr(tv_models_mod, "resnet34", _make_fake_resnet)
    setattr(tv_models_mod, "resnet50", _make_fake_resnet)
    setattr(tv_mod, "models", tv_models_mod)
    monkeypatch.setitem(sys.modules, "torchvision", tv_mod)
    monkeypatch.setitem(sys.modules, "torchvision.models", tv_models_mod)


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")

    class _FakeEncoder(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.layer = nn.ModuleList(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                ]
            )

    class _FakeBase(nn.Module):
        def __init__(self, *, vocab_size: int = 32, hidden_dim: int = 12):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, hidden_dim)
            self.encoder = _FakeEncoder(hidden_dim)

    class _FakeHFClassifier(nn.Module):
        base_model_prefix = "bert"

        def __init__(self, num_labels: int):
            super().__init__()
            self.bert = _FakeBase()
            self.classifier = nn.Linear(12, int(num_labels))
            self.config = SimpleNamespace(num_labels=int(num_labels), vocab_size=32)

        def forward(
            self,
            *,
            input_ids=None,
            attention_mask=None,
            output_hidden_states: bool = False,
            return_dict: bool = True,
            **_kwargs,
        ):
            _ = attention_mask
            if input_ids is None:
                raise ValueError("input_ids is required.")
            h = self.bert.embeddings(input_ids)
            hidden_states = [h]
            for layer in self.bert.encoder.layer:
                h = torch.tanh(layer(h))
                hidden_states.append(h)
            logits = self.classifier(h[:, 0, :])
            out = SimpleNamespace(
                logits=logits,
                hidden_states=tuple(hidden_states) if bool(output_hidden_states) else None,
                last_hidden_state=h,
            )
            if bool(return_dict):
                return out
            return (logits,)

    class _AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, model_name: str, num_labels: int, trust_remote_code: bool = False):
            _ = (cls, model_name, trust_remote_code)
            return _FakeHFClassifier(num_labels=num_labels)

    tf_mod = types.ModuleType("transformers")
    setattr(tf_mod, "AutoModelForSequenceClassification", _AutoModelForSequenceClassification)
    monkeypatch.setitem(sys.modules, "transformers", tf_mod)


def _assert_basic_cache_contract(model, out, cache) -> None:
    assert out is not None
    assert isinstance(cache, dict)
    assert "block_inputs" in cache
    assert "block_outputs" in cache
    assert isinstance(cache["block_inputs"], dict)
    assert isinstance(cache["block_outputs"], dict)
    block_names = [b.name for b in model.get_blocks()]
    assert block_names
    for name in block_names:
        assert name in cache["block_inputs"]
        assert name in cache["block_outputs"]


def test_builtin_model_names_include_expected_defaults() -> None:
    names = set(_models_registry().get_model_names())
    expected = {
        "bert",
        "cnn",
        "deephebb",
        "hf",
        "mlp",
        "resnet18",
    }
    assert expected.issubset(names)


def test_mlp_classifier_cache_contract() -> None:
    torch = importlib.import_module("torch")
    mlp_mod = importlib.import_module("lelabo.models.imported.mlp")
    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")

    x = torch.randn(6, 8)
    out, cache = model(x, return_cache=True)

    assert tuple(out.shape) == (6, 3)
    _assert_basic_cache_contract(model, out, cache)
    assert "steps" in cache
    assert "head" in cache["block_outputs"]
    assert tuple(cache["block_outputs"]["head"].shape) == (6, 3)


def test_convnet_classifier_cache_contract() -> None:
    torch = importlib.import_module("torch")
    conv_mod = importlib.import_module("lelabo.models.imported.convnet")
    model = conv_mod.ConvNetClassifier(
        in_channels=1,
        num_classes=10,
        channels=[8, 16],
        kernel_sizes=3,
        use_bn=False,
        pool_every=1,
    )

    x = torch.randn(4, 1, 28, 28)
    out, cache = model(x, return_cache=True)

    assert tuple(out.shape) == (4, 10)
    _assert_basic_cache_contract(model, out, cache)
    assert "head" in cache["block_outputs"]
    assert tuple(cache["block_outputs"]["head"].shape) == (4, 10)


def test_deep_softhebb_classifier_cache_contract() -> None:
    torch = importlib.import_module("torch")
    deep_mod = importlib.import_module("lelabo.models.imported.deep_softhebb")
    model = deep_mod.DeepSoftHebbClassifier(in_channels=3, num_classes=10)

    x = torch.randn(2, 3, 32, 32)
    out, cache = model(x, return_cache=True)

    assert tuple(out.shape) == (2, 10)
    assert isinstance(cache, dict)
    assert "block_inputs" in cache
    assert "block_outputs" in cache
    for name in ("conv1", "conv2", "conv3", "head"):
        assert name in cache["block_inputs"]
        assert name in cache["block_outputs"]


def test_hf_and_bert_builders_with_fake_transformers_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")
    registry = _models_registry()
    _install_fake_transformers(monkeypatch)

    args = _default_args()
    args.hf_model = "fake/tiny-bert"
    ctx = registry.ModelContext(
        dataset="glue",
        num_classes=2,
        in_dim=None,
        in_channels=None,
        input_shape=None,
    )
    batch = {
        "input_ids": torch.randint(0, 31, (3, 8), dtype=torch.long),
        "attention_mask": torch.ones(3, 8, dtype=torch.long),
        "labels": torch.randint(0, 2, (3,), dtype=torch.long),
    }

    for model_name in ("hf", "bert"):
        model = registry.build_model(model_name, ctx, args)
        out = model(**batch)
        assert hasattr(out, "logits")
        assert tuple(out.logits.shape) == (3, 2)

        out_cached, cache = model(return_cache=True, **batch)
        assert tuple(out_cached.logits.shape) == (3, 2)
        assert "block_inputs" in cache
        assert "head" in cache["block_inputs"]
        assert "embeddings" in cache["block_inputs"]
        assert any(k.startswith("encoder.layer") for k in cache["block_inputs"])


def test_resnet18_builder_with_fake_torchvision_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = importlib.import_module("torch")
    registry = _models_registry()
    _install_fake_torchvision(monkeypatch)

    ctx = registry.ModelContext(
        dataset="cifar10",
        num_classes=5,
        in_dim=None,
        in_channels=3,
        input_shape=(3, 32, 32),
    )
    model = registry.build_model("resnet18", ctx, _default_args())

    x = torch.randn(4, 3, 32, 32)
    out, cache = model(x, return_cache=True)
    assert tuple(out.shape) == (4, 5)
    _assert_basic_cache_contract(model, out, cache)
    assert "head" in cache["block_outputs"]
