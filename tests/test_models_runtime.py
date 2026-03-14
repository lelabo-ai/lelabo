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


def _forward_with_cache(model, *args, **kwargs):
    cache_provider = importlib.import_module("lelabo.models.cache_provider")
    if "cache_spec" not in kwargs:
        try:
            if cache_provider.declares_blocks(model):
                kwargs["cache_spec"] = cache_provider.CacheSpec(target_view="declared")
        except Exception:
            pass
    return cache_provider.forward_with_standard_cache(model, *args, **kwargs)


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
    assert "cache_version" in cache
    assert "module_inputs" in cache
    assert "module_outputs" in cache
    assert isinstance(cache["module_inputs"], dict)
    assert isinstance(cache["module_outputs"], dict)
    block_names: list[str] = []
    if hasattr(model, "declare_blocks"):
        try:
            raw = model.declare_blocks()
            if isinstance(raw, list):
                block_names = [str(getattr(b, "name", "")) for b in raw if str(getattr(b, "name", ""))]
        except Exception:
            block_names = []

    if block_names:
        for name in block_names:
            assert name in cache["module_inputs"]
            assert name in cache["module_outputs"]
    else:
        assert cache["module_inputs"]
        assert cache["module_outputs"]


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
    mlp_mod = importlib.import_module("lelabo.models.builtins.mlp")
    model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")

    x = torch.randn(6, 8)
    out, cache, _blocks = _forward_with_cache(model, x)

    assert tuple(out.shape) == (6, 3)
    _assert_basic_cache_contract(model, out, cache)
    assert "_runtime" in cache
    assert "steps" in cache["_runtime"]
    head_keys = [k for k in cache["module_outputs"].keys() if str(k).endswith("head")]
    assert head_keys
    assert tuple(cache["module_outputs"][head_keys[-1]].shape) == (6, 3)


def test_mlp_classifier_flattens_image_input_without_flatten_module_in_cache() -> None:
    torch = importlib.import_module("torch")
    mlp_mod = importlib.import_module("lelabo.models.builtins.mlp")
    model = mlp_mod.MLPClassifier(in_dim=28 * 28, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")

    x = torch.randn(4, 1, 28, 28)
    out, cache, _blocks = _forward_with_cache(model, x)

    assert tuple(out.shape) == (4, 3)
    assert all("flatten" not in str(name).lower() for name in cache["module_inputs"].keys())
    assert all("flatten" not in str(name).lower() for name in cache["module_outputs"].keys())


def test_mlp_builder_forwards_model_params() -> None:
    registry = _models_registry()
    args = _default_args()
    args.model_params = {
        "hidden": 21,
        "layers": 3,
        "activation": "tanh",
    }
    ctx = registry.ModelContext(
        dataset="mnist",
        num_classes=5,
        in_dim=8,
        in_channels=None,
        input_shape=None,
    )
    model = registry.build_model("mlp", ctx, args)

    assert model.hidden_dim == 21
    assert model.num_layers == 3
    assert model.num_classes == 5
    assert model.activation == "tanh"


def test_mlp_builder_derives_in_dim_from_input_shape_when_missing() -> None:
    registry = _models_registry()
    args = _default_args()
    args.model_params = {
        "hidden": 12,
        "layers": 2,
        "activation": "relu",
    }
    ctx = registry.ModelContext(
        dataset="mnist",
        num_classes=10,
        in_dim=None,
        in_channels=1,
        input_shape=(1, 28, 28),
    )
    model = registry.build_model("mlp", ctx, args)
    assert model.in_dim == 28 * 28


def test_convnet_classifier_cache_contract() -> None:
    torch = importlib.import_module("torch")
    conv_mod = importlib.import_module("lelabo.models.builtins.convnet")
    model = conv_mod.ConvNetClassifier(
        in_channels=1,
        num_classes=10,
        channels=[8, 16],
        kernel_sizes=3,
        use_bn=False,
        pool_every=1,
    )

    x = torch.randn(4, 1, 28, 28)
    out, cache, _blocks = _forward_with_cache(model, x)

    assert tuple(out.shape) == (4, 10)
    _assert_basic_cache_contract(model, out, cache)
    assert "head" in cache["module_outputs"]
    assert tuple(cache["module_outputs"]["head"].shape) == (4, 10)


def test_cnn_builder_forwards_model_params() -> None:
    registry = _models_registry()
    args = _default_args()
    args.model_params = {
        "channels": [4, 7],
        "kernel_sizes": [5, 3],
        "pools": [False, True],
        "use_bn": False,
        "pool_kernel": 3,
        "activation": "tanh",
        "output_activation": "sigmoid",
    }
    ctx = registry.ModelContext(
        dataset="mnist",
        num_classes=6,
        in_dim=None,
        in_channels=1,
        input_shape=(1, 28, 28),
    )
    model = registry.build_model("cnn", ctx, args)

    assert len(model.convs) == 2
    assert model.convs[0].out_channels == 4
    assert model.convs[0].kernel_size == (5, 5)
    assert model.convs[1].out_channels == 7
    assert model.convs[1].kernel_size == (3, 3)
    assert model.head.out_features == 6
    assert type(getattr(model, "bn1")).__name__ == "Identity"
    assert type(getattr(model, "bn2")).__name__ == "Identity"
    assert type(getattr(model, "pool1")).__name__ == "Identity"
    assert type(getattr(model, "pool2")).__name__ == "MaxPool2d"
    assert model.activation_name == "tanh"
    assert model.output_activation_name == "sigmoid"


def test_cnn_builder_supports_flatten_head_mode() -> None:
    registry = _models_registry()
    args = _default_args()
    args.model_params = {
        "channels": [8, 16],
        "kernel_sizes": [3, 3],
        "pools": [True, True],
        "use_bn": False,
        "activation": "relu",
        "output_activation": "identity",
        "head_mode": "flatten",
    }
    ctx = registry.ModelContext(
        dataset="mnist",
        num_classes=10,
        in_dim=None,
        in_channels=1,
        input_shape=(1, 28, 28),
    )
    model = registry.build_model("cnn", ctx, args)

    assert model.head_mode == "flatten"
    # 28x28 --pool2--> 14x14 --pool2--> 7x7 with 16 channels => 16*7*7=784
    assert model.head.in_features == 16 * 7 * 7


def test_deep_softhebb_classifier_cache_contract() -> None:
    torch = importlib.import_module("torch")
    deep_mod = importlib.import_module("lelabo.models.builtins.deep_softhebb")
    model = deep_mod.DeepSoftHebbClassifier(in_channels=3, num_classes=10)

    x = torch.randn(2, 3, 32, 32)
    out, cache, _blocks = _forward_with_cache(model, x)

    assert tuple(out.shape) == (2, 10)
    assert isinstance(cache, dict)
    assert "module_inputs" in cache
    assert "module_outputs" in cache
    for name in ("conv1", "conv2", "conv3", "head"):
        assert name in cache["module_inputs"]
        assert name in cache["module_outputs"]


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

        out_cached, cache, _blocks = _forward_with_cache(model, **batch)
        assert tuple(out_cached.logits.shape) == (3, 2)
        assert "module_inputs" in cache
        assert "head" in cache["module_inputs"]
        assert "embeddings" in cache["module_inputs"]
        assert any(k.startswith("encoder.layer") for k in cache["module_inputs"])


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
    out, cache, _blocks = _forward_with_cache(model, x)
    assert tuple(out.shape) == (4, 5)
    _assert_basic_cache_contract(model, out, cache)
    assert "head" in cache["module_outputs"]


def test_cache_provider_v3_selective_linear_view() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            return self.head(self.relu(self.fc1(x)))

    model = _Tiny()
    x = torch.randn(5, 8)
    out, cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            trainable_module_types=(nn.Linear,),
            observed_module_types=(nn.Linear,),
            require_single_output_head=True,
        ),
    )

    assert tuple(out.shape) == (5, 3)
    assert "module_inputs" in cache
    assert "module_outputs" in cache
    execution_blocks = views["execution"]
    assert len(execution_blocks) == 2
    assert all(isinstance(b["module"], nn.Linear) for b in execution_blocks)
    output_blocks = [b for b in execution_blocks if bool(b["is_output"])]
    assert isinstance(output_blocks, list)
    assert len(output_blocks) == 1
    assert set(views) == {"execution"}
    assert [str(b["name"]) for b in execution_blocks if b["exec_module"] is not None] == ["fc1", "head"]


def test_cache_provider_single_call_uses_call_counts() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Reuse(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x):
            h = torch.tanh(self.fc(x))
            h = torch.tanh(self.fc(h))
            return self.head(h)

    model = _Reuse()
    x = torch.randn(6, 4)
    with pytest.raises(cache_provider.ContractError, match="exactly once"):
        cache_provider.forward_with_standard_cache(
            model,
            x,
            cache_spec=cache_provider.CacheSpec(
                trainable_module_types=(nn.Linear,),
                observed_module_types=(nn.Linear,),
                require_single_call=True,
            ),
        )


def test_cache_provider_activation_pairing_with_module_activation() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            return self.head(self.relu(self.fc1(x)))

    model = _Tiny()
    x = torch.randn(4, 8)
    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            target_view="paired_execution",
            trainable_module_types=(nn.Linear,),
            observed_module_types=(nn.Linear,),
            require_single_output_head=True,
        ),
    )

    ordered_hidden = [
        b for b in views["paired_execution"]
        if bool(b.get("is_trainable", False)) and not bool(b.get("is_output", False))
    ]
    assert ordered_hidden
    assert torch.is_tensor(ordered_hidden[0]["h"])


def test_cache_provider_activation_pairing_through_batchnorm() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _TinyConvBN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()
            self.head = nn.Linear(4 * 8 * 8, 2)

        def forward(self, x):
            h = self.relu(self.bn(self.conv(x)))
            return self.head(h.flatten(1))

    model = _TinyConvBN()
    x = torch.randn(4, 3, 8, 8)
    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            target_view="paired_execution",
            trainable_module_types=(nn.Conv2d, nn.Linear),
            require_single_output_head=True,
        ),
    )

    ordered_hidden = [
        b for b in views["paired_execution"]
        if bool(b.get("is_trainable", False)) and not bool(b.get("is_output", False))
    ]
    assert ordered_hidden
    assert str(ordered_hidden[0]["name"]) == "conv"
    assert str(ordered_hidden[0]["activation_name"]) == "relu"
    assert torch.is_tensor(ordered_hidden[0]["h"])


def test_cache_provider_activation_pairing_fails_for_functional_activation() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            return self.head(torch.relu(self.fc1(x)))

    model = _Tiny()
    x = torch.randn(4, 8)
    with pytest.raises(cache_provider.ContractError, match="Post-activation pairing"):
        cache_provider.forward_with_standard_cache(
            model,
            x,
            cache_spec=cache_provider.CacheSpec(
                target_view="paired_execution",
                trainable_module_types=(nn.Linear,),
                observed_module_types=(nn.Linear,),
                require_single_output_head=True,
            ),
        )


def test_cache_provider_reconstructs_local_conv_blocks_with_intermediate_modules() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _TinyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(8)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.head = nn.Linear(8 * 4 * 4, 5)

        def forward(self, x):
            x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            return self.head(x.flatten(1))

    model = _TinyConv()
    x = torch.randn(3, 3, 16, 16)
    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            trainable_module_types=(nn.Conv2d, nn.Linear),
            require_single_call=True,
            require_single_output_head=True,
        ),
    )

    local_blocks = views["execution"]
    hidden_locals = [
        b
        for b in local_blocks
        if bool(b["is_trainable"]) and not bool(b["is_output"]) and b["exec_module"] is not None
    ]
    assert hidden_locals

    by_name = {str(b["name"]): b for b in hidden_locals}
    assert "conv1" in by_name
    conv1_local = by_name["conv1"]
    segment_names = tuple(str(n) for n in conv1_local.get("exec_span_names", ()))
    assert segment_names == ("conv1", "bn1", "relu1", "pool1")
    assert torch.is_tensor(conv1_local["x"])
    assert torch.is_tensor(conv1_local["u"])
    replay = conv1_local["exec_module"](conv1_local["x"])
    assert torch.is_tensor(replay)
    next_inputs = {str(b["name"]): b["x"] for b in views["execution"]}
    assert torch.is_tensor(next_inputs["conv2"])
    assert tuple(replay.shape) == tuple(next_inputs["conv2"].shape)


def test_cache_provider_multi_head_views_and_single_head_constraint() -> None:
    torch = importlib.import_module("torch")
    actor_mod = importlib.import_module("lelabo.models.builtins.actor_critic")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    model = actor_mod.ActorCriticDiscrete(obs_dim=4, n_actions=3, hidden_dim=8, num_layers=1)
    x = torch.randn(5, 4)

    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            trainable_module_types=(torch.nn.Linear,),
            observed_module_types=(torch.nn.Linear,),
        ),
    )

    output_blocks = [b for b in views["execution"] if bool(b.get("is_output", False))]
    names = sorted(str(b.get("name", "")) for b in output_blocks)
    assert names == ["actor.head", "critic.head"]

    with pytest.raises(cache_provider.ContractError, match="exactly one output head"):
        cache_provider.forward_with_standard_cache(
            model,
            x,
            cache_spec=cache_provider.CacheSpec(
                trainable_module_types=(torch.nn.Linear,),
                observed_module_types=(torch.nn.Linear,),
                require_single_output_head=True,
            ),
        )


def test_cache_provider_returns_only_target_view() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.head = nn.Linear(16, 3)

        def declare_blocks(self):
            return [
                SimpleNamespace(name="head", module=self.head, rep="identity", is_output=True, group="main"),
            ]

        def forward(self, x):
            return self.head(self.relu(self.fc1(x)))

    model = _Tiny()
    x = torch.randn(4, 8)

    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            trainable_module_types=(nn.Linear,),
            observed_module_types=(nn.Linear,),
            require_single_output_head=True,
        ),
    )
    assert set(views) == {"execution"}
    assert [str(b["name"]) for b in views["execution"]] == ["fc1", "head"]

    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        x,
        cache_spec=cache_provider.CacheSpec(
            target_view="declared",
            trainable_module_types=(nn.Linear,),
            observed_module_types=(nn.Linear,),
            require_single_output_head=True,
        ),
    )
    assert set(views) == {"declared"}
    assert [str(b["name"]) for b in views["declared"]] == ["head"]


def test_cache_provider_resolved_block_schema_is_canonical() -> None:
    torch = importlib.import_module("torch")
    actor_mod = importlib.import_module("lelabo.models.builtins.actor_critic")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    model = actor_mod.ActorCriticDiscrete(obs_dim=4, n_actions=2, hidden_dim=8, num_layers=1)
    _out, _cache, views = cache_provider.forward_with_standard_cache(
        model,
        torch.randn(3, 4),
        cache_spec=cache_provider.CacheSpec(
            trainable_module_types=(torch.nn.Linear,),
            observed_module_types=(torch.nn.Linear,),
            target_view="declared",
        ),
    )

    expected_keys = {
        "name",
        "module",
        "spec",
        "rep",
        "group",
        "is_trainable",
        "is_output",
        "x",
        "u",
        "h",
        "activation_name",
        "exec_module",
        "exec_span_names",
        "call_count",
        "available",
        "type",
    }
    assert set(views) == {"declared"}
    blocks = views["declared"]
    assert isinstance(blocks, list)
    assert blocks
    assert expected_keys == set(blocks[0].keys())
    assert tuple(views["declared"][0]["exec_span_names"]) == (str(views["declared"][0]["name"]),)
    assert views["declared"][0]["h"] is None
    assert views["declared"][0]["activation_name"] is None
    declared_names = {str(block["name"]) for block in views["declared"]}
    output_names = {str(block["name"]) for block in views["declared"] if bool(block["is_output"])}
    assert output_names.issubset(declared_names)


def test_cache_provider_helpers_report_declared_blocks() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.head = nn.Linear(16, 3)

        def declare_blocks(self):
            return [
                SimpleNamespace(name="fc1", module=self.fc1, rep="identity", is_output=False, group="main"),
                SimpleNamespace(name="head", module=self.head, rep="identity", is_output=True, group="main"),
            ]

        def forward(self, x):
            return self.head(self.relu(self.fc1(x)))

    model = _Tiny()
    assert cache_provider.declares_blocks(model) is True
    assert [str(block.name) for block in cache_provider.resolve_declared_blocks(model)] == ["fc1", "head"]


def test_cache_provider_helpers_handle_missing_declarations() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.head = nn.Linear(16, 3)

        def forward(self, x):
            return self.head(self.fc1(x))

    model = _Tiny()
    assert cache_provider.declares_blocks(model) is False
    assert cache_provider.resolve_declared_blocks(model) == []


def test_cache_provider_invalid_declare_blocks_raise() -> None:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    cache_provider = importlib.import_module("lelabo.models.cache_provider")

    class _Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.head = nn.Linear(16, 3)

        def declare_blocks(self):
            raise RuntimeError("broken declare_blocks")

        def forward(self, x):
            return self.head(self.fc1(x))

    model = _Broken()
    with pytest.raises(ValueError, match="declare_blocks\\(\\) raised"):
        cache_provider.forward_with_standard_cache(
            model,
            torch.randn(4, 8),
            cache_spec=cache_provider.CacheSpec(trainable_module_types=(nn.Linear,)),
        )
