from __future__ import annotations

import importlib
import os
import sys
from argparse import Namespace

import pytest

from conftest import REPO_ROOT


sys.path.insert(0, str(REPO_ROOT / "src"))


def _models_registry():
    # Keep import order stable in this environment (torch first).
    importlib.import_module("torch")
    return importlib.import_module("lelabo.models.registry")


def _default_args() -> Namespace:
    return Namespace(
        hidden=32,
        layers=2,
        hf_model=os.getenv("LELABO_HEAVY_HF_MODEL", "bert-base-uncased"),
        hf_trust_remote_code=False,
    )


def _forward_with_cache(model, *args, **kwargs):
    cache_provider = importlib.import_module("lelabo.models.cache_provider")
    return cache_provider.forward_with_standard_cache(model, *args, **kwargs)


def _require_heavy_models_enabled() -> None:
    if os.getenv("LELABO_MODEL_HEAVY", "0") != "1":
        pytest.skip("Set LELABO_MODEL_HEAVY=1 to run heavy real-model runtime tests.")


def _is_env_bound_runtime_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "connection error",
        "timed out",
        "temporary failure in name resolution",
        "name or service not known",
        "couldn't be found on the hugging face hub",
        "permission denied",
        "ssl",
    )
    return any(marker in msg for marker in markers)


@pytest.mark.heavy
def test_resnet18_runtime_cache_heavy() -> None:
    _require_heavy_models_enabled()
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    resnet_mod = importlib.import_module("lelabo.models.builtins.resnet")
    model = resnet_mod.ResNet(num_classes=7, resnet_type="resnet18", pretrained=False)

    x = torch.randn(2, 3, 64, 64)
    out, cache, _blocks = _forward_with_cache(model, x)

    assert tuple(out.shape) == (2, 7)
    assert "block_inputs" in cache
    assert "block_outputs" in cache
    assert "head" in cache["block_inputs"]
    assert "head" in cache["block_outputs"]
    assert any(k.startswith("layer1.") for k in cache["block_inputs"])


@pytest.mark.heavy
@pytest.mark.filterwarnings(
    "ignore:'maxsplit' is passed as positional argument:DeprecationWarning:fsspec.utils"
)
@pytest.mark.filterwarnings(
    "ignore:co_lnotab is deprecated, use co_lines instead\\.:DeprecationWarning:datasets.utils._dill"
)
@pytest.mark.parametrize("builder_name", ["hf", "bert"])
def test_hf_bert_runtime_cache_heavy(builder_name: str) -> None:
    _require_heavy_models_enabled()
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    registry = _models_registry()
    ctx = registry.ModelContext(
        dataset="glue",
        num_classes=2,
        in_dim=None,
        in_channels=None,
        input_shape=None,
    )
    args = _default_args()

    try:
        model = registry.build_model(builder_name, ctx, args)
    except Exception as exc:
        if _is_env_bound_runtime_error(exc):
            pytest.skip(f"HF heavy test skipped due environment/cache/network constraints: {exc}")
        raise

    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 30522) or 30522)
    vocab_max = max(8, min(vocab_size, 1024))
    input_ids = torch.randint(0, vocab_max, (2, 16), dtype=torch.long)
    attention_mask = torch.ones(2, 16, dtype=torch.long)
    labels = torch.randint(0, 2, (2,), dtype=torch.long)

    try:
        out, cache, _blocks = _forward_with_cache(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    except Exception as exc:
        if _is_env_bound_runtime_error(exc):
            pytest.skip(f"HF heavy forward skipped due environment/cache/network constraints: {exc}")
        raise

    assert hasattr(out, "logits")
    assert tuple(out.logits.shape) == (2, 2)
    assert "block_inputs" in cache
    assert "head" in cache["block_inputs"]
    assert any(k.startswith("encoder.layer") for k in cache["block_inputs"])
