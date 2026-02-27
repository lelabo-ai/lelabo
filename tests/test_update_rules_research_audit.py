from __future__ import annotations

import importlib
import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from conftest import REPO_ROOT


class AuditInfo(UserWarning):
    """Informational, non-blocking audit message."""


class _Args:
    lr = 1e-3
    weight_decay = 0.0
    max_grad_norm = 0.5
    model = "mlp"

    def __getattr__(self, _name: str):
        return None


def _stub_namespace_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, pkg)

    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.get(parent_name)
        if parent is not None:
            setattr(parent, child_name, pkg)


def _load_lab_modules(monkeypatch: pytest.MonkeyPatch):
    src_root = REPO_ROOT / "src"
    monkeypatch.syspath_prepend(str(src_root))

    _stub_namespace_package(monkeypatch, "lelabo", src_root / "lelabo")
    _stub_namespace_package(monkeypatch, "lelabo.core", src_root / "lelabo" / "core")
    _stub_namespace_package(monkeypatch, "lelabo.models", src_root / "lelabo" / "models")
    _stub_namespace_package(
        monkeypatch,
        "lelabo.update_rules",
        src_root / "lelabo" / "update_rules",
    )

    importlib.import_module("torch")

    registry = importlib.import_module("lelabo.update_rules.registry")
    importlib.import_module("lelabo.update_rules.builtins")
    mlp_mod = importlib.import_module("lelabo.models.builtins.mlp")
    conv_mod = importlib.import_module("lelabo.models.builtins.convnet")
    deephebb_mod = importlib.import_module("lelabo.models.builtins.deep_softhebb")
    ac_mod = importlib.import_module("lelabo.models.builtins.actor_critic")
    task_mod = importlib.import_module("lelabo.core.task")
    return registry, mlp_mod, conv_mod, deephebb_mod, ac_mod, task_mod


def _snapshot_params(model) -> dict[str, Any]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed_params(before: dict[str, Any], after: dict[str, Any], atol: float = 1e-12) -> set[str]:
    changed: set[str] = set()
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            continue
        if old.shape != new.shape:
            changed.add(name)
            continue
        if not old.is_floating_point():
            if not old.equal(new):
                changed.add(name)
            continue
        if not old.allclose(new, atol=atol, rtol=0.0):
            changed.add(name)
    return changed


def _block_param_names(model) -> dict[str, list[str]]:
    if not hasattr(model, "get_blocks"):
        return {}

    named_modules = {mod: name for name, mod in model.named_modules()}
    all_names = list(dict(model.named_parameters()).keys())
    block_to_params: dict[str, list[str]] = {}

    for b in model.get_blocks():
        block_name = str(getattr(b, "name", ""))
        module = getattr(b, "module", None)
        mod_name = named_modules.get(module)
        if not block_name or mod_name is None:
            continue

        prefix = f"{mod_name}."
        pnames = [n for n in all_names if n.startswith(prefix)]
        if pnames:
            block_to_params[block_name] = pnames

    return block_to_params


def _warn_block_coverage(
    algo: str,
    mode: str,
    changed_param_names: set[str],
    block_to_params: dict[str, list[str]],
) -> list[str]:
    unchanged_blocks: list[str] = []
    if not changed_param_names:
        warnings.warn(
            f"⚠ [Rule Audit][{algo}/{mode}] No parameter update detected over the configured audit run. "
            "This is warn-only and can happen during warmup or frozen phases.",
            stacklevel=2,
        )
        return sorted(block_to_params.keys())

    for block_name, pnames in block_to_params.items():
        if not any(name in changed_param_names for name in pnames):
            unchanged_blocks.append(block_name)

    if unchanged_blocks:
        warnings.warn(
            f"⚠ [Rule Audit][{algo}/{mode}] Blocks with no observed update: {', '.join(unchanged_blocks)}. "
            "This can be expected depending on the rule design and training phase.",
            stacklevel=2,
        )
    return sorted(unchanged_blocks)


def _maybe_speedup_learner(learner) -> None:
    # Keep heavy local audits usable: avoid long warmup/pretrain loops when possible.
    if hasattr(learner, "trunk_pretrain"):
        learner.trunk_pretrain = False
    if hasattr(learner, "_phase"):
        learner._phase = "finetune"
    if hasattr(learner, "sup_warmup_steps") and getattr(learner, "sup_warmup_steps", 0) > 10:
        learner.sup_warmup_steps = 10


def _resolve_supervised_model_names(requested: str) -> list[str]:
    available = {"mlp", "cnn", "transformer", "deephebb"}
    raw = str(requested).strip().lower()
    if raw in {"", "auto", "all"}:
        return ["mlp", "cnn", "transformer"]

    names = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not names:
        return ["mlp", "cnn", "transformer"]

    invalid = [n for n in names if n not in available]
    if invalid:
        raise ValueError(
            f"Unsupported supervised model(s): {invalid}. "
            f"Use one of: {sorted(available)} or 'auto'."
        )
    return names


def _build_supervised_case(model_name: str, mlp_mod, conv_mod, deephebb_mod, task_mod, torch, batch_size: int):
    model_name = str(model_name).lower()
    if model_name == "mlp":
        model = mlp_mod.MLPClassifier(in_dim=8, hidden_dim=16, num_layers=2, num_classes=3, activation="relu")
        task = task_mod.ClassificationTask(num_classes=3)
        dataset = "mnist"
        model_flag = "mlp"

        def _make_batch():
            x = torch.randn(batch_size, 8)
            y = torch.randint(0, 3, (batch_size,))
            return x, y

        return model, task, dataset, model_flag, _make_batch

    if model_name == "cnn":
        model = conv_mod.ConvNetClassifier(
            in_channels=1,
            num_classes=10,
            channels=[8, 16],
            kernel_sizes=3,
            use_bn=False,
            pool_every=1,
        )
        task = task_mod.ClassificationTask(num_classes=10)
        dataset = "mnist"
        model_flag = "cnn"

        def _make_batch():
            x = torch.randn(batch_size, 1, 28, 28)
            y = torch.randint(0, 10, (batch_size,))
            return x, y

        return model, task, dataset, model_flag, _make_batch

    if model_name == "transformer":
        import torch.nn as nn

        @dataclass
        class _BlockSpec:
            name: str
            module: nn.Module
            is_output: bool = False

        class _TinyTransformerClassifier(nn.Module):
            def __init__(self, vocab_size: int = 64, seq_len: int = 12, d_model: int = 16, num_classes: int = 3):
                super().__init__()
                self.vocab_size = vocab_size
                self.seq_len = seq_len
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.positional = nn.Embedding(seq_len, d_model)
                self.layers = nn.ModuleList(
                    [
                        nn.TransformerEncoderLayer(
                            d_model=d_model,
                            nhead=2,
                            dim_feedforward=32,
                            dropout=0.0,
                            activation="relu",
                            batch_first=True,
                        )
                        for _ in range(2)
                    ]
                )
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, num_classes)

            def get_blocks(self):
                blocks = []
                for i, layer in enumerate(self.layers):
                    blocks.append(_BlockSpec(name=f"transformer.layer{i}", module=layer, is_output=False))
                blocks.append(_BlockSpec(name="head", module=self.head, is_output=True))
                return blocks

            def forward(self, x, return_cache: bool = False):
                input_ids = x
                attn_mask = None
                if isinstance(x, dict):
                    input_ids = x.get("input_ids")
                    attn_mask = x.get("attention_mask")

                if not torch.is_tensor(input_ids):
                    raise TypeError("Tiny transformer expects a Tensor input_ids.")

                pos_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
                h = self.embedding(input_ids) + self.positional(pos_ids)

                cache = {"block_inputs": {}, "block_outputs": {}} if return_cache else None
                key_padding_mask = None
                if torch.is_tensor(attn_mask):
                    key_padding_mask = ~attn_mask.bool()

                for i, layer in enumerate(self.layers):
                    name = f"transformer.layer{i}"
                    if return_cache:
                        cache["block_inputs"][name] = h
                    h = layer(h, src_key_padding_mask=key_padding_mask)
                    if return_cache:
                        cache["block_outputs"][name] = h

                h = self.norm(h)
                pooled = h[:, 0, :]
                if return_cache:
                    cache["block_inputs"]["head"] = pooled
                logits = self.head(pooled)
                if return_cache:
                    cache["block_outputs"]["head"] = logits
                    return logits, cache
                return logits

        model = _TinyTransformerClassifier()
        task = task_mod.ClassificationTask(num_classes=3)
        dataset = "toy_transformer"
        model_flag = "transformer"

        def _make_batch():
            x = torch.randint(0, 64, (batch_size, 12))
            y = torch.randint(0, 3, (batch_size,))
            return x, y

        return model, task, dataset, model_flag, _make_batch

    if model_name == "deephebb":
        model = deephebb_mod.DeepSoftHebbClassifier(in_channels=3, num_classes=10)
        task = task_mod.ClassificationTask(num_classes=10)
        dataset = "cifar10"
        model_flag = "deephebb"

        def _make_batch():
            x = torch.randn(batch_size, 3, 32, 32)
            y = torch.randint(0, 10, (batch_size,))
            return x, y

        return model, task, dataset, model_flag, _make_batch

    raise ValueError(
        f"Unsupported supervised model '{model_name}'. Use one of: auto, mlp, cnn, transformer, deephebb."
    )


def _warn_execution_issue(algo: str, mode: str, exc: Exception) -> str:
    msg = str(exc)
    lowered = msg.lower()
    incompatible_markers = (
        "only supported in supervised mode",
        "expects tuple batch",
        "expected cache",
        "requires model.get_blocks",
        "no fallback",
        "unsupported",
        "has no get_blocks()",
        "missing",
    )
    if any(marker in lowered for marker in incompatible_markers):
        warnings.warn(
            f"❌ [Rule Audit][{algo}/{mode}] Skipped (contract mismatch): {msg}",
            stacklevel=2,
        )
        return "skipped"

    warnings.warn(
        f"❌ [Rule Audit][{algo}/{mode}] Runtime issue during audit execution: {msg}",
        stacklevel=2,
    )
    return "runtime"


def _emit_success_info(
    algo: str,
    mode_tag: str,
    *,
    epochs: int,
    steps_per_epoch: int,
    changed_param_count: int,
    total_block_count: int,
    unchanged_block_count: int,
) -> None:
    updated_blocks = max(0, total_block_count - unchanged_block_count)
    warnings.warn(
        (
            f"✅ [Rule Audit][{algo}/{mode_tag}] PASS: "
            f"executed {epochs} epoch(s) x {steps_per_epoch} step(s), "
            f"updated_params={changed_param_count}, "
            f"updated_blocks={updated_blocks}/{total_block_count}."
        ),
        category=AuditInfo,
        stacklevel=2,
    )


def _run_supervised_once(
    algo: str,
    registry,
    mlp_mod,
    conv_mod,
    deephebb_mod,
    task_mod,
    torch,
    *,
    model_name: str,
    scenario_name: str,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
) -> str:
    mode_tag = f"supervised/{scenario_name}"
    try:
        model, task, dataset, model_flag, make_batch = _build_supervised_case(
            model_name,
            mlp_mod,
            conv_mod,
            deephebb_mod,
            task_mod,
            torch,
            batch_size,
        )
    except Exception as exc:
        warnings.warn(f"❌ [Rule Audit][{algo}/{mode_tag}] Configuration issue: {exc}", stacklevel=2)
        return "runtime"

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    args = _Args()
    args.model = model_flag
    ctx = registry.UpdateRuleContext(
        args=args,
        model=model,
        task=task,
        optimizer=optimizer,
        mode="supervised",
        dataset=dataset,
    )

    try:
        learner = registry.build_update_rule(algo, ctx)
    except Exception as exc:
        return _warn_execution_issue(algo, mode_tag, exc)

    _maybe_speedup_learner(learner)

    if hasattr(learner, "on_train_start"):
        try:
            learner.on_train_start(model, task, "cpu")
        except Exception as exc:
            return _warn_execution_issue(algo, mode_tag, exc)

    before = _snapshot_params(model)
    for epoch in range(int(max(1, epochs))):
        if hasattr(learner, "on_epoch_start"):
            try:
                learner.on_epoch_start(epoch)
            except Exception:
                pass

        for _ in range(int(max(1, steps_per_epoch))):
            batch = make_batch()
            try:
                learner.train_step(model, task, batch, "cpu")
            except Exception as exc:
                return _warn_execution_issue(algo, mode_tag, exc)

        if hasattr(learner, "on_epoch_end"):
            try:
                learner.on_epoch_end(epoch)
            except Exception:
                pass

    after = _snapshot_params(model)

    changed = _changed_params(before, after)
    block_map = _block_param_names(model)
    unchanged_blocks = _warn_block_coverage(algo, mode_tag, changed, block_map)
    _emit_success_info(
        algo,
        mode_tag,
        epochs=max(1, int(epochs)),
        steps_per_epoch=max(1, int(steps_per_epoch)),
        changed_param_count=len(changed),
        total_block_count=len(block_map),
        unchanged_block_count=len(unchanged_blocks),
    )
    return "pass"


def _run_rl_once(
    algo: str,
    registry,
    ac_mod,
    task_mod,
    torch,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
) -> str:
    model = ac_mod.ActorCriticDiscrete(obs_dim=4, n_actions=2, hidden_dim=16, num_layers=1)
    task = task_mod.PPOTask(task_mod.PPOConfig())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ctx = registry.UpdateRuleContext(
        args=_Args(),
        model=model,
        task=task,
        optimizer=optimizer,
        mode="rl",
        dataset="cartpole",
        rl_algo="ppo",
    )

    try:
        learner = registry.build_update_rule(algo, ctx)
    except Exception as exc:
        return _warn_execution_issue(algo, "rl", exc)

    _maybe_speedup_learner(learner)

    if hasattr(learner, "on_train_start"):
        try:
            learner.on_train_start(model, task, "cpu")
        except Exception as exc:
            return _warn_execution_issue(algo, "rl", exc)

    before = _snapshot_params(model)
    for epoch in range(int(max(1, epochs))):
        if hasattr(learner, "on_epoch_start"):
            try:
                learner.on_epoch_start(epoch)
            except Exception:
                pass

        for _ in range(int(max(1, steps_per_epoch))):
            obs = torch.randn(batch_size, 4)
            y = {
                "actions": torch.randint(0, 2, (batch_size,)),
                "old_logprobs": torch.zeros(batch_size),
                "advantages": torch.randn(batch_size),
                "returns": torch.randn(batch_size),
                "old_values": torch.zeros(batch_size),
            }
            batch = (obs, y)
            try:
                learner.train_step(model, task, batch, "cpu")
            except Exception as exc:
                return _warn_execution_issue(algo, "rl", exc)

        if hasattr(learner, "on_epoch_end"):
            try:
                learner.on_epoch_end(epoch)
            except Exception:
                pass

    after = _snapshot_params(model)

    changed = _changed_params(before, after)
    block_map = _block_param_names(model)
    unchanged_blocks = _warn_block_coverage(algo, "rl", changed, block_map)
    _emit_success_info(
        algo,
        "rl",
        epochs=max(1, int(epochs)),
        steps_per_epoch=max(1, int(steps_per_epoch)),
        changed_param_count=len(changed),
        total_block_count=len(block_map),
        unchanged_block_count=len(unchanged_blocks),
    )
    return "pass"


@pytest.mark.heavy
def test_update_rules_research_audit_warn_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Optional local audit for researchers.
    It never fails by design: it emits warnings to show update coverage/risk.
    """
    if os.getenv("LELABO_RULE_AUDIT", "0") != "1":
        pytest.skip("Set LELABO_RULE_AUDIT=1 to run local warn-only update-rule audits.")

    torch = pytest.importorskip("torch")
    registry, mlp_mod, conv_mod, deephebb_mod, ac_mod, task_mod = _load_lab_modules(monkeypatch)

    available = set(registry.get_update_rule_names())
    raw = os.getenv("LELABO_RULE_AUDIT_ALGOS", ",".join(sorted(available)))
    algos = [x.strip().lower() for x in raw.split(",") if x.strip()]
    modes = [x.strip().lower() for x in os.getenv("LELABO_RULE_AUDIT_MODES", "supervised,rl").split(",") if x.strip()]
    model_spec = os.getenv("LELABO_RULE_AUDIT_MODEL", "auto").strip().lower()
    try:
        supervised_models = _resolve_supervised_model_names(model_spec)
    except Exception as exc:
        warnings.warn(
            f"⚠ [Rule Audit] Invalid supervised model selection '{model_spec}': {exc}. Falling back to auto.",
            stacklevel=2,
        )
        supervised_models = _resolve_supervised_model_names("auto")
    epochs = int(os.getenv("LELABO_RULE_AUDIT_EPOCHS", "1"))
    steps_per_epoch = int(os.getenv("LELABO_RULE_AUDIT_STEPS_PER_EPOCH", "1"))
    batch_size = int(os.getenv("LELABO_RULE_AUDIT_BATCH_SIZE", "16"))

    for algo in algos:
        if algo not in available:
            warnings.warn(
                f"⚠ [Rule Audit] Unknown algorithm '{algo}'. Available: {sorted(available)}",
                stacklevel=2,
            )
            continue

        for mode in modes:
            if mode == "supervised":
                for model_name in supervised_models:
                    _run_supervised_once(
                        algo,
                        registry,
                        mlp_mod,
                        conv_mod,
                        deephebb_mod,
                        task_mod,
                        torch,
                        model_name=model_name,
                        scenario_name=model_name,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        batch_size=batch_size,
                    )
            elif mode == "rl":
                _run_rl_once(
                    algo,
                    registry,
                    ac_mod,
                    task_mod,
                    torch,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,
                    batch_size=batch_size,
                )
            else:
                warnings.warn(
                    f"⚠ [Rule Audit][{algo}] Unknown mode '{mode}'. Use 'supervised' or 'rl'.",
                    stacklevel=2,
                )
