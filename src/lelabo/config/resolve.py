"""Resolve layered configs from defaults, files, and CLI overrides."""

from __future__ import annotations

import copy
import os
import re
from argparse import Namespace
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.display import normalize_display_mode
from .defaults import DEFAULT_RL_CONFIG, DEFAULT_SUPERVISED_CONFIG
from .schema import (
    CallbackSpec,
    ComponentSpec,
    HFSpec,
    MetricSpec,
    RLConfig,
    RLSpec,
    RobustnessSpec,
    RuntimeSpec,
    SchedulerSpec,
    SupervisedConfig,
    SupervisedTrainSpec,
    WandbSpec,
)
from .versioning import (
    TRAIN_CONFIG_SCHEMA_VERSION,
    resolve_config_version,
    resolve_lelabo_version,
)

import tomllib


_SET_BAREWORD_RE = re.compile(r"^[A-Za-z0-9_./:+-]+$")


def _normalize_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _normalize_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {_normalize_key(str(k)): _normalize_keys(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_keys(v) for v in value]
    return value


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for raw_key, raw_value in update.items():
        key = str(raw_key)
        value = raw_value
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(dict(out[key]), dict(value))
        else:
            out[key] = copy.deepcopy(value)
    return out


def _as_dict(value: Any, *, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{where}' to be a table/object, got {type(value).__name__}.")
    return dict(value)


def _path_exists(data: Mapping[str, Any], path: Iterable[str]) -> bool:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return False
        cur = cur[key]
    return True


def _set_path(data: dict[str, Any], path: list[str], value: Any) -> None:
    cur = data
    for key in path[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[path[-1]] = value


def _coerce_scalar_from_toml(raw: str) -> Any:
    snippet = f"value = {raw}"
    try:
        parsed = tomllib.loads(snippet)
        return parsed["value"]
    except tomllib.TOMLDecodeError as exc:
        token = str(raw).strip()
        if _SET_BAREWORD_RE.fullmatch(token):
            return token
        raise ValueError(
            f"Invalid --set value {raw!r}. Use valid TOML literals for structured values "
            "and quote strings that contain spaces or special characters."
        ) from exc


def parse_set_overrides(entries: list[str] | tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_entry in entries:
        entry = str(raw_entry).strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Invalid --set override '{entry}'. Expected KEY=VALUE.")
        key_raw, value_raw = entry.split("=", 1)
        dotted_key = _normalize_key(key_raw).strip(".")
        if not dotted_key:
            raise ValueError(f"Invalid --set override '{entry}'. Empty key.")
        path = [_normalize_key(part) for part in dotted_key.split(".") if part.strip()]
        if not path:
            raise ValueError(f"Invalid --set override '{entry}'. Empty key.")
        _set_path(out, path, _coerce_scalar_from_toml(value_raw.strip()))
    return out


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Config file does not exist: {path}")
    try:
        with path.open("rb") as f:
            loaded = tomllib.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse TOML config '{path}': {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config file '{path}' must contain a TOML table at the root.")
    return _normalize_keys(dict(loaded))


def _normalize_component(raw: Any, *, default_name: str) -> ComponentSpec:
    if raw is None:
        return ComponentSpec(name=default_name, params={})
    if isinstance(raw, str):
        return ComponentSpec(name=str(raw).strip(), params={})

    node = _as_dict(raw, where="component")
    name = str(node.get("name", default_name) or "").strip()
    params_raw = node.get("params", {})
    params = _as_dict(params_raw, where="component.params")
    extras = {
        str(k): v
        for k, v in node.items()
        if str(k) not in {"name", "params"}
    }
    merged_params = dict(params)
    merged_params.update(extras)
    return ComponentSpec(name=name, params=merged_params)


def _normalize_metrics(raw: Any) -> tuple[MetricSpec, ...]:
    if raw is None:
        return ()

    out: list[MetricSpec] = []

    if isinstance(raw, str):
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        return tuple(MetricSpec(name=tok, params={}) for tok in tokens)

    if isinstance(raw, Mapping):
        for key, val in raw.items():
            params = _as_dict(val, where=f"metrics.{key}")
            out.append(MetricSpec(name=str(key).strip(), params=params))
        return tuple(out)

    if not isinstance(raw, list):
        raise ValueError("Expected 'metrics' to be a list, string, or table.")

    for idx, item in enumerate(raw):
        if isinstance(item, str):
            out.append(MetricSpec(name=item.strip(), params={}))
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected metrics[{idx}] to be a table or string.")
        node = dict(item)
        name = str(node.get("name", "")).strip()
        if not name:
            raise ValueError(f"Expected metrics[{idx}].name to be non-empty.")
        params = _as_dict(node.get("params", {}), where=f"metrics[{idx}].params")
        extras = {str(k): v for k, v in node.items() if str(k) not in {"name", "params"}}
        merged = dict(params)
        merged.update(extras)
        out.append(MetricSpec(name=name, params=merged))
    return tuple(out)


def _normalize_callbacks(raw: Any) -> tuple[CallbackSpec, ...]:
    if raw is None:
        return ()

    out: list[CallbackSpec] = []

    if isinstance(raw, str):
        tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
        return tuple(CallbackSpec(name=tok, enabled=True, params={}) for tok in tokens)

    if isinstance(raw, Mapping):
        for key, val in raw.items():
            params = _as_dict(val, where=f"callbacks.{key}")
            out.append(CallbackSpec(name=str(key).strip(), enabled=True, params=params))
        return tuple(out)

    if not isinstance(raw, list):
        raise ValueError("Expected 'callbacks' to be a list, string, or table.")

    for idx, item in enumerate(raw):
        if isinstance(item, str):
            out.append(CallbackSpec(name=item.strip(), enabled=True, params={}))
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected callbacks[{idx}] to be a table or string.")
        node = dict(item)
        name = str(node.get("name", "")).strip()
        if not name:
            raise ValueError(f"Expected callbacks[{idx}].name to be non-empty.")
        enabled = bool(node.get("enabled", True))
        params = _as_dict(node.get("params", {}), where=f"callbacks[{idx}].params")
        extras = {str(k): v for k, v in node.items() if str(k) not in {"name", "enabled", "params"}}
        merged = dict(params)
        merged.update(extras)
        out.append(CallbackSpec(name=name, enabled=enabled, params=merged))
    return tuple(out)


def _reject_legacy_callback_config(merged: Mapping[str, Any]) -> None:
    legacy_keys = {
        "early_stopping",
        "early_stop",
        "early_monitor",
        "early_mode",
        "early_patience",
        "early_min_delta",
        "early_warmup",
        "early_restore_best",
    }
    present = sorted(k for k in legacy_keys if k in merged)
    if present:
        raise ValueError(
            "Legacy early-stopping config keys are no longer supported: "
            f"{present}. Use [[callbacks]] with name='earlystopping'."
        )


def _display_from_runtime(runtime_raw: Mapping[str, Any]) -> str:
    if "verbose" in runtime_raw:
        raise ValueError("runtime.verbose has been removed. Use runtime.display = 'none'|'compact'|'rich'.")
    if "display" not in runtime_raw:
        return "compact"
    return normalize_display_mode(runtime_raw.get("display"), where="runtime.display")


def _apply_aliases(data: dict[str, Any], *, mode: str) -> None:
    # Shared aliases for scalar top-level convenience.
    aliases: list[tuple[str, list[str]]] = [
        ("lr", ["optimizer", "params", "lr"]),
        ("weight_decay", ["optimizer", "params", "weight_decay"]),
        ("hidden", ["model", "params", "hidden"]),
        ("layers", ["model", "params", "layers"]),
        ("seed", ["runtime", "seed"]),
        ("device", ["runtime", "device"]),
        ("determinism", ["runtime", "determinism"]),
        ("display", ["runtime", "display"]),
        ("run_dir", ["runtime", "run_dir"]),
        ("save_checkpoints", ["runtime", "save_checkpoints"]),
    ]
    if mode == "supervised":
        aliases.extend(
            [
                ("rule", ["update_rule", "name"]),
                ("dataset", ["dataset", "name"]),
                ("initializer", ["initializer", "name"]),
                ("epochs", ["train", "epochs"]),
                ("batch", ["train", "batch"]),
                ("val_frac", ["train", "val_frac"]),
                ("input_noise_dataset", ["train", "input_noise_dataset"]),
                ("noise_on_test", ["train", "noise_on_test"]),
                ("robustness", ["robustness", "mode"]),
                ("noise_trials", ["robustness", "trials"]),
                ("robustness_max_samples", ["robustness", "max_samples"]),
                ("glue_task", ["hf", "glue_task"]),
                ("hf_model", ["hf", "model"]),
                ("hf_trust_remote_code", ["hf", "trust_remote_code"]),
                ("max_length", ["hf", "max_length"]),
                ("lr_scheduler", ["scheduler", "name"]),
                ("lr_scheduler_interval", ["scheduler", "interval"]),
                ("lr_scheduler_monitor", ["scheduler", "monitor"]),
            ]
        )
    if mode == "rl":
        aliases.extend(
            [
                ("rule", ["update_rule", "name"]),
                ("env", ["env"]),
                ("algo", ["rl", "algo"]),
                ("rl_steps", ["rl", "steps"]),
                ("rl_eval_episodes", ["rl", "eval_episodes"]),
            ]
        )

    for alias_key, target_path in aliases:
        if alias_key not in data:
            continue
        if _path_exists(data, target_path):
            continue
        _set_path(data, target_path, data[alias_key])

def _validate_supervised(cfg: SupervisedConfig) -> None:
    if str(cfg.config_version).strip() != TRAIN_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config_version='{cfg.config_version}'. "
            f"Expected '{TRAIN_CONFIG_SCHEMA_VERSION}'."
        )
    if not str(cfg.lelabo_version).strip():
        raise ValueError("lelabo_version must be non-empty.")
    if not str(cfg.dataset.name).strip():
        raise ValueError("--dataset cannot be empty. Set it in config (dataset.name) or via CLI.")
    if cfg.train.epochs <= 0:
        raise ValueError("train.epochs must be > 0.")
    if cfg.train.batch <= 0:
        raise ValueError("train.batch must be > 0.")
    if not (0.0 <= float(cfg.train.val_frac) < 1.0):
        raise ValueError("train.val_frac must satisfy 0 <= val_frac < 1.")
    if cfg.robustness.trials <= 0:
        raise ValueError("robustness.trials must be > 0.")
    if cfg.robustness.max_samples < 0:
        raise ValueError("robustness.max_samples must be >= 0.")
    if str(cfg.scheduler.interval).lower() not in {"epoch", "batch"}:
        raise ValueError("scheduler.interval must be one of: epoch, batch.")
    for idx, cb in enumerate(cfg.callbacks):
        name = str(cb.name).strip()
        if not name:
            raise ValueError(f"callbacks[{idx}].name must be non-empty.")
        if not isinstance(cb.params, dict):
            raise ValueError(f"callbacks[{idx}].params must be a table/object.")
        name_norm = name.lower()
        if name_norm in {"early_stopping", "earlystop", "es"}:
            raise ValueError(
                "Legacy callback name "
                f"'{name}' is not supported. Use 'earlystopping'."
            )
    if str(cfg.runtime.determinism).lower() not in {"off", "relaxed", "strict"}:
        raise ValueError("runtime.determinism must be one of: off, relaxed, strict.")
    if str(cfg.runtime.display).lower() not in {"none", "compact", "rich"}:
        raise ValueError("runtime.display must be one of: none, compact, rich.")


def _reject_removed_supervised_keys(merged: Mapping[str, Any]) -> None:
    removed_top_level = sorted(key for key in ("input_noise_training",) if key in merged)
    if removed_top_level:
        raise ValueError(
            "Removed supervised config key(s): "
            f"{removed_top_level}. Use dataset-level noise controls only: "
            "train.input_noise_dataset and train.noise_on_test."
        )

    train_raw = merged.get("train", {})
    if isinstance(train_raw, Mapping) and "input_noise_training" in train_raw:
        raise ValueError(
            "train.input_noise_training has been removed. "
            "Use dataset-level noise controls only: "
            "train.input_noise_dataset and train.noise_on_test."
        )


def _validate_rl(cfg: RLConfig) -> None:
    if str(cfg.config_version).strip() != TRAIN_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported config_version='{cfg.config_version}'. "
            f"Expected '{TRAIN_CONFIG_SCHEMA_VERSION}'."
        )
    if not str(cfg.lelabo_version).strip():
        raise ValueError("lelabo_version must be non-empty.")
    if not str(cfg.env).strip():
        raise ValueError("--env cannot be empty. Set it in config (env) or via CLI.")
    if cfg.rl.steps <= 0:
        raise ValueError("rl.steps must be > 0.")
    if cfg.rl.eval_episodes <= 0:
        raise ValueError("rl.eval_episodes must be > 0.")
    if str(cfg.runtime.determinism).lower() not in {"off", "relaxed", "strict"}:
        raise ValueError("runtime.determinism must be one of: off, relaxed, strict.")
    if str(cfg.runtime.display).lower() not in {"none", "compact", "rich"}:
        raise ValueError("runtime.display must be one of: none, compact, rich.")


def resolve_supervised_config(
    *,
    config_path: str | Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    set_overrides: list[str] | tuple[str, ...] | None = None,
    project_defaults: Mapping[str, Any] | None = None,
) -> SupervisedConfig:
    merged = copy.deepcopy(DEFAULT_SUPERVISED_CONFIG)
    if project_defaults:
        merged = _deep_merge(merged, _normalize_keys(dict(project_defaults)))
    if config_path:
        merged = _deep_merge(merged, _load_toml(Path(config_path).expanduser()))
    if cli_overrides:
        merged = _deep_merge(merged, _normalize_keys(dict(cli_overrides)))
    if set_overrides:
        merged = _deep_merge(merged, parse_set_overrides(set_overrides))

    merged = _normalize_keys(merged)
    _apply_aliases(merged, mode="supervised")
    _reject_removed_supervised_keys(merged)
    _reject_legacy_callback_config(merged)
    config_version = resolve_config_version(merged.get("config_version", TRAIN_CONFIG_SCHEMA_VERSION))
    lelabo_version = resolve_lelabo_version(merged.get("lelabo_version", "auto"))

    dataset = _normalize_component(merged.get("dataset"), default_name="")
    model = _normalize_component(merged.get("model"), default_name="mlp")
    initializer = _normalize_component(merged.get("initializer"), default_name="none")
    loss = _normalize_component(merged.get("loss"), default_name="cross_entropy")
    update_rule = _normalize_component(merged.get("update_rule"), default_name="bp")
    optimizer = _normalize_component(merged.get("optimizer"), default_name="adamw")

    sched_raw = _as_dict(merged.get("scheduler", {}), where="scheduler")
    scheduler = SchedulerSpec(
        name=str(sched_raw.get("name", "none")).strip(),
        interval=str(sched_raw.get("interval", "epoch")).strip(),
        monitor=str(sched_raw.get("monitor", "val.loss")).strip(),
        params=_as_dict(sched_raw.get("params", {}), where="scheduler.params"),
    )

    runtime_raw = _as_dict(merged.get("runtime", {}), where="runtime")
    runtime = RuntimeSpec(
        device=str(runtime_raw.get("device", "auto")),
        seed=int(runtime_raw.get("seed", 2)),
        determinism=str(runtime_raw.get("determinism", "relaxed")),
        display=_display_from_runtime(runtime_raw),
        run_dir=runtime_raw.get("run_dir"),
        save_checkpoints=bool(runtime_raw.get("save_checkpoints", False)),
    )

    train_raw = _as_dict(merged.get("train", {}), where="train")
    train = SupervisedTrainSpec(
        epochs=int(train_raw.get("epochs", 50)),
        batch=int(train_raw.get("batch", 64)),
        val_frac=float(train_raw.get("val_frac", 0.1)),
        input_noise_dataset=float(train_raw.get("input_noise_dataset", 0.0)),
        noise_on_test=bool(train_raw.get("noise_on_test", False)),
    )

    robust_raw = _as_dict(merged.get("robustness", {}), where="robustness")
    robustness = RobustnessSpec(
        mode=str(robust_raw.get("mode", "none")).strip(),
        trials=int(robust_raw.get("trials", 30)),
        max_samples=int(robust_raw.get("max_samples", 0)),
        params=_as_dict(robust_raw.get("params", {}), where="robustness.params"),
    )

    hf_raw = _as_dict(merged.get("hf", {}), where="hf")
    hf = HFSpec(
        glue_task=str(hf_raw.get("glue_task", "sst2")).strip(),
        model=str(hf_raw.get("model", "bert-base-uncased")).strip(),
        trust_remote_code=bool(hf_raw.get("trust_remote_code", False)),
        max_length=int(hf_raw.get("max_length", 128)),
    )

    metrics = _normalize_metrics(merged.get("metrics", []))
    callbacks = _normalize_callbacks(merged.get("callbacks", []))

    wandb_raw = _as_dict(merged.get("wandb", {}), where="wandb")
    wandb = WandbSpec(
        project=wandb_raw.get("project") or os.environ.get("WANDB_PROJECT") or None,
        entity=wandb_raw.get("entity") or os.environ.get("WANDB_ENTITY") or None,
        tags=tuple(wandb_raw.get("tags", ())),
        group=wandb_raw.get("group"),
        notes=str(wandb_raw.get("notes", "")),
        enabled=bool(wandb_raw.get("enabled", True)),
    )

    cfg = SupervisedConfig(
        config_version=config_version,
        lelabo_version=lelabo_version,
        task="supervised",
        dataset=dataset,
        model=model,
        initializer=initializer,
        loss=loss,
        update_rule=update_rule,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime=runtime,
        train=train,
        robustness=robustness,
        hf=hf,
        metrics=metrics,
        callbacks=callbacks,
        wandb=wandb,
    )
    _validate_supervised(cfg)
    return cfg


def resolve_rl_config(
    *,
    config_path: str | Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    set_overrides: list[str] | tuple[str, ...] | None = None,
    project_defaults: Mapping[str, Any] | None = None,
) -> RLConfig:
    merged = copy.deepcopy(DEFAULT_RL_CONFIG)
    if project_defaults:
        merged = _deep_merge(merged, _normalize_keys(dict(project_defaults)))
    if config_path:
        merged = _deep_merge(merged, _load_toml(Path(config_path).expanduser()))
    if cli_overrides:
        merged = _deep_merge(merged, _normalize_keys(dict(cli_overrides)))
    if set_overrides:
        merged = _deep_merge(merged, parse_set_overrides(set_overrides))

    merged = _normalize_keys(merged)
    _apply_aliases(merged, mode="rl")
    config_version = resolve_config_version(merged.get("config_version", TRAIN_CONFIG_SCHEMA_VERSION))
    lelabo_version = resolve_lelabo_version(merged.get("lelabo_version", "auto"))

    model = _normalize_component(merged.get("model"), default_name="default")
    update_rule = _normalize_component(merged.get("update_rule"), default_name="bp")
    optimizer = _normalize_component(merged.get("optimizer"), default_name="adamw")

    runtime_raw = _as_dict(merged.get("runtime", {}), where="runtime")
    runtime = RuntimeSpec(
        device=str(runtime_raw.get("device", "auto")),
        seed=int(runtime_raw.get("seed", 2)),
        determinism=str(runtime_raw.get("determinism", "relaxed")),
        display=_display_from_runtime(runtime_raw),
        run_dir=runtime_raw.get("run_dir"),
        save_checkpoints=bool(runtime_raw.get("save_checkpoints", False)),
    )

    rl_raw = _as_dict(merged.get("rl", {}), where="rl")
    rl = RLSpec(
        algo=str(rl_raw.get("algo", "ppo")).strip(),
        steps=int(rl_raw.get("steps", 500_000)),
        eval_episodes=int(rl_raw.get("eval_episodes", 10)),
        params=_as_dict(rl_raw.get("params", {}), where="rl.params"),
    )

    wandb_raw = _as_dict(merged.get("wandb", {}), where="wandb")
    wandb = WandbSpec(
        project=wandb_raw.get("project") or os.environ.get("WANDB_PROJECT") or None,
        entity=wandb_raw.get("entity") or os.environ.get("WANDB_ENTITY") or None,
        tags=tuple(wandb_raw.get("tags", ())),
        group=wandb_raw.get("group"),
        notes=str(wandb_raw.get("notes", "")),
        enabled=bool(wandb_raw.get("enabled", True)),
    )

    cfg = RLConfig(
        config_version=config_version,
        lelabo_version=lelabo_version,
        task="rl",
        env=str(merged.get("env", "")),
        model=model,
        update_rule=update_rule,
        optimizer=optimizer,
        runtime=runtime,
        rl=rl,
        wandb=wandb,
    )
    _validate_rl(cfg)
    return cfg


def to_supervised_namespace(
    cfg: SupervisedConfig,
    *,
    device_resolver,
) -> Namespace:
    model_params = dict(cfg.model.params)
    initializer_params = dict(cfg.initializer.params)
    loss_params = dict(cfg.loss.params)
    optimizer_params = dict(cfg.optimizer.params)
    dataset_params = dict(cfg.dataset.params)
    update_rule_params = dict(cfg.update_rule.params)
    scheduler_params = dict(cfg.scheduler.params)
    callback_specs = [
        {
            "name": cb.name,
            "enabled": bool(cb.enabled),
            "params": dict(cb.params),
        }
        for cb in cfg.callbacks
    ]

    metric_names = [m.name for m in cfg.metrics]
    metric_params = {m.name: dict(m.params) for m in cfg.metrics}

    device = str(cfg.runtime.device).strip().lower()
    resolved_device = device_resolver() if device in {"", "auto", "default"} else str(cfg.runtime.device)

    raw: dict[str, Any] = {
        "config_version": cfg.config_version,
        "lelabo_version": cfg.lelabo_version,
        "mode": "supervised",
        "task": "supervised",
        "dataset": cfg.dataset.name,
        "model": cfg.model.name,
        "initializer": cfg.initializer.name,
        "loss": cfg.loss.name,
        "rule": cfg.update_rule.name,
        "metrics": ",".join(metric_names),
        "metric_params": metric_params,
        "epochs": int(cfg.train.epochs),
        "batch": int(cfg.train.batch),
        "lr_scheduler": cfg.scheduler.name,
        "lr_scheduler_interval": cfg.scheduler.interval,
        "lr_scheduler_monitor": cfg.scheduler.monitor,
        "input_noise_dataset": float(cfg.train.input_noise_dataset),
        "noise_on_test": int(bool(cfg.train.noise_on_test)),
        "val_frac": float(cfg.train.val_frac),
        "callbacks": callback_specs,
        "robustness": cfg.robustness.mode,
        "noise_trials": int(cfg.robustness.trials),
        "robustness_max_samples": int(cfg.robustness.max_samples),
        "glue_task": cfg.hf.glue_task,
        "hf_model": cfg.hf.model,
        "hf_trust_remote_code": bool(cfg.hf.trust_remote_code),
        "max_length": int(cfg.hf.max_length),
        "hidden": int(model_params.get("hidden", 2048)),
        "layers": int(model_params.get("layers", 4)),
        "lr": float(optimizer_params.get("lr", 1e-3)),
        "device": resolved_device,
        "seed": int(cfg.runtime.seed),
        "determinism": str(cfg.runtime.determinism).lower(),
        "display": str(cfg.runtime.display).lower(),
        "optimizer": cfg.optimizer.name,
        "weight_decay": float(optimizer_params.get("weight_decay", 0.01)),
        "run_dir": cfg.runtime.run_dir,
        "save_checkpoints": bool(cfg.runtime.save_checkpoints),
        "model_params": model_params,
        "initializer_params": initializer_params,
        "loss_params": loss_params,
        "dataset_params": dataset_params,
        "update_rule_params": update_rule_params,
        "optimizer_params": optimizer_params,
        "scheduler_params": scheduler_params,
        "robustness_params": dict(cfg.robustness.params),
        "wandb_project": cfg.wandb.project,
        "wandb_entity": cfg.wandb.entity,
        "wandb_tags": list(cfg.wandb.tags),
        "wandb_group": cfg.wandb.group,
        "wandb_notes": cfg.wandb.notes,
        "wandb_enabled": cfg.wandb.enabled,
    }

    return Namespace(**raw)


def to_rl_namespace(
    cfg: RLConfig,
    *,
    device_resolver,
) -> Namespace:
    model_params = dict(cfg.model.params)
    optimizer_params = dict(cfg.optimizer.params)
    update_rule_params = dict(cfg.update_rule.params)
    rl_params = dict(cfg.rl.params)

    device = str(cfg.runtime.device).strip().lower()
    resolved_device = device_resolver() if device in {"", "auto", "default"} else str(cfg.runtime.device)

    raw: dict[str, Any] = {
        "config_version": cfg.config_version,
        "lelabo_version": cfg.lelabo_version,
        "mode": "rl",
        "task": "rl",
        "env": cfg.env,
        "dataset": f"env:{cfg.env}",
        "rule": cfg.update_rule.name,
        "algo": cfg.rl.algo,
        "rl_steps": int(cfg.rl.steps),
        "rl_eval_episodes": int(cfg.rl.eval_episodes),
        "hidden": int(model_params.get("hidden", 2048)),
        "layers": int(model_params.get("layers", 4)),
        "lr": float(optimizer_params.get("lr", 1e-3)),
        "weight_decay": float(optimizer_params.get("weight_decay", 0.01)),
        "device": resolved_device,
        "seed": int(cfg.runtime.seed),
        "determinism": str(cfg.runtime.determinism).lower(),
        "display": str(cfg.runtime.display).lower(),
        "optimizer": cfg.optimizer.name,
        "run_dir": cfg.runtime.run_dir,
        "save_checkpoints": bool(cfg.runtime.save_checkpoints),
        "model": cfg.model.name,
        "model_params": model_params,
        "update_rule_params": update_rule_params,
        "optimizer_params": optimizer_params,
        "rl_params": rl_params,
        "wandb_project": cfg.wandb.project,
        "wandb_entity": cfg.wandb.entity,
        "wandb_tags": list(cfg.wandb.tags),
        "wandb_group": cfg.wandb.group,
        "wandb_notes": cfg.wandb.notes,
        "wandb_enabled": cfg.wandb.enabled,
    }
    return Namespace(**raw)
