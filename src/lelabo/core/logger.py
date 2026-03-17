"""Run logging, metadata persistence, and checkpoint writing for training jobs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from uuid import uuid4

import yaml

from .run_artifacts import (
    RUN_CHECKPOINT_SCHEMA_VERSION,
    RUN_META_SCHEMA_VERSION,
    RUN_METRICS_SCHEMA_VERSION,
    RUN_SEEDS_SCHEMA_VERSION,
    RUN_SUMMARY_SCHEMA_VERSION,
    build_persisted_summary,
    json_like,
    public_args_dict,
    utc_now_iso,
)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """Write a formatted JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_git_commit() -> Optional[str]:
    """Return the current short git commit hash when available."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def _to_cpu_state(raw: Any) -> Any:
    """Detach tensors recursively so persisted checkpoints are device-agnostic."""
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(raw):
        return raw.detach().cpu().clone()
    if isinstance(raw, Mapping):
        return {str(k): _to_cpu_state(v) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_to_cpu_state(v) for v in raw]
    if isinstance(raw, tuple):
        return tuple(_to_cpu_state(v) for v in raw)
    return raw


def _checkpoint_mode_from_name(name: str) -> str:
    """Infer whether a monitor should be minimized or maximized from its name."""
    token = str(name).strip().lower()
    if any(item in token for item in ("acc", "f1", "precision", "recall", "r2", "auc")):
        return "max"
    return "min"


@dataclass
class RunLogger:
    """Own run-artifact writing for one training execution."""

    run_dir: Optional[Path] = None
    save_checkpoints: bool = False
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_group: Optional[str] = None
    wandb_notes: str = ""
    wandb_enabled: bool = True

    def __post_init__(self) -> None:
        """Materialize artifact paths for the configured run directory."""
        self._wandb_run: Any = None
        self._wandb_available = False
        if self.wandb_enabled and self.wandb_project:
            try:
                import wandb as _wandb  # noqa: F401
                self._wandb_available = True
            except ImportError:
                self._wandb_available = False
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_path = self.run_dir / "metrics.jsonl"
            self.meta_path = self.run_dir / "meta.json"
            self.summary_path = self.run_dir / "summary.json"
            self.config_path = self.run_dir / "resolved_config.yaml"
            self.seeds_path = self.run_dir / "seeds.json"
            self.checkpoints_path = self.run_dir / "checkpoints" if self.save_checkpoints else None
        else:
            self.metrics_path = None
            self.meta_path = None
            self.summary_path = None
            self.config_path = None
            self.seeds_path = None
            self.checkpoints_path = None
        self._meta_payload: dict[str, Any] | None = None
        self._best_monitor_name: str | None = None
        self._best_monitor_mode: str | None = None
        self._best_monitor_value: float | None = None
        self._best_checkpoint_payload: dict[str, Any] | None = None

    def _init_wandb(self, *, config: dict[str, Any] | None = None, name: str | None = None) -> None:
        """Lazily initialize a W&B run. No-op if already initialized or unavailable."""
        if not self._wandb_available or self._wandb_run is not None:
            return
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=name or self.run_id,
                group=self.wandb_group,
                tags=self.wandb_tags or None,
                notes=self.wandb_notes or None,
                config=config,
                id=self.run_id,
                resume="allow",
            )
        except Exception:
            self._wandb_run = None

    def _finish_wandb(self, exit_code: int = 0) -> None:
        """Finish the W&B run if one is active."""
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish(exit_code=exit_code)
        except Exception:
            pass
        self._wandb_run = None

    def log(self, record: Mapping[str, Any]) -> None:
        """Append one metrics/event record to ``metrics.jsonl`` when enabled."""
        if self.metrics_path is None and self._wandb_run is None:
            return
        payload = {str(k): json_like(v) for k, v in dict(record).items()}
        payload.setdefault("timestamp", utc_now_iso())
        if self.metrics_path is not None:
            append_jsonl(self.metrics_path, payload)
        if self._wandb_run is not None:
            try:
                wandb_payload = {k: v for k, v in payload.items() if isinstance(v, (int, float))}
                if wandb_payload:
                    self._wandb_run.log(wandb_payload)
            except Exception:
                pass

    def write_meta(self, args: Mapping[str, Any] | Any, *, task: str) -> None:
        """Write the initial ``meta.json`` lifecycle record for a run."""
        if self.meta_path is None:
            return
        public_args = public_args_dict(args)
        started_at = utc_now_iso()
        payload = {
            "schema_version": RUN_META_SCHEMA_VERSION,
            "run_id": str(self.run_id),
            "status": "running",
            "timestamp": started_at,
            "started_at": started_at,
            "completed_at": None,
            "git_commit": safe_git_commit(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "argv": list(map(str, os.sys.argv)),
            "cwd": str(Path.cwd().resolve()),
            "task": str(task),
            "metrics_schema": RUN_METRICS_SCHEMA_VERSION,
            "summary_schema": RUN_SUMMARY_SCHEMA_VERSION,
            "config_artifact": "resolved_config.yaml",
            "args_note": "Args are kept for compatibility. Use resolved_config.yaml as the canonical config artifact.",
            "checkpointing": {
                "enabled": bool(self.checkpoints_path is not None),
                "dir": None if self.checkpoints_path is None else "checkpoints",
            },
            "args": public_args,
        }
        self._meta_payload = payload
        write_json(self.meta_path, payload)
        self._init_wandb(config=public_args, name=self.run_id)

    def finalize_meta(self, status: str, *, error: str | None = None) -> None:
        """Finalize ``meta.json`` with a terminal status and optional error string."""
        if self.meta_path is None:
            return
        payload = dict(self._meta_payload or {})
        if not payload:
            completed_at = utc_now_iso()
            payload = {
                "schema_version": RUN_META_SCHEMA_VERSION,
                "run_id": str(self.run_id),
                "status": str(status),
                "started_at": completed_at,
                "completed_at": completed_at,
            }
        else:
            payload["status"] = str(status)
            payload["completed_at"] = utc_now_iso()
        if error is not None:
            payload["error"] = str(error)
        self._meta_payload = payload
        write_json(self.meta_path, payload)
        self._finish_wandb(exit_code=0 if status == "succeeded" else 1)

    def write_resolved_config(self, resolved_config: Mapping[str, Any] | Any) -> None:
        """Persist the fully resolved runtime config as ``resolved_config.yaml``."""
        if self.config_path is None:
            return
        payload = json_like(resolved_config)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def write_seeds(self, seed_state: Any) -> None:
        """Persist the effective seed/determinism state as ``seeds.json``."""
        if self.seeds_path is None:
            return
        payload = {
            "schema_version": RUN_SEEDS_SCHEMA_VERSION,
            "seed": int(getattr(seed_state, "seed", 0)),
            "determinism": str(getattr(seed_state, "mode", "relaxed")),
            "deterministic_algorithms": bool(getattr(seed_state, "deterministic_algorithms", False)),
            "cudnn_deterministic": getattr(seed_state, "cudnn_deterministic", None),
            "cudnn_benchmark": getattr(seed_state, "cudnn_benchmark", None),
            "cublas_workspace_config": getattr(seed_state, "cublas_workspace_config", None),
        }
        write_json(self.seeds_path, payload)

    def write_summary(self, summary: Mapping[str, Any], *, status: str) -> None:
        """Persist the compact on-disk summary artifact for a completed run."""
        if self.summary_path is None:
            return
        payload = build_persisted_summary(
            summary,
            run_id=self.run_id,
            status=status,
            checkpoints_dir=(None if self.checkpoints_path is None else "checkpoints"),
        )
        write_json(self.summary_path, payload)
        if self._wandb_run is not None:
            try:
                self._wandb_run.summary.update(
                    {k: json_like(v) for k, v in dict(summary).items() if k != "args"}
                )
            except Exception:
                pass

    def build_checkpoint_payload(
        self,
        *,
        task: str,
        model: Any,
        learner: Any | None = None,
        optimizer: Any | None = None,
        schedulers: list[Any] | None = None,
        epoch: int | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a device-agnostic checkpoint payload from trainer objects."""
        payload: dict[str, Any] = {
            "schema_version": RUN_CHECKPOINT_SCHEMA_VERSION,
            "run_id": str(self.run_id),
            "task": str(task),
            "epoch": None if epoch is None else int(epoch),
        }
        state_dict = getattr(model, "state_dict", None)
        if callable(state_dict):
            payload["model"] = _to_cpu_state(state_dict())
        learner_state = getattr(learner, "state_dict", None)
        if callable(learner_state):
            payload["learner"] = _to_cpu_state(learner_state())
        if optimizer is not None:
            payload["optimizer"] = _to_cpu_state(optimizer.state_dict())
        if schedulers:
            payload["schedulers"] = [
                _to_cpu_state(sched.state_dict()) if callable(getattr(sched, "state_dict", None)) else None
                for sched in list(schedulers)
            ]
        if meta:
            payload["meta"] = json_like(meta)
        return payload

    def write_checkpoint(self, name: str, payload: Mapping[str, Any]) -> None:
        """Write one named checkpoint payload into ``checkpoints/<name>.pt``."""
        if self.checkpoints_path is None:
            return
        import torch

        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        torch.save(dict(payload), self.checkpoints_path / f"{name}.pt")

    def _resolve_checkpoint_monitor(self, trainer: Any, epoch_record: Any) -> tuple[str, str] | tuple[None, None]:
        """Resolve the monitor used to decide whether a checkpoint is the new best."""
        if self._best_monitor_name is not None and self._best_monitor_mode is not None:
            return self._best_monitor_name, self._best_monitor_mode

        from .callbacks import EarlyStopping

        for callback in list(getattr(trainer, "callbacks", []) or []):
            if isinstance(callback, EarlyStopping):
                status = callback.status()
                self._best_monitor_name = str(status.name)
                self._best_monitor_mode = str(status.mode)
                return self._best_monitor_name, self._best_monitor_mode

        for scheduler in list(getattr(trainer, "schedulers", []) or []):
            monitor_name = str(getattr(scheduler, "monitor", "")).strip()
            if not monitor_name:
                continue
            raw_mode = getattr(getattr(scheduler, "scheduler", None), "mode", None)
            if str(raw_mode).strip().lower() in {"min", "max"}:
                mode = str(raw_mode).strip().lower()
            else:
                mode = _checkpoint_mode_from_name(monitor_name)
            self._best_monitor_name = monitor_name
            self._best_monitor_mode = mode
            return self._best_monitor_name, self._best_monitor_mode

        if getattr(epoch_record, "val", None) is not None:
            self._best_monitor_name = "val.loss"
            self._best_monitor_mode = "min"
        else:
            self._best_monitor_name = "train.loss"
            self._best_monitor_mode = "min"
        return self._best_monitor_name, self._best_monitor_mode

    def on_fit_start(self, trainer: Any, state: Any | None = None) -> None:
        """Reset best-checkpoint tracking at the beginning of a fit call."""
        _ = (trainer, state)
        self._best_monitor_name = None
        self._best_monitor_mode = None
        self._best_monitor_value = None
        self._best_checkpoint_payload = None

    def on_epoch_end(self, trainer: Any, epoch_record: Any, state: Any | None = None) -> None:
        """Capture a best checkpoint candidate after a completed epoch."""
        _ = state
        if self.checkpoints_path is None:
            return
        monitor_name, monitor_mode = self._resolve_checkpoint_monitor(trainer, epoch_record)
        if not monitor_name or not monitor_mode:
            return
        value = epoch_record.to_log_values().get(monitor_name)
        if not isinstance(value, (int, float)):
            return
        current_value = float(value)
        improved = self._best_monitor_value is None
        if self._best_monitor_value is not None:
            if monitor_mode == "max":
                improved = current_value > float(self._best_monitor_value)
            else:
                improved = current_value < float(self._best_monitor_value)
        if not improved:
            return
        self._best_monitor_value = current_value
        self._best_checkpoint_payload = self.build_checkpoint_payload(
            task="supervised",
            model=trainer.model,
            learner=trainer.learner,
            optimizer=getattr(trainer.learner, "optimizer", None),
            schedulers=list(getattr(trainer, "schedulers", []) or []),
            epoch=int(epoch_record.epoch),
            meta={
                "kind": "best",
                "monitor_name": monitor_name,
                "monitor_mode": monitor_mode,
                "monitor_value": current_value,
            },
        )

    def on_fit_end(self, trainer: Any, fit_result: Any, state: Any | None = None) -> None:
        """Write ``last.pt`` and ``best.pt`` checkpoint artifacts when enabled."""
        _ = state
        if self.checkpoints_path is None:
            return
        last_payload = self.build_checkpoint_payload(
            task="supervised",
            model=trainer.model,
            learner=trainer.learner,
            optimizer=getattr(trainer.learner, "optimizer", None),
            schedulers=list(getattr(trainer, "schedulers", []) or []),
            epoch=int(getattr(getattr(fit_result, "final_epoch", None), "epoch", 0) or 0),
            meta={
                "kind": "last",
                "best_epoch": getattr(getattr(fit_result, "best", None), "epoch", None),
                "restored_on_train_end": bool(getattr(getattr(fit_result, "restoration", None), "restored_on_train_end", False)),
            },
        )
        self.write_checkpoint("last", last_payload)
        if self._best_checkpoint_payload is not None:
            self.write_checkpoint("best", self._best_checkpoint_payload)
