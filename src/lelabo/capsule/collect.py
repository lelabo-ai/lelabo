from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .schema import CAPSULE_SCHEMA_VERSION, validate_manifest


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _capture_git_info(source: Path) -> dict[str, Any]:
    cwd = source if source.is_dir() else source.parent

    def _run(cmd: list[str]) -> str | None:
        try:
            out = subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.DEVNULL)
            return out.decode("utf-8", errors="replace").strip()
        except Exception:
            return None

    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    patch = _run(["git", "diff"])

    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(dirty),
        "status_porcelain": dirty,
        "patch": patch,
    }


def _capture_env_snapshot() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "argv": list(map(str, sys.argv)),
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }

    try:
        import torch  # type: ignore

        info["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())
            if hasattr(torch, "are_deterministic_algorithms_enabled")
            else None,
        }
    except Exception as exc:
        info["torch_error"] = str(exc)

    return info


def _capture_pip_freeze() -> str:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _guess_kind(source: Path) -> str:
    if source.is_file():
        return "config_only"
    if (source / "plan.json").exists():
        return "sweep"
    if (source / "meta.json").exists() or (source / "summary.json").exists():
        return "single_run"
    return "config_only"


def _entrypoint_from_meta(meta: dict[str, Any] | None) -> list[str] | None:
    if not meta:
        return None
    argv = meta.get("argv")
    if isinstance(argv, list) and argv:
        return [str(x) for x in argv]
    return None


def _collect_single_run(source: Path, stage: Path, artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    logs_dir = stage / "logs"
    art_dir = stage / "artifacts"

    copied: list[str] = []
    for name in ("meta.json", "summary.json", "metrics.jsonl", "resolved_config.yaml", "seeds.json"):
        src = source / name
        dst = logs_dir / name if name.endswith((".json", ".jsonl")) else stage / name
        if name in {"resolved_config.yaml", "seeds.json"}:
            dst = stage / name
        if _copy_if_exists(src, dst):
            copied.append(dst.relative_to(stage).as_posix())

    # Checkpoints if present. Accept legacy `ckpts/`, but normalize to `checkpoints/`.
    for dname in ("checkpoints", "ckpts"):
        dsrc = source / dname
        if dsrc.exists() and dsrc.is_dir():
            ddst = art_dir / "checkpoints"
            shutil.copytree(dsrc, ddst, dirs_exist_ok=True)
            copied.append(ddst.relative_to(stage).as_posix())

    meta = _safe_read_json(source / "meta.json")
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    if isinstance(args, dict):
        resolved_cfg = stage / "resolved_config.yaml"
        if not resolved_cfg.exists():
            # JSON is valid YAML, so this stays parseable as YAML while avoiding extra deps.
            resolved_cfg.write_text(json.dumps(args, indent=2, ensure_ascii=False), encoding="utf-8")
            copied.append(resolved_cfg.relative_to(stage).as_posix())

        seeds_path = stage / "seeds.json"
        if not seeds_path.exists():
            seed_payload: dict[str, Any] = {}
            if "seed" in args:
                seed_payload["seed"] = args.get("seed")
            if "determinism" in args:
                seed_payload["determinism"] = args.get("determinism")
            if seed_payload:
                seeds_path.write_text(json.dumps(seed_payload, indent=2, ensure_ascii=False), encoding="utf-8")
                copied.append(seeds_path.relative_to(stage).as_posix())

        datasets_path = stage / "data" / "datasets.json"
        datasets_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_payload = {
            "dataset": args.get("dataset"),
            "glue_task": args.get("glue_task"),
            "hf_model": args.get("hf_model"),
            "note": "Add revision/hash from dataset builders for stronger provenance.",
        }
        datasets_path.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        copied.append(datasets_path.relative_to(stage).as_posix())

    artifacts["single_run_files"] = copied

    entry = _entrypoint_from_meta(meta)
    if entry:
        return [{"name": "run", "cmd": entry}]
    return []


def _collect_sweep(source: Path, stage: Path, artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    copied: list[str] = []
    for fname in ("plan.json",):
        if _copy_if_exists(source / fname, stage / fname):
            copied.append(fname)

    runs_dst = stage / "runs"
    runs_dst.mkdir(parents=True, exist_ok=True)

    entrypoints: list[dict[str, Any]] = []
    for child in sorted(source.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if not (child / "meta.json").exists() and not (child / "summary.json").exists():
            continue

        out = runs_dst / child.name
        out.mkdir(parents=True, exist_ok=True)
        for name in ("meta.json", "summary.json", "metrics.jsonl", "resolved_config.yaml", "seeds.json"):
            _copy_if_exists(child / name, out / name)

        for dname in ("checkpoints", "ckpts"):
            dsrc = child / dname
            if dsrc.exists() and dsrc.is_dir():
                shutil.copytree(dsrc, out / "checkpoints", dirs_exist_ok=True)

        meta = _safe_read_json(child / "meta.json")
        cmd = _entrypoint_from_meta(meta)
        if cmd:
            entrypoints.append({"name": child.name, "cmd": cmd})

    copied.append("runs")
    artifacts["sweep_files"] = copied
    return entrypoints


def _collect_config_only(source: Path, stage: Path, artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    if source.is_file():
        cfg_dst = stage / "resolved_config.yaml"
        cfg_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, cfg_dst)
        artifacts["config"] = "resolved_config.yaml"
    return []


def collect_capsule(
    *,
    source: Path,
    stage_dir: Path,
    capsule_id: str,
    include_code_snapshot: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    stage = stage_dir.resolve()
    stage.mkdir(parents=True, exist_ok=True)

    kind = _guess_kind(source)
    artifacts: dict[str, Any] = {}

    entrypoints: list[dict[str, Any]]
    if kind == "single_run":
        entrypoints = _collect_single_run(source, stage, artifacts)
    elif kind == "sweep":
        entrypoints = _collect_sweep(source, stage, artifacts)
    else:
        entrypoints = _collect_config_only(source, stage, artifacts)

    git_info = _capture_git_info(source)
    env_info = _capture_env_snapshot()
    deps_snapshot = _capture_pip_freeze()

    code_dir = stage / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "git.json").write_text(json.dumps(git_info, indent=2, ensure_ascii=False), encoding="utf-8")
    if git_info.get("patch"):
        (code_dir / "dirty.patch").write_text(str(git_info["patch"]), encoding="utf-8")

    env_dir = stage / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "system.json").write_text(json.dumps(env_info, indent=2, ensure_ascii=False), encoding="utf-8")
    (env_dir / "deps_snapshot.txt").write_text(deps_snapshot, encoding="utf-8")

    data_dir = stage / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets_file = data_dir / "datasets.json"
    if not datasets_file.exists():
        dataset_stub = {
            "note": "Dataset provenance should be completed by dataset builders.",
            "source_path": str(source),
        }
        datasets_file.write_text(json.dumps(dataset_stub, indent=2, ensure_ascii=False), encoding="utf-8")

    if include_code_snapshot:
        repo_root = Path.cwd().resolve()
        src_dir = repo_root / "src"
        if src_dir.exists():
            dst = stage / "code_snapshot" / "src"
            shutil.copytree(src_dir, dst, dirs_exist_ok=True)
            artifacts["code_snapshot"] = "code_snapshot/src"

    replay_cmd: list[str] | None = None
    if entrypoints:
        replay_cmd = [str(x) for x in entrypoints[0]["cmd"]]

    manifest = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "capsule_id": capsule_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "source": {
            "path": str(source),
            "type": "file" if source.is_file() else "directory",
        },
        "entrypoints": entrypoints,
        "artifacts": artifacts,
        "replay": {
            "command": replay_cmd,
            "cwd": str(Path.cwd().resolve()),
            "tolerance_profile": "relaxed",
        },
    }
    validate_manifest(manifest)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if replay_cmd is not None:
        replay_dir = stage / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_spec = {
            "command": replay_cmd,
            "cwd": str(Path.cwd().resolve()),
            "tolerance_profile": "relaxed",
        }
        (replay_dir / "replay_spec.json").write_text(
            json.dumps(replay_spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return manifest
