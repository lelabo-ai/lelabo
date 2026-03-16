"""Replay helpers for capsules that embed runnable commands and artifacts."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Sequence

from .registry import get_capsule


def _load_manifest(capsule_path: Path) -> dict:
    p = capsule_path / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"manifest.json not found in {capsule_path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_command(manifest: dict) -> list[str]:
    replay = manifest.get("replay", {}) if isinstance(manifest, dict) else {}
    cmd = replay.get("command")
    if isinstance(cmd, list) and cmd:
        return [str(x) for x in cmd]

    eps = manifest.get("entrypoints", [])
    if isinstance(eps, list) and eps:
        first = eps[0]
        if isinstance(first, dict) and isinstance(first.get("cmd"), list) and first["cmd"]:
            return [str(x) for x in first["cmd"]]

    raise ValueError("No replay command found in capsule manifest.")


def _build_env(capsule_path: Path, base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    snap = capsule_path / "code_snapshot" / "src"
    if snap.exists():
        cur = env.get("PYTHONPATH", "")
        parts = [str(snap)]
        if cur:
            parts.append(cur)
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _run_current(cmd: Sequence[str], cwd: Path, env: dict[str, str]) -> int:
    return subprocess.call(list(cmd), cwd=str(cwd), env=env)


def _run_in_venv(cmd: Sequence[str], cwd: Path, env: dict[str, str], capsule_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="lelabo_capsule_venv_") as td:
        venv_dir = Path(td) / "venv"
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir))

        py = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"
        deps = capsule_path / "env" / "deps_snapshot.txt"
        if deps.exists() and deps.read_text(encoding="utf-8").strip():
            subprocess.call([str(py), "-m", "pip", "install", "-r", str(deps)], cwd=str(cwd), env=env)

        final = [str(py), *list(cmd)]
        return subprocess.call(final, cwd=str(cwd), env=env)


def rerun_capsule(
    *,
    capsule_or_alias: str,
    capsules_dir: Path | None = None,
    env_mode: str = "current",
    extra_args: Sequence[str] | None = None,
) -> int:
    entry = get_capsule(capsule_or_alias, capsules_dir)
    if entry is None:
        raise ValueError(f"Unknown capsule '{capsule_or_alias}'.")

    capsule_path = Path(entry["path"]).resolve()
    manifest = _load_manifest(capsule_path)

    cmd = _resolve_command(manifest)
    if extra_args:
        cmd = [*cmd, *list(extra_args)]

    cwd = Path(manifest.get("replay", {}).get("cwd", Path.cwd())).resolve()
    env = _build_env(capsule_path, os.environ)

    if env_mode == "venv":
        return _run_in_venv(cmd, cwd, env, capsule_path)
    if env_mode != "current":
        raise ValueError("env_mode must be 'current' or 'venv'.")
    return _run_current(cmd, cwd, env)
