"""Sweep job execution: build, run, and manage grid sweep jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .grid import build_run_dirname, cartesian_grid, to_cli_args


_DEFAULT_DISPLAY_KEYS = [
    "algo", "rule", "dataset", "task", "model", "lr", "wd", "weight_decay", "seed",
]


@dataclass
class Job:
    idx: int
    args: dict[str, Any]
    cmd: list[str]
    job_dir: Path
    log_path: Path
    meta_path: Path
    status: str = "pending"


@dataclass
class SweepPlan:
    name: str
    jobs: list[Job]
    out_root: Path
    gpu_list: list[str] = field(default_factory=list)

    def save(self) -> Path:
        """Write plan.json to the output root."""
        self.out_root.mkdir(parents=True, exist_ok=True)
        plan_path = self.out_root / "plan.json"
        payload = [
            {"idx": j.idx, "cmd": j.cmd, "args": j.args, "dir": str(j.job_dir)}
            for j in self.jobs
        ]
        plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return plan_path


def load_sweep_config(config_path: Path) -> dict[str, Any]:
    """Load and validate a sweep YAML config file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Sweep config must be a YAML mapping: {config_path}")
    if "grid" not in cfg:
        raise ValueError(f"Sweep config must contain a 'grid' section: {config_path}")
    return cfg


def build_sweep_plan(
    *,
    config: dict[str, Any],
    outdir: Path,
    name: str | None = None,
    python: str | None = None,
    entry: str = "lelabo.cli.main",
    gpus: str | None = None,
    wandb_group: str | None = None,
) -> SweepPlan:
    """Build a SweepPlan from a parsed sweep config dict."""
    base: dict[str, Any] = dict(config.get("base", {}))
    grid: dict[str, list[Any]] = dict(config.get("grid", {}))
    display_keys: list[str] = config.get("display_keys", _DEFAULT_DISPLAY_KEYS)

    exp_name = name or config.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    combos = cartesian_grid(grid)
    out_root = outdir / exp_name

    gpu_list: list[str] = []
    if gpus:
        gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]

    py = python or sys.executable
    group = wandb_group or exp_name

    jobs: list[Job] = []
    for i, combo in enumerate(combos):
        merged = dict(base)
        merged.update(combo)

        run_dirname = build_run_dirname(merged, display_keys)
        job_dir = out_root / run_dirname
        cmd = [py, "-m", entry]
        if entry == "lelabo.cli.main":
            cmd.append("train")
        cmd += to_cli_args(merged) + ["--run-dir", str(job_dir)]
        cmd += ["--set", f"wandb.group={group}"]

        jobs.append(Job(
            idx=i,
            args=merged,
            cmd=cmd,
            job_dir=job_dir,
            log_path=job_dir / "stdout.log",
            meta_path=job_dir / "meta.json",
        ))

    return SweepPlan(
        name=exp_name,
        jobs=jobs,
        out_root=out_root,
        gpu_list=gpu_list,
    )


def _with_src_on_pythonpath(env: dict[str, str]) -> dict[str, str]:
    """Ensure src/ is on PYTHONPATH for subprocess jobs."""
    out = dict(env)
    try:
        import lelabo
        src_path = str(Path(lelabo.__file__).resolve().parents[1])
    except Exception:
        src_path = str(Path.cwd() / "src")
    cur = out.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if src_path not in parts:
        parts.insert(0, src_path)
    out["PYTHONPATH"] = os.pathsep.join(parts)
    return out


def run_job(job: Job, *, env: dict[str, str] | None = None) -> int:
    """Execute a single sweep job as a subprocess."""
    job.job_dir.mkdir(parents=True, exist_ok=True)
    job.status = "running"

    payload = {
        "cmd": job.cmd,
        "args": job.args,
        "job_idx": job.idx,
        "cwd": str(Path.cwd()),
    }
    job.meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with job.log_path.open("w") as f:
        f.write(" ".join(job.cmd) + "\n\n")
        f.flush()
        p = subprocess.run(
            job.cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
    job.status = "succeeded" if p.returncode == 0 else "failed"
    return p.returncode


def run_sweep(
    plan: SweepPlan,
    *,
    max_parallel: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute all jobs in a sweep plan."""
    plan.save()

    if dry_run:
        for j in plan.jobs:
            print(" ".join(j.cmd))
        print(f"\n{len(plan.jobs)} commands generated. (dry-run)")
        print(f"Run directories would be under: {plan.out_root}")
        return {"total": len(plan.jobs), "failed": 0, "dry_run": True}

    max_parallel = max(1, max_parallel)

    if max_parallel == 1:
        fail_count = 0
        for j in plan.jobs:
            env = _with_src_on_pythonpath(os.environ.copy())
            if plan.gpu_list:
                env["CUDA_VISIBLE_DEVICES"] = plan.gpu_list[j.idx % len(plan.gpu_list)]
            code = run_job(j, env=env)
            if code != 0:
                fail_count += 1
                print(f"[FAIL] job {j.idx} (exit={code}) -> {j.log_path}")
            else:
                print(f"[OK]   job {j.idx} -> {j.job_dir}")
        print(f"Done. Logs in: {plan.out_root} | failed: {fail_count}/{len(plan.jobs)}")
        return {"total": len(plan.jobs), "failed": fail_count, "dry_run": False}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _submit(job: Job) -> tuple[int, int]:
        env = _with_src_on_pythonpath(os.environ.copy())
        if plan.gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = plan.gpu_list[job.idx % len(plan.gpu_list)]
        return job.idx, run_job(job, env=env)

    fail_count = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(_submit, j) for j in plan.jobs]
        for fut in as_completed(futs):
            idx, code = fut.result()
            if code != 0:
                fail_count += 1
                print(f"[FAIL] job {idx} (exit={code}) -> {plan.jobs[idx].log_path}")
            else:
                print(f"[OK]   job {idx} -> {plan.jobs[idx].job_dir}")

    print(f"Done. Logs in: {plan.out_root} | failed: {fail_count}/{len(plan.jobs)}")
    return {"total": len(plan.jobs), "failed": fail_count, "dry_run": False}
