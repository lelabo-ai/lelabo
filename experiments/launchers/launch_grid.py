# experiments/launchers/launch_grid.py
import argparse
import hashlib
import itertools
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------
# CLI helpers
# --------------------------
def flagify(key: str) -> str:
    # YAML keys like weight_decay -> --weight-decay
    return "--" + key.replace("_", "-")


def to_cli_args(d: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in d.items():
        if v is None:
            continue
        f = flagify(k)
        if isinstance(v, bool):
            if v:
                args.append(f)
            continue
        args += [f, str(v)]
    return args


def cartesian_grid(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos: List[Dict[str, Any]] = []
    for prod in itertools.product(*vals):
        combos.append(dict(zip(keys, prod)))
    return combos


def resolve_config_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    if not p.is_absolute():
        p_repo = REPO_ROOT / p
        if p_repo.exists():
            return p_repo
    return p


# --------------------------
# Run naming helpers
# --------------------------
def sanitize(val: Any) -> str:
    """
    Make values filesystem-friendly.
    Keep it readable (no heavy encoding), just remove annoying chars.
    """
    s = str(val)
    s = s.replace("/", "-").replace("\\", "-")
    s = s.replace(" ", "")
    s = s.replace(":", "-")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "_")
    return s


def stable_short_id(args_dict: Dict[str, Any], n: int = 6) -> str:
    """
    Stable ID derived from the full args dict.
    Same args -> same id. Useful for collision avoidance.
    """
    blob = json.dumps(args_dict, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:n]


def build_run_dirname(args_dict: Dict[str, Any], display_keys: List[str]) -> str:
    """
    Human-readable run folder name + stable short id.
    Example:
      algo=kp__dataset=mnist__model=mlp__lr=0.001__seed=3__id=ab12cd
    """
    parts: List[str] = []
    for k in display_keys:
        if k not in args_dict:
            continue
        parts.append(f"{k}={sanitize(args_dict[k])}")
    rid = stable_short_id(args_dict)
    parts.append(f"id={rid}")
    return "__".join(parts)


# --------------------------
# Job + execution
# --------------------------
@dataclass
class Job:
    idx: int
    args: Dict[str, Any]
    cmd: List[str]
    job_dir: Path
    log_path: Path
    meta_path: Path


def run_job(job: Job, env: Dict[str, str] | None = None) -> int:
    job.job_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata (repro)
    payload = {
        "cmd": job.cmd,
        "args": job.args,
        "job_idx": job.idx,
        "cwd": str(Path.cwd()),
    }
    job.meta_path.write_text(json.dumps(payload, indent=2))

    with job.log_path.open("w") as f:
        f.write(" ".join(job.cmd) + "\n\n")
        f.flush()
        p = subprocess.run(
            job.cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=REPO_ROOT,
        )
        return p.returncode


def with_src_on_pythonpath(env: Dict[str, str]) -> Dict[str, str]:
    out = dict(env)
    src_path = str(REPO_ROOT / "src")
    cur = out.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if src_path not in parts:
        parts.insert(0, src_path)
    out["PYTHONPATH"] = os.pathsep.join(parts)
    # Keep caches at repo root rather than under src/ or random cwd.
    out.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config describing base args + grid.")
    ap.add_argument("--python", type=str, default="python", help="Python executable")
    ap.add_argument("--entry", type=str, default="lelabo.cli.main", help="Python module executed via `python -m`.")
    ap.add_argument("--outdir", type=str, default="outputs/runs", help="Where logs/metadata are written")
    ap.add_argument("--name", type=str, default=None, help="Experiment name (default: timestamp)")
    ap.add_argument("--max-parallel", type=int, default=1, help="Number of concurrent runs")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    ap.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Optional GPU ids, e.g. '0,1'. Round-robin via CUDA_VISIBLE_DEVICES.",
    )
    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")

    config_path = resolve_config_path(args.config)
    cfg = yaml.safe_load(config_path.read_text())

    base: Dict[str, Any] = cfg.get("base", {})
    grid: Dict[str, List[Any]] = cfg.get("grid", {})

    # Which args to show in run folder names (human-friendly)
    display_keys: List[str] = cfg.get(
        "display_keys",
        ["algo", "dataset", "task", "model", "lr", "wd", "weight_decay", "seed"],
    )

    exp_name = args.name or cfg.get("name") or datetime.now().strftime("%Y%m%d_%H%M%S")
    combos = cartesian_grid(grid)
    out_root = Path(args.outdir) / exp_name

    jobs: List[Job] = []
    for i, combo in enumerate(combos):
        merged = dict(base)
        merged.update(combo)

        run_dirname = build_run_dirname(merged, display_keys)
        job_dir = out_root / run_dirname
        cmd = [args.python, "-m", args.entry]
        if args.entry == "lelabo.cli.main":
            cmd.append("train")
        cmd += to_cli_args(merged) + ["--run-dir", str(job_dir)]


        jobs.append(
            Job(
                idx=i,
                args=merged,
                cmd=cmd,
                job_dir=job_dir,
                log_path=job_dir / "stdout.log",
                meta_path=job_dir / "meta.json",
            )
        )

    # Save the full plan (helps reproducibility)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "plan.json").write_text(
        json.dumps(
            [{"idx": j.idx, "cmd": j.cmd, "args": j.args, "dir": str(j.job_dir)} for j in jobs],
            indent=2,
        )
    )

    if args.dry_run:
        for j in jobs:
            print(" ".join(j.cmd))
        print(f"\n{len(jobs)} commands generated. (dry-run)")
        print(f"Run directories would be under: {out_root}")
        return

    max_parallel = max(1, int(args.max_parallel))

    gpu_list: List[str] = []
    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]

    # Single-threaded
    if max_parallel == 1:
        fail_count = 0
        for j in jobs:
            env = with_src_on_pythonpath(os.environ.copy())
            if gpu_list:
                env["CUDA_VISIBLE_DEVICES"] = gpu_list[j.idx % len(gpu_list)]
            code = run_job(j, env=env)
            if code != 0:
                fail_count += 1
                print(f"[FAIL] job {j.idx} (exit={code}) -> {j.log_path}")
        print(f"Done. Logs in: {out_root} | failed: {fail_count}/{len(jobs)}")
        return

    # Parallel execution: thread pool (best for launching subprocesses; no pickling)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def submit_with_env(job: Job) -> Tuple[int, int]:
        env = with_src_on_pythonpath(os.environ.copy())
        if gpu_list:
            env["CUDA_VISIBLE_DEVICES"] = gpu_list[job.idx % len(gpu_list)]
        return job.idx, run_job(job, env=env)

    fail_count = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futs = [ex.submit(submit_with_env, j) for j in jobs]
        for fut in as_completed(futs):
            idx, code = fut.result()
            if code != 0:
                fail_count += 1
                print(f"[FAIL] job {idx} (exit={code}) -> {jobs[idx].log_path}")

    print(f"Done. Logs in: {out_root} | failed: {fail_count}/{len(jobs)}")


if __name__ == "__main__":
    main()
