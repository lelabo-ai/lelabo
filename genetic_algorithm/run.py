from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bootstrap import ensure_src_on_path

ensure_src_on_path()

from .config import RuntimeConfig, load_runtime_config
from .evaluator import SupervisedEvaluator
from .exporter import create_run_dir, export_results
from .ga import GeneticSearch
from .search_space import SearchSpace


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone GA search for update-rules.")
    p.add_argument("--config", required=True, type=str, help="Path to a genetic_algorithm YAML config.")
    p.add_argument("--output-dir", type=str, default=None, help="Override output_dir from config.")
    p.add_argument(
        "--max-evals",
        type=str,
        default=None,
        help="Override ga.max_evals (int or max/none).",
    )
    p.add_argument("--workers", type=int, default=None, help="Override ga.workers (0=auto, 1=sequential).")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    p.add_argument("--eval-repeats", type=int, default=None, help="Override ga.eval_repeats.")
    p.add_argument("--eval-seed-stride", type=int, default=None, help="Override ga.eval_seed_stride.")
    return p.parse_args()


def _override(cfg: RuntimeConfig, args: argparse.Namespace) -> RuntimeConfig:
    out_dir = str(args.output_dir) if args.output_dir else cfg.output_dir
    max_evals = _parse_max_evals_override(args.max_evals, cfg.ga.max_evals)
    workers = int(args.workers) if args.workers is not None else cfg.ga.workers
    show_progress = False if bool(args.no_progress) else cfg.ga.show_progress
    eval_repeats = int(args.eval_repeats) if args.eval_repeats is not None else cfg.ga.eval_repeats
    eval_seed_stride = int(args.eval_seed_stride) if args.eval_seed_stride is not None else cfg.ga.eval_seed_stride
    ga = cfg.ga.__class__(
        population_size=cfg.ga.population_size,
        generations=cfg.ga.generations,
        elite_size=cfg.ga.elite_size,
        tournament_size=cfg.ga.tournament_size,
        mutation_rate=cfg.ga.mutation_rate,
        crossover_rate=cfg.ga.crossover_rate,
        top_k=cfg.ga.top_k,
        max_evals=max_evals,
        workers=workers,
        show_progress=show_progress,
        eval_repeats=eval_repeats,
        eval_seed_stride=eval_seed_stride,
        memetic_enabled=cfg.ga.memetic_enabled,
        memetic_score_tol=cfg.ga.memetic_score_tol,
        memetic_immigrant_rate=cfg.ga.memetic_immigrant_rate,
        initial_pool_enabled=cfg.ga.initial_pool_enabled,
        initial_pool_path=cfg.ga.initial_pool_path,
        initial_pool_max_items=cfg.ga.initial_pool_max_items,
        initial_pool_strict=cfg.ga.initial_pool_strict,
    )
    return RuntimeConfig(
        name=cfg.name,
        seed=cfg.seed,
        device=cfg.device,
        output_dir=out_dir,
        fixed=cfg.fixed,
        search_space=cfg.search_space,
        fitness=cfg.fitness,
        ga=ga,
    )


def _parse_max_evals_override(raw: str | None, fallback: int | None) -> int | None:
    if raw is None:
        return fallback
    val = str(raw).strip().lower()
    if val in {"", "none", "null", "max", "unlimited", "inf", "infinite"}:
        return None
    return int(val)


def main() -> int:
    args = _parse_args()
    cfg = load_runtime_config(args.config)
    cfg = _override(cfg, args)

    run_dir = create_run_dir(base_output_dir=cfg.output_dir, run_name=cfg.name)
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": cfg.name,
                "seed": cfg.seed,
                "device": cfg.device,
                "output_dir": cfg.output_dir,
                "fixed": cfg.fixed,
                "search_space": cfg.search_space,
                "fitness": cfg.fitness.__dict__,
                "ga": cfg.ga.__dict__,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    search_space = SearchSpace(cfg.search_space)
    evaluator = SupervisedEvaluator(
        fixed_params=cfg.fixed,
        objective=cfg.fitness.objective,
        fitness_split=cfg.fitness.split,
        device=cfg.device,
        eval_repeats=cfg.ga.eval_repeats,
        eval_seed_stride=cfg.ga.eval_seed_stride,
    )
    initial_pool = _load_initial_pool(cfg)
    engine = GeneticSearch(
        search_space=search_space,
        evaluator=evaluator,
        settings=cfg.ga,
        seed=cfg.seed,
        initial_pool=initial_pool,
    )

    archive, history = engine.run()
    winners = _unique_top_k(archive, k=cfg.ga.top_k)

    exported = export_results(
        run_dir=run_dir,
        runtime_cfg=cfg,
        winners=winners,
        history=history,
        total_evaluations=engine.evaluation_count,
    )

    print(f"[GA] run_dir={run_dir}")
    print(f"[GA] workers={engine.workers} backend={engine.parallel_backend}")
    print(f"[GA] eval_repeats={cfg.ga.eval_repeats} seed_stride={cfg.ga.eval_seed_stride}")
    print(
        "[GA] selection=classic_rtr "
        f"prune_per_generation=false "
        f"immigrant_rate={cfg.ga.memetic_immigrant_rate} "
        f"(legacy_memetic_enabled={cfg.ga.memetic_enabled})"
    )
    print(
        f"[GA] initial_pool_enabled={cfg.ga.initial_pool_enabled} "
        f"path={cfg.ga.initial_pool_path!r} "
        f"loaded={len(initial_pool)} "
        f"max_items={cfg.ga.initial_pool_max_items}"
    )
    print(f"[GA] evaluations={engine.evaluation_count}")
    print(f"[GA] winners={len(winners)}")
    for path in exported:
        print(f"[GA] exported={path}")

    if not winners:
        print("[GA] No successful individual found. Check manifest.json for errors.")
        return 2
    return 0


def _unique_top_k(archive, *, k: int):
    out = []
    seen = set()
    for item in archive:
        if not item.result.ok:
            continue
        fp = json.dumps(item.genome, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        if fp in seen:
            continue
        seen.add(fp)
        out.append(item)
        if len(out) >= int(k):
            break
    return out


def _load_initial_pool(cfg: RuntimeConfig) -> list[dict[str, Any]]:
    if not bool(cfg.ga.initial_pool_enabled):
        return []

    path_raw = cfg.ga.initial_pool_path
    if not path_raw:
        return []
    path = Path(path_raw)
    if not path.exists():
        msg = f"Initial pool file not found: {path}"
        if cfg.ga.initial_pool_strict:
            raise FileNotFoundError(msg)
        print(f"[GA] WARN {msg}")
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    items: Any = data
    if isinstance(data, dict):
        if isinstance(data.get("pool"), list):
            items = data["pool"]
        elif isinstance(data.get("rules"), list):
            items = data["rules"]
        else:
            items = []

    if not isinstance(items, list):
        msg = f"Invalid initial pool format in {path}: expected list or object with `pool`/`rules` list."
        if cfg.ga.initial_pool_strict:
            raise ValueError(msg)
        print(f"[GA] WARN {msg}")
        return []

    out: list[dict[str, Any]] = []
    max_items = int(cfg.ga.initial_pool_max_items)
    for idx, raw in enumerate(items):
        if max_items > 0 and len(out) >= max_items:
            break
        if isinstance(raw, dict):
            out.append(raw)
        else:
            msg = f"Skipping invalid initial_pool item at index {idx}: expected object/dict."
            if cfg.ga.initial_pool_strict:
                raise ValueError(msg)
            print(f"[GA] WARN {msg}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
