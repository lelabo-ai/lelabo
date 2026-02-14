from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FIXED_PARAMS: dict[str, Any] = {
    "dataset": "iris",
    "model": "mlp",
    "hidden": 256,
    "layers": 5,
    "epochs": 50,
    "batch": 32,
    "val_frac": 0.2,
    "seed": 0,
    "verbose": 0,
    "robustness": "none",
    "input_noise_training": 0.0,
    "input_noise_dataset": 0.0,
    "noise_on_test": 0,
}

LOCKED_ARCH: dict[str, Any] = {
    "model": "mlp",
}


@dataclass(frozen=True)
class FitnessConfig:
    objective: str
    split: str
    final_splits: tuple[str, ...]


@dataclass(frozen=True)
class GASettings:
    population_size: int
    generations: int
    elite_size: int
    tournament_size: int
    mutation_rate: float
    crossover_rate: float
    top_k: int
    max_evals: int | None
    workers: int
    show_progress: bool
    eval_repeats: int
    eval_seed_stride: int
    memetic_enabled: bool
    memetic_score_tol: float
    memetic_immigrant_rate: float
    initial_pool_enabled: bool
    initial_pool_path: str | None
    initial_pool_max_items: int
    initial_pool_strict: bool


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    seed: int
    device: str
    output_dir: str
    fixed: dict[str, Any]
    search_space: dict[str, Any]
    fitness: FitnessConfig
    ga: GASettings


def _normalize_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            new_key = str(key).replace("-", "_")
            out[new_key] = _normalize_keys(value)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(v) for v in obj]
    return obj


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required for genetic_algorithm configs. Install: pip install pyyaml") from exc

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict.")
    return _normalize_keys(data)


def _as_tuple_of_str(values: list[Any] | tuple[Any, ...] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return default
    out = tuple(str(v).strip().lower() for v in values if str(v).strip())
    return out or default


def _parse_max_evals(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"", "none", "null", "max", "unlimited", "inf", "infinite"}:
            return None
        return int(v)
    return int(value)


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    raw = _read_yaml(Path(path))

    fixed = dict(DEFAULT_FIXED_PARAMS)
    fixed.update(raw.get("fixed", {}))

    search_space = raw.get("search_space", {})
    if not isinstance(search_space, dict):
        raise ValueError("`search_space` must be a dict.")

    for key, value in LOCKED_ARCH.items():
        if key in fixed and fixed[key] != value:
            raise ValueError(
                f"Architecture is locked for GA mode: fixed.{key} must be {value!r}, got {fixed[key]!r}."
            )
        fixed[key] = value

    forbidden = sorted(k for k in LOCKED_ARCH if k in search_space)
    if forbidden:
        raise ValueError(
            "search_space cannot include architecture keys in this mode: "
            + ", ".join(forbidden)
        )

    fitness_raw = raw.get("fitness", {})
    fitness = FitnessConfig(
        objective=str(fitness_raw.get("objective", "accuracy")).strip().lower(),
        split=str(fitness_raw.get("split", "val")).strip().lower(),
        final_splits=_as_tuple_of_str(fitness_raw.get("final_splits"), default=("train", "test")),
    )
    if fitness.objective not in {"accuracy", "loss", "bp_cosine_epoch", "bp_sign_match_epoch"}:
        raise ValueError(
            "fitness.objective must be one of: accuracy, loss, bp_cosine_epoch, bp_sign_match_epoch"
        )
    if fitness.split not in {"train", "val", "test"}:
        raise ValueError("fitness.split must be one of: train, val, test")

    ga_raw = raw.get("ga", {})
    memetic_raw = ga_raw.get("memetic", {})
    if memetic_raw is None:
        memetic_raw = {}
    if not isinstance(memetic_raw, dict):
        raise ValueError("ga.memetic must be a dict when provided.")
    initial_pool_raw = ga_raw.get("initial_pool", {})
    if initial_pool_raw is None:
        initial_pool_raw = {}
    if not isinstance(initial_pool_raw, dict):
        raise ValueError("ga.initial_pool must be a dict when provided.")
    ga = GASettings(
        population_size=int(ga_raw.get("population_size", 12)),
        generations=int(ga_raw.get("generations", 8)),
        elite_size=int(ga_raw.get("elite_size", 2)),
        tournament_size=int(ga_raw.get("tournament_size", 3)),
        mutation_rate=float(ga_raw.get("mutation_rate", 0.2)),
        crossover_rate=float(ga_raw.get("crossover_rate", 0.7)),
        top_k=int(ga_raw.get("top_k", 3)),
        max_evals=_parse_max_evals(ga_raw.get("max_evals")),
        workers=int(ga_raw.get("workers", 1)),
        show_progress=bool(ga_raw.get("show_progress", True)),
        eval_repeats=int(ga_raw.get("eval_repeats", 1)),
        eval_seed_stride=int(ga_raw.get("eval_seed_stride", 1)),
        memetic_enabled=bool(ga_raw.get("memetic_enabled", memetic_raw.get("enabled", False))),
        memetic_score_tol=float(ga_raw.get("memetic_score_tol", memetic_raw.get("score_tol", 0.005))),
        memetic_immigrant_rate=float(
            ga_raw.get("memetic_immigrant_rate", memetic_raw.get("immigrant_rate", 0.15))
        ),
        initial_pool_enabled=bool(
            ga_raw.get("initial_pool_enabled", initial_pool_raw.get("enabled", False))
        ),
        initial_pool_path=(
            str(ga_raw.get("initial_pool_path", initial_pool_raw.get("path")))
            if ga_raw.get("initial_pool_path", initial_pool_raw.get("path")) is not None
            else None
        ),
        initial_pool_max_items=int(
            ga_raw.get("initial_pool_max_items", initial_pool_raw.get("max_items", 0))
        ),
        initial_pool_strict=bool(
            ga_raw.get("initial_pool_strict", initial_pool_raw.get("strict", False))
        ),
    )

    if ga.population_size < 2:
        raise ValueError("ga.population_size must be >= 2")
    if ga.generations < 1:
        raise ValueError("ga.generations must be >= 1")
    if ga.elite_size < 0 or ga.elite_size > ga.population_size:
        raise ValueError("ga.elite_size must be between 0 and population_size")
    if ga.top_k < 1:
        raise ValueError("ga.top_k must be >= 1")
    if ga.workers < 0:
        raise ValueError("ga.workers must be >= 0")
    if ga.eval_repeats < 1:
        raise ValueError("ga.eval_repeats must be >= 1")
    if ga.eval_seed_stride < 1:
        raise ValueError("ga.eval_seed_stride must be >= 1")
    if ga.memetic_score_tol < 0.0:
        raise ValueError("ga.memetic_score_tol must be >= 0")
    if not (0.0 <= ga.memetic_immigrant_rate <= 1.0):
        raise ValueError("ga.memetic_immigrant_rate must be in [0,1]")
    if ga.initial_pool_max_items < 0:
        raise ValueError("ga.initial_pool_max_items must be >= 0")

    return RuntimeConfig(
        name=str(raw.get("name", "genetic_search")),
        seed=int(raw.get("seed", 0)),
        device=str(raw.get("device", "cpu")),
        output_dir=str(raw.get("output_dir", "genetic_algorithm/outputs")),
        fixed=fixed,
        search_space=search_space,
        fitness=fitness,
        ga=ga,
    )
