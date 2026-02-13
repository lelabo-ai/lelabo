from __future__ import annotations

import json
import sys
from pathlib import Path

from conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genetic_algorithm.config import load_runtime_config
from genetic_algorithm.evaluator import EvaluationResult
from genetic_algorithm.evolved_rule import EvolvedRuleParams, render_generated_rule_source
from genetic_algorithm.run import _load_initial_pool
from genetic_algorithm.search_space import SearchSpace


def test_load_runtime_config_and_search_space_sampling() -> None:
    cfg = load_runtime_config("genetic_algorithm/configs/iris_ga.yaml")
    assert cfg.name == "iris_ga_local_rule"
    assert cfg.fitness.objective in {"accuracy", "loss"}
    assert cfg.fixed["model"] == "mlp"
    assert int(cfg.fixed["hidden"]) > 0
    assert int(cfg.fixed["layers"]) >= 1
    assert cfg.ga.eval_repeats >= 1
    assert isinstance(cfg.ga.initial_pool_enabled, bool)
    assert cfg.ga.initial_pool_max_items >= 0

    space = SearchSpace(cfg.search_space)
    sample = space.sample(__import__("random").Random(0))
    assert "local_lr" in sample
    assert "update_expr" in sample
    assert isinstance(sample["update_expr"], dict)
    assert _tree_depth(sample["update_expr"]) <= int(cfg.search_space["update_expr"]["max_depth"])

    fp = space.fingerprint(sample)
    json.loads(fp)  # valid json fingerprint


def test_load_initial_pool_from_json(tmp_path: Path) -> None:
    cfg = load_runtime_config("genetic_algorithm/configs/iris_ga.yaml")

    pool_path = tmp_path / "pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "pool": [
                    {"local_lr": 0.001, "update_expr": {"t": "term", "name": "u"}},
                    {"local_lr": 0.002, "update_expr": {"t": "term", "name": "x"}},
                ]
            }
        ),
        encoding="utf-8",
    )

    ga = cfg.ga.__class__(
        population_size=cfg.ga.population_size,
        generations=cfg.ga.generations,
        elite_size=cfg.ga.elite_size,
        tournament_size=cfg.ga.tournament_size,
        mutation_rate=cfg.ga.mutation_rate,
        crossover_rate=cfg.ga.crossover_rate,
        top_k=cfg.ga.top_k,
        max_evals=cfg.ga.max_evals,
        workers=cfg.ga.workers,
        show_progress=cfg.ga.show_progress,
        eval_repeats=cfg.ga.eval_repeats,
        eval_seed_stride=cfg.ga.eval_seed_stride,
        memetic_enabled=cfg.ga.memetic_enabled,
        memetic_score_tol=cfg.ga.memetic_score_tol,
        memetic_immigrant_rate=cfg.ga.memetic_immigrant_rate,
        initial_pool_enabled=True,
        initial_pool_path=str(pool_path),
        initial_pool_max_items=1,
        initial_pool_strict=True,
    )
    cfg2 = cfg.__class__(
        name=cfg.name,
        seed=cfg.seed,
        device=cfg.device,
        output_dir=cfg.output_dir,
        fixed=cfg.fixed,
        search_space=cfg.search_space,
        fitness=cfg.fitness,
        ga=ga,
    )

    pool = _load_initial_pool(cfg2)
    assert len(pool) == 1
    assert pool[0]["local_lr"] == 0.001


def test_render_rule_file_contains_registration_contract() -> None:
    result = EvaluationResult(
        ok=True,
        score=0.9,
        objective_value=0.9,
        fitness_split="val",
        metrics={"val": {"acc": 0.9, "loss": 0.2}},
        train_summary=None,
        merged_params={"model": "mlp", "hidden": 256, "layers": 5},
        evolved_params={
            "local_lr": 1e-3,
            "local_weight_decay": 0.0,
            "head_lr": 1e-3,
            "head_weight_decay": 0.0,
            "normalize_update": False,
            "update_bias": True,
            "update_expr": {
                "t": "binary",
                "op": "outer",
                "a": {"t": "term", "name": "u"},
                "b": {"t": "term", "name": "x"},
            },
        },
    )

    source = render_generated_rule_source(
        rule_name="ga_test_rule",
        params=EvolvedRuleParams.from_genome(result.evolved_params),
        score=result.score,
        objective_value=result.objective_value,
    )
    assert '@register_update_rule(RULE_NAME)' in source
    assert 'RULE_NAME = "ga_test_rule"' in source
    assert "from genetic_algorithm.evolved_rule import EvolvedMLPUpdateRule, EvolvedRuleParams" in source
    assert "return EvolvedMLPUpdateRule(params)" in source


def _tree_depth(node: object) -> int:
    if not isinstance(node, dict):
        return 0
    if node.get("t") == "unary":
        return 1 + _tree_depth(node.get("a"))
    if node.get("t") == "binary":
        return 1 + max(_tree_depth(node.get("a")), _tree_depth(node.get("b")))
    return 0
