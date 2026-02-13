from __future__ import annotations

import json
import sys

from conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genetic_algorithm.config import load_runtime_config
from genetic_algorithm.evaluator import EvaluationResult
from genetic_algorithm.evolved_rule import EvolvedRuleParams, render_generated_rule_source
from genetic_algorithm.search_space import SearchSpace


def test_load_runtime_config_and_search_space_sampling() -> None:
    cfg = load_runtime_config("genetic_algorithm/configs/iris_ga.yaml")
    assert cfg.name == "iris_ga_local_rule"
    assert cfg.fitness.objective in {"accuracy", "loss"}
    assert cfg.fixed["model"] == "mlp"
    assert cfg.fixed["hidden"] == 256
    assert cfg.fixed["layers"] == 5
    assert cfg.ga.eval_repeats >= 1

    space = SearchSpace(cfg.search_space)
    sample = space.sample(__import__("random").Random(0))
    assert "local_lr" in sample
    assert "update_expr" in sample
    assert isinstance(sample["update_expr"], dict)
    assert _tree_depth(sample["update_expr"]) <= int(cfg.search_space["update_expr"]["max_depth"])

    fp = space.fingerprint(sample)
    json.loads(fp)  # valid json fingerprint


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
