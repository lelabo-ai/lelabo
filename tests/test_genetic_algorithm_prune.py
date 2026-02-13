from __future__ import annotations

import copy
import sys

from conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genetic_algorithm.prune import genome_complexity, simplify_expr_tree


def test_simplify_expr_tree_folds_square_sqrt_abs_const() -> None:
    expr = {
        "t": "unary",
        "op": "square",
        "a": {
            "t": "unary",
            "op": "sqrt_abs",
            "a": {"t": "const", "v": -1.2},
        },
    }
    out = simplify_expr_tree(expr)
    assert out["t"] == "const"
    assert abs(float(out["v"]) - 1.2) < 1e-6


def test_simplify_expr_tree_keeps_identity_for_add_zero() -> None:
    expr = {
        "t": "binary",
        "op": "add",
        "a": {"t": "term", "name": "mistake"},
        "b": {"t": "const", "v": 0.0},
    }
    out = simplify_expr_tree(expr)
    assert out == {"t": "term", "name": "mistake"}


def test_genome_complexity_prefers_simpler_genome() -> None:
    complex_genome = {
        "normalize_update": True,
        "update_bias": True,
        "local_weight_decay": 1e-4,
        "head_weight_decay": 1e-4,
        "bias_expr": {"t": "term", "name": "u"},
        "update_expr": {
            "t": "binary",
            "op": "mul",
            "a": {"t": "unary", "op": "tanh", "a": {"t": "term", "name": "layer_ratio"}},
            "b": {"t": "term", "name": "mistake"},
        },
    }
    simple_genome = copy.deepcopy(complex_genome)
    simple_genome["normalize_update"] = False
    simple_genome["update_bias"] = False
    simple_genome["local_weight_decay"] = 0.0
    simple_genome["head_weight_decay"] = 0.0
    simple_genome["bias_expr"] = None
    simple_genome["update_expr"] = {"t": "term", "name": "mistake"}

    assert genome_complexity(simple_genome) < genome_complexity(complex_genome)
