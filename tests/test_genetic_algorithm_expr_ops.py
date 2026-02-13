from __future__ import annotations

import sys

import torch

from conftest import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genetic_algorithm.evolved_rule import _eval_expr


def _signals(batch: int = 4, d_out: int = 3, d_in: int = 2) -> dict[str, torch.Tensor]:
    x = torch.randn(batch, d_in)
    u = torch.randn(batch, d_out)
    y = torch.randn(batch, d_out)
    w = torch.randn(batch, d_out, d_in)
    return {
        "x": x,
        "u": u,
        "y": y,
        "w": w,
        "weight": w,
        "label": torch.randn(batch, d_out),
        "mistake": torch.randn(batch, d_out),
        "layer": torch.randn(batch, d_out),
        "layer_ratio": torch.randn(batch, d_out),
        "ones": torch.ones(batch, 1),
        "zeros": torch.zeros(batch, 1),
    }


def test_weight_terminal_is_resolved() -> None:
    s = _signals()
    expr = {"t": "term", "name": "weight"}
    out = _eval_expr(expr, s, d_out=3, d_in=2)
    assert out.shape == (4, 3, 2)


def test_winner_flip_makes_sample_winner_positive() -> None:
    s = _signals(batch=5, d_out=4, d_in=3)
    expr = {"t": "unary", "op": "winner_flip", "c": 12.0, "a": {"t": "term", "name": "u"}}
    out = _eval_expr(expr, s, d_out=4, d_in=3)
    assert out.shape == (5, 4)
    winners = out.argmax(dim=1)
    for i in range(out.size(0)):
        row = out[i]
        assert row[winners[i]] >= 0.0
        mask = torch.ones_like(row, dtype=torch.bool)
        mask[winners[i]] = False
        assert torch.all(row[mask] <= 0.0)


def test_matmul_builds_cross_batch_matrix() -> None:
    s = _signals(batch=6, d_out=4, d_in=3)
    expr = {
        "t": "binary",
        "op": "matmul",
        "a": {"t": "term", "name": "y"},
        "b": {"t": "term", "name": "x"},
    }
    out = _eval_expr(expr, s, d_out=4, d_in=3)
    assert out.dim() == 3
    assert out.size(1) == s["y"].size(1)
    assert out.size(2) == s["x"].size(1)
    expected = s["y"].transpose(0, 1).matmul(s["x"])
    assert torch.allclose(out[0], expected, atol=1e-6, rtol=1e-6)
