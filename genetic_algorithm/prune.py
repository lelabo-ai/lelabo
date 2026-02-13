from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import ensure_src_on_path

ensure_src_on_path()

from .evaluator import EvaluationResult, SupervisedEvaluator
from .evolved_rule import EvolvedRuleParams, render_generated_rule_source


@dataclass(frozen=True)
class _Candidate:
    genome: dict[str, Any]
    reason: str


def simplify_expr_tree(expr: dict[str, Any] | None) -> dict[str, Any]:
    """Return a simplified equivalent-ish expression tree."""
    node = copy.deepcopy(expr) if isinstance(expr, dict) else {"t": "term", "name": "zeros"}
    for _ in range(10):
        nxt = _simplify_expr_node(node)
        if _expr_fingerprint(nxt) == _expr_fingerprint(node):
            return nxt
        node = nxt
    return node


def genome_complexity(genome: dict[str, Any]) -> int:
    expr = genome.get("update_expr")
    nodes = _expr_node_count(expr)
    depth = _expr_depth(expr)
    score = nodes * 100 + depth * 10

    if bool(genome.get("normalize_update", False)):
        score += 3
    if bool(genome.get("update_bias", False)):
        score += 3
    if isinstance(genome.get("bias_expr"), dict):
        score += 20 + _expr_node_count(genome.get("bias_expr"))

    if abs(float(genome.get("local_weight_decay", 0.0))) > 1e-12:
        score += 2
    if abs(float(genome.get("head_weight_decay", 0.0))) > 1e-12:
        score += 2
    return int(score)


def _simplify_expr_node(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {"t": "term", "name": "zeros"}

    t = str(node.get("t", "term")).lower()
    if t == "term":
        name = str(node.get("name", "zeros")).strip().lower() or "zeros"
        return {"t": "term", "name": name}
    if t == "const":
        return {"t": "const", "v": float(node.get("v", 0.0))}

    if t == "unary":
        op = str(node.get("op", "neg")).lower()
        a = _simplify_expr_node(node.get("a"))

        if op == "scale":
            c = float(node.get("c", 1.0))
            if math.isclose(c, 1.0, rel_tol=0.0, abs_tol=1e-12):
                return a
            if _is_const(a):
                return {"t": "const", "v": float(a["v"] * c)}
            if _is_unary(a, "scale"):
                cc = float(a.get("c", 1.0))
                merged = {"t": "unary", "op": "scale", "c": float(cc * c), "a": copy.deepcopy(a.get("a"))}
                return _simplify_expr_node(merged)
            return {"t": "unary", "op": "scale", "c": c, "a": a}

        if _is_unary(a, op) and op in {"abs", "sign", "clip", "tanh", "relu", "normalize", "l2norm"}:
            return a
        if op == "neg" and _is_unary(a, "neg"):
            child = a.get("a")
            return _simplify_expr_node(child)
        if op == "square" and _is_unary(a, "sqrt_abs"):
            child = a.get("a")
            return _simplify_expr_node({"t": "unary", "op": "abs", "a": child})

        if _is_const(a):
            v = float(a["v"])
            folded = _fold_unary_const(op, v)
            if folded is not None:
                return {"t": "const", "v": folded}

        out: dict[str, Any] = {"t": "unary", "op": op, "a": a}
        if op == "scale":
            out["c"] = float(node.get("c", 1.0))
        return out

    if t == "binary":
        op = str(node.get("op", "add")).lower()
        a = _simplify_expr_node(node.get("a"))
        b = _simplify_expr_node(node.get("b"))

        if op == "add":
            if _is_const_val(a, 0.0):
                return b
            if _is_const_val(b, 0.0):
                return a
        if op == "sub":
            if _is_const_val(b, 0.0):
                return a
            if _expr_fingerprint(a) == _expr_fingerprint(b):
                return {"t": "const", "v": 0.0}
        if op == "mul":
            if _is_const_val(a, 0.0) or _is_const_val(b, 0.0):
                return {"t": "const", "v": 0.0}
            if _is_const_val(a, 1.0):
                return b
            if _is_const_val(b, 1.0):
                return a
        if op == "div":
            if _is_const_val(b, 1.0):
                return a
            if _is_const_val(a, 0.0):
                return {"t": "const", "v": 0.0}
        if op in {"max", "min"} and _expr_fingerprint(a) == _expr_fingerprint(b):
            return a

        if _is_const(a) and _is_const(b):
            va = float(a["v"])
            vb = float(b["v"])
            folded = _fold_binary_const(op, va, vb)
            if folded is not None:
                return {"t": "const", "v": folded}

        return {"t": "binary", "op": op, "a": a, "b": b}

    return {"t": "term", "name": "zeros"}


def _fold_unary_const(op: str, v: float) -> float | None:
    if op == "neg":
        return -v
    if op == "abs":
        return abs(v)
    if op == "sign":
        return -1.0 if v < 0 else (1.0 if v > 0 else 0.0)
    if op == "square":
        return v * v
    if op == "sqrt_abs":
        return math.sqrt(abs(v) + 1e-8)
    if op == "log1p_abs":
        return math.log1p(abs(v))
    if op == "relu":
        return max(0.0, v)
    if op == "tanh":
        return math.tanh(v)
    if op == "sigmoid":
        return 1.0 / (1.0 + math.exp(-v))
    if op == "exp":
        return math.exp(max(-12.0, min(12.0, v)))
    if op == "sin":
        return math.sin(v)
    if op == "cos":
        return math.cos(v)
    if op == "clip":
        return max(-1.0, min(1.0, v))
    return None


def _fold_binary_const(op: str, va: float, vb: float) -> float | None:
    if op == "add":
        return va + vb
    if op == "sub":
        return va - vb
    if op == "mul":
        return va * vb
    if op == "div":
        return va / (abs(vb) + 1e-6)
    if op == "max":
        return max(va, vb)
    if op == "min":
        return min(va, vb)
    return None


def _is_unary(node: Any, op: str) -> bool:
    return isinstance(node, dict) and str(node.get("t", "")).lower() == "unary" and str(node.get("op", "")).lower() == op


def _is_const(node: Any) -> bool:
    return isinstance(node, dict) and str(node.get("t", "")).lower() == "const" and isinstance(node.get("v"), (int, float))


def _is_const_val(node: Any, target: float) -> bool:
    if not _is_const(node):
        return False
    return math.isclose(float(node["v"]), float(target), rel_tol=0.0, abs_tol=1e-12)


def _expr_node_count(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    t = str(node.get("t", "")).lower()
    if t == "unary":
        return 1 + _expr_node_count(node.get("a"))
    if t == "binary":
        return 1 + _expr_node_count(node.get("a")) + _expr_node_count(node.get("b"))
    if t in {"term", "const"}:
        return 1
    return 1


def _expr_depth(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    t = str(node.get("t", "")).lower()
    if t == "unary":
        return 1 + _expr_depth(node.get("a"))
    if t == "binary":
        return 1 + max(_expr_depth(node.get("a")), _expr_depth(node.get("b")))
    return 1


def _expr_fingerprint(expr: Any) -> str:
    return json.dumps(expr, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _collect_expr_paths(node: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if not isinstance(node, dict):
        return []
    paths = [prefix]
    if isinstance(node.get("a"), dict):
        paths.extend(_collect_expr_paths(node["a"], prefix + ("a",)))
    if isinstance(node.get("b"), dict):
        paths.extend(_collect_expr_paths(node["b"], prefix + ("b",)))
    return paths


def _get_at_path(node: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _replace_at_path(node: dict[str, Any], path: tuple[str, ...], replacement: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return copy.deepcopy(replacement)
    out = copy.deepcopy(node)
    cur: Any = out
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            return out
        cur = nxt
    cur[path[-1]] = copy.deepcopy(replacement)
    return out


def _generate_expr_candidates(expr: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    out: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()

    def add(candidate: dict[str, Any], reason: str) -> None:
        simp = simplify_expr_tree(candidate)
        fp = _expr_fingerprint(simp)
        if fp in seen:
            return
        seen.add(fp)
        out.append((simp, reason))

    base = simplify_expr_tree(expr)
    add(base, "canonical")

    paths = _collect_expr_paths(base)
    for path in paths:
        node = _get_at_path(base, path)
        if not isinstance(node, dict):
            continue
        t = str(node.get("t", "")).lower()

        if t == "unary" and isinstance(node.get("a"), dict):
            add(_replace_at_path(base, path, node["a"]), f"drop_unary@{'.'.join(path) or 'root'}")
        if t == "binary":
            if isinstance(node.get("a"), dict):
                add(_replace_at_path(base, path, node["a"]), f"keep_a@{'.'.join(path) or 'root'}")
            if isinstance(node.get("b"), dict):
                add(_replace_at_path(base, path, node["b"]), f"keep_b@{'.'.join(path) or 'root'}")

        add(_replace_at_path(base, path, {"t": "term", "name": "zeros"}), f"to_zeros@{'.'.join(path) or 'root'}")
        add(_replace_at_path(base, path, {"t": "term", "name": "ones"}), f"to_ones@{'.'.join(path) or 'root'}")

    return out


def _generate_genome_candidates(genome: dict[str, Any], *, limit: int) -> list[_Candidate]:
    out: list[_Candidate] = []
    seen: set[str] = {_expr_fingerprint(genome)}

    def add(g: dict[str, Any], reason: str) -> None:
        if len(out) >= limit:
            return
        fp = _expr_fingerprint(g)
        if fp in seen:
            return
        seen.add(fp)
        out.append(_Candidate(genome=g, reason=reason))

    if bool(genome.get("update_bias", False)):
        cand = copy.deepcopy(genome)
        cand["update_bias"] = False
        cand["bias_expr"] = None
        add(cand, "disable_update_bias")

    if isinstance(genome.get("bias_expr"), dict):
        cand = copy.deepcopy(genome)
        cand["bias_expr"] = None
        add(cand, "drop_bias_expr")

    if bool(genome.get("normalize_update", False)):
        cand = copy.deepcopy(genome)
        cand["normalize_update"] = False
        add(cand, "disable_normalize_update")

    if abs(float(genome.get("local_weight_decay", 0.0))) > 1e-12:
        cand = copy.deepcopy(genome)
        cand["local_weight_decay"] = 0.0
        add(cand, "zero_local_weight_decay")

    if abs(float(genome.get("head_weight_decay", 0.0))) > 1e-12:
        cand = copy.deepcopy(genome)
        cand["head_weight_decay"] = 0.0
        add(cand, "zero_head_weight_decay")

    expr = genome.get("update_expr")
    if isinstance(expr, dict):
        for expr_cand, reason in _generate_expr_candidates(expr):
            if len(out) >= limit:
                break
            cand = copy.deepcopy(genome)
            cand["update_expr"] = expr_cand
            add(cand, f"expr:{reason}")

    return out[:limit]


def _split_fixed_and_evolved(winner: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    params = copy.deepcopy(winner.get("params", {}))
    evolved = copy.deepcopy(winner.get("evolved_params", {}))
    if not isinstance(params, dict):
        params = {}
    if not isinstance(evolved, dict):
        evolved = {}

    for key in list(evolved.keys()):
        params.pop(key, None)
    return params, evolved


def _pick_best_candidate(
    *,
    current_score: float,
    current_complexity: int,
    evaluated: list[tuple[_Candidate, EvaluationResult, int]],
    score_tol: float,
) -> tuple[_Candidate, EvaluationResult, int] | None:
    admissible: list[tuple[_Candidate, EvaluationResult, int]] = []
    for item in evaluated:
        _, result, complexity = item
        if not result.ok:
            continue
        if result.score >= (current_score - score_tol):
            admissible.append(item)

    if not admissible:
        return None

    admissible.sort(key=lambda x: (x[2], -x[1].score))
    cand, result, complexity = admissible[0]

    if complexity < current_complexity:
        return cand, result, complexity
    if result.score > (current_score + score_tol):
        return cand, result, complexity
    return None


def _prune_winner(
    *,
    winner: dict[str, Any],
    objective: str,
    fitness_split: str,
    device: str,
    eval_repeats: int,
    eval_seed_stride: int,
    score_tol: float,
    max_iters: int,
    max_candidates_per_iter: int,
) -> dict[str, Any]:
    fixed_params, evolved = _split_fixed_and_evolved(winner)
    evaluator = SupervisedEvaluator(
        fixed_params=fixed_params,
        objective=objective,
        fitness_split=fitness_split,
        device=device,
        eval_repeats=eval_repeats,
        eval_seed_stride=eval_seed_stride,
    )

    base_genome = copy.deepcopy(evolved)
    base_result = evaluator.evaluate(base_genome)
    base_complexity = genome_complexity(base_genome)

    report: dict[str, Any] = {
        "rank": winner.get("rank"),
        "rule_name": winner.get("rule_name"),
        "source_file": winner.get("file"),
        "manifest_score": winner.get("score"),
        "base_score": base_result.score,
        "base_objective_value": base_result.objective_value,
        "base_ok": base_result.ok,
        "base_error": base_result.error,
        "base_complexity": base_complexity,
        "accepted_steps": [],
    }
    if not base_result.ok:
        report["final_score"] = base_result.score
        report["final_objective_value"] = base_result.objective_value
        report["final_ok"] = False
        report["final_error"] = base_result.error
        report["final_complexity"] = base_complexity
        report["final_genome"] = base_genome
        report["final_metrics"] = base_result.metrics
        report["score_delta"] = 0.0
        report["complexity_delta"] = 0
        report["fixed_params"] = fixed_params
        return report

    current_genome = copy.deepcopy(base_genome)
    current_result = base_result
    current_complexity = base_complexity

    for iteration in range(1, int(max_iters) + 1):
        candidates = _generate_genome_candidates(current_genome, limit=max_candidates_per_iter)
        if not candidates:
            break

        evaluated: list[tuple[_Candidate, EvaluationResult, int]] = []
        for cand in candidates:
            result = evaluator.evaluate(cand.genome)
            complexity = genome_complexity(cand.genome)
            evaluated.append((cand, result, complexity))

        picked = _pick_best_candidate(
            current_score=float(current_result.score),
            current_complexity=int(current_complexity),
            evaluated=evaluated,
            score_tol=float(score_tol),
        )
        if picked is None:
            break

        cand, result, complexity = picked
        current_genome = copy.deepcopy(cand.genome)
        current_result = result
        current_complexity = int(complexity)
        report["accepted_steps"].append(
            {
                "iteration": iteration,
                "reason": cand.reason,
                "score": float(result.score),
                "objective_value": result.objective_value,
                "complexity": int(complexity),
            }
        )

    report["final_score"] = current_result.score
    report["final_objective_value"] = current_result.objective_value
    report["final_ok"] = current_result.ok
    report["final_error"] = current_result.error
    report["final_complexity"] = current_complexity
    report["final_genome"] = current_genome
    report["final_metrics"] = current_result.metrics
    report["score_delta"] = float(current_result.score - base_result.score)
    report["complexity_delta"] = int(current_complexity - base_complexity)
    report["fixed_params"] = fixed_params
    return report


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune/simplify GA-evolved update rules with score-preserving ablations.")
    p.add_argument("--manifest", type=str, required=True, help="Path to GA manifest.json.")
    p.add_argument("--top-k", type=int, default=0, help="Only prune top-k winners (0 = all).")
    p.add_argument("--score-tol", type=float, default=0.005, help="Allowed score drop during simplification.")
    p.add_argument("--max-iters", type=int, default=6, help="Max prune iterations per winner.")
    p.add_argument(
        "--max-candidates-per-iter",
        type=int,
        default=64,
        help="Max candidate genomes tested at each iteration.",
    )
    p.add_argument("--eval-repeats", type=int, default=None, help="Override evaluation repeats.")
    p.add_argument("--eval-seed-stride", type=int, default=None, help="Override evaluation seed stride.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write prune_report.json and pruned_rules/ (default: manifest dir).",
    )
    p.add_argument("--no-export-rules", action="store_true", help="Do not export pruned .py rule files.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    winners = payload.get("winners", [])
    if not isinstance(winners, list):
        raise ValueError("Invalid manifest: `winners` must be a list.")
    if not winners:
        raise ValueError("Manifest has no winners to prune.")

    objective = str(payload.get("objective", "accuracy")).strip().lower()
    fitness_split = str(payload.get("fitness_split", "val")).strip().lower()
    device = str(payload.get("device", "cpu"))
    eval_repeats = int(args.eval_repeats) if args.eval_repeats is not None else int(payload.get("eval_repeats", 1))
    eval_seed_stride = (
        int(args.eval_seed_stride) if args.eval_seed_stride is not None else int(payload.get("eval_seed_stride", 1))
    )

    selected = winners
    if int(args.top_k) > 0:
        selected = winners[: int(args.top_k)]

    results: list[dict[str, Any]] = []
    for winner in selected:
        rank = winner.get("rank")
        name = winner.get("rule_name")
        print(f"[PRUNE] rank={rank} rule={name} start")
        item = _prune_winner(
            winner=winner,
            objective=objective,
            fitness_split=fitness_split,
            device=device,
            eval_repeats=eval_repeats,
            eval_seed_stride=eval_seed_stride,
            score_tol=float(args.score_tol),
            max_iters=int(args.max_iters),
            max_candidates_per_iter=int(args.max_candidates_per_iter),
        )
        results.append(item)
        print(
            "[PRUNE] "
            f"rank={item['rank']} "
            f"base={float(item['base_score']):.6f} "
            f"final={float(item['final_score']):.6f} "
            f"complexity={item['base_complexity']}->{item['final_complexity']} "
            f"steps={len(item['accepted_steps'])}"
        )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else manifest_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_rules: list[str] = []
    if not bool(args.no_export_rules):
        rules_dir = output_dir / "pruned_rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        for item in results:
            if not item.get("final_ok", False):
                continue
            merged = copy.deepcopy(item.get("fixed_params", {}))
            merged.update(copy.deepcopy(item.get("final_genome", {})))
            params = EvolvedRuleParams.from_genome(merged)
            rule_name = f"{item['rule_name']}_pruned"
            source = render_generated_rule_source(
                rule_name=rule_name,
                params=params,
                score=float(item["final_score"]),
                objective_value=item.get("final_objective_value"),
            )
            path = rules_dir / f"{rule_name}.py"
            path.write_text(source, encoding="utf-8")
            exported_rules.append(str(path))

    report = {
        "manifest": str(manifest_path),
        "config_name": payload.get("config_name"),
        "objective": objective,
        "fitness_split": fitness_split,
        "device": device,
        "eval_repeats": eval_repeats,
        "eval_seed_stride": eval_seed_stride,
        "score_tol": float(args.score_tol),
        "max_iters": int(args.max_iters),
        "max_candidates_per_iter": int(args.max_candidates_per_iter),
        "results": results,
        "exported_rules": exported_rules,
    }
    report_path = output_dir / "prune_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[PRUNE] report={report_path}")
    if exported_rules:
        for path in exported_rules:
            print(f"[PRUNE] exported={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
