from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass
from typing import Any


class Dimension:
    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def mutate(self, current: Any, rng: random.Random) -> Any:
        raise NotImplementedError

    def crossover(self, left: Any, right: Any, rng: random.Random) -> Any:
        if rng.random() < 0.5:
            return copy.deepcopy(left)
        return copy.deepcopy(right)


@dataclass(frozen=True)
class ChoiceDimension(Dimension):
    choices: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("ChoiceDimension requires at least one value.")

    def sample(self, rng: random.Random) -> Any:
        return copy.deepcopy(rng.choice(self.choices))

    def mutate(self, current: Any, rng: random.Random) -> Any:
        if len(self.choices) == 1:
            return copy.deepcopy(self.choices[0])
        candidates = [v for v in self.choices if v != current]
        if not candidates:
            return copy.deepcopy(current)
        return copy.deepcopy(rng.choice(candidates))


@dataclass(frozen=True)
class FloatUniformDimension(Dimension):
    low: float
    high: float
    log_scale: bool

    def __post_init__(self) -> None:
        if not self.high > self.low:
            raise ValueError("FloatUniformDimension requires high > low.")
        if self.log_scale and (self.low <= 0.0 or self.high <= 0.0):
            raise ValueError("log-uniform bounds must be strictly positive.")

    def sample(self, rng: random.Random) -> float:
        if self.log_scale:
            lo = math.log10(self.low)
            hi = math.log10(self.high)
            return float(10 ** rng.uniform(lo, hi))
        return float(rng.uniform(self.low, self.high))

    def mutate(self, current: Any, rng: random.Random) -> float:
        base = float(current) if current is not None else self.sample(rng)
        if self.log_scale:
            lo = math.log10(self.low)
            hi = math.log10(self.high)
            cur = math.log10(max(self.low, min(self.high, max(base, 1e-30))))
            width = (hi - lo) * 0.2
            nxt = max(lo, min(hi, rng.gauss(cur, width)))
            return float(10 ** nxt)
        width = (self.high - self.low) * 0.2
        nxt = rng.gauss(base, width)
        return float(max(self.low, min(self.high, nxt)))


@dataclass(frozen=True)
class IntUniformDimension(Dimension):
    low: int
    high: int

    def __post_init__(self) -> None:
        if not self.high >= self.low:
            raise ValueError("IntUniformDimension requires high >= low.")

    def sample(self, rng: random.Random) -> int:
        return int(rng.randint(self.low, self.high))

    def mutate(self, current: Any, rng: random.Random) -> int:
        cur = int(current) if current is not None else self.sample(rng)
        span = max(1, int((self.high - self.low) * 0.25))
        nxt = cur + rng.randint(-span, span)
        return int(max(self.low, min(self.high, nxt)))


@dataclass(frozen=True)
class ExpressionTreeDimension(Dimension):
    min_depth: int
    max_depth: int
    terminals: tuple[str, ...]
    unary_ops: tuple[str, ...]
    binary_ops: tuple[str, ...]
    const_min: float
    const_max: float
    p_const: float

    def __post_init__(self) -> None:
        if self.min_depth < 0:
            raise ValueError("expression_tree min_depth must be >= 0")
        if self.max_depth < 1:
            raise ValueError("expression_tree max_depth must be >= 1")
        if self.max_depth < self.min_depth:
            raise ValueError("expression_tree max_depth must be >= min_depth")
        if not self.terminals:
            raise ValueError("expression_tree terminals cannot be empty")
        if not self.unary_ops and not self.binary_ops:
            raise ValueError("expression_tree needs unary_ops and/or binary_ops")
        if not (0.0 <= self.p_const <= 1.0):
            raise ValueError("expression_tree p_const must be in [0,1]")

    def sample(self, rng: random.Random) -> dict[str, Any]:
        return self._sample_node(rng, depth=0)

    def mutate(self, current: Any, rng: random.Random) -> dict[str, Any]:
        if not isinstance(current, dict):
            return self.sample(rng)

        tree = copy.deepcopy(current)
        paths = _collect_node_paths(tree)
        if not paths:
            return self.sample(rng)

        path = rng.choice(paths)
        depth = _path_depth(path)

        selected = _get_at_path(tree, path)
        if isinstance(selected, dict) and selected.get("t") == "const" and rng.random() < 0.5:
            value = float(selected.get("v", 0.0))
            width = max(1e-6, (self.const_max - self.const_min) * 0.15)
            selected["v"] = float(max(self.const_min, min(self.const_max, rng.gauss(value, width))))
            _set_at_path(tree, path, selected)
            return _trim_tree_depth(tree, self.max_depth, self.terminals, self.const_min, self.const_max, rng)
        if (
            isinstance(selected, dict)
            and selected.get("t") == "unary"
            and str(selected.get("op", "")).lower() == "scale"
            and rng.random() < 0.6
        ):
            value = float(selected.get("c", 1.0))
            width = max(1e-6, (self.const_max - self.const_min) * 0.15)
            selected["c"] = float(max(self.const_min, min(self.const_max, rng.gauss(value, width))))
            _set_at_path(tree, path, selected)
            return _trim_tree_depth(tree, self.max_depth, self.terminals, self.const_min, self.const_max, rng)

        new_sub = self._sample_node(rng, depth=depth)
        _set_at_path(tree, path, new_sub)
        return _trim_tree_depth(tree, self.max_depth, self.terminals, self.const_min, self.const_max, rng)

    def crossover(self, left: Any, right: Any, rng: random.Random) -> dict[str, Any]:
        if not isinstance(left, dict):
            return self.sample(rng)
        if not isinstance(right, dict):
            return copy.deepcopy(left)

        out = copy.deepcopy(left)
        left_paths = _collect_node_paths(out)
        right_paths = _collect_node_paths(right)
        if not left_paths or not right_paths:
            return out

        dst_path = rng.choice(left_paths)
        src_path = rng.choice(right_paths)
        sub = copy.deepcopy(_get_at_path(right, src_path))
        _set_at_path(out, dst_path, sub)
        return _trim_tree_depth(out, self.max_depth, self.terminals, self.const_min, self.const_max, rng)

    def _sample_leaf(self, rng: random.Random) -> dict[str, Any]:
        if rng.random() < self.p_const:
            return {"t": "const", "v": float(rng.uniform(self.const_min, self.const_max))}
        return {"t": "term", "name": str(rng.choice(self.terminals))}

    def _sample_node(self, rng: random.Random, *, depth: int) -> dict[str, Any]:
        if depth >= self.max_depth:
            return self._sample_leaf(rng)

        if depth < self.min_depth:
            return self._sample_op_node(rng, depth=depth)

        choices: list[str] = ["leaf"]
        if self.unary_ops:
            choices.append("unary")
        if self.binary_ops:
            choices.append("binary")
        mode = rng.choice(choices)

        if mode == "leaf":
            return self._sample_leaf(rng)
        if mode == "unary":
            op = str(rng.choice(self.unary_ops))
            node: dict[str, Any] = {"t": "unary", "op": op, "a": self._sample_node(rng, depth=depth + 1)}
            if op == "scale":
                node["c"] = float(rng.uniform(self.const_min, self.const_max))
            return node
        op = str(rng.choice(self.binary_ops))
        return {
            "t": "binary",
            "op": op,
            "a": self._sample_node(rng, depth=depth + 1),
            "b": self._sample_node(rng, depth=depth + 1),
        }

    def _sample_op_node(self, rng: random.Random, *, depth: int) -> dict[str, Any]:
        if self.binary_ops and (not self.unary_ops or rng.random() < 0.6):
            op = str(rng.choice(self.binary_ops))
            return {
                "t": "binary",
                "op": op,
                "a": self._sample_node(rng, depth=depth + 1),
                "b": self._sample_node(rng, depth=depth + 1),
            }
        if self.unary_ops:
            op = str(rng.choice(self.unary_ops))
            node: dict[str, Any] = {"t": "unary", "op": op, "a": self._sample_node(rng, depth=depth + 1)}
            if op == "scale":
                node["c"] = float(rng.uniform(self.const_min, self.const_max))
            return node
        return self._sample_leaf(rng)


def _as_dimension(name: str, raw: Any) -> Dimension:
    if isinstance(raw, list):
        return ChoiceDimension(tuple(raw))

    if isinstance(raw, dict):
        kind = str(raw.get("type", "choice")).strip().lower()
        if kind == "choice":
            choices = raw.get("values", raw.get("choices"))
            if not isinstance(choices, list):
                raise ValueError(f"search_space.{name}: choice dimension needs `values` list.")
            return ChoiceDimension(tuple(choices))
        if kind in {"uniform", "float_uniform"}:
            return FloatUniformDimension(
                low=float(raw["min"]),
                high=float(raw["max"]),
                log_scale=False,
            )
        if kind in {"log_uniform", "float_log_uniform"}:
            return FloatUniformDimension(
                low=float(raw["min"]),
                high=float(raw["max"]),
                log_scale=True,
            )
        if kind in {"int_uniform", "randint"}:
            return IntUniformDimension(
                low=int(raw["min"]),
                high=int(raw["max"]),
            )
        if kind in {"expression_tree", "expr_tree", "tree"}:
            terminals = raw.get(
                "terminals",
                ["x", "u", "y", "w", "weight", "label", "mistake", "layer", "layer_ratio"],
            )
            unary_ops = raw.get(
                "unary_ops",
                [
                    "scale",
                    "winner_flip",
                    "neg",
                    "abs",
                    "sign",
                    "square",
                    "sqrt_abs",
                    "log1p_abs",
                    "relu",
                    "tanh",
                    "sigmoid",
                    "softmax",
                    "l2norm",
                    "standardize",
                    "normalize",
                    "outer_normalize",
                    "row_normalize",
                    "col_normalize",
                    "sum",
                    "batch_sum",
                    "batch_mean",
                    "feature_sum",
                    "feature_mean",
                    "row_sum",
                    "col_sum",
                    "row_mean",
                    "col_mean",
                ],
            )
            binary_ops = raw.get(
                "binary_ops",
                [
                    "add",
                    "sub",
                    "mul",
                    "div",
                    "max",
                    "min",
                    "outer",
                    "dot",
                    "matmul",
                ],
            )
            if not isinstance(terminals, list):
                raise ValueError(f"search_space.{name}: `terminals` must be a list.")
            if not isinstance(unary_ops, list):
                raise ValueError(f"search_space.{name}: `unary_ops` must be a list.")
            if not isinstance(binary_ops, list):
                raise ValueError(f"search_space.{name}: `binary_ops` must be a list.")

            return ExpressionTreeDimension(
                min_depth=int(raw.get("min_depth", 1)),
                max_depth=int(raw.get("max_depth", 4)),
                terminals=tuple(str(x) for x in terminals),
                unary_ops=tuple(str(x) for x in unary_ops),
                binary_ops=tuple(str(x) for x in binary_ops),
                const_min=float(raw.get("const_min", -2.0)),
                const_max=float(raw.get("const_max", 2.0)),
                p_const=float(raw.get("p_const", 0.2)),
            )
        raise ValueError(f"search_space.{name}: unknown dimension type `{kind}`.")

    return ChoiceDimension((raw,))


class SearchSpace:
    def __init__(self, raw: dict[str, Any]):
        raw_copy = dict(raw)
        extra_raw = raw_copy.pop("extra", {})
        if extra_raw is None:
            extra_raw = {}
        if not isinstance(extra_raw, dict):
            raise ValueError("search_space.extra must be a dict.")

        self._dims: dict[str, Dimension] = {k: _as_dimension(k, v) for k, v in raw_copy.items()}
        self._extra_dims: dict[str, Dimension] = {k: _as_dimension(f"extra.{k}", v) for k, v in extra_raw.items()}

    def sample(self, rng: random.Random) -> dict[str, Any]:
        out = {k: d.sample(rng) for k, d in self._dims.items()}
        if self._extra_dims:
            out["extra"] = {k: d.sample(rng) for k, d in self._extra_dims.items()}
        return out

    def mutate(self, genome: dict[str, Any], *, mutation_rate: float, rng: random.Random) -> dict[str, Any]:
        out = copy.deepcopy(genome)
        for key, dim in self._dims.items():
            if rng.random() < mutation_rate:
                out[key] = dim.mutate(out.get(key), rng)
        if self._extra_dims:
            extra = dict(out.get("extra", {}))
            for key, dim in self._extra_dims.items():
                if rng.random() < mutation_rate:
                    extra[key] = dim.mutate(extra.get(key), rng)
            out["extra"] = extra
        return out

    def crossover(
        self,
        left: dict[str, Any],
        right: dict[str, Any],
        *,
        crossover_rate: float,
        rng: random.Random,
    ) -> dict[str, Any]:
        if rng.random() >= crossover_rate:
            return copy.deepcopy(left)

        child: dict[str, Any] = {}
        for key, dim in self._dims.items():
            child[key] = dim.crossover(left.get(key), right.get(key), rng)

        if self._extra_dims:
            child_extra: dict[str, Any] = {}
            left_extra = left.get("extra", {}) or {}
            right_extra = right.get("extra", {}) or {}
            for key, dim in self._extra_dims.items():
                child_extra[key] = dim.crossover(left_extra.get(key), right_extra.get(key), rng)
            child["extra"] = child_extra

        return child

    @staticmethod
    def fingerprint(genome: dict[str, Any]) -> str:
        return json.dumps(genome, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _collect_node_paths(node: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    if not isinstance(node, dict):
        return []
    paths = [prefix]
    if "a" in node and isinstance(node["a"], dict):
        paths.extend(_collect_node_paths(node["a"], prefix + ("a",)))
    if "b" in node and isinstance(node["b"], dict):
        paths.extend(_collect_node_paths(node["b"], prefix + ("b",)))
    return paths


def _path_depth(path: tuple[str, ...]) -> int:
    return sum(1 for p in path if p in {"a", "b"})


def _get_at_path(node: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = node
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _set_at_path(node: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    if not path:
        node.clear()
        if isinstance(value, dict):
            node.update(value)
        return

    cur: Any = node
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _trim_tree_depth(
    node: Any,
    max_depth: int,
    terminals: tuple[str, ...],
    const_min: float,
    const_max: float,
    rng: random.Random,
    depth: int = 0,
) -> Any:
    if not isinstance(node, dict):
        return {"t": "term", "name": str(rng.choice(terminals))}

    if depth >= max_depth:
        if rng.random() < 0.2:
            return {"t": "const", "v": float(rng.uniform(const_min, const_max))}
        return {"t": "term", "name": str(rng.choice(terminals))}

    t = node.get("t")
    if t == "unary":
        out = {"t": "unary", "op": str(node.get("op", "neg"))}
        if out["op"] == "scale":
            c = float(node.get("c", 1.0))
            c = max(const_min, min(const_max, c))
            out["c"] = c
        out["a"] = _trim_tree_depth(node.get("a"), max_depth, terminals, const_min, const_max, rng, depth + 1)
        return out
    if t == "binary":
        out = {"t": "binary", "op": str(node.get("op", "add"))}
        out["a"] = _trim_tree_depth(node.get("a"), max_depth, terminals, const_min, const_max, rng, depth + 1)
        out["b"] = _trim_tree_depth(node.get("b"), max_depth, terminals, const_min, const_max, rng, depth + 1)
        return out
    if t == "const":
        v = float(node.get("v", 0.0))
        v = max(const_min, min(const_max, v))
        return {"t": "const", "v": v}
    name = str(node.get("name", rng.choice(terminals)))
    return {"t": "term", "name": name}
