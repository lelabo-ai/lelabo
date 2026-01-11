# plots/plot_sweep_table.py

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

try:
    from tabulate import tabulate  # pip install tabulate
except Exception:
    tabulate = None


# -------------------------
# Stats helpers (NO scipy)
# -------------------------
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120,
    17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_ci95(xs: List[float]) -> Tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0, 1
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var) / math.sqrt(n)
    df = n - 1
    t = _T975.get(df, 1.96)  # approx normal if df > 30
    return m, t * se, n


# -------------------------
# IO helpers
# -------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(k: str) -> str:
    # YAML uses hyphens; argparse vars(args) uses underscores
    return k.replace("-", "_")


def get_by_path(obj: Any, dotted: str) -> Any:
    cur = obj
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def fmt_val(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) < 1e-3 and v != 0.0:
            return f"{v:.2e}"
        return f"{v:g}"
    return str(v)


def fmt_cell(m: float, ci: float, n: int, digits: int = 4) -> str:
    if n == 0 or (isinstance(m, float) and math.isnan(m)):
        return "—"
    return f"{m:.{digits}f} ± {ci:.{digits}f}  (n={n})"


def unique_sorted(values: List[Any]) -> List[Any]:
    uniq = list({v for v in values if v is not None})
    try:
        return sorted(uniq)
    except TypeError:
        return sorted(uniq, key=lambda x: str(x))


def find_runs(exp_dir: Path) -> List[Path]:
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    runs = []
    for p in sorted(exp_dir.iterdir()):
        if p.is_dir() and (p / "summary.json").exists():
            runs.append(p)
    return runs


def extract_run_row(summary: Dict[str, Any], metric_path: str, keys: List[str]) -> Optional[Dict[str, Any]]:
    args = summary.get("args", {})
    row: Dict[str, Any] = {}

    # pull requested keys from args (normalized)
    for k in keys:
        kk = normalize_key(k)
        row[k] = args.get(kk, None)

    metric = get_by_path(summary, metric_path)
    if metric is None or not isinstance(metric, (int, float)):
        return None

    row["_metric"] = float(metric)
    return row


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML sweep config (same as launch_grid).")
    ap.add_argument("--results-root", type=str, default="results/runs", help="Root folder containing experiments.")
    ap.add_argument("--metric", type=str, default="eval.test.acc", help="Dotted path in summary.json (scalar).")
    ap.add_argument("--digits", type=int, default=4, help="Digits for mean/CI display.")
    ap.add_argument("--exp-name", type=str, default=None, help="Override experiment name (default: cfg.name).")
    ap.add_argument(
        "--tablefmt",
        type=str,
        default="github",
        help="tabulate format (e.g. github, grid, fancy_grid, simple, plain)",
    )
    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
    if tabulate is None:
        raise RuntimeError("tabulate not installed. Run: pip install tabulate")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    exp_name = args.exp_name or cfg.get("name")
    if not exp_name:
        raise ValueError("No experiment name found. Set cfg.name in YAML or pass --exp-name.")

    exp_dir = (repo_root() / args.results_root / exp_name).resolve()
    run_dirs = find_runs(exp_dir)
    if len(run_dirs) == 0:
        raise RuntimeError(f"No runs found in {exp_dir} (expected subfolders containing summary.json).")

    grid: Dict[str, List[Any]] = cfg.get("grid", {})
    varying = [k for k in grid.keys()]  # keep YAML order

    # We use varying + seed for aggregation
    want_keys = list(dict.fromkeys(varying + ["seed"]))

    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        s = load_json(rd / "summary.json")
        r = extract_run_row(s, args.metric, want_keys)
        if r is None:
            continue
        r["_run_dir"] = str(rd)
        rows.append(r)

    if len(rows) == 0:
        raise RuntimeError(
            f"Found runs, but none had a numeric metric at '{args.metric}'. "
            f"Check the metric path and that summary.json contains it."
        )

    # Dimensions
    has_algo = "algo" in grid
    algos = unique_sorted([r.get("algo") for r in rows]) if has_algo else ["—"]

    nontrivial = [k for k in varying if k not in ("seed", "algo")]
    col_var = nontrivial[0] if len(nontrivial) >= 1 else None
    facet_vars = nontrivial[1:] if len(nontrivial) >= 2 else []

    col_vals = unique_sorted([r.get(col_var) for r in rows]) if col_var else [None]
    facet_values: Dict[str, List[Any]] = {fv: unique_sorted([r.get(fv) for r in rows]) for fv in facet_vars}

    def iter_facets() -> List[Dict[str, Any]]:
        if not facet_vars:
            return [dict()]
        combos: List[Dict[str, Any]] = [dict()]
        for fv in facet_vars:
            new_combos: List[Dict[str, Any]] = []
            for base in combos:
                for val in facet_values[fv]:
                    d = dict(base)
                    d[fv] = val
                    new_combos.append(d)
            combos = new_combos
        return combos

    # Bucket metrics by (algo, col, facet)
    bucket: Dict[Tuple[Any, Any, Tuple[Tuple[str, Any], ...]], List[float]] = {}
    for r in rows:
        a = r.get("algo") if has_algo else "—"
        c = r.get(col_var) if col_var else None
        facet_tuple = tuple(sorted([(fv, r.get(fv)) for fv in facet_vars], key=lambda x: x[0]))
        bucket.setdefault((a, c, facet_tuple), []).append(float(r["_metric"]))

    print()
    print(f"Experiment : {exp_name}")
    print(f"Metric     : {args.metric}")
    print(f"Runs (ok)  : {len(rows)} / {len(run_dirs)}")
    print(f"Directory  : {exp_dir}")
    print()

    for facet in iter_facets():
        facet_tuple = tuple(sorted(facet.items(), key=lambda x: x[0]))

        # Title
        if facet_vars:
            title = " | ".join([f"{k}={fmt_val(v)}" for k, v in facet_tuple])
            print(title)
        else:
            print("Table")

        # Build table
        if col_var is None:
            headers = ["algo", "mean ± CI95 (n)"]
            table = []
            for a in algos:
                xs = bucket.get((a, None, facet_tuple), [])
                m, ci, n = mean_ci95(xs)
                table.append([fmt_val(a), fmt_cell(m, ci, n, digits=args.digits)])
        else:
            headers = ["algo"] + [f"{col_var}={fmt_val(v)}" for v in col_vals]
            table = []
            for a in algos:
                row = [fmt_val(a)]
                for cv in col_vals:
                    xs = bucket.get((a, cv, facet_tuple), [])
                    m, ci, n = mean_ci95(xs)
                    row.append(fmt_cell(m, ci, n, digits=args.digits))
                table.append(row)

        print(tabulate(table, headers=headers, tablefmt=args.tablefmt, stralign="center"))
        print()


if __name__ == "__main__":
    main()
