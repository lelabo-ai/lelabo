# tools/plot_sweep.py
# Usage examples at bottom.

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

import matplotlib.pyplot as plt


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


def resolve_config_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    root = repo_root()
    if not p.is_absolute():
        p_repo = root / p
        if p_repo.exists():
            return p_repo
    if len(p.parts) == 1:
        candidate = root / "experiments" / "sweeps" / p.name
        if candidate.exists():
            return candidate
    return p


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


def parse_scalar(s: str) -> Any:
    sl = s.strip().lower()
    if sl in ("none", "null"):
        return None
    if sl in ("true", "false"):
        return sl == "true"
    # int?
    try:
        if sl.startswith("0") and sl != "0" and "." not in sl and "e" not in sl:
            # keep as string (could be an id-like thing)
            return s
        return int(s)
    except Exception:
        pass
    # float?
    try:
        return float(s)
    except Exception:
        return s


def approx_equal(a: Any, b: Any, tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return str(a) == str(b)


def parse_kv_list(kvs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for kv in kvs:
        if "=" not in kv:
            raise ValueError(f"Invalid --where '{kv}' (expected key=value)")
        k, v = kv.split("=", 1)
        out[normalize_key(k.strip())] = parse_scalar(v.strip())
    return out


def split_keys(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [normalize_key(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# Curve metric parsing
# -------------------------
@dataclass(frozen=True)
class CurveSpec:
    # t can be:
    #   "train" or "eval" for supervised logs
    #   "rl_update" (or anything) for RL logs
    #   None to accept any "t" in metrics.jsonl
    t: Optional[str]
    split: Optional[str]  # only used for eval.<split>.<field>
    field: str            # the numeric key in the jsonl line


def parse_curve_y(y: str) -> CurveSpec:
    """
    Supported y specs in curve mode:
      - train.loss
      - eval.val.acc
      - eval.test.metric
      - rl_update.roll_mean_return_20
      - roll_mean_return_20          (field only; accept any t)
    """
    parts = y.split(".")

    # field-only: accept any t
    if len(parts) == 1:
        return CurveSpec(t=None, split=None, field=parts[0])

    # supervised formats
    if parts[0] == "train":
        if len(parts) != 2:
            raise ValueError(f"Invalid curve y spec '{y}' for train. Expected 'train.<field>'.")
        return CurveSpec(t="train", split=None, field=parts[1])

    if parts[0] == "eval":
        if len(parts) != 3:
            raise ValueError(f"Invalid curve y spec '{y}' for eval. Expected 'eval.<split>.<field>'.")
        return CurveSpec(t="eval", split=parts[1], field=parts[2])

    # generic "<t>.<field>" (for RL etc.)
    if len(parts) == 2:
        return CurveSpec(t=parts[0], split=None, field=parts[1])

    raise ValueError(
        f"Invalid curve y spec '{y}'. Use 'train.loss', 'eval.test.acc', "
        f"'<t>.<field>' (e.g. 'rl_update.roll_mean_return_20') or just '<field>'."
    )


def load_curve_series(metrics_path: Path, x_field: str, y_spec: CurveSpec) -> Dict[float, float]:
    # returns {x -> y}, aggregated per run (if duplicates, last wins)
    series: Dict[float, float] = {}
    current_epoch: Optional[int] = None

    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue

        # Track epoch (eval lines often don't include it)
        if "epoch" in d and isinstance(d["epoch"], int):
            current_epoch = d["epoch"]

        t = d.get("t", None)

        # Filter by type if requested
        if y_spec.t is not None and t != y_spec.t:
            continue

        # Special case: eval split filtering
        if y_spec.t == "eval":
            if d.get("split", None) != y_spec.split:
                continue

        # y field
        if y_spec.field not in d:
            continue
        yv = d.get(y_spec.field, None)
        if not isinstance(yv, (int, float)):
            continue

        # x
        if x_field == "epoch":
            if "epoch" in d and isinstance(d["epoch"], int):
                xv = float(d["epoch"])
            elif current_epoch is not None:
                xv = float(current_epoch)
            else:
                continue
        else:
            xv_raw = d.get(x_field, None)
            if isinstance(xv_raw, (int, float)):
                xv = float(xv_raw)
            else:
                continue

        series[xv] = float(yv)

    return series


# -------------------------
# Data model
# -------------------------
@dataclass
class RunRecord:
    run_dir: Path
    args: Dict[str, Any]
    summary: Dict[str, Any]
    curve: Optional[Dict[float, float]] = None


def make_label(keys: List[str], vals: Dict[str, Any]) -> str:
    if not keys:
        return "all"
    return ", ".join([f"{k}={fmt_val(vals.get(k))}" for k in keys])


def subset_matches(rec: RunRecord, where: Dict[str, Any]) -> bool:
    for k, v in where.items():
        rv = rec.args.get(k, None)

        # allow comma-separated OR
        if isinstance(v, str) and "," in v:
            allowed = [x.strip() for x in v.split(",")]
            if str(rv) not in allowed:
                return False
        else:
            if not approx_equal(rv, v):
                return False
    return True


def facet_key(rec: RunRecord, facet_keys: List[str]) -> Tuple[Any, ...]:
    return tuple(rec.args.get(k, None) for k in facet_keys)


def hue_key(rec: RunRecord, hue_keys: List[str]) -> Tuple[Any, ...]:
    return tuple(rec.args.get(k, None) for k in hue_keys)


# -------------------------
# Plotters
# -------------------------
def plot_curve(
    records: List[RunRecord],
    x_field: str,
    y_curve: str,
    hue_keys: List[str],
    facet_keys: List[str],
    agg_keys: List[str],
    only_hues: Optional[List[str]],
    title_base: str,
    out: Optional[str],
):
    y_spec = parse_curve_y(y_curve)

    # facet groups
    facets = unique_sorted([facet_key(r, facet_keys) for r in records]) if facet_keys else [tuple()]
    multi = len(facets) > 1

    # output handling
    pdf = None
    if out and out.lower().endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(out)

    for fvals in facets:
        # select facet
        sub = []
        for r in records:
            if facet_keys and facet_key(r, facet_keys) != fvals:
                continue
            sub.append(r)
        if not sub:
            continue

        # group by hue
        hue_groups: Dict[Tuple[Any, ...], List[RunRecord]] = {}
        for r in sub:
            hk = hue_key(r, hue_keys) if hue_keys else tuple()
            hue_groups.setdefault(hk, []).append(r)

        # optionally filter hues (string match on label)
        def hue_label(hk: Tuple[Any, ...]) -> str:
            vals = {k: v for k, v in zip(hue_keys, hk)} if hue_keys else {}
            return make_label(hue_keys, vals)

        if only_hues:
            keep = set(only_hues)
            hue_groups = {
                hk: rs
                for hk, rs in hue_groups.items()
                if hue_label(hk) in keep or (len(hk) >= 1 and str(hk[0]) in keep)
            }

        plt.figure()

        for hk, rs in sorted(hue_groups.items(), key=lambda kv: hue_label(kv[0])):
            # Ensure curves loaded
            curves: List[Dict[float, float]] = []
            for r in rs:
                if r.curve is None:
                    mp = r.run_dir / "metrics.jsonl"
                    if not mp.exists():
                        continue
                    r.curve = load_curve_series(mp, x_field=x_field, y_spec=y_spec)
                if r.curve:
                    curves.append(r.curve)

            if not curves:
                continue

            # Aggregate per x
            all_x = sorted({x for c in curves for x in c.keys()})
            means, lows, highs, kept_x = [], [], [], []

            for x in all_x:
                xs = [c[x] for c in curves if x in c]
                m, ci, n = mean_ci95(xs)
                if n == 0 or (isinstance(m, float) and math.isnan(m)):
                    continue
                kept_x.append(x)
                means.append(m)
                lows.append(m - ci)
                highs.append(m + ci)

            if not kept_x:
                continue

            lbl = hue_label(hk)
            plt.plot(kept_x, means, label=lbl)
            plt.fill_between(kept_x, lows, highs, alpha=0.2)

        # title
        facet_txt = ""
        if facet_keys:
            facet_txt = " | " + " | ".join([f"{k}={fmt_val(v)}" for k, v in zip(facet_keys, fvals)])
        plt.title(f"{title_base}{facet_txt}\ncurve: y={y_curve}, x={x_field} (band=CI95 over runs)")
        plt.xlabel(x_field)
        plt.ylabel(y_curve)
        plt.legend()

        plt.tight_layout()

        if pdf is not None:
            pdf.savefig()
            plt.close()
        elif out:
            p = Path(out)
            if multi:
                stem = p.stem
                suffix = p.suffix
                safe = "_".join([f"{k}={fmt_val(v)}" for k, v in zip(facet_keys, fvals)]) if facet_keys else "all"
                out_path = p.with_name(f"{stem}__{safe}{suffix}")
                plt.savefig(out_path, dpi=200)
                plt.close()
            else:
                plt.savefig(p, dpi=200)
                plt.close()
        else:
            plt.show()

    if pdf is not None:
        pdf.close()


def plot_sweep(
    records: List[RunRecord],
    x_key: str,
    y_path: str,
    hue_keys: List[str],
    facet_keys: List[str],
    agg_keys: List[str],
    only_hues: Optional[List[str]],
    title_base: str,
    out: Optional[str],
):
    # facet groups
    facets = unique_sorted([facet_key(r, facet_keys) for r in records]) if facet_keys else [tuple()]
    multi = len(facets) > 1

    # output handling
    pdf = None
    if out and out.lower().endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages(out)

    for fvals in facets:
        sub = []
        for r in records:
            if facet_keys and facet_key(r, facet_keys) != fvals:
                continue
            sub.append(r)
        if not sub:
            continue

        # bucket y by (hue, x)
        bucket: Dict[Tuple[Tuple[Any, ...], Any], List[float]] = {}
        for r in sub:
            xv = r.args.get(x_key, None)
            hk = hue_key(r, hue_keys) if hue_keys else tuple()

            yv = get_by_path(r.summary, y_path)
            if not isinstance(yv, (int, float)):
                continue

            bucket.setdefault((hk, xv), []).append(float(yv))

        # x domain (drop None)
        x_vals = unique_sorted([r.args.get(x_key, None) for r in sub if r.args.get(x_key, None) is not None])

        if not x_vals:
            raise RuntimeError(
                f"No valid x values for xvar='{x_key}'. "
                f"Check that this key exists in summary['args'] for your runs."
            )

        # numeric vs categorical
        numeric_x = all(isinstance(x, (int, float)) for x in x_vals)
        if numeric_x:
            x_plot = [float(x) for x in x_vals]
            x_ticks = None
        else:
            x_plot = list(range(len(x_vals)))
            x_ticks = [fmt_val(x) for x in x_vals]

        def hue_label(hk: Tuple[Any, ...]) -> str:
            vals = {k: v for k, v in zip(hue_keys, hk)} if hue_keys else {}
            return make_label(hue_keys, vals)

        # hue list
        hues = unique_sorted([hue_key(r, hue_keys) if hue_keys else tuple() for r in sub])
        if only_hues:
            keep = set(only_hues)
            hues = [hk for hk in hues if hue_label(hk) in keep or (len(hk) >= 1 and str(hk[0]) in keep)]

        plt.figure()

        for hk in sorted(hues, key=hue_label):
            means, cis = [], []
            for xv in x_vals:
                xs = bucket.get((hk, xv), [])
                m, ci, n = mean_ci95(xs)
                means.append(m)
                cis.append(ci)

            plt.errorbar(x_plot, means, yerr=cis, marker="o", capsize=3, label=hue_label(hk))

        facet_txt = ""
        if facet_keys:
            facet_txt = " | " + " | ".join([f"{k}={fmt_val(v)}" for k, v in zip(facet_keys, fvals)])
        plt.title(f"{title_base}{facet_txt}\nsweep: y={y_path} vs x={x_key} (errorbar=CI95 over runs)")
        plt.xlabel(x_key)
        plt.ylabel(y_path)

        if x_ticks is not None:
            plt.xticks(x_plot, x_ticks, rotation=25, ha="right")

        plt.legend()
        plt.tight_layout()

        if pdf is not None:
            pdf.savefig()
            plt.close()
        elif out:
            p = Path(out)
            if multi:
                stem = p.stem
                suffix = p.suffix
                safe = "_".join([f"{k}={fmt_val(v)}" for k, v in zip(facet_keys, fvals)]) if facet_keys else "all"
                out_path = p.with_name(f"{stem}__{safe}{suffix}")
                plt.savefig(out_path, dpi=200)
                plt.close()
            else:
                plt.savefig(p, dpi=200)
                plt.close()
        else:
            plt.show()

    if pdf is not None:
        pdf.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML sweep config (same as launch_grid).")
    ap.add_argument("--results-root", type=str, default="outputs/runs", help="Root folder containing experiments.")
    ap.add_argument("--exp-name", type=str, default=None, help="Override experiment name (default: cfg.name).")

    ap.add_argument(
        "--mode",
        type=str,
        choices=["curve", "sweep"],
        default="curve",
        help="curve: read metrics.jsonl and plot y over x (default x=epoch). "
             "sweep: read summary.json scalar y and plot vs x hyperparam.",
    )
    # curve args
    ap.add_argument("--x", type=str, default="epoch", help="x field (curve mode). Default: epoch")
    ap.add_argument(
        "--y",
        type=str,
        default="eval.val.acc",
        help="y spec. curve: 'train.loss', 'eval.val.acc', '<t>.<field>' (e.g. 'rl_update.roll_mean_return_20') "
             "or just '<field>' (accept any t). "
             "sweep: dotted path in summary.json (e.g. 'eval.test.acc').",
    )

    # sweep args
    ap.add_argument("--xvar", type=str, default=None, help="x hyperparam (sweep mode), e.g. layers, hidden, input-noise-training")

    # grouping/selection
    ap.add_argument("--hue", type=str, default="algo", help="Comma-separated keys for legend lines (default: algo).")
    ap.add_argument("--facet", type=str, default="", help="Comma-separated keys to split into multiple plots.")
    ap.add_argument(
        "--agg",
        type=str,
        default="seed",
        help="Comma-separated keys treated as repeats (usually seeds). Mainly informational; CI is computed across runs present.",
    )
    ap.add_argument("--where", action="append", default=[], help="Filter runs: key=value (repeatable). Example: --where algo=bp --where hidden=256")
    ap.add_argument("--only-hues", type=str, default="", help="Optional comma-separated whitelist of hue labels (or first hue value).")

    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output file (.png or .pdf). If facets>1 and out is .pdf => multi-page PDF. "
             "If facets>1 and out is .png => multiple files with facet suffix.",
    )

    args = ap.parse_args()

    if yaml is None:
        raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")

    cfg = yaml.safe_load(resolve_config_path(args.config).read_text(encoding="utf-8"))
    exp_name = args.exp_name or cfg.get("name")
    if not exp_name:
        raise ValueError("No experiment name found. Set cfg.name in YAML or pass --exp-name.")

    exp_dir = (repo_root() / args.results_root / exp_name).resolve()
    run_dirs = find_runs(exp_dir)
    if len(run_dirs) == 0:
        raise RuntimeError(f"No runs found in {exp_dir} (expected subfolders containing summary.json).")

    where = parse_kv_list(args.where)
    hue_keys = split_keys(args.hue)
    facet_keys = split_keys(args.facet)
    agg_keys = split_keys(args.agg)
    only_hues = [x.strip() for x in args.only_hues.split(",") if x.strip()] or None

    # Load runs
    records: List[RunRecord] = []
    for rd in run_dirs:
        s = load_json(rd / "summary.json")
        a = s.get("args", {})
        a_norm = {normalize_key(k): v for k, v in a.items()}

        rec = RunRecord(run_dir=rd, args=a_norm, summary=s, curve=None)

        if where and not subset_matches(rec, where):
            continue

        # in curve mode, ensure metrics file exists
        if args.mode == "curve":
            mp = rd / "metrics.jsonl"
            if not mp.exists():
                continue

        records.append(rec)

    if not records:
        raise RuntimeError("No runs matched your filters / mode requirements.")

    # Title base
    where_txt = ""
    if where:
        where_txt = " | " + " | ".join([f"{k}={fmt_val(v)}" for k, v in where.items()])
    title_base = f"{exp_name}{where_txt}"

    if args.mode == "curve":
        plot_curve(
            records=records,
            x_field=args.x,
            y_curve=args.y,
            hue_keys=hue_keys,
            facet_keys=facet_keys,
            agg_keys=agg_keys,
            only_hues=only_hues,
            title_base=title_base,
            out=args.out,
        )
    else:
        if not args.xvar:
            grid = cfg.get("grid", {})
            candidates = [normalize_key(k) for k in grid.keys() if normalize_key(k) not in ("seed", "algo")]
            if not candidates:
                raise ValueError("No --xvar given and couldn't infer one from YAML grid.")
            xvar = candidates[0]
        else:
            xvar = normalize_key(args.xvar)

        plot_sweep(
            records=records,
            x_key=xvar,
            y_path=args.y,
            hue_keys=hue_keys,
            facet_keys=facet_keys,
            agg_keys=agg_keys,
            only_hues=only_hues,
            title_base=title_base,
            out=args.out,
        )


if __name__ == "__main__":
    main()

"""
Paper

Scaling (layers):

python tools/plot_sweep.py \
    --config experiments/sweeps/compare_mnist_noise.yaml \
    --mode sweep --xvar layers \
    --y eval.test.acc --hue algo \
    --where hidden=2048 --where input-noise-training=0.0 \
    --out outputs/figures/layers.pdf
    
Scaling (hidden):

python tools/plot_sweep.py \
    --config experiments/sweeps/compare_mnist_noise.yaml \
    --mode sweep --xvar hidden \
    --y eval.test.acc --hue algo \
    --where layers=2 --where input-noise-training=0.0 \
    --out outputs/figures/hidden.pdf

Noise :

python tools/plot_sweep.py \
    --config experiments/sweeps/compare_mnist_noise.yaml \
    --mode sweep --xvar input-noise-training \
    --y eval.test.acc --hue algo \
    --where hidden=2048 --where layers=2 \
    --out outputs/figures/noise.pdf



"""
"""
EXAMPLES

python tools/plot_sweep.py \
  --config experiments/sweeps/compare_mnist_noise.yaml \
  --mode curve \
  --y eval.val.acc \
  --hue algo
  
python tools/plot_sweep.py \
  --config experiments/sweeps/compare_mnist_noise.yaml \
  --mode curve \
  --y eval.val.acc \
  --hue algo

# 2) Sweep plot (supervised): test acc vs input noise
python tools/plot_sweep.py \
  --config sweeps/compare_all_mnist_test2.yaml \
  --mode sweep \
  --xvar input-noise-training \
  --y eval.test.acc \
  --hue algo \
  --where hidden=2048 --where layers=2 \
  --out outputs/figures/noise.png

# 3) RL curve: rolling mean return vs total steps
python tools/plot_sweep.py \
  --config experiments/sweeps/compare_rl.yaml \
  --mode curve \
  --x total_steps \
  --y rl_update.roll_mean_return_20 \
  --hue algo \
  --out outputs/figures/rl_curve.png

# 4) RL curve (field-only y): accept any t in metrics.jsonl
python tools/plot_sweep.py \
  --config experiments/sweeps/compare_rl.yaml \
  --mode curve \
  --x total_steps \
  --y roll_mean_return_20 \
  --hue algo \
  --out outputs/figures/rl_curve.png
"""
