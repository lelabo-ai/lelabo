# lab/profile_update_rules.py
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from ..models import *
from ..core.task import *
from ..update_rules import UpdateRuleContext, build_update_rule

# -----------------------------
# Robust imports (package or local)
# -----------------------------

# -----------------------------
# Optimizer helper (fallback if core.utils not available)
# -----------------------------
def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = str(name).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)
    if name in ("sgd+momentum", "sgdm", "momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "ano":
        # optional dependency in your repo
        try:
            from ano_optimizer import Ano  # type: ignore
        except Exception as e:
            raise RuntimeError("Optimizer 'ano' requested but ano_optimizer is not installed/importable.") from e
        return Ano(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


# -----------------------------
# Batch specs
# -----------------------------
DATASET_SPECS = {
    "iris": dict(in_dim=4, num_classes=3),
    "breast_cancer": dict(in_dim=30, num_classes=2),
    "mnist": dict(in_dim=784, num_classes=10),
    "cifar10": dict(in_dim=3 * 32 * 32, num_classes=10),
    "cifar100": dict(in_dim=3 * 32 * 32, num_classes=100),
}


def _sync(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_cuda_peak(device: str):
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def _cuda_peak_mb(device: str) -> Tuple[float, float]:
    if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024**2)
        reserv = torch.cuda.max_memory_reserved() / (1024**2)
        return float(alloc), float(reserv)
    return float("nan"), float("nan")


def _sum_profiler_flops(prof) -> Optional[float]:
    # torch.profiler is best-effort: some ops won't report flops.
    total = 0
    seen = False
    for evt in prof.key_averages():
        fl = getattr(evt, "flops", None)
        if fl is not None:
            total += int(fl)
            seen = True
    return float(total) if seen else None


# -----------------------------
# Build model/task/batch
# -----------------------------
def build_classification( args):
    spec = DATASET_SPECS.get(args.dataset, None)
    if spec is None:
        # fallback: you can override manually
        in_dim = int(args.in_dim)
        num_classes = int(args.num_classes)
    else:
        in_dim = int(args.in_dim) if args.in_dim is not None else int(spec["in_dim"])
        num_classes = int(args.num_classes) if args.num_classes is not None else int(spec["num_classes"])

    Model = MLPClassifier
    model = Model(in_dim=in_dim, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes)

    Task = ClassificationTask
    task = Task(num_classes=num_classes)

    B = int(args.batch)
    x = torch.randn(B, in_dim, device=args.device, dtype=torch.float32)
    y = torch.randint(0, num_classes, (B,), device=args.device, dtype=torch.long)
    batch = (x, y)
    return model, task, batch


def build_ppo_minibatch( args):
    # obs_dim / n_actions: try gymnasium, else CartPole defaults
    obs_dim, n_actions = 4, 2
    if args.env is not None:
        try:
            import gymnasium as gym  # type: ignore
            env = gym.make(args.env)
            obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)
            env.close()
        except Exception:
            pass

    model = ActorCriticDiscrete(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=args.hidden, num_layers=args.layers)

    task = PPOTask(PPOConfig(
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        norm_adv=bool(args.norm_adv),
        clip_vloss=bool(args.clip_vloss),
        target_kl=args.target_kl,
    ))

    B = int(args.profile_batch if args.profile_batch is not None else args.batch)

    x = torch.randn(B, obs_dim, device=args.device, dtype=torch.float32)
    y = {
        "actions": torch.randint(0, n_actions, (B,), device=args.device, dtype=torch.long),
        "old_logprobs": torch.randn(B, device=args.device, dtype=torch.float32),
        "advantages": torch.randn(B, device=args.device, dtype=torch.float32),
        "returns": torch.randn(B, device=args.device, dtype=torch.float32),
        "old_values": torch.randn(B, device=args.device, dtype=torch.float32),
    }
    batch = (x, y)
    return model, task, batch


def build_learner(algo: str, model, args):
    algo = algo.strip().lower()
    aliases = {
        "backprop": "bp",
        "feedbackalignment": "fa",
        "targetprop": "tp",
        "sh": "softhebb",
    }
    if algo in aliases:
        algo = aliases[algo]
    opt = make_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    mode = "supervised" if args.task == "classification" else "rl"
    rl_algo = "ppo" if mode == "rl" else None

    extra = {}
    if algo == "bp":
        extra["grad_clip"] = args.max_grad_norm
    elif algo in ("fa", "dfa"):
        # keep profiler behavior: no implicit dataset-based clip
        extra["grad_clip"] = None

    ctx = UpdateRuleContext(
        args=args,
        model=model,
        task=None,
        optimizer=opt,
        mode=mode,
        dataset=args.dataset if mode == "supervised" else None,
        rl_algo=rl_algo,
        extra=extra or None,
    )
    return build_update_rule(algo, ctx)


# -----------------------------
# Profiling
# -----------------------------
@dataclass
class Measure:
    algo: str
    batch_size: int
    flops: Optional[float]
    time_ms_mean: float
    time_ms_std: float
    cuda_peak_alloc_mb_mean: float
    cuda_peak_alloc_mb_max: float
    cuda_peak_res_mb_mean: float
    cuda_peak_res_mb_max: float


def profile_one(learner, model, task, batch, device: str, warmup: int, iters: int) -> Tuple[Optional[float], float, float, List[float], List[float]]:
    model.to(device)

    # warmup
    for _ in range(max(0, warmup)):
        _sync(device)
        learner.train_step(model, task, batch, device)
        _sync(device)

    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # FLOPs on 1 iter with torch.profiler
    flops_val: Optional[float] = None
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.startswith("cuda") and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        _reset_cuda_peak(device)
        _sync(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as prof:
            learner.train_step(model, task, batch, device)
        _sync(device)
        flops_val = _sum_profiler_flops(prof)
    except Exception:
        flops_val = None  # best-effort

    # timing + peak mem (multiple iters, sans profiler)
    times_ms: List[float] = []
    peaks_alloc: List[float] = []
    peaks_res: List[float] = []

    for _ in range(max(1, iters)):
        _reset_cuda_peak(device)
        _sync(device)
        t0 = time.perf_counter()
        learner.train_step(model, task, batch, device)
        _sync(device)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(float(dt))

        a, r = _cuda_peak_mb(device)
        peaks_alloc.append(a)
        peaks_res.append(r)

    # mean/std time
    tmean = float(sum(times_ms) / len(times_ms))
    tvar = float(sum((t - tmean) ** 2 for t in times_ms) / max(1, len(times_ms) - 1))
    tstd = float(tvar ** 0.5)

    return flops_val, tmean, tstd, peaks_alloc, peaks_res


def _fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:  # nan
            return "n/a"
        return f"{x:.{nd}f}"
    return str(x)


def print_table(rows: List[Measure]):
    headers = [
        "algo",
        "batch",
        "MFLOPs/batch",
        "time ms (mean±std)",
        "cuda peak alloc MB (mean / max)",
        "cuda peak reserv MB (mean / max)",
    ]
    lines = [headers]
    for r in rows:
        lines.append([
            r.algo,
            str(r.batch_size),
            _fmt(r.flops/(10e6), 0),
            f"{_fmt(r.time_ms_mean)} ± {_fmt(r.time_ms_std)}",
            f"{_fmt(r.cuda_peak_alloc_mb_mean)} / {_fmt(r.cuda_peak_alloc_mb_max)}",
            f"{_fmt(r.cuda_peak_res_mb_mean)} / {_fmt(r.cuda_peak_res_mb_max)}",
        ])

    # simple pretty print
    colw = [max(len(str(lines[i][j])) for i in range(len(lines))) for j in range(len(headers))]
    for i, row in enumerate(lines):
        s = " | ".join(str(row[j]).ljust(colw[j]) for j in range(len(headers)))
        print(s)
        if i == 0:
            print("-+-".join("-" * colw[j] for j in range(len(headers))))


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "ppo"], default="classification")

    p.add_argument("--dataset", default="mnist")
    p.add_argument("--env", default="CartPole-v1")

    p.add_argument("--model", default="mlp")  # kept for symmetry; this profiler uses MLP/ActorCriticDiscrete
    p.add_argument("--algos", type=str, default="bp,fa,softhebb", help="comma-separated list, e.g. bp,fa,dfa,softhebb,tp,dni,kp")

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--profile-batch", type=int, default=None, help="override batch size (useful for PPO minibatches)")

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd+momentum", "ano"])
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=None)

    # PPO loss params (used only if --task ppo)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--norm-adv", type=int, default=1)
    p.add_argument("--clip-vloss", type=int, default=1)
    p.add_argument("--target-kl", type=float, default=None)

    # dataset overrides
    p.add_argument("--in-dim", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)

    # profiling controls
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main():
    args = build_parser().parse_args()

    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    results: List[Measure] = []

    torch.manual_seed(0)

    for algo in algos:
        # build fresh model per algo (so no carry-over state)
        if args.task == "classification":
            model, task, batch = build_classification(args)
            bs = int(args.batch)
        else:
            model, task, batch = build_ppo_minibatch(args)
            bs = int(args.profile_batch if args.profile_batch is not None else args.batch)

        try:
            learner = build_learner(algo, model, args)
        except Exception as e:
            print(f"[skip] {algo}: {e}")
            continue

        flops, tmean, tstd, peaks_alloc, peaks_res = profile_one(
            learner=learner,
            model=model,
            task=task,
            batch=batch,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
        )

        # aggregate mem
        alloc_valid = [x for x in peaks_alloc if x == x]  # not nan
        res_valid = [x for x in peaks_res if x == x]

        alloc_mean = float(sum(alloc_valid) / len(alloc_valid)) if alloc_valid else float("nan")
        alloc_max = float(max(alloc_valid)) if alloc_valid else float("nan")
        res_mean = float(sum(res_valid) / len(res_valid)) if res_valid else float("nan")
        res_max = float(max(res_valid)) if res_valid else float("nan")

        results.append(Measure(
            algo=algo,
            batch_size=bs,
            flops=flops,
            time_ms_mean=tmean,
            time_ms_std=tstd,
            cuda_peak_alloc_mb_mean=alloc_mean,
            cuda_peak_alloc_mb_max=alloc_max,
            cuda_peak_res_mb_mean=res_mean,
            cuda_peak_res_mb_max=res_max,
        ))

    print_table(results)


if __name__ == "__main__":
    main()
