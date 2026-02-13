from __future__ import annotations

import importlib
import statistics
import traceback
from dataclasses import dataclass
from typing import Any

from .bootstrap import ensure_lab_namespace

ensure_lab_namespace()

import torch

from lab.core.task import ClassificationTask
from lab.core.trainer import Trainer
from lab.core.utils.logger import RunLogger
from lab.core.utils.seed import seed_everything
from lab.datasets.registry import DATASET_REGISTRY, get_dataset
from lab.models.registry import MODEL_REGISTRY, ModelContext, build_model

from .evolved_rule import EvolvedMLPUpdateRule, EvolvedRuleParams


_DATASET_MODULES: dict[str, str] = {
    "iris": "lab.datasets.tabular.iris",
    "breast_cancer": "lab.datasets.tabular.breast_cancer",
    "mnist": "lab.datasets.vision.mnist",
    "cifar10": "lab.datasets.vision.cifar",
    "cifar100": "lab.datasets.vision.cifar",
}

_LOCKED_ARCH = {"model": "mlp", "hidden": 256, "layers": 5}


@dataclass
class EvaluationResult:
    ok: bool
    score: float
    objective_value: float | None
    fitness_split: str
    metrics: dict[str, dict[str, float]]
    train_summary: dict[str, Any] | None
    merged_params: dict[str, Any]
    evolved_params: dict[str, Any]
    replicate_seeds: list[int] | None = None
    replicate_objectives: list[float] | None = None
    score_std: float | None = None
    error: str | None = None
    traceback: str | None = None


class _Args:
    def __init__(self, values: dict[str, Any]):
        self.__dict__.update(values)

    def __getattr__(self, _name: str) -> Any:
        return None


class SupervisedEvaluator:
    def __init__(
        self,
        *,
        fixed_params: dict[str, Any],
        objective: str,
        fitness_split: str,
        device: str,
        eval_repeats: int = 1,
        eval_seed_stride: int = 1,
    ):
        self.fixed_params = dict(fixed_params)
        self.objective = str(objective).lower()
        self.fitness_split = str(fitness_split).lower()
        self.device = str(device)
        self.eval_repeats = int(eval_repeats)
        self.eval_seed_stride = int(eval_seed_stride)

        if self.objective not in {"accuracy", "loss"}:
            raise ValueError("objective must be accuracy or loss")
        if self.fitness_split not in {"train", "val", "test"}:
            raise ValueError("fitness_split must be train, val, or test")
        if self.eval_repeats < 1:
            raise ValueError("eval_repeats must be >= 1")
        if self.eval_seed_stride < 1:
            raise ValueError("eval_seed_stride must be >= 1")

    def evaluate(self, genome: dict[str, Any]) -> EvaluationResult:
        merged = self._merge(self.fixed_params, genome)
        for k, v in _LOCKED_ARCH.items():
            merged[k] = v
        merged["local_phase_epochs"] = int(merged.get("local_phase_epochs", 10))
        merged["head_phase_epochs"] = int(merged.get("head_phase_epochs", 40))
        merged["epochs"] = int(merged["local_phase_epochs"]) + int(merged["head_phase_epochs"])

        rule_params = EvolvedRuleParams.from_genome(merged)
        base_seed = int(merged.get("seed", 0))
        rep_seeds = [base_seed + i * self.eval_seed_stride for i in range(self.eval_repeats)]

        try:
            rep_metrics: list[dict[str, dict[str, float]]] = []
            rep_objectives: list[float] = []
            rep_scores: list[float] = []
            last_train_summary: dict[str, Any] | None = None

            for rep_seed in rep_seeds:
                res = self._evaluate_once(merged, rule_params, seed=rep_seed)
                rep_metrics.append(res["metrics"])
                rep_objectives.append(float(res["objective_value"]))
                rep_scores.append(float(res["score"]))
                last_train_summary = res["train_summary"]

            metrics = _average_nested_metrics(rep_metrics)
            objective_value = float(sum(rep_objectives) / len(rep_objectives))
            score = float(sum(rep_scores) / len(rep_scores))
            score_std = float(statistics.pstdev(rep_scores)) if len(rep_scores) > 1 else 0.0

            return EvaluationResult(
                ok=True,
                score=score,
                objective_value=objective_value,
                fitness_split=self.fitness_split,
                metrics=metrics,
                train_summary=last_train_summary,
                merged_params=merged,
                evolved_params=rule_params.as_dict(),
                replicate_seeds=rep_seeds,
                replicate_objectives=rep_objectives,
                score_std=score_std,
            )
        except Exception as exc:
            return EvaluationResult(
                ok=False,
                score=float("-inf"),
                objective_value=None,
                fitness_split=self.fitness_split,
                metrics={},
                train_summary=None,
                merged_params=merged,
                evolved_params=rule_params.as_dict(),
                replicate_seeds=rep_seeds,
                replicate_objectives=None,
                score_std=None,
                error=str(exc),
                traceback=traceback.format_exc(),
            )

    def _evaluate_once(
        self,
        merged: dict[str, Any],
        rule_params: EvolvedRuleParams,
        *,
        seed: int,
    ) -> dict[str, Any]:
        seed_everything(seed)

        model = None
        trainer = None
        bundle = None
        try:
            dataset_name = str(merged["dataset"])
            model_name = str(merged["model"])
            if model_name != "mlp":
                raise ValueError("Architecture is locked to `mlp` for this GA mode.")

            _ensure_dataset_available(dataset_name)
            bundle = get_dataset(
                name=dataset_name,
                batch_size=int(merged["batch"]),
                seed=seed,
                val_frac=float(merged.get("val_frac", 0.2)),
                flatten=True,
                input_noise_dataset=float(merged.get("input_noise_dataset", 0.0)),
                noise_on_test=bool(int(merged.get("noise_on_test", 0))),
                pin_memory=self.device.startswith("cuda"),
            )

            if bundle.num_classes is None:
                raise ValueError(f"Dataset `{dataset_name}` did not provide num_classes.")
            if bundle.in_dim is None:
                raise ValueError("MLP evaluation requires tabular/flattened inputs with `in_dim`.")

            _ensure_model_available(model_name)
            mctx = ModelContext(
                dataset=dataset_name,
                num_classes=int(bundle.num_classes),
                in_dim=bundle.in_dim,
                in_channels=None,
                input_shape=None,
            )

            args_payload = dict(merged)
            args_payload["seed"] = int(seed)
            args = _Args(args_payload)
            model = build_model(model_name, mctx, args)
            task = ClassificationTask(num_classes=int(bundle.num_classes))
            learner = EvolvedMLPUpdateRule(rule_params)

            trainer = Trainer(
                model=model,
                task=task,
                learner=learner,
                device=self.device,
                input_noise_training=float(merged.get("input_noise_training", 0.0)),
                verbose=bool(int(merged.get("verbose", 0))),
                logger=RunLogger(run_dir=None),
            )

            train_summary = trainer.fit(
                bundle.train_loader,
                epochs=int(merged["epochs"]),
                show_progress=False,
                val_loader=bundle.val_loader,
            )

            metrics: dict[str, dict[str, float]] = {}
            metrics["train"] = _coerce_eval_dict(trainer.evaluate(bundle.train_loader, split="train"))
            if bundle.val_loader is not None:
                metrics["val"] = _coerce_eval_dict(trainer.evaluate(bundle.val_loader, split="val"))
            if bundle.test_loader is not None:
                metrics["test"] = _coerce_eval_dict(trainer.evaluate(bundle.test_loader, split="test"))

            if self.fitness_split not in metrics:
                raise ValueError(
                    f"Fitness split `{self.fitness_split}` unavailable for this run. "
                    f"Available: {sorted(metrics)}. "
                    "Use val_frac > 0 for val-based fitness."
                )

            objective_value, score = _extract_objective(
                metrics[self.fitness_split],
                objective=self.objective,
            )

            return {
                "metrics": metrics,
                "objective_value": float(objective_value),
                "score": float(score),
                "train_summary": train_summary,
            }
        finally:
            del trainer
            del model
            del bundle
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        out = dict(left)
        out.update(right)
        return out


def _coerce_eval_dict(values: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _average_nested_metrics(rows: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not rows:
        return {}
    splits = sorted({split for row in rows for split in row})
    out: dict[str, dict[str, float]] = {}
    for split in splits:
        keys = sorted({k for row in rows if split in row for k in row[split]})
        out_split: dict[str, float] = {}
        for key in keys:
            vals = [row[split][key] for row in rows if split in row and key in row[split]]
            if vals:
                out_split[key] = float(sum(vals) / len(vals))
        out[split] = out_split
    return out


def _extract_objective(metrics: dict[str, float], *, objective: str) -> tuple[float, float]:
    objective = objective.lower()
    if objective == "accuracy":
        acc = metrics.get("acc", metrics.get("metric"))
        if acc is None:
            raise ValueError(f"No accuracy-like metric in evaluation output: {sorted(metrics)}")
        return float(acc), float(acc)

    if objective == "loss":
        if "loss" not in metrics:
            raise ValueError(f"No loss metric in evaluation output: {sorted(metrics)}")
        loss = float(metrics["loss"])
        return loss, -loss

    raise ValueError(f"Unsupported objective `{objective}`")


def _ensure_model_available(model_name: str) -> None:
    if model_name in MODEL_REGISTRY.names():
        return
    if model_name != "mlp":
        raise ValueError(f"Unsupported model `{model_name}` for this GA mode.")
    importlib.import_module("lab.models.mlp")


def _ensure_dataset_available(dataset_name: str) -> None:
    if dataset_name in DATASET_REGISTRY.names():
        return
    module = _DATASET_MODULES.get(dataset_name)
    if module is None:
        raise ValueError(f"Unsupported dataset `{dataset_name}` for genetic_algorithm.")
    importlib.import_module(module)
