# lab/runners/supervised_runner.py
from __future__ import annotations

from typing import Any, Dict
import json

import torch
from ..trainer import Trainer
from ...optimizers import make_optimizer, make_scheduler
from ..utils.logger import RunLogger

# datasets
from ...supervised.datasets import get_dataset
from ...supervised.datasets.base import dataset_to_tensors
from ..task import ClassificationTask, GLUETask
from ..robustness import test_with_noise

# models
from ...models.registry import build_model, ModelContext

from ...update_rules import UpdateRuleContext, build_update_rule
from ...metrics import (
    MetricContext,
    build_metric,
    get_metric_names,
    parse_metric_names,
    validate_metric_requests,
)
from ..utils.seed import derive_seed

from ...callbacks import build_configured_callbacks


def run_supervised(args, logger: RunLogger) -> Dict[str, Any]:
    Xte = yte = None
    val_loaders = None
    num_classes = None
    num_labels = None
    is_regression = False

    val_loader = None
    requested_metrics = parse_metric_names(getattr(args, "metrics", ""))
    available_metrics = set(get_metric_names())
    unknown_metrics = [name for name in requested_metrics if name not in available_metrics]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s): {unknown_metrics}. "
            f"Available: {sorted(available_metrics)}"
        )

    flatten = (args.model == "mlp")
    dataset_seed = derive_seed(args.seed, "supervised", "dataset", args.dataset)
    dataset_kwargs = dict(
        name=args.dataset,
        batch_size=args.batch,
        seed=dataset_seed,
        val_frac=args.val_frac,
        flatten=flatten,
        input_noise_dataset=args.input_noise_dataset,
        noise_on_test=bool(args.noise_on_test > 0),
    )
    if args.dataset == "glue":
        dataset_kwargs.update(
            glue_task=args.glue_task,
            hf_model=args.hf_model,
            max_length=args.max_length,
        )
    dataset_kwargs.update(dict(getattr(args, "dataset_params", {}) or {}))

    bundle = get_dataset(**dataset_kwargs)

    train_loader = bundle.train_loader
    val_loader = bundle.val_loader
    test_loader = bundle.test_loader
    num_classes = bundle.num_classes
    Xte = bundle.x_test
    yte = bundle.y_test

    # model + task
    if args.dataset == "glue":
        val_loaders = bundle.meta.get("val_loaders", {})
        num_labels = bundle.meta.get("num_labels", num_classes)
        is_regression = bundle.meta.get("is_regression", False)

        if args.model in {"hf", "bert"}:
            ctx = ModelContext(
                dataset=args.dataset,
                num_classes=num_labels,
                in_dim=None,
                in_channels=None,
                input_shape=None,
                extra={"hf_model": args.hf_model},
            )
            model = build_model(args.model, ctx, args)
        else:
            try:
                from transformers import AutoModelForSequenceClassification
            except ImportError as exc:
                raise ImportError(
                    "Dataset 'glue' requires 'transformers'. "
                    "Install optional deps with: pip install '.[nlp]'"
                ) from exc
            model = AutoModelForSequenceClassification.from_pretrained(args.hf_model, num_labels=num_labels)

        task = GLUETask(task_name=args.glue_task, is_regression=is_regression, num_labels=num_labels)
    else:
        in_channels = int(bundle.input_shape[0]) if bundle.input_shape is not None else None
        ctx = ModelContext(
            dataset=args.dataset,
            num_classes=num_classes,
            in_dim=bundle.in_dim,
            in_channels=in_channels,
            input_shape=bundle.input_shape,
        )
        model = build_model(args.model, ctx, args)
        task = ClassificationTask(num_classes=num_classes)

    optimizer_params = dict(getattr(args, "optimizer_params", {}) or {})
    optimizer = make_optimizer(
        args.optimizer,
        model.parameters(),
        lr=float(optimizer_params.pop("lr", args.lr)),
        weight_decay=float(optimizer_params.pop("weight_decay", args.weight_decay)),
        momentum=float(optimizer_params.pop("momentum", 0.9)),
        args=args,
        mode="supervised",
        dataset=args.dataset,
        **optimizer_params,
    )
    scheduler = None
    if getattr(args, "lr_scheduler", "none") not in (None, "none", "null", "off", ""):
        sched_kwargs = dict(getattr(args, "scheduler_params", {}) or {})
        if getattr(args, "lr_scheduler_kwargs", None):
            try:
                loaded = json.loads(args.lr_scheduler_kwargs)
            except Exception as exc:
                raise ValueError("lr-scheduler-kwargs must be valid JSON.") from exc
            if not isinstance(loaded, dict):
                raise ValueError("lr-scheduler-kwargs must decode to a JSON object.")
            sched_kwargs.update(loaded)

        steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
        scheduler = make_scheduler(
            args.lr_scheduler,
            optimizer,
            args=args,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            interval=getattr(args, "lr_scheduler_interval", "epoch"),
            monitor=getattr(args, "lr_scheduler_monitor", "val.loss"),
            **sched_kwargs,
        )
    schedulers = [scheduler] if scheduler is not None else []
    callbacks = build_configured_callbacks(
        args=args,
        mode="supervised",
        dataset=args.dataset,
        model=model,
        optimizer=optimizer,
        schedulers=schedulers,
    )

    rule_extra = {
        "update_rule_params": dict(getattr(args, "update_rule_params", {}) or {}),
    }
    metric_extra = {
        "requested_metrics": requested_metrics,
        "metric_params": dict(getattr(args, "metric_params", {}) or {}),
    }
    ctx = UpdateRuleContext(
        args=args,
        model=model,
        task=task,
        optimizer=optimizer,
        mode="supervised",
        dataset=args.dataset,
        extra=rule_extra,
    )
    learner = build_update_rule(args.algo, ctx)
    metric_ctx = MetricContext(
        args=args,
        mode="supervised",
        dataset=args.dataset,
        algo=args.algo,
        extra=metric_extra,
    )
    validate_metric_requests(
        requested_metrics,
        ctx=metric_ctx,
        task_kind="regression" if bool(is_regression) else "classification",
    )
    metric_probes = [build_metric(name, metric_ctx) for name in requested_metrics]

    trainer = Trainer(
        model=model,
        task=task,
        learner=learner,
        device=args.device,
        input_noise_training=args.input_noise_training,
        verbose=bool(args.verbose),
        callbacks=callbacks,
        logger=logger,
        schedulers=schedulers,
        scheduler_interval=getattr(args, "lr_scheduler_interval", "epoch"),
        scheduler_monitor=getattr(args, "lr_scheduler_monitor", "val.loss"),
        metric_probes=metric_probes,
    )

    summary: Dict[str, Any] = {"args": vars(args)}

    # train (pass val_loader!)
    summary["train"] = trainer.fit(
        train_loader,
        epochs=args.epochs,
        show_progress=True,
        val_loader=val_loader
    )

    # eval
    eval_block: Dict[str, Any] = {}
    if args.dataset == "glue":
        for split_name, vloader in val_loaders.items():
            res = trainer.evaluate(vloader, split=split_name)
            eval_block[split_name] = res
    else:
        eval_block["test"] = trainer.evaluate(test_loader, split="test")
    summary["eval"] = eval_block

    # robustness
    robustness_block: Dict[str, Any] = {}
    supports_robustness = args.dataset in {"iris", "mnist", "cifar10", "cifar100", "breast_cancer"}
    if supports_robustness and args.robustness != "none" and (Xte is None or yte is None):
        test_ds = getattr(bundle, "test_dataset", None)
        if test_ds is not None:
            raw_max = int(getattr(args, "robustness_max_samples", 0) or 0)
            max_items = raw_max if raw_max > 0 else None
            Xte, yte = dataset_to_tensors(test_ds, max_items=max_items)

    if supports_robustness and args.robustness != "none" and Xte is not None and yte is not None:
        sigmas_input = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_rel = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_w = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        def run(mode, sigmas):
            rows = []
            for s in sigmas:
                mean_acc, ci95 = test_with_noise(
                    model=model.to(args.device),
                    x=Xte,
                    y=yte,
                    sigma=s,
                    mode=mode,
                    device=args.device,
                    trials=args.noise_trials,
                    seed=derive_seed(args.seed, "robustness", mode, s),
                )
                rows.append({"sigma": float(s), "mean_acc": float(mean_acc), "ci95": float(ci95)})
            robustness_block[mode] = {"trials": int(args.noise_trials), "results": rows}

        if args.robustness in ["input_noise", "all"]:
            run("input_noise", sigmas_input)
        if args.robustness in ["relative_input_noise", "all"]:
            run("relative_input_noise", sigmas_rel)
        if args.robustness in ["weight_noise", "all"]:
            run("weight_noise", sigmas_w)

    if robustness_block:
        summary["robustness"] = robustness_block
    
    if args.verbose > 0:
        print("Summary:", summary)

    return summary
