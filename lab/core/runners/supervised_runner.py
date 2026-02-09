# lab/runners/supervised_runner.py
from __future__ import annotations

from typing import Any, Dict

import torch
from ..trainer import Trainer
from ..utils.optim import make_optimizer
from ..utils.logger import RunLogger

# datasets
from ...datasets import get_dataset
from ..task import ClassificationTask
from ..robustness import test_with_noise

# GLUE
from transformers import AutoModelForSequenceClassification
from ..glue_task import GLUETask

# models
from ...models.registry import build_model, ModelContext

# update rules
from ...algorithms.update_rules.backprop import Backprop
from ...algorithms.update_rules.local_probe_mlp import LocalProbeMLP

from ..callbacks import EarlyStopping, EarlyStoppingConfig


def run_supervised(args, logger: RunLogger) -> Dict[str, Any]:
    Xte = yte = None
    val_loaders = None
    num_classes = None
    num_labels = None
    is_regression = False

    val_loader = None

    callbacks = []
    if bool(args.early_stop):
        callbacks.append(EarlyStopping(EarlyStoppingConfig(
            monitor=args.early_monitor,
            mode="max" if "acc" in args.early_monitor else "min",
            patience=args.early_patience,
            min_delta=args.early_min_delta,
            warmup_epochs=max(0, args.early_warmup),
            restore_best=True,
        )))

    flatten = (args.model == "mlp")
    dataset_kwargs = dict(
        name=args.dataset,
        batch_size=args.batch,
        seed=args.seed,
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

    optimizer = make_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # algo
    if args.algo == "bp":
        learner = Backprop(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "lpl":
        if args.dataset == "glue":
            from ...algorithms.update_rules.local_probe_bert import LocalProbeBERT
            learner = LocalProbeBERT(base_optimizer=optimizer, probe_lr=args.lr)
        elif args.model == "mlp":
            learner = LocalProbeMLP(base_optimizer=optimizer, probe_lr=args.lr, weight_decay=args.weight_decay)
        else:
            from ...algorithms.update_rules.local_probe_blocks import LocalProbeBlocks
            learner = LocalProbeBlocks(base_optimizer=optimizer, probe_lr=args.lr)

    elif args.algo == "kp":
        from ...algorithms.update_rules.kp import KP
        learner = KP(learning_rate=args.lr, bp_lr=args.lr, bp_weight_decay=args.weight_decay)

    elif args.algo == 'scl':
        from ...algorithms.update_rules.scl import SoftContrastiveLearning
        learner = SoftContrastiveLearning(local_lr=args.lr, head_lr=args.lr, local_weight_decay=args.weight_decay)
        
    elif args.algo == 'kp3':
        from ...algorithms.update_rules.kp3 import KP3
        learner = KP3(local_lr=args.lr, head_lr=args.lr, head_weight_decay=args.weight_decay)
        
    elif args.algo == "softhebb":
        from ...algorithms.update_rules.softhebb import SoftHebb
        learner = SoftHebb(learning_rate=args.lr, head_lr=args.lr)

    elif args.algo == "tp":
        from ...algorithms.update_rules.targetprop import TargetPropagation
        dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
        inv_optim = torch.optim.SGD([dummy], lr=args.lr)
        learner = TargetPropagation(fwd_optimizer=optimizer, inv_optimizer=inv_optim, beta=1.0, noise_std=0.1)

    elif args.algo == "fa":
        from ...algorithms.update_rules.feedbackalignment import FeedbackAlignment
        learner = FeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "dfa":
        from ...algorithms.update_rules.dfa import DirectFeedbackAlignment
        learner = DirectFeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "dni":
        from ...algorithms.update_rules.dni import DNI
        learner = DNI(
            lr=args.lr,
            sg_lr=args.lr,
            sg_hidden=0,
            condition_on_label=False,
            lambda_mix=0.0,
            sg_scale=1.0,
            activation="relu",
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    trainer = Trainer(
        model=model,
        task=task,
        learner=learner,
        device=args.device,
        input_noise_training=args.input_noise_training,
        verbose=bool(args.verbose),
        callbacks=callbacks,
        logger=logger,
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

    # robustness (unchanged, uses Xte/yte)
    robustness_block: Dict[str, Any] = {}
    if args.dataset in ["iris", "mnist", "cifar10", "cifar100", "breast_cancer"] and args.robustness != "none" and Xte is not None and yte is not None:
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

    return summary
