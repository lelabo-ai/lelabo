# lab/runners/supervised_runner.py
from __future__ import annotations

from typing import Any, Dict

import torch
from core.trainer import Trainer
from core.utils.optim import make_optimizer
from core.utils.logger import RunLogger

# datasets
from core.data import make_iris_loaders, make_mnist_loaders, make_cifar_loaders, make_breast_cancer_loaders
from core.task import ClassificationTask
from core.robustness import test_with_noise

# GLUE
from transformers import AutoModelForSequenceClassification
from core.glue_data import make_glue_loaders
from core.glue_task import GLUETask

# models
from models.mlp import MLPClassifier
from models.convnet import ConvNetClassifier
from models.resnet import build_resnet

# update rules
from algorithms.update_rules.backprop import Backprop
from algorithms.update_rules.local_probe_mlp import LocalProbeMLP


def run_supervised(args, logger: RunLogger) -> Dict[str, Any]:
    Xte = yte = None
    val_loaders = None
    num_classes = None
    num_labels = None
    is_regression = False

    if args.dataset == "iris":
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim, num_classes = make_iris_loaders(
            batch_size=args.batch, seed=args.seed
        )

    elif args.dataset == "breast_cancer":
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim, num_classes = make_breast_cancer_loaders(
            batch_size=args.batch, seed=args.seed, flatten=True
        )

    elif args.dataset == "mnist":
        flatten = (args.model == "mlp")
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, num_classes = make_mnist_loaders(
            batch_size=args.batch, seed=args.seed, flatten=flatten
        )

    elif args.dataset in ["cifar10", "cifar100"]:
        flatten = (args.model == "mlp")
        train_loader, test_loader, in_dim_or_shape, num_classes = make_cifar_loaders(
            dataset=args.dataset, batch_size=args.batch, seed=args.seed, flatten=flatten
        )

    else:  # glue
        train_loader, val_loaders, num_labels, is_regression = make_glue_loaders(
            task_name=args.glue_task,
            model_name=args.hf_model,
            batch_size=args.batch,
            max_length=args.max_length,
            seed=args.seed,
        )
        test_loader = None

    # model + task
    if args.dataset == "glue":
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model, num_labels=num_labels)
        task = GLUETask(task_name=args.glue_task, is_regression=is_regression, num_labels=num_labels)
    else:
        if args.model == "mlp":
            if args.dataset in ["iris", "breast_cancer"]:
                in_dim = in_dim
            elif args.dataset == "mnist":
                in_dim = in_dim_or_shape
            else:
                in_dim = 3072
            model = MLPClassifier(in_dim=in_dim, hidden_dim=args.hidden, num_layers=args.layers, num_classes=num_classes)

        elif args.model == "cnn":
            in_channels = 1 if args.dataset == "mnist" else 3
            model = ConvNetClassifier(in_channels=in_channels, num_classes=num_classes)

        elif args.model in ["resnet18", "resnet34", "resnet50"]:
            in_channels = 1 if args.dataset == "mnist" else 3
            model = build_resnet(
                name=args.model,
                num_classes=num_classes,
                in_channels=in_channels,
                cifar_stem=(args.dataset in ["cifar10", "cifar100", "mnist"]),
                weights=None,
            )
        else:
            raise ValueError(f"Unknown model: {args.model}")

        task = ClassificationTask(num_classes=num_classes)

    optimizer = make_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # algo
    if args.algo == "bp":
        algo = Backprop(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "lpl":
        if args.dataset == "glue":
            from algorithms.update_rules.local_probe_bert import LocalProbeBERT
            algo = LocalProbeBERT(base_optimizer=optimizer, probe_lr=args.lr)
        elif args.model == "mlp":
            algo = LocalProbeMLP(base_optimizer=optimizer, probe_lr=args.lr, weight_decay=args.weight_decay)
        else:
            from algorithms.update_rules.local_probe_blocks import LocalProbeBlocks
            algo = LocalProbeBlocks(base_optimizer=optimizer, probe_lr=args.lr)

    elif args.algo == "kp":
        from algorithms.update_rules.kp import KP
        algo = KP(learning_rate=args.lr)

    elif args.algo == "softhebb":
        from algorithms.update_rules.softhebb import SoftHebb
        algo = SoftHebb(learning_rate=args.lr, head_lr=args.lr)

    elif args.algo == "tp":
        from algorithms.update_rules.targetprop import TargetPropagation
        dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
        inv_optim = torch.optim.SGD([dummy], lr=args.lr)
        algo = TargetPropagation(fwd_optimizer=optimizer, inv_optimizer=inv_optim, beta=1.0, noise_std=0.1)

    elif args.algo == "fa":
        from algorithms.update_rules.feedbackalignment import FeedbackAlignment
        algo = FeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "dfa":
        from algorithms.update_rules.dfa import DirectFeedbackAlignment
        algo = DirectFeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    trainer = Trainer(
        model=model,
        task=task,
        algorithm=algo,
        device=args.device,
        input_noise_training=args.input_noise_training,
        verbose=bool(args.verbose),
        logger=logger,
    )

    summary: Dict[str, Any] = {"args": vars(args)}

    # train
    summary["train"] = trainer.fit(train_loader, epochs=args.epochs, show_progress=True)

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
    if args.dataset in ["iris", "mnist", "cifar10", "cifar100", "breast_cancer"] and args.robustness != "none":
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
