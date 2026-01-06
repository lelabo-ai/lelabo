# main.py
import argparse
import torch

from core.data import make_iris_loaders, make_mnist_loaders, make_cifar_loaders, make_breast_cancer_loaders
from core.task import ClassificationTask
from core.trainer import Trainer
from core.robustness import test_with_noise

# GLUE / Transformers
from transformers import AutoModelForSequenceClassification
from core.glue_data import make_glue_loaders
from core.glue_task import GLUETask

# Models
from models.mlp import MLPClassifier
from models.convnet import ConvNetClassifier
from models.resnet import build_resnet

# Algorithms
from algorithms.backprop import Backprop
from algorithms.local_probe_mlp import LocalProbeMLP


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--dataset", choices=["breast_cancer", "iris", "mnist", "cifar10", "cifar100", "glue"], default="iris")
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet18", "resnet34", "resnet50", "bert"], default="mlp")
    parser.add_argument("--algo", choices=["bp", "lpl", "kp", 'softhebb', 'tp', 'fa', 'dfa'], default="softhebb")

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--verbose", type=int, default=0)
    
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd', 'sgd+momentum', 'ano'])
    parser.add_argument('--input-noise-training', type=float, default=0.2, help='stddev of gaussian noise added to inputs during training')
    # Robustness (only meaningful for datasets where we have Xte/yte tensors)
    parser.add_argument(
        "--robustness",
        type=str,
        default="input_noise",
        choices=["none", "input_noise", "relative_input_noise", "weight_noise", "all"],
    )
    parser.add_argument("--noise-trials", type=int, default=30)

    # GLUE-specific
    parser.add_argument(
        "--glue-task",
        type=str,
        default="cola",
        choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"],
    )
    parser.add_argument("--hf-model", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=128)

    # optimizer
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ---------------------------------------------------------------------
    # DATA + (optional) tensors for robustness
    # ---------------------------------------------------------------------
    Xte = yte = None
    val_loaders = None
    num_classes = None
    num_labels = None
    is_regression = False

    if args.dataset == "iris":
        if args.model in ["cnn", "resnet18", "resnet34", "resnet50", "bert"]:
            raise ValueError("Use --model mlp for iris in this scaffold.")
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim, num_classes = make_iris_loaders(
            batch_size=args.batch, seed=args.seed
        )
    
    elif args.dataset == "breast_cancer":
        if args.model in ["cnn", "resnet18", "resnet34", "resnet50", "bert"]:
            raise ValueError("Use --model mlp for breast_cancer in this scaffold.")
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim, num_classes = make_breast_cancer_loaders(
            batch_size=args.batch, seed=args.seed, flatten=True
        )

    elif args.dataset == "mnist":
        if args.model == "bert":
            raise ValueError("BERT is only supported for --dataset glue.")
        flatten = (args.model == "mlp")
        train_loader, test_loader, Xtr, ytr, Xte, yte, in_dim_or_shape, num_classes = make_mnist_loaders(
            batch_size=args.batch, seed=args.seed, flatten=flatten
        )

    elif args.dataset in ["cifar10", "cifar100"]:
        if args.model == "bert":
            raise ValueError("BERT is only supported for --dataset glue.")
        flatten = (args.model == "mlp")
        train_loader, test_loader, in_dim_or_shape, num_classes = make_cifar_loaders(
            dataset=args.dataset, batch_size=args.batch, seed=args.seed, flatten=flatten
        )

    else:  # glue
        if args.model != "bert":
            raise ValueError("For GLUE, set --model bert.")
        train_loader, val_loaders, num_labels, is_regression = make_glue_loaders(
            task_name=args.glue_task,
            model_name=args.hf_model,
            batch_size=args.batch,
            max_length=args.max_length,
            seed=args.seed,
        )
        test_loader = None  # GLUE uses val_loaders instead

    # ---------------------------------------------------------------------
    # MODEL + TASK
    # ---------------------------------------------------------------------
    if args.dataset == "glue":
        model = AutoModelForSequenceClassification.from_pretrained(args.hf_model, num_labels=num_labels)
        task = GLUETask(task_name=args.glue_task, is_regression=is_regression, num_labels=num_labels)

    else:
        # vision/classic tasks
        if args.model == "mlp":
            if args.dataset in ["iris", "breast_cancer"]:
                in_dim = in_dim
            elif args.dataset == "mnist":
                in_dim = in_dim_or_shape
            else:  # CIFAR
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


    # ---------------------------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------------------------
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    elif args.optimizer == 'sgd+momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    
    elif args.optimizer == 'ano':
        from ano_optimizer import Ano
        optimizer = Ano(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # ---------------------------------------------------------------------
    # ALGORITHM
    # ---------------------------------------------------------------------
    
    if args.algo == "bp":
        algo = Backprop(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "lpl":  # lpl
        if args.dataset == "glue":
            from algorithms.local_probe_bert import LocalProbeBERT
            algo = LocalProbeBERT(base_optimizer=optimizer, probe_lr=args.lr)

        elif args.model == "mlp":
            algo = LocalProbeMLP(base_optimizer=optimizer, probe_lr=args.lr, weight_decay=args.weight_decay)

        else:
            try:
                from algorithms.local_probe_blocks import LocalProbeBlocks
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "LocalProbeBlocks not found. Create algorithms/local_probe_blocks.py "
                    "or run with --algo bp / --model mlp."
                ) from e
            algo = LocalProbeBlocks(base_optimizer=optimizer, probe_lr=args.probe_lr)
    
    elif args.algo == "kp":
        from algorithms.kp import KP
        algo = KP(learning_rate=args.lr)
        
    elif args.algo == "softhebb":
        from algorithms.softhebb import SoftHebb
        algo = SoftHebb(
            learning_rate=args.lr,
            head_lr=args.lr,
        )

    elif args.algo == "tp":
        from algorithms.targetprop import TargetPropagation
        fwd_optim = optimizer
        dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=True)
        inv_optim = torch.optim.SGD([dummy], lr=args.lr)
        algo = TargetPropagation(
            fwd_optimizer=fwd_optim,
            inv_optimizer=inv_optim,
            beta=1.0,
            noise_std=0.1,
        )
    elif args.algo == "fa":
        from algorithms.feedbackalignment import FeedbackAlignment
        algo = FeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)
        
    elif args.algo == 'dfa':
        from algorithms.dfa import DirectFeedbackAlignment
        algo = DirectFeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    trainer = Trainer(model, task, algo, device=args.device, input_noise_training=args.input_noise_training, verbose=bool(args.verbose))

    # ---------------------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------------------
    trainer.fit(train_loader, epochs=args.epochs, show_progress=True)

    # ---------------------------------------------------------------------
    # EVAL
    # ---------------------------------------------------------------------
    if args.dataset == "glue":
        # Evaluate each split (MNLI has matched + mismatched)
        for split_name, vloader in val_loaders.items():
            res = trainer.evaluate(vloader)
            # print agg + all raw metrics
            raw = " | ".join(f"{k}={v:.4f}" for k, v in res.items() if k not in ["agg_name"])
            print(f"[{split_name}] {raw}")

        # Optional: MNLI single number (average acc)
        if args.glue_task == "mnli" and "validation_matched" in val_loaders and "validation_mismatched" in val_loaders:
            m = trainer.evaluate(val_loaders["validation_matched"])["accuracy"]
            mm = trainer.evaluate(val_loaders["validation_mismatched"])["accuracy"]
            print(f"[mnli_avg] accuracy={(m + mm)/2:.4f}")

    else:
        print("Eval:", trainer.evaluate(test_loader))

    # ---------------------------------------------------------------------
    # ROBUSTNESS (only when we have Xte/yte tensors)
    # ---------------------------------------------------------------------
    if args.dataset in ["iris", "mnist", "cifar10", "cifar100", "breast_cancer"] and args.robustness != "none":
        sigmas_input = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_rel = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_w = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        def run(mode, sigmas):
            print(f"\n{mode}:")
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
                print(f"sigma={s:.2f} | acc={mean_acc*100:.2f}% ± {ci95*100:.2f}%")

        if args.robustness in ["input_noise", "all"]:
            run("input_noise", sigmas_input)
        if args.robustness in ["relative_input_noise", "all"]:
            run("relative_input_noise", sigmas_rel)
        if args.robustness in ["weight_noise", "all"]:
            run("weight_noise", sigmas_w)

    elif args.robustness != "none" and args.dataset not in ["iris", "mnist", "breast_cancer"]:
        print("Note: robustness sweeps are currently only implemented for iris/mnist tensor test sets.")


if __name__ == "__main__":
    main()