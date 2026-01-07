# lab/main.py
import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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

# Algorithms (update rules)
from algorithms.update_rules.backprop import Backprop
from algorithms.update_rules.local_probe_mlp import LocalProbeMLP

# RL - DQN
from core.rl_trainer import RLTrainer
from models.qnet import QNet
from algorithms.rl.dqn import DQN, DQNConfig

# RL - PPO (modules à créer comme on a fait)
from models.actor_critic import ActorCriticDiscrete
from core.task import PPOConfig
from algorithms.rl.ppo import PPO, PPOAlgoConfig
from core.ppo_trainer import PPOTrainer


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        "--dataset",
        choices=["breast_cancer", "iris", "mnist", "cifar10", "cifar100", "glue", "cartpole"],
        default="cartpole",
    )
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet18", "resnet34", "resnet50", "bert"], default="mlp")
    parser.add_argument("--algo", choices=["bp", "lpl", "kp", "softhebb", "tp", "fa", "dfa"], default="softhebb")

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd+momentum", "ano"])
    parser.add_argument(
        "--input-noise-training",
        type=float,
        default=2,
        help="stddev of gaussian noise added to inputs during training",
    )

    # ---------------- RL common ----------------
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--rl-steps", type=int, default=500_000)
    parser.add_argument("--rl-eval-episodes", type=int, default=5)

    parser.add_argument("--rl-algo", type=str, default="ppo", choices=["dqn", "ppo"])

    # DQN hyperparams
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=50_000)
    parser.add_argument("--rl-batch-size", type=int, default=256)

    # PPO hyperparams (CartPole discret)
    parser.add_argument("--ppo-num-envs", type=int, default=8)
    parser.add_argument("--ppo-num-steps", type=int, default=128)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--ppo-num-minibatches", type=int, default=4)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--norm-adv", type=int, default=1)      # 1/0
    parser.add_argument("--clip-vloss", type=int, default=1)    # 1/0
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Robustness (supervised only)
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

    # optimizer extras
    parser.add_argument("--weight-decay", type=float, default=0.01)

    # Logging
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="If set, writes metrics.jsonl + summary.json into this directory.",
    )

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": _safe_git_commit(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "args": vars(args),
            "argv": list(map(str, os.sys.argv)),
        }
        _write_json(run_dir / "meta.json", meta)

    # ---------------------------------------------------------------------
    # RL branch (CartPole)
    # ---------------------------------------------------------------------
    if args.dataset == "cartpole":
        import gymnasium as gym

        # ---- helper: env factory (clean seed) ----
        def make_env(seed: int):
            def _thunk():
                env = gym.make(args.env)
                # record episodic returns in info["episode"]
                env = gym.wrappers.RecordEpisodeStatistics(env)
                env.reset(seed=seed)
                return env
            return _thunk

        device = args.device

        # ---------------- DQN ----------------
        if args.rl_algo == "dqn":
            env = gym.make(args.env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.reset(seed=args.seed)

            obs_dim = int(env.observation_space.shape[0])
            n_actions = int(env.action_space.n)

            qnet = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)

            # Optimizer
            if args.optimizer == "adamw":
                optimizer = torch.optim.AdamW(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd+momentum":
                optimizer = torch.optim.SGD(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
            elif args.optimizer == "ano":
                from ano_optimizer import Ano
                optimizer = Ano(qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")

            learner = Backprop(optimizer=optimizer, grad_clip=None)

            cfg = DQNConfig(
                gamma=args.gamma,
                batch_size=args.rl_batch_size,
                buffer_size=args.buffer_size,
                learning_starts=args.learning_starts,
                train_freq=args.train_freq,
                target_update_freq=args.target_update_freq,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_steps=args.eps_decay_steps,
            )

            dqn = DQN(q_net=qnet, learner=learner, cfg=cfg)
            dqn.setup(obs_dim=obs_dim, n_actions=n_actions)

            rl_trainer = RLTrainer(env=env, algo=dqn, device=device, run_dir=args.run_dir, verbose=bool(args.verbose))
            rl_summary = rl_trainer.train(total_steps=args.rl_steps, eval_episodes=args.rl_eval_episodes)

            out = {"args": vars(args), "rl": {"algo": "dqn", **rl_summary}}
            if run_dir:
                _write_json(run_dir / "summary.json", out)
            return

        # ---------------- PPO ----------------
        if args.rl_algo == "ppo":
            # Vector envs
            envs = gym.vector.SyncVectorEnv([make_env(args.seed + i) for i in range(args.ppo_num_envs)])

            obs_dim = int(envs.single_observation_space.shape[0])
            n_actions = int(envs.single_action_space.n)

            model = ActorCriticDiscrete(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)

            # Optimizer
            if args.optimizer == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer == "sgd+momentum":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
            elif args.optimizer == "ano":
                from ano_optimizer import Ano
                optimizer = Ano(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                raise ValueError(f"Unknown optimizer: {args.optimizer}")

            # Learner = ton update rule existant
            # (PPO typiquement clip grad norm à 0.5)
            if args.algo == 'bp':
                learner = Backprop(optimizer=optimizer, grad_clip=args.max_grad_norm)
            elif args.algo == 'softhebb':
                from algorithms.update_rules.softhebb import SoftHebb
                learner = SoftHebb(learning_rate=args.lr, head_lr=args.lr)
            else:
                raise ValueError(f"PPO only supports --algo bp or softhebb in this scaffold.")

            ppo_loss_cfg = PPOConfig(
                clip_coef=args.clip_coef,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                norm_adv=bool(args.norm_adv),
                clip_vloss=bool(args.clip_vloss),
                target_kl=args.target_kl,
            )
            ppo_cfg = PPOAlgoConfig(
                num_envs=args.ppo_num_envs,
                num_steps=args.ppo_num_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                update_epochs=args.ppo_update_epochs,
                num_minibatches=args.ppo_num_minibatches,
                ppo=ppo_loss_cfg,
            )

            algo = PPO(actor_critic=model, learner=learner, cfg=ppo_cfg)

            # PPO trainer
            ppo_trainer = PPOTrainer(envs=envs, algo=algo, device=device, run_dir=args.run_dir, verbose=bool(args.verbose))

            # total timesteps = args.rl_steps (on arrondit au multiple du batch PPO)
            rl_summary = ppo_trainer.train(total_timesteps=args.rl_steps, log_every=1)

            out = {"args": vars(args), "rl": {"algo": "ppo", **rl_summary}}
            if run_dir:
                _write_json(run_dir / "summary.json", out)
            envs.close()
            return

        raise ValueError(f"Unknown rl-algo: {args.rl_algo}")

    # ---------------------------------------------------------------------
    # SUPERVISED branch (unchanged)
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
        test_loader = None

    # MODEL + TASK
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

    # OPTIMIZER
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd+momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "ano":
        from ano_optimizer import Ano
        optimizer = Ano(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # ALGORITHM
    if args.algo == "bp":
        algo = Backprop(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "lpl":
        if args.dataset == "glue":
            from algorithms.update_rules.local_probe_bert import LocalProbeBERT
            algo = LocalProbeBERT(base_optimizer=optimizer, probe_lr=args.lr)
        elif args.model == "mlp":
            algo = LocalProbeMLP(base_optimizer=optimizer, probe_lr=args.lr, weight_decay=args.weight_decay)
        else:
            try:
                from algorithms.update_rules.local_probe_blocks import LocalProbeBlocks
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "LocalProbeBlocks not found. Create algorithms/local_probe_blocks.py "
                    "or run with --algo bp / --model mlp."
                ) from e
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
        algo = TargetPropagation(
            fwd_optimizer=optimizer,
            inv_optimizer=inv_optim,
            beta=1.0,
            noise_std=0.1,
        )

    elif args.algo == "fa":
        from algorithms.update_rules.feedbackalignment import FeedbackAlignment
        algo = FeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    elif args.algo == "dfa":
        from algorithms.update_rules.dfa import DirectFeedbackAlignment
        algo = DirectFeedbackAlignment(optimizer=optimizer, grad_clip=1.0 if args.dataset in ["mnist", "glue"] else None)

    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    trainer = Trainer(
        model,
        task,
        algo,
        device=args.device,
        input_noise_training=args.input_noise_training,
        verbose=bool(args.verbose),
        run_dir=str(run_dir) if run_dir else None,
    )

    summary: Dict[str, Any] = {"args": vars(args)}

    # TRAIN
    train_stats = trainer.fit(train_loader, epochs=args.epochs, show_progress=True)
    summary["train"] = train_stats

    # EVAL
    eval_block: Dict[str, Any] = {}

    if args.dataset == "glue":
        for split_name, vloader in val_loaders.items():
            res = trainer.evaluate(vloader, split=split_name)
            raw = " | ".join(f"{k}={v:.4f}" for k, v in res.items() if k not in ["agg_name"])
            print(f"[{split_name}] {raw}")
            eval_block[split_name] = res

        if args.glue_task == "mnli" and "validation_matched" in val_loaders and "validation_mismatched" in val_loaders:
            m = trainer.evaluate(val_loaders["validation_matched"], split="validation_matched")["accuracy"]
            mm = trainer.evaluate(val_loaders["validation_mismatched"], split="validation_mismatched")["accuracy"]
            eval_block["mnli_avg"] = {"accuracy": float((m + mm) / 2.0)}
            print(f"[mnli_avg] accuracy={(m + mm)/2:.4f}")

    else:
        res = trainer.evaluate(test_loader, split="test")
        print("Eval:", res)
        eval_block["test"] = res

    summary["eval"] = eval_block

    # ROBUSTNESS (store results into summary.json)
    robustness_block: Dict[str, Any] = {}

    if args.dataset in ["iris", "mnist", "cifar10", "cifar100", "breast_cancer"] and args.robustness != "none":
        sigmas_input = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_rel = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sigmas_w = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

        def run(mode, sigmas):
            print(f"\n{mode}:")
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
                print(f"sigma={s:.2f} | acc={mean_acc*100:.2f}% ± {ci95*100:.2f}%")
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

    # SAVE SUMMARY
    if run_dir:
        _write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
