# lab/main.py
import argparse
from pathlib import Path

import torch

from core.utils import RunLogger, seed_everything, make_env, make_vec_env, make_optimizer

# RL
from core.runners.rl_runner import RLRunner
from models.qnet import QNet
from models.actor_critic import ActorCriticDiscrete
from algorithms.update_rules.backprop import Backprop
from algorithms.rl.dqn import DQN, DQNConfig
from algorithms.rl.ppo import PPO, PPOAlgoConfig
from core.task import PPOConfig

# Supervised
from core.runners.supervised_runner import run_supervised


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # General
    p.add_argument("--dataset", choices=["breast_cancer", "iris", "mnist", "cifar10", "cifar100", "glue", "cartpole"], default="cartpole")
    p.add_argument("--model", choices=["mlp", "cnn", "resnet18", "resnet34", "resnet50", "bert"], default="mlp")
    p.add_argument("--algo", choices=["bp", "lpl", "kp", "softhebb", "tp", "fa", "dfa"], default="softhebb")

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--verbose", type=int, default=1)

    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd", "sgd+momentum", "ano"])
    p.add_argument("--weight-decay", type=float, default=0.01)

    p.add_argument("--input-noise-training", type=float, default=0.0, help="stddev gaussian noise on inputs during training")

    # RL common
    p.add_argument("--env", type=str, default="CartPole-v1")
    p.add_argument("--rl-steps", type=int, default=500_000)
    p.add_argument("--rl-eval-episodes", type=int, default=5)
    p.add_argument("--rl-algo", type=str, default="ppo", choices=["dqn", "ppo"])

    # DQN
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=1000)
    p.add_argument("--train-freq", type=int, default=1)
    p.add_argument("--target-update-freq", type=int, default=1000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-steps", type=int, default=50_000)
    p.add_argument("--rl-batch-size", type=int, default=256)

    # PPO
    p.add_argument("--ppo-num-envs", type=int, default=8)
    p.add_argument("--ppo-num-steps", type=int, default=128)
    p.add_argument("--ppo-update-epochs", type=int, default=4)
    p.add_argument("--ppo-num-minibatches", type=int, default=4)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--norm-adv", type=int, default=1)
    p.add_argument("--clip-vloss", type=int, default=1)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # Robustness (supervised)
    p.add_argument("--robustness", type=str, default="input_noise", choices=["none", "input_noise", "relative_input_noise", "weight_noise", "all"])
    p.add_argument("--noise-trials", type=int, default=30)

    # GLUE
    p.add_argument("--glue-task", type=str, default="cola",
                   choices=["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"])
    p.add_argument("--hf-model", type=str, default="bert-base-uncased")
    p.add_argument("--max-length", type=int, default=128)

    # Logging
    p.add_argument("--run-dir", type=str, default=None, help="writes metrics.jsonl + meta.json + summary.json")
    return p


def run_rl(args, logger: RunLogger):
    device = args.device

    if args.rl_algo == "dqn":
        env = make_env(args.env, seed=args.seed)
        eval_env = make_env(args.env, seed=args.seed + 10_000)

        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        qnet = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)
        optimizer = make_optimizer(args.optimizer, qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

        algo = DQN(q_net=qnet, learner=learner, cfg=cfg)
        algo.setup(obs_dim=obs_dim, n_actions=n_actions)

        runner = RLRunner(train_env=env, algo=algo, device=device, logger=logger, verbose=bool(args.verbose))
        return runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)

    if args.rl_algo == "ppo":
        envs = make_vec_env(args.env, seed=args.seed, num_envs=args.ppo_num_envs)
        eval_env = make_env(args.env, seed=args.seed + 10_000)

        obs_dim = int(envs.single_observation_space.shape[0])
        n_actions = int(envs.single_action_space.n)

        model = ActorCriticDiscrete(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)

        optimizer = make_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.algo == "bp":
            learner = Backprop(optimizer=optimizer, grad_clip=args.max_grad_norm)
        elif args.algo == "softhebb":
            from algorithms.update_rules.softhebb import SoftHebb
            learner = SoftHebb(learning_rate=args.lr, head_lr=args.lr)
        else:
            raise ValueError("PPO scaffold: --algo doit être bp ou softhebb")

        loss_cfg = PPOConfig(
            clip_coef=args.clip_coef,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            norm_adv=bool(args.norm_adv),
            clip_vloss=bool(args.clip_vloss),
            target_kl=args.target_kl,
        )
        cfg = PPOAlgoConfig(
            num_envs=args.ppo_num_envs,
            num_steps=args.ppo_num_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            update_epochs=args.ppo_update_epochs,
            num_minibatches=args.ppo_num_minibatches,
            ppo=loss_cfg,
        )

        algo = PPO(actor_critic=model, learner=learner, cfg=cfg)

        runner = RLRunner(train_env=envs, algo=algo, device=device, logger=logger, verbose=bool(args.verbose))
        out = runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)

        envs.close()
        return out

    raise ValueError(f"Unknown rl algo: {args.rl_algo}")


def main():
    args = build_parser().parse_args()

    seed_everything(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else None
    logger = RunLogger(run_dir=run_dir)
    logger.write_meta(vars(args))

    if args.dataset == "cartpole":
        rl_summary = run_rl(args, logger)
        summary = {"args": vars(args), "rl": {"algo": args.rl_algo, **rl_summary}}
        logger.write_summary(summary)
        return

    # supervised / glue
    summary = run_supervised(args, logger)
    logger.write_summary(summary)


if __name__ == "__main__":
    main()
