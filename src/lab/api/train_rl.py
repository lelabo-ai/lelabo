from __future__ import annotations

from argparse import Namespace
from typing import Any

from ..rl.algorithms import DQN, PPO
from ..update_rules import UpdateRuleContext, build_update_rule
from ..core.runners.rl_runner import RLRunner
from ..core.utils.envs import make_env, make_vec_env
from ..core.utils.logger import RunLogger
from ..optimizers import make_optimizer
from ..models.imported.actor_critic import ActorCriticDiscrete
from ..models.imported.qnet import QNet
from ..core.utils.seed import derive_seed
from .train_rl_config import build_rl_algo_config, parse_rl_param_overrides


def run_rl(args: Namespace, logger: RunLogger) -> dict[str, Any]:
    device = args.device
    raw_rl_params = list(getattr(args, "rl_param", []) or [])
    rl_overrides = parse_rl_param_overrides(raw_rl_params)
    train_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "train_env")
    eval_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "eval_env")

    if args.rl_algo == "dqn":
        env = make_env(args.env, seed=train_env_seed)
        eval_env = make_env(args.env, seed=eval_env_seed)

        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        qnet = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)
        optimizer = make_optimizer(args.optimizer, qnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ctx = UpdateRuleContext(
            args=args,
            model=qnet,
            task=None,
            optimizer=optimizer,
            mode="rl",
            dataset=args.dataset,
            rl_algo=args.rl_algo,
            extra={"grad_clip": None},
        )
        learner = build_update_rule(args.algo, ctx)

        cfg = build_rl_algo_config(args.rl_algo, rl_overrides)

        algo = DQN(q_net=qnet, learner=learner, cfg=cfg)
        algo.setup(obs_dim=obs_dim, n_actions=n_actions)

        runner = RLRunner(train_env=env, algo=algo, device=device, logger=logger, verbose=bool(args.verbose))
        return runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)

    if args.rl_algo == "ppo":
        cfg = build_rl_algo_config(args.rl_algo, rl_overrides)
        vec_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "vec_env")
        envs = make_vec_env(args.env, seed=vec_env_seed, num_envs=cfg.num_envs)
        eval_env = make_env(args.env, seed=eval_env_seed)

        obs_dim = int(envs.single_observation_space.shape[0])
        n_actions = int(envs.single_action_space.n)

        model = ActorCriticDiscrete(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=args.hidden, num_layers=args.layers)
        optimizer = make_optimizer(args.optimizer, model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        ctx = UpdateRuleContext(
            args=args,
            model=model,
            task=None,
            optimizer=optimizer,
            mode="rl",
            dataset=args.dataset,
            rl_algo=args.rl_algo,
        )
        learner = build_update_rule(args.algo, ctx)

        algo = PPO(actor_critic=model, learner=learner, cfg=cfg)
        runner = RLRunner(train_env=envs, algo=algo, device=device, logger=logger, verbose=bool(args.verbose))
        return runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)

    raise ValueError(f"Unknown rl algo: {args.rl_algo}")
