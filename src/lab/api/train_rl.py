from __future__ import annotations

from argparse import Namespace
from typing import Any

from ..algorithms.rl.dqn import DQN, DQNConfig
from ..algorithms.rl.ppo import PPO, PPOAlgoConfig
from ..algorithms.update_rules import UpdateRuleContext, build_update_rule
from ..core.runners.rl_runner import RLRunner
from ..core.task import PPOConfig
from ..core.utils.envs import make_env, make_vec_env
from ..core.utils.logger import RunLogger
from ..core.utils.optim import make_optimizer
from ..models.actor_critic import ActorCriticDiscrete
from ..models.qnet import QNet
from ..seed import derive_seed


def run_rl(args: Namespace, logger: RunLogger) -> dict[str, Any]:
    device = args.device
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
        learner = build_update_rule("bp", ctx)

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
        vec_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "vec_env")
        envs = make_vec_env(args.env, seed=vec_env_seed, num_envs=args.ppo_num_envs)
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

