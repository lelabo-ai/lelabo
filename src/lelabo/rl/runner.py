# lab/runners/rl_runner.py
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional

from ..core.logger import RunLogger


class RLRunner:
    """
    Runner unique pour DQN (single env) ET PPO (vec env).
    L'algo doit exposer:
      - to(device)
      - total_steps (int)
      - collect(env_or_envs, device) -> (batch, info) avec info["episodes"] = [{"return":..., "length":...}, ...] optionnel
      - update(batch, device) -> stats (dict)
      - evaluate(eval_env, eval_episodes, device) -> dict
    """
    def __init__(
        self,
        train_env,
        algo,
        device: str,
        logger: Optional[RunLogger] = None,
        show_logs: bool = False,
        reward_window: int = 20,
        length_window: int = 20,
    ):
        self.env = train_env
        self.algo = algo
        self.device = device
        self.logger = logger or RunLogger(None)
        self.show_logs = show_logs

        self.reward_window = int(reward_window)
        self.length_window = int(length_window)

        self._last_returns = deque(maxlen=self.reward_window)
        self._last_lengths = deque(maxlen=self.length_window)

    @staticmethod
    def _mean(xs) -> float:
        xs = list(xs)
        return float(sum(xs) / len(xs)) if xs else 0.0

    @staticmethod
    def _safe_close(env: Any) -> None:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                # Closing errors should not hide training/eval outcomes.
                pass

    def train(
        self,
        total_steps: int,
        eval_env,
        eval_episodes: int = 5,
        log_every_updates: int = 1,
        print_every_updates: int = 1,   # <= important: print à chaque update
    ) -> Dict[str, Any]:
        self.algo.to(self.device)

        start = time.perf_counter()
        updates = 0
        try:
            while int(self.algo.total_steps) < int(total_steps):
                batch, info = self.algo.collect(self.env, device=self.device)

                # épisodes terminés (si l'algo en renvoie)
                for ep in info.get("episodes", []) or []:
                    # accepte return/length, sinon ignore silencieusement
                    if "return" in ep:
                        self._last_returns.append(float(ep["return"]))
                    if "length" in ep:
                        self._last_lengths.append(float(ep["length"]))

                    rec = {"t": "episode", "total_steps": int(self.algo.total_steps), **ep}
                    #self.logger.log(rec)

                stats = self.algo.update(batch, device=self.device)

                if stats:
                    updates += 1

                    # log jsonl
                    if (updates % log_every_updates) == 0:
                        rec = {"t": "rl_update", "total_steps": int(self.algo.total_steps), **stats}

                        rec["roll_mean_return_20"] = self._mean(self._last_returns)
                        rec["roll_mean_length_20"] = self._mean(self._last_lengths)

                        self.logger.log(rec)

                    # print console
                    if self.show_logs and (updates % print_every_updates) == 0:
                        mean_r = self._mean(self._last_returns)
                        mean_l = self._mean(self._last_lengths)

                        # affiche quelques métriques PPO si dispo
                        extra_keys = ["policy_loss", "value_loss", "entropy", "approx_kl", "loss"]
                        extras = []
                        for k in extra_keys:
                            if k in stats and isinstance(stats[k], (int, float)):
                                extras.append(f"{k}={float(stats[k]):.4g}")
                        extras_s = (" | " + " ".join(extras)) if extras else ""

                        print(
                            f"[update {updates}] steps={int(self.algo.total_steps)} "
                            f"mean_return(last {self.reward_window})={mean_r:.3f} "
                            f"mean_len(last {self.length_window})={mean_l:.2f}"
                            f"{extras_s}"
                        )

            total_time = time.perf_counter() - start
            sps = float(int(self.algo.total_steps) / total_time) if total_time > 0 else 0.0

            eval_stats = self.algo.evaluate(eval_env, eval_episodes=eval_episodes, device=self.device)
            self.logger.log({"t": "eval_rl", "total_steps": int(self.algo.total_steps), **eval_stats})

            return {
                "total_steps": int(self.algo.total_steps),
                "total_time_sec": float(total_time),
                "steps_per_sec": float(sps),
                "updates": int(updates),
                "eval": eval_stats,
            }
        finally:
            self._safe_close(self.env)
            if eval_env is not self.env:
                self._safe_close(eval_env)


def run_rl(args, logger: RunLogger) -> dict[str, Any]:
    from .algorithms import DQN, PPO
    from .config import build_rl_algo_config
    from .envs import make_env, make_vec_env
    from ..core.seed import derive_seed
    from ..models.builtins.actor_critic import ActorCriticDiscrete
    from ..models.builtins.qnet import QNet
    from ..optimizers import make_optimizer
    from ..update_rules import UpdateRuleContext, build_update_rule

    device = args.device
    display_mode = str(getattr(args, "display", "compact")).strip().lower()
    show_logs = display_mode != "none"
    rl_overrides = dict(getattr(args, "rl_params", {}) or {})
    optimizer_params = dict(getattr(args, "optimizer_params", {}) or {})
    lr = float(optimizer_params.pop("lr", args.lr))
    weight_decay = float(optimizer_params.pop("weight_decay", args.weight_decay))
    momentum = float(optimizer_params.pop("momentum", 0.9))
    train_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "train_env")
    eval_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "eval_env")

    if args.rl_algo == "dqn":
        env = make_env(args.env, seed=train_env_seed)
        eval_env = make_env(args.env, seed=eval_env_seed)

        obs_dim = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)
        qnet = QNet(obs_dim=obs_dim, n_actions=n_actions, hidden=args.hidden, layers=args.layers)
        optimizer = make_optimizer(
            args.optimizer,
            qnet.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            args=args,
            mode="rl",
            dataset=args.dataset,
            **optimizer_params,
        )
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
        runner = RLRunner(train_env=env, algo=algo, device=device, logger=logger, show_logs=show_logs)
        result = runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)
        if bool(getattr(args, "save_checkpoints", False)):
            logger.write_checkpoint(
                "last",
                logger.build_checkpoint_payload(
                    task="rl",
                    model=qnet,
                    learner=learner,
                    optimizer=optimizer,
                    epoch=int(result.get("total_steps", 0)),
                    meta={"kind": "last", "rl_algo": args.rl_algo},
                ),
            )
        return result

    if args.rl_algo == "ppo":
        cfg = build_rl_algo_config(args.rl_algo, rl_overrides)
        vec_env_seed = derive_seed(args.seed, "rl", args.rl_algo, "vec_env")
        envs = make_vec_env(args.env, seed=vec_env_seed, num_envs=cfg.num_envs)
        eval_env = make_env(args.env, seed=eval_env_seed)

        obs_dim = int(envs.single_observation_space.shape[0])
        n_actions = int(envs.single_action_space.n)
        model = ActorCriticDiscrete(obs_dim=obs_dim, n_actions=n_actions, hidden_dim=args.hidden, num_layers=args.layers)
        optimizer = make_optimizer(
            args.optimizer,
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            args=args,
            mode="rl",
            dataset=args.dataset,
            **optimizer_params,
        )
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
        runner = RLRunner(train_env=envs, algo=algo, device=device, logger=logger, show_logs=show_logs)
        result = runner.train(total_steps=args.rl_steps, eval_env=eval_env, eval_episodes=args.rl_eval_episodes)
        if bool(getattr(args, "save_checkpoints", False)):
            logger.write_checkpoint(
                "last",
                logger.build_checkpoint_payload(
                    task="rl",
                    model=model,
                    learner=learner,
                    optimizer=optimizer,
                    epoch=int(result.get("total_steps", 0)),
                    meta={"kind": "last", "rl_algo": args.rl_algo},
                ),
            )
        return result

    raise ValueError(f"Unknown rl algo: {args.rl_algo}")


__all__ = ["RLRunner", "run_rl"]
