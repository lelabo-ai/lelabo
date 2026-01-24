# lab/core/rl_trainer.py
from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from .utils.logger import RunLogger


def _get_total_steps(algo: Any) -> int:
    for name in ("total_steps", "global_step", "steps"):
        if hasattr(algo, name):
            v = getattr(algo, name)
            try:
                return int(v() if callable(v) else v)
            except Exception:
                pass
    return 0


class RLRunner:
    def __init__(self, env_or_envs, algo, device: str = "cpu", logger: Optional[RunLogger] = None, verbose: bool = False, avg_window: int = 20):
        self.env = env_or_envs
        self.algo = algo
        self.device = device
        self.verbose = verbose
        self.logger = logger or RunLogger(run_dir=None)

        self.recent_returns = deque(maxlen=int(avg_window))
        self.recent_lengths = deque(maxlen=int(avg_window))
        self.episode_count = 0
        self.update_count = 0

    def log(self, record: Dict[str, Any]) -> None:
        self.logger.log(record)

    def _handle_episodes(self, episodes) -> None:
        for ep in episodes:
            self.episode_count += 1
            ep_r = float(ep.get("return", ep.get("r", 0.0)))
            ep_l = int(ep.get("length", ep.get("l", 0)))
            self.recent_returns.append(ep_r)
            self.recent_lengths.append(ep_l)

            rec = {
                "t": "episode",
                "episode": int(self.episode_count),
                "total_steps": int(_get_total_steps(self.algo)),
                "return": ep_r,
                "length": ep_l,
                "avg_return": float(np.mean(self.recent_returns)) if self.recent_returns else float("nan"),
                "avg_len": float(np.mean(self.recent_lengths)) if self.recent_lengths else float("nan"),
            }
            self.log(rec)

            if self.verbose:
                print(f"[ep {self.episode_count:5d}] steps={rec['total_steps']:9d} R={ep_r:8.2f} L={ep_l:4d} avgR={rec['avg_return']:8.2f}")

    def train(self, total_steps: int, *, log_every_updates: int = 1, eval_env: Optional[Any] = None, eval_episodes: int = 5, eval_every_updates: Optional[int] = None) -> Dict[str, Any]:
        if hasattr(self.algo, "to"):
            self.algo.to(self.device)

        start = time.perf_counter()
        iter_times = []

        last_log_time = time.perf_counter()
        last_log_steps = _get_total_steps(self.algo)

        while _get_total_steps(self.algo) < int(total_steps):
            it_start = time.perf_counter()

            batch, info = self.algo.collect(self.env, device=self.device)
            episodes = (info or {}).get("episodes", [])
            if episodes:
                self._handle_episodes(episodes)

            stats = self.algo.update(batch, device=self.device) or {}
            it_time = time.perf_counter() - it_start
            iter_times.append(it_time)

            if stats:
                self.update_count += 1
                if (self.update_count % max(1, int(log_every_updates))) == 0:
                    now = time.perf_counter()
                    steps_now = _get_total_steps(self.algo)
                    dt = max(1e-12, now - last_log_time)
                    dsteps = max(0, steps_now - last_log_steps)
                    sps = float(dsteps / dt)
                    last_log_time = now
                    last_log_steps = steps_now

                    rec = {
                        "t": "rl_update",
                        "update": int(self.update_count),
                        "total_steps": int(steps_now),
                        "iter_time_sec": float(it_time),
                        "steps_per_sec": float(sps),
                        "avg_return": float(np.mean(self.recent_returns)) if self.recent_returns else None,
                        **{k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
                    }
                    self.log(rec)

            if eval_every_updates is not None and eval_env is not None:
                if self.update_count > 0 and (self.update_count % int(eval_every_updates) == 0):
                    self.evaluate(eval_env, eval_episodes=eval_episodes)

        total_time = time.perf_counter() - start
        steps = int(_get_total_steps(self.algo))
        sps = float(steps / total_time) if total_time > 0 else 0.0

        summary = {
            "total_steps": steps,
            "total_time_sec": float(total_time),
            "steps_per_sec": float(sps),
            "updates": int(self.update_count),
            "episodes": int(self.episode_count),
            "avg_return": float(np.mean(self.recent_returns)) if self.recent_returns else float("nan"),
            "avg_len": float(np.mean(self.recent_lengths)) if self.recent_lengths else float("nan"),
            "mean_iter_time_sec": float(np.mean(iter_times)) if iter_times else float("nan"),
        }

        if eval_env is not None:
            summary["eval"] = self.evaluate(eval_env, eval_episodes=eval_episodes)
        return summary

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int = 5) -> Dict[str, Any]:
        if hasattr(self.algo, "evaluate"):
            out = self.algo.evaluate(env, eval_episodes=eval_episodes, device=self.device)
            if isinstance(out, dict):
                self.log({"t": "eval_rl", **{k: float(v) for k, v in out.items() if isinstance(v, (int, float))}})
                return out

        returns = []
        for _ in range(int(eval_episodes)):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                if hasattr(self.algo, "act_deterministic"):
                    a = int(self.algo.act_deterministic(obs_t))
                elif hasattr(self.algo, "act"):
                    a = int(self.algo.act(obs_t))
                else:
                    raise AttributeError("Algo has no evaluate() and no act_deterministic/act fallback.")
                obs, reward, terminated, truncated, _ = env.step(a)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
            returns.append(ep_ret)

        mean_ret = float(np.mean(returns)) if returns else 0.0
        out = {"mean_return": mean_ret, "n": int(len(returns))}
        self.log({"t": "eval_rl", **out})
        return out


RLTrainer = RLRunner
