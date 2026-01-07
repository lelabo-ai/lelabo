# core/ppo_trainer.py
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PPOTrainer:
    def __init__(self, envs, algo, device: str = "cpu", run_dir: Optional[str] = None, verbose: bool = False):
        self.envs = envs
        self.algo = algo
        self.device = device
        self.verbose = verbose

        self.run_dir = Path(run_dir) if run_dir else None
        self.metrics_path = (self.run_dir / "metrics.jsonl") if self.run_dir else None

        self.recent_returns = deque(maxlen=20)
        self.recent_lengths = deque(maxlen=20)
        self.episode_count = 0

    def log(self, record: Dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        _append_jsonl(self.metrics_path, record)

    def _handle_episodes(self, episodes: List[Dict[str, float]]) -> None:
        for ep in episodes:
            self.episode_count += 1
            ep_r = float(ep.get("r", 0.0))
            ep_l = int(ep.get("l", 0))

            self.recent_returns.append(ep_r)
            self.recent_lengths.append(ep_l)

            avg_r = float(np.mean(self.recent_returns))
            avg_l = float(np.mean(self.recent_lengths))

            rec = {
                "t": "episode",
                "episode": int(self.episode_count),
                "total_steps": int(self.algo.global_step),
                "return": ep_r,
                "length": ep_l,
                "avg_return_20": avg_r,
                "avg_len_20": avg_l,
            }
            self.log(rec)

            if self.verbose:
                print(
                    f"[ep {self.episode_count:5d}] "
                    f"steps={self.algo.global_step:9d} "
                    f"R={ep_r:8.2f} "
                    f"L={ep_l:4d} "
                    f"avgR20={avg_r:8.2f}"
                )

    def train(self, total_timesteps: int, log_every: int = 1) -> Dict[str, Any]:
        self.algo.to(self.device)

        cfg = self.algo.cfg
        batch_steps = cfg.num_envs * cfg.num_steps
        num_iters = max(1, total_timesteps // batch_steps)

        start = time.perf_counter()
        _ = self.envs.reset()

        it_times = []

        for it in range(1, num_iters + 1):
            it_start = time.perf_counter()

            roll = self.algo.rollout(self.envs, device=self.device)

            episodes = roll.get("_episodes", [])
            if episodes:
                self._handle_episodes(episodes)

            adv, ret = self.algo.compute_gae(roll)
            stats = self.algo.update(roll, adv, ret, device=self.device)

            it_time = time.perf_counter() - it_start
            it_times.append(it_time)
            sps = batch_steps / it_time if it_time > 0 else 0.0

            if (it % log_every) == 0:
                avg_r20 = float(np.mean(self.recent_returns)) if self.recent_returns else float("nan")
                avg_str = "—" if not self.recent_returns else f"{avg_r20:8.2f}"

                rec = {
                    "t": "ppo_iter",
                    "iter": int(it),
                    "total_steps": int(self.algo.global_step),
                    "iter_time_sec": float(it_time),
                    "steps_per_sec": float(sps),
                    "avg_return_20": float(avg_r20) if self.recent_returns else None,
                    **stats,
                }
                self.log(rec)

                if self.verbose:
                    loss = float(stats.get("loss", float("nan")))
                    print(
                        f"[it {it:4d}/{num_iters}] "
                        f"steps={self.algo.global_step:9d} "
                        f"sps={sps:7.0f} "
                        f"avgR20={avg_str} "
                        f"loss={loss:8.4f}"
                    )

        total_time = time.perf_counter() - start
        total_steps = int(num_iters * batch_steps)
        steps_per_sec = float(total_steps / total_time) if total_time > 0 else 0.0

        summary = {
            "total_timesteps": total_steps,
            "total_time_sec": float(total_time),
            "steps_per_sec": steps_per_sec,
            "episodes": int(self.episode_count),
            "avg_return_20": float(np.mean(self.recent_returns)) if self.recent_returns else float("nan"),
            "avg_len_20": float(np.mean(self.recent_lengths)) if self.recent_lengths else float("nan"),
            "mean_iter_time_sec": float(np.mean(it_times)) if it_times else float("nan"),
        }
        return summary
