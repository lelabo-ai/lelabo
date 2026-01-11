# lab/core/rl_trainer.py
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class RLTrainer:
    def __init__(self, env, algo, device: str = "cpu", run_dir: Optional[str] = None, verbose: bool = False):
        self.env = env
        self.algo = algo
        self.device = device
        self.verbose = verbose

        self.run_dir = Path(run_dir) if run_dir else None
        self.metrics_path = (self.run_dir / "metrics.jsonl") if self.run_dir else None

    def log(self, record: Dict[str, Any]) -> None:
        if self.metrics_path is None:
            return
        _append_jsonl(self.metrics_path, record)

    def train(self, total_steps: int = 50_000, eval_episodes: int = 5) -> Dict[str, Any]:
        self.algo.to(self.device)

        obs, _ = self.env.reset()
        ep_return = 0.0
        ep_len = 0
        ep_idx = 0

        start = time.perf_counter()
        updates = 0

        while self.algo.total_steps < total_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
            action = self.algo.act(obs_t)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = bool(terminated or truncated)

            self.algo.observe(obs, action, float(reward), next_obs, done)

            ep_return += float(reward)
            ep_len += 1

            stats = self.algo.maybe_update(device=self.device)
            if stats:
                updates += 1
                self.log({"t": "rl_update", "step": int(self.algo.total_steps), **stats})

            obs = next_obs

            if done:
                ep_idx += 1
                self.log({"t": "episode", "episode": int(ep_idx), "step": int(self.algo.total_steps),
                          "return": float(ep_return), "length": int(ep_len)})
                if self.verbose:
                    print(f"episode={ep_idx} step={self.algo.total_steps} return={ep_return:.1f} len={ep_len}")

                obs, _ = self.env.reset()
                ep_return = 0.0
                ep_len = 0

        total_time = time.perf_counter() - start
        steps = int(self.algo.total_steps)
        sps = steps / total_time if total_time > 0 else 0.0

        eval_stats = self.evaluate(eval_episodes=eval_episodes)

        summary = {
            "total_steps": steps,
            "total_time_sec": float(total_time),
            "steps_per_sec": float(sps),
            "updates": int(updates),
            "eval": eval_stats,
        }
        return summary

    @torch.no_grad()
    def evaluate(self, eval_episodes: int = 5) -> Dict[str, Any]:
        returns = []
        lengths = []

        for _ in range(eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_ret = 0.0
            ep_len = 0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
                qvals = self.algo.q(obs_t.unsqueeze(0))
                action = int(torch.argmax(qvals, dim=-1).item())

                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1

            returns.append(ep_ret)
            lengths.append(ep_len)

        mean_ret = float(np.mean(returns)) if returns else 0.0
        ci95 = 0.0
        if len(returns) > 1:
            m = mean_ret
            var = sum((x - m) ** 2 for x in returns) / (len(returns) - 1)
            se = (var ** 0.5) / (len(returns) ** 0.5)
            ci95 = float(1.96 * se)

        out = {"mean_return": mean_ret, "ci95": float(ci95), "n": int(len(returns))}
        self.log({"t": "eval_rl", **out})
        return out
