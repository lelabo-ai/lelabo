# lab/core/envs.py
from __future__ import annotations

from typing import Callable
import gymnasium as gym


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=seed)
    return env


def make_env_thunk(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def _thunk():
        return make_env(env_id, seed)
    return _thunk


def make_vec_env(env_id: str, seed: int, num_envs: int):
    thunks = [make_env_thunk(env_id, seed + i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(thunks)
