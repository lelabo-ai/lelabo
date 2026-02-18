# lab/core/envs.py
from __future__ import annotations

from typing import Callable
import gymnasium as gym
from lab.seed import derive_seed


def make_env(env_id: str, seed: int) -> gym.Env:
    root_seed = int(seed)
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.reset(seed=root_seed)
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(derive_seed(root_seed, "action_space"))
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(derive_seed(root_seed, "observation_space"))
    return env


def make_env_thunk(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def _thunk():
        return make_env(env_id, seed)
    return _thunk


def make_vec_env(env_id: str, seed: int, num_envs: int):
    thunks = [make_env_thunk(env_id, seed + i) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(thunks)
