# pcrl/envs/make_env.py (or wherever this lives)
import os
from typing import Any, Callable, Dict, Optional, Type, List, Union

import gymnasium as gym  # <-- use gymnasium
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

from data.instruction_pool import Sample
from pcrl.envs.gym_compact import GymnasiumAdapter  # ensure this file exists (as we added)
from pcrl.envs.env import BatchPromptCompEnv


def make_vec_env(
    env_ctor: Union[Type[BatchPromptCompEnv], Callable[..., BatchPromptCompEnv]],
    n_envs: int,
    samples: List[Sample],
    sample_k: int = 1,
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    monitor_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Build a SubprocVecEnv of Gymnasium-wrapped environments.
    env_ctor: the legacy env class (e.g., BatchPromptCompEnv)
    """
    assert n_envs % sample_k == 0, "n_envs must be divisible by sample_k"

    env_kwargs = {} if env_kwargs is None else env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs

    # Split samples into (n_envs // sample_k) chunks;
    # each chunk is shared by 'sample_k' worker envs.
    chunks = []
    n_chunks = n_envs // sample_k
    per_chunk = max(1, len(samples) // max(1, n_chunks))
    for i in range(n_chunks):
        chunks.append(samples[per_chunk * i : per_chunk * (i + 1)])
    if not chunks:
        chunks = [samples]  # fallback: single chunk

    def make_env(rank: int):
        def _init():
            # 1) Build legacy env
            legacy_env = env_ctor(
                rank=rank,
                samples=chunks[rank // sample_k],
                **env_kwargs,
            )

            # 2) Wrap to Gymnasium BEFORE Monitor
            env = GymnasiumAdapter(legacy_env)

            # 3) Seeding (Gymnasium-friendly)
            if seed is not None:
                try:
                    env.reset(seed=seed + rank)
                except Exception:
                    # legacy paths already seeded
                    pass
                if hasattr(env.action_space, "seed"):
                    env.action_space.seed(seed + rank)

            # 4) Monitor wrapper (SB3 expects gymnasium.Env here)
            monitor_path = None
            if monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
                monitor_path = os.path.join(monitor_dir, str(rank))
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            return env
        return _init

    return SubprocVecEnv([make_env(i) for i in range(n_envs)])
