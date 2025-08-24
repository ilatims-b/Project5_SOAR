# pcrl/envs/gym_compact.py
from __future__ import annotations

from typing import Any, Optional, Tuple, Dict
import gymnasium as gym

class GymnasiumAdapter(gym.Wrapper):
    """
    Wraps a legacy env to look like a gymnasium.Env and forwards special attrs
    (e.g., 'action_masks') required by sb3-contrib maskable algorithms.
    """

    def __init__(self, env):
        super().__init__(env)
        # If your legacy env exposes .observation_space/.action_space already,
        # these lines are optional. Otherwise, set them here:
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    # --------- required gymnasium API shims ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None and hasattr(self.env, "seed"):
            # legacy seed path
            try:
                self.env.seed(seed)
            except TypeError:
                pass
        obs = self.env.reset()
        # gymnasium expects (obs, info)
        return obs, {}

    def step(self, action):
        out = self.env.step(action)
        # Normalize to gymnasium 5-tuple
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        elif len(out) == 4:
            obs, reward, done, info = out
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Unexpected step() return format")
        return obs, reward, terminated, truncated, info

    # --------- critical: proxy 'action_masks' ----------
    @property
    def action_masks(self):
        """
        sb3-contrib calls vec_env.get_attr('action_masks') and expects a callable or mask.
        We forward from the legacy env.
        """
        am = getattr(self.env, "action_masks", None)
        if am is None:
            # Some envs expose a function like get_action_mask(obs)
            # You can turn that into a callable attribute here if needed.
            gm = getattr(self.env, "get_action_mask", None)
            if gm is not None:
                # Return a callable that computes masks from the latest obs or from input
                def _callable(*args, **kwargs):
                    # If the legacy function requires obs and none is provided,
                    # you may need to store the last obs in the adapter on reset/step.
                    return gm(*args, **kwargs)
                return _callable
        return am

    # --------- robust attribute forwarding ----------
    def get_wrapper_attr(self, name: str):
        # If the adapter has it, return it
        if hasattr(self, name):
            return getattr(self, name)
        # Otherwise, try the wrapped env (handles deep wrapper chains)
        if hasattr(self.env, "get_wrapper_attr"):
            try:
                return self.env.get_wrapper_attr(name)
            except AttributeError:
                pass
        if hasattr(self.env, name):
            return getattr(self.env, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
