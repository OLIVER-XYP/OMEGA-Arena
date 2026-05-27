"""
halite_pettingzoo.py — PettingZoo ParallelEnv wrapper around the Halite engine.

Provides a proper multi-agent (2-player) interface for PettingZoo. The "rl"
agent controls player 0 through the bridge bot; the "opponent" is player 1
driven by a fixed C++ bot. In future this can be upgraded to two learning
agents for true self-play.

Usage (single-agent RL with fixed opponent):
    from halite_pettingzoo import HalitePZEnv
    env = HalitePZEnv(opponent="eco")
    obs, infos = env.reset()
    while env.agents:
        actions = {"rl": policy(obs["rl"])}
        obs, rewards, terms, truncs, infos = env.step(actions)
"""

import copy
import functools

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from halite_env import HaliteRaiderEnv, N_CHANNELS, N_ACTIONS


class HalitePZEnv(ParallelEnv):
    metadata = {"name": "halite_raider_v1", "render_modes": []}

    def __init__(self, opponent="eco", map_size=32, turn_limit=300, **kwargs):
        super().__init__()
        self._opponent = opponent
        self._map_size = map_size
        self._turn_limit = turn_limit
        self._env_kwargs = kwargs
        self._single_env: HaliteRaiderEnv | None = None

        W, H = map_size, map_size
        self._obs_space = spaces.Box(
            low=0.0, high=10.0, shape=(N_CHANNELS, H, W), dtype=np.float32)
        self._act_space = spaces.MultiDiscrete([N_ACTIONS] * (W * H))

        self.possible_agents = ["rl", "opponent"]
        self.agents: list[str] = []
        self._obs: dict[str, np.ndarray] = {}
        self._rewards: dict[str, float] = {}
        self._terminations: dict[str, bool] = {}
        self._truncations: dict[str, bool] = {}
        self._infos: dict[str, dict] = {}

    # ---- PettingZoo API -------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_space if agent == "rl" else spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        if self._single_env is not None:
            self._single_env.close()
        kwargs = dict(self._env_kwargs)
        if seed is not None:
            kwargs["seed"] = seed
        self._single_env = HaliteRaiderEnv(
            opponent=self._opponent, map_size=self._map_size,
            turn_limit=self._turn_limit, **kwargs)
        obs, info = self._single_env.reset()
        self.agents = list(self.possible_agents)
        self._obs = {"rl": obs, "opponent": np.zeros_like(obs)}
        self._infos = {"rl": info, "opponent": {}}
        self._rewards = {"rl": 0.0, "opponent": 0.0}
        self._terminations = {"rl": False, "opponent": False}
        self._truncations = {"rl": False, "opponent": False}
        return copy.copy(self._obs), copy.copy(self._infos)

    def step(self, actions):
        rl_act = actions.get("rl", self._act_space.sample())
        # opponent action is ignored (C++ bot drives opponent)

        obs, r, term, trunc, info = self._single_env.step(rl_act)

        if term or trunc:
            self.agents = []
            self._terminations["rl"] = bool(term)
            self._terminations["opponent"] = bool(term)
            self._truncations["rl"] = bool(trunc)
            self._truncations["opponent"] = bool(trunc)
            self._obs = {"rl": obs, "opponent": np.zeros_like(obs)}
            self._rewards = {"rl": float(r), "opponent": -float(r)}
            self._infos = {"rl": info, "opponent": {}}
        else:
            self._obs = {"rl": obs, "opponent": np.zeros_like(obs)}
            self._rewards = {"rl": float(r), "opponent": -float(r)}
            self._infos = {"rl": info, "opponent": {}}

        return (copy.copy(self._obs), copy.copy(self._rewards),
                copy.copy(self._terminations), copy.copy(self._truncations),
                copy.copy(self._infos))

    def close(self):
        if self._single_env is not None:
            self._single_env.close()
            self._single_env = None


# -- PettingZoo helpers for single-agent training with a fixed opponent ---

def env_fn(opponent="eco", **kwargs):
    """Return an uninstantiated env for use with SB3's VecEnv wrappers."""
    return HalitePZEnv(opponent=opponent, **kwargs)


def pz_to_gym(reset_ret):
    """Flatten a PettingZoo reset() return into (obs, info) for the first agent."""
    obs, infos = reset_ret
    return obs["rl"], infos["rl"]


if __name__ == "__main__":
    # Quick smoke test using the PettingZoo Parallel API.
    env = HalitePZEnv(opponent="eco", turn_limit=80, seed=42)
    obs, infos = env.reset()
    print("agents:", env.agents)
    print("obs rl shape:", obs["rl"].shape, "sum:", obs["rl"].sum())
    step = 0
    while env.agents:
        actions = {"rl": env.action_space("rl").sample()}
        obs, rewards, terms, truncs, infos = env.step(actions)
        step += 1
        if step == 1 or step % 20 == 0 or not env.agents:
            print(f"  step {step:3d}  r={rewards['rl']:+.3f}  "
                  f"term={terms['rl']}  info={infos['rl']}")
    env.close()
    print(f"DONE: {step} steps")
