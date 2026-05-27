"""
train_ppo.py — train a raider policy with PPO over the real Halite engine.

The policy controls player 0 (RL) against a fixed C++ opponent (default: eco).
Observations are feature-plane tensors (see halite_env); actions are a spatial
action map. A small CNN feature extractor feeds PPO's MultiDiscrete policy head.

This is the training entry point for the B2 deep-RL path. CPU torch works but is
slow; use --n-envs to parallelise game rollouts. A converged raider needs many
hours — start with a short run to confirm the loss decreases and reward trends up.

KNOWN LIMITATION (found during validation, 2026-05-27): the flat spatial action
space MultiDiscrete([6]*W*H) is impractical for real training — 1024 independent
action heads mostly control EMPTY cells, so policy entropy pins at its maximum
(~1024*ln6 ≈ 1834 nats), the entropy term swamps reward, approx_kl explodes, and
the 6144-logit head drags throughput to ~5 fps on CPU. The loop is correct and
the bridge is solid, but the next real step is a better action representation:
a fully-convolutional spatial actor with action MASKING to our ship cells
(sb3-contrib MaskablePPO), or a per-entity policy that emits one action per ship.
Plus GPU + more envs for throughput. ent_coef is set very low below as a stopgap.

Examples:
    python rl/train_ppo.py --timesteps 20000 --n-envs 4 --turn-limit 150
    python rl/train_ppo.py --opponent control --timesteps 500000 --n-envs 8
"""

import argparse
import os

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from halite_env import HaliteRaiderEnv, N_CHANNELS


class SmallCNN(BaseFeaturesExtractor):
    """Compact conv stack for (C, 32, 32) feature-plane observations."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 32->16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 16->8
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.cnn(torch.zeros(1, *observation_space.shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.cnn(x))


def make_env(opponent, map_size, turn_limit, rank):
    def _init():
        return HaliteRaiderEnv(opponent=opponent, map_size=map_size,
                               turn_limit=turn_limit, seed=1000 + rank)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="eco", choices=["eco", "control", "aggro"])
    ap.add_argument("--timesteps", type=int, default=20000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--turn-limit", type=int, default=150)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--save", default=os.path.join(os.path.dirname(__file__), "ppo_raider"))
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    VecCls = DummyVecEnv if args.n_envs == 1 else SubprocVecEnv
    venv = VecCls([make_env(args.opponent, args.map_size, args.turn_limit, i)
                   for i in range(args.n_envs)])

    policy_kwargs = dict(
        features_extractor_class=SmallCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,        # obs are float feature planes, not images
        net_arch=dict(pi=[256], vf=[256]),
    )
    model = PPO(
        "CnnPolicy", venv, policy_kwargs=policy_kwargs,
        n_steps=args.n_steps, batch_size=args.n_steps * args.n_envs // 4,
        gamma=0.997, gae_lambda=0.95, ent_coef=1e-5, learning_rate=2.5e-4,
        verbose=1, device=device,
    )

    ckpt = CheckpointCallback(save_freq=max(1, 20000 // args.n_envs),
                              save_path=os.path.dirname(args.save),
                              name_prefix="ppo_raider_ck")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)
    model.save(args.save)
    venv.close()
    print(f"saved model to {args.save}.zip")


if __name__ == "__main__":
    main()
