"""
smoke_test.py — end-to-end check that the RL bridge drives a full engine game.

Runs HaliteRaiderEnv with random actions for one episode and prints progress.
No torch / SB3 needed — validates only the engine<->bot<->env plumbing.
"""

import time
from halite_env import HaliteRaiderEnv


def main():
    env = HaliteRaiderEnv(opponent="eco", turn_limit=120, seed=0)
    t0 = time.time()
    obs, info = env.reset()
    print(f"reset ok: obs shape {obs.shape}, sum {obs.sum():.1f}")
    total, steps = 0.0, 0
    term = trunc = False
    while not (term or trunc):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        total += r
        steps += 1
        if steps % 20 == 0 or term:
            print(f"  step {steps:3d}  r={r:+.3f}  total={total:+.2f}  info={info}")
        if steps > 200:
            print("  [safety break]")
            break
    env.close()
    print(f"DONE: {steps} steps, total reward {total:+.3f}, "
          f"{time.time()-t0:.1f}s, final info={info}")


if __name__ == "__main__":
    main()
