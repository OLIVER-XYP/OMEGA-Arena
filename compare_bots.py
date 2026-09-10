import json
import subprocess
import sys

engine = "game_engine/build_pyenv/halite"
ctl = "starter_kits/C++/build_linux/MyBot 12345 starter_kits/C++/bot_params_control.txt"
seed = 777

# learner_bot epsilon-greedy
for eps in (0.1, 0.3):
    learner = (f"{sys.executable} rl_ai/learner_bot.py --checkpoint "
               "rl_ai/runs/ppo_parity_r2/ppo_best.pt --out /tmp/comp_l.jsonl.gz "
               "--device cpu --epsilon " + str(eps))
    p = subprocess.run(
        [engine, "--width", "32", "--height", "32", "--turn-limit", "300",
         "--no-replay", "--no-logs", "--no-timeout", "--results-as-json",
         "-c", "starter_kits/C++/competitive_engine_v2.json", "-s", str(seed),
         learner, ctl],
        capture_output=True, text=True, timeout=180, cwd=".")
    js = json.loads(p.stdout)
    s0 = js["stats"]["0"]["score"]
    print(f"learner_bot eps={eps}: score {s0} vs control {js['stats']['1']['score']} "
          f"({'WIN' if s0 > js['stats']['1']['score'] else 'LOSS'})")

# inference_bot greedy (deployment eval)
inf = (f"{sys.executable} rl_ai/inference_bot.py --checkpoint "
       "rl_ai/runs/ppo_parity_r2/ppo_best.pt --device cpu --feature-schema auto")
p2 = subprocess.run(
    [engine, "--width", "32", "--height", "32", "--turn-limit", "300",
     "--no-replay", "--no-logs", "--no-timeout", "--results-as-json",
     "-c", "starter_kits/C++/competitive_engine_v2.json", "-s", str(seed),
     inf, ctl],
    capture_output=True, text=True, timeout=180, cwd=".")
js2 = json.loads(p2.stdout)
print("inference_bot greedy: score", js2["stats"]["0"]["score"],
      "vs control", js2["stats"]["1"]["score"])
