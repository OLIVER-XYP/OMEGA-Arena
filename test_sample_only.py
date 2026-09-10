import json
import subprocess
import sys

engine = "game_engine/build_pyenv/halite"
ctl = "starter_kits/C++/build_linux/MyBot 12345 starter_kits/C++/bot_params_control.txt"
seed = 777

for tag, extra in [("greedy", ""), ("sample", "--sample")]:
    inf = (f"{sys.executable} rl_ai/inference_bot.py --checkpoint "
           "rl_ai/runs/ppo_parity_r2/ppo_best.pt --device cpu --feature-schema auto " + extra)
    p = subprocess.run(
        [engine, "--width", "32", "--height", "32", "--turn-limit", "300",
         "--no-replay", "--no-logs", "--no-timeout", "--results-as-json",
         "-c", "starter_kits/C++/competitive_engine_v2.json", "-s", str(seed),
         inf, ctl],
        capture_output=True, text=True, timeout=180, cwd=".")
    js = json.loads(p.stdout)
    s0 = js["stats"]["0"]["score"]
    print(f"inference_bot {tag}: score {s0} vs control {js['stats']['1']['score']} "
          f"({'WIN' if s0 > js['stats']['1']['score'] else 'LOSS'})")
