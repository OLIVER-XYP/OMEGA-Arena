#!/usr/bin/env python3
"""Validate the ADAPTIVE meta-bot against each fixed archetype.

The adaptive bot (bot_params_adaptive.txt) scouts the opponent then commits to
the iron-triangle counter. Goal: it should BEAT aggro (->control) and control
(->eco), and at least MIRROR eco (~50%, eco-vs-eco), never losing a matchup.

Same two-seating color-balanced harness as the other validators. Adaptive is
always 'a'; the fixed archetype is 'b'.
"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
CFG    = ROOT / "starter_kits/C++/competitive_engine_v2.json"
ADAPT  = ROOT / "starter_kits/C++/bot_params_adaptive.txt"
PROFILES = {
    "aggro":   ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "eco":     ROOT / "starter_kits/C++/bot_params_eco.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
}
WORKERS = 16
SEEDS = 100  # -> 200 games per archetype


def run_one(seed, pa, pb):
    import subprocess, json
    cmd_a = f"{BOT} {seed} {pa}"; cmd_b = f"{BOT} {seed + 100000} {pb}"
    try:
        p = subprocess.run(
            [str(ENGINE), "--width", "32", "--height", "32",
             "--turn-limit", "300", "--no-replay", "--no-logs",
             "--results-as-json", "-c", str(CFG), cmd_a, cmd_b],
            cwd=str(ROOT), capture_output=True, text=True, timeout=45)
        js = json.loads(p.stdout)
        r0 = float(js["stats"]["0"]["rank"]); r1 = float(js["stats"]["1"]["rank"])
        return {"ok": True, "p0_won": r0 < r1, "tie": abs(r0 - r1) < 1e-9}
    except Exception:
        return {"ok": False}


def run_pair(a_path, b_path, seeds):
    tasks = [(s, a_path, b_path, False) for s in range(seeds)] + \
            [(s + 50000, b_path, a_path, True) for s in range(seeds)]
    wa = total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, s, pa, pb): a_is_p1 for s, pa, pb, a_is_p1 in tasks}
        for f in as_completed(futs):
            a_is_p1 = futs[f]; r = f.result()
            if not r["ok"]: continue
            total += 1
            if r.get("tie"): continue
            if not a_is_p1:
                if r["p0_won"]: wa += 1
            else:
                if not r["p0_won"]: wa += 1
    return wa, total


def wilson_lb(w, t):
    if t == 0: return 0.0
    p = w/t; z = 1.96
    return (p + z*z/(2*t) - z*math.sqrt(p*(1-p)/t + z*z/(4*t*t))) / (1 + z*z/t)


def main():
    print(f"ADAPTIVE vs each archetype. Workers={WORKERS} Seeds={SEEDS} ({2*SEEDS} games each)")
    print("=" * 72)
    results = {}
    for name, prof in PROFILES.items():
        w, t = run_pair(ADAPT, prof, SEEDS)
        p = w/t if t else 0.0
        lb = wilson_lb(w, t)
        results[name] = (p, lb)
        verdict = "BEATS" if lb > 0.5 else ("mirror" if p > 0.42 else "LOSES")
        print(f"  adaptive vs {name:8s}: {p:6.1%}  Wilson95LB={lb:5.1%}  ({w}/{t})  -> {verdict}",
              flush=True)
    print("=" * 72)
    worst = min(results.values(), key=lambda x: x[0])
    print(f"  worst matchup = {worst[0]:.1%}")
    print(f"  beats aggro&control, >=mirror eco: "
          f"{'YES' if results['aggro'][1]>0.5 and results['control'][1]>0.5 and results['eco'][0]>0.42 else 'NO'}")


if __name__ == "__main__":
    main()
