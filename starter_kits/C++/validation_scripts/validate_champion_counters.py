#!/usr/bin/env python3
"""Phase 0 of the champion-killer plan: find which scripted profile beats the
RL-eval champion (train_runs_behavior/best_params.txt).

Round-robin: each candidate vs champion_behavior at SEEDS paired games
(color-balanced, same harness as validate_500.py). The best candidate with
win-rate >= 0.60 becomes the BC expert for the killer policy; if none clears
0.60, fall back to a train_pipeline behavior-space search vs champion only.

Run:  python starter_kits/C++/validate_champion_counters.py
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
CFG    = ROOT / "starter_kits/C++/competitive_engine_v2.json"
KITS   = ROOT / "starter_kits/C++"
CHAMPION = KITS / "train_runs_behavior/best_params.txt"
CANDIDATES = {
    "eco":       KITS / "bot_params_eco.txt",
    "aggro":     KITS / "bot_params_aggro.txt",
    "control":   KITS / "bot_params_control.txt",
    "adaptive":  KITS / "bot_params_adaptive.txt",
    "super_eco": KITS / "train_runs/best_params.txt",
    "mixed":     KITS / "train_runs_mixed/best_params.txt",
}
WORKERS = 16
SEEDS = 100


def run_one(seed, pa, pb):
    import subprocess
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
    print(f"CHAMPION-COUNTER round-robin vs {CHAMPION.relative_to(ROOT)}")
    print(f"Workers={WORKERS} Seeds={SEEDS} ({2*SEEDS} games/candidate)")
    print("=" * 72)
    results = {}
    for name, path in CANDIDATES.items():
        w, t = run_pair(path, CHAMPION, SEEDS)
        p = w/t if t else 0.0
        lb = wilson_lb(w, t)
        results[name] = {"win_rate": p, "wilson_lb": lb, "wins": w, "games": t}
        print(f"  {name:10s} vs champion: {p:6.1%}  Wilson95LB={lb:.1%}  ({w}/{t})",
              flush=True)
    print("=" * 72)
    best = max(results, key=lambda k: results[k]["win_rate"])
    b = results[best]
    verdict = "EXPERT FOUND" if b["win_rate"] >= 0.60 else "NONE >=60% -> train_pipeline fallback"
    print(f"  best counter: {best} ({b['win_rate']:.1%}, LB {b['wilson_lb']:.1%}) — {verdict}")
    out = KITS / "champion_counters_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"champion": str(CHAMPION), "seeds": SEEDS, "results": results,
                   "best": best, "expert_found": b["win_rate"] >= 0.60}, fh, indent=1)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
