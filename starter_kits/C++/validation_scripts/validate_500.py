#!/usr/bin/env python3
"""HIGH-N validation of the committed iron triangle (r14/c300) at 500 seeds
(1000 games/edge, 3000 games total) to resolve the true spread BELOW the
~7% map-draw variance that capped the 100/150-seed sweeps. Reads the live
bot_params_{aggro,eco,control}.txt directly (no overrides) so it validates
exactly what is committed.

Each edge is two seatings (a as p0 seed s; a as p1 seed s+50000; opp seed +100000)
for color-balance, same harness as sweep_balance4.
"""
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
CFG    = ROOT / "starter_kits/C++/competitive_engine_v2.json"
AGGRO   = ROOT / "starter_kits/C++/bot_params_aggro.txt"
ECO     = ROOT / "starter_kits/C++/bot_params_eco.txt"
CONTROL = ROOT / "starter_kits/C++/bot_params_control.txt"
WORKERS = 16
SEEDS = 500


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


def edge(name, a, b):
    w, t = run_pair(a, b, SEEDS)
    p = w/t if t else 0.0
    lb = wilson_lb(w, t)
    print(f"  {name:18s} {p:6.1%} +-{1.96*math.sqrt(p*(1-p)/t)*100 if t else 0:.1f}%  "
          f"Wilson95LB={lb:.1%}  ({w}/{t})", flush=True)
    return p, lb


def main():
    print(f"HIGH-N VALIDATION of committed triangle (r14/c300). "
          f"Workers={WORKERS} Seeds={SEEDS} ({2*SEEDS} games/edge)")
    print("=" * 72)
    ae, lb_ae = edge("aggro > eco:",     AGGRO, ECO)
    ec, lb_ec = edge("eco > control:",   ECO,   CONTROL)
    ca, lb_ca = edge("control > aggro:", CONTROL, AGGRO)
    edges = [ae, ec, ca]
    spread = max(edges) - min(edges)
    mn = min(edges)
    lbs = [lb_ae, lb_ec, lb_ca]
    print("=" * 72)
    print(f"  CYCLE={'True' if mn > 0.5 else 'False'}  "
          f"weakest={mn:.1%}  weakest_Wilson95LB={min(lbs):.1%}")
    print(f"  SPREAD={spread:.1%}  (max {max(edges):.1%} - min {min(edges):.1%})")
    print(f"  all edges >66%: {'YES' if all(e > 0.66 for e in edges) else 'NO'}")


if __name__ == "__main__":
    main()
