#!/usr/bin/env python3
"""Where does the freshly-trained best_params stand against the genuinely strong
configs (the iron-triangle profiles)? Color-balanced A/B on 32x32, cap7.
best_params plays p0 on seed s and p1 on seed s+50000; opponent +100000.
"""
import json, math, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
CFG    = ROOT / "starter_kits/C++/train_engine_cap7.json"
# Candidate param file: argv[1] if given, else the single-opponent best.
CAND   = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else \
         ROOT / "starter_kits/C++/train_runs/best_params.txt"
OPPS   = {
    "eco":     ROOT / "starter_kits/C++/bot_params_eco.txt",
    "aggro":   ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
}
WORKERS = 14
SEEDS   = 120


def run_one(seed, pa, pb):
    try:
        p = subprocess.run(
            [str(ENGINE), "--width", "32", "--height", "32", "--turn-limit", "300",
             "--no-replay", "--no-logs", "--results-as-json", "-c", str(CFG),
             f"{BOT} {seed} {pa}", f"{BOT} {seed+100000} {pb}"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        js = json.loads(p.stdout)
        r0 = float(js["stats"]["0"]["rank"]); r1 = float(js["stats"]["1"]["rank"])
        return {"ok": True, "p0_won": r0 < r1, "tie": abs(r0 - r1) < 1e-9}
    except Exception:
        return {"ok": False}


def wilson_lb(w, n, z=1.96):
    if n == 0: return 0.0
    p = w / n
    return (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1 + z*z/n)


def run_pair(a_path, b_path, seeds):
    tasks = [(s, a_path, b_path, False) for s in range(seeds)] + \
            [(s + 50000, b_path, a_path, True) for s in range(seeds)]
    wa = total = ties = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, s, pa, pb): a1 for s, pa, pb, a1 in tasks}
        for f in as_completed(futs):
            a1 = futs[f]; r = f.result()
            if not r["ok"]: continue
            total += 1
            if r.get("tie"): ties += 1; continue
            if (not a1 and r["p0_won"]) or (a1 and not r["p0_won"]): wa += 1
    return wa, total - ties, ties


def main():
    print(f"candidate vs strong iron-triangle profiles. Seeds={SEEDS} ({2*SEEDS}/edge), 32x32 cap7")
    print(f"  cand = {CAND}")
    print("=" * 64)
    rates = {}
    for name, opp in OPPS.items():
        w, n, ties = run_pair(CAND, opp, SEEDS)
        wr = w / n if n else 0.0
        rates[name] = wr
        print(f"  cand vs {name:8} = {wr:5.1%}  (LB {wilson_lb(w,n):.1%})  [{w}/{n}, {ties} ties]", flush=True)
    if rates:
        print("-" * 64)
        print(f"  worst-case = {min(rates.values()):.1%}   average = {sum(rates.values())/len(rates):.1%}")


if __name__ == "__main__":
    main()
