#!/usr/bin/env python3
"""Phase 0 fallback: targeted sweep for a champion_behavior counter.

Round 1 found no stock profile >=60% (best: eco/adaptive 54.5%). Champion's
style = economy + 4 campers(T16) + hunters(r10) + defend. Hypothesis: the
counter is an ECONOMY bot with DEFENSE — punish its campers via defend
retaliation + dodge hunters via threat-avoid, while out-mining the ~4 ships
it wastes on camping. Defend is NOT gated by ENABLE_ATTACK (MyBot.cpp:712).

Variants are generated from bot_params_eco.txt / champion params and played
vs champion at SEEDS paired games (validate_500.py harness).
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
ECO = KITS / "bot_params_eco.txt"
OUT_DIR = KITS / "sweep_killer"
WORKERS = 16
SEEDS = 100


def load_params(path):
    params = {}
    order = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        params[key] = value.split("#")[0].strip()
        order.append(key)
    return params, order


def write_variant(name, base_path, overrides):
    params, order = load_params(base_path)
    for k, v in overrides.items():
        if k not in params:
            order.append(k)
        params[k] = str(v)
    out = OUT_DIR / f"{name}.txt"
    out.write_text("".join(f"{k}={params[k]}\n" for k in order), encoding="ascii")
    return out


DEFEND = {"DEFEND_MIN_CARGO": 150, "DEFEND_MIN_CELL_HALITE": 50, "DEFEND_TRIGGER_RANGE": 4}
TA = {"THREAT_AVOID_WEIGHT": 3.0}


def build_variants():
    OUT_DIR.mkdir(exist_ok=True)
    return {
        "eco_defend":       write_variant("eco_defend", ECO, DEFEND),
        "eco_ta":           write_variant("eco_ta", ECO, TA),
        "eco_defend_ta":    write_variant("eco_defend_ta", ECO, {**DEFEND, **TA}),
        "champ_ta":         write_variant("champ_ta", CHAMPION, TA),
        "champ_nocamp":     write_variant("champ_nocamp", CHAMPION, {"CAMP_ENABLED": 0}),
        "champ_nocamp_ta":  write_variant("champ_nocamp_ta", CHAMPION, {"CAMP_ENABLED": 0, **TA}),
    }


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
    variants = build_variants()
    print(f"KILLER SWEEP vs {CHAMPION.relative_to(ROOT)}  "
          f"Workers={WORKERS} Seeds={SEEDS} ({2*SEEDS} games/variant)")
    print("=" * 72)
    results = {}
    for name, path in variants.items():
        w, t = run_pair(path, CHAMPION, SEEDS)
        p = w/t if t else 0.0
        lb = wilson_lb(w, t)
        results[name] = {"win_rate": p, "wilson_lb": lb, "wins": w, "games": t,
                         "params": str(path)}
        print(f"  {name:16s} vs champion: {p:6.1%}  Wilson95LB={lb:.1%}  ({w}/{t})",
              flush=True)
    print("=" * 72)
    best = max(results, key=lambda k: results[k]["win_rate"])
    b = results[best]
    verdict = "EXPERT FOUND" if b["win_rate"] >= 0.60 else "still <60%"
    print(f"  best: {best} ({b['win_rate']:.1%}, LB {b['wilson_lb']:.1%}) — {verdict}")
    out = KITS / "killer_sweep_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"champion": str(CHAMPION), "seeds": SEEDS, "results": results,
                   "best": best}, fh, indent=1)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
