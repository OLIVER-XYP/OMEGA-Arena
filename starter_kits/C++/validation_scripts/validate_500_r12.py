#!/usr/bin/env python3
"""500-seed test of the BALANCE hypothesis flipped by validate_500.py:
eco>control (76.7%) is the true low edge and is aggro-INDEPENDENT, so weakening
aggro (r14->r12) should pull the two HIGH edges (aggro>eco, control>aggro) DOWN
toward the 76.7% anchor and shrink the spread.

Only the two aggro-dependent edges need re-measuring at r12; eco>control is
reused from validate_500.py (76.7%, aggro-independent).
"""
import math, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
CFG    = ROOT / "starter_kits/C++/competitive_engine_v2.json"
AGGRO   = ROOT / "starter_kits/C++/bot_params_aggro.txt"     # r14 (committed)
ECO     = ROOT / "starter_kits/C++/bot_params_eco.txt"
CONTROL = ROOT / "starter_kits/C++/bot_params_control.txt"   # c300
WORKERS = 16
SEEDS = 500
EC_ANCHOR = 0.767   # eco>control from validate_500.py (aggro-independent)


def make_variant(base_path, overrides, tmpdir, label):
    base = base_path.read_text().splitlines()
    keys = {k: str(v) for k, v in overrides.items()}
    out, seen = [], set()
    for line in base:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in keys:
                out.append(f"{k}={keys[k]}"); seen.add(k); continue
        out.append(line)
    for k, v in keys.items():
        if k not in seen: out.append(f"{k}={v}")
    p = tmpdir / f"{label}.txt"
    p.write_text("\n".join(out) + "\n")
    return p


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
    print(f"  {name:20s} {p:6.1%}  Wilson95LB={wilson_lb(w,t):.1%}  ({w}/{t})", flush=True)
    return p


def main():
    print(f"r12 BALANCE TEST. Workers={WORKERS} Seeds={SEEDS} ({2*SEEDS} games/edge)")
    print(f"eco>control anchor = {EC_ANCHOR:.1%} (aggro-independent, from validate_500)")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        a_r12 = make_variant(AGGRO, {"HUNT_MAX_RANGE": 12}, tmpdir, "aggro_r12")
        ae = edge("aggro>eco  r12:", a_r12, ECO)
        ca = edge("control>aggro r12:", CONTROL, a_r12)
        edges = [ae, EC_ANCHOR, ca]
        spread = max(edges) - min(edges)
        print("=" * 72)
        print(f"  r12 triangle: ae {ae:.1%} / ec {EC_ANCHOR:.1%} / ca {ca:.1%}")
        print(f"  SPREAD={spread:.1%}  min={min(edges):.1%}")
        print(f"  vs committed r14 spread=6.1% (ae 82.8 / ec 76.7 / ca 82.1)")
        if spread < 0.061:
            print(f"  --> r12 MORE BALANCED (spread {spread:.1%} < 6.1%)")
        else:
            print(f"  --> r12 NOT better (spread {spread:.1%} >= 6.1%)")


if __name__ == "__main__":
    main()
