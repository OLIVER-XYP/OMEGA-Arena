#!/usr/bin/env python3
"""Final triangle validation at 100 seeds (200 games per edge).
Config: ratio=0.65, eco ENABLE_ATTACK=0, OLD interference mechanic.
"""
import json, math, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "game_engine/build/Release/halite.exe"
BOT    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
BASE_CFG = ROOT / "starter_kits/C++/competitive_engine_v2.json"
WORKERS = 16
SEEDS = 100


def run_one(seed, pa, pb, cfg_path):
    import subprocess
    cmd_a = f"{BOT} {seed} {pa}"
    cmd_b = f"{BOT} {seed + 100000} {pb}"
    try:
        p = subprocess.run(
            [str(ENGINE), "--width", "32", "--height", "32",
             "--turn-limit", "300", "--no-replay", "--no-logs",
             "--results-as-json", "-c", str(cfg_path), cmd_a, cmd_b],
            cwd=str(ROOT), capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        return {"ok": False}
    try:
        js = json.loads(p.stdout)
        r0 = float(js["stats"]["0"]["rank"])
        r1 = float(js["stats"]["1"]["rank"])
        return {"ok": True, "p0_won": r0 < r1, "tie": abs(r0 - r1) < 1e-9}
    except Exception:
        return {"ok": False}


def run_pair(name_a, name_b, seeds, cfg_path):
    pa = ROOT / f"starter_kits/C++/bot_params_{name_a}.txt"
    pb = ROOT / f"starter_kits/C++/bot_params_{name_b}.txt"
    tasks = [(s, pa, pb, False) for s in range(seeds)] + \
            [(s + 50000, pb, pa, True) for s in range(seeds)]
    wins_a = total = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_one, seed, pa, pb, cfg_path): a_is_p1
                   for seed, pa, pb, a_is_p1 in tasks}
        for f in as_completed(futures):
            a_is_p1 = futures[f]
            r = f.result()
            if not r["ok"]: continue
            total += 1
            if r.get("tie"): continue
            if not a_is_p1:
                if r["p0_won"]: wins_a += 1
            else:
                if not r["p0_won"]: wins_a += 1
    return wins_a, total


def wr_se(w, t):
    p = w/t if t else 0
    se = math.sqrt(p*(1-p)/t) if t else 0
    return p, se


def main():
    base = json.loads(BASE_CFG.read_text())
    print(f"FINAL VALIDATION: ratio={base['MINING_INTERFERENCE_RATIO']}, "
          f"cargo={base['MINING_INTERFERENCE_CARGO_LOSS_RATIO']}, "
          f"eco ENABLE_ATTACK=0")
    print(f"Workers={WORKERS}  Seeds={SEEDS} (={2*SEEDS} games per edge)")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        cfg_path = Path(td) / "cfg.json"
        cfg_path.write_text(json.dumps(base))

        w_ae, t_ae = run_pair("aggro", "eco", SEEDS, cfg_path)
        w_ec, t_ec = run_pair("eco", "control", SEEDS, cfg_path)
        w_ca, t_ca = run_pair("control", "aggro", SEEDS, cfg_path)

        p_ae, se_ae = wr_se(w_ae, t_ae)
        p_ec, se_ec = wr_se(w_ec, t_ec)
        p_ca, se_ca = wr_se(w_ca, t_ca)
        weakest = min(p_ae, p_ec, p_ca)
        cycle = p_ae > 0.5 and p_ec > 0.5 and p_ca > 0.5

        # Wilson 95% lower bounds
        def wilson_lb(w, t):
            if t == 0: return 0.0
            p = w/t; z = 1.96
            denom = 1 + z*z/t
            center = p + z*z/(2*t)
            margin = z * math.sqrt(p*(1-p)/t + z*z/(4*t*t))
            return (center - margin) / denom

        lb_ae = wilson_lb(w_ae, t_ae)
        lb_ec = wilson_lb(w_ec, t_ec)
        lb_ca = wilson_lb(w_ca, t_ca)

        print(f"  aggro > eco:      {p_ae:.1%} +-{se_ae:.1%}  Wilson95LB={lb_ae:.1%}  ({w_ae}/{t_ae})", flush=True)
        print(f"  eco > control:    {p_ec:.1%} +-{se_ec:.1%}  Wilson95LB={lb_ec:.1%}  ({w_ec}/{t_ec})", flush=True)
        print(f"  control > aggro:  {p_ca:.1%} +-{se_ca:.1%}  Wilson95LB={lb_ca:.1%}  ({w_ca}/{t_ca})", flush=True)
        print(f"  CYCLE={cycle}  weakest={weakest:.1%}  weakest_Wilson95LB={min(lb_ae,lb_ec,lb_ca):.1%}", flush=True)


if __name__ == "__main__":
    main()
