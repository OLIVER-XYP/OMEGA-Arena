#!/usr/bin/env python3
"""
sweep_combat.py — sweep a single combat-reward knob and locate the cyclic window.

For each knob value we write a temp engine config (base competitive_engine.json with
one key overridden), run the three matchups (eco-aggro, eco-control, aggro-control),
and report the pairwise win rates plus whether a rock-paper-scissors cycle exists.

The premise: at low combat reward the metagame is transitive eco>control>aggro; at
high reward it is aggro>control>eco. The three pairwise matchups flip at *different*
knob thresholds, and a cycle lives in the window between the crossovers. This finds it.

Usage:
    python sweep_combat.py --key KILL_HALITE_BONUS_RATIO --values 0,0.5,1,2,4 --seeds 10
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import metagame_roundrobin as mr  # reuse run_pair / PROFILES / engine paths

ROOT = Path(__file__).resolve().parents[2]
BASE_CFG = ROOT / "starter_kits/C++/competitive_engine.json"


def write_cfg(overrides: dict, tmpdir: Path, tag: str) -> Path:
    cfg = json.loads(BASE_CFG.read_text())
    cfg.update(overrides)
    p = tmpdir / f"engine_{tag}.json"
    p.write_text(json.dumps(cfg, indent=2))
    return p


def winrate(name_a, name_b, seeds, mapsz, turns, cfg, workers):
    wa, wb, total = mr.run_pair(name_a, name_b, seeds, mapsz, turns, cfg, workers)
    return (wa / total if total else 0.5), total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="KILL_HALITE_BONUS_RATIO",
                    help="engine config key to sweep")
    ap.add_argument("--values", default="0,0.5,1,2,4",
                    help="comma-separated knob values")
    ap.add_argument("--extra", default="",
                    help="extra fixed overrides as k=v,k=v (applied to every config)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def parse_num(s):
        return float(s) if ("." in s or "e" in s.lower()) else int(s)

    values = [parse_num(v.strip()) for v in args.values.split(",") if v.strip()]
    extra = {}
    for kv in args.extra.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, v = kv.split("=")
        vl = v.strip().lower()
        if vl in ("true", "false"):
            extra[k.strip()] = (vl == "true")
        else:
            extra[k.strip()] = parse_num(v.strip())

    print(f"Sweep key={args.key}  values={values}  seeds={args.seeds} "
          f"(={2*args.seeds} games/matchup)  extra={extra or '{}'}")
    print(f"{'val':>7} | {'aggro>eco':>10} {'aggro>ctl':>10} {'ctl>eco':>10} | "
          f"{'eco avg':>8} {'ctl avg':>8} {'agg avg':>8} | result")
    print("-" * 96)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for v in values:
            ov = dict(extra)
            ov[args.key] = v
            cfg = write_cfg(ov, tmpdir, str(v).replace(".", "p"))

            ae, _ = winrate("aggro", "eco", args.seeds, args.map_size,
                            args.turn_limit, cfg, args.workers)
            ac, _ = winrate("aggro", "control", args.seeds, args.map_size,
                            args.turn_limit, cfg, args.workers)
            ce, _ = winrate("control", "eco", args.seeds, args.map_size,
                            args.turn_limit, cfg, args.workers)

            # average win rate per profile (vs the other two)
            eco_avg = ((1 - ae) + (1 - ce)) / 2
            ctl_avg = (ce + (1 - ac)) / 2
            agg_avg = (ae + ac) / 2

            # cycle detection on the 3 directed edges
            def edge(p):  # win rate -> ">" if >0.5
                return p > 0.5
            # cycle A: eco>aggro? aggro>control? control>eco?  -> aggro>ctl>eco>aggro
            cyc_aggro_ctl_eco = edge(ac) and edge(ce) and edge(1 - ae)  # aggro>ctl, ctl>eco, eco>aggro
            # cycle B: aggro>eco, eco>control, control>aggro
            cyc_aggro_eco_ctl = edge(ae) and edge(1 - ce) and edge(1 - ac)
            if cyc_aggro_ctl_eco:
                res = "CYCLE aggro>ctl>eco>aggro"
            elif cyc_aggro_eco_ctl:
                res = "CYCLE aggro>eco>ctl>aggro"
            else:
                # transitive: name the order by avg
                order = sorted([("eco", eco_avg), ("ctl", ctl_avg), ("agg", agg_avg)],
                               key=lambda t: -t[1])
                res = "transitive " + ">".join(n for n, _ in order)

            print(f"{v:>7} | {ae:>10.1%} {ac:>10.1%} {ce:>10.1%} | "
                  f"{eco_avg:>8.1%} {ctl_avg:>8.1%} {agg_avg:>8.1%} | {res}")


if __name__ == "__main__":
    main()
