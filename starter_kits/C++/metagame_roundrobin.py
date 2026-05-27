#!/usr/bin/env python3
"""
metagame_roundrobin.py — Round-robin metagame health study.

Runs each strategy pair for SEEDS symmetric pairs (A@pos0 vs B, then B@pos0 vs A)
and outputs a win-rate matrix W[i][j] = win rate of profile i against profile j.

Metagame is healthy when:
  - No single profile dominates (avg win rate < ~60%)
  - There is cyclic dominance (A>B, B>C, C>A)
  - Strategy entropy of the Nash equilibrium distribution is high

Usage:
    python metagame_roundrobin.py [--seeds 20] [--map-size 32] [--turn-limit 300]
                                  [--workers 6] [--engine-cfg competitive_engine.json]
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

ROOT       = Path(__file__).resolve().parents[2]
ENGINE_EXE = ROOT / "game_engine/build/Release/halite.exe"
BOT_EXE    = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"

PROFILES = {
    "eco":     ROOT / "starter_kits/C++/bot_params_eco.txt",
    "aggro":   ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
}


def parse_result(stdout: str) -> dict | None:
    try:
        js = json.loads(stdout)
    except Exception:
        return None
    stats = js.get("stats", {})
    if "0" not in stats or "1" not in stats:
        return None
    r0 = float(stats["0"].get("rank", 2))
    r1 = float(stats["1"].get("rank", 2))
    sc0 = float(stats["0"].get("score", 0))
    sc1 = float(stats["1"].get("score", 0))
    return {
        "p0_won": r0 < r1,
        "tie": abs(r0 - r1) < 1e-9,
        "sc0": sc0, "sc1": sc1,
    }


def run_game(seed: int, params_a: Path, params_b: Path,
             map_size: int, turn_limit: int, engine_cfg: Path) -> dict:
    """Run one game: params_a is player 0, params_b is player 1."""
    cmd_a = f"{BOT_EXE} {seed} {params_a}"
    cmd_b = f"{BOT_EXE} {seed + 100000} {params_b}"
    cmd = [
        str(ENGINE_EXE),
        "--width", str(map_size), "--height", str(map_size),
        "--turn-limit", str(turn_limit),
        "--no-replay", "--no-logs", "--results-as-json",
        "-c", str(engine_cfg),
        cmd_a, cmd_b,
    ]
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    r = parse_result(p.stdout)
    if r is None:
        return {"ok": False, "rc": p.returncode, "stderr": p.stderr[:200]}
    r["ok"] = True
    return r


def run_pair(name_a: str, name_b: str, seeds: int,
             map_size: int, turn_limit: int, engine_cfg: Path,
             workers: int) -> tuple[int, int, int]:
    """
    Return (wins_a, wins_b, total_valid) over 2*seeds games (symmetric positions).
    """
    params_a = PROFILES[name_a]
    params_b = PROFILES[name_b]

    tasks: list[tuple[int, Path, Path, bool]] = []
    for s in range(seeds):
        tasks.append((s,         params_a, params_b, False))  # A=p0, B=p1
        tasks.append((s + 50000, params_b, params_a, True))   # B=p0, A=p1

    wins_a = wins_b = total = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(run_game, seed, pa, pb, map_size, turn_limit, engine_cfg): a_is_p1
            for seed, pa, pb, a_is_p1 in tasks
        }
        for f in as_completed(futures):
            a_is_p1 = futures[f]
            r = f.result()
            if not r["ok"]:
                print(f"  [warn] game failed: rc={r.get('rc')} — {r.get('stderr','')}")
                continue
            total += 1
            if r["tie"]:
                continue
            p0_won = r["p0_won"]
            if not a_is_p1:
                if p0_won: wins_a += 1
                else:      wins_b += 1
            else:
                if p0_won: wins_b += 1
                else:      wins_a += 1

    return wins_a, wins_b, total


def compute_nash_entropy(win_matrix: dict[str, dict[str, float]],
                         names: list[str], iterations: int = 5000) -> float:
    """
    Approximate Nash equilibrium via iterative best-response (fictitious play).
    Returns the Shannon entropy of the equilibrium distribution (higher = healthier metagame).
    """
    n = len(names)
    counts = [1.0] * n  # uniform initialisation
    for _ in range(iterations):
        total = sum(counts)
        mix = [c / total for c in counts]
        payoffs = []
        for i, ni in enumerate(names):
            p = sum(mix[j] * win_matrix[ni].get(nj, 0.5) for j, nj in enumerate(names) if i != j)
            payoffs.append(p)
        best = max(range(n), key=lambda i: payoffs[i])
        counts[best] += 1

    total = sum(counts)
    mix = [c / total for c in counts]
    entropy = -sum(p * (p if p <= 0 else __import__("math").log(p)) for p in mix if p > 0)
    return entropy


def main():
    ap = argparse.ArgumentParser(description="Halite metagame round-robin study")
    ap.add_argument("--seeds",      type=int, default=20,
                    help="games per pair per side (total = 2*seeds per matchup)")
    ap.add_argument("--map-size",   type=int, default=32)
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--workers",    type=int, default=6)
    ap.add_argument("--engine-cfg", type=str,
                    default=str(ROOT / "starter_kits/C++/competitive_engine.json"),
                    help="engine JSON config path")
    args = ap.parse_args()

    # Windows consoles default to cp1252 and crash on non-Latin-1 output.
    # Force UTF-8 as a safety net; output below is also kept ASCII-only.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    engine_cfg = Path(args.engine_cfg)
    names = list(PROFILES.keys())
    pairs = list(combinations(names, 2))

    print(f"Metagame study: {len(names)} profiles x {len(pairs)} matchups x {2*args.seeds} games")
    print(f"Engine config: {engine_cfg.name}\n")

    wins: dict[str, dict[str, int]] = {n: {m: 0 for m in names} for n in names}
    played: dict[str, dict[str, int]] = {n: {m: 0 for m in names} for n in names}

    for name_a, name_b in pairs:
        print(f"  {name_a} vs {name_b} ...", end=" ", flush=True)
        wa, wb, total = run_pair(name_a, name_b, args.seeds,
                                 args.map_size, args.turn_limit,
                                 engine_cfg, args.workers)
        wins[name_a][name_b] = wa
        wins[name_b][name_a] = wb
        played[name_a][name_b] = total
        played[name_b][name_a] = total
        wr_a = wa / total if total else 0
        wr_b = wb / total if total else 0
        print(f"{name_a} {wa}/{total} ({wr_a:.1%})  {name_b} {wb}/{total} ({wr_b:.1%})")

    # Win-rate matrix W[i][j] = win rate of i against j
    win_matrix: dict[str, dict[str, float]] = {}
    for ni in names:
        win_matrix[ni] = {}
        for nj in names:
            if ni == nj:
                win_matrix[ni][nj] = 0.5
            else:
                p = played[ni][nj]
                win_matrix[ni][nj] = wins[ni][nj] / p if p else 0.5

    # Print win-rate matrix
    col_w = 9
    print(f"\n{'Win-rate matrix':=^{col_w*(len(names)+1)+2}}")
    header = f"{'':>{col_w}}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    for ni in names:
        row = f"{ni:>{col_w}}"
        for nj in names:
            if ni == nj:
                row += f"{'-':>{col_w}}"
            else:
                row += f"{win_matrix[ni][nj]:>{col_w}.1%}"
        print(row)

    # Overall win rates
    print(f"\n{'Average win rate':=^{col_w*(len(names)+1)+2}}")
    avg_wr: dict[str, float] = {}
    for ni in names:
        opponents = [nj for nj in names if nj != ni]
        avg_wr[ni] = sum(win_matrix[ni][nj] for nj in opponents) / len(opponents)
    for ni in sorted(names, key=lambda n: -avg_wr[n]):
        bar = "#" * int(avg_wr[ni] * 20)
        flag = " <- DOMINANT" if avg_wr[ni] > 0.60 else ""
        print(f"  {ni:<10} {avg_wr[ni]:.1%}  {bar}{flag}")

    # Cyclic dominance check
    print(f"\n{'Cyclic dominance':=^{col_w*(len(names)+1)+2}}")
    dominant_edges = [(ni, nj) for ni in names for nj in names if ni != nj and win_matrix[ni][nj] > 0.5]
    cycles_found = False
    for a, b, c in [(n[0], n[1], n[2]) for n in [names] if len(names) >= 3]:
        if (win_matrix[a][b] > 0.5 and win_matrix[b][c] > 0.5 and win_matrix[c][a] > 0.5):
            print(f"  OK  {a} > {b} > {c} > {a}  (healthy cycle)")
            cycles_found = True
        if (win_matrix[b][a] > 0.5 and win_matrix[a][c] > 0.5 and win_matrix[c][b] > 0.5):
            print(f"  OK  {b} > {a} > {c} > {b}  (healthy cycle)")
            cycles_found = True
    if not cycles_found:
        print("  !!  No cycle found - one strategy may dominate; re-tune the lagging edge.")
    for a, b, c in [(names[0], names[1], names[2])]:
        dominant_edges_abc = [(ni, nj) for (ni, nj) in dominant_edges if ni in names and nj in names]
        for (ni, nj) in dominant_edges_abc:
            print(f"  {ni} > {nj}  ({win_matrix[ni][nj]:.1%})")

    # Nash entropy
    entropy = compute_nash_entropy(win_matrix, names)
    max_entropy = __import__("math").log(len(names))
    print(f"\n{'Metagame health':=^{col_w*(len(names)+1)+2}}")
    print(f"  Nash entropy: {entropy:.3f} / {max_entropy:.3f}  ({entropy/max_entropy:.1%} of max)")
    print(f"  {'HEALTHY - no dominant strategy' if entropy/max_entropy > 0.75 else 'UNHEALTHY - tune the dominant edge'}")


if __name__ == "__main__":
    main()
