#!/usr/bin/env python3
"""Compare CPU and CUDA engine backends on real v3 bot games."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / "game_engine/build_cuda/Release/halite.exe"
DEFAULT_CFG = ROOT / "starter_kits/C++/competitive_engine_v2.json"
DEFAULT_BOT = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
DEFAULT_OUT = ROOT / "audit_runs/gpu_consistency.json"

PROFILES = {
    "eco": ROOT / "starter_kits/C++/bot_params_eco.txt",
    "aggro": ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
    "adaptive": ROOT / "starter_kits/C++/bot_params_adaptive.txt",
}

PAIRS = [
    ("eco", "aggro"),
    ("aggro", "control"),
    ("control", "eco"),
    ("adaptive", "eco"),
    ("adaptive", "aggro"),
    ("adaptive", "control"),
]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_json(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def normalize_csv_records(text: str, *, sort_records: bool) -> list[str]:
    records = [item for item in text.split(",") if item]
    if sort_records:
        def key(record: str) -> tuple[int, str]:
            head = record.split("-", 1)[0]
            try:
                return int(head), record
            except ValueError:
                return 0, record

        return sorted(records, key=key)
    return records


def normalize_snapshot(snapshot: str | None) -> dict[str, Any] | None:
    if not snapshot:
        return None
    parts = snapshot.split(";")
    if len(parts) < 3:
        return {"raw": snapshot}

    players = []
    index = 3
    while index + 3 < len(parts) and parts[index] != "":
        player_id = parts[index]
        players.append(
            {
                "player_id": int(player_id),
                "energy": int(parts[index + 1]),
                "bases": normalize_csv_records(parts[index + 2], sort_records=False),
                "entities": normalize_csv_records(parts[index + 3], sort_records=True),
            }
        )
        index += 4

    return {
        "version": parts[0],
        "map": parts[1],
        "cells": normalize_csv_records(parts[2], sort_records=False),
        "players": sorted(players, key=lambda player: player["player_id"]),
    }


def run_game(
    *,
    engine: Path,
    cfg: Path,
    bot: Path,
    profile0: str,
    profile1: str,
    seed: int,
    backend: str,
    turn_limit: int,
    timeout: float,
) -> dict[str, Any]:
    cmd0 = f"{bot} {seed} {PROFILES[profile0]}"
    cmd1 = f"{bot} {seed + 100000} {PROFILES[profile1]}"
    cmd = [
        str(engine),
        "--width",
        "32",
        "--height",
        "32",
        "-s",
        str(seed),
        "--turn-limit",
        str(turn_limit),
        "--no-replay",
        "--no-logs",
        "--results-as-json",
        "--engine-backend",
        backend,
        "-c",
        str(cfg),
        cmd0,
        cmd1,
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "backend": backend, "error": f"timeout {exc.timeout}s"}
    js = parse_json(proc.stdout)
    if not js:
        return {
            "ok": False,
            "backend": backend,
            "rc": proc.returncode,
            "stderr": proc.stderr[-1000:],
            "stdout_tail": proc.stdout[-1000:],
        }
    return {"ok": True, "backend": backend, "result": js}


def comparable(result: dict[str, Any]) -> dict[str, Any]:
    stats = result.get("stats", {})
    return {
        "stats": {
            player: {
                "rank": stats.get(player, {}).get("rank"),
                "score": stats.get(player, {}).get("score"),
            }
            for player in sorted(stats.keys())
        },
        "final_snapshot": normalize_snapshot(result.get("final_snapshot")),
        "map_seed": result.get("map_seed"),
        "map_total_halite": result.get("map_total_halite"),
    }


def compare_case(args: argparse.Namespace, profile0: str, profile1: str, seed: int) -> dict[str, Any]:
    cpu = run_game(
        engine=args.engine,
        cfg=args.engine_cfg,
        bot=args.bot,
        profile0=profile0,
        profile1=profile1,
        seed=seed,
        backend="cpu",
        turn_limit=args.turn_limit,
        timeout=args.timeout,
    )
    gpu = run_game(
        engine=args.engine,
        cfg=args.engine_cfg,
        bot=args.bot,
        profile0=profile0,
        profile1=profile1,
        seed=seed,
        backend="gpu",
        turn_limit=args.turn_limit,
        timeout=args.timeout,
    )
    case = {"profile0": profile0, "profile1": profile1, "seed": seed}
    if not cpu.get("ok") or not gpu.get("ok"):
        return {"ok": False, "case": case, "cpu": cpu, "gpu": gpu, "reason": "run_failed"}
    cpu_cmp = comparable(cpu["result"])
    gpu_cmp = comparable(gpu["result"])
    equal = cpu_cmp == gpu_cmp
    return {
        "ok": equal,
        "case": case,
        "reason": "match" if equal else "mismatch",
        "cpu": cpu_cmp,
        "gpu": gpu_cmp,
    }


def build_cases(seeds: int) -> list[tuple[str, str, int]]:
    cases = []
    for seed_index in range(seeds):
        for pair_index, (a, b) in enumerate(PAIRS):
            seed = seed_index * 1000 + pair_index * 10
            cases.append((a, b, seed))
            cases.append((b, a, seed + 50000))
    return cases


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=resolve, default=DEFAULT_ENGINE)
    ap.add_argument("--engine-cfg", type=resolve, default=DEFAULT_CFG)
    ap.add_argument("--bot", type=resolve, default=DEFAULT_BOT)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--out", type=resolve, default=DEFAULT_OUT)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    cases = build_cases(args.seeds)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compare_case, args, profile0, profile1, seed): (profile0, profile1, seed)
            for profile0, profile1, seed in cases
        }
        for future in as_completed(futures):
            results.append(future.result())

    report = {
        "schema": "halite3_v3_cpu_gpu_consistency_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": rel(args.engine),
        "engine_config": rel(args.engine_cfg),
        "turn_limit": args.turn_limit,
        "cases": len(cases),
        "passed": sum(1 for r in results if r["ok"]),
        "failed": sum(1 for r in results if not r["ok"]),
        "results": sorted(results, key=lambda r: (r["case"]["profile0"], r["case"]["profile1"], r["case"]["seed"])),
    }
    report["ok"] = report["failed"] == 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"CPU/GPU consistency: {report['passed']}/{report['cases']} passed "
        f"({report['failed']} failed)"
    )
    for item in report["results"]:
        if not item["ok"]:
            print(f"  FAIL {item['case']}: {item['reason']}")
    print(f"Wrote {rel(args.out)}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
