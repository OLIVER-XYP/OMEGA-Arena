#!/usr/bin/env python3
"""Run deterministic Halite regression samples and emit compact JSON summaries."""

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / "game_engine" / "build" / "Release" / "halite.exe"
DEFAULT_BOT = ROOT / "starter_kits" / "C++" / "build" / "Release" / "MyBot.exe"
DEFAULT_OUTPUT = ROOT / "tests" / "regression" / "fixtures" / "baseline_summary.json"
DEFAULT_SEEDS = [101, 202, 303, 404, 505]
DEFAULT_MAPS = [32, 40, 48]


@dataclass
class RegressionCase:
    seed: int
    width: int
    height: int
    players: int
    turn_limit: int


@dataclass
class RegressionResult:
    case: RegressionCase
    ok: bool
    return_code: int
    summary_hash: str
    summary: Dict[str, Any]
    stderr_tail: str


def parse_json(stdout: str) -> Dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def compact_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    stats = payload.get("stats", {})
    compact_stats = {}
    if isinstance(stats, dict):
        for player_id, player_stats in sorted(stats.items(), key=lambda item: str(item[0])):
            if not isinstance(player_stats, dict):
                continue
            compact_stats[str(player_id)] = {
                "rank": player_stats.get("rank"),
                "score": player_stats.get("score"),
                "total_production": player_stats.get("total_production"),
                "ships_spawned": player_stats.get("ships_spawned"),
                "last_turn_alive": player_stats.get("last_turn_alive"),
            }
    return {
        "terminated": payload.get("terminated", {}),
        "error_logs": payload.get("error_logs", {}),
        "stats": compact_stats,
    }


def stable_hash(summary: Dict[str, Any]) -> str:
    data = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def run_case(engine: Path, bot: Path, case: RegressionCase) -> RegressionResult:
    bot_command = f'"{bot}"'
    commands = [bot_command for _ in range(case.players)]
    cmd = [
        str(engine),
        "--seed", str(case.seed),
        "--width", str(case.width),
        "--height", str(case.height),
        "--turn-limit", str(case.turn_limit),
        "--no-replay",
        "--no-logs",
        "--results-as-json",
        *commands,
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    try:
        payload = parse_json(completed.stdout)
        summary = compact_summary(payload)
    except Exception as exc:
        summary = {"parse_error": str(exc), "stdout_tail": completed.stdout[-1000:]}
    return RegressionResult(
        case=case,
        ok=completed.returncode == 0 and "parse_error" not in summary,
        return_code=completed.returncode,
        summary_hash=stable_hash(summary),
        summary=summary,
        stderr_tail=completed.stderr[-1000:],
    )


def build_cases(players: List[int], maps: List[int], seeds: List[int], turn_limit: int) -> List[RegressionCase]:
    cases = []
    for player_count in players:
        for map_size in maps:
            for seed in seeds:
                cases.append(RegressionCase(seed=seed, width=map_size, height=map_size, players=player_count, turn_limit=turn_limit))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Halite regression samples.")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--bot", type=Path, default=DEFAULT_BOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--players", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--maps", type=int, nargs="+", default=DEFAULT_MAPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--turn-limit", type=int, default=300)
    args = parser.parse_args()

    if not args.engine.exists():
        raise SystemExit(f"Engine executable does not exist: {args.engine}")
    if not args.bot.exists():
        raise SystemExit(f"Bot executable does not exist: {args.bot}")

    results = [run_case(args.engine, args.bot, case) for case in build_cases(args.players, args.maps, args.seeds, args.turn_limit)]
    output = {
        "engine": str(args.engine),
        "bot": str(args.bot),
        "results": [
            {
                "case": asdict(result.case),
                "ok": result.ok,
                "return_code": result.return_code,
                "summary_hash": result.summary_hash,
                "summary": result.summary,
                "stderr_tail": result.stderr_tail,
            }
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    failed = sum(1 for result in results if not result.ok)
    print(f"Wrote {len(results)} regression results to {args.output}; failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
