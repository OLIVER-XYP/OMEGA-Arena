#!/usr/bin/env python3
"""Audit the locked v3 ruleset before expensive RL training.

This script is intentionally conservative: it makes stale assumptions visible,
checks the target ruleset identity, can run light smoke tests, validates the
ECO/AGGRO/CONTROL triangle with Wilson lower bounds, and can inspect replay
metrics for signs of one-note strategies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - dependency is optional until replay analysis is requested.
    zstd = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE_CFG = ROOT / "starter_kits/C++/competitive_engine_v2.json"
DEFAULT_ENGINE = ROOT / "game_engine/build/Release/halite.exe"
DEFAULT_CUDA_ENGINE = ROOT / "game_engine/build_cuda/Release/halite.exe"
DEFAULT_TEST_EXE = ROOT / "game_engine/build/Release/halite_test.exe"
DEFAULT_CUDA_TEST_EXE = ROOT / "game_engine/build_cuda/Release/halite_test.exe"
DEFAULT_BOT = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
DEFAULT_OUT = ROOT / "audit_runs/ruleset_v3_audit.json"

PROFILES = {
    "eco": ROOT / "starter_kits/C++/bot_params_eco.txt",
    "aggro": ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
    "adaptive": ROOT / "starter_kits/C++/bot_params_adaptive.txt",
}

TRIANGLE_EDGES = [
    ("aggro", "eco"),
    ("eco", "control"),
    ("control", "aggro"),
]

EXPECTED_V3 = {
    "RULESET_VERSION": "halite3-hp-competitive-v3",
    "ATTACK_RANGE": 4,
    "DEFEND_RETALIATION_DAMAGE": 100,
    "DEFEND_ALLOWS_MINING": True,
    "PLUNDER_HALITE_PER_TURN": 70,
    "MINING_INTERFERENCE_RATIO": 0.65,
}


@dataclass
class GameResult:
    ok: bool
    p0_won: bool = False
    tie: bool = False
    score0: float = 0.0
    score1: float = 0.0
    rc: int | None = None
    stderr: str = ""
    stdout_tail: str = ""
    error_logs: dict[str, str] | None = None
    terminated: dict[str, bool] | None = None
    replay: str | None = None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (center - margin) / denom


def parse_engine_result(stdout: str) -> dict[str, Any] | None:
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


def parse_game_result(proc: subprocess.CompletedProcess[str]) -> GameResult:
    js = parse_engine_result(proc.stdout)
    if not js:
        return GameResult(
            ok=False,
            rc=proc.returncode,
            stderr=proc.stderr[-1000:],
            stdout_tail=proc.stdout[-1000:],
        )
    stats = js.get("stats", {})
    if "0" not in stats or "1" not in stats:
        return GameResult(
            ok=False,
            rc=proc.returncode,
            stderr=proc.stderr[-1000:],
            stdout_tail=proc.stdout[-1000:],
        )
    rank0 = float(stats["0"].get("rank", 2))
    rank1 = float(stats["1"].get("rank", 2))
    return GameResult(
        ok=True,
        p0_won=rank0 < rank1,
        tie=abs(rank0 - rank1) < 1e-9,
        score0=float(stats["0"].get("score", 0)),
        score1=float(stats["1"].get("score", 0)),
        rc=proc.returncode,
        stderr=proc.stderr[-1000:],
        stdout_tail=proc.stdout[-1000:],
        error_logs=js.get("error_logs", {}),
        terminated=js.get("terminated", {}),
        replay=js.get("replay"),
    )


def run_game(
    *,
    engine: Path,
    bot: Path,
    engine_cfg: Path,
    profile_a: Path,
    profile_b: Path,
    seed: int,
    turn_limit: int,
    timeout: float,
    replay_dir: Path | None = None,
    backend: str = "cpu",
) -> GameResult:
    cmd_a = f"{bot} {seed} {profile_a}"
    cmd_b = f"{bot} {seed + 100000} {profile_b}"
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
        "--no-logs",
        "--results-as-json",
        "--engine-backend",
        backend,
        "-c",
        str(engine_cfg),
    ]
    if replay_dir is None:
        cmd.append("--no-replay")
    else:
        replay_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-i", str(replay_dir)])
    cmd.extend([cmd_a, cmd_b])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return GameResult(ok=False, rc=None, stderr=f"timeout after {exc.timeout}s")
    return parse_game_result(proc)


def audit_ruleset(path: Path) -> dict[str, Any]:
    cfg = load_json(path)
    checks: list[dict[str, Any]] = []
    for key, expected in EXPECTED_V3.items():
        actual = cfg.get(key)
        checks.append(
            {
                "key": key,
                "expected": expected,
                "actual": actual,
                "ok": actual == expected,
            }
        )

    design_risks = []
    if cfg.get("DEFEND_RETALIATION_DAMAGE", 0) >= 100 and cfg.get("DEFEND_ALLOWS_MINING") is True:
        design_risks.append(
            "DEFEND_RETALIATION_DAMAGE=100 with DEFEND_ALLOWS_MINING=true can make defense a low-risk productive action."
        )
    if cfg.get("PLUNDER_HALITE_PER_TURN", 0) >= 70:
        design_risks.append(
            "High PLUNDER_HALITE_PER_TURN needs replay review for structure-camping dominance."
        )
    if cfg.get("ATTACK_RANGE", 1) >= 4:
        design_risks.append(
            "ATTACK_RANGE=4 makes combat non-local; training can learn it, but explainability and bug surface increase."
        )
    if cfg.get("MINING_INTERFERENCE_RATIO", 0.0) >= 0.65:
        design_risks.append(
            "Strong mining interference can over-reward passive denial unless validated against replay metrics."
        )

    return {
        "path": rel(path),
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "design_risks": design_risks,
    }


def scan_known_code_risks() -> dict[str, Any]:
    engine_main = ROOT / "game_engine/main.cpp"
    store_header = ROOT / "game_engine/core/Store.hpp"
    project = ROOT / "PROJECT.md"
    risks: list[dict[str, Any]] = []

    if engine_main.exists():
        text = engine_main.read_text(encoding="utf-8", errors="replace")
        risks.append(
            {
                "risk": "map_size_fixed_32",
                "ok": "Only 32x32 maps are supported" in text,
                "detail": "Engine currently rejects non-32x32 maps; train/eval scripts should stay on 32x32.",
            }
        )

    if store_header.exists():
        text = store_header.read_text(encoding="utf-8", errors="replace")
        risks.append(
            {
                "risk": "stale_entity_guard_required",
                "ok": "get_entity" in text and "entities_ref" in text,
                "detail": "New AI/search code must guard entity ids with entities_ref().find(id) before get_entity.",
            }
        )

    if project.exists():
        text = project.read_text(encoding="utf-8", errors="replace")
        risks.append(
            {
                "risk": "stale_v1_metagame_note",
                "ok": "v1 metagame note is stale" in text or "halite3-hp-competitive-v3" in text,
                "detail": "PROJECT.md should mark the old no-cycle note as stale once docs are updated.",
            }
        )

    return {"risks": risks, "ok": all(r["ok"] for r in risks)}


def run_test_command(cmd: list[str], timeout: float) -> dict[str, Any]:
    if not Path(cmd[0]).exists():
        return {"ok": False, "skipped": True, "reason": f"missing executable: {cmd[0]}"}
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "timeout": exc.timeout, "cmd": cmd}
    return {
        "ok": proc.returncode == 0,
        "cmd": cmd,
        "rc": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def run_smoke_tests(test_exe: Path, cuda_test_exe: Path, timeout: float, include_cuda: bool) -> dict[str, Any]:
    tests = {
        "phase_scenarios": run_test_command([str(test_exe), "[phase]"], timeout),
        "modernization_cpu": run_test_command([str(test_exe), "[modernization]~[cuda]"], timeout),
    }
    if include_cuda:
        tests["cuda_consistency"] = run_test_command([str(cuda_test_exe), "[modernization][gpu][cuda]"], timeout)
    return {"ok": all(item.get("ok") or item.get("skipped") for item in tests.values()), "tests": tests}


def validate_triangle(args: argparse.Namespace) -> dict[str, Any]:
    edges: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    total_games = 0

    def submit_edge(executor: ThreadPoolExecutor, winner: str, loser: str):
        futures = {}
        for seed in range(args.seeds):
            futures[
                executor.submit(
                    run_game,
                    engine=args.engine,
                    bot=args.bot,
                    engine_cfg=args.engine_cfg,
                    profile_a=PROFILES[winner],
                    profile_b=PROFILES[loser],
                    seed=seed,
                    turn_limit=args.turn_limit,
                    timeout=args.game_timeout,
                    backend=args.backend,
                )
            ] = False
            futures[
                executor.submit(
                    run_game,
                    engine=args.engine,
                    bot=args.bot,
                    engine_cfg=args.engine_cfg,
                    profile_a=PROFILES[loser],
                    profile_b=PROFILES[winner],
                    seed=seed + 50000,
                    turn_limit=args.turn_limit,
                    timeout=args.game_timeout,
                    backend=args.backend,
                )
            ] = True
        return futures

    for winner, loser in TRIANGLE_EDGES:
        wins = 0
        losses = 0
        ties = 0
        valid = 0
        score_margin = []
        edge_failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = submit_edge(executor, winner, loser)
            for future in as_completed(futures):
                winner_is_p1 = futures[future]
                result = future.result()
                if not result.ok:
                    edge_failures.append(
                        {
                            "rc": result.rc,
                            "stderr": result.stderr,
                            "stdout_tail": result.stdout_tail,
                        }
                    )
                    continue
                valid += 1
                total_games += 1
                if result.tie:
                    ties += 1
                    continue
                winner_won = (not winner_is_p1 and result.p0_won) or (winner_is_p1 and not result.p0_won)
                if winner_won:
                    wins += 1
                else:
                    losses += 1
                if winner_is_p1:
                    score_margin.append(result.score1 - result.score0)
                else:
                    score_margin.append(result.score0 - result.score1)

        decided = wins + losses
        win_rate = wins / decided if decided else 0.0
        lb = wilson_lower_bound(wins, decided)
        edge_key = f"{winner}>{loser}"
        edges[edge_key] = {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "valid_games": valid,
            "failed_games": len(edge_failures),
            "win_rate": win_rate,
            "wilson95_lower_bound": lb,
            "mean_score_margin": sum(score_margin) / len(score_margin) if score_margin else 0.0,
            "target_lb_ok": lb >= args.target_lb,
            "minimum_lb_ok": lb >= args.min_lb,
        }
        if edge_failures:
            failures.append({"edge": edge_key, "failures": edge_failures[:5]})

    return {
        "ok": all(edge["minimum_lb_ok"] for edge in edges.values()),
        "target_ok": all(edge["target_lb_ok"] for edge in edges.values()),
        "total_valid_games": total_games,
        "edges": edges,
        "failures": failures,
        "thresholds": {"minimum_lb": args.min_lb, "target_lb": args.target_lb},
    }


def read_replay(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if path.suffix == ".hlt":
        if zstd is None:
            raise RuntimeError("zstandard is required for .hlt replay analysis")
        try:
            raw = zstd.ZstdDecompressor().decompress(raw, max_output_size=300 * 1024 * 1024)
        except Exception:
            pass
    return json.loads(raw)


def count_events(replay: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in replay.get("full_frames", []):
        for event in frame.get("events", []):
            event_type = str(event.get("type", "unknown"))
            counts[event_type] = counts.get(event_type, 0) + 1
    return counts


def player_stats_summary(replay: dict[str, Any]) -> list[dict[str, Any]]:
    stats = replay.get("game_statistics", {}).get("player_statistics", [])
    output = []
    for item in stats:
        output.append(
            {
                "player_id": item.get("player_id"),
                "rank": item.get("rank"),
                "total_production": item.get("total_production"),
                "total_mined": item.get("total_mined"),
                "total_bonus": item.get("total_bonus"),
                "mining_efficiency": item.get("mining_efficiency"),
                "ships_spawned": item.get("ships_spawned"),
                "ships_peak": item.get("ships_peak"),
                "carried_at_end": item.get("carried_at_end"),
                "interaction_opportunities": item.get("interaction_opportunities"),
                "self_collisions": item.get("self_collisions"),
                "all_collisions": item.get("all_collisions"),
                "number_dropoffs": item.get("number_dropoffs"),
            }
        )
    return output


def analyze_replay_file(path: Path) -> dict[str, Any]:
    replay = read_replay(path)
    frames = replay.get("full_frames", [])
    ship_counts: dict[str, list[int]] = {}
    deposits: dict[str, list[int]] = {}
    for frame in frames:
        for player_id, ships in (frame.get("entities") or {}).items():
            ship_counts.setdefault(player_id, []).append(len(ships or {}))
        for player_id, deposited in (frame.get("deposited") or {}).items():
            deposits.setdefault(player_id, []).append(int(deposited))

    return {
        "path": rel(path),
        "seed": replay.get("map_generator_seed"),
        "turns": len(frames),
        "events": count_events(replay),
        "player_stats": player_stats_summary(replay),
        "ship_count": {
            player: {
                "peak": max(values) if values else 0,
                "mean": sum(values) / len(values) if values else 0.0,
                "final": values[-1] if values else 0,
            }
            for player, values in ship_counts.items()
        },
        "deposit_final": {
            player: values[-1] if values else 0
            for player, values in deposits.items()
        },
        "plunder_income": {
            "available": False,
            "note": "Replay/statistics do not expose plunder income separately; inspect total_bonus, interactions, and structure-adjacent behavior as proxies.",
        },
    }


def collect_replay_samples(args: argparse.Namespace) -> list[Path]:
    replay_dir = args.replay_dir or (DEFAULT_OUT.parent / "replays")
    replay_dir.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in replay_dir.glob("*.hlt")}
    edges = TRIANGLE_EDGES[: max(1, args.replay_samples)]
    for idx, (a, b) in enumerate(edges):
        run_game(
            engine=args.engine,
            bot=args.bot,
            engine_cfg=args.engine_cfg,
            profile_a=PROFILES[a],
            profile_b=PROFILES[b],
            seed=900000 + idx,
            turn_limit=args.turn_limit,
            timeout=args.game_timeout,
            replay_dir=replay_dir,
            backend=args.backend,
        )
    after = {p.resolve() for p in replay_dir.glob("*.hlt")}
    return sorted(Path(p) for p in after - before)


def replay_analysis(args: argparse.Namespace) -> dict[str, Any]:
    replay_paths = [resolve(p) for p in args.replays]
    if args.replay_samples > 0:
        replay_paths.extend(collect_replay_samples(args))
    analyses = []
    errors = []
    for path in replay_paths:
        try:
            analyses.append(analyze_replay_file(path))
        except Exception as exc:
            errors.append({"path": rel(path), "error": str(exc)})
    return {"ok": not errors, "count": len(analyses), "replays": analyses, "errors": errors}


def export_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = report.get("triangle", {}).get("edges", {})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "edge",
                "wins",
                "losses",
                "ties",
                "valid_games",
                "failed_games",
                "win_rate",
                "wilson95_lower_bound",
                "mean_score_margin",
                "minimum_lb_ok",
                "target_lb_ok",
            ],
        )
        writer.writeheader()
        for edge, values in edges.items():
            row = {"edge": edge}
            row.update(values)
            writer.writerow(row)


def print_human_summary(report: dict[str, Any]) -> None:
    print(f"Ruleset audit: {report['ruleset']['path']}")
    print(f"  ruleset identity: {'OK' if report['ruleset']['ok'] else 'FAIL'}")
    for item in report["ruleset"]["checks"]:
        status = "OK" if item["ok"] else "FAIL"
        print(f"    {status:4s} {item['key']}: actual={item['actual']!r} expected={item['expected']!r}")
    if report["ruleset"]["design_risks"]:
        print("  design risks:")
        for risk in report["ruleset"]["design_risks"]:
            print(f"    - {risk}")

    if "triangle" in report:
        print("Triangle validation:")
        for edge, values in report["triangle"]["edges"].items():
            print(
                f"  {edge:15s} wr={values['win_rate']:.1%} "
                f"lb95={values['wilson95_lower_bound']:.1%} "
                f"({values['wins']}/{values['wins'] + values['losses']} decided, "
                f"{values['failed_games']} failed)"
            )
        if report["triangle"]["ok"]:
            print("  minimum gate: OK")
        else:
            print("  minimum gate: FAIL")
        if not report["triangle"]["target_ok"]:
            print("  target gate: WARN (some Wilson lower bounds are below target)")

    if "replay_analysis" in report:
        print(f"Replay analysis: {report['replay_analysis']['count']} replay(s)")
        for item in report["replay_analysis"]["replays"][:5]:
            print(f"  {item['path']} turns={item['turns']} events={item['events']}")

    print(f"Overall: {'PASS' if report['overall_ok'] else 'FAIL'}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-cfg", type=resolve, default=DEFAULT_ENGINE_CFG)
    ap.add_argument("--engine", type=resolve, default=DEFAULT_ENGINE)
    ap.add_argument("--bot", type=resolve, default=DEFAULT_BOT)
    ap.add_argument("--test-exe", type=resolve, default=DEFAULT_TEST_EXE)
    ap.add_argument("--cuda-test-exe", type=resolve, default=DEFAULT_CUDA_TEST_EXE)
    ap.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="cpu")
    ap.add_argument("--seeds", type=int, default=10, help="per edge per seating; total per edge is 2*seeds")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--turn-limit", type=int, default=300)
    ap.add_argument("--game-timeout", type=float, default=60.0)
    ap.add_argument("--test-timeout", type=float, default=120.0)
    ap.add_argument("--min-lb", type=float, default=0.50)
    ap.add_argument("--target-lb", type=float, default=0.55)
    ap.add_argument("--skip-triangle", action="store_true")
    ap.add_argument("--run-tests", action="store_true")
    ap.add_argument("--include-cuda-tests", action="store_true")
    ap.add_argument("--replays", action="append", default=[], help="existing replay path to analyze")
    ap.add_argument("--replay-samples", type=int, default=0, help="generate this many replay sample games")
    ap.add_argument("--replay-dir", type=resolve, default=None)
    ap.add_argument("--out", type=resolve, default=DEFAULT_OUT)
    ap.add_argument("--csv", type=resolve, default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {
        "started_at": started,
        "root": str(ROOT),
        "ruleset": audit_ruleset(args.engine_cfg),
        "known_risks": scan_known_code_risks(),
        "inputs": {
            "engine": rel(args.engine),
            "bot": rel(args.bot),
            "turn_limit": args.turn_limit,
            "backend": args.backend,
            "seeds": args.seeds,
            "workers": args.workers,
        },
    }

    if args.run_tests:
        report["smoke_tests"] = run_smoke_tests(
            args.test_exe,
            args.cuda_test_exe,
            args.test_timeout,
            args.include_cuda_tests,
        )

    if not args.skip_triangle:
        report["triangle"] = validate_triangle(args)

    if args.replays or args.replay_samples > 0:
        report["replay_analysis"] = replay_analysis(args)

    checks = [
        report["ruleset"]["ok"],
        report["known_risks"]["ok"],
    ]
    if "smoke_tests" in report:
        checks.append(report["smoke_tests"]["ok"])
    if "triangle" in report:
        checks.append(report["triangle"]["ok"])
    if "replay_analysis" in report:
        checks.append(report["replay_analysis"]["ok"])
    report["overall_ok"] = all(checks)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.csv:
        export_csv(report, args.csv)
    print_human_summary(report)
    print(f"Wrote {rel(args.out)}")
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
