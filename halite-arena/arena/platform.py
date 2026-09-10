"""Platform layer: exception normalization + agent feedback aggregation.

Owns the "platform duties" that used to be scattered across cli.py/runner.py:
- classify a GameRecord / CompileResult into a TechnicalFailure enum,
- write agent-facing feedback (compile_result.json + run.json + games/) into
  agents/<name>/last_game/ so agents get ONE consistent read surface,
- summarize a run for the organizer (stdout / run.json).

Both cli.py (practice/match) and round.py (official rounds) call into here;
nothing platform-y should live in the CLI command handlers.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict, field
from pathlib import Path

from . import paths
from .engine_runner import GameRecord
from .compile import CompileResult


@dataclass
class TechnicalFailure:
    """A non-competitive termination that the platform must surface to the agent."""
    kind: str                 # compile_error|no_main|compile_timeout|engine_crash|no_json|spawn_error|timeout
    detail: str = ""
    side: str | None = None   # which side failed, if a game failure
    to_agent: bool = True     # whether this should be shown to the agent (vs organizer-only)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_game(rec: GameRecord) -> TechnicalFailure | None:
    """Map a GameRecord to a TechnicalFailure, or None if it's a clean game.

    `ok=False` is NOT failure — it only means one side got eliminated/crashed.
    We classify only *platform*-level problems the agent needs to know about.
    """
    if rec.fatal == "timeout":
        return TechnicalFailure("timeout", "整局超过墙钟时限，进程组已杀")
    if rec.fatal == "engine_crash":
        return TechnicalFailure("engine_crash", "引擎异常退出（无结果 JSON）", side="engine")
    if rec.fatal == "no_json":
        return TechnicalFailure("no_json", "引擎未产出可解析的结果 JSON")
    if rec.fatal == "internal":
        return TechnicalFailure("spawn_error", "平台内部错误（见 stdout_tail）")
    # ok 但某方有 error_logs（非法命令/被 terminate）——信息性，不判技术性失败
    if not rec.ok and rec.error_logs:
        return TechnicalFailure("error_logs", str(rec.error_logs)[:500], side="?")
    return None


def classify_compile(res: CompileResult) -> TechnicalFailure | None:
    if res.ok:
        return None
    return TechnicalFailure(res.error_kind or "compile_error",
                            (res.log or "")[:2000])


def write_compile_feedback(agent: str, res: CompileResult) -> Path:
    """Record a failed compile into agents/<name>/last_game/compile_result.json."""
    lg = paths.last_game_dir(agent)
    lg.mkdir(parents=True, exist_ok=True)
    (lg / "compile_result.json").write_text(
        json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    shutil.rmtree(lg / "games", ignore_errors=True)   # 本轮无对局
    return lg


def write_run_feedback(agent: str, run_id: Path) -> Path:
    """Copy a finished run (run.json + games/*) into the agent's last_game/."""
    lg = paths.last_game_dir(agent)
    lg.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_id / "run.json", lg / "run.json")
    shutil.rmtree(lg / "games", ignore_errors=True)
    shutil.copytree(run_id, lg / "games", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("engine_cwd"))
    return lg


def agent_facing_payload(rec: GameRecord) -> dict:
    """The compact agent-readable view of one game (drops the huge engine_json)."""
    return {
        "ok": rec.ok,
        "players": rec.players,
        "scores": rec.scores,
        "winner": rec.winner,
        "margin": rec.margin,
        "fatal": rec.fatal,
        "error_logs": {s: (v[-300:] if isinstance(v, str) else v)
                       for s, v in (rec.error_logs or {}).items()},
        "execution_time_ms": rec.execution_time_ms,
        "trace_counts": rec.trace_counts,
    }


def print_run_summary(run_id: Path) -> None:
    """Print a concise organizer summary for a run.json (practice or match)."""
    run = json.loads((run_id / "run.json").read_text(encoding="utf-8"))
    print(f"run: {run['run_id']}  kind={run.get('kind')}  failed={run.get('failed', 0)}")
    if run.get("kind") == "practice":
        for opp, s in run.get("summary", {}).items():
            print(f"  vs {opp:<12} wr={s['win_rate']:.3f}  margin={s['mean_margin']:+.0f}  "
                  f"({s['ok']}/{s['games']} ok)")
    elif run.get("kind") == "match":
        print("  participants:", run.get("participants"))
        for a, mp in sorted(run.get("summary", {}).items()):
            for b, s in sorted(mp.items()):
                print(f"    {a:<16} vs {b:<16}  {s['games']:2d}局  {s['a_wins']}-{s['b_wins']}")
        print("  ELO:", run.get("ratings_after"))
