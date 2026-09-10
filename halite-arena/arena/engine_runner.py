"""Generic single-game runner: play ANY two BotSpecs on the real engine.

Generalized from rl_ai/evaluate.py run_game(). Key hardening vs evaluate:
- kills the whole process SESSION on timeout (engine + sh -c bot grandchildren),
- keeps engine per-turn 2s timeout ON by default (malicious bot can't hang a turn),
- `--no-compression` so replay is plain JSON (readable by trace.py),
- `-o` overrides player display names,
- error_logs read back as text for agent feedback.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import config as C
from .bots import BotSpec, build_bot_command

try:
    from rl_ai.evaluate import parse_json  # 复用 evaluate 的容错 JSON 解析
except Exception:  # pragma: no cover
    from ._rl_compat import parse_json  # standalone clone fallback


@dataclass
class GameRecord:
    ok: bool
    players: list[str]                       # [side0_name, side1_name]
    scores: list[int]
    ranks: list[int]
    winner: str | None
    margin: int
    terminated: dict = field(default_factory=dict)
    error_logs: dict = field(default_factory=dict)   # side(str) -> log text (truncated)
    fatal: str | None = None                 # "timeout"|"engine_crash"|"no_json"|None
    stdout_tail: str = ""
    stderr_tail: str = ""
    seed: int = 0
    engine_json: dict | None = None
    trace_counts: dict = field(default_factory=dict)
    replay: Path | None = None
    execution_time_ms: int | None = None
    map_total_halite: int | None = None


def _kill_session(sid: int) -> None:
    """SIGKILL every process group whose session == sid (engine started a new
    session; bot grandchildren share it). Sweep /proc to catch all pgrps."""
    try:
        with os.scandir("/proc") as it:
            for e in it:
                if not e.name.isdigit():
                    continue
                try:
                    st = Path("/proc") / e.name / "stat"
                    stat = st.read_text().split()
                    # stat: pid (comm) state ppid pgrp session ...
                    pgrp, session = int(stat[4]), int(stat[5])
                    if session == sid:
                        os.killpg(pgrp, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        os.killpg(sid, signal.SIGKILL)
    except Exception:
        pass


def _with_device(spec: BotSpec, device: str) -> BotSpec:
    """Rebind an RL spec's device so the CLI --device flag actually takes effect.

    Script/agent bots are native executables and ignore it. Without this, the
    device passed to run_one_game is dropped and the BotSpec default ("auto")
    wins, so `arena practice ... --device cuda:0` silently ran on auto.
    """
    if spec.kind == "rl" and device and device != "auto" and device != spec.device:
        from dataclasses import replace
        return replace(spec, device=device)
    return spec


def run_one_game(
    spec0: BotSpec, spec1: BotSpec, seed: int,
    *,
    game_dir: Path, engine: Path | None = None, engine_cfg: Path | None = None,
    backend: str = "cpu", turn_limit: int = 300,
    wall_timeout: float | None = None,
    engine_no_timeout: bool = False,
    save_replay: bool = True,
    trace_sides: tuple[int, ...] = (),
    device: str = "auto",
) -> GameRecord:
    """Run one engine game between spec0 (side 0) and spec1 (side 1)."""
    engine = engine or C.ENGINE
    engine_cfg = engine_cfg or C.ENGINE_CFG
    if not Path(engine).exists():
        return GameRecord(ok=False, players=[spec0.short, spec1.short], scores=[0, 0],
                          ranks=[], winner=None, margin=0, fatal="engine_crash",
                          stderr_tail=f"engine not found: {engine}")

    trace_paths: dict[int, Path] = {}
    for side in trace_sides:
        spec = spec0 if side == 0 else spec1
        if spec.kind == "rl":
            tp = game_dir / f"trace_side{side}.jsonl"
            trace_paths[side] = tp

    # The engine writes the replay into its cwd despite --replay-directory, but
    # we still create the advertised dir so the flag is honest and the fallback
    # glob below has a real target. Also apply the requested device to RL specs.
    replay_out = game_dir / "replay_out"
    replay_out.mkdir(parents=True, exist_ok=True)
    if device and device != "auto":
        spec0 = _with_device(spec0, device)
        spec1 = _with_device(spec1, device)

    cmd = [
        str(engine), "--width", "32", "--height", "32",
        "-s", str(seed), "--turn-limit", str(turn_limit),
        "--results-as-json", "--engine-backend", backend,
        "-c", str(engine_cfg),
        "--no-compression",
        "-o", spec0.short, "-o", spec1.short,
    ]
    if save_replay:
        cmd += ["--replay-directory", str(replay_out)]
    else:
        cmd.append("--no-replay")
    if engine_no_timeout:
        cmd.append("--no-timeout")

    cmd += [build_bot_command(spec0, seed, trace_path=trace_paths.get(0)),
            build_bot_command(spec1, seed, trace_path=trace_paths.get(1))]

    # default wall timeout: script/agent are fast; RL on CPU may need more
    if wall_timeout is None:
        wall_timeout = max(60.0, 30.0 + turn_limit * (3.0 if engine_no_timeout else 2.0))

    cwd = game_dir / "engine_cwd"
    cwd.mkdir(parents=True, exist_ok=True)
    start = __import__("time").time()
    # sandbox launch: prlimit core-off + fsize cap (nproc/as deliberately NOT
    # set — see sandbox.py docstring); session kill covers the whole tree.
    from . import sandbox
    full_cmd = sandbox._prlimit_prefix(core=0, fsize=int(0.5 * 1024 ** 3)) + cmd
    proc = subprocess.Popen(full_cmd, cwd=str(cwd), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=wall_timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_session(proc.pid)
        out, err = proc.communicate(timeout=10)
    elapsed_ms = int((__import__("time").time() - start) * 1000)

    js = None
    fatal = "timeout" if timed_out else None
    if not timed_out:
        js = parse_json(out)
        if js is None:
            fatal = "engine_crash" if proc.returncode else "no_json"

    # read error logs (engine wrote errorlog files; results error_logs maps side->path)
    error_logs: dict = {}
    replay_path: Path | None = None
    if js is not None:
        for side_str, logpath in (js.get("error_logs") or {}).items():
            lp = Path(str(logpath))
            if not lp.is_absolute():
                # engine writes errorlog into its cwd (engine_cwd/)
                cand = cwd / lp.name
                if cand.exists():
                    lp = cand
            try:
                error_logs[side_str] = lp.read_text(encoding="utf-8", errors="replace")[-4000:]
            except Exception:
                error_logs[side_str] = f"<unreadable {lp}>"
        # replay path: engine writes replay into cwd (engine_cwd) despite the
        # --replay-directory flag. Move it up to game_dir root for self-containment.
        replay_path = None
        hits = sorted(cwd.glob("*.hlt")) + (sorted(replay_out.glob("*.hlt")) if save_replay else [])
        for src in hits:
            dest = game_dir / src.name
            try:
                if src.resolve() != dest.resolve():
                    if dest.exists():
                        dest.unlink()
                    src.replace(dest)
                moved = dest
            except Exception:
                moved = src
            if replay_path is None:      # primary replay for GameRecord
                replay_path = moved

    players = [spec0.short, spec1.short]
    if js is None:
        return GameRecord(ok=False, players=players, scores=[0, 0], ranks=[],
                          winner=None, margin=0, fatal=fatal,
                          stdout_tail=(out or "")[-2000:], stderr_tail=(err or "")[-2000:],
                          seed=seed, execution_time_ms=elapsed_ms)

    # normalize scores/ranks from stats
    stats = js.get("stats") or {}
    scores = [int(stats.get("0", {}).get("score", 0)), int(stats.get("1", {}).get("score", 0))]
    ranks = [int(stats.get("0", {}).get("rank", 0)), int(stats.get("1", {}).get("rank", 0))]
    margin = scores[0] - scores[1]
    if scores[0] > scores[1]:
        winner = players[0]
    elif scores[1] > scores[0]:
        winner = players[1]
    else:
        winner = None
    terminated = dict(js.get("terminated") or {})

    # RL trace_counts for requested sides
    trace_counts: dict = {}
    for side, tp in trace_paths.items():
        if tp.exists():
            try:
                from rl_ai.evaluate import read_trace_counts
            except Exception:
                from ._rl_compat import read_trace_counts  # standalone clone fallback
            trace_counts[str(side)] = read_trace_counts(tp)

    ok = (fatal is None) and not bool(error_logs)   # 淘汰≠失败（evaluate 语义）
    return GameRecord(
        ok=ok, players=players, scores=scores, ranks=ranks, winner=winner, margin=margin,
        terminated=terminated, error_logs=error_logs, fatal=fatal,
        seed=seed, engine_json=js, trace_counts=trace_counts,
        replay=replay_path, execution_time_ms=elapsed_ms,
        map_total_halite=js.get("map_total_halite"),
    )
