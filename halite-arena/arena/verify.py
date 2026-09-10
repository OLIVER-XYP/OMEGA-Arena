"""arena verify — read-only integrity checks over a persisted run/game.

Four checks (per the fix-plan spec):
  1. completeness   — required artifacts exist and parse
  2. self-consistency — winner<->scores, ok<->fatal<->error_logs, players
  3. authenticity   — replay seed/size/frame-count cross-checked vs result.json
  4. disk vs record — mtimes/write-order + last_game copy vs runs/ origin

Plus three outputs: per-bot cost estimate, artifact write-order timeline,
and a time-formatted summary with the engine's 2s/turn budget.

Strictly read-only: never writes, never runs a game.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import paths

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Finding:
    level: str
    check: str
    detail: str = ""


@dataclass
class GameReport:
    game: str
    findings: list[Finding] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def add(self, level: str, check: str, detail: str = ""):
        self.findings.append(Finding(level, check, detail))

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.level == FAIL]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == WARN]

    @property
    def ok(self) -> bool:
        return not self.failures


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------------------
def verify_game(game_dir: Path, run_dir: Path | None = None) -> GameReport:
    g = Path(game_dir)
    rep = GameReport(game=str(g))
    _check_completeness(g, rep)
    result = _read_json(g / "result.json")
    if result is None:
        return rep   # nothing more can be checked
    _check_consistency(result, rep, run_dir)
    _check_authenticity(g, result, rep)
    _check_disk(g, result, rep, run_dir)
    _cost_and_timeline(g, result, rep)
    return rep


def _check_completeness(g: Path, rep: GameReport):
    required = {
        "result.json": g / "result.json",
    }
    replay = next(g.glob("replay-*.hlt"), None)
    if replay is None:
        rep.add(FAIL, "complete.replay", "缺少 replay-*.hlt")
    else:
        rep.info["replay_name"] = replay.name
        if _read_json(replay) is None:
            rep.add(FAIL, "complete.replay_parse", "replay 无法解析为 JSON")
    if not (g / "result.json").exists():
        rep.add(FAIL, "complete.result", "缺少 result.json")
    trace = g / "trace.jsonl"
    if not trace.exists():
        rep.add(WARN, "complete.trace", "缺少 trace.jsonl（逐回合轨迹未生成）")
    else:
        bad = 0
        n = 0
        with open(trace, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                n += 1
                try:
                    json.loads(line)
                except Exception:
                    bad += 1
        rep.info["trace_rows"] = n
        if bad:
            rep.add(FAIL, "complete.trace_parse", f"trace.jsonl 有 {bad} 行无法解析")
    cwd = g / "engine_cwd"
    if cwd.is_dir():
        rep.info["engine_cwd_files"] = sorted(p.name for p in cwd.iterdir())
        if not any(p.name.startswith("bot-") for p in cwd.iterdir()):
            rep.add(WARN, "complete.botlog", "engine_cwd 无 bot-*.log")
    else:
        rep.add(WARN, "complete.engine_cwd", "缺少 engine_cwd/ 目录")


def _check_consistency(result: dict, rep: GameReport, run_dir: Path | None):
    scores = result.get("scores") or []
    players = result.get("players") or []
    winner = result.get("winner")
    fatal = result.get("fatal")
    ok = result.get("ok")
    errs = result.get("error_logs") or {}

    if len(players) != 2 or len(scores) != 2:
        rep.add(FAIL, "consist.players", f"players/scores 长度异常: {players} {scores}")
        return
    # winner <-> scores
    if scores[0] > scores[1]:
        expect = players[0]
    elif scores[1] > scores[0]:
        expect = players[1]
    else:
        expect = None
    if winner != expect:
        rep.add(FAIL, "consist.winner",
                f"winner={winner!r} 与 scores={scores} 推出的 {expect!r} 不符")
    # ok <-> fatal <-> error_logs
    if fatal is not None and ok:
        rep.add(FAIL, "consist.ok_fatal", f"fatal={fatal} 但 ok=True")
    if ok and errs:
        rep.add(FAIL, "consist.ok_errlogs", f"ok=True 却有 error_logs: {list(errs)}")
    if not ok and fatal is None and not errs:
        rep.add(WARN, "consist.notok_no_reason", "ok=False 但既无 fatal 也无 error_logs")
    # error_logs sides must be a subset of player indices
    for side in errs:
        if str(side) not in ("0", "1"):
            rep.add(WARN, "consist.errlog_side", f"error_logs 含意外 side={side}")
    # errorlog must be READABLE (a `<unreadable …>` sentinel means the platform
    # failed to resolve the engine's relative errorlog path — the bug fixed in
    # engine_runner; verify surfaces it so it can't silently regress).
    for side, val in errs.items():
        if isinstance(val, str) and val.startswith("<unreadable"):
            rep.add(WARN, "consist.errlog_readable",
                    f"side {side} 的 errorlog 未能读取: {val}")


def _check_authenticity(g: Path, result: dict, rep: GameReport):
    replay = next(g.glob("replay-*.hlt"), None)
    if replay is None:
        return
    rp = _read_json(replay)
    if rp is None:
        return
    ej = result.get("engine_json") or {}
    seed = result.get("seed")
    # seed
    r_seed = rp.get("map_generator_seed")
    if r_seed is not None and seed is not None and int(r_seed) != int(seed):
        rep.add(FAIL, "auth.seed", f"replay map_generator_seed={r_seed} != result.seed={seed}")
    if ej.get("map_seed") is not None and seed is not None and int(ej["map_seed"]) != int(seed):
        rep.add(FAIL, "auth.engine_seed", f"engine_json.map_seed={ej['map_seed']} != seed={seed}")
    # size from production_map (replay top-level width/height can be null)
    pm = rp.get("production_map") or {}
    rw, rh = pm.get("width"), pm.get("height")
    ew, eh = ej.get("map_width"), ej.get("map_height")
    if None not in (rw, ew) and rw != ew:
        rep.add(FAIL, "auth.width", f"replay width={rw} != engine width={ew}")
    if None not in (rh, eh) and rh != eh:
        rep.add(FAIL, "auth.height", f"replay height={rh} != engine height={eh}")
    # total halite
    gstat = (rp.get("game_statistics") or {})
    r_mth = gstat.get("map_total_halite")
    if r_mth is not None and result.get("map_total_halite") is not None \
            and int(r_mth) != int(result["map_total_halite"]):
        rep.add(FAIL, "auth.map_halite",
                f"replay map_total_halite={r_mth} != result {result['map_total_halite']}")
    # frames vs trace rows
    frames = len(rp.get("full_frames") or [])
    rep.info["replay_frames"] = frames
    rows = rep.info.get("trace_rows")
    if rows is not None and frames:
        if rows != frames - 1:
            rep.add(WARN, "auth.frame_rows",
                    f"trace 行数 {rows} != replay 帧数-1 ({frames - 1})")
    # replay filename should encode seed & dims (engine convention)
    if seed is not None and str(seed) not in replay.name:
        rep.add(WARN, "auth.filename", f"replay 文件名未含 seed={seed}: {replay.name}")
    # players agree
    rp_players = [p.get("name") for p in (rp.get("players") or [])]
    if rp_players and sorted(rp_players) != sorted(result.get("players") or []):
        rep.add(FAIL, "auth.players", f"replay players={rp_players} != result {result.get('players')}")


def _check_disk(g: Path, result: dict, rep: GameReport, run_dir: Path | None):
    run_dir = run_dir or g.parent
    # run.json must list this game's players
    run = _read_json(run_dir / "run.json")
    if run is None:
        rep.add(WARN, "disk.run_json", f"{run_dir.name}: 无 run.json")
    else:
        who = run.get("agent") or run.get("participants") or run.get("agents")
        if who and result.get("players"):
            rep.info["run_who"] = who
    # last_game copy check (only when a feedback dir exists for a player agent)
    for name in (result.get("players") or []):
        lg = paths.last_game_dir(name)
        if not (lg / "run.json").exists():
            continue
        lg_games = lg / "games"
        if not lg_games.exists():
            rep.add(WARN, "disk.last_game", f"{name}/last_game 无 games/")
            continue
        src = run_dir / g.name
        if src.exists():
            src_files = {p.name for p in src.iterdir() if p.is_file()}
            cop_files = {p.name for p in (lg_games / g.name).iterdir()
                         if p.is_file()} if (lg_games / g.name).exists() else set()
            if src_files and cop_files and src_files != cop_files:
                rep.add(WARN, "disk.last_game_copy",
                        f"{name}/last_game games 文件集与 runs/ 不一致")


def _cost_and_timeline(g: Path, result: dict, rep: GameReport):
    # --- per-bot cost estimate -------------------------------------------------
    exec_ms = result.get("execution_time_ms")
    cwd = g / "engine_cwd"
    costs = {}
    for name in (result.get("players") or []):
        costs[name] = {"bot_log_bytes": 0, "turns_logged": 0, "errorlog_bytes": 0}
    if cwd.is_dir():
        for i, name in enumerate(result.get("players") or []):
            for pat in (f"bot-{i}.log", f"bot-{i}*.log"):
                for f in cwd.glob(pat):
                    costs[name]["bot_log_bytes"] += f.stat().st_size
                    try:
                        costs[name]["turns_logged"] = sum(
                            1 for ln in f.open(encoding="utf-8", errors="replace")
                            if "TURN" in ln)
                    except Exception:
                        pass
                    break
            for f in cwd.glob(f"errorlog-*-{i}.log"):
                costs[name]["errorlog_bytes"] += f.stat().st_size
    rep.info["costs"] = costs
    rep.info["exec_ms"] = exec_ms

    # --- write-order timeline -------------------------------------------------
    timeline = []
    if cwd.is_dir():
        for f in cwd.iterdir():
            if f.is_file():
                timeline.append((f.stat().st_mtime, f"engine_cwd/{f.name}"))
    for f in g.iterdir():
        if f.is_file() and f.name != "result.json":
            timeline.append((f.stat().st_mtime, f.name))
    for f in (g / "result.json", g.parent / "run.json"):
        if f.exists():
            timeline.append((f.stat().st_mtime, f"{f.parent.name}/{f.name}"))
    timeline.sort()
    rep.info["timeline"] = [(_fmt_ts(t), n) for t, n in timeline]
    # sanity: run.json is written last by the orchestrator. Only flag when it is
    # clearly (>=1s) older than a per-game artifact — same-second writes (all
    # artifacts land within one second for fast games) make mtime ordering noisy.
    run_json_m = (g.parent / "run.json")
    if run_json_m.exists() and timeline:
        rj_t = run_json_m.stat().st_mtime
        game_ts = [t for t, n in timeline if not n.endswith("run.json")]
        if game_ts and rj_t < max(game_ts) - 1.0:
            rep.add(WARN, "order.run_json",
                    "run.json 明显早于对局产物（可能未在收尾写入）")

    # --- time / 2s-per-turn budget -------------------------------------------
    turns = (rep.info.get("replay_frames") or 0) - 1
    budget_s = turns * 2.0
    rep.info["turns"] = turns
    rep.info["turn_budget_s"] = budget_s
    if exec_ms is not None and turns > 0:
        rep.info["budget_ok"] = exec_ms <= budget_s * 1000
        if not rep.info["budget_ok"]:
            rep.add(WARN, "budget.per_turn",
                    f"整局 {exec_ms}ms 超出 {turns}×2s={budget_s:.0f}s 的每回合预算上限")


def verify_run(run_dir: Path) -> tuple[dict, list[GameReport]]:
    """Verify every game in a run dir. Returns (run_summary, game reports)."""
    run_dir = Path(run_dir)
    run = _read_json(run_dir / "run.json") or {}
    reps = []
    for g in sorted(run_dir.glob("game_*")):
        if g.is_dir():
            reps.append(verify_game(g, run_dir))
    return run, reps


def format_report(run_dir: Path, run: dict, reps: list[GameReport]) -> str:
    lines = []
    lines.append(f"=== verify: {run_dir.name} ===")
    lines.append(f"kind={run.get('kind')}  agent={run.get('agent')}  "
                 f"players={run.get('participants') or run.get('agents') or run.get('agent')}")
    nfail = sum(len(r.failures) for r in reps)
    nwarn = sum(len(r.warnings) for r in reps)
    for r in reps:
        status = "✅ ok" if r.ok and not r.warnings else ("❌ FAIL" if r.failures else "⚠ warn")
        lines.append(f"\n[{r.game}] {status}")
        for k, v in r.info.items():
            if k in ("timeline",):
                continue
            lines.append(f"    · {k}: {v}")
        for f in r.findings:
            mark = {"fail": "❌", "warn": "⚠", "ok": "✓"}[f.level]
            lines.append(f"    {mark} {f.check}: {f.detail}")
        if r.info.get("timeline"):
            lines.append("    · 写入顺序: " + " → ".join(n for _, n in r.info["timeline"]))
    lines.append(f"\n总计: {len(reps)} 局, {nfail} fail, {nwarn} warn")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    from . import config as C
    ap = argparse.ArgumentParser(prog="arena.verify")
    ap.add_argument("target", help="run 目录或 game_* 目录")
    args = ap.parse_args(argv)
    t = Path(args.target)
    if not t.is_absolute():
        t = paths.runs_root() / t if (paths.runs_root() / t).exists() else t
    if t.name.startswith("game_") or (t / "result.json").exists():
        rep = verify_game(t)
        run = _read_json(t.parent / "run.json") or {}
        print(format_report(t.parent, run, [rep]))
        return 0 if rep.ok else 1
    run, reps = verify_run(t)
    print(format_report(t, run, reps))
    return 0 if all(r.ok for r in reps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
