"""Match orchestration: practice (agent vs targets) & match (round-robin)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import paths
from .bots import BotSpec
from .pairing import practice_cases, roundrobin_cases
from .engine_runner import GameRecord, run_one_game


def _case_game_dir(run_id: Path, ci: int) -> Path:
    return run_id / f"game_{ci:06d}"


def _run_cases(cases, *, game_dir_root: Path, engine_kwargs: dict) -> list[tuple[int, GameRecord]]:
    """Run cases in parallel; returns [(case_index, record)]."""
    results = {}
    # Don't mutate the caller's dict: popping "workers" in place silently reset
    # a reused kwargs dict's worker count back to 2 on the next call.
    workers = engine_kwargs.get("workers", 2)
    kwargs = {k: v for k, v in engine_kwargs.items() if k != "workers"}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for ci, case in enumerate(cases):
            gd = _case_game_dir(game_dir_root, ci)
            # Apply the case's seat assignment: side_a==1 means spec_b plays
            # side 0 (same seed, sides swapped) so map/first-move bias cancels.
            # GameRecord.margin is side-0 relative, so callers must read side_a.
            spec0, spec1 = ((case.spec_a, case.spec_b) if case.side_a == 0
                            else (case.spec_b, case.spec_a))
            futs[ex.submit(run_one_game, spec0, spec1, case.seed,
                           game_dir=gd, **kwargs)] = ci
        for fut in as_completed(futs):
            ci = futs[fut]
            gd = _case_game_dir(game_dir_root, ci)
            try:
                rec = fut.result()
                # 每局结果归一化落盘 result.json（agent 可读）
                rec_dict = {k: (str(v) if hasattr(v, "__fspath__") else v)
                            for k, v in vars(rec).items() if not k.startswith("_")}
                try:
                    import json
                    (gd / "result.json").write_text(
                        json.dumps(rec_dict, indent=1, default=str), encoding="utf-8")
                except Exception:
                    pass
                # 生成逐回合轨迹
                if rec.replay is not None and rec.replay.exists():
                    try:
                        from .trace import write_trace
                        write_trace(rec.replay)
                    except Exception:
                        pass
                results[ci] = rec
            except Exception as e:
                results[ci] = GameRecord(ok=False, players=["?", "?"], scores=[0, 0],
                                         ranks=[], winner=None, margin=0,
                                         fatal="internal", stderr_tail=str(e)[:500])
    return [results[i] for i in sorted(results)]


def run_practice(agent_name: str, targets: list[BotSpec], *, seeds: int = 2,
                 workers: int = 2, save_replay: bool = True, device: str = "auto",
                 wall_timeout: float | None = None, engine_no_timeout: bool = False,
                 tag: str = "practice") -> Path:
    from .bots import agent_spec
    spec = agent_spec(agent_name)
    run_id = paths.new_run_id(tag)
    run_id.mkdir(parents=True, exist_ok=True)
    cases = practice_cases(spec, targets, seeds)
    engine_kwargs = dict(workers=workers, save_replay=save_replay, device=device,
                         wall_timeout=wall_timeout, engine_no_timeout=engine_no_timeout)
    recs = _run_cases(cases, game_dir_root=run_id, engine_kwargs=engine_kwargs)

    # 汇总（按对手 + 双侧配对）
    from collections import defaultdict
    by_opp: dict[str, dict] = defaultdict(lambda: {"games": 0, "ok": 0, "wins": 0, "margins": []})
    for case, rec in zip(cases, recs):
        # 决定哪一方是 agent：按 short 名匹配
        other = case.spec_b if case.spec_a.short == agent_name else case.spec_a
        opp = other.short
        ag = by_opp[opp]
        ag["games"] += 1
        if not rec.ok:
            continue
        ag["ok"] += 1
        # margin 是 side-0 视角；先换算回 spec_a 视角，再取 agent 视角。
        # （side_a==1 时 spec_b 坐 side 0，故 spec_a 视角需取反）
        a_margin = rec.margin if case.side_a == 0 else -rec.margin
        agent_is_a = case.spec_a.short == agent_name
        agent_margin = a_margin if agent_is_a else -a_margin
        ag["margins"].append(agent_margin)
        ag["wins"] += 1 if agent_margin > 0 else 0
    summary = {}
    for opp, d in sorted(by_opp.items()):
        ok = max(1, d["ok"])
        summary[opp] = {"games": d["games"], "ok": d["ok"], "win_rate": round(d["wins"] / ok, 3),
                        "mean_margin": round(sum(d["margins"]) / len(d["margins"]), 1)
                        if d["margins"] else 0.0}
    run = {"schema": "halite-arena/run-v1", "run_id": str(run_id), "kind": "practice",
           "agent": agent_name, "seeds_per_target": seeds, "summary": summary,
           "failed": sum(1 for r in recs if not r.ok)}
    (run_id / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return run_id
def run_match(bots_specs, *, games_per_pair: int = 11, workers: int = 2,
              save_replay: bool = False, device: str = "auto",
              wall_timeout: float | None = None, engine_no_timeout: bool = False,
              tag: str = "match") -> tuple[Path, dict]:
    """Round-robin among bots_specs (agents + optional pool). Updates ELO."""
    from .pairing import roundrobin_cases
    from . import elo as elo_mod
    run_id = paths.new_run_id(tag)
    run_id.mkdir(parents=True, exist_ok=True)
    cases = roundrobin_cases(bots_specs, games_per_pair)
    engine_kwargs = dict(workers=workers, save_replay=save_replay, device=device,
                         wall_timeout=wall_timeout, engine_no_timeout=engine_no_timeout)
    recs = _run_cases(cases, game_dir_root=run_id, engine_kwargs=engine_kwargs)

    # 配对聚合: 双侧都 ok 才算完整 pair（同 paired_summary 语义）
    from collections import defaultdict
    games_by_pair: dict = defaultdict(list)   # (a,b)->[(case, rec)]
    for case, rec in zip(cases, recs):
        if not rec.ok:
            continue
        games_by_pair[(case.spec_a.short, case.spec_b.short)].append((case, rec))
    pairs = []
    summary = {}
    for (a, b), glist in games_by_pair.items():
        # 只在两侧都有局时算 pair（a 先手 + a 后手）
        sides = set(c.side_a for c, _ in glist)
        if len(sides) < 2:
            continue
        a_wins = sum(1 for c, r in glist
                     if (c.side_a == 0 and r.margin > 0) or (c.side_a == 1 and r.margin < 0))
        b_wins = sum(1 for c, r in glist
                     if (c.side_a == 0 and r.margin < 0) or (c.side_a == 1 and r.margin > 0))
        n = len(glist)
        pairs.append((a, b, a_wins, b_wins))
        summary.setdefault(a, {})[b] = {"games": n, "a_wins": a_wins, "b_wins": b_wins}
        summary.setdefault(b, {})[a] = {"games": n, "a_wins": b_wins, "b_wins": a_wins}

    ratings_before = dict(elo_mod.latest_ratings())
    ratings_after = elo_mod.apply_pair_results(
        pairs, games_per_pair=games_per_pair, run_key=str(run_id)) if pairs else ratings_before

    run = {"schema": "halite-arena/run-v1", "run_id": str(run_id), "kind": "match",
           "participants": [s.short for s in bots_specs], "games_per_pair": games_per_pair,
           "summary": summary, "failed": sum(1 for r in recs if not r.ok),
           "ratings_before": {k: round(v, 1) for k, v in ratings_before.items()},
           "ratings_after": {k: round(v, 1) for k, v in ratings_after.items()}}
    (run_id / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return run_id, ratings_after