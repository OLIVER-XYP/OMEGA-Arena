"""Official round: run every registered agent vs every other + vs script anchors.

Each round produces a runs/<ts>-roundN/ directory with:
- run.json  (summary, full pairwise matrix, ratings delta)
- game_*/   per-case artifacts
Then appends one line to state/rounds.jsonl and refreshes each agent's
last_game/ feedback.

ELO: only agents' ratings move; the script pool (eco/aggro/control/adaptive)
is a FROZEN anchor set (fixed reference ratings) so agent strength is measured
on an absolute scale, not a floating self-referential one.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from . import paths
from . import compile as compile_mod
from . import platform
from .bots import BotSpec, agent_spec, script_spec
from .pairing import round_cases
from .engine_runner import GameRecord
from . import runner as runner_mod


# script pool used as the frozen ELO anchor set in official rounds.
# Values chosen so agents start around/below the strongest script.
ANCHOR_RATINGS: dict[str, float] = {
    "eco": 1450.0,
    "aggro": 1500.0,
    "control": 1600.0,
    "adaptive": 1550.0,
}


def _anchor_specs() -> list[BotSpec]:
    return [script_spec(k) for k in C.SCRIPT_PARAMS]


def run_round(*, label: str = "round", games_av: int = 2, games_pool: int = 2,
              workers: int = 2, save_replay: bool = True, device: str = "auto",
              wall_timeout: float | None = None,
              engine_no_timeout: bool = False) -> tuple[Path, dict]:
    """Compile + run one official round over all registered agents."""
    from . import elo as elo_mod
    scratch_root = paths.agents_root() / "scratch"
    agents = [m for m in paths.agents_list() if m.get("kind", "cpp") == "cpp"
              and not (scratch_root / m["name"]).exists()]
    if not agents:
        raise RuntimeError("无已注册 agent（先 `arena init <name>`）")

    # 1) compile every agent's current submit; a compile failure marks that
    #    agent as technically failed this round but does NOT abort the round.
    agent_specs: list[BotSpec] = []
    compile_results = {}
    for meta in agents:
        name = meta["name"]
        res = compile_mod.compile_agent(name)
        compile_results[name] = res
        if res.ok:
            agent_specs.append(agent_spec(name))
        else:
            print(f"[round] {name}: 编译失败 ({res.error_kind})，本轮记技术性失败并跳过")
            platform.write_compile_feedback(name, res)

    anchors = _anchor_specs()
    run_id = paths.new_run_id(label)
    run_id.mkdir(parents=True, exist_ok=True)

    if agent_specs:
        cases = round_cases(agent_specs, anchors, games_av=games_av, games_pool=games_pool)
        engine_kwargs = dict(workers=workers, save_replay=save_replay, device=device,
                             wall_timeout=wall_timeout,
                             engine_no_timeout=engine_no_timeout)
        recs = runner_mod._run_cases(cases, game_dir_root=run_id, engine_kwargs=engine_kwargs)
    else:
        cases, recs = [], []

    # 2) aggregate per-pair (only complete both-sided pairs count, mirroring
    #    run_match). Pairs are recorded as (a_name, b_name, a_wins, b_wins).
    from collections import defaultdict
    games_by_pair: dict = defaultdict(list)
    for case, rec in zip(cases, recs):
        if not rec.ok:
            continue
        games_by_pair[(case.spec_a.short, case.spec_b.short)].append((case, rec))
    summary: dict = {}
    pairs = []
    for (a, b), glist in games_by_pair.items():
        sides = {c.side_a for c, _ in glist}
        if len(sides) < 2:
            continue
        a_wins = sum(1 for c, r in glist
                     if (c.side_a == 0 and r.margin > 0) or (c.side_a == 1 and r.margin < 0))
        b_wins = sum(1 for c, r in glist
                     if (c.side_a == 0 and r.margin < 0) or (c.side_a == 1 and r.margin > 0))
        pairs.append((a, b, a_wins, b_wins))
        summary.setdefault(a, {})[b] = {"games": len(glist), "a_wins": a_wins, "b_wins": b_wins}
        summary.setdefault(b, {})[a] = {"games": len(glist), "a_wins": b_wins, "b_wins": a_wins}

    # 3) ELO: recompute history + this round's pairs; script anchors frozen.
    ratings_before = dict(elo_mod.latest_ratings())
    ratings_after = ratings_before
    if pairs:
        ratings_after = elo_mod.apply_pair_results(
            pairs, games_per_pair=max(3, games_av + games_pool),
            anchors=ANCHOR_RATINGS)

    run = {
        "schema": "halite-arena/run-v2", "run_id": str(run_id), "kind": "round",
        "label": label,
        "agents": [s.short for s in agent_specs],
        "anchors": [s.short for s in anchors],
        "compile_ok": {k: v.ok for k, v in compile_results.items()},
        "games_av": games_av, "games_pool": games_pool,
        "summary": summary,
        "failed": sum(1 for r in recs if not r.ok),
        "ratings_before": {k: round(v, 1) for k, v in ratings_before.items()},
        "ratings_after": {k: round(v, 1) for k, v in ratings_after.items()},
    }
    (run_id / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")

    # 4) persist a rounds.jsonl line + refresh every agent's last_game feedback
    rounds_path = paths.state_dir() / "rounds.jsonl"
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    with open(rounds_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"run_id": str(run_id), "label": label,
                             "date": __import__("datetime").datetime.now().isoformat(
                                 timespec="seconds"),
                             "agents": run["agents"], "summary": summary}) + "\n")
    for s in agent_specs:
        platform.write_run_feedback(s.name.split(":", 1)[1], run_id)
    return run_id, ratings_after
