"""ELO persistence for arena matches (thin wrapper over rl_ai.ranking.elo.Ladder)."""
from __future__ import annotations

import json
from pathlib import Path

from . import paths
try:  # prefer the training-side implementation when it is available
    from rl_ai.ranking.elo import Ladder
except Exception:  # standalone clone: use the vendored pure-stdlib copy
    from ._elo import Ladder


def _matches_path() -> Path:
    return paths.state_dir() / "matches.jsonl"


def _ratings_path() -> Path:
    return paths.state_dir() / "ratings.json"


def load_matches() -> list[dict]:
    p = _matches_path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def apply_pair_results(pairs: list[tuple[str, str, int, int]],
                       games_per_pair: int = 11,
                       anchors: dict[str, float] | None = None,
                       run_key: str | None = None) -> dict[str, float]:
    """Recompute ratings from history + new pairs. Pairs: (bot_a, bot_b, a_wins, b_wins).

    anchors: {bot_name: fixed_rating} — those bots keep a fixed reference score
    (the script pool / known RL models). Ratings are computed over all pairs,
    then anchor entries are RESET to their fixed value, so agent ratings drift
    relative to a stable absolute scale instead of floating together.

    run_key: identifier of the run these pairs belong to (normally the run dir).
    Previously-recorded rows with the same key are replaced, so re-running or
    retrying a match is idempotent instead of silently double-counting its
    games (which inflated the ELO weight of that match).
    """
    # Create the state dir BEFORE opening matches.jsonl: on a fresh checkout the
    # append/open used to raise FileNotFoundError.
    state = paths.state_dir()
    state.mkdir(parents=True, exist_ok=True)

    matches = load_matches()
    if run_key is not None:
        matches = [m for m in matches if m.get("run") != run_key]

    history_pairs = [(m["a"], m["b"], m["a_wins"], m["b_wins"]) for m in matches]
    stamp = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    new_rows = []
    for pr in pairs:
        history_pairs.append(pr)
        new_rows.append({"a": pr[0], "b": pr[1], "a_wins": pr[2], "b_wins": pr[3],
                         "date": stamp, "run": run_key})
    # Rewrite history (deduped) + the new rows, instead of blind-appending.
    with open(_matches_path(), "w", encoding="utf-8") as fh:
        for m in matches:
            fh.write(json.dumps(m) + "\n")
        for row in new_rows:
            fh.write(json.dumps(row) + "\n")

    ladder = Ladder(games_per_pair=games_per_pair)
    for a, b, aw, bw in history_pairs:
        ladder.add_pair_result(a, b, aw, bw)
    ratings = ladder.compute()
    # 锚点冻结：参与对局的锚点强制回写固定分（不参与漂移）
    for k, v in (anchors or {}).items():
        if k in ratings:
            ratings[k] = v
    # 合并历史（保留未参赛的旧 rating）
    old = latest_ratings()
    for k, v in old.items():
        ratings.setdefault(k, v)
    # 锚点即使历史上没参赛也写入（rating 榜一开始就有参照）
    for k, v in (anchors or {}).items():
        ratings.setdefault(k, v)
    paths.state_dir().mkdir(parents=True, exist_ok=True)
    _ratings_path().write_text(json.dumps(ratings, indent=1), encoding="utf-8")
    return ratings


def latest_ratings() -> dict[str, float]:
    p = _ratings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}
