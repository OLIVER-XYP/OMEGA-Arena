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
                       anchors: dict[str, float] | None = None) -> dict[str, float]:
    """Recompute ratings from history + new pairs. Pairs: (bot_a, bot_b, a_wins, b_wins).

    anchors: {bot_name: fixed_rating} — those bots keep a fixed reference score
    (the script pool / known RL models). Ratings are computed over all pairs,
    then anchor entries are RESET to their fixed value, so agent ratings drift
    relative to a stable absolute scale instead of floating together.
    """
    matches = load_matches()
    history_pairs = [(m["a"], m["b"], m["a_wins"], m["b_wins"]) for m in matches]
    for pr in pairs:
        history_pairs.append(pr)
        matches.append({"a": pr[0], "b": pr[1], "a_wins": pr[2], "b_wins": pr[3],
                        "date": __import__("datetime").datetime.now().isoformat(timespec="seconds")})
    # 只保留新增的对局到 matches.jsonl
    with open(_matches_path(), "a", encoding="utf-8") as fh:
        for pr in pairs:
            fh.write(json.dumps({"a": pr[0], "b": pr[1], "a_wins": pr[2], "b_wins": pr[3],
                                 "date": __import__("datetime").datetime.now().isoformat(
                                     timespec="seconds")}) + "\n")

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
