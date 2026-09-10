"""Logistic ELO implementation for the bot ladder.

Standard chess ELO: expected score = 1/(1+10^((b-a)/400)); update with K.
Also helpers to convert a win-rate to an ELO difference and to bulk-update
from pairwise match counts (each pair plays G games, A wins W).

The pilot spec: every pair plays 51 games; 26 wins (>50%) marks A as stronger.
51 games / 26 wins corresponds to an ELO diff of ~+7, which is below game
noise (~±100 for 51 games) — the pairwise result is a per-pair SIGN, not a
precise rating. Ratings are accumulated across all pairs so the aggregate is
the meaningful signal.
"""
from __future__ import annotations

import math


def expected(a: float, b: float) -> float:
    """Expected score of rating a vs b (0..1)."""
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def elo_diff_from_winrate(wr: float) -> float:
    """ELO difference implied by a win-rate (positive = stronger)."""
    if wr <= 0.0:
        return -1000.0
    if wr >= 1.0:
        return 1000.0
    return 400.0 * math.log10(wr / (1.0 - wr))


def update(a: float, b: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """One match outcome: score_a in {0, 0.5, 1}. Returns (new_a, new_b)."""
    ea = expected(a, b)
    eb = 1.0 - ea
    return a + k * (score_a - ea), b + k * ((1.0 - score_a) - eb)


class Ladder:
    """Accumulates pairwise match counts and computes ratings.

    Scores follow the 51/26 pilot spec: each pair plays ``games_per_pair``
    games (odd), a pair "win" for A is wins >= (games_per_pair+1)//2. Ratings
    are computed by repeated round-robin batch updates (standard for offline
    Elo) over every pair's aggregate outcome, seeded at 1500.
    """

    def __init__(self, initial: float = 1500.0, k: float = 24.0, games_per_pair: int = 51):
        self.initial = initial
        self.k = k
        self.games_per_pair = games_per_pair
        self._ratings: dict[str, float] = {}
        self._pairs: list[tuple[str, str, int, int]] = []  # (a, b, a_wins, b_wins)

    def add_pair_result(self, a: str, b: str, a_wins: int, b_wins: int) -> None:
        self._pairs.append((a, b, a_wins, b_wins))
        for bot in (a, b):
            self._ratings.setdefault(bot, self.initial)

    def pairwise_winner(self, a: str, b: str, a_wins: int, b_wins: int) -> str | None:
        """Return the winner's name under the 'X of N' rule (None on tie).

        Both bots and their win counts are required, so the result is directly
        usable (previously the body referenced undefined names and raised
        NameError; it now takes identity as arguments).
        """
        need = (self.games_per_pair + 1) // 2
        if a_wins >= need:
            return a
        if b_wins >= need:
            return b
        return None

    def compute(self, iterations: int = 200) -> dict[str, float]:
        """Iterative batch update until convergence; returns {bot: rating}."""
        r = dict(self._ratings)
        for _ in range(iterations):
            max_delta = 0.0
            for a, b, aw, bw in self._pairs:
                n = aw + bw
                if n == 0:
                    continue
                # aggregate score over the pair (0..1 for A)
                score_a = aw / n
                ea = expected(r[a], r[b])
                k = self.k * math.sqrt(n) / (1.0 + 0.5 * abs(score_a - ea))
                k = min(k, 160.0)
                delta = k * (score_a - ea)
                r[a] += delta
                r[b] -= delta
                max_delta = max(max_delta, abs(delta))
            if max_delta < 0.5:
                break
        self._ratings = r
        return dict(r)

    def rating(self, bot: str) -> float:
        return self._ratings.get(bot, self.initial)

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self._ratings.items(), key=lambda kv: -kv[1])
