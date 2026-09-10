"""Pairing / case generation for practice & round-robin matches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .bots import BotSpec


@dataclass(frozen=True)
class Case:
    spec_a: BotSpec
    spec_b: BotSpec
    seed: int
    side_a: int          # 0: a 先手; 1: a 后手（同 seed 双局消偏）


def seed_for(name: str) -> int:
    """Deterministic per-opponent seed offset (mirror evaluate.seed_offset)."""
    return abs(hash(name)) % 997


def round_cases(agents: Sequence[BotSpec], pool: Sequence[BotSpec],
                games_av: int = 2, games_pool: int = 2, base: int = 0) -> list[Case]:
    """Official round: every agent vs every other agent (games_av each, both
    sides) + every agent vs every pool target (games_pool each, both sides).
    Deterministic seeds; same seed swapped sides for fairness."""
    cases: list[Case] = []
    n = len(agents)
    base_off = base
    # agent vs agent
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = agents[i], agents[j]
            for g in range(games_av):
                s = base_off + pair_idx * 100_000 + g
                cases.append(Case(a, b, s, 0))
                cases.append(Case(a, b, s, 1))
            pair_idx += 1
    # agent vs pool
    for i, a in enumerate(agents):
        for pi, t in enumerate(pool):
            s = base_off + 1_000_000 + i * 10_000 + pi * 10
            for g in range(games_pool):
                cases.append(Case(a, t, s + g, 0))
                cases.append(Case(a, t, s + g, 1))
    return cases


def practice_cases(spec_a: BotSpec, targets: Sequence[BotSpec], seeds: int,
                   base: int = 0) -> list[Case]:
    cases = []
    for t in targets:
        off = seed_for(t.name)
        for i in range(seeds):
            s = base + i * 1000 + off
            cases.append(Case(spec_a, t, s, 0))
            cases.append(Case(spec_a, t, s, 1))   # 同 seed 互换先后手
    return cases


def roundrobin_cases(bots: Sequence[BotSpec], games_per_pair: int = 11,
                     base: int = 0) -> list[Case]:
    """All unordered pairs (i<j); same-seed both sides for fairness.
    games_per_pair must be odd >= 3 so each pair has at least one same-seed
    double (side 0 + side 1)."""
    if games_per_pair < 3 or games_per_pair % 2 == 0:
        games_per_pair = 3
    cases = []
    n = len(bots)
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bots[i], bots[j]
            seed_base = base + pair_idx * 100_000
            halves = games_per_pair // 2
            for k in range(halves):
                s = seed_base + k * 1000
                cases.append(Case(a, b, s, 0))
                cases.append(Case(a, b, s, 1))
            if games_per_pair % 2 == 1:      # 奇数补一局
                cases.append(Case(a, b, seed_base + 999, 0))
            pair_idx += 1
    return cases
