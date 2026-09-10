"""S1 unit tests for the arena platform (pure stdlib — no torch, no weights, no engine).

Run from the repo root:

    python3 -m unittest discover -s tests/regression -p 'test_s1_*.py' -v
    # or directly
    python3 tests/regression/test_s1_unit.py -v

Covers the pure functions the platform's correctness rests on:
ELO math, tolerant JSON parsing, trace counting, deterministic pairing
(including the cross-process seed regression), pool id→file rules, and bot
command construction.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARENA = REPO / "halite-arena"
sys.path.insert(0, str(ARENA))

from arena import _elo, _rl_compat, pairing  # noqa: E402
from arena import bots as bots_mod  # noqa: E402
from arena import elo as elo_mod  # noqa: E402
from arena import paths as paths_mod  # noqa: E402
from arena import pool as pool_mod  # noqa: E402


class TestEloMath(unittest.TestCase):
    def test_expected_symmetric(self):
        self.assertAlmostEqual(_elo.expected(1500, 1500), 0.5, places=9)
        for a, b in [(1500, 1800), (2000, 1200), (900, 1600)]:
            self.assertAlmostEqual(_elo.expected(a, b) + _elo.expected(b, a), 1.0, places=9)

    def test_expected_monotonic(self):
        self.assertGreater(_elo.expected(1800, 1500), 0.5)
        self.assertLess(_elo.expected(1500, 1800), 0.5)

    def test_diff_from_winrate(self):
        self.assertAlmostEqual(_elo.elo_diff_from_winrate(0.5), 0.0, places=9)
        self.assertGreater(_elo.elo_diff_from_winrate(0.75), 0.0)
        self.assertLess(_elo.elo_diff_from_winrate(0.25), 0.0)
        # clamp at the extremes (no ZeroDivisionError)
        self.assertEqual(_elo.elo_diff_from_winrate(1.0), 1000.0)
        self.assertEqual(_elo.elo_diff_from_winrate(0.0), -1000.0)

    def test_update_is_zero_sum(self):
        a, b = 1500.0, 1500.0
        na, nb = _elo.update(a, b, 1.0)
        self.assertGreater(na, a)
        self.assertLess(nb, b)
        self.assertAlmostEqual((na - a) + (nb - b), 0.0, places=9)


class TestLadder(unittest.TestCase):
    def test_pairwise_winner_threshold_51(self):
        lad = _elo.Ladder(games_per_pair=51)
        need = (51 + 1) // 2  # 26
        self.assertEqual(lad.pairwise_winner(need, need - 1), "a")
        self.assertEqual(lad.pairwise_winner(need - 1, need), "b")
        self.assertIsNone(lad.pairwise_winner(need - 1, need - 1))

    def test_compute_equal_results_stay_at_initial(self):
        lad = _elo.Ladder()
        for _ in range(3):
            lad.add_pair_result("a", "b", 1, 1)
        r = lad.compute()
        self.assertAlmostEqual(r["a"], r["b"], places=6)
        self.assertAlmostEqual(r["a"], 1500.0, places=6)

    def test_compute_orders_stronger_higher(self):
        lad = _elo.Ladder()
        lad.add_pair_result("strong", "weak", 9, 1)
        lad.add_pair_result("weak", "mid", 1, 9)
        lad.add_pair_result("strong", "mid", 8, 2)
        r = lad.compute()
        self.assertGreater(r["strong"], r["mid"])
        self.assertGreater(r["mid"], r["weak"])

    def test_compute_is_deterministic(self):
        def build():
            lad = _elo.Ladder()
            for a, b, aw, bw in [("x", "y", 7, 3), ("y", "z", 6, 4), ("x", "z", 9, 1)]:
                lad.add_pair_result(a, b, aw, bw)
            return lad.compute()

        self.assertEqual(build(), build())

    def test_empty_pair_ignored(self):
        lad = _elo.Ladder()
        lad.add_pair_result("a", "b", 0, 0)
        r = lad.compute()
        self.assertAlmostEqual(r["a"], 1500.0, places=6)


class TestEloPersistenceWithAnchors(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = paths_mod.state_dir
        paths_mod.state_dir = lambda: Path(self._tmp.name)  # type: ignore[assignment]

    def tearDown(self):
        paths_mod.state_dir = self._orig  # type: ignore[assignment]
        self._tmp.cleanup()

    def test_anchor_is_frozen(self):
        ratings = elo_mod.apply_pair_results(
            [("agent:x", "script:eco", 1, 1)],
            games_per_pair=3,
            anchors={"script:eco": 1500.0},
        )
        self.assertEqual(ratings["script:eco"], 1500.0)  # anchor never drifts
        self.assertIn("agent:x", ratings)

    def test_history_accumulates(self):
        elo_mod.apply_pair_results([("a", "b", 1, 0)], games_per_pair=3)
        elo_mod.apply_pair_results([("a", "b", 0, 1)], games_per_pair=3)
        matches = elo_mod.load_matches()
        self.assertEqual(len(matches), 2)  # both rounds persisted


class TestTolerantParsing(unittest.TestCase):
    def test_parse_clean(self):
        self.assertEqual(_rl_compat.parse_json('{"a": 1}'), {"a": 1})

    def test_parse_with_surrounding_noise(self):
        self.assertEqual(_rl_compat.parse_json('log line\n{"a": 1}\n'), {"a": 1})

    def test_parse_nested_braces(self):
        self.assertEqual(_rl_compat.parse_json('junk {"a": {"b": 2}} tail'), {"a": {"b": 2}})

    def test_parse_empty_and_none(self):
        for bad in ["", "   ", None]:
            self.assertIsNone(_rl_compat.parse_json(bad))

    def test_parse_garbage_returns_none(self):
        self.assertIsNone(_rl_compat.parse_json("no json here at all"))

    def test_read_trace_counts_aggregates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text(
                json.dumps({"action_counts": {"mine": 2, "move": 1}}) + "\n"
                + json.dumps({"action_counts": {"mine": 3}}) + "\n"
            )
            self.assertEqual(_rl_compat.read_trace_counts(p), {"mine": 5, "move": 1})

    def test_read_trace_counts_skips_torn_last_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            p.write_text(json.dumps({"action_counts": {"move": 1}}) + "\n" + '{"action_counts": {"mo')
            self.assertEqual(_rl_compat.read_trace_counts(p), {"move": 1})

    def test_read_trace_counts_missing_file(self):
        self.assertEqual(_rl_compat.read_trace_counts(Path("/nonexistent/x.jsonl")), {})


class TestPairingDeterminism(unittest.TestCase):
    """practice seeds must be reproducible: seed_for must not use salted hash()."""

    def test_seed_for_matches_stable_reference(self):
        for name in ["eco", "aggro", "control", "adaptive", "rl:cycle002"]:
            self.assertEqual(pairing.seed_for(name), zlib.crc32(name.encode("utf-8")) % 997)

    def test_seed_for_in_range(self):
        for name in ["eco", "x" * 200, "", "rl:perturb_sig00.19_s24"]:
            self.assertGreaterEqual(pairing.seed_for(name), 0)
            self.assertLess(pairing.seed_for(name), 997)

    def test_seed_for_stable_across_processes(self):
        """Regression: hash() is salted per process; CRC32 must not be."""
        code = "import sys;sys.path.insert(0,%r);from arena.pairing import seed_for;print(seed_for('eco'))" % str(ARENA)
        outs = set()
        for hseed in ["random", "0", "12345"]:
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = hseed
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            self.assertEqual(out.returncode, 0, out.stderr)
            outs.add(out.stdout.strip())
        self.assertEqual(len(outs), 1, f"seed_for differs across processes: {outs}")

    def test_practice_cases_shape(self):
        a = bots_mod.BotSpec(name="agent:testbot", kind="agent", tag="testbot")
        t1 = bots_mod.script_spec("eco")
        t2 = bots_mod.script_spec("aggro")
        cases = pairing.practice_cases(a, [t1, t2], seeds=3)
        self.assertEqual(len(cases), 2 * 2 * 3)  # 2 targets * 3 seeds * 2 sides
        # same seed is played from both sides (fairness)
        for i in range(0, len(cases), 2):
            self.assertEqual(cases[i].seed, cases[i + 1].seed)
            self.assertEqual({cases[i].side_a, cases[i + 1].side_a}, {0, 1})

    def test_practice_cases_deterministic(self):
        a = bots_mod.BotSpec(name="agent:testbot", kind="agent", tag="testbot")
        t = [bots_mod.script_spec("eco")]
        self.assertEqual(pairing.practice_cases(a, t, seeds=2), pairing.practice_cases(a, t, seeds=2))

    def test_round_cases_both_sides(self):
        agents = [bots_mod.BotSpec(name=f"agent:a{i}", kind="agent", tag=f"a{i}") for i in range(3)]
        specs = pairing.round_cases(agents, [bots_mod.script_spec("eco")], games_av=2, games_pool=2)
        # 3 agent pairs * 2 games * 2 sides  +  3 agents * 1 pool * 2 games * 2 sides
        self.assertEqual(len(specs), 3 * 2 * 2 + 3 * 1 * 2 * 2)
        self.assertTrue(all(c.side_a in (0, 1) for c in specs))

    def test_roundrobin_rejects_even_games(self):
        bots = [bots_mod.BotSpec(name=f"b{i}", kind="agent", tag=f"b{i}") for i in range(3)]
        # even count would give an unfair pairing; must fall back to an odd >= 3
        cases = pairing.roundrobin_cases(bots, games_per_pair=4)
        self.assertTrue(len(cases) > 0)


class TestPoolIdRule(unittest.TestCase):
    def test_perturb_prefix_stripped(self):
        self.assertEqual(pool_mod._id_to_file("perturb_sig00.19_s24"), "sig00.19_s24.pt")

    def test_plain_id_passthrough(self):
        self.assertEqual(pool_mod._id_to_file("distill_h128"), "distill_h128.pt")
        self.assertEqual(pool_mod._id_to_file("hist_frozen_v4_r3_gen2_cycle002"),
                         "hist_frozen_v4_r3_gen2_cycle002.pt")

    def test_script_specs_keys(self):
        self.assertEqual(set(pool_mod.script_specs()), {"eco", "aggro", "control", "adaptive"})


class TestBotCommands(unittest.TestCase):
    def test_unknown_script_raises(self):
        with self.assertRaises(ValueError):
            bots_mod.script_spec("does-not-exist")

    def test_script_command_has_seed_offset_and_params(self):
        spec = bots_mod.script_spec("eco")
        cmd = bots_mod.build_bot_command(spec, 7)
        self.assertIn(str(7 + 100000), cmd)
        self.assertIn("bot_params_eco.txt", cmd)

    def test_rl_command_flags(self):
        spec = bots_mod.rl_spec("cycle002", checkpoint=Path("/tmp/ck.pt"), device="cpu")
        cmd = bots_mod.build_bot_command(spec, 1, trace_path=Path("/tmp/tr.jsonl"))
        for token in ["--checkpoint", "--device", "--feature-schema", "--trace", "cpu"]:
            self.assertIn(token, cmd)

    def test_rl_requires_checkpoint(self):
        spec = bots_mod.rl_spec("cycle002", checkpoint=None)
        with self.assertRaises(ValueError):
            bots_mod.build_bot_command(spec, 1)

    def test_agent_short_name(self):
        spec = bots_mod.BotSpec(name="agent:testbot", kind="agent", tag="testbot")
        self.assertEqual(spec.short, "testbot")


if __name__ == "__main__":
    unittest.main(verbosity=2)
