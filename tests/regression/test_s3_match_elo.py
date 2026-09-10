"""S3 integration (CPU-only): a round-robin match produces sound ELO.

Exercises the full match path — case generation, seat-swapped games, pair
aggregation, and ELO persistence — between two built-in script bots. No torch
and no RL pool are needed, so it runs anywhere S2 runs.

State is redirected to a temp dir (runs/ and state/) so the real workspace's
ELO ladder and run archive are never touched.

    python3 -m unittest discover -s tests/regression -p 'test_s3_*.py' -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARENA = REPO / "halite-arena"
sys.path.insert(0, str(ARENA))

from arena import config as C  # noqa: E402

SKIP = os.environ.get("ARENA_SKIP_INTEGRATION") == "1"
GAMES = 3


@unittest.skipIf(SKIP, "ARENA_SKIP_INTEGRATION=1")
@unittest.skipUnless(C.ENGINE.exists(), f"engine not built: {C.ENGINE}")
@unittest.skipUnless(C.SCRIPT_EXE.exists(), f"script bot not built: {C.SCRIPT_EXE}")
class TestMatchElo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from arena import bots as bots_mod
        from arena import paths as paths_mod
        from arena import runner

        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls._orig_runs, cls._orig_state = paths_mod.runs_root, paths_mod.state_dir
        cls._repo_state = cls._orig_state()
        before = sorted(p.name for p in cls._repo_state.glob("*")) if cls._repo_state.exists() else []
        paths_mod.runs_root = lambda: root / "runs"      # type: ignore[assignment]
        paths_mod.state_dir = lambda: root / "state"     # type: ignore[assignment]
        try:
            specs = [bots_mod.script_spec("eco"), bots_mod.script_spec("aggro")]
            cls.run_id, cls.ratings = runner.run_match(
                specs, games_per_pair=GAMES, workers=2, save_replay=False)
            cls.state_files = sorted(p.name for p in (root / "state").glob("*"))
        finally:
            paths_mod.runs_root, paths_mod.state_dir = cls._orig_runs, cls._orig_state
        after = sorted(p.name for p in cls._repo_state.glob("*")) if cls._repo_state.exists() else []
        cls.repo_state_unchanged = before == after
        cls.run_meta = json.loads((cls.run_id / "run.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_run_completed_without_failures(self):
        self.assertEqual(self.run_meta["kind"], "match")
        self.assertEqual(self.run_meta["failed"], 0)
        self.assertEqual(self.run_meta["games_per_pair"], GAMES)

    def test_expected_number_of_games_and_both_sides(self):
        games = sorted(p for p in self.run_id.glob("game_*") if p.is_dir())
        self.assertEqual(len(games), GAMES, "games_per_pair must equal games actually played")
        seats = [json.loads((g / "result.json").read_text())["players"][0] for g in games]
        self.assertIn("eco", seats)
        self.assertIn("aggro", seats)   # both bots played side 0 at least once

    def test_pair_summary_is_consistent(self):
        pair = self.run_meta["summary"]["eco"]["aggro"]
        self.assertEqual(pair["games"], GAMES)
        self.assertEqual(pair["a_wins"] + pair["b_wins"], GAMES)

    def test_elo_is_zero_sum_and_covers_both_bots(self):
        self.assertEqual(set(self.ratings), {"eco", "aggro"})
        self.assertAlmostEqual(sum(self.ratings.values()), 2 * 1500.0, places=3)

    def test_ratings_and_matches_persisted(self):
        # writes landed in the redirected temp state dir
        self.assertIn("matches.jsonl", self.state_files)
        self.assertIn("ratings.json", self.state_files)
        self.assertIn("ratings_after", self.run_meta)
        # and the real repository state dir was not polluted by this test
        self.assertTrue(self.repo_state_unchanged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
