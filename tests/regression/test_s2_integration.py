"""S2 integration test: compile a bot, play a 1v1 practice match on CPU, verify the archive.

This is the end-to-end path the platform exists for:

    arena.cli compile testbot        # C++ overlay compile
    arena.cli practice testbot eco   # 1v1 vs a script opponent, both sides
    arena.cli verify <run_dir>       # archive integrity / consistency / authenticity

It needs a built engine (game_engine/build_pyenv/halite) and a C++ toolchain.
Set ARENA_SKIP_INTEGRATION=1 to skip (the test also self-skips when the engine
or a script bot is missing), so it is safe in environments without them.

    python3 -m unittest discover -s tests/regression -p 'test_s2_*.py' -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARENA = REPO / "halite-arena"
sys.path.insert(0, str(ARENA))

from arena import config as C  # noqa: E402

ENGINE = C.ENGINE
SCRIPT_BOT = C.SCRIPT_EXE
SKIP = os.environ.get("ARENA_SKIP_INTEGRATION") == "1"

AGENT = "testbot"
TARGET = "eco"


def _run_cli(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ARENA)
    return subprocess.run(
        [sys.executable, "-m", "arena.cli", *args],
        cwd=str(ARENA), env=env, capture_output=True, text=True, timeout=timeout,
    )


@unittest.skipIf(SKIP, "ARENA_SKIP_INTEGRATION=1")
@unittest.skipUnless(ENGINE.exists(), f"engine not built: {ENGINE}")
@unittest.skipUnless(SCRIPT_BOT.exists(), f"script bot not built: {SCRIPT_BOT}")
class TestPracticeOneVOne(unittest.TestCase):
    """The S2 gate: a full 1v1 practice round-trips to a verifiable archive."""

    @classmethod
    def setUpClass(cls):
        cls.compile_res = _run_cli("compile", AGENT)
        # find the run dir created by this practice (before/after diff, robust to
        # the timestamped naming and to a pre-existing runs/ dir)
        before = set((ARENA / "runs").glob("*")) if (ARENA / "runs").exists() else set()
        cls.practice_res = _run_cli("practice", AGENT, TARGET, "--seeds", "1")
        after = set((ARENA / "runs").glob("*")) if (ARENA / "runs").exists() else set()
        new = sorted(after - before)
        cls.run_dir = next((p for p in reversed(new) if "practice" in p.name), None)
        cls.verify_res = _run_cli("verify", cls.run_dir.name) if cls.run_dir else None

    def test_compile_succeeds(self):
        self.assertEqual(self.compile_res.returncode, 0, self.compile_res.stdout + self.compile_res.stderr)
        self.assertIn("ok=True", self.compile_res.stdout)

    def test_practice_succeeds_with_no_failed_games(self):
        self.assertEqual(self.practice_res.returncode, 0,
                         self.practice_res.stdout + self.practice_res.stderr)
        self.assertIn("failed=0", self.practice_res.stdout)

    def test_run_dir_and_archives_exist(self):
        self.assertIsNotNone(self.run_dir, "no practice run dir was created")
        self.assertTrue((self.run_dir / "run.json").exists())
        games = sorted(p for p in self.run_dir.glob("game_*") if p.is_dir())
        self.assertGreaterEqual(len(games), 2, "seeds=1 with both sides should yield >=2 games")
        for g in games:
            self.assertTrue((g / "result.json").exists(), f"{g} missing result.json")
            self.assertTrue((g / "trace.jsonl").exists(), f"{g} missing trace.jsonl")
            self.assertTrue(list(g.glob("replay-*.hlt")), f"{g} missing replay")

    def test_result_json_is_consistent(self):
        self.assertIsNotNone(self.run_dir)
        for g in sorted(self.run_dir.glob("game_*")):
            res = json.loads((g / "result.json").read_text())
            self.assertIn("ok", res)
            self.assertTrue(res["ok"], f"{g} result not ok: {res}")
            # a finished game must name a winner or record a fatal/errors
            has_winner = bool(res.get("winner"))
            has_problem = bool(res.get("fatal")) or bool(res.get("error_logs"))
            self.assertTrue(has_winner or has_problem,
                            f"{g}: neither winner nor fatal/error_logs: {res}")

    def test_same_seed_played_from_both_sides(self):
        """Regression: side_a must actually swap seats (was a no-op).

        seeds=1 with both sides yields two games on the SAME seed. They must be
        played with opposite seat order, otherwise the 'same seed, swap sides'
        anti-bias mechanism does nothing and the win tally is mis-attributed.
        """
        self.assertIsNotNone(self.run_dir)
        games = sorted(p for p in self.run_dir.glob("game_*") if p.is_dir())
        self.assertEqual(len(games), 2, "seeds=1 should yield exactly 2 games (both sides)")
        first_seat = []
        for g in games:
            res = json.loads((g / "result.json").read_text())
            first_seat.append(res["players"][0])
        self.assertNotEqual(first_seat[0], first_seat[1],
                            "both games seated the same bot first: side_a swap is a no-op")

    def test_verify_reports_clean(self):
        self.assertIsNotNone(self.verify_res)
        self.assertEqual(self.verify_res.returncode, 0,
                         f"verify failed:\n{self.verify_res.stdout}\n{self.verify_res.stderr}")
        self.assertIn("0 fail", self.verify_res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
