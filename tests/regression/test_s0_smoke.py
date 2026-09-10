"""S0 smoke tests — the fastest gate: the CLI loads and the pool is addressable.

Seconds to run, no engine / weights / torch required (it self-skips the weight
check when the pool has not been fetched). Run from the repo root:

    python3 -m unittest discover -s tests/regression -p 'test_s0_*.py' -v
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARENA = REPO / "halite-arena"
sys.path.insert(0, str(ARENA))


def _cli(*args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ARENA)
    return subprocess.run([sys.executable, "-m", "arena.cli", *args],
                          cwd=str(ARENA), env=env, capture_output=True, text=True, timeout=timeout)


class TestArenaImports(unittest.TestCase):
    def test_all_modules_import_without_torch_or_rl_ai(self):
        # arena must be usable on a fresh clone: no rl_ai, no torch.
        # Block rl_ai to prove the vendored fallbacks are what makes it work.
        import importlib.abc

        class BlockRL(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "rl_ai" or name.startswith("rl_ai."):
                    raise ImportError("rl_ai blocked (simulating a fresh clone)")
                return None

        sys.meta_path.insert(0, BlockRL())
        try:
            for mod in ["arena.config", "arena.paths", "arena.pool", "arena.bots",
                        "arena.pairing", "arena.elo", "arena._elo", "arena._rl_compat",
                        "arena.engine_runner", "arena.runner", "arena.round",
                        "arena.verify", "arena.cli", "arena.web"]:
                with self.subTest(module=mod):
                    __import__(mod)
        finally:
            sys.meta_path.pop(0)


class TestCliSmoke(unittest.TestCase):
    def test_help(self):
        r = _cli("--help")
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_targets_lists_script_opponents(self):
        r = _cli("targets")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("script:eco", r.stdout)


class TestWeightPool(unittest.TestCase):
    def test_manifest_matches_local_pool_when_fetched(self):
        manifest = REPO / "bots" / "_meta" / "MANIFEST.sha256"
        pool = REPO / "bots" / "pool"
        if not manifest.exists():
            self.skipTest("no MANIFEST.sha256")
        names = [line.split("  ", 1)[1].strip() for line in manifest.read_text().splitlines() if line.strip()]
        if not any((pool / n).exists() for n in names):
            self.skipTest("weights not fetched (run scripts/fetch_weights.py)")
        missing = [n for n in names if not (pool / n).exists()]
        self.assertEqual(missing, [], f"{len(missing)} weights missing from bots/pool")


if __name__ == "__main__":
    unittest.main(verbosity=2)
