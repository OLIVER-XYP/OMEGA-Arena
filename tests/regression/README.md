# Regression tests

Pure-stdlib `unittest` suites for the arena platform. No torch, no bot weights
and no C++ toolchain are needed for S1; S2 needs the built engine + a compiler.

```bash
cd /mnt/OMEGA-Arena          # repo root

# everything (S0 fast + S1 unit + S2/S3 integration) — ~30 s
python3 -m unittest discover -s tests/regression -v

# by tier
python3 -m unittest discover -s tests/regression -p 'test_s0_*.py' -v   # ~1 s
python3 -m unittest discover -s tests/regression -p 'test_s1_*.py' -v   # ~0.6 s
python3 -m unittest discover -s tests/regression -p 'test_s2_*.py' -v   # ~30 s
python3 -m unittest discover -s tests/regression -p 'test_s3_*.py' -v   # ~0.3 s
```

## Suites

| Suite | What it locks down |
|---|---|
| `test_s0_smoke.py` | CLI loads (`--help`, `targets`); every `arena.*` module imports with `rl_ai` blocked (proves the vendored fallbacks); weight manifest matches the local pool when fetched |
| `test_s1_unit.py` | ELO math + Ladder zero-sum/idempotence; anchor freezing; tolerant JSON parse / trace counting; **deterministic pairing** (`seed_for` stable across processes); pool `id→file` rule; bot commands; `--device` override; run-id uniqueness; state-dir auto-create; run-key idempotence |
| `test_s2_integration.py` | the end-to-end 1v1 path: C++ overlay compile, `practice <agent> <script>` on CPU, archive layout, `result.json` consistency, **same-seed games must swap seats**, `arena.cli verify` 0 failures, and failed compiles leave no stale bot |
| `test_s3_match_elo.py` | `match` round-robin + ELO: `games_per_pair` matches games played, both seats occur, ratings are zero-sum and persisted — state redirected to a temp dir so the real ladder is untouched |

S2/S3 self-skip when the engine or the script bot is missing, and can be forced
off with `ARENA_SKIP_INTEGRATION=1` (e.g. on a machine without the C++ toolchain).
