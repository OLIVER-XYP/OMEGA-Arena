# Regression tests

Pure-stdlib `unittest` suites for the arena platform. No torch, no bot weights
and no C++ toolchain are needed for S1; S2 needs the built engine + a compiler.

```bash
cd /mnt/OMEGA-Arena          # repo root

# S1 — fast unit tests (~0.5 s)
python3 -m unittest discover -s tests/regression -p 'test_s1_*.py' -v

# S2 — 1v1 integration: compile → practice vs eco → verify (~30 s)
python3 -m unittest discover -s tests/regression -p 'test_s2_*.py' -v

# run both
python3 -m unittest discover -s tests/regression -v
```

## Suites

| Suite | What it locks down |
|---|---|
| `test_s1_unit.py` | ELO math + Ladder convergence; anchor freezing; tolerant JSON parse / trace counting; **deterministic pairing** (`seed_for` must be stable across processes); pool `id→file` rule; bot command construction |
| `test_s2_integration.py` | the end-to-end 1v1 path: C++ overlay compile, `practice <agent> <script>` on CPU, archive layout, `result.json` consistency, **same-seed games must swap seats**, and `arena.cli verify` reporting 0 failures |

S2 self-skips when the engine or the script bot is missing, and can be forced
off with `ARENA_SKIP_INTEGRATION=1` (e.g. on a machine without the C++ toolchain).
