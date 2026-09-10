"""Sandbox wrappers for untrusted agent code.

Defense-in-depth used by the CLI MVP:

1. **Compile**: g++ never executes agent source, but a hostile `#include` could
   still make the preprocessor wander; we bound core dumps / output file size
   and run in its own session so any runaway is killable.
2. **Engine run**: engine spawns bot via `sh -c`; engine's own per-turn 2s
   timeout + arena's wall-clock session kill already bound runaway bots.

Deliberately NOT used: prlimit --nproc / --as. RLIMIT_NPROC is a per-uid total
over the whole host (not per-process) — on a shared many-core box it instantly
caps threads and segfaults the engine; RLIMIT_AS breaks RL bots whose CUDA/torch
context legitimately maps huge virtual space. Real isolation (uid / container /
cgroup) is a documented deployment follow-up, not a per-run prlimit.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path


def _prlimit_prefix(*, core: int = 0, fsize: int | None = None) -> list[str]:
    """Optional prlimit(1) prefix; empty list if unavailable.

    Only core + fsize are set: core dumps off (no disk fill), output files
    capped. nproc/as are intentionally omitted (see module docstring).
    """
    exe = shutil.which("prlimit")
    if exe is None:
        return []
    parts = [exe, "--core=%d" % core]
    if fsize is not None:
        parts += ["--fsize=%d" % fsize]
    return parts


def run_isolated(cmd: list[str], *, cwd: Path | None = None,
                 timeout_s: float | None = None,
                 capture: bool = True) -> subprocess.CompletedProcess:
    """Run cmd in a fresh session with prlimit bounds. Returns CompletedProcess
    (timeout => raises TimeoutExpired after killing the whole session)."""
    pre = _prlimit_prefix(core=0, fsize=int(0.5 * 1024 ** 3))
    full = pre + cmd
    proc = subprocess.Popen(full, cwd=str(cwd) if cwd else None,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None,
                            text=capture, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(full, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        _kill_session(proc.pid)
        out, err = proc.communicate(timeout=5)
        raise subprocess.TimeoutExpired(cmd, timeout_s, out, err)


def _kill_session(sid: int) -> None:
    """SIGKILL every process group whose session == sid (mirrors engine_runner)."""
    try:
        with os.scandir("/proc") as it:
            for e in it:
                if not e.name.isdigit():
                    continue
                try:
                    st = (Path("/proc") / e.name / "stat").read_text().split()
                    pgrp, session = int(st[4]), int(st[5])
                    if session == sid:
                        os.killpg(pgrp, signal.SIGKILL)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        os.killpg(sid, signal.SIGKILL)
    except Exception:
        pass
