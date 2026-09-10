"""Compile an agent's C++ submit/ into a runnable bot.

Compile contract: overlay submit/ onto the full materials/cpp kit, then g++ all
.cpp. Agent may change only MyBot.cpp, or rewrite the whole tree.
Isolated: subprocess starts a new session + timeout; failure recorded for feedback.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

from . import paths


@dataclass
class CompileResult:
    ok: bool
    error_kind: str | None = None   # no_main|compile_error|timeout|internal
    log: str = ""
    exe: str = ""
    elapsed_ms: int = 0

    def to_dict(self):
        return asdict(self)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("build*", "__pycache__", ".git",
                                                  "*.o", "*.log", "*.exe", "*.so"))


def _find_main_cpp(src: Path) -> Path | None:
    for cpp in src.rglob("*.cpp"):
        try:
            text = cpp.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\bmain\s*\([^)]*\)", text):
                return cpp
        except Exception:
            continue
    return None


def compile_agent(name: str, *, gxx: str = "g++",
                  timeout_s: float = 90.0) -> CompileResult:
    import time
    a = paths.agent_dir(name)
    materials = a / "materials" / "cpp"
    if not materials.exists():
        # 只读材料缺失（如被精简/首次）→ 从套件补全后重试
        paths.scaffold_agent(name)
    if not materials.exists():
        return CompileResult(ok=False, error_kind="internal",
                             log="materials/cpp 不存在且无法生成（检查 starter_kits/C++）")

    build_dir = a / "build"
    src = build_dir / "src"
    exe = build_dir / "bot"
    build_dir.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.rmtree(src)
    try:
        _copy_tree(materials, src)
        _copy_tree(a / "submit", src)   # overlay 交付物
    except Exception as e:
        return CompileResult(ok=False, error_kind="internal", log=f"overlay 失败: {e}")

    if _find_main_cpp(src) is None:
        return CompileResult(ok=False, error_kind="no_main",
                             log="submit 里找不到含 main() 的 .cpp")

    cpps = sorted(str(p) for p in src.rglob("*.cpp"))
    cmd = [gxx, "-std=c++17", "-g", "-O2", "-Wall", "-Wno-unused-function", "-pedantic",
           "-I", str(src), "-I", str(src / "hlt"), *cpps, "-o", str(exe)]
    start = time.time()
    try:
        from . import sandbox
        proc = sandbox.run_isolated(cmd, cwd=build_dir, timeout_s=timeout_s)
        elapsed = int((time.time() - start) * 1000)
    except subprocess.TimeoutExpired:
        return CompileResult(ok=False, error_kind="timeout",
                             log=f"编译超时 (> {timeout_s}s)", elapsed_ms=int((time.time()-start)*1000))
    log = (proc.stdout or "") + (proc.stderr or "")
    log = log[-100_000:]
    if proc.returncode != 0:
        return CompileResult(ok=False, error_kind="compile_error", log=log, elapsed_ms=elapsed)
    if not exe.exists() or not (exe.stat().st_mode & 0o111):
        return CompileResult(ok=False, error_kind="internal", log="编译成功但无可执行产物", elapsed_ms=elapsed)
    # 落盘 compile.log + compile_result.json
    (build_dir / "compile.log").write_text(log, encoding="utf-8")
    res = CompileResult(ok=True, log=log[:2000], exe=str(exe), elapsed_ms=elapsed)
    (build_dir / "compile_result.json").write_text(
        __import__("json").dumps(res.to_dict(), indent=1), encoding="utf-8")
    return res
