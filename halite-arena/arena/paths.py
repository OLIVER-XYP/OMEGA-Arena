"""Arena path helpers: agent workspace, run/game dir allocation."""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from . import config as C


def agents_root() -> Path:
    return C.ARENA_ROOT / "agents"


def runs_root() -> Path:
    return C.ARENA_ROOT / "runs"


def state_dir() -> Path:
    return C.ARENA_ROOT / "state"


def agent_dir(name: str) -> Path:
    return agents_root() / name


def _copy_tree(src: Path, dst: Path, ignore_names: set[str]) -> None:
    """Copy a tree, skipping build artifacts & logs & dotfiles dirs."""
    def _ig(d, names):
        return {n for n in names if n in ignore_names or n.endswith((".o", ".exe", ".log", ".so", ".pyc"))}
    shutil.copytree(src, dst, ignore=_ig, dirs_exist_ok=True)


# engine source files we copy into materials/engine_src (readonly docs for agents).
_ENGINE_SRC_EXTS = {".cpp", ".hpp", ".h", ".cc", ".json", ".md", ".txt", ".sh"}
_ENGINE_SRC_SKIP_DIRS = {
    "build", "build_pyenv", "build_pyenv_310", "build_gpu", "external",
    "vendor", "deps", "node_modules", "CMakeFiles", ".git", "__pycache__",
}


def _copy_engine_src(a: Path) -> Path:
    """Copy game_engine/** (skipping builds/vendor/binaries) into materials/engine_src."""
    engine = C.ROOT / "game_engine"
    dst = a / "materials" / "engine_src"
    for d in engine.rglob("*"):
        rel = d.relative_to(engine)
        if any(part in _ENGINE_SRC_SKIP_DIRS for part in rel.parts):
            continue
        if d.is_dir():
            (dst / rel).mkdir(parents=True, exist_ok=True)
            continue
        if d.suffix in _ENGINE_SRC_EXTS or d.name == "CMakeLists.txt":
            try:
                p = dst / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(d, p)
            except (OSError, shutil.Error):
                pass
    return dst


def prepare_rules_for(a: Path) -> Path:
    """Generate materials/rules.md for this agent (idempotent)."""
    from . import rules as rules_mod
    out = (a / "materials" / "rules.md").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rules_mod.build_rules_md(out=out)
    return out


def refresh_materials(name: str) -> Path:
    """Re-copy engine_src + regenerate rules.md for an existing agent."""
    a = agent_dir(name)
    _copy_engine_src(a)
    prepare_rules_for(a)
    return a


def scaffold_agent(name: str, *, kit_root: Path | None = None) -> Path:
    """Create agents/<name> workspace: materials/(C++ kit 副本 + engine_src +
    rules.md) + params/ + submit/ + README."""
    kit_root = kit_root or C.KIT_CPP
    a = agent_dir(name)
    materials_cpp = a / "materials" / "cpp"
    if not (a / "agent.json").exists():
        a.mkdir(parents=True, exist_ok=True)
        # 只读材料：完整 C++ 套件源码副本
        materials_cpp.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(kit_root, materials_cpp,
                   {"build", "build_linux", "build_gpu", "__pycache__", ".git", "hlt_archive"})
        # submit/ 与 params/
        (a / "submit").mkdir(parents=True, exist_ok=True)
        (a / "params").mkdir(parents=True, exist_ok=True)
        shutil.copy(C.SCRIPT_PARAMS["eco"], a / "params" / "bot_params.txt")
        # agent.json
        (a / "agent.json").write_text(json.dumps({
            "name": name, "kind": "cpp", "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "params": str(a / "params" / "bot_params.txt"),
        }, indent=1), encoding="utf-8")
    # 已有 agent：补缺的只读材料（kit cpp/、engine_src、rules.md），不覆盖 agent submit 改动
    materials_cpp.mkdir(parents=True, exist_ok=True)
    if not (materials_cpp / "MyBot.cpp").exists():
        _copy_tree(kit_root, materials_cpp,
                   {"build", "build_linux", "build_gpu", "__pycache__", ".git", "hlt_archive"})
    if not (a / "materials" / "engine_src" / "config").exists():
        _copy_engine_src(a)
    prepare_rules_for(a)   # rules.md 幂等重生成，始终与当前引擎一致
    _ensure_readmes(a)
    return a


def _ensure_readmes(a: Path) -> None:
    """材料与 submit 的说明文档（agent 接入契约）。只写一次，不覆盖 agent 改动。"""
    materials_readme = a / "materials" / "README.md"
    if not materials_readme.exists():
        materials_readme.write_text(
            "## Agent 只读材料\n\n"
            "- `cpp/`：完整 C++ starter kit 源码副本（MyBot.cpp + hlt/ + bot_params*.txt）。\n"
            "- `engine_src/`：game_engine 引擎源码只读副本（规则/阶段实现，可研究机制细节）。\n"
            "- `rules.md`：当前竞技场规则（由引擎实际常量的生成，含数值表/阶段顺序/命令协议）。\n"
            "- 你每轮交付到 `../submit/`，平台会自动 overlay 到 cpp 全套件后编译。\n"
            "- 上局反馈在 `../last_game/`（run.json 汇总 + games/*/trace.jsonl 逐回合 + replay）。\n",
            encoding="utf-8")
    submit_readme = a / "submit" / "README.md"
    if not submit_readme.exists():
        submit_readme.write_text(
            "## 交付目录\n\n"
            "把你要交付的 C++ 源码放这里。可以：\n"
            "- 只放改动的文件（如 MyBot.cpp），平台 overlay 到 materials/cpp 全套件后编译；\n"
            "- 或放整棵树（自定义 main、增删 hlt 文件都行）。\n"
            "提交后运行 `arena practice <name> <target...>` 或 `arena compile <name>`。\n"
            "编译结果在 ../build/，最近反馈在 ../last_game/。\n",
            encoding="utf-8")


def ensure_agent(name: str, *, scaffold: bool = True) -> Path:
    a = agent_dir(name)
    if not a.exists() and scaffold:
        return scaffold_agent(name)
    return a


def agents_list() -> list[dict]:
    out = []
    for a in agents_root().glob("*/agent.json"):
        try:
            out.append(json.loads(a.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


def new_run_id(tag: str = "") -> Path:
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return runs_root() / f"{ts}-{tag or 'run'}"


def new_game_dir(run_id: Path, index: int) -> Path:
    g = run_id / f"game_{index:06d}"
    g.mkdir(parents=True, exist_ok=True)
    (g / "engine_cwd").mkdir(exist_ok=True)
    (g / "replay_out").mkdir(exist_ok=True)
    return g


def last_game_dir(name: str) -> Path:
    return agent_dir(name) / "last_game"
