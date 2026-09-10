"""Arena global paths and defaults. All paths absolute; probe existence lazily.

OMEGA 自洽设计：
- 框架/引擎/bot 池物理位于 /mnt/OMEGA-Arena（本文件自动探测：__file__ 的物理父级）。
- 训练侧 /mnt/Halite-III 保留 rl_ai（训练脚本/数据/模型），并通过**软链回指**
  game_engine / starter_kits / libhaliteviz / halite-arena → OMEGA，故训练脚本的
  相对 ROOT 路径照常解析，两侧共用同一份物理文件。
- rl_ai 的推理入口（inference_bot.py）与 torch 依赖仍在训练侧（TRAIN_ROOT）。

路径选择策略：优先 OMEGA 本地，缺失则回落到 _LEGACY（/mnt/Halite-III）——兼容
两种入口（物理目录 / 软链目录）。
"""
from __future__ import annotations

import sys
from pathlib import Path

# __file__ = <arena根>/halite-arena/arena/config.py；resolve() 穿透软链到物理位置
_ARENA_PKG = Path(__file__).resolve().parents[1]     # .../halite-arena
OMEGA_ROOT = _ARENA_PKG.parent                        # .../OMEGA-Arena
ARENA_ROOT = _ARENA_PKG
_LEGACY = Path("/mnt/Halite-III")                     # 训练侧（rl_ai 所在）

# 兼容别名：旧代码（paths.py/rules.py 等）用 C.ROOT 指引擎/套件根；
# 语义等同 OMEGA_ROOT，一处失效即可点改。
ROOT = OMEGA_ROOT
TRAIN_ROOT = _LEGACY

for _p in (str(OMEGA_ROOT), str(_LEGACY)):
    if Path(_p).exists() and _p not in sys.path:
        sys.path.insert(0, _p)


def _pick(*cands: Path) -> Path:
    """Return the first existing candidate (else the first) — path fallback."""
    for c in cands:
        if c.exists():
            return c
    return cands[0]


# --- engine & rules（物理在 OMEGA；若经软链访问亦然）---
ENGINE = _pick(OMEGA_ROOT / "game_engine/build_pyenv/halite",
               _LEGACY / "game_engine/build_pyenv/halite")
ENGINE_GPU = _pick(OMEGA_ROOT / "game_engine/build_gpu/halite",
                   _LEGACY / "game_engine/build_gpu/halite")
ENGINE_CFG = _pick(OMEGA_ROOT / "starter_kits/C++/competitive_engine_v2.json",
                   _LEGACY / "starter_kits/C++/competitive_engine_v2.json")
ENGINE_SRC = _pick(OMEGA_ROOT / "game_engine", _LEGACY / "game_engine")

# --- C++ starter kit ---
KIT_CPP = _pick(OMEGA_ROOT / "starter_kits/C++", _LEGACY / "starter_kits/C++")
SCRIPT_PARAMS = {
    "eco": KIT_CPP / "bot_params_eco.txt",
    "aggro": KIT_CPP / "bot_params_aggro.txt",
    "control": KIT_CPP / "bot_params_control.txt",
    "adaptive": KIT_CPP / "bot_params_adaptive.txt",
}
SCRIPT_EXE = KIT_CPP / "build_linux/MyBot"

# --- RL inference bot（训练侧 rl_ai；torch 依赖在那边）---
PYTHON = Path("/usr/bin/python3")
INFERENCE_BOT = _pick(_LEGACY / "rl_ai/inference_bot.py", OMEGA_ROOT / "rl_ai/inference_bot.py")
FEATURE_SCHEMA = "halite3_v3_observable_features_v3"

# --- RL bot 池（物理在 OMEGA/bots；权威索引 = _meta/bot_index.json）---
BOTS_ROOT = OMEGA_ROOT / "bots"
BOT_POOL = BOTS_ROOT / "pool"
BOT_INDEX = BOTS_ROOT / "_meta/bot_index.json"
BOT_REGISTRY = BOTS_ROOT / "bot_registry.json"
# 默认最强 RL 对手（champion）：OMEGA 池中的 hist... 命名
FROZEN_TOP = _pick(BOT_POOL / "hist_frozen_v4_r3_gen2_cycle002.pt",
                   _LEGACY / "rl_ai/runs/frozen_v4_r3_gen2/cycle002.pt")
# 训练侧清单（roster）
POOL_ROSTER = _pick(BOTS_ROOT / "_meta/rank_merged374.json",
                    _LEGACY / "rl_ai/runs/ranking/rank_merged374.json")

# --- elo 复用（训练侧优先；独立 clone 回落随包 vendored 纯标准库版）---
try:
    from rl_ai.ranking import elo as _rl_elo  # noqa: E402,F401
except Exception:  # pragma: no cover
    from . import _elo as _rl_elo  # noqa: E402,F401


def resolve_path(p: Path, what: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{what} 不存在: {p}")
    return p
