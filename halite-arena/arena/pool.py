"""Opponent pool: script targets + RL pool (374 roster) with id->checkpoint.

Single source of truth = OMEGA/bots/_meta/bot_index.json (id → pool file), backed
physically by OMEGA/bots/pool/*.pt. `bots/bot_registry.json` is a compat view.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import config as C
from .bots import BotSpec, rl_spec, script_spec


def script_specs() -> dict[str, BotSpec]:
    return {k: script_spec(k) for k in C.SCRIPT_PARAMS}


def load_merged_ranking() -> list[dict]:
    """Roster (374 ids, ranked). Prefers OMEGA _meta, falls back to legacy."""
    if not C.POOL_ROSTER.exists():
        return []
    return json.loads(C.POOL_ROSTER.read_text(encoding="utf-8")).get("ranking", [])


def load_bot_index() -> dict:
    if not C.BOT_INDEX.exists():
        return {"entries": {}, "missing_ids": []}
    return json.loads(C.BOT_INDEX.read_text(encoding="utf-8"))


def _id_to_file(pool_id: str) -> str:
    """Canonical id→filename rule (mirrors bot_index generation)."""
    return (pool_id[len("perturb_"):] if pool_id.startswith("perturb_") else pool_id) + ".pt"


def top_pool_ids(k: int, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    index = load_bot_index().get("entries", {})
    out = []
    for r in load_merged_ranking():
        i = r["id"]
        if i in exclude:
            continue
        # 只取实体存在的（missing 的不作对手）
        if index and not index.get(i, {}).get("present", True):
            continue
        out.append(i)
        if len(out) >= k:
            break
    return out


def resolve_checkpoint(pool_id: str) -> Path:
    """Map pool_id (roster / registry) to a checkpoint path in OMEGA/bots/pool."""
    # 0) 友好别名：cycle002 = 默认最强 RL（champion）
    if pool_id in ("cycle002", C.FROZEN_TOP.stem) and C.FROZEN_TOP.exists():
        return C.FROZEN_TOP
    index = load_bot_index()
    # 1) 权威索引
    ent = index.get("entries", {}).get(pool_id)
    if ent and ent.get("present") and ent.get("file"):
        return C.OMEGA_ROOT / ent["file"]            # file 相对 OMEGA 根（bots/pool/..）
    # 2) 兼容 registry（path 已指向 OMEGA pool）
    if C.BOT_REGISTRY.exists():
        try:
            reg = json.loads(C.BOT_REGISTRY.read_text(encoding="utf-8"))
            for b in reg.get("bots", []):
                if b.get("id") == pool_id and b.get("path") and Path(b["path"]).exists():
                    return Path(b["path"])
        except Exception:
            pass
    # 3) 规范 id→文件 规则直取 pool
    cand = C.BOT_POOL / _id_to_file(pool_id)
    if cand.exists():
        return cand
    # 4) cycle002 别名（默认最强 RL）
    if pool_id == C.FROZEN_TOP.stem and C.FROZEN_TOP.exists():
        return C.FROZEN_TOP
    raise FileNotFoundError(
        f"无法解析 RL 池对手 {pool_id!r}（索引/registry/pool 均未命中；"
        f"该 id 可能属于无实体的 missing 集）")


def resolve_rl(pool_id: str, *, device: str = "auto") -> BotSpec:
    """Resolve a roster/registry id to an RL BotSpec (checkpoint in OMEGA pool)."""
    return rl_spec(pool_id, checkpoint=resolve_checkpoint(pool_id), device=device)


def rl_pool_specs(k: int) -> list[BotSpec]:
    """Top-k roster ids that have a real checkpoint → BotSpecs."""
    specs = []
    for pid in top_pool_ids(k):
        try:
            specs.append(resolve_rl(pid))
        except FileNotFoundError:
            continue
    return specs
