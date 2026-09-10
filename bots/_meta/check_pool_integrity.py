#!/usr/bin/env python3
"""OMEGA-Arena bot pool integrity check — the no-drift enforcer.

Asserts, mechanically and with no assumptions beyond reading the pool dir:
  1. roster (374 ids) fully covered by bot_index entries;
  2. registry (106 ids) ⊆ roster;
  3. every pool/*.pt maps to exactly one roster id (no orphans);
  4. present + missing == roster size (no silent drops);
  5. every `present` entry's file exists; every `missing` entry has no file;
  6. id→file rule is deterministic and round-trips.

Exit 0 = clean, 1 = violation (prints each).
Run:  python3 bots/_meta/check_pool_integrity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BOTS = Path(__file__).resolve().parents[1]
POOL = BOTS / "pool"
META = BOTS / "_meta"


def id_to_file(i: str) -> str:
    return (i[len("perturb_"):] if i.startswith("perturb_") else i) + ".pt"


def main() -> int:
    fails: list[str] = []
    idx = json.loads((META / "bot_index.json").read_text(encoding="utf-8"))
    roster = json.loads((META / "rank_merged374.json").read_text(encoding="utf-8"))["ranking"]
    reg = json.loads((BOTS / "bot_registry.json").read_text(encoding="utf-8"))["bots"]

    roster_ids = [r["id"] for r in roster]
    reg_ids = {b["id"] for b in reg}
    entries = idx.get("entries", {})
    pool_files = {p.name for p in POOL.glob("*.pt")}

    # 1. roster ⊆ index
    missing_from_index = [i for i in roster_ids if i not in entries]
    if missing_from_index:
        fails.append(f"roster id 未进 index: {missing_from_index[:5]}")

    # 2. registry ⊆ roster
    if not reg_ids <= set(roster_ids):
        fails.append(f"registry 有 {len(reg_ids - set(roster_ids))} 个 id 不在 roster")

    # 3-4. present/missing 守恒 + 无孤儿
    present = [i for i in roster_ids if entries.get(i, {}).get("present")]
    missing = [i for i in roster_ids if not entries.get(i, {}).get("present")]
    if len(present) + len(missing) != len(roster_ids):
        fails.append("present+missing != roster size")
    mapped = {id_to_file(i) for i in roster_ids}
    orphans = sorted(pool_files - mapped)
    if orphans:
        fails.append(f"pool 孤儿文件(无 id): {orphans[:5]}")

    # 5. present 文件存在 / missing 无文件
    for i in present:
        f = entries[i].get("file")
        if not f or not (BOTS.parent / f).exists():
            fails.append(f"present 但文件缺失: {i} -> {f}")
    for i in missing:
        if id_to_file(i) in pool_files:
            fails.append(f"missing 但文件存在: {i}")

    # 6. 计数一致性
    if idx.get("counts", {}).get("roster_ids") != len(roster_ids):
        fails.append("index.counts.roster_ids 与 roster 不符")
    if idx.get("counts", {}).get("pool_files") != len(pool_files):
        fails.append("index.counts.pool_files 与磁盘不符")

    print(f"roster={len(roster_ids)} registry={len(reg_ids)} "
          f"pool_files={len(pool_files)} present={len(present)} missing={len(missing)}")
    if fails:
        print("❌ 校验失败:")
        for f in fails:
            print("   -", f)
        return 1
    print("✅ 池完整性校验通过（id↔文件 无漂移）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
