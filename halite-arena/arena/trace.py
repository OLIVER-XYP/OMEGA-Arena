"""Convert engine replay JSON (--no-compression) into a per-turn JSONL that
agents can read: both players' halite/deposited/ship-count/commands per turn.
Commands are lightly classified (move direction / spawn / construct / attack /
defend / heal); full semantics analysis is left to the agent.
"""
from __future__ import annotations

import json
from pathlib import Path


def _classify_commands(raw_moves: dict) -> dict:
    """raw_moves: {player_id: [cmd, ...]} where cmd is a DICT (engine replay
    serializes Command objects to dicts, e.g. {"type":"m","direction":"n","id":1}
    or {"type":"s"}/{"type":"g"}). Return per-side lightly-classified list."""
    type_map = {"m": "move", "s": "spawn", "g": "generate", "c": "construct",
                "d": "defend", "h": "heal", "x": "attack_structure",
                "a": "attack_ship"}
    out = {}
    for pid, cmds in (raw_moves or {}).items():
        parsed = []
        for c in cmds or []:
            if isinstance(c, dict):
                typ = type_map.get(str(c.get("type", "")), str(c.get("type", "?")))
                if typ == "move":
                    parsed.append({"type": "move", "ship": str(c.get("id")),
                                   "dir": str(c.get("direction", "o"))})
                elif typ == "spawn" or typ == "generate":
                    parsed.append({"type": typ})
                elif "id" in c:
                    parsed.append({"type": typ, "ship": str(c.get("id"))})
                else:
                    parsed.append({"type": typ})
            elif isinstance(c, str):   # 兜底：字符串命令
                parts = c.split()
                if parts:
                    parsed.append({"type": parts[0].lower(), "raw": " ".join(parts[1:])})
        out[str(pid)] = parsed
    return out


def _summarize_entities(entities: dict, side_ids: set[str]) -> dict:
    """entities: {player_id: {ship_id: {x,y,energy}}}. Count per side."""
    counts = {}
    for pid, ships in (entities or {}).items():
        if str(pid) in side_ids:
            counts.setdefault(str(pid), {"ships": 0, "halite_cargo": 0})
            for sid, sh in (ships or {}).items():
                counts[str(pid)]["ships"] += 1
                counts[str(pid)]["halite_cargo"] += sh.get("energy", 0)
    return counts


def replay_json_to_trace(replay: dict) -> list[dict]:
    players = replay.get("players") or []
    side_ids = {str(p.get("player_id")): p.get("name", str(p.get("player_id")))
                for p in players}
    factory = {str(p.get("player_id")): p.get("factory_location")
               for p in players if p.get("factory_location")}
    frames = replay.get("full_frames") or []
    prev_deposited: dict[str, int] = {}
    trace_rows = []
    for t, f in enumerate(frames):
        if t == 0:   # 开局帧（无实际命令）
            continue
        ent = f.get("entities") or {}
        ships = _summarize_entities(ent, set(side_ids))
        energy = f.get("energy") or {}
        deposited = f.get("deposited") or {}
        row = {
            "turn": t,
            "players": {},
            "events_n": len(f.get("events") or []),
            "cells_n": len(f.get("cells") or []),
        }
        row_delta: dict[str, int] = {}
        row_score: dict[str, int] = {}
        for sid, name in side_ids.items():
            dep = int(deposited.get(sid, 0) or 0)
            row_delta[name] = dep - prev_deposited.get(sid, dep)
            prev_deposited[sid] = dep
            # 结构位置推算分 = 存款 + 所有已存（当前引擎得分 = factory_halite 总额）
            row_score[name] = int(energy.get(sid, 0) or 0) + dep
            row["players"][name] = {
                "halite": energy.get(sid),
                "deposited": dep,
                "ships": ships.get(sid, {}).get("ships", 0),
                "cargo": ships.get(sid, {}).get("halite_cargo", 0),
                # v2: 每船位置/能量（LLM 分析各船动作价值用）
                "ship_list": _ship_list(ent, sid),
                "factory": factory.get(sid),
            }
        cmds = _classify_commands(f.get("moves"))
        for sid, name in side_ids.items():
            if name in row["players"]:
                row["players"][name]["commands"] = cmds.get(sid, [])
        # v2: 全局演化视角（t-1 -> t 的存款增量 + 得分估算）
        row["delta_deposited"] = row_delta
        row["turn_score_est"] = row_score
        trace_rows.append(row)
    return trace_rows


def _ship_list(entities: dict, side: str) -> list[dict]:
    """All ships of one side with id / x / y / energy (for per-ship reasoning)."""
    out = []
    for sid, sh in (entities or {}).items():
        if str(sid) != str(side):
            continue
        for ship_id, s in (sh or {}).items():
            out.append({"id": str(ship_id), "x": s.get("x"), "y": s.get("y"),
                        "energy": s.get("energy")})
    # deterministic order by id
    out.sort(key=lambda s: int(s["id"]))
    return out


def write_trace(replay_path: Path, out_path: Path | None = None) -> Path:
    """Load replay .hlt (plain JSON) and write a per-turn trace.jsonl next to it."""
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    rows = replay_json_to_trace(replay)
    out = out_path or replay_path.with_name("trace.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out
