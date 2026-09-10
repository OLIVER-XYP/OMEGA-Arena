"""Generate rules.md for agents from the live engine constants + phase order.

Sources (all read from disk, so the doc always matches the engine that runs):
1. Constant semantics: game_engine/config/Constants.hpp inline comments.
2. Active values: starter_kits/C++/competitive_engine_v2.json (the cfg the
   arena actually passes to the engine).
3. Turn order: game_engine/core/engine/TurnEngine.cpp ruleset.add_phase() lines.
4. Game shape: 32x32, MAX_TURNS / costs from the resolved constants.

Idempotent: `python -m arena.rules` regenerates halite-arena/rules.md in place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import config as C

# game-loop phases as registered in TurnEngine.cpp (fallback order)
PHASE_ORDER = [
    "Inspiration", "Validation", "Construction", "Defend", "Movement",
    "Combat", "Heal", "Dump", "Spawn", "Mining", "Plunder", "Regen",
    "Capture", "OverShipTax", "HaliteRebalance",
]

PHASE_LABEL = {
    "Inspiration": "算灵感加成（敌方船簇内采矿/移动倍率）",
    "Validation": "校验命令合法性与超时（非法命令/超时记入 error_logs）",
    "Construction": "执行建 dropoff（g 命令）",
    "Defend": "设置/清除 defend 状态（d 命令）——is_defending 船对攻击反弹 DEFEND_RETALIATION_DAMAGE",
    "Movement": "执行移动（n/s/e/w/o 命令 + 碰撞结算）",
    "Combat": "执行攻击（a/x 命令 + 击杀结算 + cargo 转移/击杀奖金）",
    "Heal": "执行治疗（h 命令）",
    "Dump": "货舱倾倒（到基地自动入库）",
    "Spawn": "生成 ship（s 命令）",
    "Mining": "矿藏在位（未被干扰/守卫时提取 halite）",
    "Plunder": "劫掠（敌方结构附近被动收入）",
    "Regen": "细胞再生（领地恢复规则）",
    "Capture": "捕获（敌方结构附近兵力优势时接管）",
    "OverShipTax": "船数超额税（每回合船数惩罚）",
    "HaliteRebalance": "halite 再平衡（落后方周期补助）",
}

# strategy hints for the rules that shape decisions most. Values come from the
# ACTIVE cfg at generation time; these notes explain *why* they matter.
HINTS = {
    "KILL_CREDIT_TO_ATTACKER": "被击杀船的剩余 cargo 直接进击杀者银行——攻击高负载矿工是可观收入。",
    "KILL_HALITE_BONUS_RATIO": "击杀奖金 = 被击杀船 cargo × 该系数 额外进击杀者银行。满载矿工是移动的钱袋。",
    "ATTACK_HP_SELF_DAMAGE": "攻击方每次命中自己掉 ATTACK_HP_SELF_DAMAGE HP。",
    "DEFEND_RETALIATION_DAMAGE": "⚠️ 结构机制：is_defending 船对攻击者反弹该数值伤害（100 = 满血攻击防守船 = 自杀）。不要主动攻击 defend 中的船。",
    "DEFEND_ALLOWS_MINING": "防守船在防守期间仍采矿，经济不受损——防守不是'冻结'。",
    "ATTACK_RANGE": "最大攻击距离（环面曼哈顿）。",
    "CAPTURE_ENABLED": "敌我数量差 ≥ SHIPS_ABOVE_FOR_CAPTURE 且距离 ≤ CAPTURE_RADIUS 时船可被捕获（接管为敌方）。别让船落单进敌方队形。",
    "MINING_INTERFERENCE_RATIO": "敌方船在采矿船 MINING_INTERFERENCE_RANGE 内时采矿效率下降——围到敌方矿区可压制其经济。",
    "PLUNDER_HALITE_PER_TURN": "停在敌方结构 PLUNDER_RANGE 内的船每回合被动收入。",
    "OVER_SHIP_TAX_PER_TURN": "每艘超出 OVER_SHIP_TAX_THRESHOLD 的船每回合付税；SHIP_COUNT_TARGET 是船数甜点参考。",
    "SPAWN_COST_GROWTH": "第 N 艘船实际造价 = SHIP_COST × (1 + growth × N)——越造越贵。",
    "DROPOFF_COST_GROWTH": "第 N 个 dropoff 造价 = DROPOFF_COST × (1 + growth × N)。",
    "HALITE_REBALANCE_ENABLED": "每周期从领先方转移 halite 给落后方（MIN_GAP_FRAC 抑制过冲），缩小滚雪球。",
    "EMERGENCY_SPAWN_ENABLED": "被淘汰玩家每 EMERGENCY_SPAWN_PERIOD 回合获免费救援船 + 银行补助。",
}


def _extract_phase_order() -> list[str]:
    """Pull the phase registration order straight from TurnEngine.cpp."""
    try:
        src = (C.ROOT / "game_engine/core/engine/TurnEngine.cpp").read_text(
            encoding="utf-8", errors="replace")
    except OSError:
        return list(PHASE_ORDER)
    order = [m.group(1) for m in re.finditer(
        r"add_phase\(std::make_unique<[^>]+::(\w+)Phase\(\)>\)", src)]
    return order or list(PHASE_ORDER)


def _constant_docs() -> dict[str, str]:
    """Scrape doc-comment for each constant from Constants.hpp."""
    text = (C.ROOT / "game_engine/config/Constants.hpp").read_text(
        encoding="utf-8", errors="replace")
    docs: dict[str, str] = {}
    # /** ... */ immediately preceding a declaration naming CONSTANT (or = default)
    pat = re.compile(
        r"/\*\*(?P<doc>.*?)\*/\s*"
        r"(?:static\s+)?(?:constexpr\s+)?(?:unsigned\s+long|energy_type|dimension_type|"
        r"int|double|bool|long|float|std::string|string)\s+"
        r"(?P<name>[A-Z][A-Z0-9_]*)\b",
        re.S)
    for m in pat.finditer(text):
        name = m.group("name")
        doc = " ".join(m.group("doc").split()).strip()
        if doc and name not in docs:
            docs[name] = doc
    return docs


def _md_cell(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ").strip()


def _resolved_constants() -> dict:
    """Ask the actual engine binary for its merged constant set
    (defaults overlaid by the -c cfg). This is what a real game runs with."""
    try:
        import subprocess
        proc = subprocess.run(
            [str(C.ENGINE), "--print-constants", "-c", str(C.ENGINE_CFG),
             "bot0", "false"],   # placeholder bot args; engine exits after printing
            capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            txt = proc.stdout.strip()
            i, j = txt.find("{"), txt.rfind("}")
            if 0 <= i < j:
                d = json.loads(txt[i:j + 1])
                if isinstance(d, dict) and d:
                    return d
    except Exception:
        pass
    # fallback: the raw cfg (sparse; only overridden keys)
    return json.loads(C.ENGINE_CFG.read_text(encoding="utf-8"))


def build_rules_md(out: Path | None = None) -> Path:
    """Write agent-facing rules.md. Returns its path. Idempotent."""
    cfg = _resolved_constants()
    docs = _constant_docs()
    phase_order = _extract_phase_order()

    ship_cost = int(cfg.get("NEW_ENTITY_ENERGY_COST", cfg.get("SHIP_COST", 1000)))
    max_turns = int(cfg.get("MAX_TURNS", 200))

    rows = []
    # II. active rule table
    md = ["# Halite III 竞技场规则", ""]
    md.append("> 本文件由 `arena rules` 从**平台实际使用的引擎**生成，非手写。")
    md.append("> - 数值来源：`competitive_engine_v2.json`（引擎 `-c` 参数）")
    md.append("> - 语义来源：`game_engine/config/Constants.hpp` 注释 + 引擎阶段源码")
    md.append("> - 重新生成：`/usr/bin/python3 -m arena.rules` 或 `arena prepare <agent>`")
    md.append("")
    md.append("## 1. 一句话规则")
    md.append("")
    md.append(f"- 32×32 **环面**地图。每个玩家一个船坞（shipyard），初始存款 {cfg.get('INITIAL_ENERGY', 5000)}。")
    md.append(f"- 船造价 `{ship_cost}`（随数量涨）、dropoff 造价 `{cfg.get('DROPOFF_COST', 4000)}`。")
    md.append(f"- 每回合每个 bot 输出命令（`n/e/s/w/o` 移动、`a` 攻击船、`x` 攻击建筑、`d` defend、`h` 治疗、`s` 造舰、`g` 建 dropoff）。")
    md.append(f"- 最终得分 = `factory_halite` 总额（存款 + 各 dropoff 存款）。回合上限 {max_turns}，平台单局跑 `--turn-limit 300`。")
    md.append("")
    md.append("## 2. 当前生效的规则数值（引擎解析后）")
    md.append("")
    md.append("| 键 | 值 | 语义 / 提示 |")
    md.append("|---|---|---|")
    for k in sorted(cfg):
        doc = docs.get(k, "")
        hint = HINTS.get(k)
        desc = hint if hint is not None else (doc or "-")
        md.append(f"| `{k}` | `{_md_cell(cfg[k])}` | {_md_cell(desc)} |")

    # III. phase order
    md.append("")
    md.append("## 3. 一回合的执行顺序")
    md.append("")
    md.append("引擎按下列阶段每回合顺序执行。理解顺序 = 理解命令生效时机：")
    md.append("")
    md.append("| # | 阶段 | 作用 |")
    md.append("|---|---|---|")
    for i, ph in enumerate(phase_order, 1):
        md.append(f"| {i} | **{ph}** | {PHASE_LABEL.get(ph, '')} |")

    # IV. protocol
    md.append("")
    md.append("## 4. 命令协议")
    md.append("")
    md.append("你每回合输出命令，一行一条：")
    md.append("")
    md.append("```")
    md.append("o  <船id>  n|s|e|w|o    # 移动")
    md.append("a  <攻击船id> <目标船id>   # 攻击船")
    md.append("x  <船id> <owner> <x> <y> # 攻击建筑")
    md.append("d  <船id>                 # 设 defend")
    md.append("s                        # spawn 新船")
    md.append("g                        # 建 dropoff")
    md.append("```")
    md.append("")
    md.append("非法命令 / 超时会记入引擎 error_logs，可能被淘汰；平台会把它反馈给你。")
    md.append("")
    md.append("## 5. 样例脚本")
    md.append("")
    md.append(f"- 完整 C++ starter kit 在 `cpp/`（本材料只读副本）。")
    md.append(f"- 参考对手参数：`../params/` 或平台 `bot_params_{{eco,aggro,control,adaptive}}.txt`。")
    md.append("")

    out = out or (C.ARENA_ROOT / "rules.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build_rules_md()
    print(f"wrote {p}")
