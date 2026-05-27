#!/usr/bin/env python3
"""
GreedyBot v7 — Halite III

游戏基本信息：
  MAX_TURNS = 1000（由引擎动态下发，可用 constants.MAX_TURNS 读取）
  地图环形拓扑：走出边界从对侧出现，这是官方设计。

v7 改进（相对 v6）：
  1. 战略占领：超过 STRATEGIC_DIST 且 halite >= STRATEGIC_RICH 的格子
     绕过效率计算，直接用绝对 halite 量排序（类似"插旗占领"逻辑）
  2. 终局回家意识：
     a. return buffer 从 +3 扩大到 +6，给路径绕行留余量
     b. turns_left <= END_RUSH_TURNS(20) 时，所有带货船强制返回，
        不再采矿，确保 halite 在游戏结束前运回基地
"""
import hlt
from hlt import constants
from hlt.positionals import Direction, Position

game = hlt.Game()
game.ready("GreedyBot v7")

# ── 持久状态 ──────────────────────────────────────────────────────────────────
ship_mode        = {}   # ship_id -> 'collect' | 'return'
ship_target      = {}   # ship_id -> Position
ship_is_fallback = set()
dropoff_ships    = {}   # ship_id -> 目标 Position（待建造的 dropoff 船）

# ── 策略参数 ──────────────────────────────────────────────────────────────────
STOP_RATIO = 0.03852
SPAWN_END_RATIO = 0.42676
SAFE_MARGIN      = 1
RUSH_TURNS       = 20
MIN_RETURN_CARGO = 235
DROPOFF_BUFFER   = 500
DROPOFF_BUFFER2  = 250
MAX_DROPOFFS     = 2

RETURN_CLOSE = 0.55067
RETURN_FAR = 0.82827
RETURN_DIST = 15

# 战略占领参数
STRATEGIC_DIST = 5
STRATEGIC_RICH = 565
STRATEGIC_EFF_THRESH = 20.27296

FALLBACK_MIN_HALITE = 44

# 攻击参数
ENABLE_ATTACK = True   # 是否启用主动拦截逻辑
ATTACK_RATIO = 1.55982

# 终局参数（1000 轮游戏）
END_RUSH_TURNS = 32
RETURN_BUFFER = 12


def adaptive_return_threshold(home_dist):
    t = min(home_dist / RETURN_DIST, 1.0)
    return int(constants.MAX_HALITE * (RETURN_CLOSE + (RETURN_FAR - RETURN_CLOSE) * t))

# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def get_drops(me):
    return [me.shipyard.position] + [d.position for d in me.get_dropoffs()]

def get_enemy_drops(game_obj):
    return [pos
            for pid, p in game_obj.players.items() if pid != game_obj.me.id
            for pos in [p.shipyard.position] + [d.position for d in p.get_dropoffs()]]

def nearest_drop(gmap, pos, drops):
    return min(drops, key=lambda d: gmap.calculate_distance(pos, d))

def nearest_drop_dist(gmap, pos, drops):
    return gmap.calculate_distance(pos, nearest_drop(gmap, pos, drops))

def moveable(gmap, ship, is_inspired=False):
    ratio = constants.INSPIRED_MOVE_COST_RATIO if is_inspired else constants.MOVE_COST_RATIO
    return ship.halite_amount >= gmap[ship.position].halite_amount // ratio

def mining_sim(cell_halite, stop, inspired, current_cargo=0):
    """模拟采矿；返回 (本次净收集量, 消耗回合数)。"""
    capacity_left = constants.MAX_HALITE - current_cargo
    if capacity_left <= 0:
        return 0, 0
    ratio = constants.INSPIRED_EXTRACT_RATIO if inspired else constants.EXTRACT_RATIO
    mult  = 1 + (constants.INSPIRED_BONUS_MULTIPLIER if inspired else 0)
    h, col, t = cell_halite, 0, 0
    while h > stop and col < capacity_left:
        extracted = h // ratio
        if extracted == 0:
            break
        gained = min(int(extracted * mult), capacity_left - col)
        if gained <= 0:
            break
        col += gained
        h   -= extracted
        t   += 1
    return col, t

def cell_score(gmap, pos, ship_pos, drops, inspired, stop,
               current_cargo=0, turns_left=9999):
    """halite/turn 效率得分；turns_left 过滤回合内无法完成往返的格子。"""
    halite = gmap[pos].halite_amount
    if halite <= stop:
        return 0.0
    travel = gmap.calculate_distance(ship_pos, pos)
    collected, mine_turns = mining_sim(halite, stop, inspired, current_cargo)
    if collected == 0:
        return 0.0
    ret_dist    = nearest_drop_dist(gmap, pos, drops)
    total_turns = travel + mine_turns + ret_dist
    if total_turns > turns_left:
        return 0.0
    return collected / total_turns if total_turns > 0 else 0.0

def build_inspiration_map(gmap, enemies):
    r, threshold = constants.INSPIRATION_RADIUS, constants.INSPIRATION_SHIP_COUNT
    result = {}
    for y in range(gmap.height):
        for x in range(gmap.width):
            pos   = Position(x, y)
            count = sum(1 for e in enemies
                        if gmap.calculate_distance(pos, e.position) <= r)
            result[pos] = count >= threshold
    return result

def find_target(gmap, ship, drops, enemy_drops, insp_map, claimed, stop,
                safe_margin, turns_left=9999):
    """
    全图扫描找最优未占用采矿格。
    双路径：
      - 效率路径：score = collected / total_turns（全部格子参与）
      - 战略路径：score = raw halite（仅 travel >= STRATEGIC_DIST 且 halite >= STRATEGIC_RICH）
    当本地效率偏低且船仓较空时，优先选战略目标（占领远处富矿区）。
    """
    best_eff_pos, best_eff_sc   = None, -1.0
    best_str_pos, best_str_hal  = None, -1
    cargo = ship.halite_amount

    for y in range(gmap.height):
        for x in range(gmap.width):
            pos = Position(x, y)
            if claimed.get(pos, 0) >= 1:
                continue
            our_dist = nearest_drop_dist(gmap, pos, drops)
            if enemy_drops:
                enemy_dist = min(gmap.calculate_distance(pos, ep) for ep in enemy_drops)
                if enemy_dist - our_dist < safe_margin:
                    continue

            halite = gmap[pos].halite_amount
            travel = gmap.calculate_distance(ship.position, pos)

            # ── 效率路径（全部格子）──────────────────────────────────────────
            sc = cell_score(gmap, pos, ship.position, drops,
                            insp_map.get(pos, False), stop, cargo, turns_left)
            if sc > best_eff_sc:
                best_eff_sc, best_eff_pos = sc, pos

            # ── 战略路径（远距离富矿）────────────────────────────────────────
            if travel >= STRATEGIC_DIST and halite >= STRATEGIC_RICH:
                # 保守估计：至少挖 2 回合；超出 turns_left 则放弃
                if travel + 2 + our_dist <= turns_left and halite > best_str_hal:
                    best_str_hal, best_str_pos = halite, pos

    # ── 选择：战略 vs 效率 ────────────────────────────────────────────────────
    # 战略目标胜出条件：
    #   1. 存在战略目标
    #   2. 船仓 < 50%（有足够空间收集大量矿）
    #   3. 本地效率得分偏低（说明附近矿已稀疏，应扩张）
    if (best_str_pos is not None
            and cargo < constants.MAX_HALITE * 0.5
            and best_eff_sc < STRATEGIC_EFF_THRESH):
        return best_str_pos
    return best_eff_pos

def find_best_dropoff_pos(gmap, drops, service_points, enemy_drops):
    best_pos, best_val = None, -1
    for y in range(gmap.height):
        for x in range(gmap.width):
            pos = Position(x, y)
            our_dist = nearest_drop_dist(gmap, pos, drops)
            if enemy_drops:
                enemy_dist = min(gmap.calculate_distance(pos, ep) for ep in enemy_drops)
                if enemy_dist - our_dist < SAFE_MARGIN:
                    continue
            min_svc = min(gmap.calculate_distance(pos, sp) for sp in service_points)
            if min_svc < 5:
                continue
            val = 0
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    npos = Position((x + dx) % gmap.width, (y + dy) % gmap.height)
                    if gmap.calculate_distance(pos, npos) <= 4:
                        val += gmap[npos].halite_amount
            if val > best_val:
                best_val, best_pos = val, pos
    return best_pos

# ── 导航 ──────────────────────────────────────────────────────────────────────

def nav(gmap, ship, dest, queue):
    for direction in gmap.get_unsafe_moves(ship.position, dest):
        tp = ship.position.directional_offset(direction)
        if not gmap[tp].is_occupied:
            gmap[tp].mark_unsafe(ship)
            queue.append(ship.move(direction))
            return
    for direction in Direction.get_all_cardinals():
        tp = ship.position.directional_offset(direction)
        if not gmap[tp].is_occupied:
            gmap[tp].mark_unsafe(ship)
            queue.append(ship.move(direction))
            return
    stay(gmap, ship, queue)

def stay(gmap, ship, queue):
    gmap[ship.position].mark_unsafe(ship)
    queue.append(ship.stay_still())

def handle_no_target(gmap, ship, drops, enemy_drops, insp_map, claimed,
                     stop, home, queue, safe_margin, turns_left, is_inspired):
    if ship.halite_amount >= MIN_RETURN_CARGO:
        ship_mode[ship.id] = 'return'
        if moveable(gmap, ship, is_inspired): nav(gmap, ship, home, queue)
        else: stay(gmap, ship, queue)
        return
    if gmap[ship.position].halite_amount > 0:
        stay(gmap, ship, queue)
        return
    fallback = find_target(gmap, ship, drops, enemy_drops, insp_map,
                           claimed, 0, safe_margin, turns_left)
    if fallback is None:
        fallback = find_target(gmap, ship, drops, enemy_drops, insp_map,
                               {}, 0, safe_margin, turns_left)
    # 如果 fallback 格子 halite 太少（残矿），不值得追，直接回家
    if fallback is not None and gmap[fallback].halite_amount < FALLBACK_MIN_HALITE:
        fallback = None
    if fallback is not None:
        ship_target[ship.id] = fallback
        claimed[fallback] = claimed.get(fallback, 0) + 1
        ship_is_fallback.add(ship.id)
        if moveable(gmap, ship, is_inspired): nav(gmap, ship, fallback, queue)
        else: stay(gmap, ship, queue)
    else:
        ship_mode[ship.id] = 'return'
        if moveable(gmap, ship, is_inspired): nav(gmap, ship, home, queue)
        else: stay(gmap, ship, queue)

# ── 主循环 ────────────────────────────────────────────────────────────────────

while True:
    game.update_frame()
    me, gmap, turn = game.me, game.game_map, game.turn_number
    # MAX_TURNS = 1000（由引擎下发，此处动态读取）
    turns_left  = constants.MAX_TURNS - turn
    drops       = get_drops(me)
    enemy_drops = get_enemy_drops(game)
    stop        = int(constants.MAX_HALITE * STOP_RATIO)
    enemies     = [s for pid, p in game.players.items()
                   if pid != me.id for s in p.get_ships()]

    safe_margin = 0 if turn <= RUSH_TURNS else SAFE_MARGIN
    command_queue = []
    # 本回合可用银行余额；在 dropoff/spawn 提交时同步扣减，
    # 防止同回合两艘建造船各自独立通过检查导致超支
    halite_budget = me.halite_amount

    # ── 清理已死亡船只的状态 ──────────────────────────────────────────────────
    alive = {s.id for s in me.get_ships()}
    for sid in list(ship_mode):
        if sid not in alive:
            ship_mode.pop(sid)
            ship_target.pop(sid, None)
            ship_is_fallback.discard(sid)
    for sid in list(dropoff_ships):
        if sid not in alive:
            dropoff_ships.pop(sid)
    if turns_left < 50:
        dropoff_ships.clear()

    insp_map = build_inspiration_map(gmap, enemies)

    claimed = {}
    for sid, pos in ship_target.items():
        if sid in alive:
            claimed[pos] = claimed.get(pos, 0) + 1

    # ── Dropoff 建造指定 ──────────────────────────────────────────────────────
    all_service = drops + list(dropoff_ships.values())
    num_built   = len(me.get_dropoffs())
    num_pending = len(dropoff_ships)
    buffer      = DROPOFF_BUFFER if num_built == 0 else DROPOFF_BUFFER2

    if (num_built + num_pending < MAX_DROPOFFS
            and me.halite_amount >= constants.DROPOFF_COST + buffer
            and turn < 500
            and me.get_ships()):
        new_pos = find_best_dropoff_pos(gmap, drops, all_service, enemy_drops)
        if new_pos is not None:
            candidates = [s for s in me.get_ships() if s.id not in dropoff_ships]
            if candidates:
                builder = min(candidates,
                              key=lambda s: gmap.calculate_distance(s.position, new_pos))
                dropoff_ships[builder.id] = new_pos

    # ── 初始化新生成船只的模式 ────────────────────────────────────────────────
    for ship in me.get_ships():
        ship_mode.setdefault(ship.id, 'collect')

    # ── 按优先级排序：返程船先处理，防止堵塞 dropoff 入口 ────────────────────
    def ship_key(s):
        m = ship_mode.get(s.id, 'collect')
        if m == 'return':         return (0, s.id)
        if s.id in dropoff_ships: return (1, s.id)
        return                           (2, s.id)

    for ship in sorted(me.get_ships(), key=ship_key):
        home        = nearest_drop(gmap, ship.position, drops)
        home_dist   = gmap.calculate_distance(ship.position, home)
        is_inspired = insp_map.get(ship.position, False)

        # ── 终局紧急回家（turns_left 极少，强制所有带货船返回）────────────────
        # 确保 MAX_TURNS=1000 结束时所有 halite 都已运回基地
        if turns_left <= END_RUSH_TURNS and ship.halite_amount > 0:
            ship_mode[ship.id] = 'return'
            ship_target.pop(ship.id, None)

        # ── 普通回家触发（buffer 加大到 RETURN_BUFFER 留给路径绕行）────────────
        if turns_left <= home_dist + RETURN_BUFFER:
            ship_mode[ship.id] = 'return'
            ship_target.pop(ship.id, None)

        ret_threshold = adaptive_return_threshold(home_dist)
        if ship.halite_amount >= ret_threshold:
            ship_mode[ship.id] = 'return'

        if ship_mode[ship.id] == 'return' and ship.position == home:
            ship_mode[ship.id] = 'collect'
            ship_target.pop(ship.id, None)

        # ── Dropoff 建造船逻辑 ────────────────────────────────────────────────
        if ship.id in dropoff_ships:
            dpos = dropoff_ships[ship.id]
            if ship.position == dpos:
                eff_cost = (constants.DROPOFF_COST
                            - ship.halite_amount
                            - gmap[dpos].halite_amount)
                if halite_budget >= eff_cost + buffer:
                    command_queue.append(ship.make_dropoff())
                    halite_budget -= max(0, eff_cost)  # 即时扣减，防止同回合第二艘建造船重复通过
                    dropoff_ships.pop(ship.id)
                else:
                    stay(gmap, ship, command_queue)
            else:
                if moveable(gmap, ship, is_inspired): nav(gmap, ship, dpos, command_queue)
                else: stay(gmap, ship, command_queue)
            continue

        # ── 敌船感知 ──────────────────────────────────────────────────────────
        adj_e = [e for e in enemies
                 if gmap.calculate_distance(ship.position, e.position) <= 2]
        if adj_e:
            cell_h = gmap[ship.position].halite_amount
            if is_inspired and cell_h > stop * 3 and not ship.is_full:
                stay(gmap, ship, command_queue)
                continue

            # 攻击判断：若有载货量远超自己的敌船在距离 1 格以内，主动拦截
            # 碰撞双方同归于尽，敌船 halite > 自船 halite * ATTACK_RATIO 时合算
            if ENABLE_ATTACK:
                close_e = [e for e in adj_e
                           if gmap.calculate_distance(ship.position, e.position) == 1]
                if close_e:
                    target_e = max(close_e, key=lambda e: e.halite_amount)
                    if (target_e.halite_amount > ship.halite_amount * ATTACK_RATIO
                            and ship.id not in dropoff_ships):
                        # 向敌船方向移动，强迫碰撞
                        if moveable(gmap, ship, is_inspired):
                            nav(gmap, ship, target_e.position, command_queue)
                        else:
                            stay(gmap, ship, command_queue)
                        continue

            if ship.id in ship_target:
                claimed[ship_target[ship.id]] = max(0, claimed.get(ship_target[ship.id], 0) - 1)
                ship_target.pop(ship.id, None)
                ship_is_fallback.discard(ship.id)
            if moveable(gmap, ship, is_inspired):
                nav(gmap, ship, home, command_queue)
            else:
                stay(gmap, ship, command_queue)
            continue

        # ── 返程模式 ──────────────────────────────────────────────────────────
        if ship_mode[ship.id] == 'return':
            if moveable(gmap, ship, is_inspired):
                nav(gmap, ship, home, command_queue)
            else:
                stay(gmap, ship, command_queue)
            continue

        # ── 采集模式 ──────────────────────────────────────────────────────────
        target = ship_target.get(ship.id)

        eff_stop = 0 if ship.id in ship_is_fallback else stop
        if target is not None and gmap[target].halite_amount <= eff_stop:
            claimed[target] = max(0, claimed.get(target, 0) - 1)
            ship_target.pop(ship.id, None)
            ship_is_fallback.discard(ship.id)
            target = None

        if target is None:
            target = find_target(gmap, ship, drops, enemy_drops, insp_map,
                                 claimed, stop, safe_margin, turns_left)
            if target is not None:
                ship_is_fallback.discard(ship.id)
            else:
                handle_no_target(gmap, ship, drops, enemy_drops, insp_map,
                                 claimed, stop, home, command_queue,
                                 safe_margin, turns_left, is_inspired)
                continue
            ship_target[ship.id] = target
            claimed[target] = claimed.get(target, 0) + 1

        mine_threshold = stop if ship.id not in ship_is_fallback else 0
        if ship.position == target:
            if gmap[target].halite_amount > mine_threshold and not ship.is_full:
                stay(gmap, ship, command_queue)
            else:
                claimed[target] = max(0, claimed.get(target, 0) - 1)
                ship_target.pop(ship.id, None)
                ship_is_fallback.discard(ship.id)
                target = find_target(gmap, ship, drops, enemy_drops, insp_map,
                                     claimed, stop, safe_margin, turns_left)
                if target is not None:
                    ship_is_fallback.discard(ship.id)
                    ship_target[ship.id] = target
                    claimed[target] = claimed.get(target, 0) + 1
                    if moveable(gmap, ship, is_inspired): nav(gmap, ship, target, command_queue)
                    else: stay(gmap, ship, command_queue)
                else:
                    handle_no_target(gmap, ship, drops, enemy_drops, insp_map,
                                     claimed, stop, home, command_queue,
                                     safe_margin, turns_left, is_inspired)
            continue

        if moveable(gmap, ship, is_inspired):
            nav(gmap, ship, target, command_queue)
        else:
            stay(gmap, ship, command_queue)

    # ── 生成新船 ──────────────────────────────────────────────────────────────
    num_dropoffs = len(me.get_dropoffs())
    max_ships    = max(10, gmap.width * gmap.height // max(40, 55 - 5 * num_dropoffs))
    if (turn <= constants.MAX_TURNS * SPAWN_END_RATIO
            and halite_budget >= constants.SHIP_COST
            and not gmap[me.shipyard.position].is_occupied
            and turns_left >= 50
            and len(alive) < max_ships):
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)
