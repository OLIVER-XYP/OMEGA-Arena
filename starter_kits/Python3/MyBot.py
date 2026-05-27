#!/usr/bin/env python3
"""
HaliteCombatBot  —  单文件，双实例对战
游戏可以通过以下命令启动（引擎会将两个 bot 分配到地图对角）：
  ./halite --replay-directory replays/ "python MyBot.py" "python MyBot.py"
"""

import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging

# ── 初始化 ─────────────────────────────────────────────────────────────────
game = hlt.Game()
game.ready("HaliteCombatBot")
logging.info("Bot started, player_id=%d", game.my_id)

# ── 可调参数 ───────────────────────────────────────────────────────────────
HP_DEFEND_THRESHOLD = 35    # HP <= 此值且有相邻敌舰 → 防御
HP_HEAL_THRESHOLD   = 45    # HP <= 此值且无相邻敌舰 → 回血
HP_ATTACK_MIN       = 55    # HP > 此值才主动攻击
HP_HUNT_MIN         = 70    # HP > 此值才主动追击远处敌舰

HALITE_RETURN_RATIO = 0.75  # 货仓达到此比例时返回基地
MINE_STAY_THRESHOLD = 0.10  # 当前格子 halite > MAX_HALITE * 此值时原地挖矿
MAX_FLEET           = 6     # 舰队上限（控制 HP 风险）
SPAWN_UNTIL_TURN    = 250   # 超过此回合停止建造新船
HUNT_RADIUS         = 5     # 在此范围内发现敌舰才进入追击模式

# ── 每艘船的跨回合状态 ──────────────────────────────────────────────────────
# state: 'collect' | 'return' | 'hunt'
ship_state  = {}   # ship_id -> state
hunt_target = {}   # ship_id -> enemy ship_id

# ── 工具函数 ───────────────────────────────────────────────────────────────

def all_enemy_ships():
    """返回所有敌方船只列表。"""
    result = []
    for pid, player in game.players.items():
        if pid != game.my_id:
            result.extend(player.get_ships())
    return result


def adjacent_enemy_ships(ship, enemies, gmap):
    """返回与 ship 严格相邻（曼哈顿距离 = 1）的敌舰列表。"""
    adj = {gmap.normalize(ship.position.directional_offset(d))
           for d in Direction.get_all_cardinals()}
    return [e for e in enemies if e.position in adj]


def hunt_score(my_ship, enemy, gmap):
    """追击评分：越高越值得追（高货物 + 低 HP + 距离近）。"""
    dist = gmap.calculate_distance(my_ship.position, enemy.position)
    return enemy.halite_amount * 0.05 - enemy.hp * 0.3 - dist * 4


def best_mining_target(ship, gmap, radius=8):
    """在半径内寻找 halite/距离比值最高的格子。"""
    best_pos, best_score = None, -1
    sx, sy = ship.position.x, ship.position.y
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius:
                continue
            pos = gmap.normalize(Position(sx + dx, sy + dy))
            dist = gmap.calculate_distance(ship.position, pos)
            halite = gmap[pos].halite_amount
            score = halite if dist == 0 else halite / (dist + 1)
            if score > best_score:
                best_score, best_pos = score, pos
    return best_pos


def move_cost(ship, gmap):
    """当前格子要求的移动燃料（halite）。"""
    return gmap[ship.position].halite_amount // constants.MOVE_COST_RATIO


def can_move(ship, gmap):
    return ship.halite_amount >= move_cost(ship, gmap)


def move_toward(ship, destination, gmap):
    """向目标移动一步；燃料不足时原地停留（顺带挖矿）。"""
    if not can_move(ship, gmap):
        return ship.stay_still()
    direction = gmap.naive_navigate(ship, destination)
    return ship.move(direction)

# ── 主循环 ─────────────────────────────────────────────────────────────────
while True:
    game.update_frame()
    me      = game.me
    gmap    = game.game_map
    turn    = game.turn_number
    enemies = all_enemy_ships()

    command_queue = []
    decided       = set()   # 已确定行动的 ship_id

    my_ships = me.get_ships()
    alive_ids = {s.id for s in my_ships}

    # 清理已死亡船只的状态
    ship_state  = {k: v for k, v in ship_state.items()  if k in alive_ids}
    hunt_target = {k: v for k, v in hunt_target.items() if k in alive_ids}

    # 为新生船只初始化状态
    for ship in my_ships:
        if ship.id not in ship_state:
            ship_state[ship.id] = 'collect'

    # ──────────────────────────────────────────────────────────────────────
    # 第一优先级：HP 危急处理
    # ──────────────────────────────────────────────────────────────────────
    for ship in my_ships:
        adj_e = adjacent_enemy_ships(ship, enemies, gmap)

        if ship.hp <= HP_DEFEND_THRESHOLD and adj_e:
            # HP 过低且被包围 → 撤退；燃料不足才防御
            if can_move(ship, gmap):
                command_queue.append(move_toward(ship, me.shipyard.position, gmap))
            else:
                command_queue.append(ship.defend())
            decided.add(ship.id)
            logging.debug("ship %d RETREAT/DEFEND (hp=%d, adj_enemies=%d)",
                          ship.id, ship.hp, len(adj_e))

        elif ship.hp <= HP_HEAL_THRESHOLD and not adj_e:
            # HP 低但安全 → 回血
            command_queue.append(ship.heal())
            decided.add(ship.id)
            logging.debug("ship %d HEAL (hp=%d)", ship.id, ship.hp)

    # ──────────────────────────────────────────────────────────────────────
    # 第二优先级：与相邻敌舰交战
    # ──────────────────────────────────────────────────────────────────────
    for ship in my_ships:
        if ship.id in decided:
            continue
        adj_e = adjacent_enemy_ships(ship, enemies, gmap)
        if not adj_e:
            continue

        if ship.hp > HP_ATTACK_MIN:
            # 攻击 HP 最低的相邻敌舰（争取击杀）
            target = min(adj_e, key=lambda e: e.hp)
            command_queue.append(ship.attack(target))
            logging.debug("ship %d ATTACK ship %d (enemy_hp=%d, our_hp=%d)",
                          ship.id, target.id, target.hp, ship.hp)
        else:
            # HP 不足攻击 → 撤退；燃料不足才防御
            if can_move(ship, gmap):
                command_queue.append(move_toward(ship, me.shipyard.position, gmap))
            else:
                command_queue.append(ship.defend())
            logging.debug("ship %d RETREAT/DEFEND vs adj enemy (hp=%d)", ship.id, ship.hp)

        decided.add(ship.id)

    # ──────────────────────────────────────────────────────────────────────
    # 第三优先级：采集 / 返回 / 追击
    # ──────────────────────────────────────────────────────────────────────
    for ship in my_ships:
        if ship.id in decided:
            continue

        # ── 状态转换 ──────────────────────────────────────────────────────
        if ship.halite_amount >= int(constants.MAX_HALITE * HALITE_RETURN_RATIO):
            ship_state[ship.id] = 'return'

        if ship_state[ship.id] == 'return' and ship.position == me.shipyard.position:
            ship_state[ship.id] = 'collect'

        # ── 追击判断（仅在 collect 状态且 HP 充足时触发）──────────────────
        if (ship_state[ship.id] == 'collect'
                and ship.hp >= HP_HUNT_MIN
                and enemies):
            # 在追击半径内寻找最佳目标
            in_range = [e for e in enemies
                        if gmap.calculate_distance(ship.position, e.position) <= HUNT_RADIUS]
            if in_range:
                best_enemy = max(in_range, key=lambda e: hunt_score(ship, e, gmap))
                ship_state[ship.id] = 'hunt'
                hunt_target[ship.id] = best_enemy.id
                logging.debug("ship %d HUNT -> enemy %d", ship.id, best_enemy.id)

        # ── 执行当前状态 ───────────────────────────────────────────────────
        state = ship_state[ship.id]

        if state == 'return':
            command_queue.append(move_toward(ship, me.shipyard.position, gmap))

        elif state == 'hunt':
            tid = hunt_target.get(ship.id)
            enemy_ship = next((e for e in enemies if e.id == tid), None)
            if enemy_ship is None:
                # 目标消失，重新进入采集状态
                ship_state[ship.id] = 'collect'
                state = 'collect'
            else:
                command_queue.append(move_toward(ship, enemy_ship.position, gmap))

        if state == 'collect':
            cell = gmap[ship.position]
            mine_threshold = constants.MAX_HALITE * MINE_STAY_THRESHOLD
            if cell.halite_amount >= mine_threshold and not ship.is_full:
                # 当前格子值得挖，原地停留
                command_queue.append(ship.stay_still())
            else:
                # 移动到最佳采矿目标
                tgt = best_mining_target(ship, gmap)
                if tgt and tgt != ship.position:
                    command_queue.append(move_toward(ship, tgt, gmap))
                else:
                    command_queue.append(ship.stay_still())

        decided.add(ship.id)

    # ──────────────────────────────────────────────────────────────────────
    # 建造新船
    # ──────────────────────────────────────────────────────────────────────
    if (turn <= SPAWN_UNTIL_TURN
            and len(my_ships) < MAX_FLEET
            and me.halite_amount >= constants.SHIP_COST
            and not gmap[me.shipyard].is_occupied):
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)
