"""
rl_bot.py — Halite-protocol bridge bot for reinforcement learning.

The C++ engine launches this exactly like any other bot (one per player). Instead
of computing moves itself, it RELAYS: each turn it ships the parsed game state to
the RL environment over a local TCP socket, then applies the action the env sends
back. This inverts control so an external RL policy (running in the trainer
process) drives the bot turn-by-turn through the standard stdin/stdout protocol.

Launch (the env does this):
    python rl/rl_bot.py <env_port> [host]

Wire protocol with the env (newline-delimited JSON, one message per turn):
  bot -> env : {"t","done","mh","eh","msc","esc","ships","enemies",
                "my_struct","enemy_struct","yard","halite","w","h"}
  env -> bot : {"acts": {ship_id: code}, "spawn": 0|1}
               code: 0 stay, 1 N, 2 S, 3 E, 4 W, 5 attack-adjacent-enemy
"""

import json
import os
import socket
import sys

# Make the Python3 starter kit's hlt package importable.
_KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "starter_kits", "Python3")
sys.path.insert(0, os.path.abspath(_KIT))

from hlt import Game, constants                      # noqa: E402
from hlt.positionals import Position                 # noqa: E402

CODE_DIR = {1: "n", 2: "s", 3: "e", 4: "w"}          # move codes -> engine tokens


def main():
    port = int(sys.argv[1])
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"

    game = Game()
    game.ready("RLBridge")

    sock = socket.create_connection((host, port))
    chan = sock.makefile("rw")

    def send(obj):
        chan.write(json.dumps(obj) + "\n")
        chan.flush()

    def recv():
        line = chan.readline()
        if not line:
            raise EOFError("env closed")
        return json.loads(line)

    me_id = game.my_id

    while True:
        # update_frame raises SystemExit on EOF (engine ended the game).
        try:
            game.update_frame()
        except (SystemExit, EOFError):
            try:
                send({"done": True})
            except Exception:
                pass
            break

        me = game.me
        gm = game.game_map
        W, H = gm.width, gm.height

        ships = [[s.id, s.position.x, s.position.y, s.halite_amount, s.hp]
                 for s in me.get_ships()]
        enemies, enemy_struct, enemy_halite = [], [], 0
        for pid, p in game.players.items():
            if pid == me_id:
                continue
            enemy_halite += p.halite_amount
            for s in p.get_ships():
                enemies.append([s.position.x, s.position.y, s.halite_amount, s.hp])
            enemy_struct.append([p.shipyard.position.x, p.shipyard.position.y])
            for d in p.get_dropoffs():
                enemy_struct.append([d.position.x, d.position.y])

        my_struct = [[me.shipyard.position.x, me.shipyard.position.y]]
        for d in me.get_dropoffs():
            my_struct.append([d.position.x, d.position.y])

        halite = [gm[Position(x, y)].halite_amount for y in range(H) for x in range(W)]

        send({
            "t": game.turn_number, "done": False,
            "mh": me.halite_amount, "eh": enemy_halite,
            "msc": me.halite_amount, "esc": enemy_halite,
            "ships": ships, "enemies": enemies,
            "my_struct": my_struct, "enemy_struct": enemy_struct,
            "yard": [me.shipyard.position.x, me.shipyard.position.y],
            "halite": halite, "w": W, "h": H,
        })

        msg = recv()
        acts = msg.get("acts", {})
        spawn = msg.get("spawn", 0)

        commands = []
        ship_by_id = {s.id: s for s in me.get_ships()}
        enemy_ships = []
        for pid, p in game.players.items():
            if pid == me_id:
                continue
            enemy_ships.extend(p.get_ships())

        for sid_str, code in acts.items():
            sid = int(sid_str)
            ship = ship_by_id.get(sid)
            if ship is None:
                continue
            cell_h = gm[ship.position].halite_amount
            can_move = ship.halite_amount >= cell_h // constants.MOVE_COST_RATIO
            if code == 0:
                commands.append(ship.move("o"))
            elif code in CODE_DIR:
                commands.append(ship.move(CODE_DIR[code]) if can_move else ship.move("o"))
            elif code == 5:
                # attack the nearest adjacent enemy; if none adjacent, hold.
                tgt, best = None, 2
                for e in enemy_ships:
                    d = gm.calculate_distance(ship.position, e.position)
                    if d < best:
                        best, tgt = d, e
                if tgt is not None and best == 1:
                    commands.append(ship.attack(tgt))
                else:
                    commands.append(ship.move("o"))

        if spawn:
            commands.append(me.shipyard.spawn())

        game.end_turn(commands)

    try:
        chan.close()
        sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
