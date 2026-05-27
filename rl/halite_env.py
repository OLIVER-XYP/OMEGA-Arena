"""
halite_env.py — Gymnasium environment that drives the real C++ Halite engine.

One env step = one game turn for the RL player (player 0). The opponent (player 1)
is a fixed C++ bot launched by the engine. Control is brokered through rl_bot.py
over a local TCP socket (see that file for the wire protocol).

Observation: float32 tensor (C, H, W) of feature planes (halite, own/enemy ship
presence + cargo + hp, structures, time, treasury).

Action: MultiDiscrete([6] * H*W) — a spatial action map. The entry at y*W+x is
applied to the RL player's ship occupying (x, y); cells without our ship are
ignored. Codes: 0 stay, 1 N, 2 S, 3 E, 4 W, 5 attack-adjacent-enemy. Spawning is
handled by a simple env-side heuristic in v1 so the policy can focus on maneuver.

Reward (per turn): scaled (Δ my treasury − Δ enemy treasury) plus a raider term
(+ for enemy ships lost, − for own ships lost), with a terminal win/loss bonus.
This rewards both out-earning and actively shrinking the enemy fleet.
"""

import json
import os
import platform
import socket
import subprocess
import tempfile

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _find_binary(base, stem):
    """Return the first existing path from a platform-ordered list."""
    candidates = []
    if platform.system() == "Windows":
        candidates = [
            os.path.join(base, "Release", f"{stem}.exe"),
            os.path.join(base, f"{stem}.exe"),
        ]
    else:
        candidates = [
            os.path.join(base, stem),
            os.path.join(base, "Release", stem),
        ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # Fall back to the bare stem (rely on PATH)
    return stem if platform.system() != "Windows" else f"{stem}.exe"

ENGINE = _find_binary(os.path.join(ROOT, "game_engine", "build"), "halite")
BOT_EXE = _find_binary(os.path.join(ROOT, "starter_kits", "C++", "build_cmake"), "MyBot")
RL_BOT = os.path.join(ROOT, "rl", "rl_bot.py")
DEFAULT_CFG = os.path.join(ROOT, "starter_kits", "C++", "competitive_engine.json")
OPP_PARAMS = {
    "eco": os.path.join(ROOT, "starter_kits", "C++", "bot_params_eco.txt"),
    "control": os.path.join(ROOT, "starter_kits", "C++", "bot_params_control.txt"),
    "aggro": os.path.join(ROOT, "starter_kits", "C++", "bot_params_aggro.txt"),
}

N_CHANNELS = 11
N_ACTIONS = 6
SHIP_COST_GUESS = 1500          # treasury buffer before the heuristic spawns
import sys
PY = sys.executable


class HaliteRaiderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, opponent="eco", map_size=32, turn_limit=300,
                 engine_cfg=DEFAULT_CFG, ship_cap=12, spawn_until_frac=0.55,
                 kill_reward=1.0, loss_penalty=1.0, score_scale=1000.0,
                 win_bonus=5.0, accept_timeout=40.0, seed=None):
        super().__init__()
        self.opponent = opponent
        self.W = self.H = map_size
        self.turn_limit = turn_limit
        self.engine_cfg = engine_cfg
        self.ship_cap = ship_cap
        self.spawn_until = int(turn_limit * spawn_until_frac)
        self.kill_reward = kill_reward
        self.loss_penalty = loss_penalty
        self.score_scale = score_scale
        self.win_bonus = win_bonus
        self.accept_timeout = accept_timeout
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(N_CHANNELS, self.H, self.W), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([N_ACTIONS] * (self.H * self.W))

        self._proc = None
        self._lsock = None
        self._conn = None
        self._chan = None
        self._last = None          # last state dict from the bot
        self._prev_my = 0
        self._prev_en = 0
        self._prev_my_ships = 0
        self._prev_en_ships = 0

    # ---- lifecycle -------------------------------------------------------
    def _cleanup(self):
        for closer in (self._chan, self._conn, self._lsock):
            try:
                if closer:
                    closer.close()
            except Exception:
                pass
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self._proc = self._lsock = self._conn = self._chan = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cleanup()
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Listen on an ephemeral local port for the bridge bot to connect back.
        self._lsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._lsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._lsock.bind(("127.0.0.1", 0))
        self._lsock.listen(1)
        port = self._lsock.getsockname()[1]
        self._lsock.settimeout(self.accept_timeout)

        bseed = int(self._rng.integers(0, 1_000_000))
        # Unquoted backslash paths — matches the proven roundrobin launch format
        # (our paths contain no spaces). Quoting risks the engine arg parser.
        bot0 = f'{PY} {RL_BOT} {port}'
        bot1 = f'{BOT_EXE} {bseed} {OPP_PARAMS[self.opponent]}'
        cmd = [
            ENGINE, "--width", str(self.W), "--height", str(self.H),
            "--turn-limit", str(self.turn_limit),
            "--no-replay", "--no-logs", "--no-timeout", "--results-as-json",
            "-c", self.engine_cfg, bot0, bot1,
        ]
        self._proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        self._conn, _ = self._lsock.accept()
        self._chan = self._conn.makefile("rw")

        state = self._recv()
        if state is None or state.get("done"):
            # Game ended before it began (launch failure) — return zeros.
            self._last = None
            return np.zeros(self.observation_space.shape, np.float32), {}
        self._last = state
        self._prev_my = state["msc"]
        self._prev_en = state["esc"]
        self._prev_my_ships = len(state["ships"])
        self._prev_en_ships = len(state["enemies"])
        return self._encode(state), {}

    # ---- socket I/O ------------------------------------------------------
    def _recv(self):
        try:
            line = self._chan.readline()
        except Exception:
            return None
        if not line:
            return None
        try:
            return json.loads(line)
        except Exception:
            return None

    def _send(self, obj):
        try:
            self._chan.write(json.dumps(obj) + "\n")
            self._chan.flush()
            return True
        except Exception:
            return False

    # ---- observation encoding -------------------------------------------
    def _encode(self, s):
        W, H = self.W, self.H
        obs = np.zeros((N_CHANNELS, H, W), np.float32)
        hal = np.asarray(s["halite"], np.float32).reshape(H, W)
        obs[0] = np.clip(hal / 1000.0, 0, 10)
        for sid, x, y, cargo, hp in s["ships"]:
            obs[1, y, x] = 1.0
            obs[2, y, x] = cargo / 1000.0
            obs[3, y, x] = hp / 100.0
        for x, y, cargo, hp in s["enemies"]:
            obs[4, y, x] = 1.0
            obs[5, y, x] = cargo / 1000.0
            obs[6, y, x] = hp / 100.0
        for x, y in s["my_struct"]:
            obs[7, y, x] = 1.0
        for x, y in s["enemy_struct"]:
            obs[8, y, x] = 1.0
        obs[9] = max(0.0, 1.0 - s["t"] / self.turn_limit)
        obs[10] = min(10.0, s["mh"] / 1000.0)
        return obs

    # ---- step ------------------------------------------------------------
    def step(self, action):
        if self._last is None:
            return (np.zeros(self.observation_space.shape, np.float32),
                    0.0, True, False, {"reason": "no_game"})

        act = np.asarray(action).reshape(self.H, self.W)
        s = self._last
        acts = {}
        for sid, x, y, cargo, hp in s["ships"]:
            acts[str(sid)] = int(act[y, x])

        # Heuristic spawn: affordable, early enough, under cap, yard clear.
        yard = tuple(s["yard"])
        yard_clear = all((x, y) != yard for _, x, y, _, _ in s["ships"])
        spawn = int(s["mh"] >= SHIP_COST_GUESS and s["t"] <= self.spawn_until
                    and len(s["ships"]) < self.ship_cap and yard_clear)

        if not self._send({"acts": acts, "spawn": spawn}):
            return self._terminal("send_failed")

        nxt = self._recv()
        if nxt is None or nxt.get("done"):
            return self._terminal("game_over")

        # per-turn reward
        d_my = nxt["msc"] - self._prev_my
        d_en = nxt["esc"] - self._prev_en
        en_lost = max(0, self._prev_en_ships - len(nxt["enemies"]))
        my_lost = max(0, self._prev_my_ships - len(nxt["ships"]))
        reward = ((d_my - d_en) / self.score_scale
                  + self.kill_reward * en_lost
                  - self.loss_penalty * my_lost)

        self._last = nxt
        self._prev_my, self._prev_en = nxt["msc"], nxt["esc"]
        self._prev_my_ships = len(nxt["ships"])
        self._prev_en_ships = len(nxt["enemies"])

        info = {"my_score": nxt["msc"], "en_score": nxt["esc"],
                "my_ships": self._prev_my_ships, "en_ships": self._prev_en_ships,
                "turn": nxt["t"]}
        return self._encode(nxt), float(reward), False, False, info

    def _terminal(self, reason):
        # Terminal win/loss bonus from the last known scores.
        won = self._prev_my > self._prev_en
        bonus = self.win_bonus if won else -self.win_bonus
        obs = (self._encode(self._last) if self._last is not None
               else np.zeros(self.observation_space.shape, np.float32))
        info = {"reason": reason, "won": bool(won),
                "my_score": self._prev_my, "en_score": self._prev_en}
        self._cleanup()
        return obs, float(bonus), True, False, info

    def close(self):
        self._cleanup()
