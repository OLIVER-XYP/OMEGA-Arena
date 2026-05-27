# Halite III — Competitive Ruleset & RL Training Pipeline

Fork of the [Halite III](https://halite.io) AI programming competition engine.
This project adds a **competitive ruleset** (configurable engine mechanics for
strategy diversity), **strategy bot profiles**, **metagame analysis tools**, and
a **deep reinforcement learning pipeline** that trains policies against the real
C++ engine via a Gymnasium environment.

## Directory structure

```
Halite-III/
├── game_engine/              C++ game engine (CMake, MSVC/GCC/Clang)
│   ├── config/               Constants.hpp/cpp — all engine knobs (JSON-live)
│   ├── core/                 Game loop, state store, HaliteImpl
│   ├── core/config/          GameConfig — JSON ↔ constants bridge
│   ├── mapgen/               FractalValueNoise tile-based map generator
│   ├── model/                Cell, Entity, Ship, Player, Dropoff
│   ├── rules/phases/         Per-turn rule phases (Combat, Mining, Capture,
│   │                         Inspiration, Regen, OverShipTax, HaliteRebalance…)
│   └── test/                 Catch2 unit tests (304/304 pass)
│
├── starter_kits/C++/         C++ reference bot + metagame tooling
│   ├── MyBot.cpp             Bot logic — nav, mining, combat, hunt, defend,
│   │                         focus-fire, threat-map, spawn-guard
│   ├── hlt/                  Halite C++ SDK (game.hpp, command.hpp, …)
│   ├── bot_params.hpp        All tunable parameters with file loader + clamps
│   ├── bot_params_eco.txt    ECO strategy — pure economy, max mining
│   ├── bot_params_aggro.txt  AGGRO strategy — coordinated wolfpack raider
│   ├── bot_params_control.txt CONTROL strategy — turtle with threat-aware banking
│   ├── competitive_engine.json  Ruleset for competitive play
│   ├── metagame_roundrobin.py   Round-robin tournament + cycle detection
│   ├── sweep_combat.py          Parametric sweep of engine knobs for cycles
│   ├── measure_combat.py        Combat-effectiveness instrument
│   ├── analyze_scores.py        Score-distribution analysis
│   └── CMakeLists.txt        Build definition for MyBot.exe
│
├── starter_kits/Python3/     Python3 starter kit — used by the RL bridge
│   └── hlt/                  Protocol layer (networking, commands, entity, …)
│
├── rl/                       Deep RL pipeline (B2 path)
│   ├── rl_bot.py             Bridge bot — relays engine state ↔ env over TCP
│   ├── halite_env.py         Gymnasium env — launches engine as subprocess
│   ├── halite_pettingzoo.py  PettingZoo ParallelEnv — 2-agent wrapper
│   ├── train_ppo.py          SB3 PPO training entry point
│   └── smoke_test.py         End-to-end bridge validation (no torch needed)
│
├── debug_replay.py           Reads zstd .hlt replays — ship counts, scores, deposits
├── libhaliteviz/             JavaScript replay visualizer
├── apiserver/                (Original) Flask API server
├── website/                  (Original) Competition website
└── PROJECT.md                This file
```

## Prerequisites

### Build tools
- **C++17 compiler**: MSVC 2019+ (Windows), GCC 9+/Clang 10+ (Linux/macOS)
- **CMake ≥ 3.14**
- **Python ≥ 3.10**

### Python packages (for metagame tools + RL)

```bash
pip install numpy zstandard           # replay analysis + roundrobin
pip install gymnasium                 # RL env
pip install "stable-baselines3[extra]" # PPO training (includes torch)
```

Optional: `sb3-contrib` for masked actions (not yet wired).

## Build

### Windows (MSVC)

```powershell
# Engine
cd game_engine
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
# → game_engine/build/Release/halite.exe

# C++ bot
cd starter_kits/C++
cmake -B build_cmake -G "Visual Studio 17 2022"
cmake --build build_cmake --config Release
# → starter_kits/C++/build_cmake/Release/MyBot.exe
```

### Linux / macOS (GCC / Clang)

```bash
# Engine
cd game_engine
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
# → game_engine/build/halite  (or game_engine/build/Release/halite)

# C++ bot
cd starter_kits/C++
cmake -B build_cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake
# → starter_kits/C++/build_cmake/MyBot  (or build_cmake/Release/MyBot)
```

The RL env auto-detects the binary location (tries `Release/` and bare `build/`
subdirectories, with `.exe` suffix on Windows).

### Tests

```bash
cd game_engine
cmake --build build --target test --config Release
# 304/304 passing
```

## Engine architecture

### Turn pipeline

Each turn the engine runs these phases in order:

| # | Phase | What it does |
|---|-------|-------------|
| 1 | Inspiration | Ships near ≥2 enemies mine at 2× rate |
| 2 | Validation | Commands validated, invalid discarded |
| 3 | Construction | Dropoffs built, ships spawn |
| 4 | Defend | `is_defending` set (immune to damage) |
| 5 | Movement | Ships move (halite cost = cell/ratio) |
| 6 | Combat | Attacks resolved (damage 70, steal 60%, kill-credit) |
| 7 | Heal | Ships on structures heal to full HP |
| 8 | Dump | Dead ships' cargo returned to cell |
| 9 | Spawn | `GENERATE` commands create ships |
| 10 | Mining | Ships extract halite from their cell |
| 11 | [Regen] | Cell halite regrowth toward initial (default off) |
| 12 | Capture | Outnumbered player's isolated ships captured |
| 13 | OverShipTax | Fleet upkeep on ships above threshold |
| 14 | HaliteRebalance | After-game rubber-band (disabled for competitive) |

### Combat mechanics (competitive ruleset)

- Ship HP = 100; attack damage = 70 (2-hit kill)
- Attacker steals 60% of target's cargo on hit
- Attacker takes 20 self-damage per hit (`ATTACK_HP_SELF_DAMAGE`)
- Killing a ship transfers its remaining cargo to the killer's score (`KILL_CREDIT_TO_ATTACKER`)
- Defending ships are immune to damage; `DEFEND_RETALIATION_DAMAGE` (default 0) reflects damage
- Ships heal to full HP only at structures (shipyard/dropoff)

### Configurable engine knobs (partial list)

All knobs in `Constants.hpp/cpp` are **JSON-live** — write them into a config
file and pass `-c config.json` to the engine. Key knobs:

| Key | Default | Description |
|-----|---------|-------------|
| `ATTACK_HP_DAMAGE` | 70 | Damage per hit |
| `ATTACK_HP_SELF_DAMAGE` | 20 | Self-damage per hit |
| `ATTACK_HALITE_STEAL_RATIO` | 0.60 | Fraction of target cargo stolen |
| `KILL_CREDIT_TO_ATTACKER` | true | Cargo → killer's score on kill |
| `KILL_HALITE_BONUS_RATIO` | 0.0 | Extra multiplier on kill credit |
| `ENABLE_ATTACKER_SELF_DAMAGE` | true | Apply self-damage |
| `DEFEND_RETALIATION_DAMAGE` | 0 | HP reflected when hitting a defender |
| `DEFEND_ALLOWS_MINING` | false | Defending ships still mine |
| `CAPTURE_ENABLED` | true | Isolated ships get captured |
| `CAPTURE_RADIUS` | 3 | Capture scan radius |
| `SHIPS_ABOVE_FOR_CAPTURE` | 2 | Ships needed above local parity |
| `OVER_SHIP_TAX_THRESHOLD` | 7 | Ships before tax applies |
| `OVER_SHIP_TAX_PER_TURN` | 0 | Halite deducted per excess ship |
| `SHIP_INCOME_PER_TURN` | 0 | Halite credited per ship (score inflation — keep 0) |
| `INSPIRATION_ENABLED` | true | Bonus mining near enemies |
| `INSPIRED_BONUS_MULTIPLIER` | 2 | Mining multiplier when inspired |
| `INSPIRATION_SHIP_COUNT` | 2 | Enemy ships needed to trigger |
| `SPAWN_COST_GROWTH` | 0.03 | Extra cost per existing ship |

## Strategy bots (C++)

Three hand-tuned strategy profiles in `starter_kits/C++/bot_params_*.txt`.
Each is a key=value file loaded by `MyBot.exe <seed> <params_file>`.

### ECO — pure economy
- No hunting (`HUNT_MAX_HUNTERS=0`), reactive attack only at home
- Wide search radius, strong stay-bonus, deep cargo thresholds
- Heavy territory penalty — never invades enemy half
- 3 dropoffs for map coverage
- **Beats**: CONTROL (out-produces turtle)
- **Loses to**: AGGRO (unprotected miners are raided)

### AGGRO — coordinated wolfpack raider
- `HUNTERS_PER_TARGET=2` packs land 2-hit kills (140 > 100 HP)
- `HUNT_MAX_HUNTERS=4`, all-in raiding wing
- Hit-and-run discipline: retreat to heal at HP 45, only strike laden ships
- No territory filter, invades freely
- **Beats**: ECO (kills undefended miners, shrinks their fleet)
- **Loses to**: CONTROL (clustered defense + capture punishes isolated hunters)

### CONTROL — turtle with threat-aware banking
- `HUNT_MAX_HUNTERS=2` opportunistic, defend off
- `THREAT_AVOID_WEIGHT=3.0` — loaded ships route AROUND raiders
- Balanced economy: 2 dropoffs, moderate spawn
- **Beats**: AGGRO (threat-nav protects cargo; capture eats isolated hunters)
- **Loses to**: ECO (can't out-produce full-map economy)

### Bot parameter reference

All parameters defined in `bot_params.hpp` with defaults and clamp ranges.
Key groups:

| Group | Parameters |
|-------|-----------|
| Economy | `STOP_RATIO`, `SPAWN_END_RATIO`, `MIN_RETURN_CARGO`, `RETURN_CLOSE/FAR` |
| Mining | `SEARCH_RADIUS_EARLY/MID/LATE`, `STAY_BONUS`, `MIN_CELL_HALITE_CONSIDER` |
| Territory | `INVASION_PENALTY`, `DOMINANCE_RELAX_RATIO`, `EXPAND_RADIUS_BONUS` |
| Combat | `ATTACK_RATIO`, `ATTACK_MIN_TARGET_HALITE`, `ATTACK_MAX_SELF_HALITE_FOR_RISK` |
| Hunt | `HUNT_MIN_ENEMY_HALITE`, `HUNT_MAX_RANGE`, `HUNT_MAX_HUNTERS`, `HUNTERS_PER_TARGET` |
| Defense | `DEFEND_MIN_CARGO`, `FOCUS_FIRE_MIN_SHIPS`, `THREAT_AVOID_WEIGHT` |
| Spawn | `SPAWN_PAYBACK_TURNS`, `SPAWN_MIN_TURNS_LEFT`, `SHIPYARD_TARGET_PENALTY` |
| Dropoff | `MAX_DROPOFFS`, `DROPOFF_MIN_LOCAL_HALITE`, `DROPOFF_MIN_DIST_FROM_OWN` |

## Metagame tooling (C++ kit / Python)

### `competitive_engine.json`

The competitive ruleset. Clean canonical baseline:
- No per-ship income or tax (those are ML-tuning artifacts that inflate score)
- Kill-credit on, attacker self-damage 20, steal 0.60, retaliation 0
- Capture enabled (radius 3, ships-above 2)
- HaliteRebalance and emergency spawn disabled

### `metagame_roundrobin.py` — round-robin tournament

Runs every strategy pair for N seeds (each side gets P0), outputs a win-rate
matrix and checks for cyclic dominance and Nash equilibrium entropy.

```bash
cd starter_kits/C++
python metagame_roundrobin.py --seeds 20 --workers 8
```

Output:
- Pairwise win rates (e.g., `aggro>eco 60.0%`)
- Win-rate matrix
- Average win rates with dominance flags
- Cyclic dominance check (rock-paper-scissors)
- Nash entropy / max entropy (health metric; >75% = healthy)

### `sweep_combat.py` — parametric sweep

Sweeps one engine config key over a list of values, runs the 3 matchups at each
value, reports which settings produce a cycle. Reusable for any knob.

```bash
python sweep_combat.py --key KILL_HALITE_BONUS_RATIO --values 0,1,2,4,8 --seeds 10
```

### `debug_replay.py` — replay inspector

Reads a zstd-compressed `.hlt` replay file and prints per-player statistics and
turn-by-turn ship counts / deposits.

```bash
python debug_replay.py path/to/replay.hlt
```

### `measure_combat.py` / `analyze_scores.py`

Combat-effectiveness measurement and score-distribution analysis (used during
balance tuning).

## RL pipeline (`rl/`)

The B2 deep-RL path: train a neural-net policy that controls player 0 against a
fixed C++ opponent, using the **real engine** (no reimplementation).

### Architecture

```
┌──────────────┐     TCP (JSON/line)     ┌────────────┐
│  halite.exe  │◄────stdin/stdout───────►│  rl_bot.py │
│  (C++ eng)   │                         │  (bridge)  │
│              │     The engine launches  │            │
│  vs fixed    │     rl_bot as player 0. │            │
│  C++ opp.    │                         │            │
└──────────────┘                         └─────┬──────┘
                                               │
                                    state JSON  │  action JSON
                                               │
                                         ┌─────▼──────┐
                                         │ halite_env │
                                         │ (Gymnasium)│
                                         │ or         │
                                         │halite_pz.py│
                                         │ (PettingZoo│
                                         │ Parallel)  │
                                         └─────┬──────┘
                                               │
                                               │ obs, reward
                                               ▼
                                         ┌────────────┐
                                         │ train_ppo  │
                                         │ (SB3 PPO)  │
                                         └────────────┘
```

### `rl_bot.py` — bridge bot

A Halite-protocol bot (launched by the engine as any other bot) that:
1. Receives init and per-turn game state from the engine over stdin
2. Parses it using the Python3 `hlt` library
3. Sends the state as JSON over a TCP socket to the env process
4. Receives an action map back, translates codes to engine commands
5. Sends engine commands over stdout

Action codes: `0` stay, `1` N, `2` S, `3` E, `4` W, `5` attack-adjacent-enemy.
Spawn is a boolean the env appends.

### `halite_env.py` — Gymnasium environment

`HaliteRaiderEnv(gym.Env)`:

| Feature | Detail |
|---------|--------|
| Observation | `float32 (11, 32, 32)` — 11 feature planes |
| Action | `MultiDiscrete([6] * 1024)` — per-cell spatial action map |
| Reward | Δscore/1000 + 1.0 per enemy ship killed − 1.0 per own ship lost + ±5 win bonus |
| Opponent | Fixed C++ bot (`eco`, `control`, or `aggro`) |
| Engine | Launched as subprocess with `--no-timeout` |

Feature planes (11 channels):
0. Cell halite (clipped to [0,10] × 1000)
1. Own ship presence (binary)
2. Own ship cargo / 1000
3. Own ship HP / 100
4. Enemy ship presence
5. Enemy ship cargo / 1000
6. Enemy ship HP / 100
7. Own structures
8. Enemy structures
9. Remaining turns (0→1 decreasing)
10. Own treasury / 1000

```
python rl/smoke_test.py
```

### `halite_pettingzoo.py` — PettingZoo ParallelEnv

`HalitePZEnv(pettingzoo.ParallelEnv)`: a proper 2-agent multi-agent wrapper
around the same engine infrastructure. Agents: `"rl"` (player 0, learning) and
`"opponent"` (player 1, fixed C++ bot). Full PettingZoo Parallel API:
`reset()` returns `{agent: obs}`, `step(actions)` returns `(obs, rewards, terms,
truncs, infos)`. The opponent agent gets dummy zero observations; its actions
are ignored (the C++ bot drives it). This enables future multi-agent self-play
(two RL policies co-trained) without changing the engine bridge.

```bash
python rl/halite_pettingzoo.py   # self-contained smoke test
```

For single-agent RL with SB3, use either:
- `halite_env.HaliteRaiderEnv` directly (Gymnasium Env, simpler), or
- `halite_pettingzoo.HalitePZEnv` wrapped with `pz_to_gym()` for the first agent

### `train_ppo.py` — PPO training

SB3 PPO with a custom `SmallCNN` feature extractor (3 conv layers → 256-dim
latent). Uses `SubprocVecEnv` for parallel game rollouts.

```bash
# Quick validation
python rl/train_ppo.py --timesteps 3000 --n-envs 2 --turn-limit 100

# Real training on GPU
python rl/train_ppo.py \
    --opponent eco \
    --timesteps 2000000 \
    --n-envs 8 \
    --turn-limit 300 \
    --device cuda \
    --save ./ppo_raider_v1

# Evaluate a saved model later:
# python rl/eval_ppo.py --model ./ppo_raider_v1.zip --episodes 50
```

Device auto-detection: `--device auto` (default) selects cuda > mps > cpu.

**Known limitation (2026-05-27):** The flat spatial action space
`MultiDiscrete([6]*W*H)` is impractical for real training — 1024 independent
heads mostly control empty cells, policy entropy pins at its maximum (~1834 nats),
and throughput on CPU is ~5 fps. The PPO loop is correct and validates, but
convergence needs a better action representation:
- **Masked spatial actor** (sb3-contrib `MaskablePPO`): only count entropy on
  cells we occupy
- **Per-entity policy**: emit one action per ship, not per cell

Both the env and bridge support arbitrary action formats — only the policy
architecture needs changing.

### `smoke_test.py` — bridge validation

Runs one full game with random actions. Validates the engine↔bot↔env plumbing
without torch/SB3.

```bash
python rl/smoke_test.py
# Expected: ~120 turns, obs (11,32,32), reward trends, no crashes
```

## GPU training setup

1. Clone this repo to a GPU machine.
2. Build engine + C++ bot (see Build section above).
3. Install packages:
   ```bash
   pip install -r rl/requirements.txt   # or:
   pip install gymnasium numpy "stable_baselines3[extra]"
   ```
4. Launch training:
   ```bash
   python rl/train_ppo.py --device cuda --n-envs 8 --timesteps 2000000
   ```
5. Model saves to `rl/ppo_raider.zip` with periodic checkpoints.

## Known issues & next steps

### Engine
- **Map size fixed at 32×32.** Other sizes (40/48/56/64) fail silently at
  startup — the restriction is in the engine, not the map generator, and hasn't
  been chased.
- **`Store::get_entity` asserts then dereferences in Release** (assert is no-op
  under NDEBUG). Any new engine code calling `get_entity` with a possibly-stale
  ID must guard with `entities_ref().find(id)` first.

### Metagame
- **No parameter produces a rock-paper-scissors cycle.** Swept 6 engine knobs
  exhaustively: `KILL_HALITE_BONUS_RATIO`, `ATTACK_HALITE_STEAL_RATIO`,
  `OVER_SHIP_TAX_PER_TURN`, `INSPIRED_BONUS_MULTIPLIER`, `ATTACK_HP_SELF_DAMAGE`,
  and capture radius/threshold — `eco>aggro` is invariant because on an open map
  raiding is always a net economic loss vs pure mining. A solid cycle needs a
  genuinely smarter raider (the RL path).

### RL pipeline
- **Policy architecture needs improvement** (see Known limitation above).
- **Throughput** (~60 turns/s raw, ~5 fps with current head) — needs GPU +
  more parallel envs + lighter policy head.
- **No evaluation script yet** — after training, use the smoke test with a
  loaded model or write a dedicated evaluation loop.
- **`rl/ppo_raider.zip` is 33 MB** — gitignored by `*.zip` rule; re-download
  or re-train on each machine.
- **Engine output is sent to `subprocess.DEVNULL`** — useful for headless
  training; for debugging set `capture_output` in the env or run a single
  game manually.

## Running a single game manually

```bash
# eco vs aggro, with replay written to replays/
./game_engine/build/Release/halite \
    --width 32 --height 32 --turn-limit 300 \
    -i replays --results-as-json \
    -c starter_kits/C++/competitive_engine.json \
    "starter_kits/C++/build_cmake/Release/MyBot.exe 0 starter_kits/C++/bot_params_eco.txt" \
    "starter_kits/C++/build_cmake/Release/MyBot.exe 100 starter_kits/C++/bot_params_aggro.txt"
```

Then inspect with `python debug_replay.py replays/replay-*.hlt`.
