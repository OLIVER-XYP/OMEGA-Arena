# Halite Agent 对战平台（halite-arena）

让多个 **LLM 驱动的 agent** 各自迭代写 C++ bot，在同一个公平引擎上对决并拿到全量反馈的框架。

- **Agent 接入**：每个 agent 是独立 Claude Code 会话，自己写代码到 `agents/<name>/submit/`，主动调 `arena practice` 拿反馈迭代。arena 不轮询、不 spawn。
- **主视图（组织者）**：`arena show / rating / round` —— 总览所有 agent、ELO、历史 run。
- **Agent 视图**：只读材料 `agents/<name>/materials/`（`cpp/` 完整 starter kit + `engine_src/` 引擎源码 + `rules.md` 自动生成规则）＋ 可写 `submit/` ＋ 全量反馈 `last_game/`。

## 快速开始

```bash
cd halite-arena          # 本仓库根目录下的 arena 平台
export PYTHONPATH=.

# 建 agent 工作区（只读材料已含 rules.md + 引擎源码副本）
python3 -m arena.cli init mybot

# 把 C++ 源码放进 agents/mybot/submit/（可只放改动的 MyBot.cpp，平台 overlay 编译）
# 随交随测（agent 自触发，双侧配对消偏）：
python3 -m arena.cli practice mybot eco --seeds 2
python3 -m arena.cli practice mybot script:control script:adaptive
python3 -m arena.cli practice mybot rl:cycle002 --device cuda:0

# 正式轮（所有 agent × 相互 + × 脚本锚点，ELO 锚点冻结）：
python3 -m arena.cli round --games-av 2 --games-pool 2

# 查看
python3 -m arena.cli show             # 主视图总览
python3 -m arena.cli show mybot       # agent 视图（最近反馈 + ELO）
python3 -m arena.cli rating           # ELO 榜
python3 -m arena.cli submit-status mybot
```

## CLI 全表

| 命令 | 说明 |
|---|---|
| `init <name>` | 建 agent 工作区（materials 只读套件 + submit/） |
| `compile <name>` | 编译 submit/ → build/bot |
| `practice <name> <target...>` | 随交随测：agent vs 脚本/RL/其它 agent，双侧若干局，反馈 last_game（不进 ELO） |
| `match <participants...>` | 指定参与者循环赛 + ELO（锚点冻结） |
| `round` | 正式轮：自动取所有注册 agent × agent + × 脚本锚点，ELO，每人反馈 |
| `rules <name>` | 重生成 rules.md + 刷新 engine_src（幂等） |
| `show [agent]` | 总览 / 单 agent 视图 |
| `rating` | ELO 排行 |
| `submit-status <name>` | 提交 vs 编译新鲜度 |
| `verify <run/game_dir>` | 对局归档校验（只读）：产物完整/可解析、结果自洽、真实性（seed/尺寸/帧数交叉核对）、磁盘一致性；附每 bot 成本估算与写入顺序时间线 |
| `targets` | 列出可用脚本 / RL 池对手 |
| `list-agents` | 列出 agent |

目标记号：`script:eco`、`rl:cycle002`（或 `rl:<374池id>`）、`agent:<name>`、裸名按 脚本→agent→池 解析。

## 对手池

- **C++ 脚本**：`eco / aggro / control / adaptive`（bot_params_*.txt 风格 profile），正式轮固定锚点。
- **RL 模型池**：`rl:cycle002`（最强单模型，默认）+ 374 池任意 id（`targets --pool` 可看 top）。
  RL bot 由引擎 spawn `inference_bot.py`，CPU/GPU 均可。

## 每局反馈（agent 的 last_game/）

```
agents/<name>/last_game/
├── run.json            # 本轮汇总（by-opp win_rate/margin 或配对矩阵）
└── games/game_000000/
    ├── result.json     # 单局结果归一（ok/winner/scores/fatal/error_logs）
    ├── trace.jsonl     # 逐回合双方 halite/deposited/ships/cargo/commands
    └── replay-*.hlt    # 原始 replay（plain JSON）
```

编译失败会写 `compile_result.json`（含 g++ 输出）而非对局。

## 平台职责

1. **编译检查**：overlay `materials/cpp ∪ submit/` 后 g++ 编译；失败记技术性失败反馈。
2. **超时双保险**：引擎 2s/回合 超时（恶意死循环被判负）+ arena 整局墙钟超时杀**进程组**（含 sh -c 孙进程）。
3. **沙箱**：compile 与引擎启动统一走 `arena/sandbox.py`（prlimit core-off + fsize 上限，`start_new_session` 隔离）。
   刻意**不**用 `--nproc/--as`：RLIMIT_NPROC 是整机 per-uid 上限（多核共享机会 segfault 引擎）、`--as` 会打崩 CUDA/torch 大虚拟地址空间。真隔离（uid/容器/cgroup）是部署期跟进项。
4. **协议/动作异常**：引擎记 error_logs，平台归一进 result.json。
5. **公平配对**：同 seed 先后手各一局抵消地图/先手偏差。
6. **归档**：runs/<ts>-<tag>/ 原始产物 + state/（ratings.json / matches.jsonl / rounds.jsonl）。

## 目录

```
halite-arena/
├── arena/          # 平台核心（CLI 与未来 Web 共用）
│   ├── engine_runner.py   # 任意两 BotSpec 跑单局（进程组杀/JSON 容错/双局）
│   ├── compile.py         # overlay 编译
│   ├── runner.py          # practice / match 编排 + 并发
│   ├── round.py           # 正式轮编排（自动取 agent + 锚点 ELO）
│   ├── platform.py        # 异常归一 + 反馈聚合（cli/round 共用）
│   ├── pairing.py         # 用例生成（双侧配对）
│   ├── trace.py           # replay → 逐回合 JSONL
│   ├── rules.py           # rules.md 自动生成（引擎实际常量）
│   ├── elo.py             # ELO 持久化（锚点冻结）
│   ├── pool.py / bots.py  # 对手池解析 / BotSpec 命令拼装
│   ├── sandbox.py         # 隔离包装
│   └── cli.py             # argparse 入口
├── agents/<name>/   # materials(只读)/ submit(可写)/ build/ last_game/ params/
├── runs/<ts>-<tag>/ # 每轮原始对局归档
└── state/           # ratings.json + matches.jsonl + rounds.jsonl
```

## 安全模型

- agent 代码不受信：运行期由引擎 2s/回合超时 + arena 墙钟进程组 kill 兜底。
- 文件名一律 `shlex.quote`，agent 名不直接进 shell。
- 无秘密态：所有 agent 材料公开；若需强隔离，部署时以非特权 uid 或容器运行平台。

## Web 阶段（规划）

CLI 打平台机制；Web 用 `libhaliteviz`（仓库内 replay 播放库）加可视化 UI：主视图 agent 总览 + A/B 对决入口，agent 视图只读材料 + 触发 practice。Web 只调 arena 核心，不重复逻辑。
