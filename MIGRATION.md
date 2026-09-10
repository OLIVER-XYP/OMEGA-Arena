# OMEGA-Arena — 框架 / 引擎 / Bot 池

> **公开仓库说明**：本文档记录的是本机（`/mnt/...`）的物理布局历史。公开 clone 中，
> 权重不在 git 里 —— 用 `python3 scripts/fetch_weights.py` 从 HuggingFace / ModelScope
> 拉取到 `bots/pool/`；`bots/_meta/bot_index.json` 用的是仓库相对路径 `bots/pool/<file>.pt`。
> `arena/config.py` 里的 `/mnt/Halite-III`（训练侧）路径在 clone 中不存在时会自动回落。

本目录是从 `Halite-III` 拆分出的 **对战框架 + 引擎 + bot 池**。训练侧（脚本/数据/模型）
留在 `/mnt/Halite-III`；两边通过**软链回指**共用同一份物理文件。

## 布局

```
/mnt/OMEGA-Arena/
├── game_engine/     引擎 C++ 实现（build_pyenv / build_gpu 产物）
├── starter_kits/    C++ starter kit + 4 风格 bot_params
├── halite-arena/    ★ 平台本体（arena/ 纯 Python 包 + web/）
├── libhaliteviz/    replay 可视化库（Web 回放）
├── website/ tools/ apiserver/ archive/ tests/
├── bots/
│   ├── pool/                314 个 .pt（374 roster 中磁盘有实体的全部）
│   ├── bot_registry.json    registry 兼容视图（path → pool/）
│   └── _meta/
│       ├── bot_index.json           ★ 单一事实：id → pool 文件
│       ├── rank_merged374.json      roster（374 id）
│       ├── MANIFEST.sha256          314 个 .pt 的校验和
│       └── check_pool_integrity.py  机器校验（id↔文件 无漂移）
└── *_probe/compare/test 脚本（引擎/竞技类）
```

`/mnt/Halite-III/` 保留：`rl_ai/`（训练脚本/数据/模型）+ 训练类探针脚本；
`game_engine → /mnt/OMEGA-Arena/game_engine` 等 4 个**软链回指**，训练脚本的相对
`ROOT` 路径照常解析。

## 入口

```bash
cd halite-arena
export PYTHONPATH=.
python3 -m arena.cli targets             # 对手池（脚本 + RL roster top）
python3 -m arena.cli practice testbot eco --seeds 1
python3 -m arena.cli practice testbot rl:cycle002 --device cuda:0
python3 -m arena.cli verify <run_dir>    # 对局归档校验（只读）
python3 -m arena.web --port 8081         # Web：主视图 / agent 视图 / replay
python3 bots/_meta/check_pool_integrity.py   # 池完整性
```

## Bot 池口径（重要）

- **roster = 374**（`rank_merged374.json` 的 id 列表，含 rank/mean/src）。
- **实体 = 314**（`bots/pool/*.pt`）；**60 个 roster id 无 .pt**（历史清理遗留，已显式记录在
  `bot_index.json.missing_ids`）。`id→文件` 规则确定：`perturb_<sig>_sN → <sig>_sN.pt`，
  其余 `→ <id>.pt`。
- **registry = 106**，是 roster 的子集（`registry ⊆ roster`），其 `path` 已指向 pool。
- 单一事实 = `bots/_meta/bot_index.json`；`check_pool_integrity.py` 断言
  `present+missing=roster`、无孤儿文件、present 文件存在、missing 无文件。

## 验收（可复现）

```bash
# 1) 池完整性（应输出 ✅）
python3 bots/_meta/check_pool_integrity.py
# 2) 训练侧不受影响（仅在本机训练侧存在时；公开 clone 可跳过）
python3 -c "import sys;sys.path.insert(0,'/mnt/Halite-III');import rl_ai.autotrain as a;print(a.ENGINE.exists(),a.CFG.exists())"
# 3) 平台全链路（compile → practice vs 脚本/RL → verify）
cd halite-arena && export PYTHONPATH=.
python3 -m arena.cli compile testbot
python3 -m arena.cli practice testbot eco --seeds 1
python3 -m arena.cli practice testbot rl:cycle002 --seeds 1 --device cuda:0
python3 -m arena.cli verify $(ls -dt runs/* | head -1 | xargs basename)
# 4) 池文件校验和（权重需先 fetch_weights.py 拉到位）
python3 scripts/fetch_weights.py --verify-only
```
