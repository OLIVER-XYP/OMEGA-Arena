# OMEGA-Arena

**OptiMization-oriented and Elo-Graded Agent Arena** — a Halite-III-based
对战框架 / 引擎 / **Agent 评估测试**平台。用它来评测任意 Agent（脚本 bot、RL 策略、
蒸馏小网）在真实 C++ 引擎上的表现，并用 ELO 阶梯排名。

老 Halite 内容已归档；这是把 `/mnt/OMEGA-Arena` 整理成可公开分发 monorepo 后的新写照。

## 仓库内容

| 路径 | 说明 |
|---|---|
| `halite-arena/` | **平台本体**（`arena/` 纯 Python 包 + web 视图，无 torch 依赖即可 import） |
| `game_engine/` | Halite III C++ 引擎源码（CPU/GPU、batched runner） |
| `starter_kits/` | 多语言 starter kit（C++ / Python / JS / …）与 bot_params 风格 |
| `libhaliteviz/` | replay 可视化库（Web 回放） |
| `bots/_meta/` | ★ 单一事实索引（bot_index / roster / MANIFEST） |
| `scripts/fetch_weights.py` | 一键从 HF / ModelScope 拉取权重池并校验（**权重不在 git 里**） |
| `website/ tools/ apiserver/ tests/` | 网站源码、工具、API 服务、回归测试 |

> ⚠️ **权重在 HF / ModelScope，不在本仓库。** 283+ 个 RL 策略（≈14 GB）通过
> `python3 scripts/fetch_weights.py` 下载后用 `SHA256` 校验，落盘到 `bots/pool/`。

## 快速开始

```bash
git clone https://github.com/OLIVER-XYP/OMEGA-Arena.git && cd OMEGA-Arena

# 1. 拉取权重池（HF 或 ModelScope 二选一；默认 HF）
python3 scripts/fetch_weights.py                 # 或 --source modelscope

# 2. 编译引擎（或从 Release 下载 prebuilt 到 game_engine/build_pyenv/halite）
cd game_engine && bash pyenv/build.sh            # 参考 game_engine/README.md

# 3. 跑平台
cd halite-arena && export PYTHONPATH=.
python3 -m arena.cli targets                     # 对手池（脚本 + RL roster top）
python3 -m arena.cli practice testbot eco --seeds 1
python3 -m arena.cli verify <run_dir>            # 对局归档校验（只读）
python3 -m arena.web --port 8081                 # Web：主视图 / agent 视图 / replay
```

## 依赖

- `arena/` 是**纯标准库** Python 包（无需 torch）。当训练侧 `rl_ai`（含 torch）存在时
  自动复用；缺失时回落到随包 vendored 的 `arena/_elo.py` / `arena/_rl_compat.py`。
- 引擎二进制在 [GitHub Releases](../../releases) 提供 Linux x86_64 构建
  （`halite-cpu-x86_64` / `halite-batched_runner-cpu-x86_64` / GPU CUDA 构建）。
- Bot 权重在 [HuggingFace `yyn-0120/OMEGA-Arena-bots`](https://huggingface.co/yyn-0120/OMEGA-Arena-bots)
  或 [ModelScope `yyn0120/OMEGA-Arena-bots`](https://www.modelscope.cn/models/yyn0120/OMEGA-Arena-bots)（国内镜像）。
  `scripts/fetch_weights.py --source modelscope` 可切到 ModelScope 下载。

## 目录结构（完整布局）

```
OMEGA-Arena/
├── game_engine/      % 引擎 C++ 实现（CMake + batched runner + GPU）
├── starter_kits/     C++ starter kit + 4 风格 bot_params
├── halite-arena/     ★ 平台本体（arena/ 纯 Python 包 + web/）
├── bots/
│   ├── pool/         ← 权重下载到这里（314 .pt，git 忽略）
│   ├── _meta/        ← ★ 单一事实：bot_index.json / rank_merged374.json / MANIFEST.sha256
│   └── bot_registry.json
├── scripts/          fetch_weights.py 等发布/运维工具
├── website/ tools/ apiserver/ archive/ tests/
└── MIGRATION.md      历史迁移说明（旧路径 /mnt/Halite-III 等）
```

## 验证

```bash
# 池完整性（本地权重都在后运行）
python3 scripts/fetch_weights.py --verify-only     # ✅ 314/314 SHA256 全通过
# 最小自检（无需引擎/权重）
cd halite-arena && export PYTHONPATH=. && python3 -m arena.cli --help
```

## License

MIT（引擎与框架源码沿用 [Halite-III](https://github.com/HaliteChallenge/Halite) 上游版权）；
权重文件版权归各自训练运行所属，公开发布仅供评估与对比研究。