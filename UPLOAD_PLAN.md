# 发布说明（PUBLISHING）

本仓库 = `/mnt/OMEGA-Arena` 的**代码 monorepo**。权重（314 个 `.pt`，≈14 GB）**不在 git 里**，
托管在 HuggingFace / ModelScope，用 `scripts/fetch_weights.py` 拉取并 SHA256 校验。

## 分工

| 内容 | 位置 |
|---|---|
| 代码 / 引擎 / 脚本 / 文档 | GitHub `OLIVER-XYP/OMEGA-Arena`（本仓库） |
| 引擎二进制（Linux x86_64） | 本仓库 **Releases** |
| 权重池 314×`.pt`（14 GB） | HuggingFace `yyn-0120/OMEGA-Arena-bots` + ModelScope `yyn0120/OMEGA-Arena-bots` |
| 训练侧 `rl_ai`（38 GB，含 torch） | 不发布（留在 `/mnt/Halite-III`） |

## 为什么权重不进 git

- 单文件虽 ≤46 MB（未超 GitHub 100 MB 硬限），但整池 14 GB 远超仓库 1 GB 推荐上限；
- 权重属于**可再分发的大对象**，用 LFS/对象存储更合适 → 放 HF / ModelScope。

## 权重引用关系

- `bots/_meta/bot_index.json`：**单一事实**，`id → bots/pool/<file>.pt`（仓库相对路径）。
- `bots/bot_registry.json`：兼容视图，`path`/`path_omega` 已归一为 `bots/pool/<file>.pt`。
- 发布后本地把权重放到 `bots/pool/` 即可，路径无需再改；远程副本见 HF/ModelScope。

## 可复现性

```bash
python3 scripts/fetch_weights.py                 # 默认 HF；--source modelscope
python3 scripts/fetch_weights.py --verify-only   # 314/314 SHA256 校验
python3 bots/_meta/check_pool_integrity.py       # id↔文件 无漂移
```

## 独立运行

`arena/` 为纯标准库包。训练侧 `rl_ai` 缺失时自动回落到随包 vendored 实现：
`arena/_elo.py`（ELO Ladder）、`arena/_rl_compat.py`（容错 JSON 解析 / trace 计数）。
