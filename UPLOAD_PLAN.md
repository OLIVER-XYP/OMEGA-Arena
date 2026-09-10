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

## 决策记录（2026-09-10）

- **仓库保持公开**（不做私有化）。HF / ModelScope 权重继续公开。
- **CUDA 二进制不上 HF**：只放 GitHub Release 的 4 个资产（CPU/GPU × 主程序/批跑器）。
  `build_pyenv/halite`（CPU）是 `engine_runner` 实际使用的引擎，评测不需要 GPU。
- 1v1 集成测试走 **CPU + 脚本对手（eco）**，不引入 torch。

## 测试与验收

`tests/regression/` 分四层（纯 stdlib unittest，详见该目录 README）：

| 层 | 内容 | 用时 |
|---|---|---|
| S0 | CLI 冒烟 + 全模块在无 `rl_ai` 下可导入 + 权重清单一致 | ~1s |
| S1 | ELO/配对/解析/pool 规则的单元与不变量 | ~0.6s |
| S2 | `compile → practice vs eco → verify` 全链路 | ~30s |
| S3 | `match` 循环赛 + ELO 零和/持久化（state 重定向到临时目录） | ~0.3s |

全部 70 条用例通过；无引擎/工具链环境自动跳过 S2/S3（`ARENA_SKIP_INTEGRATION=1` 强制关）。

### 本轮修复的缺陷（测试驱动）

1. `pairing.seed_for` 用加盐 `hash()` → practice 种子每进程不同、不可复现 → 改 CRC32。
2. `_elo.Ladder.pairwise_winner` 返回未定义名 → `NameError` → 返回 `"a"`/`"b"`。
3. `runner._run_cases` 未应用 `Case.side_a` → 同种子双局空转且 match/round 反向记一半胜场
   （`testbot vs eco` 误报 wr=1.000，修复后 0.500）→ 按 side 交换座位并修正 margin 视角。
4. `engine_runner` 忽略 `--device`；`replay_out` 未创建且只搬第一个 replay。
5. `elo.apply_pair_results` 先 open 后 mkdir（新检出即崩）；matches.jsonl 无去重（重跑重复计分）
   → 先建目录 + `run_key` 幂等。
6. `compile` 失败残留旧 `build/bot` 与 `ok=True` 的 `compile_result.json`（假成功）→ 先清产物、失败也落盘结果。
7. `paths.new_run_id` 同秒冲突 → 原子分配 `-N` 后缀。
8. `roundrobin_cases` 静默把偶数局改成 3 → 改为显式报错；`round_cases` 两个种子命名空间重叠 → 分离。
9. `pool.resolve_checkpoint` registry 分支用相对路径（engine_cwd 下解析不到）→ 相对 `OMEGA_ROOT` 解析并优先 `path_omega`；
   `top_pool_ids` 对索引中缺失的 id 默认 `present=True` → 默认不可用。
10. 删除死 fixture `tests/regression/fixtures/smoke_summary.json`（含 Windows 私有路径，无任何测试引用）。
