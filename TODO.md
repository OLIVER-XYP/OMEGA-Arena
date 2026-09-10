## 1. 测试部分：先测「1v1 对战 + verify」的建议

你的目标（先跑**一对一（单对局/单配对）**再往上扩）符合工程避险直觉，我建议按 **测试金字塔+回归门槛** 布局，且给每层设“最小成本门槛”：

| 层 | 内容 | 成本 | 放哪 | 该轮优先级 |
|---|---|---|---|---|
| S0 导入/冒烟 | `python3 -m arena.cli --help`、`import arena.*`、`fetch_weights.py --verify-only` | 秒级 | 脚本或 pytest | ★ 先做（已人工验证过，转成自动化） |
| S1 单元 | `_elo`（`Ladder.compute`/分数回归）、`_rl_compat`（`parse_json` 容错/`read_trace_counts` 断行）、`pairing` 双侧配对、`verify` 的归档/写入顺序校验 | 秒级 | `tests/regression/` 已有占地正合适 | ★★ |
| S2 集成：1v1 | 编译 `MyBot`（`testbot`）→ `practice testbot eco --seeds 2` → `verify` 该 run 目录 | 分钟级（CPU） | 本轮重点 | ★★★ |
| S3 端到端 | `match/round` 带 ELO 锚点 + `rl:` 池对手 | 需 `torch`/训练侧 | 私有化后用容器做 | 暂缓 |

你问的两个嵌入问题给方案：

1. “脚本逻辑的测试（如：trace 里检测……）” → 建议把 `engine_runner` 的 JSON 容错（`parse_json`）、`read_trace_counts` 断行容错、`verify` 的真实性交叉核对做成 fixture 驱动的快照测试：把一段手工构造的“坏输出”（截断行/内嵌花括号/脏 JSON）写进 `tests/fixtures/`，让每次改动都过一遍。这是收益/成本比最高的一层。
2. “一前一版”（我按“1v1 版”理解；若你指的是“上一版本的东西”请纠正）：即 `practice/match` 的单配对路径。S2 里先各跑对 2 局（种子配对抵消先后手）。

**关于「把 smoke 执行用在测评结果上」的趋势 —— 我的判断**

方向正确，并且是当前 MLOps/RL 评测的标准姿态。关键在正确分层：

- 两位一体：评估 ≠ 测试。测试验证“引擎/平台行为正确”；评估验证“策略强度”。把两者压在同一个门槛上很容易被玩坏（要么噪音大到你不管，要么种子少到假阳性）。
- 实操三条规则：
  a. seed 固定（同 seed 同 side 对）→ 结果可复现，跑差分才有意义。
  b. 容量上限（smoke 用 `--jobs 4 --seeds 2`）→ 提交时全绿，结果不当“正式 ELO”。
  c. 结果分级：smoke 只判 status（fatal / ok），不判谁强；强弱的量级结论属于 full run，用 `verify/rating` 分开归档。

也就是说：我认同“冒烟只跑评估”的趋势，但建议把冒烟与评估两层的信号区分开 —— 这正是你本仓库 `arena.cli` 天然已经具备的分层（`practice` 随验 vs `round` 正式轮），把它转成 CI 就可以。

---

## 2. 私有化的影响与适配（你要的 checklist）

**直接影响（先告知）**

- GitHub 仓库转 private：代码、README、Releases（含4个二进制）对非协作者不可见；无 Pages/Wiki/Pages/Forks 受影响（目前这些都没开）；没有破坏性，可随时转回公开。
- HF / ModelScope 权重维持 public（这是你上一轮的决定）。所以私有化 ≠ 端到端私有：代码私有，权重仍然任何人可拉。若你要“完全私有化测试”，权重也要转 private/gated（HF 支持 gated，ModelScope 有私有选项）——这是你要决策的点（见第3节）。
- 权限：转到 private 后任何下载需你的授权（token/登录）。

**适配点 checklist（转了之后的清单）**

- [ ] CI 若未来配置：secrets 走 repo Settings，不落地到代码 `.github/` 里（现在仓库没有 `.github/workflows`，所以无历史风险；如果加，按这个规范）。
- [ ] `fetch_weights.py`：不用改 — 权重在 HF 公开，即使 repo private 也仍可拉（除非你权转 gate 后需要 `HF_TOKEN`，脚本已内置该开关）。
- [ ] Release 二进制：私有化后 release 页对非成员拒绝下载，但内部服务器 clone 后仍可从 Release 拉（需授权）；如果测试机无法访问 GitHub（无网集群），需在本地留一份 `_staging/release-eng/`（已留，或走内网拷贝）。
- [ ] 重新公开时的排雷：转公开前跑一遍你自己好该命令 `git ls-files | grep -E '\.pt$|_private|/mnt/'` 应为 0（现在是 0，保持）。
- [ ] tag/历史：已重写成单提交，公开转门无历史包袱。

**回答你“私有化状态下发布二进制会不会冲突”**：不会冲突，反而是一致行为。Release 的可见性跟随仓库可见性；私有仓库 + Release 完全合法，授权协作者能下载。唯一注意点：若未来转公开，该二进制会自动公开（不需要额外操作，也没有 'leak' 风险，因为你本就是 public 目标）。所以放心。

---

## 3. CUDA 二进制要不要连 HF？两个版本的适配

**结论先说：不建议把 CUDA 二进制上传到 HF Hub；建议只上 GitHub Release（已做），并且二进制本来就有 CPU/GPU 两份，分开打标签就够。**
理由基于本次工作流对引擎资产的实测：

- `build_pyenv/halite`（4.1MB，静态链接 CPU 版）= `engine_runner.py` 实际使用的默认引擎（`C.ENGINE`）。你的 1v1 测试跑它不需要任何 GPU 依赖。
- `build_gpu/halite`（2.4MB，动态链接需 `libcudart.so.13`）= 目前没有任何 caller 使用（backend 默认 cpu）。GPU 版主要价值在大批量自对弈/训练挂载，对“评测跑分”不是必需。

**「CUDA vs 非 CUDA」的区别（一句话解释给你）：**

- 非 CUDA（CPU）：编译进引擎本体，零运行时依赖，任何机器可跑；约等于“默认沙箱引擎实验”，秒标配。
- CUDA（GPU）：依赖宿主机驱动+运行时，不规则拟合、但大场次跑得快；是与“评测代码无关”的性能选项。

**适配建议：**

1. 当前阶段只保留 GitHub Release 的4个二进制即可，不必把 GPU 版又发到 HF（会制造两份来源、两处对不齐）。
2. 若是通过免 IPA 或集群想用 GPU 版：从 Release 拉下来放到 `game_engine/build_gpu/halite` 即可被 `C.ENGINE` 识别（`config.py` 自动 `_pick`，GPU build 存在时可用）；无需改代码。
3. 私有化下没有任何冲突 — 引擎运行跟 repo 可见性无关；唯一注意是非私有 repo 下，Release download 需要 auth（上面 checklist 已列）。

---

## 4. 决策点（需要你拍板）

5. **私有化的目的**：如果是“测试期间不给外部看到”，那 GitHub repo private + 权重仍公开即可，零负担；如果“端到端全程私有”——那么 HF / ModelScope 权重也要转成 gated/私有，否则任何知道路径的人仍能拿到全部权重。（建议先 private repo + 公开权重，后续再决定权重的门槛。）
6. **1v1 测试运行环境**：纯 CPU（`--device cpu`，无需 GPU，无需装 `torch`）还是带 `torch` 的 RL 对手（`rl:cycle002`）？——决定 S2 里要不要做 `rl:` 池对手的集成。
7. **测试改跑哪个函数**：`tests/regression/` 现有 fixture 是否继续沿用（推荐沿用）还是要按 `pytest` 体系重搭。
8. **GitHub turn-public 的时间点**：当前 Release 已带二进制，若未来要正式开源/上 hackathon 展示，是否需要把 `atleast v0.2.0` 的“正式留档” tag 打在这个私有单 commit 上？（推荐：现在打 tag，避免 rebase 又动。）
9. **Backlog（高级话题，本轮不做，登记在案会）**
   - Spaces / 可视化 demo：用 HF Spaces（或 ModelScope 云空间）部署 arena web（`arena.web` + `libhaliteviz replay`），公开 URL 演示。高级话题，与私有化无冲突（Space 可见性独立于 repo），等表审完再议。
   - 整体访问控制升级：HF gated / ModelScope 私有 + `MODELSCOPE_API_TOKEN` 下发的整套（`fetch_weights.py` 已支持 header）。
   - 训练侧 `rl_ai` 的发布策略（若未来要连同 38G torch 训练侧发布，单独评估许可/匹配/入口；目前不发布判断已定）。
   - CI + 评价门槛：加 GitHub Actions eval-smoke workflow，把第1节的 S0/S2 转成 PR 门禁；`SMOKE_SEED` 固定 rule。
   - Build 可复现性 pin：引擎二进制 build 的 toolchain/commit 记录，防一 epoch 后无法复现。

---

## 5. 附件：本报告结论的依据（本次已产出的工具/工件）

- 工作流 `omega-publish-plan`（`wf_f59d4296-e41`，6 子代理，288k tokens）：完成了权重家族分组、公开目录清单、引擎二进制资产文件、git 现状与风险、runnable 检查；其 5/6 子代理结论直接支撑“CUDA 不上 HF”“私有化列 checklist”。
- 脚本 `scripts/fetch_weights.py`（HF/ModelScope 双源 + SHA256 verify）、`arena/_elo.py`、`arena/_rl_compat.py`（独立可运行依据）。
- 操作页 `UPLOAD_PLAN.md`（发布分工）已经过一轮更新，包括了“私有化影响”和“CUDA/Release”的决策说明。建议本轮结束后我把这篇汇报的要点再合并进该文件并提交一次。