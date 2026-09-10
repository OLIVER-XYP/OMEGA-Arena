# Archive - 已过时的训练和测试组件

本目录存放已被证明无效或已被更好方案替代的训练组件。保留用于历史参考。

## 📁 目录结构

### bc_scripts/ - BC训练时代的脚本
迁移自根目录的临时测试脚本，现已被 `rl_ai/` 体系替代：

- `train_champion.sh`, `train_expert.sh` - BC训练启动脚本
- `gate_bc_models.py` - BC模型评估（现用 `rl_ai/gate_eval.py`）
- `test_*.py` - 各种临时测试脚本
- `monitor_training.sh` - 训练监控

**失败原因**：BC（行为克隆）上限仅 42%，无法学习长期策略。已被 PPO 自对弈替代（达到 65%）。

### old_configs/ - 过时的训练配置
迁移自 `starter_kits/C++/`：

- `train_config_behavior.json` - behavior-EXPANDED 参数搜索配置
- `train_config_mixed.json` - 混合对手训练配置
- `train_engine_cap7.json` - v3 引擎实验配置

**失败原因**：Memory 记录显示 "NO static param set beats the iron triangle"。扩展搜索空间后结果更差（avg 39.3% vs 51%）。

### old_train_runs/ - 失败的参数扫描结果
迁移自 `starter_kits/C++/`：

- `train_runs/` - 基础参数扫描（最佳 56.9%，但被 aggro 压制至 31%）
- `train_runs_behavior/` - behavior-EXPANDED 搜索（最差，avg 39.3%）
- `train_runs_mixed/` - 混合对手训练（avg 51%）
- `sweep_killer/` - killer 扫描临时输出

**结论**：静态参数配置无法突破对称 RPS（Rock-Paper-Scissors）上限。任何静态配置对均衡场的期望胜率 ≈ 50%。

## 📊 当前有效方案

根据 Memory 和 `rl_ai/RL_BOT.md`：

1. **RL 训练路径**（有效）：
   - `rl_ai/train_ppo.py` - PPO 自对弈训练
   - `rl_ai/rank_candidates.py` - 真实引擎 gate 选择
   - `rl_ai/inference_bot.py` - 部署
   - **结果**：65% 胜率 vs 完整专家场（BC 仅 42%）

2. **铁三角策略**（原型保留）：
   - `starter_kits/C++/bot_params_eco.txt` - 经济型
   - `starter_kits/C++/bot_params_aggro.txt` - 侵略型
   - `starter_kits/C++/bot_params_control.txt` - 控制型
   - `starter_kits/C++/bot_params_adaptive.txt` - 自适应（当前部署）

## 🗑️ 已删除内容

以下内容因可重建且占用空间大（~680MB）已完全删除：

- `game_engine/build` - 旧的常规构建（432 MB）
- `game_engine/build-modernization-check` - 一次性测试构建（107 MB）
- `game_engine/build_cuda` - CUDA 构建（140 MB）
- `game_engine/build_cuda_ninja` - Ninja 变体（0.3 MB）

**保留**：`game_engine/build_pyenv` - PPO 训练依赖

## 📅 归档日期

2026-07-17

## 🔗 相关 Memory

- `project_train_pipeline_optimum.md` - 证明静态参数无效
- `project_selfplay_ppo.md` - PPO 自对弈突破 BC 上限
- `project_iron_triangle_v6.md` - 铁三角平衡研究
