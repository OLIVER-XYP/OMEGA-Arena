#!/usr/bin/env python3
"""
用 Optuna 贝叶斯优化自动调参 MyBot_cursor.py 的策略参数。

运行：python tune_bot.py
结果：打印最优参数，保存到 best_params.json，并写入 MyBot_tuned.py
"""
import subprocess
import json
import os
import re
import sys
import tempfile
import shutil
import optuna
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 路径配置 ──────────────────────────────────────────────────────────────────
HALITE_EXE      = Path(r"E:\Halite-III\game_engine\build\Release\halite.exe")
BOT_BASE        = Path(r"E:\Halite-III\starter_kits\Python3\MyBot_cursor.py")
PYTHON_EXE      = sys.executable
RESULTS_FILE    = Path(r"E:\Halite-III\starter_kits\Python3\best_params.json")
TUNED_BOT       = Path(r"E:\Halite-III\starter_kits\Python3\MyBot_tuned.py")

MAP_SIZE        = 32    # 32×32 最小标准图，速度快
TURNS           = 300   # 每局回合数（300 足以分出胜负）
GAMES_PER_TRIAL = 3     # 每组参数跑几局（奇数防平局）
N_TRIALS        = 60    # 总优化次数（约 60×3×5s ≈ 15 分钟）

# ── 基准 bot 路径（固定，调参目标就是击败它）──────────────────────────────────
BASELINE_BOT = BOT_BASE   # 用未调参版自对弈

# ── 参数搜索空间 ──────────────────────────────────────────────────────────────
def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "STOP_RATIO":           trial.suggest_float("STOP_RATIO",           0.01, 0.10),
        "SPAWN_END_RATIO":      trial.suggest_float("SPAWN_END_RATIO",      0.40, 0.75),
        "MIN_RETURN_CARGO":     trial.suggest_int(  "MIN_RETURN_CARGO",     50,   300),
        "RETURN_CLOSE":         trial.suggest_float("RETURN_CLOSE",         0.30, 0.70),
        "RETURN_FAR":           trial.suggest_float("RETURN_FAR",           0.70, 0.99),
        "RETURN_DIST":          trial.suggest_int(  "RETURN_DIST",          8,    24),
        "STRATEGIC_DIST":       trial.suggest_int(  "STRATEGIC_DIST",       4,    14),
        "STRATEGIC_RICH":       trial.suggest_int(  "STRATEGIC_RICH",       80,   600),
        "STRATEGIC_EFF_THRESH": trial.suggest_float("STRATEGIC_EFF_THRESH", 5.0,  40.0),
        "FALLBACK_MIN_HALITE":  trial.suggest_int(  "FALLBACK_MIN_HALITE",  20,   150),
        "ATTACK_RATIO":         trial.suggest_float("ATTACK_RATIO",         1.0,  4.0),
        "END_RUSH_TURNS":       trial.suggest_int(  "END_RUSH_TURNS",       10,   40),
        "RETURN_BUFFER":        trial.suggest_int(  "RETURN_BUFFER",        3,    12),
    }

def make_bot(params: dict, path: Path):
    """将参数注入 bot 脚本并写出。"""
    src = BOT_BASE.read_text(encoding="utf-8")
    for key, val in params.items():
        if isinstance(val, float):
            src = re.sub(
                rf"^({key}\s*=\s*)[^\n]+",
                lambda m, v=val: f"{key} = {v:.5f}",
                src, flags=re.MULTILINE
            )
        else:
            src = re.sub(
                rf"^({key}\s*=\s*)[^\n]+",
                lambda m, v=val: f"{key} = {v}",
                src, flags=re.MULTILINE
            )
    path.write_text(src, encoding="utf-8")

def run_game(bot_cmd_a: str, bot_cmd_b: str) -> tuple:
    """跑一局，返回 (score_player0, score_player1)。"""
    cmd = [
        str(HALITE_EXE),
        "--width",      str(MAP_SIZE),
        "--height",     str(MAP_SIZE),
        "--turn-limit", str(TURNS),
        "--no-replay",
        "--no-logs",
        "--no-timeout",
        "--results-as-json",
        bot_cmd_a,
        bot_cmd_b,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180,
            cwd=str(BOT_BASE.parent)
        )
        data = json.loads(result.stdout)
        s = data.get("stats", {})
        return s.get("0", {}).get("score", 0), s.get("1", {}).get("score", 0)
    except Exception as e:
        return 0, 0

def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)

    # 写出临时 trial bot
    trial_path = BOT_BASE.parent / f"_trial_{trial.number}.py"
    make_bot(params, trial_path)
    bot_cmd_trial    = f"{PYTHON_EXE} {trial_path}"
    bot_cmd_baseline = f"{PYTHON_EXE} {BASELINE_BOT}"

    wins, total_margin = 0, 0
    for game_idx in range(GAMES_PER_TRIAL):
        if game_idx % 2 == 0:
            sa, sb = run_game(bot_cmd_trial, bot_cmd_baseline)
            margin = sa - sb
        else:
            sb, sa = run_game(bot_cmd_baseline, bot_cmd_trial)
            margin = sa - sb
        if margin > 0:
            wins += 1
        total_margin += margin

    trial_path.unlink(missing_ok=True)

    win_rate   = wins / GAMES_PER_TRIAL
    avg_margin = total_margin / GAMES_PER_TRIAL
    score = win_rate * 10 + avg_margin / 1000.0
    print(f"  Trial {trial.number:3d} | wins {wins}/{GAMES_PER_TRIAL} | margin {avg_margin:+.0f} | score {score:.3f}", flush=True)
    return score

if __name__ == "__main__":
    print(f"Halite engine : {HALITE_EXE}", flush=True)
    print(f"Baseline bot  : {BASELINE_BOT}", flush=True)
    print(f"Map           : {MAP_SIZE}x{MAP_SIZE}, {TURNS} turns", flush=True)
    print(f"Trials        : {N_TRIALS}, {GAMES_PER_TRIAL} games each", flush=True)
    print(flush=True)

    db_path = BOT_BASE.parent / "optuna_study.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="halite_v7",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    # 先用当前参数作为起点（热启动）
    current_params = {
        "STOP_RATIO": 0.04, "SPAWN_END_RATIO": 0.55, "MIN_RETURN_CARGO": 150,
        "RETURN_CLOSE": 0.50, "RETURN_FAR": 0.90, "RETURN_DIST": 16,
        "STRATEGIC_DIST": 8, "STRATEGIC_RICH": 200, "STRATEGIC_EFF_THRESH": 18.0,
        "FALLBACK_MIN_HALITE": 50, "ATTACK_RATIO": 1.5,
        "END_RUSH_TURNS": 20, "RETURN_BUFFER": 6,
    }
    study.enqueue_trial(current_params)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    best_val = study.best_value
    print(f"\n{'='*50}", flush=True)
    print(f"Best score: {best_val:.3f}", flush=True)
    print("Best params:", flush=True)
    for k, v in best.items():
        orig = current_params.get(k, "?")
        print(f"  {k:<26} {str(orig):<10} ->  {v}", flush=True)

    RESULTS_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(f"\nParams saved: {RESULTS_FILE}", flush=True)

    make_bot(best, TUNED_BOT)
    print(f"Tuned bot written: {TUNED_BOT}", flush=True)
