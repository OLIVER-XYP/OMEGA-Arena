#!/usr/bin/env python3
"""
Tune C++ MyBot parameters with Optuna.
- BO11 per step (first to 6 wins)
- Mixed map sizes (32/40/48)
- Failure case auto logging
"""
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(r"E:\Halite-III")
HALITE_EXE = ROOT / "game_engine" / "build" / "Release" / "halite.exe"
CPP_DIR = ROOT / "starter_kits" / "C++"
BOT_EXE_RELEASE = CPP_DIR / "build" / "Release" / "MyBot.exe"
BOT_EXE_DEBUG = CPP_DIR / "build" / "Debug" / "MyBot.exe"
BOT_EXE = BOT_EXE_RELEASE if BOT_EXE_RELEASE.exists() else BOT_EXE_DEBUG
PARAMS_FILE = CPP_DIR / "bot_params.txt"
RESULTS_FILE = CPP_DIR / "best_params_cpp.json"
FAIL_LOG_DIR = CPP_DIR / "tune_fail_logs"

MAP_SIZES = [32, 40, 48]
TURNS = 300
GAMES_PER_TRIAL = 11
WINS_TO_CLINCH = 6
N_TRIALS = 1000


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "STOP_RATIO": trial.suggest_float("STOP_RATIO", 0.01, 0.10),
        "SPAWN_END_RATIO": trial.suggest_float("SPAWN_END_RATIO", 0.35, 0.75),
        "SAFE_MARGIN": trial.suggest_int("SAFE_MARGIN", 0, 3),
        "END_RUSH_TURNS": trial.suggest_int("END_RUSH_TURNS", 10, 50),
        "RETURN_BUFFER": trial.suggest_int("RETURN_BUFFER", 4, 20),
        "MIN_RETURN_CARGO": trial.suggest_int("MIN_RETURN_CARGO", 80, 400),
        "STRATEGIC_DIST": trial.suggest_int("STRATEGIC_DIST", 3, 14),
        "STRATEGIC_RICH": trial.suggest_int("STRATEGIC_RICH", 80, 700),
        "STRATEGIC_EFF_THRESH": trial.suggest_float("STRATEGIC_EFF_THRESH", 3.0, 40.0),
        "RETURN_CLOSE": trial.suggest_float("RETURN_CLOSE", 0.30, 0.80),
        "RETURN_FAR": trial.suggest_float("RETURN_FAR", 0.65, 0.99),
        "RETURN_DIST": trial.suggest_int("RETURN_DIST", 8, 28),
        "ENABLE_ATTACK": trial.suggest_categorical("ENABLE_ATTACK", [0, 1]),
        "ATTACK_RATIO": trial.suggest_float("ATTACK_RATIO", 1.0, 4.0),
    }


def write_params(params: dict):
    lines = []
    for k, v in params.items():
        if isinstance(v, float):
            lines.append(f"{k}={v:.6f}")
        else:
            lines.append(f"{k}={v}")
    PARAMS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_failure_log(*, trial_num: int, game_idx: int, map_size: int, bot_a: str, bot_b: str,
                      cmd: list[str], proc: subprocess.CompletedProcess, params: dict, reason: str):
    FAIL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = FAIL_LOG_DIR / f"trial{trial_num:04d}_game{game_idx:02d}_{map_size}x{map_size}_{ts}.json"
    payload = {
        "reason": reason,
        "trial": trial_num,
        "game_index": game_idx,
        "map_size": map_size,
        "turn_limit": TURNS,
        "bot_a": bot_a,
        "bot_b": bot_b,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "params": params,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_game(bot_a: str, bot_b: str, map_size: int, trial_num: int, game_idx: int, params: dict) -> tuple[int, int]:
    cmd = [
        str(HALITE_EXE),
        "--width", str(map_size),
        "--height", str(map_size),
        "--turn-limit", str(TURNS),
        "--no-replay",
        "--no-logs",
        "--no-timeout",
        "--results-as-json",
        bot_a,
        bot_b,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, cwd=str(CPP_DIR))

    if proc.returncode != 0:
        _save_failure_log(
            trial_num=trial_num,
            game_idx=game_idx,
            map_size=map_size,
            bot_a=bot_a,
            bot_b=bot_b,
            cmd=cmd,
            proc=proc,
            params=params,
            reason="non_zero_returncode",
        )
        return 0, 1

    try:
        data = json.loads(proc.stdout)
    except Exception:
        _save_failure_log(
            trial_num=trial_num,
            game_idx=game_idx,
            map_size=map_size,
            bot_a=bot_a,
            bot_b=bot_b,
            cmd=cmd,
            proc=proc,
            params=params,
            reason="json_parse_failed",
        )
        return 0, 1

    error_logs = data.get("error_logs", {})
    if error_logs:
        _save_failure_log(
            trial_num=trial_num,
            game_idx=game_idx,
            map_size=map_size,
            bot_a=bot_a,
            bot_b=bot_b,
            cmd=cmd,
            proc=proc,
            params=params,
            reason="engine_error_logs_present",
        )

    terminated = data.get("terminated", {})
    if "0" in terminated:
        _save_failure_log(
            trial_num=trial_num,
            game_idx=game_idx,
            map_size=map_size,
            bot_a=bot_a,
            bot_b=bot_b,
            cmd=cmd,
            proc=proc,
            params=params,
            reason="player0_terminated",
        )
    if "1" in terminated:
        _save_failure_log(
            trial_num=trial_num,
            game_idx=game_idx,
            map_size=map_size,
            bot_a=bot_a,
            bot_b=bot_b,
            cmd=cmd,
            proc=proc,
            params=params,
            reason="player1_terminated",
        )

    stats = data.get("stats", {})
    rank0 = stats.get("0", {}).get("rank", 99)
    rank1 = stats.get("1", {}).get("rank", 99)

    if rank0 < rank1:
        return 1, 0
    if rank1 < rank0:
        return 0, 1

    # rare tie/failure fallback
    return 0, 0


def objective(trial: optuna.Trial) -> float:
    params = suggest_params(trial)
    write_params(params)

    bot_trial = f"{BOT_EXE} 12345 {PARAMS_FILE}"
    bot_base = f"{BOT_EXE} 54321"

    wins, losses, played = 0, 0, 0

    for gi in range(GAMES_PER_TRIAL):
        map_size = random.choice(MAP_SIZES)

        if gi % 2 == 0:
            wa, wb = run_game(bot_trial, bot_base, map_size, trial.number, gi, params)
            wins += wa
            losses += wb
        else:
            wb, wa = run_game(bot_base, bot_trial, map_size, trial.number, gi, params)
            wins += wa
            losses += wb

        played += 1

        # BO11: first to 6 wins can end early
        if wins >= WINS_TO_CLINCH or losses >= WINS_TO_CLINCH:
            break

    score = (wins - losses) / max(played, 1)
    print(
        f"Trial {trial.number:04d} | W/L {wins}/{losses} | played {played} | score {score:+.3f}",
        flush=True,
    )
    return score


if __name__ == "__main__":
    print(f"Halite engine: {HALITE_EXE}")
    print(f"C++ bot exe  : {BOT_EXE}")
    print(f"Map sizes    : {MAP_SIZES}")
    print(f"BO format    : BO{GAMES_PER_TRIAL}, first to {WINS_TO_CLINCH}")
    print(f"Trials       : {N_TRIALS}")

    if not HALITE_EXE.exists():
        raise SystemExit(f"Engine not found: {HALITE_EXE}")
    if not BOT_EXE.exists():
        raise SystemExit(f"Bot executable not found: {BOT_EXE}. Build starter_kits/C++ first.")

    db_path = CPP_DIR / "optuna_cpp_study.db"
    study = optuna.create_study(
        direction="maximize",
        study_name="halite_cpp_bot",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    warm = {
        "STOP_RATIO": 0.04,
        "SPAWN_END_RATIO": 0.42,
        "SAFE_MARGIN": 1,
        "END_RUSH_TURNS": 30,
        "RETURN_BUFFER": 12,
        "MIN_RETURN_CARGO": 220,
        "STRATEGIC_DIST": 5,
        "STRATEGIC_RICH": 550,
        "STRATEGIC_EFF_THRESH": 20.0,
        "RETURN_CLOSE": 0.55,
        "RETURN_FAR": 0.83,
        "RETURN_DIST": 15,
        "ENABLE_ATTACK": 1,
        "ATTACK_RATIO": 1.55,
    }
    study.enqueue_trial(warm)

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best = study.best_params
    RESULTS_FILE.write_text(json.dumps(best, indent=2), encoding="utf-8")
    write_params(best)

    print("\nBest score:", study.best_value)
    print("Best params saved to:", RESULTS_FILE)
    print("bot_params.txt updated:", PARAMS_FILE)
    print("Failure logs dir:", FAIL_LOG_DIR)
