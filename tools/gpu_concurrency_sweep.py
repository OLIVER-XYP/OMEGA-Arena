#!/usr/bin/env python3
"""Sweep v3 GPU engine concurrency with no replay/log IO."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = ROOT / "game_engine/build_cuda/Release/halite.exe"
DEFAULT_CONFIG = ROOT / "starter_kits/C++/competitive_engine_v2.json"
DEFAULT_BOT = ROOT / "starter_kits/C++/build_cmake/Release/MyBot.exe"
DEFAULT_OUT = ROOT / "rl_ai/runs/gpu_concurrency_sweep.json"

PROFILES = {
    "eco": ROOT / "starter_kits/C++/bot_params_eco.txt",
    "aggro": ROOT / "starter_kits/C++/bot_params_aggro.txt",
    "control": ROOT / "starter_kits/C++/bot_params_control.txt",
    "adaptive": ROOT / "starter_kits/C++/bot_params_adaptive.txt",
}
PROFILE_PAIRS = [
    ("eco", "aggro"),
    ("aggro", "control"),
    ("control", "eco"),
    ("adaptive", "eco"),
    ("adaptive", "aggro"),
    ("adaptive", "control"),
]


@dataclass
class GpuSample:
    util: int = 0
    mem_used: int = 0
    mem_total: int = 0
    temp: int = 0
    power: float = 0.0


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_json(stdout: str) -> dict[str, Any] | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def query_gpu() -> GpuSample | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 5:
        return None
    try:
        return GpuSample(
            util=int(float(parts[0])),
            mem_used=int(float(parts[1])),
            mem_total=int(float(parts[2])),
            temp=int(float(parts[3])),
            power=float(parts[4]),
        )
    except ValueError:
        return None


def monitor_gpu(stop: threading.Event, interval: float, samples: list[GpuSample]) -> None:
    while not stop.is_set():
        sample = query_gpu()
        if sample is not None:
            samples.append(sample)
        stop.wait(interval)


def run_game(args: argparse.Namespace, job_index: int, concurrency: int) -> dict[str, Any]:
    pair = PROFILE_PAIRS[job_index % len(PROFILE_PAIRS)]
    seed = args.seed_base + concurrency * 100000 + job_index
    bot0 = f"{args.bot} {seed + 17} {PROFILES[pair[0]]}"
    bot1 = f"{args.bot} {seed + 31} {PROFILES[pair[1]]}"
    cmd = [
        str(args.engine),
        "--width",
        "32",
        "--height",
        "32",
        "-s",
        str(seed),
        "--turn-limit",
        str(args.turn_limit),
        "--no-replay",
        "--no-logs",
        "--no-timeout",
        "--results-as-json",
        "--engine-backend",
        args.backend,
        "-c",
        str(args.config),
        bot0,
        bot1,
    ]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "job_index": job_index,
            "seed": seed,
            "profiles": pair,
            "error": f"timeout {exc.timeout}s",
            "elapsed_s": time.perf_counter() - start,
        }
    js = parse_json(proc.stdout)
    elapsed = time.perf_counter() - start
    if js is None:
        return {
            "ok": False,
            "job_index": job_index,
            "seed": seed,
            "profiles": pair,
            "rc": proc.returncode,
            "stdout_tail": proc.stdout[-1000:],
            "stderr_tail": proc.stderr[-1000:],
            "elapsed_s": elapsed,
        }
    error_logs = js.get("error_logs") or {}
    terminated = js.get("terminated") or {}
    return {
        "ok": not bool(error_logs) and not bool(terminated) and proc.returncode == 0,
        "job_index": job_index,
        "seed": seed,
        "profiles": pair,
        "elapsed_s": elapsed,
        "engine_execution_time_ms": js.get("execution_time"),
        "error_logs": error_logs,
        "terminated": terminated,
        "rc": proc.returncode,
    }


def summarize_gpu(samples: list[GpuSample]) -> dict[str, Any]:
    if not samples:
        return {}
    utils = [s.util for s in samples]
    mem = [s.mem_used for s in samples]
    temps = [s.temp for s in samples]
    powers = [s.power for s in samples]
    return {
        "samples": len(samples),
        "max_util": max(utils),
        "mean_util": sum(utils) / len(utils),
        "max_mem_used_mib": max(mem),
        "last_mem_total_mib": samples[-1].mem_total,
        "max_temp_c": max(temps),
        "max_power_w": max(powers),
    }


def run_level(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    jobs = args.games_per_level or concurrency
    stop = threading.Event()
    gpu_samples: list[GpuSample] = []
    monitor = threading.Thread(target=monitor_gpu, args=(stop, args.gpu_sample_interval, gpu_samples), daemon=True)
    start = time.perf_counter()
    monitor.start()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_game, args, i, concurrency) for i in range(jobs)]
        for future in as_completed(futures):
            results.append(future.result())
    stop.set()
    monitor.join(timeout=2)
    elapsed = time.perf_counter() - start
    ok = sum(1 for result in results if result.get("ok"))
    failed = len(results) - ok
    total_turns = jobs * args.turn_limit
    return {
        "concurrency": concurrency,
        "jobs": jobs,
        "ok": ok,
        "failed": failed,
        "elapsed_s": elapsed,
        "games_per_second": jobs / elapsed if elapsed > 0 else 0.0,
        "turns_per_second_nominal": total_turns / elapsed if elapsed > 0 else 0.0,
        "gpu": summarize_gpu(gpu_samples),
        "results": sorted(results, key=lambda item: item["job_index"]),
    }


def parse_concurrency_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", type=resolve, default=DEFAULT_ENGINE)
    ap.add_argument("--config", type=resolve, default=DEFAULT_CONFIG)
    ap.add_argument("--bot", type=resolve, default=DEFAULT_BOT)
    ap.add_argument("--backend", choices=["gpu", "cpu", "auto"], default="gpu")
    ap.add_argument("--concurrency", default="1,2,4,8,12,16")
    ap.add_argument("--games-per-level", type=int, default=0, help="0 means equal to concurrency")
    ap.add_argument("--turn-limit", type=int, default=80)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--gpu-sample-interval", type=float, default=0.25)
    ap.add_argument("--stop-on-failure", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", type=resolve, default=DEFAULT_OUT)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    levels = parse_concurrency_list(args.concurrency)
    report = {
        "schema": "halite3_v3_gpu_concurrency_sweep_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "engine": rel(args.engine),
        "config": rel(args.config),
        "bot": rel(args.bot),
        "backend": args.backend,
        "turn_limit": args.turn_limit,
        "levels": [],
    }
    exit_code = 0
    for level in levels:
        item = run_level(args, level)
        report["levels"].append(item)
        gpu = item.get("gpu") or {}
        print(
            f"workers={level:>3} ok={item['ok']}/{item['jobs']} failed={item['failed']} "
            f"elapsed={item['elapsed_s']:.1f}s games/s={item['games_per_second']:.3f} "
            f"turns/s={item['turns_per_second_nominal']:.1f} "
            f"gpu_util(max/mean)={gpu.get('max_util', 0):.0f}/{gpu.get('mean_util', 0):.1f}% "
            f"mem_peak={gpu.get('max_mem_used_mib', 0)}MiB"
        )
        if item["failed"]:
            exit_code = 1
            if args.stop_on_failure:
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {rel(args.out)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
