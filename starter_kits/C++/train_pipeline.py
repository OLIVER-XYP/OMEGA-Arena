#!/usr/bin/env python3
import argparse, hashlib, json, random, statistics, subprocess, time, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "train_config.json"


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


@dataclass
class RewardWeights:
    win_bonus: float = 1200.0
    tie_bonus: float = 250.0
    loss_penalty: float = -900.0
    score_margin_weight: float = 0.20
    score_abs_weight: float = 0.02
    draw_penalty: float = 120.0


@dataclass
class TrialResult:
    score: float
    details: Dict


class TrainingContext:
    def __init__(self, config: Dict):
        self.config = config
        configured_root = config.get("root")
        self.root = Path(configured_root) if configured_root else SCRIPT_DIR.parents[1]
        self.cpp_dir = resolve_path(self.root, config.get("cpp_dir", "starter_kits/C++"))
        self.bot_exe = resolve_path(self.root, config.get("bot_exe", "starter_kits/C++/build/Release/MyBot.exe"))
        self.engine_exe = resolve_path(self.root, config.get("engine_exe", "game_engine/build/Release/halite.exe"))
        engine_config = config.get("engine_config")
        self.engine_config = resolve_path(self.root, engine_config) if engine_config else None
        self.base_params_file = resolve_path(self.root, config.get("base_params_file", "starter_kits/C++/bot_params.txt"))
        self.out_dir = resolve_path(self.root, config.get("out_dir", "starter_kits/C++/train_runs"))
        self.err_dir = self.out_dir / "error_logs"
        self.maps = [int(x) for x in config.get("maps", [32, 40, 48])]
        self.turn_limit = int(config.get("turn_limit", 300))
        self.bools = list(config.get("bools", ["ENABLE_DROPOFF", "ENABLE_ATTACK", "ENABLE_SPAWN_ROI"]))
        self.parameter_groups = config.get("parameter_groups", {})
        self.max_workers = int(config.get("max_workers", 6))
        self.reward_candidates = [RewardWeights(**candidate) for candidate in config.get("reward_candidates", [RewardWeights().__dict__])]
        self.opponents = self._build_opponents()

    def _build_opponents(self) -> List[str]:
        if "opponents" in self.config:
            return [str(x) for x in self.config["opponents"]]
        seeds = self.config.get("opponent_seeds", [54321, 67890, 24680])
        return [f"{self.bot_exe} {seed} {self.base_params_file}" for seed in seeds]


def parse_params(path: Path) -> Dict[str, str]:
    d = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, v = s.split("=", 1)
            value = v.strip().split("#", 1)[0].strip()
            d[k.strip()] = value
    return d


def write_params(path: Path, params: Dict[str, str]) -> None:
    path.write_text("\n".join(f"{k}={v}" for k, v in params.items()) + "\n", encoding="utf-8")


def sample_discrete(lo, hi, step):
    n = int(round((hi - lo) / step))
    i = random.randint(0, max(0, n))
    v = lo + i * step
    return int(round(v)) if isinstance(step, int) else round(v, 6)


def mutate(base: Dict[str, str], groups: List[Dict[str, Tuple[float, float, float]]], bools: List[str], bool_flip_prob=0.0):
    out = dict(base)
    for g in groups:
        for k, values in g.items():
            lo, hi, st = values
            out[k] = str(sample_discrete(lo, hi, st))
    for b in bools:
        if random.random() < bool_flip_prob:
            out[b] = "0" if out.get(b, "1") in ("1", "true", "True", "TRUE") else "1"
    return out


def cleanup_root_error_logs(ctx: TrainingContext):
    ctx.err_dir.mkdir(parents=True, exist_ok=True)
    for p in ctx.root.glob("errorlog-*.log"):
        try:
            p.unlink()
        except Exception:
            pass


def archive_error_logs(ctx: TrainingContext, tag: str) -> List[str]:
    ctx.err_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in ctx.root.glob("errorlog-*.log"):
        t = ctx.err_dir / f"{tag}_{p.name}"
        try:
            p.replace(t)
            moved.append(str(t))
        except Exception:
            pass
    return moved


def parse_engine_json(stdout: str) -> Dict:
    txt = stdout.strip()
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except Exception:
        l, r = txt.find("{"), txt.rfind("}")
        if l != -1 and r != -1 and r > l:
            try:
                return json.loads(txt[l:r + 1])
            except Exception:
                return {}
        return {}


def write_attempt_logs(log_dir: Path, game_id: str, attempt: int, cmd: List[str], rc: int, stdout: str, stderr: str):
    if log_dir is None:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    base = log_dir / f"{game_id}_attempt{attempt}"
    (base.with_suffix(".cmd.txt")).write_text(" ".join(cmd), encoding="utf-8")
    (base.with_suffix(".stdout.txt")).write_text(stdout or "", encoding="utf-8")
    (base.with_suffix(".stderr.txt")).write_text(stderr or "", encoding="utf-8")
    (base.with_suffix(".meta.json")).write_text(json.dumps({"return_code": rc}, indent=2), encoding="utf-8")


def run_one_game(ctx: TrainingContext, params_path: Path, map_size: int, seed_a: int, opponent_cmd: str, retry: int = 2, log_dir: Path = None, game_id: str = "game", swap_positions: bool = False) -> Dict:
    bot_cmd = f"{ctx.bot_exe} {seed_a} {params_path}"
    cmd = [str(ctx.engine_exe), "--width", str(map_size), "--height", str(map_size), "--turn-limit", str(ctx.turn_limit), "--no-replay", "--no-logs", "--results-as-json"]
    if ctx.engine_config:
        cmd.extend(["-c", str(ctx.engine_config)])
    if swap_positions:
        cmd.extend([opponent_cmd, bot_cmd])
    else:
        cmd.extend([bot_cmd, opponent_cmd])
    for attempt in range(retry + 1):
        p = subprocess.run(cmd, cwd=str(ctx.root), capture_output=True, text=True)
        js = parse_engine_json(p.stdout)
        if p.returncode != 0 or not js:
            write_attempt_logs(log_dir, game_id, attempt, cmd, p.returncode, p.stdout, p.stderr)
            if attempt == retry:
                return {"ok": False, "reason": "process_or_json", "rc": p.returncode, "error_logs": archive_error_logs(ctx, f"{game_id}_proc_a{attempt}")}
            continue
        err = js.get("error_logs", {})
        if isinstance(err, dict) and len(err) > 0:
            write_attempt_logs(log_dir, game_id, attempt, cmd, p.returncode, p.stdout, p.stderr)
            archive_error_logs(ctx, f"{game_id}_engine_a{attempt}")
            if attempt == retry:
                return {"ok": False, "reason": "engine_error_logs", "json": js}
            continue
        return {"ok": True, "json": js}
    return {"ok": False, "reason": "unknown"}


def extract_features_for_player(js: Dict, player_idx: int) -> Dict[str, float]:
    s = js.get("stats") or {}
    if not isinstance(s, dict):
        return {"r0": 2.0, "r1": 1.0, "sc0": 0.0, "sc1": 0.0, "m": -1.0, "w": 0.0, "t": 0.0, "l": 1.0}
    me, opp = str(player_idx), ("1" if player_idx == 0 else "0")
    if me in s and opp in s:
        a, b = s.get(me, {}), s.get(opp, {})
        r0, r1 = float(a.get("rank", 2)), float(b.get("rank", 1))
        sc0, sc1 = float(a.get("score", 0.0)), float(b.get("score", 0.0))
        return {"r0": r0, "r1": r1, "sc0": sc0, "sc1": sc1, "m": sc0 - sc1, "w": 1.0 if r0 < r1 else 0.0, "t": 1.0 if abs(r0 - r1) < 1e-9 else 0.0, "l": 1.0 if r0 > r1 else 0.0}
    return {"r0": 2.0, "r1": 1.0, "sc0": 0.0, "sc1": 0.0, "m": -1.0, "w": 0.0, "t": 0.0, "l": 1.0}


def extract_features(js: Dict) -> Dict[str, float]:
    s = js.get("stats") or {}
    if isinstance(s, dict) and "0" in s and "1" in s:
        a, b = s.get("0", {}), s.get("1", {})
        r0, r1 = float(a.get("rank", 2)), float(b.get("rank", 1))
        sc0, sc1 = float(a.get("score", 0.0)), float(b.get("score", 0.0))
        return {"r0": r0, "r1": r1, "sc0": sc0, "sc1": sc1, "m": sc0 - sc1, "w": 1.0 if r0 < r1 else 0.0, "t": 1.0 if abs(r0 - r1) < 1e-9 else 0.0, "l": 1.0 if r0 > r1 else 0.0}
    return {"r0": 2.0, "r1": 1.0, "sc0": 0.0, "sc1": 0.0, "m": -1.0, "w": 0.0, "t": 0.0, "l": 1.0}


def reward(f: Dict[str, float], w: RewardWeights) -> float:
    x = w.win_bonus * f["w"] + w.tie_bonus * f["t"] + w.loss_penalty * f["l"] + w.score_margin_weight * f["m"] + w.score_abs_weight * f["sc0"]
    if f["t"] > 0.5:
        x -= w.draw_penalty
    return x


def evaluate_weights_pilot(ctx: TrainingContext, base_params: Dict[str, str], pilot_games: int) -> Dict:
    pdir = ctx.out_dir / "pilot"
    pdir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / "pilot_params.txt"
    write_params(ppath, base_params)
    rec, fail = [], 0
    for i in range(pilot_games):
        m = random.choice(ctx.maps)
        seed = random.randint(1, 10_000_000)
        opp = random.choice(ctx.opponents)
        g0 = run_one_game(ctx, ppath, m, seed, opp, retry=2, log_dir=pdir / "attempt_logs", game_id=f"pilot_g{i:04d}_a", swap_positions=False)
        g1 = run_one_game(ctx, ppath, m, seed, opp, retry=2, log_dir=pdir / "attempt_logs", game_id=f"pilot_g{i:04d}_b", swap_positions=True)
        if (not g0["ok"]) or (not g1["ok"]):
            fail += 1
            continue
        rec.append(extract_features_for_player(g0["json"], 0))
        rec.append(extract_features_for_player(g1["json"], 1))
    if not rec:
        out = {"chosen_idx": 0, "games": 0, "fail_count": fail}
        (pdir / "pilot_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    scored = []
    for i, cw in enumerate(ctx.reward_candidates):
        vals = [reward(f, cw) for f in rec]
        mv, sv = statistics.mean(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        scored.append((mv + 0.35 * sv, i, mv, sv))
    scored.sort(reverse=True)
    out = {"chosen_idx": scored[0][1], "games": len(rec), "fail_count": fail, "candidates": [{"idx": i, "quality": q, "mean": m, "std": s} for (q, i, m, s) in scored]}
    (pdir / "pilot_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def run_trial(ctx: TrainingContext, base: Dict[str, str], trial_id: int, phase: str, groups, games_per_trial: int, flip: float, rw: RewardWeights, min_valid: int) -> TrialResult:
    tdir = ctx.out_dir / phase / f"trial_{trial_id:04d}"
    tdir.mkdir(parents=True, exist_ok=True)
    params = mutate(base, groups, ctx.bools, bool_flip_prob=flip)
    ppath = tdir / "bot_params.txt"
    write_params(ppath, params)
    vals, games, valid = [], [], 0
    for gi in range(games_per_trial):
        m, opp = random.choice(ctx.maps), random.choice(ctx.opponents)
        seed = random.randint(1, 10_000_000)
        gid = f"{phase}_t{trial_id:04d}_g{gi:04d}"
        g0 = run_one_game(ctx, ppath, m, seed, opp, retry=2, log_dir=tdir / "attempt_logs", game_id=f"{gid}_a", swap_positions=False)
        g1 = run_one_game(ctx, ppath, m, seed, opp, retry=2, log_dir=tdir / "attempt_logs", game_id=f"{gid}_b", swap_positions=True)
        if (not g0["ok"]) or (not g1["ok"]):
            vals.append(rw.loss_penalty * 1.2)
            games.append({"ok": False, "game": gi, "map": m, "reason": "paired_failed", "a_reason": g0.get("reason"), "b_reason": g1.get("reason")})
            continue
        f0 = extract_features_for_player(g0["json"], 0)
        f1 = extract_features_for_player(g1["json"], 1)
        r0, r1 = reward(f0, rw), reward(f1, rw)
        r = 0.5 * (r0 + r1)
        valid += 1
        vals.append(r)
        games.append({"ok": True, "game": gi, "map": m, "reward": r, "reward_a": r0, "reward_b": r1, "rank0_a": f0["r0"], "rank0_b": f1["r0"], "score0_a": f0["sc0"], "score0_b": f1["sc0"], "margin_a": f0["m"], "margin_b": f1["m"]})
    agg = statistics.mean(vals) if vals else -1e9
    if valid < min_valid:
        agg -= 5000.0
    out = {"trial": trial_id, "phase": phase, "aggregate_reward": agg, "valid_games": valid, "required_valid_games": min_valid, "games": games, "params": params}
    (tdir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return TrialResult(score=agg, details=out)


def topk(results: List[TrialResult], k: int) -> List[TrialResult]:
    return sorted(results, key=lambda r: r.score, reverse=True)[:k]


def config_hash(config: Dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def run_phase_parallel(ctx: TrainingContext, phase_name: str, count: int, base_picker, groups, games_per_trial: int, flip: float, rw: RewardWeights, min_valid: int, max_workers: int) -> List[TrialResult]:
    print(f"[{phase_name}] start trials={count} max_workers={max_workers} games_per_trial={games_per_trial}")
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(run_trial, ctx, base_picker(i), i, phase_name, groups, games_per_trial, flip, rw, min_valid): i for i in range(count)}
        done = 0
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                tr = fut.result()
                results.append(tr)
                done += 1
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (count - done) / rate if rate > 0 else 0.0
                print(f"[{phase_name}] trial={tid:04d} done={done}/{count} score={tr.score:.2f} valid={tr.details['valid_games']}/{tr.details['required_valid_games']} rate={rate:.2f}/s eta={eta:.1f}s")
            except Exception as e:
                done += 1
                print(f"[{phase_name}] trial={tid:04d} FAILED exception={e}")
    return results

def evaluate_head_to_head(ctx: TrainingContext, champ_params: Dict[str, str], cand_params: Dict[str, str], rw: RewardWeights, rounds: int, log_dir: Path, tag: str = "gate") -> Dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="halite_gate_") as td:
        tdir = Path(td)
        champ_path = tdir / "champ_params.txt"
        cand_path = tdir / "cand_params.txt"
        write_params(champ_path, champ_params)
        write_params(cand_path, cand_params)

        vals, wins, fails = [], 0, 0
        for i in range(rounds):
            m = random.choice(ctx.maps)
            seed = random.randint(1, 10_000_000)
            opp_champ = f"{ctx.bot_exe} {seed} {champ_path}"
            g0 = run_one_game(ctx, cand_path, m, seed, opp_champ, retry=2, log_dir=log_dir, game_id=f"{tag}_{i:04d}_a", swap_positions=False)
            g1 = run_one_game(ctx, cand_path, m, seed, opp_champ, retry=2, log_dir=log_dir, game_id=f"{tag}_{i:04d}_b", swap_positions=True)
            if (not g0["ok"]) or (not g1["ok"]):
                fails += 1
                continue
            f0 = extract_features_for_player(g0["json"], 0)
            f1 = extract_features_for_player(g1["json"], 1)
            r = 0.5 * (reward(f0, rw) + reward(f1, rw))
            vals.append(r)
            if r > 0:
                wins += 1
        played = len(vals)
        return {
            "played": played,
            "failed": fails,
            "mean_reward": (statistics.mean(vals) if vals else -1e9),
            "win_rate": (wins / played if played > 0 else 0.0)
        }


def run_global_search(ctx: TrainingContext, base: Dict[str, str], rw: RewardWeights, groups: Dict, games: int, max_workers: int, starts: int, iters_per_start: int, topk_per_iter: int, gate_rounds: int, gate_winrate: float, min_valid: int) -> Dict:
    out = {"mode": "global", "starts": []}
    champion = dict(base)
    champion_meta = {"source": "base", "score": -1e9}

    for sidx in range(starts):
        sdir = ctx.out_dir / "global" / f"start_{sidx:03d}"
        sdir.mkdir(parents=True, exist_ok=True)
        seed_base = mutate(base, [groups["important"], groups["medium"]], ctx.bools, bool_flip_prob=0.03)
        start_best = seed_base
        start_hist = []

        for it in range(iters_per_start):
            phase_name = f"g{sidx:03d}_iter{it:03d}"
            results = run_phase_parallel(ctx, phase_name, max(topk_per_iter * 3, 12), lambda _i: start_best, [groups["important"], groups["medium"], groups["minor"]], games, 0.02, rw, min_valid, max_workers)
            local_top = topk(results, topk_per_iter)
            if not local_top:
                continue
            start_best = local_top[0].details["params"]

            gate = evaluate_head_to_head(ctx, champion, start_best, rw, rounds=gate_rounds, log_dir=sdir / "gate_logs", tag=f"iter{it:03d}")
            accepted = gate["played"] >= max(4, gate_rounds // 3) and gate["win_rate"] >= gate_winrate and gate["mean_reward"] > 0
            start_hist.append({"iter": it, "candidate_score": local_top[0].score, "gate": gate, "accepted": accepted})
            if accepted:
                champion = dict(start_best)
                champion_meta = {"source": f"start_{sidx}_iter_{it}", "score": local_top[0].score, "gate": gate}
                write_params(ctx.out_dir / "best_params.txt", champion)

        out["starts"].append({"start": sidx, "history": start_hist})

    out["champion"] = champion_meta
    write_params(ctx.out_dir / "best_params.txt", champion)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--phase1", type=int, default=40)
    ap.add_argument("--phase2", type=int, default=40)
    ap.add_argument("--phase3", type=int, default=24)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--pilot-games", type=int, default=40)
    ap.add_argument("--min-valid-ratio", type=float, default=0.75)
    ap.add_argument("--max-workers", type=int, default=6)
    ap.add_argument("--mode", choices=["standard", "global"], default="standard")
    ap.add_argument("--global-starts", type=int, default=4)
    ap.add_argument("--global-iters", type=int, default=6)
    ap.add_argument("--global-topk", type=int, default=4)
    ap.add_argument("--gate-rounds", type=int, default=32)
    ap.add_argument("--gate-winrate", type=float, default=0.55)
    a = ap.parse_args()

    config = load_json(a.config)
    ctx = TrainingContext(config)
    if not ctx.bot_exe.exists() or not ctx.engine_exe.exists():
        raise SystemExit("Missing executable(s).")
    if ctx.engine_config and not ctx.engine_config.exists():
        raise SystemExit("Missing engine config.")
    max_workers = max(1, int(a.max_workers))
    print(f"[setup] bot={ctx.bot_exe} engine={ctx.engine_exe} engine_cfg={ctx.engine_config} max_workers={max_workers}")
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    cleanup_root_error_logs(ctx)
    base = parse_params(ctx.base_params_file)

    pilot = evaluate_weights_pilot(ctx, base, pilot_games=a.pilot_games)
    rw = ctx.reward_candidates[int(pilot.get("chosen_idx", 0))]
    min_valid = max(1, int(round(a.games * a.min_valid_ratio)))
    groups = ctx.parameter_groups

    summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(a.config),
        "config_hash": config_hash(config),
        "engine_exe": str(ctx.engine_exe),
        "engine_config": str(ctx.engine_config) if ctx.engine_config else None,
        "bot_exe": str(ctx.bot_exe),
        "max_workers": max_workers,
        "pilot": pilot,
        "reward_weights": rw.__dict__,
        "min_valid_games": min_valid,
        "phases": {},
    }

    if a.mode == "global":
        global_summary = run_global_search(
            ctx=ctx,
            base=base,
            rw=rw,
            groups=groups,
            games=a.games,
            max_workers=max_workers,
            starts=max(1, a.global_starts),
            iters_per_start=max(1, a.global_iters),
            topk_per_iter=max(1, a.global_topk),
            gate_rounds=max(8, a.gate_rounds),
            gate_winrate=a.gate_winrate,
            min_valid=min_valid,
        )
        summary["mode"] = "global"
        summary["global"] = global_summary
    else:
        p1 = run_phase_parallel(ctx, "phase1", a.phase1, lambda _i: base, [groups["important"]], a.games, 0.05, rw, min_valid, max_workers)
        b1 = topk(p1, max(4, a.phase1 // 5))
        summary["phases"]["phase1"] = {"best": [x.details for x in b1]}

        p2 = run_phase_parallel(ctx, "phase2", a.phase2, lambda _i: random.choice(b1).details["params"], [groups["important"], groups["medium"]], a.games, 0.03, rw, min_valid, max_workers)
        b2 = topk(p2, max(4, a.phase2 // 5)) if p2 else b1
        summary["phases"]["phase2"] = {"best": [x.details for x in b2]}

        p3 = run_phase_parallel(ctx, "phase3", a.phase3, lambda _i: random.choice(b2).details["params"], [groups["medium"], groups["minor"]], a.games, 0.01, rw, min_valid, max_workers)
        b3 = topk(p3, max(5, a.phase3 // 4)) if p3 else b2
        summary["phases"]["phase3"] = {"best": [x.details for x in b3]}
        write_params(ctx.out_dir / "best_params.txt", b3[0].details["params"])

    (ctx.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training complete. Best params at: {ctx.out_dir / 'best_params.txt'}")
    print(f"Summary at: {ctx.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
