"""Arena CLI entry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config as C
from . import paths
from . import compile as compile_mod
from . import bots
from . import pool
from . import runner
from . import platform


def _resolve_target(tok: str) -> bots.BotSpec:
    """script:X | rl:<id> | agent:<name> | bare (script->agent)."""
    if tok.startswith("script:"):
        return bots.script_spec(tok.split(":", 1)[1])
    if tok.startswith("rl:"):
        return pool.resolve_rl(tok.split(":", 1)[1])
    if tok.startswith("agent:"):
        return bots.agent_spec(tok.split(":", 1)[1])
    # bare: 脚本名 → agent 名 → pool
    if tok in C.SCRIPT_PARAMS:
        return bots.script_spec(tok)
    if (paths.agents_root() / tok).exists():
        return bots.agent_spec(tok)
    return pool.resolve_rl(tok)  # 当作 pool id（resolve 失败会给清晰报错）


def cmd_init(args) -> int:
    a = paths.scaffold_agent(args.name)
    print(f"agent 工作区就绪: {a}")
    print(f"  materials/ 只读套件  |  submit/ 交付  |  params/ 默认参数")
    return 0


def cmd_list_agents(args) -> int:
    for meta in paths.agents_list():
        print(f"{meta['name']:<20} kind={meta.get('kind','?')}  {meta.get('params','')}")
    if not paths.agents_list():
        print("(无 agent，先 `arena init <name>`)")
    return 0


def cmd_compile(args) -> int:
    paths.ensure_agent(args.name)
    res = compile_mod.compile_agent(args.name)
    print(f"compile {args.name}: ok={res.ok}" + (f" ({res.error_kind})" if not res.ok else ""))
    if not res.ok and res.log:
        print("--- g++ 输出(前 40 行) ---")
        print("\n".join(res.log.splitlines()[:40]))
    return 0 if res.ok else 1


def cmd_practice(args) -> int:
    paths.ensure_agent(args.name, scaffold=False)   # 需已 init；不存在则报错
    # 编译 agent 最新 submit
    if not args.no_compile:
        res = compile_mod.compile_agent(args.name)
        print(f"[compile] {args.name}: ok={res.ok}" + (f" ({res.error_kind})" if not res.ok else ""))
        if not res.ok:
            if res.log:
                print("--- g++ 输出(前 30 行) ---")
                print("\n".join(res.log.splitlines()[:30]))
            platform.write_compile_feedback(args.name, res)
            return 1
    targets = [_resolve_target(t) for t in args.target]
    print(f"[practice] {args.name} vs {[t.short for t in targets]} "
          f"(seeds={args.seeds}, workers={args.workers})", flush=True)
    try:
        run_id = runner.run_practice(args.name, targets, seeds=args.seeds,
                                     workers=args.workers, save_replay=not args.no_replay,
                                     device=args.device, wall_timeout=args.wall_timeout,
                                     engine_no_timeout=args.engine_no_timeout, tag=args.tag)
    except FileNotFoundError as e:
        print(f"目标解析失败: {e}")
        return 2
    platform.write_run_feedback(args.name, run_id)
    platform.print_run_summary(run_id)
    return 0


def cmd_targets(args) -> int:
    print("脚本对手:", ", ".join(f"script:{k}" for k in C.SCRIPT_PARAMS))
    print(f"最强 RL:  rl:{C.FROZEN_TOP.stem}")
    if C.POOL_ROSTER.exists():
        roster = json.loads(C.POOL_ROSTER.read_text(encoding="utf-8"))["ranking"]
        print(f"374 池 top {args.pool}（按 vs-脚本 mean 降序）:")
        for r in roster[: args.pool]:
            print(f"   rl:{r['id']:<48} mean={r['mean']:.3f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena", description="Halite agent 对战平台")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="建一个 agent 工作区")
    s.add_argument("name")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("list-agents", help="列所有 agent")
    s.set_defaults(func=cmd_list_agents)

    s = sub.add_parser("compile", help="编译某 agent 的 submit/ → build/bot")
    s.add_argument("name")
    s.set_defaults(func=cmd_compile)

    s = sub.add_parser("targets", help="列出可用脚本/RL 对手")
    s.add_argument("--pool", type=int, default=5)
    s.set_defaults(func=cmd_targets)

    s = sub.add_parser("practice", help="随交随测: agent vs 指定目标")
    s.add_argument("name")
    s.add_argument("target", nargs="+", help="script:eco / rl:<id> / agent:<name> / 裸名")
    s.add_argument("--seeds", type=int, default=2)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-replay", action="store_true")
    s.add_argument("--no-compile", action="store_true")
    s.add_argument("--wall-timeout", type=float, default=None)
    s.add_argument("--engine-no-timeout", action="store_true")
    s.add_argument("--tag", default="practice")
    s.set_defaults(func=cmd_practice)

    s = sub.add_parser("match", help="正式赛: 参与者间循环赛 + ELO")
    s.add_argument("participants", nargs="+", help="agent:<name> / script:<x> / rl:<id> / 裸名")
    s.add_argument("--games", type=int, default=11, help="每对局数（双侧配对，需奇数>=3）")
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-replay", action="store_true")
    s.add_argument("--wall-timeout", type=float, default=None)
    s.add_argument("--engine-no-timeout", action="store_true")
    s.add_argument("--tag", default="match")
    s.set_defaults(func=cmd_match)

    s = sub.add_parser("rating", help="显示 ELO 排行")
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(func=cmd_rating)

    s = sub.add_parser("round", help="正式轮: 所有注册 agent × agent + × 脚本锚点")
    s.add_argument("--label", default="round")
    s.add_argument("--games-av", type=int, default=2)
    s.add_argument("--games-pool", type=int, default=2)
    s.add_argument("--workers", type=int, default=2)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-replay", action="store_true")
    s.add_argument("--wall-timeout", type=float, default=None)
    s.add_argument("--engine-no-timeout", action="store_true")
    s.set_defaults(func=cmd_round)

    s = sub.add_parser("show", help="总览 / agent 视图")
    s.add_argument("agent", nargs="?", default=None)
    s.add_argument("--recent", type=int, default=5)
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("rules", help="重生成某 agent 的 rules.md + engine_src")
    s.add_argument("name")
    s.set_defaults(func=cmd_rules)

    s = sub.add_parser("submit-status", help="显示某 agent 提交 vs 编译的新鲜度")
    s.add_argument("name")
    s.set_defaults(func=cmd_submit_status)

    s = sub.add_parser("verify", help="对 run/对局做完整性+自洽性+真实性校验（只读）")
    s.add_argument("target", help="run 目录或 game_NNNNNN 目录")
    s.set_defaults(func=cmd_verify)

    return p


def cmd_match(args) -> int:
    specs = []
    for tok in args.participants:
        spec = _resolve_target(tok)
        if spec.kind == "agent":
            res = compile_mod.compile_agent(spec.name.split(":", 1)[1])
            if not res.ok:
                print(f"[compile] {spec.name}: 失败 ({res.error_kind})")
                return 1
        specs.append(spec)
    print(f"[match] {[s.short for s in specs]} games/pair={args.games}", flush=True)
    run_id, ratings = runner.run_match(specs, games_per_pair=args.games,
                                       workers=args.workers, save_replay=not args.no_replay,
                                       device=args.device, wall_timeout=args.wall_timeout,
                                       engine_no_timeout=args.engine_no_timeout, tag=args.tag)
    run = json.loads((run_id / "run.json").read_text(encoding="utf-8"))
    print(f"run: {run_id}  failed={run.get('failed', 0)}")
    print("--- 配对结果 (games, a_wins-b_wins) ---")
    for a, mp in sorted(run.get("summary", {}).items()):
        for b, s in sorted(mp.items()):
            print(f"  {a:<16} vs {b:<16}  {s['games']:2d}局  {s['a_wins']}-{s['b_wins']}")
    print("--- ELO 变化 ---")
    for k in sorted(set(run["ratings_before"]) | set(run["ratings_after"]),
                    key=lambda x: -run["ratings_after"].get(x, 0)):
        before = run["ratings_before"].get(k)
        after = run["ratings_after"].get(k, 1500)
        arrow = "" if before is None else f"  ({after - before:+.0f})"
        print(f"  {k:<24} {after:7.1f}{arrow}")
    return 0


def cmd_rating(args) -> int:
    from . import elo as elo_mod
    ratings = elo_mod.latest_ratings()
    for i, (k, v) in enumerate(sorted(ratings.items(), key=lambda x: -x[1])[: args.top]):
        print(f"  {i + 1:3d}. {k:<24} {v:7.1f}")
    if not ratings:
        print("(暂无 rating，先跑 match)")
    return 0


def cmd_round(args) -> int:
    from . import round as round_mod
    print(f"[round] {args.label}: 编译并跑全部注册 agent × agent + × 脚本锚点", flush=True)
    run_id, ratings = round_mod.run_round(
        label=args.label, games_av=args.games_av, games_pool=args.games_pool,
        workers=args.workers, save_replay=not args.no_replay, device=args.device,
        wall_timeout=args.wall_timeout, engine_no_timeout=args.engine_no_timeout)
    run = json.loads((run_id / "run.json").read_text(encoding="utf-8"))
    print(f"run: {run_id}  failed={run.get('failed', 0)}")
    print("participants:", run.get("agents"))
    print("anchors(frozen):", run.get("anchors"))
    print("--- 配对结果 (games, a_wins-b_wins) ---")
    for a, mp in sorted(run.get("summary", {}).items()):
        for b, s in sorted(mp.items()):
            print(f"  {a:<16} vs {b:<16}  {s['games']:2d}局  {s['a_wins']}-{s['b_wins']}")
    print("--- ELO 变化 ---")
    for k in sorted(set(run["ratings_before"]) | set(run["ratings_after"]),
                    key=lambda x: -run["ratings_after"].get(x, 0)):
        before = run["ratings_before"].get(k)
        after = run["ratings_after"].get(k, 1500)
        arrow = "" if before is None else f"  ({after - before:+.0f})"
        print(f"  {k:<24} {after:7.1f}{arrow}")
    for n in run.get("agents", []):
        print(f"[round] {n}: last_game 已刷新")
    return 0


def cmd_show(args) -> int:
    """Agent 视图：最近一次反馈 + ELO + 历史 run 摘要。组织者视角总览。"""
    from . import elo as elo_mod
    ratings = elo_mod.latest_ratings()
    if args.agent:
        meta = next((m for m in paths.agents_list() if m["name"] == args.agent), None)
        if meta is None:
            print(f"未知 agent: {args.agent}")
            return 2
        print(f"agent: {args.agent}  kind={meta.get('kind')}")
        lg = paths.last_game_dir(args.agent)
        runf = lg / "run.json"
        if runf.exists():
            run = json.loads(runf.read_text(encoding="utf-8"))
            kind = run.get("kind")
            print(f"最近反馈: {kind} @ {run.get('run_id')} failed={run.get('failed',0)}")
            if kind == "practice":
                for opp, s in run.get("summary", {}).items():
                    print(f"   vs {opp:<12} wr={s.get('win_rate', '?'):<6} "
                          f"margin={s.get('mean_margin')}")
            else:   # match/round: pairwise matrix 视角，汇总 agent 总胜场
                tot = {"w": 0, "l": 0}
                for a, mp in run.get("summary", {}).items():
                    if a == args.agent:
                        for b, s in mp.items():
                            w, l = s.get("a_wins", 0), s.get("b_wins", 0)
                            tot["w"] += w
                            tot["l"] += l
                            print(f"   vs {b:<12} {w}-{l}")
                print(f"   合计: {tot['w']} 胜 {tot['l']} 负")
        else:
            print("（无最近反馈，先跑 practice/round）")
        r = ratings.get(args.agent)
        print(f"ELO: {r:.1f}" if r is not None else "ELO: (未参赛)")
        print(f"工作区: {paths.agent_dir(args.agent)}")
        return 0
    # 总览：agents + ratings + 最近 run
    print("=== agents ===")
    for meta in paths.agents_list():
        r = ratings.get(meta["name"])
        print(f"  {meta['name']:<20} elo={r:7.1f}" if r is not None
              else f"  {meta['name']:<20} elo=(—)")
    runs = sorted((p for p in paths.runs_root().glob("*/run.json")),
                  key=lambda p: p.parent.name, reverse=True)
    print("=== 最近 run ===")
    for p in runs[:args.recent]:
        run = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {p.parent.name}  kind={run.get('kind'):<10} failed={run.get('failed',0):<3} "
              f"agents={run.get('agent') or run.get('participants') or run.get('agents')}")
    return 0


def cmd_rules(args) -> int:
    """Regenerate materials/rules.md + engine_src for an agent (readonly prep)."""
    paths.ensure_agent(args.name, scaffold=False)
    a = paths.refresh_materials(args.name)
    print(f"rules.md 已生成: {a / 'materials' / 'rules.md'}")
    print(f"engine_src 已刷新: {a / 'materials' / 'engine_src'}")
    return 0


def cmd_submit_status(args) -> int:
    import os
    a = paths.agent_dir(args.name)
    if not (a / "agent.json").exists():
        print(f"未知 agent: {args.name}")
        return 2
    submit_mtime = max((os.path.getmtime(p) for p in (a / "submit").rglob("*")
                        if p.is_file()), default=0)
    build = a / "build"
    exe = build / "bot"
    compile_res = build / "compile_result.json"
    if exe.exists():
        import datetime as dt
        print(f"submit/ 最后改动: {dt.datetime.fromtimestamp(submit_mtime).strftime('%H:%M:%S')}")
        print(f"build/bot 最后编译: {dt.datetime.fromtimestamp(exe.stat().st_mtime).strftime('%H:%M:%S')}")
        print(f"需重新编译: {submit_mtime > exe.stat().st_mtime}")
    else:
        print("submit/ 最后改动:", submit_mtime)
        print("build/bot: 不存在（未编译）")
    if compile_res.exists():
        cr = json.loads(compile_res.read_text(encoding="utf-8"))
        print(f"上次编译: ok={cr.get('ok')} kind={cr.get('error_kind')} "
              f"elapsed_ms={cr.get('elapsed_ms')}")
    return 0


def cmd_verify(args) -> int:
    from . import verify as verify_mod
    t = Path(args.target)
    if not t.is_absolute():
        cand = paths.runs_root() / t
        t = cand if cand.exists() else t
    if (t / "result.json").exists() or t.name.startswith("game_"):
        rep = verify_mod.verify_game(t)
        run = verify_mod._read_json(t.parent / "run.json") or {}
        print(verify_mod.format_report(t.parent, run, [rep]))
        return 0 if rep.ok else 1
    run, reps = verify_mod.verify_run(t)
    print(verify_mod.format_report(t, run, reps))
    return 0 if all(r.ok for r in reps) else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
