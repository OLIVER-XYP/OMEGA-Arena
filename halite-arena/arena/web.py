"""Arena Web UI: standard-library HTTP server, zero new dependencies.

Run:  python -m arena.web [--port 8000] [--host 127.0.0.1]

Pages:
  /                       主视图（组织者）：agent 总览 + run 列表 + 对决入口
  /agent/<name>           agent 视图：只读材料 + last_game 反馈 + 触发 practice
  /replay?run=...&game=..  replay 播放（canvas 简化渲染）

JSON APIs:
  GET  /api/agents                [{name, kind, elo, feedback, materials}]
  GET  /api/agent/<name>          workspace + latest feedback digest
  GET  /api/runs?kind=...         recent runs (newest first)
  GET  /api/run/latest            {busy, run_id, run}  (poll during a run)
  GET  /api/run/<id>              run.json + per-game digests
  GET  /api/replay?run=<id>&game=<game_N>  full plain-JSON replay
  GET  /api/targets               可用对手列表（脚本 + 最强 RL + 池 top）
  GET  /materials/<name>/<path>   只读材料文件（rules.md / cpp/... / engine_src/...）
  POST /api/practice              {agent, targets[], seeds}  （异步；轮询 /api/run/latest）
  POST /api/compile               {agent}

Engine runs run on a background thread so the UI stays responsive; /api/run/latest
reports busy state and the finished run id.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

from . import config as C
from . import paths
from . import platform
from . import storage
from . import compile as compile_mod
from . import runner as runner_mod
from . import cli as cli_mod   # _resolve_target 复用


_state = {"busy": False, "last_run_id": None, "lock": threading.Lock()}


def _start_background(fn):
    """Run fn (returns run_id Path | None) on a background thread."""
    with _state["lock"]:
        if _state["busy"]:
            return False
        _state["busy"] = True

    def _worker():
        try:
            rid = fn()
            _state["last_run_id"] = str(rid) if rid is not None else None
        except Exception as e:
            _state["last_run_id"] = f"ERROR: {e}"
        finally:
            _state["busy"] = False

    threading.Thread(target=_worker, daemon=True).start()
    return True


# ----------------------------------------------------------------------------
# API data builders


def _feedback_digest(fb):
    if not fb:
        return None
    kind = fb.get("kind")
    if kind == "practice":
        return {"kind": kind, "run_id": fb.get("run_id"),
                "summary": fb.get("summary", {})}
    return {"kind": kind, "run_id": fb.get("run_id"),
            "participants": fb.get("participants") or fb.get("agents"),
            "summary": fb.get("summary", {}),
            "ratings_after": fb.get("ratings_after")}


def api_agents():
    ratings = storage.ratings()
    out = []
    for m in storage.agents():
        name = m["name"]
        out.append({"name": name, "kind": m.get("kind"),
                    "elo": ratings.get(name),
                    "feedback": _feedback_digest(storage.latest_agent_feedback(name)),
                    "materials": storage.agent_material_files(name)})
    return out


def api_agent(name):
    meta = storage.agent(name)
    if meta is None:
        return None
    return {"name": name, "kind": meta.get("kind"),
            "created_at": meta.get("created_at"),
            "elo": storage.ratings().get(name),
            "feedback": _feedback_digest(storage.latest_agent_feedback(name)),
            "history": storage.practice_history(name, limit=10),
            "materials": storage.agent_material_files(name),
            "workspace": str(paths.agent_dir(name))}


def api_runs(kind, limit):
    out = []
    for d in storage.list_runs(kind, limit=limit):
        run = storage.read_run(d) or {}
        out.append({"id": d.name, "dir": str(d), "kind": run.get("kind"),
                    "label": run.get("label"), "agent": run.get("agent"),
                    "participants": run.get("participants") or run.get("agents"),
                    "failed": run.get("failed", 0),
                    "ratings_after": run.get("ratings_after")})
    return out


def api_run(run_id):
    r = paths.runs_root() / run_id
    run = storage.read_run(r)
    if run is None:
        return None
    run["games"] = storage.run_games(r)
    return run


def api_replay(run_id, game):
    """Return the plain-JSON replay for a game dir inside a run."""
    g = paths.runs_root() / run_id / game
    if not g.is_dir():
        return None
    rp = next(g.glob("replay-*.hlt"), None)
    if rp is None:
        return None
    try:
        return json.loads(rp.read_text(encoding="utf-8"))
    except Exception:
        return None


def api_materials_tree(name: str):
    """List the agent's materials/ file tree (rules.md + cpp/ + engine_src/)."""
    base = paths.agent_dir(name) / "materials"
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.rglob("*")):
        if p.is_file():
            try:
                rel = str(p.relative_to(base))
                out.append({"path": rel, "size": p.stat().st_size})
            except OSError:
                pass
    return out


def api_targets():
    from . import config as C
    scripts = [f"script:{k}" for k in C.SCRIPT_PARAMS]
    rl = [f"rl:{C.FROZEN_TOP.stem}"]
    try:
        import json as _json
        if C.POOL_ROSTER.exists():
            rank = _json.loads(C.POOL_ROSTER.read_text(encoding="utf-8")).get("ranking", [])
            rl += [f"rl:{b['id']}" for b in rank[:20]]
    except Exception:
        pass
    return {"scripts": scripts, "rl": rl,
            "agents": [m["name"] for m in storage.agents()]}


class ArenaHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass   # quiet by default

    # -- helpers ----------------------------------------------------------
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype=None):
        try:
            body = path.read_bytes()
        except OSError:
            self.send_error(404, f"missing {path}")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype or mimetypes.guess_type(path.name)[0]
                         or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)
        try:
            if path in ("/", "/index.html"):
                return self._send_file(C.ARENA_ROOT / "web" / "index.html")
            if path == "/agent.html":
                return self._send_file(C.ARENA_ROOT / "web" / "agent.html")
            if path == "/replay":
                return self._send_file(C.ARENA_ROOT / "web" / "replay.html")
            m = re.match(r"^/web/(.+)$", path)
            if m:
                fp = (C.ARENA_ROOT / "web" / m.group(1)).resolve()
                if str(fp).startswith(str((C.ARENA_ROOT / "web").resolve())):
                    return self._send_file(fp)
                return self._json({"error": "forbidden"}, 403)
            # ---- APIs ----
            if path == "/api/agents":
                return self._json(api_agents())
            m = re.match(r"^/api/agent/([^/]+)$", path)
            if m:
                d = api_agent(m.group(1))
                if d is None:
                    return self._json({"error": "unknown agent"}, 404)
                return self._json(d)
            if path == "/api/runs":
                return self._json(api_runs(q.get("kind", [None])[0],
                                           int(q.get("limit", ["20"])[0])))
            if path == "/api/run/latest":
                rid = _state["last_run_id"]
                is_err = isinstance(rid, str) and rid.startswith("ERROR")
                run = None if (rid is None or is_err) else api_run(rid.split("/")[-1])
                return self._json({"busy": _state["busy"], "run_id": None if is_err else rid,
                                   "run": run, "error": rid[6:] if is_err else None})
            m = re.match(r"^/api/run/([^/]+)$", path)
            if m:
                return self._json(api_run(m.group(1)) or {"error": "unknown run"}, 404)
            m = re.match(r"^/api/replay/([^/]+)/([^/]+)$", path)
            if m:
                d = api_replay(m.group(1), m.group(2))
                if d is None:
                    return self._json({"error": "replay not found"}, 404)
                return self._json(d)
            if path == "/api/targets":
                return self._json(api_targets())
            m = re.match(r"^/api/verify/([^/]+)/([^/]+)$", path)
            if m:
                from . import verify as verify_mod
                t = paths.runs_root() / m.group(1) / m.group(2)
                if not (t / "result.json").exists():
                    return self._json({"error": "game not found"}, 404)
                rep = verify_mod.verify_game(t)
                return self._json({"ok": rep.ok,
                                   "findings": [{"level": f.level, "check": f.check,
                                                 "detail": f.detail} for f in rep.findings],
                                   "info": rep.info})
            m = re.match(r"^/api/materials/([^/]+)$", path)
            if m:
                if not storage.agent(m.group(1)):
                    return self._json({"error": "unknown agent"}, 404)
                return self._json(api_materials_tree(m.group(1)))
            m = re.match(r"^/materials/([^/]+)/(.+)$", path)
            if m:
                agent, rel = m.group(1), unquote(m.group(2))
                base = (paths.agent_dir(agent) / "materials").resolve()
                fp = (base / rel).resolve()
                if not str(fp).startswith(str(base)) or not fp.is_file():
                    return self._json({"error": "not found"}, 404)
                return self._send_file(fp)
            return self._json({"error": f"not found: {path}"}, 404)
        except Exception:
            traceback.print_exc()
            self._json({"error": "internal"}, 500)

    # -- POST ----------------------------------------------------------------
    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except Exception:
            body = {}
        try:
            if u.path == "/api/practice":
                agent, targets = body.get("agent"), body.get("targets") or []
                if not agent or not targets:
                    return self._json({"ok": False, "error": "agent+targets 必填"}, 400)
                specs = []
                for t in targets:
                    spec = cli_mod._resolve_target(t)
                    if spec.kind == "agent" and spec.name.split(":", 1)[1] == agent:
                        return self._json({"ok": False,
                                           "error": "不能与自己打（选 script:/rl: 对手）"}, 400)
                    specs.append(spec)
                seeds = int(body.get("seeds", 2))

                def _run():
                    paths.ensure_agent(agent, scaffold=False)
                    res = compile_mod.compile_agent(agent)
                    if not res.ok:
                        platform.write_compile_feedback(agent, res)
                        raise RuntimeError(f"compile 失败: {res.error_kind}")
                    rid = runner_mod.run_practice(agent, specs, seeds=seeds, workers=2,
                                                  save_replay=True, device="auto")
                    platform.write_run_feedback(agent, rid)
                    return rid

                ok = _start_background(_run)
                return self._json({"ok": ok, "error": None if ok else "已有对局在跑"})
            if u.path == "/api/compile":
                agent = body.get("agent")
                res = compile_mod.compile_agent(agent)
                if not res.ok:
                    platform.write_compile_feedback(agent, res)
                return self._json({"ok": res.ok, "kind": res.error_kind,
                                   "log": (res.log or "")[-2000:]})
            return self._json({"error": f"unknown POST {u.path}"}, 404)
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": str(e)}, 500)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="arena.web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    srv = ThreadingHTTPServer((args.host, args.port), ArenaHandler)
    print(f"arena web  http://{args.host}:{args.port}   (Ctrl-C 停止)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
