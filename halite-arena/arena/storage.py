"""Storage layer: thin read wrappers over runs/, state/, agents/ for CLI + Web.

Everything here is read-only views of what runner/round/elo already persisted.
Both the CLI (show / rating / submit-status) and the future Web UI call into
this instead of poking files themselves. No platform state is written here.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import paths
from . import elo as elo_mod


def agents() -> list[dict]:
    return paths.agents_list()


def agent(name: str) -> dict | None:
    return next((m for m in agents() if m["name"] == name), None)


def ratings() -> dict[str, float]:
    return elo_mod.latest_ratings()


def list_runs(kind: str | None = None, limit: int | None = None) -> list[Path]:
    """Runs newest-first; optionally filtered by kind (practice/match/round)."""
    runs = sorted(paths.runs_root().glob("*/run.json"),
                  key=lambda p: p.parent.name, reverse=True)
    out = []
    for p in runs:
        try:
            run = json.loads(p.read_text(encoding="utf-8"))
            if kind is not None and run.get("kind") != kind:
                continue
            out.append(p.parent)
            if limit and len(out) >= limit:
                break
        except Exception:
            continue
    return out


def read_run(run_dir: Path | str) -> dict | None:
    p = Path(run_dir)
    runf = p / "run.json"
    if not runf.exists():
        return None
    try:
        return json.loads(runf.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_games(run_dir: Path | str) -> list[dict]:
    """List of per-game summaries inside a run (game_NNNNNN/result.json)."""
    p = Path(run_dir)
    out = []
    for g in sorted(p.glob("game_*")):
        resf = g / "result.json"
        if resf.exists():
            try:
                d = json.loads(resf.read_text(encoding="utf-8"))
                d["_dir"] = str(g.relative_to(p))
                out.append(d)
            except Exception:
                pass
    return out


def game_files(game_dir: Path | str) -> dict[str, str]:
    """Names of artifacts inside one game dir (trace/replay/result)."""
    g = Path(game_dir)
    out = {}
    for f in sorted(g.iterdir()):
        if f.is_file() and f.name != "result.json":
            out[f.name] = str(f.name)
    return out


def latest_agent_feedback(name: str) -> dict | None:
    """agents/<name>/last_game/run.json (or None if no practice/round yet)."""
    lg = paths.last_game_dir(name)
    runf = lg / "run.json"
    if not runf.exists():
        return None
    try:
        return json.loads(runf.read_text(encoding="utf-8"))
    except Exception:
        return None


def practice_history(name: str, limit: int = 20) -> list[dict]:
    """All practice runs that featured this agent, newest-first."""
    out = []
    for d in list_runs("practice"):
        run = read_run(d)
        if run and run.get("agent") == name:
            out.append(run)
            if len(out) >= limit:
                break
    return out


def round_history(limit: int = 20) -> list[dict]:
    """state/rounds.jsonl rows, newest-first."""
    p = paths.state_dir() / "rounds.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows[::-1][:limit]


def agent_material_files(name: str) -> dict:
    """The readonly materials an agent can read (rules.md / engine_src name)."""
    a = paths.agent_dir(name)
    out = {"rules_md": (a / "materials" / "rules.md").exists(),
           "engine_src": (a / "materials" / "engine_src").is_dir(),
           "cpp": (a / "materials" / "cpp").is_dir()}
    return out