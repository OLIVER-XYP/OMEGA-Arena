"""Vendored, pure-stdlib stand-ins for the two ``rl_ai.evaluate`` helpers the
arena runner uses. The training-side ``rl_ai`` package (which pulls in torch)
lives outside this repository; these copies keep a standalone clone functional.
When ``rl_ai`` is importable the caller prefers it (see engine_runner.py).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def parse_json(stdout: str):
    """Tolerant JSON parse of engine stdout: full text, else the outer {...}."""
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return None
    return None


def read_trace_counts(path: Path) -> dict[str, int]:
    """Aggregate per-action counts from a JSONL trace file (skips torn lines)."""
    counts: Counter[str] = Counter()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts.update({k: int(v) for k, v in (row.get("action_counts") or {}).items()})
    return dict(counts)
