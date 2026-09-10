#!/usr/bin/env python3
"""Fetch the OMEGA-Arena bot pool weights and verify them against MANIFEST.sha256.

The 314 trained policies (~14 GB) are hosted on HuggingFace / ModelScope and are
deliberately kept out of this git repository. This script downloads them into
``bots/pool/`` and verifies every file's SHA256 against ``bots/_meta/MANIFEST.sha256``.

Usage
-----
    python3 scripts/fetch_weights.py                 # default source: huggingface
    python3 scripts/fetch_weights.py --source modelscope
    python3 scripts/fetch_weights.py --jobs 8        # parallel downloads
    python3 scripts/fetch_weights.py --verify-only   # just re-check existing files

Environment
-----------
    HF_TOKEN / MODELSCOPE_API_TOKEN   optional, for gated/private mirrors
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "bots" / "pool"
MANIFEST = ROOT / "bots" / "_meta" / "MANIFEST.sha256"

# --- remote mirrors (weights/ subdir in each repo) ---
SOURCES = {
    "huggingface": {
        "repo": "yyn-0120/OMEGA-Arena-bots",
        "url": "https://huggingface.co/{repo}/resolve/main/weights/{name}",
        "token_env": "HF_TOKEN",
    },
    "modelscope": {
        "repo": "yyn-0120/OMEGA-Arena-bots",
        "url": "https://www.modelscope.cn/models/{repo}/resolve/master/weights/{name}",
        "token_env": "MODELSCOPE_API_TOKEN",
    },
}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest() -> dict[str, str]:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    out: dict[str, str] = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        out[name.strip()] = digest.strip()
    return out


def download(src: dict, name: str, dest: Path) -> None:
    headers = {}
    tok = os.environ.get(src["token_env"])
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    url = src["url"].format(repo=src["repo"], name=name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), default="huggingface")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    src = SOURCES[args.source]
    manifest = read_manifest()
    POOL.mkdir(parents=True, exist_ok=True)

    todo = [n for n in sorted(manifest) if not POOL.joinpath(n).exists()]
    print(f"manifest: {len(manifest)} files | present: {len(manifest) - len(todo)} | to fetch: {len(todo)}")

    if todo and not args.verify_only:
        print(f"source: {args.source} ({src['repo']})")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(download, src, n, POOL / n): n for n in todo}
            done = 0
            for fut in cf.as_completed(futures):
                name = futures[fut]
                try:
                    fut.result()
                    done += 1
                    print(f"  [{done}/{len(todo)}] {name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAILED {name}: {exc}", file=sys.stderr)

    bad = [n for n in manifest if not POOL.joinpath(n).exists() or sha256(POOL / n) != manifest[n]]
    if bad:
        print(f"\n{len(bad)} file(s) missing or checksum-mismatched:", file=sys.stderr)
        for n in bad[:20]:
            print(f"  {n}", file=sys.stderr)
        return 1
    print(f"\nOK: {len(manifest)}/{len(manifest)} weights present and SHA256-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
