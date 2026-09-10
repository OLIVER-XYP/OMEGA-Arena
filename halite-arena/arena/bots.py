"""Bot descriptors & shell-command construction.

Each bot passed to the engine is a SINGLE shell command string (engine spawns it
via /bin/sh -c). We build those from typed specs. All dynamic filenames are
shlex-quoted (agent-provided names never reach a shell unquoted).
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from . import config as C
from . import paths


@dataclass(frozen=True)
class BotSpec:
    name: str                       # ELO / display key
    kind: str                       # "agent" | "script" | "rl"
    exe: Path | None = None         # agent/script 可执行
    params: Path | None = None      # C++ argv[2] params 文件
    python: Path = C.PYTHON         # rl
    script: Path = C.INFERENCE_BOT  # rl
    checkpoint: Path | None = None  # rl
    device: str = "auto"            # rl
    extra_args: tuple[str, ...] = ()
    tag: str = ""                   # engine -o 短名

    @property
    def short(self) -> str:
        return self.tag or self.name.split(":")[-1]


def _q(p) -> str:
    return shlex.quote(str(p))


def agent_spec(name: str, *, build: Path | None = None, params: Path | None = None) -> BotSpec:
    a = paths.agent_dir(name)
    return BotSpec(
        name=f"agent:{name}", kind="agent",
        exe=build or (a / "build" / "bot"),
        params=params or (a / "params" / "bot_params.txt"),
        tag=name,
    )


def script_spec(name: str, *, exe: Path | None = None) -> BotSpec:
    if name not in C.SCRIPT_PARAMS:
        raise ValueError(f"未知脚本对手 {name!r}; 可选 {list(C.SCRIPT_PARAMS)}")
    return BotSpec(
        name=f"script:{name}", kind="script",
        exe=exe or C.SCRIPT_EXE, params=C.SCRIPT_PARAMS[name],
        tag=name,
    )


def rl_spec(pool_id: str, *, checkpoint: Path | None = None, device: str = "auto",
            extra: tuple[str, ...] = ()) -> BotSpec:
    # tag 用于引擎 -o 覆盖名（局内展示 + agent 识别），尽量保可读但防 -o 过长
    tag = pool_id.replace("perturb_", "p_").replace("distill_", "d_") \
        .replace("hist_", "h_")[:28] or "rl"
    return BotSpec(
        name=f"rl:{pool_id}", kind="rl",
        checkpoint=checkpoint, device=device,
        extra_args=extra,
        tag=tag,
    )


def _bot_exe(spec: BotSpec) -> Path:
    if spec.kind == "agent":
        exe = spec.exe
        if exe is None:
            raise ValueError(f"agent {spec.name} 未编译: 先跑 arena compile")
        if not Path(exe).exists():
            raise FileNotFoundError(f"agent 可执行不存在: {exe}（先 arena compile）")
    elif spec.kind == "script":
        exe = spec.exe or C.SCRIPT_EXE
    return Path(exe)


def build_bot_command(spec: BotSpec, seed: int, *, trace_path: Path | None = None) -> str:
    """Return the single shell command string the engine will run for this bot."""
    if spec.kind in ("agent", "script"):
        exe = _bot_exe(spec)
        parts = [_q(exe), str(seed + 100000)]
        if spec.params is not None:
            parts.append(_q(spec.params))
        return " ".join(parts)
    if spec.kind == "rl":
        ckpt = spec.checkpoint
        if ckpt is None:
            raise ValueError(f"rl bot {spec.name} 缺 checkpoint")
        parts = [_q(spec.python), _q(spec.script),
                 "--checkpoint", _q(ckpt),
                 "--device", spec.device,
                 "--feature-schema", C.FEATURE_SCHEMA]
        if trace_path is not None:
            parts += ["--trace", _q(trace_path)]
        if spec.extra_args:
            parts += [_q(a) for a in spec.extra_args]
        return " ".join(parts)
    raise ValueError(f"未知 bot kind: {spec.kind}")
