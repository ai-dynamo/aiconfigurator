# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small runtime helpers shared by the vLLM layerwise scheduler and worker."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _stable_hash(payload: Any, *, n: int = 16) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(raw).hexdigest()[:n]

def _parse_ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]

def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

def _get_system_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"

def _get_vllm_version() -> str:
    """Query vLLM in a child process so the scheduler never imports it."""
    code = "import vllm; print(getattr(vllm, '__version__', 'unknown'))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except Exception:
        pass
    return "unknown"

def _get_vllm_default_max_num_seqs(world_size: int = 1) -> int | None:
    """Return vLLM's hardware-dependent LLM default max_num_seqs if available."""

    code = f"""
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
_, default_max_num_seqs = EngineArgs.get_batch_defaults({int(world_size)})
print(default_max_num_seqs.get(UsageContext.LLM_CLASS))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
            return int(raw) if raw else None
    except Exception:
        pass
    return None

def _infer_default_max_num_seqs_from_system(system: str | None) -> int:
    """Fallback approximation for vLLM's hardware-dependent max_num_seqs."""

    normalized = str(system or "").lower()
    high_memory_markers = (
        "h100",
        "h200",
        "b100",
        "b200",
        "b300",
        "gb200",
        "gb300",
        "mi300",
        "mi325",
        "80gb",
        "96gb",
        "141gb",
        "192gb",
    )
    if "a100" not in normalized and any(marker in normalized for marker in high_memory_markers):
        return 1024
    return 256

def _detect_gpus(gpus_arg: str | None) -> list[str]:
    if gpus_arg:
        return [x.strip() for x in gpus_arg.split(",") if x.strip()]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible not in ("-1", "NoDevFiles"):
        return [x.strip() for x in visible.split(",") if x.strip()]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if gpus:
            return gpus
    except Exception:
        pass
    return ["0"]

def _tail(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])
