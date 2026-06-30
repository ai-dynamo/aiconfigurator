# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small runtime helpers shared by the vLLM layerwise scheduler and worker."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import select
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SYSTEM_NAME_CACHE: str | None = None
_VLLM_VERSION_CACHE: str | None = None
_VLLM_DEFAULT_MAX_NUM_SEQS_CACHE: dict[int, int | None] = {}


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
    global _SYSTEM_NAME_CACHE
    if _SYSTEM_NAME_CACHE is not None:
        return _SYSTEM_NAME_CACHE
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        _SYSTEM_NAME_CACHE = result.stdout.strip() or "unknown"
        return _SYSTEM_NAME_CACHE
    except Exception:
        _SYSTEM_NAME_CACHE = "unknown"
        return _SYSTEM_NAME_CACHE


def _get_vllm_version() -> str:
    """Query vLLM in a child process so the scheduler never imports it."""
    global _VLLM_VERSION_CACHE
    if _VLLM_VERSION_CACHE is not None:
        return _VLLM_VERSION_CACHE
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
            _VLLM_VERSION_CACHE = result.stdout.strip() or "unknown"
            return _VLLM_VERSION_CACHE
    except Exception:
        pass
    _VLLM_VERSION_CACHE = "unknown"
    return _VLLM_VERSION_CACHE


def _get_vllm_default_max_num_seqs(world_size: int = 1) -> int | None:
    """Return vLLM's hardware-dependent LLM default max_num_seqs if available."""

    world_size = int(world_size)
    if world_size in _VLLM_DEFAULT_MAX_NUM_SEQS_CACHE:
        return _VLLM_DEFAULT_MAX_NUM_SEQS_CACHE[world_size]
    code = f"""
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext
_, default_max_num_seqs = EngineArgs.get_batch_defaults({world_size})
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
            value = int(raw) if raw else None
            _VLLM_DEFAULT_MAX_NUM_SEQS_CACHE[world_size] = value
            return value
    except Exception:
        pass
    _VLLM_DEFAULT_MAX_NUM_SEQS_CACHE[world_size] = None
    return None


_DEPLOYMENT_EFFECTIVE_CONFIG_CACHE: dict[tuple[Any, ...], dict[str, Any] | None] = {}
_DEPLOYMENT_CONFIG_HELPER_SENTINEL = "__AIC_VLLM_EFFECTIVE_CONFIG__"
_DEPLOYMENT_CONFIG_HELPER_TIMEOUT_S = 90.0
_DEPLOYMENT_CONFIG_HELPER: subprocess.Popen[str] | None = None
_DEPLOYMENT_CONFIG_HELPER_ATEXIT_REGISTERED = False
_DEPLOYMENT_CONFIG_HELPER_CODE = r"""
import argparse
import json
import sys
from pathlib import Path

common_dir = Path(sys.argv[1])
sys.path.insert(0, str(common_dir))
from vllm_deployment import VllmDeploymentConfig, build_engine_args, snapshot_effective_config_from_args

try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform
    from vllm_deployment import summarize_vllm_config

    parser_args = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser_args)
except BaseException:
    EngineArgs = None
    current_platform = None
    summarize_vllm_config = None
    parser_args = None

def resolve(payload):
    tokens = build_engine_args(VllmDeploymentConfig(**payload))
    if EngineArgs is not None and parser_args is not None and current_platform is not None:
        try:
            parsed = parser_args.parse_args(tokens)
            engine_args = EngineArgs.from_cli_args(parsed)
            vllm_config = engine_args.create_engine_config()
            current_platform.update_block_size_for_backend(vllm_config)
            return summarize_vllm_config(vllm_config)
        except BaseException:
            pass
    return snapshot_effective_config_from_args(tokens)

for line in sys.stdin:
    try:
        payload = json.loads(line)
        reply = {"ok": True, "summary": resolve(payload)}
    except BaseException as exc:
        reply = {"ok": False, "error": repr(exc)}
    print("__AIC_VLLM_EFFECTIVE_CONFIG__" + json.dumps(reply, sort_keys=True), flush=True)
"""


def _deployment_effective_config_payload(
    *,
    model: str,
    tensor_parallel_size: int,
    max_num_seqs: int | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    normalized_extra_args: tuple[str, ...],
) -> dict[str, Any]:
    snapshot_extra_args = list(normalized_extra_args)
    if int(tensor_parallel_size) > 1 and not any(
        arg == "--nnodes" or arg.startswith("--nnodes=") for arg in snapshot_extra_args
    ):
        snapshot_extra_args.extend(["--nnodes", str(int(tensor_parallel_size))])
    return {
        "model": model,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": None,
        "block_size": None,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": int(tensor_parallel_size),
        "pipeline_parallel_size": None,
        "dtype": None,
        "kv_cache_dtype": None,
        "enforce_eager": False,
        "disable_prefix_caching": False,
        "no_async_scheduling": False,
        "extra_args": tuple(snapshot_extra_args),
    }


def _stop_deployment_config_helper() -> None:
    global _DEPLOYMENT_CONFIG_HELPER
    proc = _DEPLOYMENT_CONFIG_HELPER
    _DEPLOYMENT_CONFIG_HELPER = None
    if proc is None:
        return
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def _start_deployment_config_helper(common_dir: Path) -> subprocess.Popen[str] | None:
    global _DEPLOYMENT_CONFIG_HELPER, _DEPLOYMENT_CONFIG_HELPER_ATEXIT_REGISTERED
    if os.environ.get("AIC_VLLM_DEPLOYMENT_CONFIG_HELPER") == "0":
        return None
    if _DEPLOYMENT_CONFIG_HELPER is not None and _DEPLOYMENT_CONFIG_HELPER.poll() is None:
        return _DEPLOYMENT_CONFIG_HELPER
    _DEPLOYMENT_CONFIG_HELPER = None
    try:
        proc: subprocess.Popen[str] = subprocess.Popen(
            [sys.executable, "-u", "-c", _DEPLOYMENT_CONFIG_HELPER_CODE, str(common_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
    except Exception:
        return None
    _DEPLOYMENT_CONFIG_HELPER = proc
    if not _DEPLOYMENT_CONFIG_HELPER_ATEXIT_REGISTERED:
        atexit.register(_stop_deployment_config_helper)
        _DEPLOYMENT_CONFIG_HELPER_ATEXIT_REGISTERED = True
    return proc


def _query_deployment_config_helper(common_dir: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    proc = _start_deployment_config_helper(common_dir)
    if proc is None or proc.stdin is None or proc.stdout is None:
        return None
    try:
        proc.stdin.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        proc.stdin.flush()
        deadline = time.monotonic() + _DEPLOYMENT_CONFIG_HELPER_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _stop_deployment_config_helper()
                return None
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
            if not ready:
                _stop_deployment_config_helper()
                return None
            line = proc.stdout.readline()
            if not line:
                _stop_deployment_config_helper()
                return None
            line = line.strip()
            if not line.startswith(_DEPLOYMENT_CONFIG_HELPER_SENTINEL):
                continue
            reply = json.loads(line[len(_DEPLOYMENT_CONFIG_HELPER_SENTINEL) :])
            if reply.get("ok"):
                summary = reply.get("summary")
                return summary if isinstance(summary, dict) else None
            return None
    except Exception:
        _stop_deployment_config_helper()
        return None


def _resolve_deployment_effective_config_once(
    common_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    code = """
import argparse
import json
import sys
from pathlib import Path

common_dir = Path(sys.argv[1])
sys.path.insert(0, str(common_dir))
from vllm_deployment import VllmDeploymentConfig, build_engine_args, snapshot_effective_config_from_args

payload = json.loads(sys.argv[2])
tokens = build_engine_args(VllmDeploymentConfig(**payload))
try:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    parser_args = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser_args)
    parsed = parser_args.parse_args(tokens)
    engine_args = EngineArgs.from_cli_args(parsed)
    vllm_config = engine_args.create_engine_config()
    current_platform.update_block_size_for_backend(vllm_config)
    from vllm_deployment import summarize_vllm_config

    summary = summarize_vllm_config(vllm_config)
except BaseException:
    summary = snapshot_effective_config_from_args(tokens)
print(json.dumps(summary, sort_keys=True))
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(common_dir), json.dumps(payload)],
            text=True,
            capture_output=True,
            timeout=int(_DEPLOYMENT_CONFIG_HELPER_TIMEOUT_S),
            check=False,
        )
        if result.returncode == 0:
            raw = result.stdout.strip().splitlines()[-1:] or [""]
            return json.loads(raw[0]) if raw[0] else None
    except Exception:
        pass
    return None


def _get_vllm_deployment_effective_config(
    *,
    model: str,
    tensor_parallel_size: int,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = 0.9,
    extra_args: tuple[str, ...] | list[str] = (),
) -> dict[str, Any] | None:
    """Return vLLM's resolved deployment config summary for the real model."""

    normalized_extra_args = tuple(str(arg) for arg in extra_args)
    cache_key = (
        str(model),
        int(tensor_parallel_size),
        int(max_num_seqs) if max_num_seqs is not None else None,
        int(max_model_len) if max_model_len is not None else None,
        float(gpu_memory_utilization) if gpu_memory_utilization is not None else None,
        normalized_extra_args,
    )
    if cache_key in _DEPLOYMENT_EFFECTIVE_CONFIG_CACHE:
        return _DEPLOYMENT_EFFECTIVE_CONFIG_CACHE[cache_key]

    common_dir = Path(__file__).resolve().parent.parent / "common"
    payload = _deployment_effective_config_payload(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        normalized_extra_args=normalized_extra_args,
    )
    resolved = _query_deployment_config_helper(common_dir, payload)
    if resolved is None:
        resolved = _resolve_deployment_effective_config_once(common_dir, payload)
    if resolved is not None:
        _DEPLOYMENT_EFFECTIVE_CONFIG_CACHE[cache_key] = resolved
    return resolved


def _get_vllm_deployment_max_num_batched_tokens(
    *,
    model: str,
    tensor_parallel_size: int,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = 0.9,
    extra_args: tuple[str, ...] | list[str] = (),
) -> int | None:
    """Return vLLM's resolved deployment max_num_batched_tokens.

    The one-GPU layerwise worker uses a patched TP=1 model to measure one
    target rank, but context scheduling has to match the real target
    deployment. Query vLLM in a child process with the real TP size so the
    scheduler metadata follows the same path as FPM collection without forcing
    the scheduler process to import vLLM.
    """

    summary = _get_vllm_deployment_effective_config(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_args=extra_args,
    )
    if not summary:
        return None
    value = summary.get("scheduler_config.max_num_batched_tokens")
    block_size = summary.get("cache_config.block_size")
    mamba_cache_mode = summary.get("cache_config.mamba_cache_mode")
    if value is not None and mamba_cache_mode == "align" and block_size is not None:
        value = max(int(value), int(block_size))
    return int(value) if value is not None else None


def _get_vllm_deployment_cache_block_size(
    *,
    model: str,
    tensor_parallel_size: int,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float | None = 0.9,
    extra_args: tuple[str, ...] | list[str] = (),
) -> int | None:
    """Return vLLM's resolved cache block size for the real deployment."""

    summary = _get_vllm_deployment_effective_config(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_args=extra_args,
    )
    if not summary:
        return None
    value = summary.get("cache_config.block_size")
    return int(value) if value is not None else None


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
