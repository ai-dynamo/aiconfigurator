# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side vLLM engine execution and layerwise measurement drivers."""

from __future__ import annotations

import fcntl
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from random_prompt_tokens import (
    RandomPromptTokenConfig,
    load_random_prompt_token_config,
    make_prompt_token_ids,
    sample_prompt_token_ids,
)
from vllm_deployment import find_runtime_vllm_config, make_metadata, summarize_vllm_config, write_metadata

try:
    from .data import DataPoint
    from .datapoint_generator import _max_num_batched_tokens_for_datapoints
    from .engine import _create_llm, _engine_tokens
    from .runtime import _utc_now
    from .scheduler import _is_fatal_cuda_text, _is_oom_text, oom_dominates
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint
    from datapoint_generator import _max_num_batched_tokens_for_datapoints
    from engine import _create_llm, _engine_tokens
    from runtime import _utc_now
    from scheduler import _is_fatal_cuda_text, _is_oom_text, oom_dominates


def _worker_append_event(
    status_path: Path,
    event: str,
    *,
    work_unit_id: str,
    datapoint_id: str | None = None,
    **extra: Any,
) -> None:
    row = {
        "event": event,
        "work_unit_id": work_unit_id,
        "datapoint_id": datapoint_id,
        "ts": _utc_now(),
        **extra,
    }
    row = {k: v for k, v in row.items() if v is not None}
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)

def _dummy_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
):
    import random

    return [
        {"prompt_token_ids": sample_prompt_token_ids(random, input_len, token_config)}
        for _ in range(batch_size)
    ]

def _deterministic_token_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        prompt: dict[str, Any] = {
            "prompt_token_ids": make_prompt_token_ids(
                prompt_token_seed=0,
                token_count=input_len,
                request_index=request_index_offset + request_idx,
                token_config=token_config,
            )
        }
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts

def _deterministic_prefix_suffix_prompts(
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    prefix_request_index_offset: int,
    suffix_request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        prefix = make_prompt_token_ids(
            prompt_token_seed=0,
            token_count=prefix_len,
            request_index=prefix_request_index_offset + request_idx,
            token_config=token_config,
        )
        suffix = make_prompt_token_ids(
            prompt_token_seed=0,
            token_count=suffix_len,
            request_index=suffix_request_index_offset + request_idx,
            token_config=token_config,
        )
        prompt: dict[str, Any] = {"prompt_token_ids": [*prefix, *suffix]}
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts

def _run_generate(
    llm,
    sampling_params,
    *,
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
    request_index_offset: int | None = None,
    cache_salt_prefix: str | None = None,
) -> None:
    if request_index_offset is None and cache_salt_prefix is None:
        prompts = _dummy_prompts(batch_size, input_len, token_config)
    else:
        prompts = _deterministic_token_prompts(
            batch_size,
            input_len,
            token_config,
            request_index_offset=0 if request_index_offset is None else request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
    llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

def _run_generate_prefix_suffix(
    llm,
    sampling_params,
    *,
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    token_config: RandomPromptTokenConfig,
    prefix_request_index_offset: int,
    suffix_request_index_offset: int,
    cache_salt_prefix: str | None = None,
) -> None:
    prompts = _deterministic_prefix_suffix_prompts(
        batch_size,
        prefix_len,
        suffix_len,
        token_config,
        prefix_request_index_offset=prefix_request_index_offset,
        suffix_request_index_offset=suffix_request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )
    llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

def _classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if _is_oom_text(text):
        return "oom"
    if _is_fatal_cuda_text(text):
        return "fatal_cuda"
    return "error"

def _worker_datapoint_id(work_unit_id: str, dp: DataPoint) -> str:
    return dp.datapoint_id(work_unit_id)

def _write_marker_control(
    *,
    active_iterations: str | Iterable[int] = "",
    trigger: str | None = None,
    phase: str | None = None,
    step: int | None = None,
    bs: int | None = None,
    past: int | None = None,
    run: int | None = None,
    **extras: Any,
) -> None:
    raw_path = os.environ.get("LAYERWISE_CONTROL_FILE")
    if not raw_path:
        return
    if isinstance(active_iterations, str):
        active = [int(x) for x in active_iterations.split(",") if x.strip()]
    else:
        active = [int(x) for x in active_iterations]
    payload = {
        "active_iterations": active,
        "trigger": trigger,
        "phase": phase,
        "step": step,
        "bs": bs,
        "past": past,
        "run": run,
    }
    payload.update(extras)
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
    os.replace(tmp, path)

def _set_marker_state(
    marker_mod,
    *,
    active_iterations: str | Iterable[int] = "",
    trigger: str | None = None,
    phase: str | None = None,
    step: int | None = None,
    bs: int | None = None,
    past: int | None = None,
    run: int | None = None,
    **extras: Any,
) -> None:
    if isinstance(active_iterations, str):
        active_text = active_iterations
    else:
        active_text = ",".join(str(int(x)) for x in active_iterations)
    os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = active_text
    if phase is not None:
        os.environ["LAYERWISE_PROGRESS_PHASE"] = phase
    else:
        phase = os.environ.get("LAYERWISE_PROGRESS_PHASE")
    if run is None:
        os.environ.pop("LAYERWISE_MEASURE_RUN", None)
    else:
        os.environ["LAYERWISE_MEASURE_RUN"] = str(run)
    if step is None and bs is None and past is None and run is None:
        marker_mod.clear_forced_step_meta()
    else:
        marker_mod.set_forced_step_meta(step=step, bs=bs, past=past, run=run)
    _write_marker_control(
        active_iterations=active_iterations,
        trigger=trigger,
        phase=phase,
        step=step,
        bs=bs,
        past=past,
        run=run,
        **extras,
    )

def _ctx_marker_iteration(
    dp: DataPoint,
    max_num_batched_tokens: int,
) -> int:
    if max_num_batched_tokens < 1:
        raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
    return 1

def run_worker(spec_path: Path) -> None:
    """Execute one worker spec inside an nsys-profiled subprocess."""

    spec = json.loads(spec_path.read_text())
    status_path = Path(spec["status_path"])
    work_unit_id = spec["work_unit_id"]
    datapoints = [DataPoint(**raw) for raw in spec["datapoints"]]
    _worker_append_event(
        status_path,
        "worker_entered",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    # Layerwise simulates TP/EP by patching the model config; the runtime engine
    # itself stays single-process on one physical GPU.
    _worker_append_event(
        status_path,
        "worker_environment_started",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["LAYERWISE_TARGET_LAYERS"] = ",".join(str(x) for x in spec["target_layers"])
    if spec.get("moe_noop"):
        os.environ["LAYERWISE_MOE_NOOP"] = "1"
    else:
        os.environ.pop("LAYERWISE_MOE_NOOP", None)
    max_num_batched_tokens = _max_num_batched_tokens_for_datapoints(
        datapoints,
    )
    iterations = {1}
    iterations.update(
        _ctx_marker_iteration(dp, max_num_batched_tokens)
        for dp in datapoints
        if dp.phase == "ctx"
    )
    iterations.update(dp.past_kv + 1 for dp in datapoints if dp.phase == "gen")
    os.environ["LAYERWISE_STEP_ITERATIONS"] = ",".join(str(x) for x in sorted(iterations))
    os.environ["LAYERWISE_BENCH_MIN_NEW"] = "1"
    os.environ["LAYERWISE_PROGRESS_FILE"] = str(status_path)
    os.environ["LAYERWISE_WORK_UNIT_ID"] = work_unit_id
    os.environ["LAYERWISE_CONTROL_FILE"] = str(
        status_path.parent / "marker_control" / f"{work_unit_id}_a{spec['attempt_id']}.json"
    )
    # Keep the marker installed but inactive during vLLM profile/capture work.
    os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = ""
    _write_marker_control(active_iterations="")
    _worker_append_event(
        status_path,
        "worker_environment_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    sys.path.insert(0, str(_THIS_DIR))
    import multiprocessing as mp

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # These patches install vLLM hooks at import time.  Keep them inside worker
    # mode so scheduler/test imports remain vLLM-free.
    _worker_append_event(
        status_path,
        "worker_imports_started",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    import vllm_layer_skip_patch  # noqa: F401
    import vllm_step_marker
    from vllm import SamplingParams
    _worker_append_event(
        status_path,
        "worker_imports_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )

    _worker_append_event(status_path, "work_unit_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_started", work_unit_id=work_unit_id)
    _worker_append_event(status_path, "engine_args_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    prompt_token_config = load_random_prompt_token_config(spec["model_dir"])
    engine_tokens = _engine_tokens(
        model_dir=spec["model_dir"],
        datapoints=datapoints,
        extra_vllm_args=spec["extra_vllm_args"],
    )
    _worker_append_event(status_path, "engine_args_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_create_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    llm = _create_llm(engine_tokens)
    _worker_append_event(status_path, "engine_create_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_metadata_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    effective_config = None
    runtime_vllm_config = find_runtime_vllm_config(llm)
    has_ctx = any(dp.phase == "ctx" for dp in datapoints)
    has_gen = any(dp.phase == "gen" for dp in datapoints)
    if (has_ctx or has_gen) and runtime_vllm_config is not None:
        cache_config = getattr(runtime_vllm_config, "cache_config", None)
        if getattr(cache_config, "enable_prefix_caching", None) is False:
            raise RuntimeError("prefix-cache ctx/gen driver requires vLLM prefix caching")
    if runtime_vllm_config is not None:
        effective_config = summarize_vllm_config(runtime_vllm_config)
    metadata = make_metadata(
        artifact_kind="layerwise",
        measurement_mode="deployment-parity",
        engine_args=engine_tokens,
        effective_config=effective_config,
        extra={
            "work_unit_id": work_unit_id,
            "attempt_id": spec["attempt_id"],
            "target_layers": spec["target_layers"],
            "moe_noop": bool(spec.get("moe_noop")),
        },
    )
    if spec.get("metadata_path"):
        write_metadata(spec["metadata_path"], metadata)
    _worker_append_event(status_path, "engine_metadata_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(
        status_path,
        "engine_metadata_written",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
        metadata_path=spec.get("metadata_path"),
        vllm_config_hash=metadata["vllm_config_hash"],
    )
    _worker_append_event(status_path, "engine_ready", work_unit_id=work_unit_id)
    cuda_profiler_capture = str(spec.get("nsys_capture", "cuda_profiler_api")) == "cuda_profiler_api"
    try:
        _worker_append_event(status_path, "measurement_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
        _worker_set_cuda_profiler_capture(
            status_path=status_path,
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
            enabled=cuda_profiler_capture,
            action="start",
        )
        try:
            ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
            if ctx_points:
                os.environ["LAYERWISE_PROGRESS_PHASE"] = "ctx"
                os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = "1"
                _worker_run_ctx(
                    status_path,
                    work_unit_id,
                    llm,
                    SamplingParams,
                    vllm_step_marker,
                    ctx_points,
                    prompt_token_config=prompt_token_config,
                    warmup_runs=int(spec.get("ctx_warmup_runs", 0)),
                    measured_runs=int(spec.get("ctx_measured_runs", 1)),
                    max_num_batched_tokens=max_num_batched_tokens,
                )

            gen_points = [dp for dp in datapoints if dp.phase == "gen"]
            if gen_points:
                gen_iterations = sorted({dp.past_kv + 1 for dp in gen_points})
                os.environ["LAYERWISE_PROGRESS_PHASE"] = "gen"
                os.environ["LAYERWISE_ACTIVE_ITERATIONS"] = ",".join(str(x) for x in gen_iterations)
                _worker_run_gen(
                    status_path,
                    work_unit_id,
                    llm,
                    SamplingParams,
                    vllm_step_marker,
                    gen_points,
                    prompt_token_config=prompt_token_config,
                    warmup_runs=int(spec.get("gen_warmup_runs", 0)),
                    measured_runs=int(spec.get("gen_measured_runs", 1)),
                )
        finally:
            _worker_set_cuda_profiler_capture(
                status_path=status_path,
                work_unit_id=work_unit_id,
                attempt_id=spec["attempt_id"],
                enabled=cuda_profiler_capture,
                action="stop",
            )
        _worker_append_event(status_path, "measurement_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    finally:
        _worker_append_event(status_path, "worker_cleanup_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        os.environ.pop("LAYERWISE_ACTIVE_ITERATIONS", None)
        _worker_append_event(status_path, "worker_cleanup_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "work_unit_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])

def _worker_run_ctx(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    marker_mod,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    max_num_batched_tokens: int = 1,
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    pruned: set[str] = set()
    filled_prefixes: set[tuple[int, int]] = set()
    for dp in datapoints:
        dpid = _worker_datapoint_id(work_unit_id, dp)
        if dpid in pruned:
            continue
        marker_iteration = _ctx_marker_iteration(
            dp,
            max_num_batched_tokens,
        )
        _set_marker_state(
            marker_mod,
            active_iterations="",
            phase="ctx",
            step=dp.new_tokens,
            bs=dp.batch_size,
            past=dp.past_kv,
        )
        try:
            prefix_key = (int(dp.batch_size), int(dp.past_kv))
            for run_idx in range(warmup_runs):
                fill_prefix = dp.past_kv > 0 and prefix_key not in filled_prefixes
                _run_prefix_cached_ctx_iteration(
                    llm,
                    sampling_params,
                    dp,
                    work_unit_id=work_unit_id,
                    run_idx=run_idx,
                    warmup=True,
                    fill_prefix=fill_prefix,
                    token_config=prompt_token_config,
                    active_iteration="",
                    marker_mod=marker_mod,
                )
                if fill_prefix:
                    filled_prefixes.add(prefix_key)
            for run_idx in range(measured_runs):
                fill_prefix = dp.past_kv > 0 and prefix_key not in filled_prefixes
                _set_marker_state(
                    marker_mod,
                    active_iterations=str(marker_iteration),
                    phase="ctx",
                    step=dp.new_tokens,
                    bs=dp.batch_size,
                    past=dp.past_kv,
                    run=run_idx,
                )
                _run_prefix_cached_ctx_iteration(
                    llm,
                    sampling_params,
                    dp,
                    work_unit_id=work_unit_id,
                    run_idx=run_idx,
                    warmup=False,
                    fill_prefix=fill_prefix,
                    token_config=prompt_token_config,
                    active_iteration=str(marker_iteration),
                    marker_mod=marker_mod,
                )
                if fill_prefix:
                    filled_prefixes.add(prefix_key)
        except Exception as exc:
            kind = _classify_exception(exc)
            if kind == "oom":
                _worker_append_event(
                    status_path,
                    "failed_oom",
                    work_unit_id=work_unit_id,
                    datapoint_id=dpid,
                    message=str(exc),
                )
                _worker_empty_cache()
                for candidate in datapoints:
                    cid = _worker_datapoint_id(work_unit_id, candidate)
                    if cid != dpid and oom_dominates(dp, candidate):
                        pruned.add(cid)
                        _worker_append_event(
                            status_path,
                            "skipped_oom_dominated",
                            work_unit_id=work_unit_id,
                            datapoint_id=cid,
                            caused_by=dpid,
                        )
                continue
            if kind == "fatal_cuda":
                _worker_append_event(
                    status_path,
                    "failed_fatal_cuda",
                    work_unit_id=work_unit_id,
                    datapoint_id=dpid,
                    message=str(exc),
                )
                raise
            _worker_append_event(
                status_path,
                "failed_error",
                work_unit_id=work_unit_id,
                datapoint_id=dpid,
                message=str(exc),
                traceback=traceback.format_exc()[-4000:],
            )
            _worker_empty_cache()
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="ctx")

def _ctx_prefix_request_index_offset(dp: DataPoint) -> int:
    return int(dp.past_kv) * 1_000_003 + int(dp.batch_size) * 101

def _ctx_suffix_request_index_offset(dp: DataPoint, run_idx: int, *, warmup: bool) -> int:
    phase_offset = 0 if warmup else 100_000_000
    return (
        phase_offset
        + int(dp.past_kv) * 1_000_003
        + int(dp.new_tokens) * 10_007
        + int(dp.batch_size) * 101
        + int(run_idx) * 10_000
    )

def _ctx_cache_salt_prefix(
    work_unit_id: str,
    dp: DataPoint,
) -> str:
    return f"layerwise-ctx:{work_unit_id}:bs{dp.batch_size}:past{dp.past_kv}"

def _run_prefix_cached_ctx_iteration(
    llm,
    sampling_params,
    dp: DataPoint,
    *,
    work_unit_id: str,
    run_idx: int,
    warmup: bool,
    fill_prefix: bool,
    token_config: RandomPromptTokenConfig,
    active_iteration: str,
    marker_mod,
) -> None:
    prefix_request_index_offset = _ctx_prefix_request_index_offset(dp)
    suffix_request_index_offset = _ctx_suffix_request_index_offset(dp, run_idx, warmup=warmup)
    cache_salt_prefix = _ctx_cache_salt_prefix(work_unit_id, dp)
    if fill_prefix and dp.past_kv > 0:
        _set_marker_state(marker_mod, active_iterations="", phase="ctx")
        _run_generate(
            llm,
            sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            request_index_offset=prefix_request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
    _set_marker_state(
        marker_mod,
        active_iterations=active_iteration,
        phase="ctx",
        step=dp.new_tokens,
        bs=dp.batch_size,
        past=dp.past_kv,
        run=run_idx if not warmup else None,
    )
    _run_generate_prefix_suffix(
        llm,
        sampling_params,
        batch_size=dp.batch_size,
        prefix_len=dp.past_kv,
        suffix_len=dp.new_tokens,
        token_config=token_config,
        prefix_request_index_offset=prefix_request_index_offset,
        suffix_request_index_offset=suffix_request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )

def _gen_prompt_request_index_offset(dp: DataPoint) -> int:
    return 0

def _gen_cache_salt_prefix(work_unit_id: str, dp: DataPoint) -> str:
    return f"layerwise-gen:{work_unit_id}"

def _run_prefix_cached_gen_iteration(
    llm,
    fill_sampling_params,
    measure_sampling_params,
    dp: DataPoint,
    *,
    token_config: RandomPromptTokenConfig,
    fill_cache: bool,
) -> None:
    if dp.past_kv <= 0:
        raise ValueError("prefix-cache decode requires past_kv > 0")
    request_index_offset = _gen_prompt_request_index_offset(dp)
    cache_salt_prefix = _gen_cache_salt_prefix(os.environ.get("LAYERWISE_WORK_UNIT_ID", ""), dp)
    if fill_cache:
        _run_generate(
            llm,
            fill_sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            request_index_offset=request_index_offset,
            cache_salt_prefix=cache_salt_prefix,
        )
        return
    _run_generate(
        llm,
        measure_sampling_params,
        batch_size=dp.batch_size,
        input_len=dp.past_kv,
        token_config=token_config,
        request_index_offset=request_index_offset,
        cache_salt_prefix=cache_salt_prefix,
    )

def _worker_run_gen(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    marker_mod,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
    warmup_runs: int = 0,
    measured_runs: int = 1,
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    by_batch: dict[int, list[DataPoint]] = {}
    for dp in datapoints:
        by_batch.setdefault(dp.batch_size, []).append(dp)
    max_past = max(dp.past_kv for dp in datapoints)
    sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=2,
        detokenize=False,
    )
    fill_sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    max_batch_size = max(by_batch)
    fill_dp = DataPoint("gen", max_batch_size, 1, max_past)
    _set_marker_state(marker_mod, active_iterations="", phase="gen")
    _run_prefix_cached_gen_iteration(
        llm,
        fill_sampling_params,
        sampling_params,
        fill_dp,
        token_config=prompt_token_config,
        fill_cache=True,
    )
    for batch_size in sorted(by_batch):
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(
            status_path,
            "batch_started",
            work_unit_id=work_unit_id,
            batch_size=batch_size,
            max_past_kv=max_past,
        )
        try:
            for dp in sorted(by_batch[batch_size], key=lambda item: item.past_kv):
                _set_marker_state(marker_mod, active_iterations="", phase="gen")
                for _ in range(warmup_runs):
                    _set_marker_state(marker_mod, active_iterations="", phase="gen")
                    _run_prefix_cached_gen_iteration(
                        llm,
                        fill_sampling_params,
                        sampling_params,
                        dp,
                        token_config=prompt_token_config,
                        fill_cache=False,
                    )
                for run_idx in range(measured_runs):
                    _set_marker_state(
                        marker_mod,
                        active_iterations="",
                        trigger="decode_only",
                        allow_new_cached=True,
                        phase="gen",
                        step=dp.past_kv + 1,
                        bs=batch_size,
                        past=dp.past_kv,
                        run=run_idx,
                    )
                    _run_prefix_cached_gen_iteration(
                        llm,
                        fill_sampling_params,
                        sampling_params,
                        dp,
                        token_config=prompt_token_config,
                        fill_cache=False,
                    )
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)

def _worker_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def _cuda_profiler_call(action: str) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            cudart = torch.cuda.cudart()
            if action == "start":
                cudart.cudaProfilerStart()
            elif action == "stop":
                cudart.cudaProfilerStop()
            else:
                raise ValueError(f"unsupported cuda profiler action: {action}")
            return
    except Exception:
        pass

    try:
        from cuda.bindings import runtime as cuda_runtime

        if action == "start":
            cuda_runtime.cudaProfilerStart()
        elif action == "stop":
            cuda_runtime.cudaProfilerStop()
        else:
            raise ValueError(f"unsupported cuda profiler action: {action}")
    except Exception as exc:
        raise RuntimeError(f"cudaProfiler{action.title()} failed") from exc

def _worker_set_cuda_profiler_capture(
    *,
    status_path: Path,
    work_unit_id: str,
    attempt_id: int,
    enabled: bool,
    action: str,
) -> None:
    if not enabled:
        return
    event_prefix = f"cuda_profiler_{action}"
    _worker_append_event(
        status_path,
        f"{event_prefix}_started",
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
    )
    try:
        _cuda_profiler_call(action)
    except Exception as exc:
        _worker_append_event(
            status_path,
            f"{event_prefix}_failed",
            work_unit_id=work_unit_id,
            attempt_id=attempt_id,
            error=repr(exc),
        )
        raise
    _worker_append_event(
        status_path,
        f"{event_prefix}_finished",
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
    )
