# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-side vLLM engine execution and layerwise measurement drivers."""

from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from random_prompt_tokens import (
    RandomPromptTokenConfig,
    load_random_prompt_token_config,
    sample_prompt_token_ids,
)
from vllm_deployment import find_runtime_vllm_config, make_metadata, summarize_vllm_config, write_metadata

try:
    from .data import DataPoint
    from .datapoint_generator import LIVE_DECODE_OUTPUT_TOKENS
    from .engine import _create_llm, _engine_tokens
    from .runtime import _utc_now
    from .scheduler import _is_fatal_cuda_text, _is_oom_text, oom_dominates
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint
    from datapoint_generator import LIVE_DECODE_OUTPUT_TOKENS
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
    if "attempt_id" not in extra:
        raw_attempt_id = os.environ.get("LAYERWISE_ATTEMPT_ID")
        if raw_attempt_id not in (None, ""):
            extra["attempt_id"] = int(raw_attempt_id)
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


def _preload_flashinfer_comm(status_path: Path, work_unit_id: str, attempt_id: int) -> None:
    """Bind FlashInfer comm to the real CUDA runtime before TileLang loads its stub."""

    try:
        import flashinfer.comm  # noqa: F401
    except ModuleNotFoundError:
        return
    except Exception as exc:
        _worker_append_event(
            status_path,
            "flashinfer_comm_preload_failed",
            work_unit_id=work_unit_id,
            attempt_id=attempt_id,
            error=repr(exc),
        )
        return
    _worker_append_event(
        status_path,
        "flashinfer_comm_preloaded",
        work_unit_id=work_unit_id,
        attempt_id=attempt_id,
    )


def _dummy_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
):
    rng = random.Random()
    return [{"prompt_token_ids": sample_prompt_token_ids(rng, input_len, token_config)} for _ in range(batch_size)]


class PromptTokenFactory:
    """Sample synthetic prompt tokens, optionally reproducibly from a seed."""

    def __init__(self, seed: int | None):
        self.rng = random.Random(seed)
        self.streams: dict[tuple[Any, ...], list[int]] = {}

    def sample(self, token_count: int, token_config: RandomPromptTokenConfig) -> list[int]:
        """Return one fresh random token-id sequence."""

        return sample_prompt_token_ids(self.rng, int(token_count), token_config)

    def stream(
        self,
        key: tuple[Any, ...],
        token_count: int,
        token_config: RandomPromptTokenConfig,
    ) -> list[int]:
        """Return a stable token stream prefix for prefix-cache reuse."""

        token_count = int(token_count)
        tokens = self.streams.setdefault(key, [])
        while len(tokens) < token_count:
            tokens.extend(sample_prompt_token_ids(self.rng, token_count - len(tokens), token_config))
        return list(tokens[:token_count])


class PromptBatchCache:
    """Reuse large prompt-token batches across repeated decode measurements."""

    def __init__(self) -> None:
        self._token_prompts: dict[tuple[Any, ...], list[dict[str, Any]]] = {}

    def get_token_prompts(
        self,
        key: tuple[Any, ...],
        *,
        batch_size: int,
        input_len: int,
        token_config: RandomPromptTokenConfig,
        prompt_factory: PromptTokenFactory,
        stream_key_prefix: tuple[Any, ...],
        cache_salt_prefix: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Return cached prefix-cache prompts and whether this was a cache hit."""

        cached = self._token_prompts.get(key)
        hit = cached is not None
        if cached is None:
            cached = _token_prompts(
                batch_size,
                input_len,
                token_config,
                prompt_factory=prompt_factory,
                stream_key_prefix=stream_key_prefix,
                cache_salt_prefix=cache_salt_prefix,
            )
            self._token_prompts[key] = cached
        return [dict(prompt) for prompt in cached], hit

    def clear(self) -> None:
        """Drop cached prompt batches while keeping token streams elsewhere."""

        self._token_prompts.clear()


def _token_prompts(
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    prompt_factory: PromptTokenFactory,
    stream_key_prefix: tuple[Any, ...] | None = None,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        if stream_key_prefix is None:
            token_ids = prompt_factory.sample(input_len, token_config)
        else:
            token_ids = prompt_factory.stream((*stream_key_prefix, request_idx), input_len, token_config)
        prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts


def _variable_token_prompts(
    input_lens: list[int],
    token_config: RandomPromptTokenConfig,
    *,
    prompt_factory: PromptTokenFactory,
    stream_key_prefix: tuple[Any, ...] | None = None,
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Build prompts when requests in one batch have different input lengths."""

    prompts: list[dict[str, Any]] = []
    for request_idx, input_len in enumerate(input_lens):
        if stream_key_prefix is None:
            token_ids = prompt_factory.sample(input_len, token_config)
        else:
            token_ids = prompt_factory.stream((*stream_key_prefix, request_idx), input_len, token_config)
        prompt: dict[str, Any] = {"prompt_token_ids": token_ids}
        if cache_salt_prefix is not None:
            prompt["cache_salt"] = f"{cache_salt_prefix}:req{request_idx}"
        prompts.append(prompt)
    return prompts


def _prefix_suffix_prompts(
    batch_size: int,
    prefix_len: int,
    suffix_len: int,
    token_config: RandomPromptTokenConfig,
    *,
    prompt_factory: PromptTokenFactory,
    prefix_stream_key_prefix: tuple[Any, ...],
    cache_salt_prefix: str | None = None,
) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for request_idx in range(batch_size):
        prefix = prompt_factory.stream(
            (*prefix_stream_key_prefix, request_idx),
            prefix_len,
            token_config,
        )
        suffix = prompt_factory.sample(suffix_len, token_config)
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
    prompt_factory: PromptTokenFactory,
    stream_key_prefix: tuple[Any, ...] | None = None,
    cache_salt_prefix: str | None = None,
    prompt_cache: PromptBatchCache | None = None,
    prompt_cache_key: tuple[Any, ...] | None = None,
    status_path: Path | None = None,
    work_unit_id: str | None = None,
    datapoint_id: str | None = None,
    timing_phase: str | None = None,
    timing_batch_size: int | None = None,
    timing_past_kv: int | None = None,
    timing_run: int | None = None,
) -> None:
    build_start = time.perf_counter()
    prompt_cache_hit: bool | None = None
    if (
        prompt_cache is not None
        and prompt_cache_key is not None
        and stream_key_prefix is not None
        and cache_salt_prefix is not None
    ):
        prompts, prompt_cache_hit = prompt_cache.get_token_prompts(
            prompt_cache_key,
            batch_size=batch_size,
            input_len=input_len,
            token_config=token_config,
            prompt_factory=prompt_factory,
            stream_key_prefix=stream_key_prefix,
            cache_salt_prefix=cache_salt_prefix,
        )
    else:
        prompts = _token_prompts(
            batch_size,
            input_len,
            token_config,
            prompt_factory=prompt_factory,
            stream_key_prefix=stream_key_prefix,
            cache_salt_prefix=cache_salt_prefix,
        )
    prompt_build_ms = (time.perf_counter() - build_start) * 1000.0
    generate_start = time.perf_counter()
    profile_paths = _profile_generate_call(
        lambda: llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        ),
        status_path=status_path,
        work_unit_id=work_unit_id,
        datapoint_id=datapoint_id,
        phase=timing_phase,
        batch_size=timing_batch_size,
        past_kv=timing_past_kv,
        run=timing_run,
    )
    generate_ms = (time.perf_counter() - generate_start) * 1000.0
    if status_path is not None and work_unit_id is not None and timing_phase is not None:
        _worker_append_event(
            status_path,
            "generate_wall_time",
            work_unit_id=work_unit_id,
            datapoint_id=datapoint_id,
            phase=timing_phase,
            batch_size=timing_batch_size,
            past_kv=timing_past_kv,
            run=timing_run,
            prompt_build_ms=prompt_build_ms,
            generate_ms=generate_ms,
            prompt_cache_hit=prompt_cache_hit,
            profile_stats=profile_paths.get("stats"),
            profile_text=profile_paths.get("text"),
        )


def _profile_generate_call(
    func,
    *,
    status_path: Path | None,
    work_unit_id: str | None,
    datapoint_id: str | None,
    phase: str | None,
    batch_size: int | None,
    past_kv: int | None,
    run: int | None,
) -> dict[str, str]:
    """Run ``func`` with optional cProfile output for generate diagnostics."""

    if os.environ.get("LAYERWISE_PROFILE_GENERATE", "0") != "1":
        func()
        return {}
    min_bs = int(os.environ.get("LAYERWISE_PROFILE_GENERATE_MIN_BS", "0"))
    if batch_size is None or int(batch_size) < min_bs:
        func()
        return {}
    import cProfile
    import io
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        func()
    finally:
        profiler.disable()
    base_dir = Path(
        os.environ.get(
            "LAYERWISE_PROFILE_GENERATE_DIR",
            str((status_path.parent if status_path is not None else Path.cwd()) / "generate_profiles"),
        )
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    safe_dpid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", datapoint_id or "unknown")
    label = f"{work_unit_id or 'wu'}_{phase or 'phase'}_bs{batch_size}_past{past_kv}_run{run}"
    label = re.sub(r"[^A-Za-z0-9_.:-]+", "_", label)
    stats_path = base_dir / f"{label}.pstats"
    text_path = base_dir / f"{label}.txt"
    profiler.dump_stats(str(stats_path))
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(80)
    text_path.write_text(
        f"datapoint_id={safe_dpid}\n"
        f"phase={phase} batch_size={batch_size} past_kv={past_kv} run={run}\n\n" + stream.getvalue()
    )
    return {"stats": str(stats_path), "text": str(text_path)}


def _run_generate_variable_lengths(
    llm,
    sampling_params,
    *,
    input_lens: list[int],
    token_config: RandomPromptTokenConfig,
    prompt_factory: PromptTokenFactory,
    stream_key_prefix: tuple[Any, ...] | None = None,
    cache_salt_prefix: str | None = None,
) -> None:
    """Run generation for one batch with per-request prompt lengths."""

    prompts = _variable_token_prompts(
        input_lens,
        token_config,
        prompt_factory=prompt_factory,
        stream_key_prefix=stream_key_prefix,
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
    prompt_factory: PromptTokenFactory,
    prefix_stream_key_prefix: tuple[Any, ...],
    cache_salt_prefix: str | None = None,
) -> None:
    prompts = _prefix_suffix_prompts(
        batch_size,
        prefix_len,
        suffix_len,
        token_config,
        prompt_factory=prompt_factory,
        prefix_stream_key_prefix=prefix_stream_key_prefix,
        cache_salt_prefix=cache_salt_prefix,
    )
    llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def _use_live_step_driver(dp: DataPoint) -> bool:
    """Return whether the opt-in live engine-step driver should handle a row."""

    if os.environ.get("LAYERWISE_USE_LIVE_STEP_DRIVER", "0") != "1":
        return False
    if dp.phase == "ctx":
        return int(dp.batch_size) > 0 and int(dp.new_tokens) > 0
    if dp.phase == "gen":
        min_past_kv = int(os.environ.get("LAYERWISE_LIVE_STEP_GEN_MIN_PAST_KV", "8192"))
        if int(dp.past_kv) >= min_past_kv:
            return True
        min_batch_size = int(os.environ.get("LAYERWISE_LIVE_STEP_GEN_MIN_BATCH_SIZE", "256"))
        return min_batch_size > 0 and int(dp.batch_size) >= min_batch_size
    return False


def _make_final_only_sampling_params(sampling_cls, **kwargs):
    """Build sampling params matching ``LLM.generate`` output processing."""

    params = sampling_cls(**kwargs)
    try:
        from vllm.sampling_params import RequestOutputKind

        params.output_kind = RequestOutputKind.FINAL_ONLY
    except Exception:
        pass
    return params


def _engine_core_scheduler(llm):
    """Return the in-process vLLM scheduler used by ``LLMEngine.step``."""

    engine_core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    scheduler = getattr(engine_core, "scheduler", None)
    if scheduler is None:
        raise RuntimeError("live LLMEngine driver requires an in-process vLLM scheduler")
    return scheduler


@contextmanager
def _temporary_scheduler_token_budget(llm, token_budget: int):
    """Temporarily cap scheduler tokens per step without resizing engine buffers."""

    scheduler = _engine_core_scheduler(llm)
    old_budget = scheduler.max_num_scheduled_tokens
    scheduler.max_num_scheduled_tokens = int(token_budget)
    try:
        yield
    finally:
        scheduler.max_num_scheduled_tokens = old_budget


def _engine_requests_by_id(llm, request_ids: list[str]) -> dict[str, Any]:
    """Return currently known scheduler request objects for the given IDs."""

    scheduler = _engine_core_scheduler(llm)
    requests = dict(getattr(scheduler, "requests", {}))

    def _track_request_container(container: Any) -> None:
        if isinstance(container, dict):
            iterable = container.values()
        else:
            iterable = container or []
        for req in iterable:
            req_id = getattr(req, "request_id", getattr(req, "req_id", None))
            if req_id is not None:
                requests[str(req_id)] = req

    _track_request_container(getattr(scheduler, "running", []))
    _track_request_container(getattr(scheduler, "waiting", []))
    return {request_id: requests[request_id] for request_id in request_ids if request_id in requests}


def _all_requests_computed_at_least(llm, request_ids: list[str], token_count: int) -> bool:
    """Return whether all live requests have computed at least ``token_count`` tokens."""

    requests = _engine_requests_by_id(llm, request_ids)
    if len(requests) != len(request_ids):
        return False
    return all(int(getattr(req, "num_computed_tokens", 0)) >= token_count for req in requests.values())


def _all_requests_done_or_computed_at_least(
    llm,
    request_ids: list[str],
    token_count: int,
    *,
    seen_request_ids: set[str] | None = None,
) -> bool:
    """Return whether requests are finished or have reached ``token_count``."""

    requests = _engine_requests_by_id(llm, request_ids)
    if seen_request_ids is not None:
        seen_request_ids.update(requests)
        if any(request_id not in seen_request_ids for request_id in request_ids):
            return False
    elif len(requests) != len(request_ids):
        return False
    return all(
        request_id not in requests or int(getattr(requests[request_id], "num_computed_tokens", 0)) >= token_count
        for request_id in request_ids
    )


def _add_engine_requests(llm, prompts: list[dict[str, Any]], sampling_params, *, request_prefix: str) -> list[str]:
    """Submit prompts directly to the lower-level vLLM engine."""

    request_ids = []
    for idx, prompt in enumerate(prompts):
        request_id = f"{request_prefix}:{idx}:{time.time_ns()}"
        request_ids.append(llm.llm_engine.add_request(request_id, prompt, sampling_params))
    return request_ids


def _prime_engine_requests_for_prompt_prefix(
    llm,
    request_ids: list[str],
    prefix_tokens: int,
    *,
    label: str,
    append_decode_token: bool = False,
) -> int:
    """Allocate KV slots and advance newly submitted requests to ``prefix_tokens``.

    vLLM tracks both the token counter and allocated KV blocks for each
    request.  Allocate the prompt's historical KV slots through the normal
    cache manager in scheduler-sized rounds, then leave the request in WAITING.
    The next scheduler iteration can admit either the next prefill chunk or,
    when requested, a one-token decode step with a full block table.

    Chunking matters for hybrid caches such as DeepSeek V4: sliding-window and
    compressed groups recycle blocks as ``num_computed_tokens`` advances.  A
    one-shot allocation for a long prefix can ask those groups to hold blocks
    that real chunked prefill would already have freed.  The scheduler token
    budget is batch-wide, not per request, so split each round across all active
    requests.
    """

    prefix_tokens = int(prefix_tokens)
    if prefix_tokens <= 0:
        return 0

    scheduler = _engine_core_scheduler(llm)
    requests = getattr(scheduler, "requests", {})
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        raise RuntimeError("vLLM scheduler does not expose kv_cache_manager")
    scheduled_requests: list[tuple[str, Any]] = []
    for request_id in request_ids:
        request = requests.get(request_id)
        if request is None:
            raise RuntimeError(f"request {request_id} was not registered with the scheduler")
        if int(getattr(request, "num_prompt_tokens", 0)) < prefix_tokens:
            raise RuntimeError(f"request {request_id} prompt is shorter than synthetic {label} prefix={prefix_tokens}")
        if int(getattr(request, "num_computed_tokens", 0)) != 0:
            raise RuntimeError(f"request {request_id} was already partially computed")
        scheduled_requests.append((request_id, request))

    scheduler_token_budget = int(getattr(scheduler, "max_num_scheduled_tokens", 0) or 0)
    if scheduler_token_budget <= 0:
        scheduler_token_budget = prefix_tokens * max(1, len(scheduled_requests))
    scheduler_token_budget = max(1, scheduler_token_budget)

    def _free_blocks() -> Any:
        return getattr(
            getattr(kv_cache_manager, "block_pool", None),
            "get_num_free_blocks",
            lambda: None,
        )()

    total_rounds = 0
    while True:
        active = [
            (request_id, request)
            for request_id, request in scheduled_requests
            if int(getattr(request, "num_computed_tokens", 0)) < prefix_tokens
        ]
        if not active:
            break
        round_tokens = 0
        per_request_budget = max(1, scheduler_token_budget // max(1, len(active)))
        for request_id, request in active:
            available_tokens = scheduler_token_budget - round_tokens
            if available_tokens <= 0:
                break
            remaining = prefix_tokens - int(getattr(request, "num_computed_tokens", 0))
            num_new_tokens = min(per_request_budget, remaining, available_tokens)
            try:
                new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to prime {label} prefix for request {request_id}: "
                    f"prefix={prefix_tokens} computed={getattr(request, 'num_computed_tokens', 0)} "
                    f"round_tokens={round_tokens} chunk_tokens={num_new_tokens} "
                    f"active_requests={len(active)} free_blocks={_free_blocks()}: {exc}"
                ) from exc
            if new_blocks is None:
                raise RuntimeError(
                    f"insufficient KV blocks to prime {label} prefix for request {request_id}: "
                    f"prefix={prefix_tokens} computed={getattr(request, 'num_computed_tokens', 0)} "
                    f"round_tokens={round_tokens} chunk_tokens={num_new_tokens} "
                    f"active_requests={len(active)} free_blocks={_free_blocks()}"
                )
            request.num_computed_tokens += num_new_tokens
            round_tokens += num_new_tokens
        if round_tokens <= 0:
            raise RuntimeError(
                f"failed to advance {label} prefix priming: prefix={prefix_tokens} "
                f"active_requests={len(active)} scheduler_token_budget={scheduler_token_budget}"
            )
        total_rounds += 1

    if append_decode_token:
        for _, request in scheduled_requests:
            if int(getattr(request, "num_tokens", 0)) <= prefix_tokens:
                token_ids = getattr(request, "prompt_token_ids", None) or [0]
                request.append_output_token_ids(int(token_ids[-1]))
    return total_rounds


def _prime_engine_requests_for_context(llm, request_ids: list[str], past_kv: int) -> int:
    """Move newly submitted requests to an exact-past prefill scheduler state."""

    return _prime_engine_requests_for_prompt_prefix(
        llm,
        request_ids,
        past_kv,
        label="context",
        append_decode_token=False,
    )


def _prime_engine_requests_for_decode(llm, request_ids: list[str], past_kv: int) -> int:
    """Move newly submitted requests to a decode-ready scheduler state."""

    return _prime_engine_requests_for_prompt_prefix(
        llm,
        request_ids,
        past_kv,
        label="decode",
        append_decode_token=True,
    )


def _live_decode_past_tolerance(dp: DataPoint, total_decode_steps: int, decode_headroom: int) -> float:
    """Return the existing live-decode past tolerance for a datapoint."""

    return max(
        1.0,
        float(dp.past_kv) * 0.01,
        float((total_decode_steps + decode_headroom) * max(1, dp.new_tokens)),
    )


def _short_retry_past_kv(dp: DataPoint, past_tolerance: float) -> int:
    """Return a shorter prompt length that remains in the target past bucket."""

    retry_delta = max(1, int(past_tolerance))
    return max(1, int(dp.past_kv) - retry_delta)


def _abort_engine_requests(llm, request_ids: list[str]) -> None:
    """Abort live engine requests, ignoring already-finished IDs."""

    if not request_ids:
        return
    try:
        llm.llm_engine.abort_request(request_ids, internal=True)
    except Exception:
        pass


def _step_engine_until(
    llm,
    predicate,
    *,
    max_steps: int,
    status_path: Path | None = None,
    work_unit_id: str | None = None,
    datapoint_id: str | None = None,
    event: str = "live_step_wait",
) -> int:
    """Step the engine until ``predicate`` is true or ``max_steps`` is exceeded."""

    steps = 0
    while not predicate():
        if steps >= max_steps:
            raise RuntimeError(f"timed out waiting for live LLMEngine state after {steps} steps")
        llm.llm_engine.step()
        steps += 1
    if status_path is not None and work_unit_id is not None:
        _worker_append_event(
            status_path,
            event,
            work_unit_id=work_unit_id,
            datapoint_id=datapoint_id,
            steps=steps,
        )
    return steps


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


def _live_ctx_chunk_plan(
    dp: DataPoint,
    max_num_batched_tokens: int | None,
) -> list[tuple[int, int]]:
    """Return per-request ``(new_tokens, past_kv)`` scheduler chunks."""

    if dp.phase != "ctx":
        raise ValueError(f"live ctx chunk plan requires ctx datapoint, got {dp.phase}")
    if dp.new_tokens <= 0:
        raise ValueError(f"ctx new_tokens must be positive, got {dp.new_tokens}")
    request_count = int(dp.batch_size)
    if request_count < 1:
        raise ValueError(f"ctx batch_size must be positive, got {dp.batch_size}")

    step_budget = int(max_num_batched_tokens or 0)
    if step_budget > 0 and step_budget < request_count:
        raise ValueError(
            "live ctx batch cannot fit one token per request within "
            f"max_num_batched_tokens={step_budget}, batch_size={request_count}"
        )
    if step_budget <= 0 and request_count == 1:
        return [(int(dp.new_tokens), int(dp.past_kv))]
    per_request_budget = int(dp.new_tokens) if step_budget <= 0 else max(1, step_budget // request_count)
    remaining = int(dp.new_tokens)
    past_kv = int(dp.past_kv)
    chunks = []
    while remaining > 0:
        chunk_tokens = min(per_request_budget, remaining)
        chunks.append((chunk_tokens, past_kv))
        remaining -= chunk_tokens
        past_kv += chunk_tokens
    return chunks


def _resolve_attr_path(obj: Any, path: str) -> Any | None:
    """Return a nested attribute path if every segment exists."""

    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _find_torch_model(llm: Any) -> Any:
    """Find the inner torch module inside a vLLM ``LLM`` instance."""

    candidate_paths = (
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.model_runner.model",
        "llm_engine.engine_core.model_executor.driver_worker.model_runner.model",
        "engine_core.model_executor.driver_worker.model_runner.model",
    )
    for path in candidate_paths:
        candidate = _resolve_attr_path(llm, path)
        if hasattr(candidate, "named_modules"):
            return candidate

    queue: list[tuple[Any, int]] = [(llm, 0)]
    seen: set[int] = set()
    attrs = (
        "llm_engine",
        "engine_core",
        "model_executor",
        "driver_worker",
        "worker",
        "model_runner",
        "model",
    )
    while queue:
        current, depth = queue.pop(0)
        ident = id(current)
        if ident in seen:
            continue
        seen.add(ident)
        if depth > 6:
            continue
        if hasattr(current, "named_modules"):
            names = [name for name, _ in current.named_modules()]
            if any(".mlp" in name or name.endswith("mlp") for name in names):
                return current
        for attr in attrs:
            child = getattr(current, attr, None)
            if child is not None:
                queue.append((child, depth + 1))
    raise RuntimeError("could not locate inner torch model for router weight loading")


def _load_safetensor_weights(model_id: str, keys: set[str]) -> dict[str, Any]:
    """Load selected tensors from an HF safetensors checkpoint."""

    if not keys:
        return {}

    if os.path.isdir(model_id):
        model_dir = Path(model_id)
        index_path = model_dir / "model.safetensors.index.json"

        def resolve_file(filename: str) -> Path:
            return model_dir / filename
    else:
        from huggingface_hub import hf_hub_download

        index_path = Path(hf_hub_download(model_id, "model.safetensors.index.json"))

        def resolve_file(filename: str) -> Path:
            return Path(hf_hub_download(model_id, filename))

    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    missing = sorted(key for key in keys if key not in weight_map)
    if missing:
        raise RuntimeError(f"router checkpoint keys missing: {missing[:5]}")

    from safetensors import safe_open

    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    tensors: dict[str, Any] = {}
    for filename, file_keys in by_file.items():
        with safe_open(str(resolve_file(filename)), framework="pt", device="cpu") as f:
            for key in file_keys:
                tensors[key] = f.get_tensor(key)
    return tensors


def _install_router_weights(
    llm: Any,
    *,
    source_model: str,
    target_layers: list[int],
) -> int:
    """Copy compatible real MoE router tensors into a dummy-weight vLLM model."""

    import torch

    model = _find_torch_model(llm)
    layer_modules: list[tuple[int, int, str, Any]] = []
    key_suffixes = {
        "gate": "gate.weight",
        "shared_expert_gate": "shared_expert_gate.weight",
    }
    for module_name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", module_name)
        if match is None:
            continue
        layer_position = int(match.group(1))
        source_layer = int(target_layers[layer_position]) if layer_position < len(target_layers) else layer_position
        if any(hasattr(module, attr) for attr in key_suffixes):
            layer_modules.append((layer_position, source_layer, module_name, module))

    needed: set[str] = set()
    assignments: list[tuple[Any, str, str]] = []
    for _layer_position, source_layer, _module_name, module in layer_modules:
        for attr, suffix in key_suffixes.items():
            linear = getattr(module, attr, None)
            param = getattr(linear, "weight", None)
            if param is None:
                continue
            source_key = f"model.language_model.layers.{source_layer}.mlp.{suffix}"
            needed.add(source_key)
            assignments.append((param, source_key, attr))

    tensors = _load_safetensor_weights(source_model, needed)
    loaded = 0
    with torch.no_grad():
        for param, source_key, attr in assignments:
            tensor = tensors[source_key]
            if tuple(param.shape) != tuple(tensor.shape):
                param_shape = tuple(param.shape)
                tensor_shape = tuple(tensor.shape)
                if (
                    len(param_shape) == len(tensor_shape)
                    and len(param_shape) >= 1
                    and tensor_shape[0] > param_shape[0]
                    and tensor_shape[0] % param_shape[0] == 0
                    and tensor_shape[1:] == param_shape[1:]
                ):
                    tensor = tensor.narrow(0, 0, param_shape[0]).contiguous()
            if tuple(param.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"router tensor shape mismatch for {source_key} ({attr}): "
                    f"checkpoint={tuple(tensor.shape)} param={tuple(param.shape)}"
                )
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            loaded += 1
    return loaded


def run_worker(spec_path: Path) -> None:
    """Execute one worker spec inside an nsys-profiled subprocess."""

    spec = json.loads(spec_path.read_text())
    status_path = Path(spec["status_path"])
    work_unit_id = spec["work_unit_id"]
    os.environ["LAYERWISE_ATTEMPT_ID"] = str(spec["attempt_id"])
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
    if int(spec.get("physical_gpus") or 1) <= 1:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["LAYERWISE_TARGET_LAYERS"] = ",".join(str(x) for x in spec["target_layers"])
    if spec.get("enable_layer_patch", True):
        os.environ["LAYERWISE_SKIP_ENABLE"] = "1"
    else:
        os.environ["LAYERWISE_SKIP_ENABLE"] = "0"
    if spec.get("enable_step_marker", True):
        os.environ["LAYERWISE_STEP_MARKER"] = "1"
    else:
        os.environ["LAYERWISE_STEP_MARKER"] = "0"
    if spec.get("moe_noop"):
        os.environ["LAYERWISE_MOE_NOOP"] = "1"
    else:
        os.environ.pop("LAYERWISE_MOE_NOOP", None)
    if spec.get("moe_weight_mode") == "dummy":
        os.environ["LAYERWISE_SYNTHETIC_HASH_ROUTING"] = "1"
    else:
        os.environ.pop("LAYERWISE_SYNTHETIC_HASH_ROUTING", None)
    physical_gpus = int(spec.get("physical_gpus") or 1)
    if spec.get("router_weight_model") and physical_gpus > 1:
        os.environ["LAYERWISE_ROUTER_WEIGHT_MODEL"] = str(spec["router_weight_model"])
    else:
        os.environ.pop("LAYERWISE_ROUTER_WEIGHT_MODEL", None)
    max_num_batched_tokens = spec.get("max_num_batched_tokens") or 1
    iterations = {1}
    iterations.update(_ctx_marker_iteration(dp, max_num_batched_tokens) for dp in datapoints if dp.phase == "ctx")
    iterations.update(dp.past_kv + 1 for dp in datapoints if dp.phase == "gen")
    os.environ["LAYERWISE_STEP_ITERATIONS"] = ",".join(str(x) for x in sorted(iterations))
    os.environ["LAYERWISE_BENCH_MIN_NEW"] = "1"
    os.environ["LAYERWISE_PROGRESS_FILE"] = str(status_path)
    os.environ["LAYERWISE_WORK_UNIT_ID"] = work_unit_id
    os.environ["LAYERWISE_ATTEMPT_ID"] = str(spec["attempt_id"])
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
    pythonpath_parts = [str(_THIS_DIR)]
    if os.environ.get("PYTHONPATH"):
        pythonpath_parts.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
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
    _preload_flashinfer_comm(status_path, work_unit_id, int(spec["attempt_id"]))
    import vllm_layer_skip_patch  # noqa: F401
    import vllm_scheduler_timing_patch  # noqa: F401
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
    prompt_factory = PromptTokenFactory(spec.get("prompt_seed"))
    engine_tokens = _engine_tokens(
        model_dir=spec["model_dir"],
        datapoints=datapoints,
        extra_vllm_args=spec["extra_vllm_args"],
        max_num_seqs=spec.get("max_num_seqs"),
        max_num_batched_tokens=spec.get("max_num_batched_tokens"),
        cache_block_size=spec.get("cache_block_size"),
        max_model_len=spec.get("max_model_len"),
        gpu_memory_utilization=spec.get("gpu_memory_utilization", 0.9),
        gen_driver=spec.get("gen_driver", "prefix_cache"),
    )
    _worker_append_event(status_path, "engine_args_finished", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_create_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    llm = _create_llm(
        engine_tokens,
        enable_layerwise_nvtx_tracing=bool(spec.get("enable_layerwise_nvtx_tracing", True)),
    )
    _worker_append_event(
        status_path,
        "engine_create_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    if spec.get("router_weight_model") and physical_gpus <= 1:
        _worker_append_event(
            status_path,
            "router_weights_started",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
            source_model=spec.get("router_weight_model"),
        )
        loaded_router_tensors = _install_router_weights(
            llm,
            source_model=str(spec["router_weight_model"]),
            target_layers=[int(x) for x in spec["target_layers"]],
        )
        _worker_append_event(
            status_path,
            "router_weights_finished",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
            loaded_tensors=loaded_router_tensors,
        )
    _worker_append_event(
        status_path,
        "engine_metadata_started",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    effective_config = None
    runtime_vllm_config = find_runtime_vllm_config(llm)
    has_ctx = any(dp.phase == "ctx" for dp in datapoints)
    has_gen = any(dp.phase == "gen" for dp in datapoints)
    gen_driver = str(spec.get("gen_driver", "prefix_cache"))
    if (has_ctx or has_gen) and runtime_vllm_config is not None:
        cache_config = getattr(runtime_vllm_config, "cache_config", None)
        needs_prefix_cache = (
            has_gen
            and gen_driver == "prefix_cache"
            and any(dp.phase == "gen" and not _use_live_step_driver(dp) for dp in datapoints)
        )
        if needs_prefix_cache and getattr(cache_config, "enable_prefix_caching", None) is False:
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
            "model_layer_count": spec.get("model_layer_count"),
            "enable_layer_patch": bool(spec.get("enable_layer_patch", True)),
            "enable_step_marker": bool(spec.get("enable_step_marker", True)),
            "moe_noop": bool(spec.get("moe_noop")),
            "moe_weight_mode": spec.get("moe_weight_mode") or "",
            "gen_driver": gen_driver,
            "router_weight_model": spec.get("router_weight_model") or "",
            "physical_gpus": int(spec.get("physical_gpus") or 1),
        },
    )
    if spec.get("metadata_path"):
        write_metadata(spec["metadata_path"], metadata)
    _worker_append_event(
        status_path,
        "engine_metadata_finished",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
    )
    _worker_append_event(
        status_path,
        "engine_metadata_written",
        work_unit_id=work_unit_id,
        attempt_id=spec["attempt_id"],
        metadata_path=spec.get("metadata_path"),
        vllm_config_hash=metadata["vllm_config_hash"],
    )
    _worker_append_event(status_path, "engine_ready", work_unit_id=work_unit_id)
    cuda_profiler_capture = str(spec.get("nsys_capture", "full")) == "cuda_profiler_api"
    try:
        _worker_append_event(
            status_path,
            "measurement_started",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
        )
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
                    prompt_factory=prompt_factory,
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
                    prompt_factory=prompt_factory,
                    warmup_runs=int(spec.get("gen_warmup_runs", 0)),
                    measured_runs=int(spec.get("gen_measured_runs", 1)),
                    gen_driver=gen_driver,
                    max_num_seqs=spec.get("max_num_seqs"),
                )
        finally:
            _worker_set_cuda_profiler_capture(
                status_path=status_path,
                work_unit_id=work_unit_id,
                attempt_id=spec["attempt_id"],
                enabled=cuda_profiler_capture,
                action="stop",
            )
        _worker_append_event(
            status_path,
            "measurement_finished",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
        )
    finally:
        _worker_append_event(
            status_path,
            "worker_cleanup_started",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
        )
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
        _worker_append_event(
            status_path,
            "worker_cleanup_finished",
            work_unit_id=work_unit_id,
            attempt_id=spec["attempt_id"],
        )
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
    prompt_factory: PromptTokenFactory,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    max_num_batched_tokens: int = 1,
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    sampling_params = sampling_cls(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=True,
    )
    live_sampling_params = _make_final_only_sampling_params(
        sampling_cls,
        temperature=0.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    pruned: set[str] = set()
    filled_prefixes: set[tuple[int, int]] = set()

    def run_one(dp: DataPoint, run_idx: int, *, warmup: bool) -> None:
        dpid = _worker_datapoint_id(work_unit_id, dp)
        if dpid in pruned:
            return
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
            fill_prefix = dp.past_kv > 0 and prefix_key not in filled_prefixes
            if _use_live_step_driver(dp):
                wall_start = time.perf_counter()
                _run_live_ctx_iteration(
                    llm,
                    live_sampling_params,
                    dp,
                    work_unit_id=work_unit_id,
                    run_idx=run_idx,
                    warmup=warmup,
                    token_config=prompt_token_config,
                    prompt_factory=prompt_factory,
                    marker_mod=marker_mod,
                    status_path=status_path,
                    datapoint_id=dpid,
                    max_num_batched_tokens=max_num_batched_tokens,
                )
                if not warmup:
                    _worker_append_event(
                        status_path,
                        "measurement_wall_time",
                        work_unit_id=work_unit_id,
                        datapoint_id=dpid,
                        phase="ctx",
                        batch_size=dp.batch_size,
                        new_tokens=dp.new_tokens,
                        past_kv=dp.past_kv,
                        run=run_idx,
                        live_step_driver=True,
                        wall_latency_ms=(time.perf_counter() - wall_start) * 1000.0,
                    )
            elif warmup:
                active_iteration = ""
                _set_marker_state(
                    marker_mod,
                    active_iterations="",
                    phase="ctx",
                    step=dp.new_tokens,
                    bs=dp.batch_size,
                    past=dp.past_kv,
                    run=None,
                )
                _run_prefix_cached_ctx_iteration(
                    llm,
                    sampling_params,
                    dp,
                    work_unit_id=work_unit_id,
                    run_idx=run_idx,
                    warmup=True,
                    fill_prefix=fill_prefix,
                    token_config=prompt_token_config,
                    prompt_factory=prompt_factory,
                    active_iteration=active_iteration,
                    marker_mod=marker_mod,
                )
            else:
                active_iteration = str(marker_iteration)
                _set_marker_state(
                    marker_mod,
                    active_iterations=active_iteration,
                    phase="ctx",
                    step=dp.new_tokens,
                    bs=dp.batch_size,
                    past=dp.past_kv,
                    run=run_idx,
                )
                wall_start = time.perf_counter()
                _run_prefix_cached_ctx_iteration(
                    llm,
                    sampling_params,
                    dp,
                    work_unit_id=work_unit_id,
                    run_idx=run_idx,
                    warmup=False,
                    fill_prefix=fill_prefix,
                    token_config=prompt_token_config,
                    prompt_factory=prompt_factory,
                    active_iteration=str(marker_iteration),
                    marker_mod=marker_mod,
                )
                _worker_append_event(
                    status_path,
                    "measurement_wall_time",
                    work_unit_id=work_unit_id,
                    datapoint_id=dpid,
                    phase="ctx",
                    batch_size=dp.batch_size,
                    new_tokens=dp.new_tokens,
                    past_kv=dp.past_kv,
                    run=run_idx,
                    wall_latency_ms=(time.perf_counter() - wall_start) * 1000.0,
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
                return
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

    for run_idx in range(warmup_runs):
        for dp in datapoints:
            run_one(dp, run_idx, warmup=True)
    for run_idx in range(measured_runs):
        for dp in datapoints:
            run_one(dp, run_idx, warmup=False)


def _ctx_cache_salt_prefix(
    work_unit_id: str,
    dp: DataPoint,
) -> str:
    return f"layerwise-ctx:{work_unit_id}:bs{dp.batch_size}:past{dp.past_kv}"


def _ctx_prefix_stream_key(work_unit_id: str, dp: DataPoint) -> tuple[Any, ...]:
    """Return the token stream key shared by context prefix-cache prompts."""

    return ("ctx-prefix", work_unit_id, int(dp.batch_size), int(dp.past_kv))


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
    prompt_factory: PromptTokenFactory,
    active_iteration: str,
    marker_mod,
) -> None:
    cache_salt_prefix = _ctx_cache_salt_prefix(work_unit_id, dp) if dp.past_kv > 0 else None
    prefix_stream_key = _ctx_prefix_stream_key(work_unit_id, dp)
    if fill_prefix and dp.past_kv > 0:
        _set_marker_state(marker_mod, active_iterations="", phase="ctx")
        _run_generate(
            llm,
            sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            prompt_factory=prompt_factory,
            stream_key_prefix=prefix_stream_key,
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
        prompt_factory=prompt_factory,
        prefix_stream_key_prefix=prefix_stream_key,
        cache_salt_prefix=cache_salt_prefix,
    )


def _run_live_ctx_iteration(
    llm,
    sampling_params,
    dp: DataPoint,
    *,
    work_unit_id: str,
    run_idx: int,
    warmup: bool,
    token_config: RandomPromptTokenConfig,
    prompt_factory: PromptTokenFactory,
    marker_mod,
    status_path: Path | None = None,
    datapoint_id: str | None = None,
    max_num_batched_tokens: int | None = None,
) -> None:
    """Measure a ctx row by stepping live chunked-prefill request(s)."""

    if dp.batch_size < 1:
        raise ValueError(f"live ctx driver requires batch_size >= 1, got {dp.batch_size}")
    cache_salt_prefix = _ctx_cache_salt_prefix(work_unit_id, dp)
    prefix_stream_key = _ctx_prefix_stream_key(work_unit_id, dp)
    prompts = _prefix_suffix_prompts(
        dp.batch_size,
        dp.past_kv,
        dp.new_tokens,
        token_config,
        prompt_factory=prompt_factory,
        prefix_stream_key_prefix=prefix_stream_key,
        cache_salt_prefix=cache_salt_prefix,
    )
    request_ids = _add_engine_requests(
        llm,
        prompts,
        sampling_params,
        request_prefix=f"ctx-live:{work_unit_id}:run{run_idx}",
    )
    try:
        prime_rounds = _prime_engine_requests_for_context(llm, request_ids, dp.past_kv)
        _worker_append_event(
            status_path,
            "live_ctx_prefix_primed",
            work_unit_id=work_unit_id,
            datapoint_id=datapoint_id,
            batch_size=dp.batch_size,
            new_tokens=dp.new_tokens,
            past_kv=dp.past_kv,
            run=run_idx,
            rounds=prime_rounds,
        )
        live_start = time.perf_counter()
        if dp.batch_size > 1:
            scheduler_token_budget = int(max_num_batched_tokens or 0)
            if scheduler_token_budget <= 0:
                scheduler_token_budget = int(dp.batch_size) * int(dp.new_tokens)
            target_prompt_tokens = int(dp.past_kv) + int(dp.new_tokens)
            total_requested_tokens = int(dp.batch_size) * int(dp.new_tokens)
            max_steps = max(
                8,
                math.ceil(total_requested_tokens / max(1, scheduler_token_budget)) + int(dp.batch_size) + 4,
            )
            with _temporary_scheduler_token_budget(llm, scheduler_token_budget):
                if warmup:
                    _set_marker_state(marker_mod, active_iterations="", phase="ctx")
                else:
                    _set_marker_state(
                        marker_mod,
                        active_iterations="",
                        trigger="ctx_batch",
                        phase="ctx",
                        step=dp.new_tokens,
                        bs=dp.batch_size,
                        past=dp.past_kv,
                        run=run_idx,
                        live_step_driver=True,
                        sync_execute_model_wall_time=True,
                        batched_ctx_driver=True,
                        requested_new_tokens=dp.new_tokens,
                        requested_past_kv=dp.past_kv,
                        scheduler_token_budget=scheduler_token_budget,
                    )
                executed_steps = 0
                seen_request_ids: set[str] = set()
                for _step_idx in range(max_steps):
                    llm.llm_engine.step()
                    executed_steps += 1
                    if _all_requests_done_or_computed_at_least(
                        llm,
                        request_ids,
                        target_prompt_tokens,
                        seen_request_ids=seen_request_ids,
                    ):
                        break
                else:
                    raise RuntimeError(
                        "live batched ctx driver did not finish target prompt "
                        f"for {dp.shape_key}: target_prompt_tokens={target_prompt_tokens}, "
                        f"scheduler_token_budget={scheduler_token_budget}, max_steps={max_steps}"
                    )
            if not warmup:
                _worker_append_event(
                    status_path,
                    "live_step_wall_time",
                    work_unit_id=work_unit_id,
                    datapoint_id=datapoint_id,
                    phase="ctx",
                    batch_size=dp.batch_size,
                    new_tokens=dp.new_tokens,
                    past_kv=dp.past_kv,
                    run=run_idx,
                    live_step_driver=True,
                    batched_ctx_driver=True,
                    chunks=executed_steps,
                    steps=executed_steps,
                    wall_latency_ms=(time.perf_counter() - live_start) * 1000.0,
                )
            return
        chunks = _live_ctx_chunk_plan(dp, max_num_batched_tokens)
        executed_steps = 0
        for chunk_index, (chunk_tokens, chunk_past_kv) in enumerate(chunks):
            scheduler_token_budget = int(chunk_tokens) * int(dp.batch_size)
            with _temporary_scheduler_token_budget(llm, scheduler_token_budget):
                if warmup:
                    _set_marker_state(marker_mod, active_iterations="", phase="ctx")
                else:
                    _set_marker_state(
                        marker_mod,
                        active_iterations="",
                        trigger="ctx_chunk",
                        phase="ctx",
                        step=chunk_tokens,
                        bs=dp.batch_size,
                        past=chunk_past_kv,
                        run=run_idx,
                        live_step_driver=True,
                        sync_execute_model_wall_time=True,
                        requested_new_tokens=dp.new_tokens,
                        requested_past_kv=dp.past_kv,
                        chunk_index=chunk_index,
                        chunk_count=len(chunks),
                        scheduler_token_budget=scheduler_token_budget,
                    )
                marker_mod._LAST_CTX_MATCH_META = {}
                max_steps = 8
                for step_idx in range(max_steps):
                    llm.llm_engine.step()
                    executed_steps += 1
                    if warmup:
                        break
                    matched_after = dict(getattr(marker_mod, "_LAST_CTX_MATCH_META", {}))
                    if matched_after:
                        break
                else:
                    raise RuntimeError(
                        "live ctx driver did not capture target chunk "
                        f"for {dp.shape_key}: chunk_index={chunk_index}, "
                        f"chunk_tokens={chunk_tokens}, chunk_past_kv={chunk_past_kv}"
                    )
        if not warmup:
            _worker_append_event(
                status_path,
                "live_step_wall_time",
                work_unit_id=work_unit_id,
                datapoint_id=datapoint_id,
                phase="ctx",
                batch_size=dp.batch_size,
                new_tokens=dp.new_tokens,
                past_kv=dp.past_kv,
                run=run_idx,
                live_step_driver=True,
                chunks=len(chunks),
                steps=executed_steps,
                wall_latency_ms=(time.perf_counter() - live_start) * 1000.0,
            )
    finally:
        _set_marker_state(marker_mod, active_iterations="", phase="ctx")
        _abort_engine_requests(llm, request_ids)


def _live_decode_past_lengths(dp: DataPoint) -> list[int]:
    """Return per-request past lengths for live-decode diagnostics."""

    raw_offsets = os.environ.get("LAYERWISE_LIVE_DECODE_PAST_OFFSETS", "").strip()
    if not raw_offsets:
        return [dp.past_kv] * dp.batch_size
    offsets = [int(part.strip()) for part in raw_offsets.split(",") if part.strip()]
    if len(offsets) != dp.batch_size:
        raise ValueError(
            "LAYERWISE_LIVE_DECODE_PAST_OFFSETS must contain exactly "
            f"{dp.batch_size} values for batch_size={dp.batch_size}, got {len(offsets)}"
        )
    input_lens = [dp.past_kv + offset for offset in offsets]
    if any(length < 1 for length in input_lens):
        raise ValueError(f"live decode input lengths must be positive, got {input_lens}")
    return input_lens


def _gen_cache_salt_prefix(work_unit_id: str, dp: DataPoint) -> str:
    return f"layerwise-gen:{work_unit_id}"


def _gen_prefix_stream_key(work_unit_id: str) -> tuple[Any, ...]:
    """Return the token stream key shared by prefix-cache decode prompts."""

    return ("gen-prefix", work_unit_id)


def _run_prefix_cached_gen_iteration(
    llm,
    fill_sampling_params,
    measure_sampling_params,
    dp: DataPoint,
    *,
    token_config: RandomPromptTokenConfig,
    prompt_factory: PromptTokenFactory,
    fill_cache: bool,
    prompt_cache: PromptBatchCache | None = None,
    status_path: Path | None = None,
    datapoint_id: str | None = None,
    run_idx: int | None = None,
) -> None:
    if dp.past_kv <= 0:
        raise ValueError("prefix-cache decode requires past_kv > 0")
    work_unit_id = os.environ.get("LAYERWISE_WORK_UNIT_ID", "")
    cache_salt_prefix = _gen_cache_salt_prefix(work_unit_id, dp)
    stream_key_prefix = _gen_prefix_stream_key(work_unit_id)
    prompt_cache_key = ("gen", int(dp.batch_size), int(dp.past_kv), cache_salt_prefix)
    if fill_cache:
        _run_generate(
            llm,
            fill_sampling_params,
            batch_size=dp.batch_size,
            input_len=dp.past_kv,
            token_config=token_config,
            prompt_factory=prompt_factory,
            stream_key_prefix=stream_key_prefix,
            cache_salt_prefix=cache_salt_prefix,
            prompt_cache=prompt_cache,
            prompt_cache_key=prompt_cache_key,
            status_path=status_path,
            work_unit_id=work_unit_id,
            datapoint_id=datapoint_id,
            timing_phase="gen_fill",
            timing_batch_size=dp.batch_size,
            timing_past_kv=dp.past_kv,
            timing_run=run_idx,
        )
        return
    _run_generate(
        llm,
        measure_sampling_params,
        batch_size=dp.batch_size,
        input_len=dp.past_kv,
        token_config=token_config,
        prompt_factory=prompt_factory,
        stream_key_prefix=stream_key_prefix,
        cache_salt_prefix=cache_salt_prefix,
        prompt_cache=prompt_cache,
        prompt_cache_key=prompt_cache_key,
        status_path=status_path,
        work_unit_id=work_unit_id,
        datapoint_id=datapoint_id,
        timing_phase="gen",
        timing_batch_size=dp.batch_size,
        timing_past_kv=dp.past_kv,
        timing_run=run_idx,
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
    prompt_factory: PromptTokenFactory,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    gen_driver: str = "prefix_cache",
    max_num_seqs: int | None = None,
) -> None:
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1, got {measured_runs}")
    if gen_driver == "live_decode":
        _worker_run_gen_live_decode(
            status_path,
            work_unit_id,
            llm,
            sampling_cls,
            marker_mod,
            datapoints,
            prompt_token_config=prompt_token_config,
            prompt_factory=prompt_factory,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            max_num_seqs=max_num_seqs,
        )
        return
    if gen_driver != "prefix_cache":
        raise ValueError(f"unsupported gen_driver: {gen_driver}")

    by_batch: dict[int, list[DataPoint]] = {}
    for dp in datapoints:
        by_batch.setdefault(dp.batch_size, []).append(dp)
    max_past = max(dp.past_kv for dp in datapoints)
    sampling_params = sampling_cls(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=LIVE_DECODE_OUTPUT_TOKENS,
        detokenize=False,
    )
    fill_sampling_params = sampling_cls(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=1,
        detokenize=False,
    )
    prompt_cache = PromptBatchCache()
    prefix_cached_points = [dp for dp in datapoints if not _use_live_step_driver(dp)]
    if prefix_cached_points:
        max_batch_size = max(dp.batch_size for dp in prefix_cached_points)
        max_prefix_past = max(dp.past_kv for dp in prefix_cached_points)
        fill_dp = DataPoint("gen", max_batch_size, 1, max_prefix_past)
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _run_prefix_cached_gen_iteration(
            llm,
            fill_sampling_params,
            sampling_params,
            fill_dp,
            token_config=prompt_token_config,
            prompt_factory=prompt_factory,
            fill_cache=True,
            prompt_cache=prompt_cache,
            status_path=status_path,
            datapoint_id=_worker_datapoint_id(work_unit_id, fill_dp),
        )
        prompt_cache.clear()
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
                if _use_live_step_driver(dp):
                    _worker_append_event(
                        status_path,
                        "live_step_driver_started",
                        work_unit_id=work_unit_id,
                        datapoint_id=_worker_datapoint_id(work_unit_id, dp),
                        phase="gen",
                        batch_size=dp.batch_size,
                        past_kv=dp.past_kv,
                        measured_runs=measured_runs,
                    )
                    _run_live_gen_datapoint(
                        llm,
                        sampling_cls,
                        marker_mod,
                        dp,
                        work_unit_id=work_unit_id,
                        token_config=prompt_token_config,
                        prompt_factory=prompt_factory,
                        warmup_runs=warmup_runs,
                        measured_runs=measured_runs,
                        status_path=status_path,
                        datapoint_id=_worker_datapoint_id(work_unit_id, dp),
                    )
                    prompt_cache.clear()
                    continue
                _set_marker_state(marker_mod, active_iterations="", phase="gen")
                for _ in range(warmup_runs):
                    _set_marker_state(marker_mod, active_iterations="", phase="gen")
                    _run_prefix_cached_gen_iteration(
                        llm,
                        fill_sampling_params,
                        sampling_params,
                        dp,
                        token_config=prompt_token_config,
                        prompt_factory=prompt_factory,
                        fill_cache=False,
                        prompt_cache=prompt_cache,
                        status_path=status_path,
                        datapoint_id=_worker_datapoint_id(work_unit_id, dp),
                    )
                for run_idx in range(measured_runs):
                    _set_marker_state(
                        marker_mod,
                        active_iterations="",
                        trigger="decode_only",
                        allow_new_cached=True,
                        allow_partial_decode=os.environ.get("LAYERWISE_ALLOW_PARTIAL_DECODE", "0") == "1",
                        phase="gen",
                        step=dp.past_kv + 1,
                        bs=batch_size,
                        past=dp.past_kv,
                        run=run_idx,
                        measure_execute_model_gpu_time=True,
                    )
                    _run_prefix_cached_gen_iteration(
                        llm,
                        fill_sampling_params,
                        sampling_params,
                        dp,
                        token_config=prompt_token_config,
                        prompt_factory=prompt_factory,
                        fill_cache=False,
                        prompt_cache=prompt_cache,
                        status_path=status_path,
                        datapoint_id=_worker_datapoint_id(work_unit_id, dp),
                        run_idx=run_idx,
                    )
                prompt_cache.clear()
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)


def _run_live_gen_datapoint(
    llm,
    sampling_cls,
    marker_mod,
    dp: DataPoint,
    *,
    work_unit_id: str,
    token_config: RandomPromptTokenConfig,
    prompt_factory: PromptTokenFactory,
    warmup_runs: int,
    measured_runs: int,
    status_path: Path | None = None,
    datapoint_id: str | None = None,
) -> None:
    """Measure decode rows by keeping one vLLM request batch live."""

    total_decode_steps = int(warmup_runs) + int(measured_runs)
    if total_decode_steps < 1:
        return
    decode_headroom = 64 if int(dp.past_kv) <= 16384 else 1
    past_tolerance = _live_decode_past_tolerance(dp, total_decode_steps, decode_headroom)
    sampling_params = _make_final_only_sampling_params(
        sampling_cls,
        temperature=0.0,
        ignore_eos=True,
        max_tokens=total_decode_steps + decode_headroom,
        detokenize=False,
    )
    request_ids: list[str] = []
    primed_past_kv = int(dp.past_kv)
    try:
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        prefill_start = time.perf_counter()
        retry_past_kv = _short_retry_past_kv(dp, past_tolerance)
        retry_candidates = [int(dp.past_kv)]
        if retry_past_kv < int(dp.past_kv):
            retry_candidates.append(retry_past_kv)
        prime_chunks = 0
        last_error: RuntimeError | None = None
        for candidate_past_kv in retry_candidates:
            prompts = _token_prompts(
                dp.batch_size,
                candidate_past_kv,
                token_config,
                prompt_factory=prompt_factory,
                stream_key_prefix=_gen_prefix_stream_key(work_unit_id),
                cache_salt_prefix=_gen_cache_salt_prefix(work_unit_id, dp),
            )
            request_ids = _add_engine_requests(
                llm,
                prompts,
                sampling_params,
                request_prefix=f"gen-live:{work_unit_id}:bs{dp.batch_size}:past{candidate_past_kv}",
            )
            try:
                prime_chunks = _prime_engine_requests_for_decode(llm, request_ids, candidate_past_kv)
                primed_past_kv = candidate_past_kv
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
                if candidate_past_kv == int(dp.past_kv) and retry_past_kv < int(dp.past_kv) and _is_oom_text(str(exc)):
                    _abort_engine_requests(llm, request_ids)
                    request_ids = []
                    if status_path is not None and work_unit_id is not None:
                        _worker_append_event(
                            status_path,
                            "live_decode_prefix_retry",
                            work_unit_id=work_unit_id,
                            datapoint_id=datapoint_id,
                            phase="gen_prefill",
                            batch_size=dp.batch_size,
                            requested_past_kv=dp.past_kv,
                            retry_past_kv=retry_past_kv,
                            past_tolerance=past_tolerance,
                            reason="kv_block_exhaustion",
                        )
                    continue
                raise
        if last_error is not None:
            raise last_error
        if status_path is not None and work_unit_id is not None:
            _worker_append_event(
                status_path,
                "synthetic_decode_prefill_ready",
                work_unit_id=work_unit_id,
                datapoint_id=datapoint_id,
                phase="gen_prefill",
                batch_size=dp.batch_size,
                past_kv=dp.past_kv,
                primed_past_kv=primed_past_kv,
                prime_chunks=prime_chunks,
                wall_latency_ms=(time.perf_counter() - prefill_start) * 1000.0,
            )

        for _ in range(warmup_runs):
            _set_marker_state(marker_mod, active_iterations="", phase="gen")
            llm.llm_engine.step()

        for run_idx in range(measured_runs):
            _set_marker_state(
                marker_mod,
                active_iterations="",
                trigger="decode_only",
                allow_new_cached=False,
                allow_partial_decode=os.environ.get("LAYERWISE_ALLOW_PARTIAL_DECODE", "0") == "1",
                allow_variable_past=True,
                past_tolerance=past_tolerance,
                match_once=True,
                phase="gen",
                step=dp.past_kv + 1,
                bs=dp.batch_size,
                past=dp.past_kv,
                run=run_idx,
                live_step_driver=True,
            )
            live_start = time.perf_counter()
            marker_mod._LAST_DECODE_MATCH_META = {}
            max_steps = max(4, total_decode_steps + 4)
            for step_idx in range(max_steps):
                llm.llm_engine.step()
                matched_after = dict(getattr(marker_mod, "_LAST_DECODE_MATCH_META", {}))
                if matched_after:
                    if status_path is not None and work_unit_id is not None:
                        _worker_append_event(
                            status_path,
                            "live_step_wall_time",
                            work_unit_id=work_unit_id,
                            datapoint_id=datapoint_id,
                            phase="gen",
                            batch_size=dp.batch_size,
                            past_kv=dp.past_kv,
                            run=run_idx,
                            steps=step_idx + 1,
                            wall_latency_ms=(time.perf_counter() - live_start) * 1000.0,
                        )
                    break
            else:
                raise RuntimeError(f"live gen driver did not capture target decode step for {dp.shape_key}")
    finally:
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _abort_engine_requests(llm, request_ids)


def _worker_run_gen_live_decode(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    marker_mod,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
    prompt_factory: PromptTokenFactory,
    warmup_runs: int = 0,
    measured_runs: int = 1,
    max_num_seqs: int | None = None,
) -> None:
    """Measure decode from a live generation stream without prefix caching."""

    sampling_params = sampling_cls(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=LIVE_DECODE_OUTPUT_TOKENS,
        detokenize=False,
    )
    sorted_datapoints = sorted(datapoints, key=lambda item: (item.batch_size, item.past_kv))
    for dp in sorted_datapoints:
        live_batch_size = dp.batch_size
        live_input_lens = _live_decode_past_lengths(dp)
        _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(
            status_path,
            "batch_started",
            work_unit_id=work_unit_id,
            batch_size=dp.batch_size,
            live_batch_size=live_batch_size,
            max_past_kv=dp.past_kv,
            live_past_min=min(live_input_lens),
            live_past_max=max(live_input_lens),
            live_past_mean=sum(live_input_lens) / len(live_input_lens),
            gen_driver="live_decode",
        )
        try:
            for run_idx in range(warmup_runs):
                _set_marker_state(marker_mod, active_iterations="", phase="gen")
                _run_generate_variable_lengths(
                    llm,
                    sampling_params,
                    input_lens=live_input_lens,
                    token_config=prompt_token_config,
                    prompt_factory=prompt_factory,
                )
            for run_idx in range(measured_runs):
                _set_marker_state(
                    marker_mod,
                    active_iterations="",
                    trigger="decode_only",
                    allow_new_cached=False,
                    allow_partial_decode=os.environ.get("LAYERWISE_ALLOW_PARTIAL_DECODE", "0") == "1",
                    allow_variable_past=True,
                    past_tolerance=4.0,
                    match_once=True,
                    phase="gen",
                    step=dp.past_kv + 1,
                    bs=dp.batch_size,
                    past=dp.past_kv,
                    run=run_idx,
                )
                _run_generate_variable_lengths(
                    llm,
                    sampling_params,
                    input_lens=live_input_lens,
                    token_config=prompt_token_config,
                    prompt_factory=prompt_factory,
                )
        finally:
            _set_marker_state(marker_mod, active_iterations="", phase="gen")
        _worker_append_event(
            status_path,
            "batch_finished",
            work_unit_id=work_unit_id,
            batch_size=dp.batch_size,
            gen_driver="live_decode",
        )


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
