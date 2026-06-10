"""NVTX step-marker: wraps `GPUModelRunner.execute_model` at target iteration
numbers so that per-step kernel attribution can be sliced from the nsys trace.

Run style:
  - Submit bs=128 requests with isl=1, max_tokens=8192.
  - vLLM engine schedules step 1 = prefill (bs=128 x 1 token), step k (k>=2) =
    pure decode (bs=128, past_kv = k-1).
  - At each target iteration we push an outer NVTX range:
      bench_step::N<NNNNNNN>::bs<B>::past<PPPPPP>
    Example: `bench_step::N0000016::bs128::past000015`.
  - vLLM's layerwise NVTX hooks (enabled by `--enable-layerwise-nvtx-tracing`)
    push their own inner `{'Module': '...'}` ranges; our outer range becomes a
    parent. The sweep parser can then attribute kernels per (step, module).

Env:
  LAYERWISE_STEP_ITERATIONS="1,16,32,64,128,256,512,1024,2048,4096,8192"
                             comma list of step numbers to mark (1-indexed)
  LAYERWISE_ACTIVE_ITERATIONS="1,16"
                             optional runtime subset of configured iterations.
                             A shared vLLM engine can switch this between ctx
                             and gen phases without reinstalling the wrapper.
  LAYERWISE_STEP_MARKER=0    disable
  LAYERWISE_BENCH_MIN_NEW=2  min `scheduled_new_reqs` to treat a call as the
                             start of a real bench iteration (resets counter).
                             Keeps vLLM warmup / profile-run (new_reqs=1) from
                             colliding with the real bs=N prefill at step 1.
  LAYERWISE_BENCH_SKIP_STARTS=0
                             number of matching starts to ignore before
                             counting. Useful when collecting bs=1.

Non-target iterations run as-is (no outer marker). Keeps nsys overhead low
outside the sweep points.
"""
import fcntl
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)

_DEFAULT_ITERATIONS = "1,16,32,64,128,256,512,1024,2048,4096,8192"
_FORCED_STEP_META = {"step": None, "bs": None, "past": None, "run": None}


def _read_control() -> dict:
    path = os.environ.get("LAYERWISE_CONTROL_FILE")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("[step-marker] failed to read control file %s", path)
        return {}


def set_forced_step_meta(step=None, bs=None, past=None, run=None):
    """Override the next marked step label in this process.

    The batched sweep harness uses this for context shapes: every prefill is
    internally step 1, so the label's `N...` field is set to `new_tokens` to
    make parser keys unique without changing the sqlite schema.
    """
    _FORCED_STEP_META.update({"step": step, "bs": bs, "past": past, "run": run})


def clear_forced_step_meta():
    """Clear any per-process marker label override."""

    _FORCED_STEP_META.update({"step": None, "bs": None, "past": None, "run": None})


def _progress_datapoint_id(work_unit_id, phase, batch_size, step, past_kv):
    new_tokens = step if phase == "ctx" else 1
    return f"{work_unit_id}:{phase}:bs{batch_size}:new{new_tokens}:past{past_kv}"


def _write_progress(event, *, step, batch_size, past_kv, phase=None):
    """Append scheduler progress for iteration-level crash attribution.

    Generation intentionally runs many past_kv datapoints in one generate()
    call.  These events let the parent identify the active datapoint if that
    call dies halfway through.
    """
    path = os.environ.get("LAYERWISE_PROGRESS_FILE")
    work_unit_id = os.environ.get("LAYERWISE_WORK_UNIT_ID")
    phase = phase or os.environ.get("LAYERWISE_PROGRESS_PHASE")
    if not path or not work_unit_id or not phase:
        return
    row = {
        "event": event,
        "work_unit_id": work_unit_id,
        "datapoint_id": _progress_datapoint_id(
            work_unit_id, phase, batch_size, step, past_kv
        ),
        "phase": phase,
        "batch_size": int(batch_size),
        "new_tokens": int(step if phase == "ctx" else 1),
        "past_kv": int(past_kv),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _parse_iterations() -> set[int]:
    raw = os.environ.get("LAYERWISE_STEP_ITERATIONS", _DEFAULT_ITERATIONS)
    return {int(x) for x in raw.split(",") if x.strip()}


def _parse_active_iterations(default: set[int], control: dict | None = None) -> set[int]:
    if control:
        raw_control = control.get("active_iterations")
        if raw_control is not None:
            if isinstance(raw_control, str):
                return {int(x) for x in raw_control.split(",") if x.strip()}
            return {int(x) for x in raw_control}
    raw = os.environ.get("LAYERWISE_ACTIVE_ITERATIONS")
    if raw is None:
        return default
    return {int(x) for x in raw.split(",") if x.strip()}


def _request_prompt_len(req) -> int | None:
    raw = getattr(req, "num_prompt_tokens", None)
    if raw is not None:
        return int(raw)
    prompt_token_ids = getattr(req, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        return len(prompt_token_ids)
    prompt_embeds = getattr(req, "prompt_embeds", None)
    if prompt_embeds is not None:
        return len(prompt_embeds)
    return None


def _cached_num_computed_tokens(scheduler_output) -> dict[str, int]:
    cached = scheduler_output.scheduled_cached_reqs
    return {
        req_id: int(num_computed)
        for req_id, num_computed in zip(cached.req_ids, cached.num_computed_tokens)
    }


def _decode_only_match(runner, scheduler_output, control: dict) -> tuple[bool, int, int, int]:
    """Return whether this scheduler iteration is the target pure-decode step.

    vLLM may split prefill across several scheduler iterations.  We only mark
    the first decode step for the requested past_kv, where every scheduled
    request has already computed exactly the target prompt length before the
    step starts.
    """

    scheduled = scheduler_output.num_scheduled_tokens
    allow_new_cached = bool(control.get("allow_new_cached"))
    if not scheduled:
        return False, 0, 0, 0
    if scheduler_output.scheduled_new_reqs and not allow_new_cached:
        return False, 0, 0, 0

    target_bs = control.get("bs")
    target_past = control.get("past")
    if target_bs is not None and len(scheduled) != int(target_bs):
        return False, 0, 0, 0

    computed_by_req = _cached_num_computed_tokens(scheduler_output)
    request_by_req = dict(getattr(runner, "requests", {}))
    if allow_new_cached:
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = getattr(new_req, "req_id", None)
            if req_id is None:
                continue
            request_by_req[req_id] = new_req
            if req_id not in computed_by_req:
                computed_by_req[req_id] = int(getattr(new_req, "num_computed_tokens", 0))

    past_values = []
    for req_id, num_tokens in scheduled.items():
        if int(num_tokens) != 1:
            return False, 0, 0, 0
        req = request_by_req.get(req_id)
        if req is None:
            return False, 0, 0, 0
        prompt_len = _request_prompt_len(req)
        if prompt_len is None:
            return False, 0, 0, 0
        computed = computed_by_req.get(req_id, getattr(req, "num_computed_tokens", None))
        if computed is None:
            return False, 0, 0, 0
        computed = int(computed)
        if target_past is not None and computed != int(target_past):
            return False, 0, 0, 0
        if computed < int(prompt_len):
            return False, 0, 0, 0
        past_values.append(computed)

    if not past_values or len(set(past_values)) != 1:
        return False, 0, 0, 0
    past_kv = past_values[0] if target_past is None else int(target_past)
    batch_size = len(scheduled)
    step = past_kv + 1
    return True, step, batch_size, past_kv


def _run_marked_step(
    orig,
    runner,
    scheduler_output,
    intermediate_tensors,
    *,
    step: int,
    batch_size: int,
    past_kv: int,
    control: dict,
):
    forced_step = control.get("step", _FORCED_STEP_META["step"])
    forced_bs = control.get("bs", _FORCED_STEP_META["bs"])
    forced_past = control.get("past", _FORCED_STEP_META["past"])
    forced_run = control.get("run", _FORCED_STEP_META["run"])
    if forced_run is None and os.environ.get("LAYERWISE_MEASURE_RUN") is not None:
        forced_run = int(os.environ["LAYERWISE_MEASURE_RUN"])
    forced_phase = control.get("phase")
    label_step = step if forced_step is None else int(forced_step)
    label_bs = batch_size if forced_bs is None else int(forced_bs)
    label_past = past_kv if forced_past is None else int(forced_past)
    label = f"bench_step::N{label_step:07d}::bs{label_bs}::past{label_past:06d}"
    if forced_run is not None:
        label += f"::run{int(forced_run):03d}"
    _write_progress(
        "started",
        step=label_step,
        batch_size=label_bs,
        past_kv=label_past,
        phase=forced_phase,
    )
    nvtx.range_push(label)
    try:
        ret = orig(runner, scheduler_output, intermediate_tensors)
        _write_progress(
            "completed_execution",
            step=label_step,
            batch_size=label_bs,
            past_kv=label_past,
            phase=forced_phase,
        )
        return ret
    finally:
        nvtx.range_pop()


def _install():
    if os.environ.get("LAYERWISE_STEP_MARKER", "1") != "1":
        logger.info("[step-marker] disabled via LAYERWISE_STEP_MARKER=0")
        return

    iterations = _parse_iterations()
    if not iterations:
        logger.info("[step-marker] no target iterations configured; no-op")
        return

    from vllm.v1.worker import gpu_model_runner as _gmr

    orig = _gmr.GPUModelRunner.execute_model
    state = {"n": 0, "started": False}
    min_new = int(os.environ.get("LAYERWISE_BENCH_MIN_NEW", "2"))
    skip_starts = int(os.environ.get("LAYERWISE_BENCH_SKIP_STARTS", "0"))

    def patched(self, scheduler_output, intermediate_tensors=None):
        """Wrapped execute_model that emits NVTX markers for selected steps."""

        control = _read_control()
        if control.get("trigger") == "decode_only":
            matched, step, batch_size, past_kv = _decode_only_match(
                self, scheduler_output, control
            )
            if not matched:
                return orig(self, scheduler_output, intermediate_tensors)
            return _run_marked_step(
                orig,
                self,
                scheduler_output,
                intermediate_tensors,
                step=step,
                batch_size=batch_size,
                past_kv=past_kv,
                control=control,
            )

        # Ignore pre-bench calls (profile_run, single-req sanity) entirely —
        # only start counting once we see a prefill with ≥ min_new new reqs.
        nonlocal skip_starts
        num_new = len(scheduler_output.scheduled_new_reqs)
        if num_new >= min_new:
            if skip_starts > 0:
                skip_starts -= 1
                state["n"] = 0
                state["started"] = False
                return orig(self, scheduler_output, intermediate_tensors)
            state["n"] = 1
            state["started"] = True
        elif state["started"]:
            state["n"] += 1
        else:
            return orig(self, scheduler_output, intermediate_tensors)
        n = state["n"]
        if n not in _parse_active_iterations(iterations, control):
            return orig(self, scheduler_output, intermediate_tensors)

        num_reqs = (
            len(scheduler_output.scheduled_new_reqs)
            + len(scheduler_output.scheduled_cached_reqs.req_ids)
        )
        # past_kv at this step = n - 1 under isl=1 driver
        # (step 1 = prefill, past_kv=0; step k = decode, past_kv=k-1).
        past_kv = n - 1
        return _run_marked_step(
            orig,
            self,
            scheduler_output,
            intermediate_tensors,
            step=n,
            batch_size=num_reqs,
            past_kv=past_kv,
            control=control,
        )

    _gmr.GPUModelRunner.execute_model = patched
    logger.warning(
        f"[step-marker] installed GPUModelRunner.execute_model wrapper, "
        f"iterations={sorted(iterations)}"
    )


_install()
