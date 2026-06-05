"""NVTX step-marker: wraps `GPUModelRunner.execute_model` at milestone step
numbers so that per-step kernel attribution can be sliced from the nsys trace.

Run style:
  - Submit bs=128 requests with isl=1, max_tokens=8192.
  - vLLM engine schedules step 1 = prefill (bs=128 x 1 token), step k (k>=2) =
    pure decode (bs=128, past_kv = k-1).
  - At each milestone step we push an outer NVTX range:
      bench_step::N<NNNNNNN>::bs<B>::past<PPPPPP>
    Example: `bench_step::N0000016::bs128::past000015`.
  - vLLM's layerwise NVTX hooks (enabled by `--enable-layerwise-nvtx-tracing`)
    push their own inner `{'Module': '...'}` ranges; our outer range becomes a
    parent. The sweep parser can then attribute kernels per (step, module).

Env:
  LAYERWISE_STEP_MILESTONES="1,16,32,64,128,256,512,1024,2048,4096,8192"
                             comma list of step numbers to mark (1-indexed)
  LAYERWISE_ACTIVE_MILESTONES="1,16"
                             optional runtime subset of configured milestones.
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

Non-milestone steps run as-is (no outer marker). Keeps nsys overhead low
outside the sweep points.
"""
import fcntl
import json
import logging
import os
from datetime import datetime, timezone

import torch.cuda.nvtx as nvtx

logger = logging.getLogger(__name__)

_DEFAULT_MILESTONES = "1,16,32,64,128,256,512,1024,2048,4096,8192"
_FORCED_STEP_META = {"step": None, "bs": None, "past": None, "run": None}


def set_forced_step_meta(step=None, bs=None, past=None, run=None):
    """Override the next marked step label in this process.

    The batched sweep harness uses this for context shapes: every prefill is
    internally step 1, so the label's `N...` field is set to `new_tokens` to
    make parser keys unique without changing the sqlite schema.
    """
    _FORCED_STEP_META.update({"step": step, "bs": bs, "past": past, "run": run})


def clear_forced_step_meta():
    _FORCED_STEP_META.update({"step": None, "bs": None, "past": None, "run": None})


def _progress_datapoint_id(work_unit_id, phase, batch_size, step, past_kv):
    new_tokens = step if phase == "ctx" else 1
    return f"{work_unit_id}:{phase}:bs{batch_size}:new{new_tokens}:past{past_kv}"


def _write_progress(event, *, step, batch_size, past_kv):
    """Append scheduler progress for milestone-level crash attribution.

    Generation intentionally runs many past_kv datapoints in one generate()
    call.  These events let the parent identify the active datapoint if that
    call dies halfway through.
    """
    path = os.environ.get("LAYERWISE_PROGRESS_FILE")
    work_unit_id = os.environ.get("LAYERWISE_WORK_UNIT_ID")
    phase = os.environ.get("LAYERWISE_PROGRESS_PHASE")
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


def _parse_milestones() -> set[int]:
    raw = os.environ.get("LAYERWISE_STEP_MILESTONES", _DEFAULT_MILESTONES)
    return {int(x) for x in raw.split(",") if x.strip()}


def _parse_active_milestones(default: set[int]) -> set[int]:
    raw = os.environ.get("LAYERWISE_ACTIVE_MILESTONES")
    if raw is None:
        return default
    return {int(x) for x in raw.split(",") if x.strip()}


def _install():
    if os.environ.get("LAYERWISE_STEP_MARKER", "1") != "1":
        logger.info("[step-marker] disabled via LAYERWISE_STEP_MARKER=0")
        return

    milestones = _parse_milestones()
    if not milestones:
        logger.info("[step-marker] no milestones configured; no-op")
        return

    from vllm.v1.worker import gpu_model_runner as _gmr

    orig = _gmr.GPUModelRunner.execute_model
    state = {"n": 0, "started": False}
    min_new = int(os.environ.get("LAYERWISE_BENCH_MIN_NEW", "2"))
    skip_starts = int(os.environ.get("LAYERWISE_BENCH_SKIP_STARTS", "0"))

    def patched(self, scheduler_output, intermediate_tensors=None):
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
        if n not in _parse_active_milestones(milestones):
            return orig(self, scheduler_output, intermediate_tensors)

        num_reqs = (
            len(scheduler_output.scheduled_new_reqs)
            + len(scheduler_output.scheduled_cached_reqs.req_ids)
        )
        # past_kv at this step = n - 1 under isl=1 driver
        # (step 1 = prefill, past_kv=0; step k = decode, past_kv=k-1).
        past_kv = n - 1
        forced_step = _FORCED_STEP_META["step"]
        forced_bs = _FORCED_STEP_META["bs"]
        forced_past = _FORCED_STEP_META["past"]
        forced_run = _FORCED_STEP_META["run"]
        label_step = n if forced_step is None else int(forced_step)
        label_bs = num_reqs if forced_bs is None else int(forced_bs)
        label_past = past_kv if forced_past is None else int(forced_past)
        label = f"bench_step::N{label_step:07d}::bs{label_bs}::past{label_past:06d}"
        if forced_run is not None:
            label += f"::run{int(forced_run):03d}"
        _write_progress(
            "started",
            step=label_step,
            batch_size=label_bs,
            past_kv=label_past,
        )
        nvtx.range_push(label)
        try:
            ret = orig(self, scheduler_output, intermediate_tensors)
            _write_progress(
                "completed_execution",
                step=label_step,
                batch_size=label_bs,
                past_kv=label_past,
            )
            return ret
        finally:
            nvtx.range_pop()

    _gmr.GPUModelRunner.execute_model = patched
    logger.warning(
        f"[step-marker] installed GPUModelRunner.execute_model wrapper, "
        f"milestones={sorted(milestones)}"
    )


_install()
