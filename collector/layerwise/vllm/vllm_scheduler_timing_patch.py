"""Record vLLM scheduler update timing for layerwise/FPM comparisons.

The Dynamo FPM collector reports the wall interval between scheduler
``update_from_output`` calls for steady-state batches and falls back to
``schedule`` to ``update_from_output`` for the first batch after the scheduler
has been idle.  The layerwise nsys parser measures CUDA graph wrapper spans
instead.  This patch records both envelopes so context collection can use the
same timing boundary as FPM while retaining schedule-to-update diagnostics.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_control() -> dict[str, Any]:
    path = os.environ.get("LAYERWISE_CONTROL_FILE")
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("[scheduler-timing] failed to read control file %s", path)
        return {}


def _append_event(event: dict[str, Any]) -> None:
    path = os.environ.get("LAYERWISE_PROGRESS_FILE")
    work_unit_id = os.environ.get("LAYERWISE_WORK_UNIT_ID")
    if not path or not work_unit_id:
        return
    row = {
        "event": "scheduler_update_wall_time",
        "work_unit_id": work_unit_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


def _scheduled_count(scheduler_output: Any, attr: str) -> int:
    value = getattr(scheduler_output, attr, None)
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return int(bool(value))


def _install_patch() -> None:
    if os.environ.get("LAYERWISE_SCHEDULER_TIMING", "0") != "1":
        return

    from vllm.v1.core.sched import scheduler as _scheduler

    orig_update = _scheduler.Scheduler.update_from_output
    if getattr(orig_update, "_layerwise_scheduler_timing_patch", False):
        return
    orig_schedule = _scheduler.Scheduler.schedule

    def patched_schedule(self, *args, **kwargs):
        scheduler_output = orig_schedule(self, *args, **kwargs)
        schedule_times = getattr(self, "_layerwise_schedule_times", None)
        if schedule_times is None:
            schedule_times = {}
            setattr(self, "_layerwise_schedule_times", schedule_times)
        schedule_times[id(scheduler_output)] = time.monotonic()
        return scheduler_output

    def patched(self, scheduler_output, model_runner_output):
        schedule_times = getattr(self, "_layerwise_schedule_times", {})
        scheduled_at = schedule_times.pop(id(scheduler_output), None)

        scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        total_tokens = sum(int(v) for v in scheduled.values())
        engine_outputs = orig_update(self, scheduler_output, model_runner_output)
        now = time.monotonic()
        control = _read_control()
        last = getattr(self, "_layerwise_last_update_time", 0.0)
        if total_tokens > 0:
            if last > 0.0:
                fpm_wall_ms = (now - last) * 1000.0
            elif scheduled_at is not None:
                fpm_wall_ms = (now - scheduled_at) * 1000.0
            else:
                fpm_wall_ms = None
            event = {
                "scheduled_tokens": total_tokens,
                "scheduled_requests": len(scheduled),
                "scheduled_new_reqs": _scheduled_count(scheduler_output, "scheduled_new_reqs"),
                "scheduled_cached_reqs": _scheduled_count(scheduler_output, "scheduled_cached_reqs"),
                "control_phase": control.get("phase"),
                "control_step": control.get("step"),
                "control_bs": control.get("bs"),
                "control_past": control.get("past"),
                "control_run": control.get("run"),
            }
            if fpm_wall_ms is not None:
                event["fpm_wall_time_ms"] = fpm_wall_ms
                event["wall_latency_ms"] = fpm_wall_ms
            if scheduled_at is not None:
                event["schedule_to_update_ms"] = (now - scheduled_at) * 1000.0
            _append_event(event)
            setattr(self, "_layerwise_last_update_time", now)
        else:
            setattr(self, "_layerwise_last_update_time", 0.0)
        return engine_outputs

    patched._layerwise_scheduler_timing_patch = True
    _scheduler.Scheduler.schedule = patched_schedule
    _scheduler.Scheduler.update_from_output = patched
    logger.warning("[scheduler-timing] installed Scheduler schedule/update patch")


_install_patch()
