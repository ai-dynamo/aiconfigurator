# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-slot scheduler, retry handling, and nsys result ingestion for layerwise runs."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from parse_nsys_step_sweep import parse_step_sweep
from vllm_deployment import gpt_oss_runtime_defaults, has_cli_flag

try:
    from .data import DataPoint, WorkUnit
    from .engine import _append_default_vllm_args
    from .nsys import _aggregate_step_rows, _effective_rollup, _lookup_aggs, _reduce_agg_latency
    from .results import _append_success_row, _work_unit_includes_moe, _write_csv_header_if_needed
    from .runtime import (
        _detect_gpus,
        _get_vllm_deployment_max_num_batched_tokens,
        _json_dump,
        _tail,
        _utc_now,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint, WorkUnit
    from engine import _append_default_vllm_args
    from nsys import _aggregate_step_rows, _effective_rollup, _lookup_aggs, _reduce_agg_latency
    from results import _append_success_row, _work_unit_includes_moe, _write_csv_header_if_needed
    from runtime import (
        _detect_gpus,
        _get_vllm_deployment_max_num_batched_tokens,
        _json_dump,
        _tail,
        _utc_now,
    )


TERMINAL_EVENTS = {
    "success", "failed_oom", "failed_error", "failed_fatal_cuda", "failed_parse",
    "skipped_oom_dominated", "skipped_same_error", "skipped_not_started",
}
FATAL_STREAK_LIMIT = 3
FPM_PORT_ENV = "DYN_FORWARDPASS_METRIC_PORT"
FPM_PORT_BASE = 20380


@dataclass
class Attempt:
    """Running worker subprocess and its profiling artifact paths."""

    work_unit: WorkUnit
    gpu: str
    attempt_id: int
    spec_path: Path
    report_base: Path
    stdout_path: Path
    stderr_path: Path
    process: subprocess.Popen
    stdout_handle: Any
    stderr_handle: Any
    pending_ids: set[str]

class StatusIndex:
    """Reconstructed view of the append-only status log."""

    def __init__(self, events: list[dict[str, Any]]):
        self.events = events
        self.terminal: dict[str, dict[str, Any]] = {}
        self.started: dict[str, list[dict[str, Any]]] = {}
        self.completed: set[str] = set()
        for event in events:
            dpid = event.get("datapoint_id")
            if not dpid:
                continue
            name = event.get("event")
            if name in {"started", "live_step_driver_started"}:
                self.started.setdefault(dpid, []).append(event)
            elif name == "completed_execution":
                self.completed.add(dpid)
            elif name in TERMINAL_EVENTS:
                self.terminal[dpid] = event

    def is_terminal(self, datapoint_id: str) -> bool:
        """Return whether a datapoint already has a terminal status event."""

        return datapoint_id in self.terminal

    def terminal_ids(self) -> set[str]:
        """Return all datapoints with terminal status events."""

        return set(self.terminal)

    def active_started(self, work_unit_id: str, pending_ids: set[str]) -> str | None:
        """Return newest started datapoint without a terminal event.

        This is the crash contract between worker and scheduler.  If the
        worker dies, the newest non-terminal started datapoint is considered
        the one that caused the crash and is never retried.
        """
        start_events = {"started", "live_step_driver_started"}
        for event in reversed(self.events):
            if event.get("event") not in start_events:
                continue
            if event.get("work_unit_id") != work_unit_id:
                continue
            dpid = event.get("datapoint_id")
            if dpid in pending_ids and dpid not in self.terminal:
                return dpid
        return None

class StatusStore:
    """Append-only manifest/status files shared by scheduler and workers."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.manifest_path = work_dir / "manifest.jsonl"
        self.status_path = work_dir / "status.jsonl"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def append_jsonl(self, path: Path, row: dict[str, Any]) -> None:
        """Append one locked JSONL row and fsync it for crash recovery."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def append_event(self, event: str, *, work_unit_id: str, datapoint_id: str | None = None, **extra: Any) -> None:
        """Append one scheduler or worker status event."""

        row = {
            "event": event,
            "work_unit_id": work_unit_id,
            "datapoint_id": datapoint_id,
            "ts": _utc_now(),
            **extra,
        }
        self.append_jsonl(self.status_path, {k: v for k, v in row.items() if v is not None})

    def load_events(self) -> list[dict[str, Any]]:
        """Load all status events written so far."""

        if not self.status_path.exists():
            return []
        events = []
        with self.status_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    def max_attempt_id(self) -> int:
        """Return the largest attempt ID already recorded in the status log."""

        max_id = 0
        for event in self.load_events():
            try:
                attempt_id = int(event.get("attempt_id", 0))
            except (TypeError, ValueError):
                continue
            max_id = max(max_id, attempt_id)
        return max_id

    def index(self) -> StatusIndex:
        """Return a reconstructed status index from the append-only log."""

        return StatusIndex(self.load_events())

    def existing_manifest_ids(self) -> set[str]:
        """Return datapoint IDs already present in the manifest."""

        if not self.manifest_path.exists():
            return set()
        ids = set()
        with self.manifest_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                ids.add(json.loads(line)["datapoint_id"])
        return ids

    def write_missing_manifest(self, work_units: Iterable[WorkUnit]) -> None:
        """Append manifest rows that are not already present."""

        seen = self.existing_manifest_ids()
        for unit in work_units:
            for row in unit.manifest_rows():
                if row["datapoint_id"] in seen:
                    continue
                self.append_jsonl(self.manifest_path, row)
                seen.add(row["datapoint_id"])

def _is_oom_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "out of memory" in lowered
        or "cuda oom" in lowered
        or "cublas_status_alloc_failed" in lowered
        or "insufficient kv blocks" in lowered
        or "failed to prime" in lowered and "prefix" in lowered and "free_blocks=0" in lowered
    )

def _is_fatal_cuda_text(text: str) -> bool:
    lowered = text.lower()
    return "illegal memory access" in lowered or "device-side assert" in lowered or "cuda error" in lowered

def _attempt_signature(returncode: int, stderr_tail: str) -> str:
    if _is_oom_text(stderr_tail):
        return "oom"
    if _is_fatal_cuda_text(stderr_tail):
        return "fatal_cuda"
    error_lines = [
        line.strip() for line in stderr_tail.splitlines()
        if any(marker in line for marker in ("Error:", "Exception:", "RuntimeError", "ValueError"))
    ]
    detail = error_lines[-1] if error_lines else stderr_tail.strip().splitlines()[-1:]
    if isinstance(detail, list):
        detail = detail[0] if detail else ""
    fingerprint = hashlib.sha1(str(detail).encode()).hexdigest()[:12]
    return f"exit_{returncode}:{fingerprint}:{str(detail)[:160]}"

def oom_dominates(failed: DataPoint, candidate: DataPoint) -> bool:
    """Return whether a failed OOM point should prune a candidate.

    OOM pruning is phase-local.  A ctx OOM says larger ctx tokens/past are
    unsafe; a gen OOM says larger batch/past points are unsafe.  It never
    prunes the other phase.
    """
    if failed.phase != candidate.phase:
        return False
    if failed.phase == "ctx":
        same_or_larger = (
            candidate.new_tokens >= failed.new_tokens
            and candidate.past_kv >= failed.past_kv
        )
        strictly_larger = (
            candidate.new_tokens > failed.new_tokens
            or candidate.past_kv > failed.past_kv
        )
        return same_or_larger and strictly_larger
    same_or_larger = (
        candidate.batch_size >= failed.batch_size
        and candidate.past_kv >= failed.past_kv
    )
    strictly_larger = (
        candidate.batch_size > failed.batch_size
        or candidate.past_kv > failed.past_kv
    )
    if same_or_larger and strictly_larger:
        return True

    if candidate.batch_size < failed.batch_size:
        return False
    failed_kv_footprint = int(failed.batch_size) * max(
        1,
        int(failed.past_kv) + max(1, int(failed.new_tokens)),
    )
    candidate_kv_footprint = int(candidate.batch_size) * max(
        1,
        int(candidate.past_kv) + max(1, int(candidate.new_tokens)),
    )
    return candidate_kv_footprint >= failed_kv_footprint

def _attempt_config_hash(attempt: Attempt, store: StatusStore) -> str:
    for event in reversed(store.index().events):
        if event.get("event") != "engine_metadata_written":
            continue
        if event.get("work_unit_id") != attempt.work_unit.work_unit_id:
            continue
        if event.get("attempt_id") != attempt.attempt_id:
            continue
        return str(event.get("vllm_config_hash") or "")
    return ""

def _attempt_max_num_batched_tokens(attempt: Attempt) -> int | None:
    """Return the context-token budget used by a launched attempt."""

    if attempt.work_unit.max_num_batched_tokens is not None:
        return attempt.work_unit.max_num_batched_tokens
    try:
        spec = json.loads(attempt.spec_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    value = spec.get("max_num_batched_tokens")
    if value in (None, ""):
        return None
    return int(value)

def _lookup_scheduler_timing_aggs(
    events: list[dict[str, Any]],
    work_unit_id: str,
    datapoint: DataPoint,
    *,
    attempt_id: int | None = None,
    prefer_schedule_to_update_for_gen: bool = False,
) -> list[dict[str, Any]]:
    """Return scheduler wall-envelope repeats for one datapoint.

    The scheduler timing patch records a Dynamo-FPM-style wall interval between
    non-empty ``update_from_output`` calls for the active marker-control shape,
    with schedule-to-update timing as a diagnostic fallback for older logs.
    Convert those rows to the same aggregate shape used by the nsys reducer so
    repeat aggregation remains shared.
    """

    step, batch_size, past_kv = datapoint.parse_key()
    aggs = []

    model_execute_aggs = []
    for event in events:
        if event.get("event") != "completed_execution":
            continue
        if event.get("work_unit_id") != work_unit_id:
            continue
        if datapoint.phase != "ctx":
            continue
        if not event.get("live_step_driver"):
            continue
        if not event.get("sync_execute_model_wall_time"):
            continue
        event_attempt_id = event.get("attempt_id")
        if attempt_id is not None and event_attempt_id not in (None, "") and event_attempt_id != attempt_id:
            continue
        if event.get("phase") != datapoint.phase:
            continue
        if int(event.get("batch_size", -1)) != int(batch_size):
            continue
        if int(event.get("past_kv", -1)) != int(past_kv):
            continue
        if datapoint.phase == "ctx" and int(event.get("new_tokens", -1)) != int(step):
            continue
        if event.get("run") in (None, ""):
            continue
        latency_ms = event.get("execute_model_wall_time_ms")
        if latency_ms in (None, ""):
            continue
        latency_us = float(latency_ms) * 1000.0
        model_execute_aggs.append({
            "gpu_us": latency_us,
            "rms_us": 0.0,
            "span_us": latency_us,
            "kernel_count": 0,
            "rms_kernel_count": 0,
        })
    if model_execute_aggs:
        return model_execute_aggs

    live_step_aggs = []
    for event in events:
        if event.get("event") != "live_step_wall_time":
            continue
        if event.get("work_unit_id") != work_unit_id:
            continue
        if datapoint.phase != "ctx":
            continue
        event_attempt_id = event.get("attempt_id")
        if attempt_id is not None and event_attempt_id not in (None, "") and event_attempt_id != attempt_id:
            continue
        if event.get("phase") != datapoint.phase:
            continue
        if int(event.get("batch_size", -1)) != int(batch_size):
            continue
        if int(event.get("past_kv", -1)) != int(past_kv):
            continue
        if datapoint.phase == "ctx" and int(event.get("new_tokens", -1)) != int(step):
            continue
        if event.get("run") in (None, ""):
            continue
        latency_ms = event.get("wall_latency_ms")
        if latency_ms in (None, ""):
            continue
        latency_us = float(latency_ms) * 1000.0
        live_step_aggs.append({
            "gpu_us": latency_us,
            "rms_us": 0.0,
            "span_us": latency_us,
            "kernel_count": 0,
            "rms_kernel_count": 0,
        })
    if live_step_aggs:
        return live_step_aggs

    def _event_int(event: dict[str, Any], key: str) -> int | None:
        value = event.get(key)
        if value in (None, ""):
            return None
        return int(value)

    by_run: dict[int, dict[str, float]] = {}
    for event in events:
        if event.get("event") != "scheduler_update_wall_time":
            continue
        if event.get("work_unit_id") != work_unit_id:
            continue
        event_attempt_id = event.get("attempt_id")
        if attempt_id is not None and event_attempt_id not in (None, "") and event_attempt_id != attempt_id:
            continue
        if event.get("control_phase") != datapoint.phase:
            continue
        if _event_int(event, "control_step") != int(step):
            continue
        if _event_int(event, "control_bs") != int(batch_size):
            continue
        if _event_int(event, "control_past") != int(past_kv):
            continue
        if datapoint.phase == "ctx":
            scheduled_new_reqs = _event_int(event, "scheduled_new_reqs")
            if scheduled_new_reqs == 0:
                continue
        if datapoint.phase == "gen":
            scheduled_new_reqs = _event_int(event, "scheduled_new_reqs")
            scheduled_tokens = _event_int(event, "scheduled_tokens")
            if scheduled_new_reqs != 0 or scheduled_tokens != int(batch_size):
                continue
        run = _event_int(event, "control_run")
        if run is None:
            continue
        if datapoint.phase == "gen" and prefer_schedule_to_update_for_gen:
            latency_ms = event.get("schedule_to_update_ms", event.get("fpm_wall_time_ms"))
        else:
            latency_ms = event.get("fpm_wall_time_ms", event.get("schedule_to_update_ms"))
        if latency_ms in (None, ""):
            continue
        latency_us = float(latency_ms) * 1000.0
        run_agg = by_run.setdefault(
            run,
            {
                "gpu_us": 0.0,
                "rms_us": 0.0,
                "span_us": 0.0,
                "kernel_count": 0.0,
                "rms_kernel_count": 0.0,
            },
        )
        run_agg["gpu_us"] += latency_us
        run_agg["span_us"] += latency_us
    for _run, run_agg in sorted(by_run.items()):
        aggs.append({
            "gpu_us": run_agg["gpu_us"],
            "rms_us": run_agg["rms_us"],
            "span_us": run_agg["span_us"],
            "kernel_count": int(run_agg["kernel_count"]),
            "rms_kernel_count": int(run_agg["rms_kernel_count"]),
        })
    return aggs

def _lookup_worker_wall_aggs(
    events: list[dict[str, Any]],
    work_unit_id: str,
    datapoint: DataPoint,
    *,
    attempt_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return outer worker generate-call wall-time repeats for one datapoint."""

    aggs = []
    for event in events:
        if event.get("work_unit_id") != work_unit_id:
            continue
        event_attempt_id = event.get("attempt_id")
        if attempt_id is not None and event_attempt_id not in (None, "") and event_attempt_id != attempt_id:
            continue
        if event.get("event") == "measurement_wall_time":
            if event.get("phase") != datapoint.phase:
                continue
            if int(event.get("batch_size", -1)) != datapoint.batch_size:
                continue
            if int(event.get("new_tokens", -1)) != datapoint.new_tokens:
                continue
            if int(event.get("past_kv", -1)) != datapoint.past_kv:
                continue
            latency_ms = event.get("wall_latency_ms")
        elif event.get("event") == "generate_wall_time" and datapoint.phase == "gen":
            if event.get("phase") != "gen":
                continue
            if int(event.get("batch_size", -1)) != datapoint.batch_size:
                continue
            if int(event.get("past_kv", -1)) != datapoint.past_kv:
                continue
            latency_ms = event.get("generate_ms")
        else:
            continue
        if event.get("run") in (None, ""):
            continue
        if latency_ms in (None, ""):
            continue
        latency_us = float(latency_ms) * 1000.0
        aggs.append({
            "gpu_us": latency_us,
            "rms_us": 0.0,
            "span_us": latency_us,
            "kernel_count": 0,
            "rms_kernel_count": 0,
        })
    return aggs

def _effective_latency_source(
    requested_source: str,
    datapoint: DataPoint,
    *,
    includes_moe: bool = False,
    moe_decode_gpu_batch_threshold: int = 8,
) -> str:
    """Return the concrete latency source used for one datapoint."""

    if requested_source == "auto":
        if datapoint.phase in {"ctx", "gen"}:
            return "schedule_to_update"
        return "span"
    return requested_source

def _live_step_driver_would_handle(datapoint: DataPoint, *, gen_min_past_kv: int = 8192) -> bool:
    """Mirror the worker's live-step eligibility without importing worker code."""

    if datapoint.phase == "ctx":
        return datapoint.batch_size == 1 and int(datapoint.new_tokens) > 0
    if datapoint.phase == "gen":
        return int(datapoint.past_kv) >= int(gen_min_past_kv)
    return False


def _lookup_timing_source_aggs(
    *,
    requested_source: str,
    effective_source: str,
    events: list[dict[str, Any]],
    work_unit_id: str,
    datapoint: DataPoint,
    attempt_id: int | None = None,
    includes_moe: bool = False,
    moe_noop: bool = False,
) -> tuple[str, list[dict[str, Any]]] | None:
    """Return scheduler/worker timing repeats, falling back only for auto mode."""

    if effective_source == "schedule_to_update":
        aggs = _lookup_scheduler_timing_aggs(
            events,
            work_unit_id,
            datapoint,
            attempt_id=attempt_id,
            prefer_schedule_to_update_for_gen=False,
        )
        if aggs:
            return effective_source, aggs
        if requested_source == "auto":
            worker_aggs = _lookup_worker_wall_aggs(events, work_unit_id, datapoint, attempt_id=attempt_id)
            if worker_aggs:
                return "worker_wall", worker_aggs
        return None
    if effective_source == "worker_wall":
        aggs = _lookup_worker_wall_aggs(events, work_unit_id, datapoint, attempt_id=attempt_id)
        if aggs:
            return effective_source, aggs
        return None
    return None


class Scheduler:
    """One-GPU-slot scheduler for nsys-wrapped workers."""

    def __init__(self, args: argparse.Namespace, work_units: list[WorkUnit], worker_entrypoint: Path | None = None):
        """Create a scheduler for one layerwise collection run."""
        self.args = args
        self.worker_entrypoint = worker_entrypoint or (_THIS_DIR / "collect.py")
        self.work_units = work_units
        self._validate_capture_mode()
        self.work_dir = Path(args.work_dir).resolve()
        self.store = StatusStore(self.work_dir)
        self.output_path = Path(args.output).resolve()
        self.gpus = _detect_gpus(args.gpus)
        if not self.gpus:
            raise RuntimeError("No GPU slots available")
        self.max_workers = args.max_workers
        self.attempt_counter = self.store.max_attempt_id()
        self.fatal_streak: dict[tuple[str, str], int] = {}

    def _validate_capture_mode(self) -> None:
        """Reject nsys capture modes that cannot support requested attribution."""

        if self.args.nsys_capture != "cuda_profiler_api":
            return
        for unit in self.work_units:
            for dp in unit.datapoints:
                source = _effective_latency_source(
                    self.args.latency_source,
                    dp,
                    includes_moe=_work_unit_includes_moe(unit),
                    moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
                )
                if source not in {"schedule_to_update", "worker_wall"}:
                    raise ValueError(
                        "--nsys-capture cuda_profiler_api cannot be used for "
                        "per-layer latency sources because CUDA graph module "
                        "attribution is captured outside the profiler API range. "
                        "Use --nsys-capture full or --nsys-capture none with a "
                        "scheduler/worker-wall latency source."
                    )

    def run(self) -> None:
        """Run all queued work units and append successful CSV rows."""

        _write_csv_header_if_needed(self.output_path)
        self.store.write_missing_manifest(self.work_units)

        queue = list(self.work_units)
        active: dict[str, Attempt] = {}
        print(f"[scheduler] GPU slots: {','.join(self.gpus)}")

        while queue or active:
            launched = True
            while queue and launched:
                if self.max_workers is not None and len(active) >= self.max_workers:
                    break
                launched = False
                for queue_index, unit in enumerate(queue):
                    pending = self._pending_datapoints(unit)
                    if not pending:
                        queue.pop(queue_index)
                        launched = True
                        break
                    gpu_group = self._acquire_gpu_group(active, max(1, int(unit.physical_gpus or 1)))
                    if gpu_group is None:
                        continue
                    queue.pop(queue_index)
                    active[gpu_group] = self._launch_attempt(unit, gpu_group, pending)
                    launched = True
                    break

            finished = []
            for gpu, attempt in active.items():
                rc = attempt.process.poll()
                if rc is not None:
                    finished.append((gpu, attempt, rc))

            for gpu, attempt, rc in finished:
                del active[gpu]
                still_pending = self._finish_attempt(attempt, rc)
                if still_pending:
                    queue.append(attempt.work_unit)

            if active:
                time.sleep(1.0)

        print(f"[scheduler] Done. Results written to {self.output_path}")
        print(f"[scheduler] Status written to {self.store.status_path}")

    def _active_gpu_ids(self, active: dict[str, Attempt]) -> set[str]:
        """Return physical GPU IDs currently reserved by active attempts."""

        used: set[str] = set()
        for group in active:
            used.update(part.strip() for part in group.split(",") if part.strip())
        return used

    def _acquire_gpu_group(self, active: dict[str, Attempt], width: int) -> str | None:
        """Return a comma-separated visible-GPU group of the requested width."""

        if width <= 1:
            for gpu in self.gpus:
                if gpu not in active and gpu not in self._active_gpu_ids(active):
                    return gpu
            return None
        used = self._active_gpu_ids(active)
        available = [gpu for gpu in self.gpus if gpu not in used]
        if len(available) < width:
            return None
        return ",".join(available[:width])

    def _pending_datapoints(self, unit: WorkUnit) -> list[DataPoint]:
        terminal = self.store.index().terminal_ids()
        return [dp for dp in unit.datapoints if dp.datapoint_id(unit.work_unit_id) not in terminal]

    def _launch_attempt(self, unit: WorkUnit, gpu: str, pending: list[DataPoint]) -> Attempt:
        self.attempt_counter += 1
        attempt_id = self.attempt_counter
        paths = {
            "spec": self.work_dir / "specs" / f"{unit.work_unit_id}_a{attempt_id}.json",
            "report": self.work_dir / "nsys" / f"{unit.work_unit_id}_a{attempt_id}",
            "stdout": self.work_dir / "logs" / f"{unit.work_unit_id}_a{attempt_id}.out",
            "stderr": self.work_dir / "logs" / f"{unit.work_unit_id}_a{attempt_id}.err",
            "metadata": self.work_dir / "metadata" / f"{unit.work_unit_id}_a{attempt_id}.json",
        }
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        spec = self._make_spec(unit, pending, attempt_id)
        spec["metadata_path"] = str(paths["metadata"])
        _json_dump(paths["spec"], spec)
        paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = paths["stdout"].open("w")
        stderr_handle = paths["stderr"].open("w")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        fpm_port = FPM_PORT_BASE + attempt_id
        env[FPM_PORT_ENV] = str(fpm_port)
        if any(
            _effective_latency_source(
                self.args.latency_source,
                dp,
                includes_moe=_work_unit_includes_moe(unit),
                moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
            ) == "schedule_to_update"
            for dp in pending
        ):
            env["LAYERWISE_SCHEDULER_TIMING"] = "1"
        if getattr(self.args, "live_step_driver", False):
            env["LAYERWISE_USE_LIVE_STEP_DRIVER"] = "1"
            env["LAYERWISE_LIVE_STEP_GEN_MIN_PAST_KV"] = str(
                int(getattr(self.args, "live_step_gen_min_past_kv", 8192))
            )
        cmd = self._worker_cmd(
            paths["spec"],
            paths["report"],
            capture_nsys=self._attempt_needs_nsys(unit, pending),
        )
        print(
            f"[scheduler] launch gpu={gpu} attempt={attempt_id} "
            f"{unit.work_unit_id} pending={len(pending)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(_THIS_DIR),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        self.store.append_event(
            "attempt_started",
            work_unit_id=unit.work_unit_id,
            attempt_id=attempt_id,
            gpu=gpu,
            fpm_port=fpm_port,
            report_base=str(paths["report"]),
            spec=str(paths["spec"]),
        )
        return Attempt(
            work_unit=unit,
            gpu=gpu,
            attempt_id=attempt_id,
            spec_path=paths["spec"],
            report_base=paths["report"],
            stdout_path=paths["stdout"],
            stderr_path=paths["stderr"],
            process=process,
            stdout_handle=stdout_handle,
            stderr_handle=stderr_handle,
            pending_ids={dp.datapoint_id(unit.work_unit_id) for dp in pending},
        )

    def _make_spec(self, unit: WorkUnit, pending: list[DataPoint], attempt_id: int) -> dict[str, Any]:
        extra_vllm_args = list(unit.extra_vllm_args)
        if unit.row_base["kv_quant"] == "fp8":
            extra_vllm_args.extend(["--kv-cache-dtype", "fp8"])
        extra_vllm_args.extend(self.args.extra_vllm_arg)
        _append_default_vllm_args(extra_vllm_args)
        has_ctx = any(dp.phase == "ctx" for dp in pending)
        has_gen = any(dp.phase == "gen" for dp in pending)
        gen_datapoints = [dp for dp in pending if dp.phase == "gen"]
        live_step_gen_min_past_kv = int(getattr(self.args, "live_step_gen_min_past_kv", 8192))
        has_live_step_gen = (
            getattr(self.args, "live_step_driver", False)
            and any(
                _live_step_driver_would_handle(dp, gen_min_past_kv=live_step_gen_min_past_kv)
                for dp in gen_datapoints
            )
        )
        live_step_gen_deployment = (
            has_gen
            and (
                unit.gen_driver == "live_decode"
                or has_live_step_gen
            )
        )
        needs_prefix_cache = has_gen and unit.gen_driver == "prefix_cache"
        disables_prefix_cache = live_step_gen_deployment and unit.gen_driver == "live_decode"
        runtime_defaults = gpt_oss_runtime_defaults(
            model=unit.row_base["model"],
            system=unit.row_base["system"],
            disable_prefix_caching=disables_prefix_cache,
            extra_args=tuple(extra_vllm_args),
        )
        extra_vllm_args = list(runtime_defaults.extra_args)
        if runtime_defaults.kv_cache_dtype:
            extra_vllm_args = [
                "--kv-cache-dtype",
                runtime_defaults.kv_cache_dtype,
                *extra_vllm_args,
            ]
        if runtime_defaults.disable_prefix_caching and not has_cli_flag(
            extra_vllm_args, "--no-enable-prefix-caching"
        ):
            extra_vllm_args.append("--no-enable-prefix-caching")
        if needs_prefix_cache and not has_cli_flag(
            extra_vllm_args, "--enable-prefix-caching", "--no-enable-prefix-caching"
        ):
            extra_vllm_args.append("--enable-prefix-caching")
        effective_latency_sources = [
            _effective_latency_source(
                self.args.latency_source,
                dp,
                includes_moe=_work_unit_includes_moe(unit),
                moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
            )
            for dp in pending
        ]
        enable_layerwise_nvtx = any(
            source not in {"schedule_to_update", "worker_wall"}
            for source in effective_latency_sources
        )
        enable_layer_patch = unit.needs_layer_patch(
            enable_layerwise_nvtx_tracing=enable_layerwise_nvtx
        )
        live_step_marker = (
            getattr(self.args, "live_step_driver", False)
            and any(
                _live_step_driver_would_handle(dp, gen_min_past_kv=live_step_gen_min_past_kv)
                for dp in pending
            )
        )
        enable_step_marker = (
            self._attempt_needs_nsys(unit, pending)
            or enable_layerwise_nvtx
            or live_step_marker
            or (enable_layer_patch and self.args.nsys_capture != "none")
        )
        max_num_batched_tokens = unit.max_num_batched_tokens
        if max_num_batched_tokens is None and (has_ctx or live_step_gen_deployment):
            max_num_batched_tokens = _get_vllm_deployment_max_num_batched_tokens(
                model=unit.row_base["model"],
                tensor_parallel_size=int(unit.row_base["attn_tp"]),
                max_num_seqs=unit.max_num_seqs,
                max_model_len=unit.max_model_len,
                gpu_memory_utilization=unit.gpu_memory_utilization,
                extra_args=tuple(extra_vllm_args),
            )
        cache_block_size = unit.cache_block_size

        return {
            "attempt_id": attempt_id,
            "work_unit_id": unit.work_unit_id,
            "model_dir": unit.model_dir,
            "target_layers": unit.target_layers,
            "model_layer_count": unit.model_layer_count,
            "enable_layer_patch": enable_layer_patch,
            "moe_noop": unit.moe_noop,
            "moe_weight_mode": unit.moe_weight_mode,
            "datapoints": [asdict(dp) for dp in pending],
            "max_num_seqs": unit.max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
            "cache_block_size": cache_block_size,
            "max_model_len": unit.max_model_len,
            "gpu_memory_utilization": unit.gpu_memory_utilization,
            "gen_driver": unit.gen_driver,
            "status_path": str(self.store.status_path),
            "extra_vllm_args": extra_vllm_args,
            "router_weight_model": unit.router_weight_model,
            "physical_gpus": unit.physical_gpus,
            "enable_layerwise_nvtx_tracing": enable_layerwise_nvtx,
            "enable_step_marker": enable_step_marker,
            "ctx_warmup_runs": self.args.ctx_warmup_runs,
            "ctx_measured_runs": self.args.ctx_measured_runs,
            "gen_warmup_runs": self.args.gen_warmup_runs,
            "gen_measured_runs": self.args.gen_measured_runs,
            "prompt_seed": self.args.prompt_seed,
            "nsys_capture": self.args.nsys_capture,
        }

    def _attempt_needs_nsys(self, unit: WorkUnit, pending: list[DataPoint]) -> bool:
        """Return whether an attempt needs an nsys trace for its latency source."""

        if self.args.nsys_capture == "none":
            return False
        return any(
            _effective_latency_source(
                self.args.latency_source,
                dp,
                includes_moe=_work_unit_includes_moe(unit),
                moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
            )
            not in {"schedule_to_update", "worker_wall"}
            for dp in pending
        )

    def _worker_cmd(self, spec_path: Path, report_base: Path, *, capture_nsys: bool = True) -> list[str]:
        worker_cmd = [sys.executable, str(self.worker_entrypoint), "worker", "--spec", str(spec_path)]
        if not capture_nsys:
            return worker_cmd
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
        ]
        if self.args.nsys_capture == "cuda_profiler_api":
            cmd.extend([
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
            ])
        elif self.args.nsys_capture != "full":
            raise ValueError(f"unsupported nsys capture mode: {self.args.nsys_capture}")
        cmd.extend(["-o", str(report_base), *worker_cmd])
        return cmd

    def _finish_attempt(self, attempt: Attempt, returncode: int) -> bool:
        attempt.stdout_handle.close()
        attempt.stderr_handle.close()
        stderr_tail = _tail(attempt.stderr_path, 120)
        print(
            f"[scheduler] finish gpu={attempt.gpu} attempt={attempt.attempt_id} "
            f"rc={returncode} {attempt.work_unit.work_unit_id}"
        )

        successes = self._parse_attempt_report(attempt)
        if successes:
            key = (attempt.work_unit.work_unit_id, "fatal_cuda")
            self.fatal_streak.pop(key, None)

        if returncode != 0:
            self._mark_crashed_attempt(attempt, returncode, stderr_tail, successes)
        else:
            self._mark_clean_parse_failures(attempt)

        self.store.append_event(
            "attempt_finished",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            returncode=returncode,
            parsed_successes=successes,
        )
        return bool(self._pending_datapoints(attempt.work_unit))

    def _parse_attempt_report(self, attempt: Attempt) -> int:
        sqlite_path = attempt.report_base.with_suffix(".sqlite")
        rep_path = attempt.report_base.with_suffix(".nsys-rep")
        if not rep_path.exists() or rep_path.stat().st_size == 0:
            return self._parse_scheduler_timing_only(attempt)
        self.store.append_event(
            "nsys_export_started",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            rep=str(rep_path),
            rep_bytes=rep_path.stat().st_size,
        )
        export_cmd = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(rep_path),
        ]
        try:
            result = subprocess.run(
                export_cmd,
                text=True,
                capture_output=True,
                timeout=self.args.timeout,
                check=False,
            )
            if result.returncode != 0:
                self.store.append_event(
                    "nsys_export_failed",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    attempt_id=attempt.attempt_id,
                    returncode=result.returncode,
                    stderr=result.stderr[-4000:],
                )
                return 0
        except Exception as exc:
            self.store.append_event(
                "nsys_export_failed",
                work_unit_id=attempt.work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                error=repr(exc),
            )
            return 0

        self.store.append_event(
            "nsys_export_succeeded",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
            sqlite_bytes=sqlite_path.stat().st_size if sqlite_path.exists() else 0,
        )
        self.store.append_event(
            "nsys_parse_started",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
        )
        try:
            rows, meta = parse_step_sweep(
                str(sqlite_path),
                rollup=_effective_rollup(self.args),
                layer=None,
                rank_reduce=self.args.rank_reduce,
            )
        except Exception as exc:
            self.store.append_event(
                "nsys_parse_failed",
                work_unit_id=attempt.work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                sqlite=str(sqlite_path),
                error=repr(exc),
            )
            return 0

        parsed = _aggregate_step_rows(rows)
        self.store.append_event(
            "nsys_parse_succeeded",
            work_unit_id=attempt.work_unit.work_unit_id,
            attempt_id=attempt.attempt_id,
            sqlite=str(sqlite_path),
            rows=len(rows),
            meta=meta,
        )
        index = self.store.index()
        successes = 0
        for dp in attempt.work_unit.datapoints:
            dpid = dp.datapoint_id(attempt.work_unit.work_unit_id)
            if dpid not in attempt.pending_ids or index.is_terminal(dpid):
                continue
            effective_latency_source = _effective_latency_source(
                self.args.latency_source,
                dp,
                includes_moe=_work_unit_includes_moe(attempt.work_unit),
                moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
            )
            if effective_latency_source in {"schedule_to_update", "worker_wall"}:
                timing = _lookup_timing_source_aggs(
                    requested_source=self.args.latency_source,
                    effective_source=effective_latency_source,
                    events=index.events,
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint=dp,
                    attempt_id=attempt.attempt_id,
                    includes_moe=_work_unit_includes_moe(attempt.work_unit),
                    moe_noop=bool(attempt.work_unit.moe_noop),
                )
                if timing is None:
                    continue
                effective_latency_source, aggs = timing
                reduce_latency_source = "span"
            else:
                aggs = _lookup_aggs(parsed, dp.parse_key())
                reduce_latency_source = effective_latency_source
            if not aggs:
                continue
            latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = _reduce_agg_latency(
                aggs,
                latency_source=reduce_latency_source,
                aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation,
            )
            representative = asdict(attempt.work_unit.representative)
            physical_representative = (
                int(attempt.work_unit.physical_gpus or 1) > 1
                and not attempt.work_unit.uses_full_layer_depth()
            )
            if (
                effective_latency_source in {"schedule_to_update", "worker_wall"}
                and not physical_representative
            ):
                representative["layer_multiplier"] = representative["measured_layer_count"]
            row = {
                **attempt.work_unit.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
                **representative,
                "latency_ms": latency_us / 1000.0,
                "rms_latency_ms": rms_us / 1000.0,
                "rms_kernel_count": rms_kernel_count,
                "includes_moe": _work_unit_includes_moe(attempt.work_unit),
                "moe_weight_mode": attempt.work_unit.moe_weight_mode,
                "latency_source": effective_latency_source,
                "physical_gpus": attempt.work_unit.physical_gpus,
                "max_num_batched_tokens": _attempt_max_num_batched_tokens(attempt) or "",
                "vllm_config_hash": _attempt_config_hash(attempt, self.store),
            }
            _append_success_row(self.output_path, row)
            self.store.append_event(
                "success",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                latency_ms=row["latency_ms"],
                latency_source=effective_latency_source,
                requested_latency_source=self.args.latency_source,
                repeat_aggregation=(
                    self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation
                ),
                measure_count=measure_count,
                kernel_count=kernel_count,
                rms_latency_ms=row["rms_latency_ms"],
                rms_kernel_count=rms_kernel_count,
                sqlite=str(sqlite_path),
            )
            successes += 1
        return successes

    def _parse_scheduler_timing_only(self, attempt: Attempt) -> int:
        """Append schedule-envelope rows when no nsys report was captured."""

        index = self.store.index()
        successes = 0
        for dp in attempt.work_unit.datapoints:
            dpid = dp.datapoint_id(attempt.work_unit.work_unit_id)
            if dpid not in attempt.pending_ids or index.is_terminal(dpid):
                continue
            effective_latency_source = _effective_latency_source(
                self.args.latency_source,
                dp,
                includes_moe=_work_unit_includes_moe(attempt.work_unit),
                moe_decode_gpu_batch_threshold=self.args.moe_decode_gpu_batch_threshold,
            )
            if effective_latency_source not in {"schedule_to_update", "worker_wall"}:
                if dp.phase != "gen":
                    continue
                effective_latency_source = "worker_wall"
            timing = _lookup_timing_source_aggs(
                requested_source=self.args.latency_source,
                effective_source=effective_latency_source,
                events=index.events,
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint=dp,
                attempt_id=attempt.attempt_id,
                includes_moe=_work_unit_includes_moe(attempt.work_unit),
                moe_noop=bool(attempt.work_unit.moe_noop),
            )
            if timing is None:
                continue
            effective_latency_source, aggs = timing
            latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = _reduce_agg_latency(
                aggs,
                latency_source="span",
                aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation,
            )
            representative = asdict(attempt.work_unit.representative)
            physical_representative = (
                int(attempt.work_unit.physical_gpus or 1) > 1
                and not attempt.work_unit.uses_full_layer_depth()
            )
            if not physical_representative:
                representative["layer_multiplier"] = representative["measured_layer_count"]
            row = {
                **attempt.work_unit.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
                **representative,
                "latency_ms": latency_us / 1000.0,
                "rms_latency_ms": rms_us / 1000.0,
                "rms_kernel_count": rms_kernel_count,
                "includes_moe": _work_unit_includes_moe(attempt.work_unit),
                "moe_weight_mode": attempt.work_unit.moe_weight_mode,
                "latency_source": effective_latency_source,
                "physical_gpus": attempt.work_unit.physical_gpus,
                "max_num_batched_tokens": _attempt_max_num_batched_tokens(attempt) or "",
                "vllm_config_hash": _attempt_config_hash(attempt, self.store),
            }
            _append_success_row(self.output_path, row)
            self.store.append_event(
                "success",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                latency_ms=row["latency_ms"],
                latency_source=effective_latency_source,
                requested_latency_source=self.args.latency_source,
                repeat_aggregation=(
                    self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation
                ),
                measure_count=measure_count,
                kernel_count=kernel_count,
                rms_latency_ms=row["rms_latency_ms"],
                rms_kernel_count=rms_kernel_count,
                sqlite="",
            )
            successes += 1
        return successes

    def _mark_clean_parse_failures(self, attempt: Attempt) -> None:
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            if dpid in index.completed:
                self.store.append_event(
                    "failed_parse",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint_id=dpid,
                    attempt_id=attempt.attempt_id,
                    message="worker exited cleanly but no parsed latency row was found",
                )
            else:
                self.store.append_event(
                    "skipped_not_started",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint_id=dpid,
                    attempt_id=attempt.attempt_id,
                    message="worker exited cleanly before this datapoint produced a target marker",
                )

    def _mark_crashed_attempt(
        self,
        attempt: Attempt,
        returncode: int,
        stderr_tail: str,
        successes: int,
    ) -> None:
        index = self.store.index()
        signature = _attempt_signature(returncode, stderr_tail)
        active = index.active_started(attempt.work_unit.work_unit_id, attempt.pending_ids)

        if active and not index.is_terminal(active):
            if signature == "oom":
                event = "failed_oom"
            elif signature == "fatal_cuda":
                event = "failed_fatal_cuda"
            else:
                event = "failed_error"
            self.store.append_event(
                event,
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=active,
                attempt_id=attempt.attempt_id,
                returncode=returncode,
                signature=signature,
                stderr_tail=stderr_tail[-4000:],
            )
            if event == "failed_oom":
                failed_dp = self._find_datapoint(attempt.work_unit, active)
                if failed_dp:
                    self._mark_oom_dominated(attempt.work_unit, failed_dp, active)

        # Completed-but-unparsed datapoints were attempted.  Mark them terminal
        # so crash recovery never reruns work merely because sqlite parsing lost it.
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            if dpid in index.completed:
                self.store.append_event(
                    "failed_parse",
                    work_unit_id=attempt.work_unit.work_unit_id,
                    datapoint_id=dpid,
                    attempt_id=attempt.attempt_id,
                    message="worker crashed after execution but no parsed row was found",
                )

        if successes:
            return
        streak_key = (attempt.work_unit.work_unit_id, signature)
        self.fatal_streak[streak_key] = self.fatal_streak.get(streak_key, 0) + 1
        if self.fatal_streak[streak_key] >= FATAL_STREAK_LIMIT:
            self.store.append_event(
                "work_unit_omitted",
                work_unit_id=attempt.work_unit.work_unit_id,
                attempt_id=attempt.attempt_id,
                signature=signature,
                message=f"{FATAL_STREAK_LIMIT} consecutive crashes with no parsed success",
                stderr_tail=stderr_tail[-4000:],
                row_base=attempt.work_unit.row_base,
            )
            index = self.store.index()
            for dp in attempt.work_unit.datapoints:
                dpid = dp.datapoint_id(attempt.work_unit.work_unit_id)
                if dpid in attempt.pending_ids and not index.is_terminal(dpid):
                    self.store.append_event(
                        "skipped_same_error",
                        work_unit_id=attempt.work_unit.work_unit_id,
                        datapoint_id=dpid,
                        attempt_id=attempt.attempt_id,
                        signature=signature,
                        message=f"{FATAL_STREAK_LIMIT} consecutive crashes with no parsed success",
                    )

    def _mark_oom_dominated(self, unit: WorkUnit, failed_dp: DataPoint, failed_id: str) -> None:
        index = self.store.index()
        for dp in unit.datapoints:
            dpid = dp.datapoint_id(unit.work_unit_id)
            if dpid == failed_id or index.is_terminal(dpid):
                continue
            if oom_dominates(failed_dp, dp):
                self.store.append_event(
                    "skipped_oom_dominated",
                    work_unit_id=unit.work_unit_id,
                    datapoint_id=dpid,
                    caused_by=failed_id,
                )

    @staticmethod
    def _find_datapoint(unit: WorkUnit, datapoint_id: str) -> DataPoint | None:
        for dp in unit.datapoints:
            if dp.datapoint_id(unit.work_unit_id) == datapoint_id:
                return dp
        return None
