# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-slot scheduler, retry handling, and nsys result ingestion for layerwise runs."""

from __future__ import annotations

import argparse
import fcntl
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
from vllm_deployment import find_runtime_vllm_config, gpt_oss_runtime_defaults, has_cli_flag

try:
    from .data import DataPoint, WorkUnit
    from .datapoint_generator import _max_num_batched_tokens_for_datapoints
    from .engine import _append_default_vllm_args
    from .nsys import DEFAULT_ROLLUP, _aggregate_step_rows, _effective_rollup, _lookup_aggs, _reduce_agg_latency
    from .results import _append_success_row, _write_csv_header_if_needed, _work_unit_includes_moe
    from .runtime import _detect_gpus, _json_dump, _tail, _utc_now
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint, WorkUnit
    from datapoint_generator import _max_num_batched_tokens_for_datapoints
    from engine import _append_default_vllm_args
    from nsys import DEFAULT_ROLLUP, _aggregate_step_rows, _effective_rollup, _lookup_aggs, _reduce_agg_latency
    from results import _append_success_row, _write_csv_header_if_needed, _work_unit_includes_moe
    from runtime import _detect_gpus, _json_dump, _tail, _utc_now


TERMINAL_EVENTS = {
    "success", "failed_oom", "failed_error", "failed_fatal_cuda", "failed_parse",
    "skipped_oom_dominated", "skipped_same_error",
}
FATAL_STREAK_LIMIT = 3


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
            if name == "started":
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
        for event in reversed(self.events):
            if event.get("event") != "started":
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
                    events.append(json.loads(line))
        return events

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
    return "out of memory" in lowered or "cuda oom" in lowered or "cublas_status_alloc_failed" in lowered

def _is_fatal_cuda_text(text: str) -> bool:
    lowered = text.lower()
    return "illegal memory access" in lowered or "device-side assert" in lowered or "cuda error" in lowered

def _attempt_signature(returncode: int, stderr_tail: str) -> str:
    if _is_oom_text(stderr_tail):
        return "oom"
    if _is_fatal_cuda_text(stderr_tail):
        return "fatal_cuda"
    return f"exit_{returncode}"

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
    return same_or_larger and strictly_larger

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

class Scheduler:
    """One-GPU-slot scheduler for nsys-wrapped workers."""

    def __init__(self, args: argparse.Namespace, work_units: list[WorkUnit], worker_entrypoint: Path | None = None):
        """Create a scheduler for one layerwise collection run."""
        self.args = args
        self.worker_entrypoint = worker_entrypoint or (_THIS_DIR / "collect.py")
        self.work_units = work_units
        self.work_dir = Path(args.work_dir).resolve()
        self.store = StatusStore(self.work_dir)
        self.output_path = Path(args.output).resolve()
        self.gpus = _detect_gpus(args.gpus)
        if args.max_workers:
            self.gpus = self.gpus[: args.max_workers]
        if not self.gpus:
            raise RuntimeError("No GPU slots available")
        self.attempt_counter = 0
        self.fatal_streak: dict[tuple[str, str], int] = {}

    def run(self) -> None:
        """Run all queued work units and append successful CSV rows."""

        _write_csv_header_if_needed(self.output_path)
        self.store.write_missing_manifest(self.work_units)

        queue = list(self.work_units)
        active: dict[str, Attempt] = {}
        print(f"[scheduler] GPU slots: {','.join(self.gpus)}")

        while queue or active:
            for gpu in self.gpus:
                if gpu in active or not queue:
                    continue
                unit = queue.pop(0)
                pending = self._pending_datapoints(unit)
                if not pending:
                    continue
                active[gpu] = self._launch_attempt(unit, gpu, pending)

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
        cmd = self._worker_cmd(paths["spec"], paths["report"])
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
        extra_vllm_args = []
        if unit.row_base["kv_quant"] == "fp8":
            extra_vllm_args.extend(["--kv-cache-dtype", "fp8"])
        extra_vllm_args.extend(self.args.extra_vllm_arg)
        _append_default_vllm_args(extra_vllm_args)
        has_ctx = any(dp.phase == "ctx" for dp in pending)
        has_gen = any(dp.phase == "gen" for dp in pending)
        runtime_defaults = gpt_oss_runtime_defaults(
            model=unit.row_base["model"],
            system=unit.row_base["system"],
            disable_prefix_caching=not (has_ctx or has_gen),
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
        if (has_ctx or has_gen) and not has_cli_flag(
            extra_vllm_args, "--enable-prefix-caching", "--no-enable-prefix-caching"
        ):
            extra_vllm_args.append("--enable-prefix-caching")

        return {
            "attempt_id": attempt_id,
            "work_unit_id": unit.work_unit_id,
            "model_dir": unit.model_dir,
            "target_layers": unit.target_layers,
            "moe_noop": unit.moe_noop,
            "datapoints": [asdict(dp) for dp in pending],
            "status_path": str(self.store.status_path),
            "extra_vllm_args": extra_vllm_args,
            "ctx_warmup_runs": self.args.ctx_warmup_runs,
            "ctx_measured_runs": self.args.ctx_measured_runs,
            "gen_warmup_runs": self.args.gen_warmup_runs,
            "gen_measured_runs": self.args.gen_measured_runs,
            "nsys_capture": self.args.nsys_capture,
        }

    def _worker_cmd(self, spec_path: Path, report_base: Path) -> list[str]:
        worker_cmd = [sys.executable, str(self.worker_entrypoint), "worker", "--spec", str(spec_path)]
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
            return 0
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
                force_nvtx_span=(
                    _effective_rollup(self.args) == DEFAULT_ROLLUP
                    and self.args.latency_source == "span"
                ),
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
            aggs = _lookup_aggs(parsed, dp.parse_key())
            if not aggs:
                continue
            latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = _reduce_agg_latency(
                aggs,
                latency_source=self.args.latency_source,
                aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else self.args.gen_repeat_aggregation,
            )
            row = {
                **attempt.work_unit.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
                **asdict(attempt.work_unit.representative),
                "latency_ms": latency_us / 1000.0,
                "rms_latency_ms": rms_us / 1000.0,
                "rms_kernel_count": rms_kernel_count,
                "includes_moe": _work_unit_includes_moe(attempt.work_unit),
                "vllm_config_hash": _attempt_config_hash(attempt, self.store),
            }
            _append_success_row(self.output_path, row)
            self.store.append_event(
                "success",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                latency_ms=row["latency_ms"],
                latency_source=self.args.latency_source,
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

    def _mark_clean_parse_failures(self, attempt: Attempt) -> None:
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            self.store.append_event(
                "failed_parse",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                message="worker exited cleanly but no parsed latency row was found",
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
            event = "failed_oom" if signature == "oom" else "failed_fatal_cuda"
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
