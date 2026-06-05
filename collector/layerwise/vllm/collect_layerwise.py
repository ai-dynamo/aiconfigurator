#!/usr/bin/env python3
"""Resilient vLLM layerwise collector.

The public CLI is the scheduler.  It builds one work unit per mocked model
configuration, assigns work units to one-GPU slots, and launches this same file
in hidden ``worker`` mode under ``nsys profile``.  The scheduler never imports
vLLM; CUDA/vLLM failures stay inside worker subprocesses.

Workers append progress events before/after datapoints.  The scheduler parses
complete or partial nsys reports, writes successful CSV rows, and marks failed
datapoints terminal so they are not retried on resume.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from parallel_config_patch import _load_original_config, patch_for_parallelism
from parse_nsys_step_sweep import parse_step_sweep
from random_prompt_tokens import (
    RandomPromptTokenConfig,
    load_random_prompt_token_config,
    sample_prompt_token_ids,
)

CTX_NEW_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
CTX_PAST_KV = [0]
GEN_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GEN_PAST_KV = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

CSV_COLUMNS = [
    "framework",
    "framework_version",
    "system",
    "model",
    "attn_tp",
    "moe_tp",
    "ep",
    "num_slots",
    "gemm_quant",
    "moe_quant",
    "attn_quant",
    "kv_quant",
    "phase",
    "batch_size",
    "new_tokens",
    "past_kv",
    "latency_ms",
]

TERMINAL_EVENTS = {
    "success",
    "failed_oom",
    "failed_error",
    "failed_fatal_cuda",
    "failed_parse",
    "skipped_oom_dominated",
    "skipped_same_error",
}

FATAL_STREAK_LIMIT = 3


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


@dataclass(frozen=True)
class DataPoint:
    phase: str
    batch_size: int
    new_tokens: int
    past_kv: int

    @property
    def shape_key(self) -> str:
        return (
            f"{self.phase}:bs{self.batch_size}:"
            f"new{self.new_tokens}:past{self.past_kv}"
        )

    def datapoint_id(self, work_unit_id: str) -> str:
        return f"{work_unit_id}:{self.shape_key}"

    def parse_key(self) -> tuple[int, int, int]:
        if self.phase == "ctx":
            return self.new_tokens, self.batch_size, self.past_kv
        return self.past_kv + 1, self.batch_size, self.past_kv


@dataclass(frozen=True)
class WorkUnit:
    work_unit_id: str
    model_dir: str
    row_base: dict[str, Any]
    target_layers: list[int]
    datapoints: list[DataPoint]

    def manifest_rows(self) -> list[dict[str, Any]]:
        rows = []
        for dp in self.datapoints:
            rows.append({
                "work_unit_id": self.work_unit_id,
                "datapoint_id": dp.datapoint_id(self.work_unit_id),
                **self.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
            })
        return rows


@dataclass
class Attempt:
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
        return datapoint_id in self.terminal

    def terminal_ids(self) -> set[str]:
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)

    def append_event(self, event: str, *, work_unit_id: str, datapoint_id: str | None = None, **extra: Any) -> None:
        row = {
            "event": event,
            "work_unit_id": work_unit_id,
            "datapoint_id": datapoint_id,
            "ts": _utc_now(),
            **extra,
        }
        self.append_jsonl(self.status_path, {k: v for k, v in row.items() if v is not None})

    def load_events(self) -> list[dict[str, Any]]:
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
        return StatusIndex(self.load_events())

    def existing_manifest_ids(self) -> set[str]:
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
        seen = self.existing_manifest_ids()
        for unit in work_units:
            for row in unit.manifest_rows():
                if row["datapoint_id"] in seen:
                    continue
                self.append_jsonl(self.manifest_path, row)
                seen.add(row["datapoint_id"])


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


def _detect_layer_schedule(
    config: dict[str, Any],
    include_moe_layer: bool,
    target_layer_count: int = 1,
    target_layers: list[int] | None = None,
    target_layer_config_depth: int | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any] | None]:
    max_config_layers = int(config.get("num_hidden_layers") or 0)
    if target_layers is not None:
        if not target_layers:
            raise ValueError("target_layers must not be empty")
        if any(i < 0 for i in target_layers):
            raise ValueError(f"target_layers must be non-negative, got {target_layers}")
        if max_config_layers and max(target_layers) >= max_config_layers:
            raise ValueError(
                f"target_layers {target_layers} exceed config num_hidden_layers="
                f"{max_config_layers}"
            )
        if _is_moe_config(config):
            raise ValueError("explicit target_layers is currently supported for dense models only")
        sorted_layers = sorted(set(target_layers))
        num_hidden_layers = max(sorted_layers) + 1
        if target_layer_config_depth is not None:
            if target_layer_config_depth < num_hidden_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than required depth {num_hidden_layers}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            {"layer_index": i, "layer_type": "dense"}
            for i in sorted_layers
        ], num_hidden_layers, None

    if target_layer_count < 1:
        raise ValueError(f"target_layer_count must be >= 1, got {target_layer_count}")
    if not _is_moe_config(config):
        num_hidden_layers = target_layer_count
        if target_layer_config_depth is not None:
            if target_layer_config_depth < target_layer_count:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than target_layer_count={target_layer_count}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            {"layer_index": i, "layer_type": "dense"}
            for i in range(target_layer_count)
        ], num_hidden_layers, None

    # vLLM MoE is opt-in because dummy-weight routing still underestimates MoE.
    if target_layer_count != 1:
        raise ValueError(
            "target_layer_count > 1 is currently only supported for dense models"
        )
    layer_schedule = [{"layer_index": 0, "layer_type": "dense"}]
    num_hidden_layers = 1
    if include_moe_layer:
        layer_schedule.append({"layer_index": 1, "layer_type": "moe"})
        num_hidden_layers = 2

    overrides = {"first_k_dense_replace": 1}
    if "decoder_sparse_step" in config:
        overrides["decoder_sparse_step"] = 1
        overrides["mlp_only_layers"] = []
    return layer_schedule, num_hidden_layers, overrides


def _is_moe_config(config: dict[str, Any]) -> bool:
    return any((config.get(k, 0) or 0) > 0 for k in ("n_routed_experts", "num_experts"))


def _work_unit_id(
    row_base: dict[str, Any],
    target_layers: list[int],
    num_hidden_layers: int,
) -> str:
    payload = {
        **row_base,
        "target_layers": target_layers,
        "num_hidden_layers": num_hidden_layers,
    }
    return "wu_" + _stable_hash(payload)


def _filter_rows_to_target_layers(
    rows: list[dict[str, Any]],
    target_layers: Iterable[int],
) -> list[dict[str, Any]]:
    targets = {int(x) for x in target_layers}
    if not targets:
        return rows
    out = []
    for row in rows:
        parts = row.get("rollup_parts") or ()
        if parts and int(parts[0]) in targets:
            out.append(row)
    return out


def _build_datapoints(
    *,
    phases: str,
    ctx_new_tokens: list[int],
    ctx_past_kv: list[int],
    gen_batch_sizes: list[int],
    gen_past_kv: list[int],
) -> list[DataPoint]:
    datapoints: list[DataPoint] = []
    if phases in ("ctx", "both"):
        for past_kv in ctx_past_kv:
            for new_tokens in ctx_new_tokens:
                datapoints.append(DataPoint("ctx", 1, new_tokens, past_kv))
    if phases in ("gen", "both"):
        for batch_size in gen_batch_sizes:
            for past_kv in gen_past_kv:
                datapoints.append(DataPoint("gen", batch_size, 1, past_kv))
    return datapoints


def _max_num_batched_tokens_for_datapoints(
    datapoints: list[DataPoint],
    min_max_num_batched_tokens: int = 1,
) -> int:
    ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
    gen_points = [dp for dp in datapoints if dp.phase == "gen"]
    ctx_max_new = max((dp.new_tokens for dp in ctx_points), default=0)
    gen_batch_sizes = sorted({dp.batch_size for dp in gen_points})
    return max(
        1,
        min_max_num_batched_tokens,
        ctx_max_new,
        max(gen_batch_sizes, default=0),
    )


def _validate_ctx_past_kv(datapoints: list[DataPoint], max_num_batched_tokens: int) -> None:
    for dp in datapoints:
        if dp.phase != "ctx" or dp.past_kv == 0:
            continue
        if dp.past_kv % max_num_batched_tokens != 0:
            raise ValueError(
                "ctx past_kv measurements must start on a chunk boundary: "
                f"past_kv={dp.past_kv}, max_num_batched_tokens={max_num_batched_tokens}"
            )


def build_work_units(args: argparse.Namespace) -> list[WorkUnit]:
    ctx_new_tokens = _parse_ints(args.ctx_new_tokens)
    ctx_past_kv = _parse_ints(args.ctx_past_kv)
    gen_batch_sizes = _parse_ints(args.gen_batch_sizes)
    gen_past_kv = _parse_ints(args.gen_past_kv)
    tp_sizes = _parse_ints(args.tp_sizes)

    orig_config = _load_original_config(args.model)
    is_moe = _is_moe_config(orig_config)
    explicit_target_layers = (
        _parse_ints(args.target_layers) if getattr(args, "target_layers", None) else None
    )
    layer_schedule, num_hidden_layers, extra_overrides = _detect_layer_schedule(
        orig_config, args.include_moe_layer, args.target_layer_count,
        explicit_target_layers, args.target_layer_config_depth,
    )
    target_layers = [int(x["layer_index"]) for x in layer_schedule]

    work_dir = Path(args.work_dir).resolve()
    config_cache_dir = None if args.no_config_cache else (args.config_cache_dir or str(work_dir / "config_cache"))

    system = args.system or _get_system_name()
    version = args.framework_version or _get_vllm_version()
    datapoints = _build_datapoints(
        phases=args.phases,
        ctx_new_tokens=ctx_new_tokens,
        ctx_past_kv=ctx_past_kv,
        gen_batch_sizes=gen_batch_sizes,
        gen_past_kv=gen_past_kv,
    )
    _validate_ctx_past_kv(
        datapoints,
        _max_num_batched_tokens_for_datapoints(datapoints, args.min_max_num_batched_tokens),
    )

    work_units: list[WorkUnit] = []
    for tp in tp_sizes:
        if is_moe and tp % args.moe_tp != 0:
            print(f"[skip] tp={tp} not divisible by moe_tp={args.moe_tp}")
            continue
        attn_tp = tp
        moe_tp = args.moe_tp if is_moe else 1
        ep = (tp // moe_tp) if is_moe else 1
        num_slots = args.num_slots if is_moe else None
        model_dir = patch_for_parallelism(
            args.model,
            attn_tp=attn_tp,
            moe_tp=moe_tp,
            ep=ep,
            num_slots=num_slots,
            num_hidden_layers=num_hidden_layers,
            extra_overrides=extra_overrides,
            model_type_rewrites={"glm_moe_dsa": "deepseek_v3"},
            cache_dir=config_cache_dir,
            original_config=orig_config,
        )
        row_base = {
            "framework": "vLLM",
            "framework_version": version,
            "system": system,
            "model": args.model,
            "attn_tp": attn_tp,
            "moe_tp": moe_tp,
            "ep": ep,
            "num_slots": num_slots or "",
            "gemm_quant": args.gemm_quant,
            "moe_quant": args.moe_quant,
            "attn_quant": args.attn_quant,
            "kv_quant": args.kv_quant,
        }
        work_units.append(WorkUnit(
            work_unit_id=_work_unit_id(
                row_base,
                target_layers,
                num_hidden_layers,
            ),
            model_dir=model_dir,
            row_base=row_base,
            target_layers=target_layers,
            datapoints=datapoints,
        ))
    return work_units


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


def _aggregate_step_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int, int], dict[str, Any]]:
    out: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["step"],
            row["batch_size"],
            row["past_kv"],
            int(row.get("measure_run", 0)),
        )
        agg = out.setdefault(key, {
            "gpu_us": 0.0,
            "span_us": 0.0,
            "start_ns": None,
            "end_ns": None,
            "kernel_count": 0,
        })
        agg["gpu_us"] += row["gpu_us"]
        if "start_ns" in row and "end_ns" in row:
            start_ns = int(row["start_ns"])
            end_ns = int(row["end_ns"])
            if agg["start_ns"] is None or start_ns < agg["start_ns"]:
                agg["start_ns"] = start_ns
            if agg["end_ns"] is None or end_ns > agg["end_ns"]:
                agg["end_ns"] = end_ns
            agg["span_us"] = (agg["end_ns"] - agg["start_ns"]) / 1000.0
        else:
            # Backward-compatible path for older parser rows.
            agg["span_us"] += row.get("span_us", row["gpu_us"])
        agg["kernel_count"] += row["kernel_count"]
    return out


def _latency_us_from_agg(agg: dict[str, Any], latency_source: str) -> float:
    if latency_source == "span":
        return float(agg["span_us"])
    if latency_source == "gpu":
        return float(agg["gpu_us"])
    if latency_source == "gpu_capped":
        return min(float(agg["gpu_us"]), float(agg["span_us"]))
    raise ValueError(f"unsupported latency source: {latency_source}")


def _lookup_aggs(
    parsed: dict[tuple[int, int, int, int], dict[str, Any]],
    expected_key: tuple[int, int, int],
) -> list[dict[str, Any]]:
    exact = [
        value for key, value in sorted(parsed.items())
        if key[:3] == expected_key
    ]
    if exact:
        return exact

    step, _batch_size, past_kv = expected_key
    candidate_items = [
        (key, value) for key, value in sorted(parsed.items())
        if key[0] == step and key[2] == past_kv
    ]
    candidate_batches = {key[1] for key, _ in candidate_items}
    if len(candidate_batches) == 1:
        return [value for _, value in candidate_items]
    return []


def _reduce_agg_latency(
    aggs: list[dict[str, Any]],
    *,
    latency_source: str,
    aggregation: str,
) -> tuple[float, int, int]:
    if not aggs:
        raise ValueError("cannot reduce empty aggregate list")
    values = [_latency_us_from_agg(agg, latency_source) for agg in aggs]
    if aggregation == "median":
        latency_us = float(statistics.median(values))
    elif aggregation == "mean":
        latency_us = float(statistics.fmean(values))
    elif aggregation == "trimmed_mean":
        if len(values) < 3:
            raise ValueError("trimmed_mean requires at least 3 measured runs")
        latency_us = float(statistics.fmean(sorted(values)[1:-1]))
    elif aggregation == "min":
        latency_us = float(min(values))
    else:
        raise ValueError(f"unsupported repeat aggregation: {aggregation}")
    kernel_count = int(statistics.median([int(agg["kernel_count"]) for agg in aggs]))
    return latency_us, kernel_count, len(aggs)


def _write_csv_header_if_needed(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()


def _append_success_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f, fcntl.LOCK_UN)


class Scheduler:
    """One-GPU-slot scheduler for nsys-wrapped workers."""

    def __init__(self, args: argparse.Namespace, work_units: list[WorkUnit]):
        self.args = args
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
        }
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        spec = self._make_spec(unit, pending, attempt_id)
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
        extra_vllm_args.extend(shlex.split(self.args.extra_vllm_args))
        extra_vllm_args.extend(self.args.extra_vllm_arg)

        return {
            "attempt_id": attempt_id,
            "work_unit_id": unit.work_unit_id,
            "model_dir": unit.model_dir,
            "target_layers": unit.target_layers,
            "datapoints": [asdict(dp) for dp in pending],
            "status_path": str(self.store.status_path),
            "restrict_cudagraph_sizes": not self.args.no_restrict_cudagraph_sizes,
            "extra_vllm_args": extra_vllm_args,
            "min_max_num_batched_tokens": self.args.min_max_num_batched_tokens,
            "ctx_warmup_runs": self.args.ctx_warmup_runs,
            "ctx_measured_runs": self.args.ctx_measured_runs,
        }

    def _worker_cmd(self, spec_path: Path, report_base: Path) -> list[str]:
        worker_cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--spec", str(spec_path)]
        return [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-graph-trace=node",
            "--force-overwrite=true",
            "-o",
            str(report_base),
            *worker_cmd,
        ]

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

        try:
            rows, meta = parse_step_sweep(
                str(sqlite_path),
                rollup=self.args.rollup,
                layer=None,
                rank_reduce=self.args.rank_reduce,
            )
            rows = _filter_rows_to_target_layers(rows, attempt.work_unit.target_layers)
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
            latency_us, kernel_count, measure_count = _reduce_agg_latency(
                aggs,
                latency_source=self.args.latency_source,
                aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else "median",
            )
            row = {
                **attempt.work_unit.row_base,
                "phase": dp.phase,
                "batch_size": dp.batch_size,
                "new_tokens": dp.new_tokens,
                "past_kv": dp.past_kv,
                "latency_ms": latency_us / 1000.0,
            }
            _append_success_row(self.output_path, row)
            self.store.append_event(
                "success",
                work_unit_id=attempt.work_unit.work_unit_id,
                datapoint_id=dpid,
                attempt_id=attempt.attempt_id,
                latency_ms=row["latency_ms"],
                latency_source=self.args.latency_source,
                repeat_aggregation=self.args.ctx_repeat_aggregation if dp.phase == "ctx" else "median",
                measure_count=measure_count,
                kernel_count=kernel_count,
                sqlite=str(sqlite_path),
            )
            successes += 1
        return successes

    def _mark_clean_parse_failures(self, attempt: Attempt) -> None:
        index = self.store.index()
        for dpid in sorted(attempt.pending_ids):
            if index.is_terminal(dpid):
                continue
            if dpid in index.started or dpid in index.completed:
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


def _tail(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


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


def _run_generate(
    llm,
    sampling_params,
    *,
    batch_size: int,
    input_len: int,
    token_config: RandomPromptTokenConfig,
) -> None:
    llm.generate(
        _dummy_prompts(batch_size, input_len, token_config),
        sampling_params=sampling_params,
        use_tqdm=False,
    )


def _engine_tokens(
    *,
    model_dir: str,
    datapoints: list[DataPoint],
    restrict_cudagraph_sizes: bool,
    extra_vllm_args: list[str],
    min_max_num_batched_tokens: int = 1,
) -> list[str]:
    ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
    gen_points = [dp for dp in datapoints if dp.phase == "gen"]
    ctx_max_new = max((dp.new_tokens for dp in ctx_points), default=0)
    ctx_max_total = max((dp.new_tokens + dp.past_kv for dp in ctx_points), default=0)
    gen_max_past = max((dp.past_kv for dp in gen_points), default=0)
    gen_batch_sizes = sorted({dp.batch_size for dp in gen_points})
    max_seq_len = max(
        2,
        ctx_max_total + 1 if ctx_points else 0,
        gen_max_past + 2 if gen_points else 0,
    )
    max_num_batched_tokens = _max_num_batched_tokens_for_datapoints(
        datapoints,
        min_max_num_batched_tokens,
    )

    tokens = [
        "--model",
        model_dir,
        "--no-async-scheduling",
        "--max-model-len",
        str(max_seq_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
    ]

    if gen_batch_sizes:
        tokens.extend(["--max-num-seqs", str(max(gen_batch_sizes))])
        # Match real vLLM deployment shape: one engine.  FULL_DECODE_ONLY lets
        # vLLM use full CUDA graphs for uniform decode batches while running
        # prefill/mixed batches through its normal non-full-graph path.
        compilation_config: dict[str, Any] = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
        }
        if restrict_cudagraph_sizes:
            compilation_config.update({
                "cudagraph_capture_sizes": gen_batch_sizes,
                "max_cudagraph_capture_size": max(gen_batch_sizes),
            })
    else:
        compilation_config = {"mode": 0, "cudagraph_mode": "NONE"}

    tokens.extend(["--compilation-config", json.dumps(compilation_config)])
    tokens.extend(extra_vllm_args)
    return tokens


def _create_llm(engine_tokens: list[str]):
    from vllm.engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(add_help=False)
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(
        load_format="dummy",
        trust_remote_code=True,
        enable_layerwise_nvtx_tracing=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=False,
    )
    args = parser.parse_args(engine_tokens)
    engine_args = EngineArgs.from_cli_args(args)
    from vllm import LLM

    return LLM.from_engine_args(engine_args)


def _classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if _is_oom_text(text):
        return "oom"
    if _is_fatal_cuda_text(text):
        return "fatal_cuda"
    return "error"


def _worker_datapoint_id(work_unit_id: str, dp: DataPoint) -> str:
    return dp.datapoint_id(work_unit_id)


def _ctx_marker_milestone(dp: DataPoint, max_num_batched_tokens: int) -> int:
    if max_num_batched_tokens < 1:
        raise ValueError(f"max_num_batched_tokens must be >= 1, got {max_num_batched_tokens}")
    if dp.past_kv == 0:
        return 1
    return int(dp.past_kv // max_num_batched_tokens) + 1


def run_worker(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text())
    status_path = Path(spec["status_path"])
    work_unit_id = spec["work_unit_id"]
    datapoints = [DataPoint(**raw) for raw in spec["datapoints"]]

    # Physical GPU allocation is deliberately one GPU regardless of simulated
    # TP/EP.  The parent sets CUDA_VISIBLE_DEVICES to the slot assigned here.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ["LAYERWISE_TARGET_LAYERS"] = ",".join(str(x) for x in spec["target_layers"])
    max_num_batched_tokens = _max_num_batched_tokens_for_datapoints(
        datapoints,
        int(spec.get("min_max_num_batched_tokens", 1)),
    )
    milestones = {1}
    milestones.update(
        _ctx_marker_milestone(dp, max_num_batched_tokens)
        for dp in datapoints
        if dp.phase == "ctx"
    )
    milestones.update(dp.past_kv + 1 for dp in datapoints if dp.phase == "gen")
    os.environ["LAYERWISE_STEP_MILESTONES"] = ",".join(str(x) for x in sorted(milestones))
    os.environ["LAYERWISE_BENCH_MIN_NEW"] = "1"
    os.environ["LAYERWISE_PROGRESS_FILE"] = str(status_path)
    os.environ["LAYERWISE_WORK_UNIT_ID"] = work_unit_id
    # Keep the marker installed but inactive during vLLM profile/capture work.
    os.environ["LAYERWISE_ACTIVE_MILESTONES"] = ""

    sys.path.insert(0, str(_THIS_DIR))
    import multiprocessing as mp

    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

    # These patches install vLLM hooks at import time.  Keep them inside worker
    # mode so scheduler/test imports remain vLLM-free.
    import vllm_layer_skip_patch  # noqa: F401
    import vllm_step_marker
    from vllm import SamplingParams

    _worker_append_event(status_path, "work_unit_started", work_unit_id=work_unit_id, attempt_id=spec["attempt_id"])
    _worker_append_event(status_path, "engine_started", work_unit_id=work_unit_id)
    prompt_token_config = load_random_prompt_token_config(spec["model_dir"])
    llm = _create_llm(_engine_tokens(
        model_dir=spec["model_dir"],
        datapoints=datapoints,
        restrict_cudagraph_sizes=spec["restrict_cudagraph_sizes"],
        extra_vllm_args=spec["extra_vllm_args"],
        min_max_num_batched_tokens=spec.get("min_max_num_batched_tokens", 1),
    ))
    _worker_append_event(status_path, "engine_ready", work_unit_id=work_unit_id)
    try:
        ctx_points = [dp for dp in datapoints if dp.phase == "ctx"]
        if ctx_points:
            os.environ["LAYERWISE_PROGRESS_PHASE"] = "ctx"
            os.environ["LAYERWISE_ACTIVE_MILESTONES"] = "1"
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
            gen_milestones = sorted({dp.past_kv + 1 for dp in gen_points})
            os.environ["LAYERWISE_PROGRESS_PHASE"] = "gen"
            os.environ["LAYERWISE_ACTIVE_MILESTONES"] = ",".join(str(x) for x in gen_milestones)
            _worker_run_gen(
                status_path,
                work_unit_id,
                llm,
                SamplingParams,
                gen_points,
                prompt_token_config=prompt_token_config,
            )
    finally:
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        os.environ.pop("LAYERWISE_ACTIVE_MILESTONES", None)
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
    for dp in datapoints:
        dpid = _worker_datapoint_id(work_unit_id, dp)
        if dpid in pruned:
            continue
        input_len = dp.past_kv + dp.new_tokens
        marker_milestone = _ctx_marker_milestone(dp, max_num_batched_tokens)
        marker_mod.set_forced_step_meta(step=dp.new_tokens, bs=dp.batch_size, past=dp.past_kv)
        try:
            os.environ["LAYERWISE_ACTIVE_MILESTONES"] = ""
            for _ in range(warmup_runs):
                _run_generate(
                    llm,
                    sampling_params,
                    batch_size=dp.batch_size,
                    input_len=input_len,
                    token_config=prompt_token_config,
                )
            os.environ["LAYERWISE_ACTIVE_MILESTONES"] = str(marker_milestone)
            for run_idx in range(measured_runs):
                marker_mod.set_forced_step_meta(
                    step=dp.new_tokens,
                    bs=dp.batch_size,
                    past=dp.past_kv,
                    run=run_idx,
                )
                _run_generate(
                    llm,
                    sampling_params,
                    batch_size=dp.batch_size,
                    input_len=input_len,
                    token_config=prompt_token_config,
                )
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
            marker_mod.clear_forced_step_meta()


def _worker_run_gen(
    status_path: Path,
    work_unit_id: str,
    llm,
    sampling_cls,
    datapoints: list[DataPoint],
    *,
    prompt_token_config: RandomPromptTokenConfig,
) -> None:
    by_batch: dict[int, list[DataPoint]] = {}
    for dp in datapoints:
        by_batch.setdefault(dp.batch_size, []).append(dp)
    max_past = max(dp.past_kv for dp in datapoints)
    sampling_params = sampling_cls(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=max_past + 1,
        detokenize=False,
    )
    for batch_size in sorted(by_batch):
        os.environ["LAYERWISE_PROGRESS_PHASE"] = "gen"
        _worker_append_event(
            status_path,
            "batch_started",
            work_unit_id=work_unit_id,
            batch_size=batch_size,
            max_past_kv=max_past,
        )
        # Gen datapoint starts/completions are emitted by vllm_step_marker at
        # each milestone.  One generate call intentionally covers many past_kv
        # datapoints for the same batch size.
        _run_generate(
            llm,
            sampling_params,
            batch_size=batch_size,
            input_len=1,
            token_config=prompt_token_config,
        )
        _worker_append_event(status_path, "batch_finished", work_unit_id=work_unit_id, batch_size=batch_size)


def _worker_empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--output", default="vllm_layerwise_perf.csv")
    parser.add_argument("--work-dir", default="profiles/vllm_layerwise")
    parser.add_argument("--config-cache-dir", default=None)
    parser.add_argument("--no-config-cache", action="store_true")
    parser.add_argument("--system", default=None)
    parser.add_argument("--framework-version", default=None)
    parser.add_argument("--tp-sizes", default="1,2,4,8")
    parser.add_argument("--moe-tp", type=int, default=1)
    parser.add_argument("--num-slots", type=int, default=None)
    parser.add_argument("--include-moe-layer", action="store_true")
    parser.add_argument(
        "--target-layer-count",
        type=int,
        default=1,
        help="Number of initial dense layers to keep in the patched model.",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help=(
            "Comma-separated explicit dense layer indices to keep. Overrides "
            "--target-layer-count and patches num_hidden_layers to max(index)+1."
        ),
    )
    parser.add_argument(
        "--target-layer-config-depth",
        type=int,
        default=None,
        help=(
            "Dense config depth to instantiate when using layer skipping. "
            "Defaults to the minimum depth needed for the kept layers."
        ),
    )
    parser.add_argument("--phases", choices=("ctx", "gen", "both"), default="both")
    parser.add_argument("--ctx-new-tokens", default=",".join(map(str, CTX_NEW_TOKENS)))
    parser.add_argument("--ctx-past-kv", default=",".join(map(str, CTX_PAST_KV)))
    parser.add_argument("--gen-batch-sizes", default=",".join(map(str, GEN_BATCH_SIZES)))
    parser.add_argument("--gen-past-kv", default=",".join(map(str, GEN_PAST_KV)))
    parser.add_argument("--gemm-quant", default="bf16")
    parser.add_argument("--moe-quant", default="bf16")
    parser.add_argument("--attn-quant", default="bf16")
    parser.add_argument("--kv-quant", default="bf16")
    parser.add_argument(
        "--rollup",
        default=r"layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)",
    )
    parser.add_argument("--rank-reduce", choices=("sum", "max"), default="sum")
    parser.add_argument(
        "--latency-source",
        choices=("span", "gpu", "gpu_capped"),
        default="span",
        help=(
            "Write latency_ms from attributed kernel wall span, summed GPU time, or GPU time capped by span. "
            "Default matches decode span collection."
        ),
    )
    parser.add_argument(
        "--min-max-num-batched-tokens",
        type=int,
        default=1,
        help=(
            "Floor for vLLM --max-num-batched-tokens; useful for context-only grids that need "
            "FlashInfer warmup headroom."
        ),
    )
    parser.add_argument(
        "--ctx-warmup-runs",
        type=int,
        default=0,
        help="Unmarked context runs to execute per datapoint before measurement.",
    )
    parser.add_argument(
        "--ctx-measured-runs",
        type=int,
        default=1,
        help="Marked context runs to execute per datapoint and aggregate.",
    )
    parser.add_argument(
        "--ctx-repeat-aggregation",
        choices=("median", "mean", "trimmed_mean", "min"),
        default="median",
        help="Aggregation for repeated context measurements.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--gpus", default=None, help="Comma-separated physical GPU IDs. Defaults to visible GPUs.")
    parser.add_argument("--max-workers", type=int, default=None, help="Limit concurrent one-GPU workers.")
    parser.add_argument("--no-restrict-cudagraph-sizes", action="store_true")
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    parser.add_argument("--extra-vllm-args", default="")
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker_parser = argparse.ArgumentParser(description="Internal vLLM layerwise worker")
        worker_parser.add_argument("worker")
        worker_parser.add_argument("--spec", required=True)
        worker_args = worker_parser.parse_args()
        run_worker(Path(worker_args.spec))
        return

    parser = _build_arg_parser()
    args = parser.parse_args()
    work_units = build_work_units(args)
    Scheduler(args, work_units).run()


if __name__ == "__main__":
    main()
