# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit pure-prefill/pure-decode cases on Dynamo's native benchmark.

The base scheduler continues to own request construction, prefix seeding,
fake-decode construction, scheduling, model execution, KV management and
cleanup. This adapter changes only the explicit grid/repeat/result hooks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from dynamo.vllm.instrumented_scheduler import (
    BenchmarkPoint,
    BenchmarkPointResult,
)
from dynamo.vllm.instrumented_scheduler import (
    InstrumentedScheduler as _BaseInstrumentedScheduler,
)

logger = logging.getLogger(__name__)
ENV_CASE_CONFIG = "DYN_FPM_CASE_CONFIG"
_LEGACY_POINT_FIELDS = {
    "point_type",
    "isl",
    "kv_read_tokens",
    "context_length",
    "batch_size",
}
_GRAPH_AWARE_POINT_FIELDS = {
    "point_type",
    "benchmark_id",
    "total_prefill_tokens",
    "total_kv_read_tokens",
    "batch_size",
}
_POINT_FIELDS = set(getattr(BenchmarkPoint, "__dataclass_fields__", {}))
_GRAPH_AWARE_RUNTIME = _GRAPH_AWARE_POINT_FIELDS <= _POINT_FIELDS
if not _GRAPH_AWARE_RUNTIME and not _LEGACY_POINT_FIELDS <= _POINT_FIELDS:
    raise RuntimeError(f"unsupported Dynamo BenchmarkPoint contract: fields={sorted(_POINT_FIELDS)}")


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


class InstrumentedScheduler(_BaseInstrumentedScheduler):
    """Run a frozen, latency-blind ordered FPM point design."""

    def _capacity(self, item: dict) -> tuple[BenchmarkPoint | None, dict, str | None]:
        point_type = str(item["workload_kind"])
        batch = int(item["batch_size"])
        suffix = int(item["suffix_length"])
        prefix = int(item["prefix_length"])
        if point_type not in {"prefill", "decode"}:
            raise ValueError(f"invalid workload_kind: {point_type!r}")
        if batch < 1 or suffix < 1 or prefix < 0:
            raise ValueError(f"invalid FPM point: {item}")
        if point_type == "decode" and (suffix != 1 or prefix < 1):
            raise ValueError(f"decode requires suffix_length=1 and prefix_length>0: {item}")
        if point_type == "prefill" and prefix and prefix % self.block_size:
            raise ValueError(f"prefill P must align to block_size={self.block_size}: {item}")

        if _GRAPH_AWARE_RUNTIME:
            available = int(self._bench_available_blocks())
            effective_available = int(
                self._bench_usable_blocks(
                    batch,
                    reserve_watermark=batch > 1 or point_type == "decode",
                )
            )
            watermark = max(0, available - effective_available)
        else:
            block_pool = self.kv_cache_manager.block_pool
            available = int(block_pool.get_num_free_blocks())
            watermark = int(getattr(self.kv_cache_manager, "watermark_blocks", 0))
            effective_available = max(0, available - watermark) if batch > 1 or point_type == "decode" else available
        total_length = prefix + suffix
        blocks_per_request = 0
        required = 0
        reason = None

        if batch > self.max_num_running_reqs:
            reason = "MaxRunningRequests"
        elif total_length >= self.max_model_len:
            reason = "MaxModelLength"
        elif batch * suffix > self.max_num_scheduled_tokens:
            reason = "MaxScheduledTokens"
        elif point_type == "prefill":
            scheduled = self._bench_prefill_scheduled_tokens_per_req(total_length, prefix)
            if scheduled != suffix:
                reason = "NonAtomicFreshSuffix"
            else:
                blocks_per_request = int(self._bench_prefill_blocks_per_req(total_length, prefix))
                required = batch * blocks_per_request
                if _GRAPH_AWARE_RUNTIME and prefix > 0:
                    seed_prompt_length = int(self._bench_seed_prompt_len(prefix))
                    seed_required = batch * int(
                        self._bench_blocks_per_req(
                            seed_prompt_length,
                            has_cache_hit=False,
                            apply_admission_cap=False,
                        )
                    )
                    required = max(required, seed_required)
        else:
            blocks_per_request = int(self._bench_blocks_per_req(prefix + 1))
            required = batch * blocks_per_request

        if reason is None and required > effective_available:
            reason = "RuntimeKVCapacity"
        if reason is None and _GRAPH_AWARE_RUNTIME:
            feasible = (
                self._bench_prefill_point_feasible(batch * suffix, batch, batch * prefix)
                if point_type == "prefill"
                else self._bench_decode_point_feasible(batch, batch * prefix)
            )
            if not feasible:
                reason = "RuntimeFeasibility"
        capacity = {
            "blocks_per_request": blocks_per_request,
            "required_physical_blocks": required,
            "available_physical_blocks": available,
            "effective_available_physical_blocks": effective_available,
            "watermark_blocks": watermark,
        }
        if reason is not None:
            return None, capacity, reason
        if _GRAPH_AWARE_RUNTIME:
            scheduled_tokens = batch * suffix if point_type == "prefill" else batch
            capture_sizes = (
                self._bench_prefill_capture_sizes if point_type == "prefill" else self._bench_decode_capture_sizes
            )
            axis_limit = (
                self.max_num_scheduled_tokens
                if point_type == "prefill"
                else min(self.max_num_running_reqs, self.max_num_scheduled_tokens)
            )
            capture_size, padding_tokens, sample_reasons = self._bench_cudagraph_metadata(
                scheduled_tokens,
                capture_sizes,
                axis_limit,
            )
            mode = self._bench_prefill_cudagraph_mode if point_type == "prefill" else self._bench_decode_cudagraph_mode
            point = BenchmarkPoint(
                point_type=point_type,
                total_prefill_tokens=batch * suffix if point_type == "prefill" else 0,
                total_kv_read_tokens=batch * prefix,
                batch_size=batch,
                expected_cudagraph_mode=mode if capture_size is not None else "NONE",
                expected_capture_size=capture_size,
                padding_tokens=padding_tokens,
                sample_reasons=[*sample_reasons, "collector_explicit"],
            )
        elif point_type == "prefill":
            point = BenchmarkPoint(
                point_type="prefill",
                isl=total_length,
                kv_read_tokens=prefix,
                context_length=suffix,
                batch_size=batch,
            )
        else:
            point = BenchmarkPoint(
                point_type="decode",
                context_length=prefix,
                batch_size=batch,
            )
        return point, capacity, None

    def _bench_build_grid(self) -> None:
        if self._bench_grid_built:
            return
        config_path_raw = os.environ.get(ENV_CASE_CONFIG)
        if not config_path_raw:
            raise ValueError(f"{ENV_CASE_CONFIG} is required")
        config_path = Path(config_path_raw)
        raw = json.loads(config_path.read_text())
        global_warmup_iterations = int(raw.get("global_warmup_iterations", 0))
        per_point_warmups = int(raw.get("warmup_repeats", 0))
        repeats = int(raw.get("measured_repeats", 1))
        target_count = int(raw["selected_point_count"])
        shapes = raw.get("ordered_shapes")
        if global_warmup_iterations < 0 or per_point_warmups < 0 or repeats < 1:
            raise ValueError("global_warmup_iterations and warmup_repeats must be >=0; measured_repeats must be >=1")
        if per_point_warmups != 0:
            raise ValueError(
                "per-point warmup_repeats is unsupported; use the Dynamo scheduler global_warmup_iterations"
            )
        if _GRAPH_AWARE_RUNTIME and repeats != 1:
            raise ValueError("PR11509 runtime requires measured_repeats=1")
        runtime_warmup_iterations = int(self._bench_config.warmup_iterations)
        if runtime_warmup_iterations != global_warmup_iterations:
            raise ValueError(
                "global warmup mismatch between case manifest and Dynamo scheduler: "
                f"manifest={global_warmup_iterations}, runtime={runtime_warmup_iterations}"
            )
        if not isinstance(shapes, list) or not shapes:
            raise ValueError("ordered_shapes must be a non-empty list")
        if target_count < 1 or target_count > len(shapes):
            raise ValueError("selected_point_count is outside ordered_shapes")

        eligible = []
        cancelled = []
        seen = set()
        for design_index, item in enumerate(shapes):
            key = (
                str(item["workload_kind"]),
                int(item["batch_size"]),
                int(item["suffix_length"]),
                int(item["prefix_length"]),
            )
            if key in seen:
                raise ValueError(f"duplicate FPM point: {item}")
            seen.add(key)
            point, capacity, reason = self._capacity(item)
            point_dict = {
                "workload_kind": key[0],
                "batch_size": key[1],
                "suffix_length": key[2],
                "prefix_length": key[3],
            }
            if reason is not None:
                cancelled.append(
                    {
                        "design_index": design_index,
                        "point": point_dict,
                        "reason_type": reason,
                        **capacity,
                    }
                )
            else:
                eligible.append((design_index, point_dict, point, capacity))

        capacity_sufficient = len(eligible) >= target_count
        selected = eligible[:target_count] if capacity_sufficient else []
        points = []
        execution_meta = []
        for design_index, point_dict, point, capacity in selected:
            repeat_range = range(0, 1) if _GRAPH_AWARE_RUNTIME else range(repeats)
            for repeat in repeat_range:
                points.append(point)
                execution_meta.append(
                    {
                        "design_index": design_index,
                        "point": point_dict,
                        "repeat": repeat,
                        "measured": True,
                        **capacity,
                    }
                )
        if _GRAPH_AWARE_RUNTIME:
            for benchmark_id, (point, meta) in enumerate(
                zip(points, execution_meta, strict=True),
                start=1,
            ):
                point.benchmark_id = benchmark_id
                meta["benchmark_id"] = benchmark_id

        self._fpm_case_path = config_path
        self._fpm_case_raw = raw
        self._fpm_global_warmup_iterations = global_warmup_iterations
        self._fpm_repeats = repeats
        self._fpm_population_count = len(shapes)
        self._fpm_target_count = target_count
        self._fpm_eligible = eligible
        self._fpm_selected = selected
        self._fpm_cancelled = cancelled
        self._fpm_execution_meta = execution_meta
        self._fpm_pending_execution_meta = deque(execution_meta)
        self._fpm_canary_completed = False
        self._fpm_started_unix_ns = time.time_ns()
        self._fpm_started_monotonic_ns = time.monotonic_ns()
        self._fpm_result_written = False

        self._bench_grid.extend(points)
        self._bench_expected_points = len(points) if capacity_sufficient else target_count * repeats
        self._bench_grid_built = True
        if _GRAPH_AWARE_RUNTIME:
            grid_payload = json.dumps(
                [asdict(point) for point in self._bench_grid],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            self._bench_grid_digest = hashlib.sha256(grid_payload).hexdigest()
        if not capacity_sufficient:
            logger.error(
                "AIC FPM runtime capacity admits only %d/%d selected points; "
                "writing an invalid terminal result without executing forwards",
                len(eligible),
                target_count,
            )
        logger.info(
            "AIC FPM explicit grid: population=%d eligible=%d selected=%d executions=%d "
            "global_warmup_iterations=%d per_point_warmups=0 repeats=%d",
            len(shapes),
            len(eligible),
            len(selected),
            len(points),
            global_warmup_iterations,
            repeats,
        )

    def _bench_pop_next(self, point_type: str) -> BenchmarkPoint | None:
        point = super()._bench_pop_next(point_type)
        if point is not None:
            self._fpm_active_execution_meta = self._fpm_pending_execution_meta.popleft()
        return point

    def _bench_skip_point(self, point: BenchmarkPoint, reason: str) -> None:
        super()._bench_skip_point(point, reason)
        if _GRAPH_AWARE_RUNTIME:
            return
        meta = getattr(self, "_fpm_active_execution_meta", None)
        if meta is not None and meta["measured"] and not self._fpm_canary_completed:
            self._bench_grid.clear()
            self._fpm_pending_execution_meta.clear()
            logger.error(
                "AIC FPM integrated canary failed (%s); aborting the remaining formal points",
                reason,
            )

    def _bench_save_current_point(self) -> None:
        if _GRAPH_AWARE_RUNTIME:
            completed_before = len(self._bench_results)
            skipped_before = len(self._bench_skipped_points)
            meta = getattr(self, "_fpm_active_execution_meta", None)
            super()._bench_save_current_point()
            if len(self._bench_results) == completed_before + 1 and meta is not None and meta["measured"]:
                self._fpm_canary_completed = True
            elif (
                len(self._bench_skipped_points) == skipped_before + 1
                and meta is not None
                and meta["measured"]
                and not self._fpm_canary_completed
            ):
                # PR11509 has already synchronized and validated every ADP
                # rank before returning from the base result hook.  Clearing
                # the remaining grid here therefore preserves Collector's
                # first-point canary without creating rank-local divergence.
                self._bench_grid.clear()
                self._fpm_pending_execution_meta.clear()
                logger.error("AIC FPM integrated canary failed; aborting the remaining formal points")
            return

        point = self._bench_current_point
        if point is not None and self._bench_current_fpms:
            meta = self._fpm_active_execution_meta
            if len(self._bench_current_fpms) != 1:
                self._bench_skip_point(point, "multiple_fpms_for_one_execution")
            else:
                fpm = self._bench_current_fpms[0]
                scheduled = fpm.get("scheduled_requests", {})
                item = meta["point"]
                batch = int(item["batch_size"])
                suffix = int(item["suffix_length"])
                prefix = int(item["prefix_length"])
                if item["workload_kind"] == "prefill":
                    expected = {
                        "num_prefill_requests": batch,
                        "sum_prefill_tokens": batch * suffix,
                        "sum_prefill_kv_tokens": batch * prefix,
                        "num_decode_requests": 0,
                        "sum_decode_kv_tokens": 0,
                    }
                else:
                    expected = {
                        "num_prefill_requests": 0,
                        "sum_prefill_tokens": 0,
                        "sum_prefill_kv_tokens": 0,
                        "num_decode_requests": batch,
                        "sum_decode_kv_tokens": batch * prefix,
                    }
                mismatches = {
                    name: {"actual": scheduled.get(name), "expected": value}
                    for name, value in expected.items()
                    if scheduled.get(name) != value
                }
                wall_time = float(fpm.get("wall_time", 0.0))
                if not math.isfinite(wall_time) or wall_time <= 0:
                    mismatches["wall_time"] = {"actual": wall_time, "expected": "finite and > 0"}
                if mismatches:
                    self._bench_skip_point(point, f"explicit_fpm_mismatch:{mismatches}")
                else:
                    self._bench_results.append(BenchmarkPointResult(point=point, fpms=list(self._bench_current_fpms)))
                    if meta["measured"] and not self._fpm_canary_completed:
                        self._fpm_canary_completed = True
        self._bench_current_point = None
        self._bench_current_fpms = []

    def _bench_write_results(self) -> None:
        if self._fpm_result_written:
            return
        if _GRAPH_AWARE_RUNTIME:
            self._bench_write_graph_aware_results()
            return

        self._bench_write_legacy_results()

    def _collector_envelope(self, *, elapsed_seconds: float) -> dict[str, object]:
        return {
            "schema_name": "aic_fpm_raw_result",
            "schema_version": 1,
            "runtime_contract": ("dynamo_pr11509_schema_v2" if _GRAPH_AWARE_RUNTIME else "dynamo_legacy_schema_v1"),
            "plan_sha256": self._fpm_case_raw.get("plan_sha256"),
            "cell_id": self._fpm_case_raw.get("cell_id"),
            "global_warmup_iterations": self._fpm_global_warmup_iterations,
            "warmup_repeats": 0,
            "measured_repeats": self._fpm_repeats,
            "measurement_policy": "single_sample_v1",
            "population_count": self._fpm_population_count,
            "selected_point_count": self._fpm_target_count,
            "capacity_eligible_count": len(self._fpm_eligible),
            "capacity_cancelled_count": len(self._fpm_cancelled),
            "latency_read_during_capacity_filter": False,
            "case_sha256": hashlib.sha256(self._fpm_case_path.read_bytes()).hexdigest(),
            "elapsed_seconds": elapsed_seconds,
        }

    def _bench_write_graph_aware_results(self) -> None:
        destination = Path(self._bench_config.output_path)
        upstream_destination = destination.with_name(f".{destination.name}.upstream")
        original_destination = self._bench_config.output_path
        self._bench_config.output_path = str(upstream_destination)
        try:
            super()._bench_write_results()
        finally:
            self._bench_config.output_path = original_destination

        try:
            payload = json.loads(upstream_destination.read_text())
            if payload.get("schema_version") != 2:
                raise ValueError(
                    f"PR11509 base scheduler emitted an unexpected schema: {payload.get('schema_version')!r}"
                )
            config = payload.get("config")
            if isinstance(config, dict):
                config["output_path"] = original_destination
            timing = payload.get("timing")
            elapsed_seconds = float(timing.get("benchmark_elapsed_seconds", 0.0)) if isinstance(timing, dict) else 0.0
            payload["collector"] = self._collector_envelope(elapsed_seconds=elapsed_seconds)
            meta_by_id = {int(meta["benchmark_id"]): meta for meta in self._fpm_execution_meta}
            campaign_results = []
            if payload.get("valid") is True:
                for result in payload.get("results", []):
                    benchmark_id = int(result["point"]["benchmark_id"])
                    meta = meta_by_id.get(benchmark_id)
                    if meta is None:
                        raise ValueError(f"PR11509 result has unknown benchmark_id={benchmark_id}")
                    campaign_results.append(
                        {
                            "design_index": int(meta["design_index"]),
                            "point": meta["point"],
                            "warmup_fpms": [],
                            "fpms": result["fpms"],
                        }
                    )
            payload["campaign_results"] = campaign_results
            payload["cancelled_points"] = self._fpm_cancelled
            _atomic_json(destination, payload)
            self._fpm_result_written = True
        finally:
            upstream_destination.unlink(missing_ok=True)

    def _bench_write_legacy_results(self) -> None:
        completed = len(self._bench_results)
        skipped = len(self._bench_skipped_points)
        missing_phases = list(getattr(self, "_bench_missing_phases", []))
        valid = completed == self._bench_expected_points and skipped == 0 and not missing_phases
        raw_results = [{"point": asdict(result.point), "fpms": result.fpms} for result in self._bench_results]
        grouped = {}
        if valid:
            for result, meta in zip(raw_results, self._fpm_execution_meta, strict=True):
                index = int(meta["design_index"])
                row = grouped.setdefault(
                    index,
                    {
                        "design_index": index,
                        "point": meta["point"],
                        "warmup_fpms": [],
                        "fpms": [],
                    },
                )
                row["fpms" if meta["measured"] else "warmup_fpms"].extend(result["fpms"])
        output = {
            "schema_version": 1,
            "status": "complete",
            "valid": valid,
            "coverage": {
                "expected_points": self._bench_expected_points,
                "completed_points": completed,
                "skipped_points": skipped,
            },
            "config": asdict(self._bench_config),
            "collector": self._collector_envelope(
                elapsed_seconds=(time.monotonic_ns() - self._fpm_started_monotonic_ns) / 1e9
            ),
            "campaign_results": [grouped[index] for index in sorted(grouped)],
            "cancelled_points": self._fpm_cancelled,
            "results": raw_results,
            "skipped_points": [
                {"point": asdict(item.point), "reason": item.reason} for item in self._bench_skipped_points
            ],
            "missing_phases": missing_phases,
        }
        destination = self._bench_config.output_path
        temporary = destination + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(output, handle, indent=2)
        os.replace(temporary, destination)
        self._fpm_result_written = True
