# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _BenchmarkPoint:
    point_type: str
    isl: int = 0
    kv_read_tokens: int = 0
    context_length: int = 0
    batch_size: int = 1


@dataclass
class _BenchmarkPointResult:
    point: _BenchmarkPoint
    fpms: list = field(default_factory=list)


class _BaseScheduler:
    def _bench_pop_next(self, point_type):
        if self._bench_grid and self._bench_grid[0].point_type == point_type:
            return self._bench_grid.popleft()
        return None

    def _bench_skip_point(self, point, reason):
        self._bench_skipped_points.append((point, reason))


@dataclass
class _BenchConfig:
    mode: str
    output_path: str
    warmup_iterations: int = 0


@dataclass
class _GraphAwareBenchmarkPoint:
    point_type: str
    benchmark_id: int = 0
    total_prefill_tokens: int = 0
    total_kv_read_tokens: int = 0
    batch_size: int = 1
    expected_cudagraph_mode: str = "NONE"
    expected_capture_size: int | None = None
    padding_tokens: int | None = None
    sample_reasons: list[str] = field(default_factory=list)


class _GraphAwareBaseScheduler(_BaseScheduler):
    def _bench_save_current_point(self):
        failure_reason = getattr(self, "_synchronized_failure_reason", None)
        if self._bench_current_point is not None and failure_reason is not None:
            self._bench_skip_point(self._bench_current_point, failure_reason)
        elif self._bench_current_point is not None and len(self._bench_current_fpms) == 1:
            self._bench_results.append(
                _BenchmarkPointResult(
                    point=self._bench_current_point,
                    fpms=list(self._bench_current_fpms),
                )
            )
        self._bench_current_point = None
        self._bench_current_fpms = []

    def _bench_write_results(self):
        completed = len(self._bench_results)
        valid = completed == self._bench_expected_points and not self._bench_skipped_points
        output = {
            "schema_version": 2,
            "artifact_type": "rank",
            "status": "complete",
            "valid": valid,
            "usable": valid,
            "coverage": {
                "expected_points": self._bench_expected_points,
                "completed_points": completed,
                "skipped_points": len(self._bench_skipped_points),
            },
            "config": asdict(self._bench_config),
            "timing": {
                "started_at": "start",
                "completed_at": "end",
                "benchmark_elapsed_seconds": 1.25,
                "measured_iteration_seconds": 0.01,
            },
            "results": [{"point": asdict(result.point), "fpms": result.fpms} for result in self._bench_results],
            "iteration_groups": [],
            "skipped_points": [],
            "missing_phases": [],
            "error": None,
        }
        Path(self._bench_config.output_path).write_text(json.dumps(output))


def _load_adapter(monkeypatch, point_type, scheduler_type, module_name):
    module = types.ModuleType("dynamo.vllm.instrumented_scheduler")
    module.BenchmarkPoint = point_type
    module.BenchmarkPointResult = _BenchmarkPointResult
    module.InstrumentedScheduler = scheduler_type
    monkeypatch.setitem(sys.modules, "dynamo", types.ModuleType("dynamo"))
    monkeypatch.setitem(sys.modules, "dynamo.vllm", types.ModuleType("dynamo.vllm"))
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", module)

    path = Path(__file__).parents[3] / "collector" / "fpm_forward" / "runtime" / "vllm_scheduler.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def adapter(monkeypatch):
    loaded = _load_adapter(
        monkeypatch,
        _BenchmarkPoint,
        _BaseScheduler,
        "test_fpm_scheduler",
    )
    scheduler = object.__new__(loaded.InstrumentedScheduler)
    scheduler.block_size = 64
    scheduler.max_num_running_reqs = 32
    scheduler.max_model_len = 8192
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 1000),
        watermark_blocks=1,
    )
    scheduler._bench_prefill_scheduled_tokens_per_req = lambda total, prefix: total - prefix
    scheduler._bench_prefill_blocks_per_req = lambda total, prefix: (total + 63) // 64
    scheduler._bench_blocks_per_req = lambda total: (total + 63) // 64
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path="", warmup_iterations=0)
    return scheduler, loaded


@pytest.fixture
def graph_aware_adapter(monkeypatch):
    loaded = _load_adapter(
        monkeypatch,
        _GraphAwareBenchmarkPoint,
        _GraphAwareBaseScheduler,
        "test_fpm_scheduler_pr11509",
    )
    scheduler = object.__new__(loaded.InstrumentedScheduler)
    scheduler.block_size = 64
    scheduler.max_num_running_reqs = 32
    scheduler.max_model_len = 8192
    scheduler.max_num_scheduled_tokens = 4096
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=SimpleNamespace(get_num_free_blocks=lambda: 1000),
        watermark_blocks=1,
    )
    scheduler._bench_available_blocks = lambda: 1000
    scheduler._bench_usable_blocks = lambda _batch, reserve_watermark=False: 999 if reserve_watermark else 1000
    scheduler._bench_prefill_scheduled_tokens_per_req = lambda total, prefix: total - prefix
    scheduler._bench_prefill_blocks_per_req = lambda total, _prefix: (total + 63) // 64
    scheduler._bench_blocks_per_req = lambda total, **_kwargs: (total + 63) // 64
    scheduler._bench_seed_prompt_len = lambda prefix: prefix
    scheduler._bench_prefill_point_feasible = (
        lambda total, batch, total_kv: total <= 4096 and batch <= 32 and total_kv >= 0
    )
    scheduler._bench_decode_point_feasible = lambda batch, total_kv: batch <= 32 and total_kv >= batch
    scheduler._bench_prefill_capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    scheduler._bench_decode_capture_sizes = [1, 2, 4, 8, 16, 32]
    scheduler._bench_prefill_cudagraph_mode = "PIECEWISE"
    scheduler._bench_decode_cudagraph_mode = "FULL"

    def cudagraph_metadata(tokens, capture_sizes, axis_limit):
        capture = next((size for size in capture_sizes if size >= tokens), None)
        return (
            capture,
            None if capture is None else capture - tokens,
            ["engine_limit" if tokens == axis_limit else "explicit"],
        )

    scheduler._bench_cudagraph_metadata = cudagraph_metadata
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path="", warmup_iterations=0)
    return scheduler, loaded


def test_runtime_capacity_supports_prefill_past_kv_and_decode(adapter):
    scheduler, _ = adapter
    prefill, _, reason = scheduler._capacity(
        {
            "workload_kind": "prefill",
            "batch_size": 4,
            "suffix_length": 128,
            "prefix_length": 1024,
        }
    )
    assert reason is None
    assert prefill.kv_read_tokens == 1024
    assert prefill.isl == 1152

    decode, _, reason = scheduler._capacity(
        {
            "workload_kind": "decode",
            "batch_size": 4,
            "suffix_length": 1,
            "prefix_length": 1024,
        }
    )
    assert reason is None
    assert decode.point_type == "decode"
    assert decode.context_length == 1024


def test_global_warmup_does_not_duplicate_legacy_measurement_points(adapter, monkeypatch, tmp_path):
    scheduler, loaded = adapter
    case = tmp_path / "cases.json"
    case.write_text(
        json.dumps(
            {
                "global_warmup_iterations": 4,
                "warmup_repeats": 0,
                "measured_repeats": 1,
                "selected_point_count": 1,
                "ordered_shapes": [
                    {
                        "workload_kind": "prefill",
                        "batch_size": 1,
                        "suffix_length": 64,
                        "prefix_length": 0,
                    }
                ],
            }
        )
    )
    monkeypatch.setenv(loaded.ENV_CASE_CONFIG, str(case))
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path="", warmup_iterations=4)
    scheduler._bench_grid = deque()
    scheduler._bench_grid_built = False

    scheduler._bench_build_grid()

    assert len(scheduler._bench_grid) == 1
    assert scheduler._bench_expected_points == 1
    assert scheduler._fpm_global_warmup_iterations == 4
    assert [meta["measured"] for meta in scheduler._fpm_execution_meta] == [True]


def test_adapter_rejects_per_point_warmup_and_global_mismatch(adapter, monkeypatch, tmp_path):
    scheduler, loaded = adapter
    case = tmp_path / "cases.json"
    payload = {
        "global_warmup_iterations": 2,
        "warmup_repeats": 1,
        "measured_repeats": 1,
        "selected_point_count": 1,
        "ordered_shapes": [
            {
                "workload_kind": "prefill",
                "batch_size": 1,
                "suffix_length": 64,
                "prefix_length": 0,
            }
        ],
    }
    case.write_text(json.dumps(payload))
    monkeypatch.setenv(loaded.ENV_CASE_CONFIG, str(case))
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path="", warmup_iterations=2)
    scheduler._bench_grid = deque()
    scheduler._bench_grid_built = False

    with pytest.raises(ValueError, match="per-point warmup_repeats is unsupported"):
        scheduler._bench_build_grid()

    payload["warmup_repeats"] = 0
    case.write_text(json.dumps(payload))
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path="", warmup_iterations=3)
    with pytest.raises(ValueError, match="global warmup mismatch"):
        scheduler._bench_build_grid()


def test_pr11509_runtime_maps_homogeneous_points_to_iteration_totals(graph_aware_adapter):
    scheduler, loaded = graph_aware_adapter
    assert loaded._GRAPH_AWARE_RUNTIME is True

    prefill, capacity, reason = scheduler._capacity(
        {
            "workload_kind": "prefill",
            "batch_size": 4,
            "suffix_length": 128,
            "prefix_length": 1024,
        }
    )
    assert reason is None
    assert prefill.total_prefill_tokens == 512
    assert prefill.total_kv_read_tokens == 4096
    assert prefill.expected_capture_size == 512
    assert "collector_explicit" in prefill.sample_reasons
    assert capacity["required_physical_blocks"] == 72

    decode, _, reason = scheduler._capacity(
        {
            "workload_kind": "decode",
            "batch_size": 4,
            "suffix_length": 1,
            "prefix_length": 1024,
        }
    )
    assert reason is None
    assert decode.total_prefill_tokens == 0
    assert decode.total_kv_read_tokens == 4096
    assert decode.expected_capture_size == 4


def test_runtime_preflight_recognizes_pr11509_contract(graph_aware_adapter):
    assert graph_aware_adapter[1]._GRAPH_AWARE_RUNTIME is True
    path = Path(__file__).parents[3] / "collector" / "fpm_forward" / "runtime" / "preflight.py"
    spec = importlib.util.spec_from_file_location("test_fpm_preflight_pr11509", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)

    contract, fields, methods = loaded._contract(set(_GraphAwareBenchmarkPoint.__dataclass_fields__))

    assert contract == "dynamo_pr11509_schema_v2"
    assert fields == loaded.GRAPH_AWARE_FIELDS
    assert "_bench_cache_fake_prefixes" in methods


def test_runtime_capacity_shortfall_writes_no_forward_grid(adapter, monkeypatch, tmp_path):
    scheduler, loaded = adapter
    case = tmp_path / "cases.json"
    case.write_text(
        json.dumps(
            {
                "selected_point_count": 2,
                "ordered_shapes": [
                    {
                        "workload_kind": "prefill",
                        "batch_size": 1,
                        "suffix_length": 64,
                        "prefix_length": 0,
                    },
                    {
                        "workload_kind": "prefill",
                        "batch_size": 64,
                        "suffix_length": 64,
                        "prefix_length": 0,
                    },
                ],
            }
        )
    )
    monkeypatch.setenv(loaded.ENV_CASE_CONFIG, str(case))
    scheduler._bench_grid = deque()
    scheduler._bench_grid_built = False

    scheduler._bench_build_grid()

    assert list(scheduler._bench_grid) == []
    assert scheduler._bench_expected_points == 2
    assert len(scheduler._fpm_eligible) == 1
    assert scheduler._fpm_target_count == 2

    output = tmp_path / "benchmark.json"
    scheduler._bench_results = []
    scheduler._bench_skipped_points = []
    scheduler._bench_missing_phases = []
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path=str(output))
    scheduler._bench_write_results()
    payload = json.loads(output.read_text())
    assert payload["valid"] is False
    assert payload["coverage"] == {"expected_points": 2, "completed_points": 0, "skipped_points": 0}


def test_pr11509_grid_delegates_results_and_adds_collector_contract(
    graph_aware_adapter,
    monkeypatch,
    tmp_path,
):
    scheduler, loaded = graph_aware_adapter
    case = tmp_path / "cases.json"
    case.write_text(
        json.dumps(
            {
                "plan_sha256": "plan",
                "cell_id": "cell",
                "global_warmup_iterations": 3,
                "warmup_repeats": 0,
                "measured_repeats": 1,
                "selected_point_count": 2,
                "ordered_shapes": [
                    {
                        "workload_kind": "prefill",
                        "batch_size": 2,
                        "suffix_length": 64,
                        "prefix_length": 128,
                    },
                    {
                        "workload_kind": "prefill",
                        "batch_size": 1,
                        "suffix_length": 256,
                        "prefix_length": 0,
                    },
                ],
            }
        )
    )
    monkeypatch.setenv(loaded.ENV_CASE_CONFIG, str(case))
    output = tmp_path / "benchmark.json"
    scheduler._bench_grid = deque()
    scheduler._bench_grid_built = False
    scheduler._bench_results = []
    scheduler._bench_skipped_points = []
    scheduler._bench_missing_phases = []
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path=str(output), warmup_iterations=3)

    scheduler._bench_build_grid()

    points = list(scheduler._bench_grid)
    assert [point.benchmark_id for point in points] == [1, 2]
    assert len(scheduler._bench_grid_digest) == 64
    for point in points:
        popped = scheduler._bench_pop_next("prefill")
        assert popped == point
        scheduler._bench_current_point = point
        scheduler._bench_current_fpms = [
            {
                "counter_id": point.benchmark_id,
                "dp_rank": 0,
                "wall_time": 0.01,
            }
        ]
        scheduler._bench_save_current_point()

    scheduler._bench_write_results()

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert payload["artifact_type"] == "rank"
    assert payload["config"]["output_path"] == str(output)
    assert payload["collector"]["runtime_contract"] == "dynamo_pr11509_schema_v2"
    assert payload["collector"]["plan_sha256"] == "plan"
    assert payload["collector"]["global_warmup_iterations"] == 3
    assert payload["collector"]["warmup_repeats"] == 0
    assert payload["campaign_results"] == [
        {
            "design_index": 0,
            "point": {
                "workload_kind": "prefill",
                "batch_size": 2,
                "suffix_length": 64,
                "prefix_length": 128,
            },
            "warmup_fpms": [],
            "fpms": [{"counter_id": 1, "dp_rank": 0, "wall_time": 0.01}],
        },
        {
            "design_index": 1,
            "point": {
                "workload_kind": "prefill",
                "batch_size": 1,
                "suffix_length": 256,
                "prefix_length": 0,
            },
            "warmup_fpms": [],
            "fpms": [{"counter_id": 2, "dp_rank": 0, "wall_time": 0.01}],
        },
    ]
    assert not (tmp_path / ".benchmark.json.upstream").exists()


def test_pr11509_capacity_shortfall_writes_invalid_schema_v2_without_forwards(
    graph_aware_adapter,
    monkeypatch,
    tmp_path,
):
    scheduler, loaded = graph_aware_adapter
    scheduler._bench_available_blocks = lambda: 1
    scheduler._bench_usable_blocks = lambda _batch, reserve_watermark=False: 1
    case = tmp_path / "cases.json"
    case.write_text(
        json.dumps(
            {
                "plan_sha256": "plan",
                "cell_id": "cell",
                "global_warmup_iterations": 2,
                "warmup_repeats": 0,
                "measured_repeats": 1,
                "selected_point_count": 2,
                "ordered_shapes": [
                    {
                        "workload_kind": "prefill",
                        "batch_size": 1,
                        "suffix_length": 1,
                        "prefix_length": 0,
                    },
                    {
                        "workload_kind": "prefill",
                        "batch_size": 1,
                        "suffix_length": 128,
                        "prefix_length": 0,
                    },
                ],
            }
        )
    )
    monkeypatch.setenv(loaded.ENV_CASE_CONFIG, str(case))
    output = tmp_path / "benchmark.json"
    scheduler._bench_grid = deque()
    scheduler._bench_grid_built = False
    scheduler._bench_results = []
    scheduler._bench_skipped_points = []
    scheduler._bench_missing_phases = []
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path=str(output), warmup_iterations=2)

    scheduler._bench_build_grid()
    scheduler._bench_write_results()

    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 2
    assert payload["valid"] is False
    assert payload["coverage"] == {"expected_points": 2, "completed_points": 0, "skipped_points": 0}
    assert payload["collector"]["capacity_eligible_count"] == 1
    assert payload["collector"]["capacity_cancelled_count"] == 1
    assert payload["campaign_results"] == []


def test_integrated_canary_failure_aborts_remaining_grid(adapter):
    scheduler, _ = adapter
    first = _BenchmarkPoint(point_type="prefill", isl=64)
    second = _BenchmarkPoint(point_type="prefill", isl=128)
    scheduler._bench_skipped_points = []
    scheduler._bench_grid = deque([second])
    scheduler._fpm_pending_execution_meta = deque([{"measured": True}])
    scheduler._fpm_active_execution_meta = {"measured": True}
    scheduler._fpm_canary_completed = False

    scheduler._bench_skip_point(first, "canary_failed")

    assert list(scheduler._bench_grid) == []
    assert list(scheduler._fpm_pending_execution_meta) == []
    assert scheduler._bench_skipped_points == [(first, "canary_failed")]


def test_pr11509_skip_semantics_remain_owned_by_synchronized_base(graph_aware_adapter):
    scheduler, _ = graph_aware_adapter
    first = _GraphAwareBenchmarkPoint(point_type="prefill", benchmark_id=1)
    second = _GraphAwareBenchmarkPoint(point_type="prefill", benchmark_id=2)
    scheduler._bench_skipped_points = []
    scheduler._bench_grid = deque([second])
    scheduler._fpm_canary_completed = False

    scheduler._bench_skip_point(first, "rank_synchronized_failure")

    assert list(scheduler._bench_grid) == [second]
    assert scheduler._bench_skipped_points == [(first, "rank_synchronized_failure")]


def test_pr11509_synchronized_canary_failure_aborts_remaining_grid(graph_aware_adapter):
    scheduler, _ = graph_aware_adapter
    first = _GraphAwareBenchmarkPoint(point_type="prefill", benchmark_id=1)
    second = _GraphAwareBenchmarkPoint(point_type="prefill", benchmark_id=2)
    scheduler._bench_current_point = first
    scheduler._bench_current_fpms = [{"counter_id": 1, "dp_rank": 0}]
    scheduler._bench_results = []
    scheduler._bench_skipped_points = []
    scheduler._bench_grid = deque([second])
    scheduler._fpm_pending_execution_meta = deque([{"measured": True}])
    scheduler._fpm_active_execution_meta = {"measured": True}
    scheduler._fpm_canary_completed = False
    scheduler._synchronized_failure_reason = "measured_shape_mismatch"

    scheduler._bench_save_current_point()

    assert not scheduler._bench_grid
    assert not scheduler._fpm_pending_execution_meta
    assert scheduler._bench_skipped_points == [(first, "measured_shape_mismatch")]


def test_runtime_result_envelope_matches_generator_contract(adapter, tmp_path):
    scheduler, loaded = adapter
    output = tmp_path / "benchmark.json"
    case = tmp_path / "cases.json"
    case.write_text("{}")
    point = _BenchmarkPoint(point_type="prefill", isl=192, kv_read_tokens=128, context_length=64, batch_size=2)
    fpm = {
        "dp_rank": 0,
        "wall_time": 0.004,
        "scheduled_requests": {
            "num_prefill_requests": 2,
            "sum_prefill_tokens": 128,
            "sum_prefill_kv_tokens": 256,
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        },
    }
    scheduler._fpm_result_written = False
    scheduler._bench_results = [loaded.BenchmarkPointResult(point=point, fpms=[fpm])]
    scheduler._bench_skipped_points = []
    scheduler._bench_expected_points = 1
    scheduler._bench_missing_phases = []
    scheduler._fpm_execution_meta = [
        {
            "design_index": 0,
            "point": {
                "workload_kind": "prefill",
                "batch_size": 2,
                "suffix_length": 64,
                "prefix_length": 128,
            },
            "measured": True,
        }
    ]
    scheduler._fpm_case_raw = {"plan_sha256": "plan", "cell_id": "cell"}
    scheduler._fpm_case_path = case
    scheduler._fpm_global_warmup_iterations = 5
    scheduler._fpm_repeats = 1
    scheduler._fpm_population_count = 10
    scheduler._fpm_target_count = 1
    scheduler._fpm_eligible = [object()]
    scheduler._fpm_cancelled = []
    scheduler._fpm_started_monotonic_ns = 0
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path=str(output), warmup_iterations=5)

    scheduler._bench_write_results()
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["status"] == "complete"
    assert payload["valid"] is True
    assert payload["coverage"] == {"expected_points": 1, "completed_points": 1, "skipped_points": 0}
    assert payload["collector"]["global_warmup_iterations"] == 5
    assert payload["collector"]["warmup_repeats"] == 0
    assert payload["collector"]["measurement_policy"] == "single_sample_v1"
    assert payload["campaign_results"][0]["warmup_fpms"] == []
