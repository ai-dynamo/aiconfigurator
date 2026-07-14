# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import deque
from dataclasses import dataclass, field
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
    def _bench_skip_point(self, point, reason):
        self._bench_skipped_points.append((point, reason))


@dataclass
class _BenchConfig:
    mode: str
    output_path: str


@pytest.fixture
def adapter(monkeypatch):
    module = types.ModuleType("dynamo.vllm.instrumented_scheduler")
    module.BenchmarkPoint = _BenchmarkPoint
    module.BenchmarkPointResult = _BenchmarkPointResult
    module.InstrumentedScheduler = _BaseScheduler
    monkeypatch.setitem(sys.modules, "dynamo", types.ModuleType("dynamo"))
    monkeypatch.setitem(sys.modules, "dynamo.vllm", types.ModuleType("dynamo.vllm"))
    monkeypatch.setitem(sys.modules, "dynamo.vllm.instrumented_scheduler", module)

    path = Path(__file__).parents[3] / "collector" / "fpm_forward" / "runtime" / "vllm_scheduler.py"
    spec = importlib.util.spec_from_file_location("test_fpm_scheduler", path)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
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
    scheduler._fpm_warmups = 0
    scheduler._fpm_repeats = 1
    scheduler._fpm_population_count = 10
    scheduler._fpm_target_count = 1
    scheduler._fpm_eligible = [object()]
    scheduler._fpm_cancelled = []
    scheduler._fpm_started_monotonic_ns = 0
    scheduler._bench_config = _BenchConfig(mode="prefill", output_path=str(output))

    scheduler._bench_write_results()
    payload = json.loads(output.read_text())
    assert payload["schema_version"] == 1
    assert payload["status"] == "complete"
    assert payload["valid"] is True
    assert payload["coverage"] == {"expected_points": 1, "completed_points": 1, "skipped_points": 0}
    assert payload["collector"]["warmup_repeats"] == 0
    assert payload["collector"]["measurement_policy"] == "single_sample_v1"
    assert payload["campaign_results"][0]["warmup_fpms"] == []
