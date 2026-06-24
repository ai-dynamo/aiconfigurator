# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sweep.py helpers and sweep_disagg placeholder.

sweep_agg end-to-end correctness is validated by the integration
parity test (tests/integration/test_task_v1_v2_parity.py) against the
legacy CLI path; mocking it at unit level provides little signal.
"""

import pytest

from aiconfigurator.sdk.sweep import (
    _DEFAULT_AGG_BATCH_SCHEDULE,
    _agg_ctx_tokens_list,
    sweep_disagg,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _agg_ctx_tokens_list — parity with legacy base_backend._get_ctx_tokens_list_for_agg_sweep
# ---------------------------------------------------------------------------


def _legacy_ctx_tokens_list(isl: int, ctx_stride: int, enable_chunked_prefill: bool) -> list[int]:
    """Wrap the legacy helper on BaseBackend for parity comparison."""
    from aiconfigurator.sdk.backends.factory import get_backend

    legacy = get_backend("trtllm")  # any backend exposes the helper, it's on BaseBackend
    return legacy._get_ctx_tokens_list_for_agg_sweep(
        isl=isl,
        ctx_stride=ctx_stride,
        enable_chunked_prefill=enable_chunked_prefill,
    )


@pytest.mark.parametrize("isl", [1024, 2048, 4000, 8000, 16384])
@pytest.mark.parametrize("ctx_stride", [128, 256, 512, 1024])
@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
def test_agg_ctx_tokens_list_matches_legacy(isl, ctx_stride, enable_chunked_prefill):
    new = _agg_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)
    old = _legacy_ctx_tokens_list(isl, ctx_stride, enable_chunked_prefill)
    assert new == old, (
        f"Mismatch for isl={isl}, ctx_stride={ctx_stride}, "
        f"enable_chunked_prefill={enable_chunked_prefill}\nnew={new}\nold={old}"
    )


# ---------------------------------------------------------------------------
# Batch schedule shape
# ---------------------------------------------------------------------------


def test_default_agg_batch_schedule_is_monotonic_and_capped():
    assert sorted(_DEFAULT_AGG_BATCH_SCHEDULE) == _DEFAULT_AGG_BATCH_SCHEDULE
    assert _DEFAULT_AGG_BATCH_SCHEDULE[0] == 1
    assert _DEFAULT_AGG_BATCH_SCHEDULE[-1] == 1024


# ---------------------------------------------------------------------------
# sweep_disagg validation
# ---------------------------------------------------------------------------


def test_sweep_disagg_rejects_invalid_max_prefill_gpus():
    with pytest.raises(ValueError, match="max_prefill_gpus must be > 0"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            max_prefill_gpus=0,
        )


def test_sweep_disagg_rejects_invalid_max_decode_gpus():
    with pytest.raises(ValueError, match="max_decode_gpus must be > 0"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            max_decode_gpus=-5,
        )


def test_sweep_disagg_rejects_empty_num_worker_lists():
    """Empty worker lists silently skipped the rate-match inner loop in earlier
    versions; now fail loud to avoid surprising zero-result sweeps."""
    with pytest.raises(ValueError, match="non-empty prefill_num_worker_list and decode_num_worker_list"):
        sweep_disagg(
            model_path="x",
            runtime_config=None,
            prefill_database=None,
            prefill_backend_name="trtllm",
            prefill_model_config=None,
            prefill_parallel_config_list=[],
            prefill_latency_correction=1.0,
            decode_database=None,
            decode_backend_name="trtllm",
            decode_model_config=None,
            decode_parallel_config_list=[],
            decode_latency_correction=1.0,
            prefill_num_worker_list=[],
            decode_num_worker_list=[1, 2, 4],
        )
