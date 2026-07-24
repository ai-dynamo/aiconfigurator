# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for predict_disagg_worker and predict_agg_worker.

These verify the wrappers forward arguments correctly and return whatever
the backend returns.  Real-database parity is covered by the integration
test against the old CLI (tests/integration/test_task_v1_v2_parity.py).
"""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker
from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile

pytestmark = pytest.mark.unit


def _make_mocks(return_value: str = "sentinel-summary"):
    model = MagicMock(name="model")
    backend = MagicMock(name="backend")
    backend.run_static.return_value = return_value
    backend.run_agg.return_value = return_value
    database = MagicMock(name="database")
    runtime_config = MagicMock(spec=RuntimeConfig)
    return model, backend, database, runtime_config


def test_predict_disagg_worker_prefill_calls_run_static_ctx():
    model, backend, database, rt = _make_mocks()

    result = predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="prefill",
    )

    assert result == "sentinel-summary"
    backend.run_static.assert_called_once_with(model, database, rt, "static_ctx", 32, 1.0)
    backend.run_agg.assert_not_called()


def test_predict_disagg_worker_decode_calls_run_static_gen():
    model, backend, database, rt = _make_mocks()

    predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="decode",
    )

    backend.run_static.assert_called_once_with(model, database, rt, "static_gen", 32, 1.0)


def test_predict_disagg_worker_passes_latency_correction_and_stride():
    model, backend, database, rt = _make_mocks()

    predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="prefill",
        latency_correction=1.25,
        stride=64,
    )

    backend.run_static.assert_called_once_with(model, database, rt, "static_ctx", 64, 1.25)


def test_predict_disagg_worker_rejects_unknown_role():
    model, backend, database, rt = _make_mocks()

    with pytest.raises(KeyError):
        predict_disagg_worker(
            model=model,
            backend=backend,
            database=database,
            runtime_config=rt,
            role="mixed",  # type: ignore[arg-type]
        )


def test_predict_agg_worker_calls_run_agg_with_ctx_tokens():
    model, backend, database, rt = _make_mocks()

    result = predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=4096,
    )

    assert result == "sentinel-summary"
    backend.run_agg.assert_called_once_with(model, database, rt, ctx_tokens=4096)
    backend.run_static.assert_not_called()


def test_predict_agg_worker_forwards_extra_kwargs():
    model, backend, database, rt = _make_mocks()

    predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=2048,
        enable_chunked_prefill=True,
        some_backend_specific_flag="hello",
    )

    backend.run_agg.assert_called_once_with(
        model,
        database,
        rt,
        ctx_tokens=2048,
        enable_chunked_prefill=True,
        some_backend_specific_flag="hello",
    )


def test_predict_agg_worker_passes_speculative_progress_to_scheduler():
    model, backend, database, rt = _make_mocks(return_value="raw-summary")
    profile = MagicMock(spec=SpeculativeDecodingProfile)
    profile.expected_accepted_tokens = 1.0
    profile.tokens_per_iteration = 2.0
    profile.project_summary.return_value = "projected-summary"

    result = predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=2048,
        speculative_profile=profile,
    )

    backend.run_agg.assert_called_once_with(
        model,
        database,
        rt,
        ctx_tokens=2048,
        decode_tokens_per_iteration=2.0,
    )
    profile.project_summary.assert_called_once_with("raw-summary", role="agg")
    assert result == "projected-summary"
