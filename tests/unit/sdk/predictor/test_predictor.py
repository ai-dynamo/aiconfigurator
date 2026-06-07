# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sdk/predictor.py.

AnalyticPredictor must wrap backend.run_agg / backend.run_static exactly
(bit-identical to direct calls), and the Predictor Protocol must accept
AnalyticPredictor as a valid implementation.
"""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk.predictor import (
    DEFAULT_PREDICTOR,
    AnalyticPredictor,
    Predictor,
)

pytestmark = pytest.mark.unit


def _make_mocks(return_value: str = "sentinel-summary"):
    model = MagicMock(name="model")
    backend = MagicMock(name="backend")
    backend.run_agg.return_value = return_value
    backend.run_static.return_value = return_value
    database = MagicMock(name="database")
    runtime_config = MagicMock(name="runtime_config")
    return model, backend, database, runtime_config


def test_default_predictor_is_analytic():
    assert isinstance(DEFAULT_PREDICTOR, AnalyticPredictor)


def test_analytic_predictor_satisfies_protocol():
    assert isinstance(AnalyticPredictor(), Predictor)


def test_analytic_predictor_predict_agg_worker_wraps_run_agg():
    model, backend, database, rt = _make_mocks()

    result = AnalyticPredictor().predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=4096,
    )

    assert result == "sentinel-summary"
    backend.run_agg.assert_called_once_with(model, database, rt, ctx_tokens=4096)
    backend.run_static.assert_not_called()


def test_analytic_predictor_predict_agg_forwards_backend_kwargs():
    model, backend, database, rt = _make_mocks()

    AnalyticPredictor().predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=2048,
        max_seq_len=8000,
        free_gpu_memory_fraction=0.85,
    )

    backend.run_agg.assert_called_once_with(
        model,
        database,
        rt,
        ctx_tokens=2048,
        max_seq_len=8000,
        free_gpu_memory_fraction=0.85,
    )


def test_analytic_predictor_predict_disagg_worker_prefill_uses_static_ctx():
    model, backend, database, rt = _make_mocks()

    AnalyticPredictor().predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="prefill",
    )

    backend.run_static.assert_called_once_with(model, database, rt, "static_ctx", 32, 1.0)
    backend.run_agg.assert_not_called()


def test_analytic_predictor_predict_disagg_worker_decode_uses_static_gen():
    model, backend, database, rt = _make_mocks()

    AnalyticPredictor().predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="decode",
        latency_correction=1.25,
        stride=64,
    )

    backend.run_static.assert_called_once_with(model, database, rt, "static_gen", 64, 1.25)


def test_predict_functions_default_to_analytic_predictor():
    """When no predictor is passed, predict_* uses DEFAULT_PREDICTOR -- which
    means backend.run_agg / run_static is called directly (same as before
    the Predictor abstraction was introduced)."""
    from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker

    model, backend, database, rt = _make_mocks()

    predict_agg_worker(model=model, backend=backend, database=database, runtime_config=rt, ctx_tokens=1024)
    backend.run_agg.assert_called_once()

    predict_disagg_worker(model=model, backend=backend, database=database, runtime_config=rt, role="prefill")
    backend.run_static.assert_called_once()


def test_predict_functions_route_through_explicit_predictor():
    """When a custom predictor is passed, predict_* delegates to it (not the default)."""
    from aiconfigurator.sdk.predict import predict_agg_worker, predict_disagg_worker

    model, backend, database, rt = _make_mocks()
    custom = MagicMock(spec=Predictor, name="custom_predictor")
    custom.predict_agg_worker.return_value = "custom-agg-result"
    custom.predict_disagg_worker.return_value = "custom-phase-result"

    result_agg = predict_agg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        ctx_tokens=512,
        predictor=custom,
    )
    assert result_agg == "custom-agg-result"
    custom.predict_agg_worker.assert_called_once()
    backend.run_agg.assert_not_called()  # default predictor bypassed

    result_phase = predict_disagg_worker(
        model=model,
        backend=backend,
        database=database,
        runtime_config=rt,
        role="decode",
        predictor=custom,
    )
    assert result_phase == "custom-phase-result"
    custom.predict_disagg_worker.assert_called_once()
    backend.run_static.assert_not_called()
