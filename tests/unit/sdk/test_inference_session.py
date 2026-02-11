# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for DisaggInferenceSession with require_same_tp filtering.

When require_same_tp=True (SGLang non-wideep disagg), only prefill/decode
worker combinations whose tensor-parallel sizes match should survive the
rate-matching step inside find_best_disagg_result_under_constraints.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import DisaggInferenceSession
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _static_row(
    *,
    tp: int,
    pp: int = 1,
    dp: int = 1,
    moe_tp: int = 1,
    moe_ep: int = 1,
    bs: int = 1,
    mode: str = "static_ctx",
    isl: int = 4000,
    osl: int = 500,
) -> dict:
    """Return one row dict that conforms to ``common.ColumnsStatic``."""
    num_gpus = tp * pp * dp
    # Make ttft small enough (< ttft constraint / 1.8 correction) so prefill
    # candidates are not filtered out.  tpot must be < constraint.
    ttft = 50.0 / tp if mode == "static_ctx" else 0.0
    tpot = 5.0 / tp if mode == "static_gen" else 0.0
    seq_s = bs * 10.0
    return {
        "model": "test-model",
        "isl": isl,
        "osl": osl,
        "prefix": 0,
        "concurrency": bs,
        "request_rate": seq_s,
        "bs": bs,
        "global_bs": bs * dp,
        "ttft": ttft,
        "tpot": tpot,
        "seq/s": seq_s,
        "seq/s/gpu": seq_s / num_gpus,
        "tokens/s": seq_s * osl,
        "tokens/s/gpu": seq_s * osl / num_gpus,
        "tokens/s/user": osl / max(tpot, 1e-9),
        "request_latency": ttft + tpot * max(osl - 1, 0),
        "context_latency": ttft,
        "generation_latency": tpot * max(osl - 1, 0),
        "num_total_gpus": num_gpus,
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "moe_tp": moe_tp,
        "moe_ep": moe_ep,
        "parallel": f"tp{tp}_pp{pp}_dp{dp}",
        "gemm": "fp16",
        "kvcache": "fp16",
        "fmha": "fp16",
        "moe": "none",
        "comm": "half",
        "memory": 10.0,
        "backend": "sglang",
        "version": "v1",
        "system": "test_system",
        "power_w": 300.0,
    }


def _make_summary(row: dict, runtime_config: RuntimeConfig) -> InferenceSummary:
    """Wrap a single row dict into a non-OOM InferenceSummary."""
    s = InferenceSummary(runtime_config=runtime_config)
    s.set_oom(False)
    s.set_summary_df(pd.DataFrame([row], columns=common.ColumnsStatic))
    return s


def _build_mock_backend():
    """
    Return a mock backend whose ``run_static`` produces a deterministic
    InferenceSummary.  The ``tp`` column value comes from the model's
    ``tp_size`` (which ``_get_summary_df`` sets from the parallel config).
    """
    backend = MagicMock()
    backend.name = SimpleNamespace(value="sglang")

    def _run_static(model, database, runtime_config, mode, stride=32, latency_correction_scale=1.0):
        tp = model._tp
        pp = model._pp
        dp = model._dp
        row = _static_row(
            tp=tp,
            pp=pp,
            dp=dp,
            bs=runtime_config.batch_size,
            mode=mode,
            isl=runtime_config.isl,
            osl=runtime_config.osl,
        )
        return _make_summary(row, runtime_config)

    backend.run_static = _run_static
    return backend


@pytest.fixture(autouse=True)
def _patch_get_model(monkeypatch):
    """Replace ``models.get_model`` so no real model files are needed."""

    def _fake_get_model(model_path, model_config, backend_name):
        m = MagicMock()
        m._tp = model_config.tp_size
        m._pp = model_config.pp_size
        m._dp = model_config.attention_dp_size
        return m

    monkeypatch.setattr(
        "aiconfigurator.sdk.inference_session.models.get_model",
        _fake_get_model,
    )


@pytest.fixture
def runtime_config():
    return RuntimeConfig(isl=4000, osl=500, ttft=2000.0, tpot=30.0)


@pytest.fixture
def model_config():
    return ModelConfig()


@pytest.fixture
def disagg_session():
    """DisaggInferenceSession backed by mock backends / databases."""
    return DisaggInferenceSession(
        prefill_database=MagicMock(),
        prefill_backend=_build_mock_backend(),
        decode_database=MagicMock(),
        decode_backend=_build_mock_backend(),
    )


def _run(
    sess: DisaggInferenceSession,
    runtime_config: RuntimeConfig,
    model_config: ModelConfig,
    prefill_cfgs: list[tuple[int, int, int, int, int]],
    decode_cfgs: list[tuple[int, int, int, int, int]],
    require_same_tp: bool,
) -> InferenceSummary | None:
    return sess.find_best_disagg_result_under_constraints(
        model_path="test-model",
        runtime_config=runtime_config,
        prefill_model_config=model_config,
        prefill_parallel_config_list=prefill_cfgs,
        prefill_max_num_tokens=4000,
        prefill_num_worker_list=[1, 2, 4],
        decode_model_config=model_config,
        decode_parallel_config_list=decode_cfgs,
        decode_max_num_tokens=512,
        decode_num_worker_list=[1, 2, 4],
        num_gpu_list=None,
        require_same_tp=require_same_tp,
    )


class TestRequireSameTPFiltering:
    """Verify the TP-matching filter inside find_best_disagg_result_under_constraints."""

    def test_true_filters_mismatched_tp(self, disagg_session, runtime_config, model_config):
        """require_same_tp=True → every result row has (p)tp == (d)tp."""
        # prefill tp=2 only; decode tp=2 and tp=4
        result = _run(
            disagg_session,
            runtime_config,
            model_config,
            prefill_cfgs=[(2, 1, 1, 1, 1)],
            decode_cfgs=[(2, 1, 1, 1, 1), (4, 1, 1, 1, 1)],
            require_same_tp=True,
        )
        assert result is not None
        df = result.get_summary_df()
        if df is not None and not df.empty:
            mismatched = df[df["(p)tp"] != df["(d)tp"]]
            assert mismatched.empty, (
                f"require_same_tp=True but found mismatched rows:\n{mismatched[['(p)tp', '(d)tp']]}"
            )

    def test_false_allows_mismatched_tp(self, disagg_session, runtime_config, model_config):
        """require_same_tp=False → results are non-empty (mismatched TP is fine)."""
        result = _run(
            disagg_session,
            runtime_config,
            model_config,
            prefill_cfgs=[(2, 1, 1, 1, 1)],
            decode_cfgs=[(2, 1, 1, 1, 1), (4, 1, 1, 1, 1)],
            require_same_tp=False,
        )
        assert result is not None
        df = result.get_summary_df()
        assert df is not None and not df.empty, "Expected non-empty results with require_same_tp=False"

    def test_true_no_overlapping_tp_returns_empty(self, disagg_session, runtime_config, model_config):
        """require_same_tp=True with zero TP overlap → empty result."""
        # prefill tp=2, decode tp=4 - no common TP
        result = _run(
            disagg_session,
            runtime_config,
            model_config,
            prefill_cfgs=[(2, 1, 1, 1, 1)],
            decode_cfgs=[(4, 1, 1, 1, 1)],
            require_same_tp=True,
        )
        assert result is not None
        df = result.get_summary_df()
        assert df is None or df.empty, "Expected empty result when require_same_tp=True and no TP values overlap"

    def test_true_multiple_overlapping_tps(self, disagg_session, runtime_config, model_config):
        """require_same_tp=True with several common TPs → all surviving rows match."""
        # Both sides offer tp=1 and tp=2
        result = _run(
            disagg_session,
            runtime_config,
            model_config,
            prefill_cfgs=[(1, 1, 1, 1, 1), (2, 1, 1, 1, 1)],
            decode_cfgs=[(1, 1, 1, 1, 1), (2, 1, 1, 1, 1)],
            require_same_tp=True,
        )
        assert result is not None
        df = result.get_summary_df()
        assert df is not None and not df.empty
        for _, row in df.iterrows():
            assert row["(p)tp"] == row["(d)tp"], f"Mismatch: (p)tp={row['(p)tp']}, (d)tp={row['(d)tp']}"
