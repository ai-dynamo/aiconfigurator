# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sweep.py helpers and sweep_disagg placeholder.

Sweep output correctness is validated by the integration parity test
(``tests/integration/test_task_v1_v2_parity.py``) against the legacy CLI path;
the unit coverage here targets local control flow and terminal classification.
"""

from unittest.mock import MagicMock

import pytest

from aiconfigurator.sdk import config, sweep
from aiconfigurator.sdk.errors import (
    InsufficientMemoryError,
    KVCacheCapacityError,
    NoFeasibleConfigError,
)
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
# sweep_agg no-result classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("memory_states", "expected_error"),
    [
        ([(True, False), (True, False)], InsufficientMemoryError),
        ([(False, True), (True, False)], KVCacheCapacityError),
        ([(False, False), (True, False)], NoFeasibleConfigError),
    ],
)
def test_sweep_agg_classifies_no_result_outcomes(monkeypatch, memory_states, expected_error):
    summaries = []
    for model_oom, kv_cache_oom in memory_states:
        summary = MagicMock()
        summary.check_oom.return_value = model_oom
        summary.check_kv_cache_oom.return_value = kv_cache_oom
        summary.get_result_dict.return_value = {"ttft": 2.0, "tpot": 2.0}
        summaries.append(summary)

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", MagicMock(side_effect=summaries))

    with pytest.raises(expected_error):
        sweep.sweep_agg(
            model_path="test-model",
            runtime_config=config.RuntimeConfig(isl=1024, osl=1, ttft=1.0, tpot=1.0),
            database=MagicMock(),
            backend_name="trtllm",
            model_config=config.ModelConfig(),
            parallel_config_list=[(1, 1, 1, 1, 1, 1), (2, 1, 1, 2, 1, 1)],
            max_batch_size=1,
            ctx_stride=1024,
        )


def test_sweep_agg_point_config_preserves_multimodal_fields(monkeypatch):
    """Regression for NVBug 6401839: the agg per-batch RuntimeConfig must carry
    every multimodal field from the base runtime_config. The old field-by-field
    construction dropped image_height/width, num_images_per_request, and
    num_image_tokens, zeroing the image encoder workload in agg while disagg
    (which deep-copies) stayed correct."""
    captured: list[config.RuntimeConfig] = []

    def _record(*, runtime_config, **_kwargs):
        captured.append(runtime_config)
        summary = MagicMock()
        summary.check_oom.return_value = False
        summary.check_kv_cache_oom.return_value = False
        summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0}
        summary.get_per_ops_source.return_value = {}
        return summary

    monkeypatch.setattr(sweep, "get_backend", lambda _backend_name: MagicMock())
    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "predict_agg_worker", _record)

    base_rt = config.RuntimeConfig(
        isl=256,
        osl=256,
        ttft=1e9,
        tpot=1e9,
        image_height=1024,
        image_width=1024,
        num_images_per_request=2,
        num_image_tokens=333,
        seq_imbalance_correction_scale=1.5,
        engine_step_backend="rust",
    )

    sweep.sweep_agg(
        model_path="test-model",
        runtime_config=base_rt,
        database=MagicMock(),
        backend_name="trtllm",
        model_config=config.ModelConfig(),
        parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        max_batch_size=1,
        ctx_stride=1024,
    )

    assert captured, "expected at least one agg point to be evaluated"
    for point_rt in captured:
        assert point_rt.image_height == 1024
        assert point_rt.image_width == 1024
        assert point_rt.num_images_per_request == 2
        assert point_rt.num_image_tokens == 333
        # Non-multimodal fields must survive too (the deep-copy carries them all).
        assert point_rt.seq_imbalance_correction_scale == 1.5
        assert point_rt.engine_step_backend == "rust"
        assert point_rt.batch_size == 1


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


def test_sweep_disagg_epd_composes_encoder_stage(monkeypatch):
    """EPD end-to-end semantics on synthetic candidates.

    Pins the three EPD invariants: (1) enable_epd flips the prefill workers
    to language-only (encoder_colocated=False -> pure ttft 60 instead of the
    colocated 100), (2) TTFT composes sequentially as encode latency +
    corrected prefill ttft, (3) the encode pool is sized into the rate
    matching and its GPUs dilute the per-GPU metrics.
    """
    import pandas as pd

    def _worker_row(**overrides) -> dict:
        base = {
            "model": "m",
            "isl": 1000,
            "osl": 100,
            "prefix": 0,
            "concurrency": 1,
            "request_rate": 0.0,
            "bs": 1,
            "global_bs": 1,
            "ttft": 100.0,
            "tpot": 0.0,
            "seq/s": 10.0,
            "seq/s/gpu": 2.5,
            "tokens/s": 10.0,
            "tokens/s/gpu": 2.5,
            "tokens/s/user": 0.0,
            "request_latency": 100.0,
            "encoder_latency": 40.0,
            "encoder_memory": 1.5,
            "num_total_gpus": 4,
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "moe_tp": 1,
            "moe_ep": 1,
            "cp": 1,
            "parallel": "tp4pp1dp1",
            "gemm": "fp8",
            "kvcache": "fp8",
            "fmha": "fp8",
            "moe": "fp8",
            "comm": "half",
            "memory": 30.0,
            "backend": "sglang",
            "version": "0.5.10",
            "system": "h200_sxm",
            "power_w": 500.0,
        }
        base.update(overrides)
        return base

    # Colocated prefill (PD): ttft = 40ms encoder + 60ms context.  Language-only
    # prefill (EPD, encoder_colocated=False): the same worker without the ViT.
    colocated_prefill_df = pd.DataFrame([_worker_row()])
    pure_prefill_df = pd.DataFrame(
        [
            _worker_row(
                ttft=60.0,
                request_latency=60.0,
                encoder_latency=0.0,
                encoder_memory=0.0,
                **{"seq/s": 1000.0 / 60, "seq/s/gpu": 250.0 / 60, "tokens/s": 1000.0 / 60, "tokens/s/gpu": 250.0 / 60},
            )
        ]
    )
    decode_df = pd.DataFrame(
        [
            _worker_row(
                bs=32,
                global_bs=32,
                concurrency=32,
                ttft=0.0,
                tpot=8.0,
                **{"seq/s": 20.0, "seq/s/gpu": 2.5, "tokens/s/user": 125.0},
                encoder_latency=0.0,
                num_total_gpus=8,
                tp=8,
                parallel="tp8pp1dp1",
            )
        ]
    )

    def _fake_candidates(*, role, model_config, **_kwargs):
        if role == "decode":
            return decode_df.copy()
        return (colocated_prefill_df if model_config.encoder_colocated else pure_prefill_df).copy()

    monkeypatch.setattr(sweep, "_get_disagg_worker_candidates", _fake_candidates)
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 80.0, "num_total_gpus": 2, "tp": 2, "bs": 4, "memory": 1.5}
        ],
    )

    common_kwargs = dict(
        model_path="m",
        runtime_config=config.RuntimeConfig(isl=1000, osl=100, ttft=200.0, tpot=10.0),
        prefill_database=MagicMock(),
        prefill_backend_name="sglang",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(4, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1.0,
        decode_database=MagicMock(),
        decode_backend_name="sglang",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(8, 1, 1, 1, 1, 1)],
        decode_latency_correction=1.0,
        prefill_num_worker_list=[1, 2, 3, 4],
        decode_num_worker_list=[1, 2, 3, 4],
        rate_matching_prefill_degradation=1.0,
        rate_matching_decode_degradation=1.0,
    )

    # Plain PD: encoder stays colocated (ttft 100 * 1.8 correction).
    pd_row = sweep_disagg(**common_kwargs).iloc[0]
    assert pd_row["(e)workers"] == 0
    assert pd_row["ttft"] == pytest.approx(180.0)
    assert pd_row["encoder_latency"] == pytest.approx(40.0)

    # EPD: language-only prefill ttft = 60 -> corrected 108; + encode batch 50 = 158.
    epd_row = sweep_disagg(**common_kwargs, enable_epd=True, encoder_tp_list=[2]).iloc[0]
    assert epd_row["ttft"] == pytest.approx(158.0)
    assert epd_row["encoder_latency"] == pytest.approx(50.0)
    # Rate matching: p 16.667/w (4 gpus), d 20/w (8 gpus), e 80/w * 0.9 deg (2 gpus)
    # -> optimum (4p, 3d, 1e): seq/s 60, gpus 16+24+2=42.
    assert (epd_row["(p)workers"], epd_row["(d)workers"], epd_row["(e)workers"]) == (4, 3, 1)
    assert epd_row["num_total_gpus"] == 42
    assert epd_row["seq/s"] == pytest.approx(60.0)
    assert epd_row["tokens/s/gpu"] == pytest.approx(6000.0 / 42, abs=1e-3)
    assert (epd_row["(e)tp"], epd_row["(e)bs"], epd_row["(e)parallel"]) == (2, 4, "tp2")
    # request latency = corrected prefill ttft + tpot*(osl-1) + encode latency.
    assert epd_row["request_latency"] == pytest.approx(108.0 + 8.0 * 99 + 50.0)


def test_sweep_agg_epd_composes_encoder_stage(monkeypatch):
    """E+agg end-to-end semantics on synthetic candidates.

    Pins the E+agg invariants: (1) enable_epd flips the agg workers to
    language-only (encoder_colocated=False) while the default keeps the
    encoder inline, (2) TTFT composes sequentially as encode latency + agg
    ttft, (3) the cell rate match picks the best integer agg:encode worker
    ratio (encode pool sized to not bottleneck, GPUs dilute per-GPU metrics).
    """
    import pandas as pd

    agg_row = {
        "model": "m",
        "isl": 1000,
        "osl": 100,
        "prefix": 0,
        "concurrency": 32,
        "request_rate": 10.0,
        "bs": 32,
        "global_bs": 32,
        "ttft": 100.0,
        "tpot": 8.0,
        "request_latency": 100.0 + 8.0 * 99,
        "encoder_latency": 0.0,
        "encoder_memory": 0.0,
        "seq/s": 10.0,
        "seq/s/gpu": 2.5,
        "tokens/s": 1000.0,
        "tokens/s/gpu": 250.0,
        "tokens/s/user": 125.0,
        "num_total_gpus": 4,
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "parallel": "tp4pp1dp1",
        "gemm": "fp8",
        "kvcache": "fp8",
        "fmha": "fp8",
        "moe": "fp8",
        "comm": "half",
        "memory": 30.0,
        "backend": "sglang",
        "version": "0.5.10",
        "system": "h200_sxm",
        "power_w": 500.0,
        "balance_score": 1.0,
        "num_ctx_reqs": 1,
        "num_gen_reqs": 31,
        "num_tokens": 1000,
        "ctx_tokens": 1000,
        "gen_tokens": 31,
    }
    captured: dict = {}

    def _fake_get_model(*, model_path, model_config, backend_name):
        captured["encoder_colocated"] = model_config.encoder_colocated
        return MagicMock()

    monkeypatch.setattr(sweep, "get_model", _fake_get_model)
    monkeypatch.setattr(sweep, "get_backend", lambda name: MagicMock())
    monkeypatch.setattr(
        sweep, "_sweep_one_parallel_agg", lambda **_kwargs: (pd.DataFrame([agg_row]), True, True)
    )
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 80.0, "num_total_gpus": 2, "tp": 2, "bs": 4, "memory": 1.5}
        ],
    )

    common_kwargs = dict(
        model_path="m",
        runtime_config=config.RuntimeConfig(isl=1000, osl=100, ttft=200.0, tpot=10.0),
        database=MagicMock(),
        backend_name="sglang",
        model_config=config.ModelConfig(),
        parallel_config_list=[(4, 1, 1, 1, 1, 1)],
    )

    # Default: encoder stays inline (colocated); rows pass through untouched.
    agg_df = sweep.sweep_agg(**common_kwargs)
    assert captured["encoder_colocated"] is True
    assert "(e)workers" not in agg_df.columns
    assert agg_df.iloc[0]["ttft"] == pytest.approx(100.0)

    # E+agg: language-only agg workers + encode pool.
    epd_df = sweep.sweep_agg(**common_kwargs, enable_epd=True, encoder_tp_list=[2])
    assert captured["encoder_colocated"] is False
    row = epd_df.iloc[0]
    # TTFT = agg ttft + encode batch latency; request latency follows.
    assert row["ttft"] == pytest.approx(150.0)
    assert row["encoder_latency"] == pytest.approx(50.0)
    assert row["request_latency"] == pytest.approx(100.0 + 8.0 * 99 + 50.0)
    # Cell match: agg 10/w (4 gpus) vs encode capacity 80*0.9=72 (2 gpus)
    # -> optimum (7 agg, 1 e): seq/s 70, gpus 28+2=30 (ties keep the smaller cell).
    assert (row["(a)workers"], row["(e)workers"]) == (7, 1)
    assert row["num_total_gpus"] == 30
    assert row["seq/s"] == pytest.approx(70.0)
    assert row["tokens/s/gpu"] == pytest.approx(7000.0 / 30, abs=1e-3)
    assert row["concurrency"] == 32 * 7
    assert (row["(e)tp"], row["(e)bs"], row["(e)parallel"]) == (2, 4, "tp2")


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
