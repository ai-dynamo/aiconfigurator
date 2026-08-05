# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for sweep.py helpers and sweep_disagg placeholder.

Sweep output correctness is validated by the integration parity test
(``tests/integration/test_task_v1_v2_parity.py``) against the legacy CLI path;
the unit coverage here targets local control flow and terminal classification.
"""

from unittest.mock import MagicMock

import pandas as pd
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
# sweep_afd parameter forwarding
# ---------------------------------------------------------------------------


def test_sweep_afd_forwards_max_a_batch_size(monkeypatch):
    from aiconfigurator.sdk import pareto_analysis

    captured = {}
    expected = object()

    def fake_afd_pareto(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(pareto_analysis, "afd_pareto", fake_afd_pareto)

    result = sweep.sweep_afd(
        model_path="Qwen/Qwen3-32B",
        runtime_config=config.RuntimeConfig(isl=128, osl=32),
        database=object(),
        backend_name="trtllm",
        model_config=config.ModelConfig(),
        afd_parallel_config_list=[(1, 1, 2, 1, 3, "optimistic")],
        gpus_per_node=8,
        combined_with_pd=False,
        max_a_batch_size=1536,
    )

    assert result is expected
    assert captured["max_a_batch_size"] == 1536


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


def test_sweep_agg_disables_gen_dedup_for_speculative_schedules(monkeypatch):
    """The capped-gen dedup key assumes the non-speculative schedule; under
    fractional decode iterations it merges batches whose real backend
    schedules differ (e.g. vLLM runs b - ceil(ctx/isl) decode requests, which
    distinguishes b=5 from the batch its capped key collides with). With an
    active profile every guard-passing point must therefore be evaluated,
    while an inactive profile must reproduce the legacy point set exactly."""
    from aiconfigurator.sdk.speculative import SpeculativeDecodingProfile

    def _run(profile):
        points: list[tuple[int, int]] = []

        def _record(*, runtime_config, ctx_tokens, **_kwargs):
            points.append((runtime_config.batch_size, ctx_tokens))
            summary = MagicMock()
            summary.check_oom.return_value = False
            summary.check_kv_cache_oom.return_value = False
            summary.get_result_dict.return_value = {"ttft": 1.0, "tpot": 1.0}
            summary.get_per_ops_source.return_value = {}
            return summary

        monkeypatch.setattr(sweep, "predict_agg_worker", _record)
        sweep._sweep_one_parallel_agg(
            model=MagicMock(),
            backend=MagicMock(),
            database=MagicMock(),
            runtime_config=config.RuntimeConfig(isl=1024, osl=4, ttft=1e9, tpot=1e9),
            top_k=0,
            max_batch_size=64,
            ctx_stride=512,
            enable_chunked_prefill=False,
            free_gpu_memory_fraction=None,
            max_seq_len=None,
            speculative_profile=profile,
        )
        return points

    baseline = _run(None)
    inactive = _run(SpeculativeDecodingProfile(0.0))
    speculative = _run(SpeculativeDecodingProfile(1.0))  # progress = 2.0

    assert inactive == baseline
    # No dedup under speculation: the baseline's evaluated points are a strict
    # subset (dedup only ever removes points, and at osl=4 it removes some).
    assert set(baseline) < set(speculative)
    # Hand-checked point the legacy key drops: at isl=ctx_tokens=1024, b=6
    # gives balance 6/4 = 1.5 -> capped gen 6//1.5 = 4, colliding with b=5's
    # group (5//1.25 = 4), so the baseline dedups it away; with speculation
    # active it must be evaluated.
    assert (6, 1024) not in baseline
    assert (6, 1024) in speculative
    assert len(speculative) == len(set(speculative))


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


def _worker_row(**overrides) -> dict:
    """Synthetic ColumnsStatic-shaped disagg worker row shared by the EPD tests."""
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


def test_sweep_disagg_epd_composes_encoder_stage(monkeypatch):
    """EPD end-to-end semantics on synthetic candidates.

    Pins the three EPD invariants: (1) enable_epd flips the prefill workers
    to language-only (pure ttft 60 instead of the inline-encoder 100),
    (2) TTFT composes sequentially with the same queueing correction applied
    to both stages (corr x encode latency + corr x prefill ttft -- symmetric
    with the inline PD path, whose full E+P ttft is corrected), (3) the
    encode pool is sized into the rate matching and its GPUs dilute the
    per-GPU metrics.
    """
    import pandas as pd

    # Inline-encoder prefill (PD): ttft = 40ms encoder + 60ms context.
    # Language-only prefill (EPD): the same worker without the ViT.
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
        return (pure_prefill_df if model_config.language_only else colocated_prefill_df).copy()

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

    # Plain PD: the inline encoder is part of the corrected prefill ttft
    # (1.8 x (40 encoder + 60 context) = 180).
    pd_row = sweep_disagg(**common_kwargs).iloc[0]
    assert pd_row["(e)workers"] == 0
    assert pd_row["ttft"] == pytest.approx(180.0)
    assert pd_row["encoder_latency"] == pytest.approx(40.0)

    # EPD: language-only prefill ttft = 60 -> corrected 108; the encode stage
    # carries the same correction (1.8 x 50 = 90) so the PD comparison stays
    # apples-to-apples: ttft = 90 + 108 = 198.
    epd_row = sweep_disagg(**common_kwargs, enable_epd=True, encoder_tp_list=[2]).iloc[0]
    assert epd_row["ttft"] == pytest.approx(198.0)
    # encoder_latency stays the raw stage latency, like the inline PD row.
    assert epd_row["encoder_latency"] == pytest.approx(50.0)
    # Rate matching: p 16.667/w (4 gpus), d 20/w (8 gpus), e 80/w * 0.9 deg (2 gpus)
    # -> optimum (4p, 3d, 1e): seq/s 60, gpus 16+24+2=42.
    assert (epd_row["(p)workers"], epd_row["(d)workers"], epd_row["(e)workers"]) == (4, 3, 1)
    assert epd_row["num_total_gpus"] == 42
    assert epd_row["seq/s"] == pytest.approx(60.0)
    assert epd_row["tokens/s/gpu"] == pytest.approx(6000.0 / 42, abs=1e-3)
    assert (epd_row["(e)tp"], epd_row["(e)bs"]) == (2, 4)
    # request latency = corrected prefill ttft + tpot*(osl-1) + corrected encode stage.
    assert epd_row["request_latency"] == pytest.approx(108.0 + 8.0 * 99 + 90.0)


def test_sweep_agg_epd_language_only_pin_survives_config_builder(monkeypatch):
    """Task hands sweep_* a per-point ModelConfig builder, not an instance;
    the EPD language-only pin must apply to what the builder produces."""
    captured: dict = {}

    def _fake_get_model(*, model_path, model_config, backend_name):
        captured["language_only"] = model_config.language_only
        return MagicMock()

    monkeypatch.setattr(sweep, "get_model", _fake_get_model)
    monkeypatch.setattr(sweep, "get_backend", lambda name: MagicMock())
    monkeypatch.setattr(
        sweep,
        "_sweep_one_parallel_agg",
        lambda **_kwargs: (pd.DataFrame(), True, True, 0),
    )
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 80.0, "num_total_gpus": 2, "tp": 2, "bs": 4, "memory": 1.5}
        ],
    )
    with pytest.raises(NoFeasibleConfigError):
        sweep.sweep_agg(
            model_path="m",
            runtime_config=config.RuntimeConfig(isl=1000, osl=100, ttft=200.0, tpot=10.0),
            database=MagicMock(),
            backend_name="sglang",
            model_config=lambda parallel: config.ModelConfig(),
            parallel_config_list=[(4, 1, 1, 1, 1, 1)],
            enable_epd=True,
            encoder_tp_list=[2],
        )
    assert captured["language_only"] is True


def test_sweep_agg_epd_composes_encoder_stage(monkeypatch):
    """E+agg end-to-end semantics on synthetic candidates.

    Pins the E+agg invariants: (1) enable_epd flips the agg workers to
    language-only while the default keeps the encoder inline, (2) TTFT
    composes sequentially as raw encode latency + agg ttft (mirroring
    run_agg's inline convention, which adds the encoder before its queueing
    factor), (3) the cell rate match picks the best integer agg:encode
    worker ratio (encode pool sized to not bottleneck, GPUs dilute per-GPU
    metrics).
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
        captured["language_only"] = model_config.language_only
        return MagicMock()

    monkeypatch.setattr(sweep, "get_model", _fake_get_model)
    monkeypatch.setattr(sweep, "get_backend", lambda name: MagicMock())
    monkeypatch.setattr(sweep, "_sweep_one_parallel_agg", lambda **_kwargs: (pd.DataFrame([agg_row]), True, True, 0))
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
    assert captured["language_only"] is False
    assert agg_df.iloc[0]["ttft"] == pytest.approx(100.0)

    # E+agg: language-only agg workers + encode pool.
    epd_df = sweep.sweep_agg(**common_kwargs, enable_epd=True, encoder_tp_list=[2])
    assert captured["language_only"] is True
    row = epd_df.iloc[0]
    # TTFT = agg ttft + raw encode batch latency (run_agg's inline convention).
    assert row["ttft"] == pytest.approx(150.0)
    assert row["encoder_latency"] == pytest.approx(50.0)
    # Cell match: agg 10/w (4 gpus) vs encode capacity 80*0.9=72 (2 gpus)
    # -> optimum (7 agg, 1 e): seq/s 70, gpus 28+2=30 (ties keep the smaller cell).
    assert (row["(a)workers"], row["(e)workers"]) == (7, 1)
    assert row["num_total_gpus"] == 30
    assert row["seq/s"] == pytest.approx(70.0)
    assert row["concurrency"] == 32 * 7

    # Replica budget: num_gpu_list=[6] leaves exactly one feasible cell.
    row = sweep.sweep_agg(**common_kwargs, enable_epd=True, encoder_tp_list=[2], num_gpu_list=[6]).iloc[0]
    assert (row["(a)workers"], row["(e)workers"]) == (1, 1)
    assert row["num_total_gpus"] == 6


def test_sweep_disagg_epd_encoder_pool_sizing_under_replica_budget(monkeypatch):
    """E-pool sizing under a per-replica GPU budget: bottleneck and padding.

    Small pools trade capped throughput for GPUs (num_gpu_list=[3] forces
    1p1d1e with the encode pool binding at 9 seq/s); over-provisioned pools
    can be the only counts landing on the grid (num_gpu_list=[4] pads a
    never-binding encoder to e=2, throughput uncapped).
    """
    import pandas as pd

    # Language-only 1-GPU workers riding the shared row builder.
    lm = dict(
        ttft=60.0,
        request_latency=60.0,
        encoder_latency=0.0,
        encoder_memory=0.0,
        num_total_gpus=1,
        tp=1,
        parallel="tp1pp1dp1",
    )
    prefill_df = pd.DataFrame([_worker_row(**lm)])
    decode_df = pd.DataFrame(
        [_worker_row(**{**lm, "bs": 32, "global_bs": 32, "concurrency": 32, "ttft": 0.0, "tpot": 8.0})]
    )

    monkeypatch.setattr(
        sweep,
        "_get_disagg_worker_candidates",
        lambda *, role, **_kwargs: (decode_df if role == "decode" else prefill_df).copy(),
    )
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 10.0, "num_total_gpus": 1, "tp": 1, "bs": 4, "memory": 1.0}
        ],
    )

    kwargs = dict(
        model_path="m",
        runtime_config=config.RuntimeConfig(isl=1000, osl=100, ttft=200.0, tpot=10.0),
        prefill_database=MagicMock(),
        prefill_backend_name="sglang",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        prefill_latency_correction=1.0,
        decode_database=MagicMock(),
        decode_backend_name="sglang",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[(1, 1, 1, 1, 1, 1)],
        decode_latency_correction=1.0,
        rate_matching_prefill_degradation=1.0,
        rate_matching_decode_degradation=1.0,
        enable_epd=True,
        encoder_tp_list=[1],
    )
    row = sweep_disagg(**kwargs, prefill_num_worker_list=[1, 2], decode_num_worker_list=[1, 2], num_gpu_list=[3]).iloc[
        0
    ]

    assert (row["(p)workers"], row["(d)workers"], row["(e)workers"]) == (1, 1, 1)
    assert row["num_total_gpus"] == 3
    # Encode pool binds: 10 * 1 * 0.9 = 9 < min(p, d) = 10.
    assert row["seq/s"] == pytest.approx(9.0)
    assert row["tokens/s/gpu"] == pytest.approx(900.0 / 3, abs=1e-3)

    # Padding: a fast encode worker never binds at e=1, but only e=2 lands on
    # the [4] grid.
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 100.0, "num_total_gpus": 1, "tp": 1, "bs": 4, "memory": 1.0}
        ],
    )
    row = sweep_disagg(**kwargs, prefill_num_worker_list=[1], decode_num_worker_list=[1], num_gpu_list=[4]).iloc[0]
    assert (row["(p)workers"], row["(d)workers"], row["(e)workers"]) == (1, 1, 2)
    assert row["num_total_gpus"] == 4
    assert row["seq/s"] == pytest.approx(10.0)

    # max_encoder_workers=1 pins the pool: without it this encoder (capacity
    # 5.4/worker) would size to e=2 for seq/s 10; capped, the cell is E-bound.
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 6.0, "num_total_gpus": 1, "tp": 1, "bs": 4, "memory": 1.0}
        ],
    )
    row = sweep_disagg(**kwargs, prefill_num_worker_list=[1], decode_num_worker_list=[1], max_encoder_workers=1).iloc[0]
    assert (row["(p)workers"], row["(d)workers"], row["(e)workers"]) == (1, 1, 1)
    assert row["seq/s"] == pytest.approx(5.4)


def test_encoder_worker_candidates_gated_by_gpu_memory(monkeypatch):
    """_get_encoder_worker_candidates drops (tp, batch) points that exceed the
    encoder system's GPU memory."""
    from aiconfigurator.sdk import common

    enc_cfg = common.VisionEncoderConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=64,
        projector_dims=((64, 64),),
        projector_n_instances=1,
        partial_rotary_factor=0.0,
    )
    monkeypatch.setattr(sweep, "get_model_config_from_model_path", lambda _path: {"extra_params": enc_cfg})
    backend = MagicMock()
    backend.OTHERS_OVERHEAD_FRAC = 0.0
    memory_by_batch = {1: 10.0, 2: 15.0, 4: 25.0}
    backend.run_encoder_static.side_effect = lambda model, database, rc, b, latency_correction_scale=1.0: (
        50.0,
        100.0,
        {"total": memory_by_batch[b]},
        1.0,
    )
    monkeypatch.setattr(sweep, "get_backend", lambda _name: backend)

    database = MagicMock()
    database.system_spec = {
        "gpu": {"mem_capacity": 20 * (1 << 30)},
        "misc": {"nccl_mem": {1: 0}, "other_mem": 5 * (1 << 30)},
    }
    rows = sweep._get_encoder_worker_candidates(
        model_path="m",
        tp_list=[1],
        b_list=None,
        runtime_config=config.RuntimeConfig(
            isl=1000, osl=100, ttft=200.0, tpot=10.0, image_height=256, image_width=256, num_images_per_request=1
        ),
        database=database,
        backend_name="sglang",
        latency_correction=1.0,
    )
    # The gate charges the framework overhead too: batch 2 (15 + 5 other_mem
    # >= 20 GiB) stops the schedule; only batch 1 survives.
    assert [r["bs"] for r in rows] == [1]


def _encoder_candidates_env(monkeypatch, *, latency: float = 50.0):
    """Fixture env for _get_encoder_worker_candidates: real encoder ops,
    mocked perf query, nccl_mem table with keys {1, 2} only."""
    from aiconfigurator.sdk import common

    enc_cfg = common.VisionEncoderConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        patch_size=16,
        temporal_patch_size=1,
        spatial_merge_size=2,
        out_hidden_size=64,
        projector_dims=((64, 64),),
        projector_n_instances=1,
        partial_rotary_factor=0.0,
    )
    monkeypatch.setattr(sweep, "get_model_config_from_model_path", lambda _path: {"extra_params": enc_cfg})
    backend = MagicMock()
    backend.OTHERS_OVERHEAD_FRAC = 0.0
    backend.run_encoder_static.side_effect = lambda model, database, rc, b, latency_correction_scale=1.0: (
        latency,
        100.0,
        {"total": 1.0},
        1.0,
    )
    monkeypatch.setattr(sweep, "get_backend", lambda _name: backend)
    database = MagicMock()
    database.system_spec = {
        "gpu": {"mem_capacity": 20 * (1 << 30)},
        "misc": {"nccl_mem": {1: 0, 2: 0}, "other_mem": 0},
    }
    return database


_ENC_RC = config.RuntimeConfig(
    isl=1000, osl=100, ttft=200.0, tpot=10.0, image_height=256, image_width=256, num_images_per_request=1
)


def test_encoder_tp_without_comm_data_is_skipped_not_crashed(monkeypatch):
    """tp values absent from the system's nccl_mem table are skipped with a
    warning instead of raising a raw KeyError; all-skipped fails loud."""
    database = _encoder_candidates_env(monkeypatch)
    rows = sweep._get_encoder_worker_candidates(
        model_path="m",
        tp_list=[2, 3],
        b_list=[1],
        runtime_config=_ENC_RC,
        database=database,
        backend_name="sglang",
        latency_correction=1.0,
    )
    assert [r["tp"] for r in rows] == [2]
    with pytest.raises(NoFeasibleConfigError):
        sweep._get_encoder_worker_candidates(
            model_path="m",
            tp_list=[3],
            b_list=[1],
            runtime_config=_ENC_RC,
            database=database,
            backend_name="sglang",
            latency_correction=1.0,
        )


def test_encoder_zero_latency_fails_loud(monkeypatch):
    """A non-positive batch latency (corrupt perf data or zero correction)
    raises instead of ZeroDivisionError / silently skewing the sweep."""
    database = _encoder_candidates_env(monkeypatch, latency=0.0)
    with pytest.raises(ValueError, match="invalid batch latency"):
        sweep._get_encoder_worker_candidates(
            model_path="m",
            tp_list=[1],
            b_list=[1],
            runtime_config=_ENC_RC,
            database=database,
            backend_name="sglang",
            latency_correction=1.0,
        )


def test_overlay_encoder_stage_degradation_knob_and_coverage_blend():
    """The overlay honors a custom encoder degradation and re-weights
    power_coverage with the same time weights as power_w; rows without a
    coverage key must not gain one."""
    row = {
        "ttft": 100.0,
        "tpot": 10.0,
        "osl": 11,
        "request_latency": 200.0,
        "seq/s": 10.0,
        "tokens/s": 1000.0,
        "request_rate": 10.0,
        "num_total_gpus": 4,
        "power_w": 0.0,
        "power_coverage": 1.0,
    }
    enc = {
        "encoder_latency": 50.0,
        "seq/s": 8.0,
        "num_total_gpus": 1,
        "tp": 1,
        "bs": 2,
        "memory": 1.0,
        "power_w": 0.0,
        "power_coverage": 0.0,
    }
    out = sweep._overlay_encoder_stage(row, enc, 1, encoder_degradation=0.5)
    assert out["seq/s"] == pytest.approx(4.0)  # 8.0 x 0.5 x 1 caps the row's 10.0
    # decode_time = 10 x (11 - 1); coverage = (0 x 50 + 1.0 x 200) / 250.
    assert out["power_coverage"] == pytest.approx(0.8)
    # Sweep rows (no coverage channel): anything short of full encoder
    # energy data zeroes the blended power instead of understating it.
    bare = {k: v for k, v in row.items() if k != "power_coverage"}
    for partial in (0.99, float("nan")):
        zeroed = sweep._overlay_encoder_stage(bare, dict(enc, power_coverage=partial), 1, prefill_power=100.0)
        assert "power_coverage" not in zeroed
        assert zeroed["power_w"] == 0.0
    covered = sweep._overlay_encoder_stage(bare, dict(enc, power_coverage=1.0), 1, prefill_power=100.0)
    assert covered["power_w"] > 0.0


def test_sweep_agg_epd_top_k_defers_to_encoder_pairing(monkeypatch):
    """The top_k cut must happen per encode pairing, not on the language-only
    rows: with top_k=1 the higher-throughput row (no encode headroom,
    50 + 180 >= 200) would otherwise shadow the only pairable row."""
    import pandas as pd

    # Only the keys the (agg x encode) matcher and the encoder overlay read.
    def _row(ttft: float, seq_s: float) -> dict:
        return {
            "osl": 100,
            "concurrency": 32,
            "ttft": ttft,
            "tpot": 8.0,
            "request_latency": ttft + 8.0 * 99,
            "seq/s": seq_s,
            "tokens/s": seq_s * 100,
            "num_total_gpus": 4,
        }

    def _fake_sweep_one(**kwargs):
        # Honors top_k like the real sweep (seq/s-descending cut), so the
        # outcome below fails if sweep_agg ever cuts before the pairing.
        df = pd.DataFrame([_row(ttft=180.0, seq_s=50.0), _row(ttft=60.0, seq_s=10.0)])
        if kwargs["top_k"] > 0:
            df = df.head(kwargs["top_k"])
        return df, True, True, 0

    monkeypatch.setattr(sweep, "get_model", lambda **_kwargs: MagicMock())
    monkeypatch.setattr(sweep, "get_backend", lambda name: MagicMock())
    monkeypatch.setattr(sweep, "_sweep_one_parallel_agg", _fake_sweep_one)
    monkeypatch.setattr(
        sweep,
        "_get_encoder_worker_candidates",
        lambda **_kwargs: [
            {"encoder_latency": 50.0, "seq/s": 8.0, "num_total_gpus": 2, "tp": 2, "bs": 4, "memory": 1.5}
        ],
    )

    df = sweep.sweep_agg(
        model_path="m",
        runtime_config=config.RuntimeConfig(isl=1000, osl=100, ttft=200.0, tpot=10.0),
        database=MagicMock(),
        backend_name="sglang",
        model_config=config.ModelConfig(),
        parallel_config_list=[(4, 1, 1, 1, 1, 1)],
        top_k=1,
        enable_epd=True,
        encoder_tp_list=[2],
    )

    # The pairable row (ttft 60 + encode 50) wins the per-choice slot; a cut
    # before the pairing would have dropped it (fake returns rows seq/s-desc).
    assert len(df) == 1
    assert df.iloc[0]["ttft"] == pytest.approx(110.0)


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


def test_sweep_disagg_autoscale_forwards_degradation_factors(monkeypatch):
    """The Task-facing sweep path must not drop calibrated autoscale factors."""
    candidates = pd.DataFrame([{"candidate": 1}])
    captured = {}

    monkeypatch.setattr(sweep, "_get_disagg_worker_candidates", lambda **_kwargs: candidates)

    def fake_pick_autoscale(**kwargs):
        captured.update(kwargs)
        return {"best_config_df": pd.DataFrame([{"selected": True}])}

    monkeypatch.setattr("aiconfigurator.sdk.picking.pick_autoscale", fake_pick_autoscale)

    result = sweep_disagg(
        model_path="x",
        runtime_config=config.RuntimeConfig(isl=128, osl=32, ttft=100.0, tpot=10.0),
        prefill_database=object(),
        prefill_backend_name="trtllm",
        prefill_model_config=config.ModelConfig(),
        prefill_parallel_config_list=[],
        prefill_latency_correction=1.0,
        decode_database=object(),
        decode_backend_name="trtllm",
        decode_model_config=config.ModelConfig(),
        decode_parallel_config_list=[],
        decode_latency_correction=1.0,
        prefill_num_worker_list=[1],
        decode_num_worker_list=[1],
        autoscale=True,
        rate_matching_prefill_degradation=0.61,
        rate_matching_decode_degradation=0.73,
    )

    assert result.iloc[0]["selected"]
    assert captured["prefill_degradation_factor"] == 0.61
    assert captured["decode_degradation_factor"] == 0.73
