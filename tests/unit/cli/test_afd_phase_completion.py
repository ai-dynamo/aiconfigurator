# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for completing single-phase AFD estimates with regular static phases."""

import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

from aiconfigurator.cli import api
from aiconfigurator.cli.api import EstimateResult, _combine_afd_static_estimate_results
from aiconfigurator.sdk.config import AFDConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession
from aiconfigurator.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


def _fake_phase_metrics(
    *,
    t_a_layer: float,
    t_f_layer: float,
    balance_ratio: float,
    t_a2f_layer: float = 0.1,
    t_f2a_layer: float = 0.1,
    t_step: float = 50.0,
    comm_hidden: bool = True,
) -> dict:
    """Build a minimal ``_simulate_phase``-style metrics dict for AFD tests.

    Only the per-phase layer scalars vary across cases; memory/per-op shape
    stays trivial so ``_build_summary`` can run without further plumbing.
    """
    return {
        "t_a_layer": t_a_layer,
        "t_f_layer": t_f_layer,
        "t_a2f_layer": t_a2f_layer,
        "t_f2a_layer": t_f2a_layer,
        "t_c_layer": t_a2f_layer + t_f2a_layer,
        "t_step": t_step,
        "comm_hidden": comm_hidden,
        "balance_ratio": balance_ratio,
        "a_per_op": {},
        "f_per_op": {},
        "a_memory": {"total": 1.0, "weights": 1.0, "activations": 0.0, "kvcache": 0.0, "nccl": 0.0, "others": 0.0},
        "f_memory": {"total": 1.0, "weights": 1.0, "activations": 0.0, "kvcache": 0.0, "nccl": 0.0, "others": 0.0},
        "a_is_oom": False,
        "f_is_oom": False,
        "a_is_kv_cache_oom": False,
        "f_is_kv_cache_oom": False,
        "num_layers": 4,
    }


def _build_afd_session_with_phase_metrics(
    monkeypatch, *, prefill_metrics, decode_metrics, combined_with_pd: bool = False
) -> AFDInferenceSession:
    """Wire ``AFDInferenceSession`` so ``_simulate_phase`` returns the
    caller-supplied prefill / decode metrics dicts.

    Lets tests inject *different* per-phase scalars (e.g. distinct
    ``t_a_layer`` between prefill and decode) and inspect how
    ``_build_summary`` lays them out in ``result_dict``.
    """

    def fake_simulate_phase(self, *, phase, **_kwargs):
        return dict(prefill_metrics if phase == "prefill" else decode_metrics)

    monkeypatch.setattr(
        AFDInferenceSession,
        "_build_models",
        lambda self: (SimpleNamespace(_num_layers=4), SimpleNamespace(_num_layers=4)),
    )
    monkeypatch.setattr(AFDInferenceSession, "_simulate_phase", fake_simulate_phase)

    class FakeDatabase:
        version = "test-version"
        system = "test-system"
        system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": 80 * (1 << 30)}}

    afd_config = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=2,
        a_batch_size=4,
        num_microbatches=3,
        f_moe_ep_size=1,
        combined_with_pd=combined_with_pd,
    )
    return AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(),
        f_model_config=SimpleNamespace(),
        database=FakeDatabase(),
        backend=SimpleNamespace(
            name=SimpleNamespace(value="test-backend"),
            get_default_free_gpu_memory_fraction=lambda: 0.9,
        ),
        afd_config=afd_config,
    )


def _estimate_result(*, raw: dict, mode: str = "afd", summary=None) -> EstimateResult:
    return EstimateResult(
        ttft=float(raw.get("ttft", 0.0) or 0.0),
        tpot=float(raw.get("tpot", 0.0) or 0.0),
        power_w=float(raw.get("power_w", 0.0) or 0.0),
        isl=int(raw.get("isl", 1024) or 1024),
        osl=int(raw.get("osl", 100) or 100),
        batch_size=int(raw.get("bs", raw.get("b_total", 1)) or 1),
        ctx_tokens=0,
        tp_size=1,
        pp_size=1,
        model_path="test-model",
        system_name="test-system",
        backend_name="test-backend",
        backend_version="test-version",
        raw=raw,
        mode=mode,
        summary=summary,
    )


def test_decode_afd_uses_regular_prefill_metrics():
    afd_decode = _estimate_result(
        raw={
            "phase": "decode",
            "osl": 100,
            "ttft": 0.0,
            "tpot": 10.0,
            "request_latency": 990.0,
            "b_total": 64,
            "concurrency": 64,
            "seq/s": 7.0,
            "tokens/s": 700.0,
            "tokens/s/user": 100.0,
            "num_total_gpus": 12,
            "memory": 40.0,
            "power_w": 0.0,
        }
    )
    regular_prefill = _estimate_result(
        mode="static_ctx",
        raw={
            "osl": 100,
            "ttft": 123.0,
            "seq/s": 5.0,
            "num_total_gpus": 4,
            "memory": 30.0,
            "power_w": 300.0,
        },
    )

    combined = _combine_afd_static_estimate_results(
        afd_result=afd_decode,
        static_result=regular_prefill,
        afd_phase="decode",
    )

    assert combined.ttft == 123.0
    assert combined.tpot == 10.0
    assert combined.request_latency == 1113.0
    assert combined.seq_per_second == 5.0
    assert combined.tokens_per_second == 500.0
    assert combined.num_total_gpus == 16
    assert combined.raw["(p)impl"] == "static_ctx"
    assert combined.raw["(d)impl"] == "afd"


def test_prefill_afd_uses_regular_decode_metrics():
    afd_prefill = _estimate_result(
        raw={
            "phase": "prefill",
            "osl": 50,
            "ttft": 200.0,
            "tpot": 0.0,
            "request_latency": 200.0,
            "b_total": 32,
            "seq/s": 0.0,
            "tokens/s": 0.0,
            "tokens/s/user": 0.0,
            "num_total_gpus": 10,
            "memory": 50.0,
            "power_w": 0.0,
        }
    )
    regular_decode = _estimate_result(
        mode="static_gen",
        raw={
            "osl": 50,
            "tpot": 20.0,
            "seq/s": 100.0,
            "concurrency": 16,
            "num_total_gpus": 2,
            "memory": 20.0,
            "power_w": 250.0,
        },
    )

    combined = _combine_afd_static_estimate_results(
        afd_result=afd_prefill,
        static_result=regular_decode,
        afd_phase="prefill",
    )

    assert combined.ttft == 200.0
    assert combined.tpot == 20.0
    assert combined.request_latency == 1180.0
    assert combined.seq_per_second == 100.0
    assert combined.tokens_per_second == 5000.0
    assert combined.concurrency == 16
    assert combined.num_total_gpus == 12
    assert combined.raw["(p)impl"] == "afd"
    assert combined.raw["(d)impl"] == "static_gen"


def test_run_afd_estimate_passes_prefix_and_nextn(monkeypatch):
    captured = {}

    class FakeDatabase:
        system_spec: ClassVar[dict] = {
            "node": {"num_gpus_per_node": 8},
            "gpu": {"mem_capacity": 80 * (1 << 30)},
        }

    class FakeSession:
        def __init__(self, **kwargs):
            captured["a_model_config"] = kwargs["a_model_config"]
            captured["f_model_config"] = kwargs["f_model_config"]

        def run_afd(self, runtime_config, **kwargs):
            captured["runtime_config"] = runtime_config
            captured["speculative_profile"] = kwargs["speculative_profile"]
            summary = InferenceSummary(runtime_config)
            summary.set_oom(False)
            summary.set_result_dict(
                {
                    "phase": "decode",
                    "ttft": 0.0,
                    "tpot": 1.0,
                    "request_latency": 9.0,
                    "b_total": 1,
                    "num_total_gpus": 16,
                    "memory": 1.0,
                    "seq/s": 1.0,
                    "tokens/s": 9.0,
                    "tokens/s/gpu": 0.5625,
                    "tokens/s/user": 9.0,
                    "power_w": 0.0,
                }
            )
            return summary

    monkeypatch.setattr("aiconfigurator.sdk.inference_session.AFDInferenceSession", FakeSession)

    api._run_afd_estimate(
        model_path="test-model",
        system_name="test-system",
        backend_name="test-backend",
        resolved_version="test-version",
        isl=128,
        osl=10,
        tp_size=1,
        a_tp_size=1,
        n_a_nodes=1,
        n_f_nodes=1,
        a_batch_size=1,
        f_moe_ep_size=1,
        num_microbatches=None,
        pipeline_model="serial",
        comm_overhead_factor=1.0,
        afd_phase="decode",
        afd_combined_with_pd=False,
        afd_boundary_on_attn=True,
        gemm_quant_mode=None,
        kvcache_quant_mode=None,
        fmha_quant_mode=None,
        moe_quant_mode=None,
        comm_quant_mode=None,
        load_database=lambda _system_name: FakeDatabase(),
        get_backend=lambda _backend_name: SimpleNamespace(name=SimpleNamespace(value="test-backend")),
        get_model=lambda *_args, **_kwargs: None,
        free_gpu_memory_fraction=None,
        max_seq_len=None,
        prefix=32,
        nextn=2,
        nextn_accepted=0.85,
    )

    assert captured["runtime_config"].prefix == 32
    assert captured["a_model_config"].nextn == 2
    assert captured["f_model_config"].nextn == 2
    assert not hasattr(captured["a_model_config"], "nextn_accepted")
    assert not hasattr(captured["f_model_config"], "nextn_accepted")
    assert captured["speculative_profile"].expected_accepted_tokens == 0.85


def test_afd_prefill_uses_uncached_prefix_suffix_for_token_math(monkeypatch):
    """Prefill comm/compute math must size token volume by the uncached
    suffix ``isl - prefix``, not the raw ``isl``.

    All five AFD comm ops (cross-pool A↔F transfers + F-node AG/RS +
    A-side combine) receive ``x = a_batch_size * (isl - prefix)`` —
    the per-A-rank token count for that phase. ``_sum_latency`` runs
    once per pool with the same suffix length. Regressing this silently
    over-counts prefill bandwidth by the prefix-cache hit rate.
    """
    from aiconfigurator.sdk.inference_session import _AFDCommOps

    captured = {"x_queries": [], "sum_latency_seq_lens": []}

    class FakeCommOp:
        def __init__(self, name):
            self._name = name

        def query(self, _database, *, x):
            captured["x_queries"].append((self._name, x))
            # Any float-like works: the session only calls ``float(result)``
            # before folding it into the breakdown dict.
            return 0.0

    def fake_build_comm_ops(self, _a_model, _f_model, *, rank_mapping="one_to_one"):
        return _AFDCommOps(
            a2f=FakeCommOp("afd_a2f_transfer"),
            f2a=FakeCommOp("afd_f2a_transfer"),
            f_ag=FakeCommOp("afd_f_node_allgather"),
            f_rs=FakeCommOp("afd_f_node_reducescatter"),
            a_combine=FakeCommOp("afd_a_side_combine"),
        )

    def fake_sum_latency(self, _ops, *, batch_size, seq_len, model, runtime_config, is_context):
        captured["sum_latency_seq_lens"].append(seq_len)
        return 2.0, {}

    def fake_memory_summary(self, _memory, runtime_config, _free_gpu_memory_fraction):
        summary = InferenceSummary(runtime_config)
        summary.set_oom(False)
        summary.set_kv_cache_oom(False)
        return summary

    monkeypatch.setattr(
        "aiconfigurator.sdk.afd_partition.build_afd_ops_partition",
        lambda *_args, **_kwargs: SimpleNamespace(attn_ops=[], ffn_ops=[]),
    )
    monkeypatch.setattr(AFDInferenceSession, "_build_afd_comm_ops", fake_build_comm_ops)
    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", fake_sum_latency)
    monkeypatch.setattr(AFDInferenceSession, "_estimate_a_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_estimate_f_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_check_memory_dict", fake_memory_summary)

    afd_config = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=2,
        a_batch_size=3,
        num_microbatches=1,
        f_moe_ep_size=1,
    )
    session = AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(),
        f_model_config=SimpleNamespace(),
        database=object(),
        backend=object(),
        afd_config=afd_config,
    )

    session._simulate_phase(
        phase="prefill",
        runtime_config=RuntimeConfig(isl=128, osl=10, prefix=48),
        a_model=SimpleNamespace(_num_layers=2),
        f_model=SimpleNamespace(_num_layers=2),
        free_gpu_memory_fraction=None,
        max_seq_len=None,
    )

    expected_x = afd_config.a_batch_size * 80  # a_batch_size * (isl - prefix)
    assert [x for _, x in captured["x_queries"]] == [expected_x] * 5
    assert {name for name, _ in captured["x_queries"]} == {
        "afd_a2f_transfer",
        "afd_f2a_transfer",
        "afd_f_node_allgather",
        "afd_f_node_reducescatter",
        "afd_a_side_combine",
    }
    assert captured["sum_latency_seq_lens"] == [80, 80]


def test_afd_summary_concurrency_reflects_total_in_flight_batch(monkeypatch):
    """``concurrency`` in the AFD summary equals the configured total batch.

    ``a_batch_size`` is the total in-flight batch per A-Worker.  The
    pipeline executes derived microbatches internally, so summary
    concurrency must not multiply the total batch by ``num_microbatches``.
    """
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=1.0, balance_ratio=1.0)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )
    expected_b_total = session._afd_config.n_a_workers * session._afd_config.a_batch_size

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    result = summary.get_result_dict()

    assert result["b_total"] == expected_b_total
    assert result["concurrency"] == expected_b_total
    assert result["b_micro_total"] == session._afd_config.n_a_workers * 2


def test_afd_summary_phase_both_paired_scalars_and_nan_unprefixed(monkeypatch):
    """``phase='both'`` writes both ``prefill_*`` and ``decode_*`` paired
    scalars and leaves the un-prefixed "headline" form NaN/None.

    Picking decode-only (or prefill-only) values as the un-prefixed scalar
    would silently discard the other phase's estimate when the two diverge,
    so NaN/None on the un-prefixed form makes the paired columns the only
    readable source of truth and prevents accidental misuse downstream.
    """
    prefill_metrics = _fake_phase_metrics(
        t_a_layer=0.5,
        t_f_layer=0.7,
        balance_ratio=0.71,
        t_a2f_layer=0.05,
        t_f2a_layer=0.05,
        t_step=12.5,
        comm_hidden=True,
    )
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.2,
        t_f_layer=0.9,
        balance_ratio=1.33,
        t_a2f_layer=0.1,
        t_f2a_layer=0.1,
        t_step=50.0,
        comm_hidden=False,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=prefill_metrics,
        decode_metrics=decode_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="both")
    result = summary.get_result_dict()

    assert result["phase"] == "both"

    # Paired scalars carry the per-phase values directly.
    assert result["prefill_t_a_layer"] == pytest.approx(0.5)
    assert result["prefill_t_f_layer"] == pytest.approx(0.7)
    assert result["prefill_balance_ratio"] == pytest.approx(0.71)
    assert result["prefill_t_step"] == pytest.approx(12.5)
    assert result["prefill_comm_hidden"] is True
    assert result["decode_t_a_layer"] == pytest.approx(1.2)
    assert result["decode_t_f_layer"] == pytest.approx(0.9)
    assert result["decode_balance_ratio"] == pytest.approx(1.33)
    assert result["decode_t_step"] == pytest.approx(50.0)
    assert result["decode_comm_hidden"] is False

    # Un-prefixed scalars are NaN (numeric) / None (bool) so consumers
    # cannot accidentally treat decode-only values as the both-phase answer.
    for key in ("t_a_layer", "t_f_layer", "t_a2f_layer", "t_f2a_layer", "t_c_layer", "t_step", "balance_ratio"):
        assert math.isnan(result[key]), f"expected NaN un-prefixed {key} in phase=both, got {result[key]!r}"
    assert result["comm_hidden"] is None


def test_afd_summary_phase_prefill_mirrors_unprefixed_into_prefill_pair(monkeypatch):
    """Single-phase prefill: un-prefixed == ``prefill_*``; ``decode_*`` NaN/None.

    Guards the back-compat contract for existing single-phase AFD users
    (the un-prefixed columns still carry the headline values) while still
    populating the new paired columns so combined-with-PD merges can rely
    on a uniform schema.
    """
    prefill_metrics = _fake_phase_metrics(
        t_a_layer=0.5,
        t_f_layer=0.7,
        balance_ratio=0.71,
        t_step=12.5,
        comm_hidden=True,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=prefill_metrics,
        decode_metrics=prefill_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="prefill")
    result = summary.get_result_dict()

    assert result["phase"] == "prefill"

    # Un-prefixed mirrors the prefill values.
    assert result["t_a_layer"] == pytest.approx(0.5)
    assert result["t_f_layer"] == pytest.approx(0.7)
    assert result["balance_ratio"] == pytest.approx(0.71)
    assert result["t_step"] == pytest.approx(12.5)
    assert result["comm_hidden"] is True

    # Prefill-pair matches; decode-pair is NaN/None.
    assert result["prefill_t_a_layer"] == pytest.approx(0.5)
    assert result["prefill_comm_hidden"] is True
    for key in (
        "decode_t_a_layer",
        "decode_t_f_layer",
        "decode_t_a2f_layer",
        "decode_t_f2a_layer",
        "decode_t_c_layer",
        "decode_t_step",
        "decode_balance_ratio",
    ):
        assert math.isnan(result[key]), f"expected NaN {key} in phase=prefill, got {result[key]!r}"
    assert result["decode_comm_hidden"] is None


def test_afd_summary_phase_decode_mirrors_unprefixed_into_decode_pair(monkeypatch):
    """Mirror of the prefill case: single-phase decode populates ``decode_*``
    and the un-prefixed form; ``prefill_*`` are NaN/None.
    """
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.2,
        t_f_layer=0.9,
        balance_ratio=1.33,
        t_step=50.0,
        comm_hidden=False,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=decode_metrics,
        decode_metrics=decode_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    result = summary.get_result_dict()

    assert result["phase"] == "decode"

    assert result["t_a_layer"] == pytest.approx(1.2)
    assert result["balance_ratio"] == pytest.approx(1.33)
    assert result["comm_hidden"] is False

    assert result["decode_t_a_layer"] == pytest.approx(1.2)
    assert result["decode_comm_hidden"] is False
    for key in (
        "prefill_t_a_layer",
        "prefill_t_f_layer",
        "prefill_t_a2f_layer",
        "prefill_t_f2a_layer",
        "prefill_t_c_layer",
        "prefill_t_step",
        "prefill_balance_ratio",
    ):
        assert math.isnan(result[key]), f"expected NaN {key} in phase=decode, got {result[key]!r}"
    assert result["prefill_comm_hidden"] is None


def test_combined_with_pd_keeps_afd_side_pair_and_marks_static_side_nan(monkeypatch):
    """When AFD-decode is combined with static-prefill, the merged result
    keeps the decode-side AFD pair populated and leaves the prefill pair
    NaN/None, flagging "prefill was not modeled by AFD".

    On the ``combined_with_pd=True`` path the merged result advertises
    end-to-end TTFT/TPOT, but the layer-scalar schema must still make it
    clear which phase actually went through the AFD model; the static
    side stays NaN/None so readers do not mistake its rows for AFD output.
    """
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.2,
        t_f_layer=0.9,
        balance_ratio=1.33,
        t_step=50.0,
        comm_hidden=False,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=decode_metrics,
        decode_metrics=decode_metrics,
        combined_with_pd=True,
    )

    afd_summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    afd_raw = afd_summary.get_result_dict()
    afd_result = _estimate_result(raw=afd_raw, mode="afd", summary=afd_summary)

    # Static-prefill stand-in: a minimal EstimateResult that the combiner
    # treats as the "other phase". Only the fields the combiner reads
    # (mode/raw/throughput numbers) need to be plausible.
    static_raw = {
        "model": afd_raw.get("model", "test-model"),
        "(p)tp": 1,
        "(p)pp": 1,
        "(p)dp": 1,
        "(p)bs": 1,
        "(p)gpus": 1,
        "(p)impl": "trtllm",
        "ttft": 25.0,
        "seq/s": 5.0,
        "tokens/s/user": 100.0,
    }
    static_result = _estimate_result(raw=static_raw, mode="static")

    merged = _combine_afd_static_estimate_results(
        afd_result=afd_result,
        static_result=static_result,
        afd_phase="decode",
    )

    # Decode-side pair (AFD): real values.
    assert merged.raw["decode_t_a_layer"] == pytest.approx(1.2)
    assert merged.raw["decode_balance_ratio"] == pytest.approx(1.33)
    assert merged.raw["decode_comm_hidden"] is False

    # Prefill-side pair (static, not AFD): NaN/None marks "not estimated
    # under the AFD model".
    for key in (
        "prefill_t_a_layer",
        "prefill_t_f_layer",
        "prefill_t_a2f_layer",
        "prefill_t_f2a_layer",
        "prefill_t_c_layer",
        "prefill_t_step",
        "prefill_balance_ratio",
    ):
        assert math.isnan(merged.raw[key]), f"expected NaN {key} on static prefill side, got {merged.raw[key]!r}"
    assert merged.raw["prefill_comm_hidden"] is None


def test_afd_config_phase_both_with_combined_with_pd_raises():
    """``phase='both'`` + ``combined_with_pd=True`` must fail at construction.

    AFD already covers prefill+decode internally in 'both' mode, so combining
    it with a separate static pool is conceptually inconsistent; the invariant
    is enforced in ``AFDConfig.__post_init__`` for defense-in-depth.
    """
    with pytest.raises(ValueError, match="combined_with_pd=True is incompatible with phase='both'"):
        AFDConfig(
            n_a_nodes=1,
            n_f_nodes=1,
            gpus_per_node=8,
            tp_a=1,
            phase="both",
            combined_with_pd=True,
        )


def test_afd_config_phase_decode_with_combined_with_pd_allowed():
    """``phase='decode'`` + ``combined_with_pd=True`` must construct cleanly.

    This is the default sizing scenario (AFD decode pool + regular prefill
    pool); regressing it would break every default CLI invocation.
    """
    cfg = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=1,
        phase="decode",
        combined_with_pd=True,
    )
    assert cfg.combined_with_pd is True
    assert cfg.phase == "decode"


def _afd_cli_estimate_kwargs(**overrides):
    """Minimal kwargs to drive ``cli_estimate(mode='afd', ...)`` in tests."""
    base = dict(
        model_path="test-model",
        system_name="test-system",
        mode="afd",
        database_mode="SOL",
        n_a_nodes=1,
        n_f_nodes=1,
        a_tp_size=1,
        a_batch_size=1,
        f_moe_ep_size=1,
        num_microbatches=3,
        pipeline_model="serial",
        comm_overhead_factor=1.0,
        afd_phase="decode",
        afd_boundary_on_attn=True,
    )
    base.update(overrides)
    return base


def _install_estimate_perf_db_stubs(monkeypatch):
    """Stub out perf_database lookups so cli_estimate(mode='afd') can run."""
    import aiconfigurator.sdk.perf_database as perf_database

    monkeypatch.setattr(
        perf_database,
        "get_latest_database_version",
        lambda system, backend, systems_paths=None: "test-version",
    )
    monkeypatch.setattr(
        perf_database,
        "get_database",
        lambda system, backend, version, systems_paths=None, allow_missing_data=False: object(),
    )


def test_cli_estimate_afd_combined_with_pd_false_skips_static(monkeypatch):
    """``combined_with_pd=False`` must return the AFD-only result.

    The static estimate for the other phase is the dominant cost of this
    CLI path; gating it on ``combined_with_pd`` is the whole point of the
    flag. Verify ``_run_static_estimate`` is NOT invoked in this branch.
    """
    _install_estimate_perf_db_stubs(monkeypatch)
    afd_only = _estimate_result(
        raw={"phase": "decode", "num_total_gpus": 8, "memory": 1.0},
    )
    static_calls = []

    monkeypatch.setattr(api, "_run_afd_estimate", lambda **_kwargs: afd_only)
    monkeypatch.setattr(
        api,
        "_run_static_estimate",
        lambda **kwargs: static_calls.append(kwargs) or _estimate_result(raw={}),
    )

    result = api.cli_estimate(
        **_afd_cli_estimate_kwargs(afd_combined_with_pd=False),
    )

    assert result is afd_only
    assert static_calls == []


def test_cli_estimate_afd_combined_with_pd_true_runs_static_combine(monkeypatch):
    """``combined_with_pd=True`` (the default) must invoke the static path.

    This is the legacy behavior we must preserve: a single ``--afd-phase=decode``
    CLI call still has to size the regular prefill pool and merge results,
    otherwise users silently lose the rate-matched throughput / total-GPU
    accounting.
    """
    _install_estimate_perf_db_stubs(monkeypatch)
    afd_decode = _estimate_result(
        raw={
            "phase": "decode",
            "osl": 10,
            "tpot": 5.0,
            "b_total": 4,
            "concurrency": 4,
            "seq/s": 10.0,
            "num_total_gpus": 8,
            "memory": 1.0,
            "power_w": 0.0,
        },
    )
    static_ctx = _estimate_result(
        mode="static_ctx",
        raw={
            "osl": 10,
            "ttft": 50.0,
            "seq/s": 12.0,
            "num_total_gpus": 4,
            "memory": 1.0,
            "power_w": 0.0,
        },
    )
    captured = {}

    def fake_run_afd_estimate(**kwargs):
        captured["afd_kwargs"] = kwargs
        return afd_decode

    def fake_run_static_estimate(**kwargs):
        captured["static_kwargs"] = kwargs
        return static_ctx

    monkeypatch.setattr(api, "_run_afd_estimate", fake_run_afd_estimate)
    monkeypatch.setattr(api, "_run_static_estimate", fake_run_static_estimate)

    result = api.cli_estimate(
        **_afd_cli_estimate_kwargs(afd_combined_with_pd=True),
    )

    # Merged result should reflect static_ctx TTFT and AFD TPOT, plus the
    # summed GPU budget — the canonical "AFD-decode + regular-prefill" sizing.
    assert "afd_kwargs" in captured and "static_kwargs" in captured
    assert captured["static_kwargs"]["static_mode"] == "static_ctx"
    assert result.ttft == 50.0
    assert result.tpot == 5.0
    assert result.num_total_gpus == 12


def test_cli_estimate_afd_phase_both_with_combined_with_pd_raises(monkeypatch):
    """``phase='both'`` + ``combined_with_pd=True`` is mutually exclusive.

    The CLI layer rejects this combination up-front (before any database or
    model is loaded) so the user gets a clear, actionable error instead of a
    deep failure inside ``AFDConfig.__post_init__`` or the static estimator.
    """
    _install_estimate_perf_db_stubs(monkeypatch)
    monkeypatch.setattr(api, "_run_afd_estimate", lambda **_kwargs: _estimate_result(raw={}))

    with pytest.raises(ValueError, match="afd_combined_with_pd=True is incompatible with afd_phase='both'"):
        api.cli_estimate(
            **_afd_cli_estimate_kwargs(afd_phase="both", afd_combined_with_pd=True),
        )


def test_cli_main_estimate_value_error_exits_without_traceback(monkeypatch):
    """Estimate-mode validation errors surface as a concise CLI error.

    Invalid AFD sizing (e.g. ``f_moe_ep_size`` not dividing ``f_tp_size``)
    raises ``ValueError`` deep in the SDK. ``cli.main.main`` must convert it
    to ``SystemExit`` so the user sees the actionable message instead of a
    full Python traceback.
    """
    from types import SimpleNamespace

    import aiconfigurator.cli.main as cli_main
    import aiconfigurator.sdk.perf_database as perf_database

    monkeypatch.setattr(perf_database, "set_systems_paths", lambda _paths: None)
    monkeypatch.setattr(
        cli_main,
        "_run_estimate_mode",
        lambda _args: (_ for _ in ()).throw(ValueError("f_moe_ep_size (7) must be a positive divisor")),
    )

    args = SimpleNamespace(mode="estimate", systems_paths=None, top_n=5, no_color=True)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(args)

    # SystemExit carries the plain message (no traceback, exit code 1).
    assert "f_moe_ep_size (7) must be a positive divisor" in str(exc_info.value)
