# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-EP emission of ``build_moe_block_ops``: oracle equivalence vs the legacy wideEP graphs.

RECORDED-VALUE ORACLE FORM. The class-based form of this file instantiated
the live legacy model classes (``WideEPDeepSeekModel`` /
``TrtllmWideEPDeepSeekModel``) and pinned the builder op-for-op
(name/type/``__dict__``) and value-for-value (rel <= 1e-9 over six token
points per phase, EPLB on/off, num_slots=288) against them — 24/24 green
before conversion. This committed form outlives the Task 6 class deletion:

- structural expectations are RECORDED transcriptions of the legacy
  constructor calls (cited per site), validated verbatim in that run;
- value expectations are recomputed through the surviving legacy QUERY
  functions (``query_wideep_deepep_normal``/``ll``, ``query_moe`` with
  ``moe_backend="deepep_moe"``, ``query_trtllm_alltoall``,
  ``query_wideep_moe_compute``) exactly the way the legacy ops called them.

Oracle configs (pp=1 PRECONDITION): both legacy graphs derive the comm node
span from the whole MoE group (``moe_ep * moe_tp`` GPUs), which coincides
with the builder's ``nodes_for(moe_ep * moe_tp, gpus_per_node)`` (A5) only
when the worker spans exactly that group — i.e. pp=1 and attention width ==
MoE width. Both oracle configs use pp=1.

A6 name mapping between the two graphs:

- sglang: legacy ``{p}_moe_pre_dispatch`` (one op whose deepep table row sums
  dispatch+combine) == builder ``{p}_moe_dispatch`` + ``{p}_moe_combine``;
  there is no legacy post_dispatch op.
- trtllm: legacy ``{p}_moe_pre_dispatch`` (queries alltoall prepare+dispatch)
  == builder ``{p}_moe_prepare`` + ``{p}_moe_dispatch``; legacy
  ``{p}_moe_post_dispatch`` == builder ``{p}_moe_combine``.

Workload-distribution strings are transcribed from the legacy classes into
this test (the builder takes the string as a parameter; Task 6 moves the
computation into the model classes): sglang prefill alpha 0.6-if-eplb else
1.01, decode alpha 1.01 (deepseek.py:1089-1101); trtllm ``power_law_1.01``
with ``_eplb`` suffix when EPLB is on (deepseek.py:626-636).
"""

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common, config
from aiconfigurator_core.sdk.models.blocks import MoEBlockShape, build_moe_block_ops
from aiconfigurator_core.sdk.models.helpers import _get_model_info
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.perf_database import get_database

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SYSTEMS_ROOT = REPO_ROOT / "aic-core" / "src" / "aiconfigurator_core" / "systems"
SYSTEMS_DATA_ROOT = SYSTEMS_ROOT / "data"

DSR1 = "deepseek-ai/DeepSeek-R1"

# Recorded from the legacy oracle instances (nextn=0): the classes scale MoE
# ops by their full layer count; trtllm generation additionally carries the
# PDL factor (TrtllmWideEPDeepSeekModel._pdl_factor = 0.9, deepseek.py:604).
NUM_LAYERS = 61
PDL_FACTOR = 0.9

# Recorded quant resolution of the oracle ModelConfigs after get_model's
# checkpoint-driven defaults (DeepSeek-R1 is an FP8 checkpoint).
SGLANG_GEMM_QUANT = common.GEMMQuantMode.fp8_block
SGLANG_MOE_QUANT = common.MoEQuantMode.fp8_block
TRTLLM_GEMM_QUANT = common.GEMMQuantMode.fp8_block
TRTLLM_MOE_QUANT = common.MoEQuantMode.nvfp4

# Shipped-data gates (same resolution as the PR 1 query-equivalence tests).
SGLANG_DATA_PATHS = [
    resolve_op_data_path(str(SYSTEMS_DATA_ROOT / "h200_sxm"), "sglang", "0.5.6.post2", filename)
    for filename in (
        "wideep_deepep_normal_perf.parquet",
        "wideep_deepep_ll_perf.parquet",
        "wideep_context_moe_perf.parquet",
        "wideep_generation_moe_perf.parquet",
    )
]
TRTLLM_DATA_PATHS = [
    resolve_op_data_path(str(SYSTEMS_DATA_ROOT / "gb200"), "trtllm", "1.3.0rc10", filename)
    for filename in ("trtllm_alltoall_perf.parquet", "wideep_moe_perf.parquet")
]

sglang_data_present = pytest.mark.skipif(
    not all(os.path.exists(p) for p in SGLANG_DATA_PATHS),
    reason="shipped h200_sxm sglang 0.5.6.post2 wideEP parquets not present",
)
trtllm_data_present = pytest.mark.skipif(
    not all(os.path.exists(p) for p in TRTLLM_DATA_PATHS),
    reason="shipped gb200 trtllm 1.3.0rc10 wideEP parquets not present",
)

REL_TOL = 1e-9

# Six token points per phase spanning small -> in-range -> beyond-range
# overflow of the shipped comm/compute token grids.
SGLANG_CONTEXT_TOKENS = (2, 8, 96, 1024, 16384, 262144)
SGLANG_GENERATION_TOKENS = (2, 8, 32, 128, 256, 2048)
TRTLLM_TOKENS = (1, 3, 48, 512, 4096, 131072)


def _dsr1_shape():
    shape = MoEBlockShape.from_model_info(_get_model_info(DSR1))
    # The oracle transcriptions below hardcode this geometry.
    assert (shape.hidden_size, shape.moe_inter_size, shape.topk, shape.num_experts) == (7168, 2048, 8, 256)
    assert shape.num_shared_experts == 1
    return shape


def _names(op_list):
    return [op._name for op in op_list]


def _assert_ops_identical(built, expected):
    """Same op classes in order, same names, same constructor-derived state."""
    assert _names(built) == _names(expected)
    for got, want in zip(built, expected, strict=True):
        assert type(got) is type(want), got._name
        assert got.__dict__ == want.__dict__, got._name


def _lat(op, db, x):
    return float(op.query(db, x=x))


def _assert_close(got, want, context):
    assert got == pytest.approx(want, rel=REL_TOL), context


# ---------------------------------------------------------------------------
# (a) sglang DeepSeek-R1 EP32 (h200-like, 4 nodes) vs the legacy deepep queries
# ---------------------------------------------------------------------------

SGLANG_COMM = {"context": "deepep_ht", "generation": "deepep_ll"}


def _sglang_distribution(phase, enable_eplb):
    # Transcribed from WideEPDeepSeekModel (deepseek.py:1089-1101): prefill
    # alpha is 0.6 when EPLB is on else 1.01; decode alpha is always 1.01.
    alpha = (0.6 if enable_eplb else 1.01) if phase == "context" else 1.01
    return f"power_law_{alpha}"


def _sglang_cfg(enable_eplb):
    cfg = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=32,
        attention_dp_size=32,
        gemm_quant_mode=SGLANG_GEMM_QUANT,
        moe_quant_mode=SGLANG_MOE_QUANT,
        moe_backend="deepep_moe",
        enable_eplb=enable_eplb,
    )
    cfg.moe_comm_backend = dict(SGLANG_COMM)
    return cfg


def _build_sglang(cfg, phase, enable_eplb, model_family="DEEPSEEK"):
    return build_moe_block_ops(
        phase,
        _dsr1_shape(),
        cfg,
        cfg.moe_quant_mode,
        _sglang_distribution(phase, enable_eplb),
        scale_factor=NUM_LAYERS,  # mtp factor is 1.0 at nextn=0
        backend_name="sglang",
        inference_phase=phase,
        model_family=model_family,
        gpus_per_node=8,
    )


def _legacy_sglang_comm_latency(db, phase, x):
    """The legacy graph's single MoEDispatch op (deepep branch), recomputed.

    WideEPDeepSeekModel wires ONE dispatch op whose deepep table row sums the
    dispatch+combine legs: context -> ``query_wideep_deepep_normal`` (node
    span = 32 GPUs / 8 per node, tokens // scale_num_tokens with tp=1),
    generation -> ``query_wideep_deepep_ll``. Latency scales by the layer
    count (deepseek.py:1206-1227, 1320-1340).
    """
    if phase == "context":
        result = db.query_wideep_deepep_normal(
            node_num=4, num_tokens=x, num_experts=256, topk=8, hidden_size=7168, sms=20
        )
    else:
        result = db.query_wideep_deepep_ll(node_num=4, num_tokens=x, num_experts=256, topk=8, hidden_size=7168)
    return float(result) * NUM_LAYERS


def _legacy_sglang_moe_latency(db, phase, x, enable_eplb):
    """The legacy graph's MoE op, recomputed via ``query_moe``.

    Tokens globalize by attention_dp (x * 32); the EPLB 0.8 prefill token
    correction applies inside the query. The legacy class passed
    ``enable_eplb=False`` for generation unconditionally (deepseek.py:1359).
    """
    result = db.query_moe(
        num_tokens=x * 32,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        moe_tp_size=1,
        moe_ep_size=32,
        quant_mode=SGLANG_MOE_QUANT,
        workload_distribution=_sglang_distribution(phase, enable_eplb),
        is_context=(phase == "context"),
        moe_backend="deepep_moe",
        enable_eplb=enable_eplb and phase == "context",
    )
    return float(result) * NUM_LAYERS


def _sglang_expected_shared_triplet(phase):
    """Recorded transcription of the legacy deepep shared-expert triplet.

    deepseek.py:1174-1204 (context: ``scale_num_tokens=tp_size`` +
    ``seq_split=cp``; here tp=1, cp=1) and :1294-1318 (generation: neither
    kwarg). Full ``2 * moe_inter_size`` — WideEP ADP mode shared_tp=1.
    """
    token_kwargs = {"scale_num_tokens": 1, "seq_split": 1} if phase == "context" else {}
    return [
        ops.GEMM(f"{phase}_gate_ffn1_gemm", NUM_LAYERS, 2 * 2048, 7168, SGLANG_GEMM_QUANT, **token_kwargs),
        ops.ElementWise(f"{phase}_act_gate", NUM_LAYERS, 2 * 2048, 2048, 0.8, **token_kwargs),
        ops.GEMM(f"{phase}_ffn2_gemm", NUM_LAYERS, 7168, 2048, SGLANG_GEMM_QUANT, **token_kwargs),
    ]


class TestSglangLargeEPStructure:
    def test_context_block_structure_and_shared_triplet(self):
        built = _build_sglang(_sglang_cfg(enable_eplb=False), "context", enable_eplb=False)
        # DEEPSEEK-sglang A3 variant strips the router on the deepep path
        # (the legacy class never wires one); shared experts keep the legacy
        # deepep names and the scale_num_tokens=tp flavor.
        assert _names(built) == [
            "context_gate_ffn1_gemm",
            "context_act_gate",
            "context_ffn2_gemm",
            "context_moe_dispatch",
            "context_moe",
            "context_moe_combine",
        ]
        _assert_ops_identical(built[:3], _sglang_expected_shared_triplet("context"))

        dispatch, moe, combine = built[3:]
        for a2a, phase in ((dispatch, "dispatch"), (combine, "combine")):
            assert isinstance(a2a, ops.MoEAllToAll)
            assert a2a._phase == phase
            assert a2a._comm_backend == "deepep_ht"
            assert a2a._comm_dtype == "default"
            assert a2a._node_num == 4  # nodes_for(32 * 1, 8)
            assert a2a._sms == 20  # cfg.sms rides only the HT backend
            assert a2a._attention_tp_size == 1  # cfg.tp_size
            assert a2a._scale_factor == NUM_LAYERS
        assert isinstance(moe, ops.EPMoE)
        assert moe._workload_distribution == "power_law_1.01"
        assert moe._enable_eplb is False
        assert moe._num_slots == 256  # sglang has no EPLB slot axis
        assert moe._attention_dp_size == 32

    def test_generation_block_structure(self):
        built = _build_sglang(_sglang_cfg(enable_eplb=False), "generation", enable_eplb=False)
        assert _names(built) == [
            "generation_gate_ffn1_gemm",
            "generation_act_gate",
            "generation_ffn2_gemm",
            "generation_moe_dispatch",
            "generation_moe",
            "generation_moe_combine",
        ]
        _assert_ops_identical(built[:3], _sglang_expected_shared_triplet("generation"))
        dispatch, moe, combine = built[3:]
        assert dispatch._comm_backend == combine._comm_backend == "deepep_ll"
        assert dispatch._sms == combine._sms == 0  # LL has no SM budget
        assert moe._workload_distribution == "power_law_1.01"

    def test_eplb_flips_context_distribution_and_eplb_flag(self):
        built = _build_sglang(_sglang_cfg(enable_eplb=True), "context", enable_eplb=True)
        moe = built[4]
        assert moe._workload_distribution == "power_law_0.6"
        assert moe._enable_eplb is True


@sglang_data_present
class TestSglangLargeEPValues:
    """Summed builder query latencies == legacy query latencies (rel<=1e-9)."""

    @pytest.fixture(scope="class")
    def db(self):
        db = get_database("h200_sxm", "sglang", "0.5.6.post2")
        assert db is not None
        return db

    @pytest.mark.parametrize("enable_eplb", [False, True])
    @pytest.mark.parametrize("phase", ["context", "generation"])
    def test_comm_and_compute_match_legacy_queries(self, db, phase, enable_eplb):
        built = _build_sglang(_sglang_cfg(enable_eplb), phase, enable_eplb)
        dispatch, moe, combine = built[3:]

        tokens = SGLANG_CONTEXT_TOKENS if phase == "context" else SGLANG_GENERATION_TOKENS
        for x in tokens:
            context = f"sglang {phase} eplb={enable_eplb} x={x}"
            # A6: legacy pre_dispatch rode a summed dispatch+combine table row.
            _assert_close(
                _lat(dispatch, db, x) + _lat(combine, db, x), _legacy_sglang_comm_latency(db, phase, x), context
            )
            _assert_close(_lat(moe, db, x), _legacy_sglang_moe_latency(db, phase, x, enable_eplb), context)


# ---------------------------------------------------------------------------
# (b) trtllm DeepSeek-R1 GB200 EP16 nvfp4 vs the legacy alltoall/compute queries
# ---------------------------------------------------------------------------

TRTLLM_COMM = {"context": "nvlink_two_sided", "generation": "nvlink_two_sided"}

# (enable_eplb, wideep_num_slots) oracle scenarios: EPLB off, EPLB on without
# redundant slots, and the EPLB-redundant num_slots=288 variant.
TRTLLM_SCENARIOS = [(False, None), (True, None), (True, 288)]


def _trtllm_distribution(enable_eplb):
    # Transcribed from TrtllmWideEPDeepSeekModel (deepseek.py:626-636).
    return "power_law_1.01_eplb" if enable_eplb else "power_law_1.01"


def _trtllm_scale(phase):
    return NUM_LAYERS if phase == "context" else NUM_LAYERS * PDL_FACTOR  # mtp factor 1.0 at nextn=0


def _trtllm_cfg(enable_eplb, wideep_num_slots):
    # enable_wideep=True dropped: the deprecated flag is inert -- the large-EP
    # graph is selected by moe_comm_backend (set below), which is what this
    # oracle exercises.
    cfg = config.ModelConfig(
        tp_size=1,
        moe_tp_size=1,
        moe_ep_size=16,
        attention_dp_size=16,
        gemm_quant_mode=TRTLLM_GEMM_QUANT,
        moe_quant_mode=TRTLLM_MOE_QUANT,
        enable_eplb=enable_eplb,
        wideep_num_slots=wideep_num_slots,
    )
    cfg.moe_comm_backend = dict(TRTLLM_COMM)
    return cfg


def _build_trtllm(cfg, phase, enable_eplb):
    return build_moe_block_ops(
        phase,
        _dsr1_shape(),
        cfg,
        cfg.moe_quant_mode,
        _trtllm_distribution(enable_eplb),
        scale_factor=_trtllm_scale(phase),
        backend_name="trtllm",
        inference_phase=phase,
        model_family="DEEPSEEK",
        gpus_per_node=4,
    )


def _legacy_trtllm_a2a_latency(db, op_name, phase, x):
    """One leg of the legacy TrtLLMWideEPMoEDispatch query, recomputed.

    The legacy op queried ``query_trtllm_alltoall`` per leg with
    ``moe_backend="wideep"`` (NVLinkTwoSided on SM100) and ``node_num=None``
    (derived as ep//4 = 4 on GB200 NVL4); pre_dispatch = prepare + dispatch,
    post_dispatch = combine (deepseek.py:759-813, 973-1011).
    """
    result = db.query_trtllm_alltoall(
        op_name=op_name,
        num_tokens=x,
        hidden_size=7168,
        topk=8,
        num_experts=256,
        moe_ep_size=16,
        quant_mode=TRTLLM_MOE_QUANT,
        moe_backend="wideep",
        node_num=None,
    )
    return float(result) * _trtllm_scale(phase)


def _legacy_trtllm_combine_op_name(phase):
    # The legacy graph enables the low-precision combine kernel only in
    # generation for nvfp4 runs (use_low_precision_combine=(quant==nvfp4),
    # deepseek.py:1005-1011); the context post_dispatch site (deepseek.py:
    # 798-812) never passes the flag.
    return "alltoall_combine_low_precision" if phase == "generation" else "alltoall_combine"


def _legacy_trtllm_moe_latency(db, phase, x, enable_eplb, num_slots):
    """The legacy TrtLLMWideEPMoE query, recomputed via ``query_wideep_moe_compute``."""
    result = db.query_wideep_moe_compute(
        num_tokens=x * 16,  # attention_dp globalizes tokens
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=num_slots or 256,
        moe_tp_size=1,
        moe_ep_size=16,
        quant_mode=TRTLLM_MOE_QUANT,
        workload_distribution=_trtllm_distribution(enable_eplb),
    )
    return float(result) * _trtllm_scale(phase)


def _trtllm_expected_router(phase):
    # Recorded transcription: deepseek.py:747-757 (context; the builder's
    # context seq_split=1 is state-identical to the legacy no-kwarg call)
    # and :966-972 (generation, x PDL factor).
    return ops.GEMM(f"{phase}_router_gemm", _trtllm_scale(phase), 256, 7168, common.GEMMQuantMode.bfloat16)


def _trtllm_expected_shared_triplet(phase):
    # Recorded transcription: deepseek.py:720-744 (context) and :940-962
    # (generation) — FULL 2 * moe_inter_size, no ÷tp ("WideEP ADP mode
    # shared_tp_size=1"), no token scaling.
    scale = _trtllm_scale(phase)
    return [
        ops.GEMM(f"{phase}_shared_gate_up_gemm", scale, 2 * 2048, 7168, TRTLLM_GEMM_QUANT),
        ops.ElementWise(f"{phase}_shared_act_gate", scale, 2 * 2048, 2048, 0.8),
        ops.GEMM(f"{phase}_shared_ffn2_gemm", scale, 7168, 2048, TRTLLM_GEMM_QUANT),
    ]


def _trtllm_expected_reduce_add(phase):
    # Recorded transcription: deepseek.py:816-824 (context), :1022-1032
    # (generation) — moe_reduce_add_shared_output, 2h -> h.
    return ops.ElementWise(f"{phase}_moe_reduce_add", _trtllm_scale(phase), 2 * 7168, 7168, 0.8)


class TestTrtllmLargeEPStructure:
    def test_context_block_structure(self):
        cfg = _trtllm_cfg(enable_eplb=False, wideep_num_slots=None)
        built = _build_trtllm(cfg, "context", enable_eplb=False)
        assert _names(built) == [
            "context_router_gemm",
            "context_shared_gate_up_gemm",
            "context_shared_act_gate",
            "context_shared_ffn2_gemm",
            "context_moe_prepare",
            "context_moe_dispatch",
            "context_moe",
            "context_moe_combine",
            "context_moe_reduce_add",
        ]
        _assert_ops_identical([built[0]], [_trtllm_expected_router("context")])
        _assert_ops_identical(built[1:4], _trtllm_expected_shared_triplet("context"))
        _assert_ops_identical([built[8]], [_trtllm_expected_reduce_add("context")])
        prepare, dispatch, moe, combine = built[4:8]
        for a2a, phase in ((prepare, "prepare"), (dispatch, "dispatch"), (combine, "combine")):
            assert isinstance(a2a, ops.MoEAllToAll)
            assert a2a._phase == phase
            assert a2a._comm_backend == "nvlink_two_sided"
            assert a2a._node_num == 4  # nodes_for(16 * 1, 4)
            assert a2a._sms == 0
            assert a2a._attention_tp_size == 1  # legacy trtllm alltoall gets undivided tokens
        # nvlink dispatch/prepare key the run dtype; the CONTEXT combine stays
        # at the run dtype — the legacy context post_dispatch never passes
        # use_low_precision_combine (deepseek.py:798-812).
        assert prepare._comm_dtype == dispatch._comm_dtype == "nvfp4"
        assert combine._comm_dtype == "nvfp4"
        assert isinstance(moe, ops.EPMoE)
        assert moe._enable_eplb is False  # trtllm EPLB rides the _eplb distribution, never the 0.8

    def test_generation_block_structure_overlap_and_low_precision_combine(self):
        cfg = _trtllm_cfg(enable_eplb=False, wideep_num_slots=None)
        built = _build_trtllm(cfg, "generation", enable_eplb=False)
        assert len(built) == 2
        overlap, reduce_add = built
        assert isinstance(overlap, ops.OverlapOp)
        assert overlap._name == "generation_moe_overlap"
        assert _names(overlap._group_a) == [
            "generation_router_gemm",
            "generation_moe_prepare",
            "generation_moe_dispatch",
            "generation_moe",
            "generation_moe_combine",
        ]
        _assert_ops_identical([overlap._group_a[0]], [_trtllm_expected_router("generation")])
        _assert_ops_identical(overlap._group_b, _trtllm_expected_shared_triplet("generation"))
        _assert_ops_identical([reduce_add], [_trtllm_expected_reduce_add("generation")])
        # nvfp4 generation combine uses the low-precision kernel rows
        # (legacy use_low_precision_combine=(quant==nvfp4), deepseek.py:1005-1011).
        combine = overlap._group_a[-1]
        assert combine._comm_dtype == "fp4"

    @pytest.mark.parametrize(("enable_eplb", "num_slots"), TRTLLM_SCENARIOS)
    def test_epmoe_mirrors_legacy_eplb_axes(self, enable_eplb, num_slots):
        cfg = _trtllm_cfg(enable_eplb, num_slots)
        built = _build_trtllm(cfg, "context", enable_eplb)
        moe = built[6]
        assert moe._workload_distribution == _trtllm_distribution(enable_eplb)
        assert moe._num_slots == (num_slots or 256)
        assert moe._quant_mode is common.MoEQuantMode.nvfp4
        assert moe._enable_eplb is False

    @pytest.mark.parametrize("phase", ["context", "generation"])
    def test_sharedless_shape_emits_no_reduce_add_or_overlap(self, phase):
        # moe_reduce_add models the routed-topk + SHARED add (2h -> h); a
        # shape without shared experts has no shared output to add, so the
        # block stays flat: router + prepare/dispatch/moe/combine only.
        cfg = _trtllm_cfg(enable_eplb=False, wideep_num_slots=None)
        shape = MoEBlockShape(
            hidden_size=1024, moe_inter_size=512, topk=4, num_experts=64, num_shared_experts=0, num_moe_layers=10
        )
        built = build_moe_block_ops(
            phase,
            shape,
            cfg,
            cfg.moe_quant_mode,
            _trtllm_distribution(False),
            scale_factor=_trtllm_scale(phase),
            backend_name="trtllm",
            inference_phase=phase,
            model_family="DEEPSEEK",
            gpus_per_node=4,
        )
        assert _names(built) == [
            f"{phase}_router_gemm",
            f"{phase}_moe_prepare",
            f"{phase}_moe_dispatch",
            f"{phase}_moe",
            f"{phase}_moe_combine",
        ]
        assert not any(isinstance(op, ops.OverlapOp) for op in built)


@trtllm_data_present
class TestTrtllmLargeEPValues:
    @pytest.fixture(scope="class")
    def db(self):
        db = get_database("gb200", "trtllm", "1.3.0rc10")
        assert db is not None
        return db

    @pytest.mark.parametrize(("enable_eplb", "num_slots"), TRTLLM_SCENARIOS)
    @pytest.mark.parametrize("phase", ["context", "generation"])
    def test_comm_and_compute_match_legacy_queries(self, db, phase, enable_eplb, num_slots):
        cfg = _trtllm_cfg(enable_eplb, num_slots)
        built = _build_trtllm(cfg, phase, enable_eplb)
        if phase == "context":
            prepare, dispatch, moe, combine = built[4:8]
        else:
            prepare, dispatch, moe, combine = built[0]._group_a[1:]

        for x in TRTLLM_TOKENS:
            context = f"trtllm {phase} eplb={enable_eplb} slots={num_slots} x={x}"
            # A6: legacy pre_dispatch = prepare + dispatch; post_dispatch = combine.
            _assert_close(
                _lat(prepare, db, x) + _lat(dispatch, db, x),
                _legacy_trtllm_a2a_latency(db, "alltoall_prepare", phase, x)
                + _legacy_trtllm_a2a_latency(db, "alltoall_dispatch", phase, x),
                context,
            )
            _assert_close(
                _lat(combine, db, x),
                _legacy_trtllm_a2a_latency(db, _legacy_trtllm_combine_op_name(phase), phase, x),
                context,
            )
            _assert_close(_lat(moe, db, x), _legacy_trtllm_moe_latency(db, phase, x, enable_eplb, num_slots), context)


# ---------------------------------------------------------------------------
# (c) A3 router-fidelity variants (registered in blocks/moe.py itself)
# ---------------------------------------------------------------------------


class TestA3RouterVariants:
    def _build(self, backend_name, model_family, comm_backend, phase="context"):
        cfg = config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        cfg.moe_comm_backend = {phase: comm_backend} if comm_backend else {}
        shape = MoEBlockShape(
            hidden_size=1024, moe_inter_size=512, topk=4, num_experts=64, num_shared_experts=0, num_moe_layers=10
        )
        return build_moe_block_ops(
            phase,
            shape,
            cfg,
            cfg.moe_quant_mode,
            "uniform",
            scale_factor=10,
            backend_name=backend_name,
            inference_phase=phase,
            model_family=model_family,
        )

    @pytest.mark.parametrize("family", ["DEEPSEEK", "DEEPSEEKV32"])
    def test_router_stripped_for_deepseek_sglang_deepep(self, family):
        built = self._build("sglang", family, "deepep_ht")
        assert "context_router_gemm" not in _names(built)
        assert _names(built) == ["context_moe_dispatch", "context_moe", "context_moe_combine"]

    def test_router_present_for_generic_family_on_sglang_deepep(self):
        built = self._build("sglang", "SOMEMOE", "deepep_ht")
        assert _names(built)[0] == "context_router_gemm"

    def test_router_present_for_deepseek_trtllm_nvlink(self):
        built = self._build("trtllm", "DEEPSEEK", "nvlink_two_sided")
        assert "context_router_gemm" in _names(built)

    def test_router_present_for_deepseek_sglang_fused_phase(self):
        # No comm backend for the phase -> default() returned unchanged.
        built = self._build("sglang", "DEEPSEEK", None)
        assert _names(built)[0] == "context_router_gemm"
        assert _names(built)[-1] == "context_moe_post_dispatch"


# ---------------------------------------------------------------------------
# (d) vllm G2 seed: synthetic parquets, deepep emission, end-to-end queries
# ---------------------------------------------------------------------------


def _write_parquet(path, rows):
    """Write one synthetic table and keep the version dir's Collector V3
    sidecar covering every table written into it so far. Without the sidecar
    a family-layout dir fails ``get_database()``'s strict-provenance check
    (design §5/§7.4), which CI runs with ``AIC_STRICT_PROVENANCE=1``; the
    synthetic data is complete, not partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)

    meta_path = path.parent / "collection_meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {"schema_version": 2, "tables": {}}
    meta["tables"][path.stem] = {"status": "complete"}
    meta_path.write_text(yaml.safe_dump(meta))


@pytest.fixture
def vllm_toy_db(tmp_path):
    """A real PerfDatabase over a synthetic systems tree: one vllm version
    carrying only the unified ``moe_a2a_perf`` / ``moe_ep_perf`` tables for a
    non-DeepSeek MoE shape (hidden 4096, topk 4, experts 64, ep 8)."""
    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    spec = yaml.safe_load((SYSTEMS_ROOT / "h200_sxm.yaml").read_text())
    spec["data_dir"] = "data/toy_sys"
    (systems_root / "toy_sys.yaml").write_text(yaml.safe_dump(spec))

    version_dir = systems_root / "data" / "toy_sys" / "moe" / "vllm" / "1.0"
    shape = {"hidden_size": 4096, "topk": 4, "num_experts": 64}
    a2a_rows = []
    for backend, sms, base_us in (("deepep_ht", 20, 100.0), ("deepep_ll", 0, 50.0)):
        for phase, factor in (("dispatch", 1.0), ("combine", 2.0)):
            for tokens in (64, 512):
                a2a_rows.append(
                    {
                        "comm_backend": backend,
                        "phase": phase,
                        "comm_dtype": "default",
                        "ep_size": 8,
                        "node_num": 1,
                        "sms": sms,
                        "num_tokens": tokens,
                        "latency": base_us * factor * tokens / 64.0,  # us
                        **shape,
                    }
                )
    _write_parquet(version_dir / "moe_a2a_perf.parquet", a2a_rows)

    ep_rows = [
        {
            "kernel_source": "deepep_moe",
            "moe_dtype": "bfloat16",
            "distribution": "uniform",
            "inference_phase": phase,
            "num_slots": 64,
            "inter_size": 1408,
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "num_tokens": tokens,
            "latency": 0.5 * tokens / 128.0,  # ms
            **shape,
        }
        for phase in ("context", "generation")
        for tokens in (128, 4096)
    ]
    _write_parquet(version_dir / "moe_ep_perf.parquet", ep_rows)

    db = get_database("toy_sys", "vllm", "1.0", systems_paths=str(systems_root), allow_missing_data=True)
    assert db is not None
    return db


class TestVllmG2Seed:
    def test_builder_emits_deepep_ops_and_queries_succeed(self, vllm_toy_db):
        cfg = config.ModelConfig(
            tp_size=1,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=8,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        cfg.moe_comm_backend = {"context": "deepep_ht", "generation": "deepep_ll"}
        shape = MoEBlockShape(
            hidden_size=4096, moe_inter_size=1408, topk=4, num_experts=64, num_shared_experts=0, num_moe_layers=10
        )
        for phase in ("context", "generation"):
            built = build_moe_block_ops(
                phase,
                shape,
                cfg,
                cfg.moe_quant_mode,
                "uniform",
                scale_factor=10,
                backend_name="vllm",
                inference_phase=phase,
            )
            # Generic family: the router stays (G2 — no DeepSeek variant fires).
            assert _names(built) == [
                f"{phase}_router_gemm",
                f"{phase}_moe_dispatch",
                f"{phase}_moe",
                f"{phase}_moe_combine",
            ]
            dispatch, moe, combine = built[1:]
            expected_backend = cfg.moe_comm_backend[phase]
            assert dispatch._comm_backend == combine._comm_backend == expected_backend
            # Exact arithmetic against the hand-built rows at x=64 (an exact
            # collected point): a2a leaves are base_us*factor us -> ms, x10
            # scale; EPMoE globalizes tokens by dp (64*8=512), and the toy
            # token curve is linear so the lerp between 128 and 4096 is exact:
            # 0.5 ms * 512/128 = 2.0 ms, x10 scale.
            base_us = 100.0 if phase == "context" else 50.0
            assert _lat(dispatch, vllm_toy_db, 64) == pytest.approx(base_us / 1000.0 * 10, rel=1e-9)
            assert _lat(combine, vllm_toy_db, 64) == pytest.approx(2 * base_us / 1000.0 * 10, rel=1e-9)
            assert _lat(moe, vllm_toy_db, 64) == pytest.approx(2.0 * 10, rel=1e-9)


class TestDeepEPAttentionTpTokenScaling:
    """attention_tp_size divides deepep A2A tokens in CONTEXT only.

    Every legacy generation dispatch site uses the default token divisor 1
    (deepseek.py:1321-1340, deepseek_v32 generation site, moe.py generation
    site); ``scale_num_tokens=tp_size`` appears at the CONTEXT sites only
    (deepseek.py:1206-1227, moe.py context site). The toy tables are exactly
    linear in tokens, so expectations are hand-computable (see vllm_toy_db).
    """

    def _build_tp2(self, phase):
        cfg = config.ModelConfig(
            tp_size=2,
            moe_tp_size=1,
            moe_ep_size=8,
            attention_dp_size=4,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            moe_quant_mode=common.MoEQuantMode.bfloat16,
        )
        cfg.moe_comm_backend = {"context": "deepep_ht", "generation": "deepep_ll"}
        shape = MoEBlockShape(
            hidden_size=4096, moe_inter_size=1408, topk=4, num_experts=64, num_shared_experts=0, num_moe_layers=10
        )
        built = build_moe_block_ops(
            phase,
            shape,
            cfg,
            cfg.moe_quant_mode,
            "uniform",
            scale_factor=10,
            backend_name="vllm",
            inference_phase=phase,
        )
        router, dispatch, _moe, combine = built
        assert router._name == f"{phase}_router_gemm"
        return dispatch, combine

    def test_context_divides_tokens_by_attention_tp(self, vllm_toy_db):
        dispatch, combine = self._build_tp2("context")
        assert dispatch._attention_tp_size == combine._attention_tp_size == 2
        # x=128 -> 128 // 2 = 64, the exact deepep_ht collected point:
        # dispatch 100 us, combine 200 us; ms x10 scale.
        assert _lat(dispatch, vllm_toy_db, 128) == pytest.approx(100.0 / 1000.0 * 10, rel=1e-9)
        assert _lat(combine, vllm_toy_db, 128) == pytest.approx(200.0 / 1000.0 * 10, rel=1e-9)

    def test_generation_does_not_divide_tokens(self, vllm_toy_db):
        dispatch, combine = self._build_tp2("generation")
        assert dispatch._attention_tp_size == combine._attention_tp_size == 1
        # x=128 stays undivided; the linear deepep_ll curve lerps exactly:
        # dispatch 50 us * 128/64 = 100 us, combine 200 us; ms x10 scale.
        assert _lat(dispatch, vllm_toy_db, 128) == pytest.approx(100.0 / 1000.0 * 10, rel=1e-9)
        assert _lat(combine, vllm_toy_db, 128) == pytest.approx(200.0 / 1000.0 * 10, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
