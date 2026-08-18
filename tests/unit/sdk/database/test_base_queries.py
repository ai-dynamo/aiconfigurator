# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Basic PerfDatabase construction sanity.

The per-call query behaviour this file used to pin on the stub/comprehensive
fixtures (query_gemm exact hits, empirical util calibration, cross-profile
borrowing, interpolation faithfulness, query_trtllm_alltoall normalization
and its not-enabled zero, custom-allreduce/nccl/p2p mode math, the HYBRID
fallback) retired to the compiled engine with #1357 PR-5 —
query_trtllm_alltoall is a tombstone, the rest are engine-routed shims whose
values are anchored by
the frozen parity goldens.
"""

import pytest

pytestmark = pytest.mark.unit


def test_system_spec_was_loaded_correctly(stub_perf_db):
    """
    Sanity check: PerfDatabase.system_spec should be exactly what our patched yaml.load returned.
    """
    spec = stub_perf_db.system_spec
    assert isinstance(spec, dict)
    assert spec["gpu"]["bfloat16_tc_flops"] == 1_000.0
    assert spec["node"]["inter_node_bw"] == 100.0


# ---------------------------------------------------------------------------
# Shim phase inference (base Operation._engine_query_is_context)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expected"),
    [("context", True), ("prefill", True), ("generation", False), ("decode", False), ("verify", False)],
)
def test_module_shape_phase_marker_inference(phase, expected):
    """The "module" shim shape must map BOTH phase vocabularies: the mamba/gdn
    kernels store context/generation (KDA adds verify — speculative decode,
    generation-like), FPMForwardOp stores prefill/decode (``fpm_forward._PHASES``)."""
    from aiconfigurator_core.sdk.operations.base import PythonOperation

    op = PythonOperation("probe", 1.0)
    op._phase = phase
    assert op._engine_query_is_context({}) is expected


def test_module_shape_phase_inference_hint_and_error():
    from aiconfigurator_core.sdk.operations.base import PythonOperation

    op = PythonOperation("probe", 1.0)
    # explicit hint wins even with no marker
    assert op._engine_query_is_context({"is_context": False}) is False
    # no marker, no hint -> loud error naming the escape hatch
    with pytest.raises(ValueError, match="is_context"):
        op._engine_query_is_context({})


def test_fpm_forward_phase_tokens_stay_mapped():
    """Deliberate-edit tripwire: if ``fpm_forward._PHASES`` ever grows a token
    the base inference does not recognize, fail here instead of at query time."""
    from aiconfigurator_core.sdk.operations import fpm_forward
    from aiconfigurator_core.sdk.operations.base import PythonOperation

    for phase in fpm_forward._PHASES:
        op = PythonOperation("probe", 1.0)
        op._phase = phase
        assert op._engine_query_is_context({}) in (True, False)


def test_get_weights_survives_ops_without_an_opspec_variant():
    """The ``moe_backend='deepep_moe'`` MoEDispatch (still built by qwen35)
    constructs as the RetiredDeepEp tombstone flavor — spec assembly and
    query refuse it. ``get_weights`` must stay 0.0 (the Rust
    ``Op::MoeDispatch`` weight arm is flavor-independent) instead of
    crashing memory estimation."""
    from aiconfigurator_core.sdk.operations.moe import MoEDispatch

    op = MoEDispatch("d", 1.0, 7168, 8, 256, 1, 16, 1, False, backend="sglang", moe_backend="deepep_moe")
    assert op.get_weights() == 0.0


def test_dsa_partial_projection_map_normalizes_and_survives_the_weight_route():
    """A PARTIAL attn_projection_quant_modes mapping (checkpoint fact for one
    group only) must normalize to all four groups at construction — the opspec
    emission and the Rust DsaProjectionQuants deserialization require every
    field."""
    from aiconfigurator_core.sdk import common
    from aiconfigurator_core.sdk.operations.dsa import ContextDSAModule

    op = ContextDSAModule(
        "ctx_dsa",
        1.0,
        64,
        common.KVCacheQuantMode.fp8,
        common.FMHAQuantMode.bfloat16,
        common.GEMMQuantMode.nvfp4,
        attn_projection_quant_modes={"o": common.GEMMQuantMode.bfloat16},
    )
    assert set(op._attn_projection_quant_modes) == {"q", "kv", "o", "indexer"}
    assert op._attn_projection_quant_modes["o"] is common.GEMMQuantMode.bfloat16
    for defaulted in ("q", "kv", "indexer"):
        assert op._attn_projection_quant_modes[defaulted] is common.GEMMQuantMode.nvfp4
    assert op.get_weights() > 0.0

    # An unknown group name must fail loudly, not be silently dropped.
    with pytest.raises(ValueError, match="unknown DSA projection group"):
        ContextDSAModule(
            "ctx_dsa",
            1.0,
            64,
            common.KVCacheQuantMode.fp8,
            common.FMHAQuantMode.bfloat16,
            common.GEMMQuantMode.nvfp4,
            attn_projection_quant_modes={"o_proj": common.GEMMQuantMode.bfloat16},
        )
