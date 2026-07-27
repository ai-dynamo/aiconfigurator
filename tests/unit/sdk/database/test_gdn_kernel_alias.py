# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import LoadedOpData

pytestmark = pytest.mark.unit

MODEL_KEY = (2048, 16, 128, 32, 128, 4)
MODEL_SHAPE = {
    "d_model": 2048,
    "num_k_heads": 16,
    "head_k_dim": 128,
    "num_v_heads": 32,
    "head_v_dim": 128,
    "d_conv": 4,
}


def _context_table(latency):
    return {1: {128: {"latency": latency, "power": 0.0, "energy": 0.0}}}


def _generation_table(latency):
    return {1: {"latency": latency, "power": 0.0, "energy": 0.0}}


@pytest.fixture
def vllm_gdn_db(stub_perf_db):
    stub_perf_db.backend = "vllm"
    stub_perf_db.version = "0.24.0"
    return stub_perf_db


@pytest.mark.parametrize(
    "physical_source",
    (
        "chunk_gated_delta_rule_flashinfer",
        "chunk_gated_delta_rule_triton",
        "chunk_gated_delta_rule_cutedsl",
    ),
)
def test_query_gdn_vllm024_uses_exact_context_physical_alias(vllm_gdn_db, physical_source):
    vllm_gdn_db._gdn_data = LoadedOpData(
        {physical_source: {"context": {MODEL_KEY: _context_table(7.25)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(7.25)
    assert result.source == "silicon"


def test_query_gdn_vllm024_uses_exact_generation_physical_alias(vllm_gdn_db):
    vllm_gdn_db._gdn_data = LoadedOpData(
        {"fused_recurrent_gated_delta_rule_packed_decode": {"generation": {MODEL_KEY: _generation_table(3.5)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="generation",
        kernel_source="fused_sigmoid_gating_delta_rule_update",
        batch_size=1,
        seq_len=None,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(3.5)
    assert result.source == "silicon"


def test_query_gdn_exact_logical_shape_wins_over_physical_aliases(vllm_gdn_db):
    vllm_gdn_db._gdn_data = LoadedOpData(
        {
            "chunk_gated_delta_rule": {"context": {MODEL_KEY: _context_table(2.0)}},
            "chunk_gated_delta_rule_flashinfer": {"context": {MODEL_KEY: _context_table(7.0)}},
            "chunk_gated_delta_rule_triton": {"context": {MODEL_KEY: _context_table(8.0)}},
        },
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(2.0)
    assert result.source == "silicon"


def test_query_gdn_multiple_exact_physical_aliases_fail_closed(vllm_gdn_db):
    vllm_gdn_db._gdn_data = LoadedOpData(
        {
            "chunk_gated_delta_rule_flashinfer": {"context": {MODEL_KEY: _context_table(7.0)}},
            "chunk_gated_delta_rule_triton": {"context": {MODEL_KEY: _context_table(8.0)}},
        },
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert result.source == "sol"


def test_query_gdn_does_not_use_nearest_shape_from_physical_alias(vllm_gdn_db):
    other_shape = (2048, 16, 128, 31, 128, 4)
    vllm_gdn_db._gdn_data = LoadedOpData(
        {"chunk_gated_delta_rule_flashinfer": {"context": {other_shape: _context_table(7.0)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert result.source == "sol"


def test_query_gdn_preserves_nearest_shape_fallback_within_logical_source(vllm_gdn_db):
    farther_logical_shape = (2048, 16, 128, 8, 128, 4)
    nearer_logical_shape = (2048, 16, 128, 24, 128, 4)
    nearest_alias_shape = (2048, 16, 128, 31, 128, 4)
    vllm_gdn_db._gdn_data = LoadedOpData(
        {
            "chunk_gated_delta_rule": {
                "context": {
                    farther_logical_shape: _context_table(2.0),
                    nearer_logical_shape: _context_table(4.0),
                }
            },
            "chunk_gated_delta_rule_flashinfer": {"context": {nearest_alias_shape: _context_table(9.0)}},
        },
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = vllm_gdn_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert float(result) == pytest.approx(4.0)
    assert result.source == "silicon"


@pytest.mark.parametrize(("backend", "version"), (("vllm", "0.23.0"), ("sglang", "0.24.0")))
def test_query_gdn_physical_aliases_are_vllm024_only(stub_perf_db, backend, version):
    stub_perf_db.backend = backend
    stub_perf_db.version = version
    stub_perf_db._gdn_data = LoadedOpData(
        {"chunk_gated_delta_rule_flashinfer": {"context": {MODEL_KEY: _context_table(7.0)}}},
        common.PerfDataFilename.gdn,
        "gdn_perf.txt",
    )

    result = stub_perf_db.query_gdn(
        phase="context",
        kernel_source="chunk_gated_delta_rule",
        batch_size=1,
        seq_len=128,
        **MODEL_SHAPE,
    )

    assert result.source == "sol"
