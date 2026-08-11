# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lane-preserving behaviour of the context and generation attention loaders.

AIC-1715: both loaders must keep ``kernel_source`` as the outermost dict level
instead of collapsing all sources into a single first-writer-wins table.
"""

import logging

import pytest

from aiconfigurator.sdk.common import FMHAQuantMode, KVCacheQuantMode

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_CTX_HEADERS_WITH_KS = (
    "framework,version,device,op_name,kernel_source,"
    "batch_size,isl,num_heads,num_key_value_heads,head_dim,"
    "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
)
_CTX_HEADERS_NO_KS = (
    "framework,version,device,op_name,"
    "batch_size,isl,num_heads,num_key_value_heads,head_dim,"
    "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
)
_GEN_HEADERS_WITH_KS = (
    "framework,version,device,op_name,kernel_source,"
    "batch_size,isl,num_heads,num_key_value_heads,head_dim,"
    "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
)
_GEN_HEADERS_NO_KS = (
    "framework,version,device,op_name,"
    "batch_size,isl,num_heads,num_key_value_heads,head_dim,"
    "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
)

_CTX_ROW_KS = "trt,2.0,hwX,context_attention,{ks},1,{isl},8,8,128,1,bfloat16,bfloat16,1,{lat}"
_CTX_ROW_NO_KS = "trt,2.0,hwX,context_attention,1,{isl},8,8,128,1,bfloat16,bfloat16,1,{lat}"
_GEN_ROW_KS = "trt,2.0,hwX,generation_attention,{ks},1,{isl},8,8,128,1,bfloat16,bfloat16,1,{lat}"
_GEN_ROW_NO_KS = "trt,2.0,hwX,generation_attention,1,{isl},8,8,128,1,bfloat16,bfloat16,1,{lat}"

# Fixed physical-key components used across tests
QM = FMHAQuantMode.bfloat16
KCD = KVCacheQuantMode.bfloat16
KV_N = 0  # n == kv_n → normalised to 0 by loader
HEAD = 128
WIN = 0
N = 8
B = 1
ISL_A = 64  # "point A" shared between both lanes
ISL_B = 128  # "point B" shared between both lanes
# Generation stores s = isl + step (step == 1 in the row template)
S_A = ISL_A + 1  # 65
S_B = ISL_B + 1  # 129


# ---------------------------------------------------------------------------
# Context-attention loader
# ---------------------------------------------------------------------------


def test_context_loader_two_lanes_retain_separate_values(tmp_path):
    """4-row fixture: two lanes x two points, with conflicting latencies at
    the shared physical key ISL_A.  Both lanes must be retained with their
    own values; neither overwrites the other."""
    from aiconfigurator.sdk.perf_database import load_context_attention_data

    csv = tmp_path / "ctx_two_lanes.csv"
    rows = [
        _CTX_ROW_KS.format(ks="triton", isl=ISL_A, lat="1.0"),
        _CTX_ROW_KS.format(ks="trtllm_mha", isl=ISL_A, lat="2.0"),  # same physical key, different lane
        _CTX_ROW_KS.format(ks="triton", isl=ISL_B, lat="3.0"),
        _CTX_ROW_KS.format(ks="trtllm_mha", isl=ISL_B, lat="4.0"),
    ]
    csv.write_text(_CTX_HEADERS_WITH_KS + "\n".join(rows) + "\n")

    data = load_context_attention_data(str(csv))

    assert "triton" in data, "lane 'triton' must be the outermost key"
    assert "trtllm_mha" in data, "lane 'trtllm_mha' must be the outermost key"

    triton_a = data["triton"][QM][KCD][KV_N][HEAD][WIN][N][ISL_A][B]["latency"]
    trtllm_a = data["trtllm_mha"][QM][KCD][KV_N][HEAD][WIN][N][ISL_A][B]["latency"]

    assert triton_a == pytest.approx(1.0), "triton lane must preserve its latency at point A"
    assert trtllm_a == pytest.approx(2.0), "trtllm_mha lane must preserve its latency at point A"
    assert triton_a != trtllm_a, "the two lanes must differ at the shared physical key"

    assert data["triton"][QM][KCD][KV_N][HEAD][WIN][N][ISL_B][B]["latency"] == pytest.approx(3.0)
    assert data["trtllm_mha"][QM][KCD][KV_N][HEAD][WIN][N][ISL_B][B]["latency"] == pytest.approx(4.0)


def test_context_loader_missing_kernel_source_lands_in_default_lane(tmp_path):
    """A row whose CSV has no kernel_source column must land in lane 'default'."""
    from aiconfigurator.sdk.perf_database import load_context_attention_data

    csv = tmp_path / "ctx_no_ks.csv"
    csv.write_text(_CTX_HEADERS_NO_KS + _CTX_ROW_NO_KS.format(isl=ISL_A, lat="5.0") + "\n")

    data = load_context_attention_data(str(csv))

    assert "default" in data, "missing kernel_source must produce lane 'default'"
    assert data["default"][QM][KCD][KV_N][HEAD][WIN][N][ISL_A][B]["latency"] == pytest.approx(5.0)


def test_context_loader_within_lane_duplicate_first_writer_wins(tmp_path, caplog):
    """Two rows in the SAME lane at the same physical key: first writer wins
    and the duplicate is logged at DEBUG."""
    from aiconfigurator.sdk.perf_database import load_context_attention_data

    csv = tmp_path / "ctx_intra_dup.csv"
    rows = [
        _CTX_ROW_KS.format(ks="triton", isl=ISL_A, lat="1.0"),  # first writer
        _CTX_ROW_KS.format(ks="triton", isl=ISL_A, lat="9.9"),  # duplicate in same lane
    ]
    csv.write_text(_CTX_HEADERS_WITH_KS + "\n".join(rows) + "\n")

    with caplog.at_level(logging.DEBUG):
        data = load_context_attention_data(str(csv))

    # First writer must win
    assert data["triton"][QM][KCD][KV_N][HEAD][WIN][N][ISL_A][B]["latency"] == pytest.approx(1.0)
    # Conflict must be logged at DEBUG
    conflict_records = [r for r in caplog.records if "value conflict" in r.message]
    assert len(conflict_records) >= 1
    assert all(r.levelno == logging.DEBUG for r in conflict_records)


# ---------------------------------------------------------------------------
# Generation-attention loader
# ---------------------------------------------------------------------------


def test_generation_loader_two_lanes_retain_separate_values(tmp_path):
    """Mirror of the context test for the generation-attention loader.

    The stored s key is ``isl + step`` (step == 1 in the row template).
    """
    from aiconfigurator.sdk.perf_database import load_generation_attention_data

    csv = tmp_path / "gen_two_lanes.csv"
    rows = [
        _GEN_ROW_KS.format(ks="triton", isl=ISL_A, lat="1.0"),
        _GEN_ROW_KS.format(ks="trtllm_mha", isl=ISL_A, lat="2.0"),  # same physical key, different lane
        _GEN_ROW_KS.format(ks="triton", isl=ISL_B, lat="3.0"),
        _GEN_ROW_KS.format(ks="trtllm_mha", isl=ISL_B, lat="4.0"),
    ]
    csv.write_text(_GEN_HEADERS_WITH_KS + "\n".join(rows) + "\n")

    data = load_generation_attention_data(str(csv))

    assert "triton" in data, "lane 'triton' must be the outermost key"
    assert "trtllm_mha" in data, "lane 'trtllm_mha' must be the outermost key"

    # stored s = isl + step
    triton_a = data["triton"][KCD][KV_N][HEAD][WIN][N][B][S_A]["latency"]
    trtllm_a = data["trtllm_mha"][KCD][KV_N][HEAD][WIN][N][B][S_A]["latency"]

    assert triton_a == pytest.approx(1.0)
    assert trtllm_a == pytest.approx(2.0)
    assert triton_a != trtllm_a

    assert data["triton"][KCD][KV_N][HEAD][WIN][N][B][S_B]["latency"] == pytest.approx(3.0)
    assert data["trtllm_mha"][KCD][KV_N][HEAD][WIN][N][B][S_B]["latency"] == pytest.approx(4.0)


def test_generation_loader_missing_kernel_source_lands_in_default_lane(tmp_path):
    """A row without a kernel_source column must land in lane 'default'."""
    from aiconfigurator.sdk.perf_database import load_generation_attention_data

    csv = tmp_path / "gen_no_ks.csv"
    csv.write_text(_GEN_HEADERS_NO_KS + _GEN_ROW_NO_KS.format(isl=ISL_A, lat="5.0") + "\n")

    data = load_generation_attention_data(str(csv))

    assert "default" in data, "missing kernel_source must produce lane 'default'"
    assert data["default"][KCD][KV_N][HEAD][WIN][N][B][S_A]["latency"] == pytest.approx(5.0)


def test_generation_loader_within_lane_duplicate_first_writer_wins(tmp_path, caplog):
    """Within one lane, a duplicate physical key: first writer wins and the
    duplicate is logged at DEBUG."""
    from aiconfigurator.sdk.perf_database import load_generation_attention_data

    csv = tmp_path / "gen_intra_dup.csv"
    rows = [
        _GEN_ROW_KS.format(ks="triton", isl=ISL_A, lat="1.0"),
        _GEN_ROW_KS.format(ks="triton", isl=ISL_A, lat="9.9"),
    ]
    csv.write_text(_GEN_HEADERS_WITH_KS + "\n".join(rows) + "\n")

    with caplog.at_level(logging.DEBUG):
        data = load_generation_attention_data(str(csv))

    assert data["triton"][KCD][KV_N][HEAD][WIN][N][B][S_A]["latency"] == pytest.approx(1.0)
    conflict_records = [r for r in caplog.records if "value conflict" in r.message]
    assert len(conflict_records) >= 1
    assert all(r.levelno == logging.DEBUG for r in conflict_records)
