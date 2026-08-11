# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lane-preserving behaviour of the context and generation attention
loaders, and for the lane-aware query paths that consume them.

AIC-1715: both loaders must keep ``kernel_source`` as the outermost dict level
instead of collapsing all sources into a single first-writer-wins table, and the
query paths must walk the resolved lane precedence order (override → framework
default → remaining known lanes → ``"default"``), with later lanes acting as
donor gap-fill for shapes the head lane never collected.
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


def test_context_loader_empty_kernel_source_lands_in_default_lane(tmp_path):
    """An EMPTY kernel_source cell (column present, value blank) must also land
    in lane 'default' — not in a lane literally named ''."""
    from aiconfigurator.sdk.perf_database import load_context_attention_data

    csv = tmp_path / "ctx_empty_ks.csv"
    csv.write_text(_CTX_HEADERS_WITH_KS + _CTX_ROW_KS.format(ks="", isl=ISL_A, lat="6.0") + "\n")

    data = load_context_attention_data(str(csv))

    assert "" not in data, "an empty kernel_source must not create an empty-string lane"
    assert "default" in data
    assert data["default"][QM][KCD][KV_N][HEAD][WIN][N][ISL_A][B]["latency"] == pytest.approx(6.0)


def test_generation_loader_empty_kernel_source_lands_in_default_lane(tmp_path):
    """Generation twin of the empty-kernel_source case."""
    from aiconfigurator.sdk.perf_database import load_generation_attention_data

    csv = tmp_path / "gen_empty_ks.csv"
    csv.write_text(_GEN_HEADERS_WITH_KS + _GEN_ROW_KS.format(ks="", isl=ISL_A, lat="6.0") + "\n")

    data = load_generation_attention_data(str(csv))

    assert "" not in data, "an empty kernel_source must not create an empty-string lane"
    assert "default" in data
    assert data["default"][KCD][KV_N][HEAD][WIN][N][B][S_A]["latency"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Lane-aware query paths (AIC-1715 Task 3)
# ---------------------------------------------------------------------------

_LANE_DEFAULTS_YAML = """\
# Test fixture: minimal copy of attention_lane_defaults.yaml
sglang:
  "0.5.14":
    90: fa3
    100: triton
    103: triton
    120: flashinfer
"""

# The two lanes carry the SAME grid with latencies differing by a constant
# factor, so any homogeneous interpolation reproduces the ratio exactly.
_SLOW_LATENCY = 1.0
_FAST_LATENCY = 0.25
_LANE_RATIO = _FAST_LATENCY / _SLOW_LATENCY

_GRID_N = (8, 16)
_GRID_S = (32, 64, 128)
_GRID_B = (1, 2)


@pytest.fixture
def lane_systems_root(tmp_path):
    """systems_root holding the framework-default lane map (sglang 0.5.14)."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "attention_lane_defaults.yaml").write_text(_LANE_DEFAULTS_YAML, encoding="utf-8")
    return str(root)


class _LoadedLanes(dict):
    """Minimal stand-in for ``LoadedOpData``: a mapping that is always loaded."""

    loaded = True

    def raise_if_not_loaded(self):
        return None


def _leaf(latency):
    return {"latency": latency, "power": 0.0, "energy": 0.0}


def _ctx_lane(latency, head_size=HEAD):
    """[fmha][kv][kv_n][head][window][n][s][b] slice for one lane."""
    return {
        QM: {
            KCD: {
                KV_N: {
                    head_size: {WIN: {n: {s: {b: _leaf(latency) for b in _GRID_B} for s in _GRID_S} for n in _GRID_N}}
                }
            }
        }
    }


def _gen_lane(latency, head_size=HEAD):
    """[kv][kv_n][head][window][n][b][s] slice for one lane."""
    return {
        KCD: {
            KV_N: {head_size: {WIN: {n: {b: {s: _leaf(latency) for s in _GRID_S} for b in _GRID_B} for n in _GRID_N}}}
        }
    }


class _StubDatabase:
    """Minimal PerfDatabase stand-in exercising the real lane-walk query bodies.

    Only the attributes the attention query paths touch are provided; the two
    delegate methods mirror ``PerfDatabase.query_*_attention`` (including the
    ``lane_order`` pass-through) so op-level ``query()`` calls take the same
    route they do in production.
    """

    def __init__(self, systems_root, context_lanes=None, generation_lanes=None, *, sm_version=103):
        from aiconfigurator.sdk import common

        self.system = "test_system"
        self.backend = "sglang"
        self.version = "0.5.14"
        self.systems_root = systems_root
        self.enable_shared_layer = False
        self.system_spec = {
            "data_dir": "data",
            "gpu": {
                "sm_version": sm_version,
                "mem_bw": 1.0e6,
                "bfloat16_tc_flops": 1.0e6,
                "fp8_tc_flops": 2.0e6,
            },
        }
        self._default_database_mode = common.DatabaseMode.SILICON
        self.transfer_policy = frozenset()
        self._context_attention_data = _LoadedLanes(context_lanes or {})
        self._generation_attention_data = _LoadedLanes(generation_lanes or {})
        self._raw_generation_attention_data = self._generation_attention_data

    def query_context_attention(self, *args, **kwargs):
        from aiconfigurator.sdk.operations.attention import ContextAttention

        return ContextAttention._query_context_attention_table(self, *args, **kwargs)

    def query_generation_attention(self, *args, **kwargs):
        from aiconfigurator.sdk.operations.attention import GenerationAttention

        return GenerationAttention._query_generation_attention_table(self, *args, **kwargs)

    def query_mem_op(self, num_bytes):
        return 0.0

    @staticmethod
    def _interp_pr(latency, energy=0.0):
        from aiconfigurator.sdk.performance_result import PerformanceResult

        return PerformanceResult(latency, energy=energy, source="silicon")

    def _query_silicon_or_hybrid(self, get_silicon, get_empirical, database_mode, error_msg):
        return get_silicon()


@pytest.fixture
def no_op_load_data(monkeypatch):
    """Neutralize the CSV load: the stub database pre-binds its own tables."""
    from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

    monkeypatch.setattr(ContextAttention, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(GenerationAttention, "load_data", classmethod(lambda cls, database: None))


def _ctx_query(db, *, head_size=HEAD, lane_order=None):
    from aiconfigurator.sdk.operations.attention import ContextAttention

    return ContextAttention._query_context_attention_table(
        db, B, 64, 0, N, N, KCD, QM, None, WIN, head_size, lane_order
    )


def _gen_query(db, *, head_size=HEAD, lane_order=None):
    from aiconfigurator.sdk.operations.attention import GenerationAttention

    return GenerationAttention._query_generation_attention_table(db, B, 64, N, N, KCD, None, WIN, head_size, lane_order)


def test_context_query_default_order_serves_the_framework_default_lane(lane_systems_root, no_op_load_data):
    """sglang 0.5.14 @ sm103 defaults to the triton lane: with both lanes present
    and disagreeing at the same key, the triton value is what the query returns."""
    db = _StubDatabase(
        lane_systems_root,
        context_lanes={"triton": _ctx_lane(_SLOW_LATENCY), "trtllm_mha": _ctx_lane(_FAST_LATENCY)},
    )

    default = _ctx_query(db)
    triton_only = _ctx_query(db, lane_order=("triton",))
    fast_only = _ctx_query(db, lane_order=("trtllm_mha",))

    assert float(default) == pytest.approx(float(triton_only))
    assert float(fast_only) == pytest.approx(float(default) * _LANE_RATIO)
    assert float(default) == pytest.approx(_SLOW_LATENCY)


def test_generation_query_default_order_serves_the_framework_default_lane(lane_systems_root, no_op_load_data):
    """Generation twin: the triton lane serves under the default order."""
    db = _StubDatabase(
        lane_systems_root,
        generation_lanes={"triton": _gen_lane(_SLOW_LATENCY), "trtllm_mha": _gen_lane(_FAST_LATENCY)},
    )

    default = _gen_query(db)
    triton_only = _gen_query(db, lane_order=("triton",))
    fast_only = _gen_query(db, lane_order=("trtllm_mha",))

    assert float(default) == pytest.approx(float(triton_only))
    assert float(fast_only) == pytest.approx(float(default) * _LANE_RATIO)


def test_context_op_override_selects_the_requested_lane(lane_systems_root, no_op_load_data):
    """``attention_backend="trtllm_mha"`` on the op makes the fast lane serve the
    query end-to-end (op.query -> database delegate -> lane walk)."""
    from aiconfigurator.sdk.operations.attention import ContextAttention

    db = _StubDatabase(
        lane_systems_root,
        context_lanes={"triton": _ctx_lane(_SLOW_LATENCY), "trtllm_mha": _ctx_lane(_FAST_LATENCY)},
    )

    def _op(attention_backend):
        return ContextAttention(
            "context_attention",
            1.0,
            N,
            N,
            KCD,
            QM,
            window_size=WIN,
            head_size=HEAD,
            attention_backend=attention_backend,
        )

    baseline = _op(None).query(db, batch_size=B, s=64, prefix=0)
    overridden = _op("trtllm_mha").query(db, batch_size=B, s=64, prefix=0)

    assert float(baseline) == pytest.approx(_SLOW_LATENCY)
    assert float(overridden) == pytest.approx(_FAST_LATENCY)


def test_generation_op_override_selects_the_requested_lane(lane_systems_root, no_op_load_data):
    """Generation twin of the override test."""
    from aiconfigurator.sdk.operations.attention import GenerationAttention

    db = _StubDatabase(
        lane_systems_root,
        generation_lanes={"triton": _gen_lane(_SLOW_LATENCY), "trtllm_mha": _gen_lane(_FAST_LATENCY)},
    )

    def _op(attention_backend):
        return GenerationAttention(
            "generation_attention",
            1.0,
            N,
            N,
            KCD,
            window_size=WIN,
            head_size=HEAD,
            attention_backend=attention_backend,
        )

    baseline = float(_op(None).query(db, batch_size=B, s=64, beam_width=1))
    overridden = float(_op("trtllm_mha").query(db, batch_size=B, s=64, beam_width=1))

    assert overridden == pytest.approx(baseline * _LANE_RATIO)


def test_context_donor_lane_fills_a_shape_the_head_lane_never_collected(lane_systems_root, no_op_load_data, caplog):
    """head_size=64 exists only in the trtllm_mha lane: the default (triton-headed)
    order must still resolve it from the donor lane, and say so at DEBUG."""
    db = _StubDatabase(
        lane_systems_root,
        context_lanes={
            "triton": _ctx_lane(_SLOW_LATENCY, head_size=128),
            "trtllm_mha": _ctx_lane(_FAST_LATENCY, head_size=64),
        },
    )

    with caplog.at_level(logging.DEBUG, logger="aiconfigurator_core.sdk.operations.attention"):
        donor = _ctx_query(db, head_size=64)

    assert float(donor) == pytest.approx(_FAST_LATENCY)
    assert float(_ctx_query(db, head_size=128)) == pytest.approx(_SLOW_LATENCY)
    assert any("trtllm_mha" in r.message for r in caplog.records), "donor lane must be logged at DEBUG"


def test_generation_donor_lane_fills_a_shape_the_head_lane_never_collected(lane_systems_root, no_op_load_data):
    """Generation twin of the donor gap-fill test."""
    db = _StubDatabase(
        lane_systems_root,
        generation_lanes={
            "triton": _gen_lane(_SLOW_LATENCY, head_size=128),
            "trtllm_mha": _gen_lane(_FAST_LATENCY, head_size=64),
        },
    )

    donor = float(_gen_query(db, head_size=64))
    head = float(_gen_query(db, head_size=128))

    assert donor == pytest.approx(head * _LANE_RATIO)


def test_no_lane_carries_the_slice_raises_the_usual_coverage_error(lane_systems_root, no_op_load_data):
    """When NO lane has the slice the query fails exactly as before, so the
    SOL/empirical fallbacks downstream are untouched."""
    from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError

    db = _StubDatabase(lane_systems_root, context_lanes={"triton": _ctx_lane(_SLOW_LATENCY, head_size=128)})

    with pytest.raises(PerfDataNotAvailableError):
        _ctx_query(db, head_size=256)


def test_correct_sol_clamps_every_lane(lane_systems_root, no_op_load_data):
    """``_correct_sol`` must clamp sub-SOL latencies in EVERY lane, not just the first."""
    from aiconfigurator.sdk import common
    from aiconfigurator.sdk.operations.attention import GenerationAttention

    db = _StubDatabase(
        lane_systems_root,
        generation_lanes={"triton": _gen_lane(_SLOW_LATENCY), "trtllm_mha": _gen_lane(_FAST_LATENCY)},
    )
    # Second lane stores bare floats (the loader's legacy leaf form) to cover both
    # leaf shapes in one pass.
    fast = db._generation_attention_data["trtllm_mha"]
    for n in _GRID_N:
        for b in _GRID_B:
            for s in _GRID_S:
                fast[KCD][KV_N][HEAD][WIN][n][b][s] = _FAST_LATENCY

    sol = float(
        GenerationAttention._query_generation_attention_table(
            db,
            _GRID_B[0],
            _GRID_S[0],
            _GRID_N[0],
            _GRID_N[0],
            KCD,
            database_mode=common.DatabaseMode.SOL,
            window_size=WIN,
            head_size=HEAD,
        )
    )
    assert sol > _SLOW_LATENCY, "fixture must be sub-SOL for the clamp to be observable"

    GenerationAttention._correct_sol(db, db._generation_attention_data)

    slow_leaf = db._generation_attention_data["triton"][KCD][KV_N][HEAD][WIN][_GRID_N[0]][_GRID_B[0]][_GRID_S[0]]
    fast_leaf = db._generation_attention_data["trtllm_mha"][KCD][KV_N][HEAD][WIN][_GRID_N[0]][_GRID_B[0]][_GRID_S[0]]
    assert slow_leaf["latency"] == pytest.approx(sol), "first lane must be clamped to SOL"
    assert float(fast_leaf) == pytest.approx(sol), "donor lane must be clamped to SOL as well"


def test_opspec_serializes_the_resolved_lane_order(lane_systems_root, no_op_load_data):
    """Both attention op specs carry the RESOLVED lane order so Rust replays it verbatim."""
    from aiconfigurator.sdk import engine
    from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

    db = _StubDatabase(lane_systems_root)

    ctx = ContextAttention("context_attention", 1.0, N, N, KCD, QM, attention_backend="trtllm_mha")
    gen = GenerationAttention("generation_attention", 1.0, N, N, KCD)

    ctx_spec = engine._to_opspec(ctx, backend="sglang", architecture="Qwen35", database=db)["ContextAttention"]
    gen_spec = engine._to_opspec(gen, backend="sglang", architecture="Qwen35", database=db)["GenerationAttention"]

    assert ctx_spec["lane_order"][0] == "trtllm_mha", "override heads the serialized order"
    assert gen_spec["lane_order"][0] == "triton", "framework default heads the serialized order"
    assert ctx_spec["lane_order"][-1] == "default"
    assert gen_spec["lane_order"][-1] == "default"


def test_opspec_lane_order_without_a_database_falls_back_to_default():
    """Spec build with no database handle must still emit a valid (default-only) order."""
    from aiconfigurator.sdk import engine
    from aiconfigurator.sdk.operations.attention import ContextAttention, GenerationAttention

    ctx = ContextAttention("context_attention", 1.0, N, N, KCD, QM)
    gen = GenerationAttention("generation_attention", 1.0, N, N, KCD)

    ctx_spec = engine._to_opspec(ctx, backend="sglang", architecture="Qwen35", database=None)["ContextAttention"]
    gen_spec = engine._to_opspec(gen, backend="sglang", architecture="Qwen35", database=None)["GenerationAttention"]

    assert ctx_spec["lane_order"] == ["default"]
    assert gen_spec["lane_order"] == ["default"]


def test_engine_spec_schema_version_is_eight():
    """The lane_order field is an always-serialized positional payload change."""
    from aiconfigurator.sdk import engine

    assert engine.ENGINE_SPEC_SCHEMA_VERSION == 8


def test_lanes_outside_the_known_vocabulary_stay_reachable(lane_systems_root, no_op_load_data):
    """Collected ``kernel_source`` labels are richer than the resolver's lane
    vocabulary — trtllm ships ``torch_flow*``, vllm ships ``vllm_*``, and neither
    has a ``"default"`` lane. Those rows must still serve queries (they are the
    ONLY rows there), after every named lane in the resolved order."""
    from aiconfigurator.sdk.operations.attention import lane_walk_order

    db = _StubDatabase(
        lane_systems_root,
        context_lanes={"torch_flow": _ctx_lane(_SLOW_LATENCY), "torch_flow_flashinfer": _ctx_lane(_FAST_LATENCY)},
    )

    assert float(_ctx_query(db)) == pytest.approx(_SLOW_LATENCY), "first table lane serves when no named lane exists"

    order = lane_walk_order(db._context_attention_data, ("triton", "default"))
    assert order == ("triton", "default", "torch_flow", "torch_flow_flashinfer")
    assert lane_walk_order({}, ("triton", "default")) == ("triton", "default")


def test_opspec_lane_order_carries_the_table_leftovers(lane_systems_root, no_op_load_data):
    """The serialized order is the COMPLETE walk (resolved + table leftovers),
    so the Rust twin replays it without knowing the lane vocabulary."""
    from aiconfigurator.sdk import engine
    from aiconfigurator.sdk.operations.attention import ContextAttention

    db = _StubDatabase(lane_systems_root, context_lanes={"torch_flow": _ctx_lane(_SLOW_LATENCY)})
    op = ContextAttention("context_attention", 1.0, N, N, KCD, QM)

    spec = engine._to_opspec(op, backend="sglang", architecture="Qwen35", database=db)["ContextAttention"]

    assert spec["lane_order"][0] == "triton"
    assert spec["lane_order"][-1] == "torch_flow", "table leftovers ride at the end of the serialized order"
