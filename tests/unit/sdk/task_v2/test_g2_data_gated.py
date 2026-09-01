# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""G2 goal verification: data-gated enablement -- the executable
"collect data -> supported" contract (spec sections 4.4.3 / 4.5).

A MoE shape NO shipped checkpoint has (hidden 5120, topk 4, 64 experts;
verified against the bundled model_configs) becomes large-EP-explorable the
moment its ``moe_a2a_perf`` + ``moe_expert_compute_perf`` parquets exist for a system --
with ZERO source changes: no new model class, no new family registration, no
builder variant, no flag. On sglang AND vllm:

(a) the PR 1 coverage probes report the collected shape;
(b) Task resolution offers large-EP candidate tuples for exactly the covered
    EPs, resolving the right per-phase comm backends (context -> deepep_ht,
    generation -> deepep_ll);
(c) one candidate's op graph builds through ``get_model`` and every emitted
    large-EP op ``.query()``s the synthetic tables end to end with finite
    latencies -- while an uncovered tuple of the same task builds the fused
    graph with no large-EP op at all.

Fixtures fabricate only DATA (a synthetic systems root holding the two
tables, in the family-first layout of ``test_coverage_candidates.py``) and a
MODEL CONFIG (a ``config.json`` in a tmp dir, resolved through the shipped
local-path branch of ``_get_model_info`` into the existing ``MOE`` family);
``test_source_tree_is_untouched_fixtures_only`` pins that nothing was
registered into the package to make this pass.
"""

from __future__ import annotations

import json
import math
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import get_model, get_model_family
from aiconfigurator.sdk.operations.moe_comm import MoEAllToAll, MoEExpertCompute
from aiconfigurator.sdk.perf_database import databases_cache, get_database, set_systems_paths
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

# A shape no shipped model has (hidden 5120 + 64 experts has no collision in
# aic-core/src/aiconfigurator_core/model_configs): the contract must work for
# a model the source tree has never seen, not just for re-collected DeepSeek.
SYNTH_HIDDEN, SYNTH_INTER, SYNTH_TOPK, SYNTH_EXPERTS = 5120, 1536, 4, 64
SYNTH_LAYERS = 48

SYNTH_SYSTEM = "synth_g2"
SYNTH_VERSION = "9.9.9"
BACKENDS = ("sglang", "vllm")

# Collected EP ladder: (ep_size, node_num) on an 8-GPU/node system.
_PAIRS = ((8, 1), (16, 2))
_TOKEN_POINTS = (16, 4096)

# Fabricated HF config: Qwen3-MoE architecture (-> existing "MOE" family) with
# the synthetic geometry. bfloat16 checkpoint, so the task's inferred quant
# modes match the moe_dtype the synthetic compute rows carry.
_SYNTH_HF_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": SYNTH_HIDDEN,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "mlp_only_layers": [],
    "decoder_sparse_step": 1,
    "model_type": "qwen3_moe",
    "moe_intermediate_size": SYNTH_INTER,
    "num_attention_heads": 40,
    "num_experts": SYNTH_EXPERTS,
    "num_experts_per_tok": SYNTH_TOPK,
    "num_hidden_layers": SYNTH_LAYERS,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
    "vocab_size": 151936,
}


def _a2a_rows() -> list[dict]:
    """moe_a2a rows for the synthetic shape: deepep_ht (context) and deepep_ll
    (generation), dispatch+combine, both collected EPs. ``sms`` matches what
    the builder passes at query time: ``cfg.sms`` (default 20) rides deepep_ht
    only; every other backend queries sms=0."""
    rows = []
    for backend, sms in (("deepep_ht", 20), ("deepep_ll", 0)):
        for ep_size, node_num in _PAIRS:
            for phase in ("dispatch", "combine"):
                for num_tokens in _TOKEN_POINTS:
                    rows.append(
                        {
                            "comm_backend": backend,
                            "phase": phase,
                            "comm_dtype": "default",
                            "ep_size": ep_size,
                            "node_num": node_num,
                            "hidden_size": SYNTH_HIDDEN,
                            "topk": SYNTH_TOPK,
                            "num_experts": SYNTH_EXPERTS,
                            "sms": sms,
                            "num_tokens": num_tokens,
                            "latency": 50.0,  # us (loader divides by 1000)
                            "power": 300.0,
                        }
                    )
    return rows


def _ep_rows() -> list[dict]:
    """moe_ep expert-compute rows: ``kernel_source="deepep_moe"`` (what
    ``MoEExpertCompute._resolve_kernel_source`` picks on sglang/vllm) and the
    ``power_law_1.2`` distribution MOEModel passes for this family."""
    rows = []
    for phase in ("context", "generation"):
        for ep_size, _node in _PAIRS:
            for num_tokens in _TOKEN_POINTS:
                rows.append(
                    {
                        "kernel_source": "deepep_moe",
                        "moe_dtype": "bfloat16",
                        "distribution": "power_law_1.2",
                        "inference_phase": phase,
                        "topk": SYNTH_TOPK,
                        "num_experts": SYNTH_EXPERTS,
                        "num_slots": SYNTH_EXPERTS,
                        "hidden_size": SYNTH_HIDDEN,
                        "inter_size": SYNTH_INTER,
                        "moe_tp_size": 1,
                        "moe_ep_size": ep_size,
                        "num_tokens": num_tokens,
                        "latency": 1.5,  # ms (stored raw)
                        "power": 400.0,
                    }
                )
    return rows


def _trtllm_fp8_a2a_rows() -> list[dict]:
    """Hopper FP8 serving dtypes: HT is BF16 while LL is FP8."""
    rows = []
    for backend, comm_dtype in (("trtllm_deepep_ht", "bfloat16"), ("trtllm_deepep_ll", "fp8")):
        for ep_size, node_num in _PAIRS:
            for phase in ("dispatch", "combine"):
                for num_tokens in _TOKEN_POINTS:
                    rows.append(
                        {
                            "comm_backend": backend,
                            "phase": phase,
                            "comm_dtype": comm_dtype,
                            "ep_size": ep_size,
                            "node_num": node_num,
                            "hidden_size": SYNTH_HIDDEN,
                            "topk": SYNTH_TOPK,
                            "num_experts": SYNTH_EXPERTS,
                            "sms": 0,
                            "num_tokens": num_tokens,
                            "latency": 50.0,
                            "power": 300.0,
                        }
                    )
    return rows


def _trtllm_fp8_ep_rows() -> list[dict]:
    rows = _ep_rows()
    for row in rows:
        row["kernel_source"] = "trtllm_deepep"
        row["moe_dtype"] = "fp8_block"
    return rows


def _write_version_dir(root: str, family: str, backend: str, filename: str, rows: list[dict]) -> None:
    version_dir = os.path.join(root, "data", family, backend, SYNTH_VERSION)
    os.makedirs(version_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(version_dir, filename))
    stem = filename.split(".")[0]
    with open(os.path.join(version_dir, "collection_meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"status": "complete", "schema_version": 2, "tables": {stem: {"status": "complete"}}}, f)


@pytest.fixture(scope="module")
def synth_model_path(tmp_path_factory) -> str:
    """A local model dir with the fabricated config.json -- resolved by the
    SHIPPED ``os.path.isdir`` branch of the model-info loader (module-scoped:
    ``_get_model_info`` is @cache'd on the path string)."""
    model_dir = tmp_path_factory.mktemp("synth-moe-5120x64")
    (model_dir / "config.json").write_text(json.dumps(_SYNTH_HF_CONFIG), encoding="utf-8")
    return str(model_dir)


@pytest.fixture
def synth_systems(tmp_path):
    """A synthetic systems root (8 GPUs/node, SM 90) holding ONLY the two
    large-EP tables, for sglang AND vllm, mounted alongside the shipped
    default systems path (pattern from ``test_coverage_candidates.py``)."""
    root = str(tmp_path / "systems")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, f"{SYNTH_SYSTEM}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "data_dir": "data",
                "misc": {"nccl_version": "v1"},
                "gpu": {
                    "sm_version": 90,
                    "bfloat16_tc_flops": 1000.0,
                    "mem_bw": 100.0,
                    "mem_empirical_constant_latency": 1.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 100.0,
                    "intra_node_bw": 100.0,
                    "p2p_latency": 0.000001,
                },
            },
            f,
        )
    for backend in BACKENDS:
        _write_version_dir(root, "comm", backend, "moe_a2a_perf.parquet", _a2a_rows())
        _write_version_dir(root, "moe", backend, "moe_expert_compute_perf.parquet", _ep_rows())
    _write_version_dir(root, "comm", "trtllm", "moe_a2a_perf.parquet", _trtllm_fp8_a2a_rows())
    _write_version_dir(root, "moe", "trtllm", "moe_expert_compute_perf.parquet", _trtllm_fp8_ep_rows())
    databases_cache.clear()
    set_systems_paths(["default", root])
    try:
        yield root
    finally:
        set_systems_paths(None)
        databases_cache.clear()


def _synth_task(model_path: str, backend: str, **overrides) -> Task:
    kwargs = {
        "serving_mode": "agg",
        "model_path": model_path,
        "system_name": SYNTH_SYSTEM,
        "backend_name": backend,
        "backend_version": SYNTH_VERSION,
    }
    kwargs.update(overrides)
    return Task(**kwargs)


DEEPEP_BOTH_PHASES = {"context": "deepep_ht", "generation": "deepep_ll"}


# ---------------------------------------------------------------------------
# (a) the coverage probes report the collected shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_coverage_probes_report_the_synthetic_shape(synth_systems, backend):
    database = get_database(SYNTH_SYSTEM, backend, SYNTH_VERSION)
    a2a = database.moe_a2a_coverage(SYNTH_HIDDEN, SYNTH_TOPK, SYNTH_EXPERTS)
    assert a2a == {"deepep_ht": set(_PAIRS), "deepep_ll": set(_PAIRS)}
    for phase in ("context", "generation"):
        compute = database.moe_expert_compute_coverage(
            SYNTH_HIDDEN, SYNTH_INTER, SYNTH_TOPK, SYNTH_EXPERTS, common.MoEQuantMode.bfloat16, phase
        )
        assert compute == {ep for ep, _node in _PAIRS}


# ---------------------------------------------------------------------------
# (b) Task resolution offers large-EP tuples for exactly the covered EPs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_resolution_produces_large_ep_candidates(synth_systems, synth_model_path, backend):
    t = _synth_task(synth_model_path, backend)
    assert t.model_family == "MOE"

    # Per-tuple rule: covered EP-only tuples resolve the deepep pair ...
    for ep in (8, 16):
        assert t._resolve_moe_comm_backend("agg", (1, 1, ep, 1, ep, 1)) == DEEPEP_BOTH_PHASES
    # ... uncovered EP / MoE-TP / trivial-EP tuples stay fused.
    assert t._resolve_moe_comm_backend("agg", (1, 1, 4, 1, 4, 1)) is None
    assert t._resolve_moe_comm_backend("agg", (2, 1, 8, 2, 8, 1)) is None
    assert t._resolve_moe_comm_backend("agg", (1, 1, 1, 1, 1, 1)) is None

    # The DEFAULT enumeration reaches the covered EPs: sglang unions its
    # static multi-node ladder in when coverage is non-empty; vllm derives
    # its ladder FROM the covered EP sizes (data-only enablement, no static
    # ladder ships), so both reach every covered EP.
    large = [tup for tup in t.iter_parallel("agg") if t._resolve_moe_comm_backend("agg", tup)]
    assert large, "default ladders must produce large-EP candidates for the covered shape"
    assert all(tup[3] == 1 for tup in large)  # moe_tp == 1 by the rule
    assert {tup[4] for tup in large} == {8, 16}
    if backend == "sglang":
        assert 16 in t.agg_moe_ep_candidates  # the unioned multi-node ladder


@pytest.mark.parametrize("backend", BACKENDS)
def test_build_model_config_carries_backend_and_node_width(synth_systems, synth_model_path, backend):
    t = _synth_task(synth_model_path, backend)
    mc = t.build_model_config(role="agg", parallel=(1, 1, 8, 1, 8, 1))
    assert mc.moe_comm_backend == DEEPEP_BOTH_PHASES
    assert mc.num_gpus_per_node == 8  # set together with the comm backend
    fused = t.build_model_config(role="agg", parallel=(1, 1, 4, 1, 4, 1))
    assert fused.moe_comm_backend is None


# ---------------------------------------------------------------------------
# (c) one candidate builds and every large-EP op queries end to end
# ---------------------------------------------------------------------------


def _large_ep_ops(op_list) -> list:
    """All MoEAllToAll/MoEExpertCompute instances, recursing into OverlapOp groups."""
    found = []
    stack = list(op_list)
    while stack:
        op = stack.pop()
        if isinstance(op, (MoEAllToAll, MoEExpertCompute)):
            found.append(op)
        for group in ("_group_a", "_group_b"):
            stack.extend(getattr(op, group, None) or [])
    return found


def _built_model(t: Task, model_path: str, backend: str, parallel):
    """What the sweep does per point: the task's ModelConfig for the tuple,
    the tuple's widths, then ``get_model``."""
    tp, pp, dp, moe_tp, moe_ep, cp = parallel
    mc = t.build_model_config(role="agg", parallel=parallel)
    mc.tp_size, mc.pp_size, mc.attention_dp_size = tp, pp, dp
    mc.moe_tp_size, mc.moe_ep_size, mc.cp_size = moe_tp, moe_ep, cp
    return get_model(model_path, mc, backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_candidate_graph_builds_and_large_ep_ops_query_finitely(synth_systems, synth_model_path, backend):
    t = _synth_task(synth_model_path, backend)
    model = _built_model(t, synth_model_path, backend, (1, 1, 8, 1, 8, 1))

    large_ops = _large_ep_ops(model.context_ops) + _large_ep_ops(model.generation_ops)
    assert sorted(op._name for op in large_ops) == [
        "context_moe",
        "context_moe_combine",
        "context_moe_dispatch",
        "generation_moe",
        "generation_moe_combine",
        "generation_moe_dispatch",
    ]
    assert sorted(type(op).__name__ for op in large_ops) == ["MoEAllToAll"] * 4 + ["MoEExpertCompute"] * 2

    # Every emitted large-EP op queries the synthetic tables successfully:
    # x=128 per-rank tokens -> 128 on the comm curve (in range) and
    # 128 * dp8 = 1024 on the compute curve (in range).
    database = get_database(SYNTH_SYSTEM, backend, SYNTH_VERSION)
    for op in large_ops:
        latency = float(op._engine_query(database, x=128))
        assert math.isfinite(latency) and latency > 0.0, op._name


def test_hopper_fp8_trtllm_coverage_emission_and_rust_lookup(synth_systems, synth_model_path):
    """Coverage and graph emission use the same serving wire-dtype contract."""
    task = _synth_task(synth_model_path, "trtllm")
    task.moe_quant_mode = common.MoEQuantMode.fp8_block
    task._large_ep_coverage_cache.clear()

    assert task._large_ep_coverage("agg") == {
        "context": {"trtllm_deepep_ht": {8, 16}},
        "generation": {"trtllm_deepep_ll": {8, 16}},
    }
    model_config = task.build_model_config(role="agg", parallel=(1, 1, 8, 1, 8, 1))
    assert model_config.system == SYNTH_SYSTEM
    assert model_config.moe_comm_backend == {
        "context": "trtllm_deepep_ht",
        "generation": "trtllm_deepep_ll",
    }

    model = get_model(synth_model_path, model_config, "trtllm")
    context_a2a = [op for op in _large_ep_ops(model.context_ops) if isinstance(op, MoEAllToAll)]
    generation_a2a = [op for op in _large_ep_ops(model.generation_ops) if isinstance(op, MoEAllToAll)]
    assert {op._comm_dtype for op in context_a2a} == {"bfloat16"}
    assert {op._comm_dtype for op in generation_a2a} == {"fp8"}

    database = get_database(SYNTH_SYSTEM, "trtllm", SYNTH_VERSION)
    for op in context_a2a + generation_a2a:
        latency = float(op._engine_query(database, x=128))
        assert math.isfinite(latency) and latency > 0.0, op._name


@pytest.mark.parametrize("backend", BACKENDS)
def test_uncovered_tuple_of_the_same_task_builds_the_fused_graph(synth_systems, synth_model_path, backend):
    """The gate is per tuple, not per model: ep=4 has no rows, so the same
    task's ep=4 candidate builds the fused emission with zero large-EP ops."""
    t = _synth_task(synth_model_path, backend)
    model = _built_model(t, synth_model_path, backend, (1, 1, 4, 1, 4, 1))
    assert _large_ep_ops(model.context_ops) + _large_ep_ops(model.generation_ops) == []
    fused_names = {op._name for op in model.context_ops}
    assert {"context_moe_pre_dispatch", "context_moe", "context_moe_post_dispatch"} <= fused_names


# ---------------------------------------------------------------------------
# guard: a str-typed moe_quant_mode is a caller bug, not "no coverage"
# ---------------------------------------------------------------------------


def test_str_moe_quant_mode_raises_type_error(synth_systems, synth_model_path):
    """The compute-coverage probe keys the moe_ep table on ``MoEQuantMode``
    members; a str (a caller bypassing ``_resolve_quant_str``) would miss every
    key and silently disable large-EP exploration — raise loudly instead."""
    t = _synth_task(synth_model_path, "sglang")
    t.moe_quant_mode = "bfloat16"
    t._large_ep_coverage_cache.clear()
    with pytest.raises(TypeError, match="MoEQuantMode"):
        t._large_ep_coverage("agg")


# ---------------------------------------------------------------------------
# fixtures only: nothing was added to the source tree to make this pass
# ---------------------------------------------------------------------------


def test_source_tree_is_untouched_fixtures_only(synth_systems, synth_model_path):
    """The contract ran entirely on shipped code paths: the synthetic model
    resolved into the existing MOE family (no new family, no new model class),
    and the builder registry still holds exactly the two variants the package
    registers at import (no test-local ``register_moe_block``)."""
    from aiconfigurator.sdk.models.blocks.moe import _MOE_BLOCK_REGISTRY, LARGE_EP_READY_FAMILIES

    assert get_model_family(synth_model_path) == "MOE"
    assert "MOE" in LARGE_EP_READY_FAMILIES
    assert set(_MOE_BLOCK_REGISTRY) == {("DEEPSEEK", "sglang", "*"), ("DEEPSEEKV32", "sglang", "*")}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
