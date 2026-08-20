# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage-driven large-EP candidate participation (spec sections 4.4.3 / 4.5 / 4.6).

Large EP is no longer a flag: ``Task`` probes the perf database's coverage API
for the model's MoE shape and lets every parallel tuple whose ``moe_ep`` is
covered (with ``moe_tp == 1``) participate, resolving the per-phase comm
backend for it. These tests drive that resolution against a SYNTHETIC systems
root (parquets written here, parsed by the real PR 1 loaders) so the covered /
uncovered split is controlled rather than inherited from shipped data, plus
two shipped-data checks (trtllm nvlink, and the no-coverage control).
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.cli.api import cli_estimate
from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.moe_comm_resolver import a2a_covers_parallel, resolve_model_config_moe_comm
from aiconfigurator.sdk.perf_database import (
    PerfDataNotAvailableError,
    databases_cache,
    load_system_spec,
    set_systems_paths,
)
from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator_core.sdk.models import get_model

pytestmark = pytest.mark.unit

# Qwen3-235B-A22B: the synthetic tables are written for this shape so the
# resolution runs against a real checkpoint parse (MOE family, large-EP-ready).
SYNTH_MODEL = "Qwen/Qwen3-235B-A22B"
SYNTH_HIDDEN, SYNTH_INTER, SYNTH_TOPK, SYNTH_EXPERTS = 4096, 1536, 8, 128

SYNTH_SYSTEM = "synth_h8"
SYNTH_BACKEND = "sglang"
SYNTH_VERSION = "9.9.9"

# Context (deepep_ht) is collected one EP step further than generation
# (deepep_ll) — the asymmetry every per-phase assertion below keys on.
_HT_PAIRS = ((8, 1), (16, 2), (32, 4))
_LL_PAIRS = ((8, 1), (16, 2))


def _a2a_rows(backends=(("deepep_ht", _HT_PAIRS), ("deepep_ll", _LL_PAIRS))) -> list[dict]:
    rows = []
    for backend, pairs in backends:
        for ep_size, node_num in pairs:
            for phase in ("dispatch", "combine"):
                for num_tokens in (128, 1024):
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
                            "sms": 20,
                            "num_tokens": num_tokens,
                            "latency": 50.0,
                            "power": 300.0,
                        }
                    )
    return rows


def _write_version_dir(root: str, family: str, filename: str, rows: list[dict]) -> None:
    version_dir = os.path.join(root, "data", family, SYNTH_BACKEND, SYNTH_VERSION)
    os.makedirs(version_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(version_dir, filename))
    # Collector V3 sidecar: without it (or without a matching ``tables`` entry)
    # the loader warns per table; the synthetic data is complete, not partial.
    stem = filename.split(".")[0]
    with open(os.path.join(version_dir, "collection_meta.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"status": "complete", "schema_version": 2, "tables": {stem: {"status": "complete"}}}, f)


@pytest.fixture(autouse=True)
def _one_shot_log_state():
    """Snapshot/restore the module-level one-shot log dedupe sets.

    They are process-global by design (one log per model/system, not per Task),
    so a test that needs a fresh log must not leave the set emptied for the
    tests that run after it."""
    import aiconfigurator.sdk.task_v2 as task_v2

    empty_before = set(task_v2._LARGE_EP_EMPTY_COVERAGE_LOGGED)
    asym_before = set(task_v2._LARGE_EP_ASYMMETRIC_COVERAGE_WARNED)
    try:
        yield
    finally:
        task_v2._LARGE_EP_EMPTY_COVERAGE_LOGGED.clear()
        task_v2._LARGE_EP_EMPTY_COVERAGE_LOGGED.update(empty_before)
        task_v2._LARGE_EP_ASYMMETRIC_COVERAGE_WARNED.clear()
        task_v2._LARGE_EP_ASYMMETRIC_COVERAGE_WARNED.update(asym_before)


def _build_synth_root(tmp_path, a2a_rows) -> str:
    """A synthetic systems root holding only the large-EP A2A table."""
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
    _write_version_dir(root, "comm", "moe_a2a_perf.parquet", a2a_rows)
    return root


@pytest.fixture
def synth_systems(tmp_path):
    """Both phases collected (context via deepep_ht, generation via deepep_ll)."""
    root = _build_synth_root(tmp_path, _a2a_rows())
    databases_cache.clear()
    set_systems_paths(["default", root])
    try:
        yield root
    finally:
        set_systems_paths(None)
        databases_cache.clear()


@pytest.fixture
def synth_systems_generation_only(tmp_path):
    """GENERATION rows only: the day a collection lands one phase ahead of the
    other. No context communication rows at all."""
    root = _build_synth_root(
        tmp_path,
        _a2a_rows((("deepep_ll", _LL_PAIRS),)),
    )
    databases_cache.clear()
    set_systems_paths(["default", root])
    try:
        yield root
    finally:
        set_systems_paths(None)
        databases_cache.clear()


@pytest.fixture
def synth_systems_node1_fallback(tmp_path):
    """Only node-1 DeepEP comm is measured; compute covers a wider EP."""
    node1 = ((8, 1),)
    compute = ((8, 1), (32, 4))
    root = _build_synth_root(
        tmp_path,
        _a2a_rows((("deepep_ht", node1), ("deepep_ll", node1))),
        _ep_rows((("context", compute), ("generation", compute))),
    )
    databases_cache.clear()
    set_systems_paths(["default", root])
    try:
        yield root
    finally:
        set_systems_paths(None)
        databases_cache.clear()


def _synth_task(**overrides) -> Task:
    kwargs = {
        "serving_mode": "agg",
        "model_path": SYNTH_MODEL,
        "system_name": SYNTH_SYSTEM,
        "backend_name": SYNTH_BACKEND,
        "backend_version": SYNTH_VERSION,
    }
    kwargs.update(overrides)
    return Task(**kwargs)


def _tuple(tp=1, pp=1, dp=8, moe_tp=1, moe_ep=8, cp=1):
    return (tp, pp, dp, moe_tp, moe_ep, cp)


# ---------------------------------------------------------------------------
# (a) coverage present -> wideep-shaped ladders + per-tuple resolution
# ---------------------------------------------------------------------------


def test_covered_model_gets_union_of_wideep_and_fused_ladders(synth_systems):
    """Coverage replaces the enable_wideep flag: the agg ladders become the
    union of today's wideep lists and today's fused defaults, so ONE task
    explores both regimes."""
    t = _synth_task()
    assert t.agg_num_gpu_candidates == [1, 2, 4, 8, 16, 32, 64]
    assert t.agg_tp_candidates == [1, 2, 4, 8, 16]
    assert t.agg_pp_candidates == [1]
    assert t.agg_dp_candidates == [1, 2, 4, 8, 16, 32, 64]
    assert t.agg_moe_tp_candidates == [1, 2, 4, 8, 16]
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8, 16, 32, 64]


def test_covered_tuple_resolves_per_phase_deepep_backends(synth_systems):
    t = _synth_task()
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=8, moe_ep=8)) == {
        "context": "deepep_ht",
        "generation": "deepep_ll",
    }
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=16, moe_ep=16)) == {
        "context": "deepep_ht",
        "generation": "deepep_ll",
    }


def test_sglang_node1_deepep_coverage_represents_multi_node_ep(synth_systems_node1_fallback):
    t = _synth_task()
    point = _tuple(dp=32, moe_ep=32)
    both = {"context": "deepep_ht", "generation": "deepep_ll"}
    assert t._resolve_moe_comm_backend("agg", point) == both
    assert t.build_model_config(role="agg", parallel=point).moe_comm_backend == both


@pytest.mark.parametrize("noncanonical_pair", [{(4, 1)}, {(16, 1)}, {(32, 2)}])
def test_sglang_node1_fallback_requires_legacy_canonical_coordinate(noncanonical_pair):
    assert not a2a_covers_parallel(
        noncanonical_pair,
        framework="sglang",
        comm_backend="deepep_ht",
        moe_ep_size=32,
        expected_nodes=8,
    )


def test_exact_resolver_accepts_sglang_node1_deepep_substitution():
    database = SimpleNamespace(
        system="synthetic",
        version="1.0",
        system_spec={"node": {"num_gpus_per_node": 4}, "gpu": {"sm_version": 100}},
        moe_a2a_coverage=lambda *_args: {"deepep_ht": {(8, 1)}, "deepep_ll": {(8, 1)}},
        moe_expert_compute_coverage=lambda *_args: {128},
    )
    model_config = ModelConfig(attention_dp_size=128, moe_tp_size=1, moe_ep_size=128)

    resolved = resolve_model_config_moe_comm(
        model_config,
        model_path=SYNTH_MODEL,
        backend_name="sglang",
        database=database,
        required_phases=("context", "generation"),
    )

    assert resolved == {"context": "deepep_ht", "generation": "deepep_ll"}


def test_sglang_deepseek_node1_substitution_restores_wideep_mla_defaults():
    database = SimpleNamespace(
        system="synthetic",
        version="1.0",
        system_spec={"node": {"num_gpus_per_node": 4}, "gpu": {"sm_version": 100}},
        moe_a2a_coverage=lambda *_args: {"deepep_ht": {(8, 1)}},
        moe_expert_compute_coverage=lambda *_args: {128},
    )
    model_config = ModelConfig(attention_dp_size=128, moe_tp_size=1, moe_ep_size=128)

    resolve_model_config_moe_comm(
        model_config,
        model_path="deepseek-ai/DeepSeek-R1",
        backend_name="sglang",
        database=database,
        required_phases=("context",),
    )

    assert model_config.fmha_quant_mode == common.FMHAQuantMode.fp8_block
    assert model_config.kvcache_quant_mode == common.KVCacheQuantMode.fp8


def test_agg_requires_both_phases_covered(synth_systems):
    """ep=32 is collected for deepep_ht (context) only; an agg worker runs both
    phases, so the capability probe does not resolve a backend. The exact
    model-config build then rejects the cross-node fused fallback."""
    t = _synth_task()
    point = _tuple(dp=32, moe_ep=32)
    assert t._resolve_moe_comm_backend("agg", point) is None
    with pytest.raises(PerfDataNotAvailableError, match=r"Cross-node EP.*DeepEP A2A.*moe_ep=32"):
        t.build_model_config(role="agg", parallel=point)


@pytest.mark.parametrize(
    ("system", "gpus_per_node"),
    [
        ("gb200", 4),
        ("gb300", 4),
        ("b200_sxm", 8),
        ("b300_sxm", 8),
        ("h100_sxm", 8),
        ("h200_sxm", 8),
    ],
)
def test_cross_node_boundary_comes_from_system_topology(system, gpus_per_node):
    assert load_system_spec(system)["node"]["num_gpus_per_node"] == gpus_per_node


@pytest.mark.parametrize(("gpus_per_node", "local_ep", "cross_node_ep"), [(4, 4, 8), (8, 8, 16)])
def test_exact_config_allows_intra_node_fused_but_requires_deepep_cross_node(
    synth_systems, monkeypatch, gpus_per_node, local_ep, cross_node_ep
):
    t = _synth_task()
    monkeypatch.setattr(t, "_num_gpus_per_node", lambda _role: gpus_per_node)
    monkeypatch.setattr(t, "_large_ep_coverage", lambda _role: {})
    database = SimpleNamespace(
        system=SYNTH_SYSTEM,
        version=SYNTH_VERSION,
        system_spec={"node": {"num_gpus_per_node": gpus_per_node}, "gpu": {"sm_version": 90}},
    )
    monkeypatch.setattr(t, "_try_load_role_database", lambda _role: database)

    local = t.build_model_config(role="agg", parallel=_tuple(dp=local_ep, moe_ep=local_ep))
    assert local.moe_comm_backend is None
    with pytest.raises(PerfDataNotAvailableError, match=rf"Cross-node EP.*moe_ep={cross_node_ep}"):
        t.build_model_config(role="agg", parallel=_tuple(dp=cross_node_ep, moe_ep=cross_node_ep))


def test_trtllm_attention_dp_gate_falls_through_to_cross_node_data_error():
    database = SimpleNamespace(
        system="synthetic",
        version="1.0",
        system_spec={"node": {"num_gpus_per_node": 8}, "gpu": {"sm_version": 100}},
        moe_a2a_coverage=lambda *_args: {"nvlink_two_sided": {(16, 2)}},
        moe_expert_compute_coverage=lambda *_args: {16},
    )
    model_config = ModelConfig(attention_dp_size=1, moe_tp_size=1, moe_ep_size=16)

    with pytest.raises(PerfDataNotAvailableError, match=r"Cross-node EP.*moe_ep=16"):
        resolve_model_config_moe_comm(
            model_config,
            model_path=SYNTH_MODEL,
            backend_name="trtllm",
            database=database,
            required_phases=("context", "generation"),
        )


def test_trtllm_attention_dp_gate_keeps_intra_node_ep_fused():
    database = SimpleNamespace(
        system="synthetic",
        version="1.0",
        system_spec={"node": {"num_gpus_per_node": 8}, "gpu": {"sm_version": 100}},
    )
    model_config = ModelConfig(attention_dp_size=1, moe_tp_size=1, moe_ep_size=8)

    assert (
        resolve_model_config_moe_comm(
            model_config,
            model_path=SYNTH_MODEL,
            backend_name="trtllm",
            database=database,
            required_phases=("context", "generation"),
        )
        is None
    )


def test_uncovered_ep_and_moe_tp_gt_1_stay_fused(synth_systems):
    t = _synth_task()
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=4, moe_ep=4)) is None  # ep not collected
    assert t._resolve_moe_comm_backend("agg", _tuple(tp=2, dp=8, moe_tp=2, moe_ep=8)) is None
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=1, moe_ep=1)) is None  # ep == 1 is fused


def test_a2a_coverage_is_independent_of_compute_quant(synth_systems):
    t = _synth_task(moe_quant_mode=common.MoEQuantMode.fp8_block)
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=8, moe_ep=8)) is None
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8, 16]  # fused defaults


def test_unready_family_never_resolves_a_backend(synth_systems, monkeypatch):
    """Gate on the families whose model classes are wired for large-EP
    emission; an unlisted family keeps the fused path even with full data."""
    t = _synth_task()
    monkeypatch.setattr(t, "_model_family", "HYBRIDMOE", raising=False)
    t._large_ep_coverage_cache.clear()
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=8, moe_ep=8)) is None


def test_build_model_config_sets_backend_and_node_width(synth_systems):
    t = _synth_task()
    mc = t.build_model_config(role="agg", parallel=_tuple(dp=8, moe_ep=8))
    assert mc.moe_comm_backend == {"context": "deepep_ht", "generation": "deepep_ll"}
    assert mc.moe_comm_query_profile == {"context": (8, 1), "generation": (8, 1)}
    assert mc.num_gpus_per_node == 8
    fused = t.build_model_config(role="agg", parallel=_tuple(dp=4, moe_ep=4))
    assert fused.moe_comm_backend is None
    assert fused.num_gpus_per_node == 8  # always injected; only large EP reads it


def test_build_model_config_reuses_task_coverage_snapshot(synth_systems, monkeypatch):
    t = _synth_task()
    expected = t._large_ep_coverage("agg")
    database = t._try_load_role_database("agg")

    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("build_model_config must reuse the Task coverage cache")

    monkeypatch.setattr(database, "moe_a2a_coverage", unexpected_probe)
    monkeypatch.setattr(database, "moe_expert_compute_coverage", unexpected_probe)

    point = _tuple(dp=16, moe_ep=16)
    model_config = t.build_model_config(role="agg", parallel=point)
    assert expected
    assert model_config.moe_comm_backend == {"context": "deepep_ht", "generation": "deepep_ll"}


# ---------------------------------------------------------------------------
# (d) user candidate lists restrict the tuples, never the modeling rule
# ---------------------------------------------------------------------------


def test_user_moe_ep_candidates_restrict_tuples_not_the_rule(synth_systems):
    """The user list wins over the coverage-derived defaults (``_set``
    semantics), and the per-tuple rule is unchanged: every surviving EP-only
    tuple is still large-EP, MoE-TP ones are still fused."""
    t = _synth_task(agg_moe_ep_candidates=[16])
    assert t.agg_moe_ep_candidates == [16]
    tuples = list(t.iter_parallel("agg"))
    assert {tup[4] for tup in tuples} == {16}
    ep_only = [tup for tup in tuples if tup[3] == 1]
    assert ep_only
    assert all(t._resolve_moe_comm_backend("agg", tup) for tup in ep_only)
    assert all(t._resolve_moe_comm_backend("agg", tup) is None for tup in tuples if tup[3] > 1)


# ---------------------------------------------------------------------------
# (e) disagg: per-role phases, asymmetric coverage, require_same_tp exemption
# ---------------------------------------------------------------------------


def _disagg_task(**overrides) -> Task:
    kwargs = {
        "serving_mode": "disagg",
        "prefill_model_path": SYNTH_MODEL,
        "prefill_system_name": SYNTH_SYSTEM,
        "prefill_backend_name": SYNTH_BACKEND,
        "prefill_backend_version": SYNTH_VERSION,
        "decode_model_path": SYNTH_MODEL,
        "decode_system_name": SYNTH_SYSTEM,
        "decode_backend_name": SYNTH_BACKEND,
        "decode_backend_version": SYNTH_VERSION,
    }
    kwargs.update(overrides)
    return Task(**kwargs)


def test_disagg_roles_gate_on_their_own_phase(synth_systems):
    """The phase a role RUNS gates the tuple (prefill->context,
    decode->generation) and the context phase is required on top for every
    role, because the role's model object holds one whole graph and its
    context ops size the memory model even on a decode worker."""
    t = _disagg_task()
    # ep=32: context-only coverage -> prefill takes it, decode has no
    # generation rows there and stays fused.
    assert t._resolve_moe_comm_backend("prefill", _tuple(dp=32, moe_ep=32)) == {"context": "deepep_ht"}
    assert t._resolve_moe_comm_backend("decode", _tuple(dp=32, moe_ep=32)) is None
    # ep=16: both phases covered -> both roles carry the full per-phase dict.
    both = {"context": "deepep_ht", "generation": "deepep_ll"}
    assert t._resolve_moe_comm_backend("decode", _tuple(dp=16, moe_ep=16)) == both
    assert t._resolve_moe_comm_backend("prefill", _tuple(dp=16, moe_ep=16)) == both


def test_generation_only_coverage_keeps_decode_fused_and_warns(synth_systems_generation_only, caplog):
    """Generation collected ahead of context: a decode tuple must NOT resolve a
    generation-only comm backend. Its model would emit a FUSED context span
    whose (÷tp shared experts, router GEMM) weights are what
    base_backend._get_memory_usage sizes the worker from -- the same
    mis-pricing class the disagg-decode capture caught, in the other
    direction. One warning names the asymmetry."""
    with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.task_v2"):
        t = _disagg_task()
        assert t._resolve_moe_comm_backend("decode", _tuple(dp=8, moe_ep=8)) is None
        assert t._resolve_moe_comm_backend("decode", _tuple(dp=16, moe_ep=16)) is None
        assert t._resolve_moe_comm_backend("prefill", _tuple(dp=8, moe_ep=8)) is None
    warnings = [r.message for r in caplog.records if "asymmetric" in r.message]
    assert len(warnings) == 1, warnings
    assert "context phase is not" in warnings[0]
    # ...and the whole task falls back to the fused ladders/tuples.
    assert t.decode_moe_ep_candidates == [1, 2, 4, 8, 16]
    assert all(t._resolve_moe_comm_backend("decode", tup) is None for tup in t.iter_parallel("decode"))


def test_disagg_require_same_tp_is_exempt_per_pair(synth_systems):
    """SGLang disagg requires matching prefill/decode TP (KV transfer layout);
    a pair with a large-EP side is exempt, a fused pair is not."""
    t = _disagg_task(prefill_moe_ep_candidates=[32], decode_moe_ep_candidates=[32])
    gate = t.sweep_disagg_kwargs(prefill_database=None, decode_database=None)["require_same_tp"]
    assert callable(gate)
    large_ep_prefill = {"tp": 1, "pp": 1, "dp": 32, "moe_tp": 1, "moe_ep": 32, "cp": 1}
    fused_decode = {"tp": 8, "pp": 1, "dp": 4, "moe_tp": 1, "moe_ep": 32, "cp": 1}
    fused_prefill = {"tp": 8, "pp": 1, "dp": 4, "moe_tp": 4, "moe_ep": 8, "cp": 1}
    assert gate(large_ep_prefill, fused_decode) is False  # prefill side is large EP -> exempt
    assert gate(fused_prefill, fused_decode) is True  # both fused -> TP must match


def test_disagg_replica_budget_follows_coverage(synth_systems):
    t = _disagg_task()
    assert t.max_gpu_per_replica == 512
    assert t.num_gpu_per_replica is None


# ---------------------------------------------------------------------------
# (b) no coverage -> fused defaults everywhere, one INFO log
# ---------------------------------------------------------------------------


def test_uncovered_model_keeps_fused_defaults_and_logs_once(caplog):
    """A model+system+backend combo whose shipped moe_a2a table doesn't cover
    the shape keeps the fused ladders and states which collector to run.

    Uses a100_sxm/sglang, which ships no wideep DeepEP data and therefore
    can never adapt into moe_a2a coverage for any shape."""
    import aiconfigurator.sdk.task_v2 as task_v2

    task_v2._LARGE_EP_EMPTY_COVERAGE_LOGGED.clear()  # restored by the autouse fixture
    with caplog.at_level(logging.INFO, logger="aiconfigurator.sdk.task_v2"):
        t = Task(
            serving_mode="agg",
            model_path=SYNTH_MODEL,
            system_name="a100_sxm",
            backend_name="sglang",
            total_gpus=8,
        )
        t._large_ep_coverage("agg")  # a second probe must not re-log
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8, 16]
    assert t.agg_num_gpu_candidates == [1, 2, 4, 8]  # capped to total_gpus=8
    assert all(t._resolve_moe_comm_backend("agg", tup) is None for tup in t.iter_parallel("agg"))
    hits = [r for r in caplog.records if "large-EP" in r.message and "collector" in r.message]
    assert len(hits) == 1, [r.message for r in caplog.records]


# ---------------------------------------------------------------------------
# (c) shipped vllm single-node profile -> explicit unscaled cross-node proxy
# ---------------------------------------------------------------------------


def test_shipped_vllm_ep4_profile_drives_ep32_static_generation():
    """The imported GB200/vLLM profile is measured only at EP4/node1.

    EP32 is deliberately admitted by the dataset-specific proxy policy, and
    the emitted communication ops query that unchanged donor geometry.  The
    final static-gen call proves the profile survives candidate resolution,
    model construction, Rust engine compilation, and silicon lookup.
    """
    parallel = _tuple(dp=32, moe_ep=32)
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-R1",
        system_name="gb200",
        backend_name="vllm",
        backend_version="0.24.0",
        total_gpus=32,
    )
    assert t._resolve_moe_comm_backend("agg", parallel) == {
        "context": "deepep_ht",
        "generation": "deepep_ll",
    }
    base = t.build_model_config(role="agg", parallel=parallel)
    assert base.moe_comm_query_profile == {"context": (4, 1), "generation": (4, 1)}

    model_config = replace(
        base,
        tp_size=1,
        pp_size=1,
        attention_dp_size=32,
        moe_tp_size=1,
        moe_ep_size=32,
        cp_size=1,
    )
    model = get_model("deepseek-ai/DeepSeek-R1", model_config, "vllm")
    generation_a2a = [op for op in model.generation_ops if type(op).__name__ == "MoEAllToAll"]
    assert generation_a2a
    assert {(op._moe_ep_size, op._node_num) for op in generation_a2a} == {(4, 1)}

    result = cli_estimate(
        "deepseek-ai/DeepSeek-R1",
        "gb200",
        mode="static_gen",
        backend_name="vllm",
        backend_version="0.24.0",
        isl=8,
        osl=2,
        batch_size=1,
        tp_size=1,
        attention_dp_size=32,
        moe_tp_size=1,
        moe_ep_size=32,
        engine_step_backend="rust",
    )
    assert result.tpot > 0
    assert result.raw["parallel"] == "tp1pp1dp32etp1ep32"


# ---------------------------------------------------------------------------
# (c) shipped trtllm data -> nvlink_two_sided on both phases
# ---------------------------------------------------------------------------


def test_shipped_trtllm_nvfp4_resolves_nvlink_two_sided_both_phases():
    """gb200 ships both NVLink kernels; the two-sided spec wins by registry
    order where both cover an EP, and is the only one collected at ep=64."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-R1",
        system_name="gb200",
        backend_name="trtllm",
        backend_version="1.3.0rc10",
        moe_quant_mode=common.MoEQuantMode.nvfp4,
        gemm_quant_mode=common.GEMMQuantMode.nvfp4,
        total_gpus=64,
    )
    both = {"context": "nvlink_two_sided", "generation": "nvlink_two_sided"}
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=8, moe_ep=8)) == both
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=64, moe_ep=64)) == both
    mc = t.build_model_config(role="agg", parallel=_tuple(dp=8, moe_ep=8))
    assert mc.moe_comm_query_profile == {"context": (8, 2), "generation": (8, 2)}
    assert mc.num_gpus_per_node == 4  # GB200 NVL4 — not the 8-GPU HGX default


# ---------------------------------------------------------------------------
# Attention-table capability: which regime's table decides validate()
# ---------------------------------------------------------------------------


class _SupportedOverride:
    """Real database with an overridden ``supported_quant_mode`` map."""

    def __init__(self, database, supported):
        self._database = database
        self.supported_quant_mode = supported

    def __getattr__(self, name):
        return getattr(self._database, name)


def _override_supported(monkeypatch, drop=()):
    """Patch every Task DB load to hide ``drop`` from supported_quant_mode."""
    from aiconfigurator.sdk.perf_database import get_database

    database = get_database("h200_sxm", "sglang", "0.5.14")
    supported = {k: v for k, v in (database.supported_quant_mode or {}).items() if k not in drop}
    monkeypatch.setattr(Task, "_try_load_role_database", lambda self, role: _SupportedOverride(database, supported))


def _mixed_regime_task(**overrides) -> Task:
    """DeepSeek-R1 on h200/sglang: covered, default ladders -> fused AND
    large-EP tuples, i.e. two reachable attention tables
    (context_mla=[bfloat16], wideep_context_mla=[fp8_block])."""
    kwargs = {
        "serving_mode": "agg",
        "model_path": "deepseek-ai/DeepSeek-R1",
        "system_name": "h200_sxm",
        "backend_name": "sglang",
        "backend_version": "0.5.14",
    }
    kwargs.update(overrides)
    return Task(**kwargs)


def test_mixed_regime_validate_keys_on_the_fused_table():
    """Regression: an explicit fmha the FUSED table cannot serve must still
    fail fast, exactly as before large EP became per-tuple -- the large-EP
    table supporting it does not rescue the (majority) fused tuples, which
    would otherwise die one by one inside the sweep."""
    t = _mixed_regime_task(fmha_quant_mode=common.FMHAQuantMode.fp8_block)
    assert len(t._reachable_attention_op_keys("agg")) == 2  # both regimes reachable
    with pytest.raises(ValueError, match="Unsupported context_mla quant mode 'fp8_block'"):
        t.validate()


def test_uninformative_table_abstains_instead_of_green_lighting(monkeypatch):
    """An op the DB records no supported_quant_mode for carries no capability
    information: it must not green-light the check. With the fused entry gone,
    the large-EP table becomes the deciding one."""
    from aiconfigurator.sdk.errors import UnsupportedWideepConfigError

    _override_supported(monkeypatch, drop=("context_mla", "context_mla_granular"))
    t = _mixed_regime_task(fmha_quant_mode=common.FMHAQuantMode.bfloat16)
    with pytest.raises(UnsupportedWideepConfigError, match="wideep_context_mla"):
        t.validate()


def test_all_tables_uninformative_abstains(monkeypatch):
    """No information anywhere -> benefit of the doubt (legacy behavior)."""
    _override_supported(monkeypatch, drop=("context_mla", "context_mla_granular", "wideep_context_mla"))
    t = _mixed_regime_task(fmha_quant_mode=common.FMHAQuantMode.fp8_block)
    t.validate()  # must not raise


@pytest.mark.parametrize(
    "kwargs, expect_large_ep, expect_raises",
    [
        # No wideep DeepEP data on a100_sxm/sglang -> fused regime only.
        (dict(model_path=SYNTH_MODEL, system_name="a100_sxm", backend_name="sglang", total_gpus=8), False, False),
        # Dense model -> never large EP.
        (
            dict(model_path="meta-llama/Meta-Llama-3.1-70B", system_name="h100_sxm", backend_name="sglang"),
            False,
            False,
        ),
        # Covered model pinned to EP-only tuples -> large-EP regime only, and
        # the inferred fp8 fmha has no wideep_context_mla slice (fp8_block).
        (
            dict(
                model_path="deepseek-ai/DeepSeek-R1",
                system_name="h200_sxm",
                backend_name="sglang",
                backend_version="0.5.14",
                total_gpus=32,
                agg_num_gpu_candidates=[8, 16, 32],
                agg_tp_candidates=[1],
                agg_pp_candidates=[1],
                agg_dp_candidates=[8, 16, 32],
                agg_moe_tp_candidates=[1],
                agg_moe_ep_candidates=[8, 16, 32],
            ),
            True,
            True,
        ),
    ],
)
def test_single_regime_tasks_match_the_pre_change_key_logic(kwargs, expect_large_ep, expect_raises):
    """A task whose tuples all sit in ONE regime must resolve exactly the keys
    the old three-branch ``attention_op_keys(family, backend, flag)`` call
    produced, and validate to the same outcome -- the per-regime machinery is
    only allowed to change MIXED tasks."""
    from aiconfigurator.sdk.models import attention_op_keys

    t = Task(serving_mode="agg", **kwargs)
    pairs = t._reachable_attention_op_keys("agg")
    assert len(pairs) == 1, pairs
    assert pairs[0] == attention_op_keys(t.model_family, t.backend_name, expect_large_ep)
    assert t._attention_op_keys("agg") == pairs[0]
    if expect_raises:
        from aiconfigurator.sdk.errors import UnsupportedWideepConfigError

        with pytest.raises(UnsupportedWideepConfigError):
            t.validate()
    else:
        t.validate()


# ---------------------------------------------------------------------------
# D1: the deprecated moe_backend selector is inert for fused tuples
# ---------------------------------------------------------------------------


def test_deepep_moe_selector_is_neutralized_on_the_model_config(synth_systems):
    """``moe_backend="deepep_moe"`` used to select the wideep compute tables
    for the FUSED MoE op; large EP is coverage-driven now, so the per-tuple
    ModelConfig must not carry it."""
    t = _synth_task(moe_backend="deepep_moe")
    fused = t.build_model_config(role="agg", parallel=_tuple(dp=4, moe_ep=4))
    assert fused.moe_backend is None
    large_ep = t.build_model_config(role="agg", parallel=_tuple(dp=8, moe_ep=8))
    assert large_ep.moe_backend is None


def test_megamoe_selector_passes_through(synth_systems):
    """MegaMoE (DeepSeek-V4) is a real kernel selection, not a wideep flag."""
    t = Task(
        serving_mode="agg",
        model_path="deepseek-ai/DeepSeek-V4-Pro",
        system_name="b200_sxm",
        backend_name="sglang",
        moe_backend="megamoe",
    )
    mc = t.build_model_config(role="agg", parallel=_tuple(dp=8, moe_ep=8))
    assert mc.moe_backend == "megamoe"
    assert mc.moe_comm_backend is None  # DEEPSEEKV4 is not large-EP-wired


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
