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

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import databases_cache, set_systems_paths
from aiconfigurator.sdk.task_v2 import Task

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
# Expert-compute rows exist for bfloat16 only: a task on another MoE quant has
# comm coverage but no compute coverage, so it stays fused.
_EP_QUANT = "bfloat16"


def _a2a_rows() -> list[dict]:
    rows = []
    for backend, pairs in (("deepep_ht", _HT_PAIRS), ("deepep_ll", _LL_PAIRS)):
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


def _ep_rows() -> list[dict]:
    rows = []
    for phase, pairs in (("context", _HT_PAIRS), ("generation", _LL_PAIRS)):
        for ep_size, _node in pairs:
            for num_tokens in (128, 1024):
                rows.append(
                    {
                        "kernel_source": "deepep",
                        "moe_dtype": _EP_QUANT,
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
                        "latency": 1.5,
                        "power": 400.0,
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


@pytest.fixture
def synth_systems(tmp_path):
    """A synthetic systems root (8 GPUs/node, SM 90) holding ONLY the two
    large-EP tables, mounted alongside the shipped default systems path."""
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
    _write_version_dir(root, "comm", "moe_a2a_perf.parquet", _a2a_rows())
    _write_version_dir(root, "moe", "moe_ep_perf.parquet", _ep_rows())

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
    assert t.agg_tp_candidates == [1, 2, 4, 8]
    assert t.agg_pp_candidates == [1]
    assert t.agg_dp_candidates == [1, 2, 4, 8, 16, 32, 64]
    assert t.agg_moe_tp_candidates == [1, 2, 4, 8]
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


def test_agg_requires_both_phases_covered(synth_systems):
    """ep=32 is collected for deepep_ht (context) only; an agg worker runs both
    phases, so the tuple stays fused."""
    t = _synth_task()
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=32, moe_ep=32)) is None


def test_uncovered_ep_and_moe_tp_gt_1_stay_fused(synth_systems):
    t = _synth_task()
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=4, moe_ep=4)) is None  # ep not collected
    assert t._resolve_moe_comm_backend("agg", _tuple(tp=2, dp=8, moe_tp=2, moe_ep=8)) is None
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=1, moe_ep=1)) is None  # ep == 1 is fused


def test_compute_coverage_is_quant_specific(synth_systems):
    """Comm coverage alone is not enough: the EP expert-compute table is keyed
    by the run's MoE quant mode, so another quant gets no large-EP tuples."""
    t = _synth_task(moe_quant_mode=common.MoEQuantMode.fp8_block)
    assert t._resolve_moe_comm_backend("agg", _tuple(dp=8, moe_ep=8)) is None
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8]  # fused defaults


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
    assert mc.num_gpus_per_node == 8
    fused = t.build_model_config(role="agg", parallel=_tuple(dp=4, moe_ep=4))
    assert fused.moe_comm_backend is None
    assert fused.num_gpus_per_node == 8  # always injected; only large EP reads it


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
    """Only the phase a role RUNS gates the tuple (prefill->context,
    decode->generation); the other phase joins the dict whenever the data
    covers it, because the role's model object holds one whole graph (its
    context ops size the memory model even on a decode worker)."""
    t = _disagg_task()
    # ep=32: context-only coverage -> prefill takes it, decode stays fused.
    assert t._resolve_moe_comm_backend("prefill", _tuple(dp=32, moe_ep=32)) == {"context": "deepep_ht"}
    assert t._resolve_moe_comm_backend("decode", _tuple(dp=32, moe_ep=32)) is None
    # ep=16: both phases covered -> both roles carry the full per-phase dict.
    both = {"context": "deepep_ht", "generation": "deepep_ll"}
    assert t._resolve_moe_comm_backend("decode", _tuple(dp=16, moe_ep=16)) == both
    assert t._resolve_moe_comm_backend("prefill", _tuple(dp=16, moe_ep=16)) == both


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
    """Shipped h200_sxm/sglang carries no moe_a2a rows for the Qwen3 shape, so
    the task keeps the fused ladders and states which collector to run."""
    import aiconfigurator.sdk.task_v2 as task_v2

    task_v2._LARGE_EP_EMPTY_COVERAGE_LOGGED.clear()
    with caplog.at_level(logging.INFO, logger="aiconfigurator.sdk.task_v2"):
        t = Task(
            serving_mode="agg",
            model_path=SYNTH_MODEL,
            system_name="h200_sxm",
            backend_name="sglang",
            total_gpus=8,
        )
        t._large_ep_coverage("agg")  # a second probe must not re-log
    assert t.agg_moe_ep_candidates == [1, 2, 4, 8]
    assert t.agg_num_gpu_candidates == [1, 2, 4, 8]
    assert all(t._resolve_moe_comm_backend("agg", tup) is None for tup in t.iter_parallel("agg"))
    hits = [r for r in caplog.records if "large-EP" in r.message and "collector" in r.message]
    assert len(hits) == 1, [r.message for r in caplog.records]


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
    assert mc.num_gpus_per_node == 4  # GB200 NVL4 — not the 8-GPU HGX default


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
