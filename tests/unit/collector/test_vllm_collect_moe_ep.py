# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contract tests for the DORMANT vllm ``moe_ep`` collector (D3).

``collector/wideep/vllm/collect_moe_ep.py`` imports torch and vllm at module
scope, so it cannot be imported here. The population/writer helpers are pure
and are AST-extracted from the source, and the bench callable is injectable —
the whole row/population contract is covered with a mocked bench (same
``MOE_EP_HEADER`` literal as the sglang/trtllm twins: one consumer contract).

Dormancy per plan decision D3 is itself under test: the vLLM WideEP runtime
and empty registry serve standalone A2A only; no ``moe_ep`` registry or hash
closure entry exists until the compute path is verified.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "wideep" / "vllm" / "collect_moe_ep.py"
SOURCE_TEXT = SOURCE_PATH.read_text()

# The frozen moe_ep CSV header — identical to the sglang twin's literal
# (tests/unit/collector/sglang/test_collect_moe_ep.py::MOE_EP_HEADER). The
# SDK-side twin is tests/unit/sdk/database/test_collector_schema_contract.py::
# MOE_EP_HEADER.
MOE_EP_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "moe_dtype,distribution,inference_phase,num_tokens,hidden_size,inter_size,"
    "topk,num_experts,num_slots,moe_tp_size,moe_ep_size,latency"
)


def _load_symbols(*names: str) -> dict:
    """Exec the named module-level functions/constants without importing vllm."""
    tree = ast.parse(SOURCE_TEXT, filename=str(SOURCE_PATH))

    def wanted(node) -> bool:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            return node.name in names
        if isinstance(node, ast.Assign):
            return any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        return False

    selected = [node for node in tree.body if wanted(node)]
    loaded: dict = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), loaded)
    return loaded


@pytest.fixture
def moe_ep_symbols():
    return _load_symbols(
        "MOE_EP_QUANT_MODE",
        "MOE_EP_KERNEL_SOURCE",
        "MOE_EP_OP_NAME",
        "MOE_EP_CONTEXT_NUM_TOKENS",
        "MOE_EP_GENERATION_NUM_TOKENS",
        "MOE_EP_POWER_LAW_ALPHAS",
        "MoeEpBenchmarkError",
        "get_moe_ep_test_cases",
        "_build_moe_ep_row",
        "_global_num_tokens",
        "_phase_cases",
        "_collect_phase_rows",
    )


# ---------------------------------------------------------------------------
# Population (declared shapes, backend="vllm")
# ---------------------------------------------------------------------------


def _stage_counts() -> dict[str, int]:
    """Recreate the getter's stages independently, for the population table."""
    from collector.case_generator import (
        get_common_moe_test_cases,
        is_wideep_moe_model,
        moe_model_allows_quantization,
    )

    recipes = list(get_common_moe_test_cases(backend="vllm"))
    wideep = [case for case in recipes if is_wideep_moe_model(case.model_name)]
    ep_only = [case for case in wideep if case.tp == 1 and case.ep > 1]
    quantized = [case for case in ep_only if moe_model_allows_quantization("vllm", case.model_name, "fp8_block")]
    unique = {(case.model_name, case.ep) for case in quantized}
    return {
        "recipes": len(recipes),
        "wideep_declared": len(wideep),
        "ep_only": len(ep_only),
        "quant_allowed": len(quantized),
        "unique_invocations": len(unique),
    }


def test_deepseek_v3_population_counts_per_stage(monkeypatch, moe_ep_symbols):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")

    assert _stage_counts() == {
        "recipes": 42,
        "wideep_declared": 42,
        "ep_only": 24,
        "quant_allowed": 24,
        "unique_invocations": 8,
    }

    cases = moe_ep_symbols["get_moe_ep_test_cases"]()
    # Same 8 declared (model, ep) invocations as the sglang twin — EP 2..256,
    # halving local expert counts.
    assert cases == [
        [128, 2, 7168, 2048, 8, 256, 256, "fp8_block"],
        [64, 4, 7168, 2048, 8, 256, 256, "fp8_block"],
        [32, 8, 7168, 2048, 8, 256, 256, "fp8_block"],
        [16, 16, 7168, 2048, 8, 256, 256, "fp8_block"],
        [8, 32, 7168, 2048, 8, 256, 256, "fp8_block"],
        [4, 64, 7168, 2048, 8, 256, 256, "fp8_block"],
        [2, 128, 7168, 2048, 8, 256, 256, "fp8_block"],
        [1, 256, 7168, 2048, 8, 256, 256, "fp8_block"],
    ]


def test_models_without_a_declared_wideep_row_expand_to_zero(monkeypatch, moe_ep_symbols):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-235B-A22B")
    assert moe_ep_symbols["get_moe_ep_test_cases"]() == []


def test_artifact_without_the_collected_quant_mode_expands_to_zero(monkeypatch, moe_ep_symbols):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/DeepSeek-V3.1-NVFP4")
    assert moe_ep_symbols["get_moe_ep_test_cases"]() == []


def test_full_population_covers_every_declared_wideep_model(monkeypatch, moe_ep_symbols):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    counts = _stage_counts()
    cases = moe_ep_symbols["get_moe_ep_test_cases"]()
    assert counts["unique_invocations"] == len(cases) == 110
    for local_experts, ep_size, _hidden, _inter, _topk, num_experts, num_slots, quant in cases:
        assert local_experts * ep_size == num_experts
        assert num_slots == num_experts
        assert ep_size > 1
        assert quant == "fp8_block"


# ---------------------------------------------------------------------------
# Phase sweeps + row collection through a mocked bench
# ---------------------------------------------------------------------------


def test_phase_sweeps_are_sorted_and_mirror_the_sglang_grids(moe_ep_symbols):
    context = moe_ep_symbols["_phase_cases"]("context")
    generation = moe_ep_symbols["_phase_cases"]("generation")
    # 13 context token points x (uniform + 5 alphas); 8 decode batch sizes.
    assert len(context) == 13 * 6
    assert len(generation) == 8 * 6
    for cases in (context, generation):
        keys = [
            (
                case["distributed"],
                case["power_law_alpha"] if case["power_law_alpha"] is not None else -1.0,
                case["num_tokens"],
            )
            for case in cases
        ]
        assert keys == sorted(keys)  # D5 sorted emission
    with pytest.raises(ValueError, match="unknown inference_phase"):
        moe_ep_symbols["_phase_cases"]("prefill")


def test_collect_phase_rows_builds_the_frozen_payload_from_the_bench(moe_ep_symbols):
    calls = []

    def bench(inference_phase, global_num_tokens, distributed, power_law_alpha):
        calls.append((inference_phase, global_num_tokens, distributed, power_law_alpha))
        return 0.5 + global_num_tokens / 1000.0, None

    phase_cases = [
        {"num_tokens": 4, "distributed": "uniform", "power_law_alpha": None},
        {"num_tokens": 8, "distributed": "power_law", "power_law_alpha": 1.01},
    ]
    rows = moe_ep_symbols["_collect_phase_rows"](
        inference_phase="context",
        phase_cases=phase_cases,
        bench=bench,
        moe_ep_size=32,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=256,
    )

    # The bench receives the GLOBAL token count — the exact value persisted
    # in num_tokens (both flow through _global_num_tokens once): benchmarked
    # population and key cannot diverge.
    assert calls == [("context", 4 * 32, "uniform", None), ("context", 8 * 32, "power_law", 1.01)]
    assert [(row["distribution"], row["num_tokens"]) for row, _power in rows] == [
        ("uniform", 4 * 32),
        ("power_law_1.01", 8 * 32),
    ]
    assert [call[1] for call in calls] == [row["num_tokens"] for row, _power in rows]
    for row, power_stats in rows:
        assert power_stats is None
        assert list(row.keys()) == [
            "moe_dtype",
            "distribution",
            "inference_phase",
            "num_tokens",
            "hidden_size",
            "inter_size",
            "topk",
            "num_experts",
            "num_slots",
            "moe_tp_size",
            "moe_ep_size",
            "latency",
        ]
        assert row["moe_dtype"] == "fp8_block"
        assert row["inference_phase"] == "context"
        assert row["moe_tp_size"] == 1


def test_a_failing_bench_raises_classified_with_the_case_parameters(moe_ep_symbols):
    def bench(inference_phase, global_num_tokens, distributed, power_law_alpha):
        raise RuntimeError("CUDA error: out of resources")

    with pytest.raises(
        moe_ep_symbols["MoeEpBenchmarkError"],
        match=r"num_tokens=16.*global_num_tokens=128.*alpha=1.2.*moe_ep_size=8",
    ):
        moe_ep_symbols["_collect_phase_rows"](
            inference_phase="generation",
            phase_cases=[{"num_tokens": 16, "distributed": "power_law", "power_law_alpha": 1.2}],
            bench=bench,
            moe_ep_size=8,
            hidden_size=7168,
            inter_size=2048,
            topk=8,
            num_experts=256,
            num_slots=256,
        )


def test_token_accounting_mirrors_the_sglang_twin(moe_ep_symbols):
    """Both moe_ep collectors must benchmark and persist the SAME token
    population expression, or vllm rows would land on keys shared with sglang
    rows that ran an ep-fold different population under the one
    kernel_source="deepep_moe" leg.

    vllm side is behavioral: the value handed to the bench, the persisted
    num_tokens and _global_num_tokens are one expression (per_rank * ep).
    sglang side is a source contract: the same global expression
    (num_token * simulated_ep_size) feeds BOTH distribution generators and the
    persisted column; the vllm bench feeds its pre-globalized argument into
    the same generators with no re-division.
    """
    # vllm: bench input == persisted num_tokens == _global_num_tokens(4, 32).
    assert moe_ep_symbols["_global_num_tokens"](4, 32) == 128
    seen = []
    [(row, _power)] = moe_ep_symbols["_collect_phase_rows"](
        inference_phase="generation",
        phase_cases=[{"num_tokens": 4, "distributed": "power_law", "power_law_alpha": 1.01}],
        bench=lambda phase, global_tokens, dist, alpha: (seen.append(global_tokens), (1.0, None))[1],
        moe_ep_size=32,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=256,
    )
    assert seen == [128]
    assert row["num_tokens"] == 128 == moe_ep_symbols["_global_num_tokens"](4, 32)

    # sglang: the GLOBAL expression feeds both generators and the persisted
    # column (collect_deepep_moe.py) ...
    import re

    sglang_text = (REPO_ROOT / "collector" / "wideep" / "sglang" / "collect_deepep_moe.py").read_text()
    assert "num_tokens_iter = hidden_states_per_token_iter.shape[0]" in sglang_text  # == num_token * ep
    assert "int(num_token * simulated_ep_size)" in sglang_text  # sizes the prefill tensors
    assert sglang_text.count("num_tokens=num_token * simulated_ep_size,") == 2  # both persisted rows
    assert re.search(r"power_law_deepep_decode\(\s*num_token \* simulated_ep_size,", sglang_text)
    assert re.search(r"power_law_deepep_prefill\(\s*num_tokens_iter,", sglang_text)
    # ... and the vllm bench feeds its (global) argument straight into the
    # same generators — no per-rank re-division anywhere in the module.
    assert re.search(r"power_law_deepep_prefill\(\s*global_num_tokens,", SOURCE_TEXT)
    assert re.search(r"power_law_deepep_decode\(\s*global_num_tokens,", SOURCE_TEXT)
    assert "num_tokens // moe_ep_size" not in SOURCE_TEXT
    assert "global_num_tokens //" not in SOURCE_TEXT


def test_mocked_bench_rows_round_trip_the_frozen_header(tmp_path, moe_ep_symbols):
    from collector.helper import finalize_perf_files, log_perf

    [(row, _power)] = moe_ep_symbols["_collect_phase_rows"](
        inference_phase="generation",
        phase_cases=[{"num_tokens": 2, "distributed": "uniform", "power_law_alpha": None}],
        bench=lambda *args: (0.125, None),
        moe_ep_size=16,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=256,
    )

    perf_file = tmp_path / "moe_expert_compute_perf.txt"
    assert log_perf(
        item_list=[row],
        framework="VLLM",
        version="0.24.0",
        device_name="NVIDIA B200",
        op_name=moe_ep_symbols["MOE_EP_OP_NAME"],
        kernel_source=moe_ep_symbols["MOE_EP_KERNEL_SOURCE"],
        perf_filename=str(perf_file),
    )
    assert perf_file.read_text().splitlines()[0] == MOE_EP_HEADER

    [parquet_path] = finalize_perf_files([perf_file])
    import pyarrow.parquet as pq

    record = pq.read_table(parquet_path).to_pylist()[0]
    # The consumer resolves vllm large-EP compute onto the same kernel leg as
    # sglang (moe_comm.py::MoEExpertCompute._resolve_kernel_source returns "deepep_moe"
    # for both backends).
    assert record["kernel_source"] == "deepep_moe"
    assert record["op_name"] == "moe_ep"
    assert record["latency"] == pytest.approx(0.125)
    assert record["num_tokens"] == 32


# ---------------------------------------------------------------------------
# Dormancy (D3)
# ---------------------------------------------------------------------------


def test_vllm_wideep_registry_exists_for_a2a_but_enrolls_no_single_host_op():
    from collector.wideep.vllm.registry import REGISTRY

    assert REGISTRY == []
    assert not any(entry.op == "moe_ep" for entry in REGISTRY)


def test_manifest_pin_is_for_the_verified_a2a_runtime():
    from collector.framework_manifest import load_manifest

    runtime = load_manifest()["frameworks"]["wideep_vllm"]["default"]
    assert runtime["version"] == "0.24.0"
    assert runtime["source_commit"] == "ee0da84ab9e04ac7610e28580af62c365e898389"


def test_hash_closures_has_no_entry_for_the_unregistered_module():
    # Task-1 sequencing rule: a closures entry may not precede registration.
    from collector.provenance import load_closures

    closures = load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    assert "collector.wideep.vllm.collect_moe_ep" not in closures


def test_registry_module_maps_the_standalone_a2a_runtime_to_an_empty_registry():
    from collector.framework_manifest import _REGISTRY_MODULES

    assert _REGISTRY_MODULES["wideep_vllm"] == "collector.wideep.vllm.registry"


def test_kernel_source_backends_have_no_vllm_deepep_moe_mapping():
    # Enrollment surface #3 (README step 4): the module writes
    # kernel_source="deepep_moe", which the curated fact table maps only for
    # framework: sglang today. Activation adds the vllm mapping WITH a
    # verified citation — until then its absence is pinned so activated rows
    # cannot silently resolve to `unverified` facts.
    import yaml

    table = yaml.safe_load((REPO_ROOT / "collector" / "kernel_source_backends.yaml").read_text())
    vllm_deepep = [
        entry
        for entry in table["mappings"]
        if entry.get("framework") == "vllm" and entry.get("kernel_source") == "deepep_moe"
    ]
    assert vllm_deepep == []


def test_source_documents_the_activation_procedure():
    assert "DORMANT" in SOURCE_TEXT
    assert "wideep_vllm" in SOURCE_TEXT  # names the manifest key enrollment adds
    assert "skipping..." not in SOURCE_TEXT
    assert "raise MoeEpBenchmarkError" in SOURCE_TEXT
