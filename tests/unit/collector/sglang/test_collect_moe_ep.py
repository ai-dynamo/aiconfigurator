# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contract tests for the sglang unified ``moe_ep`` collector.

``collector/wideep/sglang/collect_deepep_moe.py`` imports torch and sglang at
module scope, so it cannot be imported here. The population/writer helpers are
pure and are AST-extracted from the source (same pattern as
``test_collect_moe_population.py``); the remaining guarantees are asserted
against the module text and the registry.
"""

import ast
import csv
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "wideep" / "sglang" / "collect_deepep_moe.py"
SOURCE_TEXT = SOURCE_PATH.read_text()

# The frozen moe_ep CSV header: the five helper.log_perf prefix columns plus the
# payload owned by this collector, in the order load_moe_ep_data keys them
# (aic-core .../sdk/operations/moe_comm.py::load_moe_ep_data). The SDK-side
# twin is tests/unit/sdk/database/test_collector_schema_contract.py::
# MOE_EP_HEADER.
MOE_EP_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "moe_dtype,distribution,inference_phase,num_tokens,hidden_size,inter_size,"
    "topk,num_experts,num_slots,moe_tp_size,moe_ep_size,latency"
)


def _load_symbols(*names: str) -> dict:
    """Exec the named module-level functions/constants without importing sglang."""
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
        "get_moe_ep_test_cases",
        "_build_moe_ep_row",
    )


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _stage_counts(backend: str = "sglang") -> dict[str, int]:
    """Recreate the getter's stages independently, for the population table."""
    from collector.case_generator import (
        get_common_moe_test_cases,
        is_wideep_moe_model,
        moe_model_allows_quantization,
    )

    recipes = list(get_common_moe_test_cases(backend=backend))
    wideep = [case for case in recipes if is_wideep_moe_model(case.model_name)]
    ep_only = [case for case in wideep if case.tp == 1 and case.ep > 1]
    quantized = [case for case in ep_only if moe_model_allows_quantization(backend, case.model_name, "fp8_block")]
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
        "recipes": 117,
        "wideep_declared": 117,
        "ep_only": 24,
        "quant_allowed": 24,
        # num_gpu x token-distribution recipes collapse: this collector
        # simulates EP on one GPU and sweeps distributions internally, so
        # (model, ep) is the whole invocation identity.
        "unique_invocations": 8,
    }

    cases = moe_ep_symbols["get_moe_ep_test_cases"]()
    assert len(cases) == 8
    # Sorted emission on the non-token key axis (moe_ep_size), and every case
    # carries the DECLARED DeepSeek-V3 geometry, not a live HF read.
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


def test_local_expert_counts_match_the_retired_hardcoded_sweep(monkeypatch, moe_ep_symbols):
    # The retired get_wideep_moe_test_cases(total_experts) halved the expert
    # count from EP=2 down to 1 local expert. Declared expansion must reproduce
    # exactly that coverage — no rows gained, none lost.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    local_experts = [case[0] for case in moe_ep_symbols["get_moe_ep_test_cases"]()]
    assert local_experts == [128, 64, 32, 16, 8, 4, 2, 1]


def test_models_without_a_declared_wideep_row_expand_to_zero(monkeypatch, moe_ep_symbols):
    # Qwen3-235B has moe model_case_values but no `wideep: true` — the op is
    # simply not declared for it. Zero cases, explainable from the declaration.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3-235B-A22B")
    assert moe_ep_symbols["get_moe_ep_test_cases"]() == []


def test_artifact_without_the_collected_quant_mode_expands_to_zero(monkeypatch, moe_ep_symbols):
    # The NVFP4 DeepSeek artifact declares allowed_modes [nvfp4]; this collector
    # benchmarks the fp8_block DeepEP path only, so its cases are dropped by the
    # declared quant policy rather than mislabeled as fp8_block.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/DeepSeek-V3.1-NVFP4")
    assert moe_ep_symbols["get_moe_ep_test_cases"]() == []


def test_full_population_covers_every_declared_wideep_model(monkeypatch, moe_ep_symbols):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    counts = _stage_counts()
    cases = moe_ep_symbols["get_moe_ep_test_cases"]()
    assert counts["unique_invocations"] == len(cases) == 62
    # Dedup is not a no-op: the raw stage carries 186 recipes for 62 invocations.
    assert counts["quant_allowed"] == 186
    # Every case is a valid EP shard of its declared expert count.
    for local_experts, ep_size, _hidden, _inter, _topk, num_experts, num_slots, quant in cases:
        assert local_experts * ep_size == num_experts
        assert num_slots == num_experts
        assert ep_size > 1
        assert quant == "fp8_block"


# ---------------------------------------------------------------------------
# Writer contract
# ---------------------------------------------------------------------------


def test_row_builder_emits_the_frozen_moe_ep_payload(tmp_path, moe_ep_symbols):
    from collector.helper import finalize_perf_files, log_perf

    row = moe_ep_symbols["_build_moe_ep_row"](
        moe_dtype="fp8_block",
        distribution="power_law_1.01",
        inference_phase="generation",
        num_tokens=128,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=256,
        moe_tp_size=1,
        moe_ep_size=32,
        latency_ms=0.4321,
    )

    perf_file = tmp_path / "moe_ep_perf.txt"
    assert log_perf(
        item_list=[row],
        framework="SGLang",
        version="0.5.10",
        device_name="NVIDIA B200",
        op_name="moe_ep",
        kernel_source=moe_ep_symbols["MOE_EP_KERNEL_SOURCE"],
        perf_filename=str(perf_file),
    )

    lines = perf_file.read_text().splitlines()
    assert lines[0] == MOE_EP_HEADER

    [parquet_path] = finalize_perf_files([perf_file])
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    assert table.column_names == MOE_EP_HEADER.split(",")
    record = table.to_pylist()[0]
    assert record == {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA B200",
        "op_name": "moe_ep",
        # The consumer keys sglang large-EP compute on this exact label
        # (moe_comm.py::_SGLANG_ADAPTED_KERNEL_SOURCES / _resolve_kernel_source).
        "kernel_source": "deepep_moe",
        "moe_dtype": "fp8_block",
        "distribution": "power_law_1.01",
        "inference_phase": "generation",
        "num_tokens": 128,
        "hidden_size": 7168,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "num_slots": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 32,
        # Latency is stored in milliseconds — the loader reads the column raw.
        "latency": pytest.approx(0.4321),
    }


def test_context_and_generation_rows_share_one_table(tmp_path, moe_ep_symbols):
    from collector.helper import log_perf

    perf_file = tmp_path / "moe_ep_perf.txt"
    for phase in ("context", "generation"):
        log_perf(
            item_list=[
                moe_ep_symbols["_build_moe_ep_row"](
                    moe_dtype="fp8_block",
                    distribution="uniform",
                    inference_phase=phase,
                    num_tokens=64,
                    hidden_size=7168,
                    inter_size=2048,
                    topk=8,
                    num_experts=256,
                    num_slots=256,
                    moe_tp_size=1,
                    moe_ep_size=32,
                    latency_ms=1.5,
                )
            ],
            framework="SGLang",
            version="0.5.10",
            device_name="NVIDIA B200",
            op_name="moe_ep",
            kernel_source=moe_ep_symbols["MOE_EP_KERNEL_SOURCE"],
            perf_filename=str(perf_file),
        )

    with open(perf_file, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["inference_phase"] for row in rows] == ["context", "generation"]
    # One header, one schema — the phase split is a column, not two files.
    assert perf_file.read_text().count("inference_phase") == 1


# ---------------------------------------------------------------------------
# Registry / source contract
# ---------------------------------------------------------------------------


def test_registry_exposes_moe_ep_and_retires_wideep_moe():
    from collector.registry_types import PerfFile
    from collector.wideep.sglang.registry import REGISTRY

    entries = {entry.op: entry for entry in REGISTRY}
    assert "wideep_moe" not in entries
    entry = entries["moe_ep"]
    assert entry.module == "collector.wideep.sglang.collect_deepep_moe"
    assert entry.get_func == "get_moe_ep_test_cases"
    assert entry.run_func == "run_moe_ep"
    assert entry.perf_filename is PerfFile.MOE_EP


def test_hash_closures_declare_the_case_yaml_the_module_now_reads():
    from collector.provenance import load_closures

    closures = load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    entry = closures["collector.wideep.sglang.collect_deepep_moe"]
    assert "collector/cases/base_ops/moe.yaml" in entry
    assert "__model_cases__" in entry


def test_no_silent_case_skipping_in_the_benchmark_loops():
    # failure_handling.md: a queued case is executed or raises a classified
    # error. The retired `except Exception: ... skipping` handlers must not
    # come back.
    assert "skipping..." not in SOURCE_TEXT
    assert "MoeEpBenchmarkError" in SOURCE_TEXT
    assert "MoeEpDeclarationMismatchError" in SOURCE_TEXT


def test_deepep_buffer_bound_is_parked_as_a_kernel_limit():
    # layer_permissions.md: an unverified framework kernel limit lives as a
    # FIXME(kernel-limit) note at the invocation site, so the next wideep
    # version bump greps it and either verifies or deletes it.
    assert "FIXME(kernel-limit)" in SOURCE_TEXT
    assert "num_max_dispatch_tokens_per_rank" in SOURCE_TEXT


def test_rows_persist_the_declared_topk_not_the_live_read():
    # The declared value is the contract; the live moe_layer.topk read is only
    # the subject of run_moe's _assert_declared("topk", ...) check.
    # Both _build_moe_ep_row call sites (context + generation) pass the
    # declared value; neither passes the live `topk` / `top_k` locals.
    assert SOURCE_TEXT.count("\n                            topk=model_topk,\n") == 2
    assert "\n                            topk=topk,\n" not in SOURCE_TEXT
    assert "\n                            topk=top_k,\n" not in SOURCE_TEXT
    assert '_assert_declared("topk"' in SOURCE_TEXT


def test_perf_path_comes_from_the_registry_not_ad_hoc_filenames():
    assert "wideep_context_moe_perf.txt" not in SOURCE_TEXT
    assert "wideep_generation_moe_perf.txt" not in SOURCE_TEXT
    assert "PerfFile.WIDEEP_MOE" not in SOURCE_TEXT


def test_power_is_measured_or_absent_never_zero():
    # D7: the generation bench measures power via benchmark_with_power and the
    # context bench via power_monitoring_only; neither writer substitutes 0.0.
    assert "power_monitoring_only" in SOURCE_TEXT
    assert "power_stats" in SOURCE_TEXT


def test_source_path_is_the_registered_module():
    assert SOURCE_PATH.exists()
    assert os.path.relpath(SOURCE_PATH, REPO_ROOT) == "collector/wideep/sglang/collect_deepep_moe.py"
