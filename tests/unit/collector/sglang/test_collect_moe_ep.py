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
# payload owned by this collector, in the order load_moe_expert_compute_data keys them
# (aic-core .../sdk/operations/moe_comm.py::load_moe_expert_compute_data). The SDK-side
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
    # carries the DECLARED DeepSeek-V3 geometry, not a live HF read — plus the
    # model identity the subprocess loads (F14: never a defaulted checkpoint).
    model = "deepseek-ai/DeepSeek-V3"
    assert cases == [
        [128, 2, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [64, 4, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [32, 8, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [16, 16, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [8, 32, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [4, 64, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [2, 128, 7168, 2048, 8, 256, 256, "fp8_block", model],
        [1, 256, 7168, 2048, 8, 256, 256, "fp8_block", model],
    ]


def test_local_expert_counts_match_the_retired_hardcoded_sweep(monkeypatch, moe_ep_symbols):
    # The retired get_wideep_moe_test_cases(total_experts) halved the expert
    # count from EP=2 down to 1 local expert. Declared expansion must reproduce
    # exactly that coverage — no rows gained, none lost.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    local_experts = [case[0] for case in moe_ep_symbols["get_moe_ep_test_cases"]()]
    assert local_experts == [128, 64, 32, 16, 8, 4, 2, 1]


def test_cases_resolve_their_own_declared_model(moe_ep_symbols):
    # F14: the executor sets COLLECTOR_MODEL_PATH only for single-model plans;
    # a full plan runs cases from several model families, so each subprocess
    # loads the case's OWN declared model — never a defaulted checkpoint that
    # would fail (or silently pass with the wrong geometry through) the
    # declaration asserts.
    loaded = _load_symbols("_case_model_path")
    loaded["os"] = os
    loaded["_resolve_moe_model_path"] = lambda model_id: f"resolved:{model_id}"

    os_environ_backup = os.environ.get("COLLECTOR_MODEL_PATH")
    try:
        os.environ.pop("COLLECTOR_MODEL_PATH", None)
        assert loaded["_case_model_path"]("moonshotai/Kimi-K2.5") == "resolved:moonshotai/Kimi-K2.5"
        os.environ["COLLECTOR_MODEL_PATH"] = "/scratch/artifacts/DeepSeek-V3"
        assert loaded["_case_model_path"]("deepseek-ai/DeepSeek-V3") == "resolved:/scratch/artifacts/DeepSeek-V3"
    finally:
        if os_environ_backup is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = os_environ_backup


def test_prefill_manifest_excludes_empty_uniform_workloads_at_generation(capsys):
    # F15 twin: `num_token * topk * ep // num_experts == 0` is pure declared
    # arithmetic, so the empty uniform points are resolved in the token-list
    # builder — the benchmark loop runs the manifest exactly (reaching the
    # runtime zero-workload branch raises as an invariant breach). Power-law
    # variants sample their own counts and keep every token point.
    loaded = _load_symbols("get_moe_prefill_test_cases", "_sorted_phase_cases")

    # DeepSeek-V3 at any declared EP: 16 * 8 * 2 // 256 == 1 — nothing drops.
    dsv3 = loaded["get_moe_prefill_test_cases"](2, topk=8, num_experts=256)
    assert {"num_tokens": 16, "distributed": "uniform", "power_law_alpha": None} in dsv3

    # Kimi-K2.5-like (384 experts) at EP 2: 16 * 8 * 2 // 384 == 0 — the
    # uniform point drops at generation, its power-law variants stay.
    kimi = loaded["get_moe_prefill_test_cases"](2, topk=8, num_experts=384)
    assert {"num_tokens": 16, "distributed": "uniform", "power_law_alpha": None} not in kimi
    assert {"num_tokens": 16, "distributed": "power_law", "power_law_alpha": 0.8} in kimi
    assert {"num_tokens": 32, "distributed": "uniform", "power_law_alpha": None} in kimi

    # Every unconditional generation-time filter is also counted and logged.
    # EP 64 drops 4/8 below the measurable floor and 16384 above the per-rank
    # token budget; neither point may disappear silently from the plan.
    loaded["get_moe_prefill_test_cases"](64, topk=8, num_experts=256)
    output = capsys.readouterr().out
    assert "dropped 2 token points [4, 8] below the minimum measurable workload" in output
    assert "dropped 1 token points [16384] above the per-rank token budget" in output


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
    # Every case is a valid EP shard of its declared expert count, tied to the
    # model whose checkpoint the subprocess loads.
    for local_experts, ep_size, _hidden, _inter, _topk, num_experts, num_slots, quant, model in cases:
        assert local_experts * ep_size == num_experts
        assert num_slots == num_experts
        assert ep_size > 1
        assert quant == "fp8_block"
        assert isinstance(model, str) and model
    # F14 registry surface: every declared wideep model family reaches the
    # plan under its own identity — including families whose shape arguments
    # collide — so no case can load another model's checkpoint.
    from collector.case_generator import (
        get_common_moe_test_cases,
        is_wideep_moe_model,
        moe_model_allows_quantization,
    )

    declared_models = {
        recipe.model_name
        for recipe in get_common_moe_test_cases(backend="sglang")
        if is_wideep_moe_model(recipe.model_name)
        and recipe.tp == 1
        and recipe.ep > 1
        and moe_model_allows_quantization("sglang", recipe.model_name, "fp8_block")
    }
    case_models = {case[8] for case in cases}
    assert case_models == declared_models
    assert len(case_models) >= 2, "the full path must span at least two model families"
    # Cross-model shape collisions exist — without the model argument those
    # cases would share task identity (and a default checkpoint).
    assert len({tuple(case[:8]) for case in cases}) < len(cases)
    assert len({tuple(case) for case in cases}) == len(cases)


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

    perf_file = tmp_path / "moe_expert_compute_perf.txt"
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

    perf_file = tmp_path / "moe_expert_compute_perf.txt"
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
    assert entry.perf_filename is PerfFile.MOE_EXPERT_COMPUTE


def test_hash_closures_declare_the_case_yaml_the_module_now_reads():
    from collector.provenance import load_closures

    closures = load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    entry = closures["collector.wideep.sglang.collect_deepep_moe"]
    assert "collector/cases/base_ops/moe.yaml" in entry
    assert "__model_cases__" in entry


def test_no_silent_case_skipping_in_the_benchmark_loops():
    # failure_handling.md: a queued case is executed or raises a classified
    # error. The retired `except Exception: ... skipping` handlers must not
    # come back, and neither may the runtime token-point drops (F15): the
    # deterministic empty-uniform predicate lives in the token-list builder,
    # and the stochastic LL buffer exceedance raises classified.
    assert "skipping..." not in SOURCE_TEXT
    assert "dropping num_tokens=" not in SOURCE_TEXT
    assert "the generation-time manifest should" in SOURCE_TEXT
    assert "drew a routing sample" in SOURCE_TEXT
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
