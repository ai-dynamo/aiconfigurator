# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contract tests for the trtllm unified ``moe_ep`` collector.

``collector/wideep/trtllm/collect_moe_compute.py`` imports torch and
tensorrt_llm at module scope, so it cannot be imported here. The population/
writer helpers are pure and are AST-extracted from the source (same pattern as
``tests/unit/collector/sglang/test_collect_moe_ep.py`` — the sglang twin whose
``MOE_EP_HEADER`` literal this file repeats verbatim: one consumer contract,
one frozen header); the remaining guarantees are asserted against the module
text, the registry and the manifest.
"""

import ast
import csv
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "wideep" / "trtllm" / "collect_moe_compute.py"
SOURCE_TEXT = SOURCE_PATH.read_text()

# The frozen moe_ep CSV header — identical to the sglang twin's literal
# (tests/unit/collector/sglang/test_collect_moe_ep.py::MOE_EP_HEADER): the five
# helper.log_perf prefix columns plus the payload owned by this collector, in
# the order load_moe_ep_data keys them (aic-core
# .../sdk/operations/moe_comm.py::load_moe_ep_data). The SDK-side twin is
# tests/unit/sdk/database/test_collector_schema_contract.py::MOE_EP_HEADER.
MOE_EP_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "moe_dtype,distribution,inference_phase,num_tokens,hidden_size,inter_size,"
    "topk,num_experts,num_slots,moe_tp_size,moe_ep_size,latency"
)


def _load_symbols(*names: str) -> dict:
    """Exec the named module-level functions/constants without importing trtllm."""
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


def _getter(sm_version: int):
    """The extracted getter with its module-scope dependencies injected."""
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    symbols = _load_symbols("get_moe_ep_test_cases")
    symbols["get_common_moe_test_cases"] = get_common_moe_test_cases
    symbols["is_wideep_moe_model"] = is_wideep_moe_model
    symbols["get_sm_version"] = lambda: sm_version
    symbols["aic_accurate_wideep_sim"] = True
    return symbols["get_moe_ep_test_cases"]


@pytest.fixture
def moe_ep_symbols():
    return _load_symbols("MOE_EP_OP_NAME", "_build_moe_ep_row", "_build_moe_ep_phase_rows")


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


def _retired_case_set(sm_version: int) -> set[tuple]:
    """The retired get_wideep_moe_compute_all_test_cases filter chain, inline.

    Used as change-specific kept/added/removed evidence: the rename must not
    gain or lose a single invocation (incl. the EPLB axis) — only the emission
    order changed (D5 sort).
    """
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    cases = set()
    for recipe in get_common_moe_test_cases():
        if not is_wideep_moe_model(recipe.model_name):
            continue
        if recipe.token_expert_distribution != "power_law":
            continue
        if recipe.tp != 1 or recipe.ep <= 1:
            continue
        moe_list = []
        # rc20 chain (#1356): fp8_block floor moved to min_sm 90 (SM87-89 lost
        # it, SM100+ gained it alongside nvfp4) and the two rc5-era SM120
        # nvfp4 kernel-limit skips were removed (hardware-verified clean).
        if sm_version >= 90:
            moe_list += ["fp8_block"]
        if sm_version >= 100:
            moe_list += ["nvfp4"]
        for moe_type in moe_list:
            eplb_configs = [(False, recipe.num_experts), (True, recipe.num_experts)]
            if recipe.num_experts <= 288:
                # replication identity: 384-expert models cannot lay out on 288 slots
                eplb_configs.append((True, 288))
            for use_eplb, num_slots in eplb_configs:
                if num_slots % recipe.ep != 0:
                    continue
                cases.add(
                    (
                        moe_type,
                        recipe.model_name,
                        recipe.ep,
                        recipe.power_law_alpha,
                        use_eplb,
                        num_slots,
                    )
                )
    return cases


def _stage_counts() -> dict[str, int]:
    """Recreate the getter's recipe-level stages independently."""
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = list(get_common_moe_test_cases())
    wideep = [case for case in recipes if is_wideep_moe_model(case.model_name)]
    power_law = [case for case in wideep if case.token_expert_distribution == "power_law"]
    ep_only = [case for case in power_law if case.tp == 1 and case.ep > 1]
    return {
        "recipes": len(recipes),
        "wideep_declared": len(wideep),
        "power_law": len(power_law),
        "ep_only": len(ep_only),
    }


def test_deepseek_v3_population_counts_per_stage(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")

    assert _stage_counts() == {
        "recipes": 117,
        "wideep_declared": 117,
        "power_law": 78,
        "ep_only": 16,  # 8 EP sizes x 2 power-law alphas
    }

    # 16 recipes x 1 SM-gated quant mode x 3 EPLB configs = 48, minus the 6
    # (True, 288) configs whose slots do not shard over ep in {64, 128, 256}.
    cases = _getter(90)()
    assert len(cases) == 42
    assert {case[0] for case in cases} == {"fp8_block"}
    # EPLB axis preserved exactly: baseline and redundant slot counts.
    assert {(case[13], case[14]) for case in cases} == {(False, 256), (True, 256), (True, 288)}
    # Blackwell collects BOTH quant modes since rc20 (#1356 moved the
    # fp8_block floor to min_sm 90, so SM100 gains it alongside nvfp4):
    # 16 recipes x 2 modes x 3 EPLB configs = 96, minus 2x6 slot drops.
    sm100_cases = _getter(100)()
    assert len(sm100_cases) == 84
    assert {case[0] for case in sm100_cases} == {"fp8_block", "nvfp4"}


def test_population_is_identical_to_the_retired_getter(monkeypatch):
    # kept 42 / added 0 / removed 0 on the invocation identity
    # (quant, model, ep, alpha, use_eplb, num_slots) — for every SM branch,
    # including the SM120 kernel-limit drops.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    for sm_version in (80, 90, 100, 120):
        new = {(case[0], case[10], case[8], case[12], case[13], case[14]) for case in _getter(sm_version)()}
        assert new == _retired_case_set(sm_version), f"coverage drift at SM{sm_version}"


def test_pre_hopper_sm_expands_to_zero_with_an_explained_drop(monkeypatch, capsys):
    # No quant-mode axis exists below SM87 (the SM-gated moe_list is empty) —
    # zero cases, explainable from the logged population line.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    assert _getter(80)() == []
    out = capsys.readouterr().out
    assert "0 cases from 117 moe recipes" in out
    assert "x 0 quant mode(s) x 2-3 EPLB configs" in out
    assert "= 0 expanded" in out


def test_sm90_42_case_reconciliation_is_log_pinned(monkeypatch, capsys):
    # The SM90 twin of the SM80/SM120 capsys pins: the 42-case population must
    # reconcile arithmetically in the getter's logged drop accounting, not
    # just in the returned count (117 recipes -> 16 kept x 1 mode x 3 EPLB
    # configs = 48, minus 6 slot-alignment drops, zero kernel-limit drops).
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    cases = _getter(90)()
    out = capsys.readouterr().out
    assert len(cases) == 42
    assert "moe_ep: 42 cases from 117 moe recipes" in out
    assert "-> 16 recipes kept) x 1 quant mode(s) x 2-3 EPLB configs" in out
    assert "= 48 expanded - 6 num_slots%ep!=0" in out


def test_sm120_collects_both_quants_with_no_kernel_limit_drops(monkeypatch, capsys):
    # The rc5-era SM120 nvfp4 kernel-limit drops are gone (rc20,
    # hardware-verified on RTX PRO 6000 2026-07-26 — see the getter's note),
    # and rc20's min_sm 90 floor gives SM120 fp8_block alongside nvfp4:
    # 16 recipes x 2 modes x 3 EPLB configs = 96, minus 2x6 slot drops.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    cases = _getter(120)()
    out = capsys.readouterr().out
    assert len(cases) == 84
    assert {case[0] for case in cases} == {"fp8_block", "nvfp4"}
    assert "= 96 expanded - 12 num_slots%ep!=0" in out
    # The EPLB axis survives in full on SM120 now.
    assert {case[13] for case in cases} == {False, True}


def test_full_population_covers_every_declared_wideep_model(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = _getter(90)()
    # 770 at the dc4caca merge-base: rc20 (#1356) restricts the (True, 288)
    # EPLB layout to models with <= 288 experts (replication identity), so
    # the five 384-expert entries (Kimi-K2/K2.5 trio, DeepSeek-V4-Pro pair)
    # collect 28 instead of 38 cases each.
    assert len(cases) == 770
    models = {case[10] for case in cases}
    assert "deepseek-ai/DeepSeek-V3" in models
    assert "moonshotai/Kimi-K2-Instruct" in models  # the 384-expert family
    # Invocation identities are unique — no dedup path is needed (skill: do
    # not ship dedup unless repository-owned YAML produces a duplicate).
    identities = [(case[0], case[10], case[8], case[12], case[13], case[14]) for case in cases]
    assert len(identities) == len(set(identities))
    for case in cases:
        moe_type, num_experts, tp, ep, dist, slots = (case[0], case[6], case[7], case[8], case[11], case[14])
        assert moe_type == "fp8_block"
        assert tp == 1 and ep > 1
        assert dist == "power_law"
        assert slots % ep == 0
        assert slots in (num_experts, 288)


def test_sorted_emission_is_deterministic(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "deepseek-ai/DeepSeek-V3")
    cases = _getter(90)()
    keys = [(case[10], case[0], case[8], case[12], case[13], case[14]) for case in cases]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Writer contract
# ---------------------------------------------------------------------------


def test_row_builder_emits_the_frozen_moe_ep_payload(tmp_path, moe_ep_symbols):
    from collector.helper import finalize_perf_files, log_perf

    row = moe_ep_symbols["_build_moe_ep_row"](
        moe_dtype="nvfp4",
        distribution="power_law_1.01_eplb",
        inference_phase="generation",
        num_tokens=128,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=288,
        moe_tp_size=1,
        moe_ep_size=32,
        latency_ms=0.4321,
    )

    perf_file = tmp_path / "moe_ep_perf.txt"
    assert log_perf(
        item_list=[row],
        framework="TRTLLM",
        version="1.3.0rc20",
        device_name="NVIDIA GB200",
        op_name=moe_ep_symbols["MOE_EP_OP_NAME"],
        # kernel_source stays the framework-dispatch ground truth the module
        # records in `source` — the same string the shipped legacy gb200
        # table carries and the PR-1 adapter passes through natively.
        kernel_source="wideep_compute_cutlass",
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
        "framework": "TRTLLM",
        "version": "1.3.0rc20",
        "device": "NVIDIA GB200",
        "op_name": "moe_ep",
        "kernel_source": "wideep_compute_cutlass",
        "moe_dtype": "nvfp4",
        # EPLB rides the distribution suffix and the num_slots column, exactly
        # as in the legacy table.
        "distribution": "power_law_1.01_eplb",
        "inference_phase": "generation",
        "num_tokens": 128,
        "hidden_size": 7168,
        "inter_size": 2048,
        "topk": 8,
        "num_experts": 256,
        "num_slots": 288,
        "moe_tp_size": 1,
        "moe_ep_size": 32,
        # Latency is stored in milliseconds — the loader reads the column raw.
        "latency": pytest.approx(0.4321),
    }


def test_one_measurement_emits_both_phases_matching_the_legacy_adapter(moe_ep_symbols):
    # The legacy wideep_moe_perf table has no phase split, and the PR-1
    # adapter (moe_comm._adapt_legacy_trtllm_wideep_moe) registers every
    # legacy row under BOTH inference phases. The unified writer emits both
    # phases per measurement so new rows key identically to adapted legacy
    # rows for the same configs.
    rows = moe_ep_symbols["_build_moe_ep_phase_rows"](
        moe_dtype="nvfp4",
        distribution="power_law_1.2_eplb",
        num_tokens=256,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=256,
        num_slots=288,
        moe_tp_size=1,
        moe_ep_size=8,
        latency_ms=1.25,
    )
    assert [row["inference_phase"] for row in rows] == ["context", "generation"]
    for row in rows:
        without_phase = {key: value for key, value in row.items() if key != "inference_phase"}
        assert without_phase == {
            "moe_dtype": "nvfp4",
            "distribution": "power_law_1.2_eplb",
            "num_tokens": 256,
            "hidden_size": 7168,
            "inter_size": 2048,
            "topk": 8,
            "num_experts": 256,
            "num_slots": 288,  # EPLB redundancy preserved: 288 slots, 256 experts
            "moe_tp_size": 1,
            "moe_ep_size": 8,
            "latency": pytest.approx(1.25),
        }


def test_both_phase_rows_share_one_table(tmp_path, moe_ep_symbols):
    from collector.helper import log_perf

    perf_file = tmp_path / "moe_ep_perf.txt"
    log_perf(
        item_list=moe_ep_symbols["_build_moe_ep_phase_rows"](
            moe_dtype="fp8_block",
            distribution="power_law_1.01",
            num_tokens=64,
            hidden_size=7168,
            inter_size=2048,
            topk=8,
            num_experts=256,
            num_slots=256,
            moe_tp_size=1,
            moe_ep_size=32,
            latency_ms=1.5,
        ),
        framework="TRTLLM",
        version="1.3.0rc20",
        device_name="NVIDIA GB200",
        op_name="moe_ep",
        kernel_source="wideep_compute_cutlass",
        perf_filename=str(perf_file),
    )

    with open(perf_file, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["inference_phase"] for row in rows] == ["context", "generation"]
    # One header, one schema — the phase split is a column, not two files.
    assert perf_file.read_text().count("inference_phase") == 1


# ---------------------------------------------------------------------------
# Registry / manifest / source contract
# ---------------------------------------------------------------------------


def test_registry_exposes_moe_ep_and_retires_trtllm_moe_wideep():
    from collector.registry_types import PerfFile
    from collector.wideep.trtllm.registry import REGISTRY

    entries = {entry.op: entry for entry in REGISTRY}
    assert "trtllm_moe_wideep" not in entries
    entry = entries["moe_ep"]
    assert entry.module == "collector.wideep.trtllm.collect_moe_compute"
    assert entry.get_func == "get_moe_ep_test_cases"
    assert entry.run_func == "run_moe_ep"
    assert entry.perf_filename is PerfFile.MOE_EP


def test_moe_ep_alone_resolves_to_the_wideep_trtllm_pin():
    from collector.framework_manifest import require_collector_runtime

    runtime = require_collector_runtime("trtllm", "1.3.0rc20", requested_ops={"moe_ep"}, wideep_ops={"moe_ep"})
    assert runtime.framework == "wideep_trtllm"
    assert runtime.workload == "wideep"
    assert runtime.version == "1.3.0rc20"


def test_same_pin_mixing_with_stock_trtllm_ops_is_accepted():
    # Unlike wideep_sglang (0.5.10 vs stock 0.5.14 — mixing fail-closes),
    # wideep_trtllm pins the SAME version AND image digest as stock trtllm
    # (framework_manifest.yaml), so require_collector_runtime accepts a mixed
    # stock+wideep request — the reason moe_ep can stay in the default trtllm
    # model plans.
    from collector.framework_manifest import require_collector_runtime

    mixed = require_collector_runtime(
        "trtllm", "1.3.0rc20", requested_ops={"moe", "gemm", "moe_ep"}, wideep_ops={"moe_ep"}
    )
    stock = require_collector_runtime("trtllm", "1.3.0rc20", requested_ops={"moe", "gemm"}, wideep_ops={"moe_ep"})
    assert mixed.version == stock.version == "1.3.0rc20"
    assert mixed.images == stock.images


def test_moe_ep_is_in_the_default_trtllm_plan_for_wideep_models():
    from collector.model_cases import build_collection_case_plan

    trtllm_plan = build_collection_case_plan(backend="trtllm", model_path="deepseek-ai/DeepSeek-V3")
    assert "moe_ep" in trtllm_plan.selected_ops
    assert "trtllm_moe_wideep" not in trtllm_plan.selected_ops
    # sglang keeps it out of the default plan (separate 0.5.10 runtime).
    sglang_plan = build_collection_case_plan(backend="sglang", model_path="deepseek-ai/DeepSeek-V3")
    assert "moe_ep" not in sglang_plan.selected_ops


def test_hash_closures_declare_the_case_yaml_the_module_reads():
    from collector.provenance import load_closures

    closures = load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    entry = closures["collector.wideep.trtllm.collect_moe_compute"]
    assert "collector/cases/base_ops/moe.yaml" in entry
    assert "__model_cases__" in entry


def test_retired_extras_are_not_persisted_columns():
    # dp_num_tokens / rank0_num_tokens / simulation_mode / moe_kernel stay in
    # the log line; they are NOT payload dict keys (the loader reads none of
    # them, and a stray key would widen the frozen header). The dict-key form
    # `"<name>":` is what the retired writer used.
    for retired_key in ('"dp_num_tokens":', '"rank0_num_tokens":', '"simulation_mode":', '"moe_kernel":'):
        assert retired_key not in SOURCE_TEXT, f"{retired_key} must not be a persisted column"
    # The only item_list the module writes is built by the shared row helper.
    assert "item_list=_build_moe_ep_phase_rows(" in SOURCE_TEXT


def test_no_silent_case_skipping_and_the_dry_run_raises():
    # failure_handling.md: a queued case is executed or raises a classified
    # error. The dry-run loop previously constructed a RuntimeError without
    # raising it, silently completing the case with zero rows.
    assert "skipping..." not in SOURCE_TEXT
    assert "raise MoeEpBenchmarkError" in SOURCE_TEXT
    assert 'RuntimeError(f"dry run failed' not in SOURCE_TEXT
    # Token points above the dry-run ceiling are dropped with a logged reason.
    assert "dropping num_tokens=" in SOURCE_TEXT


def test_sm120_drops_are_parked_as_a_kernel_limit():
    # layer_permissions.md: an unverified framework kernel limit lives as a
    # FIXME(kernel-limit) note at its site, re-audited on every version bump.
    assert "FIXME(kernel-limit)" in SOURCE_TEXT


def test_power_is_measured_or_absent_never_zero():
    # D7: the power columns go through _power_columns — absent when the run
    # does not measure power (never a present-null column, which would crash
    # load_moe_ep_data), NaN when the sampler yields nothing.
    assert "power_stats=_power_columns(power_stats)" in SOURCE_TEXT
    assert "power_stats=power_stats" not in SOURCE_TEXT


def test_perf_file_comes_from_the_registry_not_the_retired_table():
    assert "PerfFile.WIDEEP_MOE" not in SOURCE_TEXT
    assert "wideep_moe_perf.txt" not in SOURCE_TEXT
    assert "PerfFile.MOE_EP" in SOURCE_TEXT


def test_source_path_is_the_registered_module():
    assert SOURCE_PATH.exists()
    assert os.path.relpath(SOURCE_PATH, REPO_ROOT) == "collector/wideep/trtllm/collect_moe_compute.py"
