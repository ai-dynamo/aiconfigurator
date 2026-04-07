# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_helper_imports(monkeypatch):
    """Ensure the module can be imported without CUDA / SGLang installed."""
    import types

    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90  # default Hopper
    fake_helper.log_perf = lambda **kw: None
    fake_helper.benchmark_with_power = lambda **kw: None
    fake_helper._get_deepseek_model_path = lambda: "/fake"
    monkeypatch.setitem(__import__("sys").modules, "helper", fake_helper)


def _import_module():
    """Import collect_mla_module after mocking."""
    import importlib.util
    import sys

    mod_name = "collector.sglang.collect_mla_module"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        "collector/sglang/collect_mla_module.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestGetPrecisionCombos:
    def test_hopper_sm90(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            combos = mod._get_precision_combos("context")
        assert ("bfloat16", "bfloat16", "bfloat16") in combos
        assert ("bfloat16", "fp8", "bfloat16") in combos
        assert len(combos) == 2

    def test_ada_sm89_no_fp8(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=89):
            combos = mod._get_precision_combos("context")
        assert combos == [("bfloat16", "bfloat16", "bfloat16")]

    def test_blackwell_sm100(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            combos = mod._get_precision_combos("generation")
        assert ("bfloat16", "fp8", "bfloat16") in combos
        assert len(combos) == 2

    def test_no_phase_difference(self):
        """SGLang precision combos are the same for context and generation."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            ctx = mod._get_precision_combos("context")
            gen = mod._get_precision_combos("generation")
        assert ctx == gen


class TestGetBackends:
    def test_dsa_always_nsa(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_backends("dsa") == "nsa"
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_backends("dsa") == "nsa"

    def test_mla_hopper(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_backends("mla") == "fa3"

    def test_mla_blackwell(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_backends("mla") == "trtllm_mla"

    def test_mla_older(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=80):
            assert mod._get_backends("mla") == "flashinfer"


class TestGetContextTestCases:
    def test_filenames(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_context_test_cases("dsa")
        fnames = {c[6] for c in cases}
        assert fnames == {"dsa_context_module_perf.txt"}

    def test_memory_guard(self):
        """No test case exceeds batch_size * seq_len > 128K."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("mla"):
                assert case[0] * case[1] <= 128 * 1024

    def test_large_seq_guard(self):
        """seq_len >= 8192 only with batch_size <= 8."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("dsa"):
                if case[0] >= 8192:
                    assert case[1] <= 8

    def test_format_length(self):
        """Each test case has 7 elements."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("mla"):
                assert len(case) == 7


class TestGetGenerationTestCases:
    def test_filenames(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_generation_test_cases("mla")
        fnames = {c[6] for c in cases}
        assert fnames == {"mla_generation_module_perf.txt"}

    def test_memory_guard(self):
        """No test case exceeds batch_size * seq_len > 256K."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_generation_test_cases("dsa"):
                assert case[0] * case[1] <= 256 * 1024


class TestBuildModuleTestCases:
    def test_dsa_includes_both_models(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        model_paths = {c[7] for c in cases}
        assert "deepseek-ai/DeepSeek-V3.2" in model_paths
        assert "zai-org/GLM-5" in model_paths

    def test_mla_includes_v3_only(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("mla", "context")
        model_paths = {c[7] for c in cases}
        assert model_paths == {"deepseek-ai/DeepSeek-V3"}

    def test_format_length_10(self):
        """Each built test case has 10 elements (7 + model_path + attn_type + backend)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("dsa", "generation"):
                assert len(case) == 10
                assert case[8] == "dsa"
                assert case[9] is None  # DSA backend resolved at runtime

    def test_deduplication(self):
        """One entry per (model, precision_combo, head_num) — not per (batch, seq)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        # DSV3.2: 2 dtypes x 4 heads = 8, GLM-5: 2 dtypes x 3 heads (64 native, skip 128) = 6 -> 14
        assert len(cases) == 8 + 6

    def test_placeholder_seq_batch(self):
        """seq_len and batch_size are placeholders (0) — subprocess sweeps internally."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("mla", "context"):
                assert case[0] == 0  # seq_len placeholder
                assert case[1] == 0  # batch_size placeholder


class TestEntryPoints:
    def test_wideep_mla_context_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_wideep_mla_context_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_wideep_mla_generation_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_wideep_mla_generation_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_dsa_generation_returns_list(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod.get_dsa_generation_module_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0


class TestGetMlaBackendList:
    def test_hopper(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            assert mod._get_mla_backend_list() == ["flashinfer", "fa3"]

    def test_blackwell(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            assert mod._get_mla_backend_list() == ["trtllm_mla"]

    def test_older(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=80):
            assert mod._get_mla_backend_list() == ["flashinfer"]


class TestBuildWideepMlaTestCases:
    def test_format_length_10(self):
        """Each wideep MLA test case has 10 elements (9 + attention_backend)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert len(case) == 10
                assert case[8] == "mla"

    def test_context_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        fnames = {c[6] for c in cases}
        assert fnames == {"wideep_context_mla_perf.txt"}

    def test_generation_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("generation")
        fnames = {c[6] for c in cases}
        assert fnames == {"wideep_generation_mla_perf.txt"}

    def test_only_mla_models(self):
        """Wideep MLA only includes MLA-type models (DeepSeek-V3), not DSA."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        model_paths = {c[7] for c in cases}
        assert model_paths == {"deepseek-ai/DeepSeek-V3"}

    def test_sweeps_backends(self):
        """Hopper should sweep flashinfer and fa3 backends."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        backends = {c[9] for c in cases}
        assert backends == {"flashinfer", "fa3"}

    def test_single_precision_bfloat16(self):
        """All wideep MLA cases use bfloat16 precision (logged as fp8_block/fp8)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert case[3] == "bfloat16"  # kv_cache_dtype
                assert case[4] == "bfloat16"  # compute_dtype
                assert case[5] == "bfloat16"  # gemm_type
