# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_helper_imports(monkeypatch):
    """Ensure the module can be imported without CUDA / SGLang installed."""

    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90  # default Hopper
    fake_helper.log_perf = lambda **kw: None
    fake_helper.benchmark_with_power = lambda **kw: None
    monkeypatch.setitem(__import__("sys").modules, "helper", fake_helper)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        get_device_capability=lambda *_a, **_kw: (9, 0),
        get_device_name=lambda *_a, **_kw: "Fake GPU",
        is_available=lambda: False,
        set_device=lambda *_a, **_kw: None,
    )
    fake_torch.randn = lambda *_a, **_kw: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def _import_module():
    """Import collect_mla_module after mocking."""
    import importlib.util

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
        assert ("bfloat16", "bfloat16", "nvfp4") in combos
        assert ("bfloat16", "fp8", "nvfp4") in combos
        assert len(combos) == 4

    def test_ada_sm89_no_fp8(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=89):
            combos = mod._get_precision_combos("context")
        assert combos == [
            ("bfloat16", "bfloat16", "bfloat16"),
            ("bfloat16", "bfloat16", "nvfp4"),
        ]

    def test_blackwell_sm100(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            combos = mod._get_precision_combos("generation")
        assert ("bfloat16", "fp8", "bfloat16") in combos
        assert len(combos) == 4

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
            assert mod._get_backends("mla") == "triton"


class TestGetContextTestCases:
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
        """Each inner sweep test case has 6 elements."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_context_test_cases("mla"):
                assert len(case) == 6


class TestGetGenerationTestCases:
    def test_memory_guard(self):
        """No test case exceeds batch_size * seq_len > 256K."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod.get_generation_test_cases("dsa"):
                assert case[0] * case[1] <= 256 * 1024


class TestDsaContextPrefixShape:
    def test_rejects_single_token_prefill(self):
        mod = _import_module()
        assert not mod._dsa_context_prefix_shape_is_valid(1, 1, 0)
        assert not mod._dsa_context_prefix_shape_is_valid(32, 1, 1024)

    def test_accepts_multi_token_prefill_with_prefix(self):
        mod = _import_module()
        assert mod._dsa_context_prefix_shape_is_valid(1, 2, 0)
        assert mod._dsa_context_prefix_shape_is_valid(32, 4, 1024)

    def test_rejects_negative_prefix(self):
        mod = _import_module()
        assert not mod._dsa_context_prefix_shape_is_valid(1, 2, -1)


class TestDsaRuntimeLimits:
    def test_total_kv_token_limit_matches_indexer_offset_capacity(self):
        from collector.sglang.runtime_limits import dsa_indexer_total_kv_tokens_supported

        assert dsa_indexer_total_kv_tokens_supported(32, 4, 1_000_000, is_prefill=True)
        assert not dsa_indexer_total_kv_tokens_supported(32, 4, 1_048_575, is_prefill=True)
        assert not dsa_indexer_total_kv_tokens_supported(64, 4, 524_288, is_prefill=True)


class TestDsaPiecewiseCudaGraph:
    def test_glm5_dsa_piecewise_graph_is_only_for_trtllm_prefill(self):
        mod = _import_module()

        assert not mod._enable_glm5_dsa_piecewise_graph("dsa", "nvidia/GLM-5-NVFP4")
        assert mod._enable_glm5_dsa_piecewise_graph("dsa", "zai-org/GLM-5", "trtllm")
        assert not mod._enable_glm5_dsa_piecewise_graph("mla", "zai-org/GLM-5", "trtllm")
        assert not mod._enable_glm5_dsa_piecewise_graph("dsa", "deepseek-ai/DeepSeek-V3.2", "trtllm")

    def test_generation_cuda_graph_uses_max_batch_tokens(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disable_cuda_graph=False, cuda_graph_bs=None, cuda_graph_max_bs=256)
        )

        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 256)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 257)

    def test_generation_cuda_graph_uses_explicit_batch_list(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disable_cuda_graph=False, cuda_graph_bs=[1, 4, 16])
        )

        assert mod._generation_cuda_graph_enabled_for_tokens(runner, 4)
        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 8)

    def test_generation_cuda_graph_can_be_disabled(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            server_args=types.SimpleNamespace(disable_cuda_graph=True, cuda_graph_bs=[1, 4, 16])
        )

        assert not mod._generation_cuda_graph_enabled_for_tokens(runner, 4)


class TestBuildModuleTestCases:
    def test_module_precision_includes_fp8_kv_for_sglang(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            combos = mod._get_module_precision_combos()
        assert ("bfloat16", "bfloat16", "bfloat16") in combos
        assert ("bfloat16", "fp8", "bfloat16") in combos

    def test_dsa_includes_both_models(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        model_paths = {c[6] for c in cases}
        assert "deepseek-ai/DeepSeek-V3.2" in model_paths
        assert "zai-org/GLM-5" in model_paths
        assert "nvidia/GLM-5-NVFP4" in model_paths

    def test_mla_includes_v3_family(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("mla", "context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {"deepseek-ai/DeepSeek-V3"}

    def test_format_length_11(self):
        """Each DSA module case includes target TP and prefill backend."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("dsa", "generation"):
                assert len(case) == 11
                assert case[7] == "dsa"
                assert case[8] is None  # DSA backend resolved at runtime
                assert case[9] in {1, 2, 4, 8}
                assert case[10] == "flashmla_kv"

    def test_deduplication(self):
        """One entry per top-level sweep tuple, not per inner (seq, batch) shape."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        keys = {(c[6], c[2], c[3], c[4], c[5], c[1], c[9], c[10]) for c in cases}
        assert len(cases) == len(keys)
        assert all(c[0] == 0 for c in cases)

    def test_dsa_context_is_sharded_by_batch_size(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        batch_sizes = {c[1] for c in cases}
        assert batch_sizes == set(mod.get_mla_module_sweep_spec("sglang").context_batch_sizes)

    def test_glm5_nvfp4_context_includes_fp8_kv_module_case(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        assert any(
            c[6] == "nvidia/GLM-5-NVFP4" and c[3] == "fp8" and c[4] == "bfloat16" and c[5] == "bfloat16" for c in cases
        )

    def test_placeholder_seq_batch(self):
        """seq_len and batch_size are placeholders (0) — subprocess sweeps internally."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_module_test_cases("mla", "context"):
                assert case[0] == 0  # seq_len placeholder
                assert case[1] == 0  # batch_size placeholder


class TestDsaTpShapeValidation:
    def _runner(self, attn):
        return types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)]))
        )

    def test_accepts_local_rank_projection_shapes(self):
        mod = _import_module()
        attn = types.SimpleNamespace(
            num_heads=8,
            num_local_heads=8,
            qk_nope_head_dim=384,
            qk_rope_head_dim=128,
            v_head_dim=256,
            hidden_size=6144,
            q_b_proj=types.SimpleNamespace(output_size=4096, output_size_per_partition=4096),
            kv_b_proj=types.SimpleNamespace(output_size=5120, output_size_per_partition=5120),
            o_proj=types.SimpleNamespace(input_size=2048, input_size_per_partition=2048, output_size=6144),
        )
        mod._validate_dsa_tp_module_shapes(self._runner(attn), local_num_heads=8, target_tp_size=8)

    def test_rejects_native_projection_shapes_for_tp_rank(self):
        mod = _import_module()
        attn = types.SimpleNamespace(
            num_heads=8,
            num_local_heads=8,
            qk_nope_head_dim=384,
            qk_rope_head_dim=128,
            v_head_dim=256,
            hidden_size=6144,
            q_b_proj=types.SimpleNamespace(output_size=64 * 512, output_size_per_partition=64 * 512),
            kv_b_proj=types.SimpleNamespace(output_size=8 * (384 + 256), output_size_per_partition=8 * (384 + 256)),
            o_proj=types.SimpleNamespace(input_size=8 * 256, input_size_per_partition=8 * 256, output_size=6144),
        )
        with pytest.raises(RuntimeError, match=r"q_b_proj\.output_size"):
            mod._validate_dsa_tp_module_shapes(self._runner(attn), local_num_heads=8, target_tp_size=8)


class TestPerfLogFailures:
    def test_log_failure_is_wrapped_with_case_context(self):
        mod = _import_module()

        with (
            patch.object(mod, "log_perf", side_effect=OSError("disk full")),
            pytest.raises(mod.PerfLogWriteError, match=r"prefill metrics for b=2, s=128: disk full") as exc_info,
        ):
            mod._log_perf_strict(
                phase="prefill",
                batch_size=2,
                seq_length=128,
                item_list=[],
                perf_filename="unused.txt",
            )

        assert isinstance(exc_info.value.__cause__, OSError)

    def test_partial_sweep_log_failure_propagates(self):
        mod = _import_module()
        runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace())])
            ),
            server_args=types.SimpleNamespace(attention_backend="fake_backend"),
            req_to_token_pool=types.SimpleNamespace(clear=lambda: None),
            token_to_kv_pool_allocator=types.SimpleNamespace(clear=lambda: None),
        )
        log_error = mod.PerfLogWriteError("failed to persist second row")

        with (
            patch.object(mod, "get_version", return_value="test"),
            patch.object(mod, "_run_prefill", side_effect=[True, log_error]) as run_prefill,
            pytest.raises(mod.PerfLogWriteError, match="failed to persist second row"),
        ):
            mod.run_attention_torch(
                model_runner=runner,
                test_cases=[(1, 128, True), (2, 128, True)],
                head_num=8,
                test_layer=0,
                num_warmup=0,
                num_iterations=1,
                device="cuda:0",
                output_path=None,
                attn_type="dsa",
                model_path="test/model",
                kv_cache_dtype="bfloat16",
                compute_dtype="bfloat16",
                gemm_type="bfloat16",
            )

        assert run_prefill.call_count == 2


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
            assert mod._get_mla_backend_list() == []


class TestBuildWideepMlaTestCases:
    def test_format_length_10(self):
        """Each wideep MLA test case has 9 elements (6 + model_path + attn_type + backend)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert len(case) == 9
                assert case[7] == "mla"

    def test_context_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        assert {c[7] for c in cases} == {"mla"}

    def test_generation_filename(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("generation")
        assert {c[7] for c in cases} == {"mla"}

    def test_only_mla_models(self):
        """WideEP uses one canonical checkpoint, not its artifact aliases."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {"deepseek-ai/DeepSeek-V3"}

    def test_sweeps_backends(self):
        """Hopper should sweep flashinfer and fa3 backends."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        backends = {c[8] for c in cases}
        assert backends == {"flashinfer", "fa3"}

    def test_uses_real_checkpoint_and_kv_precisions(self):
        """DeepSeek-V3 runs native block-FP8 GEMMs with both real KV dtypes."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        assert {case[3] for case in cases} == {"bfloat16", "fp8"}
        assert {case[4] for case in cases} == {"bfloat16"}
        assert {case[5] for case in cases} == {"fp8_block"}

    def test_blackwell_kernel_source_matches_query_key(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_wideep_mla_test_cases("generation")
        assert {case[8] for case in cases} == {"trtllm_mla"}
