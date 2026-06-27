# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import sys
import textwrap
import types
from inspect import Parameter, getsource, signature
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_helper_imports(monkeypatch):
    """Ensure the module can be imported without CUDA / SGLang installed."""

    fake_helper = types.ModuleType("helper")
    fake_helper.get_sm_version = lambda: 90  # default Hopper
    fake_helper.log_perf = lambda **kw: True
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

    def test_dsa_keeps_checkpoint_needed_for_each_native_quant(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=100):
            cases = mod._build_module_test_cases("dsa", "context")
        model_paths = {case[6] for case in cases}
        assert "deepseek-ai/DeepSeek-V3.2" in model_paths
        assert "zai-org/GLM-5" in model_paths
        assert "zai-org/GLM-5-FP8" in model_paths
        assert "nvidia/GLM-5-NVFP4" in model_paths

    def test_mla_includes_v3_family(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("mla", "context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {
            "deepseek-ai/DeepSeek-V3",
            "nvidia/DeepSeek-V3.1-NVFP4",
        }

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
        keys = {
            (
                mod._module_model_architecture(c[6]),
                mod._model_native_gemm_quant(c[6]),
                c[2],
                c[3],
                c[4],
                c[5],
                c[1],
                c[9],
                c[10],
            )
            for c in cases
        }
        assert len(cases) == len(keys)
        assert all(c[0] == 0 for c in cases)

    def test_dsa_context_is_sharded_by_batch_size(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        batch_sizes = {c[1] for c in cases}
        assert batch_sizes == set(mod.get_mla_module_sweep_spec("sglang").context_batch_sizes)

    def test_glm5_canonical_context_includes_fp8_kv_module_case(self):
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_module_test_cases("dsa", "context")
        assert any(
            c[6] == "zai-org/GLM-5" and c[3] == "fp8" and c[4] == "bfloat16" and c[5] == "bfloat16" for c in cases
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


class TestWideepModeRouting:
    @pytest.mark.parametrize("function_name", ["_run_prefill", "_run_decode"])
    def test_benchmark_only_reports_success_after_log_write(self, function_name):
        mod = _import_module()
        tree = ast.parse(textwrap.dedent(getsource(getattr(mod, function_name))))
        guarded_writes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Call)
            and isinstance(node.test.operand.func, ast.Name)
            and node.test.operand.func.id == "log_perf"
        ]

        assert len(guarded_writes) == 1
        assert any(
            isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "PerfLogWriteError"
            for node in ast.walk(guarded_writes[0])
        )

    def test_mode_defaults_are_false_and_worker_flag_is_keyword_only(self):
        mod = _import_module()

        for func in (
            mod.run_mla_module_worker,
            mod._run_mla_subprocess,
            mod.run_mla_module,
            mod.run_attention_torch,
            mod._run_prefill,
            mod._run_decode,
        ):
            assert signature(func).parameters["is_wideep_mla"].default is False
        assert signature(mod.run_mla_module_worker).parameters["is_wideep_mla"].kind is Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize(
        ("is_wideep_mla", "is_prefill", "expected"),
        [
            (False, True, ("mla_context_module_perf.txt", "mla_context_module", "mla_fa3", 128, 64)),
            (True, True, ("wideep_context_mla_perf.txt", "mla_context", "fa3", 128, 64)),
            (False, False, ("mla_generation_module_perf.txt", "mla_generation_module", "mla_fa3", 1, 128)),
            (True, False, ("wideep_generation_mla_perf.txt", "mla_generation", "fa3", 128, 0)),
        ],
    )
    def test_regular_and_wideep_mla_use_distinct_perf_schemas(self, is_wideep_mla, is_prefill, expected):
        mod = _import_module()

        assert (
            mod._mla_module_perf_fields(
                is_wideep_mla=is_wideep_mla,
                is_prefill=is_prefill,
                attn_type="mla",
                backend_name="fa3",
                seq_length=128,
                prefix_len=64,
            )
            == expected
        )

    @pytest.mark.parametrize(
        ("is_wideep_mla", "expected_dtypes"),
        [
            (False, ("float16", "bfloat16", "nvfp4")),
            (True, ("fp8_block", "fp8", "fp8_block")),
        ],
    )
    def test_attention_runner_forwards_explicit_mode_and_log_dtypes(self, is_wideep_mla, expected_dtypes):
        mod = _import_module()
        attention_module = types.SimpleNamespace(q_lora_rank=1536, kv_lora_rank=512, qk_rope_head_dim=64)
        runner = types.SimpleNamespace(
            model=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attention_module)])
            ),
            server_args=types.SimpleNamespace(attention_backend="fa3"),
            req_to_token_pool=types.SimpleNamespace(clear=lambda: None),
            token_to_kv_pool_allocator=types.SimpleNamespace(clear=lambda: None),
        )
        captured = {}

        def fake_run_prefill(**kwargs):
            captured.update(kwargs)
            return True

        with (
            patch.object(mod, "_module_model_architecture", return_value="DeepseekV3ForCausalLM"),
            patch.object(mod, "get_version", return_value="0.5.0"),
            patch.object(mod, "_run_prefill", side_effect=fake_run_prefill),
        ):
            logged_count = mod.run_attention_torch(
                model_runner=runner,
                test_cases=[(1, 128, True, 64)],
                head_num=64,
                test_layer=0,
                num_warmup=1,
                num_iterations=1,
                device="cuda:0",
                output_path=None,
                attn_type="mla",
                model_path="deepseek-ai/DeepSeek-V3",
                kv_cache_dtype="bfloat16",
                compute_dtype="float16",
                gemm_type="nvfp4",
                is_wideep_mla=is_wideep_mla,
            )

        assert logged_count == 1
        assert captured["is_wideep_mla"] is is_wideep_mla
        assert (
            captured["log_mla_dtype"],
            captured["log_kv_dtype"],
            captured["log_gemm_type"],
        ) == expected_dtypes

    def test_base_worker_forwards_default_regular_mode(self):
        mod = _import_module()

        with patch.object(mod, "_run_mla_subprocess") as run_subprocess:
            mod.run_mla_module_worker(
                0,
                0,
                64,
                "bfloat16",
                "bfloat16",
                "bfloat16",
                "deepseek-ai/DeepSeek-V3",
                "mla",
                "fa3",
                perf_filename="mla_context_module_perf.txt",
            )

        assert run_subprocess.call_args.kwargs["is_wideep_mla"] is False

    def test_wideep_wrapper_forces_wideep_mode(self):
        from collector.wideep.sglang import collect_mla_module as wideep_module

        with patch.object(wideep_module, "_run_mla_module_worker", return_value="ok") as base_worker:
            result = wideep_module.run_mla_module_worker("case", is_wideep_mla=False)

        assert result == "ok"
        assert base_worker.call_args.kwargs["is_wideep_mla"] is True

    @pytest.mark.parametrize("is_wideep_mla", [False, True])
    def test_subprocess_command_preserves_explicit_mode(self, is_wideep_mla):
        mod = _import_module()
        process = types.SimpleNamespace(
            returncode=0,
            communicate=lambda timeout=None: (b"", None),
        )

        with patch.object(mod.subprocess, "Popen", return_value=process) as popen:
            mod._run_mla_subprocess(
                attn_type="mla",
                head_num=64,
                model_path="deepseek-ai/DeepSeek-V3",
                kv_cache_dtype="bfloat16",
                compute_dtype="bfloat16",
                gemm_type="bfloat16",
                is_prefill=True,
                gpu_id=0,
                is_wideep_mla=is_wideep_mla,
            )

        command = popen.call_args.args[0]
        assert f"is_wideep_mla={is_wideep_mla!r}" in command[2]

    def test_module_runner_forwards_wideep_mode_to_attention_runner(self):
        mod = _import_module()
        cases = [[128, 1, 64, "bfloat16", "bfloat16", "bfloat16"]]

        with (
            patch.object(mod, "get_context_test_cases", return_value=cases),
            patch.object(mod, "required_kv_tokens", return_value=128),
            patch.object(mod, "cleanup_distributed"),
            patch.object(mod, "load_model_runner", return_value=object()),
            patch.object(mod, "run_attention_torch", return_value=1) as run_attention,
        ):
            mod.run_mla_module(
                attn_type="mla",
                head_num=64,
                model_path="deepseek-ai/DeepSeek-V3",
                kv_cache_dtype="bfloat16",
                compute_dtype="bfloat16",
                gemm_type="bfloat16",
                is_prefill=True,
                gpu_id=0,
                attention_backend="fa3",
                is_wideep_mla=True,
            )

        assert run_attention.call_args.kwargs["is_wideep_mla"] is True


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
        """Wideep MLA only includes MLA-type models, not DSA."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        model_paths = {c[6] for c in cases}
        assert model_paths == {
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "nvidia/DeepSeek-V3.1-NVFP4",
        }

    def test_sweeps_backends(self):
        """Hopper should sweep flashinfer and fa3 backends."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            cases = mod._build_wideep_mla_test_cases("context")
        backends = {c[8] for c in cases}
        assert backends == {"flashinfer", "fa3"}

    def test_single_precision_bfloat16(self):
        """All wideep MLA cases use bfloat16 precision (logged as fp8_block/fp8)."""
        mod = _import_module()
        with patch.object(mod, "get_sm_version", return_value=90):
            for case in mod._build_wideep_mla_test_cases("context"):
                assert case[3] == "bfloat16"  # kv_cache_dtype
                assert case[4] == "bfloat16"  # compute_dtype
                assert case[5] == "bfloat16"  # gemm_type
