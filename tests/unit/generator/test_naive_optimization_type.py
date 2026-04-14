# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for optimization-type-aware parallelization in generator/naive.py."""

from typing import ClassVar
from unittest.mock import patch

import pytest

from aiconfigurator.generator.naive import (
    _resolve_parallelization,
    build_naive_generator_params,
)

# ---------------------------------------------------------------------------
# _resolve_parallelization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveParallelization:
    """Verify _resolve_parallelization selects TP / TEP / DEP correctly."""

    def test_dense_model_uses_tp(self):
        result = _resolve_parallelization(
            architecture="LlamaForCausalLM",
            is_moe=False,
            num_gpus=4,
        )
        assert result["tensor_parallel_size"] == 4
        assert result["pipeline_parallel_size"] == 1
        assert result["data_parallel_size"] == 1
        assert result["moe_tensor_parallel_size"] == 1
        assert result["moe_expert_parallel_size"] == 1

    def test_dense_model_ignores_optimization_type(self):
        for opt in ("throughput", "latency", None):
            r = _resolve_parallelization("LlamaForCausalLM", False, 8, opt)
            assert r["tensor_parallel_size"] == 8
            assert r["moe_tensor_parallel_size"] == 1

    def test_moe_non_mla_uses_tep(self):
        """Mixtral-style MoE should use TEP (moe_tensor_parallel_size = num_gpus)."""
        result = _resolve_parallelization(
            architecture="MixtralForCausalLM",
            is_moe=True,
            num_gpus=8,
            optimization_type="throughput",
        )
        assert result["tensor_parallel_size"] == 1
        assert result["moe_tensor_parallel_size"] == 8
        assert result["moe_expert_parallel_size"] == 1

    def test_mla_moe_throughput_uses_dep(self):
        """DeepSeek-V3 + throughput should use DEP (data_parallel + moe_expert_parallel)."""
        result = _resolve_parallelization(
            architecture="DeepseekV3ForCausalLM",
            is_moe=True,
            num_gpus=8,
            optimization_type="throughput",
        )
        assert result["tensor_parallel_size"] == 1
        assert result["data_parallel_size"] == 8
        assert result["moe_expert_parallel_size"] == 8
        assert result["moe_tensor_parallel_size"] == 1

    def test_mla_moe_latency_uses_tep(self):
        """DeepSeek-V3 + latency should fall through to TEP."""
        result = _resolve_parallelization(
            architecture="DeepseekV3ForCausalLM",
            is_moe=True,
            num_gpus=8,
            optimization_type="latency",
        )
        assert result["tensor_parallel_size"] == 1
        assert result["moe_tensor_parallel_size"] == 8
        assert result["moe_expert_parallel_size"] == 1

    def test_mla_moe_no_optimization_type_uses_tep(self):
        """DeepSeek-V3 with no optimization_type → TEP (safe default)."""
        result = _resolve_parallelization(
            architecture="DeepseekV3ForCausalLM",
            is_moe=True,
            num_gpus=8,
            optimization_type=None,
        )
        assert result["moe_tensor_parallel_size"] == 8
        assert result["data_parallel_size"] == 1

    def test_deepseek_v3_variant_recognized_as_mla_moe(self):
        """DeepseekV32ForCausalLM is a V3-family architecture."""
        result = _resolve_parallelization(
            architecture="DeepseekV32ForCausalLM",
            is_moe=True,
            num_gpus=4,
            optimization_type="throughput",
        )
        assert result["data_parallel_size"] == 4  # DEP path
        assert result["moe_expert_parallel_size"] == 4

    def test_qwen2moe_uses_tep(self):
        """Qwen2MoeForCausalLM is MoE but NOT MLA, so TEP even with throughput."""
        result = _resolve_parallelization(
            architecture="Qwen2MoeForCausalLM",
            is_moe=True,
            num_gpus=4,
            optimization_type="throughput",
        )
        assert result["moe_tensor_parallel_size"] == 4
        assert result["data_parallel_size"] == 1


# ---------------------------------------------------------------------------
# build_naive_generator_params — mode and optimization_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildNaiveOptimizationType:
    """Test build_naive_generator_params with mode and optimization_type args."""

    _SYS: ClassVar[dict] = {"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3}
    _WEIGHT = 30 * 1024**3  # 30 GiB -> fits in 1 GPU

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_default_mode_is_agg(self, _sys, _est):
        """Calling without mode/optimization_type preserves backward compat."""
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["DynConfig"]["mode"] == "agg"
        assert "agg" in result["params"]

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_disagg_mode_creates_prefill_decode(self, _sys, _est):
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
            mode="disagg",
        )
        assert result["DynConfig"]["mode"] == "disagg"
        assert "prefill" in result["params"]
        assert "decode" in result["params"]
        assert "agg" not in result["params"]

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_disagg_prefill_decode_have_same_parallelization(self, _sys, _est):
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
            mode="disagg",
        )
        p = result["params"]["prefill"]
        d = result["params"]["decode"]
        assert p["tensor_parallel_size"] == d["tensor_parallel_size"]
        assert p["moe_tensor_parallel_size"] == d["moe_tensor_parallel_size"]
        assert p["moe_expert_parallel_size"] == d["moe_expert_parallel_size"]

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_agg_worker_params_contain_parallelization_keys(self, _sys, _est):
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        agg = result["params"]["agg"]
        for key in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "moe_tensor_parallel_size",
            "moe_expert_parallel_size",
            "gpus_per_worker",
            "max_batch_size",
        ):
            assert key in agg, f"missing key {key} in agg worker params"

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_disagg_worker_config_has_prefill_decode_counts(self, _sys, _est):
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
            mode="disagg",
        )
        wc = result["WorkerConfig"]
        assert "prefill_workers" in wc
        assert "decode_workers" in wc
        assert "prefill_gpus_per_worker" in wc
        assert "decode_gpus_per_worker" in wc

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_optimization_type_passed_through(self, _sys, _est):
        """With optimization_type set, parallelization should still be valid."""
        for opt in ("throughput", "latency"):
            result = build_naive_generator_params(
                model_name="test/model",
                total_gpus=8,
                system_name="h200_sxm",
                backend_name="vllm",
                optimization_type=opt,
            )
            agg = result["params"]["agg"]
            assert agg["tensor_parallel_size"] >= 1
            assert agg["gpus_per_worker"] >= 1

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=300 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_model_too_large_for_one_gpu_selects_higher_tp(self, _sys, _est):
        result = build_naive_generator_params(
            model_name="test/model",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        agg = result["params"]["agg"]
        assert agg["gpus_per_worker"] > 1
