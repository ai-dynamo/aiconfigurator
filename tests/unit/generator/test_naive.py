# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generator naive module — nvbug 5941223."""

import re
from unittest.mock import patch

import pytest

from aiconfigurator.generator.naive import (
    _calculate_min_tp,
    _estimate_model_weight_bytes,
    _sanitize_rfc1123,
    build_naive_generator_params,
)

_RFC1123_LABEL_RE = re.compile(r"^[a-z0-9]([a-z0-9\-.]*[a-z0-9])?$")


@pytest.mark.unit
class TestSanitizeRfc1123:
    """Verify _sanitize_rfc1123 produces valid RFC 1123 subdomain labels."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Qwen/Qwen3-32B", "qwen-qwen3-32b"),
            ("meta-llama/Llama-3.1-70B", "meta-llama-llama-3.1-70b"),
            ("deepseek-ai/DeepSeek-V3", "deepseek-ai-deepseek-v3"),
            ("simple-model", "simple-model"),
            ("ALLCAPS", "allcaps"),
            ("a" * 100, "a" * 63),
        ],
    )
    def test_known_models(self, raw, expected):
        assert _sanitize_rfc1123(raw) == expected

    @pytest.mark.parametrize("bad_input", [None, "", "---", "///"])
    def test_fallback_to_dynamo(self, bad_input):
        assert _sanitize_rfc1123(bad_input) == "dynamo"

    @pytest.mark.parametrize(
        "raw",
        [
            "Qwen/Qwen3-32B",
            "meta-llama/Llama-3.1-70B",
            "nvidia/Nemotron-4-340B",
            "a",
            "a-b.c",
        ],
    )
    def test_result_matches_rfc1123(self, raw):
        result = _sanitize_rfc1123(raw)
        assert _RFC1123_LABEL_RE.match(result), f"{result!r} is not RFC 1123 compliant"
        assert len(result) <= 63


@pytest.mark.unit
class TestBuildNaiveGeneratorParams:
    """Verify build_naive_generator_params produces correct keys for the rendering engine."""

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_uses_service_config_and_k8s_config_keys(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert "ServiceConfig" in result, "expected ServiceConfig key, got 'service'"
        assert "K8sConfig" in result, "expected K8sConfig key, got 'k8s'"
        assert "service" not in result
        assert "k8s" not in result

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_name_prefix_is_rfc1123_valid(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        prefix = result["K8sConfig"]["name_prefix"]
        assert prefix is not None
        assert _RFC1123_LABEL_RE.match(prefix), f"{prefix!r} is not RFC 1123 compliant"

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_model_path_propagated(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["ServiceConfig"]["model_path"] == "Qwen/Qwen3-32B"
        assert result["ServiceConfig"]["model_name"] == "Qwen/Qwen3-32B"

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_agg_mode_set(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["DynConfig"]["mode"] == "agg"

    @patch(
        "aiconfigurator.generator.naive._estimate_model_weight_bytes",
        return_value=30 * 1024**3,
    )
    @patch(
        "aiconfigurator.generator.naive._get_system_config",
        return_value={"gpus_per_node": 8, "vram_per_gpu": 141 * 1024**3},
    )
    def test_include_frontend_true(self, _mock_sys, _mock_est):
        result = build_naive_generator_params(
            model_name="Qwen/Qwen3-32B",
            total_gpus=8,
            system_name="h200_sxm",
            backend_name="vllm",
        )
        assert result["ServiceConfig"]["include_frontend"] is True


@pytest.mark.unit
class TestEstimateModelWeightBytesFailsOnMissingModel:
    @patch("aiconfigurator.sdk.utils.get_model_config_from_model_path")
    def test_raises_when_config_download_fails(self, mock_get_config):
        mock_get_config.side_effect = Exception(
            "Failed to download nonexistent-org/fake-model-12345's config.json from HuggingFace: "
            "HuggingFace returned HTTP error 401: Unauthorized."
        )
        with pytest.raises(RuntimeError, match=r"Model .* not found or config unavailable"):
            _estimate_model_weight_bytes("nonexistent-org/fake-model-12345")

    @patch("aiconfigurator.generator.naive._get_system_config")
    @patch("aiconfigurator.generator.naive._estimate_model_weight_bytes")
    def test_build_naive_generator_params_propagates_model_not_found(self, mock_est, _mock_sys):
        mock_est.side_effect = RuntimeError("Model 'nonexistent-org/fake-model-12345' not found or config unavailable")
        with pytest.raises(RuntimeError, match="not found or config unavailable"):
            build_naive_generator_params(
                model_name="nonexistent-org/fake-model-12345",
                total_gpus=8,
                system_name="h200_sxm",
                backend_name="vllm",
            )


@pytest.mark.unit
class TestCalculateMinTp:
    """Verify _calculate_min_tp memory-fit floor, including multi-node sweeps."""

    # H100: 80 GiB per GPU.  DeepSeek R1 FP8: ~671 GiB of weights.
    _H100_VRAM = 80 * 1024**3
    _R1_FP8_WEIGHTS = 671 * 1024**3

    def test_single_node_caps_at_gpus_per_node(self):
        """Dense default: min_tp capped at gpus_per_node even if model is larger."""
        tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=32,
        )
        assert tp == 8  # capped at node even though model wants more

    def test_multi_node_allows_crossing_node_boundary(self):
        """MoE wide-EP: min_tp can span nodes, floored to power-of-2 fit."""
        tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=32,
            allow_multi_node=True,
        )
        # 1.5 * 671 / 80 = ~12.6 -> 13 -> round to 16
        assert tp == 16

    def test_multi_node_capped_by_total_gpus(self):
        """Even in multi-node mode, result cannot exceed total_gpus budget."""
        tp = _calculate_min_tp(
            model_weight_bytes=self._R1_FP8_WEIGHTS,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=8,
            allow_multi_node=True,
        )
        assert tp == 8  # clamped to budget

    def test_small_model_fits_on_one_gpu(self):
        tp = _calculate_min_tp(
            model_weight_bytes=10 * 1024**3,
            vram_per_gpu=self._H100_VRAM,
            gpus_per_node=8,
            total_gpus=8,
        )
        assert tp == 1


@pytest.mark.unit
class TestRenderingNameFallback:
    """Verify prepare_template_context uses 'dynamo' fallback for missing name_prefix."""

    def test_none_name_prefix_becomes_dynamo(self):
        from aiconfigurator.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {},
            "ServiceConfig": {"model_path": "test"},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["name_prefix"] == "dynamo"
        assert ctx["name"] == "dynamo-agg"

    def test_include_frontend_yields_one_replica(self):
        from aiconfigurator.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {"name_prefix": "test"},
            "ServiceConfig": {"model_path": "test", "include_frontend": True},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["frontend_replicas"] == 1

    def test_no_include_frontend_yields_zero_replica(self):
        from aiconfigurator.generator.rendering.engine import prepare_template_context

        params = {
            "K8sConfig": {"name_prefix": "test"},
            "ServiceConfig": {"model_path": "test"},
            "DynConfig": {"mode": "agg"},
            "params": {"agg": {}},
            "WorkerConfig": {},
        }
        ctx = prepare_template_context(params, "vllm")
        assert ctx["frontend_replicas"] == 0
