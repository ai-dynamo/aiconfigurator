# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for generator aggregators module."""

import pytest

from aiconfigurator.generator.aggregators import collect_generator_params


@pytest.mark.unit
class TestK8sHfHomeDefaulting:
    """Test k8s_hf_home auto-defaulting behavior."""

    @pytest.mark.parametrize(
        "k8s_model_cache,k8s_hf_home,expected_hf_home",
        [
            # Auto-default: model cache set, hf_home not set
            ("model-cache-pvc", None, "/workspace/model_cache"),
            # Explicit value: should not be overridden
            ("model-cache-pvc", "/custom/path", "/custom/path"),
            # No model cache: hf_home should remain empty
            (None, None, ""),
            # hf_home set without model cache: independent configuration
            (None, "/custom/hf/home", "/custom/hf/home"),
            # Empty string hf_home: treated as unset, triggers auto-default
            ("model-cache-pvc", "", "/workspace/model_cache"),
        ],
    )
    def test_k8s_hf_home_behavior(self, k8s_model_cache, k8s_hf_home, expected_hf_home):
        """Test various k8s_hf_home and k8s_model_cache combinations."""
        service = {"model_path": "test/model", "served_model_name": "test-model"}
        k8s = {"k8s_namespace": "dynamo"}

        if k8s_model_cache is not None:
            k8s["k8s_model_cache"] = k8s_model_cache
        if k8s_hf_home is not None:
            k8s["k8s_hf_home"] = k8s_hf_home

        result = collect_generator_params(service=service, k8s=k8s, backend="trtllm")

        assert result["K8sConfig"]["k8s_hf_home"] == expected_hf_home

    @pytest.mark.parametrize("backend", ["trtllm", "vllm", "sglang"])
    def test_k8s_hf_home_consistent_across_backends(self, backend):
        """k8s_hf_home defaulting should work consistently across all backends."""
        service = {"model_path": "test/model", "served_model_name": "test-model"}
        k8s = {"k8s_model_cache": "model-cache-pvc", "k8s_namespace": "dynamo"}

        result = collect_generator_params(service=service, k8s=k8s, backend=backend)

        assert result["K8sConfig"]["k8s_hf_home"] == "/workspace/model_cache"
