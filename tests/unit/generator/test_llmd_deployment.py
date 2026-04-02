# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llm-d deployment target support."""

import pytest

from aiconfigurator.generator.rendering.engine import render_backend_templates


@pytest.mark.unit
def test_llmd_values_template_renders():
    """Test that llm-d values template renders correctly for disaggregated mode."""
    # Set up context exactly as the template expects
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "LlmdConfig": {
            "vllm_image": "vllm/vllm-openai:v0.6.0",
            "model_cache_size": "50Gi",
            "routing_proxy_enabled": True,
            "multinode": False,
        },
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        # Parallelism settings (used directly by template)
        "prefill_tensor_parallel_size": 2,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 4,
        "decode_data_parallel_size": 1,
        # CLI args (used directly by template)
        "prefill_cli_args_list": ["--tensor-parallel-size", "2", "--max-num-seqs", "1"],
        "decode_cli_args_list": ["--tensor-parallel-size", "4", "--max-num-seqs", "128"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d")

    assert "llm-d-values.yaml" in artifacts
    assert "k8s_deploy.yaml" not in artifacts  # Should not render Dynamo manifest

    values_content = artifacts["llm-d-values.yaml"]

    # Verify key sections are present
    assert "modelArtifacts:" in values_content
    assert "name: meta-llama/Llama-3.1-8B" in values_content
    assert "uri: hf://meta-llama/Llama-3.1-8B" in values_content
    assert "size: 50Gi" in values_content

    # Verify disaggregated mode
    assert "multinode: False" in values_content or "multinode: false" in values_content
    assert "decode:" in values_content
    assert "prefill:" in values_content
    assert "replicas: 2" in values_content  # prefill replicas
    assert "replicas: 4" in values_content  # decode replicas

    # Verify parallelism
    assert "tensor: 2" in values_content  # prefill TP
    assert "tensor: 4" in values_content  # decode TP

    # Verify vLLM image
    assert "image: vllm/vllm-openai:v0.6.0" in values_content


@pytest.mark.unit
def test_llmd_values_template_aggregated_mode():
    """Test that llm-d values template renders correctly for aggregated mode."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "served_model_name": "qwen3-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {},
        "WorkerConfig": {
            "agg_workers": 4,
            "agg_gpus_per_worker": 2,
        },
        "params": {
            "agg": {"gpus_per_worker": 2},
        },
        # Parallelism settings
        "agg_tensor_parallel_size": 2,
        "agg_data_parallel_size": 1,
        # CLI args
        "agg_cli_args_list": ["--tensor-parallel-size", "2"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="llm-d")

    values_content = artifacts["llm-d-values.yaml"]

    # Verify aggregated mode
    assert "multinode: false" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "prefill:" in values_content
    assert "create: false" in values_content  # prefill should be disabled in agg mode
    assert "replicas: 4" in values_content  # agg workers
    assert "tensor: 2" in values_content  # TP


@pytest.mark.unit
def test_llmd_sglang_disagg_mode():
    """Test that sglang backend works with llm-d deployment target in disagg mode."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-32B",
            "served_model_name": "qwen3-32b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "LlmdConfig": {
            "sglang_image": "lmsysorg/sglang:v0.3.0",
            "model_cache_size": "200Gi",
            "routing_proxy_enabled": True,
            "multinode": False,
        },
        "WorkerConfig": {
            "prefill_workers": 1,
            "decode_workers": 2,
            "prefill_gpus_per_worker": 4,
            "decode_gpus_per_worker": 8,
        },
        "params": {
            "prefill": {"gpus_per_worker": 4},
            "decode": {"gpus_per_worker": 8},
        },
        "prefill_tensor_parallel_size": 4,
        "prefill_data_parallel_size": 1,
        "decode_tensor_parallel_size": 8,
        "decode_data_parallel_size": 1,
        "prefill_cli_args_list": ["--tp-size", "4", "--mem-fraction-static", "0.9"],
        "decode_cli_args_list": ["--tp-size", "8", "--mem-fraction-static", "0.9"],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d")

    assert "llm-d-values.yaml" in artifacts
    assert "k8s_deploy.yaml" not in artifacts  # Should not render Dynamo manifest

    values_content = artifacts["llm-d-values.yaml"]

    # Verify sglang-specific content
    assert "image: lmsysorg/sglang:v0.3.0" in values_content
    assert "modelCommand: sglangServe" in values_content
    assert "size: 200Gi" in values_content
    assert "name: sglang" in values_content

    # Verify disaggregated mode structure
    assert "multinode: False" in values_content or "multinode: false" in values_content
    assert "prefill:" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "replicas: 1" in values_content  # prefill
    assert "replicas: 2" in values_content  # decode

    # Verify parallelism
    assert "tensor: 4" in values_content  # prefill TP
    assert "tensor: 8" in values_content  # decode TP

    # Verify model configuration
    assert "model-path" in values_content
    assert "Qwen/Qwen3-32B" in values_content


@pytest.mark.unit
def test_llmd_sglang_agg_mode():
    """Test that sglang backend works with llm-d deployment target in agg mode."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-70B",
            "served_model_name": "llama-3.1-70b",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {
            "sglang_image": "lmsysorg/sglang:latest",
            "model_cache_size": "300Gi",
        },
        "WorkerConfig": {
            "agg_workers": 2,
            "agg_gpus_per_worker": 8,
        },
        "params": {
            "agg": {"gpus_per_worker": 8},
        },
        "agg_tensor_parallel_size": 8,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": ["--tp-size", "8", "--context-length", "8192"],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d")

    assert "llm-d-values.yaml" in artifacts
    values_content = artifacts["llm-d-values.yaml"]

    # Verify aggregated mode
    assert "multinode: false" in values_content
    assert "decode:" in values_content
    assert "create: true" in values_content
    assert "replicas: 2" in values_content  # agg workers
    assert "prefill:" in values_content
    assert "create: false" in values_content  # prefill should be disabled in agg mode

    # Verify parallelism
    assert "tensor: 8" in values_content

    # Verify sglang-specific content
    assert "image: lmsysorg/sglang:latest" in values_content
    assert "modelCommand: sglangServe" in values_content
    assert "model-path" in values_content
    assert "meta-llama/Llama-3.1-70B" in values_content


@pytest.mark.unit
def test_llmd_sglang_default_image():
    """Test that sglang uses default image when not specified."""
    params = {
        "ServiceConfig": {
            "model_path": "Qwen/Qwen3-8B",
            "port": 8000,
        },
        "DynConfig": {"mode": "agg"},
        "LlmdConfig": {},  # No sglang_image specified
        "WorkerConfig": {
            "agg_workers": 1,
            "agg_gpus_per_worker": 2,
        },
        "params": {
            "agg": {"gpus_per_worker": 2},
        },
        "agg_tensor_parallel_size": 2,
        "agg_data_parallel_size": 1,
        "agg_cli_args_list": [],
    }

    artifacts = render_backend_templates(param_values=params, backend="sglang", deployment_target="llm-d")

    values_content = artifacts["llm-d-values.yaml"]
    # Should use default sglang image
    assert "image: lmsysorg/sglang:latest" in values_content


@pytest.mark.unit
def test_dynamo_deployment_still_works():
    """Test that Dynamo deployment target still works (default behavior)."""
    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "K8sConfig": {},
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        "prefill_cli_args_list": [],
        "decode_cli_args_list": [],
        "name": "test-deployment",
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="dynamo-j2")

    # Should render Dynamo K8s manifest, not llm-d values
    assert "k8s_deploy.yaml" in artifacts
    assert "llm-d-values.yaml" not in artifacts

    k8s_content = artifacts["k8s_deploy.yaml"]
    assert "DynamoGraphDeployment" in k8s_content


@pytest.mark.unit
def test_dynamo_python_deployment():
    """Test that dynamo-python deployment target attempts to use Python config modifiers."""
    # First check if dynamo package is available
    try:
        import dynamo.profiler.utils.config_modifiers  # noqa: F401

        dynamo_available = True
    except ImportError:
        dynamo_available = False

    params = {
        "ServiceConfig": {
            "model_path": "meta-llama/Llama-3.1-8B",
            "served_model_name": "llama-3.1-8b",
            "port": 8000,
        },
        "DynConfig": {"mode": "disagg"},
        "K8sConfig": {
            "k8s_image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.0",
            "k8s_namespace": "default",
        },
        "WorkerConfig": {
            "prefill_workers": 2,
            "decode_workers": 4,
            "prefill_gpus_per_worker": 2,
            "decode_gpus_per_worker": 4,
        },
        "params": {
            "prefill": {"gpus_per_worker": 2},
            "decode": {"gpus_per_worker": 4},
        },
        "prefill_cli_args_list": ["--tensor-parallel-size", "2"],
        "decode_cli_args_list": ["--tensor-parallel-size", "4"],
    }

    artifacts = render_backend_templates(param_values=params, backend="vllm", deployment_target="dynamo-python")

    if dynamo_available:
        # If dynamo is available, should generate k8s_deploy.yaml via Python config modifiers
        assert "k8s_deploy.yaml" in artifacts
        k8s_content = artifacts["k8s_deploy.yaml"]
        assert len(k8s_content) > 0  # Should have content
    else:
        # If dynamo is not available, generation should fail gracefully (logged as warning)
        # but other artifacts should still be generated (run scripts, etc)
        assert "k8s_deploy.yaml" not in artifacts or artifacts["k8s_deploy.yaml"] == ""
        # Should still have other artifacts like run scripts
        assert any(k.startswith("run_") for k in artifacts)
