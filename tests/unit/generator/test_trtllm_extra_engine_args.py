# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trtllm k8s_deploy generation with extra_engine_args vs cli_args.

Verifies that:
- use_dynamo_generator=False → k8s_deploy uses --extra-engine-args file approach
  (no redundant CLI flags like --tensor-parallel-size in container args)
- use_dynamo_generator=True  → cli_args_list is computed and available for
  _generate_k8s_via_dynamo / build_dgd_config (profiler path)
"""

import shlex

import pytest
import yaml

from aiconfigurator.generator.rendering.engine import render_backend_templates

# Shared params dict that mirrors a typical CLI invocation:
#   aiconfigurator cli default --backend trtllm --model-path Qwen/Qwen3-32B-FP8
#   --system h200_sxm --total-gpus 8 --isl 5000 --osl 1000 --ttft 2000 --tpot 50
_BASE_PARAMS = {
    "ServiceConfig": {
        "model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_name": "Qwen/Qwen3-32B-FP8",
    },
    "K8sConfig": {
        "k8s_image": "nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.0.0",
        "k8s_namespace": "ets-dynamo",
    },
    "DynConfig": {"mode": "agg"},
    "WorkerConfig": {
        "agg_workers": 8,
        "agg_gpus_per_worker": 1,
        "prefill_workers": 0,
        "decode_workers": 0,
    },
    "NodeConfig": {"num_gpus_per_node": 8},
    "params": {
        "agg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_batch_size": 128,
            "max_num_tokens": 16384,
            "kv_cache_free_gpu_memory_fraction": 0.80,
            "enable_chunked_prefill": False,
            "disable_overlap_scheduler": True,
            "gpus_per_worker": 1,
        }
    },
}


def _build_params(**overrides):
    """Deep-copy base params and apply top-level overrides."""
    import copy

    params = copy.deepcopy(_BASE_PARAMS)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(params.get(key), dict):
            params[key].update(val)
        else:
            params[key] = val
    return params


def _extract_args_block(k8s_yaml: str) -> str:
    """Extract the bash ``args=(...)`` block from rendered k8s_deploy.yaml."""
    lines = k8s_yaml.split("\n")
    collecting = False
    block: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "args=(" in stripped:
            collecting = True
            block = [stripped]
            continue
        if collecting:
            block.append(stripped)
            if stripped == ")":
                break
    return "\n".join(block)


# ---------------------------------------------------------------------------
# use_dynamo_generator=False  (normal / standalone path)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestExtraEngineArgsMode:
    """When use_dynamo_generator=False, trtllm k8s_deploy should use the
    --extra-engine-args file approach with NO redundant inline CLI flags."""

    def test_k8s_deploy_uses_extra_engine_args_file(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        k8s = artifacts["k8s_deploy.yaml"]
        args_block = _extract_args_block(k8s)

        assert "--extra-engine-args" in args_block
        assert "--model-path" in args_block
        assert "--served-model-name" in args_block

    def test_k8s_deploy_no_redundant_cli_flags(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        args_block = _extract_args_block(artifacts["k8s_deploy.yaml"])

        assert "--tensor-parallel-size" not in args_block
        assert "--pipeline-parallel-size" not in args_block
        assert "--max-batch-size" not in args_block
        assert "--override-engine-args" not in args_block
        assert "--free-gpu-memory-fraction" not in args_block

    def test_extra_engine_args_yaml_embedded(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        k8s = artifacts["k8s_deploy.yaml"]

        assert "tensor_parallel_size:" in k8s
        assert "kv_cache_config:" in k8s
        assert "YAML" in k8s  # heredoc marker

    def test_extra_engine_args_artifact_generated(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        assert "extra_engine_args_agg.yaml" in artifacts
        engine_yaml = yaml.safe_load(artifacts["extra_engine_args_agg.yaml"])
        assert engine_yaml["tensor_parallel_size"] == 1
        assert "kv_cache_config" in engine_yaml

    def test_disagg_mode(self):
        params = _build_params(
            DynConfig={"mode": "disagg"},
            WorkerConfig={
                "prefill_workers": 2,
                "prefill_gpus_per_worker": 2,
                "decode_workers": 4,
                "decode_gpus_per_worker": 1,
                "agg_workers": 0,
            },
            params={
                "prefill": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 64,
                    "max_num_tokens": 8192,
                    "kv_cache_free_gpu_memory_fraction": 0.85,
                    "gpus_per_worker": 2,
                },
                "decode": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 256,
                    "max_num_tokens": 16384,
                    "kv_cache_free_gpu_memory_fraction": 0.80,
                    "gpus_per_worker": 1,
                },
            },
        )
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        k8s = artifacts["k8s_deploy.yaml"]

        assert "extra_engine_args_prefill.yaml" in artifacts
        assert "extra_engine_args_decode.yaml" in artifacts
        assert "--tensor-parallel-size" not in _extract_args_block(k8s)


# ---------------------------------------------------------------------------
# use_dynamo_generator=True  (profiler path)
# ---------------------------------------------------------------------------
try:
    from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS  # noqa: F401

    _has_dynamo = True
except ImportError:
    _has_dynamo = False

_requires_dynamo = pytest.mark.skipif(
    not _has_dynamo,
    reason="dynamo.profiler.utils.config_modifiers not installed",
)


@pytest.mark.unit
class TestProfilerCliArgsMode:
    """Verify cli_args artifacts are correctly computed (no dynamo needed)."""

    def test_cli_args_artifact_has_direct_flags_only(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False)
        cli = artifacts["cli_args_agg"]
        args = shlex.split(cli)

        # Direct argparser flags present
        assert "--tensor-parallel-size" in args
        assert "--max-batch-size" in args
        # model-path and override-engine-args no longer in cli_args.j2
        assert "--model-path" not in args
        assert "--override-engine-args" not in args


@pytest.mark.unit
class TestDynamoGeneratorPath:
    """use_dynamo_generator=True end-to-end tests.

    These call _generate_k8s_via_dynamo → build_dgd_config and verify the
    resulting DGD config has the correct structure.  Skipped when dynamo is
    not installed.
    """

    @_requires_dynamo
    def test_agg_k8s_deploy_has_worker_args(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        assert "k8s_deploy.yaml" in artifacts
        dgd = yaml.safe_load(artifacts["k8s_deploy.yaml"])

        services = dgd["spec"]["services"]
        worker_names = [k for k in services if k != "Frontend"]
        assert len(worker_names) >= 1

        worker = services[worker_names[0]]
        container = worker["extraPodSpec"]["mainContainer"]
        args = container.get("args", [])
        args_str = " ".join(str(a) for a in args)
        assert "--model-path" in args_str

    @_requires_dynamo
    def test_agg_k8s_deploy_replicas_and_gpu(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        dgd = yaml.safe_load(artifacts["k8s_deploy.yaml"])
        services = dgd["spec"]["services"]
        worker = next(v for k, v in services.items() if k != "Frontend")

        assert worker["replicas"] == 8
        assert str(worker["resources"]["limits"]["gpu"]) == "1"

    @_requires_dynamo
    def test_disagg_k8s_deploy_has_both_workers(self):
        params = _build_params(
            DynConfig={"mode": "disagg"},
            WorkerConfig={
                "prefill_workers": 2,
                "prefill_gpus_per_worker": 2,
                "decode_workers": 4,
                "decode_gpus_per_worker": 1,
                "agg_workers": 0,
            },
            params={
                "prefill": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 64,
                    "max_num_tokens": 8192,
                    "kv_cache_free_gpu_memory_fraction": 0.85,
                    "gpus_per_worker": 2,
                },
                "decode": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 256,
                    "max_num_tokens": 16384,
                    "kv_cache_free_gpu_memory_fraction": 0.80,
                    "gpus_per_worker": 1,
                },
            },
        )
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        dgd = yaml.safe_load(artifacts["k8s_deploy.yaml"])
        services = dgd["spec"]["services"]
        worker_names = [k for k in services if k != "Frontend"]
        assert len(worker_names) == 2

    @_requires_dynamo
    def test_extra_engine_args_still_generated(self):
        """Even with use_dynamo_generator=True, extra_engine_args artifacts
        should still be rendered (they are always produced)."""
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        assert "extra_engine_args_agg.yaml" in artifacts
        assert "cli_args_agg" in artifacts


# ---------------------------------------------------------------------------
# vllm / sglang remain unaffected
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestOtherBackendsUnaffected:
    """vllm and sglang have no extra_engine_args templates, so their
    k8s_deploy should continue using cli_args as before."""

    def test_vllm_k8s_deploy_has_cli_args(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "vllm", use_dynamo_generator=False)
        assert "extra_engine_args_agg.yaml" not in artifacts
        assert "cli_args_agg" in artifacts

    def test_sglang_k8s_deploy_has_cli_args(self):
        params = _build_params()
        artifacts = render_backend_templates(params, "sglang", use_dynamo_generator=False)
        assert "extra_engine_args_agg.yaml" not in artifacts
        assert "cli_args_agg" in artifacts


# ---------------------------------------------------------------------------
# use_dynamo_generator=True  (profiler / translate path)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDynamicFlagsMode:
    """When use_dynamo_generator=True and backend=trtllm, cli_args_list must
    contain --trtllm.* flags converted from the rendered extra_engine_args YAML."""

    def test_agg_mode(self):
        """Agg mode: --trtllm.* flags present, direct flags present, no duplication,
        no --override-engine-args, extra_engine_args YAML still generated."""
        params = _build_params()
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        cli = artifacts.get("cli_args_agg", "")

        # --trtllm.* engine params present
        assert "--trtllm.disable_overlap_scheduler" in cli
        assert "--trtllm.kv_cache_config." in cli

        # Direct CLI flags still present
        assert "--tensor-parallel-size" in cli
        assert "--max-batch-size" in cli

        # No --override-engine-args JSON blob
        assert "--override-engine-args" not in cli

        # Skipped keys not duplicated as --trtllm.*
        assert "--trtllm.tensor_parallel_size" not in cli
        assert "--trtllm.max_batch_size" not in cli
        assert "--trtllm.backend" not in cli

        # List values skipped
        assert "--trtllm.cuda_graph_config.batch_sizes" not in cli

        # extra_engine_args YAML artifact still generated
        assert "extra_engine_args_agg.yaml" in artifacts
        assert "kv_cache_config" in artifacts["extra_engine_args_agg.yaml"]

    def test_disagg_mode(self):
        """Disagg mode: both prefill and decode workers get --trtllm.* flags."""
        params = _build_params(
            DynConfig={"mode": "disagg"},
            WorkerConfig={
                "prefill_workers": 2,
                "prefill_gpus_per_worker": 2,
                "decode_workers": 4,
                "decode_gpus_per_worker": 1,
                "agg_workers": 0,
            },
            params={
                "prefill": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 64,
                    "max_num_tokens": 8192,
                    "kv_cache_free_gpu_memory_fraction": 0.85,
                    "gpus_per_worker": 2,
                },
                "decode": {
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 256,
                    "max_num_tokens": 16384,
                    "kv_cache_free_gpu_memory_fraction": 0.80,
                    "gpus_per_worker": 1,
                },
            },
        )
        artifacts = render_backend_templates(params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True)
        for role in ("prefill", "decode"):
            cli = artifacts.get(f"cli_args_{role}", "")
            assert "--trtllm." in cli, f"{role} worker missing --trtllm.* flags"
            assert "--override-engine-args" not in cli
