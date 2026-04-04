# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trtllm k8s_deploy generation with extra_engine_args vs cli_args.

Verifies that:
- use_dynamo_generator=False → k8s_deploy uses --extra-engine-args file approach
  (no redundant CLI flags like --tensor-parallel-size in container args)
- use_dynamo_generator=True  → cli_args_list is computed and available for
  _generate_k8s_via_dynamo / build_dgd_config (profiler path)
"""

import json
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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        k8s = artifacts["k8s_deploy.yaml"]
        args_block = _extract_args_block(k8s)

        assert "--extra-engine-args" in args_block
        assert "--model-path" in args_block
        assert "--served-model-name" in args_block

    def test_k8s_deploy_no_redundant_cli_flags(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        args_block = _extract_args_block(artifacts["k8s_deploy.yaml"])

        assert "--tensor-parallel-size" not in args_block
        assert "--pipeline-parallel-size" not in args_block
        assert "--max-batch-size" not in args_block
        assert "--override-engine-args" not in args_block
        assert "--free-gpu-memory-fraction" not in args_block

    def test_extra_engine_args_yaml_embedded(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        k8s = artifacts["k8s_deploy.yaml"]

        assert "tensor_parallel_size:" in k8s
        assert "kv_cache_config:" in k8s
        assert "YAML" in k8s  # heredoc marker

    def test_extra_engine_args_artifact_generated(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
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

    def test_cli_args_artifact_has_required_flags(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        cli = artifacts["cli_args_agg"]
        args = shlex.split(cli)

        # model-path is no longer in cli_args.j2 (handled by k8s_deploy fallback)
        assert "--model-path" not in args
        assert "--tensor-parallel-size" in args
        assert "--max-batch-size" in args

    def test_cli_args_override_engine_args_valid_json(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        cli = artifacts["cli_args_agg"]
        args = shlex.split(cli)

        if "--override-engine-args" in args:
            idx = args.index("--override-engine-args")
            override = json.loads(args[idx + 1])
            assert isinstance(override, dict)

    def test_cli_args_disagg_both_roles(self):
        params = _build_params(
            DynConfig={"mode": "disagg"},
            WorkerConfig={
                "prefill_workers": 1,
                "prefill_gpus_per_worker": 4,
                "decode_workers": 1,
                "decode_gpus_per_worker": 4,
                "agg_workers": 0,
            },
            params={
                "prefill": {
                    "tensor_parallel_size": 4,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 64,
                    "max_num_tokens": 8192,
                    "gpus_per_worker": 4,
                },
                "decode": {
                    "tensor_parallel_size": 4,
                    "pipeline_parallel_size": 1,
                    "max_batch_size": 128,
                    "max_num_tokens": 16384,
                    "gpus_per_worker": 4,
                },
            },
        )
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=False
        )
        assert "cli_args_prefill" in artifacts
        assert "cli_args_decode" in artifacts
        assert "--tensor-parallel-size" in artifacts["cli_args_prefill"]
        assert "--tensor-parallel-size" in artifacts["cli_args_decode"]


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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        dgd = yaml.safe_load(artifacts["k8s_deploy.yaml"])
        services = dgd["spec"]["services"]
        worker_names = [k for k in services if k != "Frontend"]
        assert len(worker_names) == 2

    @_requires_dynamo
    def test_extra_engine_args_still_generated(self):
        """Even with use_dynamo_generator=True, extra_engine_args artifacts
        should still be rendered (they are always produced)."""
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
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
        artifacts = render_backend_templates(
            params, "vllm", use_dynamo_generator=False
        )
        assert "extra_engine_args_agg.yaml" not in artifacts
        assert "cli_args_agg" in artifacts

    def test_sglang_k8s_deploy_has_cli_args(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "sglang", use_dynamo_generator=False
        )
        assert "extra_engine_args_agg.yaml" not in artifacts
        assert "cli_args_agg" in artifacts


# ---------------------------------------------------------------------------
# use_dynamo_generator=True  (profiler / translate path)
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestDynamicFlagsMode:
    """When use_dynamo_generator=True and backend=trtllm, cli_args_list must
    contain --trtllm.* flags converted from the rendered extra_engine_args YAML.
    The --override-engine-args JSON blob must NOT appear."""

    def test_cli_args_contain_trtllm_flags(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        cli = artifacts.get("cli_args_agg", "")
        assert "--trtllm.disable_overlap_scheduler" in cli
        assert "--trtllm.kv_cache_config.dtype" in cli or "--trtllm.kv_cache_config.enable_block_reuse" in cli

    def test_no_override_engine_args_json(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        cli = artifacts.get("cli_args_agg", "")
        assert "--override-engine-args" not in cli

    def test_direct_cli_flags_still_present(self):
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        cli = artifacts.get("cli_args_agg", "")
        assert "--tensor-parallel-size" in cli
        assert "--max-batch-size" in cli

    def test_skipped_keys_not_duplicated(self):
        """Keys already emitted as direct CLI flags must not appear as --trtllm.* flags."""
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        cli = artifacts.get("cli_args_agg", "")
        assert "--trtllm.tensor_parallel_size" not in cli
        assert "--trtllm.max_batch_size" not in cli
        assert "--trtllm.backend" not in cli

    def test_list_values_skipped(self):
        """List values (cuda_graph_config.batch_sizes) must not appear as --trtllm.* flags."""
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        cli = artifacts.get("cli_args_agg", "")
        assert "--trtllm.cuda_graph_config.batch_sizes" not in cli

    def test_disagg_mode_both_workers_get_flags(self):
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
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        prefill_cli = artifacts.get("cli_args_prefill", "")
        decode_cli = artifacts.get("cli_args_decode", "")
        # Both workers should have --trtllm.* flags
        assert "--trtllm." in prefill_cli
        assert "--trtllm." in decode_cli
        # Neither should have --override-engine-args
        assert "--override-engine-args" not in prefill_cli
        assert "--override-engine-args" not in decode_cli

    def test_extra_engine_args_yaml_still_generated(self):
        """The extra_engine_args YAML artifact is still generated (consumed by
        translate and potentially by --extra-engine-args file path)."""
        params = _build_params()
        artifacts = render_backend_templates(
            params, "trtllm", version="1.3.0rc5.post1", use_dynamo_generator=True
        )
        assert "extra_engine_args_agg.yaml" in artifacts
        assert "kv_cache_config" in artifacts["extra_engine_args_agg.yaml"]
