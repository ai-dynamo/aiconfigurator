# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight generator golden contracts for the Dynamo 1.2.0 backend set."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
from functools import cache
from pathlib import Path

import pytest
import yaml

from aiconfigurator.generator.api import generate_backend_artifacts
from aiconfigurator.generator.utils import resolve_backend_version_for_dynamo

pytestmark = pytest.mark.unit

_DYNAMO_VERSION = "1.2.0"
_BACKEND_VERSIONS = {
    "vllm": "0.20.1",
    "sglang": "0.5.10.post1",
    "trtllm": "1.3.0rc14",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent

_BACKEND_SOURCES = {
    ("vllm", "0.20.1"): ("vllm", "v0.20.1", "vllm/engine/arg_utils.py"),
    ("sglang", "0.5.10.post1"): ("sglang", "v0.5.10.post1", "python/sglang/srt/server_args.py"),
    ("trtllm", "1.3.0rc14"): ("TensorRT-LLM", "v1.3.0rc14", "tensorrt_llm/llmapi/llm_args.py"),
}

_ALLOWED_CLI_FLAGS = {
    "vllm": {
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
        "--enable-expert-parallel",
        "--block-size",
        "--kv-cache-dtype",
        "--max-model-len",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--skip-tokenizer-init",
        "--trust-remote-code",
        "--enforce-eager",
        "--cudagraph-capture-sizes",
        "--no-enable-prefix-caching",
        "--speculative-config",
    },
    "sglang": {
        "--tensor-parallel-size",
        "--pipeline-parallel-size",
        "--data-parallel-size",
        "--page-size",
        "--kv-cache-dtype",
        "--max-prefill-tokens",
        "--enable-mixed-chunk",
        "--context-length",
        "--max-running-requests",
        "--skip-tokenizer-init",
        "--trust-remote-code",
        "--enable-dp-attention",
        "--expert-parallel-size",
        "--moe-dense-tp-size",
        "--disable-cuda-graph",
        "--cuda-graph-bs",
        "--speculative-algorithm",
        "--speculative-num-steps",
        "--disable-cuda-graph-padding",
        "--cuda-graph-max-bs",
        "--disaggregation-transfer-backend",
    },
}

_TRTLLM_TOP_LEVEL_KEYS = {
    "backend",
    "moe_expert_parallel_size",
    "moe_tensor_parallel_size",
    "moe_config",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "enable_attention_dp",
    "enable_chunked_prefill",
    "max_batch_size",
    "max_num_tokens",
    "max_seq_len",
    "kv_cache_config",
    "cache_transceiver_config",
    "cuda_graph_config",
    "disable_overlap_scheduler",
    "print_iter_log",
    "speculative_config",
}

_TRTLLM_NESTED_KEYS = {
    "kv_cache_config": {"free_gpu_memory_fraction", "dtype", "tokens_per_block", "enable_block_reuse"},
    "cache_transceiver_config": {"backend", "max_tokens_in_buffer"},
    "cuda_graph_config": {"enable_padding", "batch_sizes"},
    "speculative_config": {"decoding_type", "num_nextn_predict_layers"},
}

_GOLDEN_PARAMS = {
    "ServiceConfig": {
        "model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_path": "Qwen/Qwen3-32B-FP8",
        "served_model_name": "qwen3-golden",
        "include_frontend": True,
    },
    "K8sConfig": {
        "name_prefix": "golden",
        "k8s_image": "nvcr.io/nvidia/ai-dynamo/runtime:test",
        "k8s_namespace": "default",
    },
    "DynConfig": {"mode": "agg"},
    "WorkerConfig": {
        "agg_workers": 1,
        "agg_gpus_per_worker": 1,
        "prefill_workers": 0,
        "decode_workers": 0,
    },
    "NodeConfig": {"num_gpus_per_node": 8},
    "SlaConfig": {"isl": 2048, "osl": 512},
    "ModelConfig": {"is_moe": True, "prefix": 1024, "nextn": 2},
    "BenchConfig": {},
    "params": {
        "agg": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "moe_tensor_parallel_size": 2,
            "moe_expert_parallel_size": 4,
            "max_batch_size": 64,
            "max_num_tokens": 4096,
            "max_seq_len": 4096,
            "kv_cache_dtype": "bfloat16",
            "kv_cache_free_gpu_memory_fraction": 0.82,
            "tokens_per_block": 32,
            "enable_chunked_prefill": False,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "disable_cuda_graph": True,
        }
    },
}


@cache
def _backend_source(backend: str, version: str) -> str:
    source_ref = _BACKEND_SOURCES.get((backend, version))
    if source_ref is None:
        return ""
    repo_name, ref, source_path = source_ref
    repo_path = _WORKSPACE_ROOT / "third_party" / repo_name
    if not (repo_path / ".git").exists():
        return ""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "show", f"{ref}:{source_path}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Timed out reading {backend} {version} source from {repo_path}: {exc}")
    return result.stdout if result.returncode == 0 else ""


def _render(backend: str) -> dict[str, str]:
    return generate_backend_artifacts(
        copy.deepcopy(_GOLDEN_PARAMS),
        backend,
        backend_version=_BACKEND_VERSIONS[backend],
        deployment_target="dynamo-j2",
    )


def _split_cli(cli: str) -> list[str]:
    return shlex.split(cli)


def _flag_set(tokens: list[str]) -> set[str]:
    return {token for token in tokens if token.startswith("--")}


def _value_after(tokens: list[str], flag: str) -> str:
    idx = tokens.index(flag)
    assert idx + 1 < len(tokens), f"{flag} is missing a value"
    return tokens[idx + 1]


def _assert_generated_flags_are_known(backend: str, version: str, flags: set[str]) -> None:
    unexpected = flags - _ALLOWED_CLI_FLAGS[backend]
    assert not unexpected

    source = _backend_source(backend, version)
    if not source:
        return

    for flag in flags:
        if flag == "--no-enable-prefix-caching":
            assert "--enable-prefix-caching" in source
        else:
            assert flag in source


def _assert_trtllm_engine_keys_are_known(engine_args: dict[str, object]) -> None:
    unexpected = set(engine_args) - _TRTLLM_TOP_LEVEL_KEYS
    assert not unexpected

    source = _backend_source("trtllm", _BACKEND_VERSIONS["trtllm"])
    if not source:
        return

    for key in engine_args:
        assert key in source

    for group, allowed_keys in _TRTLLM_NESTED_KEYS.items():
        nested = engine_args.get(group)
        if not isinstance(nested, dict):
            continue
        unexpected_nested = set(nested) - allowed_keys
        assert not unexpected_nested
        for key in nested:
            assert key in source


def test_release_1_2_0_backend_version_matrix():
    for backend, version in _BACKEND_VERSIONS.items():
        assert resolve_backend_version_for_dynamo(_DYNAMO_VERSION, backend) == version


def test_vllm_0_20_1_cli_args_golden_contract():
    tokens = _split_cli(_render("vllm")["cli_args_agg"])
    flags = _flag_set(tokens)

    _assert_generated_flags_are_known("vllm", _BACKEND_VERSIONS["vllm"], flags)
    assert _value_after(tokens, "--tensor-parallel-size") == "8"
    assert _value_after(tokens, "--data-parallel-size") == "1"
    assert _value_after(tokens, "--kv-cache-dtype") == "auto"
    assert _value_after(tokens, "--max-num-batched-tokens") == "4060"
    assert "--enable-expert-parallel" in flags
    assert "--enforce-eager" in flags
    assert "--no-enable-prefix-caching" not in flags

    speculative = json.loads(_value_after(tokens, "--speculative-config"))
    assert speculative == {"method": "mtp", "num_speculative_tokens": 2}


def test_sglang_0_5_10_post1_cli_args_golden_contract():
    tokens = _split_cli(_render("sglang")["cli_args_agg"])
    flags = _flag_set(tokens)

    _assert_generated_flags_are_known("sglang", _BACKEND_VERSIONS["sglang"], flags)
    assert _value_after(tokens, "--tensor-parallel-size") == "8"
    assert _value_after(tokens, "--expert-parallel-size") == "4"
    assert _value_after(tokens, "--kv-cache-dtype") == "auto"
    assert _value_after(tokens, "--max-prefill-tokens") == "3548"
    assert _value_after(tokens, "--context-length") == "4096"
    assert _value_after(tokens, "--max-running-requests") == "512"
    assert _value_after(tokens, "--speculative-algorithm") == "NEXTN"
    assert _value_after(tokens, "--speculative-num-steps") == "2"
    assert "--disable-cuda-graph" in flags
    assert "--moe-dense-tp-size" not in flags


def test_trtllm_1_3_0rc14_extra_engine_args_golden_contract():
    artifacts = _render("trtllm")
    engine_args = yaml.safe_load(artifacts["extra_engine_args_agg.yaml"])

    _assert_trtllm_engine_keys_are_known(engine_args)
    assert engine_args["backend"] == "pytorch"
    assert engine_args["tensor_parallel_size"] == 8
    assert engine_args["pipeline_parallel_size"] == 1
    assert engine_args["moe_expert_parallel_size"] == 4
    assert engine_args["moe_tensor_parallel_size"] == 2
    assert engine_args["max_batch_size"] == 64
    assert engine_args["max_num_tokens"] == 2624
    assert engine_args["max_seq_len"] == 4096

    assert engine_args["kv_cache_config"]["free_gpu_memory_fraction"] == 0.82
    assert engine_args["kv_cache_config"]["dtype"] == "auto"
    assert engine_args["kv_cache_config"]["tokens_per_block"] == 32
    assert engine_args["cuda_graph_config"]["enable_padding"] is True
    assert engine_args["cuda_graph_config"]["batch_sizes"][-1] == 72
    assert engine_args["speculative_config"] == {
        "decoding_type": "MTP",
        "num_nextn_predict_layers": 2,
    }
