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
from aiconfigurator.generator.rendering.engine import _select_versioned_template
from aiconfigurator.generator.utils import resolve_backend_version_for_dynamo

pytestmark = pytest.mark.unit

_DYNAMO_VERSION = "1.2.0"
_BACKEND_VERSIONS = {
    "vllm": "0.20.1",
    "sglang": "0.5.11",
    "trtllm": "1.3.0rc14",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT.parent

_BACKEND_SOURCES = {
    ("vllm", "0.20.1"): ("vllm", "v0.20.1", "vllm/engine/arg_utils.py"),
    ("sglang", "0.5.11"): ("sglang", "v0.5.11", "python/sglang/srt/server_args.py"),
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
        "--mem-fraction-static",
        "--max-total-tokens",
        "--chunked-prefill-size",
        "--max-prefill-tokens",
        "--enable-mixed-chunk",
        "--context-length",
        "--max-running-requests",
        "--skip-tokenizer-init",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--enable-dp-attention",
        "--expert-parallel-size",
        "--moe-runner-backend",
        "--moe-a2a-backend",
        "--attention-backend",
        "--disaggregation-transfer-backend",
        "--disaggregation-bootstrap-port",
        "--elastic-ep-backend",
        "--disable-cuda-graph",
        "--cuda-graph-bs",
        "--disable-cuda-graph-padding",
        "--cuda-graph-max-bs",
        "--speculative-algorithm",
        "--speculative-num-steps",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
        "--disable-overlap-schedule",
        "--load-balance-method",
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


def _render_router_kvbm_candidate(backend: str) -> dict[str, str]:
    params = copy.deepcopy(_GOLDEN_PARAMS)
    role_params = params["params"].pop("agg")
    params["params"].update(
        {
            "prefill": role_params,
            "decode": copy.deepcopy(role_params),
        }
    )
    params["ServiceConfig"]["port"] = 9123
    params["WorkerConfig"] = {
        "agg_workers": 0,
        "prefill_workers": 1,
        "decode_workers": 1,
        "prefill_gpus_per_worker": 1,
        "decode_gpus_per_worker": 1,
    }
    params["DynConfig"] = {
        "mode": "disagg",
        "enable_router": True,
        "router_mode": "kv_router",
        "router_config": {
            "host_cache_hit_weight": 0.75,
            "disk_cache_hit_weight": 0.25,
        },
        "kvbm_config": {
            "cpu_cache_override_num_blocks": 4096,
            "max_transfer_batch_size": 32,
            "max_concurrent_transfers": 4,
        },
    }
    return generate_backend_artifacts(
        params,
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


def test_sglang_0_5_11_cli_template_is_version_specific():
    template_dir = _REPO_ROOT / "src" / "aiconfigurator" / "generator" / "config" / "backend_templates" / "sglang"
    selected = _select_versioned_template(
        list(template_dir.glob("cli_args*.j2")),
        "cli_args",
        ".j2",
        _BACKEND_VERSIONS["sglang"],
    )

    assert selected is not None
    assert selected.name == "cli_args.0.5.11.j2"


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


def test_vllm_0_20_1_k8s_router_contract():
    k8s = yaml.safe_load(_render("vllm")["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]

    frontend_env = {item["name"]: item["value"] for item in services["Frontend"]["envs"]}
    assert frontend_env["DYN_ROUTER_MODE"] == "kv"

    worker_args = services["VllmWorker"]["extraPodSpec"]["mainContainer"]["args"]
    assert "--kv-events-config" in worker_args
    kv_events = json.loads(_value_after(worker_args, "--kv-events-config"))
    assert kv_events == {
        "publisher": "zmq",
        "topic": "kv-events",
        "endpoint": "tcp://*:20081",
        "enable_kv_cache_events": True,
    }


def test_routed_vllm_run_preserves_frontend_port_and_cache_weights():
    artifacts = _render_router_kvbm_candidate("vllm")
    run_sh = next(
        content for name, content in artifacts.items() if name.startswith("run_") and "dynamo.frontend" in content
    )
    frontend_line = next(line for line in run_sh.splitlines() if "python3 -m dynamo.frontend" in line)

    assert "--router-host-cache-hit-weight 0.75" in frontend_line
    assert "--router-disk-cache-hit-weight 0.25" in frontend_line
    assert '--http-port "9123"' in frontend_line


def test_sglang_0_5_11_cli_args_golden_contract():
    tokens = _split_cli(_render("sglang")["cli_args_agg"])
    flags = _flag_set(tokens)

    _assert_generated_flags_are_known("sglang", _BACKEND_VERSIONS["sglang"], flags)
    assert _value_after(tokens, "--tensor-parallel-size") == "8"
    assert _value_after(tokens, "--expert-parallel-size") == "4"
    assert _value_after(tokens, "--kv-cache-dtype") == "auto"
    assert _value_after(tokens, "--mem-fraction-static") == "0.82"
    assert _value_after(tokens, "--chunked-prefill-size") == "-1"
    assert _value_after(tokens, "--max-prefill-tokens") == "3548"
    assert _value_after(tokens, "--context-length") == "4096"
    assert _value_after(tokens, "--max-running-requests") == "512"
    assert _value_after(tokens, "--speculative-algorithm") == "NEXTN"
    assert _value_after(tokens, "--speculative-num-steps") == "2"
    assert "--disable-cuda-graph" in flags
    assert "--moe-dense-tp-size" not in flags


def test_sglang_0_5_11_k8s_router_contract():
    k8s = yaml.safe_load(_render("sglang")["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]

    frontend_env = {item["name"]: item["value"] for item in services["Frontend"]["envs"]}
    assert frontend_env["DYN_ROUTER_MODE"] == "kv"

    worker_script = services["SGLangWorker"]["extraPodSpec"]["mainContainer"]["args"][0]
    assert "--kv-events-config" in worker_script
    assert '"publisher":"zmq"' in worker_script
    assert '"topic":"kv-events"' in worker_script
    assert '"endpoint":"tcp://*:5557"' in worker_script


def test_sglang_0_5_11_run_router_contract():
    run_sh = _render("sglang")["run_0.sh"]

    assert "SGLANG_KV_EVENT_PORT_BASE=${SGLANG_KV_EVENT_PORT_BASE:-5557}" in run_sh
    assert "python3 -m dynamo.frontend --router-mode kv --http-port" in run_sh
    assert "--kv-events-config" in run_sh
    assert "tcp://*:${EVENT_PORT}" in run_sh


def test_sglang_rejects_unsupported_kvbm_config():
    with pytest.raises(
        ValueError,
        match=r"DynConfig\.kvbm_config is not supported for backend 'sglang'; supported backends: trtllm, vllm",
    ):
        _render_router_kvbm_candidate("sglang")


def test_sglang_0_5_11_optional_cli_args_golden_contract():
    params = copy.deepcopy(_GOLDEN_PARAMS)
    params["ModelConfig"]["prefix"] = 0
    params["params"]["agg"].update(
        {
            "kv_cache_max_tokens": 8192,
            "disable_prefix_cache": True,
            "moe_backend": "triton",
            "moe_all2all_backend": "none",
            "kv_transfer_backend": "nixl",
            "disaggregation_bootstrap_port": 12346,
            "attention_backend": "fa3",
            "elastic_ep_backend": "nixl",
            "cuda_graph_enable_padding": False,
            "cuda_graph_max_batch_size": 256,
            "disable_overlap_scheduler": True,
            "moe_load_balancer": "round_robin",
        }
    )

    tokens = _split_cli(
        generate_backend_artifacts(
            params,
            "sglang",
            backend_version=_BACKEND_VERSIONS["sglang"],
            deployment_target="dynamo-j2",
        )["cli_args_agg"]
    )
    flags = _flag_set(tokens)

    _assert_generated_flags_are_known("sglang", _BACKEND_VERSIONS["sglang"], flags)
    assert _value_after(tokens, "--max-total-tokens") == "8192"
    assert _value_after(tokens, "--moe-runner-backend") == "triton"
    assert _value_after(tokens, "--moe-a2a-backend") == "none"
    assert _value_after(tokens, "--disaggregation-transfer-backend") == "nixl"
    assert _value_after(tokens, "--disaggregation-bootstrap-port") == "12346"
    assert _value_after(tokens, "--attention-backend") == "fa3"
    assert _value_after(tokens, "--elastic-ep-backend") == "nixl"
    assert _value_after(tokens, "--cuda-graph-max-bs") == "256"
    assert _value_after(tokens, "--load-balance-method") == "round_robin"
    assert "--disable-radix-cache" in flags
    assert "--disable-overlap-schedule" in flags
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
    assert engine_args["kv_cache_config"]["enable_block_reuse"] is True
    assert engine_args["cuda_graph_config"]["enable_padding"] is True
    assert engine_args["cuda_graph_config"]["batch_sizes"][-1] == 72
    assert engine_args["speculative_config"] == {
        "decoding_type": "MTP",
        "num_nextn_predict_layers": 2,
    }


def test_trtllm_dynamo_j2_router_and_kvbm_artifact_fidelity():
    artifacts = _render_router_kvbm_candidate("trtllm")
    k8s = yaml.safe_load(artifacts["k8s_deploy.yaml"])
    services = k8s["spec"]["services"]

    frontend_args = services["Frontend"]["extraPodSpec"]["mainContainer"]["args"]
    assert _value_after(frontend_args, "--http-port") == "9123"
    assert _value_after(frontend_args, "--router-mode") == "kv"
    assert _value_after(frontend_args, "--router-host-cache-hit-weight") == "0.75"
    assert _value_after(frontend_args, "--router-disk-cache-hit-weight") == "0.25"

    prefill = services["TRTLLMPrefillWorker"]
    prefill_env = {entry["name"]: entry["value"] for entry in prefill["envs"]}
    assert prefill_env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] == "4096"
    assert prefill_env["DYN_KVBM_MAX_TRANSFER_BATCH_SIZE"] == "32"
    assert prefill_env["DYN_KVBM_MAX_CONCURRENT_TRANSFERS"] == "4"
    prefill_script = prefill["extraPodSpec"]["mainContainer"]["args"][0]
    assert "args+=(--connector kvbm)" in prefill_script

    decode = services["TRTLLMDecodeWorker"]
    assert decode.get("envs") is None
    decode_script = decode["extraPodSpec"]["mainContainer"]["args"][0]
    assert "--connector kvbm" not in decode_script

    run_scripts = [content for name, content in artifacts.items() if name.startswith("run_")]
    prefill_run = next(content for content in run_scripts if "--disaggregation-mode prefill" in content)
    decode_run = next(content for content in run_scripts if "--disaggregation-mode decode" in content)
    assert "export DYN_KVBM_MAX_TRANSFER_BATCH_SIZE=32" in prefill_run
    assert "export DYN_KVBM_MAX_CONCURRENT_TRANSFERS=4" in prefill_run
    assert "--connector kvbm" in prefill_run
    assert "DYN_KVBM_" not in decode_run
    assert "--connector kvbm" not in decode_run


def test_benchmark_prefix_defaults_from_model_config():
    bench = _render("trtllm")["bench_run.sh"]

    assert 'BENCH_PREFIX="${AICONFIGURATOR_BENCH_PREFIX:-1024}"' in bench
    assert 'BENCH_PREFIX_PROMPTS="${AICONFIGURATOR_BENCH_PREFIX_PROMPTS:-1}"' in bench
    assert "--prefix-prompt-length" in bench
    assert "--num-prefix-prompts" in bench
