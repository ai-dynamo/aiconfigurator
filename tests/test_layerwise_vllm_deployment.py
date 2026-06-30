import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))

from vllm_deployment import (
    VllmDeploymentConfig,
    build_engine_args,
    compare_metadata,
    gpt_oss_runtime_defaults,
    make_metadata,
)


def test_build_engine_args_omits_compile_defaults() -> None:
    args = build_engine_args(
        VllmDeploymentConfig(
            model="Qwen/Qwen3-32B",
            max_model_len=1026,
            max_num_seqs=64,
            max_num_batched_tokens=64,
            gpu_memory_utilization=0.85,
        )
    )

    assert args == [
        "--model",
        "Qwen/Qwen3-32B",
        "--max-model-len",
        "1026",
        "--max-num-seqs",
        "64",
        "--max-num-batched-tokens",
        "64",
        "--gpu-memory-utilization",
        "0.85",
    ]
    assert "--compilation-config" not in args


def test_build_engine_args_includes_explicit_block_size() -> None:
    args = build_engine_args(
        VllmDeploymentConfig(
            model="Qwen/Qwen3.6-35B-A3B",
            block_size=16,
            gpu_memory_utilization=0.9,
        )
    )

    assert args == [
        "--model",
        "Qwen/Qwen3.6-35B-A3B",
        "--block-size",
        "16",
        "--gpu-memory-utilization",
        "0.9",
    ]


def test_build_engine_args_includes_data_parallel_size() -> None:
    args = build_engine_args(
        VllmDeploymentConfig(
            model="deepseek-ai/DeepSeek-V4-Flash",
            tensor_parallel_size=1,
            data_parallel_size=4,
        )
    )

    assert args == [
        "--model",
        "deepseek-ai/DeepSeek-V4-Flash",
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "4",
    ]


def test_metadata_compare_flags_compile_mode_mismatch() -> None:
    common = {
        "vllm_version": "0.20.1",
        "scheduler_config.max_num_seqs": 64,
        "scheduler_config.max_num_batched_tokens": 64,
        "compilation_config.cudagraph_mode": "FULL_AND_PIECEWISE",
    }
    fpm = make_metadata(
        artifact_kind="fpm",
        measurement_mode="deployment-parity",
        engine_args=["--model", "m"],
        effective_config={**common, "compilation_config.mode": "VLLM_COMPILE"},
    )
    layerwise = make_metadata(
        artifact_kind="layerwise",
        measurement_mode="attribution",
        engine_args=["--model", "m"],
        effective_config={**common, "compilation_config.mode": "NONE"},
    )

    mismatches = compare_metadata(fpm, layerwise)

    assert {
        "key": "compilation_config.mode",
        "left": "VLLM_COMPILE",
        "right": "NONE",
    } in mismatches


def test_metadata_is_json_serializable() -> None:
    metadata = make_metadata(
        artifact_kind="layerwise",
        measurement_mode="deployment-parity",
        engine_args=["--model", "m"],
        deployment_config=VllmDeploymentConfig(model="m", extra_args=("--foo", "bar")),
        effective_config={"compilation_config.mode": "VLLM_COMPILE"},
    )

    json.dumps(metadata)
    assert metadata["vllm_config_hash"]


def test_gpt_oss_runtime_defaults_use_fp8_kv_on_blackwell() -> None:
    defaults = gpt_oss_runtime_defaults(
        model="openai/gpt-oss-120b",
        system="b300_sxm",
        disable_prefix_caching=True,
    )

    assert defaults.kv_cache_dtype == "fp8"
    assert defaults.disable_prefix_caching is True
    assert "--skip-mm-profiling" in defaults.extra_args
    assert "--generation-config" in defaults.extra_args
    assert defaults.extra_args[defaults.extra_args.index("--generation-config") + 1] == "vllm"
    assert defaults.extra_args[defaults.extra_args.index("--max-cudagraph-capture-size") + 1] == "2048"
    assert defaults.extra_args[defaults.extra_args.index("--stream-interval") + 1] == "20"


def test_gpt_oss_runtime_defaults_do_not_infer_fp8_kv_without_blackwell() -> None:
    defaults = gpt_oss_runtime_defaults(
        model="openai/gpt-oss-120b",
        system="h100",
        disable_prefix_caching=True,
    )

    assert defaults.kv_cache_dtype is None
    assert defaults.disable_prefix_caching is True
    assert "--max-cudagraph-capture-size" in defaults.extra_args


def test_runtime_defaults_use_fp8_kv_for_deepseek_v4() -> None:
    defaults = gpt_oss_runtime_defaults(
        model="deepseek-ai/DeepSeek-V4-Flash",
        system="b300_sxm",
    )

    assert defaults.kv_cache_dtype == "fp8"
    assert "--max-cudagraph-capture-size" not in defaults.extra_args
    assert defaults.extra_args[defaults.extra_args.index("--block-size") + 1] == "256"
    assert defaults.extra_args[defaults.extra_args.index("--compilation-config") + 1] == (
        '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
    )
    assert defaults.extra_args[defaults.extra_args.index("--attention-config") + 1] == (
        '{"use_fp4_indexer_cache":true}'
    )
    assert defaults.extra_args[defaults.extra_args.index("--tokenizer-mode") + 1] == "deepseek_v4"
    assert "--reasoning-parser" not in defaults.extra_args

    explicit = gpt_oss_runtime_defaults(
        model="deepseek-ai/DeepSeek-V4-Flash",
        system="b300_sxm",
        extra_args=(
            "--kv-cache-dtype=auto",
            "--block-size",
            "128",
            "--attention_config.use_fp4_indexer_cache=False",
        ),
    )
    assert explicit.kv_cache_dtype is None
    assert "--kv-cache-dtype=auto" in explicit.extra_args
    assert explicit.extra_args[explicit.extra_args.index("--block-size") + 1] == "128"
    assert "--attention-config" not in explicit.extra_args
    assert "--attention_config.use_fp4_indexer_cache=False" in explicit.extra_args


def test_gpt_oss_runtime_defaults_preserve_explicit_overrides() -> None:
    defaults = gpt_oss_runtime_defaults(
        model="openai/gpt-oss-120b",
        system="b300_sxm",
        disable_prefix_caching=True,
        kv_cache_dtype="bf16",
        extra_args=("--stream-interval=7", "--enable-prefix-caching"),
    )

    assert defaults.kv_cache_dtype == "bf16"
    assert defaults.disable_prefix_caching is False
    assert "--stream-interval" not in defaults.extra_args
    assert "--stream-interval=7" in defaults.extra_args
