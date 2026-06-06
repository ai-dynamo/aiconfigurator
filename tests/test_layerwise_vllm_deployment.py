import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))

from vllm_deployment import (  # noqa: E402
    VllmDeploymentConfig,
    build_engine_args,
    compare_metadata,
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
