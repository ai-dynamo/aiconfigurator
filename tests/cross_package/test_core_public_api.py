# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the standalone core wheel's supported Python facade."""

from __future__ import annotations

import importlib.resources
import inspect
import json
import subprocess
import sys

import pytest

import aiconfigurator_core
import aiconfigurator_core.sdk as sdk
from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine
from aiconfigurator_core.sdk.memory import estimate_kv_cache, estimate_num_gpu_blocks
from aiconfigurator_core.sdk.operations import ElementWise, Embedding, MoEDispatch
from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

EXPECTED_FACADE = {
    "EngineHandle",
    "ModelConfig",
    "RuntimeConfig",
    "RustForwardPassPerfModel",
    "compile_engine",
    "estimate_kv_cache",
    "estimate_num_gpu_blocks",
}


def test_sdk_facade_import_is_lazy_in_a_fresh_interpreter() -> None:
    script = """
import sys

import aiconfigurator_core.sdk

protected_modules = {
    "aiconfigurator_core.sdk.engine",
    "aiconfigurator_core.sdk.memory",
    "aiconfigurator_core.sdk.rust_engine_step",
}
loaded_modules = protected_modules.intersection(sys.modules)
assert not loaded_modules, f"SDK facade eagerly loaded: {sorted(loaded_modules)}"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sdk_facade_exports_the_canonical_objects() -> None:
    assert set(sdk.__all__) == EXPECTED_FACADE
    assert sdk.EngineHandle is EngineHandle
    assert sdk.ModelConfig is ModelConfig
    assert sdk.RuntimeConfig is RuntimeConfig
    assert sdk.RustForwardPassPerfModel is RustForwardPassPerfModel
    assert sdk.compile_engine is compile_engine
    assert sdk.estimate_kv_cache is estimate_kv_cache
    assert sdk.estimate_num_gpu_blocks is estimate_num_gpu_blocks


def test_native_and_ergonomic_fpm_classes_are_deliberately_distinct() -> None:
    assert aiconfigurator_core.RustForwardPassPerfModel is not RustForwardPassPerfModel
    assert RustForwardPassPerfModel.__module__ == "aiconfigurator_core.sdk.rust_engine_step"


def test_stable_function_signatures() -> None:
    assert str(inspect.signature(compile_engine)) == (
        "(model_path: 'str', system: 'str', backend: 'str', backend_version: 'str | None' = None, *, "
        "tp_size: 'int' = 1, pp_size: 'int' = 1, attention_dp_size: 'int' = 1, "
        "moe_tp_size: 'int | None' = None, moe_ep_size: 'int | None' = None, "
        "gemm_quant_mode: 'str | None' = None, moe_quant_mode: 'str | None' = None, "
        "kvcache_quant_mode: 'str | None' = None, fmha_quant_mode: 'str | None' = None, "
        "comm_quant_mode: 'str | None' = None, attention_backend: 'str | None' = None, "
        "nextn: 'int' = 0, "
        "kv_block_size: 'int | None' = None, "
        "systems_path: 'str | None' = None, "
        "forward_model: 'str | None' = None) -> 'bytes'"
    )
    assert "scheduler_block_size" in inspect.signature(estimate_num_gpu_blocks).parameters
    assert "memory_fraction_kind" in inspect.signature(estimate_kv_cache).parameters
    assert str(inspect.signature(RustForwardPassPerfModel.from_regression)) == (
        "(worker_type: 'str', options: 'dict[str, Any] | None' = None) -> 'RustForwardPassPerfModel'"
    )
    assert str(inspect.signature(RustForwardPassPerfModel.best_available)) == (
        "(config: 'dict[str, Any]', worker_type: 'str', "
        "options: 'dict[str, Any] | None' = None) -> 'RustForwardPassPerfModel'"
    )


@pytest.mark.parametrize("worker_type", ["agg", "Prefill", "AGGREGATED", ""])
def test_raw_fpm_binding_rejects_worker_type_aliases(worker_type: str) -> None:
    with pytest.raises(ValueError, match="invalid worker_type"):
        aiconfigurator_core.RustForwardPassPerfModel.from_regression(worker_type)
    config = json.dumps(
        {
            "schema_version": 1,
            "model_name": "this/model-is-not-compiled",
            "system_name": "b200_sxm",
            "backend": "vllm",
            "backend_version": "0.19.0",
            "tp_size": 1,
            "pp_size": 1,
            "attention_dp_size": 1,
            "moe_tp_size": None,
            "moe_ep_size": None,
            "weight_dtype": None,
            "activation_dtype": None,
            "moe_dtype": None,
            "kv_cache_dtype": None,
            "kv_block_size": None,
            "nextn": None,
            "extra": {},
        }
    )
    with pytest.raises(ValueError, match="invalid worker_type"):
        aiconfigurator_core.RustForwardPassPerfModel.best_available(config, worker_type)


def test_raw_fpm_binding_requires_worker_type() -> None:
    with pytest.raises(TypeError):
        aiconfigurator_core.RustForwardPassPerfModel.from_regression()
    with pytest.raises(TypeError):
        aiconfigurator_core.RustForwardPassPerfModel.best_available("{}")


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
def test_raw_fpm_binding_constructs_every_worker_type_with_weights(worker_type: str) -> None:
    options = json.dumps(
        {
            "regression_attention_kv_weight": 2.0,
            "regression_prefill_attention_pair_weight": 3.0,
            "regression_ffn_token_weight": 4.0,
        }
    )
    model = aiconfigurator_core.RustForwardPassPerfModel.from_regression(worker_type, options)
    diagnostics = json.loads(model.diagnostics())
    assert diagnostics == {
        "source": "fallback_regression",
        "readiness": "insufficient_data",
        "retained_observations": 0,
        "correction_ready_buckets": 0,
        "last_warning": None,
    }


def _raw_regression_iteration(worker_type: str, index: int) -> dict[str, object]:
    if worker_type == "prefill":
        num_prefill_requests = index % 3 + 1
        sum_prefill_tokens = index * 5 + 3
        sum_prefill_kv_tokens = index * index * 7
        scheduled_requests = {
            "num_prefill_requests": num_prefill_requests,
            "sum_prefill_tokens": sum_prefill_tokens,
            "sum_prefill_kv_tokens": sum_prefill_kv_tokens,
        }
        attention = (
            sum_prefill_kv_tokens
            + sum_prefill_kv_tokens * sum_prefill_tokens / num_prefill_requests
            + sum_prefill_tokens**2 / (2 * num_prefill_requests)
            + sum_prefill_tokens / 2
        )
        ffn = sum_prefill_tokens
    elif worker_type == "decode":
        num_decode_requests = index + 1
        sum_decode_kv_tokens = index * index * 17 + 11
        scheduled_requests = {
            "num_decode_requests": num_decode_requests,
            "sum_decode_kv_tokens": sum_decode_kv_tokens,
        }
        attention = sum_decode_kv_tokens
        ffn = num_decode_requests
    else:
        num_prefill_requests = index % 3 + 1
        sum_prefill_tokens = index * 4 + 2
        sum_prefill_kv_tokens = index * index * 5
        num_decode_requests = index + 1
        sum_decode_kv_tokens = index * index * 13 + 7
        scheduled_requests = {
            "num_prefill_requests": num_prefill_requests,
            "sum_prefill_tokens": sum_prefill_tokens,
            "sum_prefill_kv_tokens": sum_prefill_kv_tokens,
            "num_decode_requests": num_decode_requests,
            "sum_decode_kv_tokens": sum_decode_kv_tokens,
        }
        attention = (
            sum_prefill_kv_tokens
            + sum_decode_kv_tokens
            + sum_prefill_kv_tokens * sum_prefill_tokens / num_prefill_requests
            + sum_prefill_tokens**2 / (2 * num_prefill_requests)
            + sum_prefill_tokens / 2
        )
        ffn = sum_prefill_tokens + num_decode_requests

    observed_ms = 1.0 + 0.01 * attention + 0.1 * ffn
    return {
        "version": 1,
        "wall_time": observed_ms / 1000.0,
        "scheduled_requests": scheduled_requests,
    }


@pytest.mark.parametrize("worker_type", ["prefill", "decode", "aggregated"])
def test_raw_fpm_binding_regression_round_trip(worker_type: str) -> None:
    model = aiconfigurator_core.RustForwardPassPerfModel.from_regression(
        worker_type,
        '{"min_observations":5}',
    )
    iterations = [[_raw_regression_iteration(worker_type, index)] for index in range(1, 7)]

    assert model.estimate_forward_pass_time_ms(json.dumps(iterations[-1])) is None
    model.tune_with_fpms(json.dumps(iterations))

    diagnostics = json.loads(model.diagnostics())
    assert diagnostics == {
        "source": "fallback_regression",
        "readiness": "ready",
        "retained_observations": 6,
        "correction_ready_buckets": 0,
        "last_warning": None,
    }
    prediction = model.estimate_forward_pass_time_ms(json.dumps(iterations[-1]))
    assert prediction is not None and prediction > 0.0


def test_raw_fpm_binding_validates_regression_weights() -> None:
    with pytest.raises(ValueError, match="regression_attention_kv_weight"):
        aiconfigurator_core.RustForwardPassPerfModel.from_regression(
            "prefill",
            '{"regression_attention_kv_weight": 0.0}',
        )


@pytest.mark.parametrize(
    "field",
    [
        "regression_attention_kv_weight",
        "regression_prefill_attention_pair_weight",
        "regression_ffn_token_weight",
    ],
)
@pytest.mark.parametrize("sentinel", ["NaN", "Infinity", "-Infinity"])
def test_raw_fpm_binding_decodes_exact_nonfinite_weight_sentinels(
    field: str,
    sentinel: str,
) -> None:
    """Raw ``options_json`` callers use quoted sentinels rather than non-JSON numbers."""
    with pytest.raises(ValueError, match=field):
        aiconfigurator_core.RustForwardPassPerfModel.from_regression(
            "aggregated",
            json.dumps({field: sentinel}),
        )


def test_raw_fpm_binding_rejects_unknown_nonfinite_weight_sentinel() -> None:
    with pytest.raises(ValueError, match="invalid options JSON"):
        aiconfigurator_core.RustForwardPassPerfModel.from_regression(
            "decode",
            '{"regression_ffn_token_weight":"Inf"}',
        )


def test_native_operation_constructors_preserve_legacy_keyword_names() -> None:
    Embedding("embedding", 1.0, 1024, 128, empirical_bw_scaling_factor=0.4)
    ElementWise("elementwise", 1.0, 128, 64, empirical_bw_scaling_factor=0.6)
    MoEDispatch(
        "dispatch",
        1.0,
        7168,
        8,
        256,
        1,
        16,
        1,
        False,
        enable_fp4_all2all=False,
        backend="sglang",
        reduce_results=False,
    )


def test_distribution_carries_typing_contract() -> None:
    root = importlib.resources.files("aiconfigurator_core")
    assert (root / "py.typed").is_file()
    assert (root / "_aiconfigurator_core.pyi").is_file()
