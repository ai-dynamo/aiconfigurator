# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang MoE collector.

Benchmarks SGLang fused MoE kernels across BF16, FP8 block, NVFP4, and INT4
paths when supported. Shared MoE model/sweep cases come from YAML; this module
owns SGLang kernel compatibility, server-args mocking, routing-logit synthesis,
rank-local workload construction, quantized weight setup, and perf logging.
"""

__compat__ = "sglang==0.5.14"

import gc
import importlib
import itertools
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock

import pkg_resources

# Mock global server args before importing MOE modules (required by SGLang 0.5.5+)
# The fused_moe_triton_config module now requires get_global_server_args() to be set
import sglang.srt.server_args as _server_args_module
import torch

if _server_args_module._global_server_args is None:
    _mock_server_args = MagicMock()
    _mock_server_args.enable_deterministic_inference = False
    _mock_server_args.enable_fused_moe_sum_all_reduce = (
        False  # SGLang 0.5.14; prevents fused all-reduce in single-GPU benchmarks
    )
    _mock_server_args.kt_weight_path = None
    _mock_server_args.flashinfer_mxfp4_moe_precision = "default"
    _server_args_module._global_server_args = _mock_server_args

import sglang.srt.layers.moe.fused_moe_triton.layer as _moe_layer_mod
import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels as _fmoe_kernels_mod
import sglang.srt.layers.moe.token_dispatcher.standard as _std_dispatch_mod
import sglang.srt.layers.moe.topk as _topk_mod
import sglang.srt.layers.moe.utils as _moe_utils
import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_mxint4_moe as _mxint4_mod
import sglang.srt.layers.quantization.fp8 as _fp8_mod
import sglang.srt.layers.quantization.modelopt_quant as _modelopt_mod
import sglang.srt.layers.quantization.mxfp4 as _mxfp4_mod

try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
        get_config_dtype_str,
        get_default_config,
        get_moe_configs,
    )
except ImportError:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
        get_config_dtype_str,
        get_default_config,
        get_moe_configs,
    )
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    TopKConfig,
    TopKOutputFormat,
    select_experts,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend, RoutingMethodType
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config, ModelOptFp8Config
from sglang.srt.layers.quantization.mxfp4 import Mxfp4Config
from sglang.srt.utils import is_hip

try:
    from case_generator import (
        get_common_moe_test_cases,
        get_moe_quantization_modes,
        get_moe_quantization_module_config,
        get_sglang_moe_backend,
        moe_model_allows_quantization,
    )

    from helper import (
        EXIT_CODE_RESTART,
        balanced_logits,
        benchmark_with_power,
        build_rank0_local_workload,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )
except ModuleNotFoundError:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from case_generator import (
        get_common_moe_test_cases,
        get_moe_quantization_modes,
        get_moe_quantization_module_config,
        get_sglang_moe_backend,
        moe_model_allows_quantization,
    )

    from helper import (
        EXIT_CODE_RESTART,
        balanced_logits,
        benchmark_with_power,
        build_rank0_local_workload,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )


_is_hip = is_hip()


def _mxfp4_activation_precision(moe_type: str) -> str:
    """Map the persisted quant label to SGLang's explicit activation mode."""

    return "bf16" if moe_type == "w4a16_mxfp4" else "default"


def _ensure_writable_flashinfer_cubin_dir() -> None:
    """Overlay packaged cubins when FlashInfer needs to create JIT symlinks."""
    from flashinfer.jit import cubin_loader
    from flashinfer.jit import env as jit_env

    source = Path(jit_env.FLASHINFER_CUBIN_DIR)
    if os.access(source, os.W_OK):
        return
    configured = os.environ.get("FLASHINFER_CUBIN_DIR")
    target = Path(configured) if configured and Path(configured) != source else None
    if target is None:
        version = str(jit_env.flashinfer_version).replace("+", "_")
        target = Path(tempfile.gettempdir()) / f"aic_flashinfer_cubins_{os.getuid()}_{version}"
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    for artifact_root in source.iterdir():
        overlay_entry = target / artifact_root.name
        try:
            overlay_entry.symlink_to(artifact_root, target_is_directory=artifact_root.is_dir())
        except FileExistsError:
            pass
    jit_env.FLASHINFER_CUBIN_DIR = target
    cubin_loader.FLASHINFER_CUBIN_DIR = target
    os.environ["FLASHINFER_CUBIN_DIR"] = str(target)


def get_moe_test_cases():
    sm_version = get_sm_version()
    moe_list = get_moe_quantization_modes("sglang", sm_version=sm_version)

    common_cases = get_common_moe_test_cases()
    test_cases = []
    seen_physical_cases = {}

    for common_moe_testcase in common_cases:
        model_name = common_moe_testcase.model_name
        num_tokens_list = common_moe_testcase.num_tokens_list

        for moe_type, num_tokens in itertools.product(moe_list, num_tokens_list):
            if not moe_model_allows_quantization("sglang", model_name, moe_type):
                continue
            is_fp4_experts = common_moe_testcase.architecture == "DeepseekV4ForCausalLM" and moe_type in {
                "w4a16_mxfp4",
                "w4a8_mxfp4_mxfp8",
            }
            moe_backend = get_sglang_moe_backend(common_moe_testcase, moe_type, sm_version)
            base_case = [
                moe_type,
                num_tokens,
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.model_name,
                common_moe_testcase.token_expert_distribution,
                common_moe_testcase.power_law_alpha,
                common_moe_testcase.sglang_moe_swiglu_limit,
                moe_backend,
                common_moe_testcase.sglang_moe_activation,
                common_moe_testcase.sglang_moe_is_gated,
                common_moe_testcase.sglang_moe_has_bias,
                common_moe_testcase.sglang_moe_gemm1_alpha,
                common_moe_testcase.sglang_moe_gemm1_clamp_limit,
                common_moe_testcase.sglang_moe_scoring_func,
                common_moe_testcase.sglang_moe_routing_method_type,
                common_moe_testcase.sglang_moe_routed_scaling_factor,
                common_moe_testcase.sglang_moe_renormalize,
                common_moe_testcase.sglang_moe_has_correction_bias,
                common_moe_testcase.sglang_moe_num_expert_group,
                common_moe_testcase.sglang_moe_topk_group,
                common_moe_testcase.sglang_moe_apply_router_weight_on_input,
                is_fp4_experts,
            ]
            physical_key = (
                moe_type,
                num_tokens,
                common_moe_testcase.hidden_size,
                common_moe_testcase.inter_size,
                common_moe_testcase.topk,
                common_moe_testcase.num_experts,
                common_moe_testcase.tp,
                common_moe_testcase.ep,
                common_moe_testcase.token_expert_distribution,
                common_moe_testcase.power_law_alpha,
            )
            execution_signature = (
                moe_backend,
                common_moe_testcase.sglang_moe_activation,
                common_moe_testcase.sglang_moe_is_gated,
                common_moe_testcase.sglang_moe_has_bias,
                common_moe_testcase.sglang_moe_gemm1_alpha,
                common_moe_testcase.sglang_moe_gemm1_clamp_limit,
                common_moe_testcase.sglang_moe_swiglu_limit,
                common_moe_testcase.sglang_moe_scoring_func,
                common_moe_testcase.sglang_moe_routing_method_type,
                common_moe_testcase.sglang_moe_routed_scaling_factor,
                common_moe_testcase.sglang_moe_renormalize,
                common_moe_testcase.sglang_moe_has_correction_bias,
                common_moe_testcase.sglang_moe_num_expert_group,
                common_moe_testcase.sglang_moe_topk_group,
                common_moe_testcase.sglang_moe_apply_router_weight_on_input,
                is_fp4_experts,
            )
            previous_signature = seen_physical_cases.get(physical_key)
            if previous_signature is not None and previous_signature != execution_signature:
                raise ValueError(
                    "SGLang MoE cases share one perf DB key but require different "
                    f"execution semantics: key={physical_key}, first={previous_signature}, "
                    f"current={execution_signature}, model={model_name}"
                )
            if previous_signature is None:
                seen_physical_cases[physical_key] = execution_signature
                test_cases.append(base_case)

    return test_cases


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def benchmark_config(
    config: BenchmarkConfig,
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int] | None = None,
    num_iters: int = 10,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    activation: str = "silu",
    is_gated: bool = True,
    gemm1_alpha: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> tuple[float, dict]:
    device = torch.device("cuda")
    expert_intermediate_size = shard_intermediate_size // (2 if is_gated else 1)
    if workloads is not None:
        num_iters = len(workloads)
        num_tokens = max(workload["hidden_states"].shape[0] for workload in workloads)

    # 1. Gating Output Generation
    if workloads is not None:
        gating_output = None
    elif distributed == "uniform":
        gating_output = torch.randn(num_iters, num_tokens, num_experts, dtype=torch.float32, device=device)
    elif distributed == "balanced":
        gating_output = [balanced_logits(num_tokens, num_experts, topk).to(device) for _ in range(num_iters)]
    elif distributed == "power_law":
        gating_output = [
            power_law_logits_v3(num_tokens, num_experts, topk, 1, power_law_alpha).to(device) for _ in range(num_iters)
        ]
    else:
        raise ValueError(f"Unsupported distributed mode: {distributed}")

    # 2. Set up the raw Triton BF16/FP8 path.
    init_dtype = torch.bfloat16 if use_fp8_w8a8 else dtype
    x = None if workloads is not None else torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127, 127, (num_experts, shard_intermediate_size, hidden_size), dtype=torch.int8, device=device
        )
        w2 = torch.randint(
            -127, 127, (num_experts, hidden_size, expert_intermediate_size), dtype=torch.int8, device=device
        )
    else:
        w1 = torch.randn(num_experts, shard_intermediate_size, hidden_size, dtype=init_dtype, device=device)
        w2 = torch.randn(num_experts, hidden_size, expert_intermediate_size, dtype=init_dtype, device=device)

    w1_scale = w2_scale = a1_scale = a2_scale = None
    if use_int8_w8a16:
        w1_scale = torch.randn((num_experts, 2 * shard_intermediate_size), dtype=torch.float32, device=device)
        w2_scale = torch.randn((hidden_size, num_experts), dtype=torch.float32, device=device)
    elif use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            w1_scale = torch.randn(num_experts, shard_intermediate_size, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, hidden_size, dtype=torch.float32, device=device)
        elif block_shape is None:
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)
        else:
            bn, bk = block_shape
            w1_scale = torch.rand(
                (num_experts, (shard_intermediate_size + bn - 1) // bn, (hidden_size + bk - 1) // bk),
                dtype=torch.float32,
                device=device,
            )
            w2_scale = torch.rand(
                (num_experts, (hidden_size + bn - 1) // bn, (expert_intermediate_size + bk - 1) // bk),
                dtype=torch.float32,
                device=device,
            )

    if use_fp8_w8a8:
        f8_type = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn
        w1, w2 = w1.to(f8_type), w2.to(f8_type)

    topk_output = (
        None
        if workloads is not None
        else select_experts(x, torch.randn(num_tokens, num_experts, device=device), TopKConfig(top_k=topk))
    )

    def run_op(i):
        from sglang.srt.layers.moe.fused_moe_triton import override_config

        if workloads is None:
            input_gating = gating_output[i % num_iters]
            new_topk = select_experts(x, input_gating, TopKConfig(top_k=topk))
            topk_output.topk_weights.copy_(new_topk.topk_weights)
            topk_output.topk_ids.copy_(new_topk.topk_ids)
            topk_output.router_logits.copy_(new_topk.router_logits)
            current_hidden_states = x
            current_topk_output = topk_output
        else:
            current_hidden_states = workloads[i % num_iters]["hidden_states"]
            current_topk_output = workloads[i % num_iters]["topk_output"]
            # build_rank0_local_workload sets remote expert IDs to -1
            # and their weights to 0. The Triton fused_moe kernel indexes
            # weight tensors by expert ID without masking, so clamp -1 to 0;
            # the zero weight still ensures no contribution.
            current_topk_output = StandardTopKOutput(
                topk_weights=current_topk_output.topk_weights,
                topk_ids=current_topk_output.topk_ids.clamp(min=0),
                router_logits=current_topk_output.router_logits,
            )

        with override_config(config):
            fused_moe(
                current_hidden_states,
                w1,
                w2,
                current_topk_output,
                moe_runner_config=MoeRunnerConfig(
                    activation=activation,
                    is_gated=is_gated,
                    gemm1_alpha=gemm1_alpha,
                    gemm1_clamp_limit=gemm1_clamp_limit,
                    swiglu_limit=swiglu_limit,
                ),
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=False,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                block_shape=block_shape,
            )

    # 3. Unified Execution Loop
    outside_loop_count = 5  # Repeat ops within kernel_func to increase accuracy for fast kernels

    def kernel_func():
        for i in range(outside_loop_count):
            run_op(i)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=5,
        num_runs=num_iters,
        repeat_n=1,
    ) as results:
        pass

    return results["latency_ms"] / outside_loop_count, results["power_stats"]


@contextmanager
def _patch_framework_moe_parallel(*, moe_tp_size: int, moe_ep_size: int):
    """Patch the SGLang 0.5.14 helpers cached by framework MoE backends."""
    parallel = SimpleNamespace(
        tp_size=moe_tp_size,
        tp_rank=0,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=0,
        moe_ep_size=moe_ep_size,
        moe_ep_rank=0,
    )
    missing = object()
    originals = []

    def replace(module, name, value):
        originals.append((module, name, getattr(module, name, missing)))
        setattr(module, name, value)

    modules = [_moe_layer_mod, _std_dispatch_mod, _topk_mod, _mxint4_mod, _mxfp4_mod, _fp8_mod, _modelopt_mod]
    for module_name in (
        "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
        "sglang.srt.layers.moe.moe_runner.flashinfer_mxfp4",
        "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe",
    ):
        try:
            modules.append(importlib.import_module(module_name))
        except ImportError:
            pass
    try:
        for module in modules:
            replace(module, "get_tp_group", lambda: None)
            replace(module, "is_allocation_symmetric", lambda: False)
            if module in {
                _moe_layer_mod,
                _std_dispatch_mod,
                _topk_mod,
                _mxint4_mod,
                _mxfp4_mod,
                _fp8_mod,
                _modelopt_mod,
            }:
                replace(module, "get_parallel", lambda: parallel)
        replace(_moe_layer_mod, "get_moe_expert_parallel_world_size", lambda: moe_ep_size)
        replace(_moe_layer_mod, "get_moe_expert_parallel_rank", lambda: 0)
        replace(_moe_layer_mod, "get_moe_tensor_parallel_world_size", lambda: moe_tp_size)
        replace(_moe_layer_mod, "get_moe_tensor_parallel_rank", lambda: 0)
        replace(_moe_layer_mod, "create_kt_config_from_server_args", lambda _server_args, _layer_id: None)
        replace(_std_dispatch_mod, "get_moe_expert_parallel_world_size", lambda: moe_ep_size)
        replace(_std_dispatch_mod, "get_moe_expert_parallel_rank", lambda: 0)
        yield
    finally:
        for module, name, original in reversed(originals):
            if original is missing:
                delattr(module, name)
            else:
                setattr(module, name, original)


def _benchmark_framework_quantized_moe(
    *,
    moe_type: str,
    moe_backend: str,
    num_tokens: int,
    hidden_size: int,
    inter_size: int,
    topk: int,
    num_experts: int,
    moe_tp_size: int,
    moe_ep_size: int,
    model_name: str,
    distributed: str,
    power_law_alpha: float,
    activation: str,
    is_gated: bool,
    has_bias: bool,
    gemm1_alpha: float | None,
    gemm1_clamp_limit: float | None,
    swiglu_limit: float | None,
    scoring_func: str,
    routing_method_type: str | None,
    routed_scaling_factor: float | None,
    renormalize: bool,
    has_correction_bias: bool,
    num_expert_group: int | None,
    topk_group: int | None,
    apply_router_weight_on_input: bool,
    is_fp4_experts: bool,
    int4_group_size: int,
    device: str,
) -> tuple[float, dict, str]:
    """Benchmark model-aware paths through SGLang 0.5.14 FusedMoE."""
    if moe_backend.startswith("flashinfer"):
        _ensure_writable_flashinfer_cubin_dir()

    routing_method = None if routing_method_type is None else RoutingMethodType[routing_method_type]
    if moe_type == "bfloat16":
        quant_config = None
    elif moe_type == "int4_wo":
        quant_config = CompressedTensorsConfig.from_config(
            {
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {
                            "num_bits": 4,
                            "type": "int",
                            "symmetric": True,
                            "strategy": "group",
                            "group_size": int4_group_size,
                            "dynamic": False,
                        },
                        "input_activations": None,
                        "output_activations": None,
                    }
                },
                "format": "pack-quantized",
                "ignore": [],
            }
        )
    elif moe_type == "fp8_block":
        quant_config = (
            ModelOptFp8Config(is_checkpoint_fp8_serialized=True)
            if moe_backend == "flashinfer_cutlass"
            else Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
            )
        )
    elif moe_type == "nvfp4":
        quant_config = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    elif is_fp4_experts:
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
            is_fp4_experts=True,
        )
    elif moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
        quant_config = Mxfp4Config(is_checkpoint_mxfp4_serialized=True)
    else:
        raise ValueError(f"Unsupported framework quantized MoE case: {moe_type=} {model_name=}")

    previous_backend = _moe_utils.MOE_RUNNER_BACKEND
    server_args = _server_args_module._global_server_args
    previous_precision = server_args.flashinfer_mxfp4_moe_precision
    _moe_utils.MOE_RUNNER_BACKEND = MoeRunnerBackend(moe_backend)
    if moe_backend == "flashinfer_mxfp4":
        server_args.flashinfer_mxfp4_moe_precision = _mxfp4_activation_precision(moe_type)

    moe_layer = None
    try:
        with _patch_framework_moe_parallel(moe_tp_size=moe_tp_size, moe_ep_size=moe_ep_size):
            moe_layer = FusedMoE(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=inter_size,
                layer_id=0,
                top_k=topk,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                quant_config=quant_config,
                prefix="aic_sglang_moe",
                activation=activation,
                is_gated=is_gated,
                with_bias=has_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_clamp_limit=gemm1_clamp_limit,
                swiglu_limit=swiglu_limit,
                routing_method_type=routing_method,
                routed_scaling_factor=routed_scaling_factor,
                apply_router_weight_on_input=apply_router_weight_on_input,
            ).to(device)
            with torch.no_grad():
                for name, parameter in moe_layer.named_parameters():
                    parameter.fill_(1) if "scale" in name or "alpha" in name else parameter.zero_()

            quant_method = moe_layer.quant_method
            quant_method_name = type(quant_method).__name__
            source_by_backend = {
                "triton": "sglang_fused_moe_triton",
                "triton_kernel": "sglang_triton_kernel_moe",
                "marlin": "sglang_marlin_moe",
                "flashinfer_trtllm": "sglang_flashinfer_trtllm_moe",
                "flashinfer_cutlass": "sglang_flashinfer_cutlass_moe",
            }

            if is_fp4_experts:
                fp4_expert_sources = {
                    "Mxfp4FlashinferCutlassMoEMethod": "sglang_flashinfer_cutlass_moe",
                    "Mxfp4FlashinferTrtllmMoEMethod": "sglang_mxfp4_flashinfer_trtllm_moe",
                }
                kernel_source = fp4_expert_sources.get(quant_method_name)
                actual_precision = getattr(quant_method, "flashinfer_mxfp4_moe_precision", None)
                expected_precision = _mxfp4_activation_precision(moe_type)
                if moe_backend != "flashinfer_mxfp4" or kernel_source is None or actual_precision != expected_precision:
                    raise RuntimeError(
                        "SGLang did not construct an MXFP4 method for DeepSeek-V4 FP4 experts: "
                        f"requested_backend={moe_backend}, actual_method={quant_method_name}, "
                        f"precision={actual_precision}, expected_precision={expected_precision}"
                    )
            elif moe_type == "int4_wo":
                scheme = getattr(moe_layer, "scheme", None)
                runner = getattr(scheme, "runner", None)
                actual_backend = getattr(getattr(runner, "runner_backend", None), "value", None)
                if (
                    quant_method_name != "CompressedTensorsFusedMoEMethod"
                    or type(scheme).__name__ != "CompressedTensorsWNA16MoE"
                    or actual_backend != moe_backend
                ):
                    raise RuntimeError(
                        "SGLang INT4-WO did not construct the requested Marlin path: "
                        f"requested_backend={moe_backend}, actual_method={quant_method_name}, "
                        f"actual_scheme={type(scheme).__name__}, actual_backend={actual_backend}"
                    )
                kernel_source = source_by_backend[actual_backend]
            elif moe_type == "nvfp4":
                runner = getattr(quant_method, "runner", None)
                actual_backend = getattr(getattr(quant_method, "_moe_runner_backend", None), "value", None)
                runner_backend = getattr(getattr(runner, "runner_backend", None), "value", None)
                if (
                    quant_method_name != "ModelOptNvFp4FusedMoEMethod"
                    or actual_backend != moe_backend
                    or (runner is not None and runner_backend != actual_backend)
                ):
                    raise RuntimeError(
                        "SGLang NVFP4 runner does not match the requested backend: "
                        f"requested_backend={moe_backend}, actual_method={quant_method_name}, "
                        f"actual_backend={actual_backend}, runner_backend={runner_backend}"
                    )
                kernel_source = source_by_backend[actual_backend]
            elif moe_type in {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}:
                if quant_method_name != "Mxfp4MoEMethod":
                    raise RuntimeError(f"SGLang {moe_type} constructed {quant_method_name}, expected Mxfp4MoEMethod")
                runner = getattr(quant_method, "runner", None)
                if runner is not None:
                    actual_backend = runner.runner_backend.value
                    if actual_backend != moe_backend:
                        raise RuntimeError(
                            f"SGLang {moe_type} runner backend mismatch: "
                            f"requested={moe_backend}, actual={actual_backend}"
                        )
                    if actual_backend == "flashinfer_mxfp4":
                        if quant_method._fi_kernel != "cutlass_sm90":
                            raise RuntimeError(
                                f"SGLang {moe_type} has unexpected FlashInfer MXFP4 leaf {quant_method._fi_kernel!r}"
                            )
                        kernel_source = "sglang_flashinfer_cutlass_moe"
                    else:
                        kernel_source = source_by_backend[actual_backend]
                elif (
                    moe_backend == "flashinfer_mxfp4"
                    and quant_method.use_flashinfer
                    and quant_method._fi_kernel == "trtllm_sm100"
                ):
                    actual_precision = quant_method.flashinfer_mxfp4_moe_precision
                    expected_precision = _mxfp4_activation_precision(moe_type)
                    if actual_precision != expected_precision:
                        raise RuntimeError(
                            f"SGLang {moe_type} FlashInfer precision mismatch: "
                            f"expected={expected_precision}, actual={actual_precision}"
                        )
                    kernel_source = "sglang_flashinfer_trtllm_moe"
                else:
                    raise RuntimeError(
                        f"SGLang {moe_type} has no verified runner: "
                        f"requested_backend={moe_backend}, fi_kernel={quant_method._fi_kernel}"
                    )
            else:
                if moe_backend == "flashinfer_cutlass":
                    uses_cutlass = quant_method_name == "Fp8MoEMethod" or bool(
                        getattr(quant_method, "use_flashinfer_cutlass", False)
                    )
                    if not uses_cutlass:
                        raise RuntimeError(
                            "SGLang MoE did not construct the requested FlashInfer CUTLASS path: "
                            f"actual_method={quant_method_name}"
                        )
                    kernel_source = source_by_backend[moe_backend]
                else:
                    runner = getattr(quant_method, "runner", None)
                    actual_backend = getattr(getattr(runner, "runner_backend", None), "value", None)
                    if actual_backend != moe_backend:
                        raise RuntimeError(
                            "SGLang MoE runner backend mismatch: "
                            f"requested={moe_backend}, actual={actual_backend}, method={quant_method_name}"
                        )
                    kernel_source = source_by_backend[actual_backend]

            quant_method.process_weights_after_loading(moe_layer)

            correction_bias = (
                torch.zeros(num_experts, dtype=torch.float32, device=device) if has_correction_bias else None
            )
            topk_layer = TopK(
                top_k=topk,
                layer_id=0,
                use_grouped_topk=num_expert_group is not None and topk_group is not None,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                renormalize=renormalize,
                scoring_func=scoring_func,
                correction_bias=correction_bias,
                output_format=(
                    TopKOutputFormat.STANDARD
                    if routing_method_type is None and moe_backend not in {"flashinfer_mxfp4", "triton_kernel"}
                    else None
                ),
                routed_scaling_factor=routed_scaling_factor,
                apply_routed_scaling_factor_on_output=(
                    routed_scaling_factor is not None and moe_layer.should_fuse_routed_scaling_factor_in_topk
                ),
                is_fp4_experts=is_fp4_experts,
            )
            hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            if distributed == "balanced":
                logits = [
                    balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)
                    for _ in range(5)
                ]
            elif distributed == "power_law":
                logits = [
                    power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha).to(
                        device=device, dtype=torch.float32
                    )
                    for _ in range(5)
                ]
            else:
                logits = [torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device) for _ in range(5)]

            def kernel_func():
                for router_logits in logits:
                    moe_layer(hidden_states, topk_layer(hidden_states, router_logits))

            with benchmark_with_power(
                device=device,
                kernel_func=kernel_func,
                num_warmups=5,
                num_runs=10,
                repeat_n=1,
            ) as results:
                pass
            return results["latency_ms"] / len(logits), results["power_stats"], kernel_source
    finally:
        if moe_layer is not None:
            for parameter in moe_layer.parameters():
                parameter.__dict__.pop("weight_loader", None)
            parameter = None
            moe_layer = None
        _fmoe_kernels_mod._B_DESC_CACHE.clear()
        _moe_utils.MOE_RUNNER_BACKEND = previous_backend
        server_args.flashinfer_mxfp4_moe_precision = previous_precision
        gc.collect()
        torch.cuda.empty_cache()


def benchmark(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: list[int] | None = None,
    distributed: str = "power_law",
    power_law_alpha: float = 0,
    workloads: list["Rank0Workload"] | None = None,
    swiglu_limit: float | None = None,
    activation: str = "silu",
    is_gated: bool = True,
    gemm1_alpha: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> tuple[float, dict]:
    torch.cuda.manual_seed_all(0)
    benchmark_num_tokens = (
        max(workload["hidden_states"].shape[0] for workload in workloads) if workloads is not None else num_tokens
    )
    dtype_str = get_config_dtype_str(
        dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=False,
        use_fp8_w8a8=use_fp8_w8a8,
    )
    # NOTE(woosuk): The current naming convention uses w2.shape[2], which
    # is the intermediate size after silu_and_mul.
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0
    expert_intermediate_size = shard_intermediate_size // (2 if is_gated else 1)
    op_config = get_moe_configs(num_experts, expert_intermediate_size, dtype_str, block_n, block_k)
    if op_config is None:
        config = get_default_config(
            benchmark_num_tokens,
            num_experts,
            shard_intermediate_size,
            hidden_size,
            topk,
            dtype_str,
            False,
            block_shape,
        )
    else:
        config = op_config[min(op_config.keys(), key=lambda x: abs(x - benchmark_num_tokens))]
    kernel_time, power_stats = benchmark_config(
        config,
        benchmark_num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        block_shape,
        distributed=distributed,
        power_law_alpha=power_law_alpha,
        workloads=workloads,
        swiglu_limit=swiglu_limit,
        activation=activation,
        is_gated=is_gated,
        gemm1_alpha=gemm1_alpha,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )
    return kernel_time, power_stats


class Rank0Workload(TypedDict):
    hidden_states: torch.Tensor
    topk_output: StandardTopKOutput
    masked_m: torch.Tensor


def build_rank0_workloads(
    num_workloads: int,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    num_experts: int,
    moe_ep_size: int,
    distributed: str,
    power_law_alpha: float | None,
    dtype: torch.dtype,
    device: torch.device,
) -> list[Rank0Workload]:
    workloads: list[Rank0Workload] = []
    experts_per_rank = num_experts // moe_ep_size

    for _ in range(num_workloads):
        if distributed == "power_law":
            if power_law_alpha is None:
                raise ValueError("power_law_alpha is required for power_law distribution")
            _, rank0_info = power_law_logits_v3(
                num_tokens,
                num_experts,
                topk,
                moe_ep_size,
                power_law_alpha,
                return_rank0_info=True,
            )
        elif distributed == "balanced":
            router_logits = balanced_logits(num_tokens, num_experts, topk).to(device=device, dtype=torch.float32)
            rank0_selected_slots = torch.topk(router_logits, topk, dim=-1).indices.to(torch.int64)
            rank0_token_mask = (rank0_selected_slots < experts_per_rank).any(dim=1)
            rank0_info = {
                "rank0_selected_slots": rank0_selected_slots[rank0_token_mask],
                "rank0_logits": router_logits[rank0_token_mask],
                "rank0_num_tokens": int(rank0_token_mask.sum().item()),
                "slots_per_rank": experts_per_rank,
            }
        else:
            raise ValueError(f"Unsupported distribution for rank0 workloads: {distributed}")

        rank0_local = build_rank0_local_workload(rank0_info)
        rank0_num_tokens = int(rank0_local["num_tokens"])
        workloads.append(
            {
                "hidden_states": torch.randn(rank0_num_tokens, hidden_size, dtype=dtype, device=device),
                "topk_output": StandardTopKOutput(
                    topk_weights=rank0_local["topk_weights"].to(device=device, dtype=torch.float32),
                    topk_ids=rank0_local["topk_ids"].to(device=device, dtype=torch.int32),
                    router_logits=torch.empty((rank0_num_tokens, 0), dtype=torch.float32, device=device),
                ),
                "masked_m": rank0_local["masked_m"].to(device=device, dtype=torch.int32),
            }
        )

    return workloads


def run_moe_torch(
    moe_type,
    num_tokens,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0,
    swiglu_limit=None,
    moe_backend="triton",
    activation="silu",
    is_gated=True,
    has_bias=False,
    gemm1_alpha=None,
    gemm1_clamp_limit=None,
    scoring_func="softmax",
    routing_method_type=None,
    routed_scaling_factor=None,
    renormalize=True,
    has_correction_bias=False,
    num_expert_group=None,
    topk_group=None,
    apply_router_weight_on_input=False,
    is_fp4_experts=False,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    assert moe_type in [
        "fp8_block",
        "bfloat16",
        "nvfp4",
        "w4a8_mxfp4_mxfp8",
        "int4_wo",
        "w4a16_mxfp4",
    ], "only support moe type = fp8_block, bfloat16, nvfp4, int4_wo, w4a16_mxfp4, or w4a8_mxfp4_mxfp8"
    if moe_backend not in {
        "triton",
        "triton_kernel",
        "marlin",
        "flashinfer_trtllm",
        "flashinfer_cutlass",
        "flashinfer_mxfp4",
    }:
        raise ValueError(f"Unsupported SGLang MoE backend: {moe_backend}")
    if moe_backend == "marlin" and moe_type != "int4_wo":
        raise ValueError(f"SGLang Marlin is only valid for int4_wo, got moe_type={moe_type!r}")
    assert inter_size % moe_tp_size == 0, "inter_size % moe_tp_size must be 0"
    assert num_experts % moe_ep_size == 0, "num_experts must be divisible by moe_ep_size"

    num_local_experts = num_experts // moe_ep_size
    local_inter_size = inter_size // moe_tp_size
    sm_version = get_sm_version()
    if (
        moe_type == "bfloat16"
        and moe_backend == "flashinfer_cutlass"
        and (hidden_size % 8 != 0 or local_inter_size % 8 != 0)
    ):
        raise ValueError(
            "SGLang FlashInfer CUTLASS BF16 MoE requires hidden_size and local_inter_size "
            f"to be divisible by 8, got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
        )
    activation_vector_size = 16 if sm_version >= 100 else 8
    if moe_backend == "triton" and is_gated and activation == "gelu" and local_inter_size % activation_vector_size != 0:
        raise ValueError(
            "SGLang Triton gated BF16 GELU requires local_inter_size "
            f"to be divisible by {activation_vector_size}, got local_inter_size={local_inter_size}"
        )
    if (
        moe_type == "w4a16_mxfp4"
        and is_fp4_experts
        and moe_backend == "flashinfer_mxfp4"
        and sm_version == 90
        and (hidden_size % 128 != 0 or local_inter_size % 128 != 0)
    ):
        raise ValueError(
            "SGLang SM90 DeepSeek-V4 W4A16 FP4 experts require hidden_size and local_inter_size "
            f"to be divisible by 128, got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
        )
    use_int4_w4a16 = moe_type == "int4_wo"
    if use_int4_w4a16 and moe_backend != "marlin":
        raise ValueError(f"SGLang SM90 int4_wo requires the Marlin backend, got {moe_backend}")
    int4_group_size = 128
    if use_int4_w4a16:
        int4_config = get_moe_quantization_module_config("sglang", moe_type, model_name=model_name)
        int4_group_size = int(int4_config.get("group_size", 128))
        if hidden_size % int4_group_size != 0 or local_inter_size % int4_group_size != 0:
            raise ValueError(
                "SGLang INT4-WO group quantization requires hidden_size and local_inter_size "
                f"to be divisible by group_size={int4_group_size}, got "
                f"hidden_size={hidden_size}, local_inter_size={local_inter_size}"
            )
        block_shape = [0, int4_group_size]
    elif moe_type == "fp8_block":
        if moe_backend == "triton" and (hidden_size % 128 != 0 or local_inter_size % 128 != 0):
            raise ValueError(
                "SGLang Triton fp8_block requires hidden_size and local_inter_size "
                f"to be divisible by 128, got hidden_size={hidden_size}, local_inter_size={local_inter_size}"
            )
        block_shape = [128, 128]
    else:
        block_shape = None

    use_framework_layer = (
        moe_type in {"int4_wo", "nvfp4", "w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
        or scoring_func != "softmax"
        or routing_method_type is not None
        or routed_scaling_factor is not None
        or not renormalize
        or has_correction_bias
        or num_expert_group is not None
        or topk_group is not None
        or apply_router_weight_on_input
        or (moe_backend != "triton" and not use_int4_w4a16)
    )
    if use_framework_layer:
        latency, power_stats, kernel_source = _benchmark_framework_quantized_moe(
            moe_type=moe_type,
            moe_backend=moe_backend,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            model_name=model_name,
            distributed=distributed,
            power_law_alpha=power_law_alpha,
            activation=activation,
            is_gated=is_gated,
            has_bias=has_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_clamp_limit=gemm1_clamp_limit,
            swiglu_limit=swiglu_limit,
            scoring_func=scoring_func,
            routing_method_type=routing_method_type,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            has_correction_bias=has_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            apply_router_weight_on_input=apply_router_weight_on_input,
            is_fp4_experts=is_fp4_experts,
            int4_group_size=int4_group_size,
            device=device,
        )
    else:
        rank0_workloads: list[Rank0Workload] | None = None
        if moe_ep_size > 1 and distributed in ("power_law", "balanced"):
            rank0_workloads = build_rank0_workloads(
                num_workloads=5,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                topk=topk,
                num_experts=num_experts,
                moe_ep_size=moe_ep_size,
                distributed=distributed,
                power_law_alpha=power_law_alpha if distributed == "power_law" else None,
                dtype=torch.bfloat16,
                device=torch.device(device),
            )
        try:
            latency, power_stats = benchmark(
                num_tokens,
                num_local_experts,
                (2 if is_gated else 1) * local_inter_size,
                hidden_size,
                topk,
                torch.bfloat16,
                moe_type == "fp8_block",
                False,
                False,
                block_shape=block_shape,
                distributed=distributed,
                power_law_alpha=power_law_alpha,
                workloads=rank0_workloads,
                swiglu_limit=swiglu_limit,
                activation=activation,
                is_gated=is_gated,
                gemm1_alpha=gemm1_alpha,
                gemm1_clamp_limit=gemm1_clamp_limit,
            )
        except Exception:
            _fmoe_kernels_mod._B_DESC_CACHE.clear()
            gc.collect()
            torch.cuda.empty_cache()
            raise

    if not use_framework_layer:
        _fmoe_kernels_mod._B_DESC_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()

    restart_worker = torch.cuda.memory_allocated(device) > torch.cuda.get_device_properties(device).total_memory // 4
    if not use_framework_layer:
        kernel_source = {
            "triton": "sglang_fused_moe_triton",
            "triton_kernel": "sglang_triton_kernel_moe",
            "marlin": "sglang_marlin_moe",
            "flashinfer_trtllm": "sglang_flashinfer_trtllm_moe",
            "flashinfer_cutlass": "sglang_flashinfer_cutlass_moe",
        }[moe_backend]
    persisted = log_perf(
        item_list=[
            {
                "moe_dtype": moe_type,
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "inter_size": inter_size,
                "topk": topk,
                "num_experts": num_experts,
                "moe_tp_size": moe_tp_size,
                "moe_ep_size": moe_ep_size,
                "distribution": "power_law_" + str(power_law_alpha) if distributed == "power_law" else distributed,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name="moe",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=power_stats,
    )
    if not persisted:
        raise RuntimeError("Failed to persist SGLang MoE performance row")
    if restart_worker:
        return EXIT_CODE_RESTART


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    for test_case in test_cases:
        print(test_case)
        run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
