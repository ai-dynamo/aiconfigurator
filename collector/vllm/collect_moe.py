# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM MoE collector for current fused-experts APIs.

Benchmarks vLLM fused MoE kernels across BF16, FP8, FP8 block, NVFP4, MXFP4,
and INT4 paths when supported. Shared MoE cases come from YAML; this file owns
vLLM API compatibility, quant config creation, kernel filters, synthetic routing
logits, and perf logging.
"""

__compat__ = "vllm>=0.17.0"

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import fp8_w8a8_moe_quant_config, int4_w4a16_moe_quant_config

try:
    from vllm.model_executor.layers.fused_moe.layer import determine_expert_map
except ImportError:
    from vllm.model_executor.layers.fused_moe.expert_map_manager import determine_expert_map
from vllm.version import __version__ as vllm_version

# Compatibility: block FP8 helpers may differ by version.
# Priority: vllm.utils.deep_gemm -> deep_gemm extension -> None.
try:
    from vllm.utils.deep_gemm import per_block_cast_to_fp8
except Exception:
    try:
        import deep_gemm  # type: ignore

        per_block_cast_to_fp8 = getattr(deep_gemm, "per_block_cast_to_fp8", None)
    except Exception:
        per_block_cast_to_fp8 = None  # type: ignore[assignment]

# vLLM >= 0.14.0 raises AssertionError in get_current_vllm_config() when called
# outside a set_current_vllm_config() context (https://github.com/vllm-project/vllm/pull/31747).
# vLLM's custom ops (e.g. _vllm_ops.scaled_fp4_quant) requires vllm config to decide how to dispatch.
from vllm.config import SchedulerConfig, VllmConfig, set_current_vllm_config

try:
    from vllm.v1.worker.workspace import init_workspace_manager
except Exception:
    init_workspace_manager = None  # type: ignore[assignment]

# NVFP4 support: requires Blackwell (SM>=100) and FlashInfer TRTLLM FP4 kernel.
trtllm_fp4_block_scale_routed_moe = None
_vllm_ops = None
prepare_static_weights_for_trtllm_fp4_moe = None
_flashinfer_fp4_quantize = None
_nvfp4_available = False
# scaled_fp4_quant dropped is_sf_swizzled_layout in some vLLM builds.
# Probe the signature once at import time so _run_nvfp4_once doesn't branch per call.
_scaled_fp4_quant_accepts_swizzled: bool = False
# trtllm_fp4_block_scale_routed_moe dropped tile_tokens_dim in some flashinfer builds.
# Probe once at import time to avoid TypeError at call time.
_trtllm_moe_accepts_tile_tokens_dim: bool = False
try:
    import inspect

    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize  # type: ignore[assignment]
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe  # type: ignore[assignment]
    from vllm import _custom_ops as _vllm_ops  # type: ignore[assignment]
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        prepare_static_weights_for_trtllm_fp4_moe,  # type: ignore[assignment]
    )

    _scaled_fp4_quant_accepts_swizzled = (
        "is_sf_swizzled_layout" in inspect.signature(_vllm_ops.scaled_fp4_quant).parameters
    )
    _trtllm_moe_accepts_tile_tokens_dim = (
        "tile_tokens_dim" in inspect.signature(trtllm_fp4_block_scale_routed_moe).parameters
    )
    _nvfp4_available = True
except Exception:
    trtllm_fp4_block_scale_routed_moe = None
    _vllm_ops = None
    prepare_static_weights_for_trtllm_fp4_moe = None
    _flashinfer_fp4_quantize = None

# MXFP4 support: uses vLLM's high-level FusedMoE module with Mxfp4Config.
# This lets vLLM handle backend selection (FlashInfer/Triton/Marlin) and
# weight swizzle internally, so one code path works on all GPUs.
_mxfp4_available = False
_fused_moe_module_available = False
_MXFP4_MOE_TYPES = {"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"}
try:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config

    _fused_moe_module_available = True
    _mxfp4_available = True
except Exception:
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        _fused_moe_module_available = True
    except Exception:
        pass

from vllm.forward_context import get_forward_context, set_forward_context

from collector.case_generator import (
    get_common_moe_test_cases,
    get_moe_quantization_modes,
    get_moe_quantization_module_config,
    moe_model_allows_quantization,
    moe_shape_satisfies_constraints,
)
from collector.helper import (
    balanced_logits,
    benchmark_with_power,
    get_sm_version,
    log_perf,
    power_law_logits_v3,
    sampled_zipf_logits,
)

aic_debug = int(os.getenv("aic_moe_debug", "0"))  # noqa: SIM112
_WORKSPACE_MANAGER_DEVICES: set[str] = set()


class _SyntheticQwenGate(nn.Module):
    """Qwen-style router gate that can force synthetic logits after GEMM cost."""

    def __init__(self, hidden_size: int, num_experts: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        self.proj = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype, device=device)
        self.override_logits: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Return router logits, preserving the projection cost when overridden."""

        logits = self.proj(hidden_states)
        if self.override_logits is not None:
            forced = self.override_logits[: hidden_states.shape[0]].to(device=hidden_states.device, dtype=logits.dtype)
            logits = logits + (forced - logits).detach()
        return logits, None


class _QwenSharedExpert(nn.Module):
    """Minimal Qwen shared expert: sigmoid(gate) * down(silu(up(hidden)))."""

    def __init__(self, hidden_size: int, inter_size: int, *, dtype: torch.dtype, device: str):
        super().__init__()
        self.gate = nn.Linear(hidden_size, 1, bias=False, dtype=dtype, device=device)
        self.up = nn.Linear(hidden_size, inter_size, bias=False, dtype=dtype, device=device)
        self.down = nn.Linear(inter_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the Qwen shared-expert path."""

        return torch.sigmoid(self.gate(hidden_states)) * self.down(F.silu(self.up(hidden_states)))


def _is_deepseek_v4_model(model_name: str) -> bool:
    """Return whether a model name should use DeepSeek-V4 MoE routing settings."""

    return "DeepSeek-V4" in model_name


def _moe_distribution_label(distribution: str, power_law_alpha: float, *, use_cuda_graph: bool) -> str:
    """Return the perf-table distribution key for one MoE measurement row."""

    if distribution in {"power_law", "sampled_zipf"}:
        label = f"{distribution}_{power_law_alpha}"
    else:
        label = distribution
    if not use_cuda_graph:
        return f"{label}_eager"
    return label


def _deepseek_v4_module_settings(
    *,
    topk: int,
    num_experts: int,
    device: str,
) -> dict:
    """Build DeepSeek-V4 FusedMoE kwargs that affect routed-expert latency."""

    vocab_size = 129280
    return {
        "prefix": "model.layers.0.mlp.experts",
        "scoring_func": "sqrtsoftplus",
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "router_logits_dtype": torch.float32,
        # DeepSeek-V4 uses hash MoE in the first layers. The layerwise collector
        # currently measures layer 0 and scales it, so the op addback follows
        # the same layer-local routing mode without needing real weights.
        "hash_indices_table": torch.randint(
            0,
            num_experts,
            (vocab_size, topk),
            dtype=torch.int32,
            device=device,
        ),
    }


def _ensure_workspace_manager(device: str) -> None:
    if init_workspace_manager is None:
        return

    torch_device = torch.device(device)
    device_key = str(torch_device)
    if device_key in _WORKSPACE_MANAGER_DEVICES:
        return

    init_workspace_manager(torch_device)
    _WORKSPACE_MANAGER_DEVICES.add(device_key)


def _apply_vllm_config_overrides(
    vllm_config: VllmConfig,
    *,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
) -> None:
    """Apply deployment-shape overrides used by layerwise/FPM diagnostics."""

    model_config = getattr(vllm_config, "model_config", None)
    if max_model_len is not None and model_config is not None:
        model_config.max_model_len = max_model_len

    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    if scheduler_config is None:
        return
    if max_model_len is not None and hasattr(scheduler_config, "max_model_len"):
        scheduler_config.max_model_len = max_model_len
    if max_num_seqs is not None:
        scheduler_config.max_num_seqs = max_num_seqs
    if max_num_batched_tokens is not None:
        scheduler_config.max_num_batched_tokens = max_num_batched_tokens


def _make_module_vllm_config(
    *,
    max_model_len: int | None,
    max_num_seqs: int | None,
    max_num_batched_tokens: int | None,
) -> VllmConfig:
    """Build the vLLM config used by module-level FusedMoE benchmarks."""

    if max_model_len is None and max_num_seqs is None and max_num_batched_tokens is None:
        return VllmConfig()

    try:
        scheduler_config = SchedulerConfig(
            max_model_len=max_model_len or 1024,
            is_encoder_decoder=False,
            max_num_seqs=max_num_seqs or 128,
            max_num_batched_tokens=max_num_batched_tokens or 2048,
        )
        return VllmConfig(scheduler_config=scheduler_config)
    except Exception:
        vllm_config = VllmConfig()
        _apply_vllm_config_overrides(
            vllm_config,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        return vllm_config


def get_moe_test_cases():
    """Generate MoE test cases"""

    sm = get_sm_version()
    enabled_moe_types = get_moe_quantization_modes(
        "vllm",
        sm_version=sm,
        runtime_version=vllm_version,
        runtime_features={
            "per_block_fp8": per_block_cast_to_fp8 is not None,
            "nvfp4": _nvfp4_available,
            "mxfp4": _mxfp4_available,
        },
    )

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        model_name = common_moe_testcase.model_name

        # vllm does not support TP when EP is enabled.
        if common_moe_testcase.tp > 1 and common_moe_testcase.ep > 1:
            continue

        for moe_type in enabled_moe_types:
            if not moe_model_allows_quantization("vllm", model_name, moe_type):
                continue
            if not moe_shape_satisfies_constraints(
                "vllm",
                moe_type,
                hidden_size=common_moe_testcase.hidden_size,
                inter_size=common_moe_testcase.inter_size,
                tensor_parallel_size=common_moe_testcase.tp,
                topk=common_moe_testcase.topk,
            ):
                continue

            test_cases.append(
                [
                    moe_type,
                    common_moe_testcase.num_tokens_list,
                    common_moe_testcase.hidden_size,
                    common_moe_testcase.inter_size,
                    common_moe_testcase.topk,
                    common_moe_testcase.num_experts,
                    common_moe_testcase.tp,
                    common_moe_testcase.ep,
                    common_moe_testcase.model_name,
                    common_moe_testcase.token_expert_distribution,
                    common_moe_testcase.power_law_alpha,
                ]
            )

    return test_cases


def run_moe_torch(
    moe_type,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    model_name,
    distributed="power_law",
    power_law_alpha=0.0,
    *,
    perf_filename,
    device="cuda:0",
    shared_expert_inter_size: int = 0,
    max_model_len: int | None = None,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    use_cuda_graph: bool = True,
):
    """Run vLLM MoE performance benchmarking"""
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    # Configure quantization parameters
    dtype = torch.bfloat16
    quant_config = None
    block_shape: list[int] | None = None
    a1_scale = None
    a2_scale = None

    # Calculate local number of experts
    local_inter_size = inter_size // moe_tp_size
    local_num_experts, expert_map, _ = determine_expert_map(moe_ep_size, 0, num_experts)

    # Create weight tensors
    # w1: gate + up projection weights [num_experts, 2 * inter_size, hidden_size]
    # w2: down projection weights [num_experts, hidden_size, inter_size]
    w1 = torch.randn(
        local_num_experts,
        2 * local_inter_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2 = torch.randn(
        local_num_experts,
        hidden_size,
        local_inter_size,
        dtype=torch.bfloat16,
        device=device,
    )

    # INT4_WO path: W4A16 via vLLM's Marlin kernel using int4_w4a16_moe_quant_config.
    # Weights are packed uint8 (2 int4 per byte, shape K//2). Scales are per-group
    # along K (group_size=128). Zero-points are None (symmetric quantization).
    use_int4_wo = moe_type == "int4_wo"
    if use_int4_wo:
        int4_group_size = 128
        # w1: (E, 2*local_inter, hidden) packed → (E, 2*local_inter, hidden//2) uint8
        w1 = torch.randint(
            0, 127, (local_num_experts, 2 * local_inter_size, hidden_size // 2), dtype=torch.uint8, device=device
        )
        # w2: (E, hidden, local_inter) packed → (E, hidden, local_inter//2) uint8
        w2 = torch.randint(
            0, 127, (local_num_experts, hidden_size, local_inter_size // 2), dtype=torch.uint8, device=device
        )
        # Per-group scales: (E, N, K//group_size)
        w1_scale = torch.randn(
            (local_num_experts, 2 * local_inter_size, hidden_size // int4_group_size),
            dtype=torch.bfloat16,
            device=device,
        )
        w2_scale = torch.randn(
            (local_num_experts, hidden_size, local_inter_size // int4_group_size),
            dtype=torch.bfloat16,
            device=device,
        )
        quant_config = int4_w4a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, int4_group_size],
        )

    # BF16 and MXFP4 paths use vLLM's high-level FusedMoE module. This lets
    # vLLM handle router/top-k work and backend selection (FlashInfer/Triton/
    # Marlin/etc.) instead of benchmarking the lower-level fused_experts helper
    # directly.
    #
    # We keep a reference to the VllmConfig used during construction because
    # vLLM 0.17.0's MoERunner (vllm-project/vllm#32344) calls
    # get_forward_context() → get_layer_from_name() during forward, which
    # looks up the module in static_forward_context.  FusedMoE registers
    # itself there during __init__, so we must pass the *same* config to
    # set_forward_context() at benchmark time.
    use_mxfp4 = moe_type in _MXFP4_MOE_TYPES
    use_unquantized_module = moe_type == "bfloat16"
    use_qwen_shared_module = use_unquantized_module and shared_expert_inter_size > 0
    if shared_expert_inter_size > 0 and not use_unquantized_module:
        raise ValueError("shared_expert_inter_size is currently supported only for bfloat16 FusedMoE module rows.")
    moe_module = None
    module_vllm_cfg = None
    qwen_gate_module = None
    deepseek_hash_indices_table = None
    deepseek_hash_indices = None
    deepseek_hash_indices_list = None

    if use_mxfp4 or use_unquantized_module:
        if use_mxfp4 and not _mxfp4_available:
            raise ImportError("MXFP4 MoE requires vllm >= 0.17.0 with Mxfp4Config support.")
        if use_unquantized_module and not _fused_moe_module_available:
            raise ImportError("BF16 FusedMoE module benchmarking requires vLLM FusedMoE support.")

        _ensure_workspace_manager(device)

        module_quant_config = None
        if use_mxfp4:
            if "DeepSeek-V4" in model_name:
                from vllm.model_executor.models.deepseek_v4 import DeepseekV4FP8Config

                module_quant_config = DeepseekV4FP8Config.from_config(
                    {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
                        "weight_block_size": [128, 128],
                    }
                )
            else:
                module_quant_config = Mxfp4Config()
        module_config = get_moe_quantization_module_config("vllm", moe_type, model_name=model_name)

        # pcp_size=1: vLLM 0.17.0 added prefill context parallel to FusedMoE
        # (vllm-project/vllm#32344); without it, __init__ calls get_pcp_group()
        # which requires distributed init.
        # The collector benchmarks the already-sharded local expert weights on
        # one process, so keep FusedMoE's runtime parallel config single-process.
        module_vllm_cfg = _make_module_vllm_config(
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        )
        module_vllm_cfg.parallel_config.enable_expert_parallel = moe_ep_size > 1
        with set_current_vllm_config(module_vllm_cfg):
            # FusedMoE needs the same TP degree and global intermediate size as
            # serving so its parallel config and backend heuristics match vLLM.
            # With EP enabled vLLM derives the effective EP world from TP, so
            # pass EP through TP for the standalone local-rank benchmark.
            fused_moe_tp_size = moe_ep_size if moe_ep_size > 1 else moe_tp_size
            if fused_moe_tp_size > 1:
                # The op collector benchmarks one local rank without
                # initializing vLLM distributed groups. FusedMoE still asks the
                # parallel-config module for the TP rank while deriving EP, so
                # pin this synthetic process to rank 0 during construction.
                from vllm.model_executor.layers.fused_moe import config as fused_moe_config_module

                original_get_tp_rank = fused_moe_config_module.get_tensor_model_parallel_rank
                fused_moe_config_module.get_tensor_model_parallel_rank = lambda: 0
            else:
                fused_moe_config_module = None
                original_get_tp_rank = None
            fused_moe_kwargs = {
                "num_experts": num_experts,
                "top_k": topk,
                "hidden_size": hidden_size,
                "intermediate_size": inter_size,
                "params_dtype": torch.bfloat16,
                "renormalize": True,
                "quant_config": module_quant_config,
                "tp_size": fused_moe_tp_size,
                "dp_size": 1,
                "ep_size": moe_ep_size,
                "prefix": "",
                "has_bias": bool(module_config.get("has_bias", False)),
                "activation": str(module_config.get("activation", "silu")),
                "pcp_size": 1,
            }
            if _is_deepseek_v4_model(model_name):
                deepseek_settings = _deepseek_v4_module_settings(
                    topk=topk,
                    num_experts=num_experts,
                    device=device,
                )
                deepseek_hash_indices_table = deepseek_settings["hash_indices_table"]
                fused_moe_kwargs.update(deepseek_settings)
            if use_qwen_shared_module:
                qwen_gate_module = _SyntheticQwenGate(
                    hidden_size,
                    num_experts,
                    dtype=torch.bfloat16,
                    device=device,
                )
                fused_moe_kwargs["gate"] = qwen_gate_module
                fused_moe_kwargs["shared_experts"] = _QwenSharedExpert(
                    hidden_size,
                    shared_expert_inter_size,
                    dtype=torch.bfloat16,
                    device=device,
                )
            if "reduce_results" in inspect.signature(FusedMoE.__init__).parameters:
                fused_moe_kwargs["reduce_results"] = False
            try:
                moe_module = FusedMoE(**fused_moe_kwargs)
            finally:
                if fused_moe_config_module is not None and original_get_tp_rank is not None:
                    fused_moe_config_module.get_tensor_model_parallel_rank = original_get_tp_rank
            moe_module.to(device)
            moe_module.eval()
            moe_module.requires_grad_(False)

            with torch.no_grad():
                if use_mxfp4:
                    # Fill synthetic mxfp4 weights (uint8 packed, E2M1 format).
                    moe_module.w13_weight.data.random_(0, 255)
                    moe_module.w2_weight.data.random_(0, 255)
                    moe_module.w13_weight_scale.data.random_(0, 255)
                    moe_module.w2_weight_scale.data.random_(0, 255)
                else:
                    moe_module.w13_weight.copy_(w1)
                    moe_module.w2_weight.copy_(w2)
                if hasattr(moe_module, "w13_bias"):
                    moe_module.w13_bias.data.normal_()
                if hasattr(moe_module, "w2_bias"):
                    moe_module.w2_bias.data.normal_()

            # vLLM 0.19.0 consults get_current_vllm_config() while building
            # the TRTLLM MXFP4 MoE kernel, so keep the construction context open.
            moe_module.quant_method.process_weights_after_loading(moe_module)

        # Free standalone helper weights when the module path owns the weights.
        del w1, w2

    # NVFP4 path: uses FlashInfer TRTLLM FP4 monolithic kernel (not fused_experts).
    use_nvfp4 = moe_type == "nvfp4"
    nvfp4_data: dict | None = None

    if use_nvfp4:
        _missing = [
            name
            for name, obj in [
                ("trtllm_fp4_block_scale_routed_moe", trtllm_fp4_block_scale_routed_moe),
                ("_vllm_ops", _vllm_ops),
                ("prepare_static_weights_for_trtllm_fp4_moe", prepare_static_weights_for_trtllm_fp4_moe),
            ]
            if obj is None
        ]
        if _missing:
            raise ImportError(
                f"NVFP4 MoE requires flashinfer and vllm >= 0.14.0 with FP4 support, but the following "
                f"could not be imported: {', '.join(_missing)}. "
                f"Install a compatible flashinfer build and ensure vllm >= 0.14.0 with FP4 support."
            )

        # Raw packed FP4 weights and block scales
        w1_raw = torch.randint(
            0, 255, (local_num_experts, 2 * local_inter_size, hidden_size // 2), dtype=torch.uint8, device=device
        )
        w2_raw = torch.randint(
            0, 255, (local_num_experts, hidden_size, local_inter_size // 2), dtype=torch.uint8, device=device
        )
        w1_scale_raw = torch.ones(
            local_num_experts, 2 * local_inter_size, hidden_size // 16, dtype=torch.float8_e4m3fn, device=device
        )
        w2_scale_raw = torch.ones(
            local_num_experts, hidden_size, local_inter_size // 16, dtype=torch.float8_e4m3fn, device=device
        )

        # Shuffle weights and scales for TRTLLM kernel layout
        w1_shuf, w1_scale_shuf, w2_shuf, w2_scale_shuf = prepare_static_weights_for_trtllm_fp4_moe(
            w1_raw,
            w2_raw,
            w1_scale_raw,
            w2_scale_raw,
            hidden_size=hidden_size,
            intermediate_size=local_inter_size,
            num_experts=local_num_experts,
            is_gated_activation=True,
        )
        del w1_raw, w2_raw, w1_scale_raw, w2_scale_raw

        # Per-expert scales
        a13_scale = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        a2_scale_nvfp4 = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        w13_scale_2 = torch.ones(local_num_experts, dtype=torch.float32, device=device)
        w2_scale_2 = torch.ones(local_num_experts, dtype=torch.float32, device=device)

        nvfp4_data = dict(
            w1=w1_shuf,
            w1_scale=w1_scale_shuf,
            w2=w2_shuf,
            w2_scale=w2_scale_shuf,
            g1_scale_c=a13_scale * w13_scale_2 / a2_scale_nvfp4,
            a1_gscale=1.0 / a13_scale,
            g1_alphas=a13_scale * w13_scale_2,
            g2_alphas=a2_scale_nvfp4 * w2_scale_2,
        )
        # Free the bfloat16 weights; they are not used for nvfp4.
        del w1, w2

    elif moe_type in ["fp8", "fp8_block"]:
        dtype = torch.float8_e4m3fn
        if moe_type == "fp8_block":
            block_shape = [128, 128]

            if per_block_cast_to_fp8 is None:
                raise ImportError("per_block_cast_to_fp8 is unavailable; fp8_block requires a newer vLLM build.")

            w1_scale_list = []
            w2_scale_list = []
            w1_q = torch.empty_like(w1, dtype=dtype)
            w2_q = torch.empty_like(w2, dtype=dtype)
            for i in range(local_num_experts):
                w1_q[i], w1_scale_i = per_block_cast_to_fp8(w1[i], block_size=block_shape, use_ue8m0=True)
                w2_q[i], w2_scale_i = per_block_cast_to_fp8(w2[i], block_size=block_shape, use_ue8m0=True)
                w1_scale_list.append(w1_scale_i)
                w2_scale_list.append(w2_scale_i)
            w1 = w1_q
            w2 = w2_q
            w1_scale = torch.stack(w1_scale_list)
            w2_scale = torch.stack(w2_scale_list)
        else:
            w1_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(local_num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)

        quant_config = fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    if not (use_mxfp4 or use_unquantized_module) and dtype == torch.float8_e4m3fn:
        w1 = w1.to(dtype)
        w2 = w2.to(dtype)

    # Performance testing for each token count
    for num_tokens_idx, num_tokens in enumerate(num_tokens_lists):
        print("num_tokens", num_tokens)
        print("topk", topk)
        hs_dtype = torch.bfloat16
        hidden_states = torch.randn([num_tokens, hidden_size], dtype=hs_dtype, device=device)

        # Generate routing inputs.
        # Module paths use FusedMoE.forward(hidden_states, router_logits), which
        # does routing internally; other paths need pre-computed topk weights/ids.
        num_iter = 5 if distributed == "power_law" else 1
        input_ids_list = None
        input_ids = None
        if use_mxfp4 or use_unquantized_module:
            # FusedMoE.forward() takes raw router logits (num_tokens, num_experts)
            logits_dtype = torch.float32 if _is_deepseek_v4_model(model_name) else torch.bfloat16
            if distributed == "power_law":
                actual_logits_list = [
                    power_law_logits_v3(num_tokens, num_experts, topk, moe_ep_size, power_law_alpha)
                    .to(logits_dtype)
                    .to(device)
                    for _ in range(num_iter)
                ]
            elif distributed == "sampled_zipf":
                actual_logits = sampled_zipf_logits(num_tokens, num_experts, topk, power_law_alpha).to(
                    logits_dtype
                ).to(device)
            elif distributed == "balanced":
                actual_logits = balanced_logits(num_tokens, num_experts, topk).to(logits_dtype).to(device)
            else:
                raise ValueError(f"Unsupported distributed mode: {distributed}")
            if deepseek_hash_indices_table is not None:
                vocab_size = deepseek_hash_indices_table.shape[0]
                if distributed == "power_law":
                    input_ids_list = []
                    deepseek_hash_indices_list = []
                    for logits in actual_logits_list:
                        if logits.shape[0] > vocab_size:
                            raise ValueError(
                                "DeepSeek-V4 synthetic hash routing requires num_tokens <= vocab_size, "
                                f"got num_tokens={logits.shape[0]}, vocab_size={vocab_size}"
                            )
                        _, selected = torch.topk(logits, topk, dim=-1)
                        deepseek_hash_indices_list.append(selected.to(torch.int32).contiguous())
                        input_ids_list.append(torch.arange(logits.shape[0], dtype=torch.int32, device=device))
                else:
                    if num_tokens > vocab_size:
                        raise ValueError(
                            "DeepSeek-V4 synthetic hash routing requires num_tokens <= vocab_size, "
                            f"got num_tokens={num_tokens}, vocab_size={vocab_size}"
                        )
                    _, selected = torch.topk(actual_logits, topk, dim=-1)
                    deepseek_hash_indices = selected.to(torch.int32).contiguous()
                    input_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
        elif distributed in {"power_law", "sampled_zipf"}:
            topk_weights_list = []
            topk_ids_list = []

            for i in range(num_iter):
                if distributed == "power_law":
                    logits = power_law_logits_v3(
                        num_tokens,
                        num_experts,
                        topk,
                        moe_ep_size,
                        power_law_alpha,
                    )
                else:
                    logits = sampled_zipf_logits(
                        num_tokens,
                        num_experts,
                        topk,
                        power_law_alpha,
                        seed=20260612 + i,
                    )
                logits = logits.bfloat16().to(device)
                weights, ids = torch.topk(logits, topk, dim=-1)
                topk_weights = F.softmax(weights, dim=-1)
                if use_int4_wo:
                    topk_weights = topk_weights.float()
                topk_weights_list.append(topk_weights)
                topk_ids_list.append(ids)

            print("actual num_tokens: ", [topk_ids.shape[0] for topk_ids in topk_ids_list])

        elif distributed == "balanced":
            actual_logits = balanced_logits(num_tokens, num_experts, topk).bfloat16().to(device)
            topk_weights, topk_ids = torch.topk(actual_logits, topk, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)
            if use_int4_wo:
                topk_weights = topk_weights.float()

        else:
            raise ValueError(f"Unsupported distributed mode: {distributed}")

        num_warmups = 3
        num_runs = 6
        if distributed == "power_law":
            num_warmups = 1
            num_runs = 1

        def _run_nvfp4_once(hs, tw, ti):
            """Run a single nvfp4 MoE iteration via FlashInfer TRTLLM FP4 kernel."""
            num_tok = hs.shape[0]
            # Quantize input to FP4 with linear (non-swizzled) scale layout so that
            # x_scale can be reshaped to [M, K//16] for trtllm_fp4_block_scale_routed_moe.
            #
            # vLLM < 0.16.0: scaled_fp4_quant accepts is_sf_swizzled_layout=False directly.
            # vLLM >= 0.16.0: the parameter was removed and the op returns swizzled layout
            #   by default (tile-padded, incompatible shape). Fall back to flashinfer's
            #   fp4_quantize which still supports is_sf_swizzled_layout=False.
            if _scaled_fp4_quant_accepts_swizzled:
                x_fp4, x_scale = _vllm_ops.scaled_fp4_quant(
                    hs.to(torch.bfloat16),
                    nvfp4_data["a1_gscale"][0:1],
                    is_sf_swizzled_layout=False,
                )
            else:
                per_tok_scale = nvfp4_data["a1_gscale"][0:1].view(1, 1).expand(num_tok, 1).contiguous()
                x_fp4, x_scale = _flashinfer_fp4_quantize(
                    hs.to(torch.bfloat16),
                    per_tok_scale,
                    is_sf_swizzled_layout=False,
                )
            scale_cols = hs.shape[1] // 16
            # Pack topk: (expert_id << 16) | bf16_weight_as_int16
            packed = (ti.to(torch.int32) << 16) | tw.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            trtllm_fp4_block_scale_routed_moe(
                topk_ids=packed,
                routing_bias=None,
                hidden_states=x_fp4,
                hidden_states_scale=x_scale.view(num_tok, scale_cols).to(torch.float8_e4m3fn),
                gemm1_weights=nvfp4_data["w1"],
                gemm1_weights_scale=nvfp4_data["w1_scale"].view(torch.float8_e4m3fn),
                gemm1_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                gemm2_weights=nvfp4_data["w2"],
                gemm2_weights_scale=nvfp4_data["w2_scale"].view(torch.float8_e4m3fn),
                gemm2_bias=None,
                output1_scale_scalar=nvfp4_data["g1_scale_c"],
                output1_scale_gate_scalar=nvfp4_data["g1_alphas"],
                output2_scale_scalar=nvfp4_data["g2_alphas"],
                num_experts=num_experts,
                top_k=topk,
                n_group=0,
                topk_group=0,
                intermediate_size=local_inter_size,
                local_expert_offset=0,
                local_num_experts=local_num_experts,
                routed_scaling_factor=None,
                routing_method_type=1,  # Renormalize
                do_finalize=True,
                # tile_tokens_dim: avg tokens per expert, rounded to next power-of-2,
                # clamped to [8, 64]. Required by some flashinfer builds, rejected by others.
                **(
                    {
                        "tile_tokens_dim": min(
                            max(1 << (max(1, (num_tok * topk) // num_experts) - 1).bit_length(), 8), 64
                        )
                    }
                    if _trtllm_moe_accepts_tile_tokens_dim
                    else {}
                ),
            )

        def _module_forward(hs, rl, input_token_ids=None, hash_indices=None):
            # vLLM's custom MoE op increments a per-context layer index on
            # each forward call.  We only register one layer, so reset the
            # counter before every call to avoid an index-out-of-range error.
            fwd_ctx = get_forward_context()
            if hasattr(fwd_ctx, "moe_layer_index"):
                fwd_ctx.moe_layer_index = 0
            if hash_indices is not None:
                # DeepSeek-V4 hash MoE routes from ``input_token_ids`` through
                # ``hash_indices_table``. Keep that production path active, but
                # overwrite the synthetic table rows used by this microbenchmark
                # so the requested load distribution is actually measured.
                deepseek_hash_indices_table[: hash_indices.shape[0]].copy_(hash_indices)
            if qwen_gate_module is not None:
                qwen_gate_module.override_logits = rl
                router_input = hs
            else:
                router_input = rl
            if fused_moe_tp_size > 1:
                # The deployed TP/EP path reduces partial outputs across ranks.
                # This standalone op collector benchmarks only one local rank,
                # without a distributed process group, so keep the measured row
                # to local MoE compute and leave communication to separate
                # comm modeling.
                from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_module

                original_all_reduce = moe_runner_module.tensor_model_parallel_all_reduce
                moe_runner_module.tensor_model_parallel_all_reduce = lambda x: x
                try:
                    moe_module.forward(hs, router_input, input_token_ids)
                finally:
                    moe_runner_module.tensor_model_parallel_all_reduce = original_all_reduce
            else:
                moe_module.forward(hs, router_input, input_token_ids)

        def run_single_iteration():
            if use_mxfp4 or use_unquantized_module:
                # FusedMoE.forward(hidden_states, router_logits) does routing internally.
                if distributed == "power_law":
                    ids_iter = input_ids_list or [None] * len(actual_logits_list)
                    hash_iter = deepseek_hash_indices_list or [None] * len(actual_logits_list)
                    for logits, ids, hash_indices in zip(actual_logits_list, ids_iter, hash_iter, strict=True):
                        _module_forward(
                            hidden_states[: logits.shape[0]],
                            logits[: logits.shape[0]],
                            ids,
                            hash_indices,
                        )
                else:
                    _module_forward(hidden_states, actual_logits, input_ids, deepseek_hash_indices)
            elif use_nvfp4:
                if distributed == "power_law":
                    for tw, ti in zip(topk_weights_list, topk_ids_list, strict=True):
                        _run_nvfp4_once(hidden_states[: tw.shape[0]], tw, ti)
                else:
                    _run_nvfp4_once(hidden_states, topk_weights, topk_ids)
            elif distributed in {"power_law", "sampled_zipf"}:
                for i, (tw, ti) in enumerate(zip(topk_weights_list, topk_ids_list, strict=True)):
                    local_num_tokens = tw.shape[0]
                    if use_int4_wo:
                        tw = tw.float()
                    _ = fused_experts(
                        hidden_states[:local_num_tokens],
                        w1,
                        w2,
                        tw,
                        ti,
                        inplace=False,
                        quant_config=quant_config,
                        global_num_experts=num_experts,
                        expert_map=expert_map,
                    )
            else:
                routed_weights = topk_weights.float() if use_int4_wo else topk_weights
                _ = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    routed_weights,
                    topk_ids,
                    inplace=False,
                    quant_config=quant_config,
                    global_num_experts=num_experts,
                    expert_map=expert_map,
                )

        def run_iterations():
            # Use benchmark_with_power context manager
            with benchmark_with_power(
                device=device,
                kernel_func=run_single_iteration,
                num_warmups=num_warmups,
                num_runs=num_runs,
                repeat_n=1,
                allow_graph_fail=True,
                use_cuda_graph=use_cuda_graph,
            ) as results:
                pass

            return results["latency_ms"] / num_iter, results["power_stats"]

        try:
            vllm_cfg = module_vllm_cfg if (use_mxfp4 or use_unquantized_module) else VllmConfig()
            with set_current_vllm_config(vllm_cfg), set_forward_context({}, vllm_cfg):
                latency, power_stats = run_iterations()
        except torch.OutOfMemoryError:
            # If OOM, check if we had at least one successful run.
            if num_tokens_idx > 0:
                break
            raise

        print(f"moe latency: {latency}")

        if use_qwen_shared_module:
            source = "vllm_qwen_fused_moe_shared"
        elif use_mxfp4:
            source = "vllm_mxfp4_moe"
        elif use_unquantized_module:
            source = "vllm_fused_moe_module"
        elif use_nvfp4:
            source = "vllm_flashinfer_trtllm_moe_fp4"
        elif use_int4_wo:
            source = "vllm_marlin_int4_moe"
        else:
            source = "vllm_fused_moe"

        log_perf(
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
                    "distribution": _moe_distribution_label(
                        distributed,
                        power_law_alpha,
                        use_cuda_graph=use_cuda_graph,
                    ),
                    "latency": latency,
                    "vllm_max_model_len": max_model_len or "",
                    "vllm_max_num_seqs": max_num_seqs or "",
                    "vllm_max_num_batched_tokens": max_num_batched_tokens or "",
                    "used_cuda_graph": use_cuda_graph,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name="moe",
            kernel_source=source,
            perf_filename=perf_filename,
            power_stats=power_stats,
        )


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_moe_test_cases()
    print(f"Total test cases: {len(test_cases)}")

    for test_case in test_cases[:4]:
        print(f"Running test case: {test_case}")
        try:
            run_moe_torch(*test_case, perf_filename=PerfFile.MOE)
        except Exception as e:
            print(f"Test case failed: {test_case}")
            print(f"Error: {e}")
            continue
