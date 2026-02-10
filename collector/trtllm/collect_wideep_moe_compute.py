# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
WideEPMoE compute-only collector (excluding AlltoAll communication).

Simulates EP size > 1 scenario, single EP rank compute workload.
Supports EPLB (Expert Parallel Load Balancer) scenarios.

Reference: aic/collector/trtllm/collect_wideep_moe.py
"""

import gc
import glob
import json
import os
import sys
from typing import Optional

import tensorrt_llm
import torch
import torch.nn as nn
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3Gate
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.modules.fused_moe.ops.moe_op import MoEOp, MoEOpSelector
from tensorrt_llm._torch.modules.fused_moe.ops.moe_op_cutlass import CutlassMoEOp
from tensorrt_llm._torch.modules.fused_moe.ops.moe_op_deepgemm import DeepGemmMoEOp
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod,
    FP8QDQFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
    UnquantizedFusedMoEMethod,
)
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

try:
    from common_test_cases import get_common_moe_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common_test_cases import get_common_moe_test_cases

    from helper import (
        EXIT_CODE_RESTART,
        balanced_logits,
        benchmark_with_power,
        get_sm_version,
        log_perf,
        power_law_logits_v3,
    )


aic_debug = int(os.getenv("AIC_MOE_DEBUG", "0"))

# Control simulation mode via environment variable:
# AIC_ACCURATE_WIDEEP_SIM=1 (default): Accurate mode - DP split + rank0 filter
# AIC_ACCURATE_WIDEEP_SIM=0: Simple mode - all tokens directly
aic_accurate_wideep_sim = os.getenv("AIC_ACCURATE_WIDEEP_SIM", "1") == "1"

moe_tune_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wideep_moe_compute_tuned_cache_path")


def cleanup_empty_json_files(directory):
    if not os.path.exists(directory):
        return

    json_files = glob.glob(os.path.join(directory, "*.json"))
    deleted_count = 0

    for file_path in json_files:
        try:
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted empty file: {file_path}")
            else:
                with open(file_path) as f:
                    data = json.load(f)
                    if not data:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted empty JSON content: {file_path}")
        except (OSError, json.JSONDecodeError) as e:
            try:
                os.remove(file_path)
                deleted_count += 1
                print(f"Deleted invalid file: {file_path} (Error: {e})")
            except OSError:
                pass

    if deleted_count > 0:
        print(f"Total deleted {deleted_count} invalid JSON files from {directory}")


# =============================================================================
# WideEPMoE Compute Simulator - simulates WideEPMoE computation on single GPU
# =============================================================================
class WideEPMoEComputeSimulator(nn.Module):
    """
    Simulates WideEPMoE's computation path on SINGLE GPU.

    Uses the SAME kernel selection logic as WideEPMoE (MoEOpSelector.select_op).

    Simulate EP size > 1 scenario:
    - expert_size_per_partition = num_slots / ep_size (local slot count per rank)
    - weight shape: [expert_size_per_partition, ...]
    - EPLB num_slots >= num_experts
    """

    def __init__(
        self,
        routing_method,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
        quant_config: Optional[QuantConfig] = None,
        ep_size: int = 1,
        tp_size: int = 1,
        num_slots: Optional[int] = None,  # EPLB: num_slots >= num_experts
        force_kernel: str = "auto",
    ):
        super().__init__()

        self.routing_method = routing_method
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.quant_config = quant_config

        self.ep_size = ep_size
        self.tp_size = tp_size

        # EPLB support: num_slots >= num_experts for redundancy
        self.num_slots = num_slots if num_slots is not None else num_experts

        # expert_size_per_partition based on num_slots (not num_experts)
        self.expert_size_per_partition = self.num_slots // ep_size
        self.intermediate_size_per_partition = intermediate_size // tp_size
        # expand_intermediate_size_per_partition = intermediate_size_per_partition * 2 (for gated activation)
        self.expand_intermediate_size_per_partition = self.intermediate_size_per_partition * 2

        # For MoEOpSelector compatibility
        self.has_deepseek_fp8_block_scales = (
            quant_config is not None and quant_config.quant_algo == QuantAlgo.FP8_BLOCK_SCALES
        )
        self.has_fp8_qdq = quant_config is not None and quant_config.quant_algo == QuantAlgo.FP8
        self.has_nvfp4 = quant_config is not None and quant_config.quant_algo == QuantAlgo.NVFP4
        # Additional quant mode flags required by CutlassMoEOp
        self.has_w4a16_mxfp4 = False
        self.has_w4afp8 = False
        self.has_int8_woq_per_channel = False
        self.has_mxfp8_act_scaling = False

        # For moe_op compatibility
        self.is_gated_activation = True
        self.intermediate_size_expand_ratio = 2
        self.activation_type = 5  # ActivationType.Swiglu = 5 (gated activation)

        # Parallelism attributes required by MoERunner
        self.tp_rank = 0
        self.ep_rank = 0
        self.cluster_size = 1
        self.cluster_rank = 0
        self.tune_max_num_tokens = 8192

        # Swiglu parameters (None for standard Swiglu activation)
        # WideEPMoE does not support swiglu_alpha/beta/limit
        self.swiglu_alpha = None
        self.swiglu_beta = None
        self.swiglu_limit = None

        # For unpadded_hidden_size (used in compute_moe)
        self.unpadded_hidden_size = hidden_size

        self.quant_method = self._get_quant_method()
        self._create_weights()

        self.force_kernel = force_kernel
        self._moe_op = None

    def _get_quant_method(self):
        if self.quant_config is None:
            return UnquantizedFusedMoEMethod()
        if self.has_deepseek_fp8_block_scales:
            return DeepSeekFP8BlockScalesFusedMoEMethod()
        elif self.has_fp8_qdq:
            return FP8QDQFusedMoEMethod()
        elif self.has_nvfp4:
            return NVFP4CutlassFusedMoEMethod()
        return UnquantizedFusedMoEMethod()

    def _create_weights(self):
        """
        Create weights based on quantization mode, following TensorRT-LLM's implementation.
        Weights are created per expert_size_per_partition (= num_slots / ep_size).
        """
        device = "cuda"

        if self.has_nvfp4:
            # NVFP4: weights packed as int64 (16 fp4 values per int64)
            weight_dtype = torch.int64
            weight_vec_size = 16

            nvfp4_row_align = 128
            expand_inter = self.expand_intermediate_size_per_partition
            intermediate_size_expand_aligned = (expand_inter + nvfp4_row_align - 1) // nvfp4_row_align * nvfp4_row_align

            w3_w1_shape = (
                self.expert_size_per_partition,
                intermediate_size_expand_aligned,
                self.hidden_size // weight_vec_size,
            )
            w2_shape = (
                self.expert_size_per_partition,
                self.hidden_size,
                intermediate_size_expand_aligned // self.intermediate_size_expand_ratio // weight_vec_size,
            )

            self.w3_w1_weight = torch.randint(
                low=-(2**62), high=2**62, size=w3_w1_shape, dtype=weight_dtype, device=device
            )
            self.w2_weight = torch.randint(low=-(2**62), high=2**62, size=w2_shape, dtype=weight_dtype, device=device)

        elif self.has_deepseek_fp8_block_scales or self.has_fp8_qdq:
            # FP8: weights are float8_e4m3fn
            weight_dtype = torch.float8_e4m3fn

            w3_w1_shape = (
                self.expert_size_per_partition,
                self.expand_intermediate_size_per_partition,
                self.hidden_size,
            )
            w2_shape = (
                self.expert_size_per_partition,
                self.hidden_size,
                self.intermediate_size_per_partition,
            )

            # Use CPU to avoid CUDA RNG state pollution from previous tasks
            self.w3_w1_weight = torch.randn(w3_w1_shape, dtype=torch.bfloat16, device="cpu").to(weight_dtype).to(device)
            self.w2_weight = torch.randn(w2_shape, dtype=torch.bfloat16, device="cpu").to(weight_dtype).to(device)

        else:
            # Unquantized: weights are bfloat16
            weight_dtype = self.dtype

            w3_w1_shape = (
                self.expert_size_per_partition,
                self.expand_intermediate_size_per_partition,
                self.hidden_size,
            )
            w2_shape = (
                self.expert_size_per_partition,
                self.hidden_size,
                self.intermediate_size_per_partition,
            )

            # Use CPU to avoid CUDA RNG state pollution from previous tasks
            self.w3_w1_weight = torch.randn(w3_w1_shape, dtype=weight_dtype, device="cpu").to(device)
            self.w2_weight = torch.randn(w2_shape, dtype=weight_dtype, device="cpu").to(device)

        # Bias (set to None, same as WideEPMoE default)
        self.w3_w1_bias = None
        self.w2_bias = None

        self._create_input_quant_scales(device)
        self.quant_scales = self._create_quant_scales(device)

    def _create_input_quant_scales(self, device):
        if self.has_fp8_qdq:
            self.fc31_input_dequant = torch.ones(1, dtype=torch.float32, device=device)
        else:
            self.fc31_input_dequant = None

        if self.has_nvfp4:
            self.fc31_input_scale = torch.ones(1, dtype=torch.float32, device=device)
            self.scaling_vector_size = 16
        else:
            self.fc31_input_scale = None
            self.scaling_vector_size = None

    def _create_quant_scales(self, device):
        if self.quant_config is None:
            return []

        if self.has_deepseek_fp8_block_scales:
            block_size = 128
            fc_scale_shape = (
                self.expert_size_per_partition,
                2 * self.intermediate_size_per_partition // block_size,
                self.hidden_size // block_size,
            )
            proj_scale_shape = (
                self.expert_size_per_partition,
                self.hidden_size // block_size,
                self.intermediate_size_per_partition // block_size,
            )
            from tensorrt_llm._torch.modules.fused_moe.quantization import (
                FusedMoEQuantScalesDeepSeekFP8BlockScales,
            )

            return FusedMoEQuantScalesDeepSeekFP8BlockScales(
                fc_weight_scales=torch.ones(fc_scale_shape, dtype=torch.float32, device=device),
                proj_weight_scales=torch.ones(proj_scale_shape, dtype=torch.float32, device=device),
            )
        elif self.has_fp8_qdq:
            from tensorrt_llm._torch.modules.fused_moe.quantization import FusedMoEQuantScalesFP8

            return FusedMoEQuantScalesFP8(
                fc1_dequant=torch.ones(self.expert_size_per_partition, dtype=torch.float32, device=device),
                fc2_quant=torch.ones(1, dtype=torch.float32, device=device),
                fc2_dequant=torch.ones(self.expert_size_per_partition, dtype=torch.float32, device=device),
                fc1_input_dequant=torch.ones(1, dtype=torch.float32, device=device),
            )
        elif self.has_nvfp4:
            from tensorrt_llm._torch.modules.fused_moe.quantization import FusedMoEQuantScalesNVFP4

            nvfp4_row_align = 128
            block_scales_vec_size = 4

            expand_inter = self.expand_intermediate_size_per_partition
            intermediate_size_expand_aligned = (expand_inter + nvfp4_row_align - 1) // nvfp4_row_align * nvfp4_row_align

            w3_w1_weight_scale_shape = (
                self.expert_size_per_partition,
                intermediate_size_expand_aligned,
                self.hidden_size // self.scaling_vector_size // block_scales_vec_size,
            )
            scale_denom = self.intermediate_size_expand_ratio * self.scaling_vector_size * block_scales_vec_size
            w2_weight_scale_shape = (
                self.expert_size_per_partition,
                self.hidden_size,
                intermediate_size_expand_aligned // scale_denom,
            )

            return FusedMoEQuantScalesNVFP4(
                fc1_act_global=torch.tensor(1.0, dtype=torch.float32, device=device),
                fc1_weight_block=torch.ones(w3_w1_weight_scale_shape, dtype=torch.int32, device=device),
                fc1_global=torch.ones(self.expert_size_per_partition, dtype=torch.float32, device=device),
                fc2_act_global=torch.tensor(1.0, dtype=torch.float32, device=device),
                fc2_weight_block=torch.ones(w2_weight_scale_shape, dtype=torch.int32, device=device),
                fc2_global=torch.ones(self.expert_size_per_partition, dtype=torch.float32, device=device),
            )
        return []

    @property
    def moe_op(self) -> MoEOp:
        if self._moe_op is None:
            if self.force_kernel == "deepgemm":
                self._moe_op = DeepGemmMoEOp()
            elif self.force_kernel == "cutlass":
                self._moe_op = CutlassMoEOp()
            else:
                self._moe_op = MoEOpSelector.select_op(self)
        return self._moe_op

    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, do_finalize: bool = True
    ) -> torch.Tensor:
        """
        Run FULL MoE computation pipeline (no communication).
        Same steps as WideEPMoE.forward_chunk.
        """
        x = hidden_states
        x_sf = None
        output_dtype = torch.bfloat16

        # Step 1: Routing (WideEPMoE line 404-405)
        if aic_debug == 1:
            print(f"[wideep_moe_compute_eplb normal mode] router_logits.shape: {router_logits.shape}")
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        token_selected_slots = token_selected_experts

        # Step 2: Input Quantization (WideEPMoE line 501-531)
        if self.has_fp8_qdq:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.fc31_input_dequant)
        elif self.has_nvfp4:
            x_row = x.shape[0]
            x, x_sf = torch.ops.trtllm.fp4_quantize(
                x, self.fc31_input_scale, self.scaling_vector_size, sfUseUE8M0=False, isSfSwizzledLayout=False
            )
            x_sf = x_sf.view((x_row, -1))
        elif self.has_deepseek_fp8_block_scales:
            pass

        # Step 3: MoE Computation (WideEPMoE line 624-642)
        if aic_debug == 1:
            print(
                f"[wideep_moe_compute_eplb normal mode] x.shape: {x.shape}, "
                f"token_selected_slots.shape: {token_selected_slots.shape}, "
                f"token_final_scales.shape: {token_final_scales.shape}"
            )
        output = self.moe_op.run_moe(
            module=self,
            input=x,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            w3_w1_weight=self.w3_w1_weight,
            w3_w1_bias=None,
            w2_weight=self.w2_weight,
            w2_bias=None,
            output_dtype=output_dtype,
            quant_scales=self.quant_scales,
            use_all_to_all=False,
            input_sf=x_sf,
            swizzled_input_sf=False,
            min_latency_mode=False,
            use_fused_finalize=do_finalize,
        )

        if isinstance(output, (list, tuple)):
            output = output[0]

        return output

    def forward_router_only(self, router_logits: torch.Tensor):
        """
        Execute ONLY the routing computation.
        Used to simulate the router computation of a single DP rank in WideEP.

        In WideEP, DP size = EP size, so each DP rank has num_tokens/ep_size tokens.
        This method simulates that router computation cost.

        Args:
            router_logits: [num_tokens, num_experts/num_slots] tensor

        Returns:
            token_selected_slots: [num_tokens, topk] selected slot indices
            token_final_scales: [num_tokens, topk] routing weights
        """
        # print(f"[wideep_moe_compute_eplb accurate mode] router_logits.shape: {router_logits.shape}")
        token_selected_experts, token_final_scales = self.routing_method.apply(router_logits)
        token_selected_slots = token_selected_experts
        return token_selected_slots, token_final_scales

    def forward_moe_only(
        self,
        hidden_states: torch.Tensor,
        token_selected_slots: torch.Tensor,
        token_final_scales: torch.Tensor,
        do_finalize: bool = True,
    ) -> torch.Tensor:
        """
        Execute ONLY the MoE computation (quantization + MoE kernel), NO routing.
        Used to simulate the MoE computation of EP rank 0 in WideEP.

        The hidden_states and routing results should be pre-filtered to only include
        tokens that are routed to this EP rank.

        Args:
            hidden_states: [rank0_num_tokens, hidden_size] filtered hidden states
            token_selected_slots: [rank0_num_tokens, topk] pre-computed slot assignments
            token_final_scales: [rank0_num_tokens, topk] pre-computed routing weights
            do_finalize: Whether to finalize the output

        Returns:
            output: [rank0_num_tokens, hidden_size] MoE output
        """
        x = hidden_states
        x_sf = None
        output_dtype = torch.bfloat16

        # NO routing here - routing results are passed as arguments

        # Step 2: Input Quantization
        if self.has_fp8_qdq:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(x, self.fc31_input_dequant)
        elif self.has_nvfp4:
            x_row = x.shape[0]
            x, x_sf = torch.ops.trtllm.fp4_quantize(
                x, self.fc31_input_scale, self.scaling_vector_size, sfUseUE8M0=False, isSfSwizzledLayout=False
            )
            x_sf = x_sf.view((x_row, -1))
        elif self.has_deepseek_fp8_block_scales:
            pass

        # Step 3: MoE Computation

        # Count the actual number of token-slot pairs to compute on this EP rank
        # token_selected_slots < expert_size_per_partition means local slot
        # local_slot_mask = token_selected_slots < self.expert_size_per_partition
        # total_local_computations = local_slot_mask.sum().item()
        output = self.moe_op.run_moe(
            module=self,
            input=x,
            token_selected_slots=token_selected_slots,
            token_final_scales=token_final_scales,
            w3_w1_weight=self.w3_w1_weight,
            w3_w1_bias=None,
            w2_weight=self.w2_weight,
            w2_bias=None,
            output_dtype=output_dtype,
            quant_scales=self.quant_scales,
            use_all_to_all=False,
            input_sf=x_sf,
            swizzled_input_sf=False,
            min_latency_mode=False,
            use_fused_finalize=do_finalize,
        )

        if isinstance(output, (list, tuple)):
            output = output[0]

        return output


# =============================================================================
# Test Cases Generation
# =============================================================================
def get_wideep_moe_compute_all_test_cases():
    """
    Generate all test cases for WideEP MoE compute with three EPLB modes.

    Three EPLB configurations:
    1. EPLB OFF: use_eplb=False, num_slots=num_experts
    2. EPLB ON (baseline): use_eplb=True, num_slots=num_experts
    3. EPLB ON (redundant): use_eplb=True, num_slots=288

    Notes:
    - Only uses cutlass kernel (no deepgemm)
    - Only uses min_latency_mode=False
    - Only for DeepSeek-V3 model with power_law distribution
    """
    moe_list = []
    if get_sm_version() > 86 and get_sm_version() < 100:
        moe_list += ["fp8_block"]
    if get_sm_version() >= 100:
        moe_list += ["nvfp4"]  # SM100+ uses nvfp4

    test_cases = []

    for common_moe_testcase in get_common_moe_test_cases():
        # Focus on DeepSeek-V3 model only
        model_name = common_moe_testcase.model_name
        if "deepseek" not in model_name.lower() or "v3" not in model_name.lower():
            continue

        # Only power_law distribution (skip balanced)
        if common_moe_testcase.token_expert_distribution != "power_law":
            continue

        # WideEP requires: tp=1, ep>1, num_gpu>1
        if common_moe_testcase.tp != 1:
            continue
        if common_moe_testcase.ep <= 1:
            continue

        num_experts = common_moe_testcase.num_experts
        ep_size = common_moe_testcase.ep

        # Three EPLB configurations: (use_eplb, num_slots)
        eplb_configs = [
            (False, num_experts),  # EPLB OFF
            (True, num_experts),  # EPLB ON, baseline slots
            (True, 288),  # EPLB ON, 288 slots
        ]

        for moe_type in moe_list:
            for use_eplb, num_slots in eplb_configs:
                # Skip if num_slots is not divisible by ep_size
                if num_slots % ep_size != 0:
                    continue

                test_cases.append(
                    [
                        moe_type,
                        "cutlass",  # Only use cutlass kernel (no deepgemm)
                        common_moe_testcase.num_tokens_list,
                        common_moe_testcase.hidden_size,
                        common_moe_testcase.inter_size,
                        common_moe_testcase.topk,
                        num_experts,
                        common_moe_testcase.tp,
                        ep_size,
                        False,  # min_latency_mode = False (no min_latency_mode)
                        common_moe_testcase.model_name,
                        None,  # perf_filename - determined dynamically based on use_eplb
                        common_moe_testcase.token_expert_distribution,
                        common_moe_testcase.power_law_alpha,
                        use_eplb,
                        num_slots,
                        aic_accurate_wideep_sim,  # accurate_wideep_simulation (from env)
                    ]
                )

    return test_cases


# =============================================================================
# Main Benchmark Function
# =============================================================================
def run_wideep_moe_compute(
    moe_type,
    moe_kernel,
    num_tokens_lists,
    hidden_size,
    inter_size,
    topk,
    num_experts,
    moe_tp_size,
    moe_ep_size,
    min_latency_mode,
    model_name,
    perf_filename,
    distributed="power_law",
    power_law_alpha=0.0,
    use_eplb=False,
    num_slots=None,
    accurate_wideep_simulation=True,
    device="cuda:0",
):
    """
    Benchmark WideEP MoE computation (excluding communication).

    Args:
        accurate_wideep_simulation: If True, uses accurate WideEP simulation:
            - Router uses dp_num_tokens = num_tokens / ep_size (single DP rank)
            - MoE uses only tokens routed to EP rank 0
            This accurately simulates the real WideEP scenario where DP size = EP size.

            If False, uses all tokens directly (original mode):
            - Router and MoE both use full num_tokens
            This is simpler but less accurate for WideEP simulation.
    """
    # Default num_slots to num_experts (no redundancy)
    if num_slots is None:
        num_slots = num_experts

    device = torch.device(device)
    torch.cuda.set_device(device)
    torch.set_default_device(device)

    if aic_debug == 1:
        print("WideEP MOE Compute Allocated GDRAM:", torch.cuda.memory_allocated(device.index) / 1024**2, "MB")
        print("WideEP MOE Compute Reserved GDRAM:", torch.cuda.memory_reserved(device) / 1024**2, "MB")

    # =========================================================================
    # Setup quantization config (same as collect_moe.py)
    # =========================================================================
    dtype = torch.bfloat16
    quant_group_size = 128
    quant_algo = None

    if moe_type == "fp8_block":
        quant_algo = QuantAlgo.FP8_BLOCK_SCALES
        dtype = torch.float8_e4m3fn
    elif moe_type == "fp8":
        quant_algo = QuantAlgo.FP8
        dtype = torch.float8_e4m3fn
    elif moe_type == "nvfp4":
        quant_algo = QuantAlgo.NVFP4
        quant_group_size = 16

    if power_law_alpha - 0.0 < 1e-6:
        distributed = "balanced"

    quant_config = QuantConfig(
        quant_algo=quant_algo,
        kv_cache_quant_algo=None,
        group_size=quant_group_size,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    # =========================================================================
    # Setup routing method
    # =========================================================================
    router_logits_dtype = torch.bfloat16
    if min_latency_mode:
        n_group = 8
        topk_group = 4
        routed_scaling_factor = 2.5
        routing_method = DeepseekV3Gate(
            hidden_size,
            num_experts,
            top_k=topk,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            dtype=dtype,
            moe_backend="TRTLLM",
        ).routing_method
        router_logits_dtype = torch.float32
    else:
        routing_method = RenormalizeMoeRoutingMethod(topk)

    # =========================================================================
    # Create WideEPMoE Compute Simulator
    # =========================================================================
    moe = WideEPMoEComputeSimulator(
        routing_method=routing_method,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=inter_size,
        dtype=dtype,
        quant_config=quant_config,
        ep_size=moe_ep_size,
        tp_size=moe_tp_size,
        num_slots=num_slots,  # EPLB support
        force_kernel=moe_kernel,
    )
    moe.to(torch.device(device))

    # eplb_str = f", EPLB slots={num_slots}" if use_eplb else ""
    # sim_mode = "accurate (DP split + rank0 filter)" if accurate_wideep_simulation else "simple (all tokens)"
    # print(f"\n{'='*60}")
    # print(f"WideEPMoE Compute Collection")
    # print(f"{'='*60}")
    # print(f"Model: {model_name}")
    # print(f"Quantization: {moe_type}, Kernel: {moe_kernel}")
    # print(f"EP config: ep_size={moe_ep_size}, tp_size={moe_tp_size}")
    # print(f"Expert config: total={num_experts}, local_slots={moe.expert_size_per_partition}{eplb_str}")
    # print(f"Size config: hidden={hidden_size}, inter={inter_size}, inter_local={moe.intermediate_size_per_partition}")
    # print(f"Routing: topk={topk}, min_latency={min_latency_mode}")
    # print(f"Simulation mode: {sim_mode}")
    # print(f"{'='*60}\n")

    # =========================================================================
    # Dry run
    # =========================================================================
    torch.cuda.synchronize()
    max_tokens = num_tokens_lists[-1]
    for i in range(len(num_tokens_lists)):
        max_tokens = num_tokens_lists[-i - 1]
        try:
            hidden_states_max_tokens = torch.randn([max_tokens, hidden_size], device="cpu").bfloat16().to(device)
            # Use num_slots for routing (EPLB routes to slots, not experts)
            logits_max_tokens = balanced_logits(max_tokens, num_slots, topk).to(router_logits_dtype).to(device)
            moe.forward(hidden_states_max_tokens, logits_max_tokens, do_finalize=not min_latency_mode)
            torch.cuda.synchronize()
            print(f"[dry run] Successfully dry run for {max_tokens} tokens")
            break
        except Exception as e:
            print(f"[dry run] Failed for {max_tokens} tokens: {e}, trying smaller size...")
            if i == len(num_tokens_lists) - 1:
                RuntimeError(f"dry run failed for {max_tokens} tokens: {e}")
            else:
                continue

    # =========================================================================
    # AutoTuner
    # =========================================================================
    cleanup_empty_json_files(moe_tune_path)
    inter_local = inter_size // moe_tp_size
    slots_local = num_slots // moe_ep_size
    cache_path = f"{moe_tune_path}/wideep_compute_{moe_kernel}_{moe_type}_{hidden_size}_{inter_local}_{slots_local}"
    existing_files = glob.glob(f"{cache_path}*")
    cache_loaded = False
    if existing_files:
        json_path = existing_files[0]
        try:
            AutoTuner.get().profiling_cache.load_cache(json_path)
            cache_loaded = True
            print(f"Loaded profiling cache from {json_path}")
        except (OSError, json.JSONDecodeError):
            pass

    if not cache_loaded:
        torch.cuda.synchronize()
        for i in range(len(num_tokens_lists)):
            max_tokens_for_tuning = num_tokens_lists[-i - 1]
            if max_tokens_for_tuning > max_tokens:
                continue
            else:
                try:
                    with torch.inference_mode(), autotune(cache_path=cache_path):
                        moe.forward(
                            hidden_states_max_tokens[:max_tokens_for_tuning],
                            logits_max_tokens[:max_tokens_for_tuning],
                            do_finalize=not min_latency_mode,
                        )
                    torch.cuda.synchronize()
                    break  # Tuning succeeded, exit loop
                except Exception as e:
                    print(f"tune failed for {max_tokens_for_tuning} tokens: {e}, fallback to smaller tokens")
                    continue

    del hidden_states_max_tokens, logits_max_tokens

    # =========================================================================
    # Benchmark each token count
    # =========================================================================
    for num_tokens in num_tokens_lists:
        if num_tokens > max_tokens:
            continue

        num_iter = 5 if distributed == "power_law" else 1

        # In WideEP, DP size = EP size, each DP rank has num_tokens/ep_size tokens
        dp_num_tokens = num_tokens if num_tokens < moe_ep_size else num_tokens // moe_ep_size

        # Variables for logging
        actual_dp_tokens = None
        actual_rank0_tokens = None

        if accurate_wideep_simulation:
            # =====================================================================
            # ACCURATE MODE: Simulates real WideEP scenario
            # - Router uses dp_num_tokens (single DP rank)
            # - MoE uses only tokens routed to EP rank 0
            # =====================================================================
            if distributed == "power_law":
                # Generate ONE distribution to get rank0 info with EPLB slot assignments
                _, rank0_info = power_law_logits_v3(
                    num_tokens,
                    num_experts,
                    topk,
                    moe_ep_size,
                    power_law_alpha,
                    use_eplb=use_eplb,
                    num_slots=num_slots,
                    return_rank0_info=True,
                )
                rank0_num_tokens = rank0_info["rank0_num_tokens"]
                rank0_total_selections = rank0_info["rank0_total_selections"]
                rank0_logits = rank0_info["rank0_logits"].to(router_logits_dtype).to(device)

                # Use EPLB's actual slot assignments (the TRUE distribution after load balancing)
                rank0_selected_slots = rank0_info["rank0_selected_slots"].to(torch.int32).to(device)
                # Compute scales from logits using EPLB's slot assignments
                gathered_logits = torch.gather(rank0_logits, 1, rank0_selected_slots.long())
                token_final_scales = torch.softmax(gathered_logits, dim=1).to(torch.float32)

                # Create hidden states for rank0 tokens
                rank0_hidden = torch.randn([rank0_num_tokens, hidden_size]).bfloat16().to(torch.device(device))

                # Create dummy router logits for DP rank (length = num_tokens / ep_size)
                # This simulates the router computation on a single DP rank
                # Random logits for router computation (just need correct shape, not actual distribution)
                dummy_router_logits = torch.randn([dp_num_tokens, num_slots]).to(router_logits_dtype).to(device)

                # Store data for forward_router_only + forward_moe_only
                rank0_data_list = [
                    {
                        "hidden": rank0_hidden,
                        "token_selected_slots": rank0_selected_slots,
                        "token_final_scales": token_final_scales,
                        "num_tokens": rank0_num_tokens,
                        "router_logits": dummy_router_logits,  # for router computation
                    }
                    for _ in range(num_iter)
                ]
                total_selections_list = [rank0_total_selections] * num_iter

                avg_rank0_tokens = sum(d["num_tokens"] for d in rank0_data_list) / len(rank0_data_list)
                avg_rank0_selections = sum(total_selections_list) / len(total_selections_list)
                actual_dp_tokens = dp_num_tokens
                actual_rank0_tokens = int(avg_rank0_tokens)
                eplb_str = f"slots={num_slots}" if use_eplb else ""
                print(
                    f"[accurate] power_law: total={num_tokens}, rank0_tokens={avg_rank0_tokens:.0f}, "
                    f"rank0_selections={avg_rank0_selections:.0f}, ep={moe_ep_size} {eplb_str}"
                )
            else:  # balanced
                hidden_states = torch.randn([dp_num_tokens, hidden_size], device="cpu").bfloat16().to(device)
                actual_logits = balanced_logits(dp_num_tokens, num_slots, topk).to(router_logits_dtype).to(device)
                actual_dp_tokens = dp_num_tokens
                actual_rank0_tokens = dp_num_tokens
                print(f"[accurate] balanced: total={num_tokens}, dp={dp_num_tokens}, ep={moe_ep_size}")
        else:
            # =====================================================================
            # SIMPLE MODE: Uses all tokens directly (original behavior)
            # - Router and MoE both use full num_tokens
            # =====================================================================
            hidden_states = torch.randn([num_tokens, hidden_size], device="cpu").bfloat16().to(device)

            if distributed == "power_law":
                actual_logits_list = [
                    power_law_logits_v3(
                        num_tokens,
                        num_experts,
                        topk,
                        moe_ep_size,
                        power_law_alpha,
                        use_eplb=use_eplb,
                        num_slots=num_slots,
                    )
                    .to(router_logits_dtype)
                    .to(device)
                    for _ in range(num_iter)
                ]
                eplb_str = f"_eplb_slots{num_slots}" if use_eplb else ""
                print(f"[simple] power_law: num_tokens={num_tokens}, ep={moe_ep_size}{eplb_str}")
            else:  # balanced
                actual_logits = balanced_logits(num_tokens, num_slots, topk).to(router_logits_dtype).to(device)
                print(f"[simple] balanced: num_tokens={num_tokens}, ep={moe_ep_size}")

            actual_dp_tokens = num_tokens  # use all tokens
            actual_rank0_tokens = num_tokens

        # =====================================================================
        # Helper closure to encapsulate forward pass logic (same as collect_moe.py)
        # =====================================================================
        moe_ref = moe  # local ref for closure

        def run_forward_pass():
            """Execute one forward pass through WideEP MOE simulator."""
            if accurate_wideep_simulation:
                if distributed == "power_law":
                    for data in rank0_data_list:
                        moe_ref.forward_router_only(data["router_logits"])
                        moe_ref.forward_moe_only(
                            data["hidden"],
                            data["token_selected_slots"],
                            data["token_final_scales"],
                            do_finalize=not min_latency_mode,
                        )
                else:
                    moe_ref.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)
            else:
                if distributed == "power_law":
                    for logits in actual_logits_list:
                        moe_ref.forward(hidden_states, logits, do_finalize=not min_latency_mode)
                else:
                    moe_ref.forward(hidden_states, actual_logits, do_finalize=not min_latency_mode)

        # =====================================================================
        # Benchmark with automatic power measurement and graph fallback
        # (same pattern as collect_moe.py)
        # =====================================================================
        num_warmups = 1 if distributed == "power_law" else 3
        num_runs = 1 if distributed == "power_law" else 6

        with benchmark_with_power(
            device=device,
            kernel_func=run_forward_pass,
            num_warmups=num_warmups,
            num_runs=num_runs,
            repeat_n=1,
            allow_graph_fail=True,
        ) as results:
            # Calculate per-iteration latency (accounting for internal iterations)
            latency = results["latency_ms"] / num_iter
            power_stats = results["power_stats"]

            # Always print CUDA graph status for debugging
            print(f"[DEBUG] used_cuda_graph={results['used_cuda_graph']}, latency_ms={results['latency_ms']:.4f}")
            if not results["used_cuda_graph"]:
                print(f"CUDA graph capture failed for {num_tokens} tokens, used eager execution fallback")

        # Determine source name
        if min_latency_mode:
            source = f"wideep_compute_{moe_kernel}_min_latency"
        else:
            source = f"wideep_compute_{moe_kernel}"

        # Build distribution string with EPLB suffix if enabled
        if distributed == "power_law":
            dist_str = f"power_law_{power_law_alpha}"
            if use_eplb:
                dist_str += "_eplb"
        else:
            dist_str = distributed

        # Simulation mode string
        sim_mode_str = "accurate" if accurate_wideep_simulation else "simple"

        # Log performance
        log_perf(
            item_list=[
                {
                    "moe_dtype": moe_type,
                    "moe_kernel": moe_kernel,
                    "num_tokens": num_tokens,  # Total tokens (input parameter)
                    "dp_num_tokens": actual_dp_tokens,  # Tokens per DP rank (router input)
                    "rank0_num_tokens": actual_rank0_tokens,  # Tokens actually computed by EP rank 0
                    "hidden_size": hidden_size,
                    "inter_size": inter_size,
                    "topk": topk,
                    "num_experts": num_experts,
                    "num_slots": num_slots,
                    "moe_tp_size": moe_tp_size,
                    "moe_ep_size": moe_ep_size,
                    "distribution": dist_str,
                    "simulation_mode": sim_mode_str,
                    "latency": latency,
                }
            ],
            framework="TRTLLM",
            version=tensorrt_llm.__version__,
            device_name=torch.cuda.get_device_name(device),
            op_name="wideep_moe" if not use_eplb else "wideep_moe_eplb",
            kernel_source=source,
            perf_filename="wideep_moe_perf.txt",
            power_stats=power_stats,
        )

        # print(f"WideEP MOE Compute | total={num_tokens}, dp={actual_dp_tokens}, rank0={actual_rank0_tokens}, "
        #       f"kernel={moe_kernel}, latency={latency:.3f}ms, dist={dist_str}, mode={sim_mode_str}")

        # Cleanup iteration
        if accurate_wideep_simulation:
            if distributed == "power_law":
                del rank0_data_list
            else:
                del actual_logits, hidden_states
        else:
            if distributed == "power_law":
                del actual_logits_list, hidden_states
            else:
                del actual_logits, hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    # =========================================================================
    # Cleanup and exit
    # =========================================================================
    del moe
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()
    AutoTuner.get().clear_cache()

    # Exit the worker process to ensure complete resource cleanup
    sys.exit(EXIT_CODE_RESTART)


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    all_test_cases = get_wideep_moe_compute_all_test_cases()

    print(f"Running {len(all_test_cases)} WideEP MOE compute test configurations...")

    for i, test_case in enumerate(all_test_cases):
        print(f"\nProgress: {i + 1}/{len(all_test_cases)}")
        print(f"  Config: moe_type={test_case[0]}, kernel={test_case[1]}, model={test_case[10]}, eplb={test_case[14]}")
        run_wideep_moe_compute(*test_case)
