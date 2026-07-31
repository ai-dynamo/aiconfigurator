# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import aiconfigurator_core.sdk.operations as ops
from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.models.base import BaseModel, register_model
from aiconfigurator_core.sdk.models.blocks.moe import MoEBlockShape, build_moe_block_ops
from aiconfigurator_core.sdk.models.helpers import mtp_scale_factor, power_law_distribution
from aiconfigurator_core.sdk.utils import _load_model_config_from_model_path

logger = logging.getLogger(__name__)


@register_model("MOE")
class MOEModel(BaseModel):
    """
    Traditional MoE models uses this model impl: Mixtral, LLAMA4_MOE, MiniMax-M2, etc.
    Some rules to follow,
    Due to implementation, attn layer name needs to be context_attention or generation_attention,
    exact match is required. Same for logits_gemm.
    Supports MTP (Multi-Token Prediction) speculative decoding simulation.
    TODO: redesign shared moe part.
    """

    @classmethod
    def supports_cp(cls, backend_name: str) -> bool:
        # Dense GQA attention + MoE prefill CP: SGLang AllGather (zigzag attn via
        # ContextAttention cp_size, token-major seq_split, MoEDispatch attn_cp_size).
        return backend_name == "sglang"

    @classmethod
    def create(cls, model_info: dict, model_config, backend_name: str) -> BaseModel:
        moe_args = (model_info["topk"], model_info["num_experts"], model_info["moe_inter_size"])
        base_args = (
            model_info["model_path"],
            model_info["model_family"],
            model_info["architecture"],
            model_info["layers"],
            model_info["n"],
            model_info["n_kv"],
            model_info["d"],
            model_info["hidden_size"],
            model_info["inter_size"],
            model_info["vocab"],
            model_info["context"],
            model_config,
        )
        extra_params = model_info["extra_params"]
        return cls(*moe_args, *base_args, extra_params, backend_name=backend_name)

    def __init__(self, topk: int, num_experts: int, moe_inter_size: int, *args, backend_name: str = "") -> None:
        super().__init__(*args)

        self._backend_name = backend_name
        # Large EP: the enumerator picks a per-phase MoE comm backend; the MoE
        # block builder then emits the MoEAllToAll/EPMoE graph instead of the
        # fused dispatch/MoE/dispatch one. No flag selects it -- see
        # ``ModelConfig.moe_comm_backend``.
        self._is_large_ep = bool(self.config.moe_comm_backend)

        # MTP scale factor: throughput boost / compute overhead
        self._mtp_scale_factor = mtp_scale_factor(self._nextn, self._num_layers)

        # make sure the paralel width is same (cp is an independent attention
        # dimension that also contributes to the width the MoE must match)
        assert (
            self.config.tp_size * self.config.attention_dp_size * self.config.cp_size
            == self.config.moe_tp_size * self.config.moe_ep_size
        ), (
            f"tp_size ({self.config.tp_size}) * attention_dp_size "
            f"({self.config.attention_dp_size}) * cp_size ({self.config.cp_size}) should be equal to "
            f"moe_tp_size ({self.config.moe_tp_size}) * moe_ep_size ({self.config.moe_ep_size})"
        )

        assert num_experts >= self.config.moe_ep_size, f"ep size cannot be larger than num_experts {num_experts}"

        self._topk = topk
        self._num_experts = num_experts
        self._moe_inter_size = moe_inter_size

        # Validate quantized MoE block size alignment
        self._validate_fp8_block_quantized_moe_config()

        self._power_law_alpha = 1.2

        moe_quant_mode = self.config.moe_quant_mode

        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        num_kv_heads_per_gpu = self._num_kv_heads_per_gpu
        gemm_quant_mode = self.config.gemm_quant_mode
        kvcache_quant_mode = self.config.kvcache_quant_mode
        fmha_quant_mode = self.config.fmha_quant_mode
        workload_distribution = power_law_distribution(self.config.workload_distribution, self._power_law_alpha)
        # Large-EP prefill flattens to alpha 0.6 under EPLB (transcribed from the
        # deleted SGLangEPMOEModel, models/moe.py:424 at commit 8372e60). The
        # fused path and every decode phase keep the family alpha.
        context_workload_distribution = (
            power_law_distribution(self.config.workload_distribution, 0.6)
            if self._is_large_ep and self.config.enable_eplb
            else workload_distribution
        )
        # Context parallelism (sglang AllGather, prefill-only). Dense GQA attn
        # uses ContextAttention zigzag (cp_size); token-major ops seq_split=cp;
        # MoEDispatch attn_cp_size=cp (AG_hidden+RS). Generation not CP-modeled.
        cp = self.config.cp_size

        if self.architecture == "GptOssForCausalLM":
            attn_scale_factor = 2
            window_size = 128
            self.context_ops.append(
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    window_size=window_size,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
                    cp_size=cp,
                )
            )
            self.generation_ops.append(
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers * self._mtp_scale_factor / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    window_size=window_size,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
                )
            )
        else:
            attn_scale_factor = 1

        # Large EP replicates the embedding table instead of sharding it over TP
        # (transcribed from the deleted SGLangEPMOEModel, models/moe.py:490/588 at
        # commit 8372e60, which also emitted no embedding all-reduce and no P2P).
        embedding_vocab_size = self._vocab_size if self._is_large_ep else self._vocab_size // tp_size

        self.context_ops.extend(
            [
                ops.Embedding("context_embedding", 1, embedding_vocab_size, h, 0.3, seq_split=cp),
                ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
                ops.GEMM(
                    "context_qkv_gemm",
                    self._num_layers,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                    seq_split=cp,
                ),
                ops.ContextAttention(
                    "context_attention",
                    self._num_layers / attn_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    fmha_quant_mode,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
                    cp_size=cp,
                ),
                *self._cp_attn_comm_ops(),
                ops.GEMM(
                    "context_proj_gemm",
                    self._num_layers,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                    seq_split=cp,
                ),
                ops.ElementWise("context_add_norm_2", self._num_layers, 2 * h, 2 * h, 0.8, seq_split=cp),
            ]
        )

        # MoE block: router gemm + dispatch/compute/combine. One builder call per
        # phase; ``cfg.moe_comm_backend`` (set by the enumerator) picks the
        # large-EP MoEAllToAll/EPMoE emission over the fused dispatch/MoE pair.
        moe_shape = MoEBlockShape(
            hidden_size=h,
            moe_inter_size=self._moe_inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            # Neither the fused MOEModel span nor the deleted SGLangEPMOEModel
            # wired a separate shared-expert span for this family.
            num_shared_experts=0,
            # Descriptor-only here: the builder scales by the model-owned
            # scale_factor below (legacy uses the full layer count).
            num_moe_layers=self._num_layers,
        )
        self.context_ops.extend(
            build_moe_block_ops(
                "context",
                moe_shape,
                self.config,
                moe_quant_mode,
                context_workload_distribution,
                scale_factor=self._num_layers,
                backend_name=self._backend_name,
                inference_phase="context",
                model_family=self.model_family,
                attn_cp_size=cp,
                gpus_per_node=self.config.num_gpus_per_node,
            )
        )

        self.generation_ops.extend(
            [
                ops.Embedding("generation_embedding", 1 * self._mtp_scale_factor, embedding_vocab_size, h, 0.3),
                ops.ElementWise("generation_add_norm_1", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
                ops.GEMM(
                    "generation_qkv_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    self._num_heads * self._head_size // tp_size + self._head_size * num_kv_heads_per_gpu * 2,
                    h,
                    gemm_quant_mode,
                ),
                ops.GenerationAttention(
                    "generation_attention",
                    self._num_layers / attn_scale_factor * self._mtp_scale_factor,
                    self._num_heads // tp_size,
                    num_kv_heads_per_gpu,
                    kvcache_quant_mode,
                    head_size=self._head_size,
                    use_qk_norm=self._use_qk_norm,
                ),
                ops.GEMM(
                    "generation_proj_gemm",
                    self._num_layers * self._mtp_scale_factor,
                    h,
                    self._num_heads * self._head_size // tp_size,
                    gemm_quant_mode,
                    low_precision_input=True,
                ),
                ops.ElementWise("generation_add_norm_2", self._num_layers * self._mtp_scale_factor, 2 * h, 2 * h, 0.8),
            ]
        )

        # MoE block (see the context site above). Generation is not CP-modeled,
        # so the builder drops the seq_split/attn_cp kwargs for this phase.
        self.generation_ops.extend(
            build_moe_block_ops(
                "generation",
                moe_shape,
                self.config,
                moe_quant_mode,
                workload_distribution,
                scale_factor=self._num_layers * self._mtp_scale_factor,
                backend_name=self._backend_name,
                inference_phase="generation",
                model_family=self.model_family,
                attn_cp_size=cp,
                gpus_per_node=self.config.num_gpus_per_node,
            )
        )
        # logits gemm
        self.generation_ops.extend(
            [
                ops.GEMM(
                    "generation_logits_gemm",
                    1 * self._mtp_scale_factor,
                    self._vocab_size // tp_size,
                    h,
                    common.GEMMQuantMode.bfloat16,
                )
            ]
        )

        if self._is_large_ep:
            # The legacy large-EP graph ends at the logits gemm: with the
            # embedding replicated there is nothing to all-reduce, and it never
            # modeled pipeline P2P.
            return

        # All-reduce after embedding: needed when tp > 1
        # Embedding shards vocab across TP ranks and all-reduces
        self.context_ops.append(ops.CustomAllReduce("context_embedding_ar", 1, h, tp_size, seq_split=cp))
        self.generation_ops.append(
            ops.CustomAllReduce("generation_embedding_ar", 1 * self._mtp_scale_factor, h, tp_size)
        )

        # pp
        pp_scale_factor = pp_size - 1
        self.context_ops.append(ops.P2P("context_p2p", pp_scale_factor, h, pp_size, seq_split=cp))
        self.generation_ops.append(ops.P2P("generation_p2p", pp_scale_factor * self._mtp_scale_factor, h, pp_size))

    def _validate_fp8_block_quantized_moe_config(self) -> None:
        """
        Validate that quantized MoE configuration satisfies block size constraints.

        For fp8_block quantized MoE models, the constraint is:
        (moe_intermediate_size / moe_tp_size) % weight_block_size_n == 0

        This ensures proper alignment for quantized weight blocks.
        """
        # Only validate for fp8_block quantization
        if self.config.moe_quant_mode != common.MoEQuantMode.fp8_block:
            return

        # Load raw model config to get block size
        raw_config = _load_model_config_from_model_path(self.model_path)

        # Get weight_block_size from quantization_config (default to [128, 128])
        default_size = [128, 128]
        weight_block_size = raw_config.get("quantization_config", {}).get("weight_block_size", default_size)[0]

        # Check alignment
        moe_size_per_gpu = self._moe_inter_size // self.config.moe_tp_size
        if (moe_size_per_gpu % weight_block_size) != 0:
            raise ValueError(
                f"Invalid quantized MoE configuration: "
                f"(moe_intermediate_size={self._moe_inter_size} / moe_tp_size={self.config.moe_tp_size}) "
                f"% weight_block_size={weight_block_size} != 0. "
            )
