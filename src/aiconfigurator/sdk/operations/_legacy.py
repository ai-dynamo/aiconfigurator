# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy home for op classes that have not yet been migrated to their own
files. Each ISSUE-04..14 moves a family of classes out of this file into a
dedicated module. Final cleanup (ISSUE-15) deletes this file once empty."""

import logging

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)


class CustomAllReduce(Operation):
    """
    Custom AllReduce operation with power tracking.
    """

    def __init__(self, name: str, scale_factor: float, h: int, tp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._tp_size = tp_size
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query custom allreduce latency with power data."""
        if self._tp_size == 1:
            # No-op short-circuit: tp_size=1 has no allreduce. Tag as
            # ``empirical`` rather than letting the constructor default to
            # ``silicon`` so EMPIRICAL/SOL modes don't get a spurious
            # silicon leakage in the breakdown report.
            return PerformanceResult(0.0, 0.0, source="empirical")
        # count, not size in bytes
        size = kwargs.get("x") * self._h

        result = database.query_custom_allreduce(common.CommQuantMode.half, self._tp_size, size)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class P2P(Operation):
    """
    P2P operation with power tracking.
    """

    def __init__(self, name: str, scale_factor: float, h: int, pp_size: int) -> None:
        super().__init__(name, scale_factor)
        self._h = h
        self._pp_size = pp_size
        self._bytes_per_element = 2
        # self._empirical_scaling_factor = 1.1
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query P2P latency with power data."""
        if self._pp_size == 1:
            # No-op short-circuit: pp_size=1 has no P2P transfer. See note on
            # CustomAllReduce.query for source-tag rationale.
            return PerformanceResult(0.0, 0.0, source="empirical")

        size = kwargs.get("x") * self._h
        p2p_bytes = size * 2

        result = database.query_p2p(p2p_bytes)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class NCCL(Operation):
    """
    NCCL operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        nccl_op: str,
        num_elements_per_token: int,
        num_gpus: int,
        comm_quant_mode: common.CommQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._nccl_op = nccl_op
        self._num_elements_per_token = num_elements_per_token
        self._num_gpus = num_gpus
        self._comm_quant_mode = comm_quant_mode
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query NCCL latency with power data."""
        message_size = kwargs.get("x") * self._num_elements_per_token

        result = database.query_nccl(self._comm_quant_mode, self._num_gpus, self._nccl_op, message_size)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextMLA(Operation):
    """
    Context MLA operation. now only contains MHA part.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128)
        # up q, up k, up v  bfloat16 # 104MB / tpsize per layer
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context MLA latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_context_mla(
            b=batch_size,
            s=isl,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationMLA(Operation):
    """
    Generation MLA operation. now only contains MQA part.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        # 2*(1536*24576/tp_size + 128/tp_size*512*128+128/tp_size*512*128)
        # up q, up k, v up bfloat16
        self._weights = 0.0
        self._kv_cache_dtype = kv_cache_dtype

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation MLA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_mla(batch_size, s, self._num_heads, self._kv_cache_dtype)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLABmm(Operation):
    """
    MLABmm operation. consider to be contained by mla op. for now, keep it as a separate op to
    show the cost of bmm
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MLA BMM latency with power data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")

        result = database.query_mla_bmm(batch_size, self._num_heads, self._quant_mode, self._if_pre)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPGenerationMLA(Operation):
    """
    WideEP Generation MLA operation.
    This handles the MLA operations in generation/decoding mode.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP generation MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_wideep_generation_mla(
            batch_size,
            s,
            self._tp_size,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            self._attn_backend,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPContextMLA(Operation):
    """
    WideEP Context MLA operation.
    This handles the MLA operations in context/prefill mode.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query WideEP context MLA latency with power data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_wideep_context_mla(
            b=batch_size,
            s=isl,
            prefix=prefix,
            tp_size=self._tp_size,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            attention_backend=self._attn_backend,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Mamba2Kernel(Operation):
    """
    Single Mamba2 kernel op (Conv1D or SSM) using collected mamba2_perf data.

    One of four kernels: causal_conv1d_fn, mamba_chunk_scan_combined (context),
    causal_conv1d_update, selective_state_update (generation).
    Uses full (unsharded) dimensions for lookup; collector data is per-layer.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        kernel_source: str,
        phase: str,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
    ) -> None:
        super().__init__(name, scale_factor)
        self._kernel_source = kernel_source
        self._phase = phase
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        seq_len = s if self._phase == "context" else None
        result = database.query_mamba2(
            phase=self._phase,
            kernel_source=self._kernel_source,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self._hidden_size,
            d_state=self._d_state,
            d_conv=self._d_conv,
            nheads=self._nheads,
            head_dim=self._head_dim,
            n_groups=self._n_groups,
            chunk_size=self._chunk_size,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GDNKernel(Operation):
    """
    Single Gated DeltaNet (GDN) kernel op for Qwen3.5 linear_attention layers.

    Covers four kernel sources:
      Context phase:
        - "causal_conv1d_fn": Causal 1D convolution over full sequence
        - "chunk_gated_delta_rule": GDN chunked scan (core recurrence)
      Generation phase:
        - "causal_conv1d_update": Single-step causal conv state update
        - "fused_sigmoid_gating_delta_rule_update": Single-step GDN recurrence

    Uses full (unsharded) dimensions for database lookup; collector data is per-layer.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        kernel_source: str,
        phase: str,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
    ) -> None:
        super().__init__(name, scale_factor)
        self._kernel_source = kernel_source
        self._phase = phase
        self._d_model = d_model
        self._num_k_heads = num_k_heads
        self._head_k_dim = head_k_dim
        self._num_v_heads = num_v_heads
        self._head_v_dim = head_v_dim
        self._d_conv = d_conv
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        seq_len = s if self._phase == "context" else None
        result = database.query_gdn(
            phase=self._phase,
            kernel_source=self._kernel_source,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self._d_model,
            num_k_heads=self._num_k_heads,
            head_k_dim=self._head_k_dim,
            num_v_heads=self._num_v_heads,
            head_v_dim=self._head_v_dim,
            d_conv=self._d_conv,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Mamba2(Operation):
    """
    Mamba2 operation for NemotronH hybrid models.

    Models the Mamba2Mixer layer which consists of:
    - in_proj: Linear projection (hidden_size -> expanded_size)
    - conv1d: Causal 1D convolution
    - SSM: Selective State Space Model (scan operation)
    - norm: RMSNorm with gating
    - out_proj: Linear projection back to hidden_size

    This is a SOL-based approximation that models:
    - Two GEMMs for in_proj and out_proj
    - Memory operations for conv1d and SSM scan

    The internal state dimension is calculated as:
    expanded_size = 2 * (nheads * head_dim + 2 * n_groups * d_state)
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
        tp_size: int,
        quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._tp_size = tp_size
        self._quant_mode = quant_mode

        # Calculate dimensions matching TensorRT-LLM mamba2_mixer.py lines 76-78:
        # d_inner = head_dim * nheads
        # d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        # conv_dim = d_inner + 2 * n_groups * d_state
        self._d_inner = nheads * head_dim
        self._conv_dim = self._d_inner + 2 * n_groups * d_state
        self._in_proj_out_size = 2 * self._d_inner + 2 * n_groups * d_state + nheads

        # Calculate weights (in_proj + conv1d + out_proj + A + D + dt_bias + norm)
        # in_proj: hidden_size * in_proj_out_size (Linear d_model -> d_in_proj)
        # conv1d: d_conv * conv_dim (Linear d_conv -> conv_dim, stored as Linear for TP)
        # out_proj: d_inner * hidden_size (Linear d_inner -> d_model)
        # A, D, dt_bias: nheads each (small, ignored for weight calculation)
        # norm: d_inner (small, ignored)
        self._weights = (
            (
                hidden_size * self._in_proj_out_size  # in_proj
                + d_conv * self._conv_dim  # conv1d
                + self._d_inner * hidden_size  # out_proj
            )
            * quant_mode.value.memory
            // tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query Mamba2 latency using SOL-based approximation.

        Models the operation as:
        1. in_proj GEMM: (x, hidden_size) @ (hidden_size, in_proj_out_size)
        2. conv1d: Memory-bound operation
        3. SSM scan: Memory-bound recurrent operation
        4. out_proj GEMM: (x, d_inner) @ (d_inner, hidden_size)
        """
        x = kwargs.get("x")  # num tokens

        # Apply TP sharding (matching TensorRT-LLM mamba2_mixer.py lines 81-84)
        # tp_nheads = nheads // tp_size
        # tp_d_inner = d_inner // tp_size
        # tp_ngroups = n_groups // tp_size
        # tp_conv_dim = conv_dim // tp_size
        nheads_per_gpu = self._nheads // self._tp_size
        d_inner_per_gpu = nheads_per_gpu * self._head_dim
        n_groups_per_gpu = self._n_groups // self._tp_size
        conv_dim_per_gpu = d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state
        in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state + nheads_per_gpu

        total_latency = 0.0
        total_energy = 0.0

        # 1. in_proj GEMM: hidden_size -> in_proj_out_size
        in_proj_result = database.query_gemm(x, in_proj_out_per_gpu, self._hidden_size, self._quant_mode)
        total_latency += float(in_proj_result)
        total_energy += in_proj_result.energy

        # 2. conv1d: Memory-bound operation on conv_dim (not just d_inner)
        # conv1d operates on xbc which has dimension conv_dim
        # Read: x * conv_dim * d_conv (for conv states) + x * conv_dim (input)
        # Write: x * conv_dim (output)
        conv_read_bytes = x * conv_dim_per_gpu * (self._d_conv + 1) * 2  # bfloat16
        conv_write_bytes = x * conv_dim_per_gpu * 2
        conv_result = database.query_mem_op(conv_read_bytes + conv_write_bytes)
        total_latency += float(conv_result)
        total_energy += conv_result.energy

        # 3. SSM scan: Memory-bound recurrent operation
        # For prefill (context), uses chunked scan
        # For decode (generation), uses selective_state_update
        # Approximate as memory operation:
        # Read: x * (d_inner + n_groups * d_state * 2 + nheads) for x, B, C, dt
        # Write: x * d_inner for output
        ssm_read_bytes = (
            x
            * (
                d_inner_per_gpu
                + n_groups_per_gpu * self._d_state * 2  # B and C
                + nheads_per_gpu  # dt
            )
            * 2
        )
        ssm_write_bytes = x * d_inner_per_gpu * 2
        ssm_result = database.query_mem_op(ssm_read_bytes + ssm_write_bytes)
        total_latency += float(ssm_result)
        total_energy += ssm_result.energy

        # 4. norm: RMSNormGated on d_inner (TRT-LLM mamba2_mixer.py line 315)
        # Read SSM output, apply norm with gating, write normalized output
        norm_read_bytes = x * d_inner_per_gpu * 2  # bfloat16
        norm_write_bytes = x * d_inner_per_gpu * 2  # bfloat16
        norm_result = database.query_mem_op(norm_read_bytes + norm_write_bytes)
        total_latency += float(norm_result)
        total_energy += norm_result.energy

        # 5. out_proj GEMM: d_inner -> hidden_size
        out_proj_result = database.query_gemm(x, self._hidden_size, d_inner_per_gpu, self._quant_mode)
        total_latency += float(out_proj_result)
        total_energy += out_proj_result.energy

        # Merge sources from every sub-result so the composite reflects mixed
        # silicon/empirical provenance instead of defaulting to silicon.
        sub_sources = [
            getattr(r, "source", "silicon")
            for r in (in_proj_result, conv_result, ssm_result, norm_result, out_proj_result)
        ]
        merged_source = sub_sources[0] if all(s == sub_sources[0] for s in sub_sources) else "mixed"

        return PerformanceResult(
            latency=total_latency * self._scale_factor,
            energy=total_energy * self._scale_factor,
            source=merged_source,
        )

    def get_weights(self, **kwargs):  # Mamba2 weights
        return self._weights * self._scale_factor


# ═══════════════════════════════════════════════════════════════════════
# DSA (DeepSeek Sparse Attention) Operations
# ═══════════════════════════════════════════════════════════════════════


class ContextDSAModule(Operation):
    """
    Context phase DSA (DeepSeek Sparse Attention) module-level operation.

    Models the full DSA attention block including:
    - kv_a_proj_with_mqa GEMM (includes indexer K projection)
    - LayerNorm + q_b_proj GEMM
    - Indexer: wq_b GEMM, weights_proj GEMM, FP8 MQA logits, TopK selection
    - Sparse MLA attention (attends to top-k tokens instead of full sequence)
    - BMM pre/post (weight absorption + V projection)
    - o_proj GEMM
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context DSA latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)

        result = database.query_context_dsa_module(
            b=batch_size,
            s=isl,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._architecture,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationDSAModule(Operation):
    """
    Generation phase DSA (DeepSeek Sparse Attention) module-level operation.

    Models the full DSA attention block during decode:
    - Same components as ContextDSAModule
    - Uses paged MQA logits for indexer
    - Sparse MLA with KV cache lookup
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation DSA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_dsa_module(
            b=batch_size,
            s=s,
            num_heads=self._num_heads,
            kv_cache_dtype=self._kv_cache_dtype,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._architecture,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class DeepSeekV4MHCModule(Operation):
    """DeepSeek-V4 manifold-constrained hyper-connection pre/post module."""

    def __init__(
        self,
        name: str,
        scale_factor: float,
        op: str,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        if op not in {"pre", "post", "both"}:
            raise ValueError(f"Unsupported DeepSeek-V4 mHC op: {op}")
        self._op = op
        self._hidden_size = hidden_size
        self._hc_mult = hc_mult
        self._sinkhorn_iters = sinkhorn_iters
        self._quant_mode = quant_mode
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        # Two parameter sets per decoder block: attention mHC and FFN mHC.
        self._weights = 2 * (mix_hc * hc_dim + mix_hc + 3) * quant_mode.value.memory

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        result = database.query_mhc_module(
            num_tokens=kwargs.get("x"),
            hidden_size=self._hidden_size,
            hc_mult=self._hc_mult,
            sinkhorn_iters=self._sinkhorn_iters,
            op=self._op,
            quant_mode=self._quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class _BaseDeepSeekV4AttentionModule(Operation):
    """Common DeepSeek-V4 compressed attention module metadata."""

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._native_heads = native_heads
        self._tp_size = tp_size
        self._hidden_size = hidden_size
        self._q_lora_rank = q_lora_rank
        self._o_lora_rank = o_lora_rank
        self._head_dim = head_dim
        self._rope_head_dim = rope_head_dim
        self._index_n_heads = index_n_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._window_size = window_size
        self._compress_ratio = compress_ratio
        self._o_groups = o_groups
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._weights = self._estimate_weights()

    def _estimate_weights(self) -> float:
        gemm_weight_elems = (
            self._hidden_size * self._q_lora_rank
            + self._q_lora_rank * self._num_heads * self._head_dim
            + self._hidden_size * self._head_dim
            + self._o_groups * self._o_lora_rank * self._hidden_size
        )
        bfloat16_weight_elems = self._num_heads * self._head_dim * self._o_lora_rank
        float32_weight_elems = self._num_heads
        if self._compress_ratio:
            compressor_mult = 2 if self._compress_ratio == 4 else 1
            gemm_weight_elems += 2 * self._hidden_size * compressor_mult * self._head_dim
            float32_weight_elems += self._compress_ratio * compressor_mult * self._head_dim
        if self._compress_ratio == 4:
            gemm_weight_elems += self._q_lora_rank * self._index_n_heads * self._index_head_dim
            gemm_weight_elems += 2 * self._hidden_size * 2 * self._index_head_dim
            bfloat16_weight_elems += self._hidden_size * self._index_n_heads
            float32_weight_elems += self._compress_ratio * 2 * self._index_head_dim
        return (
            gemm_weight_elems * self._gemm_quant_mode.value.memory
            + bfloat16_weight_elems * common.GEMMQuantMode.bfloat16.value.memory
            + float32_weight_elems * 4
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Context-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module."""

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        result = database.query_context_deepseek_v4_attention_module(
            b=kwargs.get("batch_size"),
            s=kwargs.get("s"),
            prefix=kwargs.get("prefix", 0),
            num_heads=self._num_heads,
            native_heads=self._native_heads,
            tp_size=self._tp_size,
            hidden_size=self._hidden_size,
            q_lora_rank=self._q_lora_rank,
            o_lora_rank=self._o_lora_rank,
            head_dim=self._head_dim,
            rope_head_dim=self._rope_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            window_size=self._window_size,
            compress_ratio=self._compress_ratio,
            o_groups=self._o_groups,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )


class GenerationDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Decode-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module."""

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        result = database.query_generation_deepseek_v4_attention_module(
            b=kwargs.get("batch_size"),
            s=kwargs.get("s"),
            num_heads=self._num_heads,
            native_heads=self._native_heads,
            tp_size=self._tp_size,
            hidden_size=self._hidden_size,
            q_lora_rank=self._q_lora_rank,
            o_lora_rank=self._o_lora_rank,
            head_dim=self._head_dim,
            rope_head_dim=self._rope_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            window_size=self._window_size,
            compress_ratio=self._compress_ratio,
            o_groups=self._o_groups,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )


class MLAModule(Operation):
    """
    Module-level MLA operation for both context and generation phases.

    Models the complete MLA attention block as a single profiled operation.
    For context: replaces q_b_proj + kv_b_proj + ContextMLA + proj.
    For generation: replaces MLABmm(pre) + GenerationMLA + MLABmm(post).
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        is_context: bool,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._is_context = is_context
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MLA module latency with energy data."""
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        if self._is_context:
            prefix = kwargs.get("prefix", 0)
            result = database.query_context_mla_module(
                b=batch_size,
                s=s,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
            )
        else:
            beam_width = kwargs.get("beam_width")
            if beam_width != 1:
                raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
            result = database.query_generation_mla_module(
                b=batch_size,
                s=s,
                num_heads=self._num_heads,
                kv_cache_dtype=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
            )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class FallbackOp(Operation):
    """
    Try a primary operation first; if it raises PerfDataNotAvailableError,
    fall back to a sequence of fallback operations (summed).

    This supports transitional periods where some systems have module-level
    profiling data (single op) while others still have granular per-kernel data
    (multiple ops). The fallback is symmetric: either group can be primary.

    In HYBRID mode, the primary is queried in SILICON mode so that HYBRID does
    not silently swallow a miss with an empirical estimate — the fallback ops
    (which have real data) should be preferred over an empirical guess. In
    explicit EMPIRICAL/SOL modes, the primary respects the requested mode.

    Once the primary fails on the first call, it is skipped on all subsequent
    calls to avoid redundant work.

    Latency = primary.query()  OR  sum(fallback[i].query())
    Energy  = same source as whichever succeeds
    Weights = sum of whichever group is used (primary or fallback)
    """

    def __init__(self, name: str, primary: Operation, fallback: list[Operation]) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            primary: Single operation to try first.
            fallback: List of operations to sum if primary fails.
        """
        super().__init__(name, 1.0)  # scale_factor handled by inner ops
        self._primary = primary
        self._fallback = fallback
        self._primary_unavailable = False

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        import logging as _logging

        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        if not self._primary_unavailable:
            prev_mode = database._default_database_mode
            force_primary_silicon = prev_mode == common.DatabaseMode.HYBRID
            if force_primary_silicon:
                # Force SILICON mode on the primary so HYBRID does not silently
                # return an empirical estimate when module data is missing.
                database._default_database_mode = common.DatabaseMode.SILICON

            # Suppress ERROR-level logs from perf_database during the primary
            # attempt, since a failure here is expected and handled by fallback.
            perf_db_logger = _logging.getLogger("aiconfigurator.sdk.perf_database")
            prev_log_level = perf_db_logger.level
            perf_db_logger.setLevel(_logging.CRITICAL)
            try:
                return self._primary.query(database, **kwargs)
            except (PerfDataNotAvailableError, KeyError, AssertionError) as e:
                if isinstance(e, PerfDataNotAvailableError):
                    self._primary_unavailable = True
                logger.debug(
                    "FallbackOp '%s': primary op '%s' failed (%s: %s), using fallback ops",
                    self._name,
                    self._primary._name,
                    type(e).__name__,
                    e,
                )
            finally:
                if force_primary_silicon:
                    database._default_database_mode = prev_mode
                perf_db_logger.setLevel(prev_log_level)

        total = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._fallback:
            total += op.query(database, **kwargs)
        return total

    def get_weights(self, **kwargs):
        # Use primary weights if available, otherwise sum fallback weights.
        # In practice both should be equivalent since they model the same block.
        if not self._primary_unavailable:
            primary_w = self._primary.get_weights(**kwargs)
            if primary_w > 0:
                return primary_w
        return sum(op.get_weights(**kwargs) for op in self._fallback)


class OverlapOp(Operation):
    """
    Two groups of operations that execute in parallel (overlap).

    This models the TRT-LLM `maybe_execute_in_parallel` behavior where two
    operation groups run concurrently on different CUDA streams during
    generation phase (CUDA Graph enabled).

    Latency = max(sum(group_a latencies), sum(group_b latencies))
    Energy  = sum(all ops in both groups)  # both groups consume power
    Weights = sum(all ops in both groups)
    """

    def __init__(self, name: str, group_a: list, group_b: list) -> None:
        """
        Args:
            name: Operation name for latency breakdown reporting.
            group_a: List of Operation objects for the first parallel group
                     (e.g., routed expert path on main stream).
            group_b: List of Operation objects for the second parallel group
                     (e.g., shared expert path on aux stream).
        """
        super().__init__(name, 1.0)  # scale_factor handled by inner ops
        self._group_a = group_a
        self._group_b = group_b

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query overlap operation latency.

        Returns:
            PerformanceResult with latency = max(group_a, group_b)
            and energy = sum of all ops.
        """
        total_a = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_a:
            total_a += op.query(database, **kwargs)

        total_b = PerformanceResult(0.0, energy=0.0, source="empirical")
        for op in self._group_b:
            total_b += op.query(database, **kwargs)

        merged = total_a + total_b
        return PerformanceResult(
            latency=max(float(total_a), float(total_b)),
            energy=total_a.energy + total_b.energy,
            source=merged.source,
        )

    def get_weights(self, **kwargs):
        weights = 0.0
        for op in self._group_a + self._group_b:
            weights += op.get_weights(**kwargs)
        return weights
