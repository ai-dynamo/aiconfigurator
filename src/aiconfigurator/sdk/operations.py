# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)


class Operation:
    """
    Base operation class.

    Note: query() now returns PerformanceResult (float-like) instead of plain float.
    This maintains backward compatibility while adding power data.
    """

    def __init__(self, name: str, scale_factor: float) -> None:
        self._name = name
        self._scale_factor = scale_factor

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query operation latency with power data.

        Returns:
            PerformanceResult: PerformanceResult (behaves like float) with latency in milliseconds
                   (scaled by scale_factor). Power data available via .power attribute.
        """
        raise NotImplementedError

    def get_weights(self, **kwargs):
        raise NotImplementedError


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
            return PerformanceResult(0.0, 0.0)
        # count, not size in bytes
        size = kwargs.get("x") * self._h

        result = database.query_custom_allreduce(common.CommQuantMode.half, self._tp_size, size)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
            return PerformanceResult(0.0, 0.0)

        size = kwargs.get("x") * self._h
        p2p_bytes = size * 2

        result = database.query_p2p(p2p_bytes)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GEMM(Operation):
    """
    GEMM operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        k: int,
        quant_mode: common.GEMMQuantMode,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._k = k
        self._quant_mode = quant_mode
        self._weights = self._n * self._k * quant_mode.value.memory
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query GEMM latency with energy data.

        Returns:
            PerformanceResult: Behaves like float (scaled latency in ms).
                              Energy data accessible via .energy attribute.
                              Power can be derived as energy/latency.
        """
        x = kwargs.get("x")
        x //= self._scale_num_tokens
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        # Query with energy
        result = database.query_gemm(x, self._n, self._k, quant_mode)

        # Return PerformanceResult: scale BOTH latency and energy
        # (energy scales with latency for serial execution)
        return PerformanceResult(
            latency=float(result) * self._scale_factor,  # Scaled latency
            energy=result.energy * self._scale_factor,  # Scaled energy (proportional to latency)
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MoE(Operation):
    """
    MoE operation with power tracking.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        is_context: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_context = is_context
        self._moe_backend = kwargs.get("moe_backend")
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * 3
            // self._moe_ep_size
            // self._moe_tp_size
        )  # 3 for ffn1,gate,ffn2; 2 for float16

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query MoE latency with energy data."""
        # attention dp size will scale up the total input tokens.
        x = kwargs.get("x") * self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        result = database.query_moe(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
            is_context=self._is_context,
            moe_backend=self._moe_backend,
        )

        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# a comm op to deduce the communication cost of MoE
class MoEDispatch(Operation):
    """
    MoE dispatch operation. For fine grained moe dispatch
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        attention_dp_size: int,
        pre_dispatch: bool,
        enable_fp4_all2all: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._weights = 0.0
        self._enable_fp4_all2all = enable_fp4_all2all
        self._pre_dispatch = pre_dispatch
        self.num_gpus = self._moe_ep_size * self._moe_tp_size
        self._attention_tp_size = moe_tp_size * moe_ep_size // self._attention_dp_size
        self._sms = kwargs.get("sms", 12)
        self._moe_backend = kwargs.get("moe_backend")
        self._is_context = kwargs.get("is_context", True)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size
        _sm_version = database.system_spec["gpu"]["sm_version"]
        _num_gpus_per_node = database.system_spec["node"]["num_gpus_per_node"]
        _node_num = self.num_gpus / _num_gpus_per_node

        if database.backend == common.BackendName.trtllm.value:
            assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                "trtllm does not support TP>1 and DP>1 for attn simultaneously"
            )
            if _sm_version == 100:
                logger.debug("MoEDispatch: In trtllm SM100 execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        if _num_gpus_per_node == 72 and self.num_gpus > 4:  # to do: nvl72, node per gpu
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                            )
                        else:
                            comm_latency = database.query_custom_allreduce(
                                common.CommQuantMode.half, self.num_gpus, volume
                            )
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # Calculate all2all communication volume for nvfp4 all2all operation
                            # Volume calculation considers the average case between best and worst
                            # scenarios:
                            # - Best case: volume * 1/4 (all selected experts are in one GPU for
                            #   all tokens)
                            # - Worst case: volume * min(topk, attention_dp_size)/4 (every selected
                            #   expert is in different GPU)
                            # - Final volume: average of best and worst cases, divided by 4 for
                            #   nvfp4 quantization
                            all2all_volume = (
                                volume * (1 + min(self._topk, self._attention_dp_size)) / 2 / 4
                            )  # mean of best and worst
                            # to do: nvfp4 custom all2all
                            all2all_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "alltoall", all2all_volume
                            )
                            all2all_sf_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "alltoall",
                                all2all_volume / 8,
                            )  # volume_scale_factor = 1/8 volume
                            comm_latency = all2all_latency + all2all_sf_latency + 1e-2  # msg size static latency 10us
                        else:
                            all_gather_volume = volume * self._attention_dp_size / 4
                            all_gather_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "all_gather",
                                all_gather_volume,
                            )  # nvfp4 allgather
                            all_gather_sf_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "all_gather",
                                all_gather_volume / 8,
                            )  # volume_scale_factor = 1/8 volume
                            comm_latency = all_gather_latency + all_gather_sf_latency
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        if _num_gpus_per_node == 72 and self.num_gpus > 4:  # to do: nvl72, node per gpu
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                            )
                        else:
                            comm_latency = database.query_custom_allreduce(
                                common.CommQuantMode.half, self.num_gpus, volume
                            )
                    elif self._attention_dp_size > 1:
                        if self._enable_fp4_all2all:
                            # to do: nvfp4 all2all
                            all2all_volume = (
                                volume * (1 + min(self._topk, self._attention_dp_size)) / 2 / 4
                            )  # nvfp4 all2all
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half, self.num_gpus, "alltoall", all2all_volume
                            )
                        else:
                            comm_latency = database.query_nccl(
                                common.CommQuantMode.half,
                                self.num_gpus,
                                "reduce_scatter",
                                volume * self._attention_dp_size,
                            )
                    else:
                        comm_latency = 0
            else:  # sm < 100 or > 100 (for now)
                logger.debug("MoEDispatch: In trtllm SM<100 or >100 execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        elif database.backend == common.BackendName.vllm.value:
            assert self._moe_tp_size == 1 or self._moe_ep_size == 1, (
                "vllm does not support MoE TP and MoE EP at the same time"
            )

            comm_latency = 0

            # Add allreduce latency when TP > 1
            if self._attention_tp_size > 1:
                comm_latency += database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)

            if self._attention_dp_size > 1:
                comm_latency += database.query_nccl(
                    common.CommQuantMode.half,
                    self.num_gpus,
                    "all_gather" if self._pre_dispatch else "reduce_scatter",
                    volume * self._attention_dp_size,
                )
        elif database.backend == common.BackendName.sglang.value:
            assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                "We don't enable the path for SGLang to support TP>1 and DP>1 for attn simultaneously"
            )
            if self._moe_backend == "deepep_moe":
                logger.debug("MoEDispatch: In SGLang DeepEP execution path")
                if self._is_context:
                    comm_latency = database.query_wideep_deepep_normal(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                        sms=self._sms,
                    )
                else:
                    comm_latency = database.query_wideep_deepep_ll(
                        node_num=_node_num,
                        num_tokens=num_tokens,
                        num_experts=self._num_experts,
                        topk=self._topk,
                        hidden_size=self._hidden_size,
                    )
            else:
                logger.debug("MoEDispatch: In SGLang non-DeepEP execution path")
                if self._pre_dispatch:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
                else:
                    if self._attention_tp_size > 1:  # tp>1, use allreduce
                        # to do: custom allreduce
                        comm_latency = database.query_custom_allreduce(common.CommQuantMode.half, self.num_gpus, volume)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    else:
                        comm_latency = 0
        else:  # other backends
            raise NotImplementedError(f"MoEDispatch: Not implemented for backend {database.backend}")

        # MoEDispatch calculates latency rather than querying, so energy=0
        return PerformanceResult(comm_latency * self._scale_factor, energy=0.0)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor

    def query_ideal(self, database: PerfDatabase, **kwargs):
        """
        Ideal communication cost for MoE dispatch. For reference only.
        """
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size

        if self._pre_dispatch:
            reduce_scatter1_v = volume / self.num_gpus
            reduce_scatter1_num_gpus = self._attention_tp_size

            all2all1_v = volume * self._topk / self.num_gpus
            all2all1_num_gpus = self.num_gpus

            allgather1_v = volume / self._moe_tp_size
            allgather1_num_gpus = self._moe_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter1_num_gpus,
                    "reduce_scatter",
                    reduce_scatter1_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all1_num_gpus, "alltoall", all2all1_v)
                + database.query_nccl(common.CommQuantMode.half, allgather1_num_gpus, "all_gather", allgather1_v)
            )
        else:
            reduce_scatter2_v = volume
            reduce_scatter2_num_gpus = self._moe_tp_size

            all2all2_v = volume * self._topk / self.num_gpus
            all2all2_num_gpus = self.num_gpus

            allgather2_v = volume / self.num_gpus
            allgather2_num_gpus = self._attention_tp_size

            comm_latency = (
                database.query_nccl(
                    common.CommQuantMode.half,
                    reduce_scatter2_num_gpus,
                    "reduce_scatter",
                    reduce_scatter2_v,
                )
                + database.query_nccl(common.CommQuantMode.half, all2all2_num_gpus, "alltoall", all2all2_v)
                + database.query_nccl(common.CommQuantMode.half, allgather2_num_gpus, "all_gather", allgather2_v)
            )

        return comm_latency * self._scale_factor


class ContextAttention(Operation):
    """
    Context attention operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context attention latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix")

        result = database.query_context_attention(
            batch_size,
            isl,
            prefix,
            self._n,
            self._n_kv,
            self._kvcache_quant_mode,
            self._fmha_quant_mode,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationAttention(Operation):
    """
    Generation attention operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
    ) -> None:
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation attention latency with energy data."""
        beam_width = kwargs.get("beam_width")
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_attention(
            batch_size,
            s,
            self._n,
            self._n_kv,
            self._kv_cache_dtype,
            window_size=self._window_size,
            head_size=self._head_size,
        )
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        # up q, up k, up v  float16 # 104MB / tpsize per layer
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
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        # up q, up k, v up float16
        self._weights = 0.0
        self._kv_cache_dtype = kv_cache_dtype

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation MLA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_mla(batch_size, s, self._num_heads, self._kv_cache_dtype)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        assert beam_width == 1, "only support beam_width=1"
        batch_size = kwargs.get("batch_size")

        result = database.query_mla_bmm(batch_size, self._num_heads, self._quant_mode, self._if_pre)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Embedding(Operation):
    """
    Embedding operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        row_size: int,
        column_size: int,
        empirical_bw_scaling_factor: float = 0.3,
    ) -> None:
        super().__init__(name, scale_factor)
        self._row_size = row_size
        self._column_size = column_size
        self._weights = row_size * column_size * 2
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query embedding latency with power data."""
        x = kwargs.get("x")
        d2d_bytes = x * self._column_size * 2

        result = database.query_mem_op(d2d_bytes)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ElementWise(Operation):
    """
    Element-wise operation.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        dim_in: int,
        dim_out: int,
        empirical_bw_scaling_factor: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._weights = 0.0
        self._empirical_bw_scaling_factor = empirical_bw_scaling_factor
        self._constant_latency = 5e-6  # 5us
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)

    # sol only
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query element-wise operation latency with power data."""
        x = kwargs.get("x")  # num tokens
        x //= self._scale_num_tokens
        read_bytes = x * self._dim_in * 2  # fp16 for act
        write_bytes = x * self._dim_out * 2

        result = database.query_mem_op(read_bytes + write_bytes)
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

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
        return PerformanceResult(float(result) * self._scale_factor, energy=result.energy * self._scale_factor)

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
