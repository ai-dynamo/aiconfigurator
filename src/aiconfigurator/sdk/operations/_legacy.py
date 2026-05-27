# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy home for op classes that have not yet been migrated to their own
files. Each ISSUE-04..14 moves a family of classes out of this file into a
dedicated module. Final cleanup (ISSUE-15) deletes this file once empty."""

import logging
from typing import Optional

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)


class TrtLLMWideEPMoE(Operation):
    """
    TensorRT-LLM WideEP MoE operation with configurable EPLB modes.

    This class is specifically designed for TensorRT-LLM backend's WideEP MoE computation.
    It handles the pure computation aspect of MoE, excluding All2All communication which
    is handled by TrtLLMWideEPMoEDispatch.

    Supports three EPLB modes:
    - EPLB off: workload_distribution without "_eplb" suffix, num_slots = num_experts
    - EPLB on: workload_distribution with "_eplb" suffix, num_slots = num_experts
    - EPLB redundant: workload_distribution with "_eplb" suffix, num_slots > num_experts

    Args:
        name: Operation name
        scale_factor: Scaling factor for the operation
        hidden_size: Hidden dimension size
        inter_size: Intermediate dimension size
        topk: Number of top experts to select
        num_experts: Total number of experts
        num_slots: Number of expert slots (= num_experts for EPLB off/on, > num_experts for redundant)
        moe_tp_size: MoE tensor parallelism size
        moe_ep_size: MoE expert parallelism size
        quant_mode: Quantization mode for MoE computation
        workload_distribution: Workload distribution pattern (e.g., "power_law_1.01" or "power_law_1.01_eplb")
        attention_dp_size: Attention data parallelism size (scales input tokens)
        is_gated: Whether MoE uses gated activation (default: True)
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
        num_slots: Optional[int] = None,  # EPLB slots, defaults to num_experts
        is_gated: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._quant_mode = quant_mode
        self._topk = topk
        self._num_experts = num_experts
        self._num_slots = num_slots if num_slots is not None else num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._workload_distribution = workload_distribution
        self._is_gated = is_gated

        # Calculate weights: 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query TrtLLM WideEP MoE compute latency with energy data.

        Supports three EPLB modes based on workload_distribution and num_slots:
        - EPLB off: distribution without "_eplb" suffix, num_slots = num_experts
        - EPLB on: distribution with "_eplb" suffix, num_slots = num_experts
        - EPLB redundant: distribution with "_eplb" suffix, num_slots > num_experts

        Args:
            database: Performance database instance
            **kwargs: Additional arguments including:
                - x: Number of input tokens (will be scaled by attention_dp_size)
                - quant_mode: Optional override for quantization mode

        Returns:
            PerformanceResult with latency and energy data
        """
        # Scale input tokens by attention_dp_size
        x = kwargs.get("x") * self._attention_dp_size
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        logger.debug(f"TrtLLMWideEPMoE: Querying compute with num_slots={self._num_slots}")

        # Query WideEP MoE compute performance
        result = database.query_wideep_moe_compute(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            num_slots=self._num_slots,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        """Get the weight memory size for this MoE layer."""
        return self._weights * self._scale_factor


class TrtLLMWideEPMoEDispatch(Operation):
    """
    TensorRT-LLM WideEP MoE dispatch operation using NVLink Two-Sided All2All.

    This class handles WideEP-specific All2All communication for expert parallelism
    in TensorRT-LLM, including prepare, dispatch, and combine phases.

    Communication phases:
    - Pre-dispatch: prepare + dispatch operations
    - Post-dispatch: combine or combine_low_precision operation

    Args:
        name: Operation name
        scale_factor: Scaling factor for the operation
        hidden_size: Hidden dimension size
        topk: Number of top experts to select
        num_experts: Total number of experts
        moe_tp_size: MoE tensor parallelism size
        moe_ep_size: MoE expert parallelism size
        attention_dp_size: Attention data parallelism size
        pre_dispatch: If True, performs prepare+dispatch; if False, performs combine
        quant_mode: Quantization mode for All2All operations (required)
        use_low_precision_combine: If True, uses FP8 optimized combine (default: False)
        node_num: Explicit node count for All2All; None means auto-compute from EP size
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
        quant_mode: common.MoEQuantMode,
        use_low_precision_combine: bool = False,
        node_num: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._attention_dp_size = attention_dp_size
        self._pre_dispatch = pre_dispatch
        self._quant_mode = quant_mode
        self._use_low_precision_combine = use_low_precision_combine
        self._node_num = node_num
        self._weights = 0.0  # MoEDispatch has no weight memory
        self.num_gpus = self._moe_ep_size * self._moe_tp_size

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query TrtLLM WideEP All2All communication latency.

        Args:
            database: Performance database instance
            **kwargs: Additional arguments including:
                - x: Number of input tokens

        Returns:
            PerformanceResult with latency (no energy for communication ops)
        """
        num_tokens = kwargs.get("x")

        phase = "Pre-dispatch" if self._pre_dispatch else "Post-dispatch"
        precision = (
            "low-precision combine"
            if self._use_low_precision_combine and not self._pre_dispatch
            else "standard precision"
        )
        logger.debug(f"TrtLLMWideEPMoEDispatch: {phase} with {precision}")

        def _as_performance_result(result) -> PerformanceResult:
            if isinstance(result, PerformanceResult):
                return result

            energy = getattr(result, "energy", 0.0)
            if not isinstance(energy, int | float):
                energy = 0.0

            source = getattr(result, "source", "silicon")
            if not isinstance(source, str):
                source = "silicon"

            return PerformanceResult(float(result), energy=energy, source=source)

        if self._pre_dispatch:
            prepare_result = database.query_trtllm_alltoall(
                op_name="alltoall_prepare",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            dispatch_result = database.query_trtllm_alltoall(
                op_name="alltoall_dispatch",
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            comm_latency = _as_performance_result(prepare_result) + _as_performance_result(dispatch_result)
        else:
            combine_op = "alltoall_combine_low_precision" if self._use_low_precision_combine else "alltoall_combine"
            combine_result = database.query_trtllm_alltoall(
                op_name=combine_op,
                num_tokens=num_tokens,
                hidden_size=self._hidden_size,
                topk=self._topk,
                num_experts=self._num_experts,
                moe_ep_size=self._moe_ep_size,
                quant_mode=self._quant_mode,
                moe_backend="wideep",
                node_num=self._node_num,
            )
            comm_latency = _as_performance_result(combine_result)

        scaled = comm_latency * self._scale_factor
        return PerformanceResult(
            float(scaled),
            energy=getattr(scaled, "energy", 0.0),
            source=getattr(scaled, "source", "empirical"),
        )

    def get_weights(self, **kwargs):
        """MoE dispatch has no weight memory."""
        return 0.0


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
        is_gated: bool = True,
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
        self._is_gated = is_gated
        self._moe_backend = kwargs.get("moe_backend")
        self._enable_eplb = kwargs.get("enable_eplb", False)
        # 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down)
        num_gemms = 3 if is_gated else 2
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            * num_gemms
            // self._moe_ep_size
            // self._moe_tp_size
        )

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
            is_gated=self._is_gated,
            enable_eplb=self._enable_eplb,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

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
        self._scale_num_tokens = kwargs.get("scale_num_tokens", 1)
        self._quant_mode = kwargs.get("quant_mode")
        self._reduce_results = kwargs.get("reduce_results", True)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        num_tokens = kwargs.get("x")
        volume = num_tokens * self._hidden_size
        _sm_version = database.system_spec["gpu"].get("sm_version", -1)
        _num_gpus_per_node = database.system_spec["node"]["num_gpus_per_node"]
        _node_num = self.num_gpus / _num_gpus_per_node

        if self._quant_mode is not None:
            _quant_compress = self._quant_mode.value.memory / 2.0
        else:
            _quant_compress = 0.25

        if database.backend == common.BackendName.trtllm.value:
            assert self._attention_tp_size == 1 or self._attention_dp_size == 1, (
                "trtllm does not support TP>1 and DP>1 for attn simultaneously"
            )
            if _sm_version == 100:
                logger.debug("MoEDispatch: In trtllm SM100 execution path")

                _alltoall_backends = {"CUTLASS", "TRTLLM"}
                backend_supports_alltoall = self._moe_backend is None or self._moe_backend.upper() in _alltoall_backends
                is_nvl72 = _num_gpus_per_node >= 72
                enable_alltoall = (
                    backend_supports_alltoall and self._attention_dp_size > 1 and self._moe_tp_size == 1 and is_nvl72
                )

                # Quantize-aware communication volume.
                # When quant_mode is known, compute compressed volume:
                #   nvfp4: volume/4 + scale_factor volume
                #   fp8:   volume/2
                #   others / unknown: full volume (BF16)
                quant_mode = self._quant_mode
                if quant_mode is not None and quant_mode == common.MoEQuantMode.nvfp4:
                    dispatch_x_volume = volume / 4
                    dispatch_sf_volume = volume / 4 / 8
                elif quant_mode is not None and quant_mode in (common.MoEQuantMode.fp8, common.MoEQuantMode.fp8_block):
                    dispatch_x_volume = volume / 2
                    dispatch_sf_volume = 0
                else:
                    dispatch_x_volume = volume
                    dispatch_sf_volume = 0

                if enable_alltoall and quant_mode is None:
                    raise ValueError("MoEDispatch requires quant_mode when TRTLLM alltoall path is enabled.")

                if self._pre_dispatch:
                    if enable_alltoall:
                        dispatch_result = database.query_trtllm_alltoall(
                            op_name="alltoall_dispatch",
                            num_tokens=num_tokens,
                            hidden_size=self._hidden_size,
                            topk=self._topk,
                            num_experts=self._num_experts,
                            moe_ep_size=self._moe_ep_size,
                            quant_mode=quant_mode,
                            moe_backend=self._moe_backend,
                        )
                        comm_latency = float(dispatch_result)
                    elif self._attention_dp_size > 1:
                        all_gather_volume = (dispatch_x_volume + dispatch_sf_volume) * self._attention_dp_size
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half, self.num_gpus, "all_gather", all_gather_volume
                        )
                    elif self._attention_tp_size > 1:
                        if self._reduce_results:
                            if _num_gpus_per_node == 72 and self.num_gpus > 4:
                                comm_latency = database.query_nccl(
                                    common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                                )
                            else:
                                comm_latency = database.query_custom_allreduce(
                                    common.CommQuantMode.half, self.num_gpus, volume
                                )
                        else:
                            comm_latency = 0
                    else:
                        comm_latency = 0
                else:
                    if enable_alltoall:
                        combine_result = database.query_trtllm_alltoall(
                            op_name="alltoall_combine",
                            num_tokens=num_tokens,
                            hidden_size=self._hidden_size,
                            topk=self._topk,
                            num_experts=self._num_experts,
                            moe_ep_size=self._moe_ep_size,
                            quant_mode=quant_mode,
                            moe_backend=self._moe_backend,
                        )
                        comm_latency = float(combine_result)
                    elif self._attention_dp_size > 1:
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                    elif self._attention_tp_size > 1:
                        if self._reduce_results:
                            if _num_gpus_per_node == 72 and self.num_gpus > 4:
                                comm_latency = database.query_nccl(
                                    common.CommQuantMode.half, self.num_gpus, "all_reduce", volume
                                )
                            else:
                                comm_latency = database.query_custom_allreduce(
                                    common.CommQuantMode.half, self.num_gpus, volume
                                )
                        else:
                            comm_latency = 0
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
            if self._moe_backend == "deepep_moe":
                logger.debug("MoEDispatch: In SGLang DeepEP execution path")
                num_tokens = num_tokens // self._scale_num_tokens
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
                combined_attention_tpdp = self._attention_tp_size > 1 and self._attention_dp_size > 1
                if self._pre_dispatch:
                    if combined_attention_tpdp:
                        # Matches SGLang DP attention: shard across attention TP, then gather across the full TP world.
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self._attention_tp_size,
                            "reduce_scatter",
                            volume,
                        )
                        comm_latency += database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "all_gather",
                            volume * self._attention_dp_size,
                        )
                    elif self._attention_tp_size > 1:  # tp>1, use allreduce
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
                    if combined_attention_tpdp:
                        # Reverse path: reduce-scatter across the full TP world, then rebuild each attention TP group.
                        comm_latency = database.query_nccl(
                            common.CommQuantMode.half,
                            self.num_gpus,
                            "reduce_scatter",
                            volume * self._attention_dp_size,
                        )
                        comm_latency += database.query_nccl(
                            common.CommQuantMode.half,
                            self._attention_tp_size,
                            "all_gather",
                            volume,
                        )
                    elif self._attention_tp_size > 1:  # tp>1, use allreduce
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

        scaled = comm_latency * self._scale_factor
        return PerformanceResult(
            float(scaled),
            energy=getattr(scaled, "energy", 0.0),
            source=getattr(scaled, "source", "empirical"),
        )

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
