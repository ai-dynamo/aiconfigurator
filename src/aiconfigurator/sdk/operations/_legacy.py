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
