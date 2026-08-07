# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conservative AIC memory admission for FPM parallel topologies."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from aiconfigurator.sdk.memory import KVCacheEstimator

from .capabilities import ModelCapabilityProfile
from .types import ParallelTopology

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DTypeMemoryEstimate:
    kv_cache_dtype: str
    disposition: str
    estimated_non_kv_bytes: int | None
    gpu_capacity_bytes: int | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kv_cache_dtype": self.kv_cache_dtype,
            "disposition": self.disposition,
            "estimated_non_kv_bytes": self.estimated_non_kv_bytes,
            "gpu_capacity_bytes": self.gpu_capacity_bytes,
            "headroom_bytes": (
                self.gpu_capacity_bytes - self.estimated_non_kv_bytes
                if self.estimated_non_kv_bytes is not None and self.gpu_capacity_bytes is not None
                else None
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TopologyMemoryDecision:
    topology: ParallelTopology
    disposition: str
    max_new_tokens: int
    estimates: tuple[DTypeMemoryEstimate, ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "disposition": self.disposition,
            "source": "aic_native_configured_max_new_tokens",
            "activation_envelope": {
                "max_new_tokens": self.max_new_tokens,
                "aic_max_num_tokens": self.max_new_tokens,
                "max_batch_size": 1,
            },
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "reason": self.reason,
        }


def _estimate_dtype(
    *,
    backend: str,
    model_path: str,
    system: str,
    capability: ModelCapabilityProfile,
    topology: ParallelTopology,
    kv_cache_dtype: str,
    max_new_tokens: int,
) -> DTypeMemoryEstimate:
    try:
        breakdown = KVCacheEstimator.from_request(
            model_path,
            system,
            backend,
            capability.aic_database_version,
            max_num_tokens=max_new_tokens,
            max_batch_size=1,
            tp_size=topology.tp,
            pp_size=topology.pp,
            attention_dp_size=topology.dp,
            moe_tp_size=topology.moe_tp,
            moe_ep_size=topology.moe_ep,
            gemm_quant_mode=capability.dtype.gemm_quant_mode,
            moe_quant_mode=capability.dtype.moe_quant_mode,
            kvcache_quant_mode=kv_cache_dtype,
            fmha_quant_mode=capability.dtype.fmha_by_kv_dtype[kv_cache_dtype],
            comm_quant_mode=capability.dtype.comm_quant_mode,
        ).breakdown
        non_kv = math.ceil(float(breakdown["non_kv_bytes"]))
        capacity = math.floor(float(breakdown["gpu_memory_capacity_bytes"]))
        if non_kv < 0 or capacity <= 0:
            raise ValueError(f"invalid AIC memory estimate: non_kv={non_kv}, capacity={capacity}")
    except Exception as error:
        return DTypeMemoryEstimate(
            kv_cache_dtype=kv_cache_dtype,
            disposition="unknown",
            estimated_non_kv_bytes=None,
            gpu_capacity_bytes=None,
            reason=f"AIC memory estimate unavailable: {type(error).__name__}: {error}",
        )

    rejected = non_kv >= capacity
    return DTypeMemoryEstimate(
        kv_cache_dtype=kv_cache_dtype,
        disposition="rejected" if rejected else "admitted",
        estimated_non_kv_bytes=non_kv,
        gpu_capacity_bytes=capacity,
        reason=(
            "AIC configured max-new-token non-KV memory is not below GPU capacity"
            if rejected
            else "AIC configured max-new-token non-KV memory is below GPU capacity"
        ),
    )


def filter_memory_infeasible_topologies(
    *,
    backend: str,
    model_path: str,
    system: str,
    capability: ModelCapabilityProfile,
    topologies: tuple[ParallelTopology, ...],
    max_new_tokens: int,
) -> tuple[tuple[ParallelTopology, ...], tuple[TopologyMemoryDecision, ...]]:
    """Drop topologies that cannot fit the configured max-new-token envelope.

    This is intentionally a one-sided generation-time filter. A topology is
    rejected only when every requested KV dtype has a successful AIC estimate
    and all estimates exceed rank-local physical capacity. Unknown estimates
    remain runnable so the target runtime stays authoritative.
    """

    if max_new_tokens < 1:
        raise ValueError("FPM topology memory admission requires positive max_new_tokens")

    decisions = []
    admitted = []
    rejected = []
    for topology in topologies:
        estimates = tuple(
            _estimate_dtype(
                backend=backend,
                model_path=model_path,
                system=system,
                capability=capability,
                topology=topology,
                kv_cache_dtype=kv_cache_dtype,
                max_new_tokens=max_new_tokens,
            )
            for kv_cache_dtype in capability.dtype.kv_cache_dtypes
        )
        dispositions = {estimate.disposition for estimate in estimates}
        if "admitted" in dispositions:
            disposition = "admitted"
            reason = "at least one requested KV dtype fits the configured max-new-token envelope"
            admitted.append(topology)
        elif "unknown" in dispositions:
            disposition = "unknown"
            reason = "AIC could not prove the topology is impossible; runtime verification is required"
            admitted.append(topology)
        else:
            disposition = "rejected"
            reason = "all requested KV dtypes exceed rank-local GPU capacity at configured max new tokens"
            rejected.append((topology, estimates))
        decisions.append(
            TopologyMemoryDecision(
                topology=topology,
                disposition=disposition,
                max_new_tokens=max_new_tokens,
                estimates=estimates,
                reason=reason,
            )
        )

    if rejected:
        details = []
        for topology, estimates in rejected:
            best = min(
                estimates,
                key=lambda estimate: estimate.estimated_non_kv_bytes
                if estimate.estimated_non_kv_bytes is not None
                else math.inf,
            )
            details.append(
                f"{topology.to_dict()}="
                f"{best.estimated_non_kv_bytes / 2**30:.2f}/"
                f"{best.gpu_capacity_bytes / 2**30:.2f} GiB"
            )
        logger.warning(
            "fpm_forward: dropped %d/%d topologies (AIC configured max-new-token non-KV memory "
            "exceeds GPU capacity, system=%s, max_new_tokens=%d): %s",
            len(rejected),
            len(topologies),
            system,
            max_new_tokens,
            "; ".join(details),
        )
    if not admitted:
        raise ValueError(
            "AIC max-new-token memory admission rejected every structurally valid FPM topology; "
            f"model={model_path!r}, system={system!r}, max_new_tokens={max_new_tokens}"
        )
    return tuple(admitted), tuple(decisions)
