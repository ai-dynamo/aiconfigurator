# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-residency admission for typical FPM deployment candidates."""

from __future__ import annotations

from dataclasses import dataclass

from aiconfigurator.sdk.memory import estimate_kv_cache
from aiconfigurator.sdk.perf_database import load_system_spec

from .capabilities import ModelCapabilityProfile
from .types import ParallelTopology

_VLLM_DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    topology: ParallelTopology
    admitted: bool
    source: str
    reason: str
    rank_local_kv_capacity_tokens: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "topology": self.topology.to_dict(),
            "admitted": self.admitted,
            "source": self.source,
            "reason": self.reason,
            "rank_local_kv_capacity_tokens": self.rank_local_kv_capacity_tokens,
        }


def _generator_memory_fraction(generator_overrides: dict[str, object]) -> float:
    params = generator_overrides.get("params")
    agg = params.get("agg") if isinstance(params, dict) else None
    raw = agg.get("gpu_memory_utilization") if isinstance(agg, dict) else None
    value = _VLLM_DEFAULT_GPU_MEMORY_UTILIZATION if raw is None else float(raw)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"params.agg.gpu_memory_utilization must be in (0, 1], got {value}")
    return value


def admit_model_residency(
    *,
    model_path: str,
    system: str,
    backend: str,
    capability: ModelCapabilityProfile,
    topologies: tuple[ParallelTopology, ...],
    generator_overrides: dict[str, object],
) -> tuple[tuple[ParallelTopology, ...], tuple[CapacityDecision, ...], float]:
    """Keep tuples whose rank-local non-KV state leaves a positive KV budget.

    This is deliberately a model-residency check (one token, one request), not
    a claim that every workload point fits. Point-level capacity remains a
    separate collection concern.
    """

    memory_fraction = _generator_memory_fraction(generator_overrides)
    system_spec = load_system_spec(system)
    capacity_bytes = int((system_spec.get("gpu") or {}).get("mem_capacity") or 0)
    if capacity_bytes <= 0:
        raise ValueError(f"AIC system spec has no positive GPU memory capacity for {system!r}")
    decisions: list[CapacityDecision] = []
    admitted: list[ParallelTopology] = []
    for topology in topologies:
        try:
            estimate = estimate_kv_cache(
                model_path,
                system,
                backend,
                capability.aic_database_version,
                max_num_tokens=1,
                max_batch_size=1,
                memory_fraction_kind="of_total",
                memory_fraction_value=memory_fraction,
                tp_size=topology.tp,
                pp_size=topology.pp,
                attention_dp_size=topology.dp,
                moe_tp_size=topology.moe_tp,
                moe_ep_size=topology.moe_ep,
                gemm_quant_mode=capability.dtype.gemm_quant_mode,
                moe_quant_mode=capability.dtype.moe_quant_mode,
                kvcache_quant_mode=capability.dtype.native_kv_cache_dtype,
                fmha_quant_mode=capability.dtype.fmha_quant_mode,
                comm_quant_mode=capability.dtype.comm_quant_mode,
                gpu_memory_capacity_bytes_override=capacity_bytes,
                allow_naive_fallback=capability.support_level == "bootstrap_template",
            )
        except ValueError as error:
            if capability.support_level == "bootstrap_template" and str(error).startswith(
                "unsupported model/backend/GPU for KV-cache estimation"
            ):
                # The generic template is specifically the path for models AIC
                # cannot build. Keep the candidate, but make the unverified
                # residency explicit so execution cannot mistake it for proof.
                decisions.append(
                    CapacityDecision(
                        topology,
                        True,
                        "bootstrap_unverified",
                        f"AIC and naive residency estimators unavailable: {error}",
                        None,
                    )
                )
                admitted.append(topology)
                continue
            decisions.append(CapacityDecision(topology, False, "aic_memory", str(error), None))
            continue

        kv_tokens = int(estimate["total_kv_size_tokens"])
        source = str(estimate["source"])
        decisions.append(
            CapacityDecision(
                topology,
                True,
                source,
                "rank-local model state leaves a positive baseline KV budget",
                kv_tokens,
            )
        )
        admitted.append(topology)

    if not admitted:
        reasons = "; ".join(decision.reason for decision in decisions)
        raise ValueError(f"no typical topology passes AIC model-residency admission: {reasons}")
    return tuple(admitted), tuple(decisions), memory_fraction
