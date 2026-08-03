# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-pass workload and estimate contracts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MixedStepInput:
    """Scheduled work for one mixed prefill/decode engine iteration.

    ``num_decode_requests`` is the number of logical decode sequences. It is
    deliberately distinct from the target-model query-token count, which is
    derived from the model's speculative draft depth.
    """

    context_tokens: int
    num_decode_requests: int

    def __post_init__(self) -> None:
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive for a mixed step")
        if self.num_decode_requests < 0:
            raise ValueError("num_decode_requests must be non-negative")


@dataclass(frozen=True)
class StepEstimate:
    """Raw wall-time estimate for one scheduled engine iteration."""

    latency_ms: float
    energy_wms: float
    component_latency_ms: dict[str, float] = field(default_factory=dict)
    component_energy_wms: dict[str, float] = field(default_factory=dict)
    per_op_latency_ms: dict[str, float] = field(default_factory=dict)
    per_op_source: dict[str, str] = field(default_factory=dict)
    context_tokens: int = 0
    num_decode_requests: int = 0
    num_decode_query_tokens: int = 0

    def legacy_tuple(self) -> tuple[float, float, dict[str, float], dict[str, str]]:
        """Return the pre-contract private-helper shape."""
        return self.latency_ms, self.energy_wms, self.per_op_latency_ms, self.per_op_source


__all__ = ["MixedStepInput", "StepEstimate"]
