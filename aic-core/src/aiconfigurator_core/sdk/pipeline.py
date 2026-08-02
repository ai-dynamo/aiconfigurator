# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipeline-parallel (PP) modeling, split into layout and schedule.

Before this module, PP was modeled as an ideal linear speedup: a step's
latency was the *whole* model's latency (``_num_layers`` is never divided by
``pp_size``) and throughput was simply multiplied by ``pp_size``. That is
exact only when every stage costs the same and the pipe is always full.

Both assumptions break in practice:

* **Stage imbalance.** A pipeline advances at ``max_i(t_i)``, not
  ``avg_i(t_i)``. The un-sharded ``lm_head`` sits alone on the last stage, so
  for a decode step it makes that stage the critical path. Qwen3-32B at
  ``pp=8`` decodes at ~83% of the ideal cycle -- i.e. ~6.7x, not 8x.
* **Uneven layer split.** ``num_layers % pp_size != 0`` used to emit a warning
  saying "we're nothing to correct this". The cycle is set by the *fattest*
  stage, so 64 layers over pp=6 costs 11/10.67 of the even split.
* **Pipe starvation.** With fewer in-flight microbatches than stages, some
  stages idle every cycle.

Two layers, deliberately separated
----------------------------------

:class:`PipelineLayout` -- **where the work lives**: the layer partition, op
placement, per-stage compute times, and the per-hop link cost. Pure geometry
plus cost lookup, no scheduling policy. This is model and hardware knowledge,
and it is the piece every consumer needs, including a request-level
discrete-event simulator (the Dynamo Mocker) driving the compiled engine.

:class:`PipelineSteadyState` -- **how the pipe runs**: microbatch count, P2P
overlap, cycle time, and the closed-form ``balance_factor`` / ``fill_factor``
that AIC's mean-field step model collapses onto scalars.

The split matters because those scalars are *surrogates* for what an
event-driven simulator computes natively. A simulator that models stage
occupancy directly must consume :class:`PipelineLayout` and derive its own
bubbles -- applying ``fill_factor`` on top would charge for the same bubble
twice. AIC needs the closed form only because it has no simulator.

This all lives in **orchestration**, above the op layer: it consumes the
per-op latency breakdown the step model already produces. No op query math
changes, so the Rust engine-step parity surface (see
``.claude/rules/rust-core/parity.md``) is untouched and both engines benefit.

``pp_size == 1`` always yields ``balance_factor == fill_factor == 1.0``
exactly, so single-stage results are bit-identical to the previous model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Op-name -> pipeline placement. Leans on the naming contract the model
# classes already require (see the GPTModel docstring: "attn layer name needs
# to be context_attention or generation_attention, exact match is required.
# Same for logits_gemm"). Everything not matched here is per-layer and follows
# the layer partition.
_FIRST_STAGE_MARKERS = ("embedding",)
_LAST_STAGE_MARKERS = ("logits_gemm",)
_LINK_MARKERS = ("p2p",)

# An op whose scale factor is below this fraction of num_layers is almost
# certainly NOT per-layer. Used only to warn when a new non-per-layer op
# appears that the markers above do not know about.
_PER_LAYER_SCALE_FLOOR = 0.25


class Placement:
    FIRST = "first"
    LAST = "last"
    LINK = "link"
    PER_LAYER = "per_layer"


def classify(op_name: str) -> str:
    """Map an op name to its pipeline placement."""
    lowered = op_name.lower()
    if any(marker in lowered for marker in _LINK_MARKERS):
        return Placement.LINK
    if any(marker in lowered for marker in _FIRST_STAGE_MARKERS):
        return Placement.FIRST
    if any(marker in lowered for marker in _LAST_STAGE_MARKERS):
        return Placement.LAST
    return Placement.PER_LAYER


def even_partition(num_layers: int, pp_size: int) -> tuple[int, ...]:
    """Layers per stage: even split, remainder on the leading stages.

    Matches the default rule in vLLM (``get_pp_indices``) and TRT-LLM.
    """
    if pp_size <= 1:
        return (num_layers,)
    base, rem = divmod(num_layers, pp_size)
    return tuple(base + (1 if i < rem else 0) for i in range(pp_size))


# ---------------------------------------------------------------------------
# Layer 1: where the work lives -- shared by every consumer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineLayout:
    """Which ops land on which stage, and what each stage costs.

    Carries no scheduling policy: it answers "what does stage *i* cost for
    this batch" and "what does one hop cost", nothing about occupancy. That
    keeps it usable by both AIC's closed-form model and an event-driven
    simulator that derives bubbles from occupancy itself.

    Args:
        pp_size: Number of pipeline stages.
        partition: Explicit layers-per-stage. ``None`` uses
            :func:`even_partition`.
    """

    pp_size: int = 1
    partition: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.pp_size < 1:
            raise ValueError(f"pp_size must be >= 1, got {self.pp_size}")
        if self.partition is not None and len(self.partition) != self.pp_size:
            raise ValueError(f"partition has {len(self.partition)} entries but pp_size is {self.pp_size}")

    def layer_partition(self, num_layers: int) -> tuple[int, ...]:
        return self.partition if self.partition is not None else even_partition(num_layers, self.pp_size)

    def stage_times(self, per_op_ms: dict[str, float], num_layers: int) -> list[float]:
        """Fold a per-op latency breakdown into per-stage compute times.

        Link (P2P) ops are excluded -- they are not stage compute. Use
        :meth:`per_hop_latency` for the transfer cost.
        """
        parts = self.layer_partition(num_layers)
        per_layer_total = 0.0
        first_total = 0.0
        last_total = 0.0
        for name, latency in per_op_ms.items():
            place = classify(name)
            if place == Placement.LINK:
                continue
            if place == Placement.FIRST:
                first_total += latency
            elif place == Placement.LAST:
                last_total += latency
            else:
                per_layer_total += latency

        per_layer_ms = per_layer_total / num_layers if num_layers else 0.0
        times = [per_layer_ms * n for n in parts]
        times[0] += first_total
        times[-1] += last_total
        return times

    @staticmethod
    def link_latency(per_op_ms: dict[str, float]) -> float:
        """Total P2P latency in the step (already scaled by ``pp_size - 1``)."""
        return sum(v for k, v in per_op_ms.items() if classify(k) == Placement.LINK)

    def per_hop_latency(self, per_op_ms: dict[str, float]) -> float:
        """P2P cost of a single stage-to-stage hop."""
        if self.pp_size <= 1:
            return 0.0
        return self.link_latency(per_op_ms) / (self.pp_size - 1)


# ---------------------------------------------------------------------------
# Layer 2: how the pipe runs -- AIC's mean-field closed form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineSteadyState:
    """Closed-form steady-state collapse of a pipeline's occupancy.

    Produces the two scalars AIC's mean-field step model needs. An
    event-driven simulator should NOT use these -- it consumes
    :class:`PipelineLayout` and derives the same effects from simulated
    occupancy. Applying both double-counts the bubble.

    Args:
        layout: Stage geometry and per-stage costs.
        num_microbatches: In-flight microbatches. ``None`` means "exactly
            enough to fill the pipe" (``= pp_size``), which is the assumption
            the pre-existing model made unconditionally.
        p2p_overlap: Whether the stage-to-stage transfer is hidden behind
            compute. Defaults to False (serialized), matching how the P2P op
            is already summed into the step latency.
    """

    layout: PipelineLayout = field(default_factory=PipelineLayout)
    num_microbatches: int | None = None
    p2p_overlap: bool = False

    def __post_init__(self) -> None:
        if self.num_microbatches is not None and self.num_microbatches < 1:
            raise ValueError(f"num_microbatches must be >= 1, got {self.num_microbatches}")

    @property
    def pp_size(self) -> int:
        return self.layout.pp_size

    def layer_partition(self, num_layers: int) -> tuple[int, ...]:
        return self.layout.layer_partition(num_layers)

    def cycle_time(self, per_op_ms: dict[str, float], num_layers: int) -> float:
        """Steady-state time for one stage to retire one microbatch."""
        stages = self.layout.stage_times(per_op_ms, num_layers)
        hop = 0.0 if self.p2p_overlap else self.layout.per_hop_latency(per_op_ms)
        return max(stages) + hop

    def fill_factor(self) -> float:
        """Fraction of stages busy in steady state (1.0 when the pipe is full)."""
        m = self.num_microbatches if self.num_microbatches is not None else self.pp_size
        return min(1.0, m / self.pp_size)

    def balance_factor(self, per_op_ms: dict[str, float], num_layers: int) -> float:
        """How close the stages are to even, in ``(0, 1]``.

        ``ideal_cycle`` is what the pre-existing model assumed implicitly: the
        whole-model step latency divided evenly across stages. The realized
        cycle is set by the fattest stage plus one P2P hop, so a microbatch's
        real traversal time is ``step_total / balance_factor``.

        Divide per-microbatch latency by this; it also gates throughput.
        """
        if self.pp_size <= 1:
            return 1.0

        step_total = sum(per_op_ms.values())
        if step_total <= 0.0:
            return 1.0

        cycle = self.cycle_time(per_op_ms, num_layers)
        if cycle <= 0.0:
            return 1.0

        return min(1.0, (step_total / self.pp_size) / cycle)

    def efficiency(self, per_op_ms: dict[str, float], num_layers: int) -> float:
        """Realized fraction of the ideal ``pp_size`` speedup, in ``(0, 1]``.

        Throughput scales as ``pp_size * efficiency``. This is
        ``balance_factor`` (a per-microbatch latency effect) times
        ``fill_factor`` (a throughput-only effect).
        """
        return self.balance_factor(per_op_ms, num_layers) * self.fill_factor()

    def describe(self, per_op_ms: dict[str, float], num_layers: int) -> dict[str, object]:
        """Per-stage breakdown, for debugging and reports."""
        stages = self.layout.stage_times(per_op_ms, num_layers)
        step_total = sum(per_op_ms.values())
        return {
            "pp_size": self.pp_size,
            "partition": list(self.layer_partition(num_layers)),
            "stage_times_ms": stages,
            "critical_stage": stages.index(max(stages)) if stages else -1,
            "per_hop_p2p_ms": self.layout.per_hop_latency(per_op_ms),
            "cycle_time_ms": self.cycle_time(per_op_ms, num_layers),
            "ideal_cycle_ms": step_total / self.pp_size if self.pp_size else step_total,
            "fill_factor": self.fill_factor(),
            "balance_factor": self.balance_factor(per_op_ms, num_layers),
            "efficiency": self.efficiency(per_op_ms, num_layers),
        }


def warn_on_unclassified_ops(ops, num_layers: int) -> None:
    """Warn when an op looks non-per-layer but no placement marker matched.

    Guards against a new embedding-/head-like op being added without teaching
    this module about it, which would silently smear it across every stage.
    """
    if num_layers <= 0:
        return
    floor = num_layers * _PER_LAYER_SCALE_FLOOR
    for op in ops:
        name = getattr(op, "_name", "")
        if classify(name) != Placement.PER_LAYER:
            continue
        scale = getattr(op, "_scale_factor", None)
        if scale is not None and 0 < float(scale) < floor:
            logger.warning(
                "PP placement: op %r has scale_factor=%s (< %.1f for %d layers) but is "
                "treated as per-layer. If it is a first-/last-stage op, add a marker in "
                "aiconfigurator_core.sdk.pipeline.",
                name,
                scale,
                floor,
                num_layers,
            )
