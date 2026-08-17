# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD communication ops — cross-pool P2P transfer and intra-pool collectives.

Four ops model the full AFD communication path:

* ``AFDTransfer`` — unidirectional cross-pool P2P (A→F or F→A)
* ``AFDFAllGather`` — F-node intra-node AllGather along the token dimension
* ``AFDFReduceScatter`` — F-node intra-node ReduceScatter after F compute
* ``AFDCombine`` — A-side cross-EP local HBM reduce-add

These are ORCHESTRATION ops (their ``query`` overrides are whitelisted by the
single-oracle contract): the A/F topology math — send probability, per-link
volumes, rank mapping — stays here in Python, while the per-message latency
value comes from the compiled engine via a single-element op-list evaluation
of the standard comm twin (``P2P`` / ``NCCL`` / ``ElementWise``). Byte-exact
volumes are expressed as ``ceil(bytes/2)`` bf16 elements on the probe op —
at most 1 byte of rounding on multi-MB messages.

TODO(#1357 follow-up): consider porting these four ops into the compiled
engine as real ``Op`` variants. Their call surface is already op-shaped
(per-op performance queries whose values come from engine-evaluated twins),
so they would slot into the same Rust op-binding pattern as every other
family — the twin composition and the ``comm_overhead_factor`` scaling would
simply move inside the Rust ``query``, and the Python classes would become
the same thin engine-backed bindings as GEMM. Trigger points for doing it:
a Rust-side consumer needing AFD at runtime (the Dynamo Mocker's embedded
hot path cannot reach this Python-side math), or retiring the last
``query()`` orchestration whitelist entries. The partition SEARCH
(``afd_partition.py`` / the session's A-F sweep) stays with the caller
either way. Flipping this is a modeling-boundary change: it needs a
tracking issue + maintainer sign-off per ``.claude/rules/rust-core/
parity.md`` ("Known intentional splits").
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Optional

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import PythonOperation
from aiconfigurator_core.sdk.operations.communication import NCCL, P2P
from aiconfigurator_core.sdk.operations.elementwise import ElementWise
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase


def _engine_comm_query(database: PerfDatabase, op) -> PerformanceResult:
    """Engine value for one comm probe op. ``x=1``: the probe carries the full
    message in its per-token field, so no token factorization is needed."""
    from aiconfigurator_core.sdk.engine import _evaluate_single_op

    return _evaluate_single_op(database, op, is_context=True, batch_size=1, s=1, x=1)


def _afd_send_prob(num_experts: int, topk: int, num_f_nodes: int) -> float:
    """Probability that a token must be dispatched to a given F-node.

    For MoE with expert parallelism (num_experts > 0, topk > 0,
    num_f_nodes > 1): uses the combinatorial formula
    ``P_send = 1 - C(E - E/Nf, k) / C(E, k)`` -- the probability that
    at least one of a token's top-k experts resides on the target F-node.

    For dense models or degenerate configs: returns ``1 / num_f_nodes``
    (uniform distribution of tokens across F-nodes).
    """
    # dense model not verified yet
    if num_experts <= 0 or topk <= 0 or num_f_nodes <= 1:  # degenerate configs
        return 1.0 / max(num_f_nodes, 1)
    experts_per_node = num_experts // num_f_nodes
    if experts_per_node <= 0:
        return 1.0 / max(num_f_nodes, 1)

    n_other = num_experts - experts_per_node
    if topk > n_other:
        return 1.0
    return 1.0 - comb(n_other, topk) / comb(num_experts, topk)


class AFDTransfer(PythonOperation):
    """Unidirectional cross-pool P2P transfer (A→F **or** F→A).

    Construct with ``direction="a2f"`` or ``direction="f2a"`` to declare
    which leg of the round-trip this instance models.  ``query()``
    returns the single-direction latency.

    A-side operates in DP mode: each A-rank holds its own independent
    tokens and sends/receives them with the full ``hidden_size`` per
    token.  The per-link payload is therefore symmetric for both
    directions.

    Hetero A/F hardware: the A→F link spans two device types, so its
    throughput is set by the slower endpoint. Pass ``peer_database`` to
    ``query()`` and the op prices the transfer at the **bottleneck** side
    -- for a fixed payload that is the larger of the two latencies, which
    is exactly ``min`` of the two bandwidths (each side's ``p2p_latency``
    constant is already folded into ``query_p2p``).

    Super-node topology: ``span_gpus`` is how many GPUs the A and F pools
    occupy together. It selects the fabric tier -- a pool pair that fits
    inside one scale-up domain is priced on NVLink/NVSwitch, one that
    exceeds ``num_gpus_per_rack`` on the scale-out fabric. Leaving it
    ``None`` keeps the flat ``inter_node_bw`` pricing.

    The two knobs are orthogonal and compose: each side resolves its own
    tier against its own system spec (rack width is a hardware fact, so
    hetero pools can disagree on it), then bottleneck pricing picks the
    slower of the two tier-resolved latencies.
    """

    _VALID_DIRECTIONS = ("a2f", "f2a")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        direction: str,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        f_gpus_per_node: int | None = None,
        span_gpus: int | None = None,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        comm_overhead_factor: float = 1.0,
    ) -> None:
        super().__init__(name, scale_factor)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"AFDTransfer: direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}")
        self._direction = direction
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        # F-node grouping is an F-pool hardware fact; under hetero A/F it
        # differs from the A pool. Falls back to gpus_per_node.
        self._f_gpus_per_node = max(int(f_gpus_per_node or gpus_per_node), 1)
        # GPUs spanned by the A+F pools together, used to pick the fabric
        # tier. None keeps the flat inter_node_bw pricing.
        self._span_gpus = max(int(span_gpus), 1) if span_gpus else None
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        self._weights = 0.0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def num_f_nodes(self) -> int:
        """Physical F-node count: ``ceil(n_f_workers / f_gpus_per_node)``."""
        return max((self._n_f_workers + self._f_gpus_per_node - 1) // self._f_gpus_per_node, 1)

    @property
    def span_gpus(self) -> int | None:
        """GPUs spanned by the A+F pools, or ``None`` for flat pricing."""
        return self._span_gpus

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        bpe = self._comm_quant_mode.value.memory
        # x = tokens held by a single A-rank; per_link_bytes is
        # the volume on one A-rank → F-rank P2P connection.
        per_link_bytes = int(p_send * x * self._hidden_size * bpe)
        if per_link_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        # pp_size=2 passes the twin's "pp_size=1 is a no-op" gate; a single
        # P2P link is charged regardless of the actual pp depth. ``span_gpus``
        # selects the fabric tier inside the engine -- None keeps the flat
        # inter-node pricing, which is what a caller that does not care about
        # tiers wants.
        probe = P2P("afd_p2p", 1.0, -(-per_link_bytes // 2), 2, span_gpus=self._span_gpus)
        result = _engine_comm_query(database, probe)
        peer_database = kwargs.get("peer_database")
        if peer_database is not None and peer_database is not database:
            # Bottleneck pricing: same payload, so the slower side wins. The
            # probe is reused verbatim -- only the database differs, so each
            # side resolves its own fabric tier from its own system spec and
            # the two evaluations stay comparable.
            peer_result = _engine_comm_query(peer_database, probe)
            if float(peer_result) > float(result):
                result = peer_result
        lat = float(result) * self._comm_overhead_factor
        return PerformanceResult(
            lat * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDFAllGather(PythonOperation):
    """F-node intra-node AllGather along the **token** dimension before F compute.

    Each F-GPU within a node receives a subset of tokens from A-side P2P.
    The AllGather collects all token subsets across the ``gpus_per_node``
    GPUs so that every F-GPU sees the complete token batch needed for
    FFN/MoE computation.

    Returns 0 when the node has only 1 GPU or under broadcast rank mapping.
    """

    _VALID_RANK_MAPPINGS = ("one_to_one", "broadcast")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        f_gpus_per_node: int | None = None,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        rank_mapping: str = "one_to_one",
    ) -> None:
        super().__init__(name, scale_factor)
        if rank_mapping not in self._VALID_RANK_MAPPINGS:
            raise ValueError(
                f"AFDFAllGather: rank_mapping must be one of {self._VALID_RANK_MAPPINGS}, got {rank_mapping!r}"
            )
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        # F-node grouping / intra-node F-GPU count are F-pool hardware
        # facts; under hetero A/F they differ from the A pool.
        self._f_gpus_per_node = max(int(f_gpus_per_node or gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._rank_mapping = rank_mapping
        self._weights = 0.0

    @property
    def num_f_nodes(self) -> int:
        return max(
            (self._n_f_workers + self._f_gpus_per_node - 1) // self._f_gpus_per_node,
            1,
        )

    @property
    def f_gpus_in_node(self) -> int:
        """Number of F-GPUs within a single node."""
        return min(self._n_f_workers, self._f_gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._rank_mapping != "one_to_one":
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        total = x * self._n_a_workers
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        tokens_per_f_node = p_send * total
        per_rank_elements = int(tokens_per_f_node * self._hidden_size / f_local)
        if per_rank_elements <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = _engine_comm_query(
            database, NCCL("afd_all_gather", 1.0, "all_gather", per_rank_elements, f_local, self._comm_quant_mode)
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDFReduceScatter(PythonOperation):
    """F-node intra-node NCCL ReduceScatter after F compute.

    After MoE/FFN, every F-GPU within a node holds results for *all*
    tokens that were AllGathered earlier.  Because A-rank <-> F-rank is
    one-to-one mapped, a ReduceScatter along the **token** dimension
    places each A-rank's tokens onto the corresponding F-GPU, ready
    for the F->A P2P transfer.

    The number of participants is ``min(n_f_workers, gpus_per_node)``
    -- the intra-node F-GPU count -- regardless of TP or EP configuration.
    Returns 0 when the node has only 1 F-GPU or under broadcast rank
    mapping.
    """

    _VALID_RANK_MAPPINGS = ("one_to_one", "broadcast")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        f_gpus_per_node: int | None = None,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        rank_mapping: str = "one_to_one",
    ) -> None:
        super().__init__(name, scale_factor)
        if rank_mapping not in self._VALID_RANK_MAPPINGS:
            raise ValueError(
                f"AFDFReduceScatter: rank_mapping must be one of {self._VALID_RANK_MAPPINGS}, got {rank_mapping!r}"
            )
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        # F-node grouping / intra-node F-GPU count are F-pool hardware
        # facts; under hetero A/F they differ from the A pool.
        self._f_gpus_per_node = max(int(f_gpus_per_node or gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._rank_mapping = rank_mapping
        self._weights = 0.0

    @property
    def num_f_nodes(self) -> int:
        return max((self._n_f_workers + self._f_gpus_per_node - 1) // self._f_gpus_per_node, 1)

    @property
    def f_gpus_in_node(self) -> int:
        """Number of F-GPUs within a single node."""
        return min(self._n_f_workers, self._f_gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._rank_mapping != "one_to_one":
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        total = x * self._n_a_workers
        nf = self.num_f_nodes
        p_send = _afd_send_prob(self._num_experts, self._topk, nf)
        tokens_per_f_node = p_send * total
        per_rank_elements = int(tokens_per_f_node * self._hidden_size / f_local)
        if per_rank_elements <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = _engine_comm_query(
            database,
            NCCL("afd_reduce_scatter", 1.0, "reduce_scatter", per_rank_elements, f_local, self._comm_quant_mode),
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights


class AFDCombine(PythonOperation):
    """A-side cross-EP combine: local HBM reduce-add of partial results.

    When F-side uses expert parallelism (``f_moe_ep_size > 1``), each
    A-rank receives ``f_moe_ep_size`` partial results from different
    EP partitions and reduces them locally.  Each partial result carries
    the full ``hidden_size`` per token (F->A transfers full hidden).
    Returns 0 for dense FFN (``f_moe_ep_size <= 1``).
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        tp_a: int = 1,
        f_moe_ep_size: int = 1,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = int(hidden_size)
        self._tp_a = max(int(tp_a), 1)
        self._f_moe_ep_size = max(int(f_moe_ep_size), 1)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        if self._f_moe_ep_size <= 1:  # no expert parallelism
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        tokens_per_a_rank = (x + self._tp_a - 1) // self._tp_a
        bpe = self._comm_quant_mode.value.memory
        total_bytes = int((self._f_moe_ep_size + 1) * tokens_per_a_rank * self._hidden_size * bpe)
        if total_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        result = _engine_comm_query(database, ElementWise("afd_combine_mem", 1.0, -(-total_bytes // 2), 0))
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights
