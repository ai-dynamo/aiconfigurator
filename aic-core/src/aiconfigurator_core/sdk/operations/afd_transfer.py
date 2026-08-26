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

Two dispatch topologies are modeled, selected by ``dispatch_mode``:

* ``"f_side_routing"`` (default): A sends each token once per F-node that
  needs it; the F-node AllGather/ReduceScatter then move tokens between
  the GPUs inside the node.
* ``"a_side_routing"`` (DeepEP low-latency style): A knows the routing and
  sends each token directly to the GPUs holding its top-k experts, so the
  F-node AllGather/ReduceScatter return 0. MoE only. Note this only
  changes the communication model; the router GEMM op itself stays on the
  F side (it is tiny and, for the big MoE models, fused into an
  un-splittable ``OverlapOp``).

In both modes the A<->F transfer is billed as the TOTAL bytes one A-rank
sends (all destinations share the rank's single NIC), so the two modes
are directly comparable.

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
from aiconfigurator_core.sdk.config import AFD_DISPATCH_MODES
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


def afd_dest_ep_ranks(num_experts: int, topk: int, f_moe_ep_size: int) -> float:
    """How many EP ranks one token is sent to, on average (a_side_routing).

    A token goes to every EP rank holding at least one of its top-k
    experts. Reuses ``_afd_send_prob`` with the EP-rank count in place of
    the F-node count. At most ``min(topk, f_moe_ep_size)``, and less when
    several of the token's experts land on the same rank.
    """
    n_ep = max(int(f_moe_ep_size), 1)
    return n_ep * _afd_send_prob(num_experts, topk, n_ep)


class AFDTransfer(PythonOperation):
    """Unidirectional cross-pool P2P transfer (A→F **or** F→A).

    Construct with ``direction="a2f"`` or ``direction="f2a"`` to declare
    which leg of the round-trip this instance models.  ``query()``
    returns the single-direction latency.

    A-side operates in DP mode: each A-rank holds its own independent
    tokens and sends/receives them with the full ``hidden_size`` per
    token.  Billed volume = total bytes one A-rank sends across all its
    destinations (they share one NIC); byte formulas are inline in
    :meth:`query`.
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
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        comm_overhead_factor: float = 1.0,
        dispatch_mode: str = "f_side_routing",
        f_moe_ep_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        if direction not in self._VALID_DIRECTIONS:
            raise ValueError(f"AFDTransfer: direction must be one of {self._VALID_DIRECTIONS}, got {direction!r}")
        self._direction = direction
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        if dispatch_mode not in AFD_DISPATCH_MODES:
            raise ValueError(f"AFDTransfer: dispatch_mode must be one of {AFD_DISPATCH_MODES}, got {dispatch_mode!r}")
        self._dispatch_mode = dispatch_mode
        self._f_moe_ep_size = max(int(f_moe_ep_size), 1)
        if self._dispatch_mode == "a_side_routing" and (self._num_experts <= 0 or self._topk <= 0):
            raise ValueError(
                "AFDTransfer: dispatch_mode='a_side_routing' requires a MoE model "
                "(a dense FFN has no routing to move to the A side)."
            )
        self._weights = 0.0

    @property
    def direction(self) -> str:
        return self._direction

    @property
    def num_f_nodes(self) -> int:
        """Physical F-node count: ``ceil(n_f_workers / gpus_per_node)``."""
        return max((self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node, 1)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        bpe = self._comm_quant_mode.value.memory
        # Both modes: bytes = (expected destinations per token) * (copies per
        # destination) * x * H. Only the destination granularity differs.
        if self._dispatch_mode == "a_side_routing":
            # Destinations are EP ranks. a2f copies the token to every TP
            # shard of the rank; f2a returns one summed partial per rank.
            n_dest = afd_dest_ep_ranks(self._num_experts, self._topk, self._f_moe_ep_size)
            copies = max(self._n_f_workers // self._f_moe_ep_size, 1) if self._direction == "a2f" else 1
        else:
            # Destinations are F-nodes, one copy each. Dense case:
            # p_send = 1/nf, so the total is x * H exactly.
            nf = self.num_f_nodes
            n_dest = nf * _afd_send_prob(self._num_experts, self._topk, nf)
            copies = 1
        message_bytes = int(n_dest * copies * x * self._hidden_size * bpe)
        if message_bytes <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        # pp_size=2 passes the twin's "pp_size=1 is a no-op" gate; a single
        # P2P link is charged regardless of the actual pp depth.
        result = _engine_comm_query(database, P2P("afd_p2p", 1.0, -(-message_bytes // 2), 2))
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

    Only needed under ``dispatch_mode="f_side_routing"``; returns 0 under
    a_side_routing (tokens already arrive where they are needed) and when
    the node has only 1 GPU.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        dispatch_mode: str = "f_side_routing",
    ) -> None:
        super().__init__(name, scale_factor)
        if dispatch_mode not in AFD_DISPATCH_MODES:
            raise ValueError(f"AFDFAllGather: dispatch_mode must be one of {AFD_DISPATCH_MODES}, got {dispatch_mode!r}")
        self._dispatch_mode = dispatch_mode
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    @property
    def num_f_nodes(self) -> int:
        return max(
            (self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node,
            1,
        )

    @property
    def f_gpus_in_node(self) -> int:
        """Number of F-GPUs within a single node."""
        return min(self._n_f_workers, self._gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._dispatch_mode != "f_side_routing":
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
    Only needed under ``dispatch_mode="f_side_routing"``; returns 0 under
    a_side_routing (each F-GPU sends its results straight back to the
    A-rank) and when the node has only 1 F-GPU.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        num_experts: int = 0,
        topk: int = 0,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        dispatch_mode: str = "f_side_routing",
    ) -> None:
        super().__init__(name, scale_factor)
        if dispatch_mode not in AFD_DISPATCH_MODES:
            raise ValueError(
                f"AFDFReduceScatter: dispatch_mode must be one of {AFD_DISPATCH_MODES}, got {dispatch_mode!r}"
            )
        self._dispatch_mode = dispatch_mode
        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._weights = 0.0

    @property
    def num_f_nodes(self) -> int:
        return max((self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node, 1)

    @property
    def f_gpus_in_node(self) -> int:
        """Number of F-GPUs within a single node."""
        return min(self._n_f_workers, self._gpus_per_node)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        f_local = self.f_gpus_in_node
        if f_local <= 1 or self._dispatch_mode != "f_side_routing":
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

    Each A-rank sums the partial results coming back from the F-side EP
    partitions; every partial carries the full ``hidden_size`` per token.
    Partial count: all ``f_moe_ep_size`` partitions under f_side_routing,
    but only the ranks a token was actually sent to under a_side_routing
    (:func:`afd_dest_ep_ranks`). Returns 0 for dense FFN.
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
        dispatch_mode: str = "f_side_routing",
        num_experts: int = 0,
        topk: int = 0,
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = int(hidden_size)
        self._tp_a = max(int(tp_a), 1)
        self._f_moe_ep_size = max(int(f_moe_ep_size), 1)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._dispatch_mode = dispatch_mode
        self._num_experts = max(int(num_experts), 0)
        self._topk = max(int(topk), 0)
        self._weights = 0.0

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        # How many partials come back per token (see class docstring).
        if self._dispatch_mode == "a_side_routing":
            partials = afd_dest_ep_ranks(self._num_experts, self._topk, self._f_moe_ep_size)
        else:
            partials = float(self._f_moe_ep_size)
        if partials <= 1.0:  # nothing to reduce (dense FFN / single destination)
            return PerformanceResult(0.0, 0.0, source="empirical")
        x = int(kwargs.get("x", 0))
        if x <= 0:
            return PerformanceResult(0.0, 0.0, source="empirical")
        tokens_per_a_rank = (x + self._tp_a - 1) // self._tp_a
        bpe = self._comm_quant_mode.value.memory
        total_bytes = int((partials + 1) * tokens_per_a_rank * self._hidden_size * bpe)
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
