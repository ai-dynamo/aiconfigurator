# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFDTransfer op — cross-pool and intra-pool communication for AFD serving."""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Optional

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import Operation

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


def _afd_selective_send_prob(num_experts: int, top_k: int, num_f_nodes: int) -> float:
    """Probability that a given F-node holds >= 1 of a token's top-k experts.

    Formula: ``P_send = 1 - C(E - E/Nf, k) / C(E, k)``.

    Under MoE selective-send, a token's hidden state must cross to an
    F-node if and only if that F-node hosts at least one of the token's
    top-k experts.  When triggered, the *full* hidden state crosses --
    you cannot fractionally dispatch a token -- so the per-link
    payload scales with ``P_send`` rather than the looser ``topk/E``
    average-fraction approximation.

    Returns 1.0 for dense / single-node / degenerate configs (fall back to
    full broadcast).
    """
    if num_experts <= 0 or top_k <= 0 or num_f_nodes <= 1:
        return 1.0
    experts_per_node = num_experts // num_f_nodes
    if experts_per_node <= 0:
        return 1.0

    n_other = num_experts - experts_per_node
    if top_k > n_other:
        return 1.0
    return 1.0 - comb(n_other, top_k) / comb(num_experts, top_k)


class AFDTransfer(Operation):
    """Per-layer breakdown of all AFD cross-pool and intra-pool communication.

    Consolidates four conceptual ops behind one ``query()``:

      * A→F cross-pool dispatch (one-direction per-layer DMA)
      * F→A cross-pool return (symmetric: same per-link payload)
      * F-node intra-node AllGather + ReduceScatter (only under
        ``rank_mapping == "one_to_one"`` and ``tp_f > 1``)
      * A-side cross-EP combine reduce (only when ``f_moe_ep_size > 1``)

    ``query()`` returns a dict ::

        {
            "t_a2f": {"afd_transfer_a2f": float},
            "t_f2a": {"afd_transfer_f2a": float},
            "t_a":   {"afd_combine": float},
            "t_f":   {"afd_f_allgather": float,
                      "afd_f_reduce_scatter": float},
        }

    where each sub-dict maps op label → per-layer latency (ms).  The
    caller adds each sub-dict's values to the corresponding pool's
    per-layer cost (``t_a_layer`` / ``t_f_layer``) or pipeline stage
    (``t_a2f_layer`` / ``t_f2a_layer``).

    Transfer modes
    --------------
    ``"p2p"`` (default):
      Full hidden activations sent to all F-nodes.  Per-link payload =
      ``b_total * H * bpe / num_f_nodes``.
    ``"moe_selective"``:
      A token only crosses to F-nodes that host one of its top-k experts.
      Per-link payload = ``P_send * b_total * H * bpe`` with
      ``P_send = 1 - C(E - E/Nf, k) / C(E, k)`` -- a token is dispatched
      in full to an F-node if and only if that F-node hosts at least one
      of its experts, so the per-link payload scales with that
      probability rather than the looser ``topk/E`` average-fraction
      approximation.

    Rank-mapping topologies
    -----------------------
    ``"one_to_one"`` (default, **implemented**):
      A node-local A-rank dispatches to exactly one F-rank within an
      F-node (the rank-aligned slot).  Multiple A-ranks may share the
      same F-rank slot.  F-side intra-node AG/RS are required to expose
      the full token batch to ``tp_f`` ranks for TP MoE/FFN.
    ``"broadcast"`` (**stub, not yet modeled**):
      Each A-rank fans out to all ``tp_f`` F-ranks within an F-node, so
      F-side AG/RS are unnecessary.  Currently returns 0 for the
      intra-pool collectives; the cross-pool transfer formula is kept
      identical to the 1:1 case pending future modeling of the per-NIC
      fan-out factor.

    Note on ``b_total`` semantics
    -----------------------------
    ``query(b_total=...)`` expects the **total token volume** the A-pool
    sees per step, i.e. ``n_a_workers * a_batch_size * tokens_per_req``.
    In prefill ``tokens_per_req == isl``; in decode it is 1.  Callers
    must pass token volume (not request count) so per-link byte size is
    correctly scaled by sequence length.
    """

    _VALID_TRANSFER_MODES = ("p2p", "moe_selective")
    _VALID_RANK_MAPPINGS = ("one_to_one", "broadcast")

    def __init__(
        self,
        name: str,
        hidden_size: int,
        n_a_workers: int,
        n_f_workers: int,
        gpus_per_node: int = 8,
        tp_a: int = 1,
        tp_f: int = 1,
        f_moe_ep_size: int = 1,
        topk: int = 1,
        num_experts: int = 1,
        comm_quant_mode: Optional[common.CommQuantMode] = None,
        comm_overhead_factor: float = 1.0,
        transfer_mode: str = "p2p",
        rank_mapping: str = "one_to_one",
    ) -> None:
        super().__init__(name, 1.0)
        if transfer_mode not in self._VALID_TRANSFER_MODES:
            raise ValueError(
                f"AFDTransfer: transfer_mode must be one of "
                f"{self._VALID_TRANSFER_MODES}, got {transfer_mode!r}"
            )
        if rank_mapping not in self._VALID_RANK_MAPPINGS:
            raise ValueError(
                f"AFDTransfer: rank_mapping must be one of "
                f"{self._VALID_RANK_MAPPINGS}, got {rank_mapping!r}"
            )

        self._hidden_size = int(hidden_size)
        self._n_a_workers = max(int(n_a_workers), 1)
        self._n_f_workers = max(int(n_f_workers), 1)
        self._gpus_per_node = max(int(gpus_per_node), 1)
        self._tp_a = max(int(tp_a), 1)
        self._tp_f = max(int(tp_f), 1)
        self._f_moe_ep_size = max(int(f_moe_ep_size), 1)
        self._topk = max(int(topk), 0)
        self._num_experts = max(int(num_experts), 0)
        self._comm_quant_mode = comm_quant_mode or common.CommQuantMode.half
        self._comm_overhead_factor = float(comm_overhead_factor or 1.0)
        self._transfer_mode = transfer_mode
        self._rank_mapping = rank_mapping
        self._weights = 0.0

    @property
    def num_f_nodes(self) -> int:
        """Physical F-node count: ``ceil(n_f_workers / gpus_per_node)``."""
        return max(
            (self._n_f_workers + self._gpus_per_node - 1) // self._gpus_per_node,
            1,
        )

    @property
    def is_moe_selective(self) -> bool:
        return self._transfer_mode == "moe_selective" and self._f_moe_ep_size > 1

    def _tokens_per_f_node(self, b_total: int) -> float:
        """Expected tokens visible to one F-node after A→F dispatch."""
        nf = self.num_f_nodes
        if self.is_moe_selective:
            send_prob = _afd_selective_send_prob(self._num_experts, self._topk, nf)
            return b_total * send_prob
        return b_total / max(nf, 1)

    def _cross_pool_one_direction_ms(
        self, database: PerfDatabase, b_total: int
    ) -> float:
        """One-direction A→F (or F→A) per-layer cross-pool DMA latency, ms."""
        bpe = self._comm_quant_mode.value.memory
        nf = self.num_f_nodes
        if self.is_moe_selective:
            send_prob = _afd_selective_send_prob(self._num_experts, self._topk, nf)
            msg_bytes = int(send_prob * b_total * self._hidden_size * bpe)
        else:
            msg_bytes = int(b_total * self._hidden_size * bpe / max(nf, 1))
        if msg_bytes <= 0:
            return 0.0
        result = database.query_p2p(msg_bytes)
        return float(result) * self._comm_overhead_factor

    def _f_collective_ms(
        self, database: PerfDatabase, op_name: str, b_total: int
    ) -> float:
        """F-node intra-node collective latency, ms."""
        if self._tp_f <= 1 or self._rank_mapping != "one_to_one":
            return 0.0
        tokens = self._tokens_per_f_node(b_total)
        message_size = int(tokens * self._hidden_size)
        if message_size <= 0:
            return 0.0
        result = database.query_nccl(
            self._comm_quant_mode, self._tp_f, op_name, message_size
        )
        return float(result)

    def _a_combine_ms(self, database: PerfDatabase, a_batch_size: int) -> float:
        """A-side cross-EP combine latency, ms."""
        if self._f_moe_ep_size <= 1:
            return 0.0
        bpe = self._comm_quant_mode.value.memory
        tokens_per_a_rank = max(1, int(a_batch_size) // self._tp_a)
        total_bytes = int(
            (self._f_moe_ep_size + 1)
            * tokens_per_a_rank
            * self._hidden_size
            * bpe
        )
        if total_bytes <= 0:
            return 0.0
        result = database.query_mem_op(total_bytes)
        return float(result)

    def query(
        self, database: PerfDatabase, **kwargs
    ) -> dict[str, dict[str, float]]:
        """Return per-layer latency breakdown for AFD comm and collectives.

        Required kwargs:
            ``b_total`` -- total **token volume** seen by the A-pool per
            step (``n_a_workers * a_batch_size * tokens_per_req``).
            ``a_batch_size`` -- per-A-Worker token volume
            (``a_batch_size * tokens_per_req``) for combine sizing.
            Defaults to ``b_total / n_a_workers`` if omitted.

        Returns:
            ``{"t_a2f": {...}, "t_f2a": {...}, "t_a": {...}, "t_f": {...}}``
            with each sub-dict mapping op label → per-layer latency (ms).
            ``t_a2f`` / ``t_f2a`` hold the *one-direction* latency only.
        """
        b_total = int(kwargs.get("b_total", 1) or 1)
        a_batch_size = kwargs.get("a_batch_size")
        if a_batch_size is None or a_batch_size <= 0:
            a_batch_size = max(b_total // self._n_a_workers, 1)
        a_batch_size = int(a_batch_size)

        one_dir_ms = self._cross_pool_one_direction_ms(database, b_total)
        ag_ms = self._f_collective_ms(database, "all_gather", b_total)
        rs_ms = self._f_collective_ms(database, "reduce_scatter", b_total)
        combine_ms = self._a_combine_ms(database, a_batch_size)

        return {
            "t_a2f": {"afd_transfer_a2f": one_dir_ms},
            "t_f2a": {"afd_transfer_f2a": one_dir_ms},
            "t_a": {"afd_combine": combine_ms},
            "t_f": {
                "afd_f_allgather": ag_ms,
                "afd_f_reduce_scatter": rs_ms,
            },
        }

    def get_weights(self, **kwargs):
        return self._weights
