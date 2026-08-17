# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SystemSpec — hardware system spec loaded from a per-system YAML file.

Subclasses ``dict`` so existing code that does ``spec["gpu"]["mem_bw"]`` or
``isinstance(spec, dict)`` keeps working. ``get_p2p_bandwidth`` and
``get_p2p_latency`` are the only added methods; the former replaces
``PerfDatabase._get_p2p_bandwidth``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class SystemSpec(dict):
    """Hardware system spec backed by the YAML dict.

    The dict is the single source of truth — there are no parallel structured
    attributes. Construct directly with ``SystemSpec(yaml_dict)``.
    """

    def get_p2p_bandwidth(self, num_gpus: int) -> float:
        """Return point-to-point bandwidth (bytes/s) based on topology.

        Three-tier selection:

        - ``num_gpus <= num_gpus_per_node``: ``intra_node_bw`` (NVLink within node)
        - ``num_gpus <= num_gpus_per_rack``: ``inter_node_bw`` (NVSwitch within rack)
        - ``num_gpus > num_gpus_per_rack``: ``inter_rack_bw`` (InfiniBand between racks),
          falling back to ``inter_node_bw`` when ``inter_rack_bw`` is unset.

        Raises ``KeyError`` for misconfigured specs that lack required keys —
        same loud-failure behavior as the original ``_get_p2p_bandwidth``.
        """
        node_spec = self["node"]
        num_gpus_per_node = node_spec["num_gpus_per_node"]
        num_gpus_per_rack = node_spec.get("num_gpus_per_rack", float("inf"))

        if num_gpus <= num_gpus_per_node:
            return node_spec["intra_node_bw"]
        if num_gpus <= num_gpus_per_rack:
            return node_spec["inter_node_bw"]
        return node_spec.get("inter_rack_bw", node_spec["inter_node_bw"])

    def get_p2p_latency(self, num_gpus: int) -> float:
        """Return point-to-point latency (seconds) based on topology.

        The latency counterpart of :meth:`get_p2p_bandwidth`. Two tiers are
        enough here: ``p2p_latency`` covers a scale-up domain (node or rack),
        while crossing racks pays the scale-out fabric's round trip instead.

        - ``num_gpus <= num_gpus_per_rack``: ``p2p_latency``
        - ``num_gpus > num_gpus_per_rack``: ``inter_rack_latency``, falling
          back to ``p2p_latency`` when unset or when it would report a speedup

        Systems without a declared rack tier (``num_gpus_per_rack`` absent)
        therefore always return ``p2p_latency``, matching the pre-rack
        behavior.

        Leaving one rack cannot be faster than staying inside it, so the
        cross-rack value is clamped at ``p2p_latency``. That guard is load-
        bearing rather than defensive: some shipped specs raise ``p2p_latency``
        by hand as a calibration knob (gb200/gb300 label theirs a "nonofficial
        correction") while ``inter_rack_latency`` keeps the textbook InfiniBand
        figure, which leaves the pair inverted. Nothing read
        ``inter_rack_latency`` before this method existed, so the inconsistency
        had never mattered.
        """
        node_spec = self["node"]
        num_gpus_per_rack = node_spec.get("num_gpus_per_rack", float("inf"))
        p2p_latency = node_spec["p2p_latency"]

        if num_gpus <= num_gpus_per_rack:
            return p2p_latency

        inter_rack_latency = node_spec.get("inter_rack_latency")
        if inter_rack_latency is None:
            return p2p_latency
        if inter_rack_latency < p2p_latency:
            logger.warning(
                "system spec node has inter_rack_latency (%g s) below p2p_latency (%g s), which "
                "would make crossing racks faster than staying in one; clamping to p2p_latency. "
                "One of the two values is wrong -- check the spec's node section.",
                inter_rack_latency,
                p2p_latency,
            )
            return p2p_latency
        return inter_rack_latency
