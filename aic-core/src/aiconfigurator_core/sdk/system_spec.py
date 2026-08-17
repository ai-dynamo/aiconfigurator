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
        enough here: ``p2p_latency`` is measured within a scale-up domain
        (node or rack), while crossing racks adds the scale-out fabric's
        round trip.

        - ``num_gpus <= num_gpus_per_rack``: ``p2p_latency``
        - ``num_gpus > num_gpus_per_rack``: ``inter_rack_latency``, falling
          back to ``p2p_latency`` when unset

        Systems without a declared rack tier (``num_gpus_per_rack`` absent)
        therefore always return ``p2p_latency``, matching the pre-rack
        behavior.
        """
        node_spec = self["node"]
        num_gpus_per_rack = node_spec.get("num_gpus_per_rack", float("inf"))
        p2p_latency = node_spec["p2p_latency"]

        if num_gpus <= num_gpus_per_rack:
            return p2p_latency
        return node_spec.get("inter_rack_latency", p2p_latency)
