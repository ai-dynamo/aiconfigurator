# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SystemSpec — hardware system spec loaded from a per-system YAML file.

Subclasses ``dict`` so existing code that does ``spec["gpu"]["mem_bw"]`` or
``isinstance(spec, dict)`` keeps working. ``get_p2p_bandwidth`` is the only
added method, replacing ``PerfDatabase._get_p2p_bandwidth``.
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
