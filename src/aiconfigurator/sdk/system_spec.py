# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
SystemSpec — GPU constants, bandwidth, SM count, node topology.

Extracted from the ``system_spec`` YAML dict loaded in ``PerfDatabase.__init__``.
Subclasses ``dict`` so that ``isinstance(spec, dict)`` is ``True`` and existing
code like ``spec["gpu"]["mem_bw"]`` keeps working transparently.
"""

from __future__ import annotations


class SystemSpec(dict):
    """Immutable-ish representation of the hardware system specification.

    Inherits from ``dict`` to ensure full backward compatibility — existing code
    that does ``system_spec["gpu"]["mem_bw"]`` or ``isinstance(spec, dict)``
    continues to work unchanged.

    Structured attributes are populated by :meth:`from_yaml_dict` and provide a
    typed, dotted-access alternative (e.g. ``spec.mem_bw``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml_dict(cls, spec: dict) -> SystemSpec:
        """Build a *SystemSpec* from the raw YAML dict.

        The instance *is* the dict (via ``dict`` inheritance), so ``spec["gpu"]``
        returns exactly the same sub-dict as before.
        """
        instance = cls(spec)  # populates the dict portion

        gpu = spec.get("gpu", {})
        node = spec.get("node", {})
        misc = spec.get("misc", {})

        # --- gpu section ---
        instance.float16_tc_flops = gpu.get("float16_tc_flops", 0.0)
        instance.mem_bw = gpu.get("mem_bw", 0.0)
        instance.mem_bw_empirical_scaling_factor = gpu.get("mem_bw_empirical_scaling_factor", 1.0)
        instance.mem_empirical_constant_latency = gpu.get("mem_empirical_constant_latency", 0.0)
        instance.sm_version = gpu.get("sm_version", 0)

        # --- node section ---
        instance.intra_node_bw = node.get("intra_node_bw", 0.0)
        instance.inter_node_bw = node.get("inter_node_bw", 0.0)
        instance.num_gpus_per_node = node.get("num_gpus_per_node", 8)
        instance.num_gpus_per_rack = node.get("num_gpus_per_rack", 0)
        instance.inter_rack_bw = node.get("inter_rack_bw", 0.0)
        instance.p2p_latency = node.get("p2p_latency", 0.0)

        # --- misc ---
        instance.data_dir = spec.get("data_dir", "")
        instance.nccl_version = misc.get("nccl_version", "")

        return instance

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_p2p_bandwidth(self, num_gpus: int) -> float:
        """Return point-to-point bandwidth (bytes/s) based on topology.

        Three-tier selection:

        - *num_gpus <= num_gpus_per_node*: ``intra_node_bw`` (NVLink within node)
        - *num_gpus <= num_gpus_per_rack*: ``inter_node_bw`` (NVSwitch within rack)
        - *num_gpus > num_gpus_per_rack*: ``inter_rack_bw`` (InfiniBand between racks)
        """
        gpus_per_rack = self.num_gpus_per_rack if self.num_gpus_per_rack > 0 else float("inf")

        if num_gpus <= self.num_gpus_per_node:
            return self.intra_node_bw
        elif num_gpus <= gpus_per_rack:
            return self.inter_node_bw
        else:
            # Inter-rack communication, fallback to inter_node_bw if inter_rack_bw not defined
            return self.inter_rack_bw if self.inter_rack_bw else self.inter_node_bw
