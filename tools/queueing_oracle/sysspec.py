# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire AIC system specs into the oracle's KV-transfer model.

The AIC system spec (``src/aiconfigurator/systems/<system>.yaml``) carries
per-GPU single-direction link bandwidths in Byte/s::

    node:
      inter_node_bw: 50000000000   # e.g. 1:1 CX7 NDR 400Gb/s
      intra_node_bw: 450000000000  # e.g. NVLink

``transfer_spec_from_system`` maps those onto a ``TransferSpec``. The
per-GPU convention keeps the arithmetic rank-local: with TP sharding, each
rank transfers its KV shard through its own NIC in parallel, so
(per-rank bytes / per-GPU bandwidth) equals (worker bytes / worker
bandwidth) and ``kv_bytes_per_token`` should likewise be the per-rank
value (e.g. ``model.get_kvcache_bytes_per_sequence(1)`` from the SDK).

``bw_efficiency`` de-rates nominal line rate to what transfers actually
achieve; the 0.8 default mirrors the spec's own
``mem_bw_empirical_scaling_factor`` convention. Calibrate per fabric when
measurements are available.
"""

from __future__ import annotations

from vllm_sim import TransferSpec


def load_system_spec(system: str) -> dict:
    """Load the AIC system spec yaml for `system` (requires the
    aiconfigurator_core package + PyYAML; the oracle core itself stays
    stdlib-only)."""
    from importlib import resources

    import yaml

    path = resources.files("aiconfigurator_core") / "systems" / f"{system}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def transfer_spec_from_system(
    system_spec: dict | str,
    kv_bytes_per_token: int,
    *,
    cross_node: bool = True,
    bw_efficiency: float = 0.8,
) -> TransferSpec:
    """Build a ``TransferSpec`` from an AIC system spec (dict or system
    name). ``cross_node=True`` uses ``node.inter_node_bw`` (the common
    disagg placement: prefill and decode pools on different nodes);
    ``False`` uses ``node.intra_node_bw``."""
    if isinstance(system_spec, str):
        system_spec = load_system_spec(system_spec)
    node = system_spec["node"]
    link = float(node["inter_node_bw"] if cross_node else node["intra_node_bw"])
    return TransferSpec(
        kv_bytes_per_token=kv_bytes_per_token,
        egress_bytes_per_s=link,
        ingress_bytes_per_s=link,
        bw_efficiency=bw_efficiency,
    )
