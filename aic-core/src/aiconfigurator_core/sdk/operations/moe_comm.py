# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Large-EP MoE communication family, unified across SGLang, vLLM, and TRT-LLM.

Models the all-to-all communication of large-scale expert-parallel MoE
(dispatch/combine, plus TRT-LLM's prepare phase) with one comm-backend
registry shared by all three inference backends. On TRT-LLM this covers the
*wideEP* path only — non-wideEP TRT-LLM paths are untouched.

``MOE_A2A_BACKENDS`` maps backend name to its :class:`MoECommBackendSpec`
(framework/phase applicability plus feasibility rules).
``load_moe_a2a_data`` loads the unified ``moe_a2a_perf.parquet`` comm table
(with legacy per-backend adapters) into one nested dict keyed by
``[comm_backend][phase][comm_dtype][ep_size][node_num][hidden_size][topk]
[num_experts][sms][num_tokens]``.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from aiconfigurator_core.sdk.operations.base import _read_filtered_rows

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoECommBackendSpec:
    """Static description of one MoE all-to-all comm backend."""

    name: str
    frameworks: tuple[str, ...]  # ("sglang", "vllm") or ("trtllm",)
    inference_phases: tuple[str, ...]  # ("context",) | ("generation",) | ("context", "generation")
    comm_phases: tuple[str, ...]  # ("dispatch", "combine") | ("prepare", "dispatch", "combine")
    min_sm: int = 0
    max_topk: int = 8

    def feasible(
        self,
        *,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        sm_version: int | None = None,
    ) -> bool:
        """Whether this backend can serve the given MoE parallelism config."""
        return (
            topk <= self.max_topk
            and moe_tp_size == 1
            and 1 < moe_ep_size <= num_experts
            and num_experts % moe_ep_size == 0
            and (sm_version is None or sm_version >= self.min_sm)
        )


MOE_A2A_BACKENDS: dict[str, MoECommBackendSpec] = {
    "deepep_ht": MoECommBackendSpec(
        name="deepep_ht",
        frameworks=("sglang", "vllm"),
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
    ),
    "deepep_ll": MoECommBackendSpec(
        name="deepep_ll",
        frameworks=("sglang", "vllm"),
        inference_phases=("generation",),
        comm_phases=("dispatch", "combine"),
    ),
    "nvlink_two_sided": MoECommBackendSpec(
        name="nvlink_two_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("prepare", "dispatch", "combine"),
        min_sm=100,
    ),
    "nvlink_one_sided": MoECommBackendSpec(
        name="nvlink_one_sided",
        frameworks=("trtllm",),
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
        min_sm=100,
    ),
}


def nodes_for(ep_size: int, num_gpus_per_node: int) -> int:
    """Node count needed to host ``ep_size`` EP ranks (ceil division)."""
    return -(-ep_size // num_gpus_per_node)


def _moe_a2a_store() -> defaultdict:
    """Empty moe_a2a store: 9 auto-vivifying levels over a token->leaf dict.

    Key order: ``[comm_backend][phase][comm_dtype][ep_size][node_num]
    [hidden_size][topk][num_experts][sms]`` -> ``{num_tokens: leaf}``.
    """
    return defaultdict(  # comm_backend
        lambda: defaultdict(  # phase
            lambda: defaultdict(  # comm_dtype
                lambda: defaultdict(  # ep_size
                    lambda: defaultdict(  # node_num
                        lambda: defaultdict(  # hidden_size
                            lambda: defaultdict(  # topk
                                lambda: defaultdict(  # num_experts
                                    lambda: defaultdict(dict)  # sms -> {num_tokens: leaf}
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _store_a2a_leaf(data: defaultdict, key: tuple, leaf: dict, *, overwrite: bool) -> None:
    """Store one ``{"latency", "power", "energy"}`` leaf under the 10-part key.

    ``key`` is ``(comm_backend, phase, comm_dtype, ep_size, node_num,
    hidden_size, topk, num_experts, sms, num_tokens)``. With
    ``overwrite=False`` a collision keeps the first-stored leaf and logs at
    debug level — the intra-source convention every sibling loader follows.
    ``overwrite=True`` replaces whatever is there — the path new-schema rows
    use to take precedence over legacy-adapted rows at the same key.
    """
    *outer_key, num_tokens = key
    bucket = data
    for part in outer_key:
        bucket = bucket[part]
    if num_tokens in bucket and not overwrite:
        logger.debug("value conflict in moe_a2a data: %s", " ".join(str(part) for part in key))
        return
    bucket[num_tokens] = leaf


def _normalize_sms(raw: object) -> int:
    """Normalize the ``sms`` column to an int key; null/NaN/absent -> 0.

    HT-mode rows carry an SM budget; LL-mode rows leave ``sms`` null (older
    files may omit the column entirely). Parquet nulls read back as ``""``
    through ``_read_perf_rows``; an absent column arrives as ``None``.
    """
    if raw is None or raw == "":
        return 0
    value = float(raw)
    return 0 if math.isnan(value) else int(value)


def _load_legacy_a2a(
    data: defaultdict,
    legacy_normal_sources,
    legacy_ll_sources,
    legacy_trtllm_alltoall_sources,
) -> bool:
    """Adapt legacy per-backend comm tables into the unified ``data`` store.

    Mapping (spec §4.1): sglang ``wideep_deepep_normal_perf`` -> ``deepep_ht``,
    ``wideep_deepep_ll_perf`` -> ``deepep_ll``, trtllm ``trtllm_alltoall_perf``
    -> ``nvlink_two_sided``/``nvlink_one_sided``. Returns True when at least
    one legacy source contributed rows. Stub for now — the adapters land with
    the legacy-compat task; the sources are accepted and ignored.
    """
    return False


def load_moe_a2a_data(
    sources,
    legacy_normal_sources=None,
    legacy_ll_sources=None,
    legacy_trtllm_alltoall_sources=None,
) -> dict | None:
    """Load the unified MoE all-to-all comm table (``moe_a2a_perf.parquet``).

    ``sources`` is the new-schema source list (``(path, kernel_source_filter)``
    tuples, or a single path) read via ``_read_filtered_rows``; the three
    ``legacy_*_sources`` feed the per-backend legacy adapters
    (:func:`_load_legacy_a2a`). Legacy rows load first; a new-schema row
    overwrites a legacy leaf at the same key, while collisions **within** the
    new schema keep the first row (debug log) like every sibling loader.

    Returns:
        dict: ``[comm_backend][phase][comm_dtype][ep_size][node_num]
        [hidden_size][topk][num_experts][sms][num_tokens]`` -> dict with
        ``latency`` (ms — the parquet column is in microseconds), ``power``
        (W) and ``energy`` (W·ms) keys. ``phase`` is stored as collected
        (``prepare``/``dispatch``/``combine``); validation happens at query
        time. ``None`` when no source loaded anything.
    """
    data = _moe_a2a_store()
    legacy_loaded = _load_legacy_a2a(data, legacy_normal_sources, legacy_ll_sources, legacy_trtllm_alltoall_sources)

    rows = _read_filtered_rows(sources)
    if rows is None and not legacy_loaded:
        logger.debug(f"MoE A2A data sources {sources} not found.")
        return None
    rows = rows or []

    # Check if the power column exists (optional in the schema)
    has_power = len(rows) > 0 and "power" in rows[0]
    if len(rows) > 0 and not has_power:
        logger.debug("moe_a2a data has no power column - power will default to 0.0")

    seen: set[tuple] = set()
    for row in rows:
        key = (
            row["comm_backend"],
            row["phase"],  # stored as collected; validated at query time
            row["comm_dtype"],
            int(row["ep_size"]),
            int(row["node_num"]),
            int(row["hidden_size"]),
            int(row["topk"]),
            int(row["num_experts"]),
            _normalize_sms(row.get("sms")),
            int(row["num_tokens"]),
        )
        latency = float(row["latency"]) / 1000.0  # collector records us; leaves are ms
        power = float(row.get("power", 0.0))
        energy = power * latency  # watt-milliseconds

        # The first new-schema occurrence of a key overwrites any
        # legacy-adapted leaf; repeats fall into the helper's keep-first path.
        first_occurrence = key not in seen
        seen.add(key)
        _store_a2a_leaf(
            data,
            key,
            {"latency": latency, "power": power, "energy": energy},
            overwrite=first_occurrence,
        )

    return data
