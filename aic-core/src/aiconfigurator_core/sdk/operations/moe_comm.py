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

The module also owns the large-EP compute side of the same family:
``load_moe_ep_data`` loads the unified ``moe_ep_perf.parquet`` EP MoE compute
table (with legacy sglang/trtllm wideep adapters) into one nested dict keyed
by ``[kernel_source][quant][distribution][inference_phase][topk][num_experts]
[num_slots][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens]``.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from aiconfigurator_core.sdk import common
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


def _adapt_legacy_deepep(data: defaultdict, rows, *, comm_backend: str, phase_columns: dict) -> None:
    """Adapt legacy sglang DeepEP rows (normal or ll table) into ``data``.

    One legacy row becomes one dispatch row + one combine row; each phase
    latency is the sum of its ``phase_columns`` entries in **microseconds**
    (the legacy query path divides by 1000), stored as ms. ``comm_dtype`` is
    ``"default"`` and ``ep_size = node_num * 8`` — the legacy tables were
    collected on 8-GPU HGX fleets with no dtype axis. HT rows keep their
    ``dispatch_sms`` budget for both phases (the legacy table keys the whole
    row by it); LL rows have no SM budget -> 0.
    """
    for row in rows:
        node_num = int(row["node_num"])
        sms = int(row["dispatch_sms"]) if comm_backend == "deepep_ht" else 0
        power = float(row.get("power", 0.0))
        for phase, columns in phase_columns.items():
            latency_us = 0.0
            for column in columns:
                latency_us += float(row[column])
            latency = latency_us / 1000.0  # us -> ms
            key = (
                comm_backend,
                phase,
                "default",
                node_num * 8,
                node_num,
                int(row["hidden_size"]),
                int(row["num_topk"]),
                int(row["num_experts"]),
                sms,
                int(row["num_token"]),
            )
            leaf = {"latency": latency, "power": power, "energy": power * latency}
            _store_a2a_leaf(data, key, leaf, overwrite=False)


def _adapt_legacy_deepep_normal(data: defaultdict, rows) -> None:
    _adapt_legacy_deepep(
        data,
        rows,
        comm_backend="deepep_ht",
        phase_columns={
            "dispatch": ("dispatch_transmit_us", "dispatch_notify_us"),
            "combine": ("combine_transmit_us", "combine_notify_us"),
        },
    )


def _adapt_legacy_deepep_ll(data: defaultdict, rows) -> None:
    # The legacy LL table never had the four-way transmit/notify split — only
    # per-phase averages (see load_wideep_deepep_ll_data).
    _adapt_legacy_deepep(
        data,
        rows,
        comm_backend="deepep_ll",
        phase_columns={"dispatch": ("dispatch_avg_t_us",), "combine": ("combine_avg_t_us",)},
    )


_LEGACY_TRTLLM_KERNEL_TO_BACKEND = {
    "NVLinkTwoSided": "nvlink_two_sided",
    "NVLinkOneSided": "nvlink_one_sided",
}

# op_name -> (phase, comm_dtype); None means the row's ``moe_dtype`` passes
# through. comm_dtype is the table's dtype axis: the run's moe_dtype for
# prepare/dispatch/standard combine (dispatch payload == run dtype physically;
# standard-combine payload is always bf16 but is keyed by run dtype so every
# legacy leaf maps 1:1, losslessly), and "fp4" for the low-precision combine
# kernel (distinct key — an nvfp4 run's standard combine keys as "nvfp4").
_LEGACY_TRTLLM_OP_TO_PHASE_DTYPE = {
    "alltoall_prepare": ("prepare", None),
    "alltoall_dispatch": ("dispatch", None),
    "alltoall_combine": ("combine", None),
    "alltoall_combine_low_precision": ("combine", "fp4"),
}


def _adapt_legacy_trtllm_alltoall(data: defaultdict, rows) -> None:
    """Adapt legacy trtllm ``trtllm_alltoall_perf`` rows into ``data``.

    UNITS: the legacy ``latency`` column is already in **milliseconds** —
    ``load_trtllm_alltoall_data`` stores it raw and ``query_trtllm_alltoall``
    returns table values without the /1000 the DeepEP query path applies (its
    SOL tier computes ms directly; shipped gb200 values span ~0.01-17 ms).
    Stored raw here, no us->ms conversion.

    ``node_num``: the legacy GB200 NVL4 files carry no ``num_nodes`` column,
    so it is derived as ``max(1, moe_ep_size // 4)`` — here once, mirroring
    ``load_trtllm_alltoall_data``, and never anywhere else; an explicit
    ``num_nodes`` column wins when present, also mirroring the legacy loader.
    """
    for row in rows:
        kernel_source = row.get("kernel_source", "NVLinkTwoSided")
        comm_backend = _LEGACY_TRTLLM_KERNEL_TO_BACKEND.get(kernel_source)
        phase_dtype = _LEGACY_TRTLLM_OP_TO_PHASE_DTYPE.get(row["op_name"])
        if comm_backend is None or phase_dtype is None:
            logger.debug(
                "skipping legacy trtllm_alltoall row with no unified mapping: "
                f"kernel_source={kernel_source} op_name={row['op_name']}"
            )
            continue
        phase, comm_dtype = phase_dtype
        if comm_dtype is None:
            comm_dtype = row["moe_dtype"]
        ep_size = int(row["moe_ep_size"])
        node_num = int(row["num_nodes"]) if "num_nodes" in row else max(1, ep_size // 4)
        latency = float(row["latency"])  # already ms — see docstring
        power = float(row.get("power", 0.0))
        key = (
            comm_backend,
            phase,
            comm_dtype,
            ep_size,
            node_num,
            int(row["hidden_size"]),
            int(row["topk"]),
            int(row["num_experts"]),
            0,  # legacy alltoall rows carry no SM budget
            int(row["num_tokens"]),
        )
        leaf = {"latency": latency, "power": power, "energy": power * latency}
        _store_a2a_leaf(data, key, leaf, overwrite=False)


def _load_legacy_a2a(
    data: defaultdict,
    legacy_normal_sources,
    legacy_ll_sources,
    legacy_trtllm_alltoall_sources,
) -> bool:
    """Adapt legacy per-backend comm tables into the unified ``data`` store.

    Mapping (spec §4.1): sglang ``wideep_deepep_normal_perf`` -> ``deepep_ht``,
    ``wideep_deepep_ll_perf`` -> ``deepep_ll``, trtllm ``trtllm_alltoall_perf``
    -> ``nvlink_two_sided``/``nvlink_one_sided``. All adapters store with
    ``overwrite=False`` (intra-source keep-first), so a later new-schema row
    can still take precedence via its overwrite path. Returns True when at
    least one legacy source exists — an existing-but-empty file counts, the
    same exists-but-empty semantic the new-schema path has.
    """
    loaded = False
    for sources, adapt in (
        (legacy_normal_sources, _adapt_legacy_deepep_normal),
        (legacy_ll_sources, _adapt_legacy_deepep_ll),
        (legacy_trtllm_alltoall_sources, _adapt_legacy_trtllm_alltoall),
    ):
        if sources is None:
            continue
        rows = _read_filtered_rows(sources)
        if rows is None:
            continue
        loaded = True
        adapt(data, rows)
    return loaded


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


# ---------------------------------------------------------------------------
# EP MoE compute (moe_ep_perf.parquet) — same family, compute side
# ---------------------------------------------------------------------------


def _moe_ep_store() -> defaultdict:
    """Empty moe_ep store: 11 auto-vivifying levels over a token->leaf dict.

    Key order: ``[kernel_source][quant][distribution][inference_phase][topk]
    [num_experts][num_slots][hidden_size][inter_size][moe_tp_size]
    [moe_ep_size]`` -> ``{num_tokens: leaf}``. ``quant`` is a
    :class:`common.MoEQuantMode` member, matching the sibling MoE loaders.
    """
    return defaultdict(  # kernel_source
        lambda: defaultdict(  # quant
            lambda: defaultdict(  # distribution
                lambda: defaultdict(  # inference_phase
                    lambda: defaultdict(  # topk
                        lambda: defaultdict(  # num_experts
                            lambda: defaultdict(  # num_slots
                                lambda: defaultdict(  # hidden_size
                                    lambda: defaultdict(  # inter_size
                                        lambda: defaultdict(  # moe_tp_size
                                            lambda: defaultdict(dict)  # moe_ep_size -> {num_tokens: leaf}
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )


def _store_ep_leaf(data: defaultdict, key: tuple, leaf: dict, *, overwrite: bool) -> None:
    """Store one ``{"latency", "power", "energy"}`` leaf under the 12-part key.

    ``key`` is ``(kernel_source, quant, distribution, inference_phase, topk,
    num_experts, num_slots, hidden_size, inter_size, moe_tp_size, moe_ep_size,
    num_tokens)``. ``overwrite=False`` keeps the first-stored leaf on a
    collision (debug log) — the intra-source convention new-schema rows
    follow. ``overwrite=True`` replaces whatever is there — used by the
    legacy adapters (their oracles assign unconditionally, so the last legacy
    row wins) and by the first new-schema occurrence of a key to take
    precedence over legacy-adapted rows.
    """
    *outer_key, num_tokens = key
    bucket = data
    for part in outer_key:
        bucket = bucket[part]
    if num_tokens in bucket and not overwrite:
        logger.debug("value conflict in moe_ep data: %s", " ".join(str(part) for part in key))
        return
    bucket[num_tokens] = leaf


def _adapt_legacy_sglang_wideep_moe(data: defaultdict, rows, *, inference_phase: str) -> None:
    """Adapt legacy sglang ``wideep_{context,generation}_moe_perf`` rows.

    Mirrors ``load_wideep_context_moe_data`` / ``load_wideep_generation_moe_data``
    (the oracles): straight ``MoEQuantMode[moe_dtype]`` with no
    kernel-source-based quant rerouting (unlike ``load_moe_data``), and
    unconditional assignment — the last row wins on an intra-file key
    collision (``overwrite=True``). ``kernel_source`` is pinned to
    ``"deepep_moe"`` (spec §4.2; the legacy column spells it ``deepepmoe``
    and the oracles never read it), ``num_slots = num_experts`` (the legacy
    sglang tables have no EPLB redundancy axis), and ``inference_phase``
    comes from which kwarg carried the source file.
    """
    for row in rows:
        latency = float(row["latency"])
        power = float(row.get("power", 0.0))
        num_experts = int(row["num_experts"])
        key = (
            "deepep_moe",
            common.MoEQuantMode[row["moe_dtype"]],
            row["distribution"],
            inference_phase,
            int(row["topk"]),
            num_experts,
            num_experts,  # num_slots = num_experts
            int(row["hidden_size"]),
            int(row["inter_size"]),
            int(row["moe_tp_size"]),
            int(row["moe_ep_size"]),
            int(row["num_tokens"]),
        )
        _store_ep_leaf(data, key, {"latency": latency, "power": power, "energy": power * latency}, overwrite=True)


def _adapt_legacy_sglang_context_moe(data: defaultdict, rows) -> None:
    _adapt_legacy_sglang_wideep_moe(data, rows, inference_phase="context")


def _adapt_legacy_sglang_generation_moe(data: defaultdict, rows) -> None:
    _adapt_legacy_sglang_wideep_moe(data, rows, inference_phase="generation")


def _adapt_legacy_trtllm_wideep_moe(data: defaultdict, rows) -> None:
    """Adapt legacy trtllm ``wideep_moe_perf`` rows.

    Mirrors ``load_wideep_moe_compute_data`` (the oracle): native
    ``kernel_source`` (``"moe_torch_flow"`` when the column is absent),
    ``num_slots`` and ``_eplb`` distributions pass through unchanged, no
    quant rerouting, unconditional assignment (``overwrite=True``, last row
    wins). The legacy table has no context/generation split — one kernel
    measured across the token range — so each row is registered under BOTH
    ``inference_phase`` values with identical (but independent) leaves.
    """
    for row in rows:
        latency = float(row["latency"])
        power = float(row.get("power", 0.0))
        base_key = (
            row.get("kernel_source", "moe_torch_flow"),
            common.MoEQuantMode[row["moe_dtype"]],
            row["distribution"],
        )
        shape_key = (
            int(row["topk"]),
            int(row["num_experts"]),
            int(row["num_slots"]),
            int(row["hidden_size"]),
            int(row["inter_size"]),
            int(row["moe_tp_size"]),
            int(row["moe_ep_size"]),
            int(row["num_tokens"]),
        )
        for inference_phase in ("context", "generation"):
            leaf = {"latency": latency, "power": power, "energy": power * latency}
            _store_ep_leaf(data, (*base_key, inference_phase, *shape_key), leaf, overwrite=True)


def _load_legacy_ep(
    data: defaultdict,
    legacy_context_sources,
    legacy_generation_sources,
    legacy_trtllm_wideep_sources,
) -> bool:
    """Adapt legacy wideep compute tables into the unified ``data`` store.

    Mapping (spec §4.2): sglang ``wideep_context_moe_perf`` /
    ``wideep_generation_moe_perf`` -> ``kernel_source="deepep_moe"`` with the
    inference phase set per source kwarg; trtllm ``wideep_moe_perf`` -> native
    kernel sources, registered under both phases.

    UNITS: the legacy ``latency`` column is already in **milliseconds** — the
    oracle loaders store it raw and their query paths feed it through the same
    ``perf_interp`` machinery as the regular (ms) moe table with no /1000
    anywhere; shipped values span ~0.03-62 ms, physically sensible for MoE
    compute. Stored raw here, no unit conversion — bit-exact equality with the
    oracles is pinned by the shipped-data equivalence sweeps.

    Returns True when at least one legacy source exists — an
    existing-but-empty file counts, the same exists-but-empty semantic the
    new-schema path has.
    """
    loaded = False
    for sources, adapt in (
        (legacy_context_sources, _adapt_legacy_sglang_context_moe),
        (legacy_generation_sources, _adapt_legacy_sglang_generation_moe),
        (legacy_trtllm_wideep_sources, _adapt_legacy_trtllm_wideep_moe),
    ):
        if sources is None:
            continue
        rows = _read_filtered_rows(sources)
        if rows is None:
            continue
        loaded = True
        adapt(data, rows)
    return loaded


def load_moe_ep_data(
    sources,
    legacy_context_sources=None,
    legacy_generation_sources=None,
    legacy_trtllm_wideep_sources=None,
) -> dict | None:
    """Load the unified EP MoE compute table (``moe_ep_perf.parquet``).

    ``sources`` is the new-schema source list (``(path, kernel_source_filter)``
    tuples, or a single path) read via ``_read_filtered_rows``; the three
    ``legacy_*_sources`` feed the legacy wideep adapters
    (:func:`_load_legacy_ep`). Legacy rows load first; a new-schema row
    overwrites a legacy leaf at the same key, while collisions **within** the
    new schema keep the first row (debug log) like every sibling loader.

    Returns:
        dict: ``[kernel_source][quant][distribution][inference_phase][topk]
        [num_experts][num_slots][hidden_size][inter_size][moe_tp_size]
        [moe_ep_size][num_tokens]`` -> dict with ``latency`` (ms — the column
        is already in milliseconds, unlike the us-collected a2a table),
        ``power`` (W) and ``energy`` (W·ms) keys. ``quant`` is a
        :class:`common.MoEQuantMode` member; ``inference_phase`` is stored as
        collected (``context``/``generation``); validation happens at query
        time. ``None`` when no source loaded anything.
    """
    data = _moe_ep_store()
    legacy_loaded = _load_legacy_ep(
        data, legacy_context_sources, legacy_generation_sources, legacy_trtllm_wideep_sources
    )

    rows = _read_filtered_rows(sources)
    if rows is None and not legacy_loaded:
        logger.debug(f"MoE EP data sources {sources} not found.")
        return None
    rows = rows or []

    # Check if the power column exists (optional in the schema)
    has_power = len(rows) > 0 and "power" in rows[0]
    if len(rows) > 0 and not has_power:
        logger.debug("moe_ep data has no power column - power will default to 0.0")

    seen: set[tuple] = set()
    for row in rows:
        key = (
            row["kernel_source"],
            common.MoEQuantMode[row["moe_dtype"]],
            row["distribution"],
            row["inference_phase"],  # stored as collected; validated at query time
            int(row["topk"]),
            int(row["num_experts"]),
            int(row["num_slots"]),
            int(row["hidden_size"]),
            int(row["inter_size"]),
            int(row["moe_tp_size"]),
            int(row["moe_ep_size"]),
            int(row["num_tokens"]),
        )
        latency = float(row["latency"])  # already ms (spec §4.2) — stored raw
        power = float(row.get("power", 0.0))
        energy = power * latency  # watt-milliseconds

        # The first new-schema occurrence of a key overwrites any
        # legacy-adapted leaf; repeats fall into the helper's keep-first path.
        first_occurrence = key not in seen
        seen.add(key)
        _store_ep_leaf(
            data,
            key,
            {"latency": latency, "power": power, "energy": energy},
            overwrite=first_occurrence,
        )

    return data
