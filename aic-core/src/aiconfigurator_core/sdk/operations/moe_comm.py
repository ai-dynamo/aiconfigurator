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
``MoEAllToAll`` is the op class over that table: it owns the class-level
cache + ``load_data`` (the retired per-call lookup lived behind
``PerfDatabase.query_moe_a2a``).

Large-EP expert compute is modeled from stock ``moe_perf`` by
``operations.moe.ModeledEPMoE``; this module owns communication only.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


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
    "deepep_v2": MoECommBackendSpec(
        name="deepep_v2",
        frameworks=("vllm",),
        inference_phases=("context", "generation"),
        comm_phases=("dispatch", "combine"),
    ),
    "trtllm_deepep_ht": MoECommBackendSpec(
        name="trtllm_deepep_ht",
        frameworks=("trtllm",),
        inference_phases=("context",),
        comm_phases=("dispatch", "combine"),
    ),
    "trtllm_deepep_ll": MoECommBackendSpec(
        name="trtllm_deepep_ll",
        frameworks=("trtllm",),
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


def _row_power(row: dict) -> float:
    """Normalize optional ``power`` to watts; null/NaN/absent -> 0.0.

    A present-but-null power cell means "not measured", exactly like an absent
    column — but ``float(row.get("power", 0.0))`` raised ``ValueError`` on it:
    parquet nulls read back as ``""`` through ``_read_perf_rows``, and a NaN
    cell would silently poison ``energy``. ``log_perf`` freezes the CSV header
    from the first row, so within one collection run power columns are
    all-or-nothing per file and this cell is unreachable; null cells arrive
    through merged/legacy files, which every ``power`` read in this module
    must tolerate (same treatment as ``sms`` above).
    """
    raw = row.get("power")
    if raw is None or raw == "":
        return 0.0
    value = float(raw)
    if math.isnan(value):
        return 0.0
    if not math.isfinite(value):
        raise ValueError("non-finite power cell in perf data: power must be finite when measured")
    return value


def _require_latency(row: dict, table: str) -> float:
    """Read schema-required finite ``latency``; invalid values are corrupt.

    Unlike ``power`` (optional — see ``_row_power``), a latency-less row has
    no meaning: coercing null to 0.0 would silently poison every consumer, so
    the load refuses with a named error instead of the bare ``float("")``
    ValueError. New-schema loaders only — the legacy adapters keep their
    oracle loaders' bare ``float(row["latency"])`` behavior (parity).
    """
    raw = row.get("latency")
    if raw is None or raw == "":
        raise ValueError(
            f"null latency cell in a {table} row: latency is schema-required and must be finite; "
            "refusing to load corrupt perf data"
        )
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite latency cell in a {table} row: latency is schema-required and must be finite; "
            "refusing to load corrupt perf data"
        )
    return value


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
        power = _row_power(row)
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
    ``load_trtllm_alltoall_data`` stores it raw and the retired per-phase query
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
        power = _row_power(row)
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
        latency = _require_latency(row, "moe_a2a_perf") / 1000.0  # collector records us; leaves are ms
        power = _row_power(row)
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
# MoEAllToAll — the op class over the unified moe_a2a table
# ---------------------------------------------------------------------------


_A2A_PHASES = ("prepare", "dispatch", "combine")


def _validate_a2a_request(comm_backend: str, phase: str) -> None:
    """Shared ctor/query validation: unknown backend or phase is a ValueError.

    The per-backend check is a guard, not a live code path: the block builder
    iterates ``spec.comm_phases`` itself and the coverage probe walks table
    keys without validating — so a combination like ``("deepep_ht",
    "prepare")`` can only come from future misuse, and should fail here where
    the intent is expressed rather than later as a data miss.
    """
    if comm_backend not in MOE_A2A_BACKENDS:
        raise ValueError(f"Invalid comm_backend '{comm_backend}'. Must be one of {sorted(MOE_A2A_BACKENDS)}")
    if phase not in _A2A_PHASES:
        raise ValueError(f"Invalid phase '{phase}'. Must be one of {list(_A2A_PHASES)}")
    supported = MOE_A2A_BACKENDS[comm_backend].comm_phases
    if phase not in supported:
        raise ValueError(
            f"comm_backend '{comm_backend}' does not implement phase '{phase}'; supported: {list(supported)}"
        )


class MoEAllToAll(Operation):
    """Unified large-EP MoE all-to-all comm op (one phase per instance).

    Owns ``_moe_a2a_data`` — the unified comm table loaded by
    :func:`load_moe_a2a_data` (new-schema ``moe_a2a_perf.parquet`` plus the
    three legacy per-backend adapters). Loaded on every inference backend
    ({"sglang", "vllm", "trtllm"} all have legacy comm sources); ``None``
    otherwise. Comm ops see per-rank token counts: ``query(x=...)`` scales by
    ``scale_factor`` only — never by ``attention_dp_size``.
    ``attention_tp_size`` divides the token key before the lookup — legacy
    fidelity with ``MoEDispatch``'s ``num_tokens // self._scale_num_tokens``
    (plain floor division, no ``max(1, ...)`` guard). ``comm_dtype``
    resolves exact-first, then the legacy ``fp8_block`` -> ``fp8`` behavioral
    aliasing, then the sole-collected-dtype fallback (typed miss otherwise).
    """

    _data_cache: ClassVar[dict] = {}

    _SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = ("sglang", "vllm", "trtllm")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        phase: str,
        comm_backend: str,
        hidden_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        node_num: int,
        comm_dtype: str = "default",
        sms: int = 0,
        attention_tp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        _validate_a2a_request(comm_backend, phase)
        self._phase = phase
        self._comm_backend = comm_backend
        self._hidden_size = hidden_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_ep_size = moe_ep_size
        self._node_num = node_num
        self._comm_dtype = comm_dtype
        self._sms = sms
        self._attention_tp_size = attention_tp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the unified moe_a2a table (new schema + legacy
        adapters) on the three inference backends; binds ``None`` otherwise.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend in cls._SUPPORTED_BACKENDS:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

                primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.moe_a2a.value
                )
                sources = database._build_op_sources(PerfDataFilename.moe_a2a, primary, system_data_root)

                legacy_sources = {}
                for kwarg, filename_enum in (
                    ("legacy_normal_sources", PerfDataFilename.wideep_deepep_normal),
                    ("legacy_ll_sources", PerfDataFilename.wideep_deepep_ll),
                    ("legacy_trtllm_alltoall_sources", PerfDataFilename.trtllm_alltoall),
                ):
                    legacy_primary = resolve_op_data_path(
                        system_data_root, database.backend, database.version, filename_enum.value
                    )
                    legacy_sources[kwarg] = database._build_op_sources(filename_enum, legacy_primary, system_data_root)

                cls._data_cache[key] = LoadedOpData(
                    load_moe_a2a_data(sources, **legacy_sources),
                    PerfDataFilename.moe_a2a,
                    primary,
                )
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_moe_a2a_data" not in database.__dict__:
            database._moe_a2a_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "tokens"

    def get_weights(self, **kwargs) -> float:
        """All-to-all communication has no weight memory."""
        return 0.0
