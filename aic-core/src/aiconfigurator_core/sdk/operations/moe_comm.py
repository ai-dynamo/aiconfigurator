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
cache + ``load_data`` and the ``_query_a2a_table`` lookup behind
``PerfDatabase.query_moe_a2a``.

The module also owns the large-EP compute side of the same family:
``load_moe_expert_compute_data`` loads the unified ``moe_expert_compute_perf.parquet`` EP MoE compute
table (with legacy sglang/trtllm wideep adapters) into one nested dict keyed
by ``[kernel_source][quant][distribution][inference_phase][topk][num_experts]
[num_slots][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens]``.
``MoEExpertCompute`` is the op class over that table: it owns the class-level cache +
``load_data`` and the ``_query_ep_table`` lookup behind
``PerfDatabase.query_moe_expert_compute``.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError, PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path
from aiconfigurator_core.sdk.performance_result import PerformanceResult

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
    """Normalize the optional ``power`` cell to watts; null/NaN/absent -> 0.0.

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
    return 0.0 if math.isnan(value) else value


def _require_latency(row: dict, table: str) -> float:
    """Read the schema-required ``latency`` cell; a null is corrupt data.

    Unlike ``power`` (optional — see ``_row_power``), a latency-less row has
    no meaning: coercing null to 0.0 would silently poison every consumer, so
    the load refuses with a named error instead of the bare ``float("")``
    ValueError. New-schema loaders only — the legacy adapters keep their
    oracle loaders' bare ``float(row["latency"])`` behavior (parity).
    """
    raw = row.get("latency")
    if raw is None or raw == "":
        raise ValueError(
            f"null latency cell in a {table} row: latency is schema-required; refusing to load corrupt perf data"
        )
    return float(raw)


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


def _resolve_comm_dtype_slice(phase_slice, comm_dtype: str, query_context: str):
    """Three-step dtype resolution: exact key -> fp8_block alias -> sole dtype.

    1. Exact ``comm_dtype`` key.
    2. ``fp8_block`` tries ``fp8`` (debug log): fp8_block is a behavioral mode
       that reuses the fp8 comm tables — the same normalization the legacy
       ``query_trtllm_alltoall`` applies via ``_normalize_quant_mode_for_table``.
       Exact-first ordering keeps real fp8_block rows winning if a future
       collection ships them.
    3. The ``"default"`` slice when it is the sole collected key (debug log):
       the legacy DeepEP tables have no dtype axis (their adapted rows live
       under ``"default"``), so a caller asking for a payload dtype must
       still reach them — untyped data is a stand-in for any request. A sole
       TYPED key is NOT: shipped GB200 ``nvlink_one_sided`` carries only
       nvfp4, and matched two-sided rows show bf16/nvfp4 dispatch ratios of
       0.56x-3.48x, so substituting across payload dtypes is a material
       silent error where the legacy query raised. Typed slices miss.
    """
    if comm_dtype in phase_slice:
        return phase_slice[comm_dtype]
    if comm_dtype == "fp8_block" and "fp8" in phase_slice:
        logger.debug(
            "moe_a2a: comm_dtype 'fp8_block' not collected; normalizing to 'fp8' "
            "(behavioral mode reusing the fp8 comm tables; %s)",
            query_context,
        )
        return phase_slice["fp8"]
    if len(phase_slice) == 1 and "default" in phase_slice:
        logger.debug(
            "moe_a2a: comm_dtype %r not collected; falling back to the untyped 'default' slice (%s)",
            comm_dtype,
            query_context,
        )
        return phase_slice["default"]
    raise PerfDataNotAvailableError(
        f"Missing silicon data for the requested lookup; comm_dtype '{comm_dtype}' is not available for "
        f"{query_context}; collected dtypes: {sorted(phase_slice)}."
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
    # Query table (behind PerfDatabase.query_moe_a2a)
    # ------------------------------------------------------------------

    @classmethod
    def _query_a2a_table(
        cls,
        database: PerfDatabase,
        comm_backend: str,
        phase: str,
        comm_dtype: str,
        ep_size: int,
        node_num: int,
        hidden_size: int,
        topk: int,
        num_experts: int,
        num_tokens: int,
        sms: int = 0,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """Silicon lookup against the unified moe_a2a table.

        SILICON walks the slice (backend -> phase -> dtype-with-fallback ->
        ep -> node -> hidden -> topk -> experts), then resolves ``sms``: an
        exact sms key gets a 1-D token interpolation, otherwise a 2-D
        (sms, tokens) grid — the same split the legacy DeepEP-normal query
        uses. SOL/SOL_FULL/EMPIRICAL have no estimation tier yet and raise
        ``EmpiricalNotImplementedError``; HYBRID falls back to that same
        raise when silicon data misses.
        """
        cls.load_data(database)
        _validate_a2a_request(comm_backend, phase)

        if database_mode is None:
            database_mode = database._default_database_mode

        query_context = (
            f"moe_a2a {comm_backend}/{phase}: {comm_dtype=}, {ep_size=}, {node_num=}, "
            f"{hidden_size=}, {topk=}, {num_experts=}, {sms=}, {num_tokens=}"
        )

        if database_mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL, common.DatabaseMode.EMPIRICAL):
            raise EmpiricalNotImplementedError(
                f"{database_mode.name} mode is not available for {query_context}: "
                "silicon data required (estimation tier is a planned follow-up)."
            )

        def get_silicon() -> PerformanceResult:
            phase_slice = util_empirical.require_data_slice(database._moe_a2a_data, comm_backend, phase)
            dtype_slice = _resolve_comm_dtype_slice(phase_slice, comm_dtype, query_context)
            by_sms = util_empirical.require_data_slice(dtype_slice, ep_size, node_num, hidden_size, topk, num_experts)
            # 1-D/2-D token curves with a linear token proxy SOL for the
            # boundary util-hold: per-slice payload bytes scale ~linearly with
            # tokens (hidden/topk/dtype fixed), so the proxy is
            # ratio-equivalent to any bandwidth roofline (see the DeepEP notes
            # in operations/moe.py).
            if sms in by_sms:
                config = perf_interp.OpInterpConfig(
                    axes=("num_tokens",), resolver=perf_interp.Grid(), sol_fn=lambda t: float(t)
                )
                result = perf_interp.query(config, by_sms[sms], num_tokens)
            else:
                config = perf_interp.OpInterpConfig(
                    axes=("sms", "num_tokens"), resolver=perf_interp.Grid(), sol_fn=lambda _sm, t: float(t)
                )
                result = perf_interp.query(config, by_sms, sms, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat, energy=energy)

        def get_empirical() -> float:
            raise EmpiricalNotImplementedError(
                f"HYBRID empirical fallback is not available for {query_context}: "
                "silicon data required (estimation tier is a planned follow-up)."
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=f"Failed to query moe_a2a data for {query_context}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        num_tokens = kwargs.get("x")  # per-rank tokens — comm ops never ADP-scale
        # Legacy fidelity: attention TP shards the token stream ahead of the
        # A2A, mirroring MoEDispatch's ``num_tokens // self._scale_num_tokens``
        # exactly — plain floor division, no max(1, ...) guard (0 is possible).
        num_tokens = num_tokens // self._attention_tp_size
        result = database.query_moe_a2a(
            self._comm_backend,
            self._phase,
            self._comm_dtype,
            self._moe_ep_size,
            self._node_num,
            self._hidden_size,
            self._topk,
            self._num_experts,
            num_tokens,
            sms=self._sms,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs) -> float:
        """All-to-all communication has no weight memory."""
        return 0.0


# ---------------------------------------------------------------------------
# EP MoE compute (moe_expert_compute_perf.parquet) — same family, compute side
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
    collision (debug log) — the keep-first convention shared by new-schema
    rows AND the legacy adapters (their oracles adopted the
    skip-on-key-conflict shared-layer contract in #1423). ``overwrite=True``
    replaces whatever is there — used only by the first new-schema
    occurrence of a key to take precedence over legacy-adapted rows.
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
    kernel-source-based quant rerouting (unlike ``load_moe_data``), and the
    first row wins on a key collision (``overwrite=False``) — the oracles
    adopted the skip-on-key-conflict shared-layer contract in #1423, so a
    cross-source conflict (primary vs shared/fallback file) resolves to the
    first-loaded source on both paths. ``kernel_source`` is pinned to
    ``"deepep_moe"`` (spec §4.2; the legacy column spells it ``deepepmoe``
    and the oracles never read it), ``num_slots = num_experts`` (the legacy
    sglang tables have no EPLB redundancy axis), and ``inference_phase``
    comes from which kwarg carried the source file.
    """
    for row in rows:
        latency = float(row["latency"])
        power = _row_power(row)
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
        _store_ep_leaf(data, key, {"latency": latency, "power": power, "energy": power * latency}, overwrite=False)


def _adapt_legacy_sglang_context_moe(data: defaultdict, rows) -> None:
    _adapt_legacy_sglang_wideep_moe(data, rows, inference_phase="context")


def _adapt_legacy_sglang_generation_moe(data: defaultdict, rows) -> None:
    _adapt_legacy_sglang_wideep_moe(data, rows, inference_phase="generation")


def _adapt_legacy_trtllm_wideep_moe(data: defaultdict, rows) -> None:
    """Adapt legacy trtllm ``wideep_moe_perf`` rows.

    Mirrors ``load_wideep_moe_compute_data`` (the oracle): native
    ``kernel_source`` (``"moe_torch_flow"`` when the column is absent),
    ``num_slots`` and ``_eplb`` distributions pass through unchanged, no
    quant rerouting, first row wins on a key collision (``overwrite=False``
    — the oracle adopted the skip-on-key-conflict shared-layer contract in
    #1423). The legacy table has no context/generation split — one kernel
    measured across the token range — so each row is registered under BOTH
    ``inference_phase`` values with identical (but independent) leaves.
    """
    for row in rows:
        latency = float(row["latency"])
        power = _row_power(row)
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
            _store_ep_leaf(data, (*base_key, inference_phase, *shape_key), leaf, overwrite=False)


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


def load_moe_expert_compute_data(
    sources,
    legacy_context_sources=None,
    legacy_generation_sources=None,
    legacy_trtllm_wideep_sources=None,
) -> dict | None:
    """Load the unified EP MoE compute table (``moe_expert_compute_perf.parquet``).

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
        latency = _require_latency(row, "moe_expert_compute_perf")  # already ms (spec §4.2) — stored raw
        power = _row_power(row)
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


# ---------------------------------------------------------------------------
# MoEExpertCompute — the op class over the unified moe_ep table
# ---------------------------------------------------------------------------


_EP_PHASES = ("context", "generation")

#: Kernel legs adapted from the legacy sglang wideep tables — the loader pins
#: ``"deepep_moe"`` (spec §4.2) and ``_resolve_kernel_source`` returns it for
#: sglang/vllm. The EPLB context correction applies to these legs only; the
#: trtllm legs (``deepgemm``/``moe_torch_flow``) never carried it.
_SGLANG_ADAPTED_KERNEL_SOURCES = frozenset({"deepep_moe"})


def _validate_ep_phase(inference_phase: str) -> None:
    """Shared ctor/query validation: an unknown inference phase is a ValueError."""
    if inference_phase not in _EP_PHASES:
        raise ValueError(f"Invalid inference_phase '{inference_phase}'. Must be one of {list(_EP_PHASES)}")


def _resolve_ep_distribution(quant_slice, workload_distribution: str, inference_phase: str, query_context: str) -> str:
    """Distribution fallback chain: requested -> "uniform" -> first-available.

    Candidates are the distributions that actually carry ``inference_phase``
    data, in table insertion order — the legacy sglang tables are separate
    files per phase, so their fallback is inherently phase-scoped; the
    unified table nests phase below distribution and must filter to match.
    The chain reproduces both legacy oracles on their shipped tables: the
    sglang tables include "uniform", so step 2 fires there (the sglang
    oracle's own fallback); the trtllm gb200 table has no "uniform", so
    step 3 fires there with the same first-collected distribution the trtllm
    oracle picks (``available_distributions[0]`` — e.g. the production
    default request "power_law" resolves to "power_law_1.01_eplb"). The
    residual corner — a table lacking "uniform" where the sglang oracle
    would raise but the unified chain answers first-available — is strictly
    more permissive and unreachable on shipped sglang data, the same stance
    the comm-dtype fallback takes. A typed miss remains only when no
    collected distribution carries the requested phase.
    """
    candidates = [dist for dist, phases in quant_slice.items() if inference_phase in phases]
    if workload_distribution in candidates:
        return workload_distribution
    if "uniform" in candidates:
        logger.debug(
            "moe_ep: workload_distribution %r not collected; falling back to 'uniform' (%s)",
            workload_distribution,
            query_context,
        )
        return "uniform"
    if candidates:
        logger.debug(
            "moe_ep: workload_distribution %r not collected; falling back to first available %r (%s)",
            workload_distribution,
            candidates[0],
            query_context,
        )
        return candidates[0]
    raise PerfDataNotAvailableError(
        f"Missing silicon data for the requested lookup; workload_distribution '{workload_distribution}' is not "
        f"available for {query_context}; no collected distribution carries {inference_phase} data."
    )


class MoEExpertCompute(Operation):
    """Unified large-EP MoE expert-compute op (one inference phase per instance).

    Owns ``_moe_ep_data`` — the unified compute table loaded by
    :func:`load_moe_expert_compute_data` (new-schema ``moe_expert_compute_perf.parquet`` plus the
    legacy sglang wideep context/generation and trtllm wideep adapters).
    Loaded on every inference backend ({"sglang", "vllm", "trtllm"} all have
    legacy compute sources); ``None`` otherwise. ``query(x=...)`` scales
    tokens by ``attention_dp_size`` (attention DP globalizes tokens through
    the A2A dispatch — the same scaling as the legacy ``MoE`` /
    ``TrtLLMWideEPMoE`` query paths) and always queries ``moe_tp_size=1``:
    the large-EP family is EP-only. ``num_slots`` defaults to ``num_experts``
    (no EPLB redundancy); ``kernel_source=None`` auto-resolves per backend at
    query time (see :meth:`_resolve_kernel_source`). ``enable_eplb=True`` is
    legacy fidelity with the sglang MoE query: tokens become
    ``int(tokens * 0.8)`` before the table lookup when the phase is context
    AND the resolved kernel leg is sglang-adapted
    (``_SGLANG_ADAPTED_KERNEL_SOURCES``) — never on the trtllm legs, whose
    EPLB effect rides the ``_eplb`` distribution suffix instead.
    """

    _data_cache: ClassVar[dict] = {}

    _SUPPORTED_BACKENDS: ClassVar[tuple[str, ...]] = ("sglang", "vllm", "trtllm")

    def __init__(
        self,
        name: str,
        scale_factor: float,
        *,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        attention_dp_size: int,
        inference_phase: str,
        num_slots: int | None = None,
        kernel_source: str | None = None,
        is_gated: bool = True,
        enable_eplb: bool = False,
    ) -> None:
        super().__init__(name, scale_factor)
        _validate_ep_phase(inference_phase)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._topk = topk
        self._num_experts = num_experts
        self._num_slots = num_slots if num_slots is not None else num_experts
        self._moe_ep_size = moe_ep_size
        self._quant_mode = quant_mode
        self._workload_distribution = workload_distribution
        self._attention_dp_size = attention_dp_size
        self._inference_phase = inference_phase
        self._kernel_source = kernel_source
        self._is_gated = is_gated
        self._enable_eplb = enable_eplb
        # 3 GEMMs for gated (gate, up, down), 2 GEMMs for non-gated (up, down).
        # EP-only family: no moe_tp division (moe_tp == 1 by construction).
        # Parity-pinned: sized by num_experts, NOT num_slots, matching the
        # retired TrtLLMWideEPMoE._weights (operations/moe.py:1435 @ dc4caca)
        # so get_weights() byte-matches the legacy classes on the shipped
        # gb200 EPLB artifacts. Physically EPLB replicates experts across
        # slots, so num_slots-based sizing is the correct model — tracked as
        # AIC-1674 (intentional delta: moves the memory column on EPLB
        # configs). The SOL roofline's num_slots weight term mirrors its own
        # legacy twin (the retired wideep_moe.rs sol_latency_ms) — the
        # asymmetry is inherited, not invented.
        num_gemms = 3 if is_gated else 2
        self._weights = hidden_size * inter_size * num_experts * quant_mode.value.memory * num_gemms // moe_ep_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the unified moe_ep table (new schema + legacy
        adapters) on the three inference backends; binds ``None`` otherwise.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend in cls._SUPPORTED_BACKENDS:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

                primary = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.moe_expert_compute.value
                )
                sources = database._build_op_sources(PerfDataFilename.moe_expert_compute, primary, system_data_root)

                legacy_sources = {}
                for kwarg, filename_enum in (
                    ("legacy_context_sources", PerfDataFilename.wideep_context_moe),
                    ("legacy_generation_sources", PerfDataFilename.wideep_generation_moe),
                    ("legacy_trtllm_wideep_sources", PerfDataFilename.wideep_moe_compute),
                ):
                    legacy_primary = resolve_op_data_path(
                        system_data_root, database.backend, database.version, filename_enum.value
                    )
                    legacy_sources[kwarg] = database._build_op_sources(filename_enum, legacy_primary, system_data_root)

                cls._data_cache[key] = LoadedOpData(
                    load_moe_expert_compute_data(sources, **legacy_sources),
                    PerfDataFilename.moe_expert_compute,
                    primary,
                )
            else:
                cls._data_cache[key] = None

            cls._record_load()

        if "_moe_ep_data" not in database.__dict__:
            database._moe_ep_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # kernel_source auto-resolution (kernel_source=None)
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_kernel_source(cls, database: PerfDatabase, quant_mode: common.MoEQuantMode) -> str:
        """Resolve the collected kernel key when the caller pins none.

        sglang/vllm large-EP MoE has a single collected kernel
        (``"deepep_moe"``, spec §4.2). trtllm replicates
        ``TrtLLMWideEPMoE._select_kernel`` (TensorRT-LLM's
        ``MoEOpSelector.select_op``) against the unified table's kernel keys:
        Blackwell (SM >= 100) + fp8_block -> ``"deepgemm"``, otherwise
        ``"moe_torch_flow"`` (Cutlass); an absent preferred kernel falls back
        to the first collected kernel key. Copied, not imported — the legacy
        classmethod consults its trtllm-only ``_wideep_moe_compute_data``
        table, which this family retires.
        """
        if database.backend in ("sglang", "vllm"):
            return "deepep_moe"

        cls.load_data(database)
        sm_version = database.system_spec["gpu"]["sm_version"]
        is_blackwell = sm_version >= 100
        quant_mode_str = quant_mode.name if hasattr(quant_mode, "name") else str(quant_mode)
        preferred = "deepgemm" if is_blackwell and "fp8_block" in quant_mode_str else "moe_torch_flow"

        ep_data = database._moe_ep_data
        if ep_data:
            available_kernels = list(ep_data.keys())
            if preferred in available_kernels:
                return preferred
            if available_kernels:
                fallback = available_kernels[0]
                logger.debug(f"Preferred MoE kernel '{preferred}' not available, falling back to '{fallback}'")
                return fallback

        return preferred

    # ------------------------------------------------------------------
    # Query table (behind PerfDatabase.query_moe_expert_compute)
    # ------------------------------------------------------------------

    @classmethod
    def _query_ep_table(
        cls,
        database: PerfDatabase,
        kernel_source: str,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        inference_phase: str,
        topk: int,
        num_experts: int,
        num_slots: int,
        hidden_size: int,
        inter_size: int,
        moe_tp_size: int,
        moe_ep_size: int,
        num_tokens: int,
        is_gated: bool = True,
        enable_eplb: bool = False,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """Silicon lookup against the unified moe_ep table.

        SILICON walks the slice (kernel -> quant -> distribution-with-fallback
        -> phase -> topk -> experts -> slots -> hidden -> inter -> tp -> ep),
        then resolves tokens on a 1-D ``perf_interp`` Grid curve: RAW lerp in
        range; beyond the collected range the engine holds the boundary util
        and the wideep-MoE roofline SOL carries the growth — exactly the
        legacy retrieval (``MoE._query_moe_table`` deepep branch /
        ``TrtLLMWideEPMoE._query_compute_table``). The sglang oracle's
        singleton-underflow guard is adopted family-wide: a single measured
        token point cannot define the low-token launch floor, so querying
        below it is a typed miss (the trtllm oracle would boundary-hold there;
        the guard is the deliberate, safer behavior). SOL/SOL_FULL/EMPIRICAL
        have no estimation tier yet and raise
        ``EmpiricalNotImplementedError``; HYBRID falls back to that same
        raise when silicon data misses.
        """
        cls.load_data(database)
        _validate_ep_phase(inference_phase)
        if enable_eplb and inference_phase == "context" and kernel_source in _SGLANG_ADAPTED_KERNEL_SOURCES:
            # Legacy sglang EPLB prefill correction (moe.py:649): EPLB
            # rebalancing flattens hot-expert load, modeled as int(tokens*0.8)
            # before the table walk. Context + deepep tables only — the trtllm
            # EPLB effect rides the `_eplb` distribution suffix instead.
            num_tokens = int(num_tokens * 0.8)

        if database_mode is None:
            database_mode = database._default_database_mode

        query_context = (
            f"moe_ep {kernel_source}/{inference_phase}: {quant_mode=}, {workload_distribution=}, "
            f"{topk=}, {num_experts=}, {num_slots=}, {hidden_size=}, {inter_size=}, "
            f"{moe_tp_size=}, {moe_ep_size=}, {num_tokens=}"
        )

        if database_mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL, common.DatabaseMode.EMPIRICAL):
            raise EmpiricalNotImplementedError(
                f"{database_mode.name} mode is not available for {query_context}: "
                "silicon data required (estimation tier is a planned follow-up)."
            )

        # Verbatim wideep-MoE roofline (TrtLLMWideEPMoE._query_compute_table's
        # get_sol): num_slots (not num_experts) sizes the weight-read term —
        # EPLB redundant mode replicates experts across slots. sglang-adapted
        # slices pin num_slots == num_experts, where this reduces exactly to
        # the sglang oracle's SOL (MoE._query_moe_table's get_sol). num_gemms
        # is pinned to 3 (gated): the forward has no is_gated axis and both
        # Gated (SwiGLU): 3 GEMMs; non-gated (Relu2): 2 — the legacy sglang
        # oracle derives this in _query_moe_table (moe.py:309) from the op's
        # is_gated. The SOL only shapes the beyond-range boundary util-hold;
        # in-range lookups are raw lerp on measured points.
        num_gemms = 3 if is_gated else 2

        def get_sol_latency(tokens: int) -> float:
            total_tokens = tokens * topk
            ops = total_tokens * hidden_size * inter_size * num_gemms * 2 // moe_ep_size // moe_tp_size
            mem_bytes = quant_mode.value.memory * (
                total_tokens // moe_ep_size * hidden_size * 2  # input+output
                + total_tokens // moe_ep_size * inter_size * num_gemms // moe_tp_size  # intermediate activations
                + hidden_size
                * inter_size
                * num_gemms
                // moe_tp_size
                * min(num_slots // moe_ep_size, total_tokens // moe_ep_size)  # weights (use num_slots)
            )
            sol_math = ops / (database.system_spec["gpu"]["bfloat16_tc_flops"] * quant_mode.value.compute) * 1000
            sol_mem = mem_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            return max(sol_math, sol_mem)

        def get_silicon() -> PerformanceResult:
            quant_slice = util_empirical.require_data_slice(database._moe_ep_data, kernel_source, quant_mode)
            used_distribution = _resolve_ep_distribution(
                quant_slice, workload_distribution, inference_phase, query_context
            )
            moe_dict = util_empirical.require_data_slice(
                quant_slice,
                used_distribution,
                inference_phase,
                topk,
                num_experts,
                num_slots,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
            )
            token_points = sorted(moe_dict)
            if len(token_points) == 1 and num_tokens < token_points[0]:
                raise PerfDataNotAvailableError(
                    "MoE EP silicon token underflow has only one measured point; cannot infer "
                    f"low-token latency from a singleton. measured_token={token_points[0]}, {query_context}."
                )
            config = perf_interp.OpInterpConfig(
                axes=("num_tokens",), resolver=perf_interp.Grid(), sol_fn=get_sol_latency
            )
            result = perf_interp.query(config, moe_dict, num_tokens)
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat, energy=energy)

        def get_empirical() -> float:
            raise EmpiricalNotImplementedError(
                f"HYBRID empirical fallback is not available for {query_context}: "
                "silicon data required (estimation tier is a planned follow-up)."
            )

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=f"Failed to query moe_ep data for {query_context}",
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        # Attention DP globalizes tokens: the A2A dispatch delivers every DP
        # rank's tokens to the experts (same scaling as legacy MoE /
        # TrtLLMWideEPMoE query()).
        num_tokens = kwargs.get("x") * self._attention_dp_size
        # Per-call quant override — the legacy expert-compute ops both honor
        # kwargs.get("quant_mode"); it drives kernel resolution too (a
        # Blackwell fp8_block override selects deepgemm, not moe_torch_flow).
        quant_mode = kwargs.get("quant_mode") or self._quant_mode
        kernel_source = self._kernel_source
        if kernel_source is None:
            kernel_source = self._resolve_kernel_source(database, quant_mode)
        result = database.query_moe_expert_compute(
            kernel_source,
            quant_mode,
            self._workload_distribution,
            self._inference_phase,
            self._topk,
            self._num_experts,
            self._num_slots,
            self._hidden_size,
            self._inter_size,
            1,  # moe_tp_size — the large-EP family is EP-only
            self._moe_ep_size,
            num_tokens,
            is_gated=self._is_gated,
            enable_eplb=self._enable_eplb,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs) -> float:
        return self._weights * self._scale_factor
