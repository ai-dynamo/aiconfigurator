# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes bind the engine's table views, SOL correction (generation
only — context attention has no SOL clamp in the legacy
``_correct_data``), and grid extrapolation.
``PerfDatabase.query_context_attention`` / ``query_generation_attention``
delegate here.

``ContextAttention.query`` keeps its three ``query_mem_op`` callers
(QK-norm, apply-RoPE, KV-write) pointed at ``database.query_mem_op`` —
deciding a long-term home for the analytical mem-op formula is deferred
to the post-refactor cleanup.

Cache key is ``(systems_root, system, backend, version,
enable_shared_layer)``, same as GEMM (and every other migrated op).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.attention_lanes import resolve_attention_lane_order, split_attention_lane_tiers
from aiconfigurator_core.sdk.operations.base import OpShellKit

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=256)
def _lane_order_cached(backend, version, sm_version, override, systems_root) -> tuple[str, ...]:
    """Memoized :func:`resolve_attention_lane_order`.

    The resolution reads a YAML map and builds a list; the engine-spec build
    needs it per attention op, so memoize on the full input tuple. One entry
    per (database identity x override) — a sweep resolves each order exactly
    once.
    """
    return resolve_attention_lane_order(backend, version, sm_version, override, systems_root)


def resolve_lane_order(database, override: str | None = None) -> tuple[str, ...]:
    """Attention lane precedence for *database* under an optional *override*.

    The override is the user-facing ``attention_backend`` knob carried by the
    op; everything else comes off the database handle (backend, version,
    ``sm_version``, systems root). ``"default"`` is always the last element.
    """
    sm_version = database.system_spec["gpu"].get("sm_version") or -1
    return _lane_order_cached(database.backend, database.version, sm_version, override, database.systems_root)


# Depth of a measured "slice" below the lane key — the unit a lane serves in
# full. Context adds the fmha level: [fmha][kv][kv_n][head][window]; generation
# is [kv][kv_n][head][window].
_CONTEXT_SLICE_DEPTH = 5
_GENERATION_SLICE_DEPTH = 4

# Memo attribute for :func:`_lane_density`. Stashed on the (long-lived, class
# cached) table object because a per-call recount walks every measured row.
_LANE_DENSITY_ATTR = "_aic_lane_density"


def _lane_density(table, slice_depth: int) -> dict[str, tuple[int, int]]:
    """``{lane: (slice_count, row_count)}`` for *table*, memoized on the table.

    Both numbers matter: vllm's context table carries ``…trtllmprefill`` and
    ``…trtllmdecode`` with an identical 72-slice footprint, and only the row
    count (44 664 vs 3 684) identifies the prefill lane as the substantive one.
    The memo is keyed on ``(slice_depth, len(table))`` so a re-bound or
    differently-shaped table recomputes; value mutation in place (``_correct_sol``
    clamps latencies, never the key structure) cannot invalidate it.
    """
    stamp = (slice_depth, len(table))
    cached = getattr(table, _LANE_DENSITY_ATTR, None)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    density: dict[str, tuple[int, int]] = {}
    for lane in table:
        if not isinstance(lane, str):
            continue
        nodes = [table[lane]]
        for _ in range(slice_depth):
            nxt = []
            for node in nodes:
                if not isinstance(node, Mapping):
                    nxt = []
                    break
                nxt.extend(node.values())
            nodes = nxt
        slice_count = len(nodes)
        rows = 0
        for node in nodes:  # one slice = [n][s|b][b|s] -> measured points
            for lvl1 in node.values() if isinstance(node, Mapping) else ():
                for lvl2 in lvl1.values() if isinstance(lvl1, Mapping) else ():
                    rows += len(lvl2) if isinstance(lvl2, Mapping) else 1
        density[lane] = (slice_count, rows)

    try:
        setattr(table, _LANE_DENSITY_ATTR, (stamp, density))
    except (AttributeError, TypeError):  # plain dict fixtures cannot hold attrs
        pass
    return density


def lane_walk_order(table, lane_order: tuple[str, ...], slice_depth: int) -> tuple[str, ...]:
    """The concrete walk order for *table*: pinned lanes, then donors by density.

    Three tiers, in order:

    1. **Pinned** — the override and the framework-default map lane, in the
       precedence the resolver produced. Never re-ordered: explicit intent wins.
       The boundary comes from the resolver itself (``LaneOrder.pinned_count``,
       read by ``split_attention_lane_tiers``) — a lane order that is not
       resolver-produced (hand-specified, or an already-expanded walk) is pinned
       in full and replayed verbatim.
    2. **Donor tier** — the remaining known lanes (plus ``"default"`` last, per
       the resolver's contract), ranked by measured coverage in THIS table
       (slices, then rows, then name) instead of alphabetically. Gap-fill should
       come from the data-richest lane: on gb200/sglang, plain ``sorted()`` let
       ``flashinfer`` (10 slices / 2 584 rows) preempt ``trtllm_mha`` (64 /
       31 141) on 5 context + 10 generation slices for no reason but its name.
    3. **Table leftovers** — lanes present in the table but outside the resolver
       vocabulary, same density ranking. The collected ``kernel_source`` labels
       are richer than the map (trtllm ships ``torch_flow*``, vllm ``vllm_*``,
       sglang also ``flash_attention``) and those backends have no ``"default"``
       lane at all, so without this tier none of their rows would be reachable.

    The ranking is a pure function of the table, so it is stable for a data set
    and identical at spec-build time — the ENGINE SPEC carries this extended
    order and the Rust twin replays it verbatim rather than re-deriving it.
    """
    if not table:
        return tuple(lane_order)
    pinned, donors = split_attention_lane_tiers(lane_order)
    density = _lane_density(table, slice_depth)

    def _rank(lane: str) -> tuple[int, int, str]:
        slices, rows = density.get(lane, (0, 0))
        return (-slices, -rows, lane)

    known = sorted((lane for lane in donors if lane != "default"), key=_rank)
    tail = ("default",) if "default" in donors else ()
    leftovers = sorted((lane for lane in density if lane not in lane_order), key=_rank)
    return pinned + tuple(known) + tail + tuple(leftovers)


def resolved_lane_order_for_op(database, table_attr: str, override: str | None = None) -> list[str]:
    """Kernel-lane precedence for an attention op, RESOLVED python-side.

    Since the pyo3 op unification, ``ContextAttention``/``GenerationAttention``
    are constructed by the model layer WITHOUT a database handle (models are
    pure shape graphs; the database is only bound later, when
    ``engine.py::build_engine_spec_json`` walks a built model's op lists
    against a specific database — same place ``_wideep_moe`` pre-bakes its
    kernel_source). This is called from there, once the database is
    available, to set each attention op's ``_lane_order`` before
    serialization; it is NOT reachable from the op's own ``__init__``.

    ``table_attr`` is ``"_context_attention_data"`` or
    ``"_generation_attention_data"``. With no resolvable database (or any
    resolution failure) the always-valid ``["default"]`` is returned — the
    engine spec must never carry an empty lane list.
    """
    if database is None:
        return ["default"]
    try:
        order = resolve_lane_order(database, override)
        op_cls = ContextAttention if table_attr.startswith("_context") else GenerationAttention
        depth = _CONTEXT_SLICE_DEPTH if table_attr.startswith("_context") else _GENERATION_SLICE_DEPTH
        op_cls.load_data(database)
        return list(lane_walk_order(getattr(database, table_attr, None), order, depth))
    except Exception:
        logger.debug("attention lane order unresolvable for %s; serializing the default-only order", table_attr)
        return ["default"]


# Extrapolation target grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__`` so behavior stays bit-identical.

# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM's, used by both Attention ops.

    TODO: hoist to ``operations/base.py`` once a third op family (Phase 3
    NCCL / MLA / Mamba) lands and needs the same key shape — preferring
    duplication over premature abstraction with only two callers.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class ContextAttention(_core.ContextAttention, OpShellKit):
    """
    Context (prefill) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the context attention CSV —
    raw as-collected rows (no load-time clamp or grid pre-expansion; the
    engine owns interpolation and the SOL floor).
    """

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's context_attention table view
        (raw rows) and binds ``database._context_attention_data``,
        respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_attention_data", PerfDataFilename.context_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_context_attention_data" not in database.__dict__:
            database._context_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------


class GenerationAttention(_core.GenerationAttention, OpShellKit):
    """
    Generation (decode) attention operation.

    Owns the SILICON row cache (raw as-collected; the load-time SOL clamp
    and grid expansion retired with #1357 PR-5 — the engine owns both) plus
    the raw-cache alias kept for its historical consumers.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_attention table view
        (raw rows) and binds both database views.

        Mirrors ``GEMM.load_data``: loading operates on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_generation_attention_data", PerfDataFilename.generation_attention
            )
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_generation_attention_data" not in database.__dict__:
            database._generation_attention_data = cls._data_cache[key]
            database._raw_generation_attention_data = cls._raw_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()

    # NOTE(#1357 PR-5): the load-time SOL clamp (`_correct_sol`) retired with
    # the Python query math. The loaded table is now the RAW collected data
    # plane (enumeration/charts); the compiled engine applies the same clamp
    # to its own load (see perf_database/attention.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------


class EncoderAttention(_core.EncoderAttention, OpShellKit):
    """
    Non-causal encoder attention: full N^2, MHA, no KV cache, optional partial RoPE.

    Used to model bidirectional encoders — ViT (vision), audio encoders, and any
    other omni-modal encoder where the kernel runs full N^2 attention without a
    causal mask and without writing a KV cache. The optional
    ``partial_rotary_factor`` accounts for partial-rotation RoPE variants such as
    Qwen3-VL (factor=0.5, rotating half of head_dim). Defaults to 0.0 (no RoPE),
    matching CLIP / SigLIP / Whisper; set to 0.5 / 1.0 only for RoPE encoders.

    Owns ``_data_cache: {key: LoadedOpData}`` for the encoder attention CSV.
    Schema is simpler than context attention: MHA only (no n_kv), no KV cache
    (no kvcache_quant_mode), no sliding window. No SOL clamp. Grid extrapolation
    resolves on the raw grid via the engine's interpolation.
    """

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's encoder_attention table view
        (raw rows), binds ``database._encoder_attention_data``.
        """
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_encoder_attention_data", PerfDataFilename.encoder_attention)
            cls._record_load()

        # Bind instance attr (respect intentional test pre-overrides).
        if "_encoder_attention_data" not in database.__dict__:
            database._encoder_attention_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_encoder_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------
