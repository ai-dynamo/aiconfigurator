# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) module-level ops (ISSUE-10 / AIC-538).

Both ContextDSAModule and GenerationDSAModule own their CSV-backed perf
tables and grid extrapolation. ``PerfDatabase.query_context_dsa_module``
and ``query_generation_dsa_module`` delegate here.

Both classes still bind a ``_raw_data_cache`` for backward compatibility,
but with load-time pre-expansion removed the table IS the raw measurements,
so it is a plain alias. (The PR #903 topk-piecewise lookup and the hand-rolled
boundary-util anchoring it served are superseded by the engine's interpolation: linear
bracket blends cannot overshoot the topk knee the way cubic did, and
util-hold is native.)

No SOL clamping in the legacy ``_correct_data`` for either DSA op —
extrapolation only. The legacy ``__init__`` loaded DSA twice (once near
the MLA/Mamba block, once after); both loads are consolidated into a
single ``load_data`` call per class.

The DSA-specific helper ``_format_dsa_unavailable_message`` also moves
here as a module-level function. ``DSA_MODEL_DIMS`` and ``DEFAULT_DSA_ARCHITECTURE`` stay on
``perf_database.py`` as module-level constants for now — the cleanup PR
revisits their home.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# kernel_source -> configured-backend bucket(s) for FP8-KV rows. 0.5.14
# SGLang DSA collectors record the EXECUTED kernel; dense ragged prefill is
# selected by SHAPE (isl <= 2048) under either configured backend, so its
# rows back both buckets.
_DSA_KERNEL_SOURCE_BUCKETS = {
    "sglang_dsa_indexer_trtllm": ("trtllm",),
    "sglang_dsa_skip_indexer_trtllm": ("trtllm",),
    "sglang_dsa_indexer_flashmla_sparse": ("flashmla_kv",),
    "sglang_dsa_skip_indexer_flashmla_sparse": ("flashmla_kv",),
    "sglang_dsa_dense_mha_trtllm_ragged": ("trtllm", "flashmla_kv"),
}


def _dsa_kernel_source_buckets(kernel_source: str, kv_dtype) -> tuple[str, ...]:
    """Configured-backend bucket(s) a DSA perf row supports.

    The trtllm/flashmla_kv split mirrors serving's FP8-KV sub-backend selector
    (an FP8-KV rule: SM90 -> flashmla_kv, SM100+ -> trtllm; BF16 KV stays on
    framework defaults). With a BF16 KV cache there is exactly ONE real
    execution path per shape, so every bf16 row backs BOTH buckets — a bare
    substring test split one measured b200 sweep across the two buckets and
    left the default query bucket with nothing beyond 2048 tokens. FP8 rows
    bucket by executed-kernel name; legacy (pre-0.5.14) names keep the old
    substring rule.
    """
    if kv_dtype is common.KVCacheQuantMode.bfloat16:
        return ("trtllm", "flashmla_kv")
    buckets = _DSA_KERNEL_SOURCE_BUCKETS.get(kernel_source)
    if buckets is not None:
        return buckets
    return ("trtllm",) if "trtllm" in kernel_source else ("flashmla_kv",)


DSA_MODEL_DIMS: dict[str, dict] = {
    "DeepseekV32ForCausalLM": {
        "hidden_size": 7168,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 64,
    },
    "GlmMoeDsaForCausalLM": {
        "hidden_size": 6144,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_topk": 2048,
        "index_head_dim": 128,
        "index_n_heads": 32,
    },
}

DEFAULT_DSA_ARCHITECTURE = "DeepseekV32ForCausalLM"




# Extrapolation grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__``.

# fmt: on


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as GEMM, Attention, and Communication.

    Still local to ``operations/dsa.py`` (Phase 3 has 5 duplicate copies
    so far); the cleanup PR hoists this to ``operations/base.py`` once
    Phase 3 settles.
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


def _format_dsa_unavailable_message(
    phase: str,
    error: Exception,
    *,
    b: int,
    s: int,
    num_heads: int,
    architecture: str,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    prefix: int | None = None,
) -> str:
    """Format the ``PerfDataNotAvailableError`` message body. Lifted verbatim
    from ``PerfDatabase._format_dsa_unavailable_message``."""
    prefix_part = "" if prefix is None else f", prefix={prefix}"
    return (
        f"{phase} DSA module perf data unavailable for candidate "
        f"b={b}, s={s}{prefix_part}, num_heads={num_heads}, architecture={architecture}, "
        f"index_n_heads={index_n_heads}, index_head_dim={index_head_dim}, index_topk={index_topk}: {error}"
    )


class ContextDSAModule(Operation):
    """
    Context phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns ``_data_cache`` (extrapolated context_dsa_module CSV) AND
    ``_raw_data_cache`` (the same CSV pre-extrapolation, used by the
    topk-boundary piecewise interpolation path).

    Models the full DSA attention block including:
    - kv_a_proj_with_mqa GEMM (includes indexer K projection)
    - LayerNorm + q_b_proj GEMM
    - Indexer: wq_b GEMM, weights_proj GEMM, FP8 MQA logits, TopK selection
    - Sparse MLA attention (attends to top-k tokens instead of full sequence)
    - BMM pre/post (weight absorption + V projection)
    - o_proj GEMM
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _skip_data_cache: ClassVar[dict] = {}
    _raw_skip_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
        cp_size: int = 1,
        index_topk_freq: int = 1,
        dsa_full_layer_fraction: float | None = None,
        attn_projection_quant_modes: dict | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._cp_size = cp_size
        # GLM-5.2 shares one DSA topk index across a group of layers: some compute
        # the indexer (full), the rest reuse it (skip). query() amortizes
        # per_layer = full_frac*full + (1-full_frac)*skip, using the directly-
        # collected skip data (no delta). full_frac is the EXACT fraction of
        # indexer-computing layers (honors index_skip_topk_offset/pattern), passed
        # by the model; fall back to 1/freq only if not provided. full_frac==1.0
        # (DeepSeek-V3.2 / GLM-5) => pure full, skip path never taken.
        self._index_topk_freq = max(1, int(index_topk_freq or 1))
        self._full_frac = (
            float(dsa_full_layer_fraction) if dsa_full_layer_fraction is not None else 1.0 / self._index_topk_freq
        )
        # Persisted (not just consumed) so the opspec can carry the
        # per-projection checkpoint fact to the engine's weight_bytes.
        # Normalized to ALL FOUR groups: a partial mapping fills the missing
        # projections from gemm_quant_mode, so the opspec emission (and the
        # Rust DsaProjectionQuants deserialization, which requires every
        # field) never sees an incomplete map.
        self._attn_projection_quant_modes = {
            **dict.fromkeys(("q", "kv", "o", "indexer"), gemm_quant_mode),
            **(attn_projection_quant_modes or {}),
        }

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's context_dsa_module table view
        (full + skip_indexer splits), binds
        ``database._context_dsa_module_data`` and
        ``database._raw_context_dsa_module_data``."""
        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view, load_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_context_dsa_module_data", PerfDataFilename.dsa_context_module)
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid pre-expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer (GLM-5.2) rows live in the SAME file, tagged by op_name
        # (dsa_context_module_skip_indexer); the engine view splits them out.
        # Empty (no skip rows -> DeepSeek-V3.2 / GLM-5 freq==1) =>
        # slot None and the skip query path is never taken.
        if key not in cls._skip_data_cache:
            skip_view = fetch_table_view(database, "_context_dsa_module_skip_data")
            if skip_view:
                cls._skip_data_cache[key] = LoadedOpData(
                    skip_view, PerfDataFilename.dsa_context_module, cls._data_cache[key].filepath
                )
                cls._raw_skip_data_cache[key] = cls._skip_data_cache[key]
            else:
                cls._skip_data_cache[key] = None
                cls._raw_skip_data_cache[key] = None

        if "_context_dsa_module_data" not in database.__dict__:
            database._context_dsa_module_data = cls._data_cache[key]
        if "_raw_context_dsa_module_data" not in database.__dict__:
            database._raw_context_dsa_module_data = cls._raw_data_cache[key]
        if "_context_dsa_module_skip_data" not in database.__dict__:
            database._context_dsa_module_skip_data = cls._skip_data_cache[key]
        if "_raw_context_dsa_module_skip_data" not in database.__dict__:
            database._raw_context_dsa_module_skip_data = cls._raw_skip_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._skip_data_cache.clear()
        cls._raw_skip_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_dsa_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"

    # ------------------------------------------------------------------
    # Context-Parallel (CP) prefill model — GLM-5 DSA only.
    # See docs/CONTEXT_PARALLEL_DSA_MODELING.md. Per-card =
    #   base dsa_module(isl/cp, bf16-KV row)
    #   + mqa(isl/cp)*(cp-1)                          (mqa ∝ isl², xcp identity)
    #   - [topk_full(flat) - topk_full(top_last)]/cp  (topk ∝ full/cp; module is dummy/flat)
    #   + AG_KV + AG_LSE                              (the two small attention all-gathers)
    # AG_hidden + RS belong to the MoE comm (modeled by MoEDispatch), not here.
    # ------------------------------------------------------------------


class GenerationDSAModule(Operation):
    """
    Generation phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns both an extrapolated working cache and the original measured rows.
    The raw view supplies trustworthy boundary utilization when a sequence
    query falls outside a collected curve.

    Models the full DSA attention block during decode:
    - Same components as ContextDSAModule
    - Uses paged MQA logits for indexer
    - Sparse MLA with KV cache lookup
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _skip_data_cache: ClassVar[dict] = {}
    _raw_skip_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
        index_topk_freq: int = 1,
        dsa_full_layer_fraction: float | None = None,
        attn_projection_quant_modes: dict | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        # GLM-5.2 shared-index amortization (see ContextDSAModule): exact
        # full-layer fraction; fall back to 1/freq. full_frac==1.0 => pure full.
        self._index_topk_freq = max(1, int(index_topk_freq or 1))
        self._full_frac = (
            float(dsa_full_layer_fraction) if dsa_full_layer_fraction is not None else 1.0 / self._index_topk_freq
        )
        # Persisted (not just consumed) so the opspec can carry the
        # per-projection checkpoint fact to the engine's weight_bytes.
        # Normalized to ALL FOUR groups: a partial mapping fills the missing
        # projections from gemm_quant_mode, so the opspec emission (and the
        # Rust DsaProjectionQuants deserialization, which requires every
        # field) never sees an incomplete map.
        self._attn_projection_quant_modes = {
            **dict.fromkeys(("q", "kv", "o", "indexer"), gemm_quant_mode),
            **(attn_projection_quant_modes or {}),
        }

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's generation_dsa_module table view
        (full + skip_indexer splits) and binds both views on ``database``."""
        from aiconfigurator_core.sdk.engine_table_view import fetch_table_view, load_view
        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(
                database, "_generation_dsa_module_data", PerfDataFilename.dsa_generation_module
            )
            # The raw wrapper stays a plain alias of the table (no load-time
            # grid pre-expansion since PR-5).
            cls._raw_data_cache[key] = cls._data_cache[key]
            cls._record_load()

        # skip_indexer rows share the same file (op_name tag); the engine view
        # splits them out.
        if key not in cls._skip_data_cache:
            skip_view = fetch_table_view(database, "_generation_dsa_module_skip_data")
            if skip_view:
                cls._skip_data_cache[key] = LoadedOpData(
                    skip_view, PerfDataFilename.dsa_generation_module, cls._data_cache[key].filepath
                )
                cls._raw_skip_data_cache[key] = cls._skip_data_cache[key]
            else:
                cls._skip_data_cache[key] = None
                cls._raw_skip_data_cache[key] = None

        if "_generation_dsa_module_data" not in database.__dict__:
            database._generation_dsa_module_data = cls._data_cache[key]
            database._raw_generation_dsa_module_data = cls._raw_data_cache[key]
        if "_generation_dsa_module_skip_data" not in database.__dict__:
            database._generation_dsa_module_skip_data = cls._skip_data_cache[key]
            database._raw_generation_dsa_module_skip_data = cls._raw_skip_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._skip_data_cache.clear()
        cls._raw_skip_data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_dsa_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def _read_dsa_row_sources(file_or_sources):
    """Read rows while retaining priority-source boundaries.

    DSA files historically used last-row-wins for duplicates within one file.
    Shared-layer inputs add a second requirement: an earlier source (the active
    stack) must outrank every later sibling source. ``_read_filtered_rows``
    intentionally flattens sources, so DSA keeps the groups here and applies
    those two rules independently.
    """
    if isinstance(file_or_sources, str):
        rows = _read_filtered_rows(file_or_sources)
        return None if rows is None else [rows]

    row_sources = []
    any_source_exists = False
    for source in file_or_sources:
        rows = _read_filtered_rows([source])
        if rows is None:
            continue
        any_source_exists = True
        row_sources.append(rows)
    return row_sources if any_source_exists else None


def load_context_dsa_module_data(dsa_file: str, op_kind: str = "full"):
    """
    Load context DSA data.

    Dict structure:
        data[fmha_quant_mode][kv_cache_quant_mode][gemm_quant_mode][architecture][dsa_backend][num_heads][prefix][s][b]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported FMHAQuantMode names (aligned with
    ``_context_attention_data``).  ``architecture`` (e.g.
    "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM") selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".

    Full and skip-indexer (GLM-5.2 reuse-layer) rows live in the SAME file,
    distinguished by the ``op_name`` column (``dsa_context_module`` vs
    ``dsa_context_module_skip_indexer``) — no extra column. ``op_kind`` selects
    which to keep: ``"full"`` (op_name without ``skip_indexer``) or ``"skip"``.
    """
    row_sources = _read_dsa_row_sources(dsa_file)
    if row_sources is None:
        logger.debug(f"DSA context data file {dsa_file} not found.")
        return None

    def _nest():
        return defaultdict(_nest)

    dsa_data = _nest()

    first_row = next((row for source_rows in row_sources for row in source_rows), None)
    has_power = first_row is not None and "power" in first_row
    seen_coordinates = set()

    for source_rows in row_sources:
        # Preserve legacy last-row-wins behavior within each source.
        source_values = {}
        for row in source_rows:
            # full vs skip-indexer share one file, split by op_name.
            if ("skip_indexer" in (row.get("op_name") or "")) != (op_kind == "skip"):
                continue
            num_heads = int(row["num_heads"])
            b = int(row["batch_size"])
            s = int(row["isl"])
            latency = float(row["latency"])
            power = float(row.get("power", 0.0)) if has_power else 0.0
            energy = power * latency

            arch = row.get("architecture", DEFAULT_DSA_ARCHITECTURE)
            step = row.get("step")
            step_missing = step is None or (isinstance(step, str) and step.strip() == "")
            if arch == "GlmMoeDsaForCausalLM" and step_missing:
                raise ValueError(
                    "GLM-5 context DSA module data requires a non-empty step column for prefix/past_kv length"
                )
            prefix = 0 if step_missing else int(step)
            gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
            fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
            kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

            ks = row.get("kernel_source") or ""
            for dsa_backend in _dsa_kernel_source_buckets(ks, kv_dtype):
                coordinate = (fmha_mode, kv_dtype, gemm_mode, arch, dsa_backend, num_heads, prefix, s, b)
                source_values[coordinate] = {
                    "latency": latency,
                    "power": power,
                    "energy": energy,
                }

        # Sources are priority-ordered: active first, shared fallbacks later.
        for coordinate, value in source_values.items():
            if coordinate in seen_coordinates:
                continue
            seen_coordinates.add(coordinate)
            fmha_mode, kv_dtype, gemm_mode, arch, dsa_backend, num_heads, prefix, s, b = coordinate
            dsa_data[fmha_mode][kv_dtype][gemm_mode][arch][dsa_backend][num_heads][prefix][s][b] = value

    return dsa_data


def load_generation_dsa_module_data(dsa_file: str, op_kind: str = "full"):
    """
    Load generation DSA data.

    Dict structure:
        data[kv_cache_quant_mode][gemm_quant_mode][architecture][dsa_backend][num_heads][b][s]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported KVCacheQuantMode names (aligned with
    ``_generation_attention_data``).  ``architecture`` selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".

    Full and skip-indexer rows share one file, split by the ``op_name`` column;
    ``op_kind`` ("full"/"skip") selects which to keep.
    """
    row_sources = _read_dsa_row_sources(dsa_file)
    if row_sources is None:
        logger.debug(f"DSA generation data file {dsa_file} not found.")
        return None

    def _nest():
        return defaultdict(_nest)

    dsa_data = _nest()

    first_row = next((row for source_rows in row_sources for row in source_rows), None)
    has_power = first_row is not None and "power" in first_row
    seen_coordinates = set()

    for source_rows in row_sources:
        # Preserve legacy last-row-wins behavior within each source.
        source_values = {}
        for row in source_rows:
            if ("skip_indexer" in (row.get("op_name") or "")) != (op_kind == "skip"):
                continue
            num_heads = int(row["num_heads"])
            b = int(row["batch_size"])
            s = int(row["isl"]) + int(row["step"])
            latency = float(row["latency"])
            power = float(row.get("power", 0.0)) if has_power else 0.0
            energy = power * latency

            arch = row.get("architecture", DEFAULT_DSA_ARCHITECTURE)
            gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
            kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

            ks = row.get("kernel_source") or ""
            # Total decode length is the canonical coordinate even if two rows
            # decompose it into different isl/step pairs.
            for dsa_backend in _dsa_kernel_source_buckets(ks, kv_dtype):
                coordinate = (kv_dtype, gemm_mode, arch, dsa_backend, num_heads, b, s)
                source_values[coordinate] = {
                    "latency": latency,
                    "power": power,
                    "energy": energy,
                }

        # Sources are priority-ordered: active first, shared fallbacks later.
        for coordinate, value in source_values.items():
            if coordinate in seen_coordinates:
                continue
            seen_coordinates.add(coordinate)
            kv_dtype, gemm_mode, arch, dsa_backend, num_heads, b, s = coordinate
            dsa_data[kv_dtype][gemm_mode][arch][dsa_backend][num_heads][b][s] = value

    return dsa_data
