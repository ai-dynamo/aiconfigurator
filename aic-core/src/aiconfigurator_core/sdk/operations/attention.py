# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context + Generation attention ops (ISSUE-06 / AIC-543).

Both classes own their CSV-backed perf tables, SOL correction (generation
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

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)

# Attention utilisation vs head_size, relative to head_size=128. Used to transfer
# util across head_size when the exact head_size has no collected data: borrow the
# nearest collected head_size's util curve and rescale by ratio[target]/ratio[ref].
#
# Measured on Blackwell (b200) GQA where 64/128/256 are collected; the cross-GPU
# spread within a backend is small, so a per-backend table is enough. PREFILL util
# rises with head_size (compute-bound: larger heads pack the tensor cores better);
# 192 is log2-interpolated and 512 EXTRAPOLATED under the observed diminishing-
# returns trend (util saturates) -- both are estimates, not measured. 512 is held
# conservatively close to 256 (only a small bump) since there is no data and util
# is near-saturated there; under-crediting its efficiency errs toward over- rather
# than under-estimating latency. DECODE util is ~head_size-independent (memory-bound
# KV read), so its ratio is 1.0 (no table).
_ATTN_PREFILL_HS_RATIO: dict[str, dict[int, float]] = {
    "trtllm": {64: 0.58, 128: 1.00, 192: 1.10, 256: 1.17, 512: 1.20},
    "sglang": {64: 0.60, 128: 1.00, 192: 1.18, 256: 1.32, 512: 1.38},
    "vllm": {64: 0.60, 128: 1.00, 192: 1.27, 256: 1.51, 512: 1.60},
}


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


def generation_attn_mode(system_spec: dict, kvcache_quant_mode: common.KVCacheQuantMode) -> common.FMHAQuantMode:
    """Decode-attention FMHA mode implied by the kv-cache dtype.

    fp8 KV implies an fp8-MMA decode kernel only where fp8 tensor cores exist
    (SM >= 89, Ada and newer); on pre-89 hardware — and on specs without
    ``sm_version``, e.g. XPU — the kernel dequantizes KV and issues the MMA on
    the bf16 pipeline. That is how a100's shipped fp8-kv generation data was
    collected in the first place, so gating here keeps that silicon usable
    under the strict per-dtype resolution. The single home for the derivation
    rule (mirrors Rust ``perf_database::attention::generation_attn_mode``).
    """
    has_fp8_mma = (system_spec["gpu"].get("sm_version") or -1) >= 89
    if kvcache_quant_mode == common.KVCacheQuantMode.fp8 and has_fp8_mma:
        return common.FMHAQuantMode.fp8
    return common.FMHAQuantMode.bfloat16


def generation_attn_flops(system_spec: dict, kvcache_quant_mode: common.KVCacheQuantMode) -> float:
    """Strictly resolved TC FLOPS for :func:`generation_attn_mode`; used by
    the generation get_sol closures, the eager query-entry checks, and
    ``Attention._correct_sol``.
    """
    return common.get_quant_tc_flops(system_spec, generation_attn_mode(system_spec, kvcache_quant_mode))


class ContextAttention(Operation):
    """
    Context (prefill) attention operation.

    Owns ``_data_cache: {key: LoadedOpData}`` for the context attention CSV.
    No SOL clamp on the loaded table (legacy ``_correct_data`` did not
    correct context attention) — only grid extrapolation runs in ``load_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
        cp_size: int = 1,
    ) -> None:
        """Initialize context attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm
        # Context parallelism (sglang AllGather, zigzag in-seq split). When
        # cp_size > 1, query() models ONE representative CP rank (rank 0): the
        # sequence is split into 2*cp contiguous chunks, rank 0 owns chunk 0
        # (prefix=0) and chunk 2*cp-1 (prefix=isl-c). Its two halves' work sums
        # to the balanced per-rank total isl^2/(2*cp). See
        # docs/CONTEXT_PARALLEL_DSA_MODELING.md (dense analog).
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_attention CSV into the class cache,
        applies grid extrapolation, binds ``database._context_attention_data``.

        Mirrors ``GEMM.load_data``: correction/extrapolation operate on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.context_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.context_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_attention_data(sources), PerfDataFilename.context_attention, primary_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via
            # the engine's interpolation (sqrt-space blend; the truncated large-seq x large-batch
            # corner is ordinary out-of-range util-hold).
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

    _ENGINE_QUERY_SHAPE = "context"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationAttention(Operation):
    """
    Generation (decode) attention operation.

    Owns an extrapolated SILICON working cache plus a raw measured cache for
    empirical utilization calibration. ``load_data`` applies SOL clamping,
    snapshots the measured rows, then expands the working grid.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        n: int,
        n_kv: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        window_size: int = 0,
        head_size: int = 128,
        use_qk_norm: bool = False,
    ) -> None:
        """Initialize generation attention query parameters."""
        super().__init__(name, scale_factor)
        self._n = n
        self._weights = 0.0
        self._n_kv = n_kv
        self._kv_cache_dtype = kv_cache_dtype
        self._window_size = window_size
        self._head_size = head_size
        self._use_qk_norm = use_qk_norm

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_attention CSV, clamps to SOL,
        preserves measured rows, extrapolates the working grid, and binds both
        database views.

        Mirrors ``GEMM.load_data``: correction/extrapolation operate on the
        canonical class-cache value (passed explicitly), then the instance
        attr is bound, respecting any pre-set test override."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.generation_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.generation_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_attention_data(sources), PerfDataFilename.generation_attention, primary_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW table
            # via the engine's interpolation, so the table IS the raw data (the former deepcopy
            # for _raw_generation_attention_data is now just an alias).
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
    # to its own load (see perf_database/gemm.rs), so QUERY values stay
    # SOL-floored via the single oracle.

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_attention)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract: query() + get_weights()
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class EncoderAttention(Operation):
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

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        head_size: int,
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        partial_rotary_factor: float = 0.0,
    ) -> None:
        super().__init__(name, scale_factor)
        # Encoder kernels currently only have bfloat16 perf data;
        if fmha_quant_mode != common.FMHAQuantMode.bfloat16:
            raise ValueError(f"EncoderAttention only supports FMHAQuantMode.bfloat16, got {fmha_quant_mode}")
        if not 0.0 <= partial_rotary_factor <= 1.0:
            raise ValueError(f"partial_rotary_factor must be in [0.0, 1.0], got {partial_rotary_factor}")
        self._n = num_heads
        self._head_size = head_size
        self._fmha_quant_mode = fmha_quant_mode
        self._partial_rotary_factor = partial_rotary_factor
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads encoder_attention CSV into the class cache,
        applies grid extrapolation, binds ``database._encoder_attention_data``.
        """
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.encoder_attention.value
            )
            sources = database._build_op_sources(PerfDataFilename.encoder_attention, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_encoder_attention_data(sources), PerfDataFilename.encoder_attention, primary_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
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

    _ENGINE_QUERY_SHAPE = "context"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def _log_attention_row_conflict(attention_kind, key, kept_provenance, dropped_row):
    """Warn only when two named kernel sources at the same version collapse to one key.

    First-wins overlap from earlier-version or legacy reuse sources is expected and
    remains at debug level.
    """
    kept_kernel_source, kept_version = kept_provenance
    dropped_kernel_source = dropped_row.get("kernel_source")
    dropped_version = dropped_row.get("version")
    is_backend_collision = (
        kept_kernel_source
        and dropped_kernel_source
        and kept_kernel_source != dropped_kernel_source
        and kept_version
        and kept_version == dropped_version
    )
    log = logger.warning if is_backend_collision else logger.debug
    log(
        f"value conflict in {attention_kind} attention data: {key} — keeping first row "
        f"(kernel_source={kept_kernel_source}, version={kept_version}), dropping later row "
        f"(kernel_source={dropped_kernel_source}, version={dropped_version})"
    )


def load_context_attention_data(context_attention_file):
    """
    Load the context attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(context_attention_file)
    if rows is None:
        logger.debug(f"Context attention data file {context_attention_file} not found.")
        return None
    context_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                )
            )
        )
    )
    context_attention_provenance = {}

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (context_attention) - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:  # catch potential error for backward comptability
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, latency = (
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]
        key = (quant_mode, kv_cache_dtype, kv_n, head_size, window_size, n, s, b)

        try:
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b]
        except KeyError:
            # Store all three values
            context_attention_data[quant_mode][kv_cache_dtype][kv_n][head_size][window_size][n][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }
            context_attention_provenance[key] = (row.get("kernel_source"), row.get("version"))
        else:
            _log_attention_row_conflict("context", " ".join(map(str, key)), context_attention_provenance[key], row)

    return context_attention_data


def load_generation_attention_data(generation_attention_file):
    """
    Load the generation attention data with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(generation_attention_file)
    if rows is None:
        logger.debug(f"Generation attention data file {generation_attention_file} not found.")
        return None
    generation_attention_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )
    generation_attention_provenance = {}

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (generation_attention) - power will default to 0.0")

    for row in rows:
        try:
            window_size = row["window_size"]
        except KeyError:
            window_size = 0
        quant_mode, kv_cache_dtype, b, s, n, kv_n, head_size, step, latency = (  # noqa: F841
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["step"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        kv_n = int(kv_n)
        head_size = int(head_size)
        window_size = int(window_size)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        # we only have kv_n==n(MHA) and kv_n==1,2,4,8(XQA), interp/extrap all other num_kv_heads.
        # Use kv_n = 0 to mean n_kv == n.
        kv_n = 0 if n == kv_n else kv_n
        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]
        key = (kv_cache_dtype, kv_n, head_size, window_size, n, b, s)

        try:
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s]
        except KeyError:
            # Store all three values
            generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }
            generation_attention_provenance[key] = (row.get("kernel_source"), row.get("version"))
        else:
            _log_attention_row_conflict(
                "generation", " ".join(map(str, key)), generation_attention_provenance[key], row
            )

    return generation_attention_data


def load_encoder_attention_data(encoder_attention_file):
    """
    Load the non-causal encoder attention data (ViT, audio encoder, etc.).

    Schema is intentionally simplified vs. context attention:
    - MHA only (n_kv == n), so no n_kv dimension
    - No KV cache (encoder is single-pass), so no kv_cache_dtype dimension
    - No sliding window, so no window_size dimension

    Returns:
        dict: Nested dict [fmha_quant_mode][head_size][n][s][b] -> {latency, power, energy}.
    """
    rows = _read_filtered_rows(encoder_attention_file)
    if rows is None:
        logger.debug(f"Encoder attention data file {encoder_attention_file} not found.")
        return None
    encoder_attention_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (encoder_attention) - power will default to 0.0")

    for row in rows:
        quant_mode, b, s, n, head_size, latency = (
            row["attn_dtype"],
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["head_dim"],
            row["latency"],
        )
        b = int(b)
        s = int(s)
        n = int(n)
        head_size = int(head_size)
        latency = float(latency)

        power = float(row.get("power", 0.0))
        energy = power * latency

        quant_mode = common.FMHAQuantMode[quant_mode]

        try:
            encoder_attention_data[quant_mode][head_size][n][s][b]
            logger.debug(f"value conflict in encoder attention data: {quant_mode} {head_size} {n} {s} {b}")
        except KeyError:
            encoder_attention_data[quant_mode][head_size][n][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return encoder_attention_data
