# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLA (Multi-head Latent Attention) family (ISSUE-08 / AIC-540).

Six op classes migrate from ``_legacy.py`` into ``operations/mla.py``:

- ``ContextMLA`` / ``GenerationMLA`` — regular MLA ops; own
  ``_context_mla_data`` / ``_generation_mla_data`` respectively. Both
  delegate to ``PerfDatabase.query_context_mla`` / ``query_generation_mla``
  which become one-line forwards.
- ``MLABmm`` — pre/post BMM op for MLA decoding. Owns ``_mla_bmm_data``.
- ``MLAModule`` — module-level MLA (both context and generation in one
  class, dispatched by ``is_context`` flag). Owns BOTH
  ``_context_mla_module_data`` AND ``_generation_mla_module_data`` since
  ``MLAModule.query`` chooses between them at runtime.
- ``WideEPContextMLA`` / ``WideEPGenerationMLA`` — SGLang-only variants.
  Their CSV tables are loaded only when ``backend == "sglang"`` (matching
  the legacy conditional ``if backend == "sglang"`` block in
  ``PerfDatabase.__init__``).

No SOL clamping for any MLA variant in the legacy ``_correct_data``.
Extrapolation present for all 4 regular + 2 module variants + 2 WideEP
variants (the WideEP variants extrapolate only when their data was
loaded — SGLang-only).

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``. For
WideEP variants, ``backend`` in the key naturally encodes the SGLang
constraint (cache misses on non-SGLang backends).
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


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family.

    TODO: hoist to ``operations/base.py`` once Phase 3 settles (7 op
    families duplicating this helper now).
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# Native-head pin for MLA *module* tables (#1458). Module rows are single-GPU
# rank-local head sweeps (``tp_size`` is provenance, hardcoded 1 by the module
# collectors), so native CANNOT be derived as ``num_heads * tp_size`` — it
# comes from the ``model`` column via this pin. Unknown models fail the load;
# extending this map (and its Rust twin) is part of landing new module data.
_MLA_MODULE_NATIVE_HEADS = {
    "deepseek-ai/DeepSeek-V3": 128,
    # vllm 0.22.0 provenance aliases of the same 128-native DSV3 geometry —
    # they collapse into one bucket (first source wins).
    "deepseek-ai/DeepSeek-R1": 128,
    "nvidia/DeepSeek-V3.1-NVFP4": 128,
}


def _mla_module_native_heads(row: dict, mla_module_file, num_heads: int) -> int:
    """Native identity of one module row via the model pin; fails loud on
    missing/unpinned models. tp>1 rows must be rank-local (heads * tp ==
    native — the #1429 stale fingerprint, checked per row)."""
    model = str(row.get("model", "") or "")
    if not model:
        raise ValueError(
            f"MLA module row in {mla_module_file} carries no model column; the module "
            f"table keys its native-head identity off the model pin (#1458)."
        )
    native_heads = _MLA_MODULE_NATIVE_HEADS.get(model)
    if native_heads is None:
        raise ValueError(
            f"MLA module row in {mla_module_file} names unpinned model {model!r}; add its "
            f"native head count to _MLA_MODULE_NATIVE_HEADS when landing the data (#1458)."
        )
    tp_size = max(1, int(row.get("tp_size", 1) or 1))
    if tp_size > 1 and num_heads * tp_size != native_heads:
        raise ValueError(
            f"MLA module row in {mla_module_file} for model {model!r} has "
            f"num_heads={num_heads} at tp_size={tp_size}, inconsistent with native "
            f"{native_heads} (num_heads must be rank-local, #1429/#1458)."
        )
    return native_heads


# fmt: on


class ContextMLA(Operation):
    """
    Context MLA operation. Owns ``_context_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        # Context parallelism (sglang AllGather zigzag in-seq split). When cp>1,
        # query() models CP rank 0's two zigzag chunks (prev: prefix..+c; next:
        # prefix+isl-c..isl), same as ContextAttention. c = ceil(isl/(2*cp)).
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_mla CSV, applies extrapolation, binds
        ``database._context_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.context_mla.value
            )
            sources = database._build_op_sources(PerfDataFilename.context_mla, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_mla_data(sources), PerfDataFilename.context_mla, primary_path
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
            cls._record_load()

        if "_context_mla_data" not in database.__dict__:
            database._context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationMLA(Operation):
    """
    Generation MLA operation (MQA part). Owns ``_generation_mla_data``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._kv_cache_dtype = kv_cache_dtype

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_mla CSV, applies extrapolation, binds
        ``database._generation_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.generation_mla.value
            )
            sources = database._build_op_sources(PerfDataFilename.generation_mla, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_mla_data(sources), PerfDataFilename.generation_mla, primary_path
            )
            # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
            cls._record_load()

        if "_generation_mla_data" not in database.__dict__:
            database._generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLABmm(Operation):
    """
    MLABmm operation — pre/post BMM for MLA decoding. Owns ``_mla_bmm_data``.
    No extrapolation in the legacy ``__init__`` path; data is 1D-keyed by
    num_tokens within each (quant_mode, op_name, num_heads) bucket.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        quant_mode: common.GEMMQuantMode,
        if_pre: bool = True,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._weights = 0.0
        self._quant_mode = quant_mode
        self._if_pre = if_pre

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads mla_bmm CSV, binds ``database._mla_bmm_data``.
        No extrapolation (1D table)."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_bmm.value
            )
            sources = database._build_op_sources(PerfDataFilename.mla_bmm, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_mla_bmm_data(sources), PerfDataFilename.mla_bmm, primary_path)
            cls._record_load()

        if "_mla_bmm_data" not in database.__dict__:
            database._mla_bmm_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mla_bmm)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

    def _engine_query_plan(self, kwargs: dict):
        """Legacy signature has no ``s``: the BMM shape is batch-only."""
        beam_width = kwargs.get("beam_width", 1)
        if beam_width != 1:
            raise ValueError(f"{type(self).__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        if batch_size is None:
            raise ValueError(f"{type(self).__name__}.query requires 'batch_size'.")
        return self, {
            "is_context": False,
            "batch_size": int(batch_size),
            "s": int(kwargs.get("s", 1) or 1),
        }

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class MLAModule(Operation):
    """
    Module-level MLA op for both context and generation phases.

    Owns BOTH ``_context_mla_module_data`` (via ``_context_data_cache``)
    AND ``_generation_mla_module_data`` (via ``_generation_data_cache``)
    because ``query()`` chooses between them at runtime based on the
    ``is_context`` flag.

    Models the complete MLA attention block as a single profiled operation.
    For context: replaces q_b_proj + kv_b_proj + ContextMLA + proj.
    For generation: replaces MLABmm(pre) + GenerationMLA + MLABmm(post).
    """

    _context_data_cache: ClassVar[dict] = {}
    _generation_data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        is_context: bool,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        native_num_heads: int | None = None,
    ) -> None:
        super().__init__(name, scale_factor)
        self._is_context = is_context
        self._num_heads = num_heads
        # Model-native identity for the [native][local] module table (#1458).
        self._native_num_heads = native_num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership — two tables, one per phase
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads BOTH context and generation module CSVs,
        applies extrapolation to each, binds
        ``database._context_mla_module_data`` and
        ``database._generation_mla_module_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._context_data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])

            context_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_context_module.value
            )
            context_sources = database._build_op_sources(
                PerfDataFilename.mla_context_module, context_path, system_data_root
            )
            cls._context_data_cache[key] = LoadedOpData(
                load_context_mla_module_data(context_sources), PerfDataFilename.mla_context_module, context_path
            )

            gen_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mla_generation_module.value
            )
            gen_sources = database._build_op_sources(PerfDataFilename.mla_generation_module, gen_path, system_data_root)
            cls._generation_data_cache[key] = LoadedOpData(
                load_generation_mla_module_data(gen_sources), PerfDataFilename.mla_generation_module, gen_path
            )

            # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
            cls._record_load()

        if "_context_mla_module_data" not in database.__dict__:
            database._context_mla_module_data = cls._context_data_cache[key]
        if "_generation_mla_module_data" not in database.__dict__:
            database._generation_mla_module_data = cls._generation_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._context_data_cache.clear()
        cls._generation_data_cache.clear()

    # ------------------------------------------------------------------
    # Query tables (formerly PerfDatabase.query_context_mla_module /
    # query_generation_mla_module)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "module"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPGenerationMLA(Operation):
    """
    WideEP Generation MLA operation (SGLang-only). Owns
    ``_wideep_generation_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads wideep_generation_mla CSV (SGLang only),
        applies extrapolation, binds ``database._wideep_generation_mla_data``.

        Non-SGLang backends get ``None`` (matching the legacy
        ``if backend == "sglang"`` guard in ``__init__``)."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary_path = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_generation_mla.value
                )
                sources = database._build_op_sources(
                    PerfDataFilename.wideep_generation_mla, primary_path, system_data_root
                )
                cls._data_cache[key] = LoadedOpData(
                    load_wideep_generation_mla_data(sources),
                    PerfDataFilename.wideep_generation_mla,
                    primary_path,
                )
                # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
            cls._record_load()

        if "_wideep_generation_mla_data" not in database.__dict__:
            database._wideep_generation_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_generation_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "generation"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class WideEPContextMLA(Operation):
    """
    WideEP Context MLA operation (SGLang-only). Owns
    ``_wideep_context_mla_data``. Loaded only when ``backend == "sglang"``.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        tp_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        attn_backend: str = "flashinfer",
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._tp_size = tp_size
        self._weights = 0.0
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._attn_backend = attn_backend
        # CP (sglang AllGather zigzag); see ContextMLA. cp>1 -> rank-0 two-chunk.
        self._cp_size = cp_size

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads wideep_context_mla CSV (SGLang only),
        applies extrapolation, binds ``database._wideep_context_mla_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            if database.backend != "sglang":
                cls._data_cache[key] = None
            else:
                system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
                primary_path = resolve_op_data_path(
                    system_data_root, database.backend, database.version, PerfDataFilename.wideep_context_mla.value
                )
                sources = database._build_op_sources(
                    PerfDataFilename.wideep_context_mla, primary_path, system_data_root
                )
                cls._data_cache[key] = LoadedOpData(
                    load_wideep_context_mla_data(sources),
                    PerfDataFilename.wideep_context_mla,
                    primary_path,
                )
                # No load-time grid pre-expansion: queries resolve on the RAW grid via the engine's interpolation.
            cls._record_load()

        if "_wideep_context_mla_data" not in database.__dict__:
            database._wideep_context_mla_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_wideep_context_mla)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    _ENGINE_QUERY_SHAPE = "context"

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_context_mla_data(context_mla_file):
    """
    Load the context mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(context_mla_file)
    if rows is None:
        logger.debug(f"Context mla data file {context_mla_file} not found.")
        return None
    context_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (context_mla) - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        if "num_heads" not in row:
            # Retired ``128 // tp_size`` backfill: it silently mislabeled any
            # non-128-native MLA model (#1458). Fail like the DSV4 loaders do.
            raise ValueError(
                f"context MLA row in {context_mla_file} carries no num_heads column; "
                f"the rank-local head count is mandatory (#1458). Migrate the file: "
                f"num_heads = model_native_heads // tp_size."
            )
        num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(f"value conflict in context mla data: {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}")
        except KeyError:
            # Store all three values
            context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return context_mla_data


def load_generation_mla_data(generation_mla_file):
    """
    Load the generation mla data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(generation_mla_file)
    if rows is None:
        logger.debug(f"Generation mla data file {generation_mla_file} not found.")
        return None
    generation_mla_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (generation_mla) - power will default to 0.0")

    for row in rows:
        quant_mode, kv_cache_dtype, b, s, step, latency = (  # noqa: F841
            row["mla_dtype"],
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        if "num_heads" not in row:
            # Retired ``128 // tp_size`` backfill — see load_context_mla_data.
            raise ValueError(
                f"generation MLA row in {generation_mla_file} carries no num_heads column; "
                f"the rank-local head count is mandatory (#1458). Migrate the file: "
                f"num_heads = model_native_heads // tp_size."
            )
        num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            generation_mla_data[kv_cache_dtype][num_heads][b][s]
            logger.debug(f"value conflict in generation mla data: {kv_cache_dtype} {num_heads} {b} {s} ")
        except KeyError:
            # Store all three values
            generation_mla_data[kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return generation_mla_data


def load_mla_bmm_data(mla_bmm_file):
    """
    Load the mla bmm data for trtllm with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(mla_bmm_file)
    if rows is None:
        logger.debug(f"MLA BMM data file {mla_bmm_file} not found.")
        return None
    mla_bmm_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (mla_bmm) - power will default to 0.0")

    for row in rows:
        quant_mode, num_tokens, num_heads, latency, op_name = (
            row["bmm_dtype"],
            row["num_tokens"],
            row["num_heads"],
            row["latency"],
            row["op_name"],
        )
        num_tokens = int(num_tokens)
        num_heads = int(num_heads)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.GEMMQuantMode[quant_mode]

        try:
            # Check for conflict
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens]
            logger.debug(f"value conflict in mla bmm data: {op_name} {quant_mode} {num_heads} {num_tokens} ")
        except KeyError:
            # Store all three values
            mla_bmm_data[quant_mode][op_name][num_heads][num_tokens] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return mla_bmm_data


def load_wideep_context_mla_data(wideep_context_mla_file):
    """
    Load the SGLang WideEP context MLA data from wideep_context_mla_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_context_mla_file)
    if rows is None:
        logger.debug(f"SGLang wideep context mla data file {wideep_context_mla_file} not found.")
        return None
    wideep_context_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_context_mla) - power will default to 0.0")

    for row in rows:
        (
            quant_mode,
            kv_cache_dtype,
            b,
            s,
            latency,
        ) = row["mla_dtype"], row["kv_cache_dtype"], row["batch_size"], row["isl"], row["latency"]

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            # Retired ``128 // tp_size`` backfill — see load_context_mla_data.
            raise ValueError(
                f"WideEP context MLA row in {wideep_context_mla_file} carries no num_heads "
                f"column; the rank-local head count is mandatory (#1458). Migrate the file: "
                f"num_heads = model_native_heads // tp_size."
            )
        num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        quant_mode = common.FMHAQuantMode[quant_mode]
        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b]
            logger.debug(
                f"value conflict in context mla data: {kernel_source} {quant_mode} {kv_cache_dtype} {num_heads} {s} {b}"
            )
        except KeyError:
            # Store all three values
            wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_context_mla_data


def load_wideep_generation_mla_data(wideep_generation_mla_file):
    """
    Load the SGLang WideEP generation MLA data from wideep_generation_mla_perf.parquet
    with power support (backward compatible).

    Returns:
        dict: Nested dict structure where leaf values are dicts with 'latency' and 'power' keys.
    """
    rows = _read_filtered_rows(wideep_generation_mla_file)
    if rows is None:
        logger.debug(f"SGLang wideep generation mla data file {wideep_generation_mla_file} not found.")
        return None
    wideep_generation_mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
    )

    # Check if power columns exist (backward compatibility)
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (wideep_generation_mla) - power will default to 0.0")

    for row in rows:
        kv_cache_dtype, b, s, step, latency = (
            row["kv_cache_dtype"],
            row["batch_size"],
            row["isl"],
            row["step"],
            row["latency"],
        )

        kernel_source = row.get("kernel_source", "flashinfer")

        if "num_heads" not in row:
            # Retired ``128 // tp_size`` backfill — see load_context_mla_data.
            raise ValueError(
                f"WideEP generation MLA row in {wideep_generation_mla_file} carries no "
                f"num_heads column; the rank-local head count is mandatory (#1458). Migrate "
                f"the file: num_heads = model_native_heads // tp_size."
            )
        num_heads = int(row["num_heads"])

        b = int(b)
        s = int(s)
        step = int(step)
        latency = float(latency)

        # NEW: Read power with backward compatibility
        power = float(row.get("power", 0.0))

        # NEW: Calculate energy from power and latency
        energy = power * latency  # watt-milliseconds

        s = s + step

        kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype]

        try:
            # Check for conflict
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s]
            logger.debug(
                f"value conflict in generation mla data: {kernel_source} {kv_cache_dtype} {num_heads} {b} {s} "
            )
        except KeyError:
            # Store all three values
            wideep_generation_mla_data[kernel_source][kv_cache_dtype][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,  # NEW: precomputed energy
            }

    return wideep_generation_mla_data


def load_context_mla_module_data(mla_module_file: str):
    """
    Load context MLA module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source, model,
    architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
    batch_size, isl, tp_size, step, latency [, power]

    Dict structure (#1458 — native level between quant keys and the local
    head-sweep axis; native is the model identity from ``model`` via
    ``_MLA_MODULE_NATIVE_HEADS``, num_heads stays the rank-local interp axis):
        data[fmha_quant_mode][kv_cache_quant_mode][gemm_quant_mode][native][num_heads][s][b]
    """
    rows = _read_filtered_rows(mla_module_file)
    if rows is None:
        logger.debug(f"MLA context module data file {mla_module_file} not found.")
        return None

    mla_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        num_heads = int(row["num_heads"])
        native_heads = _mla_module_native_heads(row, mla_module_file, num_heads)
        b = int(row["batch_size"])
        s = int(row["isl"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]

        try:
            # Check for conflict: first source wins (shared-layer contract,
            # _read_filtered_rows orders primary before sibling fallbacks).
            mla_data[fmha_mode][kv_dtype][gemm_mode][native_heads][num_heads][s][b]
            logger.debug(
                f"value conflict in context mla module data: {fmha_mode} {kv_dtype} {gemm_mode} "
                f"{native_heads} {num_heads} {s} {b}"
            )
        except KeyError:
            mla_data[fmha_mode][kv_dtype][gemm_mode][native_heads][num_heads][s][b] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return mla_data


def load_generation_mla_module_data(mla_module_file: str):
    """
    Load generation MLA module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source, model,
    architecture, mla_dtype, kv_cache_dtype, gemm_type, num_heads,
    batch_size, isl, tp_size, step, latency [, power]

    Dict structure (#1458 — native level between quant keys and the local
    head-sweep axis, same as the context loader):
        data[kv_cache_quant_mode][gemm_quant_mode][native][num_heads][b][s]

    The ``mla_dtype`` column is ignored: decode MLA compute dtype follows the
    KV cache dtype (collectors hardcode ``bfloat16`` in that column), so it is
    not a real axis — mirroring ``load_generation_mla_data``, which likewise
    drops it.
    """
    rows = _read_filtered_rows(mla_module_file)
    if rows is None:
        logger.debug(f"MLA generation module data file {mla_module_file} not found.")
        return None

    mla_data = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
    )

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        num_heads = int(row["num_heads"])
        native_heads = _mla_module_native_heads(row, mla_module_file, num_heads)
        b = int(row["batch_size"])
        s = int(row["isl"]) + int(row["step"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

        try:
            # Check for conflict: first source wins (shared-layer contract,
            # _read_filtered_rows orders primary before sibling fallbacks).
            mla_data[kv_dtype][gemm_mode][native_heads][num_heads][b][s]
            logger.debug(
                f"value conflict in generation mla module data: {kv_dtype} {gemm_mode} "
                f"{native_heads} {num_heads} {b} {s}"
            )
        except KeyError:
            mla_data[kv_dtype][gemm_mode][native_heads][num_heads][b][s] = {
                "latency": latency,
                "power": power,
                "energy": energy,
            }

    return mla_data
