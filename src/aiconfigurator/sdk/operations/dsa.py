# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) module-level ops (ISSUE-10 / AIC-538).

Both ContextDSAModule and GenerationDSAModule own their CSV-backed perf
tables and grid extrapolation. ``PerfDatabase.query_context_dsa_module``
and ``query_generation_dsa_module`` delegate here.

ContextDSAModule additionally maintains a ``_raw_data_cache`` — a
``copy.deepcopy`` of the loaded table BEFORE extrapolation runs — because
``interpolation.interp_dsa_context_topk_piecewise_from_raw`` needs the
un-extrapolated rows for the topk-boundary regime-aware piecewise lookup
(PR #903).

No SOL clamping in the legacy ``_correct_data`` for either DSA op —
extrapolation only. The legacy ``__init__`` loaded DSA twice (once near
the MLA/Mamba block, once after); both loads are consolidated into a
single ``load_data`` call per class.

DSA-specific helpers (``_is_dsa_interpolation_miss``,
``_format_dsa_unavailable_message``) also move here as module-level
functions. ``DSA_MODEL_DIMS`` and ``DEFAULT_DSA_ARCHITECTURE`` stay on
``perf_database.py`` as module-level constants for now — the cleanup PR
revisits their home.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.operations.base import Operation
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# Extrapolation grids — lifted verbatim from the legacy blocks in
# ``PerfDatabase.__init__``.

# fmt: off
_CONTEXT_DSA_TARGET_Y: list[int] = (
    [1, 16, 32, 64, 128, 256, 512, 1024, 2048]
    + [4096 + i * 2048 for i in range(14)]
    + [32768 + 16384 * i for i in range(6)]
    + [131072 + 32768 * i for i in range(12)]
    + [524288 + 65536 * i for i in range(9)]
)  # s
_CONTEXT_DSA_TARGET_Z: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048,
]  # b

_GENERATION_DSA_TARGET_Y: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 1024, 2048, 8192,
]  # b
_GENERATION_DSA_TARGET_Z: list[int] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    16384, 32768, 65536, 131072, 262144, 2097152 * 8,
]  # s
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


def _is_dsa_interpolation_miss(error: Exception) -> bool:
    """Recognize the ``ValueError`` patterns raised by
    ``interpolation.nearest_1d_point_helper`` when DSA shape is outside
    the sampled grid. Lifted verbatim from
    ``PerfDatabase._is_dsa_interpolation_miss``."""
    message = str(error)
    return isinstance(error, ValueError) and (
        "x is not equal to the only value in the list" in message
        or "x is less than the smallest value in the list" in message
        or "x is greater than the largest value in the list" in message
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

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads context_dsa_module CSV, deepcopies the raw
        version, applies grid extrapolation to the main cache, binds
        ``database._context_dsa_module_data`` and
        ``database._raw_context_dsa_module_data``."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename, load_context_dsa_module_data

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.dsa_context_module.value)
            sources = database._build_op_sources(PerfDataFilename.dsa_context_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_dsa_module_data(sources), PerfDataFilename.dsa_context_module, primary_path
            )
            # Deepcopy BEFORE extrapolation so the raw rows survive intact
            # for ``interp_dsa_context_topk_piecewise_from_raw``.
            cls._raw_data_cache[key] = copy.deepcopy(cls._data_cache[key])
            cls._extrapolate(cls._data_cache[key])
            cls._record_load()

        if "_context_dsa_module_data" not in database.__dict__:
            database._context_dsa_module_data = cls._data_cache[key]
        if "_raw_context_dsa_module_data" not in database.__dict__:
            database._raw_context_dsa_module_data = cls._raw_data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()

    @classmethod
    def _extrapolate(cls, data_wrapper) -> None:
        """Apply the legacy 4-level (fmha_mode → kv_cache_dtype → gemm_mode
        → arch → grid) extrapolation."""
        if data_wrapper is None or not getattr(data_wrapper, "loaded", False):
            return

        for fmha_mode in data_wrapper:
            for kv_cache_dtype in data_wrapper[fmha_mode]:
                for gemm_mode in data_wrapper[fmha_mode][kv_cache_dtype]:
                    for arch in data_wrapper[fmha_mode][kv_cache_dtype][gemm_mode]:
                        data_dict = data_wrapper[fmha_mode][kv_cache_dtype][gemm_mode][arch]
                        num_heads_list = list(data_dict.keys())
                        interpolation.extrapolate_data_grid(
                            data_dict=data_dict,
                            target_x_list=num_heads_list,
                            target_y_list=_CONTEXT_DSA_TARGET_Y,
                            target_z_list=_CONTEXT_DSA_TARGET_Z,
                        )

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_dsa_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_dsa_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        prefix: int = 0,
        architecture: str | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
    ):
        """Query context DSA module table. Verbatim port of the legacy body."""
        from aiconfigurator.sdk.perf_database import DEFAULT_DSA_ARCHITECTURE, DSA_MODEL_DIMS, PerfDataNotAvailableError

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE

        dims = DSA_MODEL_DIMS.get(architecture, DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE])
        hidden_size = dims["hidden_size"]
        q_lora = dims["q_lora_rank"]
        kv_lora = dims["kv_lora_rank"]
        qk_nope = dims["qk_nope_head_dim"]
        qk_rope = dims["qk_rope_head_dim"]
        v_dim = dims["v_head_dim"]
        if index_n_heads is None:
            index_n_heads = dims["index_n_heads"]
        if index_head_dim is None:
            index_head_dim = dims["index_head_dim"]
        if index_topk is None:
            index_topk = dims["index_topk"]
        qk_head_dim = qk_nope + qk_rope
        attn_head_dim = kv_lora + qk_rope

        def get_sol(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> tuple[float, float, float]:
            """SOL estimate for the full DSA context attention block.

            Ops are split into two groups with different throughput/memory:
              - GEMM group (linear projections + absorption BMMs): gemm_quant_mode
              - Attention group (indexer logits + sparse MLA): fmha_quant_mode
            """
            full_s = s + prefix
            tokens = b * s

            # ── Compute (FLOPs) ─────────────────────────────────────────
            proj_out = q_lora + kv_lora + qk_rope + index_head_dim

            gemm_group_ops = (
                2 * tokens * hidden_size * proj_out
                + 2 * tokens * q_lora * (num_heads * qk_head_dim)
                + 2 * tokens * q_lora * (index_n_heads * index_head_dim)
                + 2 * tokens * hidden_size * index_n_heads
                + 2 * tokens * (num_heads * v_dim) * hidden_size
                + 2 * num_heads * tokens * qk_nope * kv_lora
                + 2 * num_heads * tokens * kv_lora * v_dim
            )

            # Indexer logits group — always FP8 (hardcoded in both vLLM and TRT-LLM)
            if full_s <= index_topk:
                indexer_logits_ops = 0
            else:
                indexer_logits_ops = 2 * tokens * index_n_heads * index_head_dim * full_s

            # Sparse MLA attention group — throughput governed by fmha_quant_mode
            effective_kv = min(full_s, index_topk)
            # Exact KV pair count: sum_{i=0..s-1} min(prefix+i+1, topk)
            if full_s <= index_topk:
                total_kv_pairs = b * (full_s * (full_s + 1) - prefix * (prefix + 1)) // 2
            elif prefix >= index_topk:
                total_kv_pairs = tokens * index_topk
            else:
                ramp_pairs = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) // 2
                sat_pairs = b * (full_s - index_topk) * index_topk
                total_kv_pairs = ramp_pairs + sat_pairs
            sparse_attn_ops = 2 * num_heads * (attn_head_dim + kv_lora) * total_kv_pairs

            # ── Memory (bytes) ──────────────────────────────────────────
            gemm_weight_bytes = (
                hidden_size * proj_out
                + q_lora * num_heads * qk_head_dim
                + q_lora * index_n_heads * index_head_dim
                + hidden_size * index_n_heads
                + num_heads * v_dim * hidden_size
            ) * gemm_quant_mode.value.memory

            kv_cache_bytes = b * num_heads * effective_kv * attn_head_dim * kvcache_quant_mode.value.memory
            indexer_entry_bytes = common.indexer_cache_entry_bytes(index_head_dim)
            indexer_cache_bytes = 0 if full_s <= index_topk else b * full_s * indexer_entry_bytes
            q_io_bytes = tokens * num_heads * qk_head_dim * fmha_quant_mode.value.memory * 2

            total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes

            # ── SOL ─────────────────────────────────────────────────────
            gemm_flops = database._get_quant_tc_flops(gemm_quant_mode)
            indexer_fp8_flops = database._get_quant_tc_flops(common.FMHAQuantMode.fp8)
            attn_flops = database._get_quant_tc_flops(fmha_quant_mode)

            sol_math = (
                gemm_group_ops / gemm_flops + indexer_logits_ops / indexer_fp8_flops + sparse_attn_ops / attn_flops
            ) * 1000
            sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(
            b: int,
            s: int,
            prefix: int,
            num_heads: int,
            kvcache_quant_mode: common.KVCacheQuantMode,
            fmha_quant_mode: common.FMHAQuantMode,
        ) -> float:
            latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            scale_factor = 0.5
            return latency / scale_factor

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)

        def missing_context_dsa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Context DSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"fmha_quant_mode={fmha_quant_mode}, kvcache_quant_mode={kvcache_quant_mode}, "
                f"gemm_quant_mode={gemm_quant_mode}, num_heads={num_heads}, s={s}, prefix={prefix}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        try:
            dsa_module_data = database._context_dsa_module_data
            if dsa_module_data is None:
                raise PerfDataNotAvailableError(
                    f"Context DSA module perf data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )
            try:
                dsa_dict = dsa_module_data[fmha_quant_mode][kvcache_quant_mode][gemm_quant_mode][architecture]
            except (KeyError, TypeError) as exc:
                raise missing_context_dsa_error() from exc
            full_s = s + prefix
            raw_dsa_dict = None
            raw_dsa_module_data = database._raw_context_dsa_module_data
            if raw_dsa_module_data is not None and getattr(raw_dsa_module_data, "loaded", True):
                try:
                    raw_dsa_dict = raw_dsa_module_data[fmha_quant_mode][kvcache_quant_mode][gemm_quant_mode][
                        architecture
                    ]
                except (KeyError, TypeError):
                    raw_dsa_dict = None
            try:
                result = interpolation.interp_dsa_context_topk_piecewise_from_raw(
                    num_heads, full_s, b, raw_dsa_dict, index_topk
                )
                if result is None:
                    result = database._interp_3d(num_heads, full_s, b, dsa_dict, "cubic")
                latency = result["latency"]
                energy = result.get("energy", 0.0)
            except (KeyError, TypeError, AssertionError) as exc:
                raise missing_context_dsa_error() from exc
            if prefix > 0:
                base_sol = get_sol(b, full_s, 0, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
                target_sol = get_sol(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)[0]
                correction = 1.0 if base_sol <= 0 else target_sol / base_sol
                latency *= correction
                energy *= correction
            return database._interp_pr(latency, energy=energy)
        except Exception as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query context DSA module for {b=}, {s=}, {prefix=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}; using empirical"
                )
                latency = get_empirical(b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode)
                return PerformanceResult(latency, energy=0.0, source="empirical")
            if isinstance(e, PerfDataNotAvailableError):
                logger.warning(str(e))
                raise
            if _is_dsa_interpolation_miss(e):
                message = _format_dsa_unavailable_message(
                    "Context",
                    e,
                    b=b,
                    s=s,
                    prefix=prefix,
                    num_heads=num_heads,
                    architecture=architecture,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    index_topk=index_topk,
                )
                logger.warning(message)
                raise PerfDataNotAvailableError(message) from None
            else:
                logger.exception(
                    f"Failed to query context DSA module for {b=}, {s=}, {prefix=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}, "
                    f"{kvcache_quant_mode=}, {fmha_quant_mode=}, {database_mode=}."
                )
                raise

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query context DSA latency with energy data."""
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)

        result = database.query_context_dsa_module(
            b=batch_size,
            s=isl,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._architecture,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationDSAModule(Operation):
    """
    Generation phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns ``_data_cache`` (extrapolated generation_dsa_module CSV). No
    ``_raw_data_cache`` because the generation path doesn't use the
    topk-boundary piecewise interpolation — straight 3D cubic.

    Models the full DSA attention block during decode:
    - Same components as ContextDSAModule
    - Uses paged MQA logits for indexer
    - Sparse MLA with KV cache lookup
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        architecture: str = "DeepseekV32ForCausalLM",
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kv_cache_dtype = kv_cache_dtype
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads generation_dsa_module CSV, applies grid
        extrapolation, binds ``database._generation_dsa_module_data``."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename, load_generation_dsa_module_data

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.dsa_generation_module.value)
            sources = database._build_op_sources(PerfDataFilename.dsa_generation_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_dsa_module_data(sources), PerfDataFilename.dsa_generation_module, primary_path
            )
            cls._extrapolate(cls._data_cache[key])
            cls._record_load()

        if "_generation_dsa_module_data" not in database.__dict__:
            database._generation_dsa_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _extrapolate(cls, data_wrapper) -> None:
        """Apply the legacy 3-level (kv_cache_dtype → gemm_mode → arch
        → grid) extrapolation."""
        if data_wrapper is None or not getattr(data_wrapper, "loaded", False):
            return

        for kv_cache_dtype in data_wrapper:
            for gemm_mode in data_wrapper[kv_cache_dtype]:
                for arch in data_wrapper[kv_cache_dtype][gemm_mode]:
                    data_dict = data_wrapper[kv_cache_dtype][gemm_mode][arch]
                    tp_list = list(data_dict.keys())
                    interpolation.extrapolate_data_grid(
                        data_dict=data_dict,
                        target_x_list=tp_list,
                        target_y_list=_GENERATION_DSA_TARGET_Y,
                        target_z_list=_GENERATION_DSA_TARGET_Z,
                    )

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_dsa_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_dsa_module_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        kv_cache_dtype: common.KVCacheQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        architecture: str | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
    ):
        """Query generation DSA module table. Verbatim port of the legacy body."""
        from aiconfigurator.sdk.perf_database import DEFAULT_DSA_ARCHITECTURE, DSA_MODEL_DIMS, PerfDataNotAvailableError

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE

        dims = DSA_MODEL_DIMS.get(architecture, DSA_MODEL_DIMS[DEFAULT_DSA_ARCHITECTURE])
        hidden_size = dims["hidden_size"]
        q_lora = dims["q_lora_rank"]
        kv_lora = dims["kv_lora_rank"]
        qk_nope = dims["qk_nope_head_dim"]
        qk_rope = dims["qk_rope_head_dim"]
        v_dim = dims["v_head_dim"]
        if index_n_heads is None:
            index_n_heads = dims["index_n_heads"]
        if index_head_dim is None:
            index_head_dim = dims["index_head_dim"]
        if index_topk is None:
            index_topk = dims["index_topk"]
        qk_head_dim = qk_nope + qk_rope
        attn_head_dim = kv_lora + qk_rope

        def get_sol(
            b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode
        ) -> tuple[float, float, float]:
            """SOL estimate for generation DSA module (1 token per request).

            Ops split into GEMM group (gemm_quant_mode) and attention group
            (fmha derived from kv_cache_dtype).
            """
            fmha_mode = common.FMHAQuantMode.bfloat16

            tokens = b
            proj_out = q_lora + kv_lora + qk_rope + index_head_dim
            effective_kv = min(s, index_topk)

            gemm_group_ops = (
                2 * tokens * hidden_size * proj_out
                + 2 * tokens * q_lora * num_heads * qk_head_dim
                + 2 * tokens * q_lora * index_n_heads * index_head_dim
                + 2 * tokens * hidden_size * index_n_heads
                + 2 * tokens * num_heads * v_dim * hidden_size
                + 2 * num_heads * tokens * qk_nope * kv_lora
                + 2 * num_heads * tokens * kv_lora * v_dim
            )

            indexer_logits_ops = 2 * tokens * index_n_heads * index_head_dim * s
            sparse_attn_ops = 2 * tokens * num_heads * (attn_head_dim + kv_lora) * effective_kv

            gemm_weight_bytes = (
                hidden_size * proj_out
                + q_lora * num_heads * qk_head_dim
                + q_lora * index_n_heads * index_head_dim
                + hidden_size * index_n_heads
                + num_heads * v_dim * hidden_size
            ) * gemm_quant_mode.value.memory
            indexer_entry_bytes = common.indexer_cache_entry_bytes(index_head_dim)
            indexer_cache_bytes = b * s * indexer_entry_bytes
            kv_cache_bytes = b * effective_kv * attn_head_dim * kv_cache_dtype.value.memory
            total_mem = gemm_weight_bytes + indexer_cache_bytes + kv_cache_bytes

            gemm_flops = database._get_quant_tc_flops(gemm_quant_mode)
            indexer_fp8_flops = database._get_quant_tc_flops(common.FMHAQuantMode.fp8)
            attn_flops = database._get_quant_tc_flops(fmha_mode)

            sol_math = (
                gemm_group_ops / gemm_flops + indexer_logits_ops / indexer_fp8_flops + sparse_attn_ops / attn_flops
            ) * 1000
            sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
            sol_time = max(sol_math, sol_mem)
            return sol_time, sol_math, sol_mem

        def get_empirical(b: int, s: int, num_heads: int, kv_cache_dtype: common.KVCacheQuantMode) -> float:
            latency = get_sol(b, s, num_heads, kv_cache_dtype)[0]
            scale_factor = 0.5
            return latency / scale_factor

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            sol_latency = get_sol(b, s, num_heads, kv_cache_dtype)[0]
            return PerformanceResult(sol_latency, energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol(b, s, num_heads, kv_cache_dtype)
        elif database_mode == common.DatabaseMode.EMPIRICAL:
            emp_latency = get_empirical(b, s, num_heads, kv_cache_dtype)
            return PerformanceResult(emp_latency, energy=0.0, source="empirical")

        cls.load_data(database)

        def missing_generation_dsa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Generation DSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"kv_cache_dtype={kv_cache_dtype}, gemm_quant_mode={gemm_quant_mode}, "
                f"num_heads={num_heads}, s={s}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        try:
            dsa_module_data = database._generation_dsa_module_data
            if dsa_module_data is None:
                raise PerfDataNotAvailableError(
                    f"Generation DSA module perf data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )
            try:
                dsa_dict = dsa_module_data[kv_cache_dtype][gemm_quant_mode][architecture]
                result = database._interp_3d(num_heads, b, s, dsa_dict, "cubic")
                latency = result["latency"]
                energy = result.get("energy", 0.0)
            except (KeyError, TypeError, AssertionError) as exc:
                raise missing_generation_dsa_error() from exc
            return database._interp_pr(latency, energy=energy)
        except Exception as e:
            if database_mode == common.DatabaseMode.HYBRID:
                logger.debug(
                    f"Failed to query generation DSA module for {b=}, {s=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}; using empirical"
                )
                latency = get_empirical(b, s, num_heads, kv_cache_dtype)
                return PerformanceResult(latency, energy=0.0, source="empirical")
            if isinstance(e, PerfDataNotAvailableError):
                logger.warning(str(e))
                raise
            if _is_dsa_interpolation_miss(e):
                message = _format_dsa_unavailable_message(
                    "Generation",
                    e,
                    b=b,
                    s=s,
                    num_heads=num_heads,
                    architecture=architecture,
                    index_n_heads=index_n_heads,
                    index_head_dim=index_head_dim,
                    index_topk=index_topk,
                )
                logger.warning(message)
                raise PerfDataNotAvailableError(message) from None
            else:
                logger.exception(
                    f"Failed to query generation DSA module for {b=}, {s=}, {num_heads=}, "
                    f"{index_n_heads=}, {index_head_dim=}, {index_topk=}, "
                    f"{kv_cache_dtype=}, {database_mode=}."
                )
                raise

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query generation DSA latency with energy data."""
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")

        result = database.query_generation_dsa_module(
            b=batch_size,
            s=s,
            num_heads=self._num_heads,
            kv_cache_dtype=self._kv_cache_dtype,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._architecture,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
