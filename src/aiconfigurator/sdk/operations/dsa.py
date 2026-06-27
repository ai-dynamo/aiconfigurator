# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSA (DeepSeek Sparse Attention) module-level ops (ISSUE-10 / AIC-538).

Both module ops keep their measured tables sparse and immutable. Numerical
holes and authorized exterior points are answered lazily by the shared sparse
surrogate; context and generation samples are split at the top-k boundary
before interpolation so the kernel discontinuity is never crossed.

DSA-specific helpers (``_is_dsa_interpolation_miss``,
``_format_dsa_unavailable_message``) also move here as module-level
functions. ``DSA_MODEL_DIMS`` and ``DEFAULT_DSA_ARCHITECTURE`` stay on
``perf_database.py`` as module-level constants for now — the cleanup PR
revisits their home.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator.sdk import common, interpolation
from aiconfigurator.sdk.interpolation import InterpolationDataNotAvailableError
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows
from aiconfigurator.sdk.perf_surrogate import Axis, estimate_sparse
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


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


def _dsa_backend_for_row(row) -> str:
    """Preserve the backend category when legacy rows use kernel_source=default."""
    kernel_source = str(row.get("kernel_source") or "").lower()
    framework = str(row.get("framework") or "").lower()
    return "trtllm" if "trtllm" in kernel_source or framework == "trtllm" else "flashmla_kv"


def _select_dsa_backend(arch_dict, dsa_backend):
    """Pick the per-backend sub-dict from a context-DSA architecture node.

    Context data is keyed ...[architecture][backend][num_heads]...; backend is
    "trtllm" (faster kernel, non-CP default) or "flashmla_kv" (used under CP).
    Backend is categorical, so a missing requested bucket must not silently use
    measurements from another backend. Legacy nodes without a backend axis
    (int head keys) pass through."""
    if not isinstance(arch_dict, dict) or not arch_dict:
        return arch_dict
    if not any(isinstance(k, str) for k in arch_dict):
        return arch_dict
    return arch_dict[dsa_backend]


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


def _cached_sparse_view(
    database: PerfDatabase,
    key: tuple,
    source: Mapping,
    build: Callable[[], dict],
) -> dict:
    """Cache a small filtered/reordered view by source identity and policy."""
    cache = database.__dict__.setdefault("_dsa_sparse_table_cache", {})
    cache_key = (*key, id(source))
    cached = cache.get(cache_key)
    if cached is None or cached[0] is not source:
        cached = (source, build())
        cache[cache_key] = cached
    return cached[1]


def _context_regime_view(
    database: PerfDatabase,
    table: Mapping,
    prefix: int,
    index_topk: int,
    above_topk: bool,
) -> dict:
    return _cached_sparse_view(
        database,
        ("context_dsa", prefix, index_topk, above_topk),
        table,
        lambda: {
            head: {
                sequence: by_batch
                for sequence, by_batch in by_sequence.items()
                if (sequence + prefix > index_topk) == above_topk
            }
            for head, by_sequence in table.items()
            if isinstance(by_sequence, Mapping)
        },
    )


def _generation_regime_view(
    database: PerfDatabase,
    table: Mapping,
    index_topk: int,
    above_topk: bool,
) -> dict:
    def build() -> dict:
        result: dict = {}
        for head, by_batch in table.items():
            for batch, by_sequence in by_batch.items():
                samples = {
                    sequence: metric
                    for sequence, metric in by_sequence.items()
                    if (sequence > index_topk) == above_topk
                }
                if samples:
                    result.setdefault(head, {})[batch] = samples
        return result

    return _cached_sparse_view(
        database,
        ("generation_dsa", index_topk, above_topk),
        table,
        build,
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

    Owns the measured sparse context DSA table.

    Models the full DSA attention block including:
    - kv_a_proj_with_mqa GEMM (includes indexer K projection)
    - LayerNorm + q_b_proj GEMM
    - Indexer: wq_b GEMM, weights_proj GEMM, FP8 MQA logits, TopK selection
    - Sparse MLA attention (attends to top-k tokens instead of full sequence)
    - BMM pre/post (weight absorption + V projection)
    - o_proj GEMM
    """

    _data_cache: ClassVar[dict] = {}

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
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._architecture = architecture
        self._cp_size = cp_size
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Load the measured sparse table without materializing a grid."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.dsa_context_module.value)
            sources = database._build_op_sources(PerfDataFilename.dsa_context_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_dsa_module_data(sources), PerfDataFilename.dsa_context_module, primary_path
            )
            cls._record_load()

        if "_context_dsa_module_data" not in database.__dict__:
            database._context_dsa_module_data = cls._data_cache[key]
        if "_raw_context_dsa_module_data" not in database.__dict__:
            database._raw_context_dsa_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._glm5_sparse_cache.clear()

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
        dsa_backend: str | None = None,
    ):
        """Query context DSA module table. Verbatim port of the legacy body."""
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        # ``DEFAULT_DSA_ARCHITECTURE`` and ``DSA_MODEL_DIMS`` live at module
        # level in this file — use them directly rather than round-tripping
        # through ``perf_database``'s backward-compat re-export.

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE
        if dsa_backend is None:
            dsa_backend = "trtllm" if database.backend == common.BackendName.trtllm.value else "flashmla_kv"

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
            from aiconfigurator.sdk.operations.gemm import GEMM

            gemm_flops = GEMM._get_quant_tc_flops(database.system_spec, gemm_quant_mode)
            indexer_fp8_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
            attn_flops = GEMM._get_quant_tc_flops(database.system_spec, fmha_quant_mode)

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
                dsa_dict = _select_dsa_backend(dsa_dict, dsa_backend)
            except (KeyError, TypeError) as exc:
                raise missing_context_dsa_error() from exc

            def _is_latency_leaf(value):
                return isinstance(value, dict) and "latency" in value

            def _context_prefix_data(module_dict, require_prefix_axis: bool = False):
                if not isinstance(module_dict, dict):
                    return None, False
                for head_data in module_dict.values():
                    if not isinstance(head_data, dict):
                        continue
                    for maybe_batch_slice in head_data.values():
                        if not isinstance(maybe_batch_slice, dict):
                            continue
                        # Legacy shape: [num_heads][s][b].
                        if any(_is_latency_leaf(v) for v in maybe_batch_slice.values()):
                            if require_prefix_axis:
                                return None, False
                            return _cached_sparse_view(
                                database,
                                ("context_dsa_prefix", False),
                                module_dict,
                                lambda: {0: module_dict},
                            ), False
                        break
                    break

                def build_prefix_data() -> dict:
                    result: dict = {}
                    for head, head_data in module_dict.items():
                        if not isinstance(head_data, dict):
                            continue
                        for prefix_value, prefix_slice in head_data.items():
                            if isinstance(prefix_slice, dict):
                                result.setdefault(prefix_value, {})[head] = prefix_slice
                    return result

                prefix_data = _cached_sparse_view(
                    database,
                    ("context_dsa_prefix", True),
                    module_dict,
                    build_prefix_data,
                )
                return (prefix_data or None), True

            require_prefix_axis = architecture == "GlmMoeDsaForCausalLM"
            prefix_data, has_prefix_axis = _context_prefix_data(dsa_dict, require_prefix_axis)

            def _lookup_prefix_module_at(prefix_value: int):
                if not isinstance(prefix_data, dict):
                    return None
                prefix_slice = prefix_data.get(prefix_value)
                if not isinstance(prefix_slice, dict):
                    return None
                above_topk = s + prefix_value > index_topk
                regime_table = _context_regime_view(
                    database,
                    prefix_slice,
                    prefix_value,
                    index_topk,
                    above_topk,
                )

                def baseline(point: Mapping[str, float]) -> float:
                    return get_sol(
                        point["batch"],
                        point["s"],
                        prefix_value,
                        point["heads"],
                        kvcache_quant_mode,
                        fmha_quant_mode,
                    )[0]

                try:
                    latency, energy = estimate_sparse(
                        database,
                        (
                            "context_dsa",
                            fmha_quant_mode,
                            kvcache_quant_mode,
                            gemm_quant_mode,
                            architecture,
                            dsa_backend,
                            prefix_value,
                            above_topk,
                        ),
                        regime_table,
                        {"heads": num_heads, "s": s, "batch": b},
                        axes=(
                            Axis("heads"),
                            Axis("s", extrapolate="both"),
                            Axis("batch", extrapolate="both"),
                        ),
                        varying="s",
                        baseline=baseline,
                    )
                except InterpolationDataNotAvailableError:
                    return None
                power = 0.0 if latency == 0 else energy / latency
                return {"latency": latency, "power": power, "energy": energy}

            def _interp_results_1d(x0, x1, r0, r1, x):
                return {
                    field: interpolation.interp_1d([x0, x1], [r0.get(field, 0.0), r1.get(field, 0.0)], x)
                    for field in ("latency", "power", "energy")
                }

            try:
                result = _lookup_prefix_module_at(prefix)
                if result is None:
                    if not has_prefix_axis:
                        raise missing_context_dsa_error()
                    prefix_points = sorted(p for p, slice_ in prefix_data.items() if isinstance(slice_, dict))
                    result_by_prefix = {}
                    for prefix_point in prefix_points:
                        if (s + prefix_point > index_topk) != (s + prefix > index_topk):
                            continue
                        prefix_result = _lookup_prefix_module_at(prefix_point)
                        if prefix_result is not None:
                            result_by_prefix[prefix_point] = prefix_result
                    if len(result_by_prefix) < 2:
                        raise missing_context_dsa_error()
                    left, right = interpolation.nearest_1d_point_helper(
                        prefix, list(result_by_prefix.keys()), inner_only=False
                    )
                    result = (
                        result_by_prefix[left]
                        if left == right
                        else _interp_results_1d(
                            left,
                            right,
                            result_by_prefix[left],
                            result_by_prefix[right],
                            prefix,
                        )
                    )
                latency = result["latency"]
                energy = result.get("energy", 0.0)
            except (KeyError, TypeError, ValueError, AssertionError) as exc:
                raise missing_context_dsa_error() from exc
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

        if self._cp_size and self._cp_size > 1:
            return self._query_cp(database, batch_size, isl, prefix)

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

    # ------------------------------------------------------------------
    # Context-Parallel (CP) prefill model — GLM-5 DSA only.
    # See docs/CONTEXT_PARALLEL_DSA_MODELING.md. Per-card =
    #   base dsa_module(isl/cp, bf16-KV row)
    #   + mqa(isl/cp)*(cp-1)                          (mqa ∝ isl², xcp identity)
    #   - [topk_full(flat) - topk_full(top_last)]/cp  (topk ∝ full/cp; module is dummy/flat)
    #   + AG_KV + AG_LSE                              (the two small attention all-gathers)
    # AG_hidden + RS belong to the MoE comm (modeled by MoEDispatch), not here.
    # ------------------------------------------------------------------
    _glm5_sparse_cache: ClassVar[dict] = {}

    def _query_cp(self, database: PerfDatabase, b: int, isl: int, prefix: int) -> PerformanceResult:
        """CP (round-robin split) per-layer DSA, new strategy (2026-06-11):

            result = dsa(isl/cp, prefix)
                   + [mqa(isl, prefix)/cp      - mqa(isl/cp, prefix)]
                   + [topk_last(isl, prefix)/cp - topk_flat(isl/cp, prefix)]
                   + AG_KV + AG_LSE

        The per-card monolithic dsa_module(isl/cp, prefix) is the base; its
        internal mqa(isl/cp,prefix) and topk_flat(isl/cp,prefix) are swapped out
        by the two deltas, leaving proj + dsa_attn (both prefix-independent: proj
        by construction, dsa_attn topk-capped to index_topk) plus the CP-correct
        full-chunk mqa/topk_last divided across cp ranks. All sub-kernels are
        looked up at the REAL (q_len, prefix) shape — the parquet ``step`` column
        IS the prefix (past_kv) length.
        """
        cp = self._cp_size
        per_card = max(1, -(-isl // cp))  # ceil: critical path = busiest CP rank
        sp = self._load_glm5_sparse(database)
        g = sp.get("_2d", {})
        # Fail fast: CP DSA modeling REQUIRES the sparse mqa/topk tables for
        # the mqa/topk_last deltas. _lookup_2d clamps isl + interp/extrapolates
        # step, so a None below means the table is absent entirely (parquet not
        # collected) -- degrading silently to dsa_base would hide that.
        missing = [k for k in ("mqa", "topk_last", "topk_flat") if not g.get(k)]
        if missing:
            raise ValueError(
                f"GLM5 CP DSA modeling needs sparse tables {missing} for "
                f"{self._architecture}; collect glm5_mqa_logits/glm5_topk first."
            )
        # Base: per-card monolithic dsa_module at (per_card, prefix), follows the
        # run's kv_cache_dtype like the non-CP path.
        dsa_base = float(
            database.query_context_dsa_module(
                b=b,
                s=per_card,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
                architecture=self._architecture,
                dsa_backend="flashmla_kv",
            )
        )
        mqa_full = self._lookup_2d(database, "mqa", g.get("mqa"), isl, prefix)
        mqa_perc = self._lookup_2d(database, "mqa", g.get("mqa"), per_card, prefix)
        tl_full = self._lookup_2d(database, "topk_last", g.get("topk_last"), isl, prefix)
        tf_perc = self._lookup_2d(database, "topk_flat", g.get("topk_flat"), per_card, prefix)
        latency = dsa_base
        if None not in (mqa_full, mqa_perc, tl_full, tf_perc):
            delta_mqa = mqa_full / cp - mqa_perc
            delta_topk = tl_full / cp - tf_perc
            latency += delta_mqa + delta_topk
        # CP communication: AG of compressed KV (kv_lora+rope) + AG of LSE (kv_lora).
        dims = DSA_MODEL_DIMS.get(self._architecture, {})
        kv_lora = dims.get("kv_lora_rank", 512)
        rope = dims.get("qk_rope_head_dim", 64)
        index_head_dim = dims.get("index_head_dim", 128)
        # CP attention all-gather, verified by instrumenting sglang cp_utils
        # (cp_all_gather_rerange_output): per current-chunk tokens (isl, not
        # isl+prefix; prefix KV is already replicated), bf16. Two gathers:
        #   - compressed KV latent: kv_lora_rank + qk_rope_head_dim (= 576)
        #   - DSA indexer key: index_head_dim (= 128)
        # (The hidden_states 6144 AG/RS is the MoE token dispatch, modeled in
        # context_moe_pre/post_dispatch, not here.)
        # ag_kv = MQA-stage gather: DSA indexer key (index_head_dim), bf16.
        # ag_lse = FMHA-stage gather: compressed KV latent (kv_lora_rank +
        # qk_rope_head_dim), bf16. Both over the current chunk (isl), verified by
        # instrumenting sglang (dsa_indexer index_key 128; deepseek_v2
        # rebuild_cp_kv_cache latent 576).
        ag_kv = float(database.query_nccl(common.CommQuantMode.half, cp, "all_gather", isl * index_head_dim))
        ag_lse = float(database.query_nccl(common.CommQuantMode.half, cp, "all_gather", isl * (kv_lora + rope)))
        latency += ag_kv + ag_lse
        return PerformanceResult(latency * self._scale_factor, energy=0.0, source="cp_model")

    @classmethod
    def _load_glm5_sparse(cls, database: PerfDatabase) -> dict:
        key = cls._cache_key(database)
        if key in cls._glm5_sparse_cache:
            return cls._glm5_sparse_cache[key]
        import os

        import pandas as pd

        data_dir = os.path.join(
            database.systems_root, database.system_spec["data_dir"], database.backend, database.version
        )
        # 2D grids keyed by (isl, step) for the CP composition path.
        out = {}
        out2d = {"mqa": {}, "topk_last": {}, "topk_flat": {}, "dsa_attn": {}}

        def _read(fn):
            p = os.path.join(data_dir, fn)
            return pd.read_parquet(p) if os.path.exists(p) else None

        mdf = _read("glm5_mqa_logits_module_perf.parquet")
        if mdf is not None:
            mh = mdf[mdf["num_heads"] == 64] if "num_heads" in mdf else mdf
            for _, r in mh[mh["batch_size"] == 1].iterrows():
                out2d["mqa"][(int(r["isl"]), int(r["step"]))] = float(r["latency"])
        tdf = _read("glm5_topk_module_perf.parquet")
        if tdf is not None:
            th = tdf[tdf["num_heads"] == 64] if "num_heads" in tdf else tdf
            for _, r in th[th["batch_size"] == 1].iterrows():
                mode = "topk_flat" if str(r.get("score_mode", "")) == "flat" else "topk_last"
                out2d[mode][(int(r["isl"]), int(r["step"]))] = float(r["latency"])
        adf = _read("glm5_dsa_attn_module_perf.parquet")
        if adf is not None:
            ah = adf[adf["num_heads"] == 64] if "num_heads" in adf else adf
            for _, r in ah[ah["batch_size"] == 1].iterrows():
                out2d["dsa_attn"][(int(r["isl"]), int(r["step"]))] = float(r["latency"])
        out["_2d"] = out2d
        cls._glm5_sparse_cache[key] = out
        return out

    @staticmethod
    def _lookup_2d(database: PerfDatabase, op_name: str, table, isl, step):
        """Lookup {(isl,step): latency} at a fixed isl (exact grid value), linear
        interpolation and endpoint hold on step."""
        if not table:
            return None
        isls = sorted({i for (i, _s) in table})
        if isl > isls[-1]:
            raise ValueError(
                f"GLM5 CP DSA: isl={isl} exceeds the collected sparse-kernel grid "
                f"(max isl={isls[-1]}); mqa/topk scale super-linearly with isl, so "
                f"clamping the isl axis would silently under-estimate. Re-collect with "
                f"AIC_CHUNKED_PREFILL_SIZE >= {isl} "
                f"(docs/CONTEXT_PARALLEL_DSA_MODELING.md §9.1)."
            )
        use_isl = isl if isl in isls else min(isls, key=lambda x: abs(x - isl))
        curve = _cached_sparse_view(
            database,
            ("glm5_cp", op_name, use_isl),
            table,
            lambda: {
                sampled_step: {"latency": latency}
                for (sampled_isl, sampled_step), latency in table.items()
                if sampled_isl == use_isl
            },
        )
        if not curve:
            return None
        latency, _ = estimate_sparse(
            database,
            ("glm5_cp", op_name, use_isl),
            curve,
            {"prefix": step},
            axes=(Axis("prefix", extrapolate="both"),),
            varying="prefix",
            exterior="raw",
        )
        return latency

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GenerationDSAModule(Operation):
    """
    Generation phase DSA (DeepSeek Sparse Attention) module-level operation.

    Owns the measured sparse generation DSA table.

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
        """Load the measured sparse table without materializing a grid."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.dsa_generation_module.value)
            sources = database._build_op_sources(PerfDataFilename.dsa_generation_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_dsa_module_data(sources), PerfDataFilename.dsa_generation_module, primary_path
            )
            cls._record_load()

        if "_generation_dsa_module_data" not in database.__dict__:
            database._generation_dsa_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

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
        dsa_backend: str | None = None,
    ):
        """Query generation DSA module table. Verbatim port of the legacy body."""
        from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError

        # ``DEFAULT_DSA_ARCHITECTURE`` and ``DSA_MODEL_DIMS`` live at module
        # level in this file — use them directly rather than round-tripping
        # through ``perf_database``'s backward-compat re-export.

        if architecture is None:
            architecture = DEFAULT_DSA_ARCHITECTURE
        if dsa_backend is None:
            dsa_backend = "trtllm" if database.backend == common.BackendName.trtllm.value else "flashmla_kv"

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

            from aiconfigurator.sdk.operations.gemm import GEMM

            gemm_flops = GEMM._get_quant_tc_flops(database.system_spec, gemm_quant_mode)
            indexer_fp8_flops = GEMM._get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
            attn_flops = GEMM._get_quant_tc_flops(database.system_spec, fmha_mode)

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
                dsa_dict = _select_dsa_backend(dsa_dict, dsa_backend)
                above_topk = s > index_topk
                regime_table = _generation_regime_view(database, dsa_dict, index_topk, above_topk)

                def baseline(point: Mapping[str, float]) -> float:
                    return get_sol(
                        point["batch"],
                        point["s_total"],
                        point["heads"],
                        kv_cache_dtype,
                    )[0]

                latency, energy = estimate_sparse(
                    database,
                    (
                        "generation_dsa",
                        kv_cache_dtype,
                        gemm_quant_mode,
                        architecture,
                        dsa_backend,
                        above_topk,
                    ),
                    regime_table,
                    {"heads": num_heads, "batch": b, "s_total": s},
                    axes=(
                        Axis("heads"),
                        Axis("batch", extrapolate="both"),
                        Axis("s_total", extrapolate="both"),
                    ),
                    varying="s_total",
                    baseline=baseline,
                )
            except (KeyError, TypeError, ValueError, AssertionError) as exc:
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


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_context_dsa_module_data(dsa_file: str):
    """
    Load context DSA data.

    Dict structure:
        data[fmha_quant_mode][kv_cache_quant_mode][gemm_quant_mode][architecture][backend][num_heads][prefix][s][b]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported FMHAQuantMode names (aligned with
    ``_context_attention_data``).  ``architecture`` (e.g.
    "DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM") selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".
    """
    rows = _read_filtered_rows(dsa_file)
    if rows is None:
        logger.debug(f"DSA context data file {dsa_file} not found.")
        return None

    dsa_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))
                    )
                )
            )
        )
    )

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
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
            raise ValueError("GLM-5 context DSA module data requires a non-empty step column for prefix/past_kv length")
        prefix = 0 if step_missing else int(step)
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

        dsa_backend = _dsa_backend_for_row(row)
        batch_data = dsa_data[fmha_mode][kv_dtype][gemm_mode][arch][dsa_backend][num_heads][prefix][s]
        batch_data.setdefault(b, {"latency": latency, "power": power, "energy": energy})

    return dsa_data


def load_generation_dsa_module_data(dsa_file: str):
    """
    Load generation DSA data.

    Dict structure:
        data[kv_cache_quant_mode][gemm_quant_mode][architecture][backend][num_heads][b][s]

    Quant modes are the outermost keys so that ``_enum_key_names`` can
    directly extract supported KVCacheQuantMode names (aligned with
    ``_generation_attention_data``).  ``architecture`` selects the
    model-specific structural dimensions from ``DSA_MODEL_DIMS``.
    Legacy CSV rows without an ``architecture`` column default to
    "DeepseekV32ForCausalLM".
    """
    rows = _read_filtered_rows(dsa_file)
    if rows is None:
        logger.debug(f"DSA generation data file {dsa_file} not found.")
        return None

    dsa_data = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict()))))
        )
    )

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        num_heads = int(row["num_heads"])
        b = int(row["batch_size"])
        s = int(row["isl"]) + int(row["step"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        arch = row.get("architecture", DEFAULT_DSA_ARCHITECTURE)
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[row["kv_cache_dtype"]]

        dsa_backend = _dsa_backend_for_row(row)
        sequence_data = dsa_data[kv_dtype][gemm_mode][arch][dsa_backend][num_heads][b]
        sequence_data.setdefault(s, {"latency": latency, "power": power, "energy": energy})

    return dsa_data
