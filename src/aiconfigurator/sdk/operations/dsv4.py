# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 family (ISSUE-11 / AIC-1095).

Four op classes migrate from ``_legacy.py`` into ``operations/dsv4.py``:

- ``DeepSeekV4MHCModule`` — manifold-constrained hyper-connection pre/post.
  Owns ``_mhc_module_data``. Delegates to
  ``PerfDatabase.query_mhc_module`` which becomes a one-line forward.
- ``_BaseDeepSeekV4AttentionModule`` — shared weight metadata; not
  instantiated directly. Holds the shared SOL helper used by both
  context and generation phases.
- ``ContextDeepSeekV4AttentionModule`` — context-phase SWA/CSA/HCA. Owns
  ``_context_deepseek_v4_attention_module_data`` (merged from csa+hca
  split files), ``_raw_context_deepseek_v4_attention_module_data``
  (deepcopy used for topk piecewise lookup), and the
  ``_dsv4_sparse_kernel_data`` sidecar dict (paged_mqa_logits + hca_attn)
  used for prefix kernel-Δ correction.
- ``GenerationDeepSeekV4AttentionModule`` — decode-phase. Owns
  ``_generation_deepseek_v4_attention_module_data`` (merged from
  csa+hca split files).

No SOL clamping in the legacy ``_correct_data`` for DSV4 (the per-attn
SOL formula runs inside the query path). No grid extrapolation either —
``_dsv4_robust_3d_lookup`` handles interpolation/fallback at query time.

Cache key matches every other migrated op:
``(systems_root, system, backend, version, enable_shared_layer)``.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

from aiconfigurator.sdk import common, interpolation, perf_interp
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.base import Operation, _read_filtered_rows

logger = logging.getLogger(__name__)
from aiconfigurator.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator.sdk.perf_database import PerfDatabase


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op family."""
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


# ───────────────────────────────────────────────────────────────────────
# Module-level helpers (moved from perf_database.py).
# Re-exported from perf_database for back-compat with tests that imported
# them via ``from aiconfigurator.sdk.perf_database import ...``.
# ───────────────────────────────────────────────────────────────────────


def _deep_merge_dsv4_dicts(dest, src):
    """In-place merge ``src`` nested dict into ``dest``.

    Used to combine the per-(attn_kind) CSVs into one nested dict. At any
    level where both sides have a dict, recurse; otherwise overwrite.
    """
    if src is None:
        return dest
    for k, v in src.items():
        if k in dest and isinstance(dest[k], dict) and isinstance(v, dict):
            _deep_merge_dsv4_dicts(dest[k], v)
        else:
            dest[k] = v
    return dest


def _dsv4_robust_3d_lookup(self, dict_, x, y, z, *, batch_axis: str = "z"):
    """DeepSeek-V4 3D lookup: exact, cubic, then sampled-batch fallback.

    Generic ``_interp_3d`` raises ``QhullError`` when the point cloud has
    a degenerate axis (e.g. the DSV4 sweep caps b=1 at s=8192, so the
    b axis is flat near that query point).

      1. Try an exact ``dict[x][y][z]`` lookup — handles the common case
         of querying at a measured bench point.
      2. Try the existing cubic interpolation path.
      3. If interpolation fails and batches on both sides of the query cover
         its sequence length, interpolate along sequence within each batch and
         then across batch.
      4. Otherwise, use the largest sampled batch no larger than the query
         batch, interpolate/extrapolate along sequence length, and scale by
         batch. ``batch_axis`` selects whether fallback treats ``y`` or ``z``
         as the batch axis.

    First positional arg is named ``self`` to match the legacy signature so
    existing test stubs (``PerfDatabase._lookup_dsv4_sparse_kernel`` style)
    keep working.
    """
    if batch_axis not in ("y", "z"):
        raise ValueError(f"unsupported DeepSeek-V4 fallback {batch_axis=}; expected 'y' or 'z'")

    # Use .get() chain instead of [] indexing: dict_ may be a (nested)
    # defaultdict, so [] reads would create spurious empty branches that
    # later poison _interp_3d's grid traversal.
    level1 = dict_.get(x) if isinstance(dict_, dict) else None
    level2 = level1.get(y) if isinstance(level1, dict) else None
    exact = level2.get(z) if isinstance(level2, dict) else None
    if isinstance(exact, dict) and "latency" in exact:
        return exact

    def _finite_result(result):
        if not isinstance(result, dict):
            return False
        value = result.get("latency")
        return value is not None and bool(np.all(np.isfinite(np.asarray(value))))

    try:
        result = interpolation.interp_3d(x, y, z, dict_, "cubic", self._extracted_metrics_cache)
        if _finite_result(result):
            return result
    except (PerfDataNotAvailableError, interpolation.InterpolationDataNotAvailableError):
        pass

    # Fallback: real DeepSeek-V4 data is sampled at batches like 1/2/4/8.
    # First interpolate between lower/upper batches when both batch curves
    # cover the query sequence. This preserves launch-overhead plateaus; simply
    # scaling the lower batch can nearly double latency between b=4 and b=8.
    # When no upper batch covers the sequence, retain the legacy behavior:
    # prefer the largest batch <= query batch and scale it by batch, only
    # extrapolating along sequence if no lower batch covers it.
    sub = dict_.get(x) if isinstance(dict_, dict) else None
    if isinstance(sub, dict):
        query_s, query_b = (z, y) if batch_axis == "y" else (y, z)

        def _all_batch_points():
            if batch_axis == "y":
                return sorted(bp for bp, sd in sub.items() if isinstance(sd, dict))
            return sorted({bp for sd in sub.values() if isinstance(sd, dict) for bp in sd})

        def _leaf_at(s, b):
            first_key, second_key = (b, s) if batch_axis == "y" else (s, b)
            level = sub.get(first_key)
            return level.get(second_key) if isinstance(level, dict) else None

        def _seq_points_at_batch(b):
            if batch_axis == "y":
                level = sub.get(b)
                return sorted(level.keys()) if isinstance(level, dict) else []
            return sorted(s for s, sd in sub.items() if isinstance(sd, dict) and b in sd)

        def _lookup_at_batch(bp, *, allow_extrapolate: bool):
            exact_at_batch = _leaf_at(query_s, bp)
            if exact_at_batch is not None:
                return exact_at_batch
            ss = _seq_points_at_batch(bp)
            if len(ss) < 2:
                return None
            if not allow_extrapolate and not (ss[0] <= query_s <= ss[-1]):
                return None
            sl, sr = interpolation.nearest_1d_point_helper(query_s, ss, inner_only=not allow_extrapolate)
            left = _leaf_at(sl, bp)
            right = _leaf_at(sr, bp)
            if not isinstance(left, dict) or not isinstance(right, dict):
                return None
            return {
                field: interpolation.interp_1d([sl, sr], [left.get(field, 0.0), right.get(field, 0.0)], query_s)
                for field in ("latency", "power", "energy")
            }

        batch_points = _all_batch_points()
        covered = {}
        for bp in batch_points:
            try:
                leaf = _lookup_at_batch(bp, allow_extrapolate=False)
                if _finite_result(leaf):
                    covered[bp] = leaf
            except (PerfDataNotAvailableError, interpolation.InterpolationDataNotAvailableError):
                continue

        lower_batches = [bp for bp in covered if bp <= query_b]
        upper_batches = [bp for bp in covered if bp >= query_b]
        if lower_batches and upper_batches:
            lower_b = max(lower_batches)
            upper_b = min(upper_batches)
            if lower_b == upper_b:
                return covered[lower_b]
            lower = covered[lower_b]
            upper = covered[upper_b]
            result = {
                field: interpolation.interp_1d(
                    [lower_b, upper_b],
                    [lower.get(field, 0.0), upper.get(field, 0.0)],
                    query_b,
                )
                for field in ("latency", "power", "energy")
            }
            if _finite_result(result):
                return result

        lower_batch_points = sorted((bp for bp in batch_points if bp <= query_b), reverse=True)
        for allow_extrapolate in (False, True):
            for bp in lower_batch_points:
                try:
                    leaf = _lookup_at_batch(bp, allow_extrapolate=allow_extrapolate)
                    if leaf is None:
                        continue
                    result = {f: leaf.get(f, 0.0) * query_b / bp for f in ("latency", "power", "energy")}
                    if _finite_result(result):
                        return result
                except (PerfDataNotAvailableError, interpolation.InterpolationDataNotAvailableError):
                    continue

    if batch_axis == "y":
        raise interpolation.InterpolationDataNotAvailableError(
            f"DeepSeek-V4 robust lookup failed (tp={x}, b={y}, s={z})"
        )
    raise interpolation.InterpolationDataNotAvailableError(f"DeepSeek-V4 robust lookup failed (tp={x}, s={y}, b={z})")


def _dsv4_resolve_head_key(quant_data, num_heads):
    """SCHEME A head-key resolution.

    The head axis is the rank-local head count (``native // tp``).  Prefer an
    exact match on the value the model passes.  The b300 module data is a
    universal sweep collected with a single local-head value; if the model's
    local head count is not an exact bench point, fall back to the single
    available head key so the universal data still resolves.  Returns the head
    key to index ``quant_data`` with, or ``None`` if no head data is loaded.
    """
    if not isinstance(quant_data, dict) or not quant_data:
        return None
    if num_heads in quant_data:
        return num_heads
    head_keys = [k for k in quant_data if isinstance(k, int)]
    if len(head_keys) == 1:
        return head_keys[0]
    if head_keys:
        # Multiple head keys but no exact match: pick the nearest <= request,
        # else the smallest available.
        le = [k for k in head_keys if k <= num_heads]
        return max(le) if le else min(head_keys)
    return None


def _dsv4_lookup_prefix_resolved(database, cr_dict, prefix, s, b):
    """SCHEME A prefix-resolved silicon lookup.

    ``cr_dict`` is ``{prefix: {s: {b: leaf}}}``.  At a measured prefix we wrap
    the ``{s: {b}}`` slice and run the robust 3D lookup (exact -> cubic ->
    sampled-batch fallback) with the prefix as the leading axis.  Off-grid
    prefixes are linearly interpolated between the two nearest prefix anchors
    that can answer the (s, b) query.  Returns a leaf dict or ``None``.
    """
    if cr_dict is None:
        return None
    if not isinstance(cr_dict, Mapping):
        raise TypeError(f"Malformed DeepSeek-V4 prefix table: expected a mapping, got {type(cr_dict).__name__}.")
    if not cr_dict:
        return None
    for prefix_value, sb_slice in cr_dict.items():
        if sb_slice is not None and not isinstance(sb_slice, Mapping):
            raise TypeError(
                "Malformed DeepSeek-V4 prefix slice: expected a mapping at "
                f"prefix={prefix_value}, got {type(sb_slice).__name__}."
            )

    def _finite(result):
        if not isinstance(result, dict):
            return False
        value = result.get("latency")
        return value is not None and bool(np.all(np.isfinite(np.asarray(value))))

    def _lookup_at_prefix(prefix_value):
        sb_slice = cr_dict.get(prefix_value)
        if sb_slice is None:
            return None
        wrapped = {prefix_value: sb_slice}
        try:
            result = _dsv4_robust_3d_lookup(database, wrapped, prefix_value, s, b)
        except (PerfDataNotAvailableError, interpolation.InterpolationDataNotAvailableError):
            return None
        return result if _finite(result) else None

    prefix = int(prefix)
    if prefix in cr_dict:
        exact = _lookup_at_prefix(prefix)
        if exact is not None:
            return exact

    prefix_points = sorted(p for p, sl in cr_dict.items() if isinstance(sl, dict))
    if not prefix_points:
        return None
    if len(prefix_points) == 1:
        return _lookup_at_prefix(prefix_points[0])

    result_by_prefix = {}
    for p in prefix_points:
        r = _lookup_at_prefix(p)
        if r is not None:
            result_by_prefix[p] = r
    if not result_by_prefix:
        return None
    keys = sorted(result_by_prefix)
    if prefix in result_by_prefix:
        return result_by_prefix[prefix]
    if len(keys) == 1 or prefix <= keys[0]:
        return result_by_prefix[keys[0]]
    if prefix >= keys[-1]:
        return result_by_prefix[keys[-1]]
    left = max(k for k in keys if k < prefix)
    right = min(k for k in keys if k > prefix)
    rl, rr = result_by_prefix[left], result_by_prefix[right]
    return {
        field: interpolation.interp_1d([left, right], [rl.get(field, 0.0), rr.get(field, 0.0)], prefix)
        for field in ("latency", "power", "energy")
    }


def _deepseek_v4_attention_sol(
    database: PerfDatabase,
    *,
    is_context: bool,
    b: int,
    s: int,
    prefix: int,
    num_heads: int,
    hidden_size: int,
    q_lora_rank: int,
    o_lora_rank: int,
    head_dim: int,
    rope_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    window_size: int,
    compress_ratio: int,
    o_groups: int,
    kvcache_quant_mode: common.KVCacheQuantMode,
    fmha_quant_mode: common.FMHAQuantMode,
    gemm_quant_mode: common.GEMMQuantMode,
) -> tuple[float, float, float]:
    """Shared SOL formula for both context and generation phases.

    Verbatim port of the legacy ``PerfDatabase._deepseek_v4_attention_sol``
    body. Reads ``database.system_spec``, ``database._causal_limited_pairs``,
    ``database._compressed_context_pairs``, and ``GEMM._get_quant_tc_flops``.
    """
    from aiconfigurator.sdk.operations.gemm import GEMM

    def _tc_flops(quant_mode):
        return GEMM._get_quant_tc_flops(database.system_spec, quant_mode)

    tokens = b * s if is_context else b
    kv_len = prefix + s if is_context else max(0, s - 1)
    local_groups = max(1, o_groups)

    gemm_projection_ops = (
        2 * tokens * hidden_size * q_lora_rank
        + 2 * tokens * q_lora_rank * num_heads * head_dim
        + 2 * tokens * hidden_size * head_dim
        + 2 * tokens * local_groups * o_lora_rank * hidden_size
    )
    output_absorption_ops = 2 * tokens * num_heads * head_dim * o_lora_rank

    compressor_mult = 2 if compress_ratio == 4 else 1
    compressor_ops = 0.0
    if compress_ratio:
        compressor_ops = 4 * tokens * hidden_size * compressor_mult * head_dim
        compressor_ops += 2 * tokens * compressor_mult * head_dim
        if compress_ratio == 4:
            indexer_compressor_mult = 2
            compressor_ops += 4 * tokens * hidden_size * indexer_compressor_mult * index_head_dim
            compressor_ops += 2 * tokens * indexer_compressor_mult * index_head_dim

    if is_context:
        window_pairs = database._causal_limited_pairs(b, s, prefix, window_size)
        if compress_ratio:
            compressed_limit = index_topk if compress_ratio == 4 else max(0, kv_len // compress_ratio)
            compressed_pairs = database._compressed_context_pairs(b, s, prefix, compress_ratio, compressed_limit)
        else:
            compressed_pairs = 0
    else:
        window_pairs = b * min(kv_len, window_size)
        if compress_ratio:
            compressed_limit = index_topk if compress_ratio == 4 else max(0, kv_len // compress_ratio)
            compressed_pairs = b * min(kv_len // compress_ratio, compressed_limit)
        else:
            compressed_pairs = 0

    attention_pairs = window_pairs + compressed_pairs
    attention_ops = 4 * num_heads * head_dim * attention_pairs

    indexer_ops = 0.0
    indexer_bfloat16_ops = 0.0
    indexer_cache_bytes = 0.0
    if compress_ratio == 4:
        compressed_len = kv_len // compress_ratio
        if is_context:
            indexer_query_tokens = b * s
        else:
            indexer_query_tokens = b
        indexer_pairs = indexer_query_tokens * compressed_len
        indexer_ops = (
            2 * indexer_query_tokens * q_lora_rank * index_n_heads * index_head_dim
            + 2 * indexer_pairs * index_n_heads * index_head_dim
        )
        indexer_bfloat16_ops = 2 * indexer_query_tokens * hidden_size * index_n_heads
        indexer_cache_bytes = b * compressed_len * common.deepseek_v4_indexer_cache_entry_bytes(index_head_dim)

    gemm_weight_bytes = (
        hidden_size * q_lora_rank
        + q_lora_rank * num_heads * head_dim
        + hidden_size * head_dim
        + local_groups * o_lora_rank * hidden_size
    ) * gemm_quant_mode.value.memory
    bfloat16_weight_bytes = num_heads * head_dim * o_lora_rank * common.GEMMQuantMode.bfloat16.value.memory
    if compress_ratio:
        gemm_weight_bytes += 2 * hidden_size * compressor_mult * head_dim * gemm_quant_mode.value.memory
    if compress_ratio == 4:
        gemm_weight_bytes += q_lora_rank * index_n_heads * index_head_dim * gemm_quant_mode.value.memory
        bfloat16_weight_bytes += hidden_size * index_n_heads * common.GEMMQuantMode.bfloat16.value.memory

    activation_bytes = (
        tokens
        * (hidden_size + q_lora_rank + num_heads * head_dim + head_dim + local_groups * o_lora_rank)
        * gemm_quant_mode.value.memory
    )
    # DeepSeek-V4 attention uses compressed (MLA / MQA-equivalent) KV cache: each
    # cache entry stores a single ``head_dim``-sized vector that is shared by all
    # ``num_heads`` query heads (see ``DeepSeekV4Model.get_kvcache_bytes_per_sequence``
    # which derives storage as ``head_dim * kvcache_quant_mode.value.memory``).
    # The previous formula multiplied by ``num_heads``, which double-counted KV
    # traffic by a factor of ``num_heads`` (128x for DSv4-Pro). The inflated
    # ``sol_mem`` pinned SOL TPOT above HYBRID at non-trivial decode batch sizes,
    # reproducible with a single CLI invocation:
    #
    #   aiconfigurator cli estimate \
    #     --model-path deepseek-ai/DeepSeek-V4-Pro --system gb300 \
    #     --backend sglang --estimate-mode static_gen \
    #     --isl 8192 --osl 1024 --batch-size 570 --tp 1 --dp 12 --ep 12 \
    #     --database-mode HYBRID --detail all
    #
    # whose "Latency Summary" reported HYBRID at 99.3 ms vs SOL at 216.7 ms
    # (HYBRID ~2.2x faster than SOL) and the ``generation_attention`` op alone
    # at HYBRID 52.0 s vs SOL 197.8 s (3.8x violation), an impossible ordering
    # since SOL is the per-op roofline lower bound. The corrected formula reads
    # each KV entry once (pairs * head_dim), matching the storage layout and
    # the underlying kernel's MQA broadcast pattern.
    kv_cache_bytes = attention_pairs * head_dim * kvcache_quant_mode.value.memory
    rope_bytes = tokens * num_heads * rope_head_dim * fmha_quant_mode.value.memory

    sol_math = (
        (gemm_projection_ops + compressor_ops) / _tc_flops(gemm_quant_mode)
        + (output_absorption_ops + indexer_bfloat16_ops) / _tc_flops(common.GEMMQuantMode.bfloat16)
        + indexer_ops / _tc_flops(common.GEMMQuantMode.fp8)
        + attention_ops / _tc_flops(fmha_quant_mode)
    ) * 1000
    sol_mem = (
        (
            gemm_weight_bytes
            + bfloat16_weight_bytes
            + activation_bytes
            + kv_cache_bytes
            + indexer_cache_bytes
            + rope_bytes
        )
        / database.system_spec["gpu"]["mem_bw"]
        * 1000
    )
    return max(sol_math, sol_mem), sol_math, sol_mem


# ───────────────────────────────────────────────────────────────────────
# DeepSeekV4MHCModule
# ───────────────────────────────────────────────────────────────────────


class DeepSeekV4MHCModule(Operation):
    """DeepSeek-V4 manifold-constrained hyper-connection pre/post module."""

    _data_cache: ClassVar[dict] = {}
    _CP_AWARE: ClassVar[bool] = True  # token-major: query divides num_tokens by self._seq_split

    def __init__(
        self,
        name: str,
        scale_factor: float,
        op: str,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        quant_mode: common.GEMMQuantMode,
        *,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        if op not in {"pre", "post", "both"}:
            raise ValueError(f"Unsupported DeepSeek-V4 mHC op: {op}")
        self._op = op
        self._hidden_size = hidden_size
        self._hc_mult = hc_mult
        self._sinkhorn_iters = sinkhorn_iters
        self._quant_mode = quant_mode
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * hidden_size
        # Two parameter sets per decoder block: attention mHC and FFN mHC.
        self._weights = 2 * (mix_hc * hc_dim + mix_hc + 3) * quant_mode.value.memory

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads mhc_module CSV, binds ``database._mhc_module_data``."""
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.mhc_module.value)
            sources = database._build_op_sources(PerfDataFilename.mhc_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_mhc_module_data(sources), PerfDataFilename.mhc_module, primary_path
            )
            cls._record_load()

        if "_mhc_module_data" not in database.__dict__:
            database._mhc_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mhc_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_mhc_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        hc_mult: int,
        sinkhorn_iters: int,
        op: str,
        quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_mhc_module`` body.

        The SOL estimate models the combined attention-site and FFN-site mHC work
        inside one decoder layer, matching the collector's module boundary.
        """
        from aiconfigurator.sdk.operations.gemm import GEMM

        cls.load_data(database)

        sites = 2
        hc_dim = hc_mult * hidden_size
        mix_hc = (2 + hc_mult) * hc_mult

        def get_sol(nt: int = num_tokens, op_name: str = op) -> tuple[float, float, float]:
            pre_ops = sites * (
                2 * nt * hc_dim * mix_hc
                + nt * hc_dim * 3
                + nt * (hc_mult * hc_mult + 2 * hc_mult) * sinkhorn_iters
                + 2 * nt * hc_mult * hidden_size
            )
            post_ops = sites * (2 * nt * hc_mult * hc_mult * hidden_size + 2 * nt * hc_mult * hidden_size)
            if op_name == "pre":
                ops = pre_ops
            elif op_name == "post":
                ops = post_ops
            elif op_name == "both":
                ops = pre_ops + post_ops
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 mHC op: {op_name}")

            param_bytes = sites * (mix_hc * hc_dim + mix_hc + 3) * quant_mode.value.memory
            activation_bytes = sites * nt * hc_dim * quant_mode.value.memory * (3 if op_name == "both" else 2)
            if op_name in {"pre", "both"}:
                activation_bytes += sites * nt * (2 * hc_mult + hc_mult * hc_mult) * 4
            sol_math = ops / GEMM._get_quant_tc_flops(database.system_spec, quant_mode) * 1000
            sol_mem = (param_bytes + activation_bytes) / database.system_spec["gpu"]["mem_bw"] * 1000
            return max(sol_math, sol_mem), sol_math, sol_mem

        def get_empirical() -> float:
            # SOL / util from own num_tokens curve (per op slice); raises if no data.
            mhc_data = getattr(database, "_mhc_module_data", None)

            def _emp_for_op(op_name: str) -> float:
                sol_q = get_sol(num_tokens, op_name)[0]

                def _slice():
                    if not mhc_data:
                        raise PerfDataNotAvailableError("No DeepSeek-V4 mHC data is loaded.")
                    return util_empirical.require_data_slice(mhc_data, op_name, hc_mult, hidden_size)

                grid = util_empirical.grid_for(
                    (
                        "dsv4_mhc",
                        database.system,
                        database.backend,
                        database.version,
                        op_name,
                        hc_mult,
                        hidden_size,
                        quant_mode.name,
                    ),
                    _slice,
                    lambda c: get_sol(c[0], op_name)[0],  # c=(num_tokens,)
                    depth=1,
                )
                lat, _ = util_empirical.estimate(sol_q, (num_tokens,), grid)
                return lat

            if op == "both":
                return _emp_for_op("pre") + _emp_for_op("post")
            return _emp_for_op(op)

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol()
        if database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

        def get_silicon():
            mhc_data = getattr(database, "_mhc_module_data", None)
            if not mhc_data:
                raise PerfDataNotAvailableError(
                    f"DeepSeek-V4 mHC module data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )

            def _lookup_single(op_name: str) -> PerformanceResult:
                # Validate bucket presence before chained indexing; mhc_data is
                # a nested defaultdict, so `mhc_data[op][hc_mult][hidden_size]`
                # would silently materialize empty dicts and then fall through
                # to _nearest_1d_point_helper with an empty key list, surfacing
                # as an opaque AssertionError instead of a structured
                # PerfDataNotAvailableError.
                if (
                    op_name not in mhc_data
                    or hc_mult not in mhc_data[op_name]
                    or hidden_size not in mhc_data[op_name][hc_mult]
                    or not mhc_data[op_name][hc_mult][hidden_size]
                ):
                    raise PerfDataNotAvailableError(
                        f"No mHC silicon data for op='{op_name}', hc_mult={hc_mult}, hidden_size={hidden_size}."
                    )
                mhc_dict = mhc_data[op_name][hc_mult][hidden_size]
                # 1-D tokens curve: RAW lerp in range; boundary util-hold via
                # the per-op mHC SOL beyond it.
                config = perf_interp.OpInterpConfig(
                    axes=("num_tokens",),
                    resolver=perf_interp.Grid(),
                    sol_fn=lambda t: get_sol(t, op_name)[0],
                )
                result = perf_interp.query(config, mhc_dict, num_tokens)
                latency = interpolation.get_value(result, "latency")
                energy = interpolation.get_value(result, "energy")
                return database._interp_pr(latency, energy=energy)

            # Silicon tables only store "pre" and "post" rows. For op=="both"
            # (still a supported input in DeepSeekV4MHCModule), aggregate the
            # two silicon look-ups so callers don't need to know about the
            # storage layout.
            if op == "both":
                pre_result = _lookup_single("pre")
                post_result = _lookup_single("post")
                # Use PerformanceResult's __add__ to merge sources correctly
                # (silicon + silicon -> silicon, mismatch -> mixed) instead of
                # constructing a new PR that would default-tag as silicon.
                return pre_result + post_result

            return _lookup_single(op)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query DeepSeek-V4 mHC module for {num_tokens=}, {hidden_size=}, "
                f"{hc_mult=}, {sinkhorn_iters=}, {op=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        result = database.query_mhc_module(
            num_tokens=-(-kwargs.get("x") // self._seq_split),  # CP: per-rank token count (ceil = busiest rank)
            hidden_size=self._hidden_size,
            hc_mult=self._hc_mult,
            sinkhorn_iters=self._sinkhorn_iters,
            op=self._op,
            quant_mode=self._quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# _BaseDeepSeekV4AttentionModule (shared metadata)
# ───────────────────────────────────────────────────────────────────────


class _BaseDeepSeekV4AttentionModule(Operation):
    """Common DeepSeek-V4 compressed attention module metadata.

    Not instantiated directly. Subclassed by ``ContextDeepSeekV4AttentionModule``
    and ``GenerationDeepSeekV4AttentionModule``, each of which owns its own
    silicon data cache.
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        *,
        cp_size: int = 1,
    ) -> None:
        super().__init__(name, scale_factor)
        self._cp_size = cp_size  # context parallelism (sglang AllGather); >1 only on context modules
        self._num_heads = num_heads
        self._native_heads = native_heads
        self._tp_size = tp_size
        self._hidden_size = hidden_size
        self._q_lora_rank = q_lora_rank
        self._o_lora_rank = o_lora_rank
        self._head_dim = head_dim
        self._rope_head_dim = rope_head_dim
        self._index_n_heads = index_n_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._window_size = window_size
        self._compress_ratio = compress_ratio
        self._o_groups = o_groups
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._weights = self._estimate_weights()

    def _estimate_weights(self) -> float:
        gemm_weight_elems = (
            self._hidden_size * self._q_lora_rank
            + self._q_lora_rank * self._num_heads * self._head_dim
            + self._hidden_size * self._head_dim
            + self._o_groups * self._o_lora_rank * self._hidden_size
        )
        bfloat16_weight_elems = self._num_heads * self._head_dim * self._o_lora_rank
        float32_weight_elems = self._num_heads
        if self._compress_ratio:
            compressor_mult = 2 if self._compress_ratio == 4 else 1
            gemm_weight_elems += 2 * self._hidden_size * compressor_mult * self._head_dim
            float32_weight_elems += self._compress_ratio * compressor_mult * self._head_dim
        if self._compress_ratio == 4:
            gemm_weight_elems += self._q_lora_rank * self._index_n_heads * self._index_head_dim
            gemm_weight_elems += 2 * self._hidden_size * 2 * self._index_head_dim
            bfloat16_weight_elems += self._hidden_size * self._index_n_heads
            float32_weight_elems += self._compress_ratio * 2 * self._index_head_dim
        return (
            gemm_weight_elems * self._gemm_quant_mode.value.memory
            + bfloat16_weight_elems * common.GEMMQuantMode.bfloat16.value.memory
            + float32_weight_elems * 4
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# ContextDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class ContextDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Context-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns three class-level caches:
    - ``_data_cache`` — merged ctx table (csa + hca split files combined)
    - ``_raw_data_cache`` — deepcopy of the merged table, kept untouched
      so the topk-piecewise lookup can consult the original
      compress_ratio==4 rows for boundary correctness.
    - ``_sparse_kernel_cache`` — dict ``{"paged_mqa_logits", "hca_attn"}``
      of ``LoadedOpData`` used for prefix kernel-Δ correction.
    """

    _data_cache: ClassVar[dict] = {}
    _raw_data_cache: ClassVar[dict] = {}
    _sparse_kernel_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the csa+hca context split files, merges them,
        deep-copies the merged dict for topk-piecewise lookup, and loads the
        two DSV4 sparse-kernel CSVs.

        Binds:
        - ``database._context_deepseek_v4_attention_module_data``
        - ``database._raw_context_deepseek_v4_attention_module_data``
        - ``database._dsv4_sparse_kernel_data``
        """
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)

            def _load(filename_enum):
                primary_path = os.path.join(data_dir, filename_enum.value)
                sources = database._build_op_sources(filename_enum, primary_path, system_data_root)
                return LoadedOpData(load_context_dsv4_kind_module_data(sources), filename_enum, primary_path)

            ctx_split = [
                _load(PerfDataFilename.dsv4_csa_context_module),
                _load(PerfDataFilename.dsv4_hca_context_module),
            ]
            cls._data_cache[key] = _load_dsv4_split(ctx_split)
            ctx_merged = cls._data_cache[key]
            # perf_interp resolves on the raw merged table directly; the raw
            # wrapper is kept as a plain alias for backward compatibility.
            cls._raw_data_cache[key] = ctx_merged

            def _load_sparse(filename_enum):
                primary_path = os.path.join(data_dir, filename_enum.value)
                sources = database._build_op_sources(filename_enum, primary_path, system_data_root)
                return LoadedOpData(load_dsv4_sparse_kernel_data(sources), filename_enum, primary_path)

            cls._sparse_kernel_cache[key] = {
                "paged_mqa_logits": _load_sparse(PerfDataFilename.dsv4_paged_mqa_logits_module),
                "hca_attn": _load_sparse(PerfDataFilename.dsv4_hca_attn_module),
                "csa_attn": _load_sparse(PerfDataFilename.dsv4_csa_attn_module),
            }

            cls._record_load()

        if "_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._context_deepseek_v4_attention_module_data = cls._data_cache[key]
        if "_raw_context_deepseek_v4_attention_module_data" not in database.__dict__:
            database._raw_context_deepseek_v4_attention_module_data = cls._raw_data_cache[key]
        if "_dsv4_sparse_kernel_data" not in database.__dict__:
            database._dsv4_sparse_kernel_data = cls._sparse_kernel_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()
        cls._raw_data_cache.clear()
        cls._sparse_kernel_cache.clear()
        cls._csa_topk_abs_cache.clear()

    # ------------------------------------------------------------------
    # Sparse-kernel lookup helper (formerly PerfDatabase._lookup_dsv4_sparse_kernel)
    # ------------------------------------------------------------------

    @classmethod
    def _lookup_sparse_kernel(
        cls,
        database: PerfDatabase,
        kernel: str,
        bs: int,
        isl: int,
        past_kv: int,
        tp_size: int,
        native_heads: int,
    ) -> Optional[float]:
        """Look up a sparse-kernel latency at (kernel, bs, isl, past_kv, tp).

        Strategy:
          1. Exact (bs, isl, past_kv, tp) hit  → return latency
          2. Cubic 3D interpolation on (past_kv, isl, bs) within the fixed tp slice
          3. If cubic fails, use the largest sampled batch no larger than the query
             batch that covers isl, interpolate on (past_kv, isl), then scale by batch.
        Returns None if the kernel CSV is not loaded.
        """
        from collections import defaultdict

        all_data = getattr(database, "_dsv4_sparse_kernel_data", None)
        if all_data is None or kernel not in all_data:
            return None
        loaded = all_data[kernel]
        if loaded is None or loaded.data is None:
            return None
        per_tp = loaded.data.get(native_heads)
        if per_tp is None:
            return None
        if tp_size in per_tp:
            per_tp_dict = per_tp[tp_size]
        elif 1 in per_tp:
            # paged_mqa_logits is collected at tp=1 only — kernel work itself
            # is TP-independent so we fall back when caller asks for tp>1.
            per_tp_dict = per_tp[1]
        else:
            return None
        if not per_tp_dict:
            return None

        def _finite_latency(value):
            return value is not None and bool(np.all(np.isfinite(np.asarray(value))))

        if past_kv in per_tp_dict and isl in per_tp_dict[past_kv] and bs in per_tp_dict[past_kv][isl]:
            latency = per_tp_dict[past_kv][isl][bs]["latency"]
            if _finite_latency(latency):
                return float(np.asarray(latency))

        try:
            result = interpolation.interp_3d(past_kv, isl, bs, per_tp_dict, "cubic", database._extracted_metrics_cache)
            latency = result.get("latency") if isinstance(result, dict) else None
            if _finite_latency(latency):
                return float(np.asarray(latency))
        except Exception:
            pass

        batch_points = sorted(
            {
                sampled_b
                for isl_dict in per_tp_dict.values()
                if isinstance(isl_dict, dict)
                for isl_data in isl_dict.values()
                if isinstance(isl_data, dict)
                for sampled_b in isl_data
                if sampled_b <= bs
            },
            reverse=True,
        )

        def _lookup_at_batch(bp, *, allow_extrapolate: bool):
            batch_slice = defaultdict(dict)
            for sampled_past, isl_dict in per_tp_dict.items():
                if not isinstance(isl_dict, dict):
                    continue
                for sampled_isl, isl_data in isl_dict.items():
                    if isinstance(isl_data, dict) and bp in isl_data:
                        batch_slice[sampled_past][sampled_isl] = isl_data[bp]

            if not batch_slice:
                return None

            try:
                latency = interpolation.interp_2d_linear(past_kv, isl, batch_slice, database._extracted_metrics_cache)[
                    "latency"
                ]
                if _finite_latency(latency):
                    return latency
            except Exception:
                pass

            def _lookup_isl_at_past(sampled_past):
                isl_dict = batch_slice.get(sampled_past)
                if not isinstance(isl_dict, dict):
                    return None
                if isl in isl_dict:
                    return isl_dict[isl].get("latency")
                isl_points = sorted(s for s, leaf in isl_dict.items() if isinstance(leaf, dict) and "latency" in leaf)
                if len(isl_points) < 2:
                    return None
                if not allow_extrapolate and not (isl_points[0] <= isl <= isl_points[-1]):
                    return None
                left, right = interpolation.nearest_1d_point_helper(isl, isl_points, inner_only=not allow_extrapolate)
                latency = interpolation.interp_1d(
                    [left, right],
                    [isl_dict[left].get("latency"), isl_dict[right].get("latency")],
                    isl,
                )
                return latency

            if past_kv in batch_slice:
                return _lookup_isl_at_past(past_kv)

            past_points = sorted(
                sampled_past for sampled_past in batch_slice if _lookup_isl_at_past(sampled_past) is not None
            )
            if len(past_points) < 2:
                return None
            if not allow_extrapolate and not (past_points[0] <= past_kv <= past_points[-1]):
                return None
            left, right = interpolation.nearest_1d_point_helper(past_kv, past_points, inner_only=not allow_extrapolate)
            left_latency = _lookup_isl_at_past(left)
            right_latency = _lookup_isl_at_past(right)
            if left_latency is None or right_latency is None:
                return None
            return interpolation.interp_1d([left, right], [left_latency, right_latency], past_kv)

        for allow_extrapolate in (False, True):
            for bp in batch_points:
                try:
                    latency = _lookup_at_batch(bp, allow_extrapolate=allow_extrapolate)
                    if latency is None:
                        continue
                    latency = latency * bs / bp
                    if _finite_latency(latency):
                        return latency
                except Exception:
                    continue
        return None

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_context_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_attn_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
        *,
        prefix: int = 0,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_context_deepseek_v4_attention_module``."""
        cls.load_data(database)

        def get_sol(b_: int = b, s_: int = s, prefix_: int = prefix) -> tuple[float, float, float]:
            return _deepseek_v4_attention_sol(
                database,
                is_context=True,
                b=b_,
                s=s_,
                prefix=prefix_,
                num_heads=num_heads,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                o_lora_rank=o_lora_rank,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                window_size=window_size,
                compress_ratio=compress_ratio,
                o_groups=o_groups,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
            )

        def get_empirical() -> float:
            # SOL / util from own prefix-resolved (prefix, s, b) grid; raises if no data.
            sol_q = get_sol()[0]  # true SOL(s, prefix)

            def _slice():
                data = getattr(database, "_context_deepseek_v4_attention_module_data", None)
                if not data:
                    raise PerfDataNotAvailableError("No context DeepSeek-V4 attention data is loaded.")
                quant_data = util_empirical.require_data_slice(
                    data,
                    fmha_quant_mode,
                    kvcache_quant_mode,
                    gemm_quant_mode,
                )
                head_axis = _dsv4_resolve_head_key(quant_data, num_heads)
                if head_axis is None:
                    raise PerfDataNotAvailableError("No context DeepSeek-V4 attention head slice is available.")
                return util_empirical.require_data_slice(
                    quant_data,
                    head_axis,
                    compress_ratio,
                )  # {prefix: {s: {b: leaf}}}

            try:
                prefix_keys = tuple(sorted(_slice().keys()))
            except PerfDataNotAvailableError:
                prefix_keys = ()
            # Genuine prefix interpolation needs >=2 collected prefix points bracketing the
            # query. A degenerate axis (e.g. prefix=0 only, as collected on sglang) or an
            # out-of-range query would otherwise borrow util at the query's own small-s point,
            # crossing the indexer/window regime and the launch-overhead floor and inflating the
            # estimate. Anchor instead at the prefix=0 slice at full_s = s + prefix (regime-
            # matched), with the prefix effect carried by sol_q -- same as context attention/MLA
            # and the DSA context fallback.
            interp_prefix = len(prefix_keys) >= 2 and prefix_keys[0] <= prefix <= prefix_keys[-1]

            if interp_prefix:
                depth, query, slice_fn, key_tag = 3, (prefix, s, b), _slice, "dsv4_ctx_attn"

                def _sol(c):
                    return get_sol(c[2], c[1], c[0])[0]  # c=(prefix, s, b)
            else:
                depth, query, key_tag = 2, (s + prefix, b), "dsv4_ctx_attn_p0anchor"

                def slice_fn():
                    return util_empirical.require_data_slice(_slice(), 0)  # {s: {b: leaf}}

                def _sol(c):
                    return get_sol(c[1], c[0], 0)[0]  # c=(s, b); anchor prefix=0

            grid = util_empirical.grid_for(
                (
                    key_tag,
                    database.system,
                    database.backend,
                    database.version,
                    fmha_quant_mode.name,
                    kvcache_quant_mode.name,
                    gemm_quant_mode.name,
                    num_heads,
                    compress_ratio,
                    depth,
                ),
                slice_fn,
                _sol,
                depth=depth,
            )
            lat, _ = util_empirical.estimate(sol_q, query, grid)
            return lat

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol()
        if database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

        def get_silicon():
            data = getattr(database, "_context_deepseek_v4_attention_module_data", None)
            if not data:
                raise PerfDataNotAvailableError(
                    f"DeepSeek-V4 context attention module data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )
            # SCHEME A: head axis is the rank-local head count the model passes.
            quant_data = util_empirical.require_data_slice(
                data,
                fmha_quant_mode,
                kvcache_quant_mode,
                gemm_quant_mode,
            )
            head_axis = _dsv4_resolve_head_key(quant_data, num_heads)
            if head_axis is None:
                raise PerfDataNotAvailableError(
                    f"No DeepSeek-V4 context attention silicon data for num_heads={num_heads}, "
                    f"loaded head keys={list(quant_data.keys())}."
                )
            cr_dict = quant_data[head_axis].get(compress_ratio)
            if cr_dict is None:
                raise PerfDataNotAvailableError(
                    f"No DeepSeek-V4 context attention silicon data for num_heads={num_heads}, "
                    f"compress_ratio={compress_ratio}, loaded cr keys="
                    f"{list(quant_data[head_axis].keys())}."
                )

            # SCHEME A: cr_dict is prefix-resolved -> {prefix: {s: {b: leaf}}}.
            # RAW 3-axis grid: the CSA leave-one-out (gb200) showed plain linear
            # crossing at 1.72% median (regime-aware 1.92%; knee-just-above plain
            # +0.57% vs regime -2.94%), so no regime special-casing. A prefix
            # beyond the collected range is util-hold — the prefix-aware SOL
            # carries the effect (the empirical path anchors the same way) —
            # replacing the legacy clamp-at-boundary.
            config = perf_interp.OpInterpConfig(
                axes=("prefix", "seq_len", "batch"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda p_v, s_v, b_v: get_sol(b_v, s_v, p_v)[0],
            )
            try:
                result = perf_interp.query(config, cr_dict, prefix, s, b)
            except interpolation.InterpolationDataNotAvailableError as exc:
                raise PerfDataNotAvailableError(
                    f"DeepSeek-V4 prefix-resolved context attention module data not available for "
                    f"{b=}, {s=}, {prefix=}, {num_heads=}, {compress_ratio=}."
                ) from exc
            latency = float(interpolation.get_value(result, "latency"))
            energy = float(interpolation.get_value(result, "energy"))

            # SCHEME A: for CSA (compress_ratio==4) ONLY, subtract the measured
            # topK DELTA = flat_ms - top_last_ms (degenerate collector topK vs
            # representative silicon topK). HCA (cr==128) is left untouched.
            if compress_ratio == 4 and _TOPK_CORRECTION_ENABLED:
                calib = _get_dsv4_topk_calib(database)
                delta = _dsv4_topk_delta_ms(calib, int(prefix), int(s), int(b))
                corrected_latency = max(0.0, latency - delta)
                if latency > 0.0 and energy:
                    energy *= corrected_latency / latency
                latency = corrected_latency
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query DeepSeek-V4 context attention module for {b=}, {s=}, {prefix=}, "
                f"{num_heads=}, {native_heads=}, {tp_size=}, {compress_ratio=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def _module_base(self, database: PerfDatabase, b: int, s: int, prefix: int):
        """The per-(b,s,prefix) context attention module latency (non-CP path)."""
        return database.query_context_deepseek_v4_attention_module(
            b=b,
            s=s,
            prefix=prefix,
            num_heads=self._num_heads,
            native_heads=self._native_heads,
            tp_size=self._tp_size,
            hidden_size=self._hidden_size,
            q_lora_rank=self._q_lora_rank,
            o_lora_rank=self._o_lora_rank,
            head_dim=self._head_dim,
            rope_head_dim=self._rope_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            window_size=self._window_size,
            compress_ratio=self._compress_ratio,
            o_groups=self._o_groups,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        isl = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)
        if self._cp_size and self._cp_size > 1:
            return self._query_cp(database, batch_size, isl, prefix)
        result = self._module_base(database, batch_size, isl, prefix)
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    # ------------------------------------------------------------------
    # Context-Parallel (CP) prefill model — DeepSeek-V4 CSA / HCA.
    # Self-contained branch (does NOT reuse the non-CP topk-DELTA correction).
    # Mirrors GLM-5 ContextDSAModule._query_cp: per-card monolithic base +
    # full/cp swap of the super-linear sub-kernels + CP all-gathers.
    #   CSA (compress_ratio==4): base(per_card) + mqa full/cp + topk full/cp
    #       + AG(indexer key) + AG(compressed KV).
    #   HCA (compress_ratio==128 / SWA): base(per_card) + AG(windowed dense KV)
    #       + AG(compressed KV)  — windowed dense, no indexer/topk selection.
    # AG sizes follow DeepSeekV4Model.get_kvcache_bytes_per_sequence (head_dim
    # entries; indexer index_head_dim; compressed isl//ratio; window-capped).
    # ------------------------------------------------------------------
    _csa_topk_abs_cache: ClassVar[dict] = {}

    def _query_cp(self, database: PerfDatabase, b: int, isl: int, prefix: int) -> PerformanceResult:
        cp = self._cp_size
        per_card = max(1, -(-isl // cp))  # ceil: busiest CP rank = critical path
        ratio = self._compress_ratio
        latency = float(self._module_base(database, b, per_card, prefix))
        head_dim = self._head_dim

        # x b throughout: mqa/topk are linear in batch (b independent sequences)
        # and the all-gather moves b sequences' worth of KV. The sparse lookups
        # are at bs=1, so x b matches the batch-b base. (b==1 -> unchanged.)
        def ag(message_elems):
            return float(database.query_nccl(common.CommQuantMode.half, cp, "all_gather", int(b * message_elems)))

        if ratio == 4:
            # CSA: super-linear indexer (mqa) + topk -> full/cp swap (GLM-5 form).
            # Look up at the REAL batch b (paged_mqa_logits interpolates the bs
            # axis; csa_topk keeps every collected bs), so the delta matches the
            # batch-b base WITHOUT an external x b linearity assumption.
            mqa_full = self._lookup_sparse_kernel(
                database, "paged_mqa_logits", b, isl, prefix, self._tp_size, self._native_heads
            )
            mqa_perc = self._lookup_sparse_kernel(
                database, "paged_mqa_logits", b, per_card, prefix, self._tp_size, self._native_heads
            )
            tl_full = self._csa_topk_top_last(database, isl, prefix, self._native_heads, b)
            tl_perc = self._csa_topk_top_last(database, per_card, prefix, self._native_heads, b)
            # Fail loud (like GLM-5 ContextDSAModule._query_cp): the CSA CP deltas
            # REQUIRE the sparse tables -- silently dropping them would hide a
            # missing/uncollected parquet behind a too-small base-only estimate.
            if None in (mqa_full, mqa_perc, tl_full, tl_perc):
                raise PerfDataNotAvailableError(
                    "DeepSeek-V4 CSA CP modeling needs sparse tables (paged_mqa_logits + "
                    f"csa_topk_calib top_last) at num_heads={self._native_heads}, b={b}; "
                    "collect dsv4_paged_mqa_logits_module / dsv4_csa_topk_calib first."
                )
            latency += (mqa_full / cp - mqa_perc) + (tl_full / cp - tl_perc)
            latency += ag(isl * self._index_head_dim)  # AG indexer key (mqa stage)
        else:
            # HCA (128) / SWA: windowed dense; no indexer/topk selection.
            window = self._window_size or isl
            latency += ag(min(isl, window) * head_dim)  # AG windowed dense KV (the HCA "+1")

        # Compressed-KV all-gather: both CSA and HCA gather the isl//ratio
        # compressed entries (the fmha-stage KV). Common to both branches.
        if ratio:
            latency += ag((isl // ratio) * head_dim)

        return PerformanceResult(latency * self._scale_factor, energy=0.0, source="estimated")

    @classmethod
    def _csa_topk_top_last(cls, database: PerfDatabase, isl: int, step: int, native_heads: int, b: int):
        """Absolute top_last topk latency at (isl, step) for batch ``b``. CP branch
        only — reads the raw flat/top_last rows of dsv4_csa_topk_calib (the non-CP
        path keeps only the flat-top_last DELTA), so this is a separate
        self-contained loader. Looks up at the real batch (every collected bs is
        kept) so the topk delta matches the batch-b base without an x b assumption."""
        from aiconfigurator.sdk.operations.dsa import ContextDSAModule

        by_bs = cls._load_csa_topk_top_last(database, native_heads)
        return ContextDSAModule._lookup_2d(ContextDSAModule._bs_slice(by_bs, b), isl, step)

    @classmethod
    def _load_csa_topk_top_last(cls, database: PerfDatabase, native_heads: int) -> dict:
        """{bs: {(isl, step): top_last_latency}} from dsv4_csa_topk_calib."""
        # Full database identity so two systems under the same root/backend/
        # version don't reuse each other's dsv4_csa_topk_calib table.
        key = (
            database.systems_root,
            database.system,
            database.backend,
            database.version,
            database.enable_shared_layer,
            native_heads,
        )
        if key in cls._csa_topk_abs_cache:
            return cls._csa_topk_abs_cache[key]
        import os

        import pandas as pd

        from aiconfigurator.sdk.perf_database import PerfDataFilename

        data_dir = os.path.join(
            database.systems_root, database.system_spec["data_dir"], database.backend, database.version
        )
        path = os.path.join(data_dir, PerfDataFilename.dsv4_csa_topk_calib.value)
        by_bs: dict = {}
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if "num_heads" in df:
                df = df[df["num_heads"] == native_heads]
            tl = df[df["score_mode"].astype(str) == "top_last"]
            for _, r in tl.iterrows():
                by_bs.setdefault(int(r["batch_size"]), {})[(int(r["isl"]), int(r["step"]))] = float(r["latency"])
        cls._csa_topk_abs_cache[key] = by_bs
        return by_bs


# ───────────────────────────────────────────────────────────────────────
# GenerationDeepSeekV4AttentionModule
# ───────────────────────────────────────────────────────────────────────


class GenerationDeepSeekV4AttentionModule(_BaseDeepSeekV4AttentionModule):
    """Decode-phase DeepSeek-V4 SWA/CSA/HCA compressed attention module.

    Owns ``_generation_deepseek_v4_attention_module_data`` (merged from
    csa+hca split files).
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
        """Idempotent. Loads the csa+hca generation split files, merges
        them, binds ``database._generation_deepseek_v4_attention_module_data``.
        """
        import os

        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)

            def _load(filename_enum):
                primary_path = os.path.join(data_dir, filename_enum.value)
                sources = database._build_op_sources(filename_enum, primary_path, system_data_root)
                return LoadedOpData(load_generation_dsv4_kind_module_data(sources), filename_enum, primary_path)

            gen_split = [
                _load(PerfDataFilename.dsv4_csa_generation_module),
                _load(PerfDataFilename.dsv4_hca_generation_module),
            ]
            cls._data_cache[key] = _load_dsv4_split(gen_split)

            cls._record_load()

        if "_generation_deepseek_v4_attention_module_data" not in database.__dict__:
            database._generation_deepseek_v4_attention_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_generation_deepseek_v4_attention_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_attn_table(
        cls,
        database: PerfDatabase,
        b: int,
        s: int,
        num_heads: int,
        native_heads: int,
        tp_size: int,
        hidden_size: int,
        q_lora_rank: int,
        o_lora_rank: int,
        head_dim: int,
        rope_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        window_size: int,
        compress_ratio: int,
        o_groups: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode = common.GEMMQuantMode.bfloat16,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult | tuple[float, float, float]:
        """Verbatim port of legacy ``PerfDatabase.query_generation_deepseek_v4_attention_module``."""
        cls.load_data(database)

        def get_sol(b_: int = b, s_: int = s) -> tuple[float, float, float]:
            return _deepseek_v4_attention_sol(
                database,
                is_context=False,
                b=b_,
                s=s_,
                prefix=0,
                num_heads=num_heads,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                o_lora_rank=o_lora_rank,
                head_dim=head_dim,
                rope_head_dim=rope_head_dim,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
                window_size=window_size,
                compress_ratio=compress_ratio,
                o_groups=o_groups,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
            )

        def get_empirical() -> float:
            # SOL / util from own (b, s_total) grid; raises if no data.
            sol_q = get_sol()[0]

            def _slice():
                data = getattr(database, "_generation_deepseek_v4_attention_module_data", None)
                if not data:
                    raise PerfDataNotAvailableError("No generation DeepSeek-V4 attention data is loaded.")
                quant_data = util_empirical.require_data_slice(data, kvcache_quant_mode, gemm_quant_mode)
                head_axis = _dsv4_resolve_head_key(quant_data, num_heads)
                if head_axis is None:
                    raise PerfDataNotAvailableError("No generation DeepSeek-V4 attention head slice is available.")
                return util_empirical.require_data_slice(
                    quant_data,
                    head_axis,
                    compress_ratio,
                )  # {b: {s_total: leaf}}

            grid = util_empirical.grid_for(
                (
                    "dsv4_gen_attn",
                    database.system,
                    database.backend,
                    database.version,
                    kvcache_quant_mode.name,
                    gemm_quant_mode.name,
                    num_heads,
                    compress_ratio,
                ),
                _slice,
                lambda c: get_sol(c[0], c[1])[0],  # c=(b, s_total)
                depth=2,
            )
            lat, _ = util_empirical.estimate(sol_q, (b, s), grid)
            return lat

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if database_mode == common.DatabaseMode.SOL_FULL:
            return get_sol()
        if database_mode == common.DatabaseMode.EMPIRICAL:
            return PerformanceResult(get_empirical(), energy=0.0, source="empirical")

        def get_silicon():
            data = getattr(database, "_generation_deepseek_v4_attention_module_data", None)
            if not data:
                raise PerfDataNotAvailableError(
                    f"DeepSeek-V4 generation attention module data not loaded for system='{database.system}', "
                    f"backend='{database.backend}', version='{database.version}'."
                )
            # SCHEME A: head axis is the rank-local head count the model passes.
            quant_data = util_empirical.require_data_slice(data, kvcache_quant_mode, gemm_quant_mode)
            head_axis = _dsv4_resolve_head_key(quant_data, num_heads)
            if head_axis is None:
                raise PerfDataNotAvailableError(
                    f"No DeepSeek-V4 generation attention silicon data for num_heads={num_heads}, "
                    f"loaded head keys={list(quant_data.keys())}."
                )
            deepseek_v4_dict = quant_data[head_axis].get(compress_ratio)
            if deepseek_v4_dict is None:
                raise PerfDataNotAvailableError(
                    f"No DeepSeek-V4 generation attention silicon data for num_heads={num_heads}, "
                    f"compress_ratio={compress_ratio}, loaded cr keys={list(quant_data[head_axis].keys())}."
                )
            # SCHEME A generation dict is {b: {s_total: leaf}} after head/cr
            # slicing -> 2-axis RAW grid (decode ~linear in s_total); beyond the
            # collected range is util-hold via the decode SOL.
            config = perf_interp.OpInterpConfig(
                axes=("batch", "seq_len"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda b_v, s_v: get_sol(b_v, s_v)[0],
            )
            result = perf_interp.query(config, deepseek_v4_dict, b, s)
            latency = float(interpolation.get_value(result, "latency"))
            energy = float(interpolation.get_value(result, "energy"))

            # SCHEME A: subtract the topK DELTA for CSA (cr==4) only. Decode is
            # q_len=1 with past_kv = s_total - 1.
            if compress_ratio == 4 and _TOPK_CORRECTION_ENABLED:
                calib = _get_dsv4_topk_calib(database)
                decode_prefix = max(int(s) - 1, 0)
                delta = _dsv4_topk_delta_ms(calib, decode_prefix, 1, int(b))
                corrected_latency = max(0.0, latency - delta)
                if latency > 0.0 and energy:
                    energy *= corrected_latency / latency
                latency = corrected_latency
            return database._interp_pr(latency, energy=energy)

        return database._query_silicon_or_hybrid(
            get_silicon=get_silicon,
            get_empirical=get_empirical,
            database_mode=database_mode,
            error_msg=(
                f"Failed to query DeepSeek-V4 generation attention module for {b=}, {s=}, "
                f"{num_heads=}, {native_heads=}, {tp_size=}, {compress_ratio=}"
            ),
        )

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        beam_width = kwargs.get("beam_width")
        if beam_width != 1:
            raise ValueError(f"{self.__class__.__name__} only supports beam_width=1, got {beam_width}")
        result = database.query_generation_deepseek_v4_attention_module(
            b=kwargs.get("batch_size"),
            s=kwargs.get("s"),
            num_heads=self._num_heads,
            native_heads=self._native_heads,
            tp_size=self._tp_size,
            hidden_size=self._hidden_size,
            q_lora_rank=self._q_lora_rank,
            o_lora_rank=self._o_lora_rank,
            head_dim=self._head_dim,
            rope_head_dim=self._rope_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            window_size=self._window_size,
            compress_ratio=self._compress_ratio,
            o_groups=self._o_groups,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )
        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )


class DeepSeekV4MegaMoEModule(Operation):
    """
    SGLang DeepSeek-V4 MegaMoE routed module.

    This models the measured routed MegaMoE module boundary used by
    ``collector/sglang/collect_dsv4_megamoe.py``: prepared hidden states and
    top-k tensors -> SGLang pre-dispatch -> ``deep_gemm.fp8_fp4_mega_moe`` ->
    routed output scaling. Gate/top-k and shared experts are modeled outside
    this operation.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        source_policy: str = "random",
        pre_dispatch: str = "sglang_jit",
        num_fused_shared_experts: int = 0,
        kernel_source: str = "deepgemm_megamoe",
        kernel_dtype: str = "fp8_fp4",
    ) -> None:
        super().__init__(name, scale_factor)
        self._hidden_size = hidden_size
        self._inter_size = inter_size
        self._topk = topk
        self._num_experts = num_experts
        self._moe_tp_size = moe_tp_size
        self._moe_ep_size = moe_ep_size
        self._quant_mode = quant_mode
        self._workload_distribution = self._normalize_distribution(workload_distribution)
        self._is_context = is_context
        self._source_policy = source_policy
        self._pre_dispatch = pre_dispatch
        self._num_fused_shared_experts = num_fused_shared_experts
        self._kernel_source = kernel_source
        self._kernel_dtype = kernel_dtype
        self._weights = (
            self._hidden_size
            * self._inter_size
            * self._num_experts
            * quant_mode.value.memory
            # DSv4 MegaMoE is always gated SwiGLU: 3 GEMMs (gate, up, down).
            * 3
            // self._moe_ep_size
            // self._moe_tp_size
        )

    @staticmethod
    def _normalize_distribution(workload_distribution: str) -> str:
        if workload_distribution == "uniform":
            return "balanced"
        return workload_distribution

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            data_dir = os.path.join(system_data_root, database.backend, database.version)
            primary_path = os.path.join(data_dir, PerfDataFilename.dsv4_megamoe_module.value)
            cls._data_cache[key] = LoadedOpData(
                load_dsv4_megamoe_module_data(primary_path), PerfDataFilename.dsv4_megamoe_module, primary_path
            )
            cls._record_load()

        if "_dsv4_megamoe_module_data" not in database.__dict__:
            database._dsv4_megamoe_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    @classmethod
    def _query_megamoe_table(
        cls,
        database: PerfDatabase,
        num_tokens: int,
        hidden_size: int,
        inter_size: int,
        topk: int,
        num_experts: int,
        moe_tp_size: int,
        moe_ep_size: int,
        quant_mode: common.MoEQuantMode,
        workload_distribution: str,
        is_context: bool = True,
        source_policy: str = "random",
        pre_dispatch: str = "sglang_jit",
        num_fused_shared_experts: int = 0,
        kernel_source: str = "deepgemm_megamoe",
        kernel_dtype: str = "fp8_fp4",
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        """
        Query DeepSeek-V4 MegaMoE full-module latency.

        This table is intentionally strict: it models only measured fused
        MegaMoE rows and does not fall back to uniform/random distributions or
        analytical constants when a row is missing. New databases use the
        unified ``dsv4_megamoe_module`` file for both context and generation;
        ``is_context`` selects the phase stored inside that table.
        """
        cls.load_data(database)

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode not in (common.DatabaseMode.SILICON, common.DatabaseMode.HYBRID):
            raise PerfDataNotAvailableError(
                f"DSv4 MegaMoE module only supports measured SILICON data, got {database_mode=}."
            )

        if not isinstance(quant_mode, common.MoEQuantMode):
            quant_mode = common.MoEQuantMode[str(quant_mode)]
        phase = "context" if is_context else "generation"

        module_data = getattr(database, "_dsv4_megamoe_module_data", None)
        if module_data is None:
            raise PerfDataNotAvailableError(
                f"DSv4 MegaMoE module data not loaded for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}'."
            )
        module_data.raise_if_not_loaded()

        try:
            token_dict = module_data[phase][kernel_source][kernel_dtype][quant_mode][pre_dispatch][source_policy][
                workload_distribution
            ][topk][num_experts][num_fused_shared_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
        except KeyError as exc:
            raise PerfDataNotAvailableError(
                f"No DSv4 MegaMoE {phase} module data for {kernel_source=}, {kernel_dtype=}, {quant_mode=}, "
                f"{pre_dispatch=}, {source_policy=}, {workload_distribution=}, {topk=}, {num_experts=}, "
                f"{num_fused_shared_experts=}, {hidden_size=}, {inter_size=}, "
                f"{moe_tp_size=}, {moe_ep_size=}."
            ) from exc

        # 1-D tokens curve. No analytic SOL is implemented for the fused
        # MegaMoE module, but util-hold only needs the SOL RATIO: routed-expert
        # work scales ~linearly with tokens at fixed topk/experts/hidden, so a
        # linear token proxy is ratio-equivalent (see the DeepEP note).
        config = perf_interp.OpInterpConfig(
            axes=("num_tokens",),
            resolver=perf_interp.Grid(),
            sol_fn=lambda t: float(t),
        )
        result = perf_interp.query(config, token_dict, num_tokens)
        latency = float(interpolation.get_value(result, "latency"))
        energy = float(interpolation.get_value(result, "energy"))
        return PerformanceResult(latency, energy=energy)

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """Query measured MegaMoE routed-module latency."""
        sm_version = int(database.system_spec.get("gpu", {}).get("sm_version", -1))
        if sm_version < 100:
            raise ValueError(
                "DeepSeek-V4 MegaMoE is only supported on Blackwell-class GPUs "
                f"(SM >= 100); got sm_version={sm_version}."
            )

        # DSv4 MegaMoE perf rows are indexed by local-rank tokens. Do not
        # multiply by attention_dp_size here; the old decomposed MoE table is
        # indexed differently.
        x = int(kwargs.get("x"))
        overwrite_quant_mode = kwargs.get("quant_mode")
        quant_mode = self._quant_mode if overwrite_quant_mode is None else overwrite_quant_mode

        result = database.query_dsv4_megamoe_module(
            num_tokens=x,
            hidden_size=self._hidden_size,
            inter_size=self._inter_size,
            topk=self._topk,
            num_experts=self._num_experts,
            moe_tp_size=self._moe_tp_size,
            moe_ep_size=self._moe_ep_size,
            quant_mode=quant_mode,
            workload_distribution=self._workload_distribution,
            is_context=self._is_context,
            source_policy=self._source_policy,
            pre_dispatch=self._pre_dispatch,
            num_fused_shared_experts=self._num_fused_shared_experts,
            kernel_source=self._kernel_source,
            kernel_dtype=self._kernel_dtype,
        )

        return PerformanceResult(
            float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


# ───────────────────────────────────────────────────────────────────────
# Init-time split-file merge helper (formerly in PerfDatabase.__init__)
# ───────────────────────────────────────────────────────────────────────


def _load_dsv4_split(loaded_list):
    """Merge per-(attn_kind) loaded data into one combined ``LoadedOpData``.

    Each DSV4 context/generation module CSV is collected per attention kind
    (csa/hca). Each loader returns a nested dict scoped to one
    compress_ratio. We merge into one aggregate dict so downstream queries
    do not need to know which attention kind produced each row.
    """
    from aiconfigurator.sdk.perf_database import LoadedOpData

    merged: dict = {}
    first_loaded = next((x for x in loaded_list if x is not None), None)
    if first_loaded is None:
        return None
    for loaded in loaded_list:
        if loaded is None or not loaded.loaded:
            continue
        _deep_merge_dsv4_dicts(merged, loaded.data)
    if not merged:
        return None
    return LoadedOpData(merged, first_loaded.op_name_enum, first_loaded.filepath)


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_mhc_module_data(mhc_file: str):
    """Load DeepSeek-V4 mHC pre/post module-level performance data.

    CSV columns: framework, version, device, op_name, kernel_source,
    architecture, num_tokens, hc_mult, hidden_size, latency [, power]
    Optional metadata columns: num_sites, sinkhorn_iters
    Legacy rows may include a ``model`` column; it is ignored because mHC is
    selected by compute shape.

    ``op_name`` is ``pre`` or ``post``, matching the ``op`` arg of
    ``query_mhc_module``.

    Dict structure (matches query_mhc_module silicon path):
        data[op][hc_mult][hidden_size][num_tokens]
    """
    rows = _read_filtered_rows(mhc_file)
    if rows is None:
        logger.debug(f"mHC module data file {mhc_file} not found.")
        return None

    mhc_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict())))

    has_power = len(rows) > 0 and "power" in rows[0]

    for row in rows:
        op = row["op_name"]
        hc_mult = int(row["hc_mult"])
        hidden_size = int(row["hidden_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency

        mhc_data[op][hc_mult][hidden_size][num_tokens] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }

    return mhc_data


_DSV4_DTYPE_ALIASES = {
    # CSV columns use sglang naming; aic_dev enums use canonical short names.
    "fp8_e4m3": "fp8",
}


def _dsv4_normalize_dtype(name: str) -> str:
    return _DSV4_DTYPE_ALIASES.get(name, name)


# ───────────────────────────────────────────────────────────────────────
# DSV4 CSA topk DELTA calibration (SCHEME A correction).
#
# The CSA context-module collector runs the topK kernel on DEGENERATE scores
# (dummy weights + zero/uninitialized prefix KV -> near-constant logits -> the
# Small topK path falls into its O(n^2) tie-break). That inflates the measured
# module latency vs real silicon, where logits are spread. We measured the topK
# time standalone under a degenerate "flat" construction and a representative
# "top_last" construction for every (prefix, isl, batch_size) shape, and store
# as two rows (score_mode=flat | top_last, each a single latency) per shape in
# dsv4_csa_topk_calib_perf; DELTA = flat.latency - top_last.latency. At query
# time we SUBTRACT DELTA from the CSA (compress_ratio==4) module latency only.
# Gate: AIC_DSV4_TOPK_CORRECTION (default on; set "0" to disable).
# ───────────────────────────────────────────────────────────────────────
_TOPK_CORRECTION_ENABLED = os.environ.get("AIC_DSV4_TOPK_CORRECTION", "1") != "0"


def _build_topk_calib_from_rows(by_mode):
    """Pair the flat / top_last rows the sparse-op loader produced into the topK
    DELTA table ``{'exact': {(step, isl, bs): delta_ms},
                   'by_pi': {(step, isl): [(bs, delta_ms), ...]}}`` or ``None``.

    ``by_mode`` is the ``_TOPK_CALIB_KEYS`` nesting
    ``data[step][isl][bs][score_mode] = {"latency": ms}``. The calibration is
    stored as two rows per shape — ``score_mode=flat`` is the degenerate
    worst-case the module CSV captured, ``top_last`` the representative one —
    and DELTA = flat.latency - top_last.latency.
    """
    if not by_mode:
        return None
    exact = {}
    by_pi = {}
    for step, isl_d in by_mode.items():
        for isl, bs_d in isl_d.items():
            for bs, mode_d in bs_d.items():
                flat = mode_d.get("flat")
                top_last = mode_d.get("top_last")
                if not isinstance(flat, dict) or not isinstance(top_last, dict):
                    continue
                delta = max(0.0, float(flat["latency"]) - float(top_last["latency"]))
                exact[(step, isl, bs)] = delta
                by_pi.setdefault((step, isl), []).append((bs, delta))
    if not exact:
        return None
    for k in by_pi:
        by_pi[k].sort()
    return {"exact": exact, "by_pi": by_pi}


def _dsv4_interp_1d_from_points(points, x):
    """Linear interpolation with nearest-value extrapolation."""
    if not points:
        return None
    merged = defaultdict(list)
    for coord, value in points:
        merged[int(coord)].append(float(value))
    xs = sorted(merged)
    vals = {k: sum(v) / len(v) for k, v in merged.items()}
    if x in vals:
        return vals[x]
    if len(xs) == 1:
        return vals[xs[0]]
    if x <= xs[0]:
        return vals[xs[0]]
    if x >= xs[-1]:
        return vals[xs[-1]]
    left = max(v for v in xs if v < x)
    right = min(v for v in xs if v > x)
    if left == right:
        return vals[left]
    t = (float(x) - float(left)) / (float(right) - float(left))
    return vals[left] * (1.0 - t) + vals[right] * t


def _dsv4_topk_delta_ms(calib, prefix, isl, bs):
    """Return topK DELTA (flat_ms - top_last_ms) from measured calibration.

    Exact (prefix, isl, bs) rows are preferred; off-grid shapes are
    interpolated prefix-first within a fixed (isl, bs), then isl, then bs.
    Returns 0.0 when no calibration is available.
    """
    if not calib:
        return 0.0
    prefix = int(prefix)
    isl = int(isl)
    bs = int(bs)
    exact = calib.get("exact", {})
    direct = exact.get((prefix, isl, bs))
    if direct is not None:
        return max(0.0, float(direct))

    def _prefix_interp(query_prefix, anchor_isl, anchor_bs):
        points = [(p, d) for (p, i, b), d in exact.items() if int(i) == int(anchor_isl) and int(b) == int(anchor_bs)]
        return _dsv4_interp_1d_from_points(points, query_prefix)

    def _isl_interp(query_prefix, query_isl, anchor_bs):
        isl_values = sorted({i for (_, i, b) in exact if int(b) == int(anchor_bs)})
        points = []
        for i in isl_values:
            value = _prefix_interp(query_prefix, i, anchor_bs)
            if value is not None:
                points.append((i, value))
        return _dsv4_interp_1d_from_points(points, query_isl)

    bs_values = sorted({b for (_, _, b) in exact})
    points = []
    for b in bs_values:
        value = _isl_interp(prefix, isl, b)
        if value is not None:
            points.append((b, value))
    interpolated = _dsv4_interp_1d_from_points(points, bs)
    if interpolated is None:
        return 0.0
    return max(0.0, float(interpolated))


def _get_dsv4_topk_calib(database):
    """Load (and cache on ``database``) the CSA topK DELTA calibration through
    the same sparse-op loader + source resolution the other DSV4 ops use."""
    cached = getattr(database, "_dsv4_csa_topk_calib", _MISSING)
    if cached is not _MISSING:
        return cached
    import os as _os

    from aiconfigurator.sdk.perf_database import PerfDataFilename

    system_data_root = _os.path.join(database.systems_root, database.system_spec["data_dir"])
    data_dir = _os.path.join(system_data_root, database.backend, database.version)
    enum = PerfDataFilename.dsv4_csa_topk_calib
    primary_path = _os.path.join(data_dir, enum.value)
    sources = database._build_op_sources(enum, primary_path, system_data_root)
    by_mode = load_dsv4_sparse_op_data(sources, _TOPK_CALIB_KEYS)
    calib = _build_topk_calib_from_rows(by_mode)
    try:
        database._dsv4_csa_topk_calib = calib
    except Exception:
        pass
    return calib


_MISSING = object()


def load_context_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 context CSV (single attn_kind / compress_ratio).

    SCHEME A.  Returns a 7-level prefix-resolved nested dict:
        data[fmha_quant][kv_quant][gemm_quant][num_heads_local][compress_ratio]
            [prefix][s][b] = {"latency": ms, "power": W, "energy": J}

    The head axis is the rank-LOCAL head count = ``int(row["num_heads"])``
    (the collector writes ``local_attention_heads = native // tp``).  There is
    NO separate ``tp_size`` key and NO reconstructed native-head key.

    ``prefix`` is the past-KV length, ``int(float(row["step"]))``; ``s`` is the
    context chunk length (``isl``).  Multiple files (csa/hca) merge cleanly
    because compress_ratio is a key dimension.
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None

    # 7-level nesting: fmha → kv → gemm → num_heads_local → cr → prefix → s → b
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(7)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue  # skip duplicate header rows from appended runs
        try:
            b = int(row["batch_size"])
            s = int(row["isl"])
            prefix = int(float(row.get("step", 0) or 0))
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        # SCHEME A: head key is the rank-local head count straight from the CSV.
        num_heads_local = int(row["num_heads"])
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        fmha_mode = common.FMHAQuantMode[_dsv4_normalize_dtype(row["mla_dtype"])]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        # NOTE: the topK DELTA correction (degenerate -> representative) is
        # applied ONCE at query time for compress_ratio==4 (CSA). Do NOT
        # subtract it here, or the CSA module latency would be double-corrected.
        data[fmha_mode][kv_dtype][gemm_mode][num_heads_local][cr][prefix][s][b] = {
            "latency": latency,
            "power": power,
            "energy": power * latency,
        }
    return data


def load_generation_dsv4_kind_module_data(file_path: str):
    """Load ONE DeepSeek-V4 generation CSV.

    Generation lookup uses absolute KV length ``s_total = isl + step`` (decode
    is q_len=1 with past_kv = step).  SCHEME A dict shape:
        data[kv_quant][gemm_quant][num_heads_local][compress_ratio]
            [b][s_total]
    """
    rows = _read_filtered_rows(file_path)
    if rows is None:
        logger.debug(f"DSV4 module data file {file_path} not found.")
        return None

    # SCHEME A: 5-level nesting kv → gemm → num_heads_local → cr → b → s_total
    def _make_nested(depth: int):
        if depth == 0:
            return defaultdict()
        return defaultdict(lambda d=depth: _make_nested(d - 1))

    data = _make_nested(5)
    has_power = bool(rows) and "power" in rows[0]

    for row in rows:
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            b = int(row["batch_size"])
            s_total = int(row["isl"]) + int(row["step"])
            cr = int(row["compress_ratio"])
            latency = float(row["latency"])
        except (TypeError, ValueError, KeyError):
            continue
        power = float(row.get("power", 0.0)) if has_power else 0.0

        # SCHEME A: head key is the rank-local head count straight from the CSV;
        # no tp_size key, no native reconstruction.  Generation convention puts
        # ``b`` before ``s_total`` (matches the (head, b, s) lookup order).
        num_heads_local = int(row["num_heads"])
        gemm_mode = common.GEMMQuantMode[row["gemm_type"]]
        kv_dtype = common.KVCacheQuantMode[_dsv4_normalize_dtype(row["kv_cache_dtype"])]

        data[kv_dtype][gemm_mode][num_heads_local][cr][b][s_total] = {
            "latency": latency,
            "power": power,
            "energy": power * latency,
        }
    return data


def load_dsv4_megamoe_module_data(dsv4_megamoe_module_file):
    """
    Load DeepSeek-V4 MegaMoE full-module data.

    The collected latency is the SGLang/DeepGEMM MegaMoE routed path:
    prepared hidden states and top-k tensors -> pre-dispatch -> fused MegaMoE.
    Gate/top-k generation is intentionally outside the measured region.

    Returns:
        dict: Nested dict whose leaves contain latency, power, energy and
        routing metadata.
    """
    if dsv4_megamoe_module_file is None:
        return None

    if isinstance(dsv4_megamoe_module_file, list | tuple):
        raise TypeError("DSv4 MegaMoE data loader expects a single unified perf file path")

    source_label = os.fspath(dsv4_megamoe_module_file)
    rows = _read_filtered_rows(source_label)
    if rows is None:
        logger.debug(f"DeepSeek-V4 MegaMoE data file {source_label} not found.")
        return None

    def _to_bool(value: object) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    row_bool_invariants = [
        ("used_cuda_graph", True, None, "DSv4 MegaMoE perf row was not collected with CUDA Graph"),
        (
            "includes_gate_topk",
            False,
            "true",
            "DSv4 MegaMoE perf row includes gate/top-k outside the supported boundary",
        ),
        ("includes_routed_scale", True, None, "DSv4 MegaMoE perf row does not include SGLang routed output scaling"),
    ]

    def _row_phase(row: dict[str, str]) -> str:
        phase = row.get("phase", "").strip()
        if not phase:
            raise ValueError(f"DSv4 MegaMoE unified perf file requires a phase column: {source_label} {row}")
        if phase not in {"context", "generation"}:
            raise ValueError(f"DSv4 MegaMoE perf row has unsupported phase={phase!r}: {row}")
        return phase

    def _put_nested(root: dict, keys: list[object], value: dict) -> None:
        current = root
        for key in keys[:-1]:
            current = current.setdefault(key, {})
        leaf_key = keys[-1]
        if leaf_key in current:
            raise ValueError(f"duplicate DSv4 MegaMoE data row for {source_label} {keys}")
        current[leaf_key] = value

    dsv4_megamoe_data: dict = {}
    logger.debug(f"Loading DeepSeek-V4 MegaMoE module data from: {source_label}")
    for row in rows:
        for field, expected_value, default, error in row_bool_invariants:
            if _to_bool(row.get(field, default)) != expected_value:
                raise ValueError(f"{error}: {source_label} {row}")

        kernel_source = row.get("kernel_source", "deepgemm_megamoe")
        kernel_dtype = row["kernel_dtype"]
        quant_mode = common.MoEQuantMode[row["moe_dtype"]]
        pre_dispatch = row["pre_dispatch"]
        source_policy = row["source_policy"]
        distribution = row["distribution"]
        topk = int(row["topk"])
        num_experts = int(row["num_experts"])
        num_fused_shared_experts = int(row.get("num_fused_shared_experts", 0))
        hidden_size = int(row["hidden_size"])
        inter_size = int(row["inter_size"])
        moe_tp_size = int(row.get("moe_tp_size", 1))
        moe_ep_size = int(row["moe_ep_size"])
        num_tokens = int(row["num_tokens"])
        latency = float(row["latency"])
        power = float(row.get("power") or 0.0)
        energy = power * latency
        num_max_tokens_per_rank = int(row.get("num_max_tokens_per_rank") or 0)
        effective_num_max_tokens_per_rank = int(row.get("effective_num_max_tokens_per_rank") or num_max_tokens_per_rank)

        entry = {
            "latency": latency,
            "power": power,
            "energy": energy,
            "global_num_tokens": int(row.get("global_num_tokens") or num_tokens * moe_ep_size),
            "num_max_tokens_per_rank": num_max_tokens_per_rank,
            "effective_num_max_tokens_per_rank": effective_num_max_tokens_per_rank,
            "used_cuda_graph": True,
            "kernel_dtype": kernel_dtype,
            "routed_scaling_factor": float(row["routed_scaling_factor"]),
            "includes_routed_scale": True,
            "includes_gate_topk": False,
            "buffer_policy": row.get("buffer_policy", ""),
            "includes_buffer_init": _to_bool(row.get("includes_buffer_init", "false")),
        }
        phase = _row_phase(row)
        entry["phase"] = phase
        _put_nested(
            dsv4_megamoe_data,
            [
                phase,
                kernel_source,
                kernel_dtype,
                quant_mode,
                pre_dispatch,
                source_policy,
                distribution,
                topk,
                num_experts,
                num_fused_shared_experts,
                hidden_size,
                inter_size,
                moe_tp_size,
                moe_ep_size,
                num_tokens,
            ],
            entry,
        )

    return dsv4_megamoe_data


# ───────────────────────────────────────────────────────────────────────
# DSV4 sparse-op family loader (ONE engine for all four)
# ───────────────────────────────────────────────────────────────────────
# csa_attn / hca_attn / paged_mqa_logits (FMLA & indexer kernels) and the
# csa_topk_calib DELTA rows share ONE column schema, so they all parse through
# ``load_dsv4_sparse_op_data``; each consumer just supplies the key columns it
# indexes on (declared here so callers stay in sync).
_SPARSE_KERNEL_KEYS = ("num_heads", "tp_size", "step", "isl", "batch_size")
_TOPK_CALIB_KEYS = ("step", "isl", "batch_size", "score_mode")


def load_dsv4_sparse_op_data(file_or_sources, key_columns):
    """Generic loader for the DeepSeek-V4 sparse-op family.

    Reads the shared perf schema (parquet or txt, single path or override
    ``(path, kernel_source_filter)`` sources — see ``_read_filtered_rows``) and
    nests every row under ``key_columns`` in order, leaf == ``{"latency": ms}``.

    Numeric key cells coerce to ``int``; non-numeric stay ``str`` (e.g.
    ``score_mode``). Rows with a blank or NaN/inf key cell are skipped.
    Returns ``None`` when no source file exists.

    Consumers:
      - sparse kernels: ``_SPARSE_KERNEL_KEYS`` -> data[heads][tp][past_kv][isl][bs]
      - topk calib:     ``_TOPK_CALIB_KEYS``    -> data[step][isl][bs][score_mode]
    """
    rows = _read_filtered_rows(file_or_sources)
    if rows is None:
        return None

    def _coerce(value):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return value

    def _is_bad_key(k):
        # A key cell that is blank or a NaN/inf sentinel must not become a dict
        # key: such rows are malformed and would misbucket (or KeyError) the
        # downstream calibration lookup. Legitimate non-numeric keys (e.g.
        # ``score_mode`` values like ``"default"``) are kept.
        if k is None:
            return True
        if isinstance(k, float):  # uncoerced float NaN/inf
            return k != k or k in (float("inf"), float("-inf"))
        if isinstance(k, str):
            return k.strip() == "" or k.strip().lower() in (
                "nan",
                "inf",
                "-inf",
                "+inf",
                "infinity",
                "-infinity",
            )
        return False

    root: dict = {}
    for row in rows:
        # Skip duplicate header rows (files may be appended to across runs).
        if row.get("batch_size") in (None, "", "batch_size"):
            continue
        try:
            keys = [_coerce(row[col]) for col in key_columns]
            latency = float(row["latency"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(_is_bad_key(k) for k in keys):  # blank / NaN / inf key cell
            continue
        node = root
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = {"latency": latency}
    return root or None


def load_dsv4_sparse_kernel_data(file_or_sources):
    """DSV4 sparse-kernel CSV (csa_attn / hca_attn / paged_mqa_logits).

    Thin wrapper over ``load_dsv4_sparse_op_data`` with the kernel key columns,
    yielding ``data[native_heads][tp_size][past_kv][isl][bs] = {"latency": ms}``.
    """
    return load_dsv4_sparse_op_data(file_or_sources, _SPARSE_KERNEL_KEYS)
