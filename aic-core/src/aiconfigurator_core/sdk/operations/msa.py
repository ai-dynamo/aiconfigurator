# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax Sparse Attention (MSA) module ops for MiniMax-M3.

MSA (github.com/MiniMax-AI/MSA) is structurally a GQA version of DSA: an indexer
does a cheap per-block "dense proxy" pass to score KV blocks, the top-k blocks
are selected, and full attention runs over only the selected tokens. Versus DSA
the main attention is standard GQA (not MLA-compressed), and the indexer scores
per *block* (block_size tokens) rather than per token.

MSA now has its own module-level silicon tables (msa_context_module_perf /
msa_generation_module_perf, collected by ``collector/trtllm/collect_msa_module.py``
with the exact DSA-module row schema). SILICON resolves them on the raw grids
via perf_interp exactly like the DSA modules (context ``[num_heads][prefix][s][b]``,
generation ``[num_heads][b][s]``), with the analytic SOL below as the util-hold
anchor. HYBRID / EMPIRICAL try that silicon path first; when the table (or the
requested quant slice) is absent they fall back to the legacy CROSS-OP TRANSFER
from DSA's measured utilisation at the same workload, scaled by a manual
``dsa_scale_k`` (util_scale hook): ``latency = SOL_msa / (util_dsa * k)``. SOL
only needs to capture the (b, s, prefix) shape trend; k pulls the absolute
level. The transfer is gated by the XOP transfer kind and raises honestly when
neither MSA data nor a DSA util source exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import EmpiricalNotImplementedError, InterpolationDataNotAvailableError
from aiconfigurator_core.sdk.operations import util_empirical
from aiconfigurator_core.sdk.operations.base import Operation, resolve_op_data_path
from aiconfigurator_core.sdk.operations.dsa import (
    _cache_key,
    _select_dsa_backend,
    load_context_dsa_module_data,
    load_generation_dsa_module_data,
)
from aiconfigurator_core.sdk.operations.util_empirical import note_provenance
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


# The collector always stamps this architecture on MSA rows; it is the sole
# native head geometry in the table (see docs/perf_database/head-axis-keying.md).
DEFAULT_MSA_ARCHITECTURE = "MiniMaxM3ForCausalLM"

# Native MiniMax-M3 structural dims — defaults for the public
# ``query_*_msa_module`` wrappers' SOL anchor when a caller does not pass the
# op's own (rank-local) values. ``num_heads``/``num_kv_heads`` are the NATIVE
# (tp=1) head counts; per-rank kv heads default proportionally.
MSA_MODEL_DIMS: dict[str, dict] = {
    "MiniMaxM3ForCausalLM": {
        "num_heads": 64,
        "num_kv_heads": 4,
        "hidden_size": 6144,
        "head_dim": 128,
        "v_head_dim": 128,
        "index_n_heads": 4,
        "index_head_dim": 128,
        "index_topk": 2048,
        "block_size": 128,
    },
}


def _msa_attention_sol(
    database: PerfDatabase,
    *,
    is_context: bool,
    b: int,
    s: int,
    prefix: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    block_size: int,
    kvcache_quant_mode: common.KVCacheQuantMode,
    fmha_quant_mode: common.FMHAQuantMode,
    gemm_quant_mode: common.GEMMQuantMode,
) -> tuple[float, float, float]:
    """SOL for one MSA block. GQA projections + per-block FP8 indexer + sparse
    attention over the top-k (= index_topk) selected tokens.

    Mirrors DSA/DSV4's three-group structure (gemm / fp8 indexer / fmha attn);
    the attention group uses GQA dims (compute by num_heads, KV cache by
    num_kv_heads) and the top-k saturated causal pair count.
    """

    qk_head_dim = head_dim
    tokens = b * s if is_context else b
    # context: full prefill of `s` new tokens on top of `prefix` cached.
    # generation: 1 query token, kv_len = s - 1 cached.
    full_s = prefix + s if is_context else s
    kv_len = full_s if is_context else max(0, s - 1)

    # ── GEMM group (Q / GQA-KV / O / indexer-Q projections) ──────────────
    gemm_ops = (
        2 * tokens * hidden_size * (num_heads * qk_head_dim)  # Q
        + 2 * tokens * hidden_size * (2 * num_kv_heads * head_dim)  # K, V (GQA)
        + 2 * tokens * (num_heads * v_head_dim) * hidden_size  # O
        + 2 * tokens * hidden_size * (index_n_heads * index_head_dim)  # indexer Q
    )

    # ── sparse attention: top-k saturated causal (query, kv) pair count ──
    if is_context:
        if full_s <= index_topk:
            pairs = b * (full_s * (full_s + 1) - prefix * (prefix + 1)) // 2
        elif prefix >= index_topk:
            pairs = tokens * index_topk
        else:
            ramp = b * (index_topk * (index_topk + 1) - prefix * (prefix + 1)) // 2
            sat = b * (full_s - index_topk) * index_topk
            pairs = ramp + sat
        score_len = full_s
    else:
        pairs = tokens * min(kv_len, index_topk)
        score_len = kv_len
    effective_kv = min(kv_len, index_topk) if not is_context else min(full_s, index_topk)
    attention_ops = 2 * num_heads * (qk_head_dim + v_head_dim) * pairs  # QK^T + AV

    # ── indexer: per-block scoring (block_size tokens per block), FP8 ────
    num_blocks = (score_len + block_size - 1) // block_size if score_len > index_topk else 0
    indexer_ops = 2 * tokens * index_n_heads * index_head_dim * num_blocks

    # ── memory ───────────────────────────────────────────────────────────
    gemm_weight_bytes = (
        hidden_size * num_heads * qk_head_dim
        + hidden_size * 2 * num_kv_heads * head_dim
        + num_heads * v_head_dim * hidden_size
        + hidden_size * index_n_heads * index_head_dim
    ) * gemm_quant_mode.value.memory
    kv_cache_bytes = b * num_kv_heads * effective_kv * (qk_head_dim + v_head_dim) * kvcache_quant_mode.value.memory
    indexer_cache_bytes = b * num_blocks * index_n_heads * index_head_dim  # FP8 index keys, per block
    q_io_bytes = tokens * num_heads * qk_head_dim * fmha_quant_mode.value.memory * 2
    total_mem = gemm_weight_bytes + kv_cache_bytes + indexer_cache_bytes + q_io_bytes

    gemm_flops = common.get_quant_tc_flops(database.system_spec, gemm_quant_mode)
    fp8_flops = common.get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
    attn_flops = common.get_quant_tc_flops(database.system_spec, fmha_quant_mode)

    sol_math = (gemm_ops / gemm_flops + indexer_ops / fp8_flops + attention_ops / attn_flops) * 1000
    sol_mem = total_mem / database.system_spec["gpu"]["mem_bw"] * 1000
    sol_time = max(sol_math, sol_mem)
    return sol_time, sol_math, sol_mem


def _msa_sol_dims(architecture: str, num_heads: int, overrides: dict) -> dict:
    """Resolve the SOL-anchor dims for one MSA table query: caller overrides
    first, then the architecture's native dims (per-rank kv heads scaled from
    the native q:kv ratio when only ``num_heads`` is known)."""
    if architecture not in MSA_MODEL_DIMS:
        raise ValueError(
            f"Unknown MSA architecture '{architecture}'; known: {sorted(MSA_MODEL_DIMS)}. "
            "Substituting another architecture's geometry would silently anchor the "
            "SOL model on the wrong shape."
        )
    dims = MSA_MODEL_DIMS[architecture]
    resolved = {key: dims[key] for key in dims if key not in ("num_heads", "num_kv_heads")}
    resolved["num_kv_heads"] = max(1, num_heads * dims["num_kv_heads"] // dims["num_heads"])
    for key, value in overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved


def _dsa_context_util(
    database, *, b, s, prefix, num_heads, kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode, architecture
):
    """DSA's measured utilisation (SOL/silicon) at the same workload, or None."""
    try:
        sol = float(
            database.query_context_dsa_module(
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                architecture=architecture,
                database_mode=common.DatabaseMode.SOL,
            )
        )
        sil = float(
            database.query_context_dsa_module(
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                architecture=architecture,
                database_mode=common.DatabaseMode.SILICON,
            )
        )
        return sol / sil if sol > 0 and sil > 0 else None
    except Exception:
        return None


def _dsa_generation_util(database, *, b, s, num_heads, kvcache_quant_mode, gemm_quant_mode, architecture):
    try:
        sol = float(
            database.query_generation_dsa_module(
                b,
                s,
                num_heads,
                kvcache_quant_mode,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.SOL,
                architecture=architecture,
            )
        )
        sil = float(
            database.query_generation_dsa_module(
                b,
                s,
                num_heads,
                kvcache_quant_mode,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.SILICON,
                architecture=architecture,
            )
        )
        return sol / sil if sol > 0 and sil > 0 else None
    except Exception:
        return None


class _BaseMSAModule(Operation):
    """Shared MSA op: SOL + own silicon table + DSA cross-op-transfer fallback."""

    def __init__(
        self,
        name: str,
        scale_factor: float,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        v_head_dim: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        block_size: int,
        kvcache_quant_mode: common.KVCacheQuantMode,
        fmha_quant_mode: common.FMHAQuantMode,
        gemm_quant_mode: common.GEMMQuantMode,
        dsa_architecture: str = "GlmMoeDsaForCausalLM",
        dsa_scale_k: float = 1.0,
    ) -> None:
        super().__init__(name, scale_factor)
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self._hidden_size = hidden_size
        self._head_dim = head_dim
        self._v_head_dim = v_head_dim
        self._index_n_heads = index_n_heads
        self._index_head_dim = index_head_dim
        self._index_topk = index_topk
        self._block_size = block_size
        self._kvcache_quant_mode = kvcache_quant_mode
        self._fmha_quant_mode = fmha_quant_mode
        self._gemm_quant_mode = gemm_quant_mode
        self._dsa_architecture = dsa_architecture
        self._dsa_scale_k = dsa_scale_k
        self._weights = 0.0

    def _sol(self, database, b, s, prefix, is_context):
        return _msa_attention_sol(
            database,
            is_context=is_context,
            b=b,
            s=s,
            prefix=prefix,
            num_heads=self._num_heads,
            num_kv_heads=self._num_kv_heads,
            hidden_size=self._hidden_size,
            head_dim=self._head_dim,
            v_head_dim=self._v_head_dim,
            index_n_heads=self._index_n_heads,
            index_head_dim=self._index_head_dim,
            index_topk=self._index_topk,
            block_size=self._block_size,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
        )[0]

    def _own_table_kwargs(self) -> dict:
        """The op's structural dims, forwarded so the table query's SOL anchor
        matches this op exactly (the wrapper defaults only cover the native
        MiniMax-M3 shape)."""
        return {
            "num_kv_heads": self._num_kv_heads,
            "hidden_size": self._hidden_size,
            "head_dim": self._head_dim,
            "v_head_dim": self._v_head_dim,
            "index_n_heads": self._index_n_heads,
            "index_head_dim": self._index_head_dim,
            "index_topk": self._index_topk,
            "block_size": self._block_size,
        }

    @staticmethod
    def _scaled_result(result: PerformanceResult, scale_factor: float) -> PerformanceResult:
        return PerformanceResult(
            float(result) * scale_factor,
            energy=getattr(result, "energy", 0.0) * scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class ContextMSAModule(_BaseMSAModule):
    """Context (prefill) MSA. SILICON reads the msa_context_module table;
    HYBRID/EMPIRICAL fall back to the DSA cross-op transfer on a data miss."""

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership (mirrors ContextDSAModule.load_data)
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the msa_context_module parquet and binds
        ``database._context_msa_module_data``. No pre-expansion: queries
        resolve on the raw grid via perf_interp."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # Source resolution scans sibling/cross-backend directories; keep it
            # inside the cache-miss branch (as the MLA/DSv4 loaders do) so
            # per-query load_data calls stay O(1) after the first load.
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.msa_context_module.value
            )
            sources = database._build_op_sources(PerfDataFilename.msa_context_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_context_msa_module_data(sources), PerfDataFilename.msa_context_module, primary_path
            )
            cls._record_load()
        if "_context_msa_module_data" not in database.__dict__:
            database._context_msa_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Table query (delegated to by PerfDatabase.query_context_msa_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_context_msa_module_table(
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
        num_kv_heads: int | None = None,
        hidden_size: int | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        block_size: int | None = None,
    ):
        """Query the context MSA module table (DSA-shaped, minus the
        skip_indexer / dsa_backend / CP complexity: the MSA collector emits a
        single full op per row). SILICON, HYBRID and EMPIRICAL all resolve the
        raw ``[num_heads][prefix][s][b]`` grid via perf_interp with the
        analytic MSA SOL as anchor; a data miss raises the typed
        ``PerfDataNotAvailableError`` (the op layer adds the DSA xop
        fallback for HYBRID/EMPIRICAL)."""
        # Strict eager flops resolution (parity with the Rust engine): reject a
        # missing *_tc_flops entry up front — an exact silicon hit never
        # invokes the get_sol closure.
        common.get_quant_tc_flops(database.system_spec, gemm_quant_mode)
        common.get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
        common.get_quant_tc_flops(database.system_spec, fmha_quant_mode)
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        if architecture is None:
            architecture = DEFAULT_MSA_ARCHITECTURE
        dims = _msa_sol_dims(
            architecture,
            num_heads,
            {
                "num_kv_heads": num_kv_heads,
                "hidden_size": hidden_size,
                "head_dim": head_dim,
                "v_head_dim": v_head_dim,
                "index_n_heads": index_n_heads,
                "index_head_dim": index_head_dim,
                "index_topk": index_topk,
                "block_size": block_size,
            },
        )

        def get_sol(b: int, s: int, prefix: int, num_heads: int) -> float:
            return _msa_attention_sol(
                database,
                is_context=True,
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                **dims,
            )[0]

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(b, s, prefix, num_heads), energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return _msa_attention_sol(
                database,
                is_context=True,
                b=b,
                s=s,
                prefix=prefix,
                num_heads=num_heads,
                kvcache_quant_mode=kvcache_quant_mode,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                **dims,
            )

        cls.load_data(database)

        def missing_context_msa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Context MSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"fmha_quant_mode={fmha_quant_mode}, kvcache_quant_mode={kvcache_quant_mode}, "
                f"gemm_quant_mode={gemm_quant_mode}, num_heads={num_heads}, s={s}, prefix={prefix}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        msa_module_data = getattr(database, "_context_msa_module_data", None)
        if msa_module_data is None:
            raise missing_context_msa_error()
        try:
            msa_dict = util_empirical.require_data_slice(
                msa_module_data,
                fmha_quant_mode,
                kvcache_quant_mode,
                gemm_quant_mode,
                architecture,
            )
        except PerfDataNotAvailableError as exc:
            raise missing_context_msa_error() from exc
        # The DSA-schema loader keys ...[architecture][backend][num_heads]...;
        # MSA rows carry kernel_source="default", so descend past the backend
        # axis with the same selector the loader's bucketing pairs with.
        msa_dict = _select_dsa_backend(msa_dict, "trtllm")
        try:
            config = perf_interp.OpInterpConfig(
                axes=("num_heads", "prefix", "seq_len", "batch"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda n_v, p_v, s_v, b_v: get_sol(b_v, s_v, p_v, n_v),
            )
            result = perf_interp.query(config, msa_dict, num_heads, prefix, s, b)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
        except InterpolationDataNotAvailableError as exc:
            raise missing_context_msa_error() from exc
        return database._interp_pr(latency, energy=energy)

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database, **kwargs):
        b = kwargs.get("batch_size")
        s = kwargs.get("s")
        prefix = kwargs.get("prefix", 0)
        mode = database._default_database_mode
        if mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL):
            sol = self._sol(database, b, s, prefix, is_context=True)
            return PerformanceResult(sol * self._scale_factor, energy=0.0, source="sol")

        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        # Own silicon table first (SILICON, and the preferred HYBRID/EMPIRICAL
        # source); only a typed data miss falls through to the DSA transfer.
        try:
            result = database.query_context_msa_module(
                b=b,
                s=s,
                prefix=prefix,
                num_heads=self._num_heads,
                kvcache_quant_mode=self._kvcache_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
                **self._own_table_kwargs(),
            )
            return self._scaled_result(result, self._scale_factor)
        except (PerfDataNotAvailableError, InterpolationDataNotAvailableError):
            if mode == common.DatabaseMode.SILICON:
                raise
            logger.debug(f"MSA context data unavailable for b={b}, s={s}, prefix={prefix}; trying DSA xop transfer")

        # EMPIRICAL / HYBRID fallback: cross-op (XOP) transfer from DSA util * k.
        # When XOP is disabled by the transfer policy there is nothing left to
        # fall back on -> raise honestly.
        sol = self._sol(database, b, s, prefix, is_context=True)
        if common.TransferKind.XOP not in database.transfer_policy:
            raise EmpiricalNotImplementedError(
                "MSA context: cross-op transfer (xop) is disabled by the transfer policy "
                "and no MSA silicon data is available for this workload."
            )
        util = _dsa_context_util(
            database,
            b=b,
            s=s,
            prefix=prefix,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            fmha_quant_mode=self._fmha_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._dsa_architecture,
        )
        if not (util and util > 0):
            raise EmpiricalNotImplementedError(
                f"MSA context: no DSA util to transfer from (arch={self._dsa_architecture}, "
                f"b={b}, s={s}); collect MSA/DSA data or set msa_dsa_scale_k against an available quant."
            )
        note_provenance("xop")  # cross-op transfer from DSA
        lat = sol / (util * self._dsa_scale_k)
        return PerformanceResult(lat * self._scale_factor, energy=0.0, source="empirical")


class GenerationMSAModule(_BaseMSAModule):
    """Generation (decode) MSA. s = total kv length. SILICON reads the
    msa_generation_module table; HYBRID/EMPIRICAL fall back to the DSA
    cross-op transfer on a data miss."""

    _data_cache: ClassVar[dict] = {}

    # ------------------------------------------------------------------
    # Data ownership (mirrors GenerationDSAModule.load_data)
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads the msa_generation_module parquet and binds
        ``database._generation_msa_module_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            # See ContextMSAModule.load_data: resolution stays inside the
            # cache-miss branch so per-query calls do not rescan directories.
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.msa_generation_module.value
            )
            sources = database._build_op_sources(PerfDataFilename.msa_generation_module, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(
                load_generation_msa_module_data(sources), PerfDataFilename.msa_generation_module, primary_path
            )
            cls._record_load()
        if "_generation_msa_module_data" not in database.__dict__:
            database._generation_msa_module_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Table query (delegated to by PerfDatabase.query_generation_msa_module)
    # ------------------------------------------------------------------

    @classmethod
    def _query_generation_msa_module_table(
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
        fmha_quant_mode: common.FMHAQuantMode = common.FMHAQuantMode.bfloat16,
        num_kv_heads: int | None = None,
        hidden_size: int | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        index_n_heads: int | None = None,
        index_head_dim: int | None = None,
        index_topk: int | None = None,
        block_size: int | None = None,
    ):
        """Query the generation MSA module table on the raw ``[num_heads][b][s]``
        grid (``s`` = total decode length, ``isl + step`` at load time exactly
        like DSA). ``fmha_quant_mode`` only feeds the SOL anchor — the
        generation table has no fmha axis."""
        common.get_quant_tc_flops(database.system_spec, gemm_quant_mode)
        common.get_quant_tc_flops(database.system_spec, common.FMHAQuantMode.fp8)
        common.get_quant_tc_flops(database.system_spec, fmha_quant_mode)
        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        if architecture is None:
            architecture = DEFAULT_MSA_ARCHITECTURE
        dims = _msa_sol_dims(
            architecture,
            num_heads,
            {
                "num_kv_heads": num_kv_heads,
                "hidden_size": hidden_size,
                "head_dim": head_dim,
                "v_head_dim": v_head_dim,
                "index_n_heads": index_n_heads,
                "index_head_dim": index_head_dim,
                "index_topk": index_topk,
                "block_size": block_size,
            },
        )

        def get_sol(b: int, s: int, num_heads: int) -> float:
            return _msa_attention_sol(
                database,
                is_context=False,
                b=b,
                s=s,
                prefix=0,
                num_heads=num_heads,
                kvcache_quant_mode=kv_cache_dtype,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                **dims,
            )[0]

        if database_mode is None:
            database_mode = database._default_database_mode
        if database_mode == common.DatabaseMode.SOL:
            return PerformanceResult(get_sol(b, s, num_heads), energy=0.0, source="sol")
        elif database_mode == common.DatabaseMode.SOL_FULL:
            return _msa_attention_sol(
                database,
                is_context=False,
                b=b,
                s=s,
                prefix=0,
                num_heads=num_heads,
                kvcache_quant_mode=kv_cache_dtype,
                fmha_quant_mode=fmha_quant_mode,
                gemm_quant_mode=gemm_quant_mode,
                **dims,
            )

        cls.load_data(database)

        def missing_generation_msa_error() -> PerfDataNotAvailableError:
            return PerfDataNotAvailableError(
                f"Generation MSA module data not available for system='{database.system}', "
                f"backend='{database.backend}', version='{database.version}', architecture='{architecture}', "
                f"kv_cache_dtype={kv_cache_dtype}, gemm_quant_mode={gemm_quant_mode}, "
                f"num_heads={num_heads}, s={s}, b={b}. "
                "Missing silicon data for the requested lookup."
            )

        msa_module_data = getattr(database, "_generation_msa_module_data", None)
        if msa_module_data is None:
            raise missing_generation_msa_error()
        try:
            msa_dict = util_empirical.require_data_slice(
                msa_module_data,
                kv_cache_dtype,
                gemm_quant_mode,
                architecture,
            )
        except PerfDataNotAvailableError as exc:
            raise missing_generation_msa_error() from exc
        msa_dict = _select_dsa_backend(msa_dict, "trtllm")
        try:
            config = perf_interp.generation_grid_config(sol_fn=lambda n_v, b_v, s_v: get_sol(b_v, s_v, n_v))
            result = perf_interp.query(config, msa_dict, num_heads, b, s)
            latency = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
        except InterpolationDataNotAvailableError as exc:
            raise missing_generation_msa_error() from exc
        return database._interp_pr(latency, energy=energy)

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database, **kwargs):
        b = kwargs.get("batch_size")
        s = kwargs.get("s")
        mode = database._default_database_mode
        if mode in (common.DatabaseMode.SOL, common.DatabaseMode.SOL_FULL):
            sol = self._sol(database, b, s, 0, is_context=False)
            return PerformanceResult(sol * self._scale_factor, energy=0.0, source="sol")

        from aiconfigurator_core.sdk.perf_database import PerfDataNotAvailableError

        try:
            result = database.query_generation_msa_module(
                b=b,
                s=s,
                num_heads=self._num_heads,
                kv_cache_dtype=self._kvcache_quant_mode,
                gemm_quant_mode=self._gemm_quant_mode,
                fmha_quant_mode=self._fmha_quant_mode,
                **self._own_table_kwargs(),
            )
            return self._scaled_result(result, self._scale_factor)
        except (PerfDataNotAvailableError, InterpolationDataNotAvailableError):
            if mode == common.DatabaseMode.SILICON:
                raise
            logger.debug(f"MSA generation data unavailable for b={b}, s={s}; trying DSA xop transfer")

        sol = self._sol(database, b, s, 0, is_context=False)
        if common.TransferKind.XOP not in database.transfer_policy:
            raise EmpiricalNotImplementedError(
                "MSA generation: cross-op transfer (xop) is disabled by the transfer policy "
                "and no MSA silicon data is available for this workload."
            )
        util = _dsa_generation_util(
            database,
            b=b,
            s=s,
            num_heads=self._num_heads,
            kvcache_quant_mode=self._kvcache_quant_mode,
            gemm_quant_mode=self._gemm_quant_mode,
            architecture=self._dsa_architecture,
        )
        if not (util and util > 0):
            raise EmpiricalNotImplementedError(
                f"MSA generation: no DSA util to transfer from (arch={self._dsa_architecture}, "
                f"b={b}, s={s}); collect MSA/DSA data or set msa_dsa_scale_k against an available quant."
            )
        note_provenance("xop")  # cross-op transfer from DSA
        lat = sol / (util * self._dsa_scale_k)
        return PerformanceResult(lat * self._scale_factor, energy=0.0, source="empirical")


# ─────────────────────────────────────────────────────────
# Loaders — the MSA collector emits the exact DSA-module row schema
# (op_name msa_context_module / msa_generation_module, architecture always
# present, kernel_source "default"), so both loaders delegate to the DSA
# parsers. MSA op_names never carry a skip_indexer tag, so op_kind="full"
# keeps every row; the backend axis derived from kernel_source is resolved
# at query time with the same ``_select_dsa_backend`` chain.
# ─────────────────────────────────────────────────────────


def load_context_msa_module_data(msa_file):
    """Load context MSA module data:
    ``data[fmha][kv_cache][gemm][architecture][backend][num_heads][prefix][s][b]``."""
    return load_context_dsa_module_data(msa_file, op_kind="full")


def load_generation_msa_module_data(msa_file):
    """Load generation MSA module data:
    ``data[kv_cache][gemm][architecture][backend][num_heads][b][s]`` with
    ``s = isl + step`` (total decode length)."""
    return load_generation_dsa_module_data(msa_file, op_kind="full")
