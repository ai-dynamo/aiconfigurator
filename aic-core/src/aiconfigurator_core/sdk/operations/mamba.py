# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 + GDN kernels (ISSUE-09 / AIC-539).

- ``Mamba2Kernel`` represents a single Mamba2 kernel (conv1d or SSM) and
  owns ``_data_cache`` for ``mamba2_perf.txt``.
  ``PerfDatabase.query_mamba2`` delegates here.
- ``GDNKernel`` represents a single Gated DeltaNet kernel for Qwen3.5
  linear-attention layers and owns ``_data_cache`` for ``gdn_perf.txt``.
  ``PerfDatabase.query_gdn`` delegates here.
- ``Mamba2`` is the higher-level composite op for NemotronH-style hybrid
  models — calls ``database.query_gemm`` (for in_proj + out_proj) and
  ``database.query_mem_op`` (for conv1d + SSM + norm). No CSV data of
  its own.

``Mamba2.query`` deliberately keeps its three ``database.query_mem_op``
callers — ``query_mem_op`` is an explicit non-goal of this refactor
(see project handoff Decision #7); the cleanup PR decides whether/where
to extract the analytical mem-op formula.

Neither table has SOL clamping or grid extrapolation in the legacy
``_correct_data`` / ``__init__`` path — the data is keyed by structural
config tuples (``(d_model, d_state, ...)``) rather than dense
``(num_heads, s, b)`` grids, so extrapolation wouldn't apply.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from aiconfigurator_core.sdk import common, perf_interp
from aiconfigurator_core.sdk.errors import InterpolationDataNotAvailableError
from aiconfigurator_core.sdk.operations.base import Operation, _read_filtered_rows, resolve_op_data_path
from aiconfigurator_core.sdk.performance_result import PerformanceResult

if TYPE_CHECKING:
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

logger = logging.getLogger(__name__)


def _cache_key(database: PerfDatabase) -> tuple:
    """Shared cache key — same shape as every other migrated op.

    TODO: hoist to ``operations/base.py`` once Phase 3 settles (6 op
    families duplicating this helper now).
    """
    return (
        database.systems_root,
        database.system,
        database.backend,
        database.version,
        database.enable_shared_layer,
    )


class Mamba2Kernel(Operation):
    """
    Single Mamba2 kernel op (Conv1D or SSM) using collected mamba2_perf data.

    One of four kernels: causal_conv1d_fn, mamba_chunk_scan_combined (context),
    causal_conv1d_update, selective_state_update (generation).
    Uses full (unsharded) dimensions for lookup; collector data is per-layer.

    Owns ``_data_cache`` for the mamba2_perf CSV table.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        kernel_source: str,
        phase: str,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._kernel_source = kernel_source
        self._phase = phase
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads mamba2_perf CSV, binds
        ``database._mamba2_data``. No extrapolation (data is keyed by
        structural config tuples, not a dense grid)."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.mamba2.value
            )
            sources = database._build_op_sources(PerfDataFilename.mamba2, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_mamba2_data(sources), PerfDataFilename.mamba2, primary_path)
            cls._record_load()

        if "_mamba2_data" not in database.__dict__:
            database._mamba2_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mamba2)
    # ------------------------------------------------------------------

    @classmethod
    def _query_mamba2_table(
        cls,
        database: PerfDatabase,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        d_state: int,
        d_conv: int,
        nheads: int,
        head_dim: int,
        n_groups: int,
        chunk_size: int,
    ) -> PerformanceResult:
        """Query Mamba2 kernel table. Verbatim port of the legacy body.

        Uses SOL-based fallback when mamba2_perf data is not loaded or
        when the requested ``(kernel_source, phase, model_key)`` is absent.
        """
        cls.load_data(database)
        mamba2_data = database._mamba2_data
        # ``mamba2_data`` is a ``LoadedOpData`` wrapper; treat empty/unloaded
        # tables as the all-SOL fallback path. Iterating an unloaded
        # ``LoadedOpData`` raises, so explicitly check ``.loaded``.
        if not getattr(mamba2_data, "loaded", False):
            mamba2_data = {}

        def get_sol(b: int = batch_size, s: int | None = seq_len) -> tuple[float, float, float]:
            d_inner = nheads * head_dim
            conv_dim = d_inner + 2 * n_groups * d_state
            x = (b * s) if phase == "context" and s else b
            if kernel_source in ("causal_conv1d_fn", "causal_conv1d_update"):
                conv_read_bytes = x * conv_dim * (d_conv + 1) * 2
                conv_write_bytes = x * conv_dim * 2
                total_bytes = conv_read_bytes + conv_write_bytes
            else:
                ssm_read_bytes = x * (d_inner + n_groups * d_state * 2 + nheads) * 2
                ssm_write_bytes = x * d_inner * 2
                total_bytes = ssm_read_bytes + ssm_write_bytes
            sol_mem = total_bytes / database.system_spec["gpu"]["mem_bw"] * 1000
            return sol_mem, 0, sol_mem

        if not mamba2_data:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")

        model_key = (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size)
        try:
            by_phase = mamba2_data[kernel_source]
        except KeyError:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        try:
            by_key = by_phase[phase]
        except KeyError:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
        if model_key not in by_key:
            # Nearest config by d_model
            keys_with_d_model = [k for k in by_key if k[0] == d_model]
            if keys_with_d_model:
                model_key = keys_with_d_model[0]
            else:
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")

        table = by_key[model_key]

        if phase == "context":
            if seq_len is None or seq_len <= 0:
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            # 2-axis RAW grid (memory-bound ~linear per axis); beyond-range is
            # util-hold via the kernel SOL. DEGRADATION CONTRACT preserved: a
            # genuine data miss falls back to plain SOL (returns, not raises).
            config = perf_interp.OpInterpConfig(
                axes=("batch", "seq_len"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda b_v, s_v: get_sol(b_v, s_v)[0],
            )
            try:
                result = perf_interp.query(config, table, batch_size, seq_len)
            except (InterpolationDataNotAvailableError, KeyError, ValueError):
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            return database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
        else:
            # Normalize to a flat {batch: entry} curve first (legacy tables nest
            # batch -> seq -> entry; generation rows have a single seq), then a
            # 1-axis RAW engine query with SOL-fallback degradation.
            def _mamba2_gen_entry(val):
                if isinstance(val, dict) and "latency" in val:
                    return val
                if isinstance(val, dict) and val:
                    inner = next(iter(val.values()))
                    if isinstance(inner, dict) and "latency" in inner:
                        return inner
                return None

            curve = {bb: entry for bb, entry in ((bb, _mamba2_gen_entry(v)) for bb, v in table.items()) if entry}
            config = perf_interp.OpInterpConfig(
                axes=("batch",),
                resolver=perf_interp.Grid(),
                sol_fn=lambda b_v: get_sol(b_v, seq_len)[0],
            )
            try:
                result = perf_interp.query(config, curve, batch_size)
            except (InterpolationDataNotAvailableError, KeyError, ValueError):
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat, energy=energy)

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        seq_len = s if self._phase == "context" else None
        result = database.query_mamba2(
            phase=self._phase,
            kernel_source=self._kernel_source,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self._hidden_size,
            d_state=self._d_state,
            d_conv=self._d_conv,
            nheads=self._nheads,
            head_dim=self._head_dim,
            n_groups=self._n_groups,
            chunk_size=self._chunk_size,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class GDNKernel(Operation):
    """
    Single Gated DeltaNet (GDN) kernel op for Qwen3.5 linear_attention layers.

    Covers four kernel sources:
      Context phase:
        - "causal_conv1d_fn": Causal 1D convolution over full sequence
        - "chunk_gated_delta_rule": GDN chunked scan (core recurrence)
      Generation phase:
        - "causal_conv1d_update": Single-step causal conv state update
        - "fused_sigmoid_gating_delta_rule_update": Single-step GDN recurrence

    Uses full (unsharded) dimensions for database lookup; collector data is per-layer.

    Owns ``_data_cache`` for the gdn_perf CSV table.
    """

    _data_cache: ClassVar[dict] = {}

    def __init__(
        self,
        name: str,
        scale_factor: float,
        kernel_source: str,
        phase: str,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._kernel_source = kernel_source
        self._phase = phase
        self._d_model = d_model
        self._num_k_heads = num_k_heads
        self._head_k_dim = head_k_dim
        self._num_v_heads = num_v_heads
        self._head_v_dim = head_v_dim
        self._d_conv = d_conv
        self._weights = 0.0

    # ------------------------------------------------------------------
    # Data ownership
    # ------------------------------------------------------------------

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Loads gdn_perf CSV, binds ``database._gdn_data``."""
        import os

        from aiconfigurator_core.sdk.perf_database import LoadedOpData, PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            system_data_root = os.path.join(database.systems_root, database.system_spec["data_dir"])
            primary_path = resolve_op_data_path(
                system_data_root, database.backend, database.version, PerfDataFilename.gdn.value
            )
            sources = database._build_op_sources(PerfDataFilename.gdn, primary_path, system_data_root)
            cls._data_cache[key] = LoadedOpData(load_gdn_data(sources), PerfDataFilename.gdn, primary_path)
            cls._record_load()

        if "_gdn_data" not in database.__dict__:
            database._gdn_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_gdn)
    # ------------------------------------------------------------------

    @classmethod
    def _query_gdn_table(
        cls,
        database: PerfDatabase,
        phase: str,
        kernel_source: str,
        batch_size: int,
        seq_len: int | None,
        d_model: int,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        d_conv: int,
    ) -> PerformanceResult:
        """Query GDN kernel table. Verbatim port of the legacy body."""
        cls.load_data(database)
        gdn_data = database._gdn_data
        if not getattr(gdn_data, "loaded", False):
            gdn_data = {}

        def get_sol(b: int = batch_size, s: int | None = seq_len) -> tuple[float, float, float]:
            x = (b * s) if phase == "context" and s else b
            if kernel_source in ("causal_conv1d_fn", "causal_conv1d_update"):
                conv_channels = num_k_heads * head_k_dim + num_v_heads * head_v_dim
                read_bytes = x * conv_channels * (d_conv + 1) * 2
                write_bytes = x * conv_channels * 2
            elif kernel_source == "chunk_gated_delta_rule":
                # GDN chunked scan (context phase).
                # State shape: [num_v_heads, head_k_dim, head_v_dim], stored as BF16 in global memory.
                # Intermediate h_chunks [B, NT, H, K, V] are written by chunk_delta_h and read by
                # chunk_o via global memory (separate kernel launches). Allocated via k.new_empty()
                # (no dtype override), so matches input dtype: FP16/BF16 → 2 bytes.
                chunk_size = 64  # flash-linear-attention default for chunk_gated_delta_rule
                state_size = num_v_heads * head_k_dim * head_v_dim
                num_chunks = (s // chunk_size) if s else 0
                h_chunks_bytes = num_chunks * state_size * 2 * b
                read_bytes = (
                    x * (num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2
                    + state_size * 2 * b
                    + h_chunks_bytes  # chunk_o reads h_chunks from global memory
                )
                write_bytes = (
                    x * num_v_heads * head_v_dim * 2
                    + state_size * 2 * b
                    + h_chunks_bytes  # chunk_delta_h writes h_chunks to global memory
                )
            elif kernel_source == "fused_sigmoid_gating_delta_rule_update":
                # GDN single-step decode. State stored as BF16 in global memory.
                state_size = num_v_heads * head_k_dim * head_v_dim
                read_bytes = x * (num_k_heads * head_k_dim + num_v_heads * head_v_dim) * 2 + state_size * 2 * b
                write_bytes = x * num_v_heads * head_v_dim * 2 + state_size * 2 * b
            else:
                read_bytes = x * d_model * 2
                write_bytes = x * d_model * 2
            sol_mem = (read_bytes + write_bytes) / database.system_spec["gpu"]["mem_bw"] * 1000
            return sol_mem, 0, sol_mem

        if not gdn_data:
            return PerformanceResult(get_sol()[0], energy=0.0, source="sol")

        model_key = (d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv)
        try:
            by_key = gdn_data[kernel_source][phase]
        except KeyError:
            by_key = {}
        if model_key not in by_key:
            alias_sources = ()
            if database.backend == "vllm" and database.version == "0.24.0":
                if phase == "context" and kernel_source == "chunk_gated_delta_rule":
                    alias_sources = (
                        "chunk_gated_delta_rule_flashinfer",
                        "chunk_gated_delta_rule_triton",
                        "chunk_gated_delta_rule_cutedsl",
                    )
                elif phase == "generation" and kernel_source == "fused_sigmoid_gating_delta_rule_update":
                    alias_sources = ("fused_recurrent_gated_delta_rule_packed_decode",)

            exact_aliases = []
            for alias_source in alias_sources:
                try:
                    alias_by_key = gdn_data[alias_source][phase]
                except KeyError:
                    continue
                if model_key in alias_by_key:
                    exact_aliases.append(alias_by_key)

            if len(exact_aliases) == 1:
                by_key = exact_aliases[0]
            elif exact_aliases:
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            else:
                # Preserve the legacy fallback only within the logical source;
                # physical aliases must match the complete model shape.
                keys_same_d_model = [k for k in by_key if k[0] == d_model]
                if keys_same_d_model:
                    model_key = min(keys_same_d_model, key=lambda k: abs(k[3] - num_v_heads))
                else:
                    return PerformanceResult(get_sol()[0], energy=0.0, source="sol")

        table = by_key[model_key]

        if phase == "context":
            if seq_len is None or seq_len <= 0:
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            # 2-axis RAW grid (memory-bound ~linear per axis); beyond-range is
            # util-hold via the kernel SOL. DEGRADATION CONTRACT preserved: a
            # genuine data miss falls back to plain SOL (returns, not raises).
            config = perf_interp.OpInterpConfig(
                axes=("batch", "seq_len"),
                resolver=perf_interp.Grid(),
                sol_fn=lambda b_v, s_v: get_sol(b_v, s_v)[0],
            )
            try:
                result = perf_interp.query(config, table, batch_size, seq_len)
            except (InterpolationDataNotAvailableError, KeyError, ValueError):
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            return database._interp_pr(
                perf_interp.get_value(result, "latency"),
                energy=perf_interp.get_value(result, "energy"),
            )
        else:
            # See mamba2: normalize legacy nesting to {batch: entry}, then a
            # 1-axis RAW engine query with SOL-fallback degradation.
            def _gdn_gen_entry(val):
                if isinstance(val, dict) and "latency" in val:
                    return val
                if isinstance(val, dict) and val:
                    inner = next(iter(val.values()))
                    if isinstance(inner, dict) and "latency" in inner:
                        return inner
                return None

            curve = {bb: entry for bb, entry in ((bb, _gdn_gen_entry(v)) for bb, v in table.items()) if entry}
            config = perf_interp.OpInterpConfig(
                axes=("batch",),
                resolver=perf_interp.Grid(),
                sol_fn=lambda b_v: get_sol(b_v, seq_len)[0],
            )
            try:
                result = perf_interp.query(config, curve, batch_size)
            except (InterpolationDataNotAvailableError, KeyError, ValueError):
                return PerformanceResult(get_sol()[0], energy=0.0, source="sol")
            lat = perf_interp.get_value(result, "latency")
            energy = perf_interp.get_value(result, "energy")
            return database._interp_pr(lat, energy=energy)

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        batch_size = kwargs.get("batch_size")
        s = kwargs.get("s")
        seq_len = s if self._phase == "context" else None
        result = database.query_gdn(
            phase=self._phase,
            kernel_source=self._kernel_source,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=self._d_model,
            num_k_heads=self._num_k_heads,
            head_k_dim=self._head_k_dim,
            num_v_heads=self._num_v_heads,
            head_v_dim=self._head_v_dim,
            d_conv=self._d_conv,
        )
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
            source=getattr(result, "source", "silicon"),
        )

    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor


class Mamba2(Operation):
    """
    Mamba2 operation for NemotronH hybrid models.

    Composite op — no CSV data of its own. Builds the full Mamba2Mixer
    layer cost from:
    - in_proj GEMM (``database.query_gemm``)
    - conv1d mem_op (``database.query_mem_op`` — deliberately kept on
      ``PerfDatabase``; ``query_mem_op`` is an explicit non-goal of this
      refactor, see handoff Decision #7)
    - SSM mem_op (``database.query_mem_op``)
    - norm mem_op (``database.query_mem_op``)
    - out_proj GEMM (``database.query_gemm``)

    The internal state dimension is calculated as:
    expanded_size = 2 * (nheads * head_dim + 2 * n_groups * d_state)
    """

    def __init__(
        self,
        name: str,
        scale_factor: float,
        hidden_size: int,
        nheads: int,
        head_dim: int,
        d_state: int,
        d_conv: int,
        n_groups: int,
        chunk_size: int,
        tp_size: int,
        quant_mode: common.GEMMQuantMode,
        seq_split: int = 1,
    ) -> None:
        super().__init__(name, scale_factor, seq_split=seq_split)
        self._hidden_size = hidden_size
        self._nheads = nheads
        self._head_dim = head_dim
        self._d_state = d_state
        self._d_conv = d_conv
        self._n_groups = n_groups
        self._chunk_size = chunk_size
        self._tp_size = tp_size
        self._quant_mode = quant_mode

        # Calculate dimensions matching TensorRT-LLM mamba2_mixer.py lines 76-78:
        # d_inner = head_dim * nheads
        # d_in_proj = 2 * d_inner + 2 * n_groups * d_state + nheads
        # conv_dim = d_inner + 2 * n_groups * d_state
        self._d_inner = nheads * head_dim
        self._conv_dim = self._d_inner + 2 * n_groups * d_state
        self._in_proj_out_size = 2 * self._d_inner + 2 * n_groups * d_state + nheads

        # Calculate weights (in_proj + conv1d + out_proj + A + D + dt_bias + norm)
        # in_proj: hidden_size * in_proj_out_size (Linear d_model -> d_in_proj)
        # conv1d: d_conv * conv_dim (Linear d_conv -> conv_dim, stored as Linear for TP)
        # out_proj: d_inner * hidden_size (Linear d_inner -> d_model)
        # A, D, dt_bias: nheads each (small, ignored for weight calculation)
        # norm: d_inner (small, ignored)
        self._weights = (
            (
                hidden_size * self._in_proj_out_size  # in_proj
                + d_conv * self._conv_dim  # conv1d
                + self._d_inner * hidden_size  # out_proj
            )
            * quant_mode.value.memory
            // tp_size
        )

    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query Mamba2 latency using SOL-based approximation.

        Models the operation as:
        1. in_proj GEMM: (x, hidden_size) @ (hidden_size, in_proj_out_size)
        2. conv1d: Memory-bound operation
        3. SSM scan: Memory-bound recurrent operation
        4. out_proj GEMM: (x, d_inner) @ (d_inner, hidden_size)
        """
        x = kwargs.get("x")  # num tokens
        # No ``_seq_split`` division here: the Mamba SSM scan is order-dependent,
        # so CP cannot shard its tokens across ranks (each rank sees the full
        # sequence). The ``_CP_AWARE = False`` default also makes the constructor
        # raise if a caller ever passes ``seq_split > 1`` via __init__.

        # Apply TP sharding (matching TensorRT-LLM mamba2_mixer.py lines 81-84)
        nheads_per_gpu = self._nheads // self._tp_size
        d_inner_per_gpu = nheads_per_gpu * self._head_dim
        n_groups_per_gpu = self._n_groups // self._tp_size
        conv_dim_per_gpu = d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state
        in_proj_out_per_gpu = 2 * d_inner_per_gpu + 2 * n_groups_per_gpu * self._d_state + nheads_per_gpu

        total_latency = 0.0
        total_energy = 0.0

        # 1. in_proj GEMM: hidden_size -> in_proj_out_size
        in_proj_result = database.query_gemm(x, in_proj_out_per_gpu, self._hidden_size, self._quant_mode)
        total_latency += float(in_proj_result)
        total_energy += in_proj_result.energy

        # 2. conv1d: Memory-bound operation on conv_dim (not just d_inner)
        # conv1d operates on xbc which has dimension conv_dim
        # Read: x * conv_dim * d_conv (for conv states) + x * conv_dim (input)
        # Write: x * conv_dim (output)
        conv_read_bytes = x * conv_dim_per_gpu * (self._d_conv + 1) * 2  # bfloat16
        conv_write_bytes = x * conv_dim_per_gpu * 2
        conv_result = database.query_mem_op(conv_read_bytes + conv_write_bytes)
        total_latency += float(conv_result)
        total_energy += conv_result.energy

        # 3. SSM scan: Memory-bound recurrent operation
        # For prefill (context), uses chunked scan
        # For decode (generation), uses selective_state_update
        # Approximate as memory operation:
        # Read: x * (d_inner + n_groups * d_state * 2 + nheads) for x, B, C, dt
        # Write: x * d_inner for output
        ssm_read_bytes = (
            x
            * (
                d_inner_per_gpu
                + n_groups_per_gpu * self._d_state * 2  # B and C
                + nheads_per_gpu  # dt
            )
            * 2
        )
        ssm_write_bytes = x * d_inner_per_gpu * 2
        ssm_result = database.query_mem_op(ssm_read_bytes + ssm_write_bytes)
        total_latency += float(ssm_result)
        total_energy += ssm_result.energy

        # 4. norm: RMSNormGated on d_inner (TRT-LLM mamba2_mixer.py line 315)
        # Read SSM output, apply norm with gating, write normalized output
        norm_read_bytes = x * d_inner_per_gpu * 2  # bfloat16
        norm_write_bytes = x * d_inner_per_gpu * 2  # bfloat16
        norm_result = database.query_mem_op(norm_read_bytes + norm_write_bytes)
        total_latency += float(norm_result)
        total_energy += norm_result.energy

        # 5. out_proj GEMM: d_inner -> hidden_size
        out_proj_result = database.query_gemm(x, self._hidden_size, d_inner_per_gpu, self._quant_mode)
        total_latency += float(out_proj_result)
        total_energy += out_proj_result.energy

        # Merge sources from every sub-result so the composite reflects mixed
        # silicon/empirical provenance instead of defaulting to silicon.
        sub_sources = [
            getattr(r, "source", "silicon")
            for r in (in_proj_result, conv_result, ssm_result, norm_result, out_proj_result)
        ]
        merged_source = sub_sources[0] if all(s == sub_sources[0] for s in sub_sources) else "mixed"

        return PerformanceResult(
            latency=total_latency * self._scale_factor,
            energy=total_energy * self._scale_factor,
            source=merged_source,
        )

    def get_weights(self, **kwargs):  # Mamba2 weights
        return self._weights * self._scale_factor


# ─────────────────────────────────────────────────────────
# CSV loaders (moved here from perf_database.py so each op family owns its data + parser)
# ─────────────────────────────────────────────────────────


def load_mamba2_data(mamba2_file: str):
    """
    Load Mamba2 Conv1D + SSM kernel performance data from mamba2_perf.txt.

    CSV columns: framework, version, device, op_name, kernel_source, phase,
    batch_size, seq_len, num_tokens, d_model, d_state, d_conv, nheads, head_dim,
    n_groups, chunk_size, model_name, latency (optional: power).
    All rows must have the same columns (context and generation both include
    seq_len and num_tokens so columns align).

    Returns:
        dict: data[kernel_source][phase][model_key] where model_key is
              (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size).
              For phase "context" the leaf is [batch_size][seq_len] -> {latency, power, energy}.
              For phase "generation" the leaf is [batch_size] -> {latency, power, energy}.
              Returns None if file does not exist.
    """
    rows = _read_filtered_rows(mamba2_file)
    if rows is None:
        logger.debug(f"Mamba2 data file {mamba2_file} not found.")
        return None

    # data[kernel_source][phase][model_key] -> nested batch_size [seq_len] -> {latency, power, energy}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (mamba2) - power will default to 0.0")

    for row in rows:
        kernel_source = row["kernel_source"]
        phase = row["phase"]
        batch_size = int(row["batch_size"])
        seq_len = int(row["seq_len"])
        d_model = int(row["d_model"])
        d_state = int(row["d_state"])
        d_conv = int(row["d_conv"])
        nheads = int(row["nheads"])
        head_dim = int(row["head_dim"])
        n_groups = int(row["n_groups"])
        chunk_size = int(row["chunk_size"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0))
        energy = power * latency

        model_key = (d_model, d_state, d_conv, nheads, head_dim, n_groups, chunk_size)
        entry = {"latency": latency, "power": power, "energy": energy}

        try:
            if phase == "context":
                data[kernel_source][phase][model_key][batch_size][seq_len]
                logger.debug(
                    f"value conflict in mamba2 data: {kernel_source} {phase} {model_key} {batch_size} {seq_len}"
                )
            else:
                data[kernel_source][phase][model_key][batch_size]
                logger.debug(f"value conflict in mamba2 data: {kernel_source} {phase} {model_key} {batch_size}")
        except KeyError:
            if phase == "context":
                data[kernel_source][phase][model_key][batch_size][seq_len] = entry
            else:
                data[kernel_source][phase][model_key][batch_size] = entry

    # Convert default dicts to regular dicts for predictable behavior; keep generation as 1D
    result = {}
    for ks, by_phase in data.items():
        result[ks] = {}
        for ph, by_key in by_phase.items():
            result[ks][ph] = dict(by_key)

    return result


# The GDN decode-recurrence kernel name drifted across sglang releases
# (0.5.10 recorded fused_recurrent_gated_delta_rule; 0.5.14 records the
# executed fused_recurrent_gated_delta_rule_packed_decode). Consumers (the
# qwen35 GDNKernel op and the Rust port) query one canonical modeling
# identity; normalize the LOOKUP key here so every version's measured decode
# rows are reachable. The parquet keeps the executed-kernel truth.
_GDN_DECODE_RECURRENCE_ALIASES = {
    "fused_recurrent_gated_delta_rule": "fused_sigmoid_gating_delta_rule_update",
    "fused_recurrent_gated_delta_rule_packed_decode": "fused_sigmoid_gating_delta_rule_update",
}


def load_gdn_data(gdn_file: str):
    """
    Load GDN (Gated DeltaNet) kernel performance data from gdn_perf.txt.

    CSV columns: framework, version, device, op_name, kernel_source, phase,
    batch_size, seq_len, num_tokens, d_model, d_conv, num_k_heads, head_k_dim,
    num_v_heads, head_v_dim, model_name, latency (optional: power).
    All rows must have the same columns (context and generation both include
    seq_len and num_tokens so columns align).

    Returns:
        dict: data[kernel_source][phase][model_key] where model_key is
              (d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv).
              For phase "context" the leaf is [batch_size][seq_len] -> {latency, power, energy}.
              For phase "generation" the leaf is [batch_size] -> {latency, power, energy}.
              Returns None if file does not exist.
    """
    rows = _read_filtered_rows(gdn_file)
    if rows is None:
        logger.debug(f"GDN data file {gdn_file} not found.")
        return None

    # data[kernel_source][phase][model_key] -> nested batch_size [seq_len] -> {latency, power, energy}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug("Legacy database format detected (gdn) - power will default to 0.0")

    for row in rows:
        kernel_source = _GDN_DECODE_RECURRENCE_ALIASES.get(row["kernel_source"], row["kernel_source"])
        phase = row["phase"]
        batch_size = int(row["batch_size"])
        seq_len = int(row["seq_len"])
        d_model = int(row["d_model"])
        d_conv = int(row["d_conv"])
        num_k_heads = int(row["num_k_heads"])
        head_k_dim = int(row["head_k_dim"])
        num_v_heads = int(row["num_v_heads"])
        head_v_dim = int(row["head_v_dim"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0))
        energy = power * latency

        model_key = (d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv)
        entry = {"latency": latency, "power": power, "energy": energy}

        by_model = data[kernel_source][phase][model_key]
        if phase == "context":
            if batch_size in by_model and seq_len in by_model[batch_size]:
                logger.debug(f"value conflict in gdn data: {kernel_source} {phase} {model_key} {batch_size} {seq_len}")
            else:
                by_model.setdefault(batch_size, {})[seq_len] = entry
        else:
            if batch_size in by_model:
                logger.debug(f"value conflict in gdn data: {kernel_source} {phase} {model_key} {batch_size}")
            else:
                by_model[batch_size] = entry

    # Convert defaultdicts to regular dicts for predictable behavior
    result = {}
    for ks, by_phase in data.items():
        result[ks] = {}
        for ph, by_key in by_phase.items():
            result[ks][ph] = dict(by_key)

    return result
