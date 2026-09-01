# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 + GDN kernels (ISSUE-09 / AIC-539).

- ``Mamba2Kernel`` represents a single Mamba2 kernel (conv1d or SSM) and
  owns ``_data_cache`` for ``mamba2_perf.parquet``.
- ``GDNKernel`` represents a single Gated DeltaNet kernel for Qwen3.5
  linear-attention layers and owns ``_data_cache`` for ``gdn_perf.parquet``.
- ``KDAKernel`` extends GDN with the Kimi-K3 verify phase (draft_tokens).

(The deprecated ``Mamba2`` composite — a Python leg COMPOSITION over
engine-evaluated twin ops — was removed after its one-release window,
together with the per-call query shims.)

Neither table has SOL clamping or grid extrapolation in the legacy
``_correct_data`` / ``__init__`` path — the data is keyed by structural
config tuples (``(d_model, d_state, ...)``) rather than dense
``(num_heads, s, b)`` grids, so extrapolation wouldn't apply.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import aiconfigurator_core._aiconfigurator_core as _core
from aiconfigurator_core.sdk.operations.base import OpShellKit

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


class Mamba2Kernel(_core.Mamba2Kernel, OpShellKit):
    """
    Single Mamba2 kernel op (Conv1D or SSM) using collected mamba2_perf data.

    One of four kernels: causal_conv1d_fn, mamba_chunk_scan_combined (context),
    causal_conv1d_update, selective_state_update (generation).
    Uses full (unsharded) dimensions for lookup; collector data is per-layer.

    Owns ``_data_cache`` for the packaged mamba2_perf Parquet perf table.
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
        """Idempotent. Fetches the engine's mamba2_perf table view and binds
        ``database._mamba2_data``. No extrapolation (data is keyed by
        structural config tuples, not a dense grid)."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_mamba2_data", PerfDataFilename.mamba2)
            cls._record_load()

        if "_mamba2_data" not in database.__dict__:
            database._mamba2_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_mamba2)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------


class GDNKernel(_core.GDNKernel, OpShellKit):
    """
    Single Gated DeltaNet (GDN) kernel op for Qwen3.5 linear_attention layers.

    Covers four kernel sources:
      Context phase:
        - "causal_conv1d_fn": Causal 1D convolution over full sequence
        - "chunk_gated_delta_rule": GDN chunked scan (core recurrence)
      Generation phase:
        - "causal_conv1d_update": Single-step causal conv state update
        - "fused_sigmoid_gating_delta_rule_update": Single-step GDN recurrence

    Uses the runtime kernel dimensions supplied by the model builder for database
    lookup. Tensor-parallel model builders therefore pass per-rank head counts.

    Owns ``_data_cache`` for the packaged gdn_perf Parquet perf table.
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
        """Idempotent. Fetches the engine's gdn_perf table view and binds
        ``database._gdn_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_gdn_data", PerfDataFilename.gdn)
            cls._record_load()

        if "_gdn_data" not in database.__dict__:
            database._gdn_data = cls._data_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        cls._data_cache.clear()

    # ------------------------------------------------------------------
    # Query table (formerly PerfDatabase.query_gdn)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Op contract
    # ------------------------------------------------------------------


class KDAKernel(_core.KDAKernel, OpShellKit):
    """
    Single KDA (Kimi Delta Attention) kernel op for Kimi-K3 linear_attention layers.

    Same structural key as GDN — (d_model, num_k_heads, head_k_dim, num_v_heads,
    head_v_dim, d_conv) — but a distinct kernel family (per-K full-rank gate,
    fp32 recurrent state) collected into kda_perf. Covers:
      Context phase:
        - "causal_conv1d_fn_qkv3": the 3-call Q/K/V causal-conv sequence
        - "chunk_kda": chunked delta-rule scan (raw per-K gates)
      Generation phase:
        - "causal_conv1d_update": packed single-step conv update (3P channels)
        - "fused_recurrent_kda_packed_decode": packed KDA recurrence (T=1)
      Verify phase (speculative target-verify; 2-axis batch x draft_tokens):
        - "causal_conv1d_update": conv update over batch*draft_tokens rows
        - "fused_sigmoid_gating_delta_rule_update": fused chain-verify recurrence

    Dims are passed per attention-TP shard (num heads / tp), matching the
    collector's per-shard kda rows. ``draft_tokens`` fixes the verify width
    (speculative block size + 1) for verify-phase ops.

    Owns ``_data_cache`` for the packaged kda_perf Parquet perf table.
    """

    _data_cache: ClassVar[dict] = {}

    @classmethod
    def _cache_key(cls, database: PerfDatabase) -> tuple:
        # Formerly inherited from the Python GDNKernel base; the Rust-backed
        # shells are siblings, so the key lives here explicitly.
        return _cache_key(database)

    @classmethod
    def load_data(cls, database: PerfDatabase) -> None:
        """Idempotent. Fetches the engine's kda_perf table view and binds
        ``database._kda_data``."""
        from aiconfigurator_core.sdk.engine_table_view import load_view
        from aiconfigurator_core.sdk.perf_database import PerfDataFilename

        key = cls._cache_key(database)
        if key not in cls._data_cache:
            cls._data_cache[key] = load_view(database, "_kda_data", PerfDataFilename.kda)
            cls._record_load()

        if "_kda_data" not in database.__dict__:
            database._kda_data = cls._data_cache[key]
