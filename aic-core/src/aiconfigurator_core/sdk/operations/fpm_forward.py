# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model forward-pass op backed by collected ``fpm_forward_perf`` data.

With ``ModelConfig.forward_model == "fpm"`` the model builder replaces each
phase op list with exactly one :class:`FPMForwardOp`. The op is
CONSTRUCTION-ONLY on the Python side: it carries the cell-match identity, the
whole-model weight bytes, and the original granular op list, and
``sdk/engine.py::_to_opspec`` serializes those into the compiled engine's
``Op::FpmForward`` — the Rust core owns the loader
(``perf_database/fpm_forward.rs``), the interpolation/clamp semantics
(``operators/fpm_forward.rs``), and the whole-model SOL roofline derived from
the granular list (``operators/fpm_sol.rs``). The formal database pair it
reads:

    systems/data/<system>/<backend>/<version>/fpm_forward_perf.parquet
    systems/data/<system>/<backend>/<version>/fpm_forward_perf.metadata.json

(The former Python-side query/loader machinery — the per-call ``query()``
family, the parquet/sidecar validators, and the per-op ``DatabaseMode.SOL``
roofline closure — was retired with the Python engine-step path; the Rust
implementations are the single owners now.)
"""

from __future__ import annotations

from enum import Enum

from aiconfigurator_core.sdk.operations.base import Operation

_PHASES = ("prefill", "decode")


# Identity columns that select a cell, in row-column order. ``model_path`` is
# handled separately (exact-match, never borrowed); ``weight_quantization``
# is redundant with ``gemm_quant_mode`` (the collector falls one back to the
# other) so only ``gemm_quant_mode`` participates in matching. The Rust
# loader's cell keying mirrors this order and arity (15).
_CELL_MATCH_COLUMNS = (
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "kv_cache_dtype",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
    # Explicit backend identity (schema v6): "auto" = the engine decided;
    # pinned values were plumbed to the engine and verified by the collector.
    # The two enable_* columns are real parquet booleans; _norm_identity's
    # str() lowers them to "True"/"False", matching the request side's
    # Python bools.
    "moe_backend",
    "attention_backend",
    "enable_wideep",
    "enable_eplb",
)


def _norm_backend_request(value, *, engine_default: str | None = None) -> str:
    """Request-side normalization for the string backend identity columns.

    The collector records "auto" when the knob was left to the engine.
    ``engine_default`` folds AIC's spelled-out default (ModelConfig ships
    attention_backend="flashinfer" rather than None) back to "auto" so the
    default config reaches the auto-collected cells.
    """
    if value is None or value == "" or value == engine_default:
        return "auto"
    return str(value)


def _norm_identity(value) -> str:
    """Normalize an identity field for matching: None -> "", Enum -> name."""
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.name)
    return str(value)


class FPMForwardOp(Operation):
    """One whole-model forward pass for a single phase (prefill or decode)."""

    def __init__(
        self,
        phase: str,
        model_config,
        model_path: str,
        weight_bytes: float = 0.0,
        *,
        sol_ops: list,
    ) -> None:
        """``sol_ops`` — the model's ORIGINAL op-level list for this phase —
        rides the compiled spec so the Rust FPM SOL roofline derives from the
        op-level model itself: per-op analytic max(compute, mem) with the
        real physics (MoE activation, DSA index_topk saturation, per-op
        quant)."""
        if phase not in _PHASES:
            raise ValueError(f"unknown FPM phase: {phase!r}")
        super().__init__(f"fpm_forward_{phase}", 1.0)
        self._phase = phase
        self._model_path = str(model_path)
        self._weight_bytes = float(weight_bytes)
        self._match_identity = (
            _norm_identity(model_config.gemm_quant_mode),
            _norm_identity(model_config.moe_quant_mode),
            _norm_identity(model_config.fmha_quant_mode),
            _norm_identity(model_config.comm_quant_mode),
            _norm_identity(model_config.kvcache_quant_mode),
            _norm_identity(model_config.tp_size),
            _norm_identity(model_config.pp_size),
            _norm_identity(model_config.attention_dp_size),
            _norm_identity(model_config.moe_tp_size if model_config.moe_tp_size is not None else 1),
            _norm_identity(model_config.moe_ep_size if model_config.moe_ep_size is not None else 1),
            _norm_identity(model_config.cp_size),
            _norm_backend_request(getattr(model_config, "moe_backend", None)),
            # ModelConfig spells the engine default out ("flashinfer"); the
            # collector records engine-decided knobs as "auto".
            _norm_backend_request(getattr(model_config, "attention_backend", None), engine_default="flashinfer"),
            _norm_identity(bool(getattr(model_config, "enable_wideep", False))),
            _norm_identity(bool(getattr(model_config, "enable_eplb", False))),
        )
        self._sol_ops = list(sol_ops)

    def get_weights(self, **kwargs) -> float:
        """Per-rank weight bytes of the whole model (captured from the original
        op-level lists before the rewrite), so memory estimation that sums
        ``op.get_weights()`` over the phase list keeps working."""
        return self._weight_bytes
