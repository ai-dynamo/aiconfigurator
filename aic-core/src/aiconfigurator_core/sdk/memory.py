# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-cache capacity estimation.

The single source of truth for the rank-local KV-cache capacity estimate. The
Rust ``aiconfigurator_core::memory::estimate_kv_cache`` is a pure forwarder that
calls :func:`estimate_kv_cache` here and rebuilds its result, so all of the
math -- fraction + tolerance validation, the native AIC memory model, the naive
fallback, and the tolerance margin -- lives in this module.

Two estimators, each built then asked to ``estimate()``:

- **Native** (:class:`KVCacheEstimator`): ``from_request`` reuses AIC's full
  backend memory model (``BaseBackend._get_memory_usage`` plus the model's
  MLA-aware per-token KV size), then ``estimate`` does the OfFree (TRT-LLM,
  fraction of FREE memory) / OfTotal (vLLM/SGLang, fraction of TOTAL memory)
  budget math.
- **Naive fallback** (:class:`NaiveKVCacheEstimator`): for models/backends AIC
  cannot build natively, ``from_model_path`` parses the HF ``config.json``
  (reusing the SDK's existing loaders +
  :func:`~aiconfigurator_core.sdk.utils._parse_hf_config_json`) and ``estimate``
  reserves a fraction (default 80%) of post-weight memory for KV. It handles
  standard K/V and MLA/compressed-latent attention separately, auto-selects
  hybrid sliding-window vs linear KV growth, and RAISES when the config lacks the
  metadata to compute the weights or the per-token KV size (rather than guessing
  with a placeholder).

The budget math is factored into :func:`kv_cache_budget_bytes` so the deferred
consolidation of ``InferenceSummary._check_and_set_kv_cache_oom`` onto a single
formula is a small follow-up (tracked in #1208).
"""

from __future__ import annotations

import enum
import math
import os
from collections.abc import Callable
from typing import Any

from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.backends.factory import get_backend
from aiconfigurator_core.sdk.common import DefaultHFModels
from aiconfigurator_core.sdk.config_builders import apply_nextn, build_model_config, validate_nextn
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.utils import (
    _download_hf_config,
    _load_local_config,
    _load_pre_downloaded_hf_config,
    _parse_hf_config_json,
)

_ONE_GIB = 1 << 30

# A KV byte-budget -> token-count inverse. Every estimation path produces one of
# these (native: the model's ``get_kvcache_max_tokens``; naive:
# ``NaiveKVCacheEstimator.get_kvcache_max_tokens``; the linear
# ``_linear_tokens_from_bytes`` is the default for constant-per-token growth), so
# the raw and tolerance-adjusted token counts invert the same curve through a
# single, uniform type.
TokensFromBytes = Callable[[float], int]

# Default fraction of post-weight memory reserved for KV in the naive fallback
# (exposed as the ``naive_kv_reservation`` arg so callers can tune it).
_DEFAULT_NAIVE_KV_RESERVATION = 0.80

# Backend -> the memory-fraction kind that backend accepts. TRT-LLM applies the
# fraction to FREE memory (`free_gpu_memory_fraction`); vLLM/SGLang apply it to
# TOTAL memory (`gpu_memory_utilization` / `mem_fraction_static`).
_BACKEND_FRACTION_KIND = {
    "trtllm": "of_free",
    "vllm": "of_total",
    "sglang": "of_total",
}


# --------------------------------------------------------------------------- #
# Validation (runs before any model build, so bad inputs fail cheaply).
# --------------------------------------------------------------------------- #


def _validate_memory_fraction(backend: str, memory_fraction_kind: str, memory_fraction_value: float) -> None:
    """Validate the ``memory_fraction_kind`` against the backend, then the range.

    Runs FIRST in :func:`estimate_kv_cache`, before any model build, so a wrong
    kind or out-of-range value fails deterministically without touching the perf
    database. ``of_free`` is TRT-LLM only; ``of_total`` is vLLM / SGLang.
    """
    expected = _BACKEND_FRACTION_KIND.get(backend)
    if expected is None:
        raise ValueError(f"unknown backend {backend!r} for KV-cache estimation")
    if memory_fraction_kind != expected:
        raise ValueError(
            f"incompatible memory fraction: backend {backend!r} does not accept "
            f"memory_fraction_kind {memory_fraction_kind!r} (expected {expected!r})"
        )
    f = float(memory_fraction_value)
    if not (math.isfinite(f) and 0.0 <= f <= 1.0):
        raise ValueError(f"kv_cache_memory_fraction must be finite and in [0, 1], got {f}")


def _validate_naive_reservation(naive_kv_reservation: float) -> None:
    """Validate the naive-fallback KV reservation fraction.

    Must be finite and in ``[0, 1]``. Runs FIRST in :func:`estimate_kv_cache`
    (before any model build), alongside the other fail-fast validators, so a bad
    value is rejected deterministically even on the native path where the
    fraction is never used.
    """
    r = float(naive_kv_reservation)
    if not (math.isfinite(r) and 0.0 <= r <= 1.0):
        raise ValueError(f"naive_kv_reservation must be finite and in [0, 1], got {naive_kv_reservation}")


def _validate_tolerance(tolerance_fraction: float | None) -> None:
    """Validate the optional KV-cache safety-margin ``tolerance_fraction``.

    Must be finite and in the half-open interval ``[0, 1)`` (``t = 1.0`` would
    reserve the entire budget). Runs FIRST in :func:`estimate_kv_cache`, before
    the expensive breakdown, so a bad value fails cheaply and deterministically.
    """
    if tolerance_fraction is None:
        return
    t = float(tolerance_fraction)
    if not (math.isfinite(t) and 0.0 <= t < 1.0):
        raise ValueError(f"tolerance_fraction must be finite and in [0, 1), got {tolerance_fraction}")


# --------------------------------------------------------------------------- #
# Shared budget math.
#
# The single formula both the estimate and (in a follow-up PR) the sweep's
# `InferenceSummary._check_and_set_kv_cache_oom` should use. The OOM check folds
# `reserved` (block-allocator overhead) and `tolerance` into the fraction
# multiplicatively; the estimate passes `reserved=tolerance=0` here and applies
# its tolerance separately via `_apply_tolerance` (so `tolerance_adjusted` stays
# a distinct field). Unit-agnostic: pass capacity/non_kv in the same unit.
# TODO(#1208): point `_check_and_set_kv_cache_oom` at this helper.
# --------------------------------------------------------------------------- #


def kv_cache_budget_bytes(
    *,
    capacity: float,
    non_kv: float,
    fraction: float,
    of_free: bool,
    reserved: float = 0.0,
    tolerance: float = 0.0,
) -> float:
    """KV-cache budget for one rank.

    ``of_free`` (TRT-LLM) applies the fraction to the FREE pool
    ``(capacity - non_kv)``; otherwise (vLLM/SGLang ``of_total``) it applies the
    fraction to TOTAL memory then subtracts ``non_kv``. ``reserved`` and
    ``tolerance`` shrink the fraction multiplicatively (both default to a no-op).
    """
    scale = (1.0 - reserved) * (1.0 - tolerance)
    if of_free:
        return (capacity - non_kv) * fraction * scale
    return capacity * fraction * scale - non_kv


def _linear_tokens_from_bytes(per_token_bytes: float) -> TokensFromBytes:
    """The linear-growth inverse ``floor(budget / per_token)`` as a callable.

    The default :data:`TokensFromBytes` for models with a constant per-token KV
    size (MHA / GQA / MLA), where token capacity scales linearly with the budget.
    Hybrid / compressed-attention paths supply their own non-linear inverse
    instead.
    """

    def tokens_from_bytes(budget_bytes: float) -> int:
        budget = float(budget_bytes)
        if budget <= 0.0 or per_token_bytes <= 0.0:
            return 0
        return int(budget // per_token_bytes)

    return tokens_from_bytes


def _apply_tolerance(estimate: dict[str, Any], tolerance_fraction: float | None) -> dict[str, Any]:
    """Populate ``tolerance_adjusted`` on a finished raw estimate (in place).

    ``adj_bytes = floor(total_kv_size_bytes * (1 - t))`` (a ``t`` of 0.05 keeps
    95% of raw, reserving a 5% safety margin); ``adj_tokens`` is the token count
    that fits in ``adj_bytes``, derived through the estimate's
    ``tokens_from_kv_bytes`` inverse -- the SAME callable used for the raw count,
    so the raw and adjusted counts follow the same KV curve. When
    ``tolerance_fraction`` is ``None``, ``tolerance_adjusted`` stays ``None`` (raw
    only).

    The emitted dict keys (``tolerance_fraction`` / ``total_kv_size_bytes`` /
    ``total_kv_size_tokens``) match what the Rust ``estimate_from_dict`` parses
    back, so the numbers are identical across the PyO3 boundary.
    """
    if tolerance_fraction is None:
        return estimate
    t = float(tolerance_fraction)
    raw_bytes = int(estimate["total_kv_size_bytes"])
    adj_bytes = math.floor(raw_bytes * (1.0 - t))
    tokens_from_bytes: TokensFromBytes = estimate["tokens_from_kv_bytes"]
    adj_tokens = int(tokens_from_bytes(adj_bytes))
    estimate["tolerance_adjusted"] = {
        "tolerance_fraction": t,
        "total_kv_size_bytes": adj_bytes,
        "total_kv_size_tokens": adj_tokens,
    }
    return estimate


# --------------------------------------------------------------------------- #
# Native path: AIC's full backend memory model.
#
# `KVCacheEstimator` builds the model + backend + perf DB (the part that fails
# for models AIC cannot model -> the caller falls back to NaiveKVCacheEstimator),
# computes the `BaseBackend._get_memory_usage` breakdown + per-token KV byte size
# + capacity, then sizes the rank-local KV budget via the OfFree (TRT-LLM) /
# OfTotal (vLLM/SGLang) formula. Mirrors NaiveKVCacheEstimator: build via
# from_request(...), then estimate().
#
# Everything is in BYTES (not GiB) so no downstream unit scaling is needed:
# `_get_memory_usage` returns GiB, so the breakdown components are multiplied by
# 1 GiB; `get_kvcache_bytes_per_sequence(1)` is already bytes;
# `system_spec["gpu"]["mem_capacity"]` is already bytes.
# --------------------------------------------------------------------------- #


class KVCacheEstimator:
    """Native KV-cache estimator: AIC's full backend memory model.

    :meth:`from_request` builds the model / backend / perf DB and the non-KV
    memory breakdown -- the part that raises for models AIC cannot build, which is
    the signal for the caller to fall back to :class:`NaiveKVCacheEstimator`.
    :meth:`estimate` then does the pure OfFree (TRT-LLM) / OfTotal (vLLM/SGLang)
    budget math, whose ``ValueError``\\ s (no KV budget, zero per-token) propagate
    rather than triggering the fallback.
    """

    def __init__(self, breakdown: dict[str, Any]):
        self._breakdown = breakdown

    @classmethod
    def from_request(
        cls,
        model_path: str,
        system: str,
        backend: str,
        backend_version: str | None = None,
        *,
        max_num_tokens: int,
        max_batch_size: int,
        tp_size: int = 1,
        pp_size: int = 1,
        attention_dp_size: int = 1,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        gemm_quant_mode: str | None = None,
        moe_quant_mode: str | None = None,
        kvcache_quant_mode: str | None = None,
        fmha_quant_mode: str | None = None,
        comm_quant_mode: str | None = None,
        nextn: int = 0,
        systems_path: str | None = None,
    ) -> KVCacheEstimator:
        """Build the model/backend/perf-DB and the non-KV memory breakdown.

        Reuses the exact AIC machinery the latency path uses: ``build_model_config``
        + ``apply_nextn`` (so the built model is spec-decode aware) + ``get_model``
        (quant inferred inside ``get_model``), ``get_backend``, and
        ``perf_database.get_database``. Calls ``BaseBackend._get_memory_usage`` with
        ``num_tokens = max_num_tokens`` and ``mtp_activation_scaling=False`` (so
        activations track ``BuildConfig.max_num_tokens`` -- the engine's per-iteration
        token budget that already bounds draft tokens -- the way TRT-LLM's
        ``_memory_usage_kwargs_for_agg`` does, without re-applying the ``(nextn+1)``
        decode multiplier); ``isl``/``osl``/``max_seq_len`` only
        feed the discarded ``kvcache`` key, so they are set to a neutral ``1``. Note
        ``max_batch_size`` does NOT affect non-KV memory (activations use
        ``num_tokens``, KV is recomputed per token), but it is accepted to mirror the
        request shape.

        Raises when AIC cannot build the model/backend or the perf DB is missing --
        the signal for the caller to fall back to the naive estimator.
        """
        resolved_moe_tp = moe_tp_size if moe_tp_size is not None else 1
        resolved_moe_ep = moe_ep_size if moe_ep_size is not None else 1
        model_config = build_model_config(
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=attention_dp_size,
            moe_tp_size=resolved_moe_tp,
            moe_ep_size=resolved_moe_ep,
            gemm_quant_mode=gemm_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            moe_quant_mode=moe_quant_mode,
            comm_quant_mode=comm_quant_mode,
        )
        # Apply nextn/MTP onto the config BEFORE get_model so the built model is
        # spec-decode aware (e.g. for any draft-module weights). This does NOT scale
        # the capacity activation by (nextn+1); that multiplier is suppressed below
        # (see mtp_activation_scaling). Mirrors the agg/disagg/static estimate paths.
        # Memory is cost-side only; accepted-token progress never enters
        # capacity math.
        apply_nextn(model_config, nextn)
        model = get_model(model_path, model_config, backend)
        backend_obj = get_backend(backend)
        database = perf_database.get_database(system, backend, backend_version, systems_paths=systems_path)

        # num_tokens = max_num_tokens -> activations track BuildConfig.max_num_tokens
        # (TRT-LLM `_memory_usage_kwargs_for_agg`). With num_tokens > 0 passed
        # explicitly, isl/osl/max_seq_len only feed the discarded `kvcache` key.
        #
        # mtp_activation_scaling=False: max_num_tokens is the engine's per-iteration
        # token budget, which already caps total per-forward tokens INCLUDING the nextn
        # draft tokens. The latency sweep's (nextn+1) activation multiplier would
        # double-count here -- it inflated the prefill worker's non-KV memory past GPU
        # capacity and drove the KV budget negative once the draft length grew
        # (AIC-1110). With it suppressed the breakdown is nextn-independent: the draft
        # module's marginal weight/KV cost (~nextn extra layers) is not separately
        # modeled, so the estimate slightly OVERSTATES available KV (an optimistic
        # approximation) -- but far closer to reality than the gross prior over-count.
        memory = backend_obj._get_memory_usage(
            model,
            database,
            batch_size=int(max_batch_size),
            beam_width=1,
            isl=1,
            osl=1,
            num_tokens=int(max_num_tokens),
            prefix=0,
            max_seq_len=1,
            mtp_activation_scaling=False,
        )

        weights_bytes = float(memory["weights"]) * _ONE_GIB
        activations_bytes = float(memory["activations"]) * _ONE_GIB
        runtime_overhead_bytes = float(memory["others"]) * _ONE_GIB
        comm_overhead_bytes = float(memory["nccl"]) * _ONE_GIB
        non_kv_bytes = weights_bytes + activations_bytes + runtime_overhead_bytes + comm_overhead_bytes

        return cls(
            {
                "weights_bytes": weights_bytes,
                "activations_bytes": activations_bytes,
                "runtime_overhead_bytes": runtime_overhead_bytes,
                "comm_overhead_bytes": comm_overhead_bytes,
                "non_kv_bytes": non_kv_bytes,
                "kv_size_per_token_bytes": float(model.get_kvcache_bytes_per_sequence(1)),
                "gpu_memory_capacity_bytes": float(database.system_spec["gpu"]["mem_capacity"]),
                # Model's byte-budget -> token-count inverse (KV-curve aware).
                "tokens_from_kv_bytes": model.get_kvcache_max_tokens,
            }
        )

    @property
    def breakdown(self) -> dict[str, Any]:
        """The non-KV memory breakdown (BYTE quantities + the ``tokens_from_kv_bytes``
        inverse) computed by :meth:`from_request`."""
        return self._breakdown

    def estimate(
        self,
        *,
        is_of_free: bool,
        fraction: float,
        gpu_memory_capacity_bytes_override: int | None,
    ) -> dict[str, Any]:
        """Size the rank-local KV budget from the breakdown (raw estimate dict)."""
        return self._estimate_from_breakdown(
            self._breakdown,
            is_of_free=is_of_free,
            fraction=fraction,
            gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
        )

    @staticmethod
    def _estimate_from_breakdown(
        breakdown: dict[str, Any],
        *,
        is_of_free: bool,
        fraction: float,
        gpu_memory_capacity_bytes_override: int | None,
    ) -> dict[str, Any]:
        """Native budget math over a breakdown dict (pure; no model build).

        Uses the shared :func:`kv_cache_budget_bytes` (``reserved=tolerance=0``;
        tolerance is applied separately by :func:`_apply_tolerance`). Floors to whole
        bytes/tokens. The token count comes from the breakdown's
        ``tokens_from_kv_bytes`` inverse (the model's KV-curve-aware
        ``get_kvcache_max_tokens``), so hybrid / sliding-window models do not get
        capacity extrapolated from the ``seq_len=1`` slope; when absent (e.g. a
        synthetic breakdown) it defaults to the linear
        :func:`_linear_tokens_from_bytes`. Raises ``ValueError`` for a non-positive
        per-token KV size or budget (these propagate; they do NOT trigger the naive
        fallback, which only fires when the breakdown itself cannot be built).
        """
        kv_per_token = float(breakdown["kv_size_per_token_bytes"])
        if not (math.isfinite(kv_per_token) and kv_per_token > 0.0):
            raise ValueError("insufficient model metadata; missing: kv_size_per_token_bytes")

        if gpu_memory_capacity_bytes_override is not None:
            capacity = float(gpu_memory_capacity_bytes_override)
        else:
            capacity = float(breakdown["gpu_memory_capacity_bytes"])
        if not (math.isfinite(capacity) and capacity > 0.0):
            raise ValueError(f"GPU capacity must be finite and positive, got {capacity}")

        non_kv = float(breakdown["non_kv_bytes"])
        total_kv_size_bytes_f = kv_cache_budget_bytes(
            capacity=capacity, non_kv=non_kv, fraction=fraction, of_free=is_of_free
        )

        if total_kv_size_bytes_f <= 0.0:
            raise ValueError(
                f"no KV budget: non-KV memory ({int(max(non_kv, 0.0))} bytes) meets/exceeds the "
                f"KV-cache memory limit (capacity={int(max(capacity, 0.0))} bytes)"
            )

        tokens_from_bytes: TokensFromBytes = breakdown.get("tokens_from_kv_bytes") or _linear_tokens_from_bytes(
            kv_per_token
        )
        total_kv_size_tokens = int(tokens_from_bytes(total_kv_size_bytes_f))

        return {
            "total_gpu_capacity_bytes": int(max(capacity, 0.0)),
            "total_kv_size_bytes": math.floor(total_kv_size_bytes_f),
            "kv_size_per_token_bytes": math.floor(kv_per_token),
            "total_kv_size_tokens": total_kv_size_tokens,
            # Carried so _apply_tolerance derives adjusted tokens the same way;
            # stripped by estimate_kv_cache before returning.
            "tokens_from_kv_bytes": tokens_from_bytes,
            "source": "native",
            "memory_breakdown": {
                "weights_bytes": int(max(float(breakdown["weights_bytes"]), 0.0)),
                "activations_bytes": int(max(float(breakdown["activations_bytes"]), 0.0)),
                "runtime_overhead_bytes": int(max(float(breakdown["runtime_overhead_bytes"]), 0.0)),
                "comm_overhead_bytes": int(max(float(breakdown["comm_overhead_bytes"]), 0.0)),
            },
            "tolerance_adjusted": None,
        }


# --------------------------------------------------------------------------- #
# Naive fallback: NaiveKVCacheEstimator (HF-config-only) + reserve a fraction of
# post-weight memory.
#
# The estimator reuses the SDK's config loaders + `_parse_hf_config_json`, which
# REJECTS architectures AIC does not support -- exactly the fallback's target --
# so a supported arch (perf DB missing) uses the normalized parse, while an
# unsupported arch reads the raw config keys directly. Both feed one geometry dict
# that drives the per-token KV, weight, and token-capacity math.
# --------------------------------------------------------------------------- #


class NaiveKVCacheMode(enum.Enum):
    """Which KV-growth model :class:`NaiveKVCacheEstimator` inverts."""

    LINEAR = "linear"  # constant bytes/token (MHA/GQA/MLA, or an unknown layout)
    HYBRID = "hybrid"  # SWA layers cap at the window; global layers keep growing


class NaiveKVCacheEstimator:
    """Config-only KV-cache estimator for models AIC cannot build natively.

    Owns the HF-``config.json``-derived geometry and the byte<->token math the
    native backend memory model would otherwise provide, and mirrors the native
    :meth:`~aiconfigurator_core.sdk.models.base.Model.get_kvcache_max_tokens` surface so
    the naive fallback plugs into the same :data:`TokensFromBytes` pipeline.
    Handles standard K/V and MLA attention, and (via :class:`NaiveKVCacheMode`)
    hybrid sliding-window vs linear KV growth.

    The static ``_load_config`` / ``_dtype_bytes`` / ``_swa_layout`` / ``_geometry``
    helpers turn an HF config into the geometry dict; :meth:`from_model_path` wires
    them together. The instance methods compute the per-token / weight byte sizes
    and the token-capacity inverse from that geometry.
    """

    def __init__(
        self,
        geometry: dict,
        *,
        dtype_bytes: int,
        tp_size: int,
        pp_size: int,
        moe_ep_size: int = 1,
        moe_tp_size: int = 1,
    ):
        self.geometry = geometry
        self.dtype_bytes = int(dtype_bytes)
        self.tp_size = int(tp_size)
        self.pp_size = int(pp_size)
        self.moe_ep_size = max(int(moe_ep_size), 1)
        self.moe_tp_size = max(int(moe_tp_size), 1)

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        *,
        tp_size: int,
        pp_size: int,
        allow_hf_config_download: bool,
        moe_ep_size: int = 1,
        moe_tp_size: int = 1,
    ) -> NaiveKVCacheEstimator:
        """Build from an HF ``config.json`` (local dir / pre-cached / optional download).

        Supported architectures go through the normalized parse (which also unwraps
        multimodal); unsupported architectures -- the fallback's real target -- read
        the raw config keys directly. Raises ``ValueError`` when no config can be
        obtained offline.
        """
        hf_config = cls._load_config(model_path, allow_hf_config_download)
        if hf_config is None:
            raise ValueError(
                f"insufficient model metadata: no HF config for {model_path!r} (not a local "
                "directory / not pre-cached, and HF download is disabled)"
            )
        try:
            parsed = _parse_hf_config_json(hf_config)  # supported arch -> normalized parse
        except Exception:
            parsed = None  # unsupported arch (the fallback's target) -> raw-key read
        return cls(
            cls._geometry(hf_config, parsed),
            dtype_bytes=cls._dtype_bytes(hf_config),
            tp_size=tp_size,
            pp_size=pp_size,
            moe_ep_size=moe_ep_size,
            moe_tp_size=moe_tp_size,
        )

    # ----------------------------- config -> geometry ----------------------------- #

    @staticmethod
    def _load_config(model_path: str, allow_hf_config_download: bool) -> dict | None:
        """Load a raw HF ``config.json`` dict, or ``None`` when unavailable offline.

        Reuses the SDK loaders: a local directory -> the pre-downloaded
        ``model_configs`` cache -> an HF download (ONLY when
        ``allow_hf_config_download`` is set). Returns ``None`` when no config can be
        obtained without downloading; :meth:`from_model_path` raises in that case.
        """
        if os.path.isdir(model_path):
            return _load_local_config(model_path)
        if model_path in DefaultHFModels:
            return _load_pre_downloaded_hf_config(model_path)
        if allow_hf_config_download:
            return _download_hf_config(model_path)
        return None

    @staticmethod
    def _dtype_bytes(hf_config: dict) -> int:
        """dtype byte width from ``torch_dtype``/``dtype``. Defaults to 2 (bf16)."""
        dtype_str = (hf_config.get("torch_dtype") or hf_config.get("dtype") or "").lower()
        if dtype_str in ("float32", "fp32"):
            return 4
        if dtype_str in ("bfloat16", "float16", "bf16", "fp16", "half"):
            return 2
        if "fp8" in dtype_str or "float8" in dtype_str or "int8" in dtype_str:
            return 1
        return 2

    @staticmethod
    def _swa_layout(hf_config: dict) -> tuple[int | None, int | None, int | None]:
        """Sliding-window size + (SWA, global) layer counts from an HF config.

        Returns ``(sliding_window, num_swa_layers, num_global_layers)`` for a hybrid
        sliding-window model, or ``(None, None, None)`` when the layout cannot be
        pinned down (so the estimator stays on its linear per-token mode). Two
        signals are recognized, in order:

        - ``layer_types`` (Gemma-3/4-style explicit per-layer list): count the
          ``sliding_*`` entries vs the rest.
        - ``sliding_window`` + ``sliding_window_pattern`` (an integer ``P``): the
          Gemma convention of one global layer every ``P`` layers, the rest sliding.

        A bare ``sliding_window`` with no per-layer signal is left unresolved -- the
        split is too model-specific to guess, and linear is the safe default.
        """
        sliding_window = hf_config.get("sliding_window") or hf_config.get("sliding_window_size")
        layer_types = hf_config.get("layer_types")
        if isinstance(layer_types, list) and layer_types:
            num_swa = sum(1 for t in layer_types if "sliding" in str(t).lower())
            return sliding_window, num_swa, len(layer_types) - num_swa

        num_layers = hf_config.get("num_hidden_layers")
        pattern = hf_config.get("sliding_window_pattern")
        if sliding_window and num_layers and pattern and int(pattern) > 0:
            num_global = int(num_layers) // int(pattern)
            return int(sliding_window), int(num_layers) - num_global, num_global

        return None, None, None

    @classmethod
    def _geometry(cls, hf_config: dict, parsed: dict | None) -> dict:
        """Normalized attention/FFN scalars the byte formulas consume.

        From :func:`~aiconfigurator_core.sdk.utils._parse_hf_config_json` output when the
        architecture is supported (``parsed`` not ``None``), else from the HF config
        keys directly (unsupported architecture). Missing values stay ``None`` /
        ``0`` so the formulas can fail closed.
        """
        if parsed is not None:
            extra = parsed.get("extra_params")
            if isinstance(extra, dict):
                kv_lora = extra.get("kv_lora_rank")
                qk_rope = extra.get("qk_rope_head_dim")
            else:
                # Some supported archs carry a dataclass extra_params (e.g. NemotronH,
                # Gemma4Mix, DeepSeek-V4); read MLA latent fields by attribute when
                # present. Archs whose latent geometry is NOT a simple
                # (kv_lora_rank + qk_rope_head_dim) sum -- notably DeepSeek-V4's sparse
                # attention -- expose neither and fall through to the standard branch,
                # a rough approximation acceptable for this naive fallback.
                kv_lora = getattr(extra, "kv_lora_rank", None)
                qk_rope = getattr(extra, "qk_rope_head_dim", None)
            return {
                "layers": parsed.get("layers"),
                "num_kv_heads": parsed.get("n_kv") or parsed.get("n"),
                "head_dim": parsed.get("d"),
                "kv_lora_rank": kv_lora,
                "qk_rope_head_dim": qk_rope,
                "vocab": parsed.get("vocab"),
                "hidden": parsed.get("hidden_size"),
                "inter": parsed.get("inter_size"),
                "num_experts": parsed.get("num_experts") or 0,
                "moe_inter": parsed.get("moe_inter_size") or 0,
                # Supported (native-capable) archs do not take the naive path, so the
                # sliding-window split is only mined from the HF config (below).
                "sliding_window": None,
                "num_swa_layers": None,
                "num_global_layers": None,
            }

        hidden = hf_config.get("hidden_size")
        heads = hf_config.get("num_attention_heads")
        head_dim = hf_config.get("head_dim") or hf_config.get("attention_head_dim")
        if head_dim is None and hidden and heads:
            head_dim = int(hidden) // int(heads)
        sliding_window, num_swa_layers, num_global_layers = cls._swa_layout(hf_config)
        return {
            "layers": hf_config.get("num_hidden_layers"),
            "num_kv_heads": hf_config.get("num_key_value_heads") or heads,
            "head_dim": head_dim,
            "kv_lora_rank": hf_config.get("kv_lora_rank"),
            "qk_rope_head_dim": hf_config.get("qk_rope_head_dim"),
            "vocab": hf_config.get("vocab_size"),
            "hidden": hidden,
            "inter": hf_config.get("intermediate_size"),
            "num_experts": hf_config.get("n_routed_experts") or hf_config.get("num_local_experts") or 0,
            "moe_inter": hf_config.get("moe_intermediate_size") or hf_config.get("intermediate_size") or 0,
            "sliding_window": sliding_window,
            "num_swa_layers": num_swa_layers,
            "num_global_layers": num_global_layers,
        }

    # --------------------------- geometry -> byte sizes --------------------------- #

    def kv_bytes_per_token(self) -> int | None:
        """Per-token KV byte size (the ``seq_len=1`` rate), MLA-aware, or ``None``.

        Standard K/V and MLA/compressed-latent attention are handled separately
        (MLA stores one shared latent cache per layer, not per-head K and V):

        - MLA (``kv_lora_rank`` present): ``layers * (kv_lora_rank + qk_rope_head_dim)
          * dtype_bytes`` (the latent is shared across heads, NOT sharded by TP).
          Fails closed (``None``) when ``qk_rope_head_dim`` is absent so the caller
          raises rather than guessing the latent width.
        - Standard (GQA/MHA): ``2 * ceil(num_kv_heads / tp) * head_dim * layers *
          dtype_bytes``.
        """
        geom = self.geometry
        layers = geom.get("layers")
        if not layers:
            return None

        kv_lora = geom.get("kv_lora_rank")
        if kv_lora:
            qk_rope = geom.get("qk_rope_head_dim")
            if not qk_rope:
                # MLA but latent width unclear -> fail closed (None; the caller raises).
                return None
            return int(layers) * (int(kv_lora) + int(qk_rope)) * self.dtype_bytes

        num_kv_heads = geom.get("num_kv_heads")
        head_dim = geom.get("head_dim")
        if not num_kv_heads or not head_dim:
            return None
        kv_heads_per_rank = -(-int(num_kv_heads) // max(self.tp_size, 1))  # ceil division
        return 2 * kv_heads_per_rank * int(head_dim) * int(layers) * self.dtype_bytes

    def weight_bytes(self) -> int | None:
        """Rough per-rank weight estimate (bytes) from the geometry, or ``None``.

        Coarse param count (embeddings + per-layer attention + dense/MoE FFN), times
        dtype bytes, divided by TP*PP. NOT calibrated; for models AIC cannot fully
        model. Attention-DP replicates rather than shards, so it does NOT divide
        here. ``hidden_size`` / ``num_hidden_layers`` / ``vocab_size`` /
        ``intermediate_size`` are all required: returns ``None`` (the caller raises)
        when any is missing, rather than fabricating a placeholder.
        """
        geom = self.geometry
        hidden = geom.get("hidden")
        layers = geom.get("layers")
        vocab = geom.get("vocab")
        inter = geom.get("inter")
        if not hidden or not layers or not vocab or not inter:
            return None
        hidden = int(hidden)
        layers = int(layers)
        vocab = int(vocab)
        inter = int(inter)

        # Embeddings: input + (untied) output projection ~ 2 * vocab * hidden.
        embed_params = 2 * vocab * hidden
        # Per-layer attention q/k/v/o ~ 4 * hidden**2.
        attn_params_per_layer = 4 * hidden * hidden

        n_experts = int(geom.get("num_experts") or 0)
        pp = max(self.pp_size, 1)
        tp = max(self.tp_size, 1)
        if n_experts > 0:
            moe_inter = int(geom.get("moe_inter") or inter)
            # Expert FFN is sharded by moe_tp * moe_ep, not by the attention TP.
            moe_divisor = self.moe_tp_size * self.moe_ep_size
            expert_params = n_experts * 3 * hidden * moe_inter // moe_divisor
            non_expert_params = embed_params + layers * attn_params_per_layer
            non_expert_bytes = non_expert_params * self.dtype_bytes // (tp * pp)
            expert_bytes = layers * expert_params * self.dtype_bytes // pp
            return non_expert_bytes + expert_bytes
        else:
            ffn_params_per_layer = 3 * hidden * inter
            total_params = embed_params + layers * (attn_params_per_layer + ffn_params_per_layer)
            return (total_params * self.dtype_bytes) // (tp * pp)

    # --------------------------- byte budget -> tokens ---------------------------- #

    @property
    def mode(self) -> NaiveKVCacheMode:
        """HYBRID when the config exposes a usable sliding-window layout, else LINEAR."""
        return NaiveKVCacheMode.HYBRID if self._hybrid_tokens_from_bytes() is not None else NaiveKVCacheMode.LINEAR

    def get_kvcache_max_tokens(self, kv_budget_bytes: float, mode: NaiveKVCacheMode | None = None) -> int:
        """KV byte budget -> token capacity, mirroring ``Model.get_kvcache_max_tokens``.

        ``mode`` defaults to :attr:`mode` (auto: HYBRID when a sliding-window layout
        is known, else LINEAR); pass it explicitly to force one. HYBRID caps the SWA
        layers at the window while global layers keep growing; LINEAR floor-divides
        the budget by the per-token size. An explicit HYBRID with no usable layout
        falls back to LINEAR rather than raising.
        """
        hybrid = self._hybrid_tokens_from_bytes()
        use_hybrid = mode is NaiveKVCacheMode.HYBRID or (mode is None and hybrid is not None)
        if use_hybrid and hybrid is not None:
            return hybrid(kv_budget_bytes)
        return _linear_tokens_from_bytes(float(self.kv_bytes_per_token() or 0.0))(kv_budget_bytes)

    def estimate(self, *, capacity: int | None, naive_kv_reservation: float) -> dict[str, Any]:
        """Conservative naive estimate: reserve ``naive_kv_reservation`` of post-weight memory.

        The naive counterpart to :meth:`KVCacheEstimator.estimate`. ``capacity`` is
        the per-rank GPU memory (the ``gpu_memory_capacity_bytes_override`` the caller
        must supply, since the native SystemSpec capacity is unavailable when this
        path runs). Does NOT honor the requested ``memory_fraction`` -- the
        post-weight reservation is the crude budget. The estimator auto-selects
        HYBRID (window-capped) vs LINEAR from the config, so hybrid sliding-window
        models do not get capacity extrapolated from the (larger) ``seq_len=1`` rate.
        RAISES ``ValueError`` when ``capacity`` is missing or the config lacks the
        metadata to estimate the weights / per-token KV (rather than guessing with a
        placeholder constant).
        """
        if not (capacity and int(capacity) > 0):
            raise ValueError(
                "naive fallback requires gpu_memory_capacity_bytes_override (the native "
                "path failed, so SystemSpec capacity is unavailable)"
            )
        capacity_bytes = float(capacity)

        kv_per_token = self.kv_bytes_per_token()
        if kv_per_token is None or not (math.isfinite(kv_per_token) and kv_per_token > 0.0):
            raise ValueError(
                "insufficient model metadata; missing: per-token KV geometry "
                "(num_hidden_layers + num_key_value_heads/head_dim, or the MLA latent dims)"
            )
        weight_bytes = self.weight_bytes()
        if weight_bytes is None:
            raise ValueError(
                "insufficient model metadata; missing: weight-estimate fields "
                "(hidden_size, num_hidden_layers, vocab_size, intermediate_size)"
            )
        kv_per_token = float(kv_per_token)
        weight_bytes = float(weight_bytes)

        post_weight = max(capacity_bytes - weight_bytes, 0.0)
        total_kv_size_bytes_f = post_weight * float(naive_kv_reservation)
        if total_kv_size_bytes_f <= 0.0:
            raise ValueError(
                f"no KV budget: non-KV memory ({int(max(weight_bytes, 0.0))} bytes) meets/exceeds the "
                f"KV-cache memory limit (capacity={int(max(capacity_bytes, 0.0))} bytes)"
            )

        return {
            "total_gpu_capacity_bytes": int(max(capacity_bytes, 0.0)),
            "total_kv_size_bytes": math.floor(total_kv_size_bytes_f),
            "kv_size_per_token_bytes": math.floor(kv_per_token),
            "total_kv_size_tokens": self.get_kvcache_max_tokens(total_kv_size_bytes_f),
            # Carried so _apply_tolerance derives adjusted tokens the same way;
            # stripped by estimate_kv_cache before returning.
            "tokens_from_kv_bytes": self.get_kvcache_max_tokens,
            "source": "naive_fallback",
            "memory_breakdown": None,
            "tolerance_adjusted": None,
        }

    def _hybrid_tokens_from_bytes(self) -> TokensFromBytes | None:
        """Piecewise sliding-window inverse closure, or ``None`` when not hybrid.

        For a STANDARD-attention model with a known sliding-window layout: caps the
        SWA layers at the window while the global layers keep growing, so capacity
        is not extrapolated from the (larger) ``seq_len=1`` rate. ``None`` (caller
        stays LINEAR) for non-hybrid models, MLA hybrids (latent cache; out of scope
        for this simple heuristic), or incomplete geometry. Deliberately coarse: it
        assumes one head geometry across both layer types.
        """
        geom = self.geometry
        if geom.get("kv_lora_rank"):  # MLA latent cache: piecewise split N/A here
            return None
        window = geom.get("sliding_window")
        num_swa = geom.get("num_swa_layers")
        num_global = geom.get("num_global_layers")
        head_dim = geom.get("head_dim")
        num_kv_heads = geom.get("num_kv_heads")
        if not window or not num_swa or num_global is None or not head_dim or not num_kv_heads:
            return None

        window = int(window)
        kv_heads_per_rank = -(-int(num_kv_heads) // max(self.tp_size, 1))  # ceil division
        per_layer = 2 * kv_heads_per_rank * int(head_dim) * self.dtype_bytes
        swa_const = int(num_swa) * per_layer  # bytes/token for the window-capped layers
        global_const = int(num_global) * per_layer  # bytes/token for the growing layers

        def tokens_from_bytes(budget_bytes: float) -> int:
            budget = float(budget_bytes)
            if budget <= 0.0:
                return 0
            within_window_rate = swa_const + global_const  # every layer grows up to the window
            bytes_at_window = within_window_rate * window
            if budget <= bytes_at_window:
                return int(budget // within_window_rate) if within_window_rate > 0 else 0
            if global_const <= 0:  # all layers capped -> KV can't grow past the window
                return window
            return window + int((budget - bytes_at_window) // global_const)

        return tokens_from_bytes


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #


def estimate_kv_cache(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction_kind: str,
    memory_fraction_value: float,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    nextn: int = 0,
    systems_path: str | None = None,
    gpu_memory_capacity_bytes_override: int | None = None,
    tolerance_fraction: float | None = None,
    naive_kv_reservation: float = _DEFAULT_NAIVE_KV_RESERVATION,
    allow_naive_fallback: bool = False,
    allow_hf_config_download: bool = False,
) -> dict[str, Any]:
    """Compute the KV-cache memory estimate (raw + optional tolerance margin).

    This is the complete implementation: fraction + tolerance validation, the
    native/naive budget math, AND the tolerance margin. The Rust
    ``aiconfigurator_core::memory::estimate_kv_cache`` is a pure forwarder that
    calls this function and rebuilds its result, so this is the single source of
    truth for the estimate.

    Native path: :meth:`KVCacheEstimator.from_request` builds AIC's full backend
    memory model and :meth:`KVCacheEstimator.estimate` does the OfFree (TRT-LLM) /
    OfTotal (vLLM/SGLang) budget math. Naive fallback: when the native model build
    is unsupported (``from_request`` raises) AND ``allow_naive_fallback`` is set,
    :meth:`NaiveKVCacheEstimator.estimate` parses the HF ``config.json`` and
    reserves ``naive_kv_reservation`` of post-weight memory for KV; this requires
    ``gpu_memory_capacity_bytes_override``.

    Args:
        memory_fraction_kind: ``"of_total"`` (vLLM / SGLang) or ``"of_free"``
            (TRT-LLM); validated against ``backend`` BEFORE any model build.
        memory_fraction_value: the fraction in ``[0, 1]``.
        gpu_memory_capacity_bytes_override: when set, wins over the SystemSpec
            capacity on the native path; REQUIRED on the naive fallback.
        tolerance_fraction: optional safety margin in ``[0, 1)``; when set,
            ``tolerance_adjusted`` is populated with ``floor(raw * (1 - t))``
            bytes and the recomputed token count. ``None`` -> raw estimate only.
        naive_kv_reservation: fraction of post-weight memory the naive fallback
            reserves for KV (default ``0.80``). Ignored on the native path.
        allow_naive_fallback: fall back to the naive heuristic when the native
            model build is unsupported (default off -> the error propagates).
        allow_hf_config_download: allow the naive fallback to download a
            ``config.json`` from HuggingFace when it is not local / pre-cached.

    Returns:
        A flat dict with ``total_gpu_capacity_bytes``, ``total_kv_size_bytes``,
        ``kv_size_per_token_bytes``, ``total_kv_size_tokens``, ``source``
        (``"native"`` | ``"naive_fallback"``), ``memory_breakdown`` (dict on the
        native path, ``None`` on the fallback), and ``tolerance_adjusted`` (a dict
        when ``tolerance_fraction`` is set, else ``None``).

    Raises:
        ValueError: incompatible / out-of-range memory fraction, out-of-range
            tolerance, no KV budget, insufficient model metadata, or (with the
            fallback off) an unsupported model/backend.
    """
    _validate_memory_fraction(backend, memory_fraction_kind, memory_fraction_value)
    _validate_tolerance(tolerance_fraction)
    _validate_naive_reservation(naive_kv_reservation)
    fraction = float(memory_fraction_value)
    is_of_free = memory_fraction_kind == "of_free"

    # Validate the compute-side MTP depth before any fallback path.
    validate_nextn(nextn)

    try:
        native = KVCacheEstimator.from_request(
            model_path,
            system,
            backend,
            backend_version,
            max_num_tokens=int(max_num_tokens),
            max_batch_size=int(max_batch_size),
            tp_size=int(tp_size),
            pp_size=int(pp_size),
            attention_dp_size=int(attention_dp_size),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=gemm_quant_mode,
            moe_quant_mode=moe_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            comm_quant_mode=comm_quant_mode,
            nextn=int(nextn),
            systems_path=systems_path,
        )
    except Exception as exc:  # native model build unsupported (model/backend/perf DB)
        if not allow_naive_fallback:
            raise ValueError(
                f"unsupported model/backend/GPU for KV-cache estimation: "
                f"model={model_path}, backend={backend}, gpu_sku={system}: {exc}"
            ) from exc
        estimate = NaiveKVCacheEstimator.from_model_path(
            model_path,
            tp_size=int(tp_size),
            pp_size=int(pp_size),
            allow_hf_config_download=allow_hf_config_download,
            moe_ep_size=int(moe_ep_size or 1),
            moe_tp_size=int(moe_tp_size or 1),
        ).estimate(
            capacity=gpu_memory_capacity_bytes_override,
            naive_kv_reservation=float(naive_kv_reservation),
        )
    else:
        # Native model built: budget-math failures (no KV budget, zero per-token)
        # propagate; they do NOT trigger the naive fallback.
        estimate = native.estimate(
            is_of_free=is_of_free,
            fraction=fraction,
            gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
        )

    # Apply the tolerance margin (no-op when `tolerance_fraction` is None),
    # identically for the native and naive estimates. _apply_tolerance reads the
    # carried `tokens_from_kv_bytes` inverse; strip it afterwards so the result is
    # the flat, serializable dict the Rust forwarder parses.
    _apply_tolerance(estimate, tolerance_fraction)
    estimate.pop("tokens_from_kv_bytes", None)
    return estimate


def estimate_num_gpu_blocks(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    scheduler_block_size: int,
    max_num_tokens: int,
    max_batch_size: int,
    memory_fraction_kind: str,
    memory_fraction_value: float,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    nextn: int = 0,
    systems_path: str | None = None,
    gpu_memory_capacity_bytes_override: int | None = None,
    tolerance_fraction: float | None = None,
    naive_kv_reservation: float = _DEFAULT_NAIVE_KV_RESERVATION,
    allow_naive_fallback: bool = False,
    allow_hf_config_download: bool = False,
) -> int:
    """Convert the KV-cache token capacity to a scheduler block count.

    Thin convenience over :func:`estimate_kv_cache`: computes the per-rank
    KV-cache token capacity and returns ``floor(total_kv_size_tokens /
    scheduler_block_size)``, using the tolerance-adjusted token count when
    ``tolerance_fraction`` is set (the raw count otherwise).

    Args:
        scheduler_block_size: KV block size the scheduler allocates in (tokens).
        memory_fraction_kind: ``"of_total"`` (vLLM / SGLang) or ``"of_free"``
            (TRT-LLM); validated against ``backend``.
        memory_fraction_value: the fraction in ``[0, 1]``.
        (remaining kwargs mirror :func:`estimate_kv_cache`.)

    Returns:
        ``floor(total_kv_size_tokens / scheduler_block_size)``, using the
        tolerance-adjusted tokens when ``tolerance_fraction`` is set.

    Raises:
        ValueError: propagated from :func:`estimate_kv_cache` (unsupported
            model/backend, incompatible memory fraction, out-of-range tolerance,
            no KV budget, etc.).
    """
    block_size = int(scheduler_block_size)
    if block_size <= 0 or block_size != scheduler_block_size:
        raise ValueError(f"scheduler_block_size must be a positive integer, got {scheduler_block_size!r}")

    estimate = estimate_kv_cache(
        model_path,
        system,
        backend,
        backend_version=backend_version,
        max_num_tokens=int(max_num_tokens),
        max_batch_size=int(max_batch_size),
        memory_fraction_kind=memory_fraction_kind,
        memory_fraction_value=float(memory_fraction_value),
        tp_size=int(tp_size),
        pp_size=int(pp_size),
        attention_dp_size=int(attention_dp_size),
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        gemm_quant_mode=gemm_quant_mode,
        moe_quant_mode=moe_quant_mode,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        comm_quant_mode=comm_quant_mode,
        nextn=int(nextn),
        systems_path=systems_path,
        gpu_memory_capacity_bytes_override=gpu_memory_capacity_bytes_override,
        tolerance_fraction=tolerance_fraction,
        naive_kv_reservation=float(naive_kv_reservation),
        allow_naive_fallback=allow_naive_fallback,
        allow_hf_config_download=allow_hf_config_download,
    )

    adjusted = estimate.get("tolerance_adjusted")
    if adjusted is not None:
        tokens = int(adjusted["total_kv_size_tokens"])
    else:
        tokens = int(estimate["total_kv_size_tokens"])

    return tokens // block_size
