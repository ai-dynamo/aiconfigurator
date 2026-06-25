# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.operations.layerwise import SCHEDULER_ENVELOPE_LATENCY_SOURCES
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError
from aiconfigurator.sdk.performance_result import PerformanceResult

logger = logging.getLogger(__name__)

_USE_LAYERWISE = os.environ.get("AIC_VLLM_USE_LAYERWISE", "0") == "1"
_LAYERWISE_SCHEDULER_LATENCY_SOURCES = SCHEDULER_ENVELOPE_LATENCY_SOURCES
# When a full-step scheduler envelope was collected on fewer physical GPUs than
# tp_size (e.g. single-GPU shape sweeps), it cannot contain the real tensor-parallel
# all-reduce, so the explicit generation all-reduce term must still be added. Set
# False to restore the legacy behavior (always trust a full-step envelope).
_LAYERWISE_GEN_SINGLE_GPU_COMM = True
# Real vLLM fuses the all-reduce with the residual-add + RMSNorm into one kernel,
# which is substantially cheaper than a standalone all-reduce (especially at high
# tp). Use the collected fused `allreduce_residual_rms` timings instead of the
# standalone custom all-reduce. Set False to use standalone custom all-reduce.
_LAYERWISE_USE_FUSED_ALLREDUCE_RMS = True
# The single-GPU decode-compute microbenchmark grows too gently with batch vs real
# serving (paged attention over fragmented/variable KV + per-sequence overhead).
# Scale the per-step decode COMPUTE (layerwise) by (1 + k*batch); comm is modeled
# separately via the fused all-reduce so it is NOT scaled here. k fit against FPM.
# Set to 0.0 to disable.
_DECODE_COMPUTE_BATCH_CAL = 0.0066


class VLLMBackend(BaseBackend):
    """vLLM backend.

    Currently mirrors TRT-LLM's activation-memory model (the pre-refactor
    implementation literally delegated ``_get_memory_usage`` to TRTLLMBackend),
    with no KV-cache-aware OOM accounting yet. We reuse both TRT-LLM's
    per-family coefficient table and its ``_moe_workspace_width`` hook so
    estimates stay byte-identical with the old delegation; the agg-pipeline
    hooks (``_resolve_agg_kwargs``, ``_oom_check_kwargs``, ...) remain at
    BaseBackend defaults — vLLM does not yet do KV-cache OOM probing.
    """

    # Reuse TRT-LLM's per-family activation coefficients until a vLLM-specific
    # tuning lands.
    ACTIVATION_COEFFICIENTS = TRTLLMBackend.ACTIVATION_COEFFICIENTS

    # Mirror TRT-LLM's MoE workspace accounting (raw h for DEEPSEEK family,
    # ``_hidden_size`` for GEMMA4MIX). Plain class-attribute alias to the
    # function object — Python binds it to the VLLMBackend instance at call
    # time; the function does not touch any TRTLLMBackend-specific state.
    _moe_workspace_width = TRTLLMBackend._moe_workspace_width

    def __init__(self):
        super().__init__()
        self.name = common.BackendName.vllm

    def _layerwise_runtime_config(self, runtime_config: RuntimeConfig) -> RuntimeConfig:
        if not _USE_LAYERWISE:
            return runtime_config
        return replace(runtime_config, engine_step_backend="python")

    def _run_static_breakdown(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
        img_ctx_tokens: int = 0,
    ):
        return super()._run_static_breakdown(
            model,
            database,
            self._layerwise_runtime_config(runtime_config),
            mode,
            stride,
            latency_correction_scale,
            img_ctx_tokens,
        )

    def run_agg(self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs):
        return super().run_agg(model, database, self._layerwise_runtime_config(runtime_config), **kwargs)

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ):
        return super().find_best_agg_result_under_constraints(
            model, database, self._layerwise_runtime_config(runtime_config), **kwargs
        )

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, osl: int) -> int:
        # vLLM v1 scheduler sets max_num_partial_prefills=1 by default, meaning
        # exactly one request is in partial-prefill state per forward pass.
        # The remaining b - ceil(ctx_tokens/isl) requests are in decode phase.
        # This applies regardless of whether steps_to_finish_ctx >= osl or not,
        # giving a consistent formula across both scheduling regimes.
        # Source: vllm/v1/core/sched/scheduler.py, SchedulerConfig.max_num_partial_prefills
        return max(1, b - int(np.ceil(ctx_tokens / isl)))

    def _layerwise_context_layer_ms(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int = 0,
        max_num_batched_tokens: int | None = None,
    ) -> float:
        return float(
            self._layerwise_context_layer_detail(
                database,
                model_name,
                tp_size,
                batch_size,
                seq_len,
                prefix,
                max_num_batched_tokens=max_num_batched_tokens,
            )["latency"]
        )

    def _layerwise_combined_context_chunk_detail(
        self,
        chunk_details: list[dict],
        max_num_batched_tokens: int,
    ) -> dict[str, float | bool | list[dict]]:
        components: list[dict] = []
        for detail in chunk_details:
            detail_components = detail.get("components")
            if isinstance(detail_components, list):
                components.extend(dict(component) for component in detail_components if isinstance(component, dict))
            else:
                components.append(dict(detail))
        first_mode = self._layerwise_detail_moe_weight_mode(chunk_details[0])
        if not all(self._layerwise_detail_moe_weight_mode(detail) == first_mode for detail in chunk_details):
            first_mode = ""
        return {
            "latency": sum(float(detail["latency"]) for detail in chunk_details),
            "energy": sum(float(detail.get("energy", 0.0)) for detail in chunk_details),
            "rms_latency": sum(float(detail.get("rms_latency", 0.0)) for detail in chunk_details),
            "includes_moe": all(bool(detail.get("includes_moe", False)) for detail in chunk_details),
            "latency_source": (
                chunk_details[0].get("latency_source", "")
                if all(
                    str(detail.get("latency_source", "")) == str(chunk_details[0].get("latency_source", ""))
                    for detail in chunk_details
                )
                else "mixed"
            ),
            "moe_weight_mode": first_mode,
            "max_num_batched_tokens": float(max_num_batched_tokens),
            "layer_type": chunk_details[0].get("layer_type", ""),
            "layer_index": float(chunk_details[0].get("layer_index", 0.0)),
            "measured_layer_count": float(chunk_details[0].get("measured_layer_count", 1.0)),
            "layer_multiplier": float(chunk_details[0].get("layer_multiplier", 0.0)),
            "components": components,
        }

    def _layerwise_context_layer_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int = 0,
        max_num_batched_tokens: int | None = None,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        allow_zero_prefix_chunk_fallback: bool = False,
    ) -> dict[str, float | bool]:
        if max_num_batched_tokens is None or seq_len <= max_num_batched_tokens:
            try:
                return self._query_layerwise_detail(
                    database,
                    model_name,
                    "CTX",
                    tp_size,
                    batch_size,
                    seq_len,
                    seq_len_kv_cache=max(prefix, 0),
                    moe_weight_mode=moe_weight_mode,
                    max_num_batched_tokens=max_num_batched_tokens,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
            except (KeyError, PerfDataNotAvailableError, ValueError):
                if max_num_batched_tokens is None:
                    raise
                return self._query_layerwise_detail(
                    database,
                    model_name,
                    "CTX",
                    tp_size,
                    batch_size,
                    seq_len,
                    seq_len_kv_cache=max(prefix, 0),
                    moe_weight_mode=moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )

        chunk_details = []
        remaining = seq_len
        past_kv = max(prefix, 0)
        try:
            while remaining > 0:
                chunk_tokens = min(max_num_batched_tokens, remaining)
                chunk_details.append(
                    self._query_layerwise_detail(
                        database,
                        model_name,
                        "CTX",
                        tp_size,
                        batch_size,
                        chunk_tokens,
                        seq_len_kv_cache=past_kv,
                        moe_weight_mode=moe_weight_mode,
                        max_num_batched_tokens=max_num_batched_tokens,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                )
                past_kv += chunk_tokens
                remaining -= chunk_tokens
            return self._layerwise_combined_context_chunk_detail(chunk_details, max_num_batched_tokens)
        except (KeyError, PerfDataNotAvailableError, ValueError):
            if allow_zero_prefix_chunk_fallback:
                try:
                    remaining = seq_len
                    chunk_details = []
                    while remaining > 0:
                        chunk_tokens = min(max_num_batched_tokens, remaining)
                        chunk_details.append(
                            self._query_layerwise_detail(
                                database,
                                model_name,
                                "CTX",
                                tp_size,
                                batch_size,
                                chunk_tokens,
                                seq_len_kv_cache=0,
                                moe_weight_mode=moe_weight_mode,
                                max_num_batched_tokens=max_num_batched_tokens,
                                moe_tp_size=moe_tp_size,
                                moe_ep_size=moe_ep_size,
                            )
                        )
                        remaining -= chunk_tokens
                    return self._layerwise_combined_context_chunk_detail(chunk_details, max_num_batched_tokens)
                except (KeyError, PerfDataNotAvailableError, ValueError):
                    pass
            # Older layerwise datasets have a direct 16k CTX row but no
            # nonzero-context-KV chunk rows. Use that row until chunked data
            # exists for the requested TP.
            if max_num_batched_tokens is not None:
                try:
                    return self._query_layerwise_detail(
                        database,
                        model_name,
                        "CTX",
                        tp_size,
                        batch_size,
                        seq_len,
                        moe_weight_mode=moe_weight_mode,
                        max_num_batched_tokens=max_num_batched_tokens,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                except (KeyError, PerfDataNotAvailableError, ValueError):
                    pass
            return self._query_layerwise_detail(
                database,
                model_name,
                "CTX",
                tp_size,
                batch_size,
                seq_len,
                moe_weight_mode=moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )

    def _layerwise_context_chunk_size(
        self,
        runtime_config: RuntimeConfig,
        detail: dict | None = None,
    ) -> int | None:
        """Return explicit vLLM context chunk size for layerwise estimates."""

        runtime_value = getattr(runtime_config, "vllm_max_num_batched_tokens", None)
        if runtime_value:
            return max(1, int(runtime_value))
        env_value = os.environ.get("AIC_VLLM_MAX_NUM_BATCHED_TOKENS")
        if env_value:
            return max(1, int(env_value))
        if detail is not None:
            detail_value = detail.get("max_num_batched_tokens")
            if detail_value:
                return max(1, int(float(detail_value)))
        return None

    def _layerwise_has_subquadratic_context_attention(self, model: BaseModel) -> bool:
        """Return whether continuation context should scale below full quadratic attention."""

        extra_params = getattr(model, "extra_params", None) or getattr(
            getattr(model, "config", None),
            "extra_params",
            None,
        )
        compress_ratios = getattr(extra_params, "compress_ratios", None)
        if compress_ratios:
            return True
        for attr in ("sliding_window", "sliding_window_size", "attention_chunk_size"):
            try:
                if int(getattr(extra_params, attr, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _layerwise_context_noop_moe_weight_mode(
        self,
        database: PerfDatabase,
        *,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        runtime_config: RuntimeConfig,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> str | None:
        """Return the no-op MoE row mode when chunked context data provides it."""

        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        if chunk_size is None or int(seq_len) <= chunk_size:
            return None
        try:
            detail = self._query_layerwise_detail(
                database,
                model_name,
                "CTX",
                tp_size,
                batch_size,
                min(int(seq_len), chunk_size),
                seq_len_kv_cache=0,
                moe_weight_mode="noop",
                max_num_batched_tokens=chunk_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
        except (KeyError, PerfDataNotAvailableError, ValueError):
            return None
        if self._layerwise_detail_moe_weight_mode(detail) == "noop" and not bool(detail.get("includes_moe", False)):
            return "noop"
        return None

    def _layerwise_context_noop_moe_distribution_override(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        runtime_config: RuntimeConfig,
    ) -> str | None:
        """Return a context-only MoE distribution for no-op chunked prefill."""

        if str(getattr(model.config, "workload_distribution", "power_law")) != "power_law":
            return None
        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        if chunk_size is None or int(token_count) <= chunk_size:
            return None

        query_token_count = min(int(token_count), chunk_size) * int(getattr(model.config, "attention_dp_size", 1) or 1)
        distribution = "sampled_zipf_1.2"
        if self._layerwise_moe_distribution_available(
            model,
            database,
            distribution,
            token_count=query_token_count,
        ):
            return distribution
        return None

    def _layerwise_context_lookup_tp_size(self, model: BaseModel, tp_size: int) -> int:
        """Return the measured attention-TP surface to use for context rows."""

        return max(1, int(tp_size))

    def _layerwise_context_lookup_tp_size_for_shape(
        self,
        model: BaseModel,
        runtime_config: RuntimeConfig,
        *,
        tp_size: int,
        effective_isl: int,
        ctx_requests: int,
    ) -> int:
        """Return the context surface for a concrete scheduler-step shape."""

        lookup_tp_size = self._layerwise_context_lookup_tp_size(model, tp_size)
        if lookup_tp_size <= 1 or int(ctx_requests) != 1:
            return lookup_tp_size
        if int(getattr(model, "_num_experts", 0) or 0) <= 0:
            return lookup_tp_size
        if self._layerwise_has_subquadratic_context_attention(model):
            return lookup_tp_size

        chunk_size = self._layerwise_context_chunk_size(runtime_config) or 2048
        long_context_threshold = max(512, int(chunk_size) // 4)
        if int(effective_isl) < long_context_threshold:
            return lookup_tp_size
        return 1

    def _query_layerwise_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
        *,
        moe_weight_mode: str | None = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float | bool]:
        kwargs = {
            "moe_weight_mode": moe_weight_mode,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "moe_tp_size": moe_tp_size,
            "moe_ep_size": moe_ep_size,
        }
        args = (
            model_name,
            phase,
            max(1, int(tp_size)),
            max(1, int(batch_size)),
            max(1, int(seq_len)),
            max(0, int(seq_len_kv_cache)),
        )
        if hasattr(database, "query_layerwise_detail"):
            try:
                raw_detail = database.query_layerwise_detail(*args, **kwargs)
            except TypeError:
                raw_detail = database.query_layerwise_detail(*args)
        else:
            result = database.query_layerwise(*args)
            raw_detail = {
                "latency": float(result),
                "energy": float(getattr(result, "energy", 0.0)),
                "latency_source": str(getattr(result, "source", "silicon")),
            }
        detail = dict(raw_detail)
        detail.setdefault("latency", 0.0)
        detail.setdefault("energy", 0.0)
        detail.setdefault("rms_latency", 0.0)
        detail.setdefault("includes_moe", False)
        return detail

    def _layerwise_context_detail_for_runtime(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        runtime_config: RuntimeConfig,
        *,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        allow_zero_prefix_chunk_fallback: bool = False,
    ) -> dict[str, float | bool]:
        return self._layerwise_context_layer_detail(
            database,
            model_name,
            tp_size,
            batch_size,
            seq_len,
            prefix,
            max_num_batched_tokens=self._layerwise_context_chunk_size(runtime_config),
            moe_weight_mode=moe_weight_mode,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            allow_zero_prefix_chunk_fallback=allow_zero_prefix_chunk_fallback,
        )

    def _layerwise_context_step_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        *,
        max_num_batched_tokens: int | None = None,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float | bool]:
        return self._layerwise_context_layer_detail(
            database,
            model_name,
            tp_size,
            batch_size,
            seq_len,
            prefix,
            max_num_batched_tokens=max_num_batched_tokens,
            moe_weight_mode=moe_weight_mode,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )

    def _layerwise_detail_scale(self, detail: dict, fallback_num_layers: int) -> float:
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        if raw_multiplier <= 0.0:
            return float(fallback_num_layers)
        measured = max(float(detail.get("measured_layer_count", 1.0) or 1.0), 1.0)
        represented = min(raw_multiplier, float(fallback_num_layers))
        return represented / measured

    def _layerwise_detail_represented_layers(self, detail: dict, fallback_num_layers: int) -> int:
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        if raw_multiplier <= 0.0:
            return int(fallback_num_layers)
        return max(0, round(min(raw_multiplier, float(fallback_num_layers))))

    def _layerwise_detail_represented_moe_layers(self, detail: dict, fallback_num_layers: int) -> int:
        components = detail.get("components")
        if isinstance(components, list):
            total = 0
            for component in components:
                if isinstance(component, dict) and bool(component.get("includes_moe", False)):
                    total += self._layerwise_detail_represented_layers(component, fallback_num_layers)
            return min(total, fallback_num_layers)
        if bool(detail.get("includes_moe", False)):
            return self._layerwise_detail_represented_layers(detail, fallback_num_layers)
        return 0

    def _layerwise_detail_represented_noop_moe_layers(self, detail: dict, fallback_num_layers: int) -> int:
        components = detail.get("components")
        if isinstance(components, list):
            total = 0
            for component in components:
                if isinstance(component, dict) and self._layerwise_detail_moe_weight_mode(component) == "noop":
                    total += self._layerwise_detail_represented_layers(component, fallback_num_layers)
            return min(total, fallback_num_layers)
        if self._layerwise_detail_moe_weight_mode(detail) == "noop" and not bool(detail.get("includes_moe", False)):
            return self._layerwise_detail_represented_layers(detail, fallback_num_layers)
        return 0

    def _layerwise_detail_moe_weight_mode(self, detail: dict) -> str:
        value = detail.get("moe_weight_mode")
        if value not in (None, ""):
            return str(value)
        components = detail.get("components")
        if isinstance(components, list):
            modes = {
                str(component.get("moe_weight_mode") or "") for component in components if isinstance(component, dict)
            }
            if len(modes) == 1:
                return modes.pop()
        return ""

    def _layerwise_scheduler_like_detail(self, detail: dict) -> bool:
        if str(detail.get("latency_source") or "") not in _LAYERWISE_SCHEDULER_LATENCY_SOURCES:
            return False
        max_tokens = detail.get("max_num_batched_tokens")
        seq_len_q = detail.get("seq_len_q")
        if max_tokens not in (None, "") and seq_len_q not in (None, ""):
            try:
                if float(seq_len_q) > float(max_tokens):
                    return False
            except (TypeError, ValueError):
                pass
        return True

    def _layerwise_scheduler_timed_detail(self, detail: dict) -> bool:
        return self._layerwise_scheduler_like_detail(detail)

    def _layerwise_scheduler_envelope_is_full_step(self, detail: dict) -> bool:
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        measured = float(detail.get("measured_layer_count", 1.0) or 1.0)
        return raw_multiplier > 0.0 and measured >= raw_multiplier

    def _layerwise_detail_has_scaled_module_timing(self, detail: dict) -> bool:
        def _component_is_scaled_module(component: dict) -> bool:
            if str(component.get("latency_source") or "") in _LAYERWISE_SCHEDULER_LATENCY_SOURCES:
                return False
            raw_multiplier = float(component.get("layer_multiplier", 0.0) or 0.0)
            measured = max(float(component.get("measured_layer_count", 1.0) or 1.0), 1.0)
            return raw_multiplier > measured

        components = detail.get("components")
        if isinstance(components, list):
            return any(
                _component_is_scaled_module(component) for component in components if isinstance(component, dict)
            )
        return _component_is_scaled_module(detail)

    def _validate_decode_layerwise_detail(self, detail: dict, model_name: str) -> None:
        if not self._layerwise_detail_has_scaled_module_timing(detail):
            return
        raise PerfDataNotAvailableError(
            "vLLM GEN layerwise data for "
            f"{model_name!r} is representative module timing, not a full scheduler step. "
            "Recollect decode rows with latency_source=auto/execute_model_gpu/schedule_to_update/worker_wall."
        )

    def _layerwise_structural_moe_context_ms(self, detail: dict, num_layers: int) -> float | None:
        return None

    def _layerwise_noop_moe_addback_is_bundled(
        self,
        detail: dict,
        fallback_num_layers: int,
        *,
        model_name: str,
        tp_size: int,
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> bool:
        if bool(detail.get("includes_moe", False)):
            return True
        return self._layerwise_detail_moe_weight_mode(detail) not in {"", "noop"}

    def _layerwise_tp_allreduce_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        tp_size: int,
        token_count: int,
        *,
        execution_mode: str | None = None,
        use_fused: bool = False,
    ) -> float:
        tp_size = int(tp_size)
        if tp_size <= 1 or token_count <= 0:
            return 0.0
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        if hidden_size <= 0:
            return 0.0
        quant_mode = getattr(model.config, "comm_quant_mode", None) or common.CommQuantMode.half
        size = max(1, int(token_count) * hidden_size)
        # The fused allreduce_rms kernel is only cheaper than the standalone custom
        # all-reduce for SMALL messages (decode). For large messages (prefill) it is
        # collected as substantially more expensive, so callers opt in explicitly
        # (decode does; context keeps the standalone custom all-reduce).
        if use_fused and _LAYERWISE_USE_FUSED_ALLREDUCE_RMS:
            try:
                fused = database.query_allreduce_rms(quant_mode, tp_size, size, hidden_size)
                if hasattr(fused, "latency"):
                    return float(fused.latency)
                if isinstance(fused, tuple):
                    return float(fused[0])
                return float(fused)
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError, AttributeError):
                pass  # fall back to standalone custom all-reduce
        result = database.query_custom_allreduce(
            quant_mode,
            tp_size,
            size,
            execution_mode=execution_mode,
        )
        return float(result)

    def _layerwise_context_tp_allreduce_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        tp_size: int,
        token_count: int,
        num_layers: int,
        *,
        context_source: str = "",
    ) -> float:
        if tp_size <= 1:
            return 0.0
        return self._layerwise_tp_allreduce_ms(model, database, tp_size, token_count) * 2 * num_layers

    def _layerwise_generation_tp_allreduce_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        tp_size: int,
        token_count: int,
        num_layers: int,
        *,
        layer_detail: dict,
        layer_includes_moe: bool,
        represented_moe_layers: int,
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> float:
        if tp_size <= 1:
            return 0.0
        if self._layerwise_scheduler_envelope_is_full_step(layer_detail):
            # A full-step scheduler envelope already contains the all-reduce ONLY
            # if it was measured on real multi-GPU hardware. When the layerwise data
            # was collected on fewer physical GPUs than tp_size (single-GPU shape
            # sweeps), no real all-reduce happened, so the explicit term must still
            # be added.
            physical_gpus = float(layer_detail.get("physical_gpus", 0.0) or 0.0)
            envelope_has_comm = (not _LAYERWISE_GEN_SINGLE_GPU_COMM) or physical_gpus >= float(tp_size)
            if envelope_has_comm:
                return 0.0
        return self._layerwise_tp_allreduce_ms(model, database, tp_size, token_count, use_fused=True) * num_layers

    def _layerwise_moe_ep_alltoall_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        moe_ep_size: int,
        token_count: int,
        *,
        exchange_count: float = 1.0,
    ) -> float:
        moe_ep_size = int(moe_ep_size)
        if moe_ep_size <= 1 or token_count <= 0:
            return 0.0
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        if hidden_size <= 0:
            return 0.0
        quant_mode = getattr(model.config, "comm_quant_mode", None) or common.CommQuantMode.half
        result = database.query_nccl(
            quant_mode,
            moe_ep_size,
            "alltoall",
            max(1, int(token_count) * hidden_size),
        )
        return float(result) * float(exchange_count)

    def _layerwise_moe_workload_distribution(self, model: BaseModel) -> str:
        distribution = str(getattr(model.config, "workload_distribution", "power_law") or "power_law")
        if distribution == "power_law":
            return f"power_law_{getattr(model, '_power_law_alpha', 1.2)}"
        return distribution

    def _layerwise_moe_distribution_available(
        self,
        model: BaseModel,
        database: PerfDatabase,
        distribution: str,
        *,
        token_count: int,
    ) -> bool:
        try:
            self._layerwise_moe_compute(
                model,
                database,
                token_count=token_count,
                num_layers=1,
                is_context=True,
                workload_distribution_override=distribution,
            )
        except (AttributeError, PerfDataNotAvailableError, ValueError):
            return False
        return True

    def _layerwise_moe_compute_result(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        num_layers: int,
        is_context: bool,
        workload_distribution_override: str | None = None,
    ) -> PerformanceResult:
        token_count = int(token_count)
        num_layers = int(num_layers)
        if token_count <= 0 or num_layers <= 0 or int(getattr(model, "_topk", 0) or 0) <= 0:
            return PerformanceResult(0.0, energy=0.0, source="silicon")
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        inter_size = int(
            getattr(model, "_intermediate_size", 0)
            or getattr(model, "_moe_intermediate_size", 0)
            or getattr(model, "_moe_inter_size", 0)
            or 0
        )
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        if hidden_size <= 0 or inter_size <= 0 or num_experts <= 0:
            return PerformanceResult(0.0, energy=0.0, source="silicon")
        quant_mode = getattr(model.config, "moe_quant_mode", None) or common.MoEQuantMode.bfloat16
        result = database.query_moe(
            num_tokens=token_count,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=int(getattr(model, "_topk", 1) or 1),
            num_experts=num_experts,
            moe_tp_size=int(getattr(model.config, "moe_tp_size", 1) or 1),
            moe_ep_size=int(getattr(model.config, "moe_ep_size", 1) or 1),
            quant_mode=quant_mode,
            workload_distribution=workload_distribution_override or self._layerwise_moe_workload_distribution(model),
            is_context=is_context,
            moe_backend=getattr(model.config, "moe_backend", None),
        )
        if isinstance(result, tuple):
            return PerformanceResult(float(result[0]) * num_layers, energy=0.0, source="sol")
        if not isinstance(result, PerformanceResult):
            return PerformanceResult(float(result) * num_layers, energy=0.0, source="silicon")
        return result * num_layers

    def _layerwise_moe_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        num_layers: int,
        is_context: bool,
        workload_distribution_override: str | None = None,
    ) -> tuple[float, float, str]:
        result = self._layerwise_moe_compute_result(
            model,
            database,
            token_count=token_count,
            num_layers=num_layers,
            is_context=is_context,
            workload_distribution_override=workload_distribution_override,
        )
        return (
            float(result),
            float(getattr(result, "energy", 0.0)),
            str(getattr(result, "source", "silicon")),
        )

    def _layerwise_moe_router_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
    ) -> tuple[float, float, str]:
        token_count = int(token_count)
        num_layers = int(num_layers)
        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        if hidden_size <= 0 or num_experts <= 0:
            return 0.0, 0.0, "silicon"
        quant_mode = getattr(model.config, "gemm_quant_mode", None) or common.GEMMQuantMode.bfloat16
        latency = 0.0
        energy = 0.0
        sources: set[str] = set()
        try:
            result = database.query_gemm(token_count, num_experts, hidden_size, quant_mode)
        except (AttributeError, PerfDataNotAvailableError, ValueError):
            result = None
        if result is not None:
            latency += float(result) * num_layers
            energy += float(getattr(result, "energy", 0.0)) * num_layers
            sources.add(str(getattr(result, "source", "silicon")))
        topk = int(getattr(model, "_topk", 0) or 0)
        if topk > 0:
            try:
                from aiconfigurator.sdk.operations.elementwise import ElementWise

                postprocess = ElementWise(
                    "layerwise_moe_router_postprocess",
                    num_layers,
                    num_experts,
                    topk,
                    0.8,
                ).query(database, x=token_count)
                latency += float(postprocess)
                energy += float(getattr(postprocess, "energy", 0.0))
                sources.add(str(getattr(postprocess, "source", "silicon")))
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                pass
        return latency, energy, sources.pop() if len(sources) == 1 else "mixed" if sources else "silicon"

    def _layerwise_moe_dispatch_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
        *,
        is_context: bool,
    ) -> tuple[float, float, str]:
        token_count = int(token_count)
        num_layers = int(num_layers)
        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        topk = int(getattr(model, "_topk", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        if hidden_size <= 0 or topk <= 0 or num_experts <= 0:
            return 0.0, 0.0, "silicon"

        from aiconfigurator.sdk.operations.moe import MoEDispatch

        latency = 0.0
        energy = 0.0
        sources: set[str] = set()
        try:
            results = []
            for pre_dispatch in (True, False):
                dispatch = MoEDispatch(
                    "layerwise_moe_dispatch",
                    num_layers,
                    hidden_size,
                    topk,
                    num_experts,
                    int(getattr(model.config, "moe_tp_size", 1) or 1),
                    int(getattr(model.config, "moe_ep_size", 1) or 1),
                    int(getattr(model.config, "attention_dp_size", 1) or 1),
                    pre_dispatch,
                    quant_mode=getattr(model.config, "moe_quant_mode", None),
                    moe_backend=getattr(model.config, "moe_backend", None),
                    is_context=is_context,
                )
                results.append(dispatch.query(database, x=token_count))
        except (AssertionError, AttributeError, PerfDataNotAvailableError, ValueError):
            results = []
        latency += sum(float(result) for result in results)
        energy += sum(float(getattr(result, "energy", 0.0)) for result in results)
        sources.update(str(getattr(result, "source", "silicon")) for result in results)
        try:
            from aiconfigurator.sdk.operations.elementwise import ElementWise

            for name, dim_in, dim_out in (
                ("layerwise_moe_dispatch_pack", hidden_size, hidden_size * topk),
                ("layerwise_moe_dispatch_combine", hidden_size * topk, hidden_size),
            ):
                local = ElementWise(name, num_layers, dim_in, dim_out, 0.8).query(database, x=token_count)
                latency += float(local)
                energy += float(getattr(local, "energy", 0.0))
                sources.add(str(getattr(local, "source", "silicon")))
        except (AttributeError, PerfDataNotAvailableError, ValueError):
            pass
        return latency, energy, sources.pop() if len(sources) == 1 else "mixed" if sources else "silicon"

    def _layerwise_moe_shared_expert_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
    ) -> tuple[float, float, str]:
        token_count = int(token_count)
        num_layers = int(num_layers)
        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        shared_inter_size = int(
            getattr(model, "_shared_expert_inter_size", 0)
            or getattr(model, "_moe_shared_expert_intermediate_size", 0)
            or 0
        )
        if hidden_size <= 0 or shared_inter_size <= 0:
            return 0.0, 0.0, "silicon"
        tp_size = max(1, int(getattr(model.config, "tp_size", 1) or 1))
        local_inter_size = max(1, shared_inter_size // tp_size)
        quant_mode = getattr(model.config, "gemm_quant_mode", None) or common.GEMMQuantMode.bfloat16
        try:
            up = database.query_gemm(token_count, local_inter_size, hidden_size, quant_mode)
            gate = database.query_gemm(token_count, local_inter_size, hidden_size, quant_mode)
            down = database.query_gemm(token_count, hidden_size, local_inter_size, quant_mode)
            from aiconfigurator.sdk.operations.elementwise import ElementWise

            activation = ElementWise(
                "layerwise_shared_expert_activation",
                num_layers,
                2 * local_inter_size,
                local_inter_size,
                0.8,
            ).query(database, x=token_count)
        except (AttributeError, PerfDataNotAvailableError, ValueError):
            return 0.0, 0.0, "silicon"
        latency = (float(up) + float(gate) + float(down)) * num_layers + float(activation)
        energy = (
            float(getattr(up, "energy", 0.0))
            + float(getattr(gate, "energy", 0.0))
            + float(getattr(down, "energy", 0.0))
        ) * num_layers + float(getattr(activation, "energy", 0.0))
        sources = {str(getattr(result, "source", "silicon")) for result in (up, gate, down, activation)}
        return latency, energy, sources.pop() if len(sources) == 1 else "mixed"

    def _layerwise_moe_shared_expert_overlap_adjustment_ms(
        self,
        *,
        token_count: int,
        moe_ms: float,
        moe_router_ms: float,
        moe_dispatch_ms: float,
        moe_ep_alltoall_ms: float,
        moe_shared_ms: float,
    ) -> float:
        """Return negative latency for vLLM's overlapped shared-expert stream."""

        if moe_shared_ms <= 0.0:
            return 0.0
        if os.environ.get("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0") not in {"", "0", "false", "False"}:
            return 0.0
        try:
            threshold = int(os.environ.get("VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD", "256"))
        except ValueError:
            threshold = 256
        if int(token_count) > threshold:
            return 0.0
        routed_path_ms = (
            max(0.0, moe_ms) + max(0.0, moe_router_ms) + max(0.0, moe_dispatch_ms) + max(0.0, moe_ep_alltoall_ms)
        )
        if routed_path_ms <= 0.0:
            return 0.0
        return -min(float(moe_shared_ms), routed_path_ms)

    def _layerwise_noop_moe_addback(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        num_layers: int,
        is_context: bool,
        workload_distribution_override: str | None = None,
    ) -> tuple[
        tuple[float, float, str],
        tuple[float, float, str],
        tuple[float, float, str],
        tuple[float, float, str],
        bool,
    ]:
        distribution_candidates = [workload_distribution_override]

        moe = None
        moe_result: PerformanceResult | None = None
        last_error: Exception | None = None
        for distribution in distribution_candidates:
            try:
                moe_result = self._layerwise_moe_compute_result(
                    model,
                    database,
                    token_count=token_count,
                    num_layers=num_layers,
                    is_context=is_context,
                    workload_distribution_override=distribution,
                )
                moe = (
                    float(moe_result),
                    float(getattr(moe_result, "energy", 0.0)),
                    str(getattr(moe_result, "source", "silicon")),
                )
                break
            except (AttributeError, PerfDataNotAvailableError, ValueError) as exc:
                last_error = exc
        if moe is None:
            if last_error is not None:
                raise last_error
            moe = (0.0, 0.0, "silicon")
        moe_metadata = getattr(moe_result, "metadata", {}) if moe_result is not None else {}
        # Module-level fused MoE rows (moe_module_level=True) already contain the
        # router/top-k/gather/scatter work inside the measured kernel, so adding the
        # separate router+dispatch terms double-counts them. This holds for BOTH phases:
        # the guard is kernel-source-aware (keyed on the data the MoE query actually hit),
        # not phase-gated. Decode hits the same fused rows as prefill for hybrid-MoE models
        # (e.g. Qwen3.6), where phase-gating this to context inflated decode ~1.1ms/step
        # (router+dispatch double-count, ~+30pp MAPE vs golden). Lower-level / main MoE
        # data (no moe_module_level flag) still gets the explicit router+dispatch add-back.
        if bool(moe_metadata.get("moe_module_level", False)):
            router = (0.0, 0.0, "silicon")
            dispatch = (0.0, 0.0, "silicon")
        else:
            router = self._layerwise_moe_router_compute(model, database, token_count, num_layers)
            dispatch = self._layerwise_moe_dispatch_compute(
                model,
                database,
                token_count,
                num_layers,
                is_context=is_context,
            )
        if bool(moe_metadata.get("moe_includes_shared_expert", False)):
            shared = (0.0, 0.0, "silicon")
        else:
            shared = self._layerwise_moe_shared_expert_compute(model, database, token_count, num_layers)
        return moe, router, dispatch, shared, False

    def _run_context_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        isl: int,
        prefix: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        if not _USE_LAYERWISE:
            return super()._run_context_phase(model, database, runtime_config, batch_size, isl, prefix)

        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(f"isl must be greater than 0 after removing prefix, but got {effective_isl}")
        return self._get_context_step_latency(
            model,
            database,
            runtime_config,
            ctx_tokens=batch_size * effective_isl,
            ctx_kv_tokens=batch_size * prefix,
            ctx_requests=batch_size,
        )

    def _get_context_step_latency(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        ctx_tokens: int,
        ctx_kv_tokens: int = 0,
        ctx_requests: int = 1,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """Return the latency components for one vLLM context scheduler step."""

        if not _USE_LAYERWISE:
            if ctx_requests <= 0:
                raise ValueError(f"ctx_requests must be positive, got {ctx_requests}")
            if ctx_tokens % ctx_requests != 0:
                raise ValueError(
                    "Base backend context step requires ctx_tokens divisible by ctx_requests, "
                    f"got ctx_tokens={ctx_tokens}, ctx_requests={ctx_requests}"
                )
            prefix = round(ctx_kv_tokens / ctx_requests)
            return super()._run_context_phase(
                model,
                database,
                runtime_config,
                batch_size=ctx_requests,
                isl=(ctx_tokens // ctx_requests) + prefix,
                prefix=prefix,
            )

        ctx_tokens = int(ctx_tokens)
        ctx_requests = int(ctx_requests)
        ctx_kv_tokens = int(ctx_kv_tokens)
        if ctx_requests <= 0:
            raise ValueError(f"ctx_requests must be positive, got {ctx_requests}")
        if ctx_tokens <= 0:
            raise ValueError(f"ctx_tokens must be positive, got {ctx_tokens}")
        if ctx_tokens % ctx_requests != 0:
            raise ValueError(
                "Layerwise context step requires uniform request shapes, "
                f"got ctx_tokens={ctx_tokens}, ctx_requests={ctx_requests}"
            )
        if ctx_kv_tokens % ctx_requests != 0:
            raise ValueError(
                "Layerwise context step requires uniform prefix shapes, "
                f"got ctx_kv_tokens={ctx_kv_tokens}, ctx_requests={ctx_requests}"
            )

        effective_isl = ctx_tokens // ctx_requests
        prefix = ctx_kv_tokens // ctx_requests
        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        context_lookup_tp_size = self._layerwise_context_lookup_tp_size_for_shape(
            model,
            runtime_config,
            tp_size=tp_size,
            effective_isl=effective_isl,
            ctx_requests=ctx_requests,
        )
        context_moe_weight_mode = self._layerwise_context_noop_moe_weight_mode(
            database,
            model_name=str(model_name),
            tp_size=context_lookup_tp_size,
            batch_size=ctx_requests,
            seq_len=effective_isl,
            runtime_config=runtime_config,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        allow_zero_prefix_chunk_fallback = self._layerwise_has_subquadratic_context_attention(model)

        try:
            layer_detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                context_lookup_tp_size,
                ctx_requests,
                effective_isl,
                prefix,
                runtime_config,
                moe_weight_mode=context_moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                allow_zero_prefix_chunk_fallback=allow_zero_prefix_chunk_fallback,
            )
        except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
            if context_lookup_tp_size == tp_size:
                raise
            layer_detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                tp_size,
                ctx_requests,
                effective_isl,
                prefix,
                runtime_config,
                moe_weight_mode=context_moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                allow_zero_prefix_chunk_fallback=allow_zero_prefix_chunk_fallback,
            )

        structural_moe_context_ms = self._layerwise_structural_moe_context_ms(layer_detail, num_layers)
        if structural_moe_context_ms is None:
            layer_ms = float(layer_detail["latency"]) * self._layerwise_detail_scale(layer_detail, num_layers)
        else:
            layer_ms = structural_moe_context_ms
        layer_includes_moe = bool(layer_detail.get("includes_moe", False))
        context_source = str(layer_detail.get("latency_source") or "")
        allreduce_ms = self._layerwise_context_tp_allreduce_ms(
            model,
            database,
            tp_size,
            ctx_tokens,
            num_layers,
            context_source=context_source,
        )

        represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)
        moe_tp_allreduce_ms = 0.0
        if represented_moe_layers > 0 and moe_tp_size > 1:
            try:
                moe_tp_allreduce_ms = (
                    self._layerwise_tp_allreduce_ms(
                        model,
                        database,
                        moe_tp_size,
                        ctx_tokens,
                        execution_mode="eager",
                    )
                    * represented_moe_layers
                )
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no explicit vLLM CTX MoE-TP allreduce for model=%s, moe_tp_size=%s, "
                    "token_count=%s because allreduce data is unavailable",
                    model_name,
                    moe_tp_size,
                    ctx_tokens,
                )

        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        moe_router_ms, moe_router_energy, moe_router_source = 0.0, 0.0, "silicon"
        moe_dispatch_ms, moe_dispatch_energy, moe_dispatch_source = 0.0, 0.0, "silicon"
        moe_shared_ms, moe_shared_energy, moe_shared_source = 0.0, 0.0, "silicon"
        moe_addback_is_bundled = False
        if structural_moe_context_ms is None and not layer_includes_moe:
            represented_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
            if represented_layers > 0:
                context_moe_distribution_override = self._layerwise_context_noop_moe_distribution_override(
                    model,
                    database,
                    token_count=ctx_tokens,
                    runtime_config=runtime_config,
                )
                (
                    (moe_ms, moe_energy, moe_source),
                    (moe_router_ms, moe_router_energy, moe_router_source),
                    (moe_dispatch_ms, moe_dispatch_energy, moe_dispatch_source),
                    (moe_shared_ms, moe_shared_energy, moe_shared_source),
                    moe_addback_is_bundled,
                ) = self._layerwise_noop_moe_addback(
                    model,
                    database,
                    token_count=ctx_tokens,
                    num_layers=represented_layers,
                    is_context=True,
                    workload_distribution_override=context_moe_distribution_override,
                )
                if moe_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                    try:
                        moe_tp_allreduce_ms = (
                            self._layerwise_tp_allreduce_ms(
                                model,
                                database,
                                moe_tp_size,
                                ctx_tokens,
                                execution_mode="eager",
                            )
                            * represented_layers
                        )
                    except (AttributeError, PerfDataNotAvailableError, ValueError):
                        logger.debug(
                            "Falling back to no explicit vLLM CTX no-op MoE-TP allreduce for model=%s, "
                            "moe_tp_size=%s, token_count=%s because allreduce data is unavailable",
                            model_name,
                            moe_tp_size,
                            ctx_tokens,
                        )

        moe_ep_alltoall_layers = represented_moe_layers
        if moe_ep_alltoall_layers <= 0 and moe_ms > 0.0:
            moe_ep_alltoall_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
        moe_ep_alltoall_ms = 0.0
        if moe_ep_alltoall_layers > 0 and moe_ep_size > 1 and not layer_includes_moe and not moe_addback_is_bundled:
            try:
                moe_ep_alltoall_ms = (
                    self._layerwise_moe_ep_alltoall_ms(
                        model,
                        database,
                        moe_ep_size,
                        ctx_tokens,
                        exchange_count=1.0,
                    )
                    * moe_ep_alltoall_layers
                )
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no explicit vLLM CTX MoE-EP all-to-all for model=%s, moe_ep_size=%s, "
                    "token_count=%s because NCCL data is unavailable",
                    model_name,
                    moe_ep_size,
                    ctx_tokens,
                )
        moe_shared_overlap_adjustment_ms = self._layerwise_moe_shared_expert_overlap_adjustment_ms(
            token_count=ctx_tokens,
            moe_ms=moe_ms,
            moe_router_ms=moe_router_ms,
            moe_dispatch_ms=moe_dispatch_ms,
            moe_ep_alltoall_ms=moe_ep_alltoall_ms,
            moe_shared_ms=moe_shared_ms,
        )

        latency_dict = defaultdict(float, {"context_layerwise": layer_ms, "context_tp_allreduce": allreduce_ms})
        energy_dict = defaultdict(float, {"context_layerwise": 0.0, "context_tp_allreduce": 0.0})
        source_dict = {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}
        if moe_tp_allreduce_ms > 0.0:
            latency_dict["context_moe_tp_allreduce"] = moe_tp_allreduce_ms
            energy_dict["context_moe_tp_allreduce"] = 0.0
            source_dict["context_moe_tp_allreduce"] = "silicon"
        if moe_ep_alltoall_ms > 0.0:
            latency_dict["context_moe_ep_alltoall"] = moe_ep_alltoall_ms
            energy_dict["context_moe_ep_alltoall"] = 0.0
            source_dict["context_moe_ep_alltoall"] = "silicon"
        if moe_ms > 0.0:
            latency_dict["context_moe"] = moe_ms
            energy_dict["context_moe"] = moe_energy
            source_dict["context_moe"] = moe_source
        if moe_router_ms > 0.0:
            latency_dict["context_moe_router"] = moe_router_ms
            energy_dict["context_moe_router"] = moe_router_energy
            source_dict["context_moe_router"] = moe_router_source
        if moe_dispatch_ms > 0.0:
            latency_dict["context_moe_dispatch"] = moe_dispatch_ms
            energy_dict["context_moe_dispatch"] = moe_dispatch_energy
            source_dict["context_moe_dispatch"] = moe_dispatch_source
        if moe_shared_ms > 0.0:
            latency_dict["context_moe_shared_expert"] = moe_shared_ms
            energy_dict["context_moe_shared_expert"] = moe_shared_energy
            source_dict["context_moe_shared_expert"] = moe_shared_source
        if moe_shared_overlap_adjustment_ms < 0.0:
            latency_dict["context_moe_shared_expert_overlap"] = moe_shared_overlap_adjustment_ms
            energy_dict["context_moe_shared_expert_overlap"] = 0.0
            source_dict["context_moe_shared_expert_overlap"] = "silicon"
        return latency_dict, energy_dict, source_dict

    def _get_decode_step_latency(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        past_kv: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """Return the latency components for one vLLM decode scheduler step."""

        if not _USE_LAYERWISE:
            return super()._run_generation_phase(
                model,
                database,
                runtime_config,
                batch_size=batch_size,
                beam_width=1,
                isl=past_kv,
                osl=2,
                stride=1,
            )

        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        effective_bs = int(batch_size)
        if effective_bs <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        past_kv = int(past_kv)
        if past_kv <= 0:
            raise ValueError(f"past_kv must be positive, got {past_kv}")

        layer_detail = self._query_layerwise_detail(
            database,
            model_name,
            "GEN",
            tp_size,
            effective_bs,
            past_kv,
            max_num_seqs=getattr(runtime_config, "vllm_max_num_seqs", None),
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        self._validate_decode_layerwise_detail(layer_detail, str(model_name))
        layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
        layer_step_ms = float(layer_detail["latency"]) * layer_scale
        # Calibrate the decode compute's batch-scaling (single-GPU microbenchmark
        # grows too gently with batch vs real paged-attention serving). Comm is
        # modeled separately (fused all-reduce) and is not scaled here. The factor
        # was fit on DENSE decode; it does not transfer to MoE (different decode
        # scaling, and here it would scale only the expert-free backbone), so it is
        # applied to dense models only.
        is_moe_model = int(getattr(model, "_topk", 0) or 0) > 0 and int(getattr(model, "_num_experts", 0) or 0) > 0
        if _DECODE_COMPUTE_BATCH_CAL and not is_moe_model:
            layer_step_ms *= 1.0 + _DECODE_COMPUTE_BATCH_CAL * effective_bs
        layer_includes_moe = bool(layer_detail.get("includes_moe", False))
        represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)

        tp_allreduce_step_ms = self._layerwise_generation_tp_allreduce_ms(
            model,
            database,
            tp_size,
            effective_bs,
            num_layers,
            layer_detail=layer_detail,
            layer_includes_moe=layer_includes_moe,
            represented_moe_layers=represented_moe_layers,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )

        moe_tp_allreduce_step_ms = 0.0
        moe_ep_alltoall_step_ms = 0.0
        if not layer_includes_moe and represented_moe_layers > 0 and moe_ep_size > 1:
            try:
                moe_ep_alltoall_step_ms = (
                    self._layerwise_moe_ep_alltoall_ms(model, database, moe_ep_size, effective_bs)
                    * represented_moe_layers
                )
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, moe_ep_size=%s, "
                    "batch_size=%s because NCCL data is unavailable",
                    model_name,
                    moe_ep_size,
                    effective_bs,
                )
                moe_ep_alltoall_step_ms = 0.0

        moe_step_ms, moe_step_energy, moe_step_source = 0.0, 0.0, "silicon"
        moe_router_step_ms, moe_router_step_energy, moe_router_step_source = 0.0, 0.0, "silicon"
        moe_dispatch_step_ms, moe_dispatch_step_energy, moe_dispatch_step_source = 0.0, 0.0, "silicon"
        moe_shared_step_ms, moe_shared_step_energy, moe_shared_step_source = 0.0, 0.0, "silicon"
        if not layer_includes_moe and not self._layerwise_noop_moe_addback_is_bundled(
            layer_detail,
            num_layers,
            model_name=str(model_name),
            tp_size=tp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        ):
            represented_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
            (
                (moe_step_ms, moe_step_energy, moe_step_source),
                (moe_router_step_ms, moe_router_step_energy, moe_router_step_source),
                (moe_dispatch_step_ms, moe_dispatch_step_energy, moe_dispatch_step_source),
                (moe_shared_step_ms, moe_shared_step_energy, moe_shared_step_source),
                moe_addback_is_bundled,
            ) = self._layerwise_noop_moe_addback(
                model,
                database,
                token_count=effective_bs,
                num_layers=represented_layers,
                is_context=False,
            )
            if moe_step_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                try:
                    moe_tp_allreduce_step_ms = (
                        self._layerwise_tp_allreduce_ms(model, database, moe_tp_size, effective_bs) * represented_layers
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM GEN no-op MoE-TP allreduce for model=%s, "
                        "moe_tp_size=%s, batch_size=%s because allreduce data is unavailable",
                        model_name,
                        moe_tp_size,
                        effective_bs,
                    )
                    moe_tp_allreduce_step_ms = 0.0
            if moe_step_ms > 0.0 and moe_ep_size > 1 and not moe_addback_is_bundled:
                try:
                    moe_ep_alltoall_step_ms = self._layerwise_moe_ep_alltoall_ms(
                        model, database, moe_ep_size, effective_bs
                    ) * self._layerwise_detail_represented_layers(layer_detail, num_layers)
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, "
                        "moe_ep_size=%s, batch_size=%s because NCCL data is unavailable",
                        model_name,
                        moe_ep_size,
                        effective_bs,
                    )
                    moe_ep_alltoall_step_ms = 0.0
        moe_shared_overlap_step_ms = self._layerwise_moe_shared_expert_overlap_adjustment_ms(
            token_count=effective_bs,
            moe_ms=moe_step_ms,
            moe_router_ms=moe_router_step_ms,
            moe_dispatch_ms=moe_dispatch_step_ms,
            moe_ep_alltoall_ms=moe_ep_alltoall_step_ms,
            moe_shared_ms=moe_shared_step_ms,
        )

        latency_dict = defaultdict(float, {"generation_layerwise": max(0.0, layer_step_ms)})
        energy_dict = defaultdict(float, {"generation_layerwise": 0.0})
        source_dict = {"generation_layerwise": "silicon"}
        if tp_allreduce_step_ms > 0.0:
            latency_dict["generation_tp_allreduce"] = tp_allreduce_step_ms
            energy_dict["generation_tp_allreduce"] = 0.0
            source_dict["generation_tp_allreduce"] = "silicon"
        if moe_tp_allreduce_step_ms > 0.0:
            latency_dict["generation_moe_tp_allreduce"] = moe_tp_allreduce_step_ms
            energy_dict["generation_moe_tp_allreduce"] = 0.0
            source_dict["generation_moe_tp_allreduce"] = "silicon"
        if moe_ep_alltoall_step_ms > 0.0:
            latency_dict["generation_moe_ep_alltoall"] = moe_ep_alltoall_step_ms
            energy_dict["generation_moe_ep_alltoall"] = 0.0
            source_dict["generation_moe_ep_alltoall"] = "silicon"
        if moe_step_ms > 0.0:
            latency_dict["generation_moe"] = moe_step_ms
            energy_dict["generation_moe"] = moe_step_energy
            source_dict["generation_moe"] = moe_step_source
        if moe_router_step_ms > 0.0:
            latency_dict["generation_moe_router"] = moe_router_step_ms
            energy_dict["generation_moe_router"] = moe_router_step_energy
            source_dict["generation_moe_router"] = moe_router_step_source
        if moe_dispatch_step_ms > 0.0:
            latency_dict["generation_moe_dispatch"] = moe_dispatch_step_ms
            energy_dict["generation_moe_dispatch"] = moe_dispatch_step_energy
            source_dict["generation_moe_dispatch"] = moe_dispatch_step_source
        if moe_shared_step_ms > 0.0:
            latency_dict["generation_moe_shared_expert"] = moe_shared_step_ms
            energy_dict["generation_moe_shared_expert"] = moe_shared_step_energy
            source_dict["generation_moe_shared_expert"] = moe_shared_step_source
        if moe_shared_overlap_step_ms < 0.0:
            latency_dict["generation_moe_shared_expert_overlap"] = moe_shared_overlap_step_ms
            energy_dict["generation_moe_shared_expert_overlap"] = 0.0
            source_dict["generation_moe_shared_expert_overlap"] = "silicon"
        return latency_dict, energy_dict, source_dict

    def _run_generation_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        stride: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        if not _USE_LAYERWISE:
            return super()._run_generation_phase(
                model, database, runtime_config, batch_size, beam_width, isl, osl, stride
            )

        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        effective_bs = batch_size * beam_width * (model._nextn + 1)

        layer_ms_total = 0.0
        tp_allreduce_total = 0.0
        moe_tp_allreduce_total = 0.0
        moe_ep_alltoall_total = 0.0
        fused_allreduce_rms_total = 0.0
        moe_ms_total = 0.0
        moe_energy_total = 0.0
        moe_source: str | None = None
        moe_router_ms_total = 0.0
        moe_router_energy_total = 0.0
        moe_router_source: str | None = None
        moe_dispatch_ms_total = 0.0
        moe_dispatch_energy_total = 0.0
        moe_dispatch_source: str | None = None
        moe_shared_ms_total = 0.0
        moe_shared_energy_total = 0.0
        moe_shared_source: str | None = None
        moe_shared_overlap_total = 0.0
        for i in range(0, osl - 1, stride):
            kv_len = isl + i
            layer_detail = self._query_layerwise_detail(
                database,
                model_name,
                "GEN",
                tp_size,
                effective_bs,
                kv_len,
                max_num_seqs=getattr(runtime_config, "vllm_max_num_seqs", None),
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
            self._validate_decode_layerwise_detail(layer_detail, str(model_name))
            layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
            layer_step_ms = float(layer_detail["latency"]) * layer_scale
            layer_includes_moe = bool(layer_detail.get("includes_moe", False))
            represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)
            # vLLM layerwise GEN rows measure the transformer block execution
            # envelope. The block already contains RMS work, so adding a
            # separate fused all-reduce+RMS term double-counts decode latency.
            rms_step_ms = 0.0
            allreduce_rms_step_ms = 0.0
            tp_allreduce_step_ms = self._layerwise_generation_tp_allreduce_ms(
                model,
                database,
                tp_size,
                effective_bs,
                num_layers,
                layer_detail=layer_detail,
                layer_includes_moe=layer_includes_moe,
                represented_moe_layers=represented_moe_layers,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
            moe_tp_allreduce_step_ms = 0.0
            # Full-MoE vLLM decode rows already measure the MoE block envelope.
            # Adding another expert-TP collective double-counts Qwen3.6 TP2
            # decode; keep explicit MoE-TP add-back for no-op rows below.
            moe_ep_alltoall_step_ms = 0.0
            if not layer_includes_moe and represented_moe_layers > 0 and moe_ep_size > 1:
                try:
                    moe_ep_alltoall_step_ms = (
                        self._layerwise_moe_ep_alltoall_ms(model, database, moe_ep_size, effective_bs)
                        * represented_moe_layers
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, moe_ep_size=%s, "
                        "batch_size=%s because NCCL data is unavailable",
                        model_name,
                        moe_ep_size,
                        effective_bs,
                    )
                    moe_ep_alltoall_step_ms = 0.0
            repeat_count = min(stride, osl - 1 - i)
            layer_ms_total += max(0.0, layer_step_ms - rms_step_ms) * repeat_count
            tp_allreduce_total += tp_allreduce_step_ms * repeat_count
            moe_tp_allreduce_total += moe_tp_allreduce_step_ms * repeat_count
            fused_allreduce_rms_total += allreduce_rms_step_ms * repeat_count
            moe_step_ms, moe_step_energy, moe_step_source = 0.0, 0.0, "silicon"
            moe_router_step_ms, moe_router_step_energy, moe_router_step_source = 0.0, 0.0, "silicon"
            moe_dispatch_step_ms, moe_dispatch_step_energy, moe_dispatch_step_source = 0.0, 0.0, "silicon"
            moe_shared_step_ms, moe_shared_step_energy, moe_shared_step_source = 0.0, 0.0, "silicon"
            if not layer_includes_moe and not self._layerwise_noop_moe_addback_is_bundled(
                layer_detail,
                num_layers,
                model_name=str(model_name),
                tp_size=tp_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            ):
                represented_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
                (
                    (moe_step_ms, moe_step_energy, moe_step_source),
                    (moe_router_step_ms, moe_router_step_energy, moe_router_step_source),
                    (moe_dispatch_step_ms, moe_dispatch_step_energy, moe_dispatch_step_source),
                    (moe_shared_step_ms, moe_shared_step_energy, moe_shared_step_source),
                    moe_addback_is_bundled,
                ) = self._layerwise_noop_moe_addback(
                    model,
                    database,
                    token_count=effective_bs,
                    num_layers=represented_layers,
                    is_context=False,
                )
                if moe_step_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                    try:
                        moe_tp_allreduce_step_ms = (
                            self._layerwise_tp_allreduce_ms(model, database, moe_tp_size, effective_bs)
                            * represented_layers
                        )
                    except (AttributeError, PerfDataNotAvailableError, ValueError):
                        logger.debug(
                            "Falling back to no explicit vLLM GEN no-op MoE-TP allreduce for model=%s, "
                            "moe_tp_size=%s, batch_size=%s because allreduce data is unavailable",
                            model_name,
                            moe_tp_size,
                            effective_bs,
                        )
                        moe_tp_allreduce_step_ms = 0.0
                if moe_step_ms > 0.0 and moe_ep_size > 1 and not moe_addback_is_bundled:
                    try:
                        moe_ep_alltoall_step_ms = self._layerwise_moe_ep_alltoall_ms(
                            model, database, moe_ep_size, effective_bs
                        ) * self._layerwise_detail_represented_layers(layer_detail, num_layers)
                    except (AttributeError, PerfDataNotAvailableError, ValueError):
                        logger.debug(
                            "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, "
                            "moe_ep_size=%s, batch_size=%s because NCCL data is unavailable",
                            model_name,
                            moe_ep_size,
                            effective_bs,
                        )
                        moe_ep_alltoall_step_ms = 0.0
            moe_shared_overlap_step_ms = self._layerwise_moe_shared_expert_overlap_adjustment_ms(
                token_count=effective_bs,
                moe_ms=moe_step_ms,
                moe_router_ms=moe_router_step_ms,
                moe_dispatch_ms=moe_dispatch_step_ms,
                moe_ep_alltoall_ms=moe_ep_alltoall_step_ms,
                moe_shared_ms=moe_shared_step_ms,
            )
            moe_ep_alltoall_total += moe_ep_alltoall_step_ms * repeat_count
            moe_ms_total += moe_step_ms * repeat_count
            moe_energy_total += moe_step_energy * repeat_count
            moe_router_ms_total += moe_router_step_ms * repeat_count
            moe_router_energy_total += moe_router_step_energy * repeat_count
            moe_dispatch_ms_total += moe_dispatch_step_ms * repeat_count
            moe_dispatch_energy_total += moe_dispatch_step_energy * repeat_count
            moe_shared_ms_total += moe_shared_step_ms * repeat_count
            moe_shared_energy_total += moe_shared_step_energy * repeat_count
            moe_shared_overlap_total += moe_shared_overlap_step_ms * repeat_count
            if moe_source is None or moe_source == moe_step_source:
                moe_source = moe_step_source
            else:
                moe_source = "mixed"
            if moe_router_source is None or moe_router_source == moe_router_step_source:
                moe_router_source = moe_router_step_source
            else:
                moe_router_source = "mixed"
            if moe_dispatch_source is None or moe_dispatch_source == moe_dispatch_step_source:
                moe_dispatch_source = moe_dispatch_step_source
            else:
                moe_dispatch_source = "mixed"
            if moe_shared_source is None or moe_shared_source == moe_shared_step_source:
                moe_shared_source = moe_shared_step_source
            else:
                moe_shared_source = "mixed"

        latency_dict = defaultdict(float, {"generation_layerwise": layer_ms_total})
        energy_dict = defaultdict(float, {"generation_layerwise": 0.0})
        source_dict = {"generation_layerwise": "silicon"}
        if tp_allreduce_total > 0.0:
            latency_dict["generation_tp_allreduce"] = tp_allreduce_total
            energy_dict["generation_tp_allreduce"] = 0.0
            source_dict["generation_tp_allreduce"] = "silicon"
        if moe_tp_allreduce_total > 0.0:
            latency_dict["generation_moe_tp_allreduce"] = moe_tp_allreduce_total
            energy_dict["generation_moe_tp_allreduce"] = 0.0
            source_dict["generation_moe_tp_allreduce"] = "silicon"
        if moe_ep_alltoall_total > 0.0:
            latency_dict["generation_moe_ep_alltoall"] = moe_ep_alltoall_total
            energy_dict["generation_moe_ep_alltoall"] = 0.0
            source_dict["generation_moe_ep_alltoall"] = "silicon"
        if fused_allreduce_rms_total > 0.0:
            latency_dict["generation_tp_allreduce_rms"] = fused_allreduce_rms_total
            energy_dict["generation_tp_allreduce_rms"] = 0.0
            source_dict["generation_tp_allreduce_rms"] = "silicon"
        if moe_ms_total > 0.0:
            latency_dict["generation_moe"] = moe_ms_total
            energy_dict["generation_moe"] = moe_energy_total
            source_dict["generation_moe"] = moe_source or "silicon"
        if moe_router_ms_total > 0.0:
            latency_dict["generation_moe_router"] = moe_router_ms_total
            energy_dict["generation_moe_router"] = moe_router_energy_total
            source_dict["generation_moe_router"] = moe_router_source or "silicon"
        if moe_dispatch_ms_total > 0.0:
            latency_dict["generation_moe_dispatch"] = moe_dispatch_ms_total
            energy_dict["generation_moe_dispatch"] = moe_dispatch_energy_total
            source_dict["generation_moe_dispatch"] = moe_dispatch_source or "silicon"
        if moe_shared_ms_total > 0.0:
            latency_dict["generation_moe_shared_expert"] = moe_shared_ms_total
            energy_dict["generation_moe_shared_expert"] = moe_shared_energy_total
            source_dict["generation_moe_shared_expert"] = moe_shared_source or "silicon"
        if moe_shared_overlap_total < 0.0:
            latency_dict["generation_moe_shared_expert_overlap"] = moe_shared_overlap_total
            energy_dict["generation_moe_shared_expert_overlap"] = 0.0
            source_dict["generation_moe_shared_expert_overlap"] = "silicon"
        return latency_dict, energy_dict, source_dict

    def _get_mix_step_latency(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int,
        ctx_requests: int = 1,
    ) -> tuple[float, float, dict, dict]:
        """Return one mixed vLLM scheduler step from measured layerwise steps.

        Mixed rows are one scheduler iteration, not serialized ctx then decode
        iterations. The context work defines the primary envelope; decode
        contributes only the portion that exceeds that envelope.
        """

        if not _USE_LAYERWISE:
            return super()._get_mix_step_latency(
                model,
                database,
                runtime_config,
                ctx_tokens,
                gen_tokens,
                isl,
                osl,
                prefix,
            )

        ctx_tokens = max(int(ctx_tokens), 0)
        gen_tokens = max(int(gen_tokens), 0)
        ctx_requests = max(int(ctx_requests), 1)
        ctx_prefix = max(int(prefix), 0)
        if ctx_tokens <= 0 and gen_tokens <= 0:
            return 0.0, 0.0, {}, {}

        def _active_source(latencies: dict[str, float], sources: dict[str, str]) -> str:
            active = {sources.get(name, "silicon") for name, value in latencies.items() if float(value) > 0.0}
            if not active:
                return "silicon"
            if len(active) == 1:
                return next(iter(active))
            return "mixed"

        latency_ms = 0.0
        energy_wms = 0.0
        context_total_ms = 0.0
        per_ops: dict[str, float] = {
            "mixed_layerwise_context_combined": 0.0,
            "mixed_layerwise_context_tp_allreduce": 0.0,
            "mixed_layerwise_decode_delta": 0.0,
        }
        per_ops_source: dict[str, str] = dict.fromkeys(per_ops, "silicon")

        if ctx_tokens > 0:
            # FPM mixed rows report aggregate scheduled context tokens. Uniform
            # aggregates have enough request-count information to query the
            # corresponding batched context scheduler envelope. Non-uniform
            # aggregates do not. Some real mixed workload rows are uniform but
            # exceed the batch-specific layerwise context grid. Dense fallback
            # conserves total token work; MoE fallback preserves the scheduler
            # envelope before treating concurrent expert work as serialized.
            is_moe_model = int(getattr(model, "_topk", 0) or 0) > 0 and int(getattr(model, "_num_experts", 0) or 0) > 0
            context_shape_options: list[tuple[int, int, int]] = []

            def _append_context_shape(context_tokens: int, context_prefix_total: int, context_requests: int) -> None:
                option = (
                    max(1, int(context_tokens)),
                    max(0, int(context_prefix_total)),
                    max(1, int(context_requests)),
                )
                if option not in context_shape_options:
                    context_shape_options.append(option)

            if ctx_tokens % ctx_requests == 0:
                _append_context_shape(ctx_tokens, ctx_prefix * ctx_requests, ctx_requests)
            if is_moe_model and ctx_requests > 1:
                avg_ctx_tokens = (ctx_tokens + ctx_requests - 1) // ctx_requests
                _append_context_shape(avg_ctx_tokens, ctx_prefix, 1)
            _append_context_shape(ctx_tokens, ctx_prefix, 1)

            context_error: Exception | None = None
            for context_tokens, context_prefix_total, context_requests in context_shape_options:
                try:
                    context_latency, context_energy, context_sources = self._get_context_step_latency(
                        model,
                        database,
                        runtime_config,
                        ctx_tokens=context_tokens,
                        ctx_kv_tokens=context_prefix_total,
                        ctx_requests=context_requests,
                    )
                    break
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError) as exc:
                    context_error = exc
            else:
                assert context_error is not None
                raise context_error
            context_tp_allreduce_ms = float(context_latency.get("context_tp_allreduce", 0.0))
            context_total_ms = float(sum(context_latency.values()))
            context_combined_ms = max(0.0, context_total_ms - context_tp_allreduce_ms)
            latency_ms += context_total_ms
            energy_wms += float(sum(context_energy.values()))
            per_ops["mixed_layerwise_context_combined"] = context_combined_ms
            per_ops["mixed_layerwise_context_tp_allreduce"] = context_tp_allreduce_ms
            per_ops_source["mixed_layerwise_context_combined"] = _active_source(
                {name: value for name, value in context_latency.items() if name != "context_tp_allreduce"},
                context_sources,
            )
            per_ops_source["mixed_layerwise_context_tp_allreduce"] = context_sources.get(
                "context_tp_allreduce",
                "silicon",
            )

        if gen_tokens > 0:
            avg_decode_kv = max(1, int(isl) + max(int(osl), 0) // 2)
            decode_latency, decode_energy, decode_sources = self._get_decode_step_latency(
                model,
                database,
                runtime_config,
                batch_size=gen_tokens,
                past_kv=avg_decode_kv,
            )
            decode_ms = float(sum(decode_latency.values()))
            decode_delta_ms = decode_ms
            if ctx_tokens > 0:
                # A mixed scheduler step runs prefill + decode as ONE fused forward
                # (verified against vLLM V1: a single flattened batch of
                # prefill_new + decode tokens). Cost structure:
                #   * linear/MLP layers -> ONE GEMM over all tokens, dominated by
                #     the (larger) prefill token count, so already counted in
                #     ``context_total_ms``; the decode MLP increment is negligible
                #     for the small decode batches a step carries.
                #   * attention -> a unified varlen kernel where prefill and decode
                #     attention ADD (per-request work summed). Prefill attention is
                #     in ``context_total_ms``; the decode side contributes only its
                #     KV attention.
                # So the decode contribution is its KV-attention, estimated as the
                # decode step latency above its attention-free floor (same batch,
                # ~zero KV). Unlike the previous ``max(context, decode)`` envelope,
                # this is added even when the prefill dominates -- otherwise decode
                # attention is silently dropped (the dominant mixed-step bias).
                decode_floor_ms = decode_ms
                try:
                    decode_floor_latency, _, _ = self._get_decode_step_latency(
                        model,
                        database,
                        runtime_config,
                        batch_size=gen_tokens,
                        past_kv=1,
                    )
                    decode_floor_ms = float(sum(decode_floor_latency.values()))
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    decode_floor_ms = decode_ms
                decode_delta_ms = max(0.0, decode_ms - decode_floor_ms)
            latency_ms += decode_delta_ms
            energy_wms += float(sum(decode_energy.values()))
            per_ops["mixed_layerwise_decode_delta"] = decode_delta_ms
            per_ops_source["mixed_layerwise_decode_delta"] = _active_source(decode_latency, decode_sources)

        return latency_ms, energy_wms, per_ops, per_ops_source
