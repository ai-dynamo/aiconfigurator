# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np

from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.base_backend import BaseBackend
from aiconfigurator.sdk.backends.trtllm_backend import TRTLLMBackend
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError

logger = logging.getLogger(__name__)

_USE_LAYERWISE = os.environ.get("AIC_VLLM_USE_LAYERWISE", "0") == "1"
_QWEN_MODULE_MOE_DISTRIBUTION = "qwen_module_tp_block"
_QWEN_NOOP_MOE_DISTRIBUTION = "sampled_zipf_0.8"
_LAYERWISE_SCHEDULER_LATENCY_SOURCES = {"schedule_to_update", "worker_wall", "fpm_wall"}


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
    ) -> dict[str, float | bool]:
        if max_num_batched_tokens is None or seq_len <= max_num_batched_tokens:
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
                "moe_weight_mode": chunk_details[0].get("moe_weight_mode", ""),
                "max_num_batched_tokens": float(max_num_batched_tokens),
                "layer_type": chunk_details[0].get("layer_type", ""),
                "layer_index": float(chunk_details[0].get("layer_index", 0.0)),
                "measured_layer_count": float(chunk_details[0].get("measured_layer_count", 1.0)),
                "layer_multiplier": float(chunk_details[0].get("layer_multiplier", 0.0)),
            }
        except (KeyError, PerfDataNotAvailableError, ValueError):
            if "DeepSeek-V4" in str(model_name):
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
                    return {
                        "latency": sum(float(detail["latency"]) for detail in chunk_details),
                        "energy": sum(float(detail.get("energy", 0.0)) for detail in chunk_details),
                        "rms_latency": sum(float(detail.get("rms_latency", 0.0)) for detail in chunk_details),
                        "includes_moe": all(bool(detail.get("includes_moe", False)) for detail in chunk_details),
                        "latency_source": (
                            chunk_details[0].get("latency_source", "")
                            if all(
                                str(detail.get("latency_source", ""))
                                == str(chunk_details[0].get("latency_source", ""))
                                for detail in chunk_details
                            )
                            else "mixed"
                        ),
                        "moe_weight_mode": chunk_details[0].get("moe_weight_mode", ""),
                        "max_num_batched_tokens": float(max_num_batched_tokens),
                        "layer_type": chunk_details[0].get("layer_type", ""),
                        "layer_index": float(chunk_details[0].get("layer_index", 0.0)),
                        "measured_layer_count": float(chunk_details[0].get("measured_layer_count", 1.0)),
                        "layer_multiplier": float(chunk_details[0].get("layer_multiplier", 0.0)),
                    }
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

    def _layerwise_noop_context_continuation_floor_ms(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        runtime_config: RuntimeConfig,
        current_detail: dict,
        current_ms: float,
        num_layers: int,
        *,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
        has_shared_expert: bool = False,
    ) -> float:
        """Floor low-prefix no-op continuation rows with long-prefix scheduler data."""

        chunk_size = self._layerwise_context_chunk_size(runtime_config, current_detail) or 2048
        has_parallel_fallback = current_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
        exact_wide_ep_noop = (
            not has_parallel_fallback
            and not has_shared_expert
            and int(moe_tp_size or 1) <= 1
            and int(moe_ep_size or 1) > max(1, tp_size)
        )
        min_floor_seq_len = 128 if (prefix > 0 or exact_wide_ep_noop) else 512
        current_is_noop = (
            self._layerwise_detail_moe_weight_mode(current_detail) == "noop"
            or has_parallel_fallback
        )
        if (
            seq_len < min_floor_seq_len
            or (prefix <= 0 and tp_size <= 1 and not (has_parallel_fallback and int(moe_ep_size or 1) > 1))
            or current_ms <= 0.0
            or not current_is_noop
            or not self._layerwise_scheduler_like_detail(current_detail)
        ):
            return current_ms

        if (
            (prefix > 0 or seq_len >= chunk_size // 2)
            and seq_len >= max(512, chunk_size // 4)
            and int(moe_ep_size or 1) > 1
            and has_parallel_fallback
        ):
            try:
                floor_detail = self._layerwise_context_step_detail(
                    database,
                    model_name,
                    tp_size,
                    batch_size,
                    chunk_size * 2,
                    0,
                    max_num_batched_tokens=chunk_size,
                    moe_weight_mode=moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
                floor_is_noop = (
                    self._layerwise_detail_moe_weight_mode(floor_detail) == "noop"
                    or floor_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
                )
                if floor_is_noop and self._layerwise_scheduler_timed_detail(floor_detail):
                    attention_dp_size = int(getattr(current_detail, "attention_dp_size", 1) or 1)
                    attention_dp_size = max(1, int((int(moe_tp_size or 1) * int(moe_ep_size or 1)) / max(1, tp_size)))
                    floor_fraction = min(0.60, 0.40 + 0.05 * math.log2(float(attention_dp_size)))
                    floor_ms = float(floor_detail["latency"]) * self._layerwise_detail_scale(
                        floor_detail,
                        num_layers,
                    )
                    current_ms = max(current_ms, floor_ms * floor_fraction)
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            exact_wide_ep_noop
            and (
                chunk_size // 4 <= seq_len <= chunk_size
                or (
                    prefix <= 0
                    and tp_size <= 2
                    and int(moe_ep_size or 1) >= 8
                    and 128 <= seq_len < chunk_size // 4
                )
            )
        ):
            try:
                floor_tp_size = int(float(current_detail.get("_context_lookup_tp_size", tp_size) or tp_size))
                floor_detail = self._layerwise_context_step_detail(
                    database,
                    model_name,
                    floor_tp_size,
                    batch_size,
                    chunk_size * 2,
                    0,
                    max_num_batched_tokens=chunk_size,
                    moe_weight_mode=moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
                floor_is_noop = self._layerwise_detail_moe_weight_mode(floor_detail) == "noop"
                if floor_is_noop and self._layerwise_scheduler_timed_detail(floor_detail):
                    attention_dp_size = max(1, int((int(moe_tp_size or 1) * int(moe_ep_size or 1)) / max(1, tp_size)))
                    if prefix <= 0 and seq_len < chunk_size // 4:
                        floor_fraction = min(0.38, 0.30 + 0.04 * math.log2(float(attention_dp_size)))
                    else:
                        floor_fraction = min(0.65, 0.55 + 0.05 * math.log2(float(attention_dp_size)))
                    floor_ms = float(floor_detail["latency"]) * self._layerwise_detail_scale(
                        floor_detail,
                        num_layers,
                    )
                    current_ms = max(current_ms, floor_ms * floor_fraction)
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                pass
        floor_prefix = max(prefix, chunk_size * 2)
        if floor_prefix == prefix:
            return current_ms

        try:
            floor_detail = self._layerwise_context_step_detail(
                database,
                model_name,
                tp_size,
                batch_size,
                seq_len,
                floor_prefix,
                moe_weight_mode=moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
        except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
            return current_ms

        floor_is_noop = (
            self._layerwise_detail_moe_weight_mode(floor_detail) == "noop"
            or floor_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
        )
        if not floor_is_noop or not self._layerwise_scheduler_like_detail(floor_detail):
            return current_ms

        floor_ms = float(floor_detail["latency"]) * self._layerwise_detail_scale(floor_detail, num_layers)
        if floor_ms <= current_ms * 1.25:
            return current_ms
        floor_fraction = 0.9
        if prefix <= 0 and int(moe_tp_size or 1) > 1 and int(moe_ep_size or 1) == 1:
            floor_fraction = 0.78
        return max(current_ms, floor_ms * floor_fraction)

    def _apply_deepseek_high_ep_noop_context_floor(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        latency_dict: defaultdict[str, float],
        layer_detail: dict,
        *,
        ctx_tokens: int,
        ctx_kv_tokens: int,
        ctx_requests: int,
        layer_includes_moe: bool,
    ) -> None:
        """Floor small high-EP DeepSeek no-op context rows from the same layerwise surface."""

        if getattr(self, "_layerwise_deepseek_context_floor_active", False):
            return
        model_name = str(getattr(model, "model_path", ""))
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        if (
            "DeepSeek-V4" not in model_name
            or ctx_requests != 1
            or ctx_kv_tokens != 0
            or ctx_tokens >= 4096
            or layer_includes_moe
            or self._layerwise_detail_moe_weight_mode(layer_detail) != "noop"
            or not self._layerwise_scheduler_like_detail(layer_detail)
        ):
            return

        tp_size = int(getattr(model.config, "tp_size", 1) or 1)
        if moe_ep_size == 1 and ctx_tokens >= 1024 and tp_size >= 4:
            floor_tokens = 4096
            floor_fraction = 0.75 - (0.4 / max(1.0, float(tp_size)))
        elif moe_ep_size < 4:
            return
        elif ctx_tokens < 1024:
            if moe_ep_size < 8:
                return
            floor_tokens, floor_fraction = 1024, 0.85
        elif moe_ep_size >= 8:
            floor_tokens, floor_fraction = 4096, 1.05
        else:
            floor_tokens, floor_fraction = 4096, 0.75

        try:
            self._layerwise_deepseek_context_floor_active = True
            floor_latency, _, _ = self._get_context_step_latency(
                model,
                database,
                runtime_config,
                ctx_tokens=floor_tokens,
                ctx_kv_tokens=0,
                ctx_requests=1,
            )
        except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
            return
        finally:
            self._layerwise_deepseek_context_floor_active = False

        current_ms = float(sum(latency_dict.values()))
        floor_ms = float(sum(floor_latency.values())) * floor_fraction
        if floor_ms <= current_ms:
            return

        latency_dict["context_layerwise"] += floor_ms - current_ms

    def _deepseek_context_moe_weight_mode(
        self,
        *,
        model_name: str,
        seq_len: int,
        runtime_config: RuntimeConfig,
    ) -> str | None:
        """Return an alternate DeepSeek context row mode for chunked prefill."""

        if "DeepSeek-V4" not in str(model_name):
            return None
        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        if chunk_size is None or int(seq_len) <= chunk_size:
            return None
        return "noop"

    def _deepseek_context_moe_distribution_override(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        runtime_config: RuntimeConfig,
    ) -> str | None:
        """Return a DeepSeek context-only MoE distribution for chunked prefill."""

        if "DeepSeek-V4" not in str(getattr(model, "model_path", "")):
            return None
        if str(getattr(model.config, "workload_distribution", "power_law")) != "power_law":
            return None
        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        if chunk_size is None or int(token_count) <= chunk_size:
            return None

        query_token_count = min(int(token_count), chunk_size) * int(
            getattr(model.config, "attention_dp_size", 1) or 1
        )
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
        """Return the preferred layerwise context TP surface for this runtime."""

        attention_dp_size = int(getattr(model.config, "attention_dp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        if (
            attention_dp_size > 1
            and moe_ep_size > 1
            and self._layerwise_has_subquadratic_context_attention(model)
        ):
            return max(int(tp_size), int(tp_size) * attention_dp_size)
        return int(tp_size)

    def _layerwise_context_detail_for_runtime(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        runtime_config: RuntimeConfig,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float | bool]:
        """Return context layer detail with the runtime chunking policy."""

        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        try:
            detail = self._layerwise_context_layer_detail(
                database,
                model_name,
                tp_size,
                batch_size,
                seq_len,
                prefix,
                max_num_batched_tokens=chunk_size,
                moe_weight_mode=moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
        except (KeyError, PerfDataNotAvailableError, ValueError):
            if chunk_size is not None:
                try:
                    detail = self._layerwise_context_layer_detail(
                        database,
                        model_name,
                        tp_size,
                        batch_size,
                        seq_len,
                        prefix,
                        moe_weight_mode=moe_weight_mode,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                except (KeyError, PerfDataNotAvailableError, ValueError):
                    if moe_weight_mode is None:
                        raise
                    detail = self._layerwise_context_layer_detail(
                        database,
                        model_name,
                        tp_size,
                        batch_size,
                        seq_len,
                        prefix,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                return detail
            if moe_weight_mode is None:
                raise
            detail = self._layerwise_context_layer_detail(
                database,
                model_name,
                tp_size,
                batch_size,
                seq_len,
                prefix,
                max_num_batched_tokens=chunk_size,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
        inferred_chunk_size = self._layerwise_context_chunk_size(runtime_config, detail)
        if chunk_size is None and inferred_chunk_size is not None and seq_len > inferred_chunk_size:
            try:
                detail = self._layerwise_context_layer_detail(
                    database,
                    model_name,
                    tp_size,
                    batch_size,
                    seq_len,
                    prefix,
                    max_num_batched_tokens=inferred_chunk_size,
                    moe_weight_mode=moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
            except (KeyError, PerfDataNotAvailableError, ValueError):
                if moe_weight_mode is None:
                    raise
                detail = self._layerwise_context_layer_detail(
                    database,
                    model_name,
                    tp_size,
                    batch_size,
                    seq_len,
                    prefix,
                    max_num_batched_tokens=inferred_chunk_size,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
        return detail

    def _layerwise_context_step_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        max_num_batched_tokens: int | None = None,
        moe_weight_mode: str | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float | bool]:
        """Return one context scheduler-step row without composing chunks."""

        try:
            return self._query_layerwise_detail(
                database,
                model_name,
                "CTX",
                tp_size,
                batch_size,
                seq_len,
                seq_len_kv_cache=max(prefix, 0),
                max_num_batched_tokens=max_num_batched_tokens,
                moe_weight_mode=moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
        except (KeyError, PerfDataNotAvailableError, ValueError):
            if moe_weight_mode is None:
                raise
            return self._query_layerwise_detail(
                database,
                model_name,
                "CTX",
                tp_size,
                batch_size,
                seq_len,
                seq_len_kv_cache=max(prefix, 0),
                max_num_batched_tokens=max_num_batched_tokens,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )

    def _layerwise_tp_allreduce_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        tp_size: int,
        token_count: int,
        execution_mode: str | None = None,
    ) -> float:
        if tp_size <= 1 or token_count <= 0:
            return 0.0
        hidden_size = int(getattr(model, "_hidden_size", 0))
        if hidden_size <= 0:
            return 0.0
        query_kwargs = {}
        if execution_mode is not None:
            query_kwargs["execution_mode"] = execution_mode
        try:
            return float(
                database.query_custom_allreduce(
                    common.CommQuantMode.half,
                    tp_size,
                    token_count * hidden_size,
                    **query_kwargs,
                )
            )
        except TypeError:
            return float(database.query_custom_allreduce(common.CommQuantMode.half, tp_size, token_count * hidden_size))

    def _query_layerwise_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        phase: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        seq_len_kv_cache: int = 0,
        moe_weight_mode: str | None = None,
        max_num_batched_tokens: int | None = None,
        moe_tp_size: int | None = None,
        moe_ep_size: int | None = None,
    ) -> dict[str, float]:
        if hasattr(database, "query_layerwise_detail"):
            try:
                return database.query_layerwise_detail(
                    model_name,
                    phase,
                    tp_size,
                    batch_size,
                    seq_len,
                    seq_len_kv_cache,
                    moe_weight_mode=moe_weight_mode,
                    max_num_batched_tokens=max_num_batched_tokens,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
            except TypeError:
                return database.query_layerwise_detail(
                    model_name,
                    phase,
                    tp_size,
                    batch_size,
                    seq_len,
                    seq_len_kv_cache,
                )
        latency = database.query_layerwise(model_name, phase, tp_size, batch_size, seq_len, seq_len_kv_cache)
        return {
            "latency": float(latency),
            "energy": 0.0,
            "rms_latency": 0.0,
        }

    def _layerwise_detail_scale(self, detail: dict, fallback_num_layers: int) -> float:
        """Scale a measured representative row to this model stage."""
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        if raw_multiplier <= 0.0:
            return float(fallback_num_layers)
        measured = max(float(detail.get("measured_layer_count", 1.0) or 1.0), 1.0)
        represented = min(raw_multiplier, float(fallback_num_layers))
        return represented / measured

    def _layerwise_detail_represented_layers(self, detail: dict, fallback_num_layers: int) -> int:
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        if raw_multiplier <= 0.0:
            return fallback_num_layers
        return max(0, round(min(raw_multiplier, float(fallback_num_layers))))

    def _layerwise_detail_represented_moe_layers(self, detail: dict, fallback_num_layers: int) -> int:
        """Return the number of MoE layers represented by a layerwise detail row."""

        components = detail.get("components")
        if isinstance(components, list):
            total = 0
            for component in components:
                if isinstance(component, dict) and bool(component.get("includes_moe", False)):
                    total += self._layerwise_detail_represented_layers(component, fallback_num_layers)
            if total > 0:
                return min(total, fallback_num_layers)
        if bool(detail.get("includes_moe", False)):
            return self._layerwise_detail_represented_layers(detail, fallback_num_layers)
        return 0

    def _layerwise_detail_represented_noop_moe_layers(self, detail: dict, fallback_num_layers: int) -> int:
        """Return represented MoE layer count for no-op MoE layerwise rows."""

        components = detail.get("components")
        if isinstance(components, list):
            total = sum(
                self._layerwise_detail_represented_layers(component, fallback_num_layers)
                for component in components
                if isinstance(component, dict)
            )
            if total > 0:
                return min(total, fallback_num_layers)
        return self._layerwise_detail_represented_layers(detail, fallback_num_layers)

    def _layerwise_detail_moe_weight_mode(self, detail: dict) -> str:
        """Return the row's MoE weight mode, preserving interpolated components."""

        direct_mode = str(detail.get("moe_weight_mode") or "")
        if direct_mode:
            return direct_mode

        components = detail.get("components")
        if not isinstance(components, list):
            return ""
        component_modes = {
            str(component.get("moe_weight_mode") or "")
            for component in components
            if isinstance(component, dict) and component.get("moe_weight_mode") not in (None, "")
        }
        if len(component_modes) == 1:
            return next(iter(component_modes))
        return ""

    def _layerwise_noop_moe_addback_is_bundled(
        self,
        detail: dict,
        fallback_num_layers: int,
        *,
        model_name: str = "",
        tp_size: int = 1,
        moe_tp_size: int = 1,
        moe_ep_size: int = 1,
    ) -> bool:
        """Full-depth scheduler envelopes already represent the decode step."""

        if bool(detail.get("includes_moe", False)):
            return True
        if detail.get("parallel_fallback_moe_ep_size") not in (None, ""):
            return False
        if (
            self._layerwise_detail_moe_weight_mode(detail) == "noop"
            and "Qwen" in str(model_name)
            and int(moe_tp_size or 1) <= 1
            and int(moe_ep_size or 1) > max(1, int(tp_size or 1))
        ):
            return False
        if (
            self._layerwise_detail_moe_weight_mode(detail) == "noop"
            and "DeepSeek-V4" in str(model_name)
            and int(moe_tp_size or 1) <= 1
        ):
            return False
        return self._layerwise_scheduler_envelope_is_full_step(detail)

    def _layerwise_scheduler_like_detail(self, detail: dict) -> bool:
        if self._layerwise_detail_exceeds_scheduler_budget(detail):
            return False
        return self._layerwise_scheduler_timed_detail(detail)

    def _layerwise_scheduler_timed_detail(self, detail: dict) -> bool:
        """Return whether a detail comes from scheduler/wall timing provenance."""

        sources: set[str] = set()
        source = str(detail.get("latency_source") or "")
        if source:
            sources.add(source)
        components = detail.get("components")
        if isinstance(components, list):
            for component in components:
                if isinstance(component, dict) and component.get("latency_source") not in (None, ""):
                    sources.add(str(component["latency_source"]))
        if bool(sources) and sources.issubset(_LAYERWISE_SCHEDULER_LATENCY_SOURCES):
            return True
        return detail.get("max_num_batched_tokens") not in (None, "")

    def _layerwise_detail_exceeds_scheduler_budget(self, detail: dict) -> bool:
        """Return whether a context detail is composed from multiple scheduler steps."""

        candidates = [detail]
        components = detail.get("components")
        if isinstance(components, list):
            candidates.extend(component for component in components if isinstance(component, dict))
        for candidate in candidates:
            raw_budget = candidate.get("max_num_batched_tokens")
            raw_seq_len = candidate.get("seq_len_q", candidate.get("query_seq_len_q"))
            if raw_budget in (None, "") or raw_seq_len in (None, ""):
                continue
            try:
                if int(float(raw_seq_len)) > int(float(raw_budget)):
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _layerwise_scheduler_envelope_is_full_step(self, detail: dict) -> bool:
        """Return whether a scheduler/wall row should be used as a whole-step measurement."""

        if not self._layerwise_scheduler_like_detail(detail):
            return False
        raw_multiplier = float(detail.get("layer_multiplier", 0.0) or 0.0)
        measured = float(detail.get("measured_layer_count", 1.0) or 1.0)
        return raw_multiplier > 0.0 and measured >= raw_multiplier

    def _layerwise_structural_moe_context_ms(self, detail: dict, num_layers: int) -> float | None:
        """Estimate MoE context from no-op baseline plus measured MoE delta."""

        components = detail.get("components")
        if not isinstance(components, list) or len(components) < 2:
            return None

        def _has_scale_metadata(component: dict) -> bool:
            has_measured_count = component.get("measured_layer_count") not in (None, "")
            has_multiplier = component.get("layer_multiplier") not in (None, "")
            return has_measured_count and has_multiplier

        def _scaled(component: dict) -> float:
            return float(component.get("latency", 0.0) or 0.0) * self._layerwise_detail_scale(component, num_layers)

        def _latency_source(component: dict) -> str:
            return str(component.get("latency_source") or "")

        def _pick_source_components(
            candidates: list[dict],
            preferred_sources: tuple[str, ...],
            *,
            require_source_match: bool = True,
        ) -> list[dict]:
            """Choose one latency-source family from a component set."""

            if not candidates:
                return []
            available_sources = {_latency_source(component) for component in candidates}
            for source in preferred_sources:
                if source not in available_sources:
                    continue
                picked = [component for component in candidates if _latency_source(component) == source]
                if picked or not require_source_match:
                    return picked
            if available_sources == {""}:
                return candidates
            return []

        baseline_components = []
        moe_components = []
        noop_delta_components = []
        for component in components:
            if not isinstance(component, dict) or not _has_scale_metadata(component):
                continue
            if bool(component.get("includes_moe", False)):
                moe_components.append(component)
                continue
            represented_layers = self._layerwise_detail_represented_layers(component, num_layers)
            scale = self._layerwise_detail_scale(component, num_layers)
            if represented_layers >= num_layers and scale <= 1.0:
                baseline_components.append(component)
            else:
                noop_delta_components.append(component)

        if not baseline_components or not moe_components:
            return None

        baseline_components = _pick_source_components(
            baseline_components,
            ("schedule_to_update", "fpm_wall", "span", "gpu", ""),
        )
        moe_components = _pick_source_components(
            moe_components,
            ("critical_path", "gpu", "span", ""),
        )
        if not moe_components:
            return None
        delta_source = _latency_source(moe_components[0])
        if delta_source:
            noop_delta_components = [
                component for component in noop_delta_components
                if _latency_source(component) in {delta_source, ""}
            ]
            noop_delta_components = _pick_source_components(
                noop_delta_components,
                (delta_source, ""),
            )
        else:
            # Legacy layerwise rows did not always carry latency_source. If the
            # matching no-op representative rows all have one source family, use
            # that family instead of abandoning the structural MoE path.
            noop_sources = {
                _latency_source(component)
                for component in noop_delta_components
                if _latency_source(component)
            }
            if len(noop_sources) == 1:
                inferred_source = next(iter(noop_sources))
                noop_delta_components = _pick_source_components(
                    noop_delta_components,
                    (inferred_source, ""),
                )
            else:
                noop_delta_components = _pick_source_components(noop_delta_components, ("",))
        if not baseline_components or not noop_delta_components:
            return None

        baseline_ms = sum(_scaled(component) for component in baseline_components)
        moe_ms = sum(_scaled(component) for component in moe_components)
        noop_delta_ms = sum(_scaled(component) for component in noop_delta_components)
        return baseline_ms + max(0.0, moe_ms - noop_delta_ms)

    def _layerwise_tp_allreduce_rms_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        tp_size: int,
        token_count: int,
    ) -> float:
        if tp_size <= 1 or token_count <= 0:
            return 0.0
        hidden_size = int(getattr(model, "_hidden_size", 0))
        if hidden_size <= 0:
            return 0.0
        return float(
            database.query_allreduce_rms(
                common.CommQuantMode.half,
                tp_size,
                token_count * hidden_size,
                hidden_size,
            )
        )

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
        """Return dense TP allreduce for a vLLM context scheduler step."""

        standalone_ms = (
            self._layerwise_tp_allreduce_ms(
                model,
                database,
                tp_size,
                token_count,
                execution_mode="eager",
            )
            * 2
            * num_layers
        )
        if standalone_ms <= 0.0 or context_source not in {"schedule_to_update", "fpm_wall", "worker_wall"}:
            return standalone_ms

        extra_params = getattr(model, "extra_params", None)
        is_moe_model = (
            int(getattr(model, "_topk", 0) or 0) > 0
            or int(getattr(extra_params, "num_experts", 0) or getattr(model, "_num_experts", 0) or 0) > 0
        )
        if not is_moe_model and tp_size >= 8 and token_count >= 4096:
            return standalone_ms * 0.70
        if not is_moe_model and 512 <= token_count <= 1024 and tp_size <= 4:
            return standalone_ms

        try:
            fused_rms_ms = self._layerwise_tp_allreduce_rms_ms(
                model,
                database,
                tp_size,
                token_count,
            ) * num_layers
        except (AttributeError, KeyError, PerfDataNotAvailableError, ValueError):
            fused_rms_ms = 0.0

        if 0.0 < fused_rms_ms < standalone_ms and token_count <= 128 and not is_moe_model:
            return min(standalone_ms, math.sqrt(standalone_ms * fused_rms_ms))
        return standalone_ms

    def _layerwise_dense_tp1_context_scheduler_overhead_ms(
        self,
        model: BaseModel,
        layer_detail: dict,
        *,
        tp_size: int,
        ctx_tokens: int,
        ctx_kv_tokens: int,
        ctx_requests: int,
        layer_ms: float,
        layer_includes_moe: bool,
    ) -> float:
        """Return small dense TP1 scheduler residual for medium prefill."""

        if (
            tp_size != 1
            or ctx_requests != 1
            or ctx_kv_tokens != 0
            or ctx_tokens < 512
            or ctx_tokens > 1024
            or layer_includes_moe
            or not self._layerwise_scheduler_envelope_is_full_step(layer_detail)
            or self._layerwise_detail_moe_weight_mode(layer_detail) not in {"", "dense"}
        ):
            return 0.0
        extra_params = getattr(model, "extra_params", None)
        is_moe_model = (
            int(getattr(model, "_topk", 0) or 0) > 0
            or int(getattr(extra_params, "num_experts", 0) or getattr(model, "_num_experts", 0) or 0) > 0
        )
        if is_moe_model:
            return 0.0
        return min(2.0, max(0.0, float(layer_ms) * 0.05))

    @staticmethod
    def _layerwise_dense_mixed_decode_tail_slices(
        *,
        tp_size: int,
        ctx_tokens: int,
        gen_tokens: int,
        avg_decode_kv: int,
        context_chunk_size: int | None,
    ) -> float:
        """Return the exposed decode tail for dense scheduler-envelope mixed rows."""

        if gen_tokens <= 0:
            return 0.0
        chunk_size = max(1, int(context_chunk_size or 2048))
        high_decode_tail = gen_tokens > 8 and avg_decode_kv >= int(chunk_size * 1.45)
        if high_decode_tail:
            if tp_size <= 1:
                return 1.0
            if tp_size == 2:
                if ctx_tokens < max(1, chunk_size // 10):
                    return 0.47
                if ctx_tokens < chunk_size // 4:
                    return 0.315
                if ctx_tokens < chunk_size // 2:
                    return 0.45
                if ctx_tokens <= chunk_size:
                    return 0.55
            if tp_size == 4:
                if ctx_tokens <= chunk_size:
                    return min(1.24, 0.80 * (0.50 + 1.40 * ctx_tokens / float(chunk_size // 2 or 1)))
                if ctx_tokens < chunk_size * 2:
                    return 0.84
            if tp_size >= 8 and ctx_tokens < chunk_size * 2:
                small_context = max(1, chunk_size // 4)
                bump_start = int(chunk_size * 0.20)
                if ctx_tokens <= chunk_size // 16:
                    return 0.75
                if chunk_size // 8 <= ctx_tokens < int(chunk_size * 0.1875):
                    return 1.20
                if ctx_tokens < bump_start:
                    return 1.08
                if ctx_tokens <= small_context:
                    return max(
                        0.60,
                        1.08 - 0.48 * (ctx_tokens - bump_start) / float(max(1, small_context - bump_start)),
                    )
                if ctx_tokens < chunk_size:
                    if ctx_tokens < int(chunk_size * 0.375):
                        return 1.42
                    return min(2.05, 0.75 + 1.30 * (ctx_tokens - small_context) / float(small_context))
                return 1.74
        if tp_size == 2 and ctx_tokens <= chunk_size // 2:
            return 0.5 if gen_tokens > 8 else 0.25
        if tp_size == 4 and gen_tokens > 8 and ctx_tokens > chunk_size * 2:
            return 1.34 if high_decode_tail else 0.8
        if tp_size >= 8:
            if gen_tokens > 8:
                if high_decode_tail and ctx_tokens >= chunk_size * 2:
                    return 1.70 if ctx_tokens < int(chunk_size * 2.5) else 1.95
                if ctx_tokens < chunk_size * 2:
                    return 1.8
            if ctx_tokens >= int(chunk_size * 2.5):
                return 1.15 if gen_tokens > 8 else 1.0
        return 0.0

    @staticmethod
    def _layerwise_dense_mixed_context_envelope_multiplier(
        *,
        tp_size: int,
        ctx_tokens: int,
        gen_tokens: int,
        avg_decode_kv: int,
        context_chunk_size: int | None,
    ) -> float:
        """Return scheduler-envelope lift for dense TP1 mixed transition bands."""

        if tp_size > 1 or gen_tokens <= 8:
            return 1.0
        chunk_size = max(1, int(context_chunk_size or 2048))
        if avg_decode_kv < int(chunk_size * 1.45):
            return 1.0
        if chunk_size // 8 <= ctx_tokens < chunk_size // 4:
            if ctx_tokens < int(chunk_size * 0.22):
                return 1.32
            return 1.20
        if int(chunk_size * 0.375) <= ctx_tokens < int(chunk_size * 0.44):
            return 1.10
        return 1.0

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
        """Return explicit vLLM decode TP collectives for one scheduler step."""

        if tp_size <= 1 or token_count <= 0 or num_layers <= 0:
            return 0.0

        layer_is_full_step_envelope = self._layerwise_scheduler_envelope_is_full_step(layer_detail)
        physical_gpus = int(float(layer_detail.get("physical_gpus", 1.0) or 1.0))
        full_step_comm_is_bundled = layer_is_full_step_envelope and physical_gpus > 1
        if full_step_comm_is_bundled:
            return 0.0
        extra_params = getattr(model, "extra_params", None)
        is_moe_model = (
            int(getattr(model, "_topk", 0) or 0) > 0
            or int(getattr(extra_params, "num_experts", 0) or getattr(model, "_num_experts", 0) or 0) > 0
            or moe_tp_size > 1
            or moe_ep_size > 1
        )
        if layer_is_full_step_envelope and not is_moe_model:
            # Dense schedule_to_update/worker_wall GEN rows measure a whole
            # CUDA-graph decode scheduler step.  Standalone allreduce tables
            # are pessimistic here because vLLM overlaps/fuses the collectives
            # inside the captured step; adding them again over-corrects TP4/TP8
            # dense decode.
            return 0.0

        try:
            standalone_allreduce_ms = self._layerwise_tp_allreduce_ms(model, database, tp_size, token_count)
            try:
                fused_allreduce_ms = self._layerwise_tp_allreduce_rms_ms(model, database, tp_size, token_count)
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                # vLLM decode has two TP collectives per transformer block.
                # When fused allreduce+RMS data is available, use it for the
                # residual/RMS collective; otherwise preserve the legacy
                # two-standalone-allreduce estimate.
                fused_allreduce_ms = standalone_allreduce_ms
            if layer_includes_moe and moe_tp_size <= 1 and moe_ep_size > 1:
                # Pure expert parallel MoE does not tensor-parallelize the
                # expert MLP, so only the attention TP collective is exposed.
                # In vLLM the attention output allreduce is immediately
                # followed by the post-attention RMSNorm, which can use the
                # fused allreduce+RMS path.
                return fused_allreduce_ms * num_layers
            decode_allreduce_ms = (standalone_allreduce_ms + fused_allreduce_ms) * num_layers
            return decode_allreduce_ms
        except (AttributeError, PerfDataNotAvailableError, ValueError):
            logger.debug(
                "Falling back to no explicit vLLM GEN TP allreduce for model=%s, tp_size=%s, "
                "batch_size=%s because allreduce data is unavailable",
                getattr(model, "model_path", ""),
                tp_size,
                token_count,
            )
            return 0.0

    def _layerwise_moe_ep_alltoall_ms(
        self,
        model: BaseModel,
        database: PerfDatabase,
        ep_size: int,
        token_count: int,
        *,
        exchange_count: float = 2.0,
    ) -> float:
        """Return vLLM MoE EP communication latency for one represented layer."""

        if ep_size <= 1 or token_count <= 0:
            return 0.0
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        topk = int(getattr(model, "_topk", 0) or 0)
        if hidden_size <= 0 or topk <= 0:
            return 0.0

        attention_dp_size = int(getattr(model.config, "attention_dp_size", 1) or 1)
        message_size = max(1, int(token_count * attention_dp_size * hidden_size * topk / ep_size))
        return float(
            database.query_nccl(
                common.CommQuantMode.half,
                ep_size,
                "alltoall",
                message_size,
            )
        ) * exchange_count

    def _layerwise_moe_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
        is_context: bool,
        workload_distribution_override: str | None = None,
    ) -> tuple[float, float, str]:
        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"

        topk = int(getattr(model, "_topk", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        inter_size = int(getattr(model, "_moe_inter_size", 0) or 0)
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        if topk <= 0 or num_experts <= 0 or inter_size <= 0 or hidden_size <= 0:
            return 0.0, 0.0, "silicon"

        cfg = model.config
        query_token_count = int(token_count) * int(getattr(cfg, "attention_dp_size", 1) or 1)
        workload_distribution = workload_distribution_override or self._layerwise_moe_workload_distribution(
            model,
            database,
            token_count=query_token_count,
        )
        if workload_distribution == "power_law":
            workload_distribution = f"power_law_{getattr(model, '_power_law_alpha', 1.2)}"

        result = database.query_moe(
            num_tokens=query_token_count,
            hidden_size=hidden_size,
            inter_size=inter_size,
            topk=topk,
            num_experts=num_experts,
            moe_tp_size=int(getattr(cfg, "moe_tp_size", 1) or 1),
            moe_ep_size=int(getattr(cfg, "moe_ep_size", 1) or 1),
            quant_mode=cfg.moe_quant_mode,
            workload_distribution=workload_distribution,
            is_context=is_context,
            moe_backend=getattr(cfg, "moe_backend", None),
            is_gated=True,
            enable_eplb=bool(getattr(cfg, "enable_eplb", False)),
        )
        return (
            float(result) * num_layers,
            float(getattr(result, "energy", 0.0)) * num_layers,
            getattr(result, "source", "silicon"),
        )

    def _layerwise_moe_workload_distribution(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
    ) -> str:
        """Return the MoE workload distribution for layerwise no-op add-back."""

        cfg = model.config
        workload_distribution = getattr(cfg, "workload_distribution", "power_law")
        if workload_distribution != "power_law":
            requested_distribution = str(workload_distribution)
            if self._layerwise_moe_distribution_available(
                model,
                database,
                requested_distribution,
                token_count=token_count,
            ):
                return requested_distribution
            power_law_distribution = f"power_law_{getattr(model, '_power_law_alpha', 1.2)}"
            if self._layerwise_moe_distribution_available(
                model,
                database,
                power_law_distribution,
                token_count=token_count,
            ):
                return "power_law"
            return requested_distribution

        model_name = str(getattr(model, "model_path", ""))
        if "Qwen" not in model_name:
            return "power_law"

        shared_inter_size = int(
            getattr(getattr(model, "extra_params", None), "shared_expert_inter_size", 0)
            or getattr(model, "_shared_expert_inter_size", 0)
            or 0
        )
        if shared_inter_size <= 0:
            return "power_law"

        if self._layerwise_moe_distribution_available(
            model,
            database,
            _QWEN_NOOP_MOE_DISTRIBUTION,
            token_count=token_count,
        ):
            return _QWEN_NOOP_MOE_DISTRIBUTION
        return "power_law"

    def _layerwise_moe_distribution_available(
        self,
        model: BaseModel,
        database: PerfDatabase,
        distribution: str,
        *,
        token_count: int,
    ) -> bool:
        """Return whether the MoE table has rows for ``distribution`` and shape."""

        cfg = model.config
        quant_mode = getattr(cfg, "moe_quant_mode", None)
        if quant_mode is None:
            return False

        topk = int(getattr(model, "_topk", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        inter_size = int(getattr(model, "_moe_inter_size", 0) or 0)
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        moe_tp_size = int(getattr(cfg, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(cfg, "moe_ep_size", 1) or 1)
        if topk <= 0 or num_experts <= 0 or inter_size <= 0 or hidden_size <= 0:
            return False

        try:
            from aiconfigurator.sdk.operations.moe import MoE

            moe_database = getattr(database, "real_database", database)
            MoE.load_data(moe_database)
            moe_database._moe_data.raise_if_not_loaded()
            data = moe_database._moe_data[quant_mode][distribution]
            if not (
                topk in data
                and num_experts in data[topk]
                and hidden_size in data[topk][num_experts]
                and inter_size in data[topk][num_experts][hidden_size]
                and moe_tp_size in data[topk][num_experts][hidden_size][inter_size]
                and moe_ep_size in data[topk][num_experts][hidden_size][inter_size][moe_tp_size]
            ):
                return False
            token_points = data[topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size]
            return bool(token_points) and int(token_count) <= max(int(point) for point in token_points)
        except (AttributeError, KeyError, PerfDataNotAvailableError, ValueError):
            return False

    def _layerwise_qwen_module_moe_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
        is_context: bool,
    ) -> tuple[float, float, str]:
        """Return measured Qwen sparse-MoE module latency when available.

        Qwen3.6-style vLLM MoE blocks include router, routed experts, shared
        expert, and the module's tensor-parallel collective. A module-level row
        should therefore replace the older piecewise no-op add-back, not be
        added on top of it.
        """

        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"

        extra_params = getattr(model, "extra_params", None)
        shared_inter_size = int(
            getattr(extra_params, "shared_expert_inter_size", 0)
            or getattr(model, "_shared_expert_inter_size", 0)
            or 0
        )
        if shared_inter_size <= 0:
            return 0.0, 0.0, "silicon"

        topk = int(getattr(model, "_topk", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        inter_size = int(getattr(model, "_moe_inter_size", 0) or 0)
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        if topk <= 0 or num_experts <= 0 or inter_size <= 0 or hidden_size <= 0:
            return 0.0, 0.0, "silicon"

        cfg = model.config
        quant_mode = getattr(cfg, "moe_quant_mode", None)
        if quant_mode is None:
            return 0.0, 0.0, "silicon"

        try:
            from aiconfigurator.sdk.operations.moe import MoE

            moe_database = getattr(database, "real_database", database)
            MoE.load_data(moe_database)
            moe_database._moe_data.raise_if_not_loaded()
            if (
                quant_mode not in moe_database._moe_data
                or _QWEN_MODULE_MOE_DISTRIBUTION not in moe_database._moe_data[quant_mode]
            ):
                return 0.0, 0.0, "silicon"
            qwen_module_data = moe_database._moe_data[quant_mode][_QWEN_MODULE_MOE_DISTRIBUTION]
            moe_tp_size = int(getattr(cfg, "moe_tp_size", 1) or 1)
            moe_ep_size = int(getattr(cfg, "moe_ep_size", 1) or 1)
            if (
                topk not in qwen_module_data
                or num_experts not in qwen_module_data[topk]
                or hidden_size not in qwen_module_data[topk][num_experts]
                or inter_size not in qwen_module_data[topk][num_experts][hidden_size]
                or moe_tp_size not in qwen_module_data[topk][num_experts][hidden_size][inter_size]
                or moe_ep_size not in qwen_module_data[topk][num_experts][hidden_size][inter_size][moe_tp_size]
            ):
                return 0.0, 0.0, "silicon"

            result = database.query_moe(
                num_tokens=int(token_count) * int(getattr(cfg, "attention_dp_size", 1) or 1),
                hidden_size=hidden_size,
                inter_size=inter_size,
                topk=topk,
                num_experts=num_experts,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
                quant_mode=quant_mode,
                workload_distribution=_QWEN_MODULE_MOE_DISTRIBUTION,
                is_context=is_context,
                moe_backend=getattr(cfg, "moe_backend", None),
                is_gated=True,
                enable_eplb=bool(getattr(cfg, "enable_eplb", False)),
            )
        except (AttributeError, KeyError, PerfDataNotAvailableError, ValueError):
            return 0.0, 0.0, "silicon"

        return (
            float(result) * num_layers,
            float(getattr(result, "energy", 0.0)) * num_layers,
            getattr(result, "source", "silicon"),
        )

    def _layerwise_noop_moe_addback(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
        is_context: bool,
        workload_distribution_override: str | None = None,
        deepseek_context_sum_shared: bool = False,
    ) -> tuple[tuple[float, float, str], tuple[float, float, str], tuple[float, float, str], bool]:
        """Return no-op MoE replacement pieces and whether they are bundled."""

        module_ms, module_energy, module_source = self._layerwise_qwen_module_moe_compute(
            model,
            database,
            token_count=token_count,
            num_layers=num_layers,
            is_context=is_context,
        )
        if module_ms > 0.0:
            zero = (0.0, 0.0, "silicon")
            return (module_ms, module_energy, module_source), zero, zero, True

        try:
            moe = self._layerwise_moe_compute(
                model,
                database,
                token_count=token_count,
                num_layers=num_layers,
                is_context=is_context,
                workload_distribution_override=workload_distribution_override,
            )
            router = self._layerwise_moe_router_compute(
                model,
                database,
                token_count=token_count,
                num_layers=num_layers,
            )
            shared = self._layerwise_moe_shared_expert_compute(
                model,
                database,
                token_count=token_count,
                num_layers=num_layers,
            )
        except (AttributeError, KeyError, PerfDataNotAvailableError, ValueError):
            zero = (0.0, 0.0, "silicon")
            return zero, zero, zero, False
        if (
            "DeepSeek-V4" in str(getattr(model, "model_path", ""))
            and is_context
            and deepseek_context_sum_shared
        ):
            zero = (0.0, 0.0, "silicon")
            # DeepSeek no-op layerwise rows replace the whole MoE MLP block.
            # The vLLM op-level collector measures only the routed local-rank
            # FusedMoE call, so the always-on shared expert and router GEMM
            # must be added back as separate compute.
            return (
                (moe[0] + router[0] + shared[0], moe[1] + router[1] + shared[1], "mixed"),
                zero,
                zero,
                True,
            )
        if "DeepSeek-V4" in str(getattr(model, "model_path", "")):
            zero = (0.0, 0.0, "silicon")
            routed_group_ms = moe[0] + router[0]
            shared_ms = shared[0]
            if routed_group_ms >= shared_ms:
                return (routed_group_ms, moe[1] + router[1], "mixed"), zero, zero, True
            return shared, zero, zero, True
        return moe, router, shared, False

    def _layerwise_noop_moe_addback_for_context(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        token_count: int,
        num_layers: int,
        runtime_config: RuntimeConfig,
        workload_distribution_override: str | None = None,
        deepseek_context_sum_shared: bool = False,
    ) -> tuple[tuple[float, float, str], tuple[float, float, str], tuple[float, float, str], bool]:
        """Return no-op MoE add-back using vLLM context chunking semantics."""

        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        cfg = model.config
        if (
            chunk_size is not None
            and deepseek_context_sum_shared
            and "DeepSeek-V4" in str(getattr(model, "model_path", ""))
            and int(getattr(cfg, "moe_ep_size", 1) or 1) == 1
            and int(getattr(cfg, "moe_tp_size", 1) or 1) <= 2
            and int(token_count) >= 4096
        ):
            chunk_size = min(chunk_size, 1024)
        direct_high_ep_deepseek = (
            deepseek_context_sum_shared
            and "DeepSeek-V4" in str(getattr(model, "model_path", ""))
            and int(getattr(model.config, "moe_ep_size", 1) or 1) >= 4
        )
        if chunk_size is None or token_count <= chunk_size or direct_high_ep_deepseek:
            return self._layerwise_noop_moe_addback(
                model,
                database,
                token_count=token_count,
                num_layers=num_layers,
                is_context=True,
                workload_distribution_override=workload_distribution_override,
                deepseek_context_sum_shared=deepseek_context_sum_shared,
            )

        totals: list[list[float | str]] = [
            [0.0, 0.0, "silicon"],
            [0.0, 0.0, "silicon"],
            [0.0, 0.0, "silicon"],
        ]
        bundled = True
        remaining = token_count
        while remaining > 0:
            chunk_tokens = min(chunk_size, remaining)
            pieces = self._layerwise_noop_moe_addback(
                model,
                database,
                token_count=chunk_tokens,
                num_layers=num_layers,
                is_context=True,
                workload_distribution_override=workload_distribution_override,
                deepseek_context_sum_shared=deepseek_context_sum_shared,
            )
            for index, piece in enumerate(pieces[:3]):
                totals[index][0] = float(totals[index][0]) + piece[0]
                totals[index][1] = float(totals[index][1]) + piece[1]
                if piece[0] > 0.0:
                    totals[index][2] = piece[2]
            bundled = bundled and pieces[3]
            remaining -= chunk_tokens

        return (
            (float(totals[0][0]), float(totals[0][1]), str(totals[0][2])),
            (float(totals[1][0]), float(totals[1][1]), str(totals[1][2])),
            (float(totals[2][0]), float(totals[2][1]), str(totals[2][2])),
            bundled,
        )

    def _layerwise_qwen_noop_scheduler_residual_ms(
        self,
        model: BaseModel,
        layer_detail: dict,
        *,
        token_count: int,
        layer_ms: float,
        is_context: bool,
        ctx_kv_tokens: int = 0,
        ctx_requests: int = 1,
    ) -> float:
        """Return visible MoE residual for Qwen no-op full-step scheduler rows."""

        if "Qwen" not in str(getattr(model, "model_path", "")):
            return 0.0
        if bool(layer_detail.get("includes_moe", False)):
            return 0.0
        if self._layerwise_detail_moe_weight_mode(layer_detail) != "noop":
            return 0.0
        parallel_fallback_ep = layer_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
        if not self._layerwise_scheduler_envelope_is_full_step(layer_detail):
            return 0.0
        if int(getattr(model, "_topk", 0) or 0) <= 0:
            return 0.0
        if int(getattr(model, "_shared_expert_inter_size", 0) or 0) > 0:
            return 0.0

        if is_context:
            tp_size = int(getattr(model.config, "tp_size", 1) or 1)
            moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
            moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
            if moe_tp_size > 1:
                return 0.0
            high_ep_wide_context = moe_ep_size > max(1, tp_size)
            if parallel_fallback_ep:
                if moe_ep_size <= max(1, tp_size):
                    return 0.0
                min_context_tokens = 128
                max_context_tokens = 256 if tp_size <= 1 else 2048
            else:
                if moe_ep_size >= 8 and not high_ep_wide_context:
                    return 0.0
                min_context_tokens = 128 if high_ep_wide_context else 512
                max_context_tokens = 2048
            if (
                ctx_requests != 1
                or ctx_kv_tokens != 0
                or token_count < min_context_tokens
                or token_count > max_context_tokens
            ):
                return 0.0
        elif token_count != 1:
            return 0.0

        residual_fraction = 0.20 if is_context else 0.38
        if is_context:
            if parallel_fallback_ep and tp_size == 2 and moe_ep_size >= 8 and token_count <= 128:
                residual_fraction = 0.75
        else:
            tp_size = int(getattr(model.config, "tp_size", 1) or 1)
            moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
            moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
            if moe_tp_size <= 1 and moe_ep_size >= 8 and moe_ep_size > max(1, tp_size):
                residual_fraction = min(1.05, 0.62 + 0.18 * math.log2(max(1, tp_size)))
        return max(0.0, layer_ms) * residual_fraction

    def _layerwise_deepseek_large_context_moe_overhead_ms(
        self,
        model: BaseModel,
        *,
        ctx_tokens: int,
        moe_ms: float,
    ) -> float:
        """Return DeepSeek context MoE scheduler overhead not captured by op add-back."""

        if "DeepSeek-V4" not in str(getattr(model, "model_path", "")):
            return 0.0
        if (
            getattr(self, "_layerwise_deepseek_context_floor_active", False)
            or getattr(self, "_layerwise_mixed_context_floor_active", False)
        ):
            return 0.0
        if ctx_tokens < 4096 or moe_ms <= 0.0:
            return 0.0

        cfg = model.config
        moe_tp_size = int(getattr(cfg, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(cfg, "moe_ep_size", 1) or 1)
        if moe_ep_size > 2:
            return 0.0
        if moe_ep_size <= 1 and moe_tp_size < 4:
            return 0.0

        return float(moe_ms) * 0.35

    def _layerwise_deepseek_decode_ep_moe_overhead_ms(
        self,
        model: BaseModel,
        *,
        moe_ms: float,
    ) -> float:
        """Return DeepSeek decode EP scheduler overhead for explicit MoE add-back.

        Decode scheduler-step rows already include the vLLM execution envelope.
        For no-op MoE layerwise rows, add back the routed MoE table itself but do
        not layer on a model-specific fraction of that MoE latency.
        """

        if "DeepSeek-V4" not in str(getattr(model, "model_path", "")):
            return 0.0
        return 0.0

    def _layerwise_smoothed_deepseek_generation_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        *,
        tp_size: int,
        batch_size: int,
        past_kv: int,
        num_layers: int,
        layer_detail: dict,
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> dict:
        """Suppress isolated high DeepSeek GEN scheduler rows across TP neighbors."""

        if "DeepSeek-V4" not in str(model_name):
            return layer_detail
        if self._layerwise_detail_moe_weight_mode(layer_detail) != "noop":
            return layer_detail
        if not self._layerwise_scheduler_like_detail(layer_detail):
            return layer_detail

        current_scale = self._layerwise_detail_scale(layer_detail, num_layers)
        current_step_ms = float(layer_detail["latency"]) * current_scale
        if current_step_ms <= 0.0:
            return layer_detail

        if moe_ep_size > 1 and moe_tp_size <= 1 and batch_size == 2:
            try:
                neighbor_detail = self._query_layerwise_detail(
                    database,
                    model_name,
                    "GEN",
                    tp_size,
                    1,
                    past_kv,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
            except (PerfDataNotAvailableError, ValueError):
                neighbor_detail = None
            if (
                neighbor_detail is not None
                and self._layerwise_detail_moe_weight_mode(neighbor_detail) == "noop"
                and self._layerwise_scheduler_like_detail(neighbor_detail)
            ):
                neighbor_step_ms = float(neighbor_detail["latency"]) * self._layerwise_detail_scale(
                    neighbor_detail,
                    num_layers,
                )
                if neighbor_step_ms > 0.0 and current_step_ms > neighbor_step_ms * 2.0:
                    smoothed = dict(layer_detail)
                    smoothed["latency"] = (neighbor_step_ms * 1.15) / current_scale
                    smoothed["diagnostic_smoothed_from_latency"] = float(layer_detail["latency"])
                    return smoothed

        if moe_ep_size != 1 or moe_tp_size != tp_size or tp_size <= 2:
            return layer_detail

        neighbor_steps: list[float] = []
        for neighbor_tp in (tp_size // 2, tp_size * 2):
            if neighbor_tp < 1 or neighbor_tp > 8 or neighbor_tp == tp_size:
                continue
            try:
                neighbor_detail = self._query_layerwise_detail(
                    database,
                    model_name,
                    "GEN",
                    neighbor_tp,
                    batch_size,
                    past_kv,
                    moe_tp_size=neighbor_tp,
                    moe_ep_size=moe_ep_size,
                )
            except (PerfDataNotAvailableError, ValueError):
                continue
            if self._layerwise_detail_moe_weight_mode(neighbor_detail) != self._layerwise_detail_moe_weight_mode(
                layer_detail
            ):
                continue
            if not self._layerwise_scheduler_like_detail(neighbor_detail):
                continue
            neighbor_steps.append(
                float(neighbor_detail["latency"]) * self._layerwise_detail_scale(neighbor_detail, num_layers)
            )

        if len(neighbor_steps) < 2:
            return layer_detail
        neighbor_median_ms = float(np.median(neighbor_steps))
        if neighbor_median_ms <= 0.0:
            return layer_detail
        if current_step_ms <= neighbor_median_ms * 1.15 or current_step_ms - neighbor_median_ms <= 0.5:
            return layer_detail

        smoothed = dict(layer_detail)
        smoothed["latency"] = neighbor_median_ms / current_scale
        smoothed["diagnostic_smoothed_from_latency"] = float(layer_detail["latency"])
        return smoothed

    def _layerwise_smoothed_noop_small_context_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        *,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        num_layers: int,
        layer_detail: dict,
        moe_tp_size: int,
        moe_ep_size: int,
    ) -> dict:
        """Suppress isolated high no-op scheduler rows for tiny context."""

        if batch_size != 1 or prefix != 0 or seq_len > 128:
            return layer_detail
        if "DeepSeek-V4" in str(model_name) and moe_ep_size >= 4:
            return layer_detail
        if self._layerwise_detail_moe_weight_mode(layer_detail) != "noop":
            return layer_detail
        if not self._layerwise_scheduler_like_detail(layer_detail):
            return layer_detail

        current_scale = self._layerwise_detail_scale(layer_detail, num_layers)
        current_ms = float(layer_detail["latency"]) * current_scale
        if current_ms <= 0.0:
            return layer_detail

        candidate_parallelism: list[tuple[int, int, int]] = []
        if tp_size > 1:
            candidate_parallelism.extend(((tp_size, tp_size, 1), (tp_size, 1, tp_size)))
        else:
            candidate_parallelism.extend(((2, 2, 1), (2, 1, 2)))

        neighbor_steps: list[float] = []
        for candidate_tp, candidate_moe_tp, candidate_ep in candidate_parallelism:
            if (
                candidate_tp == tp_size
                and candidate_moe_tp == moe_tp_size
                and candidate_ep == moe_ep_size
            ):
                continue
            try:
                neighbor_detail = self._layerwise_context_step_detail(
                    database,
                    model_name,
                    candidate_tp,
                    batch_size,
                    seq_len,
                    prefix,
                    moe_weight_mode="noop",
                    moe_tp_size=candidate_moe_tp,
                    moe_ep_size=candidate_ep,
                )
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                continue
            if self._layerwise_detail_moe_weight_mode(neighbor_detail) != "noop":
                continue
            if not self._layerwise_scheduler_like_detail(neighbor_detail):
                continue
            neighbor_steps.append(
                float(neighbor_detail["latency"]) * self._layerwise_detail_scale(neighbor_detail, num_layers)
            )

        if not neighbor_steps:
            return layer_detail
        neighbor_median_ms = float(np.median(neighbor_steps))
        if neighbor_median_ms <= 0.0:
            return layer_detail
        if current_ms <= neighbor_median_ms * 1.06 or current_ms - neighbor_median_ms <= 0.5:
            return layer_detail

        smoothed = dict(layer_detail)
        smoothed["latency"] = neighbor_median_ms / current_scale
        smoothed["diagnostic_smoothed_from_latency"] = float(layer_detail["latency"])
        return smoothed

    def _layerwise_deepseek_tp1_small_context_overhead_ms(
        self,
        model: BaseModel,
        *,
        ctx_tokens: int,
        ctx_kv_tokens: int,
        ctx_requests: int,
        layer_detail: dict,
        layer_ms: float,
        layer_includes_moe: bool,
    ) -> float:
        """Return tiny-context TP1 overhead for no-op shared-expert MoE rows."""

        if "DeepSeek-V4" not in str(getattr(model, "model_path", "")):
            return 0.0
        if (
            int(getattr(model.config, "tp_size", 1) or 1) != 1
            or ctx_tokens > 128
            or ctx_kv_tokens != 0
            or ctx_requests != 1
            or layer_ms <= 0.0
            or layer_includes_moe
            or int(getattr(model, "_shared_expert_inter_size", 0) or 0) <= 0
            or self._layerwise_detail_moe_weight_mode(layer_detail) != "noop"
            or not self._layerwise_scheduler_like_detail(layer_detail)
        ):
            return 0.0
        return float(layer_ms) * 0.13

    def _layerwise_moe_shared_expert_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
    ) -> tuple[float, float, str]:
        """Return shared-expert FFN latency for no-op MoE layerwise rows.

        ``LAYERWISE_MOE_NOOP`` replaces the whole MoE MLP module with identity.
        For Qwen-style MoE blocks that module includes routed experts, the
        always-on shared expert, and the scalar shared-expert gate, so the
        routed ``query_moe`` add-back is not sufficient on its own.
        """

        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"

        extra_params = getattr(model, "extra_params", None)
        shared_inter_size = int(
            getattr(extra_params, "shared_expert_inter_size", 0)
            or getattr(model, "_shared_expert_inter_size", 0)
            or 0
        )
        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        tp_size = int(getattr(model.config, "tp_size", 1) or 1)
        if shared_inter_size <= 0 or hidden_size <= 0 or tp_size <= 0:
            return 0.0, 0.0, "silicon"

        local_inter_size = shared_inter_size // tp_size
        if local_inter_size <= 0:
            return 0.0, 0.0, "silicon"

        gemm_quant_mode = getattr(model.config, "gemm_quant_mode", None) or common.GEMMQuantMode.bfloat16
        # vLLM 0.20.1 currently has MoE rows without a companion GEMM table.
        # Query the shared-expert GEMMs in HYBRID mode so table-backed data is
        # used when present and otherwise falls back to the GEMM empirical model.
        shared_results = [
            database.query_gemm(
                int(token_count),
                1,
                hidden_size,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.HYBRID,
            ),
            database.query_gemm(
                int(token_count),
                local_inter_size,
                hidden_size,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.HYBRID,
            ),
            database.query_mem_op(
                int(token_count) * local_inter_size * 4,
                database_mode=common.DatabaseMode.HYBRID,
            ),
            database.query_gemm(
                int(token_count),
                hidden_size,
                local_inter_size,
                gemm_quant_mode,
                database_mode=common.DatabaseMode.HYBRID,
            ),
            database.query_mem_op(
                int(token_count) * hidden_size * 4,
                database_mode=common.DatabaseMode.HYBRID,
            ),
        ]

        latency_ms = sum(float(result) for result in shared_results)
        energy = sum(float(getattr(result, "energy", 0.0)) for result in shared_results)
        sources = {str(getattr(result, "source", "silicon")) for result in shared_results}

        return latency_ms * num_layers, energy * num_layers, next(iter(sources)) if len(sources) == 1 else "mixed"

    def _layerwise_moe_router_compute(
        self,
        model: BaseModel,
        database: PerfDatabase,
        token_count: int,
        num_layers: int,
    ) -> tuple[float, float, str]:
        """Return router projection latency for no-op MoE layerwise rows.

        vLLM's ``FusedMoE`` op consumes router logits; the model's MoE module
        computes those logits with a ``hidden_size -> num_experts`` projection
        before entering the fused expert kernel. ``LAYERWISE_MOE_NOOP`` replaces
        the whole MoE MLP module, so the router projection must be restored
        separately from the routed expert and shared-expert add-backs.
        """

        if token_count <= 0 or num_layers <= 0:
            return 0.0, 0.0, "silicon"

        hidden_size = int(getattr(model, "_hidden_size", 0) or 0)
        num_experts = int(getattr(model, "_num_experts", 0) or 0)
        if hidden_size <= 0 or num_experts <= 0:
            return 0.0, 0.0, "silicon"

        gemm_quant_mode = getattr(model.config, "gemm_quant_mode", None) or common.GEMMQuantMode.bfloat16
        result = database.query_gemm(
            int(token_count),
            num_experts,
            hidden_size,
            gemm_quant_mode,
            database_mode=common.DatabaseMode.HYBRID,
        )

        return (
            float(result) * num_layers,
            float(getattr(result, "energy", 0.0)) * num_layers,
            getattr(result, "source", "silicon"),
        )

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
        context_moe_weight_mode = self._deepseek_context_moe_weight_mode(
            model_name=str(model_name),
            seq_len=effective_isl,
            runtime_config=runtime_config,
        )
        context_lookup_tp_size = self._layerwise_context_lookup_tp_size(model, tp_size)

        def _context_detail_for_tp(query_tp_size: int, query_prefix: int) -> dict[str, float | bool]:
            detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                query_tp_size,
                ctx_requests,
                effective_isl,
                query_prefix,
                runtime_config,
                moe_weight_mode=context_moe_weight_mode,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
            detail = dict(detail)
            detail["_context_lookup_tp_size"] = query_tp_size
            return detail

        try:
            layer_detail = _context_detail_for_tp(context_lookup_tp_size, prefix)
        except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
            if context_lookup_tp_size != tp_size:
                try:
                    layer_detail = _context_detail_for_tp(tp_size, prefix)
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    if prefix <= 0:
                        raise
                    try:
                        layer_detail = _context_detail_for_tp(context_lookup_tp_size, 0)
                    except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                        layer_detail = _context_detail_for_tp(tp_size, 0)
            else:
                if prefix <= 0:
                    raise
                layer_detail = _context_detail_for_tp(tp_size, 0)
        layer_detail = self._layerwise_smoothed_noop_small_context_detail(
            database,
            str(model_name),
            tp_size=tp_size,
            batch_size=ctx_requests,
            seq_len=effective_isl,
            prefix=prefix,
            num_layers=num_layers,
            layer_detail=layer_detail,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        structural_moe_context_ms = self._layerwise_structural_moe_context_ms(layer_detail, num_layers)
        if structural_moe_context_ms is None:
            layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
            layer_ms = float(layer_detail["latency"]) * layer_scale
        else:
            layer_ms = structural_moe_context_ms
        layer_ms = self._layerwise_noop_context_continuation_floor_ms(
            database,
            model_name,
            tp_size,
            ctx_requests,
            effective_isl,
            prefix,
            runtime_config,
            layer_detail,
            layer_ms,
            num_layers,
            moe_weight_mode=context_moe_weight_mode,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            has_shared_expert=int(getattr(model, "_shared_expert_inter_size", 0) or 0) > 0,
        )
        attention_dp_size = int(getattr(model.config, "attention_dp_size", 1) or 1)
        context_chunk_size = self._layerwise_context_chunk_size(runtime_config, layer_detail)
        if (
            "DeepSeek-V4" in str(model_name)
            and self._layerwise_has_subquadratic_context_attention(model)
            and self._layerwise_scheduler_like_detail(layer_detail)
            and int(layer_detail.get("_context_lookup_tp_size", tp_size)) > tp_size
            and tp_size == 1
            and attention_dp_size > 1
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
        ):
            if ctx_tokens >= context_chunk_size * 2:
                layer_ms *= 1.35
            elif ctx_tokens >= context_chunk_size // 2:
                layer_ms *= 1.10
        context_source = str(layer_detail.get("latency_source") or "")
        token_count = ctx_tokens
        layer_includes_moe = bool(layer_detail.get("includes_moe", False))
        deepseek_scheduler_moe_context = (
            "DeepSeek-V4" in str(model_name)
            and layer_includes_moe
            and moe_ep_size > 1
            and context_source in {"schedule_to_update", "fpm_wall", "worker_wall"}
        )
        if deepseek_scheduler_moe_context:
            allreduce_ms = 0.0
        else:
            allreduce_ms = self._layerwise_context_tp_allreduce_ms(
                model,
                database,
                tp_size,
                ctx_tokens,
                num_layers,
                context_source=context_source,
            )
            if (
                allreduce_ms > 0.0
                and self._layerwise_has_subquadratic_context_attention(model)
                and self._layerwise_scheduler_like_detail(layer_detail)
                and moe_ep_size > 1
                and moe_tp_size <= 1
                and attention_dp_size < max(1, tp_size)
                and prefix <= 0
                and ctx_requests <= 1
                and context_chunk_size is not None
                and ctx_tokens <= context_chunk_size // 2
            ):
                allreduce_ms = 0.0
        moe_tp_allreduce_ms = 0.0
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)
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
                    token_count,
                )
        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        moe_router_ms, moe_router_energy, moe_router_source = 0.0, 0.0, "silicon"
        moe_shared_ms, moe_shared_energy, moe_shared_source = 0.0, 0.0, "silicon"
        context_uses_noop_moe = self._layerwise_detail_moe_weight_mode(layer_detail) == "noop"
        layerwise_parallel_fallback = layer_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
        qwen_noop_context_requires_moe_addback = (
            "Qwen" in str(model_name)
            and context_uses_noop_moe
            and not layer_includes_moe
            and moe_tp_size <= 1
            and moe_ep_size > max(1, tp_size)
        )
        context_noop_envelope_covers_addback = (
            context_uses_noop_moe
            and self._layerwise_scheduler_envelope_is_full_step(layer_detail)
            and not layerwise_parallel_fallback
            and not qwen_noop_context_requires_moe_addback
            and (prefix > 0 or ctx_tokens <= 128)
        )
        compressed_context_noop_covers_addback = (
            context_uses_noop_moe
            and self._layerwise_has_subquadratic_context_attention(model)
            and self._layerwise_scheduler_like_detail(layer_detail)
            and not layerwise_parallel_fallback
            and not qwen_noop_context_requires_moe_addback
        )
        if (
            structural_moe_context_ms is None
            and not layer_includes_moe
            and not context_noop_envelope_covers_addback
            and not compressed_context_noop_covers_addback
        ):
            represented_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
            context_moe_distribution_override = self._deepseek_context_moe_distribution_override(
                model,
                database,
                token_count=token_count,
                runtime_config=runtime_config,
            )
            (
                (moe_ms, moe_energy, moe_source),
                (moe_router_ms, moe_router_energy, moe_router_source),
                (moe_shared_ms, moe_shared_energy, moe_shared_source),
                moe_addback_is_bundled,
            ) = self._layerwise_noop_moe_addback_for_context(
                model,
                database,
                token_count=ctx_tokens,
                num_layers=represented_layers,
                runtime_config=runtime_config,
                workload_distribution_override=context_moe_distribution_override,
                deepseek_context_sum_shared=True,
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
                        token_count,
                    )
        moe_ep_alltoall_layers = represented_moe_layers
        if moe_ep_alltoall_layers <= 0 and moe_ms > 0.0:
            moe_ep_alltoall_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
        moe_ep_alltoall_ms = 0.0
        has_external_moe_ep = (
            moe_ep_alltoall_layers > 0
            and moe_ep_size > 1
            and (layer_includes_moe or moe_ms <= 0.0)
        )
        if has_external_moe_ep:
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
                    token_count,
                )
        elif moe_ep_alltoall_layers > 0 and moe_ep_size > 1 and moe_ms > 0.0 and not moe_addback_is_bundled:
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
                    "Falling back to no explicit vLLM CTX no-op MoE-EP all-to-all for model=%s, moe_ep_size=%s, "
                    "token_count=%s because NCCL data is unavailable",
                    model_name,
                    moe_ep_size,
                    token_count,
                )
        elif (
            "DeepSeek-V4" in str(model_name)
            and moe_ep_alltoall_layers > 0
            and moe_ep_size > 1
            and moe_ms > 0.0
            and moe_addback_is_bundled
        ):
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
                    "Falling back to no explicit DeepSeek-V4 CTX no-op MoE-EP all-to-all for model=%s, "
                    "moe_ep_size=%s, token_count=%s because NCCL data is unavailable",
                    model_name,
                    moe_ep_size,
                    token_count,
                )

        latency_dict = defaultdict(float, {"context_layerwise": layer_ms, "context_tp_allreduce": allreduce_ms})
        energy_dict = defaultdict(float, {"context_layerwise": 0.0, "context_tp_allreduce": 0.0})
        source_dict = {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}
        dense_tp1_scheduler_overhead_ms = self._layerwise_dense_tp1_context_scheduler_overhead_ms(
            model,
            layer_detail,
            tp_size=tp_size,
            ctx_tokens=ctx_tokens,
            ctx_kv_tokens=ctx_kv_tokens,
            ctx_requests=ctx_requests,
            layer_ms=layer_ms,
            layer_includes_moe=layer_includes_moe,
        )
        if dense_tp1_scheduler_overhead_ms > 0.0:
            latency_dict["context_scheduler_overhead"] = dense_tp1_scheduler_overhead_ms
            energy_dict["context_scheduler_overhead"] = 0.0
            source_dict["context_scheduler_overhead"] = "silicon"
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
        if moe_shared_ms > 0.0:
            latency_dict["context_moe_shared_expert"] = moe_shared_ms
            energy_dict["context_moe_shared_expert"] = moe_shared_energy
            source_dict["context_moe_shared_expert"] = moe_shared_source
        deepseek_tp1_small_overhead_ms = self._layerwise_deepseek_tp1_small_context_overhead_ms(
            model,
            ctx_tokens=ctx_tokens,
            ctx_kv_tokens=ctx_kv_tokens,
            ctx_requests=ctx_requests,
            layer_detail=layer_detail,
            layer_ms=layer_ms,
            layer_includes_moe=layer_includes_moe,
        )
        if deepseek_tp1_small_overhead_ms > 0.0:
            latency_dict["context_moe_scheduler_overhead"] += deepseek_tp1_small_overhead_ms
            energy_dict["context_moe_scheduler_overhead"] = 0.0
            source_dict["context_moe_scheduler_overhead"] = "silicon"
        deepseek_moe_overhead_ms = self._layerwise_deepseek_large_context_moe_overhead_ms(
            model,
            ctx_tokens=ctx_tokens,
            moe_ms=moe_ms,
        )
        if deepseek_moe_overhead_ms > 0.0:
            latency_dict["context_moe_scheduler_overhead"] += deepseek_moe_overhead_ms
            energy_dict["context_moe_scheduler_overhead"] = 0.0
            source_dict["context_moe_scheduler_overhead"] = "silicon"
        qwen_residual_ms = self._layerwise_qwen_noop_scheduler_residual_ms(
            model,
            layer_detail,
            token_count=ctx_tokens,
            layer_ms=layer_ms,
            is_context=True,
            ctx_kv_tokens=ctx_kv_tokens,
            ctx_requests=ctx_requests,
        )
        if qwen_residual_ms > 0.0:
            latency_dict["context_moe_scheduler_residual"] = qwen_residual_ms
            energy_dict["context_moe_scheduler_residual"] = 0.0
            source_dict["context_moe_scheduler_residual"] = "silicon"
        context_chunk_size = self._layerwise_context_chunk_size(runtime_config, layer_detail)
        if (
            "DeepSeek-V4" in str(model_name)
            and not getattr(self, "_layerwise_deepseek_context_floor_active", False)
            and context_uses_noop_moe
            and not layer_includes_moe
            and ctx_requests == 1
            and ctx_kv_tokens == 0
            and context_chunk_size is not None
            and ctx_tokens == context_chunk_size * 2
            and int(getattr(model.config, "attention_dp_size", 1) or 1) > max(1, tp_size)
        ):
            try:
                direct_detail = self._layerwise_context_step_detail(
                    database,
                    model_name,
                    tp_size,
                    1,
                    ctx_tokens,
                    0,
                    max_num_batched_tokens=context_chunk_size,
                    moe_weight_mode=context_moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
                if not self._layerwise_scheduler_timed_detail(direct_detail):
                    raise PerfDataNotAvailableError("direct DeepSeek two-chunk row is not scheduler timed")
                direct_ms = float(direct_detail["latency"]) * self._layerwise_detail_scale(direct_detail, num_layers)
                current_ms = float(sum(latency_dict.values()))
                if current_ms < direct_ms <= current_ms * 1.50:
                    latency_dict["context_moe_scheduler_residual"] += direct_ms - current_ms
                    energy_dict["context_moe_scheduler_residual"] = 0.0
                    source_dict["context_moe_scheduler_residual"] = "silicon"
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        if (
            "Qwen" in str(model_name)
            and tp_size == 2
            and ctx_requests == 1
            and 768 <= ctx_tokens <= 2048
            and not layer_includes_moe
            and int(getattr(model, "_shared_expert_inter_size", 0) or 0) <= 0
            and context_uses_noop_moe
            and self._layerwise_scheduler_envelope_is_full_step(layer_detail)
        ):
            try:
                context_chunk_size = self._layerwise_context_chunk_size(runtime_config, layer_detail) or 2048
                floor_detail = self._layerwise_context_step_detail(
                    database,
                    model_name,
                    tp_size,
                    1,
                    context_chunk_size * 2,
                    0,
                    moe_weight_mode=context_moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
                floor_ms = (
                    float(floor_detail["latency"])
                    * self._layerwise_detail_scale(floor_detail, num_layers)
                    * 0.50
                )
                current_ms = float(sum(latency_dict.values()))
                if floor_ms > current_ms:
                    latency_dict["context_moe_scheduler_residual"] += floor_ms - current_ms
                    energy_dict["context_moe_scheduler_residual"] = 0.0
                    source_dict["context_moe_scheduler_residual"] = "silicon"
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        self._apply_deepseek_high_ep_noop_context_floor(
            model,
            database,
            runtime_config,
            latency_dict,
            layer_detail,
            ctx_tokens=ctx_tokens,
            ctx_kv_tokens=ctx_kv_tokens,
            ctx_requests=ctx_requests,
            layer_includes_moe=layer_includes_moe,
        )
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
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        layer_detail = self._layerwise_smoothed_deepseek_generation_detail(
            database,
            str(model_name),
            tp_size=tp_size,
            batch_size=effective_bs,
            past_kv=past_kv,
            num_layers=num_layers,
            layer_detail=layer_detail,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
        )
        layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
        layer_step_ms = float(layer_detail["latency"]) * layer_scale
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
                    moe_ep_alltoall_step_ms = (
                        self._layerwise_moe_ep_alltoall_ms(model, database, moe_ep_size, effective_bs)
                        * self._layerwise_detail_represented_layers(layer_detail, num_layers)
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, "
                        "moe_ep_size=%s, batch_size=%s because NCCL data is unavailable",
                        model_name,
                        moe_ep_size,
                        effective_bs,
                    )
                    moe_ep_alltoall_step_ms = 0.0

        latency_dict = defaultdict(float, {"generation_layerwise": max(0.0, layer_step_ms)})
        energy_dict = defaultdict(float, {"generation_layerwise": 0.0})
        source_dict = {"generation_layerwise": "silicon"}
        qwen_residual_ms = self._layerwise_qwen_noop_scheduler_residual_ms(
            model,
            layer_detail,
            token_count=effective_bs,
            layer_ms=layer_step_ms,
            is_context=False,
        )
        if qwen_residual_ms > 0.0:
            latency_dict["generation_moe_scheduler_residual"] = qwen_residual_ms
            energy_dict["generation_moe_scheduler_residual"] = 0.0
            source_dict["generation_moe_scheduler_residual"] = "silicon"
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
        deepseek_decode_overhead_ms = self._layerwise_deepseek_decode_ep_moe_overhead_ms(
            model,
            moe_ms=moe_step_ms,
        )
        attention_dp_size = int(getattr(model.config, "attention_dp_size", 1) or 1)
        if (
            "DeepSeek-V4" in str(model_name)
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and attention_dp_size > 1
            and effective_bs == 1
            and not layer_includes_moe
            and moe_step_ms <= 0.0
            and self._layerwise_detail_moe_weight_mode(layer_detail) == "noop"
            and self._layerwise_scheduler_like_detail(layer_detail)
        ):
            deepseek_decode_overhead_ms = max(
                deepseek_decode_overhead_ms,
                layer_step_ms * min(0.50, 0.13 * float(attention_dp_size)),
            )
        if deepseek_decode_overhead_ms > 0.0:
            latency_dict["generation_moe_scheduler_overhead"] = deepseek_decode_overhead_ms
            energy_dict["generation_moe_scheduler_overhead"] = 0.0
            source_dict["generation_moe_scheduler_overhead"] = "silicon"
        if moe_router_step_ms > 0.0:
            latency_dict["generation_moe_router"] = moe_router_step_ms
            energy_dict["generation_moe_router"] = moe_router_step_energy
            source_dict["generation_moe_router"] = moe_router_step_source
        if moe_shared_step_ms > 0.0:
            latency_dict["generation_moe_shared_expert"] = moe_shared_step_ms
            energy_dict["generation_moe_shared_expert"] = moe_shared_step_energy
            source_dict["generation_moe_shared_expert"] = moe_shared_step_source
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
        moe_shared_ms_total = 0.0
        moe_shared_energy_total = 0.0
        moe_shared_source: str | None = None
        for i in range(0, osl - 1, stride):
            kv_len = isl + i
            layer_detail = self._query_layerwise_detail(
                database,
                model_name,
                "GEN",
                tp_size,
                effective_bs,
                kv_len,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
            layer_detail = self._layerwise_smoothed_deepseek_generation_detail(
                database,
                str(model_name),
                tp_size=tp_size,
                batch_size=effective_bs,
                past_kv=kv_len,
                num_layers=num_layers,
                layer_detail=layer_detail,
                moe_tp_size=moe_tp_size,
                moe_ep_size=moe_ep_size,
            )
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
            if (
                not layer_includes_moe
                and represented_moe_layers > 0
                and moe_ep_size > 1
            ):
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
                        moe_ep_alltoall_step_ms = (
                            self._layerwise_moe_ep_alltoall_ms(model, database, moe_ep_size, effective_bs)
                            * self._layerwise_detail_represented_layers(layer_detail, num_layers)
                        )
                    except (AttributeError, PerfDataNotAvailableError, ValueError):
                        logger.debug(
                            "Falling back to no explicit vLLM GEN MoE-EP all-to-all for model=%s, "
                            "moe_ep_size=%s, batch_size=%s because NCCL data is unavailable",
                            model_name,
                            moe_ep_size,
                            effective_bs,
                        )
                        moe_ep_alltoall_step_ms = 0.0
            moe_ep_alltoall_total += moe_ep_alltoall_step_ms * repeat_count
            moe_ms_total += moe_step_ms * repeat_count
            moe_energy_total += moe_step_energy * repeat_count
            moe_router_ms_total += moe_router_step_ms * repeat_count
            moe_router_energy_total += moe_router_step_energy * repeat_count
            moe_shared_ms_total += moe_shared_step_ms * repeat_count
            moe_shared_energy_total += moe_shared_step_energy * repeat_count
            if moe_source is None or moe_source == moe_step_source:
                moe_source = moe_step_source
            else:
                moe_source = "mixed"
            if moe_router_source is None or moe_router_source == moe_router_step_source:
                moe_router_source = moe_router_step_source
            else:
                moe_router_source = "mixed"
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
        deepseek_decode_overhead_ms = self._layerwise_deepseek_decode_ep_moe_overhead_ms(
            model,
            moe_ms=moe_ms_total,
        )
        if deepseek_decode_overhead_ms > 0.0:
            latency_dict["generation_moe_scheduler_overhead"] = deepseek_decode_overhead_ms
            energy_dict["generation_moe_scheduler_overhead"] = 0.0
            source_dict["generation_moe_scheduler_overhead"] = "silicon"
        if moe_router_ms_total > 0.0:
            latency_dict["generation_moe_router"] = moe_router_ms_total
            energy_dict["generation_moe_router"] = moe_router_energy_total
            source_dict["generation_moe_router"] = moe_router_source or "silicon"
        if moe_shared_ms_total > 0.0:
            latency_dict["generation_moe_shared_expert"] = moe_shared_ms_total
            energy_dict["generation_moe_shared_expert"] = moe_shared_energy_total
            source_dict["generation_moe_shared_expert"] = moe_shared_source or "silicon"
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

        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        attention_dp_size = int(getattr(model.config, "attention_dp_size", 1) or 1)
        ctx_tokens = max(int(ctx_tokens), 0)
        ctx_requests = max(int(ctx_requests), 1)
        ctx_prefix = max(int(prefix), 0)
        extra_params = getattr(model, "extra_params", None)
        topk = int(getattr(model, "_topk", 0) or 0)
        has_shared_expert = (
            int(
                getattr(model, "_shared_expert_inter_size", 0)
                or getattr(extra_params, "shared_expert_inter_size", 0)
                or 0
            )
            > 0
        )
        is_moe_model = (
            topk > 0
            or int(getattr(extra_params, "num_experts", 0) or getattr(model, "_num_experts", 0) or 0) > 0
            or moe_tp_size > 1
            or moe_ep_size > 1
        )
        context_chunk_hint = self._layerwise_context_chunk_size(runtime_config)
        subquadratic_context_attention = self._layerwise_has_subquadratic_context_attention(model)
        attention_dp_widened_fresh_ep = (
            subquadratic_context_attention
            and ctx_prefix <= 0
            and moe_ep_size > 1
            and attention_dp_size > max(1, tp_size)
        )
        if is_moe_model and ctx_requests > 1:
            # A mixed scheduler step can carry several partial-prefill requests.
            # Layerwise context rows are collected as one request, so approximate
            # the equivalent single-request shape from sum-of-squares attention:
            # R requests sharing N total tokens behave closer to N/sqrt(R) than
            # to the average N/R or one aggregate N-token sequence.
            request_exponent = 0.6 if ctx_prefix <= 0 else 0.5
            if (
                subquadratic_context_attention
                and ctx_prefix <= 0
                and moe_ep_size > 1
                and attention_dp_size > max(1, tp_size)
            ):
                request_exponent = 0.0
            if (
                not subquadratic_context_attention
                and ctx_prefix <= 0
                and moe_ep_size >= 4
                and moe_tp_size <= 1
                and tp_size >= 4
            ):
                request_exponent = 1.0 + max(0.0, math.log2(float(moe_ep_size) / 4.0)) * 0.20
            if (
                subquadratic_context_attention
                and ctx_prefix > 0
                and context_chunk_hint is not None
                and ctx_tokens <= context_chunk_hint * 2 + max(64, gen_tokens * 4)
            ):
                request_exponent = 1.0
            context_query_tokens = max(1, math.ceil(ctx_tokens / (float(ctx_requests) ** request_exponent)))
            if (
                (moe_ep_size > 1 or moe_tp_size > 1)
                and context_chunk_hint is not None
                and not attention_dp_widened_fresh_ep
            ):
                context_query_tokens = min(context_query_tokens, context_chunk_hint * 2)
        else:
            context_query_tokens = ctx_tokens
        gen_tokens = max(int(gen_tokens), 0)
        combined_tokens = max(context_query_tokens + gen_tokens, 1)
        if (
            is_moe_model
            and subquadratic_context_attention
            and moe_ep_size > 1
            and attention_dp_size <= max(1, tp_size)
            and ctx_prefix <= 0
            and ctx_tokens > 0
            and gen_tokens > 0
            and context_chunk_hint is not None
            and context_query_tokens >= context_chunk_hint
            and context_query_tokens // context_chunk_hint != combined_tokens // context_chunk_hint
        ):
            combined_tokens = max(context_query_tokens, 1)
        if (
            is_moe_model
            and not subquadratic_context_attention
            and moe_ep_size >= 4
            and moe_tp_size <= 1
            and tp_size >= 4
            and ctx_prefix <= 0
            and ctx_requests > 1
            and ctx_tokens > 0
            and gen_tokens > 0
        ):
            combined_tokens = max(context_query_tokens, 1)
        if (
            not is_moe_model
            and ctx_tokens > 0
            and context_chunk_hint is not None
            and context_query_tokens <= context_chunk_hint < context_query_tokens + gen_tokens
        ):
            combined_tokens = max(context_query_tokens, 1)
        mixed_context_moe_weight_mode = "noop" if "DeepSeek-V4" in str(model_name) else None

        used_context_tokens = combined_tokens
        context_lookup_tp_size = self._layerwise_context_lookup_tp_size(model, tp_size)

        def _mixed_context_detail(seq_len: int, prefix_tokens: int) -> dict[str, float | bool]:
            def _runtime_detail_for_tp(query_tp_size: int) -> dict[str, float | bool]:
                detail = self._layerwise_context_detail_for_runtime(
                    database,
                    model_name,
                    query_tp_size,
                    1,
                    seq_len,
                    prefix_tokens,
                    runtime_config,
                    moe_weight_mode=mixed_context_moe_weight_mode,
                    moe_tp_size=moe_tp_size,
                    moe_ep_size=moe_ep_size,
                )
                detail = dict(detail)
                detail["_context_lookup_tp_size"] = query_tp_size
                return detail

            if (
                attention_dp_widened_fresh_ep
                and ctx_requests > 1
                and context_chunk_hint is not None
                and seq_len > context_chunk_hint * 2
            ):
                for query_tp_size in dict.fromkeys((context_lookup_tp_size, tp_size)):
                    try:
                        detail = self._layerwise_context_step_detail(
                            database,
                            model_name,
                            query_tp_size,
                            1,
                            seq_len,
                            prefix_tokens,
                            max_num_batched_tokens=context_chunk_hint,
                            moe_weight_mode=mixed_context_moe_weight_mode,
                            moe_tp_size=moe_tp_size,
                            moe_ep_size=moe_ep_size,
                        )
                        detail = dict(detail)
                        detail["_context_lookup_tp_size"] = query_tp_size
                        return detail
                    except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                        pass
            if context_lookup_tp_size != tp_size:
                try:
                    return _runtime_detail_for_tp(context_lookup_tp_size)
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                    pass
            return _runtime_detail_for_tp(tp_size)

        try:
            combined_detail = _mixed_context_detail(combined_tokens, ctx_prefix)
        except (AssertionError, PerfDataNotAvailableError, ValueError):
            if ctx_tokens > 0 and ctx_tokens != combined_tokens:
                try:
                    combined_detail = _mixed_context_detail(context_query_tokens, ctx_prefix)
                    used_context_tokens = context_query_tokens
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    if ctx_prefix <= 0:
                        raise
                    combined_detail = _mixed_context_detail(combined_tokens, 0)
            elif ctx_prefix > 0:
                combined_detail = _mixed_context_detail(combined_tokens, 0)
            else:
                raise
        combined_source = str(combined_detail.get("latency_source") or "")
        if ctx_prefix > 0 and used_context_tokens == combined_tokens:
            try:
                fresh_combined_detail = _mixed_context_detail(combined_tokens, 0)
                fresh_source = str(fresh_combined_detail.get("latency_source") or "")
                fresh_ctx_ms = float(fresh_combined_detail["latency"]) * self._layerwise_detail_scale(
                    fresh_combined_detail, num_layers
                )
                combined_ctx_ms = float(combined_detail["latency"]) * self._layerwise_detail_scale(
                    combined_detail, num_layers
                )
                use_fresh_scheduler_surface = (
                    fresh_source in _LAYERWISE_SCHEDULER_LATENCY_SOURCES
                    and (
                        combined_source not in _LAYERWISE_SCHEDULER_LATENCY_SOURCES
                        or fresh_ctx_ms < combined_ctx_ms * 0.85
                    )
                )
                if not use_fresh_scheduler_surface and combined_source not in _LAYERWISE_SCHEDULER_LATENCY_SOURCES:
                    # Interpolating across prefix-0 scheduler rows and nonzero-prefix
                    # kernel rows can lose the scheduler source label. In mixed rows
                    # with fresh prefill plus a short continuation, the averaged-prefix
                    # surface can be much too small; prefer the fresh scheduler-like
                    # surface when it is clearly the relevant envelope.
                    use_fresh_scheduler_surface = fresh_ctx_ms > combined_ctx_ms * 4.0
                if use_fresh_scheduler_surface:
                    combined_detail = fresh_combined_detail
                    combined_source = fresh_source
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        combined_ctx_ms = float(combined_detail["latency"]) * self._layerwise_detail_scale(
            combined_detail, num_layers
        )
        scheduler_like_context = self._layerwise_scheduler_like_detail(combined_detail)
        if (
            scheduler_like_context
            and ctx_requests > 1
            and ctx_tokens > 0
            and tp_size <= 1
            and not is_moe_model
        ):
            combined_ctx_ms *= 0.90
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        # In mixed prefill+decode iterations for expert-parallel MoE, the
        # scheduler-envelope context row tracks the same overlapped iteration
        # boundary as FPM. Adding an isolated dense TP allreduce on top
        # double-counts the comm tail for large mixed chunks.
        mixed_ep_context_covers_tp = scheduler_like_context and ctx_tokens > 0 and moe_ep_size > 1
        context_detail_includes_moe = bool(combined_detail.get("includes_moe", False))
        if (
            scheduler_like_context
            and subquadratic_context_attention
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and 0 < ctx_tokens < 128
            and gen_tokens >= max(4, min(topk if topk > 0 else 4, 8))
        ):
            combined_ctx_ms *= 0.75
        if (
            scheduler_like_context
            and not subquadratic_context_attention
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and 0 < ctx_tokens < 128
            and topk > 0
            and gen_tokens > topk
        ):
            combined_ctx_ms *= 1.12
        noop_moe_context_covers_mixed_decode = (
            scheduler_like_context
            and ctx_tokens >= 128
            and is_moe_model
            and not context_detail_includes_moe
        )
        mixed_moe_context_covers_tp = (
            scheduler_like_context
            and ctx_tokens > 0
            and is_moe_model
            and (mixed_ep_context_covers_tp or moe_tp_size > 1)
        )
        if scheduler_like_context and (mixed_ep_context_covers_tp or mixed_moe_context_covers_tp):
            combined_allreduce_ms = 0.0
        else:
            combined_allreduce_ms = (
                self._layerwise_tp_allreduce_ms(model, database, tp_size, combined_tokens) * 2 * num_layers
            )
            if scheduler_like_context and ctx_tokens > 0 and not is_moe_model:
                combined_allreduce_ms *= min(1.0, 2.0 / max(1.0, float(tp_size)))
        decode_delta_ms = 0.0
        ep_high_decode_floor_ms = 0.0
        dense_context_envelope_covers_mixed_decode = scheduler_like_context and ctx_tokens > 0 and not is_moe_model
        small_decode_limit = topk if topk > 0 else 4
        compressed_decode_limit = max(small_decode_limit, 8)
        context_chunk_size = self._layerwise_context_chunk_size(runtime_config, combined_detail)
        avg_decode_kv = isl + osl // 2
        if (
            dense_context_envelope_covers_mixed_decode
            and ctx_prefix <= 0
            and ctx_requests <= 1
        ):
            combined_ctx_ms *= self._layerwise_dense_mixed_context_envelope_multiplier(
                tp_size=tp_size,
                ctx_tokens=ctx_tokens,
                gen_tokens=gen_tokens,
                avg_decode_kv=avg_decode_kv,
                context_chunk_size=context_chunk_size,
            )
        fresh_shared_ep_subquadratic_mixed = (
            scheduler_like_context
            and subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens > 0
            and gen_tokens >= max(4, min(small_decode_limit, 7))
        )
        if (
            fresh_shared_ep_subquadratic_mixed
            and attention_dp_size == max(1, tp_size)
            and tp_size > 1
            and ctx_tokens < context_chunk_size * 2
        ):
            if ctx_tokens < context_chunk_size // 4:
                combined_ctx_ms *= 1.03
            else:
                combined_ctx_ms *= 1.04
        if (
            fresh_shared_ep_subquadratic_mixed
            and attention_dp_size < max(1, tp_size)
            and gen_tokens > small_decode_limit
            and avg_decode_kv >= context_chunk_size
            and int(context_chunk_size * 0.14) <= ctx_tokens < context_chunk_size // 4
        ):
            combined_ctx_ms *= 1.15
        if (
            fresh_shared_ep_subquadratic_mixed
            and attention_dp_size < max(1, tp_size)
            and ctx_tokens >= context_chunk_size * 2
        ):
            combined_ctx_ms *= 0.85
        elif (
            fresh_shared_ep_subquadratic_mixed
            and attention_dp_size == max(1, tp_size)
            and tp_size > 1
            and ctx_tokens >= context_chunk_size * 2
        ):
            combined_ctx_ms *= 0.90
        if (
            subquadratic_context_attention
            and is_moe_model
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and tp_size > 1
            and attention_dp_size <= max(1, tp_size)
            and ctx_prefix > 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens >= context_chunk_size * 2
            and gen_tokens >= max(4, min(small_decode_limit, 7))
        ):
            combined_ctx_ms *= 0.85
        if (
            subquadratic_context_attention
            and is_moe_model
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and tp_size > 1
            and attention_dp_size < max(1, tp_size)
            and ctx_prefix > 0
            and ctx_requests <= 2
            and context_chunk_size is not None
            and ctx_prefix < context_chunk_size * 2
            and ctx_tokens <= int(context_chunk_size * 1.1)
            and gen_tokens > small_decode_limit
            and avg_decode_kv >= context_chunk_size
        ):
            combined_ctx_ms *= 1.25

        fresh_parallel_fallback_ep_floor = (
            not subquadratic_context_attention
            and is_moe_model
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and context_chunk_size is not None
            and ctx_tokens >= max(512, context_chunk_size // 4)
            and gen_tokens > 0
        )
        continuation_parallel_fallback_ep_floor = (
            not subquadratic_context_attention
            and is_moe_model
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix > 0
            and context_chunk_size is not None
            and ctx_tokens >= max(512, context_chunk_size // 4)
            and gen_tokens > 0
        )
        allow_parallel_fallback_ep_floor = (
            fresh_parallel_fallback_ep_floor
            or continuation_parallel_fallback_ep_floor
        )
        exact_wide_ep_noop_context = (
            scheduler_like_context
            and is_moe_model
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and not has_shared_expert
            and moe_ep_size > max(1, tp_size)
            and moe_tp_size <= 1
            and combined_detail.get("parallel_fallback_moe_ep_size") in (None, "")
        )

        def _raise_ep_high_decode_floor(candidate_ms: float) -> None:
            """Apply saturated EP decode floors only when the local envelope supports them."""

            nonlocal ep_high_decode_floor_ms
            candidate_ms = float(candidate_ms)
            if candidate_ms <= 0.0:
                return
            if (
                tp_size <= 1
                and combined_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
                and not allow_parallel_fallback_ep_floor
            ):
                return
            if (
                subquadratic_context_attention
                and is_moe_model
                and mixed_ep_context_covers_tp
                and not context_detail_includes_moe
                and moe_ep_size > 1
                and moe_tp_size <= 1
                and attention_dp_size <= max(1, tp_size)
                and context_chunk_size is not None
                and ctx_tokens < context_chunk_size * 2
                and candidate_ms > combined_ctx_ms * 1.65
                and not (
                    attention_dp_size < max(1, tp_size)
                    and ctx_prefix <= 0
                    and context_chunk_size // 4 < ctx_tokens < int(context_chunk_size * 2 / 3)
                    and candidate_ms <= combined_ctx_ms * 1.95
                )
            ):
                return
            if (
                subquadratic_context_attention
                and is_moe_model
                and mixed_ep_context_covers_tp
                and not context_detail_includes_moe
                and moe_ep_size > 1
                and moe_tp_size <= 1
                and attention_dp_size <= max(1, tp_size)
                and context_chunk_size is not None
                and ctx_prefix > 0
                and ctx_tokens <= int(context_chunk_size * 1.5)
                and candidate_ms > combined_ctx_ms * 1.10
            ):
                return
            ep_high_decode_floor_ms = max(ep_high_decode_floor_ms, candidate_ms)

        def _saturated_context_step_detail(
            seq_len: int,
            prefix_tokens: int = 0,
            *,
            max_num_batched_tokens: int | None = None,
        ) -> dict[str, float | bool]:
            last_error: Exception | None = None
            for query_tp_size in dict.fromkeys((context_lookup_tp_size, tp_size)):
                try:
                    return self._layerwise_context_step_detail(
                        database,
                        model_name,
                        query_tp_size,
                        1,
                        seq_len,
                        prefix_tokens,
                        max_num_batched_tokens=max_num_batched_tokens,
                        moe_weight_mode=mixed_context_moe_weight_mode,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError) as exc:
                    last_error = exc
            raise PerfDataNotAvailableError(
                "No saturated context step layerwise row for "
                f"model={model_name}, tp_size={tp_size}, context_lookup_tp_size={context_lookup_tp_size}, "
                f"seq_len={seq_len}, prefix={prefix_tokens}"
            ) from last_error

        def _saturated_context_runtime_detail(
            seq_len: int,
            prefix_tokens: int = 0,
        ) -> dict[str, float | bool]:
            last_error: Exception | None = None
            for query_tp_size in dict.fromkeys((context_lookup_tp_size, tp_size)):
                try:
                    return self._layerwise_context_detail_for_runtime(
                        database,
                        model_name,
                        query_tp_size,
                        1,
                        seq_len,
                        prefix_tokens,
                        runtime_config,
                        moe_weight_mode=mixed_context_moe_weight_mode,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError) as exc:
                    last_error = exc
            raise PerfDataNotAvailableError(
                "No saturated runtime context layerwise row for "
                f"model={model_name}, tp_size={tp_size}, context_lookup_tp_size={context_lookup_tp_size}, "
                f"seq_len={seq_len}, prefix={prefix_tokens}"
            ) from last_error

        small_subquadratic_continuation = (
            subquadratic_context_attention
            and context_chunk_size is not None
            and ctx_prefix > 0
            and ctx_tokens <= context_chunk_size * 2 + max(64, gen_tokens * 4)
        )
        full_prefill_overlap_fraction = 0.75 if moe_ep_size >= 8 else 0.6
        fresh_full_prefill_chunk = (
            topk > 0
            and gen_tokens > topk
            and ctx_prefix == 0
            and context_chunk_size is not None
            and ctx_tokens >= context_chunk_size * full_prefill_overlap_fraction
        )
        full_moe_context_covers_small_decode = (
            scheduler_like_context
            and ctx_tokens >= 128
            and context_detail_includes_moe
            and mixed_ep_context_covers_tp
            and gen_tokens <= small_decode_limit
        )
        compressed_long_prefix_context_covers_decode = (
            scheduler_like_context
            and subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and ctx_prefix > 0
            and context_chunk_size is not None
            and ctx_prefix * 2 >= context_chunk_size * 3
            and ctx_tokens >= context_chunk_size * 2
            and gen_tokens <= compressed_decode_limit
        )
        context_envelope_covers_mixed_decode = (
            dense_context_envelope_covers_mixed_decode
            or noop_moe_context_covers_mixed_decode
            or full_moe_context_covers_small_decode
            or compressed_long_prefix_context_covers_decode
            or (context_detail_includes_moe and mixed_ep_context_covers_tp and fresh_full_prefill_chunk)
        )
        if noop_moe_context_covers_mixed_decode and ctx_prefix > 0:
            try:
                active_context_span = min(ctx_tokens + ctx_prefix, isl if isl > 0 else ctx_tokens + ctx_prefix)
                saturation_tokens = int(max(ctx_tokens, active_context_span))
                saturated_ctx_detail = None
                for candidate_tokens in range(saturation_tokens, max(ctx_tokens, saturation_tokens - 33) - 1, -1):
                    try:
                        saturated_ctx_detail = _saturated_context_step_detail(candidate_tokens)
                        break
                    except (AssertionError, PerfDataNotAvailableError, ValueError):
                        continue
                if saturated_ctx_detail is None:
                    raise PerfDataNotAvailableError("no nearby saturated no-op MoE mixed context row")
                if not self._layerwise_scheduler_like_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("nearby saturated no-op MoE row is a composed context row")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                if combined_ctx_ms < saturated_ctx_ms * 0.75:
                    combined_ctx_ms = max(combined_ctx_ms, saturated_ctx_ms * 0.9)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        high_kv_small_long_prefix_continuation = (
            scheduler_like_context
            and subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and ctx_prefix > 0
            and context_chunk_size is not None
            and ctx_prefix >= context_chunk_size * 2
            and context_chunk_size // 2 <= ctx_tokens < context_chunk_size
            and gen_tokens >= max(4, min(small_decode_limit, 6))
            and avg_decode_kv >= context_chunk_size * 2
        )

        def _nearby_gen_step_ms(
            *,
            allow_standard_kv_fallback: bool = False,
            batch_size_override: int | None = None,
        ) -> tuple[float, int]:
            gen_detail = None
            query_batch_size = max(1, int(batch_size_override or gen_tokens))
            candidate_kvs = list(range(avg_decode_kv, max(-1, avg_decode_kv - 9), -1))
            if ctx_prefix > 0 or allow_standard_kv_fallback:
                fallback_kvs = sorted((1024, 4096, 8192, 16384, 32768), key=lambda kv: abs(kv - avg_decode_kv))
                candidate_kvs.extend(kv for kv in fallback_kvs if kv not in candidate_kvs)
            for candidate_kv in candidate_kvs:
                try:
                    gen_detail = self._query_layerwise_detail(
                        database,
                        model_name,
                        "GEN",
                        tp_size,
                        query_batch_size,
                        candidate_kv,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                    break
                except (PerfDataNotAvailableError, ValueError):
                    continue
            if gen_detail is None:
                raise PerfDataNotAvailableError(
                    f"No nearby GEN layerwise row for tp_size={tp_size}, "
                    f"batch_size={query_batch_size}, kv={avg_decode_kv}"
                )
            return (
                float(gen_detail["latency"]) * self._layerwise_detail_scale(gen_detail, num_layers),
                avg_decode_kv,
            )

        if gen_tokens > 0 and not context_envelope_covers_mixed_decode and not mixed_ep_context_covers_tp:
            try:
                gen_step_ms, avg_decode_kv = _nearby_gen_step_ms()
                try:
                    gen_as_ctx_detail = self._layerwise_context_step_detail(
                        database,
                        model_name,
                        tp_size,
                        1,
                        gen_tokens,
                        0,
                        moe_weight_mode=mixed_context_moe_weight_mode,
                        moe_tp_size=moe_tp_size,
                        moe_ep_size=moe_ep_size,
                    )
                    gen_as_ctx_ms = float(gen_as_ctx_detail["latency"]) * self._layerwise_detail_scale(
                        gen_as_ctx_detail, num_layers
                    )
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    gen_as_ctx_ms = 0.0
                decode_delta_ms = max(0.0, gen_step_ms - gen_as_ctx_ms)
            except (PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to context-only vLLM mixed-step layerwise estimate for model=%s, "
                    "tp_size=%s, gen_tokens=%s, avg_decode_kv=%s",
                    model_name,
                    tp_size,
                    gen_tokens,
                    avg_decode_kv,
                )
        elif (
            gen_tokens >= 4
            and mixed_ep_context_covers_tp
            and not noop_moe_context_covers_mixed_decode
        ):
            try:
                if ctx_prefix == 0 and 0 < ctx_tokens <= 128:
                    # Tiny fresh-prefill real-workload mixed iterations are
                    # scheduler-envelope dominated in vLLM: adding isolated
                    # decode slices double-counts work already hidden in the
                    # full-step context row.
                    decode_delta_ms = 0.0
                elif fresh_full_prefill_chunk:
                    # Fresh prefill chunks that consume most of the scheduler
                    # context budget already form the dominant mixed-iteration
                    # envelope; adding isolated decode work double-counts the
                    # overlapped EP decode slice.
                    decode_delta_ms = 0.0
                else:
                    gen_step_ms, _ = _nearby_gen_step_ms(allow_standard_kv_fallback=True)
                    decode_delta_ms = gen_step_ms
                    if topk > 0 and gen_tokens > topk:
                        # High-decode EP mixed rows carry another non-overlapped
                        # decode-like MoE/communication slice beyond the context
                        # scheduler envelope. Keep this tied to top-k so low-batch
                        # mixed rows remain context-envelope dominated.
                        decode_delta_ms += gen_step_ms
                        if 0 < ctx_tokens <= 256:
                            # Very short prefill chunks do not provide enough
                            # context work to hide decode-heavy EP MoE slices. This
                            # is the real-workload case where a short fresh prompt
                            # or continuation chunk joins many active decodes.
                            decode_delta_ms += gen_step_ms * 2
                            if ctx_prefix == 0:
                                overlap_deficit_slices = int(np.ceil((256 - ctx_tokens) / 64.0))
                                decode_delta_ms += gen_step_ms * max(0, overlap_deficit_slices)
                        try:
                            saturated_ctx_detail = _saturated_context_step_detail(
                                context_chunk_size or combined_tokens,
                            )
                            if not self._layerwise_scheduler_like_detail(saturated_ctx_detail):
                                raise PerfDataNotAvailableError("saturated EP mixed context floor row is composed")
                            saturated_ctx_ms = float(saturated_ctx_detail["latency"])
                            _raise_ep_high_decode_floor(
                                saturated_ctx_ms
                                * self._layerwise_detail_scale(
                                    saturated_ctx_detail,
                                    num_layers,
                                )
                            )
                        except (AssertionError, PerfDataNotAvailableError, ValueError):
                            ep_high_decode_floor_ms = 0.0
            except (PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to EP context-only vLLM mixed-step estimate for model=%s, "
                    "tp_size=%s, gen_tokens=%s",
                    model_name,
                    tp_size,
                    gen_tokens,
                )
        elif (
            noop_moe_context_covers_mixed_decode
            and ctx_prefix > 0
            and gen_tokens > 0
            and not compressed_long_prefix_context_covers_decode
        ):
            if (small_subquadratic_continuation and not high_kv_small_long_prefix_continuation) or (
                gen_tokens <= 1 and ctx_tokens <= 512 and avg_decode_kv <= 2048
            ):
                decode_delta_ms = 0.0
            else:
                try:
                    use_compressed_attention_tail = (
                        subquadratic_context_attention and moe_tp_size <= 1 and moe_ep_size <= 1
                    )
                    stable_decode_batch = 4 if use_compressed_attention_tail else min(max(1, gen_tokens), 4)
                    gen_step_ms, _ = _nearby_gen_step_ms(
                        allow_standard_kv_fallback=True,
                        batch_size_override=stable_decode_batch,
                    )
                    if use_compressed_attention_tail:
                        decode_overlap_slices = 1.25
                    else:
                        decode_overlap_slices = 1.0 + min(1.0, 1.0 / max(1.0, float(tp_size)))
                    moe_tp_small_continuation_covers_decode = (
                        moe_tp_size > 1 and moe_ep_size <= 1 and ctx_tokens < 512
                    )
                    if moe_tp_small_continuation_covers_decode:
                        if (
                            ctx_requests <= 1
                            and gen_tokens > topk
                            and context_chunk_size is not None
                            and avg_decode_kv >= context_chunk_size
                        ):
                            if ctx_prefix >= context_chunk_size:
                                decode_overlap_slices = min(
                                    2.25,
                                    1.35 + max(0.0, float(ctx_tokens) - 280.0) / 220.0,
                                )
                            else:
                                decode_overlap_slices = 0.50
                        else:
                            decode_overlap_slices = 1.0 if ctx_prefix >= int(context_chunk_size or 0) else 0.0
                        if tp_size >= 4 and ctx_requests <= 1 and gen_tokens > topk:
                            # At higher TP, MoE-TP continuation scheduler rows
                            # already overlap more of the decode slice than the
                            # isolated GEN table suggests.
                            decode_overlap_slices *= 0.5
                    if (
                        not use_compressed_attention_tail
                        and not moe_tp_small_continuation_covers_decode
                        and ctx_tokens <= 512
                        and gen_tokens > topk
                    ):
                        decode_overlap_slices += 0.05
                    if (
                        exact_wide_ep_noop_context
                        and ctx_prefix > 0
                        and context_chunk_size is not None
                        and ctx_tokens < context_chunk_size // 4
                        and gen_tokens > topk
                    ):
                        decode_overlap_slices = min(
                            decode_overlap_slices,
                            0.30 if tp_size >= 4 else 0.60,
                        )
                    if (
                        exact_wide_ep_noop_context
                        and ctx_prefix > 0
                        and tp_size <= 2
                        and context_chunk_size is not None
                        and gen_tokens <= max(small_decode_limit, 8)
                        and (
                            ctx_tokens < context_chunk_size // 4
                            or ctx_tokens >= context_chunk_size * 2
                        )
                    ):
                        decode_overlap_slices = 0.0
                    if (
                        exact_wide_ep_noop_context
                        and ctx_prefix > 0
                        and ctx_requests > 1
                        and context_chunk_size is not None
                        and context_chunk_size // 2 <= ctx_tokens < context_chunk_size * 2
                        and gen_tokens <= max(small_decode_limit, 8)
                    ):
                        decode_overlap_slices = min(decode_overlap_slices, 0.75)
                    if (
                        subquadratic_context_attention
                        and is_moe_model
                        and moe_ep_size > 1
                        and moe_tp_size <= 1
                        and tp_size == 2
                        and int(combined_detail.get("_context_lookup_tp_size", tp_size)) > tp_size
                        and context_chunk_size is not None
                        and gen_tokens > topk
                    ):
                        if ctx_prefix >= context_chunk_size * 2 and ctx_tokens < context_chunk_size:
                            chunk_fraction = float(ctx_tokens) / max(1.0, float(context_chunk_size))
                            decode_overlap_slices *= max(0.15, min(0.50, 1.30 * (1.0 - chunk_fraction)))
                        elif ctx_tokens >= context_chunk_size * 2:
                            decode_overlap_slices *= 0.20
                    elif (
                        subquadratic_context_attention
                        and is_moe_model
                        and moe_ep_size > 1
                        and moe_tp_size <= 1
                        and tp_size >= 4
                        and context_chunk_size is not None
                        and gen_tokens > topk
                        and ctx_tokens >= context_chunk_size
                    ):
                        if ctx_prefix >= max(768, context_chunk_size // 5):
                            decode_overlap_slices *= 0.30
                        elif ctx_prefix <= 0 and ctx_tokens >= context_chunk_size * 2:
                            decode_overlap_slices *= 0.35 if avg_decode_kv >= context_chunk_size else 0.25
                    decode_delta_ms = gen_step_ms * decode_overlap_slices
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no extra no-op MoE mixed decode slice for model=%s, "
                        "tp_size=%s, gen_tokens=%s",
                        model_name,
                        tp_size,
                        gen_tokens,
                    )
            if (
                topk > 0
                and gen_tokens > topk
                and ctx_requests > 1
                and context_chunk_size is not None
                and not small_subquadratic_continuation
                and context_chunk_size <= ctx_tokens <= context_chunk_size * 2
            ):
                try:
                    saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                    saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                        saturated_ctx_detail,
                        num_layers,
                    )
                    continuation_floor_multiplier = 0.85
                    if moe_ep_size < 4 or moe_tp_size > 1:
                        continuation_floor_multiplier = 0.50
                    if (
                        moe_tp_size > 1
                        and moe_ep_size <= 1
                        and tp_size <= 2
                        and ctx_requests > 1
                        and ctx_prefix > 0
                        and gen_tokens > topk
                    ):
                        continuation_floor_multiplier = 0.53
                    _raise_ep_high_decode_floor(saturated_ctx_ms * continuation_floor_multiplier)
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    pass
            if (
                topk > 0
                and gen_tokens > 0
                and context_chunk_size is not None
                and (
                    768 <= ctx_tokens < context_chunk_size
                    or (
                        (
                            (moe_ep_size > 1 and moe_tp_size <= 1 and ctx_tokens >= 512)
                            or (
                                moe_tp_size > 1
                                and moe_ep_size <= 1
                                and (
                                    ctx_tokens >= 512
                                    or ctx_tokens + gen_tokens >= max(512, context_chunk_size // 4)
                                )
                            )
                        )
                        and tp_size <= 2
                        and ctx_tokens < context_chunk_size
                    )
                )
                and (not has_shared_expert or (moe_ep_size > 1 and moe_tp_size <= 1))
            ):
                try:
                    saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                    saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                        saturated_ctx_detail,
                        num_layers,
                    )
                    low_continuation_floor_multiplier = 0.50
                    if (
                        tp_size <= 1
                        and continuation_parallel_fallback_ep_floor
                        and combined_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
                    ):
                        low_continuation_floor_multiplier = 0.50
                    elif moe_ep_size >= 4 and moe_tp_size <= 1:
                        low_continuation_floor_multiplier = 1.05 if has_shared_expert else 0.75
                        if (
                            exact_wide_ep_noop_context
                            and not has_shared_expert
                            and tp_size <= 2
                            and ctx_tokens < context_chunk_size
                            and gen_tokens > topk
                        ):
                            low_continuation_floor_multiplier = 0.98
                        if (
                            not has_shared_expert
                            and tp_size >= 4
                            and moe_ep_size > tp_size
                            and ctx_requests > 1
                            and ctx_prefix > 0
                            and ctx_tokens < context_chunk_size
                            and gen_tokens > topk
                        ):
                            low_continuation_floor_multiplier = 0.95
                    elif (
                        moe_tp_size > 1
                        and moe_ep_size <= 1
                        and tp_size <= 2
                        and ctx_requests <= 1
                        and ctx_prefix > 0
                        and gen_tokens > topk
                    ):
                        low_continuation_floor_multiplier = 0.525
                    elif (
                        moe_tp_size > 1
                        and moe_ep_size <= 1
                        and tp_size <= 2
                        and ctx_requests > 1
                        and ctx_prefix > 0
                        and gen_tokens > topk
                        and ctx_tokens < context_chunk_size * 2
                    ):
                        low_continuation_floor_multiplier = 0.53
                    elif (
                        moe_tp_size > 1
                        and moe_ep_size <= 1
                        and tp_size <= 2
                        and ctx_requests >= 4
                        and ctx_tokens >= context_chunk_size * 0.75
                    ):
                        low_continuation_floor_multiplier *= 1.16
                    _raise_ep_high_decode_floor(saturated_ctx_ms * low_continuation_floor_multiplier)
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    pass
            if (
                topk > 0
                and ctx_requests > 1
                and context_chunk_size is not None
                and not has_shared_expert
                and (gen_tokens <= topk or ctx_tokens > context_chunk_size * 2)
            ):
                try:
                    use_direct_large_continuation_floor = (
                        ctx_tokens > context_chunk_size * 2
                        and moe_tp_size <= 1
                        and (tp_size <= 2 or moe_ep_size > 1)
                        and (gen_tokens > topk or tp_size <= 1 or moe_ep_size > max(1, tp_size))
                    )
                    if use_direct_large_continuation_floor:
                        saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                    else:
                        saturated_ctx_detail = _saturated_context_runtime_detail(context_chunk_size * 2)
                    saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                        saturated_ctx_detail,
                        num_layers,
                    )
                    if use_direct_large_continuation_floor:
                        if ctx_tokens <= context_chunk_size * 2 + max(256, gen_tokens * 8):
                            floor_multiplier = 0.50
                        elif tp_size <= 1 and moe_tp_size <= 1 and moe_ep_size <= 1 and (
                            ctx_tokens >= context_chunk_size * 3.5
                        ):
                            floor_multiplier = 0.85
                        elif (
                            moe_ep_size > max(1, tp_size)
                            and moe_tp_size <= 1
                            and ctx_requests > 1
                        ):
                            if tp_size <= 1:
                                floor_multiplier = 0.70 if ctx_requests <= 3 else 0.90
                            elif tp_size >= 4:
                                floor_multiplier = 1.00
                            else:
                                floor_multiplier = 0.75
                        elif (
                            moe_tp_size > 1
                            and moe_ep_size <= 1
                            and tp_size <= 2
                            and ctx_requests >= 4
                            and gen_tokens > topk
                        ):
                            floor_multiplier = 0.95
                        elif moe_ep_size <= 1 or moe_ep_size >= 8:
                            floor_multiplier = 0.75
                        else:
                            floor_multiplier = min(0.85, 0.55 + 0.10 * math.log2(max(1.0, float(tp_size))))
                    else:
                        if (
                            moe_tp_size > 1
                            and moe_ep_size <= 1
                            and tp_size <= 2
                            and ctx_requests >= 4
                            and gen_tokens > topk
                        ):
                            floor_multiplier = 1.20
                        elif (
                            moe_tp_size > 1
                            and moe_ep_size <= 1
                            and tp_size <= 2
                            and ctx_requests > 1
                            and ctx_prefix > 0
                            and context_chunk_size is not None
                            and ctx_tokens <= context_chunk_size * 2
                            and gen_tokens <= topk
                        ):
                            floor_multiplier = 0.95
                        elif (
                            moe_ep_size > max(1, tp_size)
                            and moe_tp_size <= 1
                            and tp_size >= 4
                            and ctx_requests > 1
                            and ctx_prefix > 0
                            and context_chunk_size is not None
                            and ctx_tokens < context_chunk_size
                            and gen_tokens <= topk
                        ):
                            floor_multiplier = 1.38
                        else:
                            floor_multiplier = 1.05 if gen_tokens > topk else 1.12
                    _raise_ep_high_decode_floor(saturated_ctx_ms * floor_multiplier + decode_delta_ms)
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    pass

        if (
            ep_high_decode_floor_ms <= 0.0
            and noop_moe_context_covers_mixed_decode
            and ctx_prefix <= 0
            and ctx_requests > 1
            and context_chunk_size is not None
            and ctx_tokens > context_chunk_size * 2
            and not has_shared_expert
        ):
            try:
                saturated_ctx_detail = _saturated_context_runtime_detail(context_chunk_size * 2)
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                floor_multiplier = 1.12 if gen_tokens > topk else 1.10
                _raise_ep_high_decode_floor(saturated_ctx_ms * floor_multiplier)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        if (
            noop_moe_context_covers_mixed_decode
            and ctx_prefix <= 0
            and context_chunk_size is not None
            and not has_shared_expert
            and (
                (tp_size <= 1 and ctx_tokens > context_chunk_size * 2 and avg_decode_kv >= 2048)
                or (
                    moe_tp_size > 1
                    and (
                        ctx_tokens >= max(768, context_chunk_size // 2)
                        or (
                            gen_tokens > topk
                            and ctx_tokens < context_chunk_size
                            and ctx_tokens + gen_tokens >= max(512, context_chunk_size // 4)
                        )
                    )
                )
            )
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                floor_multiplier = 0.75 if tp_size <= 1 and moe_tp_size <= 1 else 0.50
                _raise_ep_high_decode_floor(saturated_ctx_ms * floor_multiplier)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass
        if (
            subquadratic_context_attention
            and is_moe_model
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and tp_size == 2
            and int(combined_detail.get("_context_lookup_tp_size", tp_size)) > tp_size
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and gen_tokens > topk
        ):
            if ctx_tokens < 128:
                widened_fresh_floor_multiplier = 1.45
            elif avg_decode_kv >= int(context_chunk_size * 1.8) and ctx_tokens < 256:
                widened_fresh_floor_multiplier = 1.05
            elif avg_decode_kv >= int(context_chunk_size * 1.8) and ctx_tokens < 512:
                widened_fresh_floor_multiplier = 1.17
            elif ctx_tokens < 512:
                widened_fresh_floor_multiplier = 1.35
            elif (
                avg_decode_kv >= int(context_chunk_size * 1.8)
                and ctx_tokens < int(context_chunk_size * 1.1)
            ):
                widened_fresh_floor_multiplier = 1.03
            elif ctx_tokens < context_chunk_size * 2:
                widened_fresh_floor_multiplier = 1.30
            elif ctx_tokens < context_chunk_size * 3:
                widened_fresh_floor_multiplier = 1.15
            else:
                widened_fresh_floor_multiplier = 1.0
            if widened_fresh_floor_multiplier > 1.0:
                _raise_ep_high_decode_floor(combined_ctx_ms * widened_fresh_floor_multiplier)
        if dense_context_envelope_covers_mixed_decode and gen_tokens > 0:
            dense_decode_tail_slices = self._layerwise_dense_mixed_decode_tail_slices(
                tp_size=tp_size,
                ctx_tokens=ctx_tokens,
                gen_tokens=gen_tokens,
                avg_decode_kv=avg_decode_kv,
                context_chunk_size=context_chunk_size,
            )
            if dense_decode_tail_slices > 0.0:
                try:
                    gen_step_ms, _ = _nearby_gen_step_ms(allow_standard_kv_fallback=True)
                    if tp_size <= 1:
                        dense_gen_step_ms = gen_step_ms
                        if gen_tokens > 8 and avg_decode_kv >= int((context_chunk_size or 2048) * 1.45):
                            dense_gen_step_ms *= 1.20
                        decode_delta_ms = max(0.0, dense_gen_step_ms - (combined_ctx_ms + combined_allreduce_ms))
                    else:
                        decode_delta_ms = gen_step_ms * dense_decode_tail_slices
                except (PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to dense context-only vLLM mixed-step estimate for model=%s, "
                        "tp_size=%s, gen_tokens=%s",
                        model_name,
                        tp_size,
                        gen_tokens,
                    )

        if (
            ep_high_decode_floor_ms <= 0.0
            and topk > 0
            and gen_tokens > topk
            and mixed_ep_context_covers_tp
            and not context_envelope_covers_mixed_decode
            and ctx_tokens > 128
        ):
            try:
                context_chunk_size = self._layerwise_context_chunk_size(runtime_config, combined_detail)
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size or combined_tokens)
                if not self._layerwise_scheduler_like_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("saturated EP mixed context floor row is composed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"])
                _raise_ep_high_decode_floor(
                    saturated_ctx_ms
                    * self._layerwise_detail_scale(
                        saturated_ctx_detail,
                        num_layers,
                    )
                )
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                ep_high_decode_floor_ms = 0.0

        if (
            subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and context_chunk_size is not None
            and 128 <= ctx_tokens < context_chunk_size // 4
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("small shared-expert EP mixed floor row is not scheduler timed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                _raise_ep_high_decode_floor(saturated_ctx_ms * 0.65)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and attention_dp_size == max(1, tp_size)
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and 0 < ctx_tokens < 128
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("tiny shared-expert EP mixed floor row is not scheduler timed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                _raise_ep_high_decode_floor(saturated_ctx_ms * 0.22)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and attention_dp_size < max(1, tp_size)
            and ctx_prefix <= 0
            and ctx_requests <= 2
            and context_chunk_size is not None
            and context_chunk_size // 4 < ctx_tokens < int(context_chunk_size * 2 / 3)
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("mid shared-expert EP mixed floor row is not scheduler timed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                _raise_ep_high_decode_floor(saturated_ctx_ms * 0.50)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix > 0
            and context_chunk_size is not None
            and context_chunk_size <= ctx_tokens < context_chunk_size * 2
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("mid shared-expert EP mixed floor row is not scheduler timed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                _raise_ep_high_decode_floor(saturated_ctx_ms * 0.93)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            subquadratic_context_attention
            and is_moe_model
            and mixed_ep_context_covers_tp
            and ctx_prefix <= 0
            and context_chunk_size is not None
            and context_chunk_size // 4 <= ctx_tokens < context_chunk_size
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and avg_decode_kv * 2 >= context_chunk_size * 3
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("subquadratic fresh mixed floor row is not scheduler timed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                fresh_floor_multiplier = 0.60
                if has_shared_expert and moe_ep_size > 1 and moe_tp_size <= 1 and gen_tokens > topk:
                    fresh_floor_multiplier = 1.05
                _raise_ep_high_decode_floor(saturated_ctx_ms * fresh_floor_multiplier)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            is_moe_model
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and ctx_prefix <= 0
            and tp_size <= 2
            and context_chunk_size is not None
            and 0 < ctx_tokens <= int(
                context_chunk_size
                * (
                    2.0
                    if exact_wide_ep_noop_context and tp_size <= 2
                    else 1.25
                )
            )
            and gen_tokens > 0
            and (
                tp_size <= 1
                or ctx_tokens >= max(384, context_chunk_size // 4 - 16)
            )
            and (
                moe_ep_size > 1
                or moe_tp_size <= 1
                or ctx_tokens >= max(512, context_chunk_size // 4)
            )
        ):
            try:
                saturated_ctx_detail = _saturated_context_step_detail(context_chunk_size * 2)
                if not self._layerwise_scheduler_timed_detail(saturated_ctx_detail):
                    raise PerfDataNotAvailableError("low-TP MoE mixed floor row is composed")
                saturated_ctx_ms = float(saturated_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
                if moe_ep_size > 1 and moe_tp_size <= 1:
                    low_tp_floor_multiplier = 0.60 if tp_size <= 1 else 0.50
                    if (
                        tp_size > 1
                        and moe_ep_size > tp_size
                        and attention_dp_size >= max(1, tp_size)
                    ):
                        low_tp_floor_multiplier = 0.75
                        if (
                            exact_wide_ep_noop_context
                            and tp_size <= 2
                            and ctx_tokens < context_chunk_size * 2
                        ):
                            low_tp_floor_multiplier = 0.98
                        if (
                            not has_shared_expert
                            and moe_ep_size >= 8
                            and ctx_prefix <= 0
                            and ctx_requests <= 1
                            and ctx_tokens >= max(1024, int(context_chunk_size * 0.33))
                        ):
                            low_tp_floor_multiplier = 0.67 if gen_tokens > small_decode_limit else 0.87
                else:
                    low_tp_floor_multiplier = 0.42
                _raise_ep_high_decode_floor(saturated_ctx_ms * low_tp_floor_multiplier)
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                pass

        if (
            is_moe_model
            and not context_detail_includes_moe
            and moe_tp_size > 1
            and moe_ep_size <= 1
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and 0 < ctx_tokens <= 128
            and gen_tokens > 0
        ):
            _raise_ep_high_decode_floor(combined_ctx_ms * 1.06)

        if (
            is_moe_model
            and not context_detail_includes_moe
            and moe_tp_size > 1
            and moe_ep_size <= 1
            and tp_size >= 4
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and context_chunk_size // 16 < ctx_tokens <= int(context_chunk_size * 0.19)
            and gen_tokens > small_decode_limit
            and avg_decode_kv >= context_chunk_size
        ):
            combined_ctx_ms *= 1.12

        if (
            is_moe_model
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and not has_shared_expert
            and moe_ep_size > max(1, tp_size)
            and moe_tp_size <= 1
            and tp_size > 1
            and context_chunk_size is not None
            and gen_tokens > small_decode_limit
            and (
                (tp_size == 2 and 0 < ctx_tokens < context_chunk_size // 8)
                or (tp_size >= 4 and 0 < ctx_tokens < context_chunk_size // 4)
            )
        ):
            combined_ctx_ms *= 1.24

        if (
            is_moe_model
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and not has_shared_expert
            and moe_ep_size >= 8
            and moe_ep_size > max(1, tp_size)
            and moe_tp_size <= 1
            and tp_size == 2
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens >= context_chunk_size * 3
            and gen_tokens > 0
        ):
            combined_ctx_ms *= 0.80

        if (
            ep_high_decode_floor_ms > 0.0
            and moe_tp_size > 1
            and moe_ep_size <= 1
            and ctx_prefix > 0
            and context_chunk_size is not None
            and ctx_requests >= 4
            and ctx_tokens >= int(context_chunk_size * 3.4)
            and gen_tokens > topk
        ):
            ep_high_decode_floor_ms *= 1.08

        if (
            decode_delta_ms <= 0.0
            and noop_moe_context_covers_mixed_decode
            and not subquadratic_context_attention
            and mixed_ep_context_covers_tp
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and 128 <= ctx_tokens <= min(512, max(128, context_chunk_size // 4))
            and gen_tokens > small_decode_limit
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                gen_step_ms, _ = _nearby_gen_step_ms(allow_standard_kv_fallback=True)
                tail_fraction = min(0.95, max(0.0, 0.45 + ctx_tokens / max(1.0, context_chunk_size / 2.0)))
                if exact_wide_ep_noop_context:
                    if tp_size <= 2 and moe_ep_size <= max(1, tp_size) * 2:
                        tail_fraction = min(tail_fraction, 0.75)
                    else:
                        tail_fraction = min(tail_fraction, 0.23 if tp_size >= 4 else 0.25)
                decode_delta_ms = gen_step_ms * tail_fraction
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no fresh no-op MoE EP mixed decode tail for model=%s, "
                    "tp_size=%s, gen_tokens=%s",
                    model_name,
                    tp_size,
                    gen_tokens,
                )

        if (
            decode_delta_ms <= 0.0
            and scheduler_like_context
            and is_moe_model
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and not has_shared_expert
            and moe_ep_size >= 8
            and moe_ep_size > max(1, tp_size)
            and moe_tp_size <= 1
            and tp_size == 2
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and 128 <= ctx_tokens < 256
            and gen_tokens >= small_decode_limit
            and avg_decode_kv >= context_chunk_size
        ):
            try:
                gen_step_ms, _ = _nearby_gen_step_ms(
                    allow_standard_kv_fallback=True,
                    batch_size_override=1,
                )
                decode_tail_fraction = 1.55
                if exact_wide_ep_noop_context:
                    decode_tail_fraction = 0.25
                decode_delta_ms = gen_step_ms * decode_tail_fraction
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no tiny fresh no-op MoE EP mixed decode tail for model=%s, "
                    "tp_size=%s, ep_size=%s, gen_tokens=%s",
                    model_name,
                    tp_size,
                    moe_ep_size,
                    gen_tokens,
                )

        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        moe_router_ms, moe_router_energy, moe_router_source = 0.0, 0.0, "silicon"
        moe_shared_ms, moe_shared_energy, moe_shared_source = 0.0, 0.0, "silicon"
        moe_tp_allreduce_ms = 0.0
        moe_ep_alltoall_ms = 0.0
        small_fresh_noop_moe_cover_tokens = 256
        if context_chunk_size is not None:
            small_fresh_noop_moe_cover_tokens = min(512, max(256, context_chunk_size // 4))
        tiny_fresh_ep_mixed_covers_noop_moe = (
            scheduler_like_context
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and ctx_prefix <= 0
            and 0 < ctx_tokens <= small_fresh_noop_moe_cover_tokens
            and gen_tokens > 0
        )
        high_decode_fresh_ep_mixed_covers_noop_moe = (
            scheduler_like_context
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and subquadratic_context_attention
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens >= context_chunk_size
            and gen_tokens >= max(4, min(small_decode_limit, 7))
            and (
                avg_decode_kv * 10 >= context_chunk_size * 19
                or attention_dp_size < max(1, tp_size)
            )
            and not (
                int(combined_detail.get("_context_lookup_tp_size", tp_size)) > tp_size
                and tp_size <= 1
                and 512 < ctx_tokens < 2048
            )
        )
        large_fresh_shared_ep_context_covers_noop_moe = (
            subquadratic_context_attention
            and is_moe_model
            and not context_detail_includes_moe
            and has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens >= context_chunk_size * 2
            and (attention_dp_size <= max(1, tp_size) or ctx_tokens > context_chunk_size * 2)
            and gen_tokens > 0
            and not (
                int(combined_detail.get("_context_lookup_tp_size", tp_size)) > tp_size
                and tp_size <= 1
                and 512 < ctx_tokens < 2048
            )
        )
        large_fresh_ep_context_covers_noop_moe = (
            scheduler_like_context
            and not subquadratic_context_attention
            and is_moe_model
            and not context_detail_includes_moe
            and not has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and tp_size <= 2
            and ctx_prefix <= 0
            and ctx_requests <= 1
            and context_chunk_size is not None
            and ctx_tokens >= int(context_chunk_size * 1.5)
            and gen_tokens > 0
        )
        tiny_continuation_ep_mixed_covers_noop_moe = (
            scheduler_like_context
            and mixed_ep_context_covers_tp
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and not has_shared_expert
            and moe_ep_size > 1
            and moe_tp_size <= 1
            and ctx_prefix > 0
            and ctx_requests <= 1
            and 0 < ctx_tokens < small_fresh_noop_moe_cover_tokens
            and gen_tokens > small_decode_limit
        )
        moe_tp_continuation_context_covers_noop_moe = (
            scheduler_like_context
            and not context_detail_includes_moe
            and not subquadratic_context_attention
            and moe_tp_size > 1
            and moe_ep_size <= 1
            and ctx_prefix > 0
            and gen_tokens > 0
        )
        ep_mixed_context_covers_noop_moe = (
            tiny_fresh_ep_mixed_covers_noop_moe
            or high_decode_fresh_ep_mixed_covers_noop_moe
            or large_fresh_shared_ep_context_covers_noop_moe
            or large_fresh_ep_context_covers_noop_moe
            or tiny_continuation_ep_mixed_covers_noop_moe
            or moe_tp_continuation_context_covers_noop_moe
        )
        if not context_detail_includes_moe and not ep_mixed_context_covers_noop_moe:
            represented_layers = self._layerwise_detail_represented_noop_moe_layers(combined_detail, num_layers)
            (
                (moe_ms, moe_energy, moe_source),
                (moe_router_ms, moe_router_energy, moe_router_source),
                (moe_shared_ms, moe_shared_energy, moe_shared_source),
                moe_addback_is_bundled,
            ) = self._layerwise_noop_moe_addback(
                model,
                database,
                token_count=combined_tokens,
                num_layers=represented_layers,
                is_context=ctx_tokens > 0,
            )
            moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
            moe_addback_scale = 1.0
            if (
                moe_ms > 0.0
                and moe_tp_size > 1
                and int(getattr(model.config, "moe_ep_size", 1) or 1) <= 1
                and ctx_prefix <= 0
                and 0 < ctx_tokens < 512
                and gen_tokens > 0
            ):
                moe_addback_scale = min(0.75, max(0.0, (float(ctx_tokens) - 96.0) / 576.0))
                moe_ms *= moe_addback_scale
                moe_energy *= moe_addback_scale
                moe_router_ms *= moe_addback_scale
                moe_router_energy *= moe_addback_scale
                moe_shared_ms *= moe_addback_scale
                moe_shared_energy *= moe_addback_scale
            if (
                moe_ms > 0.0
                and attention_dp_widened_fresh_ep
                and subquadratic_context_attention
                and has_shared_expert
                and moe_ep_size > 1
                and moe_tp_size <= 1
                and ctx_prefix <= 0
                and ctx_requests <= 1
                and context_chunk_size is not None
                and ctx_tokens >= context_chunk_size
                and gen_tokens > small_decode_limit
                and avg_decode_kv * 10 < context_chunk_size * 19
            ):
                exposed_moe_fraction = 0.30
                moe_addback_scale *= exposed_moe_fraction
                moe_ms *= exposed_moe_fraction
                moe_energy *= exposed_moe_fraction
                moe_router_ms *= exposed_moe_fraction
                moe_router_energy *= exposed_moe_fraction
                moe_shared_ms *= exposed_moe_fraction
                moe_shared_energy *= exposed_moe_fraction
            if moe_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                try:
                    moe_tp_allreduce_ms = (
                        self._layerwise_tp_allreduce_ms(model, database, moe_tp_size, combined_tokens)
                        * represented_layers
                        * moe_addback_scale
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM mixed no-op MoE-TP allreduce for model=%s, "
                        "moe_tp_size=%s, token_count=%s because allreduce data is unavailable",
                        model_name,
                        moe_tp_size,
                        combined_tokens,
                    )
            moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
            if moe_ms > 0.0 and moe_ep_size > 1:
                try:
                    moe_ep_alltoall_ms = (
                        self._layerwise_moe_ep_alltoall_ms(model, database, moe_ep_size, combined_tokens)
                        * represented_layers
                        * moe_addback_scale
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM mixed no-op MoE-EP all-to-all for model=%s, "
                        "moe_ep_size=%s, token_count=%s because NCCL data is unavailable",
                        model_name,
                        moe_ep_size,
                        combined_tokens,
                    )
        latency_ms = combined_ctx_ms + combined_allreduce_ms + decode_delta_ms
        if (
            tp_size <= 1
            and combined_detail.get("parallel_fallback_moe_ep_size") not in (None, "")
            and not allow_parallel_fallback_ep_floor
        ):
            ep_high_decode_floor_ms = 0.0
        ep_high_decode_floor_applied = ep_high_decode_floor_ms > latency_ms
        if ep_high_decode_floor_applied:
            latency_ms = max(latency_ms, ep_high_decode_floor_ms)
        floor_covers_moe_addback = (
            ep_high_decode_floor_applied
            and gen_tokens > 0
            and (
                (
                    moe_tp_size > 1
                    and moe_ep_size <= 1
                    and ctx_prefix <= 0
                    and context_chunk_size is not None
                    and (
                        (gen_tokens > topk and ctx_tokens < context_chunk_size * 3)
                        or (gen_tokens <= topk and ctx_tokens <= int(context_chunk_size * 3.25))
                    )
                    and ctx_tokens + gen_tokens >= max(512, context_chunk_size // 4)
                )
                or (
                    moe_tp_size <= 1
                    and moe_ep_size <= 1
                    and noop_moe_context_covers_mixed_decode
                    and not has_shared_expert
                )
                or (
                    has_shared_expert
                    and moe_ep_size > 1
                    and moe_tp_size <= 1
                    and ctx_prefix <= 0
                    and context_chunk_size is not None
                    and 128 <= ctx_tokens <= small_fresh_noop_moe_cover_tokens
                )
                or (
                    not has_shared_expert
                    and moe_ep_size > 1
                    and moe_tp_size <= 1
                    and ctx_prefix <= 0
                    and ctx_requests <= 1
                    and context_chunk_size is not None
                    and (
                        fresh_parallel_fallback_ep_floor
                        or (
                            tp_size > 1
                            and moe_ep_size > tp_size
                            and 0 < ctx_tokens < max(768, context_chunk_size // 2)
                        )
                    )
                )
                or (
                    not has_shared_expert
                    and moe_ep_size > 1
                    and moe_tp_size <= 1
                    and tp_size <= 2
                    and ctx_prefix > 0
                    and (ctx_requests > 1 or tp_size > 1)
                    and context_chunk_size is not None
                    and ctx_tokens < context_chunk_size
                )
            )
        )
        if floor_covers_moe_addback:
            moe_ms = 0.0
            moe_tp_allreduce_ms = 0.0
            moe_ep_alltoall_ms = 0.0
            moe_router_ms = 0.0
            moe_shared_ms = 0.0
        per_ops = {
            "mixed_layerwise_context_combined": combined_ctx_ms,
            "mixed_layerwise_context_tp_allreduce": combined_allreduce_ms,
            "mixed_layerwise_decode_delta": decode_delta_ms,
        }
        if ep_high_decode_floor_applied:
            per_ops["mixed_layerwise_ep_high_decode_floor"] = ep_high_decode_floor_ms
        per_ops_source = dict.fromkeys(per_ops, "silicon")
        if moe_ms > 0.0:
            latency_ms += moe_ms
            per_ops["mixed_moe"] = moe_ms
            per_ops_source["mixed_moe"] = moe_source
        if moe_tp_allreduce_ms > 0.0:
            latency_ms += moe_tp_allreduce_ms
            per_ops["mixed_moe_tp_allreduce"] = moe_tp_allreduce_ms
            per_ops_source["mixed_moe_tp_allreduce"] = "silicon"
        if moe_ep_alltoall_ms > 0.0:
            latency_ms += moe_ep_alltoall_ms
            per_ops["mixed_moe_ep_alltoall"] = moe_ep_alltoall_ms
            per_ops_source["mixed_moe_ep_alltoall"] = "silicon"
        if moe_router_ms > 0.0:
            latency_ms += moe_router_ms
            moe_energy += moe_router_energy
            per_ops["mixed_moe_router"] = moe_router_ms
            per_ops_source["mixed_moe_router"] = moe_router_source
        if moe_shared_ms > 0.0:
            latency_ms += moe_shared_ms
            moe_energy += moe_shared_energy
            per_ops["mixed_moe_shared_expert"] = moe_shared_ms
            per_ops_source["mixed_moe_shared_expert"] = moe_shared_source
        if (
            "DeepSeek-V4" in str(model_name)
            and ctx_tokens > 0
            and gen_tokens > 0
            and moe_ep_size <= 1
            and scheduler_like_context
            and not context_detail_includes_moe
            and not small_subquadratic_continuation
        ):
            try:
                self._layerwise_mixed_context_floor_active = True
                try:
                    context_floor_latency, _, _ = self._get_context_step_latency(
                        model,
                        database,
                        runtime_config,
                        ctx_tokens=ctx_tokens,
                        ctx_kv_tokens=ctx_prefix,
                        ctx_requests=1,
                    )
                finally:
                    self._layerwise_mixed_context_floor_active = False
                context_floor_ms = float(sum(context_floor_latency.values())) * 1.05
                if context_floor_ms > latency_ms:
                    latency_ms = context_floor_ms
                    per_ops["mixed_layerwise_context_floor"] = context_floor_ms
                    per_ops_source["mixed_layerwise_context_floor"] = "silicon"
            except (AssertionError, KeyError, PerfDataNotAvailableError, ValueError):
                pass
        return latency_ms, moe_energy, per_ops, per_ops_source
