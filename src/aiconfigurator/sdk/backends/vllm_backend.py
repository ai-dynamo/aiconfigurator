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
from aiconfigurator.sdk.perf_database import PerfDatabase, PerfDataNotAvailableError

logger = logging.getLogger(__name__)

_USE_LAYERWISE = os.environ.get("AIC_VLLM_USE_LAYERWISE", "0") == "1"
_QWEN_MODULE_MOE_DISTRIBUTION = "qwen_module_tp_block"
_QWEN_NOOP_MOE_DISTRIBUTION = "sampled_zipf_0.8"


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
                    )
                )
                past_kv += chunk_tokens
                remaining -= chunk_tokens
            return {
                "latency": sum(float(detail["latency"]) for detail in chunk_details),
                "energy": sum(float(detail.get("energy", 0.0)) for detail in chunk_details),
                "rms_latency": sum(float(detail.get("rms_latency", 0.0)) for detail in chunk_details),
                "includes_moe": all(bool(detail.get("includes_moe", False)) for detail in chunk_details),
                "layer_type": chunk_details[0].get("layer_type", ""),
                "layer_index": float(chunk_details[0].get("layer_index", 0.0)),
                "measured_layer_count": float(chunk_details[0].get("measured_layer_count", 1.0)),
                "layer_multiplier": float(chunk_details[0].get("layer_multiplier", 0.0)),
            }
        except (PerfDataNotAvailableError, ValueError):
            # Older layerwise datasets have a direct 16k CTX row but no
            # nonzero-context-KV chunk rows. Use that row until chunked data
            # exists for the requested TP.
            return self._query_layerwise_detail(database, model_name, "CTX", tp_size, batch_size, seq_len)

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

    def _layerwise_context_detail_for_runtime(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int,
        runtime_config: RuntimeConfig,
    ) -> dict[str, float | bool]:
        """Return context layer detail with the runtime chunking policy."""

        chunk_size = self._layerwise_context_chunk_size(runtime_config)
        detail = self._layerwise_context_layer_detail(
            database,
            model_name,
            tp_size,
            batch_size,
            seq_len,
            prefix,
            max_num_batched_tokens=chunk_size,
        )
        inferred_chunk_size = self._layerwise_context_chunk_size(runtime_config, detail)
        if chunk_size is None and inferred_chunk_size is not None and seq_len > inferred_chunk_size:
            detail = self._layerwise_context_layer_detail(
                database,
                model_name,
                tp_size,
                batch_size,
                seq_len,
                prefix,
                max_num_batched_tokens=inferred_chunk_size,
            )
        return detail

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
    ) -> dict[str, float]:
        if hasattr(database, "query_layerwise_detail"):
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

        message_size = max(1, int(token_count * hidden_size * topk / ep_size))
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
        workload_distribution = self._layerwise_moe_workload_distribution(
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
            return str(workload_distribution)

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

        moe = self._layerwise_moe_compute(
            model,
            database,
            token_count=token_count,
            num_layers=num_layers,
            is_context=is_context,
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
        return moe, router, shared, False

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

        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        try:
            layer_detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                tp_size,
                batch_size,
                effective_isl,
                prefix,
                runtime_config,
            )
        except (AssertionError, PerfDataNotAvailableError, ValueError):
            if prefix <= 0:
                raise
            layer_detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                tp_size,
                batch_size,
                effective_isl,
                0,
                runtime_config,
            )
        structural_moe_context_ms = self._layerwise_structural_moe_context_ms(layer_detail, num_layers)
        if structural_moe_context_ms is None:
            layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
            layer_ms = float(layer_detail["latency"]) * layer_scale
        else:
            layer_ms = structural_moe_context_ms
        context_source = str(layer_detail.get("latency_source") or "")
        token_count = batch_size * effective_isl
        layer_includes_moe = bool(layer_detail.get("includes_moe", False))
        physical_gpus = int(float(layer_detail.get("physical_gpus", 1.0) or 1.0))
        context_source_covers_tp = physical_gpus >= tp_size
        if context_source in {"schedule_to_update", "fpm_wall", "worker_wall"} and context_source_covers_tp:
            allreduce_ms = 0.0
        else:
            allreduce_ms = (
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
        moe_tp_allreduce_ms = 0.0
        moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)
        if represented_moe_layers > 0 and moe_tp_size > 1 and physical_gpus < moe_tp_size:
            try:
                moe_tp_allreduce_ms = (
                    self._layerwise_tp_allreduce_ms(
                        model,
                        database,
                        moe_tp_size,
                        token_count,
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
        if structural_moe_context_ms is None and not layer_includes_moe:
            represented_layers = self._layerwise_detail_represented_noop_moe_layers(layer_detail, num_layers)
            (
                (moe_ms, moe_energy, moe_source),
                (moe_router_ms, moe_router_energy, moe_router_source),
                (moe_shared_ms, moe_shared_energy, moe_shared_source),
                moe_addback_is_bundled,
            ) = self._layerwise_noop_moe_addback(
                model,
                database,
                token_count=token_count,
                num_layers=represented_layers,
                is_context=True,
            )
            if moe_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                try:
                    moe_tp_allreduce_ms = (
                        self._layerwise_tp_allreduce_ms(
                            model,
                            database,
                            moe_tp_size,
                            token_count,
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
            and physical_gpus < moe_ep_size
        )
        if has_external_moe_ep:
            try:
                moe_ep_alltoall_ms = (
                    self._layerwise_moe_ep_alltoall_ms(
                        model,
                        database,
                        moe_ep_size,
                        token_count,
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
        elif moe_ep_alltoall_layers > 0 and moe_ep_size > 1 and moe_ms > 0.0:
            try:
                moe_ep_alltoall_ms = (
                    self._layerwise_moe_ep_alltoall_ms(
                        model,
                        database,
                        moe_ep_size,
                        token_count,
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
        if moe_shared_ms > 0.0:
            latency_dict["context_moe_shared_expert"] = moe_shared_ms
            energy_dict["context_moe_shared_expert"] = moe_shared_energy
            source_dict["context_moe_shared_expert"] = moe_shared_source
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
            )
            layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
            layer_step_ms = float(layer_detail["latency"]) * layer_scale
            layer_includes_moe = bool(layer_detail.get("includes_moe", False))
            moe_tp_size = int(getattr(model.config, "moe_tp_size", 1) or 1)
            moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
            physical_gpus = int(float(layer_detail.get("physical_gpus", 1.0) or 1.0))
            represented_moe_layers = self._layerwise_detail_represented_moe_layers(layer_detail, num_layers)
            # vLLM layerwise GEN rows measure the transformer block execution
            # envelope. The block already contains RMS work, so adding a
            # separate fused all-reduce+RMS term double-counts decode latency.
            rms_step_ms = 0.0
            allreduce_rms_step_ms = 0.0
            try:
                standalone_allreduce_ms = self._layerwise_tp_allreduce_ms(model, database, tp_size, effective_bs)
                try:
                    fused_allreduce_ms = self._layerwise_tp_allreduce_rms_ms(
                        model, database, tp_size, effective_bs
                    )
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
                    tp_allreduce_step_ms = fused_allreduce_ms * num_layers
                else:
                    tp_allreduce_step_ms = (standalone_allreduce_ms + fused_allreduce_ms) * num_layers
            except (AttributeError, PerfDataNotAvailableError, ValueError):
                logger.debug(
                    "Falling back to no explicit vLLM GEN TP allreduce for model=%s, tp_size=%s, "
                    "batch_size=%s because allreduce data is unavailable",
                    model_name,
                    tp_size,
                    effective_bs,
                )
                tp_allreduce_step_ms = 0.0
            moe_tp_allreduce_step_ms = 0.0
            # Full-MoE vLLM decode rows already measure the MoE block envelope.
            # Adding another expert-TP collective double-counts Qwen3.6 TP2
            # decode; keep explicit MoE-TP add-back for no-op rows below.
            moe_ep_alltoall_step_ms = 0.0
            if (
                not layer_includes_moe
                and represented_moe_layers > 0
                and moe_ep_size > 1
                and physical_gpus < moe_ep_size
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
            if not layer_includes_moe:
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
                if moe_step_ms > 0.0 and moe_ep_size > 1:
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
        ctx_tokens = max(int(ctx_tokens), 0)
        gen_tokens = max(int(gen_tokens), 0)
        combined_tokens = max(ctx_tokens + gen_tokens, 1)

        ctx_prefix = max(int(prefix), 0)
        used_context_tokens = combined_tokens
        try:
            combined_detail = self._layerwise_context_detail_for_runtime(
                database,
                model_name,
                tp_size,
                1,
                combined_tokens,
                ctx_prefix,
                runtime_config,
            )
        except (AssertionError, PerfDataNotAvailableError, ValueError):
            if ctx_tokens > 0 and ctx_tokens != combined_tokens:
                try:
                    combined_detail = self._layerwise_context_detail_for_runtime(
                        database,
                        model_name,
                        tp_size,
                        1,
                        ctx_tokens,
                        ctx_prefix,
                        runtime_config,
                    )
                    used_context_tokens = ctx_tokens
                except (AssertionError, PerfDataNotAvailableError, ValueError):
                    if ctx_prefix <= 0:
                        raise
                    combined_detail = self._layerwise_context_detail_for_runtime(
                        database,
                        model_name,
                        tp_size,
                        1,
                        combined_tokens,
                        0,
                        runtime_config,
                    )
            elif ctx_prefix > 0:
                combined_detail = self._layerwise_context_detail_for_runtime(
                    database,
                    model_name,
                    tp_size,
                    1,
                    combined_tokens,
                    0,
                    runtime_config,
                )
            else:
                raise
        combined_source = str(combined_detail.get("latency_source") or "")
        if (
            ctx_prefix > 0
            and used_context_tokens == combined_tokens
            and combined_source not in {"schedule_to_update", "fpm_wall", "worker_wall"}
        ):
            try:
                fresh_combined_detail = self._layerwise_context_detail_for_runtime(
                    database,
                    model_name,
                    tp_size,
                    1,
                    combined_tokens,
                    0,
                    runtime_config,
                )
                fresh_source = str(fresh_combined_detail.get("latency_source") or "")
                fresh_ctx_ms = float(fresh_combined_detail["latency"]) * self._layerwise_detail_scale(
                    fresh_combined_detail, num_layers
                )
                combined_ctx_ms = float(combined_detail["latency"]) * self._layerwise_detail_scale(
                    combined_detail, num_layers
                )
                use_fresh_scheduler_surface = fresh_source in {"schedule_to_update", "fpm_wall", "worker_wall"}
                if not use_fresh_scheduler_surface:
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
        physical_gpus = int(float(combined_detail.get("physical_gpus", 1.0) or 1.0))
        context_source_covers_tp = physical_gpus >= tp_size
        scheduler_like_context = combined_source in {"schedule_to_update", "fpm_wall", "worker_wall"}
        extra_params = getattr(model, "extra_params", None)
        is_moe_model = (
            int(getattr(extra_params, "num_experts", 0) or getattr(model, "_num_experts", 0) or 0) > 0
        )
        moe_ep_size = int(getattr(model.config, "moe_ep_size", 1) or 1)
        # In mixed prefill+decode iterations for expert-parallel MoE, the
        # scheduler-envelope context row tracks the same overlapped iteration
        # boundary as FPM. Adding an isolated dense TP allreduce on top
        # double-counts the comm tail for large mixed chunks.
        mixed_ep_context_covers_tp = scheduler_like_context and ctx_tokens > 0 and moe_ep_size > 1
        mixed_moe_context_covers_tp = (
            scheduler_like_context
            and ctx_tokens > 0
            and is_moe_model
            and (context_source_covers_tp or mixed_ep_context_covers_tp)
        )
        if scheduler_like_context and (context_source_covers_tp or mixed_ep_context_covers_tp):
            combined_allreduce_ms = 0.0
        else:
            combined_allreduce_ms = (
                self._layerwise_tp_allreduce_ms(model, database, tp_size, combined_tokens) * 2 * num_layers
            )
        decode_delta_ms = 0.0
        ep_high_decode_floor_ms = 0.0
        dense_context_envelope_covers_mixed_decode = (
            scheduler_like_context and ctx_tokens >= 128 and not is_moe_model
        )
        context_envelope_covers_mixed_decode = (
            dense_context_envelope_covers_mixed_decode
            or (ctx_tokens >= 128 and mixed_moe_context_covers_tp)
        )
        avg_decode_kv = isl + osl // 2
        topk = int(getattr(model, "_topk", 0) or 0)

        def _nearby_gen_step_ms(*, allow_standard_kv_fallback: bool = False) -> tuple[float, int]:
            gen_detail = None
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
                        gen_tokens,
                        candidate_kv,
                    )
                    break
                except (PerfDataNotAvailableError, ValueError):
                    continue
            if gen_detail is None:
                raise PerfDataNotAvailableError(
                    f"No nearby GEN layerwise row for tp_size={tp_size}, "
                    f"batch_size={gen_tokens}, kv={avg_decode_kv}"
                )
            return (
                float(gen_detail["latency"]) * self._layerwise_detail_scale(gen_detail, num_layers),
                avg_decode_kv,
            )

        if gen_tokens > 0 and not context_envelope_covers_mixed_decode:
            try:
                gen_step_ms, avg_decode_kv = _nearby_gen_step_ms()
                try:
                    gen_as_ctx_detail = self._layerwise_context_detail_for_runtime(
                        database,
                        model_name,
                        tp_size,
                        1,
                        gen_tokens,
                        0,
                        runtime_config,
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
        elif gen_tokens >= 4 and mixed_ep_context_covers_tp:
            try:
                context_chunk_size = self._layerwise_context_chunk_size(runtime_config, combined_detail)
                full_prefill_overlap_fraction = 0.75 if moe_ep_size >= 8 else 0.6
                fresh_full_prefill_chunk = (
                    topk > 0
                    and gen_tokens > topk
                    and ctx_prefix == 0
                    and context_chunk_size is not None
                    and ctx_tokens >= context_chunk_size * full_prefill_overlap_fraction
                )
                if fresh_full_prefill_chunk:
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
                            saturated_ctx_detail = self._layerwise_context_detail_for_runtime(
                                database,
                                model_name,
                                tp_size,
                                1,
                                context_chunk_size or combined_tokens,
                                0,
                                runtime_config,
                            )
                            saturated_ctx_ms = float(saturated_ctx_detail["latency"])
                            ep_high_decode_floor_ms = saturated_ctx_ms * self._layerwise_detail_scale(
                                saturated_ctx_detail,
                                num_layers,
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

        if ep_high_decode_floor_ms <= 0.0 and topk > 0 and gen_tokens > topk and mixed_ep_context_covers_tp:
            try:
                context_chunk_size = self._layerwise_context_chunk_size(runtime_config, combined_detail)
                saturated_ctx_detail = self._layerwise_context_detail_for_runtime(
                    database,
                    model_name,
                    tp_size,
                    1,
                    context_chunk_size or combined_tokens,
                    0,
                    runtime_config,
                )
                saturated_ctx_ms = float(saturated_ctx_detail["latency"])
                ep_high_decode_floor_ms = saturated_ctx_ms * self._layerwise_detail_scale(
                    saturated_ctx_detail,
                    num_layers,
                )
            except (AssertionError, PerfDataNotAvailableError, ValueError):
                ep_high_decode_floor_ms = 0.0

        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        moe_router_ms, moe_router_energy, moe_router_source = 0.0, 0.0, "silicon"
        moe_shared_ms, moe_shared_energy, moe_shared_source = 0.0, 0.0, "silicon"
        moe_tp_allreduce_ms = 0.0
        if not bool(combined_detail.get("includes_moe", False)):
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
            if moe_ms > 0.0 and moe_tp_size > 1 and not moe_addback_is_bundled:
                try:
                    moe_tp_allreduce_ms = (
                        self._layerwise_tp_allreduce_ms(model, database, moe_tp_size, combined_tokens)
                        * represented_layers
                    )
                except (AttributeError, PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to no explicit vLLM mixed no-op MoE-TP allreduce for model=%s, "
                        "moe_tp_size=%s, token_count=%s because allreduce data is unavailable",
                        model_name,
                        moe_tp_size,
                        combined_tokens,
                    )
        latency_ms = combined_ctx_ms + combined_allreduce_ms + decode_delta_ms
        ep_high_decode_floor_applied = ep_high_decode_floor_ms > latency_ms
        if ep_high_decode_floor_applied:
            latency_ms = max(latency_ms, ep_high_decode_floor_ms)
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
        return latency_ms, moe_energy, per_ops, per_ops_source
