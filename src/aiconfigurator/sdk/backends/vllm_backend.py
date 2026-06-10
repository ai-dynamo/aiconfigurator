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
_VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192


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
    ) -> float:
        return float(
            self._layerwise_context_layer_detail(database, model_name, tp_size, batch_size, seq_len, prefix)["latency"]
        )

    def _layerwise_context_layer_detail(
        self,
        database: PerfDatabase,
        model_name: str,
        tp_size: int,
        batch_size: int,
        seq_len: int,
        prefix: int = 0,
    ) -> dict[str, float | bool]:
        if seq_len <= _VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS:
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
                chunk_tokens = min(_VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS, remaining)
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

    def _layerwise_tp_allreduce_ms(
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
        return max(0, int(round(min(raw_multiplier, float(fallback_num_layers)))))

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
        workload_distribution = getattr(cfg, "workload_distribution", "power_law")
        if workload_distribution == "power_law":
            workload_distribution = f"power_law_{getattr(model, '_power_law_alpha', 1.2)}"

        result = database.query_moe(
            num_tokens=int(token_count) * int(getattr(cfg, "attention_dp_size", 1) or 1),
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
        layer_detail = self._layerwise_context_layer_detail(
            database, model_name, tp_size, batch_size, effective_isl, prefix
        )
        layer_scale = self._layerwise_detail_scale(layer_detail, num_layers)
        layer_ms = float(layer_detail["latency"]) * layer_scale
        allreduce_ms = (
            self._layerwise_tp_allreduce_ms(model, database, tp_size, batch_size * effective_isl) * 2 * num_layers
        )
        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        if not bool(layer_detail.get("includes_moe", False)):
            moe_ms, moe_energy, moe_source = self._layerwise_moe_compute(
                model,
                database,
                token_count=batch_size * effective_isl,
                num_layers=self._layerwise_detail_represented_layers(layer_detail, num_layers),
                is_context=True,
            )

        latency_dict = defaultdict(float, {"context_layerwise": layer_ms, "context_tp_allreduce": allreduce_ms})
        energy_dict = defaultdict(float, {"context_layerwise": 0.0, "context_tp_allreduce": 0.0})
        source_dict = {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}
        if moe_ms > 0.0:
            latency_dict["context_moe"] = moe_ms
            energy_dict["context_moe"] = moe_energy
            source_dict["context_moe"] = moe_source
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
        fused_allreduce_rms_total = 0.0
        moe_ms_total = 0.0
        moe_energy_total = 0.0
        moe_source: str | None = None
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
            rms_step_ms = float(layer_detail.get("rms_latency", 0.0)) * layer_scale
            allreduce_rms_step_ms = 0.0
            if tp_size > 1 and rms_step_ms > 0.0:
                try:
                    allreduce_rms_step_ms = (
                        self._layerwise_tp_allreduce_rms_ms(model, database, tp_size, effective_bs) * 2 * num_layers
                    )
                    if allreduce_rms_step_ms <= 0.0:
                        rms_step_ms = 0.0
                except (PerfDataNotAvailableError, ValueError):
                    logger.debug(
                        "Falling back to unadjusted vLLM GEN layerwise row for model=%s, tp_size=%s, "
                        "batch_size=%s, kv_len=%s because allreduce_rms data is unavailable",
                        model_name,
                        tp_size,
                        effective_bs,
                        kv_len,
                    )
                    rms_step_ms = 0.0
            else:
                rms_step_ms = 0.0
            repeat_count = min(stride, osl - 1 - i)
            layer_ms_total += max(0.0, layer_step_ms - rms_step_ms) * repeat_count
            fused_allreduce_rms_total += allreduce_rms_step_ms * repeat_count
            moe_step_ms, moe_step_energy, moe_step_source = 0.0, 0.0, "silicon"
            if not bool(layer_detail.get("includes_moe", False)):
                moe_step_ms, moe_step_energy, moe_step_source = self._layerwise_moe_compute(
                    model,
                    database,
                    token_count=effective_bs,
                    num_layers=self._layerwise_detail_represented_layers(layer_detail, num_layers),
                    is_context=False,
                )
            moe_ms_total += moe_step_ms * repeat_count
            moe_energy_total += moe_step_energy * repeat_count
            if moe_source is None or moe_source == moe_step_source:
                moe_source = moe_step_source
            else:
                moe_source = "mixed"

        latency_dict = defaultdict(float, {"generation_layerwise": layer_ms_total})
        energy_dict = defaultdict(float, {"generation_layerwise": 0.0})
        source_dict = {"generation_layerwise": "silicon"}
        if fused_allreduce_rms_total > 0.0:
            latency_dict["generation_tp_allreduce_rms"] = fused_allreduce_rms_total
            energy_dict["generation_tp_allreduce_rms"] = 0.0
            source_dict["generation_tp_allreduce_rms"] = "silicon"
        if moe_ms_total > 0.0:
            latency_dict["generation_moe"] = moe_ms_total
            energy_dict["generation_moe"] = moe_energy_total
            source_dict["generation_moe"] = moe_source or "silicon"
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

        del runtime_config, prefix
        model_name = getattr(model, "model_path", "")
        tp_size = model.config.tp_size
        num_layers = model._num_layers // model.config.pp_size
        ctx_tokens = max(int(ctx_tokens), 0)
        gen_tokens = max(int(gen_tokens), 0)
        combined_tokens = max(ctx_tokens + gen_tokens, 1)

        combined_detail = self._layerwise_context_layer_detail(database, model_name, tp_size, 1, combined_tokens)
        combined_ctx_ms = float(combined_detail["latency"]) * self._layerwise_detail_scale(
            combined_detail, num_layers
        )
        combined_allreduce_ms = (
            self._layerwise_tp_allreduce_ms(model, database, tp_size, combined_tokens) * 2 * num_layers
        )
        decode_delta_ms = 0.0
        if gen_tokens > 0:
            avg_decode_kv = isl + osl // 2
            try:
                gen_detail = self._query_layerwise_detail(
                    database,
                    model_name,
                    "GEN",
                    tp_size,
                    gen_tokens,
                    avg_decode_kv,
                )
                gen_step_ms = (
                    float(gen_detail["latency"])
                    * self._layerwise_detail_scale(gen_detail, num_layers)
                )
                gen_as_ctx_detail = self._layerwise_context_layer_detail(database, model_name, tp_size, 1, gen_tokens)
                gen_as_ctx_ms = float(gen_as_ctx_detail["latency"]) * self._layerwise_detail_scale(
                    gen_as_ctx_detail, num_layers
                )
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

        moe_ms, moe_energy, moe_source = 0.0, 0.0, "silicon"
        if not bool(combined_detail.get("includes_moe", False)):
            moe_ms, moe_energy, moe_source = self._layerwise_moe_compute(
                model,
                database,
                token_count=combined_tokens,
                num_layers=self._layerwise_detail_represented_layers(combined_detail, num_layers),
                is_context=ctx_tokens > 0,
            )
        latency_ms = combined_ctx_ms + combined_allreduce_ms + decode_delta_ms
        per_ops = {
            "mixed_layerwise_context_combined": combined_ctx_ms,
            "mixed_layerwise_context_tp_allreduce": combined_allreduce_ms,
            "mixed_layerwise_decode_delta": decode_delta_ms,
        }
        per_ops_source = dict.fromkeys(per_ops, "silicon")
        if moe_ms > 0.0:
            latency_ms += moe_ms
            per_ops["mixed_moe"] = moe_ms
            per_ops_source["mixed_moe"] = moe_source
        return latency_ms, moe_energy, per_ops, per_ops_source
