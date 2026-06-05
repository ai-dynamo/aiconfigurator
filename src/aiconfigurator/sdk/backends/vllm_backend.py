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
from aiconfigurator.sdk.perf_database import PerfDataNotAvailableError, PerfDatabase

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
        if seq_len <= _VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS:
            return float(
                database.query_layerwise(
                    model_name,
                    "CTX",
                    tp_size,
                    batch_size,
                    seq_len,
                    seq_len_kv_cache=max(prefix, 0),
                )
            )

        chunk_latencies = []
        remaining = seq_len
        past_kv = max(prefix, 0)
        try:
            while remaining > 0:
                chunk_tokens = min(_VLLM_DEFAULT_MAX_NUM_BATCHED_TOKENS, remaining)
                chunk_latencies.append(
                    float(
                        database.query_layerwise(
                            model_name,
                            "CTX",
                            tp_size,
                            batch_size,
                            chunk_tokens,
                            seq_len_kv_cache=past_kv,
                        )
                    )
                )
                past_kv += chunk_tokens
                remaining -= chunk_tokens
            return sum(chunk_latencies)
        except PerfDataNotAvailableError:
            # Older layerwise datasets have a direct 16k CTX row but no
            # nonzero-context-KV chunk rows. Use that row until chunked data
            # exists for the requested TP.
            return float(database.query_layerwise(model_name, "CTX", tp_size, batch_size, seq_len))

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
        layer_ms = (
            self._layerwise_context_layer_ms(database, model_name, tp_size, batch_size, effective_isl, prefix)
            * num_layers
        )
        allreduce_ms = (
            self._layerwise_tp_allreduce_ms(model, database, tp_size, batch_size * effective_isl) * 2 * num_layers
        )

        latency_dict = defaultdict(float, {"context_layerwise": layer_ms, "context_tp_allreduce": allreduce_ms})
        energy_dict = defaultdict(float, {"context_layerwise": 0.0, "context_tp_allreduce": 0.0})
        source_dict = {"context_layerwise": "silicon", "context_tp_allreduce": "silicon"}
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
        for i in range(0, osl - 1, stride):
            kv_len = isl + i + 1
            layer_step_ms = (
                float(database.query_layerwise(model_name, "GEN", tp_size, effective_bs, kv_len)) * num_layers
            )
            allreduce_step_ms = (
                self._layerwise_tp_allreduce_ms(model, database, tp_size, effective_bs) * 2 * num_layers
            )
            repeat_count = min(stride, osl - 1 - i)
            layer_ms_total += (layer_step_ms + allreduce_step_ms) * repeat_count

        latency_dict = defaultdict(
            float,
            {
                "generation_layerwise": layer_ms_total,
            },
        )
        energy_dict = defaultdict(float, {"generation_layerwise": 0.0})
        source_dict = {"generation_layerwise": "silicon"}
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

        combined_ctx_ms = (
            self._layerwise_context_layer_ms(database, model_name, tp_size, 1, combined_tokens) * num_layers
        )
        combined_allreduce_ms = (
            self._layerwise_tp_allreduce_ms(model, database, tp_size, combined_tokens) * 2 * num_layers
        )
        decode_delta_ms = 0.0
        if gen_tokens > 0:
            avg_decode_kv = isl + osl // 2
            gen_step_ms = (
                float(database.query_layerwise(model_name, "GEN", tp_size, gen_tokens, avg_decode_kv)) * num_layers
            )
            gen_as_ctx_ms = (
                self._layerwise_context_layer_ms(database, model_name, tp_size, 1, gen_tokens) * num_layers
            )
            decode_delta_ms = max(0.0, gen_step_ms - gen_as_ctx_ms)

        latency_ms = combined_ctx_ms + combined_allreduce_ms + decode_delta_ms
        per_ops = {
            "mixed_layerwise_context_combined": combined_ctx_ms,
            "mixed_layerwise_context_tp_allreduce": combined_allreduce_ms,
            "mixed_layerwise_decode_delta": decode_delta_ms,
        }
        per_ops_source = dict.fromkeys(per_ops, "silicon")
        return latency_ms, 0.0, per_ops, per_ops_source
