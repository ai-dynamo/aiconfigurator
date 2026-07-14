# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import logging
from collections import defaultdict
from typing import ClassVar

import numpy as np
import pandas as pd

from aiconfigurator.sdk import common
from aiconfigurator.sdk.config import RuntimeConfig
from aiconfigurator.sdk.inference_summary import InferenceSummary
from aiconfigurator.sdk.models import BaseModel
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.rust_engine_step import (
    RustEngineUnsupportedError,
    estimate_decode_step_latency_with_rust,
    estimate_mixed_step_latency_with_rust,
    estimate_static_latency_breakdown_with_rust,
    should_use_rust_engine_step,
)

logger = logging.getLogger(__name__)


class BaseBackend:
    """Base class for all inference backends.

    Subclasses provide:
        - ``self.name`` (set in ``__init__``).
        - ``ACTIVATION_COEFFICIENTS``: per-model-family activation scaling factors.
        - Optional overrides of memory-overhead constants
          (``MIN_ACTIVATION_BYTES``, ``ACTIVATION_OVERHEAD_FRAC``, ``OTHERS_OVERHEAD_FRAC``)
          and the agg-pipeline hooks (``_resolve_agg_kwargs``, ``_make_agg_cache_key``,
          ``_memory_usage_kwargs_for_agg``, ``_oom_check_kwargs``, ``_moe_workspace_width``).

    Concrete shared implementations:
        - ``run_static`` / ``run_static_latency_only``: static-batching inference.
        - ``run_agg``: continuous-batching inference for a single (b, ctx_tokens) point.
        - ``find_best_agg_result_under_constraints``: SLA-constrained sweep over agg points.
        - ``_get_memory_usage``: weights + activations + KV + nccl + others (model-family aware).
    """

    # ---- Memory-model knobs (overridable by subclasses) ----------------
    # Per-family activation scaling: family -> {tp_size: scalar}. The "default" key
    # is used when a model_family is not in the table. Empty in BaseBackend; each
    # subclass populates with its own table.
    ACTIVATION_COEFFICIENTS: ClassVar[dict[str, dict[int, float]]] = {}

    # Model families whose MoE block-scale dispatch workspace is added on top of
    # the base activation budget.
    MOE_WORKSPACE_FAMILIES: ClassVar[tuple[str, ...]] = (
        "GEMMA4MIX",
        "DEEPSEEK",
        "DEEPSEEKV32",
        "DEEPSEEKV4",
        "KIMIK25",
    )

    # Minimum activation memory, in bytes (clamps from below).
    MIN_ACTIVATION_BYTES: int = 70 * 1024 * 1024

    # Multiplicative overhead applied after the base activation/others computation.
    # SGLang sets these > 0 to model Python/runtime overhead.
    ACTIVATION_OVERHEAD_FRAC: float = 0.0
    OTHERS_OVERHEAD_FRAC: float = 0.0

    def __init__(self):
        # Flat dict keyed by tuple from ``_make_agg_cache_key``.
        self._agg_cache: dict = {}
        # Subclasses set the canonical name.
        self.name = None

    # ============== HOOKS (overridable by subclasses) ==================

    def _moe_workspace_width(self, model: BaseModel, model_family: str, h: int) -> int:
        """Feature width per token for MoE block-scale dispatch workspace.

        Default: model's residual hidden size (``_hidden_size``), which equals
        ``num_heads*head_size`` for most models but is wider for DeepSeek-V4's
        attention expansion. TRT-LLM overrides this to use the raw ``h`` for
        the DEEPSEEK family (legacy accounting, predates V4).
        """
        return getattr(model, "_hidden_size", h)

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, osl: int) -> int:
        """Return the number of decode tokens per mix step for a batch of b requests.

        A mix step is a forward pass that contains both prefill tokens (for requests
        still completing their context phase) and decode tokens (for requests already
        generating). This method encodes the engine's scheduling policy for how many
        decode-phase requests participate alongside the prefilling request(s).

        Subclasses should override to match their engine's scheduling behaviour.
        """
        steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
        if steps_to_finish_ctx >= osl:
            return max(1, int(b // (steps_to_finish_ctx / osl)))
        return max(1, b - int(np.ceil(ctx_tokens / isl)))

    def _mix_step_efficiency(self, ctx_tokens: int, gen_tokens: int) -> float:
        """GPU batching efficiency factor for a mixed prefill/decode forward pass.

        Per-op silicon data measures each operation in isolation, overstating the
        marginal cost of prefill tokens when they share a forward pass with decode
        tokens. Weight matrices are loaded once from HBM for the combined batch.
        Default: 1.0 (no correction — preserves existing behaviour for backends
        without empirical efficiency data).
        """
        return 1.0

    def _tpot_mix_steps(self, num_mix_steps: int) -> int:
        """Return the effective mix-step count for TPOT calculation.

        Engines with pipeline-drain latency at the context/decode boundary
        (requests cannot be immediately enqueued after prefill finishes) may
        reduce the effective step count to account for that bubble. Default:
        use the full mix step count. Subclasses should override with an
        empirically calibrated correction.
        """
        return num_mix_steps

    def _ttft_queuing_factor(self, b: int, steps_to_finish_ctx: float) -> float:
        """Return the queuing factor applied to the per-request prefill time to get TTFT.

        In a batch of b requests that all arrive simultaneously, each request waits
        for the preceding ones to complete their context phase before its own first
        token is produced. Default: the legacy heuristic formula (preserves existing
        behaviour for non-vLLM backends). Subclasses should override with a model
        appropriate to their engine's scheduling policy.
        """
        return min(2 + (steps_to_finish_ctx - 3) / 2 / 10, 4)

    def _prefill_dispatch_overhead_ms(self, model: "BaseModel") -> float:
        """Return a constant per-request overhead added to T_prefill (ms).

        Silicon benchmarks measure isolated kernel time. Production inference
        engines carry a fixed per-request cost from CPU-side Python dispatch
        across all layers (tensor creation, CUDA kernel launches) that does not
        appear in per-kernel measurements and does not scale with batch size.
        The model is provided so subclasses can factor in architecture properties
        beyond layer count. Default: 0.0 (no correction).
        """
        return 0.0

    def _throughput_cap(self, step_throughput: float, ttft: float, tpot: float, b: int, osl: int) -> float:
        """Return the effective output throughput after any engine-specific cap.

        Default: returns step_throughput unchanged. Subclasses may override to
        apply a tighter constraint — e.g. a Little's Law cap that prevents the
        model from recommending operating points that cannot be sustained in
        steady state given the predicted request latency.
        """
        return step_throughput

    def _resolve_agg_kwargs(self, kwargs: dict, isl: int, osl: int) -> dict:
        """Resolve backend-specific run_agg kwargs to defaults.

        Default: returns an empty dict. TRT-LLM resolves ``max_seq_len`` /
        ``max_num_tokens`` / ``free_gpu_memory_fraction`` here so both
        ``run_agg`` and ``find_best_agg_result_under_constraints`` see the
        same values when forwarding. Idempotent — calling with already-resolved
        kwargs returns the same values.
        """
        return {}

    def _make_agg_cache_key(
        self,
        isl: int,
        osl: int,
        b: int,
        ctx_tokens: int,
        engine_step_backend_key: str,
        agg_extra: dict,
    ) -> tuple:
        """Build the cache key for ``run_agg`` results."""
        return (isl, osl, b, ctx_tokens, engine_step_backend_key)

    @staticmethod
    def _runtime_config_for_agg_candidate(runtime_config: RuntimeConfig, batch_size: int) -> RuntimeConfig:
        candidate = copy.deepcopy(runtime_config)
        candidate.batch_size = batch_size
        return candidate

    def _memory_usage_kwargs_for_agg(self, num_tokens: int, agg_extra: dict) -> dict:
        """Kwargs for the ``_get_memory_usage`` call from ``run_agg``.

        Default: pass the locally-computed ``num_tokens``. TRT-LLM passes
        ``max_num_tokens`` (BuildConfig.max_num_tokens) for activation sizing
        and forwards ``max_seq_len`` for KV cache sizing.
        """
        return {"num_tokens": num_tokens}

    def _oom_check_kwargs(self, agg_extra: dict) -> dict:
        """Extra kwargs for ``InferenceSummary.set_memory_and_check_oom``.

        Default: none. TRT-LLM passes ``free_gpu_memory_fraction``,
        ``kv_cache_reserved_fraction``, and ``kv_cache_tolerance`` to enable
        the KV-cache capacity OOM check.
        """
        return {}

    # ============== STATIC INFERENCE (shared) ==========================

    @staticmethod
    def _visual_context_tokens_from_encoder_config(enc_cfg, runtime_config: RuntimeConfig) -> int:
        if not isinstance(enc_cfg, common.VisionEncoderConfig) or runtime_config.num_images_per_request <= 0:
            return 0
        post_merge, _ = BaseBackend._encoder_pre_merge_per_visual(runtime_config, enc_cfg)
        return post_merge * runtime_config.num_images_per_request

    @staticmethod
    def _visual_context_tokens(model: BaseModel, runtime_config: RuntimeConfig) -> int:
        return BaseBackend._visual_context_tokens_from_encoder_config(
            getattr(model, "encoder_config", None), runtime_config
        )

    @staticmethod
    def _encoder_pre_merge_per_visual(
        runtime_config: RuntimeConfig,
        enc_cfg,
    ) -> tuple[int, int]:
        """Resolve the per-image pre-merge / post-merge token counts from
        RuntimeConfig + VisionEncoderConfig.

        Resolution order:
            1. image_height + image_width (computed from patch/merge sizes)
            2. num_image_tokens (explicit per-image override)

        Returns ``(tokens_post_merge_per_image, pre_merge_per_image)``.
        Returns ``(0, 0)`` when neither is set (text-only path).
        """
        has_image_dims = runtime_config.image_height > 0 and runtime_config.image_width > 0
        if has_image_dims:
            img_stride = enc_cfg.patch_size * enc_cfg.spatial_merge_size
            tokens_per_image = (runtime_config.image_height // img_stride) * (runtime_config.image_width // img_stride)
            pre_merge_per_image = (runtime_config.image_height // enc_cfg.patch_size) * (
                runtime_config.image_width // enc_cfg.patch_size
            )
        elif runtime_config.num_image_tokens > 0:
            tokens_per_image = runtime_config.num_image_tokens
            pre_merge_per_image = tokens_per_image * (enc_cfg.spatial_merge_size**2)
        else:
            return 0, 0
        if tokens_per_image <= 0 or pre_merge_per_image <= 0:
            return 0, 0
        return tokens_per_image, pre_merge_per_image

    def _run_encoder_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str], int]:
        # Run the encoder phase (Currently VL models only).
        encoder_latency_dict = defaultdict(float)
        encoder_energy_wms_dict = defaultdict(float)
        encoder_source_dict = {}

        if not model.encoder_ops:
            return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, 0

        enc_cfg = getattr(model, "encoder_config", None)
        num_images = runtime_config.num_images_per_request
        if num_images <= 0 or not isinstance(enc_cfg, common.VisionEncoderConfig):
            return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, 0

        tokens_per_image, pre_merge_per_image = self._encoder_pre_merge_per_visual(runtime_config, enc_cfg)
        if tokens_per_image == 0:
            # No image dimensions specified; skip encoder modeling.
            return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, 0

        n_img_post = tokens_per_image * num_images  # post-merge: injected into LLM context
        n_img_pre = pre_merge_per_image * num_images  # pre-merge: processed by ViT transformer

        for op in model.encoder_ops:
            use_post = "encoder_projector" in op._name
            # ViT attention uses cu_seqlens: each image is an independent
            # varlen sequence of pre_merge_per_image patches.
            use_varlen = "encoder_attention" in op._name
            n_img = n_img_post if use_post else n_img_pre
            eff_batch = batch_size * num_images if use_varlen else batch_size
            eff_s = pre_merge_per_image if use_varlen else n_img
            x = eff_batch * eff_s
            result = op.query(
                database,
                x=x,
                batch_size=eff_batch,
                beam_width=1,
                s=eff_s,
                prefix=0,
                model_name=getattr(model, "model_name", ""),
            )
            encoder_latency_dict[op._name] += float(result)
            encoder_energy_wms_dict[op._name] += getattr(result, "energy", 0.0)
            encoder_source_dict[op._name] = getattr(result, "source", "silicon")

        return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, n_img_post

    def _run_context_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        isl: int,
        prefix: int,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        context_latency_dict = defaultdict(float)
        context_energy_wms_dict = defaultdict(float)
        # Per-op data source, accumulated by merging across calls to the same op.
        # Same-source repeated calls keep the tag; mismatched calls collapse to "mixed".
        context_source_dict: dict[str, str] = {}

        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(f"isl must be greater than 0 after removing prefix, but got {effective_isl}")

        for op in model.context_ops:
            x = batch_size * effective_isl if "logits_gemm" not in op._name else batch_size
            result = op.query(
                database,
                x=x,
                batch_size=batch_size,
                beam_width=1,
                s=effective_isl,
                prefix=prefix,
                seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
            )
            context_latency_dict[op._name] += float(result)
            context_energy_wms_dict[op._name] += getattr(result, "energy", 0.0)
            new_src = getattr(result, "source", "silicon")
            existing = context_source_dict.get(op._name)
            if existing is None or existing == new_src:
                context_source_dict[op._name] = new_src
            else:
                context_source_dict[op._name] = "mixed"

        return context_latency_dict, context_energy_wms_dict, context_source_dict

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
        generation_latency_dict = defaultdict(float)
        generation_energy_wms_dict = defaultdict(float)
        generation_source_dict: dict[str, str] = {}

        batch_size = batch_size * (model._nextn + 1)

        for i in range(0, osl - 1, stride):
            latency_dict = defaultdict(float)
            energy_wms_dict = defaultdict(float)

            for op in model.generation_ops:
                result = op.query(
                    database,
                    x=batch_size * beam_width,
                    batch_size=batch_size,
                    beam_width=beam_width,
                    s=isl + i + 1,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
                latency_dict[op._name] += float(result)
                energy_wms_dict[op._name] += getattr(result, "energy", 0.0)
                new_src = getattr(result, "source", "silicon")
                existing = generation_source_dict.get(op._name)
                if existing is None or existing == new_src:
                    generation_source_dict[op._name] = new_src
                else:
                    generation_source_dict[op._name] = "mixed"

            repeat_count = min(stride, osl - 1 - i)
            for op in latency_dict:
                generation_latency_dict[op] += latency_dict[op] * repeat_count
                generation_energy_wms_dict[op] += energy_wms_dict[op] * repeat_count

        return generation_latency_dict, generation_energy_wms_dict, generation_source_dict

    # TODO: refactor this 6-tuple return into a NamedTuple (or @dataclass) for
    # readability; current call sites unpack positionally and the signature is
    # hard to scan.
    def _run_static_breakdown(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
        img_ctx_tokens: int = 0,
    ) -> tuple[
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, str],
        dict[str, str],
    ]:
        batch_size, beam_width, isl, osl, prefix = (
            runtime_config.batch_size,
            runtime_config.beam_width,
            runtime_config.isl,
            runtime_config.osl,
            runtime_config.prefix,
        )
        isl_eff = isl + img_ctx_tokens

        context_latency_dict, context_energy_wms_dict, context_source_dict = {}, {}, {}
        generation_latency_dict, generation_energy_wms_dict, generation_source_dict = {}, {}, {}

        if should_use_rust_engine_step(runtime_config, database):
            try:
                rust_runtime_config = runtime_config
                if img_ctx_tokens:
                    rust_runtime_config = copy.copy(runtime_config)
                    rust_runtime_config.isl = isl_eff
                (
                    context_latency_dict,
                    generation_latency_dict,
                    context_source_dict,
                    generation_source_dict,
                ) = estimate_static_latency_breakdown_with_rust(
                    model,
                    database,
                    rust_runtime_config,
                    mode,
                    stride,
                    latency_correction_scale,
                )
                context_energy_wms_dict = dict.fromkeys(context_latency_dict, 0.0)
                generation_energy_wms_dict = dict.fromkeys(generation_latency_dict, 0.0)
                return (
                    context_latency_dict,
                    context_energy_wms_dict,
                    generation_latency_dict,
                    generation_energy_wms_dict,
                    context_source_dict,
                    generation_source_dict,
                )
            except RustEngineUnsupportedError as exc:
                # Op graph not expressible as a compiled EngineSpec: fall back
                # to the Python step (parity by delegation — Python computes
                # what Rust cannot yet express). Perf-data misses are NOT
                # caught here; they must stay error-symmetric.
                logger.warning(
                    "engine-step backend 'rust' cannot compile this model; using the python step: %s",
                    exc,
                )

        if mode == "static_ctx":
            context_latency_dict, context_energy_wms_dict, context_source_dict = self._run_context_phase(
                model, database, runtime_config, batch_size, isl_eff, prefix
            )
        elif mode == "static_gen":
            generation_latency_dict, generation_energy_wms_dict, generation_source_dict = self._run_generation_phase(
                model, database, runtime_config, batch_size, beam_width, isl_eff, osl, stride
            )
        else:
            context_latency_dict, context_energy_wms_dict, context_source_dict = self._run_context_phase(
                model, database, runtime_config, batch_size, isl_eff, prefix
            )
            generation_latency_dict, generation_energy_wms_dict, generation_source_dict = self._run_generation_phase(
                model, database, runtime_config, batch_size, beam_width, isl_eff, osl, stride
            )

        if latency_correction_scale != 1.0:
            logger.debug(f"latency_correction_scale: {latency_correction_scale} is applied")
            for op in context_latency_dict:
                context_latency_dict[op] *= latency_correction_scale
                context_energy_wms_dict[op] *= latency_correction_scale
            for op in generation_latency_dict:
                generation_latency_dict[op] *= latency_correction_scale
                generation_energy_wms_dict[op] *= latency_correction_scale

        return (
            context_latency_dict,
            context_energy_wms_dict,
            generation_latency_dict,
            generation_energy_wms_dict,
            context_source_dict,
            generation_source_dict,
        )

    def run_static_latency_only(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> float:
        """
        Run static inference and return only the total latency in milliseconds.

        This shares the same latency breakdown path as ``run_static`` but skips
        building an ``InferenceSummary``.
        """
        if mode == "static_gen":
            encoder_latency = 0.0
            img_ctx_tokens = self._visual_context_tokens(model, runtime_config)
        else:
            encoder_latency_dict, encoder_energy_wms_dict, _, img_ctx_tokens = self._run_encoder_phase(
                model, database, runtime_config, runtime_config.batch_size
            )
            if latency_correction_scale != 1.0:
                for op in encoder_latency_dict:
                    encoder_latency_dict[op] *= latency_correction_scale
                    encoder_energy_wms_dict[op] *= latency_correction_scale
            encoder_latency = sum(encoder_latency_dict.values())

        (
            context_latency_dict,
            _,
            generation_latency_dict,
            _,
            _,
            _,
        ) = self._run_static_breakdown(
            model,
            database,
            runtime_config,
            mode,
            stride,
            latency_correction_scale,
            img_ctx_tokens=img_ctx_tokens,
        )
        return encoder_latency + sum(context_latency_dict.values()) + sum(generation_latency_dict.values())

    def run_static(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        mode: str,
        stride: int = 32,
        latency_correction_scale: float = 1.0,
    ) -> InferenceSummary:
        """
        Run the static inference.

        Args:
            model (BaseModel): the model to run inference
            database (PerfDatabase): the database to run inference
            runtime_config (RuntimeConfig): the runtime config
            mode (str): the mode to run inference, static, static_ctx, static_gen
            stride (int): the stride is used to accelerate the estimation, for a give osl,
                will only computes the i, i+stride, i+2*stride, ... step, default is 32.
            latency_correction_scale (float): the correction scale to adjust the latency,
                default is 1.0.
                corrected latency = latency * latency_correction_scale
        """

        def _run_encoder(batch_size: int) -> tuple[dict[str, float], dict[str, float], dict[str, str], int]:
            return self._run_encoder_phase(model, database, runtime_config, batch_size)

        def _run_context(bs: int, effective_isl: int, pfx: int):
            return self._run_context_phase(model, database, runtime_config, bs, effective_isl, pfx)

        def _run_generation(bs: int, bw: int, effective_isl: int, eff_osl: int, strd: int):
            return self._run_generation_phase(model, database, runtime_config, bs, bw, effective_isl, eff_osl, strd)

        summary = InferenceSummary(runtime_config)
        batch_size, beam_width, isl, osl, prefix = (
            runtime_config.batch_size,
            runtime_config.beam_width,
            runtime_config.isl,
            runtime_config.osl,
            runtime_config.prefix,
        )

        if mode == "static_gen":
            encoder_latency_dict, encoder_energy_wms_dict = defaultdict(float), defaultdict(float)
            encoder_source_dict = {}
            img_ctx_tokens = self._visual_context_tokens(model, runtime_config)
        else:
            encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, img_ctx_tokens = _run_encoder(
                batch_size
            )

        if latency_correction_scale != 1.0:
            for op in encoder_latency_dict:
                encoder_latency_dict[op] *= latency_correction_scale
                encoder_energy_wms_dict[op] *= latency_correction_scale

        encoder_memory = (
            {}
            if mode == "static_gen"
            else self._get_encoder_component_memory_for_runtime(model, runtime_config, batch_size)
        )
        encoder_memory_total = encoder_memory.get("total", 0.0)

        (
            context_latency_dict,
            context_energy_wms_dict,
            generation_latency_dict,
            generation_energy_wms_dict,
            context_source_dict,
            generation_source_dict,
        ) = self._run_static_breakdown(
            model,
            database,
            runtime_config,
            mode,
            stride,
            latency_correction_scale,
            img_ctx_tokens=img_ctx_tokens,
        )

        if mode == "static_ctx":
            memory = self._get_memory_usage(
                model,
                database,
                batch_size,
                beam_width,
                isl + img_ctx_tokens,
                1,
                prefix=prefix,
                encoder_memory=encoder_memory,
            )
        elif mode == "static_gen":
            memory = self._get_memory_usage(
                model,
                database,
                batch_size,
                beam_width,
                isl + img_ctx_tokens,
                osl,
                num_tokens=batch_size * beam_width,
                prefix=prefix,
            )
        else:
            memory = self._get_memory_usage(
                model,
                database,
                batch_size,
                beam_width,
                isl + img_ctx_tokens,
                osl,
                prefix=prefix,
                encoder_memory=encoder_memory,
            )

        # Calculate total latencies and energies (simple sums - decoupled!)
        encoder_latency_ms = sum(encoder_latency_dict.values())  # milliseconds
        encoder_energy_wms = sum(encoder_energy_wms_dict.values())  # watt-milliseconds

        context_latency_ms = sum(context_latency_dict.values())  # milliseconds
        context_energy_wms = sum(context_energy_wms_dict.values())  # watt-milliseconds

        generation_latency_ms = sum(generation_latency_dict.values())  # milliseconds
        generation_energy_wms = sum(generation_energy_wms_dict.values())  # watt-milliseconds

        # Calculate average power (SIMPLIFIED - just divide! Single operation.)
        encoder_power_avg = encoder_energy_wms / encoder_latency_ms if encoder_latency_ms > 0 else 0.0
        context_power_avg = context_energy_wms / context_latency_ms if context_latency_ms > 0 else 0.0
        generation_power_avg = generation_energy_wms / generation_latency_ms if generation_latency_ms > 0 else 0.0

        # E2E weighted average power (EVEN SIMPLER - natural weighted average!)
        total_latency_ms = encoder_latency_ms + context_latency_ms + generation_latency_ms
        total_energy_wms = encoder_energy_wms + context_energy_wms + generation_energy_wms
        e2e_power_avg = total_energy_wms / total_latency_ms if total_latency_ms > 0 else 0.0

        # For backward compatibility, keep old variable names
        encoder_latency = encoder_latency_ms
        context_latency = context_latency_ms
        generation_latency = generation_latency_ms

        bs = batch_size
        global_bs = bs * model.config.attention_dp_size
        concurrency = global_bs
        ttft = encoder_latency + context_latency
        tpot = 0.0 if osl <= 1 else generation_latency / (osl - 1)
        num_generated_tokens = max(osl - 1, 0)
        request_latency = ttft + tpot * num_generated_tokens
        if request_latency == 0.0:
            request_latency = encoder_latency + context_latency + generation_latency
        request_rate = 0.0
        seq_s = (
            0.0 if request_latency == 0.0 else global_bs / request_latency * 1000 * model.config.pp_size
        )  # handle statc_gen only with osl==1, scale by pp
        seq_s_gpu = seq_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s = seq_s * osl if mode != "static_gen" else seq_s * (osl - 1)
        if mode == "static_ctx":
            tokens_s = seq_s * 1  # only first token
        tokens_s_gpu = tokens_s / model.config.tp_size / model.config.pp_size / model.config.attention_dp_size
        tokens_s_user = 0.0 if tpot == 0.0 else 1000.0 / tpot
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        cp = model.config.cp_size
        # CP is an independent sequence-sharding dim -> folds into the per-worker
        # GPU count (tp*pp*dp*cp), so throughput-per-GPU normalizes correctly.
        num_total_gpus = model.config.total_gpus_per_worker
        parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}" + (f"cp{cp}" if cp > 1 else "")
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory["total"]

        data = [
            [
                model.model_path,
                isl,
                osl,
                prefix,
                concurrency,
                request_rate,
                bs,
                global_bs,
                ttft,
                tpot,
                seq_s,
                seq_s_gpu,
                tokens_s,
                tokens_s_gpu,
                tokens_s_user,
                request_latency,
                encoder_latency,
                encoder_memory_total,
                context_latency,
                generation_latency,
                num_total_gpus,
                tp,
                pp,
                dp,
                moe_tp,
                moe_ep,
                cp,
                parallel,
                gemm,
                kvcache,
                fmha,
                moe,
                comm,
                mem,
                database.backend,
                database.version,
                database.system,
                e2e_power_avg,  # NEW: E2E weighted average power in watts
            ]
        ]

        summary_df = pd.DataFrame(data, columns=common.ColumnsStatic).round(3)

        summary.set_encoder_latency_dict(encoder_latency_dict)
        summary.set_context_latency_dict(context_latency_dict)
        summary.set_generation_latency_dict(generation_latency_dict)
        summary.set_encoder_energy_wms_dict(encoder_energy_wms_dict)
        summary.set_context_energy_wms_dict(context_energy_wms_dict)  # UPDATED: explicit units
        summary.set_generation_energy_wms_dict(generation_energy_wms_dict)  # UPDATED: explicit units
        summary.set_encoder_source_dict(encoder_source_dict)
        summary.set_context_source_dict(context_source_dict)
        summary.set_generation_source_dict(generation_source_dict)
        summary.set_encoder_power_avg(encoder_power_avg)
        summary.set_context_power_avg(context_power_avg)
        summary.set_generation_power_avg(generation_power_avg)
        summary.set_e2e_power_avg(e2e_power_avg)
        summary.set_memory_and_check_oom(memory, database.system_spec["gpu"]["mem_capacity"])
        # KV-per-seq context for capacity probing in CLI detail reports.
        try:
            kv_seq_len_used = isl + img_ctx_tokens + beam_width * osl
            # CP shards persistent KV across cp ranks (full/cp per rank).
            kv_bytes_per_seq = model.get_kvcache_bytes_per_sequence(kv_seq_len_used) / model._cp_kv_memory_divisor()
            summary.set_kv_per_seq(kv_bytes_per_seq, kv_seq_len_used)
        except Exception:
            # Best-effort; downstream report degrades gracefully when unset.
            pass

        if encoder_memory:
            summary.set_encoder_memory(encoder_memory)

        summary.set_summary_df(summary_df)

        return summary

    def get_default_free_gpu_memory_fraction(self) -> float | None:
        """Default KV cache memory fraction for this backend, if it has one."""
        return None

    def get_kv_cache_memory_check_params(self) -> tuple[float, float]:
        """Return backend-specific KV cache reserved fraction and tolerance."""
        return 0.0, 0.0

    def get_partition_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        *,
        partition_ops,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
        prefix: int = 0,
        max_seq_len: int | None = None,
        include_kvcache: bool = True,
        kvcache_multiplier: int = 1,
    ) -> dict[str, float]:
        """Get backend memory with weights replaced by a model partition.

        AFD uses the same backend activation/KV/NCCL/other memory model as
        agg/disagg, then substitutes the weights that actually live on the
        A- or F-worker pool.
        """
        kwargs = {
            "num_tokens": num_tokens,
            "prefix": prefix,
        }
        if "max_seq_len" in inspect.signature(self._get_memory_usage).parameters:
            kwargs["max_seq_len"] = max_seq_len

        memory = self._get_memory_usage(
            model,
            database,
            batch_size,
            beam_width,
            isl,
            osl,
            **kwargs,
        )
        memory = dict(memory)
        memory["weights"] = sum(op.get_weights() for op in partition_ops) / max(model.config.pp_size, 1) / (1 << 30)
        if include_kvcache:
            memory["kvcache"] = memory.get("kvcache", 0.0) * max(kvcache_multiplier, 1)
        else:
            memory["kvcache"] = 0.0

        memory.setdefault("activations", 0.0)
        memory.setdefault("nccl", 0.0)
        memory.setdefault("others", 0.0)
        memory["total"] = (
            memory["weights"] + memory["activations"] + memory["kvcache"] + memory["nccl"] + memory["others"]
        )
        return memory

    def _get_ctx_tokens_list_for_agg_sweep(
        self,
        isl: int,
        ctx_stride: int,
        enable_chunked_prefill: bool,
        max_normal_ctx_tokens: int = 8192,
        max_ctx_tokens_multiple_of_isl: int = 2,
        max_ctx_tokens_small_search_steps: int = 16,
        max_ctx_tokens_search_steps: int = 8,
    ) -> list[int]:
        """
        Generate a list of num_context_tokens to sweep for agg inference.

        Args:
            isl: Target input sequence length during inference.
            ctx_stride: Default stride for context_tokens to sweep, ignored if enable_chunked_prefill is True.
            enable_chunked_prefill: Whether the inference framework will have chunked_prefill enabled.
            max_normal_ctx_tokens: boundary at which to increase the stride for faster sweeping.
            max_ctx_tokens_multiple_of_isl: Maximum multiple of isl to consider for ctx tokens.
            max_ctx_tokens_small_search_steps: Maximum search steps under max_normal_ctx_tokens.
            max_ctx_tokens_large_search_steps: Maximum search steps over max_normal_ctx_tokens.
        Returns:
            Sorted list of num_context_tokens to sweep.
        """

        # Largest ctx_tokens to consider for sweeping.
        max_ctx_tokens = max(max_normal_ctx_tokens, isl * max_ctx_tokens_multiple_of_isl)

        # Sweep stride under max_normal_ctx_tokens.
        ctx_stride = max(ctx_stride, max_normal_ctx_tokens // max_ctx_tokens_small_search_steps)

        # Sweep stride once ctx_tokens is larger than max_normal_ctx_tokens.
        ctx_stride_large = max(
            1024,
            ctx_stride,
            max_ctx_tokens // max_ctx_tokens_search_steps,
        )

        if not enable_chunked_prefill:
            new_ctx_stride = max(isl, ctx_stride)
            new_ctx_stride_large = int(np.ceil(ctx_stride_large / isl) * isl)
            logger.debug(
                f"enable_chunked_prefill is off, override ctx_stride: from {ctx_stride} to {new_ctx_stride}, "
                f"ctx_stride_large: from {ctx_stride_large} to {new_ctx_stride_large}"
            )
            ctx_stride = new_ctx_stride
            ctx_stride_large = new_ctx_stride_large

        # prepare ctx_tokens_list
        ctx_tokens_list = []
        ctx_tokens = 0
        while True:
            if ctx_tokens < max_normal_ctx_tokens:
                ctx_tokens += ctx_stride
            else:
                ctx_tokens += ctx_stride_large

            if ctx_tokens > max_ctx_tokens:
                break

            ctx_tokens_list.append(ctx_tokens)

        # add those just match the multiple of isl
        for i in range(1, max_ctx_tokens_multiple_of_isl + 1):
            ctx_tokens = isl * i
            if ctx_tokens not in ctx_tokens_list:
                ctx_tokens_list.append(ctx_tokens)
        ctx_tokens_list.sort()
        return ctx_tokens_list

    # ============== AGG STEP LATENCY HELPERS (shared) ==================

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
        """Latency / energy for one mixed (chunked-prefill + decode) step.

        Returns ``(latency_ms, energy_wms, per_op_latency, per_op_source)``.
        The per-op dicts represent the breakdown for this step alone; the
        caller stores them under the "mix_step" key of the run_agg
        per-ops summary.
        """
        if should_use_rust_engine_step(runtime_config, database):
            try:
                latency_ms = estimate_mixed_step_latency_with_rust(
                    model,
                    database,
                    ctx_tokens=ctx_tokens,
                    gen_tokens=gen_tokens,
                    isl=isl,
                    osl=osl,
                    prefix=prefix,
                    seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
                return (
                    latency_ms,
                    0.0,
                    {"rust_engine_step_mixed": latency_ms},
                    {"rust_engine_step_mixed": "rust"},
                )
            except RustEngineUnsupportedError as exc:
                logger.warning(
                    "engine-step backend 'rust' cannot compile this model; using the python step: %s",
                    exc,
                )

        ctx_scale = runtime_config.seq_imbalance_correction_scale
        gen_scale = runtime_config.gen_seq_imbalance_correction_scale

        # Pass 1: combined single-batch inference to extract non-attention latency.
        num_tokens_combined = ctx_tokens + gen_tokens
        summary = self.run_static(
            model,
            database,
            # num tokens for gemm needs to be adjusted for prefix, depends on the avg prefix len per request
            RuntimeConfig(
                batch_size=1,
                beam_width=1,
                isl=num_tokens_combined,
                osl=1,
                prefix=prefix * np.floor(ctx_tokens / isl),
                seq_imbalance_correction_scale=ctx_scale,
            ),
            mode="static_ctx",
        )
        latency_dict = summary.get_context_latency_dict()
        energy_wms_dict = summary.get_context_energy_wms_dict()
        source_dict = summary.get_context_source_dict()
        non_attention_latency_ms = 0.0
        non_attention_energy_wms = 0.0
        mix_non_attn_ops: dict[str, float] = {}
        mix_non_attn_sources: dict[str, str] = {}
        for layer_name, latency in latency_dict.items():
            if layer_name != "context_attention":
                non_attention_latency_ms += latency
                non_attention_energy_wms += energy_wms_dict.get(layer_name, 0.0)
                mix_non_attn_ops[layer_name] = latency
                mix_non_attn_sources[layer_name] = source_dict.get(layer_name, "silicon")

        # Pass 2: context attention split full isl over num_steps and averaged.
        batch_size = np.ceil(ctx_tokens / isl)
        summary = self.run_static(
            model,
            database,
            RuntimeConfig(
                batch_size=batch_size,
                beam_width=1,
                isl=isl,
                osl=1,
                prefix=prefix,
                seq_imbalance_correction_scale=ctx_scale,
            ),
            mode="static_ctx",
        )
        latency_dict = summary.get_context_latency_dict()
        energy_wms_dict = summary.get_context_energy_wms_dict()
        ctx_attn_source = summary.get_context_source_dict().get("context_attention", "silicon")
        scale_factor = np.ceil(isl / ctx_tokens)
        ctx_attention_latency_ms = latency_dict["context_attention"] / scale_factor
        ctx_attention_energy_wms = energy_wms_dict.get("context_attention", 0.0) / scale_factor

        # Pass 3: generation attention (use isl + osl//2 for the avg seq len).
        gen_attention_latency_ms = 0.0
        gen_attention_energy_wms = 0.0
        gen_attn_source = "silicon"
        if gen_tokens > 0:
            summary = self.run_static(
                model,
                database,
                RuntimeConfig(
                    batch_size=gen_tokens,
                    beam_width=1,
                    isl=isl + osl // 2,
                    osl=2,
                    gen_seq_imbalance_correction_scale=gen_scale,
                ),
                mode="static_gen",
            )
            latency_dict = summary.get_generation_latency_dict()
            energy_wms_dict = summary.get_generation_energy_wms_dict()
            gen_attention_latency_ms = latency_dict["generation_attention"]
            gen_attention_energy_wms = energy_wms_dict.get("generation_attention", 0.0)
            gen_attn_source = summary.get_generation_source_dict().get("generation_attention", "silicon")

        per_ops_step_data = {
            **mix_non_attn_ops,
            "context_attention (scaled)": ctx_attention_latency_ms,
            "generation_attention": gen_attention_latency_ms,
        }
        per_ops_step_source = {
            **mix_non_attn_sources,
            "context_attention (scaled)": ctx_attn_source,
            "generation_attention": gen_attn_source,
        }

        total_latency_ms = non_attention_latency_ms + ctx_attention_latency_ms + gen_attention_latency_ms
        total_energy_wms = non_attention_energy_wms + ctx_attention_energy_wms + gen_attention_energy_wms
        return total_latency_ms, total_energy_wms, per_ops_step_data, per_ops_step_source

    def _get_genonly_step_latency(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        gen_tokens: int,
        isl: int,
        osl: int,
    ) -> tuple[float, float, dict, dict]:
        """Latency / energy for one generation-only step.

        Returns ``(latency_ms, energy_wms, per_op_latency, per_op_source)``.
        When ``gen_tokens <= 0`` both totals are 0 and the per-op dicts are empty.
        """
        if gen_tokens <= 0:
            return 0.0, 0.0, {}, {}
        if should_use_rust_engine_step(runtime_config, database):
            try:
                latency_ms = estimate_decode_step_latency_with_rust(
                    model,
                    database,
                    gen_tokens=gen_tokens,
                    isl=isl,
                    osl=osl,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
                return (
                    latency_ms,
                    0.0,
                    {"rust_engine_step_generation": latency_ms},
                    {"rust_engine_step_generation": "rust"},
                )
            except RustEngineUnsupportedError as exc:
                logger.warning(
                    "engine-step backend 'rust' cannot compile this model; using the python step: %s",
                    exc,
                )

        gen_scale = runtime_config.gen_seq_imbalance_correction_scale
        summary = self.run_static(
            model,
            database,
            RuntimeConfig(
                batch_size=gen_tokens,
                beam_width=1,
                isl=isl + osl // 2,
                osl=2,
                gen_seq_imbalance_correction_scale=gen_scale,
            ),
            mode="static_gen",
        )
        latency_dict = summary.get_generation_latency_dict()
        energy_wms_dict = summary.get_generation_energy_wms_dict()
        source_dict = summary.get_generation_source_dict()
        total_latency_ms = 0.0
        total_energy_wms = 0.0
        per_ops_step_data: dict[str, float] = {}
        per_ops_step_source: dict[str, str] = {}
        for layer_name, latency in latency_dict.items():
            total_latency_ms += latency
            total_energy_wms += energy_wms_dict.get(layer_name, 0.0)
            per_ops_step_data[layer_name] = latency
            per_ops_step_source[layer_name] = source_dict.get(layer_name, "silicon")
        return total_latency_ms, total_energy_wms, per_ops_step_data, per_ops_step_source

    # ============== AGG INFERENCE (shared) =============================

    def _get_encoder_component_memory(self, model: BaseModel, num_tokens: int) -> dict[str, float]:
        """Encoder memory component colocated with the prefill/agg worker."""
        weights = sum(op.get_weights() for op in model.encoder_ops)
        enc_cfg = getattr(model, "encoder_config", None)
        activations = 0.0
        if isinstance(enc_cfg, common.VisionEncoderConfig) and num_tokens > 0:
            # ~3x hidden_size per patch covers QKV, attention output, and FFN intermediates (bfloat16)
            activations = 2 * num_tokens * enc_cfg.hidden_size * 3
            activations = max(activations, 32 * 1024 * 1024)  # 32 MiB minimum
        one_gib = 1 << 30
        return {
            "total": (weights + activations) / one_gib,
            "weights": weights / one_gib,
            "activations": activations / one_gib,
            "kvcache": 0.0,
            "nccl": 0.0,
            "others": 0.0,
        }

    def _get_encoder_component_memory_for_runtime(
        self,
        model: BaseModel,
        runtime_config: RuntimeConfig,
        batch_size: int,
    ) -> dict[str, float]:
        enc_cfg = getattr(model, "encoder_config", None)
        if not model.encoder_ops or not isinstance(enc_cfg, common.VisionEncoderConfig):
            return {}
        if runtime_config.num_images_per_request <= 0:
            return {}
        _, pre_merge_per_image = self._encoder_pre_merge_per_visual(runtime_config, enc_cfg)
        if pre_merge_per_image <= 0:
            return {}
        num_tokens = batch_size * runtime_config.num_images_per_request * pre_merge_per_image
        return self._get_encoder_component_memory(model, num_tokens)

    def run_agg(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """Run the agg (continuous-batching) inference for a single (b, ctx_tokens) point."""
        text_isl = runtime_config.isl
        osl = runtime_config.osl
        prefix = runtime_config.prefix
        b = runtime_config.batch_size
        img_ctx_tokens = self._visual_context_tokens(model, runtime_config)
        isl = text_isl + img_ctx_tokens
        engine_step_backend_key = "rust" if should_use_rust_engine_step(runtime_config, database) else "python"
        ctx_tokens = kwargs.get("ctx_tokens")
        assert ctx_tokens is not None, "ctx_tokens is required"
        balance_score = isl * b / ctx_tokens / osl

        # Backend-specific kwargs (TRT-LLM: max_seq_len / max_num_tokens /
        # free_gpu_memory_fraction; others: {}).
        agg_extra = self._resolve_agg_kwargs(kwargs, isl=isl, osl=osl)

        visual_cache_key = (
            runtime_config.image_height,
            runtime_config.image_width,
            runtime_config.num_images_per_request,
        )
        cache_key = (
            self._make_agg_cache_key(isl, osl, b, ctx_tokens, engine_step_backend_key, agg_extra),
            visual_cache_key,
        )
        cached = self._agg_cache.get(cache_key)
        if cached is not None:
            return cached

        encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, _ = self._run_encoder_phase(
            model, database, runtime_config, b
        )
        encoder_latency_ms = sum(encoder_latency_dict.values())
        encoder_energy_wms = sum(encoder_energy_wms_dict.values())
        encoder_memory = self._get_encoder_component_memory_for_runtime(model, runtime_config, b)
        encoder_memory_total = encoder_memory.get("total", 0.0)

        # Compute num_mix_steps / num_genonly_steps within osl steps such that
        # all ctx tokens are consumed.
        steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
        num_mix_steps = num_genonly_steps = 0
        num_mix_steps_for_tpot_calc = 0  # correction for tpot calc only
        if b > 1:
            num_mix_gen_tokens = self._mix_step_gen_tokens(b, ctx_tokens, isl, osl)
            assert num_mix_gen_tokens >= 1, (
                f"num_mix_gen_tokens: {num_mix_gen_tokens}, b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}"
            )
            num_mix_ctx_tokens = ctx_tokens
            if steps_to_finish_ctx >= osl:
                num_mix_steps = steps_to_finish_ctx
                num_genonly_steps = 0
                num_genonly_tokens = 0
                num_mix_steps_for_tpot_calc = num_mix_steps
            else:
                num_mix_steps = steps_to_finish_ctx
                num_genonly_steps = osl - num_mix_steps
                num_genonly_tokens = b
                num_mix_steps_for_tpot_calc = self._tpot_mix_steps(num_mix_steps)
        elif b == 1:
            # special case for b=1
            num_mix_steps = 1
            num_mix_ctx_tokens = ctx_tokens
            num_mix_gen_tokens = 0
            num_genonly_steps = osl - 1
            num_genonly_tokens = 1
            num_mix_steps_for_tpot_calc = 0

        # Step-latency helpers (return (latency_ms, energy_wms, per_op_data, per_op_source)).
        per_ops_data: dict[str, dict] = {}
        per_ops_source: dict[str, dict] = {}

        mix_step_latency_ms, mix_step_energy_wms, mix_per_ops, mix_per_ops_src = self._get_mix_step_latency(
            model, database, runtime_config, num_mix_ctx_tokens, num_mix_gen_tokens, isl, osl, prefix
        )
        mix_efficiency = self._mix_step_efficiency(num_mix_ctx_tokens, num_mix_gen_tokens)
        mix_step_latency_ms *= mix_efficiency
        mix_step_energy_wms *= mix_efficiency
        if mix_efficiency != 1.0:
            mix_per_ops = {op: v * mix_efficiency for op, v in mix_per_ops.items()}
        per_ops_data["mix_step"] = mix_per_ops
        per_ops_source["mix_step"] = mix_per_ops_src

        (
            genonly_step_latency_ms,
            genonly_step_energy_wms,
            genonly_per_ops,
            genonly_per_ops_src,
        ) = self._get_genonly_step_latency(model, database, runtime_config, num_genonly_tokens, isl, osl)
        if genonly_per_ops:
            per_ops_data["genonly_step"] = genonly_per_ops
            per_ops_source["genonly_step"] = genonly_per_ops_src

        # TTFT: per-request prefill time * queuing factor, plus encoder latency.
        # _mix_step_efficiency reduces mix_step_latency_ms based on the fraction of
        # decode tokens in the step. For TTFT we need the pure prefill cost (no decode
        # tokens alongside), so we undo that efficiency reduction first.
        _prefill_step_ms = mix_step_latency_ms / mix_efficiency if mix_efficiency > 0 else mix_step_latency_ms
        _ttft_per_request = _prefill_step_ms * np.ceil(isl / ctx_tokens) + self._prefill_dispatch_overhead_ms(model)
        ttft = encoder_latency_ms + _ttft_per_request * self._ttft_queuing_factor(b, steps_to_finish_ctx)
        logger.debug(
            f"ttft: prefill_step={_prefill_step_ms:.2f}ms qf={self._ttft_queuing_factor(b, steps_to_finish_ctx):.2f}"
        )

        # Guard against osl == 1 (no-decode), which makes both denominators zero.
        _tpot_steps = num_mix_steps_for_tpot_calc + num_genonly_steps
        tpot = (
            (mix_step_latency_ms * num_mix_steps_for_tpot_calc + genonly_step_latency_ms * num_genonly_steps)
            / _tpot_steps
            if _tpot_steps > 0
            else 0.0
        )
        _total_step_latency_ms = (
            encoder_latency_ms + num_mix_steps * mix_step_latency_ms + num_genonly_steps * genonly_step_latency_ms
        )
        _step_throughput = (
            (1000 / _total_step_latency_ms * b * (osl - 1)) if (osl > 1 and _total_step_latency_ms > 0) else 0.0
        )
        output_throughput = self._throughput_cap(_step_throughput, ttft, tpot, b, osl)
        logger.debug(
            f"ctx_tokens: {ctx_tokens}, b: {b}, osl: {osl}, isl: {isl}, "
            f"num_mix_steps: {num_mix_steps}, num_genonly_steps: {num_genonly_steps}, "
            f"num_mix_ctx_tokens: {num_mix_ctx_tokens}, "
            f"num_mix_gen_tokens: {num_mix_gen_tokens}, "
            f"num_genonly_tokens: {num_genonly_tokens}"
        )
        logger.debug(f"mix_step_latency: {mix_step_latency_ms} ms, genonly_step_latency: {genonly_step_latency_ms} ms")
        logger.debug(
            f"mix_step_energy: {mix_step_energy_wms} W·ms, genonly_step_energy: {genonly_step_energy_wms} W·ms"
        )
        logger.debug(f"ttft: {ttft}, tpot: {tpot}, output_throughput: {output_throughput}")

        # Weighted average power: total energy / total latency.
        total_energy_wms = (
            encoder_energy_wms + num_mix_steps * mix_step_energy_wms + num_genonly_steps * genonly_step_energy_wms
        )
        total_latency_ms = _total_step_latency_ms
        agg_power_avg_w = total_energy_wms / total_latency_ms if total_latency_ms > 0 else 0.0
        logger.debug(f"Aggregated power: {agg_power_avg_w}W (from {total_energy_wms} W·ms / {total_latency_ms} ms)")

        num_ctx_requests = np.ceil(ctx_tokens / isl)
        num_gen_requests = b - num_ctx_requests
        if b == 1:
            num_ctx_requests = 1
            num_gen_requests = 1

        # correct output_throughput and concurrency for attention dp (global batch)
        scale_factor = model.config.pp_size * model.config.attention_dp_size
        output_throughput = output_throughput * scale_factor
        concurrency = b * scale_factor

        request_rate = output_throughput / (osl - 1) if osl > 1 else 0.0
        if b > 1:
            # will not be corrected by balance score when it's larger than 1.0
            # in order to indicate what's happening
            num_tokens = num_gen_requests + ctx_tokens
        else:
            num_tokens = ctx_tokens

        memory = self._get_memory_usage(
            model,
            database,
            b,
            1,
            isl,
            osl,
            prefix=prefix,
            encoder_memory=encoder_memory,
            **self._memory_usage_kwargs_for_agg(num_tokens=num_tokens, agg_extra=agg_extra),
        )
        tp = model.config.tp_size
        pp = model.config.pp_size
        dp = model.config.attention_dp_size
        moe_tp = model.config.moe_tp_size
        moe_ep = model.config.moe_ep_size
        cp = model.config.cp_size
        tokens_s_gpu = output_throughput / pp / tp / dp / cp
        # tpot can be 0.0 for valid no-decode agg runs (osl<=1 / _tpot_steps==0).
        tokens_s_user = 1000.0 / tpot if (osl > 1 and tpot > 0.0) else 0.0
        seq_s = request_rate
        seq_s_gpu = seq_s / pp / tp / dp / cp
        tokens_s = output_throughput
        request_latency = ttft + tpot * max(osl - 1, 0)
        num_total_gpus = model.config.total_gpus_per_worker
        parallel = f"tp{tp}pp{pp}dp{dp}etp{moe_tp}ep{moe_ep}" + (f"cp{cp}" if cp > 1 else "")
        gemm = model.config.gemm_quant_mode.name
        kvcache = model.config.kvcache_quant_mode.name
        fmha = model.config.fmha_quant_mode.name
        moe = model.config.moe_quant_mode.name
        comm = model.config.comm_quant_mode.name
        mem = memory["total"]

        result_dict = {
            "model": model.model_path,
            "isl": text_isl,
            "osl": osl,
            "prefix": prefix,
            "concurrency": concurrency,
            "request_rate": request_rate,
            "bs": b,
            "global_bs": b * model.config.attention_dp_size,
            "ttft": ttft,
            "tpot": tpot,
            "seq/s": seq_s,
            "seq/s/gpu": seq_s_gpu,
            "tokens/s": tokens_s,
            "tokens/s/gpu": tokens_s_gpu,
            "tokens/s/user": tokens_s_user,
            "request_latency": request_latency,
            "encoder_latency": encoder_latency_ms,
            "encoder_memory": encoder_memory_total,
            "num_total_gpus": num_total_gpus,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "moe_tp": moe_tp,
            "moe_ep": moe_ep,
            "cp": cp,
            "parallel": parallel,
            "gemm": gemm,
            "kvcache": kvcache,
            "fmha": fmha,
            "moe": moe,
            "comm": comm,
            "memory": mem,
            "balance_score": balance_score,
            "num_ctx_reqs": num_ctx_requests,
            "num_gen_reqs": num_gen_requests,
            "num_tokens": num_tokens,
            "ctx_tokens": ctx_tokens,
            "gen_tokens": num_gen_requests,
            "backend": database.backend,
            "version": database.version,
            "system": database.system,
            "power_w": agg_power_avg_w,
        }
        result = pd.DataFrame([result_dict], columns=common.ColumnsAgg).round(3)
        summary = InferenceSummary(RuntimeConfig(isl=isl, osl=osl))
        summary.set_memory_and_check_oom(
            memory,
            database.system_spec["gpu"]["mem_capacity"],
            **self._oom_check_kwargs(agg_extra),
        )
        summary.set_encoder_latency_dict(encoder_latency_dict)
        summary.set_encoder_energy_wms_dict(encoder_energy_wms_dict)
        summary.set_encoder_power_avg(encoder_energy_wms / encoder_latency_ms if encoder_latency_ms > 0 else 0.0)
        summary.set_encoder_source_dict(encoder_source_dict)
        summary.set_summary_df(result)
        summary.set_result_dict(result_dict)
        if encoder_memory:
            summary.set_encoder_memory(encoder_memory)

        # Scheduling counters: aggregate sums, not DB queries — recorded in
        # per_ops_data only; no per-op source applies.
        per_ops_data["scheduling"] = {
            "num_mix_steps": float(num_mix_steps),
            "num_genonly_steps": float(num_genonly_steps),
            "mix_step_latency_ms": float(mix_step_latency_ms),
            "genonly_step_latency_ms": float(genonly_step_latency_ms),
        }
        if encoder_latency_dict:
            per_ops_data["encoder"] = dict(encoder_latency_dict)
            per_ops_source["encoder"] = dict(encoder_source_dict)
        summary.set_per_ops_data(per_ops_data)
        summary.set_per_ops_source(per_ops_source)

        self._agg_cache[cache_key] = summary
        return summary

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.

        Args:
            model: the model to be tested
            database: the database to be tested
            runtime_config: the runtime configuration
            top_k: the number of best results to return
            max_batch_size: the maximum batch size to test
            ctx_stride: the stride of ctx tokens to test, it will impact the time to run the test.
            enable_chunked_prefill: whether to enable chunked prefill, it will impact the time to
                run the test while have little impact on the result. Default off.
            **kwargs: additional backend-specific kwargs (e.g. TRT-LLM accepts
                ``max_seq_len`` and ``free_gpu_memory_fraction``).

        Returns:
            A summary of the best agg result under constraints.
        """
        isl = runtime_config.isl
        isl_eff = isl + self._visual_context_tokens(model, runtime_config)
        osl = runtime_config.osl
        ttft = runtime_config.ttft
        tpot = runtime_config.tpot
        top_k = kwargs.get("top_k", 1)
        max_batch_size = kwargs.get("max_batch_size", 512)
        ctx_stride = kwargs.get("ctx_stride", 512)
        enable_chunked_prefill = kwargs.get("enable_chunked_prefill", False)

        # Resolve backend-specific kwargs once; forward into run_agg so each
        # (b, ctx_tokens) point sees the same backend params.
        sweep_extra = self._resolve_agg_kwargs(kwargs, isl=isl_eff, osl=osl)

        # when b is larger than 1024, the result is not good as the data collection is not enough
        # to cover this.
        b_list_default = (
            list(range(1, 16, 1))
            + list(range(16, 32, 4))
            + list(range(32, 64, 8))
            + list(range(64, 256, 16))
            + list(range(256, 512, 32))
            + list(range(512, 1024, 256))
            + [1024]
        )

        # sweep for batch_size and ctx_tokens
        # ctx_tokens will have a step of ctx_stride. When it's larger than 8192, we will increase
        # the step to ctx_stride_large.
        # outer_loop is over batch_size dimention, from 1 to max_batch_size
        # inner_loop is over ctx_tokens dimention, from 0 to max_ctx_tokens where it's
        # max(8192, 4*isl).
        # during the loop, as b, ctx_tokens and system memory are monotonic, we can break the
        # inner loop when the system is oom.
        b_list = [b for b in b_list_default if b <= max_batch_size]
        ctx_tokens_list = self._get_ctx_tokens_list_for_agg_sweep(isl_eff, ctx_stride, enable_chunked_prefill)

        results_df = pd.DataFrame(columns=common.ColumnsAgg)
        results_dict_list: list[dict] = []
        results_per_ops_source: list[dict | None] = []  # aligned with results_dict_list
        capped_b: list[int] = []
        all_oom = True
        for b in b_list:
            for ctx_tokens in ctx_tokens_list:
                if b - np.ceil(ctx_tokens / isl_eff) < 0:  # allow b==1
                    break

                if b > 1 and (
                    b - np.ceil(ctx_tokens / isl_eff) < 1
                ):  # general case, to ensure there's at least one gen req
                    break

                # filter out repeated records for balance score correction
                balance_score = isl_eff * b / ctx_tokens / osl
                if balance_score > 1:
                    gen_tokens = b // balance_score
                    if gen_tokens > 1 and gen_tokens in capped_b:
                        continue
                    else:
                        capped_b.append(gen_tokens)

                summary = self.run_agg(
                    model=model,
                    database=database,
                    runtime_config=self._runtime_config_for_agg_candidate(runtime_config, b),
                    ctx_tokens=ctx_tokens,
                    **sweep_extra,
                )

                if summary.check_oom() or summary.check_kv_cache_oom():
                    break  # larger ctx tokens will cause oom
                all_oom = False
                result_dict = summary.get_result_dict()
                if result_dict and result_dict["tpot"] <= tpot and result_dict["ttft"] <= ttft:
                    results_dict_list.append(result_dict)
                    results_per_ops_source.append(summary.get_per_ops_source())

        if results_dict_list:
            results_df = pd.DataFrame(results_dict_list, columns=common.ColumnsAgg).round(3)
            # Carry per-row per_ops_source as an object column, sorted/truncated alongside the
            # standard columns. report_and_save.py strips this before writing best_config_topn.csv
            # and emits one per_ops_source.json per topN/ subdir.
            results_df["_per_ops_source"] = results_per_ops_source

        sorted_results_df = results_df.sort_values(by="seq/s", ascending=False).round(3)
        if top_k > 0:
            sorted_results_df = sorted_results_df.head(top_k)

        summary = InferenceSummary(runtime_config)
        summary.set_summary_df(sorted_results_df)
        summary.set_oom(all_oom)
        return summary

    # ============== MEMORY USAGE (shared) ==============================

    def _get_memory_usage(
        self,
        model: BaseModel,
        database: PerfDatabase,
        batch_size: int,
        beam_width: int,
        isl: int,
        osl: int,
        num_tokens: int = 0,
        prefix: int = 0,
        max_seq_len: int | None = None,
        encoder_memory: dict[str, float] | None = None,
        mtp_activation_scaling: bool = True,
    ) -> dict[str, float]:
        """
        Get the memory usage of the backend.

        Args:
            prefix: number of prefix tokens (part of isl) whose KV is already cached
                (per-request) and does not need activation computation.
            max_seq_len: per-slot KV cache pre-allocation budget. Defaults to
                ``isl + beam_width * osl`` when not supplied.
            encoder_memory: optional colocated encoder component to add to this worker.
            mtp_activation_scaling: whether to scale activation by ``(nextn + 1)`` for
                speculative decoding (see the MTP correction below). True for the
                latency sweep, where ``num_tokens`` is the per-step token count that the
                multiplier turns into the verified ``nextn + 1`` tokens. False for the
                KV-cache capacity path, where ``num_tokens`` is the engine's
                ``max_num_tokens`` budget that already caps total per-forward tokens
                (draft tokens included), so re-multiplying would double-count.
        """
        weights = 0.0
        for op in model.context_ops:
            weights += op.get_weights()
        # count weights on a single GPU
        weights /= model.config.pp_size

        h = model._num_heads * model._head_size
        if num_tokens == 0:
            num_tokens = (isl - prefix) * batch_size

        tp_clamped = min(model.config.tp_size, 8)
        family = model.model_family
        coeffs_table = self.ACTIVATION_COEFFICIENTS
        coeffs = coeffs_table.get(family, coeffs_table.get("default", {1: 10, 2: 6, 4: 5, 8: 5}))
        activations = 2 * num_tokens * h * coeffs[tp_clamped]

        # MoE block-scale dispatch workspace (only for families that pay this cost).
        # 128 = block scale; 4 = float bytes.
        if family in self.MOE_WORKSPACE_FAMILIES:
            moe_h = self._moe_workspace_width(model, family, h)
            activations += (
                num_tokens
                * moe_h
                * model.config.attention_dp_size
                * model._num_experts
                * model._topk
                / model.config.moe_ep_size
                / 128
                * 4
            )

        activations = max(activations, self.MIN_ACTIVATION_BYTES)

        # MTP correction: speculative decoding verifies nextn+1 tokens per decode step,
        # so the decode-phase activation scales with (nextn+1). Suppressed on the
        # KV-cache capacity path (mtp_activation_scaling=False), where num_tokens is the
        # engine's max_num_tokens budget that already caps total per-forward tokens
        # (draft tokens included) -- re-multiplying there double-counts and can drive the
        # prefill worker's KV budget negative.
        if mtp_activation_scaling and model.config.nextn > 0:
            activations = activations * (model.config.nextn + 1)

        # Backend-level activation overhead (SGLang only by default).
        if self.ACTIVATION_OVERHEAD_FRAC > 0:
            activations *= 1.0 + self.ACTIVATION_OVERHEAD_FRAC

        seq_tokens = max_seq_len if max_seq_len is not None else isl + beam_width * osl
        # CP shards persistent KV across cp ranks (full/cp per rank); the
        # all-gather is a transient compute buffer, not steady-state footprint.
        kvcache = batch_size * model.get_kvcache_bytes_per_sequence(seq_tokens) / model._cp_kv_memory_divisor()
        # should not be divided by pp_size as you need to hold all kvcache for stages.

        # starting from 2.22
        nccl_mem = database.system_spec["misc"]["nccl_mem"][tp_clamped]
        # cuda, cublas, etc.
        others_mem = database.system_spec["misc"]["other_mem"]
        if self.OTHERS_OVERHEAD_FRAC > 0:
            others_mem *= 1.0 + self.OTHERS_OVERHEAD_FRAC

        one_gib = 1 << 30
        if encoder_memory:
            weights += float(encoder_memory.get("weights", 0.0) or 0.0) * one_gib
            activations += float(encoder_memory.get("activations", 0.0) or 0.0) * one_gib
            kvcache += float(encoder_memory.get("kvcache", 0.0) or 0.0) * one_gib
            nccl_mem += float(encoder_memory.get("nccl", 0.0) or 0.0) * one_gib
            others_mem += float(encoder_memory.get("others", 0.0) or 0.0) * one_gib
        return {
            "total": (weights + activations + kvcache + nccl_mem + others_mem) / one_gib,
            "weights": weights / one_gib,
            "activations": activations / one_gib,
            "kvcache": kvcache / one_gib,
            "nccl": nccl_mem / one_gib,
            "others": others_mem / one_gib,
        }
