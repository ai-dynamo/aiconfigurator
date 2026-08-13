# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import dataclasses
import inspect
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import pandas as pd

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.config import RuntimeConfig
from aiconfigurator_core.sdk.inference_summary import InferenceSummary
from aiconfigurator_core.sdk.models import BaseModel
from aiconfigurator_core.sdk.perf_database import PerfDatabase
from aiconfigurator_core.sdk.rust_engine_step import (
    RustEngineUnsupportedError,
    estimate_decode_step_breakdown_with_rust,
    estimate_mixed_step_breakdown_with_rust,
    estimate_static_latency_breakdown_with_rust,
    note_python_step_fallback,
    should_use_rust_engine_step,
)
from aiconfigurator_core.sdk.step_estimate import MixedStepInput, StepEstimate

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _EncoderVisualWorkload:
    """Per-request-image token geometry after checkpoint preprocessing."""

    output_tokens_per_image: int
    context_tokens_per_image: int
    output_tokens_per_sequence: int
    transformer_tokens_per_sequence: int
    patch_tokens_per_sequence: int
    sequences_per_image: int


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
        "STEP3P7",
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

    def _mix_step_gen_tokens(self, b: int, ctx_tokens: int, isl: int, decode_iterations: float) -> int:
        """Return logical decode requests per mix step for a batch of b requests.

        A mix step is a forward pass that contains both prefill tokens (for requests
        still completing their context phase) and decode tokens (for requests already
        generating). This method encodes the engine's scheduling policy for how many
        decode-phase requests participate alongside the prefilling request(s).

        Subclasses should override to match their engine's scheduling behaviour.
        """
        steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
        if steps_to_finish_ctx >= decode_iterations:
            return max(1, int(b // (steps_to_finish_ctx / decode_iterations)))
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

    def _resolve_agg_kwargs(self, kwargs: dict, isl: int, osl: int, backend_version: str | None = None) -> dict:
        """Resolve backend-specific run_agg kwargs to defaults.

        Default: resolves ``free_gpu_memory_fraction`` — an explicit kwarg
        wins, else the backend default (possibly version-dependent, see
        ``get_default_free_gpu_memory_fraction``). Backends without a default
        return an empty dict. TRT-LLM overrides to also resolve
        ``max_seq_len`` / ``max_num_tokens``, so both ``run_agg`` and
        ``find_best_agg_result_under_constraints`` see the same values when
        forwarding. Idempotent — calling with already-resolved kwargs returns
        the same values.
        """
        fraction = kwargs.get("free_gpu_memory_fraction")
        if fraction is None:
            fraction = self.get_default_free_gpu_memory_fraction(backend_version)
        if fraction is None:
            return {}
        return {"free_gpu_memory_fraction": fraction}

    def _make_agg_cache_key(
        self,
        isl: int,
        osl: int,
        b: int,
        ctx_tokens: int,
        engine_step_backend_key: str,
        agg_extra: dict,
    ) -> tuple:
        """Build the cache key for ``run_agg`` results.

        The resolved fraction is part of the key: the cached summary embeds
        the KV-budget OOM verdict, which depends on it.
        """
        return (isl, osl, b, ctx_tokens, engine_step_backend_key, agg_extra.get("free_gpu_memory_fraction"))

    @staticmethod
    def _runtime_config_for_agg_candidate(runtime_config: RuntimeConfig, batch_size: int) -> RuntimeConfig:
        return dataclasses.replace(runtime_config, batch_size=batch_size)

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
        workload = BaseBackend._encoder_workload_per_visual(runtime_config, enc_cfg)
        return workload.context_tokens_per_image * runtime_config.num_images_per_request

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
        """Compatibility wrapper for per-image output and ViT token counts.

        Resolution order:
            1. image_height + image_width (computed from patch/merge sizes)
            2. num_image_tokens (explicit per-image override)

        Returns ``(output_tokens_per_image, transformer_tokens_per_image)``.
        The second value includes per-sequence special tokens and all local or
        global tile sequences for fixed tiled encoders.
        Returns ``(0, 0)`` when neither is set (text-only path).
        """
        workload = BaseBackend._encoder_workload_per_visual(runtime_config, enc_cfg)
        return (
            workload.output_tokens_per_image,
            workload.transformer_tokens_per_sequence * workload.sequences_per_image,
        )

    @staticmethod
    def _tiled_encoder_sequence_count(height: int, width: int, enc_cfg: common.VisionEncoderConfig) -> int:
        """Mirror Llama4ImageProcessor's best-fit canvas and global-tile rule."""
        tile = enc_cfg.image_size
        if height <= 0 or width <= 0 or tile <= 0 or enc_cfg.max_num_tiles <= 0:
            raise ValueError(
                "tiled sequence count requires positive geometry: "
                f"height={height}, width={width}, tile={tile}, max_num_tiles={enc_cfg.max_num_tiles}"
            )
        candidates: list[tuple[int, int]] = []
        for chunk_count in range(enc_cfg.max_num_tiles, 0, -1):
            for factor in range(1, int(chunk_count**0.5) + 1):
                if chunk_count % factor == 0:
                    candidates.append((factor * tile, (chunk_count // factor) * tile))
                    if factor != chunk_count // factor:
                        candidates.append(((chunk_count // factor) * tile, factor * tile))

        scales = [min(target_h / height, target_w / width) for target_h, target_w in candidates]
        upscaling = [scale for scale in scales if scale >= 1]
        if upscaling:
            selected_scale = max(upscaling) if enc_cfg.resize_to_max_canvas else min(upscaling)
        else:
            selected_scale = max(scales)
        viable = [
            candidate
            for candidate, scale in zip(candidates, scales, strict=True)
            if abs(scale - selected_scale) <= 1e-12
        ]
        target_h, target_w = min(viable, key=lambda size: size[0] * size[1])
        local_tiles = (target_h // tile) * (target_w // tile)
        return local_tiles + int(enc_cfg.add_global_tile and local_tiles > 1)

    @staticmethod
    def _encoder_workload_per_visual(
        runtime_config: RuntimeConfig,
        enc_cfg: common.VisionEncoderConfig,
    ) -> _EncoderVisualWorkload:
        """Resolve output, ViT, and raw-patch tokens for one request image.

        Dynamic-resolution encoders retain the legacy one-sequence formula.
        Fixed tiled encoders (Llama 4) use the published image-processor canvas
        selection, run each local/global tile as an independent ViT sequence,
        and account for the per-tile CLS token separately from patch embedding.
        """
        zero = _EncoderVisualWorkload(0, 0, 0, 0, 0, 0)
        has_image_dims = runtime_config.image_height > 0 and runtime_config.image_width > 0

        if enc_cfg.image_size > 0 and enc_cfg.max_num_tiles > 0:
            post_per_sequence = (enc_cfg.image_size // (enc_cfg.patch_size * enc_cfg.spatial_merge_size)) ** 2
            patch_per_sequence = (enc_cfg.image_size // enc_cfg.patch_size) ** 2
            transformer_per_sequence = patch_per_sequence + int(enc_cfg.has_cls_token)
            if has_image_dims:
                sequences = BaseBackend._tiled_encoder_sequence_count(
                    runtime_config.image_height,
                    runtime_config.image_width,
                    enc_cfg,
                )
                output_tokens = post_per_sequence * sequences
            elif runtime_config.num_image_tokens > 0:
                output_tokens = runtime_config.num_image_tokens
                if output_tokens % post_per_sequence != 0:
                    # ``num_image_tokens`` is a public workload-level override,
                    # not a declaration of processor tile geometry. Preserve
                    # its legacy arbitrary-token approximation as one synthetic
                    # sequence when the count cannot describe whole Llama 4
                    # chunks. Image dimensions still take the exact tiled path.
                    patch_tokens = output_tokens * enc_cfg.spatial_merge_size**2
                    return _EncoderVisualWorkload(
                        output_tokens,
                        output_tokens + enc_cfg.prompt_image_tokens,
                        output_tokens,
                        patch_tokens + int(enc_cfg.has_cls_token),
                        patch_tokens,
                        1,
                    )
                sequences = output_tokens // post_per_sequence
            else:
                return zero
            local_tiles = sequences - int(enc_cfg.add_global_tile and sequences > 1)
            context_tokens = output_tokens + enc_cfg.prompt_image_tokens
            if local_tiles > 1:
                context_tokens += local_tiles * enc_cfg.prompt_tokens_per_local_tile
            return _EncoderVisualWorkload(
                output_tokens,
                context_tokens,
                post_per_sequence,
                transformer_per_sequence,
                patch_per_sequence,
                sequences,
            )

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
            return zero
        if tokens_per_image <= 0 or pre_merge_per_image <= 0:
            return zero
        return _EncoderVisualWorkload(
            tokens_per_image,
            tokens_per_image,
            tokens_per_image,
            pre_merge_per_image,
            pre_merge_per_image,
            1,
        )

    def _run_encoder_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        *,
        include_energy: bool = True,
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

        workload = self._encoder_workload_per_visual(runtime_config, enc_cfg)
        if workload.output_tokens_per_image == 0:
            # No image dimensions specified; skip encoder modeling.
            return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, 0

        n_img_context = workload.context_tokens_per_image * num_images

        # Encoder DP: whole images are sharded across the tp_size ranks and the
        # busiest rank (ceil share) gates the phase.
        encoder_dp_size = model.config.tp_size if model.config.enable_encoder_dp else 1
        images_local = -(-batch_size * num_images // encoder_dp_size)
        sequences_local = images_local * workload.sequences_per_image

        # Per-op shape rules (the encoder orchestration — this token math —
        # stays Python-side; only the per-op values may come from the
        # compiled engine below). Projector ops and the DP exit AllGather run
        # on post-merge tokens; ViT attention uses cu_seqlens (each tile is an
        # independent varlen sequence containing its raw patches plus CLS).
        def _encoder_eff_s(op) -> int:
            name = op._name
            if "encoder_patch_embedding" in name:
                return workload.patch_tokens_per_sequence
            if "encoder_attention" in name:
                return workload.transformer_tokens_per_sequence
            if "encoder_projector" in name or name == "encoder_dp_all_gather":
                return workload.output_tokens_per_sequence
            return workload.transformer_tokens_per_sequence

        if should_use_rust_engine_step(runtime_config, database):
            rust = self._run_encoder_phase_with_rust(
                model,
                database,
                sequences_local,
                _encoder_eff_s,
                include_energy=include_energy,
            )
            if rust is not None:
                encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict = rust
                return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, n_img_context

        for op in model.encoder_ops:
            eff_batch, eff_s = sequences_local, _encoder_eff_s(op)
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
            if include_energy:
                encoder_energy_wms_dict[op._name] += getattr(result, "energy", 0.0)
            encoder_source_dict[op._name] = getattr(result, "source", "silicon")

        return encoder_latency_dict, encoder_energy_wms_dict, encoder_source_dict, n_img_context

    def _run_encoder_phase_with_rust(
        self,
        model: BaseModel,
        database: PerfDatabase,
        sequences_local: int,
        eff_s_of,
        *,
        include_energy: bool,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]] | None:
        """Compiled-engine path of the encoder per-op loop, or ``None`` for
        the Python ``op.query()`` fallback.

        Encoder ops are deliberately NOT in the compiled ``EngineSpec`` (the
        compile path threads no image configuration), so they travel through
        the ad-hoc op-list evaluation FFI: ops are grouped by their resolved
        ``eff_s`` (the shape math above), each group serialized to OpSpec
        JSON and evaluated at ``batch=sequences_local, s=eff_s, x=batch*s``.
        Accumulation matches the Python loop for the shipped encoder op
        lists (unique names): latency/energy fold with ``+=``; sources are
        last-wins ACROSS shape groups, while duplicate names WITHIN one
        group would merge to ``"mixed"`` inside the engine (the Python loop
        is last-wins there — divergent only for duplicate-name encoder ops,
        which ``build_encoder_ops`` never emits today).
        """
        from aiconfigurator_core.sdk.engine import OpConversionError, build_ops_json
        from aiconfigurator_core.sdk.rust_engine_step import evaluate_ops_json_with_rust

        groups: dict[int, list] = {}
        for op in model.encoder_ops:
            groups.setdefault(int(eff_s_of(op)), []).append(op)

        latency_dict: dict[str, float] = defaultdict(float)
        energy_dict: dict[str, float] = defaultdict(float)
        source_dict: dict[str, str] = {}
        backend_name = getattr(database.backend, "value", database.backend)
        try:
            for eff_s, ops in groups.items():
                ops_json = build_ops_json(ops, model=model, backend=str(backend_name), database=database)
                entries = evaluate_ops_json_with_rust(
                    model,
                    database,
                    ops_json=ops_json,
                    is_context=True,
                    batch_size=sequences_local,
                    s=eff_s,
                    prefix=0,
                    x=sequences_local * eff_s,
                )
                for name, latency_ms, energy_wms, source in entries:
                    latency_dict[name] += float(latency_ms)
                    if include_energy:
                        energy_dict[name] += float(energy_wms)
                    source_dict[name] = source
        except (OpConversionError, RustEngineUnsupportedError) as exc:
            note_python_step_fallback("unsupported_op_graph:encoder", str(exc))
            return None
        return latency_dict, energy_dict, source_dict

    def _run_context_phase(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        batch_size: int,
        isl: int,
        prefix: int,
        *,
        include_energy: bool = True,
        op_filter: Callable[[str], bool] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """One pass over ``model.context_ops``.

        ``op_filter`` (by op name) lets ``run_mixed``'s passes query ONLY the
        ops whose values they consume — the exact sets the rust mixed step
        walks (`ContextOpFilter::{Skip,Only}ContextAttention`). Querying and
        discarding the rest is not free: a raise in a discarded query blocks
        an otherwise-computable estimate, one-sidedly vs the compiled engine.
        """
        context_latency_dict = defaultdict(float)
        context_energy_wms_dict = defaultdict(float)
        # Per-op data source, accumulated by merging across calls to the same op.
        # Same-source repeated calls keep the tag; mismatched calls collapse to "mixed".
        context_source_dict: dict[str, str] = {}

        effective_isl = isl - prefix
        if effective_isl <= 0:
            raise ValueError(f"isl must be greater than 0 after removing prefix, but got {effective_isl}")

        for op in model.context_ops:
            if op_filter is not None and not op_filter(op._name):
                continue
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
            if include_energy:
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
        *,
        include_energy: bool = True,
        op_filter: Callable[[str], bool] | None = None,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, str]]:
        """One pass over ``model.generation_ops``. ``op_filter`` as in
        :meth:`_run_context_phase` (rust twin: ``only_generation_attention``)."""
        generation_latency_dict = defaultdict(float)
        generation_energy_wms_dict = defaultdict(float)
        generation_source_dict: dict[str, str] = {}

        batch_size = batch_size * (model._nextn + 1)

        for i in range(0, osl - 1, stride):
            latency_dict = defaultdict(float)
            energy_wms_dict = defaultdict(float)

            for op in model.generation_ops:
                if op_filter is not None and not op_filter(op._name):
                    continue
                result = op.query(
                    database,
                    x=batch_size * beam_width,
                    batch_size=batch_size,
                    beam_width=beam_width,
                    s=isl + i + 1,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
                latency_dict[op._name] += float(result)
                if include_energy:
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
                if include_energy:
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
        include_energy: bool = True,
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

        # FPM models compile to no Rust op variant yet; force the Python phases
        # (which handle the single whole-model op naturally).
        if model.forward_model != "fpm" and should_use_rust_engine_step(runtime_config, database):
            try:
                rust_runtime_config = runtime_config
                if img_ctx_tokens:
                    rust_runtime_config = copy.copy(runtime_config)
                    rust_runtime_config.isl = isl_eff
                (
                    context_latency_dict,
                    generation_latency_dict,
                    context_energy_wms_dict,
                    generation_energy_wms_dict,
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
                if not include_energy:
                    # Latency-only callers must not observe energy; keep the
                    # key sets identical to the latency dicts (the power
                    # coverage gate pairs latency and energy by name).
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
                note_python_step_fallback("unsupported_op_graph:static", str(exc))

        if mode == "static_ctx":
            context_latency_dict, context_energy_wms_dict, context_source_dict = self._run_context_phase(
                model,
                database,
                runtime_config,
                batch_size,
                isl_eff,
                prefix,
                include_energy=include_energy,
            )
        elif mode == "static_gen":
            generation_latency_dict, generation_energy_wms_dict, generation_source_dict = self._run_generation_phase(
                model,
                database,
                runtime_config,
                batch_size,
                beam_width,
                isl_eff,
                osl,
                stride,
                include_energy=include_energy,
            )
        else:
            context_latency_dict, context_energy_wms_dict, context_source_dict = self._run_context_phase(
                model,
                database,
                runtime_config,
                batch_size,
                isl_eff,
                prefix,
                include_energy=include_energy,
            )
            generation_latency_dict, generation_energy_wms_dict, generation_source_dict = self._run_generation_phase(
                model,
                database,
                runtime_config,
                batch_size,
                beam_width,
                isl_eff,
                osl,
                stride,
                include_energy=include_energy,
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
            encoder_latency_dict, _, _, img_ctx_tokens = self._run_encoder_phase(
                model,
                database,
                runtime_config,
                runtime_config.batch_size,
                include_energy=False,
            )
            if latency_correction_scale != 1.0:
                for op in encoder_latency_dict:
                    encoder_latency_dict[op] *= latency_correction_scale
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
            include_energy=False,
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
        free_gpu_memory_fraction: float | None = None,
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

        summary.set_deferred_row(data, common.ColumnsStatic)

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
        summary.set_memory_and_check_oom(
            memory,
            database.system_spec["gpu"]["mem_capacity"],
            **self._static_oom_check_kwargs(
                database.system_spec["gpu"]["mem_capacity"],
                free_gpu_memory_fraction=free_gpu_memory_fraction,
                backend_version=database.version,
                model_config=model.config,
            ),
        )
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

        return summary

    def get_default_free_gpu_memory_fraction(self, backend_version: str | None = None) -> float | None:
        """Default KV cache memory fraction for this backend, if it has one.

        ``backend_version`` lets backends whose framework changed the default
        across releases resolve the right value (vLLM 0.19->0.22: 0.90->0.92).
        """
        return None

    def get_kv_cache_memory_check_params(self) -> tuple[float, float]:
        """Return backend-specific KV cache reserved fraction and tolerance."""
        return 0.0, 0.0

    def memory_fraction_of_free(self) -> bool:
        """Whether the memory fraction applies to FREE memory (after non-KV).

        ``True`` for TRT-LLM's ``free_gpu_memory_fraction``; ``False`` for
        backends whose fraction caps TOTAL device memory (vLLM
        ``gpu_memory_utilization`` / SGLang ``mem_fraction_static``).
        """
        return True

    def _static_oom_check_kwargs(
        self,
        mem_capacity_bytes: int | None = None,
        free_gpu_memory_fraction: float | None = None,
        backend_version: str | None = None,
        model_config=None,
    ) -> dict:
        """Fraction-based KV budget kwargs for the static path.

        A user-configured ``free_gpu_memory_fraction`` (Task / estimate API)
        always wins; the backend default — possibly version-dependent (vLLM
        0.14/0.19 ship 0.90, 0.22+ ship 0.92) or capacity-derived (SGLang) —
        is the fallback. Backends without any default return ``{}``, which
        skips the budget check (plain capacity OOM check still applies).
        """
        fraction = (
            free_gpu_memory_fraction
            if free_gpu_memory_fraction is not None
            else self.get_default_free_gpu_memory_fraction(backend_version)
        )
        if fraction is None:
            return {}
        reserved, tolerance = self.get_kv_cache_memory_check_params()
        return {
            "free_gpu_memory_fraction": fraction,
            "kv_cache_reserved_fraction": reserved,
            "kv_cache_tolerance": tolerance,
            "fraction_of_free": self.memory_fraction_of_free(),
        }

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
        """Compatibility wrapper around :meth:`run_mixed`.

        ``isl`` must be the text-only isl; :meth:`run_mixed` derives the visual
        context tokens from ``runtime_config``'s image fields.
        """
        mixed_runtime_config = copy.copy(runtime_config)
        mixed_runtime_config.isl = isl
        mixed_runtime_config.osl = osl
        mixed_runtime_config.prefix = prefix
        estimate = self.run_mixed(
            model,
            database,
            mixed_runtime_config,
            MixedStepInput(
                context_tokens=ctx_tokens,
                num_decode_requests=gen_tokens,
            ),
        )
        return estimate.legacy_tuple()

    def run_mixed(
        self,
        model: BaseModel,
        database: PerfDatabase,
        runtime_config: RuntimeConfig,
        step: MixedStepInput,
    ) -> StepEstimate:
        """Estimate one scheduled mixed prefill/decode forward pass.

        ``runtime_config.isl`` is the text-only isl; the visual context tokens
        implied by the config's image fields are added here, exactly as
        ``run_static`` / ``run_agg`` do, so direct callers (e.g.
        ``InferenceSession.run_mixed``) get the effective sequence length
        without pre-adjusting the config.
        """
        isl = int(runtime_config.isl or 0)
        osl = int(runtime_config.osl or 0)
        prefix = int(runtime_config.prefix or 0)
        if isl <= 0:
            raise ValueError("runtime_config.isl must be positive for a mixed step")
        if osl <= 0:
            raise ValueError("runtime_config.osl must be positive for a mixed step")
        if prefix < 0:
            raise ValueError("runtime_config.prefix must be non-negative")
        # The internal pass configs below carry no image fields, so the visual
        # contribution is applied exactly once.
        isl += self._visual_context_tokens(model, runtime_config)

        decode_query_tokens = step.num_decode_requests * (model._nextn + 1)
        if should_use_rust_engine_step(runtime_config, database):
            try:
                components = estimate_mixed_step_breakdown_with_rust(
                    model,
                    database,
                    ctx_tokens=step.context_tokens,
                    gen_tokens=step.num_decode_requests,
                    isl=isl,
                    osl=osl,
                    prefix=prefix,
                    seq_imbalance_correction_scale=runtime_config.seq_imbalance_correction_scale,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
            except RustEngineUnsupportedError as exc:
                note_python_step_fallback("unsupported_op_graph:mixed", str(exc))
            else:
                return StepEstimate(
                    latency_ms=components["latency_ms"],
                    energy_wms=components["energy_wms"],
                    component_latency_ms=components["component_latency_ms"],
                    component_energy_wms=components["component_energy_wms"],
                    per_op_latency_ms=components["per_op_latency_ms"],
                    per_op_source=components["per_op_source"],
                    context_tokens=step.context_tokens,
                    num_decode_requests=step.num_decode_requests,
                    num_decode_query_tokens=decode_query_tokens,
                )

        if model.forward_model == "fpm":
            # The 3-pass split below recognizes granular attention op NAMES
            # ("context_attention"/"generation_attention"); a whole-model op
            # matches neither and would be mis-accounted. Compose pure
            # prefill + the decode work's MARGINAL cost instead. The Rust
            # route above already branches identically inside
            # Engine::mixed_step_breakdown.
            latency_ms, energy_wms, per_op_latency, per_op_source = self._get_fpm_mix_step_latency(
                model,
                database,
                runtime_config,
                step.context_tokens,
                step.num_decode_requests,
                isl,
                osl,
                prefix,
            )
            return StepEstimate(
                latency_ms=latency_ms,
                energy_wms=energy_wms,
                per_op_latency_ms=per_op_latency,
                per_op_source=per_op_source,
                context_tokens=step.context_tokens,
                num_decode_requests=step.num_decode_requests,
                num_decode_query_tokens=decode_query_tokens,
            )

        ctx_scale = runtime_config.seq_imbalance_correction_scale
        gen_scale = runtime_config.gen_seq_imbalance_correction_scale

        # The three passes below query ONLY the ops whose values they consume
        # (pass 1: everything but context_attention; pass 2/3: the attention
        # bucket alone) — the exact op sets the rust mixed step walks
        # (`ContextOpFilter::SkipContextAttention` / `OnlyContextAttention` /
        # `only_generation_attention`). The passes previously ran the full
        # lists through `run_static` and discarded the rest, so a raise in a
        # discarded query (e.g. a generation-MoE low-token miss in pass 3)
        # blocked the estimate on the Python side only — a one-sided error
        # surface vs the compiled engine (issue #1498 follow-through).

        # Pass 1: combined single-batch inference for the non-attention latency.
        # Every decode request verifies one target token plus all scheduled
        # drafts. Acceptance does not reduce this current-iteration work.
        num_tokens_combined = step.context_tokens + decode_query_tokens
        # num tokens for gemm needs to be adjusted for prefix, depends on the
        # avg prefix len per request. int(): np.floor yields a float64 that
        # contaminates the ops' s = isl - prefix (the DSV4 CP composition
        # chunks s with range()); the product is integral, so the cast is
        # lossless.
        prefix_combined = int(prefix * np.floor(step.context_tokens / isl))
        pass1_config = RuntimeConfig(
            batch_size=1,
            beam_width=1,
            isl=num_tokens_combined,
            osl=1,
            prefix=prefix_combined,
            seq_imbalance_correction_scale=ctx_scale,
        )
        latency_dict, energy_wms_dict, source_dict = self._run_context_phase(
            model,
            database,
            pass1_config,
            1,
            num_tokens_combined,
            prefix_combined,
            op_filter=lambda name: name != "context_attention",
        )
        non_attention_latency_ms = float(sum(latency_dict.values()))
        non_attention_energy_wms = float(sum(energy_wms_dict.values()))
        mix_non_attn_ops: dict[str, float] = dict(latency_dict)
        mix_non_attn_sources: dict[str, str] = {
            layer_name: source_dict.get(layer_name, "silicon") for layer_name in latency_dict
        }

        # Pass 2: context attention split full isl over num_steps and averaged.
        batch_size = np.ceil(step.context_tokens / isl)
        pass2_config = RuntimeConfig(
            batch_size=batch_size,
            beam_width=1,
            isl=isl,
            osl=1,
            prefix=prefix,
            seq_imbalance_correction_scale=ctx_scale,
        )
        latency_dict, energy_wms_dict, source_dict = self._run_context_phase(
            model,
            database,
            pass2_config,
            batch_size,
            isl,
            prefix,
            op_filter=lambda name: name == "context_attention",
        )
        ctx_attn_source = source_dict.get("context_attention", "silicon")
        scale_factor = np.ceil(isl / step.context_tokens)
        ctx_attention_latency_ms = latency_dict["context_attention"] / scale_factor
        ctx_attention_energy_wms = energy_wms_dict.get("context_attention", 0.0) / scale_factor

        # Pass 3: generation attention (use isl + osl//2 for the avg seq len).
        gen_attention_latency_ms = 0.0
        gen_attention_energy_wms = 0.0
        gen_attn_source = "silicon"
        if step.num_decode_requests > 0:
            pass3_config = RuntimeConfig(
                batch_size=step.num_decode_requests,
                beam_width=1,
                isl=isl + osl // 2,
                osl=2,
                gen_seq_imbalance_correction_scale=gen_scale,
            )
            latency_dict, energy_wms_dict, source_dict = self._run_generation_phase(
                model,
                database,
                pass3_config,
                step.num_decode_requests,
                1,
                isl + osl // 2,
                2,
                1,
                op_filter=lambda name: name == "generation_attention",
            )
            gen_attention_latency_ms = latency_dict["generation_attention"]
            gen_attention_energy_wms = energy_wms_dict.get("generation_attention", 0.0)
            gen_attn_source = source_dict.get("generation_attention", "silicon")

        per_ops_step_data: dict[str, float] = {
            **mix_non_attn_ops,
            "context_attention (scaled)": ctx_attention_latency_ms,
            "generation_attention": gen_attention_latency_ms,
        }
        per_ops_step_source = {
            **mix_non_attn_sources,
            "context_attention (scaled)": ctx_attn_source,
            "generation_attention": gen_attn_source,
        }

        component_latency_ms = {
            "shared_non_attention": non_attention_latency_ms,
            "context_attention": ctx_attention_latency_ms,
            "decode_attention": gen_attention_latency_ms,
        }
        component_energy_wms = {
            "shared_non_attention": non_attention_energy_wms,
            "context_attention": ctx_attention_energy_wms,
            "decode_attention": gen_attention_energy_wms,
        }
        return StepEstimate(
            latency_ms=sum(component_latency_ms.values()),
            energy_wms=sum(component_energy_wms.values()),
            component_latency_ms=component_latency_ms,
            component_energy_wms=component_energy_wms,
            per_op_latency_ms=per_ops_step_data,
            per_op_source=per_ops_step_source,
            context_tokens=step.context_tokens,
            num_decode_requests=step.num_decode_requests,
            num_decode_query_tokens=decode_query_tokens,
        )

    def _get_fpm_mix_step_latency(
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
        """Mixed step for FPM models: the step's SCHEDULED TOTALS priced on
        the prefill curve, plus the decode work's MARGINAL cost. No mixed
        database row exists or is synthesized.

        A mixed step is one shared forward pass: weight reads, kernel
        launches, and per-step fixed overheads are paid once, by the prefill
        component. Summing a full pure-decode step would pay them twice, so
        the decode component is composed as
        ``decode(B, KV) - decode_pass_baseline(B)`` — the KV-axis floor of the
        decode curve cancels the shared-pass part, keeping only the
        KV-read/attention cost that genuinely adds to the iteration. A
        generation-only step (``ctx_tokens == 0``) has no pass to ride on and
        keeps the full decode latency.

        The prefill component is addressed at the iteration's REAL scheduled
        totals via ``query_totals``: the engine picks its execution regime
        (CUDA-graph vs eager) and its GEMM width from the step's total token
        count — prefill chunk PLUS decode tokens — so the query coordinate
        must carry both, and the collected rows (which encode the regime
        cliffs) answer on the correct side. When ``ctx_tokens`` is a chunk of
        a longer request, each chunk is priced at its own
        ``(chunk + gen_tokens, past_kv)`` coordinates and the per-iteration
        AVERAGE is returned (callers treat the mixed step as the steady-state
        iteration); the decode tokens' GEMM/weights are then counted exactly
        once, because the baseline subtraction removed them from the decode
        term.

        CONTRACT PREREQUISITE: this pricing is only as correct as the match
        between the deployed engine configuration and the collected data —
        in particular the CUDA-graph capture surface. The cliff positions are
        not modeled; they are encoded in the data. A deployment whose capture
        config differs from the collection cannot be rescued by this formula.
        """
        per_ops_step_data: dict[str, float] = {}
        per_ops_step_source: dict[str, str] = {}
        total_latency_ms = 0.0
        total_energy_wms = 0.0

        if ctx_tokens > 0:
            prefill_op = model.context_ops[0]
            new_tokens = isl - prefix
            if new_tokens <= 0:
                raise ValueError(f"isl must be greater than prefix, got isl={isl} prefix={prefix}")
            if ctx_tokens >= new_tokens:
                # One or more whole prefills scheduled in this iteration (no
                # chunking): the step's scheduled total (ctx + decode tokens)
                # picks the regime row.
                batch = int(np.ceil(ctx_tokens / new_tokens))
                results = [
                    prefill_op.query_totals(
                        database,
                        batch_size=batch,
                        total_prefill_tokens=ctx_tokens + gen_tokens,
                        total_kv_read_tokens=batch * prefix,
                    )
                ]
            else:
                # Chunked prefill: one request spread over ceil(new/ctx)
                # iterations. Price each chunk at ITS scheduled totals —
                # chunk + decode tokens over the chunk's already-computed
                # context — and report the per-iteration average.
                results = []
                done = 0
                while done < new_tokens:
                    chunk = min(ctx_tokens, new_tokens - done)
                    results.append(
                        prefill_op.query_totals(
                            database,
                            batch_size=1,
                            total_prefill_tokens=chunk + gen_tokens,
                            total_kv_read_tokens=prefix + done,
                        )
                    )
                    done += chunk
            pre_ms = sum(float(r) for r in results) / len(results)
            sources = {getattr(r, "source", "silicon") for r in results}
            per_ops_step_data[prefill_op._name] = pre_ms
            per_ops_step_source[prefill_op._name] = sources.pop() if len(sources) == 1 else "mixed"
            total_latency_ms += pre_ms

        if gen_tokens > 0:
            summary = self.run_static(
                model,
                database,
                RuntimeConfig(
                    batch_size=gen_tokens,
                    beam_width=1,
                    isl=isl + osl // 2,
                    osl=2,
                    engine_step_backend=runtime_config.engine_step_backend,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                ),
                mode="static_gen",
            )
            pass_baseline_ms = 0.0
            if ctx_tokens > 0:
                # _run_generation_phase scaled the batch by (nextn + 1); the
                # baseline must be sampled at the same effective batch.
                pass_baseline_ms = float(
                    model.generation_ops[0].query_pass_baseline(database, batch_size=gen_tokens * (model._nextn + 1))
                )
            energy_dict = summary.get_generation_energy_wms_dict()
            source_dict = summary.get_generation_source_dict()
            for op_name, latency in summary.get_generation_latency_dict().items():
                latency = max(latency - pass_baseline_ms, 0.0)
                per_ops_step_data[op_name] = per_ops_step_data.get(op_name, 0.0) + latency
                per_ops_step_source[op_name] = source_dict.get(op_name, "silicon")
                total_latency_ms += latency
                total_energy_wms += energy_dict.get(op_name, 0.0)

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
        if model.forward_model != "fpm" and should_use_rust_engine_step(runtime_config, database):
            try:
                return estimate_decode_step_breakdown_with_rust(
                    model,
                    database,
                    gen_tokens=gen_tokens,
                    isl=isl,
                    osl=osl,
                    gen_seq_imbalance_correction_scale=runtime_config.gen_seq_imbalance_correction_scale,
                )
            except RustEngineUnsupportedError as exc:
                note_python_step_fallback("unsupported_op_graph:decode", str(exc))

        gen_scale = runtime_config.gen_seq_imbalance_correction_scale
        summary = self.run_static(
            model,
            database,
            RuntimeConfig(
                batch_size=gen_tokens,
                beam_width=1,
                isl=isl + osl // 2,
                osl=2,
                engine_step_backend=runtime_config.engine_step_backend,
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

    def _get_encoder_component_memory(self, model: BaseModel, num_tokens: int, embed_tokens: int) -> dict[str, float]:
        """Encoder memory component colocated with the prefill/agg worker.

        num_tokens: transformer tokens run through the ViT on this rank.
        embed_tokens: post-merge tokens of the projected-embeddings buffer
        every rank holds at the encoder exit (the full batch in both modes).
        """
        weights = sum(op.get_weights() for op in model.encoder_ops)
        enc_cfg = getattr(model, "encoder_config", None)
        activations = 0.0
        if isinstance(enc_cfg, common.VisionEncoderConfig) and num_tokens > 0:
            # ~3x hidden_size per patch covers QKV, attention output, and FFN intermediates (bfloat16)
            activations = 2 * num_tokens * enc_cfg.hidden_size * 3
            # Projected embeddings (all projector instances concatenated along hidden)
            activations += 2 * embed_tokens * enc_cfg.out_hidden_size * enc_cfg.projector_n_instances
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
        workload = self._encoder_workload_per_visual(runtime_config, enc_cfg)
        if workload.transformer_tokens_per_sequence <= 0:
            return {}
        # ViT activations follow the busiest rank's image share; the embeddings
        # buffer covers the full batch on every rank.
        total_images = batch_size * runtime_config.num_images_per_request
        encoder_dp_size = model.config.tp_size if model.config.enable_encoder_dp else 1
        images_local = -(-total_images // encoder_dp_size)
        num_tokens = images_local * workload.sequences_per_image * workload.transformer_tokens_per_sequence
        embed_tokens = total_images * workload.output_tokens_per_image
        return self._get_encoder_component_memory(model, num_tokens, embed_tokens)

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
        # None (or an omitted kwarg) means the caller did not model speculative
        # progress here; the summary then stays eligible for the upper-layer
        # post-hoc projection (SpeculativeDecodingProfile.project_summary).
        _explicit_progress = kwargs.pop("decode_tokens_per_iteration", None)
        speculative_scheduling = _explicit_progress is not None
        decode_tokens_per_iteration = float(_explicit_progress) if speculative_scheduling else 1.0
        max_decode_progress = float(model._nextn + 1)
        if (
            not math.isfinite(decode_tokens_per_iteration)
            or decode_tokens_per_iteration < 1.0
            or decode_tokens_per_iteration > max_decode_progress
        ):
            raise ValueError(
                f"decode_tokens_per_iteration must be finite and within [1, nextn + 1={max_decode_progress:g}]"
            )
        decode_iterations = 1.0 + max(osl - 1, 0) / decode_tokens_per_iteration
        balance_score = isl * b / ctx_tokens / decode_iterations

        # Backend-specific kwargs (TRT-LLM: max_seq_len / max_num_tokens /
        # free_gpu_memory_fraction; vLLM / SGLang: free_gpu_memory_fraction).
        agg_extra = self._resolve_agg_kwargs(kwargs, isl=isl, osl=osl, backend_version=database.version)

        visual_cache_key = (
            runtime_config.image_height,
            runtime_config.image_width,
            runtime_config.num_images_per_request,
        )
        cache_key = (
            self._make_agg_cache_key(isl, osl, b, ctx_tokens, engine_step_backend_key, agg_extra),
            visual_cache_key,
            # Explicit progress and an omitted kwarg schedule identically at
            # 1.0 but record different scheduling metadata, so they must not
            # share a cache entry.
            decode_tokens_per_iteration if speculative_scheduling else None,
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

        # Compute the mean-field number of engine iterations needed to consume
        # all context and commit the requested output tokens.
        steps_to_finish_ctx = np.ceil(isl * b / ctx_tokens)
        num_mix_steps = num_genonly_steps = 0
        num_mix_steps_for_tpot_calc = 0  # correction for tpot calc only
        if b > 1:
            num_mix_gen_tokens = self._mix_step_gen_tokens(b, ctx_tokens, isl, decode_iterations)
            assert num_mix_gen_tokens >= 1, (
                f"num_mix_gen_tokens: {num_mix_gen_tokens}, b: {b}, ctx_tokens: {ctx_tokens}, isl: {isl}"
            )
            num_mix_ctx_tokens = ctx_tokens
            if steps_to_finish_ctx >= decode_iterations:
                num_mix_steps = steps_to_finish_ctx
                num_genonly_steps = 0
                num_genonly_tokens = 0
                num_mix_steps_for_tpot_calc = num_mix_steps
            else:
                num_mix_steps = steps_to_finish_ctx
                num_genonly_steps = decode_iterations - num_mix_steps
                num_genonly_tokens = b
                num_mix_steps_for_tpot_calc = self._tpot_mix_steps(num_mix_steps)
        elif b == 1:
            # special case for b=1
            num_mix_steps = 1
            num_mix_ctx_tokens = ctx_tokens
            num_mix_gen_tokens = 0
            num_genonly_steps = max(decode_iterations - 1.0, 0.0)
            num_genonly_tokens = 1
            num_mix_steps_for_tpot_calc = 0

        # Step-latency helpers (return (latency_ms, energy_wms, per_op_data, per_op_source)).
        per_ops_data: dict[str, dict] = {}
        per_ops_source: dict[str, dict] = {}

        # run_mixed derives the image-augmented effective isl from the config's
        # own image fields, so the raw runtime_config is passed unchanged (no
        # pre-adjustment here, or the visual tokens would be counted twice).
        mix_step_estimate = self.run_mixed(
            model,
            database,
            runtime_config,
            MixedStepInput(
                context_tokens=num_mix_ctx_tokens,
                num_decode_requests=num_mix_gen_tokens,
            ),
        )
        mix_step_latency_ms = mix_step_estimate.latency_ms
        mix_step_energy_wms = mix_step_estimate.energy_wms
        mix_per_ops = mix_step_estimate.per_op_latency_ms
        mix_per_ops_src = mix_step_estimate.per_op_source
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
            / decode_tokens_per_iteration
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
            "mix_step_energy_wms": float(mix_step_energy_wms),
            "genonly_step_energy_wms": float(genonly_step_energy_wms),
            "mix_efficiency": float(mix_efficiency),
            "decode_iterations": decode_iterations,
            "mix_context_tokens": float(num_mix_ctx_tokens),
            "mix_decode_requests": float(num_mix_gen_tokens),
            "mix_decode_query_tokens": float(mix_step_estimate.num_decode_query_tokens),
        }
        if speculative_scheduling:
            # Recorded only when the caller explicitly supplied the progress:
            # its presence tells SpeculativeDecodingProfile.project_summary
            # that the scheduler already modeled it, so the post-hoc scalar
            # projection must not be applied on top.
            per_ops_data["scheduling"]["decode_tokens_per_iteration"] = decode_tokens_per_iteration
        if encoder_latency_dict:
            per_ops_data["encoder"] = dict(encoder_latency_dict)
            per_ops_source["encoder"] = dict(encoder_source_dict)
        summary.set_per_ops_data(per_ops_data)
        summary.set_per_ops_source(per_ops_source)
        summary.set_step_estimates(
            {
                # Raw run_mixed output (pre-mix_efficiency); the authoritative
                # scheduled latency is scheduling["mix_step_latency_ms"], which
                # already includes the mix_efficiency scale.
                "mixed": mix_step_estimate,
                "scheduling": dict(per_ops_data["scheduling"]),
            }
        )

        self._agg_cache[cache_key] = summary
        return summary

    def find_best_agg_result_under_constraints(
        self, model: BaseModel, database: PerfDatabase, runtime_config: RuntimeConfig, **kwargs
    ) -> InferenceSummary:
        """
        Find the best agg result under constraints.

        Note: this legacy sweep is not speculation-aware — it never forwards
        ``decode_tokens_per_iteration`` to :meth:`run_agg`, so its TPOT filter
        compares unprojected values. Speculative workloads should use the
        ``sweep.py`` path via ``predict_agg_worker``.

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
        sweep_extra = self._resolve_agg_kwargs(kwargs, isl=isl_eff, osl=osl, backend_version=database.version)

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
        if family in self.MOE_WORKSPACE_FAMILIES and model._num_experts:
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
