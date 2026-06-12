# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate vLLM layerwise datapoints and mocked model work units."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _THIS_DIR.parent / "common"
sys.path.insert(0, str(_COMMON_DIR))

from config_patch import patch_model_path
from parallel_config_patch import EXPERT_COUNT_KEYS, _load_original_config, patch_for_parallelism

try:
    from .data import DataPoint, RepresentativeLayer, WorkUnit
    from .registry import LayerwiseModel
    from .runtime import (
        _get_system_name,
        _get_vllm_default_max_num_seqs,
        _get_vllm_version,
        _infer_default_max_num_seqs_from_system,
        _parse_ints,
        _stable_hash,
    )
except ImportError:  # pragma: no cover - direct script compatibility
    from data import DataPoint, RepresentativeLayer, WorkUnit
    from registry import LayerwiseModel
    from runtime import (
        _get_system_name,
        _get_vllm_default_max_num_seqs,
        _get_vllm_version,
        _infer_default_max_num_seqs_from_system,
        _parse_ints,
        _stable_hash,
    )


CTX_NEW_TOKENS = [1, 16, 128, 256, 512, 1024, 2048, 4096, 8192]
PAST_KV_TOKENS = [16, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
CTX_PAST_KV = [0, *PAST_KV_TOKENS]
GEN_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GEN_PAST_KV = [1, *PAST_KV_TOKENS]
SMOKE_CTX_NEW_TOKENS = [1024, 4096]
SMOKE_CTX_PAST_KV = [0]
SMOKE_GEN_BATCH_SIZES = [1, 8]
SMOKE_GEN_PAST_KV = [1024]
LIVE_DECODE_OUTPUT_TOKENS = 2


def _batch_sizes_up_to(max_batch_size: int) -> list[int]:
    """Return the canonical power-of-two decode batch grid up to a max."""

    sizes = [value for value in GEN_BATCH_SIZES if value <= max_batch_size]
    if max_batch_size > 0 and max_batch_size not in sizes:
        sizes.append(max_batch_size)
    return sorted(set(sizes))


def values_for_preset(preset: str, *, max_decode_batch_size: int | None = None) -> dict[str, str]:
    """Return CSV shape defaults for the named public run preset."""
    if preset == "full":
        gen_batch_sizes = (
            _batch_sizes_up_to(max_decode_batch_size)
            if max_decode_batch_size is not None else GEN_BATCH_SIZES
        )
        return {
            "ctx_new_tokens": ",".join(map(str, CTX_NEW_TOKENS)),
            "ctx_past_kv": ",".join(map(str, CTX_PAST_KV)),
            "gen_batch_sizes": ",".join(map(str, gen_batch_sizes)),
            "gen_past_kv": ",".join(map(str, GEN_PAST_KV)),
        }
    if preset == "smoke":
        return {
            "ctx_new_tokens": ",".join(map(str, SMOKE_CTX_NEW_TOKENS)),
            "ctx_past_kv": ",".join(map(str, SMOKE_CTX_PAST_KV)),
            "gen_batch_sizes": ",".join(map(str, SMOKE_GEN_BATCH_SIZES)),
            "gen_past_kv": ",".join(map(str, SMOKE_GEN_PAST_KV)),
        }
    raise ValueError(f"unknown run preset: {preset}")


def resolve_max_decode_batch_size(raw: str, *, system: str | None, tp_size: int) -> int:
    """Resolve the public max decode batch setting to an integer."""

    if raw != "auto":
        value = int(raw)
        if value < 1:
            raise ValueError(f"max decode batch size must be >= 1, got {value}")
        return value
    detected = _get_vllm_default_max_num_seqs(world_size=tp_size)
    if detected is not None:
        return detected
    return _infer_default_max_num_seqs_from_system(system)


def parse_csv_ints_or_auto(raw: str) -> list[int] | str:
    """Parse a CSV integer list, preserving the special 'auto' value."""
    if raw == "auto":
        return "auto"
    return _parse_ints(raw)


def _resolve_real_weight_model_dir(model: str) -> str:
    """Return a local model directory suitable for real-weight loading."""

    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)


def ep_sizes_for_model(model: LayerwiseModel, raw_ep_sizes: str, tp: int) -> list[int]:
    """Resolve vLLM-parity EP sizes for one registry model and TP size.

    vLLM does not expose an independent MoE-TP axis. With expert parallelism
    enabled, MoE layers set local MoE TP to 1 and distribute experts over the
    full TP/DP group. In this collector schema that means the deployment-parity
    cases are TP-only (ep=1) and full expert parallel (ep=tp).
    """
    if not model.is_moe:
        return [1]
    parsed = parse_csv_ints_or_auto(raw_ep_sizes)
    candidates = list(model.ep_sizes) if parsed == "auto" else parsed
    return [ep for ep in candidates if ep == 1 or ep == tp]


def make_work_unit_args(
    public_args: argparse.Namespace,
    model: LayerwiseModel,
    *,
    tp_size: int,
    ep_size: int,
    phase: str | None = None,
) -> argparse.Namespace:
    """Convert public CLI args into lower-level work-unit generator args."""
    phases = phase or public_args.phases
    if public_args.target_layer_count is not None:
        target_layer_count = public_args.target_layer_count
    elif phases == "ctx":
        target_layer_count = public_args.ctx_target_layer_count
    else:
        target_layer_count = public_args.gen_target_layer_count
    system = public_args.system or _get_system_name()
    max_decode_batch_size = resolve_max_decode_batch_size(
        public_args.max_decode_batch_size,
        system=system,
        tp_size=tp_size,
    )
    preset_values = values_for_preset(
        public_args.run_preset,
        max_decode_batch_size=max_decode_batch_size if phases in ("gen", "both") else None,
    )
    moe_tp = tp_size // ep_size if model.is_moe else 1
    moe_dummy_router = bool(getattr(public_args, "moe_dummy_router", False))
    use_moe_noop = public_args.moe_noop or (
        model.is_moe
        and not public_args.moe_real_router
        and not moe_dummy_router
        and phases != "ctx"
    )
    return argparse.Namespace(
        model=model.model,
        work_dir=str(public_args.run_dir / "profiles"),
        config_cache_dir=str(public_args.run_dir / "config_cache"),
        no_config_cache=public_args.no_config_cache,
        system=system,
        framework_version=public_args.framework_version,
        tp_sizes=str(tp_size),
        moe_tp=moe_tp,
        num_slots=model.num_slots,
        moe_noop=use_moe_noop,
        target_layer_count=target_layer_count,
        target_layers=public_args.target_layers,
        target_layer_config_depth=public_args.target_layer_config_depth,
        phases=phases,
        ctx_new_tokens=public_args.ctx_new_tokens or preset_values["ctx_new_tokens"],
        ctx_past_kv=public_args.ctx_past_kv or preset_values["ctx_past_kv"],
        no_filter_model_max_len=public_args.no_filter_model_max_len,
        gen_batch_sizes=public_args.gen_batch_sizes or preset_values["gen_batch_sizes"],
        gen_past_kv=public_args.gen_past_kv or preset_values["gen_past_kv"],
        max_num_seqs=public_args.max_num_seqs,
        max_num_batched_tokens=public_args.max_num_batched_tokens,
        max_model_len=public_args.max_model_len,
        gen_driver=model.gen_driver,
        latency_source=public_args.latency_source,
        gemm_quant=model.gemm_quant,
        moe_quant=model.moe_quant,
        attn_quant=model.attn_quant,
        kv_quant=model.kv_quant,
        moe_real_router=bool(public_args.moe_real_router),
        physical_tp=public_args.physical_tp,
        physical_tp_real_weights=public_args.physical_tp_real_weights,
    )


def build_public_work_units(public_args: argparse.Namespace, models: list[LayerwiseModel]) -> list[WorkUnit]:
    """Expand public model and TP/EP filters into scheduler work units."""
    work_units: list[WorkUnit] = []
    requested_tp_sizes = _parse_ints(public_args.tp_sizes)
    phases = ("ctx", "gen") if public_args.phases == "both" else (public_args.phases,)
    for model in models:
        for tp in requested_tp_sizes:
            if tp not in model.tp_sizes:
                continue
            for ep in ep_sizes_for_model(model, public_args.ep_sizes, tp):
                for phase in phases:
                    work_units.extend(
                        build_work_units(make_work_unit_args(public_args, model, tp_size=tp, ep_size=ep, phase=phase))
                    )
    return work_units


def _detect_layer_schedule(
    config: dict[str, Any],
    target_layer_count: int = 1,
    target_layers: list[int] | None = None,
    target_layer_config_depth: int | None = None,
    expand_layer_types: bool = True,
) -> tuple[list[RepresentativeLayer], int, dict[str, Any] | None]:
    config = _decoder_config_view(config)
    max_config_layers = int(config.get("num_hidden_layers") or 0)
    if target_layers is not None:
        if not target_layers:
            raise ValueError("target_layers must not be empty")
        if any(i < 0 for i in target_layers):
            raise ValueError(f"target_layers must be non-negative, got {target_layers}")
        if max_config_layers and max(target_layers) >= max_config_layers:
            raise ValueError(
                f"target_layers {target_layers} exceed config num_hidden_layers="
                f"{max_config_layers}"
            )
        is_moe = _is_moe_config(config)
        if is_moe and not _is_all_moe_config(config):
            raise ValueError(
                "explicit target_layers for hybrid MoE configs is not supported; "
                "hybrid representative collection is not implemented yet"
            )
        sorted_layers = sorted(set(target_layers))
        num_hidden_layers = max(sorted_layers) + 1
        if target_layer_config_depth is not None:
            if target_layer_config_depth < num_hidden_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than required depth {num_hidden_layers}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
            )
            num_hidden_layers = target_layer_config_depth
        return [
            RepresentativeLayer(
                layer_index=sorted_layers[0],
                layer_type="moe" if is_moe else "dense",
                measured_layer_count=len(sorted_layers),
                layer_multiplier=len(sorted_layers),
                target_layers=tuple(sorted_layers),
            )
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    if target_layer_count < 1:
        raise ValueError(f"target_layer_count must be >= 1, got {target_layer_count}")
    if not _is_moe_config(config):
        typed_representatives = (
            _representatives_from_layer_types(config, suffix="dense")
            if expand_layer_types
            else None
        )
        if typed_representatives is not None:
            return typed_representatives
        num_hidden_layers = target_layer_count
        if target_layer_config_depth is not None:
            if target_layer_config_depth < target_layer_count:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than target_layer_count={target_layer_count}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            RepresentativeLayer(
                layer_index=0,
                layer_type="dense",
                measured_layer_count=target_layer_count,
                layer_multiplier=max_config_layers or target_layer_count,
                target_layers=tuple(range(target_layer_count)),
            )
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    if _is_all_moe_config(config):
        typed_representatives = _representatives_from_layer_types(config, suffix="moe") if expand_layer_types else None
        if typed_representatives is not None:
            return typed_representatives
        num_hidden_layers = target_layer_count
        if target_layer_config_depth is not None:
            if target_layer_config_depth < target_layer_count:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} is smaller "
                    f"than target_layer_count={target_layer_count}"
                )
            if max_config_layers and target_layer_config_depth > max_config_layers:
                raise ValueError(
                    f"target_layer_config_depth={target_layer_config_depth} exceeds "
                    f"config num_hidden_layers={max_config_layers}"
                )
            num_hidden_layers = target_layer_config_depth
        return [
            RepresentativeLayer(
                layer_index=0,
                layer_type="moe",
                measured_layer_count=target_layer_count,
                layer_multiplier=max_config_layers or target_layer_count,
                target_layers=tuple(range(target_layer_count)),
            )
        ], num_hidden_layers, _layer_types_override(config, num_hidden_layers)

    raise ValueError(
        "hybrid dense+MoE layerwise collection is not implemented yet; "
        "the collector schema supports representative layer metadata, but "
        "hybrid scheduling needs separate dense and MoE work units"
    )

def _layer_types_override(
    config: dict[str, Any],
    num_hidden_layers: int,
) -> dict[str, Any] | None:
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list):
        return None
    if len(layer_types) < num_hidden_layers:
        raise ValueError(
            f"layer_types has length {len(layer_types)} but num_hidden_layers="
            f"{num_hidden_layers}"
        )
    return {"layer_types": list(layer_types[:num_hidden_layers])}

def _representatives_from_layer_types(
    config: dict[str, Any],
    *,
    suffix: str,
) -> tuple[list[RepresentativeLayer], int, dict[str, Any] | None] | None:
    """Return one representative layer per distinct config ``layer_types`` value."""

    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        return None
    positions_by_type: dict[str, list[int]] = {}
    for index, raw_layer_type in enumerate(layer_types):
        positions_by_type.setdefault(str(raw_layer_type), []).append(index)
    if len(positions_by_type) <= 1:
        return None

    representatives = [
        RepresentativeLayer(
            layer_index=positions[0],
            layer_type=f"{layer_type}_{suffix}" if suffix else layer_type,
            measured_layer_count=1,
            layer_multiplier=len(positions),
            target_layers=(positions[0],),
        )
        for layer_type, positions in positions_by_type.items()
    ]
    num_hidden_layers = max(rep.layer_index for rep in representatives) + 1
    return representatives, num_hidden_layers, _layer_types_override(config, num_hidden_layers)

def _decoder_config_view(config: dict[str, Any]) -> dict[str, Any]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and "num_attention_heads" in text_config:
        return text_config
    return config

def _is_moe_config(config: dict[str, Any]) -> bool:
    config = _decoder_config_view(config)
    return any((config.get(k, 0) or 0) > 0 for k in EXPERT_COUNT_KEYS)

def _is_all_moe_config(config: dict[str, Any]) -> bool:
    config = _decoder_config_view(config)
    model_type = str(config.get("model_type") or "").lower()
    architectures = [str(x) for x in config.get("architectures") or []]
    if model_type == "gpt_oss" or "GptOssForCausalLM" in architectures:
        return True
    # Hybrid MoE configs usually carry replacement/sparse-step controls.  If a
    # config exposes experts but none of those controls, assume every decoder
    # block owns a routed MLP.
    return _is_moe_config(config) and not any(
        key in config
        for key in ("first_k_dense_replace", "decoder_sparse_step", "mlp_only_layers")
    )

def _work_unit_id(
    row_base: dict[str, Any],
    target_layers: list[int],
    num_hidden_layers: int,
    representative: RepresentativeLayer,
    moe_noop: bool = False,
    physical_tp: bool = False,
) -> str:
    payload = {
        **row_base,
        "target_layers": target_layers,
        "num_hidden_layers": num_hidden_layers,
        "representative": asdict(representative),
        "moe_noop": moe_noop,
        "physical_tp": physical_tp,
    }
    return "wu_" + _stable_hash(payload)


def _patch_for_layerwise_depth(
    model_id: str,
    *,
    root_config: dict[str, Any],
    num_hidden_layers: int,
    extra_overrides: dict[str, Any] | None,
    cache_dir: str | None,
) -> str:
    """Patch only layerwise depth/type metadata, leaving TP dimensions intact."""

    config = _decoder_config_view(root_config)
    override_prefix = "text_config." if config is not root_config else ""
    overrides: dict[str, Any] = {f"{override_prefix}num_hidden_layers": num_hidden_layers}
    if extra_overrides:
        for key, value in extra_overrides.items():
            if override_prefix and "." not in key:
                overrides[f"{override_prefix}{key}"] = value
            else:
                overrides[key] = value
    return patch_model_path(
        model_id,
        overrides=overrides,
        strip_auto_map=True,
        model_type_rewrites={"glm_moe_dsa": "deepseek_v3"},
        cache_dir=cache_dir,
    )

def _build_datapoints(
    *,
    phases: str,
    ctx_new_tokens: list[int],
    ctx_past_kv: list[int],
    gen_batch_sizes: list[int],
    gen_past_kv: list[int],
) -> list[DataPoint]:
    datapoints: list[DataPoint] = []
    if phases in ("ctx", "both"):
        for past_kv in ctx_past_kv:
            for new_tokens in ctx_new_tokens:
                datapoints.append(DataPoint("ctx", 1, new_tokens, past_kv))
    if phases in ("gen", "both"):
        for batch_size in gen_batch_sizes:
            for past_kv in gen_past_kv:
                datapoints.append(DataPoint("gen", batch_size, 1, past_kv))
    return datapoints

def _model_max_position_embeddings(config: dict[str, Any]) -> int | None:
    config = _decoder_config_view(config)
    for key in ("max_position_embeddings", "model_max_length", "seq_length", "n_positions"):
        raw = config.get(key)
        if isinstance(raw, bool) or raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        # Some tokenizer/model configs use a huge sentinel for "unbounded".
        if 0 < value < 1_000_000_000:
            return value
    return None

def _filter_datapoints_for_model_max_len(
    datapoints: list[DataPoint],
    max_model_len: int | None,
) -> tuple[list[DataPoint], int]:
    if max_model_len is None:
        return datapoints, 0

    filtered: list[DataPoint] = []
    skipped = 0
    for dp in datapoints:
        if dp.phase == "ctx":
            # The ctx driver uses max_tokens=1 to force vLLM to execute the
            # measured context suffix, so max_model_len must fit the prompt plus that
            # generated token.
            required_len = dp.past_kv + dp.new_tokens + 1
        else:
            required_len = dp.past_kv + 2
        if required_len > max_model_len:
            skipped += 1
            continue
        filtered.append(dp)
    return filtered, skipped

def build_work_units(args: argparse.Namespace) -> list[WorkUnit]:
    """Build legacy work units for one model and one requested TP/EP sweep."""

    ctx_new_tokens = _parse_ints(args.ctx_new_tokens)
    ctx_past_kv = _parse_ints(args.ctx_past_kv)
    gen_batch_sizes = _parse_ints(args.gen_batch_sizes)
    gen_past_kv = _parse_ints(args.gen_past_kv)
    tp_sizes = _parse_ints(args.tp_sizes)
    max_num_seqs = getattr(args, "max_num_seqs", None)
    max_num_batched_tokens = getattr(args, "max_num_batched_tokens", None)
    max_model_len = getattr(args, "max_model_len", None)

    orig_config = _load_original_config(args.model)
    orig_decoder_config = _decoder_config_view(orig_config)
    orig_layer_count = int(orig_decoder_config.get("num_hidden_layers") or 0)
    target_layer_count = args.target_layer_count
    if target_layer_count == 0:
        if orig_layer_count < 1:
            raise ValueError("target_layer_count=0 requires config num_hidden_layers")
        target_layer_count = orig_layer_count
    is_moe = _is_moe_config(orig_config)
    explicit_target_layers = (
        _parse_ints(args.target_layers) if getattr(args, "target_layers", None) else None
    )
    moe_noop = bool(getattr(args, "moe_noop", False) and is_moe)
    requested_latency_source = str(getattr(args, "latency_source", "span"))
    context_schedule_envelope = (
        args.phases == "ctx"
        and requested_latency_source in {"schedule_to_update", "worker_wall"}
    )
    layer_schedule, num_hidden_layers, extra_overrides = _detect_layer_schedule(
        orig_config, target_layer_count,
        explicit_target_layers, args.target_layer_config_depth,
        expand_layer_types=not context_schedule_envelope,
    )

    work_dir = Path(args.work_dir).resolve()
    config_cache_dir = None if args.no_config_cache else (args.config_cache_dir or str(work_dir / "config_cache"))

    system = args.system or _get_system_name()
    version = args.framework_version or _get_vllm_version()
    datapoints = _build_datapoints(
        phases=args.phases,
        ctx_new_tokens=ctx_new_tokens,
        ctx_past_kv=ctx_past_kv,
        gen_batch_sizes=gen_batch_sizes,
        gen_past_kv=gen_past_kv,
    )
    gen_driver = str(
        getattr(args, "gen_driver", "")
        or os.environ.get("LAYERWISE_GEN_DRIVER")
        or "prefix_cache"
    )
    if not getattr(args, "no_filter_model_max_len", False):
        model_max_len = _model_max_position_embeddings(orig_config)
        datapoints, skipped = _filter_datapoints_for_model_max_len(datapoints, model_max_len)
        if skipped:
            print(
                f"[skip] {skipped} datapoints require more than "
                f"model_max_len={model_max_len} tokens"
            )
        if not datapoints:
            raise ValueError("all datapoints exceed the model's configured max length")
    work_units: list[WorkUnit] = []
    for tp in tp_sizes:
        if is_moe and tp % args.moe_tp != 0:
            print(f"[skip] tp={tp} not divisible by moe_tp={args.moe_tp}")
            continue
        physical_tp = bool(getattr(args, "physical_tp", False))
        attn_tp = tp
        moe_tp = args.moe_tp if is_moe else 1
        ep = (tp // moe_tp) if is_moe else 1
        if physical_tp and ep != 1:
            print(f"[skip] physical TP diagnostic currently supports ep=1 only, got tp={tp}, ep={ep}")
            continue
        physical_tp_real_weights = bool(getattr(args, "physical_tp_real_weights", False))
        if physical_tp_real_weights and not physical_tp:
            raise ValueError("physical_tp_real_weights requires physical_tp")
        if physical_tp_real_weights and target_layer_count != orig_layer_count:
            raise ValueError("physical_tp_real_weights requires full-depth target_layer_count")
        num_slots = args.num_slots if is_moe else None
        use_real_moe_router = (
            is_moe
            and attn_tp == 1
            and moe_tp == 1
            and ep == 1
            and bool(getattr(args, "moe_real_router", False))
            and not moe_noop
        )
        router_weight_model = None
        if use_real_moe_router:
            model_dir = _resolve_real_weight_model_dir(args.model)
            work_unit_extra_vllm_args = ("--load-format", "auto")
        else:
            if physical_tp:
                if physical_tp_real_weights:
                    model_dir = _resolve_real_weight_model_dir(args.model)
                else:
                    model_dir = _patch_for_layerwise_depth(
                        args.model,
                        root_config=orig_config,
                        num_hidden_layers=num_hidden_layers,
                        extra_overrides=extra_overrides,
                        cache_dir=config_cache_dir,
                    )
                work_unit_extra_vllm_args = (
                    "--tensor-parallel-size",
                    str(tp),
                    "--worker-extension-cls",
                    "vllm_worker_extension.LayerwiseWorkerExtension",
                )
                if physical_tp_real_weights:
                    work_unit_extra_vllm_args = (
                        *work_unit_extra_vllm_args,
                        "--load-format",
                        "auto",
                    )
            else:
                model_dir = patch_for_parallelism(
                    args.model,
                    attn_tp=attn_tp,
                    moe_tp=moe_tp,
                    ep=ep,
                    num_slots=num_slots,
                    num_hidden_layers=num_hidden_layers,
                    extra_overrides=extra_overrides,
                    model_type_rewrites={"glm_moe_dsa": "deepseek_v3"},
                    cache_dir=config_cache_dir,
                    original_config=orig_config,
                )
                work_unit_extra_vllm_args = ()
            if (
                is_moe
                and bool(getattr(args, "moe_real_router", False))
                and not moe_noop
                and not physical_tp_real_weights
            ):
                router_weight_model = args.model
        row_base = {
            "framework": "vLLM",
            "framework_version": version,
            "system": system,
            "model": args.model,
            "attn_tp": attn_tp,
            "moe_tp": moe_tp,
            "ep": ep,
            "num_slots": num_slots or "",
            "gemm_quant": args.gemm_quant,
            "moe_quant": args.moe_quant,
            "attn_quant": args.attn_quant,
            "kv_quant": args.kv_quant,
        }
        for representative in layer_schedule:
            target_layers = representative.kept_layers()
            includes_moe = (
                not moe_noop
                and "moe" in representative.layer_type.lower()
            )
            final_extra_vllm_args = work_unit_extra_vllm_args
            work_units.append(WorkUnit(
                work_unit_id=_work_unit_id(
                    row_base,
                    target_layers,
                    num_hidden_layers,
                    representative,
                    moe_noop,
                    physical_tp,
                ),
                model_dir=model_dir,
                row_base=row_base,
                representative=representative,
                target_layers=target_layers,
                datapoints=datapoints,
                model_layer_count=num_hidden_layers,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                gen_driver=gen_driver,
                extra_vllm_args=final_extra_vllm_args,
                moe_noop=moe_noop,
                includes_moe=includes_moe,
                router_weight_model=router_weight_model,
                physical_gpus=tp if physical_tp else 1,
            ))
    return work_units
