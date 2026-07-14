# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project a model's AIC attention case population into FPM workloads."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from collector import case_generator

from .types import FPMPoint


@dataclass(frozen=True, slots=True)
class AttentionPopulation:
    source_name: str
    source_ops: tuple[str, str]
    prefill_points: tuple[FPMPoint, ...]
    decode_points: tuple[FPMPoint, ...]
    p_values: tuple[int, ...]
    max_model_len: int | None
    model_config_path: str | None
    native_weight_quantization: str
    is_moe: bool

    @property
    def points(self) -> tuple[FPMPoint, ...]:
        return self.prefill_points + self.decode_points

    def to_summary(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_ops": list(self.source_ops),
            "prefill_points": len(self.prefill_points),
            "decode_points": len(self.decode_points),
            "prefix_lengths": list(self.p_values),
            "max_model_len": self.max_model_len,
            "model_config_path": self.model_config_path,
            "native_weight_quantization": self.native_weight_quantization,
            "is_moe": self.is_moe,
        }


@contextmanager
def _model_path_filter(model_path: str) -> Iterator[None]:
    previous = os.environ.get("COLLECTOR_MODEL_PATH")
    os.environ["COLLECTOR_MODEL_PATH"] = model_path
    case_generator._load_model_cases_data.cache_clear()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = previous
        case_generator._load_model_cases_data.cache_clear()


def _cached_model_config_path(model_path: str) -> Path | None:
    direct = Path(model_path).expanduser()
    if direct.is_dir() and (direct / "config.json").exists():
        return direct / "config.json"
    if direct.is_file():
        return direct
    root = Path(__file__).resolve().parents[2]
    cached = root / "src" / "aiconfigurator" / "model_configs" / f"{model_path.replace('/', '--')}_config.json"
    return cached if cached.exists() else None


def _load_model_config(model_path: str) -> tuple[dict[str, object], Path | None]:
    path = _cached_model_config_path(model_path)
    if path is None:
        return {}, None
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"model config must be a mapping: {path}")
    text_config = payload.get("text_config")
    if isinstance(text_config, dict):
        payload = {**payload, **text_config}
    return payload, path


def _native_weight_quantization(config: dict[str, object], model_path: str) -> str:
    quant = config.get("quantization_config")
    if isinstance(quant, dict):
        for key in ("quant_algo", "format", "method", "quant_method"):
            if quant.get(key):
                return str(quant[key]).lower()
    lowered = model_path.lower()
    if "nvfp4" in lowered:
        return "nvfp4"
    if "fp8" in lowered:
        return "fp8"
    return "bfloat16"


def _max_model_len(config: dict[str, object]) -> int | None:
    for key in ("max_position_embeddings", "model_max_length", "max_seq_len"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return None


def _is_moe(config: dict[str, object]) -> bool:
    for key in ("n_routed_experts", "num_local_experts", "num_experts"):
        value = config.get(key)
        if value is not None and int(value) > 1:
            return True
    return False


def _valid_pairs(
    batches: list[int],
    lengths: list[int],
    *,
    max_tokens: int,
    large_sequence_min: int = 0,
    large_sequence_max_batch: int = 0,
) -> list[tuple[int, int]]:
    return [
        (batch, length)
        for batch in batches
        for length in lengths
        if batch * length <= max_tokens
        and not (
            large_sequence_min
            and large_sequence_max_batch
            and length >= large_sequence_min
            and batch > large_sequence_max_batch
        )
    ]


def _mla_pairs(backend: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sweep = case_generator.get_mla_module_sweep_spec(backend)
    context = _valid_pairs(
        sweep.context_batch_sizes,
        sweep.context_sequence_lengths,
        max_tokens=sweep.context_max_tokens,
        large_sequence_min=sweep.context_large_sequence_min,
        large_sequence_max_batch=sweep.context_large_sequence_max_batch_size,
    )
    generation = _valid_pairs(
        sweep.generation_batch_sizes,
        sweep.generation_sequence_lengths,
        max_tokens=sweep.generation_max_tokens,
        large_sequence_min=sweep.generation_large_sequence_min,
        large_sequence_max_batch=sweep.generation_large_sequence_max_batch_size,
    )
    return context, generation


def _dense_attention_pairs(
    backend: str,
    model_config: dict[str, object],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    query_heads = int(model_config.get("num_attention_heads", 1))
    kv_heads = int(model_config.get("num_key_value_heads", query_heads))

    context_pairs = set()
    for sweep in case_generator.get_attention_context_shape_sweeps(backend):
        batches = [int(value) for value in sweep.get("batch_sizes", [])]
        lengths = [int(value) for value in sweep.get("sequence_lengths", [])]
        if query_heads == kv_heads:
            max_tokens = int(sweep.get("max_tokens_self_attention", 65536))
            max_batch = int(sweep.get("max_batch_size_self_attention", max(batches)))
            batches = [batch for batch in batches if batch <= max_batch]
        else:
            max_tokens = int(sweep.get("max_tokens_grouped_query_attention", 131072))
        context_pairs.update(_valid_pairs(batches, lengths, max_tokens=max_tokens))

    generation_pairs = set()
    for sweep in case_generator.get_attention_generation_shape_sweeps(backend):
        batches = [int(value) for value in sweep.get("batch_sizes", [])]
        lengths = [int(value) for value in sweep.get("sequence_lengths", [])]
        key = "max_mha_tokens_per_step" if query_heads == kv_heads else "max_xqa_tokens_per_step"
        max_tokens = int(sweep.get(key, 8388608))
        generation_pairs.update(_valid_pairs(batches, lengths, max_tokens=max_tokens))
    return sorted(context_pairs), sorted(generation_pairs)


@dataclass(frozen=True, slots=True)
class _SourceProvider:
    name: str
    context_op: str
    generation_op: str
    build_pairs: Callable[[str, dict[str, object]], tuple[list[tuple[int, int]], list[tuple[int, int]]]]


def _mla_provider(backend: str, _config: dict[str, object]):
    return _mla_pairs(backend)


_SOURCE_PROVIDERS = (
    _SourceProvider("dsa_module", "dsa_context_module", "dsa_generation_module", _mla_provider),
    _SourceProvider("mla_module", "mla_context_module", "mla_generation_module", _mla_provider),
    _SourceProvider(
        "dense_attention",
        "attention_context",
        "attention_generation",
        _dense_attention_pairs,
    ),
)


def _resolve_provider(selected_ops: set[str]) -> _SourceProvider:
    matches = [
        provider
        for provider in _SOURCE_PROVIDERS
        if {provider.context_op, provider.generation_op}.issubset(selected_ops)
    ]
    if not matches:
        raise ValueError(
            "fpm_forward requires one supported AIC context/generation attention pair; "
            f"selected ops were {sorted(selected_ops)}"
        )
    if len(matches) > 1:
        raise ValueError(
            "fpm_forward attention source is ambiguous; model cases must select one pair: "
            + ", ".join(provider.name for provider in matches)
        )
    return matches[0]


def build_attention_population(
    *,
    backend: str,
    model_path: str,
    selected_ops: set[str],
    kv_block_size: int,
) -> AttentionPopulation:
    """Build the latency-blind physical workload population for one model."""

    model_config, config_path = _load_model_config(model_path)
    provider = _resolve_provider(selected_ops)
    with _model_path_filter(model_path):
        context_pairs, generation_pairs = provider.build_pairs(backend, model_config)
    if not context_pairs or not generation_pairs:
        raise ValueError(f"AIC attention source {provider.name} produced an empty phase")

    max_len = _max_model_len(model_config)
    prefix_values = sorted(
        {
            length
            for _, length in generation_pairs
            if length >= kv_block_size and length % kv_block_size == 0 and (max_len is None or length < max_len)
        }
    )
    if not prefix_values:
        raise ValueError(f"AIC generation population has no P>0 values aligned to KV block size {kv_block_size}")

    prefill_points = sorted(
        {
            FPMPoint("prefill", batch, suffix, prefix)
            for batch, suffix in context_pairs
            for prefix in [0, *prefix_values]
            if max_len is None or prefix + suffix <= max_len
        }
    )
    decode_points = sorted(
        {
            FPMPoint("decode", batch, 1, length)
            for batch, length in generation_pairs
            if max_len is None or length + 1 <= max_len
        }
    )
    return AttentionPopulation(
        source_name=provider.name,
        source_ops=(provider.context_op, provider.generation_op),
        prefill_points=tuple(prefill_points),
        decode_points=tuple(decode_points),
        p_values=(0, *prefix_values),
        max_model_len=max_len,
        model_config_path=str(config_path) if config_path is not None else None,
        native_weight_quantization=_native_weight_quantization(model_config, model_path),
        is_moe=_is_moe(model_config),
    )
