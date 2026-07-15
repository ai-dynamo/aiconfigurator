# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project a model's AIC attention case population into FPM workloads."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from collector import case_generator

from .model_capability import load_model_config, resolve_attention_source
from .types import FPMPoint


@dataclass(frozen=True, slots=True)
class AttentionPopulation:
    source_name: str
    source_ops: tuple[str, ...]
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
    required_ops: frozenset[str]
    source_ops: tuple[str, ...]
    build_pairs: Callable[[str, dict[str, object]], tuple[list[tuple[int, int]], list[tuple[int, int]]]] | None
    build_points: (
        Callable[
            [str, dict[str, object]],
            tuple[list[FPMPoint], list[FPMPoint]],
        ]
        | None
    ) = None


def _mla_provider(backend: str, _config: dict[str, object]):
    return _mla_pairs(backend)


def _dsv4_vllm_points(
    backend: str,
    _config: dict[str, object],
) -> tuple[list[FPMPoint], list[FPMPoint]]:
    """Project the lightweight vLLM DSV4 CSA/HCA module case domain.

    CSA and HCA use the same coordinates, so their union contains one point
    per physical workload. These bounds mirror the vLLM op-level collector's
    default case filter without importing vLLM during FPM planning.
    """

    if backend != "vllm":
        raise ValueError(f"FPM DSV4 population is not implemented for backend={backend!r}")

    max_seq_len = 65536
    max_context_query_tokens = 262144
    max_generation_kv_tokens = 1024 * 1024
    context_prefix_anchors = (0, 128, 2048, 4096)
    batch_sizes = case_generator._DSV4_MODULE_BATCH_SIZES
    sequence_lengths = case_generator._DSV4_MODULE_SEQ_LENGTHS

    prefill: list[FPMPoint] = []
    decode: list[FPMPoint] = []
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            if sequence_length > max_seq_len:
                continue
            prefixes = dict.fromkeys((*context_prefix_anchors, max_seq_len - sequence_length))
            for prefix_length in prefixes:
                total_sequence_length = prefix_length + sequence_length
                if prefix_length < 0 or total_sequence_length > max_seq_len:
                    continue
                if batch_size * sequence_length > max_context_query_tokens:
                    continue
                if batch_size * total_sequence_length > max_generation_kv_tokens:
                    continue
                prefill.append(FPMPoint("prefill", batch_size, sequence_length, prefix_length))

            if batch_size * sequence_length > max_generation_kv_tokens:
                continue
            if sequence_length >= 524288 and batch_size > 1:
                continue
            if sequence_length >= 262144 and batch_size > 2:
                continue
            if sequence_length >= 131072 and batch_size > 4:
                continue
            if sequence_length >= 65536 and batch_size > 8:
                continue
            if sequence_length >= 32768 and batch_size > 16:
                continue
            if sequence_length >= 8192 and batch_size > 64:
                continue
            decode.append(FPMPoint("decode", batch_size, 1, sequence_length))
    return prefill, decode


_SOURCE_PROVIDERS = (
    _SourceProvider(
        "dsa_module",
        frozenset({"dsa_context_module", "dsa_generation_module"}),
        ("dsa_context_module", "dsa_generation_module"),
        _mla_provider,
    ),
    _SourceProvider(
        "dsv4_module",
        frozenset(
            {
                "dsv4_csa_context_module",
                "dsv4_hca_context_module",
                "dsv4_csa_generation_module",
                "dsv4_hca_generation_module",
            }
        ),
        (
            "dsv4_csa_context_module",
            "dsv4_hca_context_module",
            "dsv4_csa_generation_module",
            "dsv4_hca_generation_module",
        ),
        None,
        _dsv4_vllm_points,
    ),
    _SourceProvider(
        "mla_module",
        frozenset({"mla_context_module", "mla_generation_module"}),
        ("mla_context_module", "mla_generation_module"),
        _mla_provider,
    ),
    _SourceProvider(
        "dense_attention",
        frozenset({"attention_context", "attention_generation"}),
        ("attention_context", "attention_generation"),
        _dense_attention_pairs,
    ),
)


def _resolve_provider(selected_ops: set[str], *, required: bool = True) -> _SourceProvider | None:
    source_name = resolve_attention_source(selected_ops, required=required)
    return _provider_by_name(source_name) if source_name is not None else None


def _provider_by_name(source_name: str) -> _SourceProvider:
    for provider in _SOURCE_PROVIDERS:
        if provider.name == source_name:
            return provider
    raise ValueError(f"unknown FPM attention source template: {source_name!r}")


def build_attention_population(
    *,
    backend: str,
    model_path: str,
    selected_ops: set[str],
    kv_block_size: int,
    attention_source: str | None = None,
) -> AttentionPopulation:
    """Build the latency-blind physical workload population for one model."""

    model_config, config_path = load_model_config(model_path)
    provider = _provider_by_name(attention_source) if attention_source is not None else _resolve_provider(selected_ops)
    assert provider is not None
    max_len = _max_model_len(model_config)
    if provider.build_points is not None:
        with _model_path_filter(model_path):
            raw_prefill, raw_decode = provider.build_points(backend, model_config)
        prefill_points = sorted(
            {point for point in raw_prefill if max_len is None or point.prefix_length + point.suffix_length <= max_len}
        )
        decode_points = sorted({point for point in raw_decode if max_len is None or point.prefix_length + 1 <= max_len})
        prefix_values = sorted({point.prefix_length for point in prefill_points if point.prefix_length > 0})
    else:
        assert provider.build_pairs is not None
        with _model_path_filter(model_path):
            context_pairs, generation_pairs = provider.build_pairs(backend, model_config)
        if not context_pairs or not generation_pairs:
            raise ValueError(f"AIC attention source {provider.name} produced an empty phase")
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
    if not prefill_points or not decode_points:
        raise ValueError(f"AIC attention source {provider.name} produced an empty phase")
    return AttentionPopulation(
        source_name=provider.name,
        source_ops=provider.source_ops,
        prefill_points=tuple(prefill_points),
        decode_points=tuple(decode_points),
        p_values=(0, *prefix_values),
        max_model_len=max_len,
        model_config_path=str(config_path) if config_path is not None else None,
        native_weight_quantization=_native_weight_quantization(model_config, model_path),
        is_moe=_is_moe(model_config),
    )
