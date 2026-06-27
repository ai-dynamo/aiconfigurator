# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-aligned identities for persisted collector rows.

The performance filename is part of the identity: two files can intentionally
use identical columns while feeding different consumers.  Each registered
table then supplies the exact normalized fields used by its SDK loader.  Fields
that only describe provenance or measurement quality (for example ``model``
aliases, latency, power, device, and framework version) are deliberately not
part of a physical key unless the consumer itself indexes them.

Unknown files are legacy/unmigrated tables and return ``None``.  This module
never guesses a key from whichever columns happen to be present.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

PHYSICAL_KEY_SCHEMA_VERSION = 1


class PhysicalKeyError(ValueError):
    """A registered table row cannot be normalized into its consumer key."""


@dataclass(frozen=True, slots=True)
class PhysicalRowKey:
    """Versioned, table-scoped identity of one consumer-visible perf row."""

    schema_version: int
    table: str
    fields: tuple[str, ...]
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if len(self.fields) != len(self.values):
            raise ValueError("physical key fields and values must have equal lengths")
        # Fail at construction time rather than much later in a planner set.
        hash(self.values)


RowNormalizer = Callable[[Mapping[str, Any]], tuple[object, ...]]


@dataclass(frozen=True, slots=True)
class PhysicalKeySchema:
    """Explicit consumer-key contract for one canonical performance table."""

    fields: tuple[str, ...]
    normalize: RowNormalizer
    version: int = PHYSICAL_KEY_SCHEMA_VERSION


_MISSING = object()


def _value(row: Mapping[str, Any], name: str, default: object = _MISSING) -> Any:
    value = row.get(name, _MISSING)
    if value is _MISSING or value is None or (isinstance(value, str) and not value.strip()):
        if default is _MISSING:
            raise PhysicalKeyError(f"missing required physical-key field {name!r}")
        return default
    return value


def _text(row: Mapping[str, Any], name: str, default: object = _MISSING) -> str:
    return str(_value(row, name, default)).strip()


def _integer(row: Mapping[str, Any], name: str, default: object = _MISSING) -> int:
    value = _value(row, name, default)
    if isinstance(value, bool):
        raise PhysicalKeyError(f"boolean is not a valid integer for {name!r}")
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicalKeyError(f"invalid integer for {name!r}: {value!r}") from exc
    if number != result:
        raise PhysicalKeyError(f"non-integral value for {name!r}: {value!r}")
    return result


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "datatype.bf16": "bfloat16",
    "float8_e4m3fn": "fp8",
    "float8_e4m3fnuz": "fp8",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e4m3fnuz": "fp8",
    "datatype.fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e4m3fn": "fp8",
}


def _dtype(row: Mapping[str, Any], name: str) -> str:
    value = _text(row, name).lower()
    return _DTYPE_ALIASES.get(value, value)


def _canonical_perf_filename(perf_file: str | os.PathLike[str]) -> str:
    name = Path(perf_file).name
    if name.endswith(".txt"):
        return f"{name[:-4]}.parquet"
    return name


def _local_mla_heads(row: Mapping[str, Any]) -> int:
    if row.get("num_heads") not in (None, ""):
        return _integer(row, "num_heads")
    tp_size = _integer(row, "tp_size")
    if tp_size <= 0 or 128 % tp_size:
        raise PhysicalKeyError(f"legacy MLA tp_size must divide 128, got {tp_size}")
    return 128 // tp_size


def _gemm(row: Mapping[str, Any]) -> tuple[object, ...]:
    return _dtype(row, "gemm_dtype"), _integer(row, "m"), _integer(row, "n"), _integer(row, "k")


def _compute_scale(row: Mapping[str, Any]) -> tuple[object, ...]:
    return _dtype(row, "quant_dtype"), _integer(row, "m"), _integer(row, "k")


def _normalized_kv_heads(row: Mapping[str, Any], num_heads: int) -> int:
    kv_heads = _integer(row, "num_key_value_heads")
    return 0 if kv_heads == num_heads else kv_heads


def _context_attention(row: Mapping[str, Any]) -> tuple[object, ...]:
    num_heads = _integer(row, "num_heads")
    return (
        _dtype(row, "attn_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _normalized_kv_heads(row, num_heads),
        _integer(row, "head_dim"),
        _integer(row, "window_size", 0),
        num_heads,
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _generation_attention(row: Mapping[str, Any]) -> tuple[object, ...]:
    num_heads = _integer(row, "num_heads")
    total_sequence_length = _integer(row, "isl") + _integer(row, "step")
    return (
        _dtype(row, "kv_cache_dtype"),
        _normalized_kv_heads(row, num_heads),
        _integer(row, "head_dim"),
        _integer(row, "window_size", 0),
        num_heads,
        _integer(row, "batch_size"),
        total_sequence_length,
    )


def _encoder_attention(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "attn_dtype"),
        _integer(row, "head_dim"),
        _integer(row, "num_heads"),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _context_mla(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _local_mla_heads(row),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _generation_mla(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "kv_cache_dtype"),
        _local_mla_heads(row),
        _integer(row, "batch_size"),
        _integer(row, "isl") + _integer(row, "step"),
    )


_BMM_OP_ALIASES = {
    "mla_bmm_gen_pre": "mla_gen_pre",
    "mla_bmm_gen_post": "mla_gen_post",
}


def _mla_bmm(row: Mapping[str, Any]) -> tuple[object, ...]:
    op_name = _text(row, "op_name").lower()
    return (
        _dtype(row, "bmm_dtype"),
        _BMM_OP_ALIASES.get(op_name, op_name),
        _integer(row, "num_heads"),
        _integer(row, "num_tokens"),
    )


def _moe_quant_mode(row: Mapping[str, Any]) -> str:
    quant_mode = _dtype(row, "moe_dtype")
    source = _text(row, "kernel_source", "")
    if quant_mode == "w4a8_mxfp4_mxfp8" and source == "sglang_mxfp4_flashinfer_trtllm_moe":
        return "w4a8_mxfp4_mxfp8_trtllm"
    if quant_mode == "w4a16_mxfp4" and source == "sglang_flashinfer_cutlass_moe":
        return "w4a16_mxfp4_cutlass"
    return quant_mode


def _moe_dimensions(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _moe_quant_mode(row),
        _text(row, "distribution"),
        _integer(row, "topk"),
        _integer(row, "num_experts"),
        _integer(row, "hidden_size"),
        _integer(row, "inter_size"),
        _integer(row, "moe_tp_size"),
        _integer(row, "moe_ep_size"),
        _integer(row, "num_tokens"),
    )


def _moe(row: Mapping[str, Any]) -> tuple[object, ...]:
    latency_bucket = "low_latency" if _text(row, "kernel_source", "") == "moe_torch_flow_min_latency" else "default"
    return latency_bucket, *_moe_dimensions(row)


def _state_space_sequence_key(row: Mapping[str, Any]) -> int | None:
    return _integer(row, "seq_len") if _text(row, "phase").lower() == "context" else None


def _mamba2(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _text(row, "kernel_source"),
        _text(row, "phase").lower(),
        _integer(row, "d_model"),
        _integer(row, "d_state"),
        _integer(row, "d_conv"),
        _integer(row, "nheads"),
        _integer(row, "head_dim"),
        _integer(row, "n_groups"),
        _integer(row, "chunk_size"),
        _integer(row, "batch_size"),
        _state_space_sequence_key(row),
    )


_GDN_KERNEL_SOURCE_ALIASES = {
    "fused_sigmoid_gating_delta_rule_update": "fused_recurrent_gated_delta_rule",
}


def _canonical_gdn_source(row: Mapping[str, Any]) -> str:
    source = _text(row, "kernel_source")
    return _GDN_KERNEL_SOURCE_ALIASES.get(source, source)


def _gdn(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _canonical_gdn_source(row),
        _text(row, "phase").lower(),
        _integer(row, "d_model"),
        _integer(row, "num_k_heads"),
        _integer(row, "head_k_dim"),
        _integer(row, "num_v_heads"),
        _integer(row, "head_v_dim"),
        _integer(row, "d_conv"),
        _integer(row, "batch_size"),
        _state_space_sequence_key(row),
    )


def _context_mla_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        _integer(row, "num_heads"),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _generation_mla_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        _integer(row, "num_heads"),
        _integer(row, "batch_size"),
        _integer(row, "isl") + _integer(row, "step"),
    )


def _dsa_backend(row: Mapping[str, Any]) -> str:
    return "trtllm" if "trtllm" in _text(row, "kernel_source", "").lower() else "flashmla_kv"


def _context_dsa_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        _text(row, "architecture", "DeepseekV32ForCausalLM"),
        _dsa_backend(row),
        _integer(row, "num_heads"),
        _integer(row, "step", 0),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _generation_dsa_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        _text(row, "architecture", "DeepseekV32ForCausalLM"),
        _dsa_backend(row),
        _integer(row, "num_heads"),
        _integer(row, "batch_size"),
        _integer(row, "isl") + _integer(row, "step"),
    )


def _wideep_context_mla(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _text(row, "kernel_source", "flashinfer"),
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _local_mla_heads(row),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _wideep_generation_mla(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _text(row, "kernel_source", "flashinfer"),
        _dtype(row, "kv_cache_dtype"),
        _local_mla_heads(row),
        _integer(row, "batch_size"),
        _integer(row, "isl") + _integer(row, "step"),
    )


def _dsv4_profile_id(model: object | None) -> str:
    model_name = str(model or "").lower()
    if "deepseek-v4-flash" in model_name:
        return "flash"
    if "deepseek-v4-pro" in model_name:
        return "pro"
    # Match the SDK loader's structural-fallback spelling when a row carries
    # an unrecognized model id and no separate structural hints.
    return "heads=0:hidden=0:topk=0"


def _dsv4_module_head_identity(row: Mapping[str, Any]) -> tuple[str | None, int | None, int]:
    model = row.get("model")
    logged_heads = _integer(row, "num_heads")
    if model is None or (isinstance(model, str) and not model.strip()):
        # The SDK loader preserves old rows on their historical integer head axis.
        return None, None, logged_heads

    profile = _dsv4_profile_id(model)
    tp_size = _integer(row, "tp_size", 1)
    native_heads = 64 if profile == "flash" else 128 if profile == "pro" else None
    if tp_size <= 0 or (native_heads is not None and native_heads % tp_size):
        raise PhysicalKeyError(f"invalid DSV4 tp_size={tp_size} for profile={profile!r}")
    local_heads = logged_heads if native_heads is None else native_heads // tp_size
    return profile, tp_size, local_heads


def _context_dsv4_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    profile, tp_size, local_heads = _dsv4_module_head_identity(row)
    return (
        _dtype(row, "mla_dtype"),
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        profile,
        tp_size,
        local_heads,
        _integer(row, "compress_ratio"),
        _integer(row, "step", 0),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _generation_dsv4_module(row: Mapping[str, Any]) -> tuple[object, ...]:
    profile, tp_size, local_heads = _dsv4_module_head_identity(row)
    return (
        _dtype(row, "kv_cache_dtype"),
        _dtype(row, "gemm_type"),
        profile,
        tp_size,
        local_heads,
        _integer(row, "compress_ratio"),
        _integer(row, "batch_size"),
        _integer(row, "isl") + _integer(row, "step"),
    )


def _dsv4_sparse(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _integer(row, "num_heads"),
        _integer(row, "tp_size"),
        _integer(row, "step"),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
    )


def _dsv4_topk(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _dsv4_profile_id(_value(row, "model")),
        _integer(row, "step"),
        _integer(row, "isl"),
        _integer(row, "batch_size"),
        _text(row, "score_mode").lower(),
    )


def _mhc(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _text(row, "op_name").lower(),
        _integer(row, "hc_mult"),
        _integer(row, "hidden_size"),
        _integer(row, "num_tokens"),
    )


def _wideep_moe(row: Mapping[str, Any]) -> tuple[object, ...]:
    # Context and generation live in separate, table-scoped files.
    return _moe_dimensions(row)


def _wideep_moe_compute(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        _text(row, "kernel_source", "moe_torch_flow"),
        _moe_quant_mode(row),
        _text(row, "distribution"),
        _integer(row, "topk"),
        _integer(row, "num_experts"),
        _integer(row, "hidden_size"),
        _integer(row, "inter_size"),
        _integer(row, "num_slots"),
        _integer(row, "moe_tp_size"),
        _integer(row, "moe_ep_size"),
        _integer(row, "num_tokens"),
    )


def _schema(fields: tuple[str, ...], normalizer: RowNormalizer) -> PhysicalKeySchema:
    return PhysicalKeySchema(fields=fields, normalize=normalizer)


_MOE_FIELDS = (
    "effective_quant_mode",
    "distribution",
    "topk",
    "num_experts",
    "hidden_size",
    "inter_size",
    "moe_tp_size",
    "moe_ep_size",
    "num_tokens",
)


PHYSICAL_KEY_REGISTRY: Mapping[str, PhysicalKeySchema] = MappingProxyType(
    {
        "gemm_perf.parquet": _schema(("gemm_dtype", "m", "n", "k"), _gemm),
        "computescale_perf.parquet": _schema(("quant_dtype", "m", "k"), _compute_scale),
        "scale_matrix_perf.parquet": _schema(("quant_dtype", "m", "k"), _compute_scale),
        "context_attention_perf.parquet": _schema(
            (
                "attn_dtype",
                "kv_cache_dtype",
                "normalized_kv_heads",
                "head_dim",
                "window_size",
                "num_heads",
                "isl",
                "batch_size",
            ),
            _context_attention,
        ),
        "generation_attention_perf.parquet": _schema(
            (
                "kv_cache_dtype",
                "normalized_kv_heads",
                "head_dim",
                "window_size",
                "num_heads",
                "batch_size",
                "total_sequence_length",
            ),
            _generation_attention,
        ),
        "encoder_attention_perf.parquet": _schema(
            ("attn_dtype", "head_dim", "num_heads", "isl", "batch_size"),
            _encoder_attention,
        ),
        "context_mla_perf.parquet": _schema(
            ("mla_dtype", "kv_cache_dtype", "local_heads", "isl", "batch_size"),
            _context_mla,
        ),
        "generation_mla_perf.parquet": _schema(
            ("kv_cache_dtype", "local_heads", "batch_size", "total_sequence_length"),
            _generation_mla,
        ),
        "mla_bmm_perf.parquet": _schema(
            ("bmm_dtype", "op_name", "num_heads", "num_tokens"),
            _mla_bmm,
        ),
        "moe_perf.parquet": _schema(("latency_bucket", *_MOE_FIELDS), _moe),
        "mamba2_perf.parquet": _schema(
            (
                "kernel_source",
                "phase",
                "d_model",
                "d_state",
                "d_conv",
                "nheads",
                "head_dim",
                "n_groups",
                "chunk_size",
                "batch_size",
                "seq_len_or_none",
            ),
            _mamba2,
        ),
        "gdn_perf.parquet": _schema(
            (
                "kernel_source",
                "phase",
                "d_model",
                "num_k_heads",
                "head_k_dim",
                "num_v_heads",
                "head_v_dim",
                "d_conv",
                "batch_size",
                "seq_len_or_none",
            ),
            _gdn,
        ),
        "mla_context_module_perf.parquet": _schema(
            ("mla_dtype", "kv_cache_dtype", "gemm_type", "num_heads", "isl", "batch_size"),
            _context_mla_module,
        ),
        "mla_generation_module_perf.parquet": _schema(
            ("mla_dtype", "kv_cache_dtype", "gemm_type", "num_heads", "batch_size", "total_sequence_length"),
            _generation_mla_module,
        ),
        "dsa_context_module_perf.parquet": _schema(
            (
                "mla_dtype",
                "kv_cache_dtype",
                "gemm_type",
                "architecture",
                "dsa_backend",
                "num_heads",
                "prefix",
                "isl",
                "batch_size",
            ),
            _context_dsa_module,
        ),
        "dsa_generation_module_perf.parquet": _schema(
            (
                "kv_cache_dtype",
                "gemm_type",
                "architecture",
                "dsa_backend",
                "num_heads",
                "batch_size",
                "total_sequence_length",
            ),
            _generation_dsa_module,
        ),
        "wideep_context_mla_perf.parquet": _schema(
            ("kernel_source", "mla_dtype", "kv_cache_dtype", "local_heads", "isl", "batch_size"),
            _wideep_context_mla,
        ),
        "wideep_generation_mla_perf.parquet": _schema(
            ("kernel_source", "kv_cache_dtype", "local_heads", "batch_size", "total_sequence_length"),
            _wideep_generation_mla,
        ),
        "dsv4_csa_context_module_perf.parquet": _schema(
            (
                "mla_dtype",
                "kv_cache_dtype",
                "gemm_type",
                "profile",
                "tp_size",
                "local_heads",
                "compress_ratio",
                "prefix",
                "isl",
                "batch_size",
            ),
            _context_dsv4_module,
        ),
        "dsv4_hca_context_module_perf.parquet": _schema(
            (
                "mla_dtype",
                "kv_cache_dtype",
                "gemm_type",
                "profile",
                "tp_size",
                "local_heads",
                "compress_ratio",
                "prefix",
                "isl",
                "batch_size",
            ),
            _context_dsv4_module,
        ),
        "dsv4_csa_generation_module_perf.parquet": _schema(
            (
                "kv_cache_dtype",
                "gemm_type",
                "profile",
                "tp_size",
                "local_heads",
                "compress_ratio",
                "batch_size",
                "total_sequence_length",
            ),
            _generation_dsv4_module,
        ),
        "dsv4_hca_generation_module_perf.parquet": _schema(
            (
                "kv_cache_dtype",
                "gemm_type",
                "profile",
                "tp_size",
                "local_heads",
                "compress_ratio",
                "batch_size",
                "total_sequence_length",
            ),
            _generation_dsv4_module,
        ),
        "dsv4_paged_mqa_logits_module_perf.parquet": _schema(
            ("num_heads", "tp_size", "past_kv", "isl", "batch_size"),
            _dsv4_sparse,
        ),
        "dsv4_hca_attn_module_perf.parquet": _schema(
            ("num_heads", "tp_size", "past_kv", "isl", "batch_size"),
            _dsv4_sparse,
        ),
        "dsv4_csa_attn_module_perf.parquet": _schema(
            ("num_heads", "tp_size", "past_kv", "isl", "batch_size"),
            _dsv4_sparse,
        ),
        "dsv4_csa_topk_calib_perf.parquet": _schema(
            ("profile", "past_kv", "isl", "batch_size", "score_mode"),
            _dsv4_topk,
        ),
        "mhc_module_perf.parquet": _schema(
            ("op_name", "hc_mult", "hidden_size", "num_tokens"),
            _mhc,
        ),
        "wideep_context_moe_perf.parquet": _schema(_MOE_FIELDS, _wideep_moe),
        "wideep_generation_moe_perf.parquet": _schema(_MOE_FIELDS, _wideep_moe),
        "wideep_moe_perf.parquet": _schema(
            (
                "kernel_source",
                "effective_quant_mode",
                "distribution",
                "topk",
                "num_experts",
                "hidden_size",
                "inter_size",
                "num_slots",
                "moe_tp_size",
                "moe_ep_size",
                "num_tokens",
            ),
            _wideep_moe_compute,
        ),
    }
)


def physical_row_key(
    perf_file: str | os.PathLike[str],
    row: Mapping[str, Any],
) -> PhysicalRowKey | None:
    """Return the registered consumer key, or ``None`` for a legacy table.

    Known tables fail explicitly on missing or malformed key fields.  Unknown
    tables are not inferred from row contents, even if they look similar to a
    registered schema.
    """

    table = _canonical_perf_filename(perf_file)
    schema = PHYSICAL_KEY_REGISTRY.get(table)
    if schema is None:
        return None
    try:
        values = schema.normalize(row)
    except PhysicalKeyError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicalKeyError(f"cannot normalize physical key for {table}: {exc}") from exc
    return PhysicalRowKey(
        schema_version=schema.version,
        table=table,
        fields=schema.fields,
        values=values,
    )


__all__ = [
    "PHYSICAL_KEY_REGISTRY",
    "PHYSICAL_KEY_SCHEMA_VERSION",
    "PhysicalKeyError",
    "PhysicalKeySchema",
    "PhysicalRowKey",
    "physical_row_key",
]
