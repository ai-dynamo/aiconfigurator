# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-free attention case population.

The runtime collectors should only discover their SM/framework version and
load the requested YAML sweeps.  All shape limits and version-specific skip
rules live here so the same case plan can be produced without importing a GPU
framework.

The returned lists deliberately retain each runtime's historical positional
ABI.  They are passed directly to ``run_attention_torch`` (or the encoder
equivalent), so changing field order here is a consumer contract change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from collector.case_generator import get_attention_encoder_head_configs, get_attention_head_configs
from collector.planner.models import LogicalCandidate, PlanContext
from collector.planner.physical_keys import PhysicalRowKey, physical_row_key
from collector.planner.registry import get_schema, legacy_passthrough, register_schema

ShapeSweep: TypeAlias = Mapping[str, object]
AttentionCaseValue: TypeAlias = int | bool
AttentionCase: TypeAlias = list[AttentionCaseValue]

_ATTENTION_BACKENDS = frozenset({"sglang", "trtllm", "vllm", "vllm_xpu"})
_ENCODER_BACKENDS = frozenset({"sglang", "trtllm", "vllm"})

_CONTEXT_LAYOUTS = {
    "sglang": (
        "batch_size",
        "sequence_length",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "use_fp8_kv_cache",
        "use_fp8_context_fmha",
        "is_context_phase",
        "window_size",
    ),
    "trtllm": (
        "batch_size",
        "sequence_length",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "window_size",
        "use_fp8_kv_cache",
        "use_fp8_context_fmha",
        "is_context_phase",
    ),
    "vllm": (
        "batch_size",
        "sequence_length",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "use_fp8_kv_cache",
        "is_context_phase",
        "window_size",
    ),
    "vllm_xpu": (
        "batch_size",
        "sequence_length",
        "num_heads",
        "num_kv_heads",
        "head_dim",
        "use_fp8_kv_cache",
        "is_context_phase",
        "window_size",
    ),
}

_GENERATION_LAYOUTS = {
    "sglang": _CONTEXT_LAYOUTS["sglang"],
    "trtllm": _CONTEXT_LAYOUTS["trtllm"],
    "vllm": _CONTEXT_LAYOUTS["vllm"],
    "vllm_xpu": _CONTEXT_LAYOUTS["vllm_xpu"],
}

_ENCODER_LAYOUT = ("batch_size", "sequence_length", "num_heads", "head_dim")
_BOOLEAN_FIELDS = frozenset({"use_fp8_kv_cache", "use_fp8_context_fmha", "is_context_phase"})
_EXPECTED_TABLES = {
    "attention_context": "context_attention_perf.parquet",
    "attention_generation": "generation_attention_perf.parquet",
    "encoder_attention": "encoder_attention_perf.parquet",
}
_MISSING = object()

# Universal safety budgets from the base attention sweeps. Sampling policy
# (for example generation's minimum batch options and largest-point drop) is
# intentionally not a schema guard: exact model deltas are additive and need
# not belong to the rectangular base grid.
_CONTEXT_MAX_MHA_TOKENS = 65536
_CONTEXT_MAX_GQA_TOKENS = 131072
_CONTEXT_MAX_MHA_BATCH = 128
_ATTENTION_MAX_INDEX_ELEMENTS = 2**31 - 1
_GENERATION_MAX_MHA_TOKENS = 8192 * 1024
_GENERATION_MAX_XQA_TOKENS = 8192 * 1024 * 2
_ENCODER_MAX_TOKENS = 131072


def _validate_backend(backend: str, supported: frozenset[str]) -> None:
    if backend not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"unsupported attention backend {backend!r}; expected one of: {choices}")


def _int_list(values: object, *, field_name: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be a sequence")
    return [int(value) for value in values]


def _precision_cases(shape_sweep: ShapeSweep) -> Iterable[Mapping[str, object]]:
    values = shape_sweep["precision_cases"]
    if not isinstance(values, Sequence) or isinstance(values, str | bytes):
        raise TypeError("precision_cases must be a sequence")
    for value in values:
        if not isinstance(value, Mapping):
            raise TypeError("precision_cases entries must be mappings")
        yield value


def _trtllm_sm120_fp8_context_fmha_is_unsupported(
    framework_version: str,
    sm_version: int,
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not (framework_version.startswith(("1.3.0rc5", "1.3.0rc10")) and sm_version >= 120):
        return False

    num_tokens = batch_size * input_len
    if framework_version.startswith("1.3.0rc5"):
        return (num_heads == num_key_value_heads == 96 and head_dim == 128 and num_tokens == 65536) or head_dim == 256

    if head_dim != 256:
        return False
    return (
        (num_heads == 96 and num_key_value_heads == 8 and num_tokens >= 81920)
        or (num_heads == 48 and num_key_value_heads == 8 and num_tokens >= 131072)
        or (num_heads == num_key_value_heads == 96 and batch_size >= 2 and input_len >= 16384)
    )


def _trtllm_sm89_rc15_long_context_gqa_is_unsupported(
    framework_version: str,
    sm_version: int,
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not (framework_version.startswith("1.3.0rc15") and sm_version == 89):
        return False
    if num_key_value_heads not in {1, 2, 4, 8}:
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 98304
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 81920
        return head_dim >= 192 and num_tokens >= 98304
    if num_heads == 48:
        if head_dim >= 256:
            return num_tokens >= 98304
        return head_dim >= 192 and num_tokens >= 131072
    if num_heads == 40:
        return head_dim >= 256 and num_tokens >= 131072
    return False


def _trtllm_sm89_rc15_fp8_context_mha_is_unsupported(
    framework_version: str,
    sm_version: int,
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    use_fp8_kv_cache: bool,
    use_fp8_context_fmha: bool,
) -> bool:
    if not (framework_version.startswith("1.3.0rc15") and sm_version == 89):
        return False
    if not (use_fp8_kv_cache and use_fp8_context_fmha and num_heads == num_key_value_heads):
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 65536
        if head_dim >= 256:
            return num_tokens >= 32768
        return head_dim >= 192 and num_tokens >= 40960
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 48:
        return head_dim >= 256 and num_tokens >= 65536
    return False


def _trtllm_gqa_ratio_is_unsupported(num_heads: int, num_kv_heads: int, sm_version: int) -> bool:
    if num_kv_heads == num_heads or sm_version < 100:
        return False
    ratio = num_heads // num_kv_heads
    return ratio >= 32 and ratio % 32 != 0


def _vllm_head_dim_is_unsupported(framework_version: str, sm_version: int, head_dim: int) -> bool:
    return framework_version.startswith("0.22.0") and sm_version == 89 and head_dim >= 512


def _vllm_fp8_is_unsupported(framework_version: str, sm_version: int, use_fp8_kv_cache: bool) -> bool:
    return use_fp8_kv_cache and framework_version.startswith("0.22.0") and sm_version == 89


def _context_precision_cases(
    backend: str,
    shape_sweep: ShapeSweep,
    *,
    sm_version: int,
) -> Iterable[tuple[bool, bool]]:
    if backend in {"sglang", "trtllm"}:
        for precision_case in _precision_cases(shape_sweep):
            yield bool(precision_case["fp8_kv_cache"]), bool(precision_case["fp8_context_fmha"])
        return

    fp8_values = [False, True] if backend == "vllm_xpu" or sm_version > 86 else [False]
    for use_fp8_kv_cache in fp8_values:
        yield use_fp8_kv_cache, False


def _context_reachability_reason(
    *,
    backend: str,
    sm_version: int,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_mha_tokens: int,
    max_gqa_tokens: int,
    max_mha_batch: int,
    max_index_elements: int,
) -> str | None:
    num_tokens = batch_size * sequence_length
    if num_kv_heads == num_heads:
        if num_tokens > max_mha_tokens:
            return "attention.context_mha_token_budget_exceeded"
        if batch_size > max_mha_batch:
            return "attention.context_mha_batch_budget_exceeded"
    elif num_tokens > max_gqa_tokens:
        return "attention.context_gqa_token_budget_exceeded"
    if num_tokens * num_kv_heads * head_dim * 2 >= max_index_elements:
        return "attention.context_kv_index_budget_exceeded"
    if backend == "sglang" and sm_version >= 120 and num_tokens * num_heads * head_dim >= max_index_elements:
        return "attention.sglang_sm120_index_budget_exceeded"
    return None


def _generation_max_batch(
    backend: str,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
    max_tokens: int,
) -> int:
    if backend == "sglang":
        return max_tokens // sequence_length // num_heads * 128 // head_dim
    if backend == "vllm":
        return max_tokens * 128 // head_dim // sequence_length // num_heads
    return max_tokens // sequence_length // num_heads


def _encoder_reachability_reason(
    *,
    batch_size: int,
    sequence_length: int,
    num_heads: int,
    head_dim: int,
) -> str | None:
    if batch_size * sequence_length > _ENCODER_MAX_TOKENS:
        return "attention.encoder_token_budget_exceeded"
    if 4 * batch_size * sequence_length * num_heads * head_dim * 2 >= 2**31:
        return "attention.encoder_index_budget_exceeded"
    return None


def build_attention_context_cases(
    backend: str,
    shape_sweeps: Iterable[ShapeSweep],
    *,
    sm_version: int,
    framework_version: str,
) -> list[AttentionCase]:
    """Build ordered context-attention runtime cases from resolved YAML sweeps."""

    _validate_backend(backend, _ATTENTION_BACKENDS)
    test_cases: list[AttentionCase] = []

    for shape_sweep in shape_sweeps:
        batch_sizes = _int_list(shape_sweep["batch_sizes"], field_name="batch_sizes")
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"], field_name="sequence_lengths")
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep.get("max_batch_size_self_attention", 128))
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        xpu_head_dims = (
            set(_int_list(shape_sweep["head_dims"], field_name="head_dims")) if backend == "vllm_xpu" else None
        )
        xpu_window_sizes = (
            set(_int_list(shape_sweep["window_sizes"], field_name="window_sizes")) if backend == "vllm_xpu" else None
        )

        for head_config in get_attention_head_configs(dict(shape_sweep), phase="context"):
            num_heads = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size

            if (
                backend == "trtllm"
                and num_kv_heads != num_heads
                and (num_kv_heads >= num_heads or num_heads % num_kv_heads != 0)
            ):
                continue
            if backend == "vllm" and _vllm_head_dim_is_unsupported(framework_version, sm_version, head_dim):
                continue
            if backend == "vllm_xpu":
                assert xpu_head_dims is not None and xpu_window_sizes is not None
                if head_dim not in xpu_head_dims or window_size not in xpu_window_sizes:
                    continue
                if num_heads // num_kv_heads > 16:
                    continue
                if window_size > 0 and head_dim == 128:
                    continue

            for sequence_length in sorted(sequence_lengths, reverse=True):
                for batch_size in sorted(batch_sizes, reverse=True):
                    if _context_reachability_reason(
                        backend=backend,
                        sm_version=sm_version,
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        num_heads=num_heads,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                        max_mha_tokens=max_tokens_self_attention,
                        max_gqa_tokens=max_tokens_grouped_query_attention,
                        max_mha_batch=max_batch_size_self_attention,
                        max_index_elements=max_kv_elements,
                    ):
                        continue
                    if backend == "trtllm":
                        if _trtllm_gqa_ratio_is_unsupported(num_heads, num_kv_heads, sm_version):
                            continue
                        if _trtllm_sm89_rc15_long_context_gqa_is_unsupported(
                            framework_version,
                            sm_version,
                            batch_size,
                            sequence_length,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        ):
                            continue

                    skip_trtllm_fp8_fmha = backend == "trtllm" and (
                        _trtllm_sm120_fp8_context_fmha_is_unsupported(
                            framework_version,
                            sm_version,
                            batch_size,
                            sequence_length,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                        )
                    )
                    for use_fp8_kv_cache, use_fp8_context_fmha in _context_precision_cases(
                        backend,
                        shape_sweep,
                        sm_version=sm_version,
                    ):
                        if backend == "sglang" and sm_version < 90 and use_fp8_kv_cache:
                            continue
                        if backend == "trtllm":
                            if sm_version <= 86 and use_fp8_kv_cache:
                                continue
                            if skip_trtllm_fp8_fmha and use_fp8_context_fmha:
                                continue
                            if _trtllm_sm89_rc15_fp8_context_mha_is_unsupported(
                                framework_version,
                                sm_version,
                                batch_size,
                                sequence_length,
                                num_heads,
                                num_kv_heads,
                                head_dim,
                                use_fp8_kv_cache,
                                use_fp8_context_fmha,
                            ):
                                continue
                        if backend == "vllm" and _vllm_fp8_is_unsupported(
                            framework_version, sm_version, use_fp8_kv_cache
                        ):
                            continue

                        if backend == "sglang":
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    use_fp8_kv_cache,
                                    use_fp8_context_fmha,
                                    True,
                                    window_size,
                                ]
                            )
                        elif backend == "trtllm":
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window_size,
                                    use_fp8_kv_cache,
                                    use_fp8_context_fmha,
                                    True,
                                ]
                            )
                        else:
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    use_fp8_kv_cache,
                                    True,
                                    window_size,
                                ]
                            )
    return test_cases


def _generation_target_sequence_lengths(
    backend: str,
    batch_sizes: Sequence[int],
    sequence_lengths: Sequence[int],
    num_heads: int,
    head_dim: int,
    max_tokens: int,
    shape_sweep: ShapeSweep,
) -> dict[int, set[int]]:
    batch_to_sequences: dict[int, set[int]] = {}
    sequence_to_batches: dict[int, set[int]] = {}
    ordered_batch_sizes = sorted(batch_sizes) if backend == "sglang" else batch_sizes
    for sequence_length in sequence_lengths:
        max_batch = _generation_max_batch(backend, sequence_length, num_heads, head_dim, max_tokens)
        for batch_size in ordered_batch_sizes:
            if batch_size > max_batch:
                break
            sequence_to_batches.setdefault(sequence_length, set()).add(batch_size)
    for sequence_length, batch_set in sequence_to_batches.items():
        if len(batch_set) < int(shape_sweep["min_batch_options_per_sequence"]):
            continue
        for batch_size in batch_set:
            batch_to_sequences.setdefault(batch_size, {sequence_length - 1}).add(sequence_length - 1)
    return batch_to_sequences


def _generation_fp8_values(backend: str, shape_sweep: ShapeSweep, sm_version: int) -> Iterable[bool]:
    if backend in {"sglang", "trtllm"}:
        for precision_case in _precision_cases(shape_sweep):
            yield bool(precision_case["fp8_kv_cache"])
        return
    yield False
    if backend == "vllm_xpu" or sm_version > 86:
        yield True


def build_attention_generation_cases(
    backend: str,
    shape_sweeps: Iterable[ShapeSweep],
    *,
    sm_version: int,
    framework_version: str,
) -> list[AttentionCase]:
    """Build ordered generation-attention runtime cases from resolved YAML sweeps."""

    _validate_backend(backend, _ATTENTION_BACKENDS)
    test_cases: list[AttentionCase] = []

    for shape_sweep in shape_sweeps:
        batch_sizes = _int_list(shape_sweep["batch_sizes"], field_name="batch_sizes")
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"], field_name="sequence_lengths")
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])
        xpu_head_dims = (
            set(_int_list(shape_sweep["head_dims"], field_name="head_dims")) if backend == "vllm_xpu" else None
        )
        xpu_window_sizes = (
            set(_int_list(shape_sweep["window_sizes"], field_name="window_sizes")) if backend == "vllm_xpu" else None
        )

        for head_config in get_attention_head_configs(dict(shape_sweep), phase="generation"):
            num_heads = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size

            if (
                backend == "trtllm"
                and num_kv_heads != num_heads
                and (num_kv_heads >= num_heads or num_heads % num_kv_heads != 0)
            ):
                continue
            if backend == "vllm" and _vllm_head_dim_is_unsupported(framework_version, sm_version, head_dim):
                continue
            if backend == "vllm_xpu":
                assert xpu_head_dims is not None and xpu_window_sizes is not None
                if head_dim not in xpu_head_dims or window_size not in xpu_window_sizes:
                    continue
                if num_heads // num_kv_heads > 16:
                    continue
                if window_size > 0 and head_dim == 128:
                    continue
            if backend == "trtllm" and _trtllm_gqa_ratio_is_unsupported(num_heads, num_kv_heads, sm_version):
                continue
            if backend == "vllm" and sm_version >= 100 and num_heads // num_kv_heads > 16:
                continue

            max_tokens_key = "max_mha_tokens_per_step" if num_heads == num_kv_heads else "max_xqa_tokens_per_step"
            batch_to_sequences = _generation_target_sequence_lengths(
                backend,
                batch_sizes,
                sequence_lengths,
                num_heads,
                head_dim,
                int(shape_sweep[max_tokens_key]),
                shape_sweep,
            )
            for batch_size, limited_sequences in batch_to_sequences.items():
                target_sequences = sorted(limited_sequences)
                if batch_size >= min_drop_batch:
                    target_sequences = target_sequences[:-1]
                for sequence_length in target_sequences:
                    for use_fp8_kv_cache in _generation_fp8_values(backend, shape_sweep, sm_version):
                        if backend == "sglang" and sm_version < 90 and use_fp8_kv_cache:
                            continue
                        if backend == "trtllm" and sm_version <= 86 and use_fp8_kv_cache:
                            continue
                        if backend == "vllm" and _vllm_fp8_is_unsupported(
                            framework_version, sm_version, use_fp8_kv_cache
                        ):
                            continue

                        if backend == "sglang":
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    use_fp8_kv_cache,
                                    False,
                                    False,
                                    window_size,
                                ]
                            )
                        elif backend == "trtllm":
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    window_size,
                                    use_fp8_kv_cache,
                                    False,
                                    False,
                                ]
                            )
                        else:
                            test_cases.append(
                                [
                                    batch_size,
                                    sequence_length,
                                    num_heads,
                                    num_kv_heads,
                                    head_dim,
                                    use_fp8_kv_cache,
                                    False,
                                    window_size,
                                ]
                            )
    return test_cases


def build_encoder_attention_cases(
    backend: str,
    shape_sweeps: Iterable[ShapeSweep],
    *,
    sm_version: int,
    framework_version: str,
) -> list[AttentionCase]:
    """Build ordered non-causal encoder-attention runtime cases."""

    _validate_backend(backend, _ENCODER_BACKENDS)
    # Reserved for backend-specific encoder capability checks.  Keeping these
    # explicit inputs makes the plan-only and runtime APIs identical.
    _ = sm_version, framework_version
    test_cases: list[AttentionCase] = []
    for shape_sweep in shape_sweeps:
        batch_sizes = _int_list(shape_sweep["batch_sizes"], field_name="batch_sizes")
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"], field_name="sequence_lengths")
        for head_config in get_attention_encoder_head_configs(dict(shape_sweep)):
            num_heads = head_config.num_heads
            head_dim = head_config.head_dim
            for sequence_length in sorted(sequence_lengths):
                for batch_size in sorted(batch_sizes):
                    if _encoder_reachability_reason(
                        batch_size=batch_size,
                        sequence_length=sequence_length,
                        num_heads=num_heads,
                        head_dim=head_dim,
                    ):
                        continue
                    test_cases.append([batch_size, sequence_length, num_heads, head_dim])
    return test_cases


def _case_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"attention case field {field_name!r} must be an integer, not bool")
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"attention case field {field_name!r} must be an integer: {value!r}") from exc
    if number != result:
        raise ValueError(f"attention case field {field_name!r} must be integral: {value!r}")
    return result


def _case_boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"attention case field {field_name!r} must be bool: {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class AttentionOpCaseSchema:
    """Validate one legacy attention ABI and key it like the SDK consumer.

    ``normalize`` preserves the positional runtime ABI as a tuple so the
    collector can still invoke ``run_*(*case)``.  ``invocation_key`` projects
    that tuple through the same :func:`physical_row_key` contract used for
    persisted rows; fields such as generation attention dtype therefore do
    not accidentally create duplicate consumer entries.
    """

    op: str

    def __post_init__(self) -> None:
        if self.op not in _EXPECTED_TABLES:
            raise ValueError(f"unsupported attention schema op: {self.op!r}")

    def _layout(self, backend: str) -> tuple[str, ...]:
        if self.op == "encoder_attention":
            _validate_backend(backend, _ENCODER_BACKENDS)
            return _ENCODER_LAYOUT
        _validate_backend(backend, _ATTENTION_BACKENDS)
        layouts = _CONTEXT_LAYOUTS if self.op == "attention_context" else _GENERATION_LAYOUTS
        return layouts[backend]

    def _positional_values(self, payload: object, context: PlanContext) -> dict[str, int | bool]:
        layout = self._layout(context.backend)
        if not isinstance(payload, list | tuple):
            raise TypeError(f"{self.op}/{context.backend} case must be a positional list or tuple")
        if len(payload) != len(layout):
            raise ValueError(
                f"{self.op}/{context.backend} case has {len(payload)} fields; "
                f"expected {len(layout)} ({', '.join(layout)})"
            )

        values: dict[str, int | bool] = {}
        for field_name, value in zip(layout, payload, strict=True):
            if field_name in _BOOLEAN_FIELDS:
                values[field_name] = _case_boolean(value, field_name)
            else:
                values[field_name] = _case_integer(value, field_name)
        return values

    @staticmethod
    def _pop_alias(
        data: dict[object, object],
        aliases: tuple[str, ...],
        *,
        required: bool = True,
        default: object = _MISSING,
    ) -> object:
        present = [name for name in aliases if name in data]
        if len(present) > 1:
            raise ValueError(f"attention semantic case sets conflicting aliases: {', '.join(present)}")
        if present:
            return data.pop(present[0])
        if required:
            raise ValueError(f"attention semantic case is missing required field: {' or '.join(aliases)}")
        return default

    def _semantic_heads(self, data: dict[object, object], *, encoder: bool) -> tuple[int, int | None]:
        raw_tp = data.pop("tensor_parallel_size", _MISSING)
        local_q_fields = ("num_heads", "query_heads")
        local_kv_fields = ("num_kv_heads", "kv_heads")
        native_fields = ("num_attention_heads",) if encoder else ("num_attention_heads", "num_key_value_heads")

        if raw_tp is _MISSING:
            unexpected_native = [field_name for field_name in native_fields if field_name in data]
            if unexpected_native:
                raise ValueError("native attention heads require tensor_parallel_size: " + ", ".join(unexpected_native))
            num_heads = _case_integer(self._pop_alias(data, local_q_fields), "num_heads")
            if encoder:
                return num_heads, None
            num_kv_heads = _case_integer(self._pop_alias(data, local_kv_fields), "num_kv_heads")
            return num_heads, num_kv_heads

        conflicting_local = [field_name for field_name in (*local_q_fields, *local_kv_fields) if field_name in data]
        if conflicting_local:
            raise ValueError(
                "tensor_parallel_size cases must use native head fields, not local fields: "
                + ", ".join(conflicting_local)
            )
        tp_size = _case_integer(raw_tp, "tensor_parallel_size")
        if tp_size <= 0:
            raise ValueError("attention case field 'tensor_parallel_size' must be positive")
        native_num_heads = _case_integer(
            self._pop_alias(data, ("num_attention_heads",)),
            "num_attention_heads",
        )
        if native_num_heads <= 0 or native_num_heads % tp_size:
            raise ValueError(
                f"native num_attention_heads={native_num_heads} must be positive and divisible by "
                f"tensor_parallel_size={tp_size}"
            )
        num_heads = native_num_heads // tp_size
        if encoder:
            return num_heads, None

        native_num_kv_heads = _case_integer(
            self._pop_alias(data, ("num_key_value_heads",)),
            "num_key_value_heads",
        )
        if native_num_kv_heads <= 0:
            raise ValueError("attention case field 'num_key_value_heads' must be positive")
        num_kv_heads = (native_num_kv_heads + tp_size - 1) // tp_size
        return num_heads, num_kv_heads

    def _semantic_values(self, payload: Mapping[object, object], context: PlanContext) -> dict[str, int | bool]:
        data = dict(payload)
        values: dict[str, int | bool] = {
            "batch_size": _case_integer(self._pop_alias(data, ("batch_size",)), "batch_size"),
            "sequence_length": _case_integer(
                self._pop_alias(data, ("sequence_length",)),
                "sequence_length",
            ),
            "head_dim": _case_integer(self._pop_alias(data, ("head_dim",)), "head_dim"),
        }
        num_heads, num_kv_heads = self._semantic_heads(data, encoder=self.op == "encoder_attention")
        values["num_heads"] = num_heads

        expected_context = self.op != "attention_generation"
        raw_phase = data.pop("is_context_phase", expected_context)
        is_context_phase = _case_boolean(raw_phase, "is_context_phase")
        if is_context_phase is not expected_context:
            raise ValueError(
                f"{self.op}/{context.backend} semantic case has wrong phase flag: is_context_phase={is_context_phase!r}"
            )

        if self.op != "encoder_attention":
            assert num_kv_heads is not None
            values.update(
                {
                    "num_kv_heads": num_kv_heads,
                    "window_size": _case_integer(
                        self._pop_alias(data, ("window_size",)),
                        "window_size",
                    ),
                    "use_fp8_kv_cache": _case_boolean(
                        self._pop_alias(data, ("use_fp8_kv_cache", "fp8_kv_cache")),
                        "use_fp8_kv_cache",
                    ),
                    "use_fp8_context_fmha": _case_boolean(
                        self._pop_alias(
                            data,
                            ("use_fp8_context_fmha", "fp8_context_fmha"),
                            required=False,
                            default=False,
                        ),
                        "use_fp8_context_fmha",
                    ),
                    "is_context_phase": is_context_phase,
                }
            )
            if context.backend in {"vllm", "vllm_xpu"} and values["use_fp8_context_fmha"]:
                raise ValueError(f"{context.backend} attention ABI cannot represent use_fp8_context_fmha=True")

        if data:
            unknown = ", ".join(sorted(repr(field_name) for field_name in data))
            raise ValueError(f"attention semantic case has unknown fields: {unknown}")
        return values

    def _validate_values(self, values: dict[str, int | bool], context: PlanContext) -> None:
        if context.op != self.op:
            raise ValueError(f"{self.op} schema cannot normalize context phase {context.op!r}")

        for field_name in ("batch_size", "sequence_length", "num_heads", "head_dim"):
            if values[field_name] <= 0:
                raise ValueError(f"attention case field {field_name!r} must be positive")
        if self.op != "encoder_attention":
            if values["num_kv_heads"] <= 0:
                raise ValueError("attention case field 'num_kv_heads' must be positive")
            if values["window_size"] < 0:
                raise ValueError("attention case field 'window_size' cannot be negative")
            expected_context = self.op == "attention_context"
            if values["is_context_phase"] is not expected_context:
                raise ValueError(
                    f"{self.op}/{context.backend} case has wrong phase flag: "
                    f"is_context_phase={values['is_context_phase']!r}"
                )
            if values.get("use_fp8_context_fmha", False) and not values["use_fp8_kv_cache"]:
                raise ValueError("FP8 context FMHA requires an FP8 KV cache")

    def _values(self, payload: object, context: PlanContext) -> dict[str, int | bool]:
        if context.op != self.op:
            raise ValueError(f"{self.op} schema cannot normalize context phase {context.op!r}")
        self._layout(context.backend)
        values = (
            self._semantic_values(payload, context)
            if isinstance(payload, Mapping)
            else self._positional_values(payload, context)
        )
        self._validate_values(values, context)
        return values

    def normalize(self, candidate: LogicalCandidate, context: PlanContext) -> tuple[int | bool, ...]:
        values = self._values(candidate.payload, context)
        return tuple(values[field_name] for field_name in self._layout(context.backend))

    def is_reachable(
        self,
        candidate: LogicalCandidate,
        context: PlanContext,
    ) -> tuple[bool, str | None]:
        values = self._values(candidate.payload, context)
        batch_size = int(values["batch_size"])
        sequence_length = int(values["sequence_length"])
        num_heads = int(values["num_heads"])
        head_dim = int(values["head_dim"])

        if self.op == "encoder_attention":
            reason = _encoder_reachability_reason(
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_heads=num_heads,
                head_dim=head_dim,
            )
            return reason is None, reason

        num_kv_heads = int(values["num_kv_heads"])
        if num_kv_heads > num_heads:
            return False, "attention.kv_heads_exceed_query_heads"

        if self.op == "attention_context":
            reason = _context_reachability_reason(
                backend=context.backend,
                sm_version=context.sm_version if context.sm_version is not None else 0,
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_mha_tokens=_CONTEXT_MAX_MHA_TOKENS,
                max_gqa_tokens=_CONTEXT_MAX_GQA_TOKENS,
                max_mha_batch=_CONTEXT_MAX_MHA_BATCH,
                max_index_elements=_ATTENTION_MAX_INDEX_ELEMENTS,
            )
            return reason is None, reason

        # Builder decode cases store the past-KV length, one less than the
        # sequence length used by its budget calculation.
        total_sequence_length = sequence_length + 1
        max_tokens = _GENERATION_MAX_MHA_TOKENS if num_heads == num_kv_heads else _GENERATION_MAX_XQA_TOKENS
        max_batch = _generation_max_batch(
            context.backend,
            total_sequence_length,
            num_heads,
            head_dim,
            max_tokens,
        )
        if batch_size > max_batch:
            return False, "attention.generation_token_budget_exceeded"
        return True, None

    def is_supported(
        self,
        candidate: LogicalCandidate,
        context: PlanContext,
    ) -> tuple[bool, str | None]:
        values = self._values(candidate.payload, context)
        if self.op == "encoder_attention":
            return True, None

        backend = context.backend
        sm_version = context.sm_version
        framework_version = context.framework_version or ""
        batch_size = int(values["batch_size"])
        sequence_length = int(values["sequence_length"])
        num_heads = int(values["num_heads"])
        num_kv_heads = int(values["num_kv_heads"])
        head_dim = int(values["head_dim"])
        window_size = int(values["window_size"])
        use_fp8_kv_cache = bool(values["use_fp8_kv_cache"])
        use_fp8_context_fmha = bool(values.get("use_fp8_context_fmha", False))

        if backend == "vllm_xpu":
            if head_dim not in {64, 128}:
                return False, "attention.vllm_xpu_head_dim_unsupported"
            if window_size not in {0, 128}:
                return False, "attention.vllm_xpu_window_unsupported"
            if num_heads // num_kv_heads > 16:
                return False, "attention.vllm_xpu_gqa_ratio_unsupported"
            if window_size > 0 and head_dim == 128:
                return False, "attention.vllm_xpu_swa_head_dim_unsupported"

        if backend == "trtllm":
            if num_kv_heads != num_heads and (num_kv_heads >= num_heads or num_heads % num_kv_heads != 0):
                return False, "attention.trtllm_gqa_head_layout_unsupported"
            if sm_version is not None and _trtllm_gqa_ratio_is_unsupported(
                num_heads,
                num_kv_heads,
                sm_version,
            ):
                return False, "attention.trtllm_gqa_ratio_unsupported"
            if self.op == "attention_context" and sm_version is not None:
                if _trtllm_sm89_rc15_long_context_gqa_is_unsupported(
                    framework_version,
                    sm_version,
                    batch_size,
                    sequence_length,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                ):
                    return False, "attention.trtllm_sm89_rc15_long_context_gqa_unsupported"
                if use_fp8_context_fmha and _trtllm_sm120_fp8_context_fmha_is_unsupported(
                    framework_version,
                    sm_version,
                    batch_size,
                    sequence_length,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                ):
                    return False, "attention.trtllm_sm120_fp8_context_fmha_unsupported"
                if _trtllm_sm89_rc15_fp8_context_mha_is_unsupported(
                    framework_version,
                    sm_version,
                    batch_size,
                    sequence_length,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    use_fp8_kv_cache,
                    use_fp8_context_fmha,
                ):
                    return False, "attention.trtllm_sm89_rc15_fp8_context_mha_unsupported"

        if backend == "vllm" and sm_version is not None:
            if _vllm_head_dim_is_unsupported(framework_version, sm_version, head_dim):
                return False, "attention.vllm_sm89_022_head_dim_unsupported"
            if _vllm_fp8_is_unsupported(framework_version, sm_version, use_fp8_kv_cache):
                return False, "attention.vllm_sm89_022_fp8_unsupported"
            if self.op == "attention_generation" and sm_version >= 100 and num_heads // num_kv_heads > 16:
                return False, "attention.vllm_sm100_gqa_ratio_unsupported"

        if use_fp8_kv_cache and sm_version is not None:
            if backend == "sglang" and sm_version < 90:
                return False, "attention.sglang_fp8_requires_sm90"
            if backend == "trtllm" and sm_version <= 86:
                return False, "attention.trtllm_fp8_requires_sm87"
            if backend == "vllm" and sm_version <= 86:
                return False, "attention.vllm_fp8_requires_sm87"

        return True, None

    def _row(self, payload: object, context: PlanContext) -> dict[str, object]:
        values = self._values(payload, context)
        if self.op == "encoder_attention":
            return {
                "attn_dtype": "bfloat16",
                "head_dim": values["head_dim"],
                "num_heads": values["num_heads"],
                "isl": values["sequence_length"],
                "batch_size": values["batch_size"],
            }

        use_fp8_fmha = bool(values.get("use_fp8_context_fmha", False))
        row = {
            "attn_dtype": "fp8" if use_fp8_fmha else "bfloat16",
            "kv_cache_dtype": "fp8" if values["use_fp8_kv_cache"] else "bfloat16",
            "num_heads": values["num_heads"],
            "num_key_value_heads": values["num_kv_heads"],
            "head_dim": values["head_dim"],
            "window_size": values["window_size"],
            "batch_size": values["batch_size"],
        }
        if self.op == "attention_context":
            row.update({"isl": values["sequence_length"], "step": 0})
        else:
            # Runtime collectors persist decode as isl=1, step=past KV.
            row.update({"isl": 1, "step": values["sequence_length"]})
        return row

    def invocation_key(self, candidate: LogicalCandidate, context: PlanContext) -> PhysicalRowKey:
        key = physical_row_key(context.perf_file, self._row(candidate.payload, context))
        expected_table = _EXPECTED_TABLES[self.op]
        if key is None or key.table != expected_table:
            actual_table = None if key is None else key.table
            raise ValueError(
                f"{self.op} requires perf table {expected_table!r}, got {actual_table!r} from {context.perf_file!r}"
            )
        return key


ATTENTION_CONTEXT_SCHEMA = AttentionOpCaseSchema("attention_context")
ATTENTION_GENERATION_SCHEMA = AttentionOpCaseSchema("attention_generation")
ENCODER_ATTENTION_SCHEMA = AttentionOpCaseSchema("encoder_attention")

_ATTENTION_SCHEMAS = (
    ATTENTION_CONTEXT_SCHEMA,
    ATTENTION_GENERATION_SCHEMA,
    ENCODER_ATTENTION_SCHEMA,
)


def register_attention_schemas(*, replace: bool = False) -> None:
    """Explicitly register attention schemas for runtime or plan-only use.

    Importing this module has no registry side effect.  Repeating this function
    is idempotent for these schemas; a foreign registration is preserved unless
    the caller opts into ``replace=True``.
    """

    for schema in _ATTENTION_SCHEMAS:
        existing = get_schema(schema.op)
        if existing is schema or (isinstance(existing, AttentionOpCaseSchema) and existing.op == schema.op):
            continue
        if existing is not legacy_passthrough and not replace:
            # Delegate to the registry so callers receive its consistent
            # duplicate-registration error and key details.
            register_schema(schema.op, schema)
            continue
        register_schema(schema.op, schema, replace=replace)
