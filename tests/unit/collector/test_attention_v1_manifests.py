# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_encoder_shape_sweeps,
    get_attention_generation_shape_sweeps,
)
from collector.planner.coverage import CoverageHeader, CoverageManifest, assert_legacy_subset, load
from collector.planner.physical_keys import PhysicalRowKey, physical_row_key
from collector.planner.schemas.attention import (
    build_attention_context_cases,
    build_attention_generation_cases,
    build_encoder_attention_cases,
)
from tools.collector.generate_v1_attention_manifests import (
    ENCODER_BASELINE_FRAMEWORK_VERSIONS,
    EXPECTED_COUNTS,
    FRAMEWORK_VERSIONS,
    HARDWARE_SCOPES,
    MANIFEST_ROOT,
    SOURCE_GIT_REFS,
    framework_version_for,
    generate_manifests,
    manifest_path,
    perf_table_for_op,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _current_case_to_row(backend: str, op: str, case: Sequence[int | bool]) -> dict[str, object]:
    """Translate each pure builder's positional ABI into its persisted row."""

    if op == "encoder_attention":
        batch, input_len, heads, head_dim = case
        return {
            "batch_size": int(batch),
            "isl": int(input_len),
            "num_heads": int(heads),
            "head_dim": int(head_dim),
            "attn_dtype": "bfloat16",
        }

    if backend == "sglang":
        batch, input_len, heads, kv_heads, head_dim, fp8_kv, fp8_fmha, is_context, window = case
    elif backend == "trtllm":
        batch, input_len, heads, kv_heads, head_dim, window, fp8_kv, fp8_fmha, is_context = case
    elif backend in {"vllm", "vllm_xpu"}:
        batch, input_len, heads, kv_heads, head_dim, fp8_kv, is_context, window = case
        fp8_fmha = False
    else:  # pragma: no cover - parametrization is tied to the active registry variants
        raise ValueError(f"unknown attention backend: {backend}")

    expected_context = op == "attention_context"
    if bool(is_context) != expected_context:
        raise ValueError(f"{backend}/{op} case has inconsistent phase flag: {case!r}")
    return {
        "batch_size": int(batch),
        "isl": int(input_len) if expected_context else 1,
        "num_heads": int(heads),
        "num_key_value_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "window_size": int(window),
        "beam_width": 1,
        "attn_dtype": "fp8" if bool(fp8_fmha) else "bfloat16",
        "kv_cache_dtype": "fp8" if bool(fp8_kv) else "bfloat16",
        "step": 0 if expected_context else int(input_len),
    }


def _build_current_manifest(backend: str, op: str) -> CoverageManifest:
    # The builders are framework-free. Project current cases into the frozen
    # baseline header scope for a physical-key set comparison; this does not
    # claim that today's configured runtime image satisfies that scope.
    framework_version = framework_version_for(backend, op)
    gpu_type, sm_version = HARDWARE_SCOPES[backend]
    if op == "attention_context":
        sweeps = get_attention_context_shape_sweeps(backend)
        cases = build_attention_context_cases(
            backend,
            sweeps,
            sm_version=sm_version,
            framework_version=framework_version,
        )
    elif op == "attention_generation":
        sweeps = get_attention_generation_shape_sweeps(backend)
        cases = build_attention_generation_cases(
            backend,
            sweeps,
            sm_version=sm_version,
            framework_version=framework_version,
        )
    elif op == "encoder_attention":
        sweeps = get_attention_encoder_shape_sweeps(backend)
        cases = build_encoder_attention_cases(
            backend,
            sweeps,
            sm_version=sm_version,
            framework_version=framework_version,
        )
    else:  # pragma: no cover - parametrization is tied to EXPECTED_COUNTS
        raise ValueError(f"unknown attention op: {op}")

    perf_table = perf_table_for_op(op)
    keys: set[PhysicalRowKey] = set()
    for case in cases:
        key = physical_row_key(perf_table, _current_case_to_row(backend, op, case))
        assert key is not None
        keys.add(key)

    return CoverageManifest(
        header=CoverageHeader(
            source_git_ref="working-tree-current",
            backend_variant=backend,
            framework_version=framework_version,
            gpu_type=gpu_type,
            sm_version=sm_version,
            perf_table=perf_table,
        ),
        keys=frozenset(keys),
    )


@pytest.mark.parametrize(
    ("backend", "op", "case", "expected"),
    [
        (
            "sglang",
            "attention_context",
            [2, 64, 8, 4, 128, True, True, True, 4096],
            (2, 64, 8, 4, 128, 4096, "fp8", "fp8", 0),
        ),
        (
            "trtllm",
            "attention_context",
            [2, 64, 8, 4, 128, 4096, True, True, True],
            (2, 64, 8, 4, 128, 4096, "fp8", "fp8", 0),
        ),
        (
            "vllm",
            "attention_context",
            [2, 64, 8, 4, 128, True, True, 4096],
            (2, 64, 8, 4, 128, 4096, "bfloat16", "fp8", 0),
        ),
        (
            "vllm_xpu",
            "attention_context",
            [2, 64, 8, 4, 128, True, True, 4096],
            (2, 64, 8, 4, 128, 4096, "bfloat16", "fp8", 0),
        ),
        (
            "sglang",
            "attention_generation",
            [2, 63, 8, 4, 128, True, False, False, 4096],
            (2, 1, 8, 4, 128, 4096, "bfloat16", "fp8", 63),
        ),
        (
            "sglang",
            "encoder_attention",
            [2, 576, 16, 72],
            (2, 576, 16, 72, "bfloat16"),
        ),
        (
            "trtllm",
            "encoder_attention",
            [2, 576, 16, 72],
            (2, 576, 16, 72, "bfloat16"),
        ),
        (
            "vllm",
            "encoder_attention",
            [2, 576, 16, 72],
            (2, 576, 16, 72, "bfloat16"),
        ),
        (
            "trtllm",
            "attention_generation",
            [2, 63, 8, 4, 128, 4096, True, False, False],
            (2, 1, 8, 4, 128, 4096, "bfloat16", "fp8", 63),
        ),
        (
            "vllm",
            "attention_generation",
            [2, 63, 8, 4, 128, True, False, 4096],
            (2, 1, 8, 4, 128, 4096, "bfloat16", "fp8", 63),
        ),
        (
            "vllm_xpu",
            "attention_generation",
            [2, 63, 8, 4, 128, True, False, 4096],
            (2, 1, 8, 4, 128, 4096, "bfloat16", "fp8", 63),
        ),
    ],
)
def test_current_attention_tuple_abi_to_row(backend, op, case, expected):
    row = _current_case_to_row(backend, op, case)
    if op == "encoder_attention":
        actual = (
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["head_dim"],
            row["attn_dtype"],
        )
    else:
        actual = (
            row["batch_size"],
            row["isl"],
            row["num_heads"],
            row["num_key_value_heads"],
            row["head_dim"],
            row["window_size"],
            row["attn_dtype"],
            row["kv_cache_dtype"],
            row["step"],
        )
    assert actual == expected


@pytest.mark.parametrize(("backend", "op", "expected_count"), [(*key, count) for key, count in EXPECTED_COUNTS.items()])
def test_current_attention_exact_keys_retain_historical_manifest(backend, op, expected_count):
    legacy = load(manifest_path(MANIFEST_ROOT, backend, op))

    assert legacy.header.source_git_ref == SOURCE_GIT_REFS[op]
    assert legacy.header.backend_variant == backend
    assert legacy.header.framework_version == framework_version_for(backend, op)
    assert (legacy.header.gpu_type, legacy.header.sm_version) == HARDWARE_SCOPES[backend]
    assert len(legacy.keys) == expected_count

    current = _build_current_manifest(backend, op)
    result = assert_legacy_subset(legacy, current)

    removed = len(result.missing)
    assert removed == 0
    assert result.retained_count == expected_count
    if op == "encoder_attention":
        assert len(result.added) == 671
        assert len(current.keys) == 7679


def test_historical_attention_manifest_generation_is_deterministic_and_current():
    paths = [manifest_path(MANIFEST_ROOT, backend, op) for backend, op in EXPECTED_COUNTS]
    before = {path: path.read_bytes() for path in paths}

    messages = generate_manifests(MANIFEST_ROOT, check=True)

    after = {path: path.read_bytes() for path in before}
    assert before == after
    assert len(messages) == len(EXPECTED_COUNTS)


def test_context_and_generation_headers_use_current_framework_manifest_versions():
    manifest = yaml.safe_load((_REPO_ROOT / "collector" / "framework_manifest.yaml").read_text())

    assert {
        "sglang": manifest["frameworks"]["sglang"]["version"],
        "trtllm": manifest["frameworks"]["trtllm"]["version"],
        "vllm": manifest["frameworks"]["vllm"]["version"],
        "vllm_xpu": manifest["frameworks"]["vllm"]["version"],
    } == FRAMEWORK_VERSIONS


def test_encoder_initial_baseline_headers_use_original_minimum_compatible_ranges():
    assert ENCODER_BASELINE_FRAMEWORK_VERSIONS == {
        "sglang": ">=0.5.11",
        "trtllm": ">=1.3.0rc5",
        "vllm": ">=0.21.0",
    }
