# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_encoder_shape_sweeps,
    get_attention_generation_shape_sweeps,
)
from collector.planner.compiler import compile_population
from collector.planner.context import use_case_catalog
from collector.planner.models import LogicalCandidate, PlanContext, RuleSource, legacy_rule
from collector.planner.physical_keys import physical_row_key
from collector.planner.registry import clear_schemas, get_schema, legacy_passthrough
from collector.planner.schemas import attention
from collector.planner.schemas.attention import (
    ATTENTION_CONTEXT_SCHEMA,
    ATTENTION_GENERATION_SCHEMA,
    ENCODER_ATTENTION_SCHEMA,
    build_attention_context_cases,
    build_attention_generation_cases,
    build_encoder_attention_cases,
    register_attention_schemas,
)


@contextmanager
def _without_repository_model_profiles():
    catalog = SimpleNamespace(model_documents=(), model_path=None)
    with use_case_catalog(catalog):
        yield


def _context_sweep(**overrides):
    sweep = {
        "batch_sizes": [1],
        "sequence_lengths": [64],
        "query_head_counts": [8],
        "kv_head_options": [0],
        "head_dims": [64],
        "window_sizes": [0],
        "max_tokens_self_attention": 65536,
        "max_tokens_grouped_query_attention": 131072,
        "max_batch_size_self_attention": 128,
        "max_kv_elements": 2**40,
        "precision_cases": [
            {"fp8_kv_cache": False, "fp8_context_fmha": False},
            {"fp8_kv_cache": True, "fp8_context_fmha": True},
        ],
    }
    sweep.update(overrides)
    return sweep


def _generation_sweep(**overrides):
    sweep = {
        "batch_sizes": [1],
        "sequence_lengths": [16],
        "mha_query_head_counts": [8],
        "xqa_query_head_counts": [],
        "kv_head_counts": [],
        "head_dims": [64],
        "window_sizes": [0],
        "max_mha_tokens_per_step": 65536,
        "max_xqa_tokens_per_step": 65536,
        "min_batch_options_per_sequence": 1,
        "drop_largest_sequence_for_batch_at_least": 99,
        "precision_cases": [
            {"fp8_kv_cache": False, "fp8_context_fmha": False},
            {"fp8_kv_cache": True, "fp8_context_fmha": True},
        ],
    }
    sweep.update(overrides)
    return sweep


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "sglang",
            [
                [1, 64, 8, 8, 64, False, False, True, 0],
                [1, 64, 8, 8, 64, True, True, True, 0],
            ],
        ),
        (
            "trtllm",
            [
                [1, 64, 8, 8, 64, 0, False, False, True],
                [1, 64, 8, 8, 64, 0, True, True, True],
            ],
        ),
        (
            "vllm",
            [
                [1, 64, 8, 8, 64, False, True, 0],
                [1, 64, 8, 8, 64, True, True, 0],
            ],
        ),
        (
            "vllm_xpu",
            [
                [1, 64, 8, 8, 64, False, True, 0],
                [1, 64, 8, 8, 64, True, True, 0],
            ],
        ),
    ],
)
def test_context_cases_preserve_backend_positional_abi(backend, expected):
    with _without_repository_model_profiles():
        actual = build_attention_context_cases(
            backend,
            [_context_sweep()],
            sm_version=90,
            framework_version="test",
        )
    assert actual == expected


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "sglang",
            [
                [1, 15, 8, 8, 64, False, False, False, 0],
                [1, 15, 8, 8, 64, True, False, False, 0],
            ],
        ),
        (
            "trtllm",
            [
                [1, 15, 8, 8, 64, 0, False, False, False],
                [1, 15, 8, 8, 64, 0, True, False, False],
            ],
        ),
        (
            "vllm",
            [
                [1, 15, 8, 8, 64, False, False, 0],
                [1, 15, 8, 8, 64, True, False, 0],
            ],
        ),
        (
            "vllm_xpu",
            [
                [1, 15, 8, 8, 64, False, False, 0],
                [1, 15, 8, 8, 64, True, False, 0],
            ],
        ),
    ],
)
def test_generation_cases_preserve_backend_positional_abi(backend, expected):
    with _without_repository_model_profiles():
        actual = build_attention_generation_cases(
            backend,
            [_generation_sweep()],
            sm_version=90,
            framework_version="test",
        )
    assert actual == expected


def test_sglang_keeps_its_stricter_fp8_sm_threshold():
    with _without_repository_model_profiles():
        sglang = build_attention_context_cases("sglang", [_context_sweep()], sm_version=89, framework_version="test")
        trtllm = build_attention_context_cases("trtllm", [_context_sweep()], sm_version=89, framework_version="test")
    assert len(sglang) == 1
    assert len(trtllm) == 2


@pytest.mark.parametrize("framework_version", ["1.3.0rc5", "1.3.0rc5.post1"])
def test_trtllm_sm120_rc5_skips_only_unsupported_fp8_fmha(framework_version):
    sweep = _context_sweep(
        query_head_counts=[96],
        head_dims=[256],
        sequence_lengths=[128],
    )
    with _without_repository_model_profiles():
        actual = build_attention_context_cases(
            "trtllm",
            [sweep],
            sm_version=120,
            framework_version=framework_version,
        )
    assert actual == [[1, 128, 96, 96, 256, 0, False, False, True]]


def test_trtllm_sm89_rc15_filters_verified_failure_regions():
    mha = _context_sweep(
        query_head_counts=[96],
        head_dims=[128],
        sequence_lengths=[65536],
        max_tokens_self_attention=65536,
    )
    gqa = _context_sweep(
        query_head_counts=[96],
        kv_head_options=[8],
        head_dims=[128],
        sequence_lengths=[98304],
        max_tokens_grouped_query_attention=98304,
    )
    with _without_repository_model_profiles():
        actual_mha = build_attention_context_cases("trtllm", [mha], sm_version=89, framework_version="1.3.0rc15")
        actual_gqa = build_attention_context_cases("trtllm", [gqa], sm_version=89, framework_version="1.3.0rc15")
    assert actual_mha == [[1, 65536, 96, 96, 128, 0, False, False, True]]
    assert actual_gqa == []


def test_vllm_sm89_022_filters_fp8_and_large_flashinfer_heads():
    sweep = _context_sweep(head_dims=[128, 512])
    with _without_repository_model_profiles():
        actual = build_attention_context_cases("vllm", [sweep], sm_version=89, framework_version="0.22.0")
    assert actual == [[1, 64, 8, 8, 128, False, True, 0]]


def test_xpu_filters_unsupported_gqa_window_head_combinations():
    sweep = _context_sweep(
        query_head_counts=[16],
        kv_head_options=[1],
        head_dims=[64, 128],
        window_sizes=[0, 128],
    )
    with _without_repository_model_profiles():
        actual = build_attention_context_cases("vllm_xpu", [sweep], sm_version=0, framework_version="test")
    assert [(case[4], case[7]) for case in actual] == [
        (64, 0),
        (64, 0),
        (64, 128),
        (64, 128),
        (128, 0),
        (128, 0),
    ]


def test_generation_gqa_limits_are_backend_specific():
    sweep = _generation_sweep(
        mha_query_head_counts=[],
        xqa_query_head_counts=[64],
        kv_head_counts=[2],
    )
    with _without_repository_model_profiles():
        vllm_sm90 = build_attention_generation_cases("vllm", [sweep], sm_version=90, framework_version="test")
        vllm_sm100 = build_attention_generation_cases("vllm", [sweep], sm_version=100, framework_version="test")
        trtllm_sm100 = build_attention_generation_cases("trtllm", [sweep], sm_version=100, framework_version="test")
        xpu = build_attention_generation_cases("vllm_xpu", [sweep], sm_version=0, framework_version="test")
    assert len(vllm_sm90) == 2
    assert vllm_sm100 == []
    # TRT-LLM supports ratios that are exact multiples of its 32-head CTA.
    assert len(trtllm_sm100) == 2
    assert xpu == []


def test_trtllm_generation_rejects_nondivisible_gqa_layout():
    sweep = _generation_sweep(
        mha_query_head_counts=[],
        xqa_query_head_counts=[10],
        kv_head_counts=[3],
    )

    with _without_repository_model_profiles():
        actual = build_attention_generation_cases("trtllm", [sweep], sm_version=100, framework_version="test")

    assert actual == []


def test_encoder_cases_are_sorted_and_apply_both_safety_budgets():
    sweep = {
        "batch_sizes": [2, 1, 131073],
        "sequence_lengths": [16, 1],
        "head_counts": [8],
        "head_dims": [64],
    }
    with _without_repository_model_profiles():
        actual = build_encoder_attention_cases("vllm", [sweep], sm_version=100, framework_version="test")
    assert actual == [
        [1, 1, 8, 64],
        [2, 1, 8, 64],
        [1, 16, 8, 64],
        [2, 16, 8, 64],
    ]


def test_planner_schema_has_no_gpu_framework_imports():
    imported_roots = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(ast.parse(inspect.getsource(attention)))
        if isinstance(node, ast.Import | ast.ImportFrom)
    }
    assert imported_roots.isdisjoint({"torch", "sglang", "tensorrt_llm", "vllm"})


def test_unknown_backend_is_rejected_before_population():
    with pytest.raises(ValueError, match="unsupported attention backend"):
        build_attention_context_cases("unknown", [], sm_version=0, framework_version="test")


def _plan_context(op, backend, perf_file, *, sm_version=100, framework_version="test"):
    return PlanContext(
        backend=backend,
        op=op,
        perf_file=perf_file,
        schema_version="planner-v1",
        sm_version=sm_version,
        framework_version=framework_version,
    )


@pytest.mark.parametrize(
    ("schema", "context", "raw_case", "equivalent_row"),
    [
        (
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "sglang", "context_attention_perf.txt"),
            [2, 64, 8, 8, 64, True, True, True, 0],
            {
                "attn_dtype": "fp8",
                "kv_cache_dtype": "fp8",
                "num_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "window_size": 0,
                "isl": 64,
                "batch_size": 2,
            },
        ),
        (
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "trtllm", "context_attention_perf.txt"),
            [2, 64, 8, 8, 64, 0, True, True, True],
            {
                "attn_dtype": "fp8",
                "kv_cache_dtype": "fp8",
                "num_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "window_size": 0,
                "isl": 64,
                "batch_size": 2,
            },
        ),
        *[
            (
                ATTENTION_CONTEXT_SCHEMA,
                _plan_context("attention_context", backend, "context_attention_perf.txt"),
                [2, 64, 8, 8, 64, True, True, 0],
                {
                    "attn_dtype": "bfloat16",
                    "kv_cache_dtype": "fp8",
                    "num_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 64,
                    "window_size": 0,
                    "isl": 64,
                    "batch_size": 2,
                },
            )
            for backend in ("vllm", "vllm_xpu")
        ],
        (
            ATTENTION_GENERATION_SCHEMA,
            _plan_context("attention_generation", "sglang", "generation_attention_perf.txt"),
            [2, 15, 8, 8, 64, True, False, False, 0],
            {
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "num_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "window_size": 0,
                "isl": 1,
                "step": 15,
                "batch_size": 2,
            },
        ),
        (
            ATTENTION_GENERATION_SCHEMA,
            _plan_context("attention_generation", "trtllm", "generation_attention_perf.txt"),
            [2, 15, 8, 8, 64, 0, True, False, False],
            {
                "attn_dtype": "bfloat16",
                "kv_cache_dtype": "fp8",
                "num_heads": 8,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "window_size": 0,
                "isl": 1,
                "step": 15,
                "batch_size": 2,
            },
        ),
        *[
            (
                ATTENTION_GENERATION_SCHEMA,
                _plan_context("attention_generation", backend, "generation_attention_perf.txt"),
                [2, 15, 8, 8, 64, True, False, 0],
                {
                    "attn_dtype": "bfloat16",
                    "kv_cache_dtype": "fp8",
                    "num_heads": 8,
                    "num_key_value_heads": 8,
                    "head_dim": 64,
                    "window_size": 0,
                    "isl": 1,
                    "step": 15,
                    "batch_size": 2,
                },
            )
            for backend in ("vllm", "vllm_xpu")
        ],
        *[
            (
                ENCODER_ATTENTION_SCHEMA,
                _plan_context("encoder_attention", backend, "encoder_attention_perf.txt"),
                [2, 64, 8, 64],
                {
                    "attn_dtype": "bfloat16",
                    "head_dim": 64,
                    "num_heads": 8,
                    "isl": 64,
                    "batch_size": 2,
                },
            )
            for backend in ("sglang", "trtllm", "vllm")
        ],
    ],
)
def test_op_schema_key_matches_physical_row_contract(schema, context, raw_case, equivalent_row):
    source = RuleSource("test")
    normalized = schema.normalize(LogicalCandidate(raw_case, source), context)
    planner_key = schema.invocation_key(LogicalCandidate(normalized, source), context)

    assert planner_key == physical_row_key(context.perf_file, equivalent_row)
    # MHA is normalized to the zero-KV-head consumer sentinel.
    if context.op != "encoder_attention":
        assert planner_key.values[1 if context.op == "attention_generation" else 2] == 0


def test_generation_schema_dedupes_flags_absent_from_consumer_key():
    context = _plan_context("attention_generation", "sglang", "generation_attention_perf.txt")
    without_context_fmha = [2, 15, 8, 8, 64, True, False, False, 0]
    with_context_fmha = [2, 15, 8, 8, 64, True, True, False, 0]

    result = compile_population(
        [legacy_rule([without_context_fmha, with_context_fmha])],
        context,
        schema=ATTENTION_GENERATION_SCHEMA,
    )

    assert result.report.scheduled == 1
    assert result.report.duplicate_invocations == 1
    assert result.cases[0].payload == tuple(without_context_fmha)


def test_op_schema_normalizes_numeric_values_but_preserves_runtime_abi():
    context = _plan_context("attention_context", "vllm", "context_attention_perf.txt")
    raw_case = ["2", 64.0, "8", 8, "64", True, True, "0"]

    normalized = ATTENTION_CONTEXT_SCHEMA.normalize(
        LogicalCandidate(raw_case, RuleSource("test")),
        context,
    )

    assert normalized == (2, 64, 8, 8, 64, True, True, 0)


def test_op_schema_rejects_wrong_length_and_phase():
    context = _plan_context("attention_context", "sglang", "context_attention_perf.txt")
    source = RuleSource("test")
    with pytest.raises(ValueError, match="has 8 fields; expected 9"):
        ATTENTION_CONTEXT_SCHEMA.normalize(LogicalCandidate([1] * 8, source), context)
    with pytest.raises(ValueError, match="wrong phase flag"):
        ATTENTION_CONTEXT_SCHEMA.normalize(
            LogicalCandidate([1, 64, 8, 8, 64, False, False, False, 0], source),
            context,
        )

    generation_context = _plan_context(
        "attention_generation",
        "sglang",
        "generation_attention_perf.txt",
    )
    with pytest.raises(ValueError, match="cannot normalize context phase"):
        ATTENTION_CONTEXT_SCHEMA.normalize(
            LogicalCandidate([1, 64, 8, 8, 64, False, False, True, 0], source),
            generation_context,
        )


def test_attention_schema_registration_is_explicit_and_idempotent():
    clear_schemas()
    try:
        assert get_schema("attention_context") is legacy_passthrough

        register_attention_schemas()
        assert get_schema("attention_context") is ATTENTION_CONTEXT_SCHEMA
        assert get_schema("attention_generation") is ATTENTION_GENERATION_SCHEMA
        assert get_schema("encoder_attention") is ENCODER_ATTENTION_SCHEMA

        register_attention_schemas()
        assert get_schema("attention_context") is ATTENTION_CONTEXT_SCHEMA
    finally:
        clear_schemas()


def _semantic_attention_case(**overrides):
    case = {
        "batch_size": 2,
        "sequence_length": 64,
        "num_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "window_size": 0,
        "use_fp8_kv_cache": True,
        "use_fp8_context_fmha": False,
    }
    case.update(overrides)
    return case


def test_semantic_mapping_normalizes_to_backend_tuple_and_physical_key():
    context = _plan_context("attention_context", "trtllm", "context_attention_perf.txt")
    semantic_case = {
        "batch_size": "2",
        "sequence_length": 64,
        "query_heads": 8,
        "kv_heads": 2,
        "head_dim": 64,
        "window_size": 4096,
        "fp8_kv_cache": True,
        "fp8_context_fmha": True,
    }
    source = RuleSource("semantic")

    normalized = ATTENTION_CONTEXT_SCHEMA.normalize(LogicalCandidate(semantic_case, source), context)
    planner_key = ATTENTION_CONTEXT_SCHEMA.invocation_key(LogicalCandidate(normalized, source), context)
    expected_key = physical_row_key(
        context.perf_file,
        {
            "attn_dtype": "fp8",
            "kv_cache_dtype": "fp8",
            "num_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "window_size": 4096,
            "isl": 64,
            "batch_size": 2,
        },
    )

    assert normalized == (2, 64, 8, 2, 64, 4096, True, True, True)
    assert planner_key == expected_key


@pytest.mark.parametrize(
    "head_fields",
    [
        {"num_heads": 8, "num_kv_heads": 2},
        {"query_heads": 8, "kv_heads": 2},
    ],
)
def test_semantic_mapping_accepts_local_head_aliases(head_fields):
    context = _plan_context("attention_generation", "vllm", "generation_attention_perf.txt")
    semantic_case = {
        "batch_size": 2,
        "sequence_length": 15,
        "head_dim": 64,
        "window_size": 0,
        "fp8_kv_cache": True,
        **head_fields,
    }

    normalized = ATTENTION_GENERATION_SCHEMA.normalize(
        LogicalCandidate(semantic_case, RuleSource("aliases")),
        context,
    )

    assert normalized == (2, 15, 8, 2, 64, True, False, 0)


def test_semantic_mapping_projects_native_heads_through_tensor_parallelism():
    context = _plan_context("attention_generation", "sglang", "generation_attention_perf.txt")
    semantic_case = {
        "batch_size": 2,
        "sequence_length": 15,
        "num_attention_heads": 32,
        "num_key_value_heads": 10,
        "tensor_parallel_size": 4,
        "head_dim": 128,
        "window_size": 0,
        "fp8_kv_cache": False,
    }

    normalized = ATTENTION_GENERATION_SCHEMA.normalize(
        LogicalCandidate(semantic_case, RuleSource("native_tp")),
        context,
    )

    # Query heads divide exactly; KV heads follow the generator's ceil-div rule.
    assert normalized == (2, 15, 8, 3, 128, False, False, False, 0)


def test_encoder_semantic_mapping_supports_local_alias_and_native_tp():
    context = _plan_context("encoder_attention", "vllm", "encoder_attention_perf.txt")
    source = RuleSource("encoder")

    local = ENCODER_ATTENTION_SCHEMA.normalize(
        LogicalCandidate(
            {"batch_size": 2, "sequence_length": 64, "query_heads": 8, "head_dim": 64},
            source,
        ),
        context,
    )
    native = ENCODER_ATTENTION_SCHEMA.normalize(
        LogicalCandidate(
            {
                "batch_size": 2,
                "sequence_length": 64,
                "num_attention_heads": 32,
                "tensor_parallel_size": 4,
                "head_dim": 64,
            },
            source,
        ),
        context,
    )

    assert local == native == (2, 64, 8, 64)
    assert ENCODER_ATTENTION_SCHEMA.invocation_key(LogicalCandidate(native, source), context) == physical_row_key(
        context.perf_file,
        {
            "attn_dtype": "bfloat16",
            "head_dim": 64,
            "num_heads": 8,
            "isl": 64,
            "batch_size": 2,
        },
    )


def test_semantic_mapping_rejects_nondivisible_native_query_heads():
    context = _plan_context("attention_context", "sglang", "context_attention_perf.txt")
    semantic_case = {
        "batch_size": 1,
        "sequence_length": 64,
        "num_attention_heads": 30,
        "num_key_value_heads": 8,
        "tensor_parallel_size": 4,
        "head_dim": 128,
        "window_size": 0,
        "fp8_kv_cache": False,
    }

    with pytest.raises(ValueError, match="must be positive and divisible"):
        ATTENTION_CONTEXT_SCHEMA.normalize(
            LogicalCandidate(semantic_case, RuleSource("bad_tp")),
            context,
        )


def test_semantic_mapping_strictly_rejects_missing_unknown_and_conflicting_fields():
    context = _plan_context("attention_context", "vllm", "context_attention_perf.txt")
    source = RuleSource("invalid")

    missing = _semantic_attention_case()
    missing.pop("head_dim")
    with pytest.raises(ValueError, match="head_dim"):
        ATTENTION_CONTEXT_SCHEMA.normalize(LogicalCandidate(missing, source), context)

    unknown = _semantic_attention_case(typo_field=1)
    with pytest.raises(ValueError, match=r"unknown fields.*typo_field"):
        ATTENTION_CONTEXT_SCHEMA.normalize(LogicalCandidate(unknown, source), context)

    conflicting = _semantic_attention_case(query_heads=8)
    with pytest.raises(ValueError, match="conflicting aliases"):
        ATTENTION_CONTEXT_SCHEMA.normalize(LogicalCandidate(conflicting, source), context)


def test_semantic_mapping_rejects_flags_the_backend_abi_cannot_represent():
    context = _plan_context("attention_context", "vllm", "context_attention_perf.txt")
    semantic_case = _semantic_attention_case(use_fp8_context_fmha=True)

    with pytest.raises(ValueError, match="cannot represent"):
        ATTENTION_CONTEXT_SCHEMA.normalize(
            LogicalCandidate(semantic_case, RuleSource("unsupported_flag")),
            context,
        )


def _compile_exact_case(schema, context, case):
    return compile_population(
        [legacy_rule([case])],
        context,
        schema=schema,
    )


@pytest.mark.parametrize(
    ("schema", "context", "case", "reason"),
    [
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "vllm", "context_attention_perf.txt"),
            _semantic_attention_case(num_heads=4, num_kv_heads=8, use_fp8_kv_cache=False),
            "attention.kv_heads_exceed_query_heads",
            id="context-invalid-head-structure",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "vllm", "context_attention_perf.txt"),
            _semantic_attention_case(
                batch_size=1,
                sequence_length=65537,
                num_heads=8,
                num_kv_heads=8,
                use_fp8_kv_cache=False,
            ),
            "attention.context_mha_token_budget_exceeded",
            id="context-token-budget",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "vllm", "context_attention_perf.txt"),
            _semantic_attention_case(
                batch_size=1,
                sequence_length=131072,
                num_heads=8,
                num_kv_heads=1,
                head_dim=8192,
                use_fp8_kv_cache=False,
            ),
            "attention.context_kv_index_budget_exceeded",
            id="context-index-budget",
        ),
        pytest.param(
            ATTENTION_GENERATION_SCHEMA,
            _plan_context("attention_generation", "trtllm", "generation_attention_perf.txt"),
            _semantic_attention_case(
                batch_size=2,
                sequence_length=131071,
                num_heads=64,
                num_kv_heads=64,
                head_dim=128,
                use_fp8_kv_cache=False,
            ),
            "attention.generation_token_budget_exceeded",
            id="generation-token-budget",
        ),
        pytest.param(
            ENCODER_ATTENTION_SCHEMA,
            _plan_context("encoder_attention", "vllm", "encoder_attention_perf.txt"),
            {"batch_size": 2, "sequence_length": 65537, "num_heads": 8, "head_dim": 64},
            "attention.encoder_token_budget_exceeded",
            id="encoder-token-budget",
        ),
        pytest.param(
            ENCODER_ATTENTION_SCHEMA,
            _plan_context("encoder_attention", "vllm", "encoder_attention_perf.txt"),
            {"batch_size": 1, "sequence_length": 131072, "num_heads": 32, "head_dim": 80},
            "attention.encoder_index_budget_exceeded",
            id="encoder-index-budget",
        ),
    ],
)
def test_exact_mapping_reports_unreachable_guards(schema, context, case, reason):
    result = _compile_exact_case(schema, context, case)

    assert result.report.scheduled == 0
    assert result.report.unreachable == 1
    assert result.report.unsupported == 0
    assert result.report.decisions[0].reason == reason


@pytest.mark.parametrize(
    ("schema", "context", "case", "reason"),
    [
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "sglang",
                "context_attention_perf.txt",
                sm_version=89,
            ),
            _semantic_attention_case(use_fp8_kv_cache=True),
            "attention.sglang_fp8_requires_sm90",
            id="sglang-fp8-sm",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "trtllm", "context_attention_perf.txt"),
            _semantic_attention_case(
                num_heads=10,
                num_kv_heads=3,
                use_fp8_kv_cache=False,
            ),
            "attention.trtllm_gqa_head_layout_unsupported",
            id="trtllm-gqa-layout",
        ),
        pytest.param(
            ATTENTION_GENERATION_SCHEMA,
            _plan_context("attention_generation", "trtllm", "generation_attention_perf.txt"),
            _semantic_attention_case(
                num_heads=10,
                num_kv_heads=3,
                use_fp8_kv_cache=False,
                is_context_phase=False,
            ),
            "attention.trtllm_gqa_head_layout_unsupported",
            id="trtllm-generation-gqa-layout",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context("attention_context", "trtllm", "context_attention_perf.txt"),
            _semantic_attention_case(
                num_heads=48,
                num_kv_heads=1,
                use_fp8_kv_cache=False,
            ),
            "attention.trtllm_gqa_ratio_unsupported",
            id="trtllm-gqa-ratio",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "trtllm",
                "context_attention_perf.txt",
                sm_version=120,
                framework_version="1.3.0rc5",
            ),
            _semantic_attention_case(
                sequence_length=128,
                num_heads=8,
                num_kv_heads=8,
                head_dim=256,
                use_fp8_kv_cache=True,
                use_fp8_context_fmha=True,
            ),
            "attention.trtllm_sm120_fp8_context_fmha_unsupported",
            id="trtllm-versioned-fp8-fmha",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "vllm",
                "context_attention_perf.txt",
                sm_version=89,
                framework_version="0.22.0",
            ),
            _semantic_attention_case(head_dim=512, use_fp8_kv_cache=False),
            "attention.vllm_sm89_022_head_dim_unsupported",
            id="vllm-versioned-head-dim",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "vllm",
                "context_attention_perf.txt",
                sm_version=89,
                framework_version="0.22.0",
            ),
            _semantic_attention_case(head_dim=128, use_fp8_kv_cache=True),
            "attention.vllm_sm89_022_fp8_unsupported",
            id="vllm-versioned-fp8",
        ),
        pytest.param(
            ATTENTION_GENERATION_SCHEMA,
            _plan_context("attention_generation", "vllm", "generation_attention_perf.txt"),
            _semantic_attention_case(
                sequence_length=15,
                num_heads=64,
                num_kv_heads=2,
                use_fp8_kv_cache=False,
            ),
            "attention.vllm_sm100_gqa_ratio_unsupported",
            id="vllm-gqa-ratio",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "vllm_xpu",
                "context_attention_perf.txt",
                sm_version=0,
            ),
            _semantic_attention_case(num_heads=32, num_kv_heads=1, use_fp8_kv_cache=False),
            "attention.vllm_xpu_gqa_ratio_unsupported",
            id="xpu-gqa-ratio",
        ),
        pytest.param(
            ATTENTION_CONTEXT_SCHEMA,
            _plan_context(
                "attention_context",
                "vllm_xpu",
                "context_attention_perf.txt",
                sm_version=0,
            ),
            _semantic_attention_case(
                head_dim=128,
                window_size=128,
                use_fp8_kv_cache=False,
            ),
            "attention.vllm_xpu_swa_head_dim_unsupported",
            id="xpu-swa-head-dim",
        ),
    ],
)
def test_exact_mapping_reports_backend_hardware_and_version_limits(schema, context, case, reason):
    result = _compile_exact_case(schema, context, case)

    assert result.report.scheduled == 0
    assert result.report.unreachable == 0
    assert result.report.unsupported == 1
    assert result.report.decisions[0].reason == reason


_CURRENT_BUILDERS = {
    "attention_context": (
        get_attention_context_shape_sweeps,
        build_attention_context_cases,
        ATTENTION_CONTEXT_SCHEMA,
        "context_attention_perf.txt",
    ),
    "attention_generation": (
        get_attention_generation_shape_sweeps,
        build_attention_generation_cases,
        ATTENTION_GENERATION_SCHEMA,
        "generation_attention_perf.txt",
    ),
    "encoder_attention": (
        get_attention_encoder_shape_sweeps,
        build_encoder_attention_cases,
        ENCODER_ATTENTION_SCHEMA,
        "encoder_attention_perf.txt",
    ),
}


@pytest.mark.parametrize(
    ("backend", "op", "sm_version", "framework_version"),
    [
        pytest.param(backend, op, sm_version, framework_version, id=f"{backend}-{op}")
        for backend, sm_version, framework_version, ops in (
            ("sglang", 100, "0.5.10", ("attention_context", "attention_generation")),
            ("trtllm", 100, "1.3.0rc10", ("attention_context", "attention_generation")),
            ("vllm", 100, "0.19.0", ("attention_context", "attention_generation")),
            ("vllm_xpu", 0, "0.19.0", ("attention_context", "attention_generation")),
            ("sglang", 100, "0.5.11", ("encoder_attention",)),
            ("trtllm", 100, "1.3.0rc5", ("encoder_attention",)),
            ("vllm", 100, "0.21.0", ("encoder_attention",)),
        )
        for op in ops
    ],
)
def test_current_builder_cases_remain_scheduled_after_schema_guards(
    monkeypatch,
    backend,
    op,
    sm_version,
    framework_version,
):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    get_sweeps, build_cases, schema, perf_file = _CURRENT_BUILDERS[op]
    with use_case_catalog(None):
        cases = build_cases(
            backend,
            get_sweeps(backend),
            sm_version=sm_version,
            framework_version=framework_version,
        )
    context = _plan_context(
        op,
        backend,
        perf_file,
        sm_version=sm_version,
        framework_version=framework_version,
    )

    result = compile_population(
        [legacy_rule(cases)],
        context,
        schema=schema,
        record_decisions=False,
    )

    assert result.report.expanded == len(cases)
    assert result.report.scheduled == len(cases)
    assert result.report.duplicate_invocations == 0
    assert result.report.unreachable == 0
    assert result.report.unsupported == 0
