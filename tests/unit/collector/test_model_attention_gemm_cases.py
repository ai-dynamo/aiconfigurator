# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for per-model GEMM/attention shape injection in ``case_generator``.

These cover the lean/additive feature: model files declare
``model_case_values.{gemm,attention_context,attention_generation,attention_encoder}``
blocks that are appended to the model-agnostic base grid (filtered by
``COLLECTOR_MODEL_PATH``), with attention entries overlaying the per-backend base
template so caps / precision_cases / window_sizes are inherited. The tests are
GPU-free: they exercise ``collector.case_generator`` directly and inject
synthetic model data by monkeypatching the model-cases loader.
"""

import pytest

from collector import case_generator
from collector.case_generator import GemmCommonTestCase

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_model_path(monkeypatch):
    """Do not inherit COLLECTOR_MODEL_PATH from the env."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)


@pytest.fixture
def gemm_base_only(monkeypatch) -> list[GemmCommonTestCase]:
    """Base-only GEMM specs (no model data), for additive/prefix assertions."""
    with monkeypatch.context() as m:
        m.setattr(case_generator, "_load_model_cases_data", lambda: ())
        return case_generator.get_gemm_case_specs("vllm")


def _set_model_cases(monkeypatch, *model_dicts: dict) -> None:
    """Replace the model-cases loader with synthetic data and clear the filter."""
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(case_generator, "_load_model_cases_data", lambda: tuple(model_dicts))


def _model_file(model_path: str, **model_case_values: object) -> dict:
    return {
        "architecture": "SyntheticArch",
        "model_path": model_path,
        "model_case_values": dict(model_case_values),
    }


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------


def test_gemm_model_shapes_are_appended_and_inherit_token_counts(monkeypatch, gemm_base_only):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", gemm=[{"output_feature_sizes": [99], "input_feature_sizes": [77]}]),
    )

    with_model = case_generator.get_gemm_case_specs("vllm")

    # Base grid is preserved verbatim as a prefix; model shapes are appended.
    assert with_model[: len(gemm_base_only)] == gemm_base_only
    assert len(with_model) > len(gemm_base_only)

    # token_counts inherited from base -> every base M value emitted for (99, 77).
    base_x = {case.x for case in gemm_base_only}
    model_x = {case.x for case in with_model if case.n == 99 and case.k == 77}
    assert model_x == base_x
    assert GemmCommonTestCase(x=1, n=99, k=77) in with_model


def test_gemm_model_explicit_token_counts_override(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            gemm=[{"output_feature_sizes": [99], "input_feature_sizes": [77], "token_counts": [3, 5]}],
        ),
    )

    cases = case_generator.get_gemm_case_specs("vllm")
    model_x = {case.x for case in cases if case.n == 99 and case.k == 77}
    assert model_x == {3, 5}


def test_gemm_model_feature_sizes_shorthand(monkeypatch):
    _set_model_cases(monkeypatch, _model_file("org/M", gemm=[{"feature_sizes": [99]}]))

    cases = case_generator.get_gemm_case_specs("vllm")
    assert any(case.n == 99 and case.k == 99 for case in cases)


def test_gemm_model_shapes_reach_all_backends(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", gemm=[{"output_feature_sizes": [99], "input_feature_sizes": [77]}]),
    )

    for backend in (None, "trtllm", "vllm", "sglang", "vllm_xpu"):
        cases = case_generator.get_gemm_case_specs(backend)
        assert any(case.n == 99 and case.k == 77 for case in cases), backend


def test_gemm_model_shapes_filtered_by_collector_model_path(monkeypatch):
    monkeypatch.setattr(
        case_generator,
        "_load_model_cases_data",
        lambda: (
            _model_file(
                "org/A", gemm=[{"model_path": "org/A", "output_feature_sizes": [99], "input_feature_sizes": [77]}]
            ),
            _model_file(
                "org/B", gemm=[{"model_path": "org/B", "output_feature_sizes": [55], "input_feature_sizes": [33]}]
            ),
        ),
    )
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "org/A")

    cases = case_generator.get_gemm_case_specs("vllm")
    assert any(case.n == 99 and case.k == 77 for case in cases)
    assert not any(case.n == 55 and case.k == 33 for case in cases)


def test_gemm_model_shapes_dedupe_against_base(monkeypatch, gemm_base_only):
    # 4096 x 7168 are both on the base feature grid, so the model entry is fully
    # redundant and must not add duplicate (x, n, k) cases.
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", gemm=[{"output_feature_sizes": [4096], "input_feature_sizes": [7168]}]),
    )
    with_model = case_generator.get_gemm_case_specs("vllm")
    assert with_model == gemm_base_only


def test_gemm_model_paths_plural_expansion(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/primary",
            gemm=[{"model_paths": ["org/a", "org/b"], "output_feature_sizes": [99], "input_feature_sizes": [77]}],
        ),
    )
    # No filter -> both aliases expand to the same sweep; dedupe means each
    # (x, n, k) for (99, 77) appears exactly once (no per-alias duplication).
    cases = case_generator.get_gemm_case_specs("vllm")
    matching = [(case.x, case.n, case.k) for case in cases if case.n == 99 and case.k == 77]
    assert matching, "expected the model (99, 77) shape to be present"
    assert len(matching) == len(set(matching)), "model shape duplicated across aliases (dedupe failed)"

    # Filter to one alias -> still present.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "org/b")
    cases_b = case_generator.get_gemm_case_specs("vllm")
    assert any(case.n == 99 and case.k == 77 for case in cases_b)


# ---------------------------------------------------------------------------
# Attention context
# ---------------------------------------------------------------------------


def test_attention_context_model_sweep_inherits_caps_and_window(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_context=[
                {
                    "model_path": "org/M",
                    "query_head_counts": [64],
                    "kv_head_options": [8],
                    "head_dims": [192],
                    "sequence_lengths": [1024, 8192],
                }
            ],
        ),
    )

    sweeps = case_generator.get_attention_context_model_sweeps("vllm")
    assert len(sweeps) == 1
    sweep = sweeps[0]
    base = case_generator.get_merged_base_op_case_specs("vllm", "attention_context")[0]

    assert sweep["id"].startswith("model_")
    assert sweep["query_head_counts"] == [64]
    assert sweep["head_dims"] == [192]
    # Caps and window_sizes (vllm-required) inherited from the base template.
    assert sweep["max_tokens_self_attention"] == base["max_tokens_self_attention"]
    assert sweep["window_sizes"] == base["window_sizes"]
    # Metadata (entry model_path + file-injected architecture) is stripped, not leaked.
    assert "model_path" not in sweep
    assert "architecture" not in sweep


@pytest.mark.parametrize("backend", ["trtllm", "sglang"])
def test_attention_context_model_sweep_carries_precision_cases(monkeypatch, backend):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", attention_context=[{"query_head_counts": [64], "head_dims": [192]}]),
    )

    sweep = case_generator.get_attention_context_model_sweeps(backend)[0]
    base = case_generator.get_merged_base_op_case_specs(backend, "attention_context")[0]
    assert sweep["precision_cases"] == base["precision_cases"]


def test_attention_context_cap_override(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_context=[{"query_head_counts": [64], "head_dims": [192], "max_tokens_self_attention": 999}],
        ),
    )
    sweep = case_generator.get_attention_context_model_sweeps("vllm")[0]
    assert sweep["max_tokens_self_attention"] == 999


def test_attention_context_bf16_only_precision_override(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_context=[
                {
                    "query_head_counts": [64],
                    "head_dims": [192],
                    "precision_cases": [{"id": "bf16", "fp8_kv_cache": False, "fp8_context_fmha": False}],
                }
            ],
        ),
    )
    sweep = case_generator.get_attention_context_model_sweeps("trtllm")[0]
    assert [pc["fp8_kv_cache"] for pc in sweep["precision_cases"]] == [False]


# ---------------------------------------------------------------------------
# Attention generation
# ---------------------------------------------------------------------------


def test_attention_generation_generic_query_heads_fan_out(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_generation=[{"query_head_counts": [64], "kv_head_counts": [8], "head_dims": [192]}],
        ),
    )
    sweep = case_generator.get_attention_generation_model_sweeps("trtllm")[0]
    assert sweep["mha_query_head_counts"] == [64]
    assert sweep["xqa_query_head_counts"] == [64]
    assert sweep["kv_head_counts"] == [8]
    assert "query_head_counts" not in sweep


def test_attention_generation_explicit_xqa_overrides_generic(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_generation=[{"query_head_counts": [64], "xqa_query_head_counts": [8], "kv_head_counts": [8]}],
        ),
    )
    sweep = case_generator.get_attention_generation_model_sweeps("trtllm")[0]
    assert sweep["mha_query_head_counts"] == [64]
    assert sweep["xqa_query_head_counts"] == [8]


# ---------------------------------------------------------------------------
# Attention encoder
# ---------------------------------------------------------------------------


def test_attention_encoder_maps_to_head_counts(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", attention_encoder=[{"query_head_counts": [16], "head_dims": [80]}]),
    )
    sweep = case_generator.get_attention_encoder_model_sweeps("trtllm")[0]
    assert sweep["head_counts"] == [16]
    assert sweep["head_dims"] == [80]
    assert "query_head_counts" not in sweep


def test_normalize_rejects_unknown_op_name():
    with pytest.raises(ValueError, match="unsupported attention op_name"):
        case_generator._normalize_attention_model_entry({"query_head_counts": [8]}, op_name="bogus")


# ---------------------------------------------------------------------------
# Dedupe
# ---------------------------------------------------------------------------


def test_attention_model_sweeps_dedupe_identical_entries(monkeypatch):
    entry = {"query_head_counts": [64], "head_dims": [192]}
    monkeypatch.setattr(
        case_generator,
        "_load_model_cases_data",
        lambda: (
            _model_file("org/A", attention_context=[dict(entry)]),
            _model_file("org/B", attention_context=[dict(entry)]),
        ),
    )
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    sweeps = case_generator.get_attention_context_model_sweeps("vllm")
    assert len(sweeps) == 1


def test_attention_context_model_sweeps_filtered_by_collector_model_path(monkeypatch):
    monkeypatch.setattr(
        case_generator,
        "_load_model_cases_data",
        lambda: (
            _model_file(
                "org/A", attention_context=[{"model_path": "org/A", "query_head_counts": [64], "head_dims": [192]}]
            ),
            _model_file(
                "org/B", attention_context=[{"model_path": "org/B", "query_head_counts": [40], "head_dims": [128]}]
            ),
        ),
    )
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "org/A")

    sweeps = case_generator.get_attention_context_model_sweeps("vllm")
    assert len(sweeps) == 1
    assert sweeps[0]["query_head_counts"] == [64]


def test_attention_overlay_identical_to_base_is_dropped(monkeypatch):
    # An overlay with no shape fields equals the base sweep, which already covers
    # those shapes, so it must not be emitted as a redundant duplicate.
    _set_model_cases(monkeypatch, _model_file("org/M", attention_context=[{"model_path": "org/M"}]))
    assert case_generator.get_attention_context_model_sweeps("vllm") == []


# ---------------------------------------------------------------------------
# Single-base-template invariant (guard against >1 base sweep)
# ---------------------------------------------------------------------------


def test_attention_model_sweeps_reject_multiple_base_specs(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", attention_context=[{"query_head_counts": [64], "head_dims": [192]}]),
    )
    monkeypatch.setattr(
        case_generator,
        "get_merged_base_op_case_specs",
        lambda backend, op_name: [{"id": "a"}, {"id": "b"}],
    )
    with pytest.raises(RuntimeError, match="exactly one base"):
        case_generator.get_attention_context_model_sweeps("vllm")


def test_gemm_model_sweeps_reject_multiple_base_specs(monkeypatch):
    _set_model_cases(
        monkeypatch,
        _model_file("org/M", gemm=[{"output_feature_sizes": [99], "input_feature_sizes": [77]}]),
    )
    monkeypatch.setattr(
        case_generator,
        "_get_base_gemm_shape_sweeps",
        lambda backend=None: [{"token_counts": [1]}, {"token_counts": [2]}],
    )
    with pytest.raises(RuntimeError, match="exactly one base"):
        case_generator._gemm_model_shape_sweeps("vllm")


# ---------------------------------------------------------------------------
# Backward-compat regression
# ---------------------------------------------------------------------------


def test_no_model_data_yields_base_only(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    monkeypatch.setattr(case_generator, "_load_model_cases_data", lambda: ())

    assert case_generator._gemm_model_shape_sweeps("vllm") == []
    assert case_generator.get_attention_context_model_sweeps("vllm") == []
    assert case_generator.get_attention_generation_model_sweeps("vllm") == []
    assert case_generator.get_attention_encoder_model_sweeps("vllm") == []

    context = case_generator.get_attention_context_shape_sweeps("vllm")
    assert context == case_generator.get_merged_base_op_case_specs("vllm", "attention_context")


def test_attention_base_sweeps_preserved_when_model_present(monkeypatch):
    # Generic base sweeps must survive unchanged (as a prefix) when a model adds
    # its own shapes, for every attention phase.
    bases = {
        op: case_generator.get_merged_base_op_case_specs("vllm", op)
        for op in ("attention_context", "attention_generation", "attention_encoder")
    }
    _set_model_cases(
        monkeypatch,
        _model_file(
            "org/M",
            attention_context=[{"query_head_counts": [64], "head_dims": [192]}],
            attention_generation=[{"query_head_counts": [64], "kv_head_counts": [8], "head_dims": [192]}],
            attention_encoder=[{"query_head_counts": [16], "head_dims": [80]}],
        ),
    )
    combined = {
        "attention_context": case_generator.get_attention_context_shape_sweeps("vllm"),
        "attention_generation": case_generator.get_attention_generation_shape_sweeps("vllm"),
        "attention_encoder": case_generator.get_attention_encoder_shape_sweeps("vllm"),
    }
    for op, base in bases.items():
        assert combined[op][: len(base)] == base, op
        assert len(combined[op]) == len(base) + 1, op
        assert combined[op][-1]["id"].startswith("model_"), op
