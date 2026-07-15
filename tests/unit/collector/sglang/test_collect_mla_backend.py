# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pins the standalone-MLA collector's default-backend map to SGLang 0.5.14
serving selection (server_args._get_default_attn_backend, MLA branch) and the
fail-closed behaviour below the audited platform set {89, 90, 100, 103, 120}."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_collector_function(name, namespace):
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_mla.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), loaded)
    return loaded[name]


def _selector(sm_version, *, cuda_ok):
    return _load_collector_function(
        "_select_default_mla_backend",
        {
            "get_sm_version": lambda: sm_version,
            "_cuda_version_at_least": lambda *_a: cuda_ok,
        },
    )


@pytest.mark.unit
def test_sglang_mla_backend_map_follows_0514_serving_selection():
    # server_args.py:3641-3650 — DeepSeek V3/R1 trtllm_mla special-case is
    # gated on is_sm100_supported (major 10 + CUDA 12.8).
    assert _selector(100, cuda_ok=True)() == "trtllm_mla"
    assert _selector(103, cuda_ok=True)() == "trtllm_mla"
    # server_args.py:4457-4472 — Hopper takes fa3; everything else on the
    # audited CUDA platform set falls through to the final ``triton``.
    assert _selector(90, cuda_ok=True)() == "fa3"
    assert _selector(90, cuda_ok=False)() == "triton"
    assert _selector(89, cuda_ok=True)() == "triton"
    assert _selector(89, cuda_ok=False)() == "triton"
    assert _selector(120, cuda_ok=False)() == "triton"

    for unsupported_sm in (80, 86):
        with pytest.raises(ValueError, match=r"No SGLang 0\.5\.14 MLA backend mapping"):
            _selector(unsupported_sm, cuda_ok=True)()


@pytest.mark.unit
def test_sglang_mla_getters_fail_closed_instead_of_returning_silent_empty():
    # Below the audited set the selector raises; the getters must propagate
    # it. A silent [] would violate the zero-cases-need-logged-drops rule.
    def _raising_selector():
        raise ValueError("No SGLang 0.5.14 MLA backend mapping for SM80")

    for name in ("get_context_mla_test_cases", "get_generation_mla_test_cases"):
        getter = _load_collector_function(
            name,
            {
                "_select_default_mla_backend": _raising_selector,
                "torch": SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8"),
                "_build_mla_test_cases": lambda *_a, **_kw: [],
                "get_context_mla_case_specs": lambda: [],
                "get_generation_mla_case_specs": lambda: [],
            },
        )
        with pytest.raises(ValueError, match=r"No SGLang 0\.5\.14 MLA backend mapping"):
            getter()


@pytest.mark.unit
def test_sglang_mla_triton_platforms_store_bf16_kv_only():
    # SGLang's Triton MLA path stores BF16 MLA KV cache; fp8 KV belongs only
    # to the fa3/trtllm_mla platforms.
    captured = {}

    def _build(_specs, *, dtype_list, tp_sizes, backend=None):
        captured["dtype_list"] = list(dtype_list)
        captured["backend"] = backend
        captured["tp_sizes"] = tp_sizes
        return []

    for name in ("get_context_mla_test_cases", "get_generation_mla_test_cases"):
        getter = _load_collector_function(
            name,
            {
                "_select_default_mla_backend": lambda: "triton",
                "torch": SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8"),
                "_build_mla_test_cases": _build,
                "get_context_mla_case_specs": lambda: [],
                "get_generation_mla_case_specs": lambda: [],
            },
        )
        getter()
        assert captured["dtype_list"] == ["bf16"]
        assert captured["backend"] == "triton"


@pytest.mark.unit
def test_dsa_module_family_op_floors_cover_skip_indexer_variants():
    # The skip-indexer variants run the same SM90+ sparse attention kernel
    # family as the full DSA modules; the capability floor must cover all
    # four so sub-SM90 platforms drop them with a logged reason.
    from collector.capabilities import _load_capabilities

    _, op_min_sm = _load_capabilities()
    for op in (
        "dsa_context_module",
        "dsa_generation_module",
        "dsa_context_module_skip_indexer",
        "dsa_generation_module_skip_indexer",
    ):
        assert op_min_sm.get(op) == 90
