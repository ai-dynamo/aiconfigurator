# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TRT-LLM dense-attention per-model backend axis (trtllm_attn_backend).

Mirrors serving's backend selection: TorchLlmArgs.attn_backend defaults to
"TRTLLM" (llmapi/llm_args.py:4544-4546@1.3.0rc20) and model classes may
override it via get_model_defaults (Gemma4 forces "FLASHINFER",
models/modeling_gemma4.py:942-952). Pure population tests — no framework
import, so the module stays runnable on CUDA-free CI hosts (same approach as
test_sglang_attention_0514.py).
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_collector_functions(names, namespace):
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in set(names)]
    assert {function.name for function in functions} == set(names)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"), loaded)
    return [loaded[name] for name in names]


def _profiles(*profiles):
    return {"head_profiles": list(profiles)}


def _profile(**overrides):
    profile = {
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 512,
        "window_size": 0,
        "tensor_parallel_sizes": [1],
    }
    profile.update(overrides)
    return profile


def test_trtllm_backend_defaults_to_the_serving_default(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(_profiles(_profile()), phase="context", backend="trtllm")

    assert [config.kernel_source for config in configs] == ["TRTLLM"]


def test_trtllm_backend_override_is_recorded(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile(trtllm_attn_backend="FLASHINFER")),
        phase="context",
        backend="trtllm",
    )

    assert [config.kernel_source for config in configs] == ["FLASHINFER"]


def test_trtllm_backend_dedup_keeps_distinct_backends_apart(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile(), _profile(trtllm_attn_backend="FLASHINFER")),
        phase="context",
        backend="trtllm",
    )

    assert [config.kernel_source for config in configs] == ["TRTLLM", "FLASHINFER"]


def test_unknown_trtllm_backend_fails_loudly(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    with pytest.raises(ValueError, match=r"Unsupported trtllm_attn_backend"):
        get_attention_head_configs(
            _profiles(_profile(trtllm_attn_backend="TRITON")),
            phase="context",
            backend="trtllm",
        )


def test_framework_neutral_callers_see_no_backend(monkeypatch):
    # Collectors that do not pass a backend keep the historical population
    # contract: kernel_source stays None.
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    configs = get_attention_head_configs(
        _profiles(_profile(trtllm_attn_backend="FLASHINFER")),
        phase="context",
    )

    assert [config.kernel_source for config in configs] == [None]


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_gemma4_profiles_route_to_flashinfer(monkeypatch, phase):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "google/gemma-4-26B-A4B")
    sweeps = (
        get_attention_context_shape_sweeps("trtllm")
        if phase == "context"
        else get_attention_generation_shape_sweeps("trtllm")
    )
    configs = [
        config for sweep in sweeps for config in get_attention_head_configs(sweep, phase=phase, backend="trtllm")
    ]

    assert configs, "Gemma4 attention profiles must populate"
    assert {config.kernel_source for config in configs} == {"FLASHINFER"}
    assert {config.head_dim for config in configs} == {256, 512}


def test_flashinfer_path_pins_the_trtllm_gen_sub_backend():
    # create_attention only selects the backend CLASS; serving additionally
    # pins the flashinfer sub-backend to "trtllm-gen" for every FLASHINFER
    # layer (Gemma4, models/modeling_gemma4.py:263-270@1.3.0rc20). Without the
    # pin FlashInferAttention defaults to "fa2" (flashinfer.py:1372), a kernel
    # serving never runs — so run_attention_torch must set it. AST-only check
    # keeps this runnable on CUDA-free CI (run_attention_torch cannot be exec'd
    # without the framework).
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch"
    )
    pins = [
        node
        for node in ast.walk(run_fn)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "flashinfer_backend"
        and isinstance(node.value, ast.Constant)
    ]
    assert [pin.value.value for pin in pins] == ["trtllm-gen"], (
        "run_attention_torch must pin the flashinfer sub-backend to serving's trtllm-gen exactly once"
    )


def test_flashinfer_path_fails_closed_on_non_blackwell():
    # trtllm-gen FMHA (the sub-backend Gemma4/flashinfer pins) hard-restricts to
    # Blackwell: TllmGenFmhaRunner asserts mSM == kSM_100 || kSM_103
    # ("Unsupported architecture", fmhaRunner.cuh:37@1.3.0rc20). run_attention_torch
    # must therefore raise a classified skip for the FLASHINFER path on any SM other
    # than 100/103, instead of building the op and eating a CUDA-level abort. AST-only
    # check (run_attention_torch cannot be exec'd on CUDA-free CI).
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch"
    )
    guards = [
        node
        for node in ast.walk(run_fn)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Name) and n.id == "is_flashinfer" for n in ast.walk(node.test))
        and any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "get_sm_version"
            for n in ast.walk(node.test)
        )
        and {c.value for c in ast.walk(node.test) if isinstance(c, ast.Constant)} >= {100, 103}
        and any(isinstance(n, ast.Raise) for n in ast.walk(node))
    ]
    assert len(guards) == 1, (
        "run_attention_torch must fail closed (raise) for the FLASHINFER path when "
        "get_sm_version() is not a Blackwell arch (100/103)"
    )


def test_dense_attention_uses_full_kv_cache_for_every_backend():
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch"
    )
    assignments = [
        node
        for node in ast.walk(run_fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "kv_cache_type" for target in node.targets)
    ]

    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.Attribute)
    assert value.attr == "SELF"


def _case_functions():
    namespace = {
        "os": __import__("os"),
        "tensorrt_llm": SimpleNamespace(__version__="1.3.0rc20"),
        "get_sm_version": lambda: 120,
        "get_attention_context_shape_sweeps": get_attention_context_shape_sweeps,
        "get_attention_generation_shape_sweeps": get_attention_generation_shape_sweeps,
        "get_attention_head_configs": get_attention_head_configs,
    }
    return _load_collector_functions(
        [
            "_int_list",
            "_skip_trtllm_sm120_fp8_context_fmha",
            "_skip_trtllm_sm89_rc15_long_context_gqa",
            "_skip_trtllm_sm89_rc15_fp8_context_mha",
            "_generation_target_sequence_lengths",
            "get_context_attention_test_cases",
            "get_generation_attention_test_cases",
        ],
        namespace,
    )


def test_backend_element_reaches_the_trtllm_case_tuples(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "google/gemma-4-26B-A4B")
    (
        _int_list,
        skip_sm120,
        skip_sm89_gqa,
        skip_sm89_fp8,
        gen_targets,
        get_context_cases,
        get_generation_cases,
    ) = _case_functions()
    loaded = get_context_cases.__globals__
    loaded["_int_list"] = _int_list
    loaded["_skip_trtllm_sm120_fp8_context_fmha"] = skip_sm120
    loaded["_skip_trtllm_sm89_rc15_long_context_gqa"] = skip_sm89_gqa
    loaded["_skip_trtllm_sm89_rc15_fp8_context_mha"] = skip_sm89_fp8
    loaded["_generation_target_sequence_lengths"] = gen_targets

    context_cases = get_context_cases()
    generation_cases = get_generation_cases()

    assert context_cases and generation_cases
    for case in [*context_cases, *generation_cases]:
        assert len(case) == 10
        assert case[9] == "FLASHINFER"
    # fp8 FMHA combinations stay generated for flashinfer-routed configs and
    # fail closed at runtime (classified) — population never silently drops
    # them.
    assert any(case[7] for case in context_cases)


def test_out_scale_mirrors_serving_use_quantize_output():
    # out_scale must mirror serving's Attention._use_quantize_output()
    # (_torch/modules/attention.py:648-670,758-761@1.3.0rc20): only a quantized
    # model supplies o_proj.inv_input_scale. Passing a scale is not inert — the
    # backend allocates fp8 attention output whenever out_scale is present
    # (is_quantize_output, attention_backend/trtllm.py:1452), so an
    # unconditional fp8-KV scale would measure the fp8-output path under an
    # attn_dtype=bfloat16 label on SMs where serving keeps bf16 output (any
    # non-SM90 context). The sanctioned condition:
    # use_fp8_context_fmha (the fp8-model context case) OR fp8-KV generation
    # (the sweep's only fp8-model generation flavor) — no SM sniffing. AST-only
    # check (run_attention_torch cannot be exec'd on CUDA-free CI).
    source_path = REPO_ROOT / "collector" / "trtllm" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch"
    )
    guards = [
        node
        for node in ast.walk(run_fn)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "out_scale" for t in n.targets)
            for n in ast.walk(node)
        )
    ]
    assert len(guards) == 1, "expected exactly one conditional out_scale assignment"
    test_names = {n.id for n in ast.walk(guards[0].test) if isinstance(n, ast.Name)}
    assert test_names == {"use_fp8_context_fmha", "use_fp8_kv_cache", "is_context_phase"}, (
        "out_scale must be given exactly for use_fp8_context_fmha or fp8-KV "
        f"generation cases, got condition over {sorted(test_names)}"
    )
    assert not any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == "get_sm_version"
        for n in ast.walk(guards[0])
    ), "out_scale must not be SM-conditioned; SM90 behavior is the framework's own"
