# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_dsv4_attn.py"


def test_dsv4_worker_fails_closed_when_phase_has_no_valid_shapes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_subprocess_entry"
    )
    namespace = {
        "Iterable": Iterable,
        "_expand_grid": lambda: ([], [1]),
        "_filter_pairs": lambda *_args: [],
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    with pytest.raises(RuntimeError, match=r"no valid prefix/sl values; ok=0 error=0 skip=0 total=0"):
        namespace["_subprocess_entry"](
            mode="generation",
            attn_kind="csa",
            model_path="sgl-project/DeepSeek-V4-Flash-FP8",
            kv_cache_dtype="fp8",
            batch_size=1,
            output_path="unused",
        )


def test_dsv4_generation_enters_sglang_model_capture_mode():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_dsv4_mla_module"
    )
    function_source = ast.get_source_segment(source, function)

    assert "model_capture_mode() if not is_prefill" in function_source


def test_dsv4_context_memory_filter_binds_exact_inner_manifest(capsys):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_attach_dsv4_context_memory_manifest"
    )

    memory_by_device = [1000, 2000]
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: len(memory_by_device),
            get_device_properties=lambda device: SimpleNamespace(total_memory=memory_by_device[device]),
        )
    )
    model_config = {"hidden_size": 10, "max_position_embeddings": 100}
    namespace = {
        "torch": fake_torch,
        "sys": SimpleNamespace(argv=["pytest"]),
        "_SEQ_LENGTHS": [10, 11],
        "_PREFIX_LENGTHS": [0, 5],
        "_filter_pairs": lambda _mode, batch_sizes, seq_lens: [
            (batch_size, seq_len) for batch_size in batch_sizes for seq_len in seq_lens
        ],
        "_dsv4_context_structural_manifest": lambda _bs, seq_lens, prefix_lens, max_pos: tuple(
            (prefix_len, tuple(seq_len for seq_len in seq_lens if prefix_len + seq_len <= max_pos))
            for prefix_len in prefix_lens
        ),
        "_load_dsv4_model_config": lambda _model_path: model_config,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    case = [0, 4, 1, "fp8", "bfloat16", "bfloat16", "deepseek-ai/DeepSeek-V4-Pro", "csa", None]
    filtered = namespace["_attach_dsv4_context_memory_manifest"]([case], "csa")

    # The smallest live device supplies an 80% budget: 1000 * 4 / 5 = 800.
    # seq=10 consumes exactly 4 * 10 * 10 * 2 = 800 bytes and is retained;
    # seq=11 exceeds the budget and is absent from both prefix groups.
    assert filtered == [[*case, ((0, (10,)), (5, (10,)))]]
    output = capsys.readouterr().out
    assert "structurally_admitted=4/4" in output
    assert "memory_dropped=2/4" in output
    assert "budget_bytes=800" in output
    assert "batch_size*sequence_length*hidden_size*2" in output

    memory_by_device[:] = [2000, 2000]
    larger = namespace["_attach_dsv4_context_memory_manifest"]([case], "csa")
    assert larger == [[*case, ((0, (11, 10)), (5, (11, 10)))]]
    assert str(larger[0]) != str(filtered[0])

    memory_by_device[:] = [150_110_011_392] * 8
    model_config.update(hidden_size=7168, max_position_embeddings=1_048_576)
    namespace["_SEQ_LENGTHS"] = [6144, 1_048_575]
    namespace["_PREFIX_LENGTHS"] = [0]
    extreme = [*case]
    extreme[1] = 1024
    pro_filtered = namespace["_attach_dsv4_context_memory_manifest"]([extreme], "csa")
    assert pro_filtered == [[*extreme, ((0, (6144,)),)]]

    namespace["sys"].argv = ["collect.py", "--smoke"]
    memory_by_device[:] = [2000]
    model_config.update(hidden_size=1, max_position_embeddings=4096)
    namespace["_SEQ_LENGTHS"] = [1, 128, 1024]
    namespace["_PREFIX_LENGTHS"] = [0, 512, 1024]
    smoke_case = [*case]
    smoke_case[1] = 1
    smoke_filtered = namespace["_attach_dsv4_context_memory_manifest"]([smoke_case], "csa")
    assert smoke_filtered == [[*smoke_case, ((0, (128, 1)), (512, (128, 1)))]]


def test_dsv4_subprocess_consumes_getter_manifest_without_rebuilding_grid():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_subprocess_entry"
    )
    captured = {}

    def _run_dsv4_mla_module(**kwargs):
        captured.update(kwargs)

    namespace = {
        "Iterable": Iterable,
        "_expand_grid": lambda: pytest.fail("getter manifest must replace worker-side grid expansion"),
        "_filter_pairs": lambda *_args: pytest.fail("getter manifest must replace worker-side shape filtering"),
        "_dsv4_max_position_embeddings": lambda *_args: pytest.fail(
            "getter manifest must already contain model-position filtering"
        ),
        "run_dsv4_mla_module": _run_dsv4_mla_module,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)

    manifest = ((1024, (1024,)),)
    namespace["_subprocess_entry"](
        mode="context",
        attn_kind="csa",
        model_path="sgl-project/DeepSeek-V4-Flash-FP8",
        kv_cache_dtype="fp8",
        batch_size=1,
        output_path="unused",
        inner_shapes=manifest,
        smoke=True,
    )

    assert captured["prefix_lens"] == (1024,)
    assert captured["seq_lens_by_prefix"] == {1024: [1024]}


def test_dsv4_generation_getters_do_not_apply_context_hidden_state_filter():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "get_dsv4_csa_context_test_cases",
            "get_dsv4_csa_generation_test_cases",
            "get_dsv4_hca_context_test_cases",
            "get_dsv4_hca_generation_test_cases",
        }
    }

    assert "_attach_dsv4_context_memory_manifest" in functions["get_dsv4_csa_context_test_cases"]
    assert "_attach_dsv4_context_memory_manifest" in functions["get_dsv4_hca_context_test_cases"]
    assert "_attach_dsv4_context_memory_manifest" not in functions["get_dsv4_csa_generation_test_cases"]
    assert "_attach_dsv4_context_memory_manifest" not in functions["get_dsv4_hca_generation_test_cases"]
