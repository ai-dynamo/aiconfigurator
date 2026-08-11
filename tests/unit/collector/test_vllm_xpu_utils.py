# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_COLLECTOR_ROOT = REPO_ROOT / "collector" / "vllm"


def test_xpu_collectors_use_the_legacy_utility_module():
    attention_source = (VLLM_COLLECTOR_ROOT / "collect_attn_xpu.py").read_text()
    gemm_source = (VLLM_COLLECTOR_ROOT / "collect_gemm_xpu.py").read_text()
    moe_source = (VLLM_COLLECTOR_ROOT / "collect_moe_xpu.py").read_text()

    assert "from collector.vllm.utils_xpu import" in attention_source
    assert "from collector.vllm.utils_xpu import" in gemm_source
    assert "from collector.vllm.utils import" not in attention_source
    assert "from collector.vllm.utils import" not in gemm_source
    # Match the import form precisely: "collector.vllm.utils" is a substring
    # of "collector.vllm.utils_xpu", so a bare substring check would reject a
    # correct utils_xpu import.
    assert "from collector.vllm.utils import" not in moe_source


def test_cuda_collectors_do_not_import_xpu_utilities():
    for collector_path in VLLM_COLLECTOR_ROOT.glob("collect_*.py"):
        if collector_path.stem.endswith("_xpu"):
            continue
        assert "collector.vllm.utils_xpu" not in collector_path.read_text()


def test_xpu_utility_module_defines_required_exports():
    tree = ast.parse((VLLM_COLLECTOR_ROOT / "utils_xpu.py").read_text())
    exports = {
        node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {
        "BatchSpec",
        "create_and_prepopulate_kv_cache",
        "create_common_attn_metadata",
        "create_standard_kv_cache_spec",
        "create_vllm_config",
        "get_attention_backend",
        "setup_distributed",
        "with_exit_stack",
    } <= exports


def _load_xpu_attention_function(name: str):
    tree = ast.parse((VLLM_COLLECTOR_ROOT / "collect_attn_xpu.py").read_text())
    needed = {"_apply_kv_cache_stride_order", "_pack_v1_fa_kv_cache", name}
    fn_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in needed]
    namespace = {"torch": pytest.importorskip("torch")}
    exec(
        compile(ast.Module(body=fn_nodes, type_ignores=[]), str(VLLM_COLLECTOR_ROOT / "collect_attn_xpu.py"), "exec"),
        namespace,
    )
    return namespace[name]


def _load_xpu_attention_function_without_torch(name: str):
    tree = ast.parse((VLLM_COLLECTOR_ROOT / "collect_attn_xpu.py").read_text())
    fn_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace: dict = {}
    exec(
        compile(ast.Module(body=[fn_node], type_ignores=[]), str(VLLM_COLLECTOR_ROOT / "collect_attn_xpu.py"), "exec"),
        namespace,
    )
    return namespace[name]


def test_xpu_attention_split_cache_update_runs_before_forward():
    forward = _load_xpu_attention_function_without_torch("_forward_with_optional_kv_cache_update")
    events = []
    layer, query, key, value, kv_cache, output = (object() for _ in range(6))

    class Backend:
        forward_includes_kv_cache_update = False

    class Impl:
        def do_kv_cache_update(self, got_layer, got_key, got_value, got_cache, got_slots):
            events.append(("update", got_layer, got_key, got_value, got_cache, got_slots))

        def forward(self, got_layer, got_query, got_key, got_value, got_cache, got_metadata, *, output):
            events.append(("forward", got_layer, got_query, got_key, got_value, got_cache, got_metadata, output))
            return "done"

    metadata = type("Metadata", (), {"slot_mapping": object()})()

    assert forward(Backend, Impl(), layer, query, key, value, kv_cache, metadata, output) == "done"
    assert events == [
        ("update", layer, key, value, kv_cache, metadata.slot_mapping),
        ("forward", layer, query, key, value, kv_cache, metadata, output),
    ]


def test_xpu_attention_skips_explicit_update_when_backend_forward_includes_it():
    forward = _load_xpu_attention_function_without_torch("_forward_with_optional_kv_cache_update")
    events = []

    class Backend:
        forward_includes_kv_cache_update = True

    class Impl:
        def do_kv_cache_update(self, *_args):
            events.append("update")

        def forward(self, *_args, output):
            events.append("forward")

    metadata = type("Metadata", (), {"slot_mapping": object()})()

    forward(Backend, Impl(), object(), object(), object(), object(), object(), metadata, object())
    assert events == ["forward"]


def test_xpu_flash_attention_kv_cache_keeps_backend_stride_order():
    torch = pytest.importorskip("torch")
    pack_cache = _load_xpu_attention_function("_pack_v1_fa_kv_cache")

    num_blocks, block_size, num_kv_heads, head_size = 3, 5, 7, 11
    helper_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_size)

    nhd_cache = pack_cache(helper_cache, (0, 2, 1, 3))
    expected_nhd_stride = torch.empty(num_blocks, block_size, num_kv_heads, 2 * head_size).permute(0, 2, 1, 3).stride()
    assert nhd_cache.shape == (num_blocks, num_kv_heads, block_size, 2 * head_size)
    assert nhd_cache.stride() == expected_nhd_stride
    key_cache, value_cache = nhd_cache.transpose(1, 2).split(head_size, dim=-1)
    expected_kernel_stride = (
        torch.empty(num_blocks, block_size, num_kv_heads, 2 * head_size).split(head_size, dim=-1)[0].stride()
    )
    assert key_cache.shape == (num_blocks, block_size, num_kv_heads, head_size)
    assert value_cache.shape == (num_blocks, block_size, num_kv_heads, head_size)
    assert key_cache.stride() == expected_kernel_stride
    assert value_cache.stride() == expected_kernel_stride

    hnd_cache = pack_cache(helper_cache, (0, 1, 2, 3))
    expected_hnd_stride = torch.empty(num_blocks, num_kv_heads, block_size, 2 * head_size).stride()
    assert hnd_cache.shape == (num_blocks, num_kv_heads, block_size, 2 * head_size)
    assert hnd_cache.stride() == expected_hnd_stride


def _load_gemm_footprint_fn():
    """Extract _gemm_peak_footprint_bytes from the XPU gemm collector without
    importing the module (its top-level torch/vllm imports are unavailable in
    unit-test envs). The function is pure arithmetic over its args."""
    tree = ast.parse((VLLM_COLLECTOR_ROOT / "collect_gemm_xpu.py").read_text())
    fn_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_gemm_peak_footprint_bytes"
    )
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn_node], type_ignores=[]), "<gemm>", "exec"), namespace)
    return namespace["_gemm_peak_footprint_bytes"]


@pytest.mark.unit
def test_gemm_peak_footprint_accounts_for_all_tensors_and_fp8_transient():
    footprint = _load_gemm_footprint_fn()
    m, n, k = 8192, 65536, 4096
    input_bytes = m * k * 2
    per_copy = (n * k + m * n) * 2

    # bf16 baseline: input + weight + output.
    assert footprint("bfloat16", m, n, k) == input_bytes + per_copy
    # fp8 stages an int8 transient weight.
    assert footprint("fp8", m, n, k) == input_bytes + per_copy + n * k
    # fp8_block stages a float32 raw weight (4x fp8).
    assert footprint("fp8_block", m, n, k) == input_bytes + per_copy + n * k * 4
    assert footprint("fp8_block", m, n, k) > footprint("fp8", m, n, k) > footprint("bfloat16", m, n, k)

    # copies scales the per-op portion but not input or staging.
    assert footprint("bfloat16", m, n, k, copies=6) == input_bytes + per_copy * 6
    assert footprint("fp8_block", m, n, k, copies=6) == input_bytes + per_copy * 6 + n * k * 4

    # copies=0 returns the copy-independent portion (used by the loop-count cap).
    assert footprint("bfloat16", m, n, k, copies=0) == input_bytes
    assert footprint("fp8", m, n, k, copies=0) == input_bytes + n * k
    assert footprint("fp8_block", m, n, k, copies=0) == input_bytes + n * k * 4


@pytest.mark.unit
def test_gemm_collector_frees_memory_per_case():
    """Cross-case OOMs are only avoided if `run_gemm`'s try/finally actually
    frees op_list/x and calls empty_cache(). Assert the structure via AST so
    the guard survives refactors that keep the strings but move them out of
    the finally block (or out of run_gemm)."""
    tree = ast.parse((VLLM_COLLECTOR_ROOT / "collect_gemm_xpu.py").read_text())
    run_gemm = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_gemm")

    try_nodes = [node for node in ast.walk(run_gemm) if isinstance(node, ast.Try) and node.finalbody]
    assert try_nodes, "run_gemm must have a try/finally around the benchmark to free per-case tensors"

    finally_bodies = [stmt for tn in try_nodes for stmt in tn.finalbody]

    # `del op_list` and `del x` release the per-case tensors before the next
    # case allocates. Both are required for the loop cap to stay honest.
    deleted_names = {
        target.id
        for stmt in finally_bodies
        if isinstance(stmt, ast.Delete)
        for target in stmt.targets
        if isinstance(target, ast.Name)
    }
    assert {"op_list", "x"} <= deleted_names, (
        f"run_gemm's finally must `del op_list` and `del x`; got dels: {deleted_names}"
    )

    # empty_cache() must be called from within the finally (directly or under
    # a size-gated `if`), not merely referenced elsewhere in the module.
    empty_cache_calls = [
        node
        for stmt in finally_bodies
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "empty_cache"
    ]
    assert empty_cache_calls, "run_gemm's finally must call empty_cache() to purge the allocator between cases"
