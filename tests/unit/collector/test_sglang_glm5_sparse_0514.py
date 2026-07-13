# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py"


def _load_pure_helpers(*names):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_glm5_topk_metadata_uses_causal_lengths_and_full_kv_offsets():
    helper = _load_pure_helpers("_glm5_topk_metadata")["_glm5_topk_metadata"]

    lengths, offsets, max_seqlen_k = helper(bs=2, isl=3, past_kv=4)

    assert lengths == [5, 6, 7, 5, 6, 7]
    assert offsets == [0, 0, 0, 7, 7, 7]
    assert max_seqlen_k == 7


def test_glm5_decode_metadata_keeps_full_kv_page_table_width():
    helper = _load_pure_helpers("_glm5_topk_metadata")["_glm5_topk_metadata"]

    lengths, offsets, max_seqlen_k = helper(bs=2, isl=1, past_kv=4095)

    assert lengths == [4096, 4096]
    assert offsets == [0, 4096]
    assert max_seqlen_k == 4096

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "torch.arange(max_seqlen_k" in source
    assert "pt.view(1, max_seqlen_k)" in source


def test_glm5_flash_mla_heads_are_architecture_specific():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "q_heads = 128 if sm_major >= 10 else native_heads" in source


def test_glm5_flash_mla_keeps_the_production_topk_width():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "torch.full((M, 1, topk), -1" in source
    assert "indices[:, :, :valid_k]" in source
    assert "if k % 64" not in source


def test_glm5_sparse_collector_is_pinned_to_sglang_0514_dsa_api():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert '__compat__ = "sglang==0.5.14"' in source
    assert "SGLANG_DSA_FUSE_TOPK" in source
    assert "SGLANG_NSA_FUSE_TOPK" not in source


def test_glm5_mqa_matches_sglang_query_row_chunking_contract():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION" in source
    assert "budget_bytes = min(int(free_mem * free_mem_fraction), int(total_mem * 0.30))" in source
    assert "budget_bytes // (num_k * 4)" in source
    assert "_glm5_score_rows_per_chunk(M, full_s, device)" in source
    assert "for start in range(0, M, query_rows_per_chunk)" in source
    assert "q[start:end]" in source
    assert "ks[start:end]" in source
    assert "ke[start:end]" in source


def test_glm5_sparse_case_plan_shards_real_context_and_decode_batches():
    helpers = _load_pure_helpers("_glm5_sparse_kernel_cases")
    helpers["_selected_glm5_models"] = lambda: ["nvidia/GLM-5-NVFP4"]
    helpers["get_sm_version"] = lambda: 90
    helpers["_dsa_context_derived_shapes"] = lambda _model: [(0, 128, 1), (64, 128, 2)]
    helpers["_dsa_generation_derived_shapes"] = lambda _model: [(128, 1, 4), (256, 1, 2)]

    for kernel in ("mqa", "topk", "dsa_attn"):
        assert helpers["_glm5_sparse_kernel_cases"](kernel) == [
            ["nvidia/GLM-5-NVFP4", kernel, 1],
            ["nvidia/GLM-5-NVFP4", kernel, 2],
            ["nvidia/GLM-5-NVFP4", kernel, 4],
        ]


def test_glm5_sparse_smoke_samples_each_batch_across_its_full_shape_range():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert 'if "--smoke" in sys.argv and len(shapes) > 8:' in source
    assert "round(i * (len(shapes) - 1) / 7)" in source


def test_glm5_sparse_platform_gating_lives_outside_the_getter():
    """The getter itself carries no SM predicate: pre-Hopper is dropped by the
    op_min_sm capability floors and SM120 is parked by registry maturity
    markers, so both stay auditable declaration-layer decisions."""
    from collector.capabilities import _load_capabilities
    from collector.sglang.registry import REGISTRY

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "not in {90, 100, 103}" not in source

    _, op_min_sm = _load_capabilities()
    glm5_ops = ("glm5_mqa_logits_module", "glm5_topk_module", "glm5_dsa_attn_module")
    for op in glm5_ops:
        assert op_min_sm.get(op) == 90
    markers = {entry.op: entry.unverified_sms for entry in REGISTRY if entry.op in glm5_ops}
    assert markers == dict.fromkeys(glm5_ops, (120,))


def test_dsv4_sparse_platform_gating_lives_outside_the_getter():
    """Same contract for the DSV4 sparse sub-kernel family; the flash_mla-less
    HCA/CSA collectors are parked whole-op unverified instead of silently
    enumerating zero cases."""
    from collector.capabilities import _load_capabilities
    from collector.sglang.registry import REGISTRY

    dsv4_source = (REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py").read_text(encoding="utf-8")
    assert "_dsv4_sparse_kernel_supported" not in dsv4_source
    assert "_dsv4_topk_kernel_supported" not in dsv4_source

    _, op_min_sm = _load_capabilities()
    floors = ("dsv4_paged_mqa_logits_module", "dsv4_hca_attn_module", "dsv4_csa_attn_module", "dsv4_csa_topk_calib")
    for op in floors:
        assert op_min_sm.get(op) == 90
    by_op = {entry.op: entry for entry in REGISTRY}
    assert by_op["dsv4_paged_mqa_logits_module"].unverified_sms == (120,)
    assert by_op["dsv4_csa_topk_calib"].unverified_sms == (120,)
    assert by_op["dsv4_hca_attn_module"].unverified is True
    assert by_op["dsv4_csa_attn_module"].unverified is True
