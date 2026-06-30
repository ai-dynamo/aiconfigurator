# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

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
    assert "collector.vllm.utils" not in moe_source


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
