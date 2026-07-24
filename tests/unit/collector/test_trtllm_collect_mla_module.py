# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST-only guards for the TRT-LLM MLA/DSA module collector.

The trtllm ``create_attention_layer`` must resolve the model id to a local,
auto_map-stripped config dir (``helper._resolve_local_model_path``) before any
transformers / TRT-LLM config load — otherwise the raw HF id routes through the
``trust_remote_code`` path whose ``_remote_code.lock`` serialized the 8 parallel
workers and produced mass file-lock Timeouts on the ``*_module`` ops. AST-only so
the check runs on CUDA-free CI (``create_attention_layer`` cannot be exec'd
without the framework).
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE = REPO_ROOT / "collector" / "trtllm" / "collect_mla_module.py"


def _function(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_resolver_is_imported_from_helper():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "collector.helper"
        for alias in node.names
    }
    assert "_resolve_local_model_path" in imported, (
        "collect_mla_module must import _resolve_local_model_path from collector.helper"
    )


def test_create_attention_layer_resolves_model_path_before_config_load():
    fn = _function("create_attention_layer")

    # The reassignment `model_path = _resolve_local_model_path(model_path)`.
    resolve_line = None
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "model_path" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "_resolve_local_model_path"
        ):
            resolve_line = node.lineno
            break
    assert resolve_line is not None, (
        "create_attention_layer must reassign model_path via _resolve_local_model_path(model_path)"
    )

    # It must run before the first transformers config read (get_config_dict),
    # so the local auto_map-stripped path — not the raw HF id — is loaded.
    config_line = min(
        (
            node.lineno
            for node in ast.walk(fn)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get_config_dict"
        ),
        default=None,
    )
    assert config_line is not None, "expected a get_config_dict(...) call in create_attention_layer"
    assert resolve_line < config_line, (
        "model_path must be resolved to a local config BEFORE get_config_dict(), "
        "or the trust_remote_code lock path is still taken"
    )
