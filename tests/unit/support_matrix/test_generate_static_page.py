# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).parents[3] / "tools" / "support_matrix" / "generate_static_page.py"
_SPEC = importlib.util.spec_from_file_location("generate_static_page", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
generate_static_page = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generate_static_page)

build_full_data = generate_static_page.build_full_data
build_target_versions = generate_static_page.build_target_versions


def _row(model, system, backend, version, mode="agg", status="PASS"):
    return {
        "HuggingFaceID": model,
        "Architecture": "ExampleForCausalLM",
        "System": system,
        "Backend": backend,
        "Version": version,
        "Mode": mode,
        "Status": status,
        "ErrMsg": "",
    }


def test_target_versions_are_derived_from_latest_tested_backend_versions():
    rows = [
        _row("model-a", "b200_sxm", "vllm", "0.14.0"),
        _row("model-a", "b200_sxm", "vllm", "0.19.0"),
        _row("model-a", "b200_sxm", "trtllm", "1.2.0rc5"),
        _row("model-a", "b200_sxm", "trtllm", "1.3.0rc10"),
        _row("model-a", "b200_sxm", "sglang", "0.5.9"),
        _row("model-a", "b200_sxm", "sglang", "0.5.10"),
    ]

    assert build_target_versions(rows) == {
        "sglang": "0.5.10",
        "trtllm": "1.3.0rc10",
        "vllm": "0.19.0",
    }


def test_full_page_data_uses_derived_target_versions():
    rows = [
        _row("model-a", "a100_sxm", "vllm", "0.14.0", "agg"),
        _row("model-a", "a100_sxm", "vllm", "0.14.0", "disagg"),
        _row("model-a", "b200_sxm", "vllm", "0.14.0", "agg"),
        _row("model-a", "b200_sxm", "vllm", "0.14.0", "disagg"),
        _row("model-a", "b200_sxm", "vllm", "0.19.0", "agg"),
        _row("model-a", "b200_sxm", "vllm", "0.19.0", "disagg"),
    ]

    data = build_full_data(rows)

    assert data["target_versions"] == {"vllm": "0.19.0"}
    assert data["systems"]["a100_sxm"]["target_versions"] == {"vllm": "0.14.0"}
    assert data["systems"]["b200_sxm"]["target_versions"] == {"vllm": "0.19.0"}
    assert data["systems"]["b200_sxm"]["matrix"]["all"][0]["cells"]["vllm"]["version"] == "0.19.0"
