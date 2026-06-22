# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

_M = Path(__file__).resolve().parents[3] / "src/aiconfigurator/generator/facts/models.yaml"


def _models():
    return yaml.safe_load(_M.read_text())["models"]


def test_deepseek_v4_profile_present_and_structured():
    dsv4 = _models()["deepseek-v4"]
    assert "moe" in dsv4["traits"]
    assert dsv4["defaults"], "deepseek-v4 must carry scoped backend_args defaults"
    for block in dsv4["defaults"]:
        assert "match" in block and "backend_args" in block, "each default block is scoped"


def test_model_ids_unique_and_nonempty():
    models = _models()
    assert all(name for name in models)
    assert len(models) == len(set(models))
