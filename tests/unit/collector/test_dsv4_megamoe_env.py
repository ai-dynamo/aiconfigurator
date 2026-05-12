# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit

_MODULE = "collector.sglang.collect_dsv4_megamoe"
_saved_module = sys.modules.pop(_MODULE, None)
_saved_torch = sys.modules.get("torch")
_saved_torch_distributed = sys.modules.get("torch.distributed")
sys.modules["torch"] = MagicMock()
sys.modules["torch.distributed"] = MagicMock()
try:
    from collector.sglang.collect_dsv4_megamoe import _env_flag, _routing_dump_layers_for_case
finally:
    sys.modules.pop(_MODULE, None)
    if _saved_module is not None:
        sys.modules[_MODULE] = _saved_module
    if _saved_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = _saved_torch
    if _saved_torch_distributed is None:
        sys.modules.pop("torch.distributed", None)
    else:
        sys.modules["torch.distributed"] = _saved_torch_distributed


def test_env_flag_treats_empty_value_as_default(monkeypatch):
    monkeypatch.setenv("AIC_TEST_FLAG", "")
    assert _env_flag("AIC_TEST_FLAG") == 0

    monkeypatch.setenv("AIC_TEST_FLAG", "   ")
    assert _env_flag("AIC_TEST_FLAG", default="1") == 1


def test_routing_dump_layers_for_case_does_not_mutate_args():
    args = SimpleNamespace(routing_dump_layer="bottleneck", routing_dump_layers="2,5-6")

    assert _routing_dump_layers_for_case(args, SimpleNamespace(distribution="sglang_trace")) == ["2", "5", "6"]
    assert args.routing_dump_layer == "bottleneck"
    assert _routing_dump_layers_for_case(args, SimpleNamespace(distribution="balanced")) == ["bottleneck"]
