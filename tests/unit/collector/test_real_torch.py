# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real tensor operations across repeated mock-preserving imports."""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

from tests.unit.collector._real_torch import real_torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("entry", [None, MagicMock()])
def test_repeated_borrows_preserve_module_entry(monkeypatch, entry):
    if entry is None:
        monkeypatch.delitem(sys.modules, "torch", raising=False)
    else:
        monkeypatch.setitem(sys.modules, "torch", entry)

    for _ in range(3):
        with real_torch() as torch:
            assert sys.modules["torch"] is torch
            assert torch.arange(3).sum().item() == 3
        assert sys.modules["torch"] is (torch if entry is None else entry)
        if entry is None:
            assert importlib.import_module("torch") is torch


def test_borrow_restores_mock_when_body_raises(monkeypatch):
    mock_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    with pytest.raises(RuntimeError, match="consumer failed"), real_torch() as torch:
        assert torch.ones(2).sum().item() == 2
        raise RuntimeError("consumer failed")

    assert sys.modules["torch"] is mock_torch


def test_borrow_preserves_existing_real_torch(monkeypatch):
    with real_torch() as torch:
        pass
    monkeypatch.setitem(sys.modules, "torch", torch)
    with real_torch() as borrowed:
        assert borrowed is torch
    assert sys.modules["torch"] is torch
