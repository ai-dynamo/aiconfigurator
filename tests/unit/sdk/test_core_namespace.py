# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upper/core SDK ownership boundary."""

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.unit


def test_task_v2_remains_upper_owned() -> None:
    task = importlib.import_module("aiconfigurator.sdk.task_v2").Task

    assert task.__module__ == "aiconfigurator.sdk.task_v2"
    assert importlib.util.find_spec("aiconfigurator_core.sdk.task_v2") is None
