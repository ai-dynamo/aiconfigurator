# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the core-namespaced SDK compatibility surface."""

import importlib

import pytest

pytestmark = pytest.mark.unit


def test_core_namespaced_task_is_canonical_sdk_task() -> None:
    core_namespaced_task = importlib.import_module("aiconfigurator_core.sdk.task_v2").Task
    canonical_task = importlib.import_module("aiconfigurator.sdk.task_v2").Task

    assert core_namespaced_task is canonical_task
    assert core_namespaced_task.__module__ == "aiconfigurator.sdk.task_v2"
