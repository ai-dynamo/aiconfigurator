# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix.compare_support_matrix import (
    check_csv_sanity,
    find_blocking_status_transitions,
    generate_pr_description,
)
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    STATUS_PASS,
    SUPPORT_MATRIX_HEADER,
)

pytestmark = pytest.mark.unit

HEADER = SUPPORT_MATRIX_HEADER


def _row(
    status: str,
    err_msg: str = "",
    command: str = "uv run aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8",
) -> list[str]:
    return [
        "Qwen/Qwen3-32B-FP8",
        "Qwen3ForCausalLM",
        "a100_sxm",
        "trtllm",
        "1.0.0",
        "agg",
        status,
        err_msg,
        command,
    ]


def _changed(old_status: str, new_status: str) -> tuple:
    return (
        "Qwen/Qwen3-32B-FP8",
        "Qwen3ForCausalLM",
        "a100_sxm",
        "trtllm",
        "1.0.0",
        "agg",
        old_status,
        new_status,
    )


def test_csv_sanity_accepts_hardware_incompatible_status_with_reason():
    errors = check_csv_sanity(
        HEADER,
        [_row(STATUS_HW_INCOMPATIBLE, "a100_sxm (SM80) does not support FP8 required by Qwen/Qwen3-32B-FP8")],
    )

    assert errors == []


def test_csv_sanity_accepts_framework_incompatible_status_with_reason():
    errors = check_csv_sanity(
        HEADER,
        [_row(STATUS_FRAMEWORK_INCOMPATIBLE, "Framework runtime rejects the required attention shape")],
    )

    assert errors == []


def test_csv_sanity_requires_framework_incompatible_reason():
    errors = check_csv_sanity(HEADER, [_row(STATUS_FRAMEWORK_INCOMPATIBLE)])

    assert any("framework incompatibility reason" in err for err in errors)


def test_csv_sanity_requires_hardware_incompatible_reason():
    errors = check_csv_sanity(HEADER, [_row(STATUS_HW_INCOMPATIBLE)])

    assert any("must include a hardware incompatibility reason" in err for err in errors)


def test_csv_sanity_requires_command_for_current_header():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command="")])

    assert any("Command column must include" in err for err in errors)


def test_pass_to_hardware_incompatible_is_blocking_transition():
    errors = find_blocking_status_transitions([_changed(STATUS_PASS, STATUS_HW_INCOMPATIBLE)])

    assert len(errors) == 1
    assert "PASS -> HW_INCOMPATIBLE" in errors[0]


def test_hardware_incompatible_to_fail_is_blocking_transition():
    errors = find_blocking_status_transitions([_changed(STATUS_HW_INCOMPATIBLE, STATUS_FAIL)])

    assert len(errors) == 1
    assert "HW_INCOMPATIBLE -> FAIL" in errors[0]


def test_framework_incompatible_to_fail_is_blocking_transition():
    errors = find_blocking_status_transitions([_changed(STATUS_FRAMEWORK_INCOMPATIBLE, STATUS_FAIL)])

    assert len(errors) == 1
    assert "FRAMEWORK_INCOMPATIBLE -> FAIL" in errors[0]


def test_fail_to_hardware_incompatible_is_reported_but_not_blocking():
    errors = find_blocking_status_transitions([_changed(STATUS_FAIL, STATUS_HW_INCOMPATIBLE)])
    pr_description = generate_pr_description([], [], [_changed(STATUS_FAIL, STATUS_HW_INCOMPATIBLE)])

    assert errors == []
    assert "Reclassified as hardware-incompatible" in pr_description
    assert "| Qwen/Qwen3-32B-FP8 |" in pr_description


def test_fail_to_framework_incompatible_is_reported_but_not_blocking():
    errors = find_blocking_status_transitions([_changed(STATUS_FAIL, STATUS_FRAMEWORK_INCOMPATIBLE)])
    pr_description = generate_pr_description([], [], [_changed(STATUS_FAIL, STATUS_FRAMEWORK_INCOMPATIBLE)])

    assert errors == []
    assert "Reclassified as framework-incompatible" in pr_description
    assert "| Qwen/Qwen3-32B-FP8 |" in pr_description
