# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.support_matrix.compare_support_matrix import (
    check_csv_sanity,
    find_blocking_status_transitions,
    find_metadata_changes,
    generate_pr_description,
)
from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    STATUS_HYBRID_PASS,
    STATUS_PASS,
    SUPPORT_MATRIX_HEADER,
)

pytestmark = pytest.mark.unit

HEADER = SUPPORT_MATRIX_HEADER
SILICON_REPLAY = "uv run aiconfigurator cli default --database-mode SILICON"
HYBRID_REPLAY = "uv run aiconfigurator cli default --database-mode HYBRID"


def _row(
    status: str,
    err_msg: str = "",
    command: str = ("uv run aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --database-mode SILICON"),
    source: str | None = None,
) -> list[str]:
    if source is None:
        source = "silicon" if status == STATUS_PASS else ""
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
        source,
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


def test_csv_sanity_accepts_transitional_9col_header():
    """Committed per-system CSVs generated at the base+Command (9-col, pre-Source) stage
    must still validate -- compare rejecting them would break the daily diff."""
    from tools.support_matrix.support_matrix import SUPPORT_MATRIX_BASE_HEADER

    header9 = SUPPORT_MATRIX_BASE_HEADER + ["Command"]
    row9 = _row(STATUS_PASS)[:9]  # drop the Source column
    errors = check_csv_sanity(header9, [row9])
    assert errors == []


def test_csv_sanity_rejects_hybrid_pass_without_source_column():
    from tools.support_matrix.support_matrix import SUPPORT_MATRIX_BASE_HEADER

    header9 = SUPPORT_MATRIX_BASE_HEADER + ["Command"]
    row9 = _row(STATUS_HYBRID_PASS)[:9]
    errors = check_csv_sanity(header9, [row9])

    assert any("HYBRID_PASS requires the current header" in err for err in errors)


def test_csv_sanity_requires_command_for_current_header():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command="")])

    assert any("Command column must include" in err for err in errors)


@pytest.mark.parametrize(
    "_case,status,command,source",
    [
        (
            "silicon-equals-form",
            STATUS_PASS,
            SILICON_REPLAY.replace("--database-mode ", "--database-mode="),
            "silicon",
        ),
        ("hybrid-transfer", STATUS_HYBRID_PASS, HYBRID_REPLAY, "xshape"),
    ],
)
def test_csv_sanity_accepts_replay_contract(_case, status, command, source):
    assert check_csv_sanity(HEADER, [_row(status, command=command, source=source)]) == []


@pytest.mark.parametrize(
    "_case,status,command,source,expected_error",
    [
        (
            "hybrid-pass-with-silicon-command",
            STATUS_HYBRID_PASS,
            SILICON_REPLAY,
            "xshape",
            "exactly one effective --database-mode HYBRID",
        ),
        (
            "pass-with-hybrid-command",
            STATUS_PASS,
            HYBRID_REPLAY,
            "silicon",
            "exactly one effective --database-mode SILICON",
        ),
        ("pass-without-silicon-source", STATUS_PASS, SILICON_REPLAY, "", "PASS is reserved for SILICON support"),
        (
            "duplicate-database-mode",
            STATUS_PASS,
            f"{SILICON_REPLAY} --database-mode HYBRID",
            "silicon",
            "exactly one effective --database-mode SILICON",
        ),
        (
            "missing-database-mode-value",
            STATUS_PASS,
            "uv run aiconfigurator cli default --database-mode",
            "silicon",
            "exactly one effective --database-mode SILICON",
        ),
        ("malformed-command", STATUS_PASS, "uv run 'unterminated", "silicon", "not valid shell syntax"),
        (
            "nonpass-with-hybrid-command",
            STATUS_FAIL,
            HYBRID_REPLAY,
            "",
            "exactly one effective --database-mode SILICON",
        ),
        ("unknown-hybrid-source", STATUS_HYBRID_PASS, HYBRID_REPLAY, "made-up-tier", "Invalid Source"),
        ("nonpass-with-source", STATUS_FAIL, SILICON_REPLAY, "xshape", "must not include Source"),
    ],
)
def test_csv_sanity_rejects_invalid_replay_contract(_case, status, command, source, expected_error):
    errors = check_csv_sanity(HEADER, [_row(status, command=command, source=source)])
    assert any(expected_error in error for error in errors)


def test_csv_sanity_rejects_duplicate_configuration_key():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS), _row(STATUS_HYBRID_PASS, source="xshape")])

    assert any("duplicate support-matrix key" in err for err in errors)


@pytest.mark.parametrize(
    "old_command,new_command,old_source,new_source",
    [
        ("old command", "new command", "xshape", "xshape"),
        ("same command", "same command", "xshape", "xop"),
    ],
)
def test_metadata_diff_detects_replay_command_or_source_changes(old_command, new_command, old_source, new_source):
    old = _row(STATUS_HYBRID_PASS, command=old_command, source=old_source)
    new = _row(STATUS_HYBRID_PASS, command=new_command, source=new_source)

    changes = find_metadata_changes([old], [new])

    assert len(changes) == 1
    assert changes[0][-4:] == (old_command, new_command, old_source, new_source)


def test_pass_to_hardware_incompatible_is_blocking_transition():
    errors = find_blocking_status_transitions([_changed(STATUS_PASS, STATUS_HW_INCOMPATIBLE)])

    assert len(errors) == 1
    assert "PASS -> HW_INCOMPATIBLE" in errors[0]


@pytest.mark.parametrize(
    "old_status,new_status",
    [(STATUS_PASS, STATUS_HYBRID_PASS), (STATUS_HYBRID_PASS, STATUS_FAIL)],
)
def test_hybrid_status_regressions_are_blocking(old_status, new_status):
    errors = find_blocking_status_transitions([_changed(old_status, new_status)])

    assert len(errors) == 1
    assert f"{old_status} -> {new_status}" in errors[0]


def test_hybrid_pass_to_pass_is_a_non_blocking_silicon_fix():
    errors = find_blocking_status_transitions([_changed(STATUS_HYBRID_PASS, STATUS_PASS)])
    pr_description = generate_pr_description([], [], [_changed(STATUS_HYBRID_PASS, STATUS_PASS)])

    assert errors == []
    assert "Silicon fixes" in pr_description


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
