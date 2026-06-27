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
    assert not any("Invalid header" in e for e in errors), errors


def test_csv_sanity_rejects_hybrid_pass_without_source_column():
    from tools.support_matrix.support_matrix import SUPPORT_MATRIX_BASE_HEADER

    header9 = SUPPORT_MATRIX_BASE_HEADER + ["Command"]
    row9 = _row(STATUS_HYBRID_PASS)[:9]
    errors = check_csv_sanity(header9, [row9])

    assert any("HYBRID_PASS requires the current header" in err for err in errors)


def test_csv_sanity_requires_command_for_current_header():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command="")])

    assert any("Command column must include" in err for err in errors)


def test_csv_sanity_accepts_explicit_hybrid_pass_with_replayable_command():
    errors = check_csv_sanity(
        HEADER,
        [
            _row(
                STATUS_HYBRID_PASS,
                command=("uv run aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --database-mode HYBRID"),
                source="xshape",
            )
        ],
    )

    assert errors == []


def test_csv_sanity_rejects_hybrid_pass_with_silicon_replay_command():
    errors = check_csv_sanity(HEADER, [_row(STATUS_HYBRID_PASS, source="xshape")])

    assert any("exactly one effective --database-mode HYBRID" in err for err in errors)


def test_csv_sanity_rejects_pass_with_hybrid_replay_command():
    errors = check_csv_sanity(
        HEADER,
        [
            _row(
                STATUS_PASS,
                command=("uv run aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --database-mode HYBRID"),
            )
        ],
    )

    assert any("exactly one effective --database-mode SILICON" in err for err in errors)


def test_csv_sanity_requires_explicit_silicon_source_for_current_pass():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, source="")])

    assert any("PASS is reserved for SILICON support" in err for err in errors)


@pytest.mark.parametrize(
    "command",
    [
        "uv run aiconfigurator cli default --database-mode SILICON --database-mode HYBRID",
        "uv run aiconfigurator cli default --database-mode HYBRID --database-mode SILICON",
    ],
)
def test_csv_sanity_rejects_duplicate_database_mode_flags(command):
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command=command)])

    assert any("exactly one effective --database-mode SILICON" in err for err in errors)


def test_csv_sanity_accepts_equals_form_database_mode():
    command = "uv run aiconfigurator cli default --database-mode=SILICON"

    assert check_csv_sanity(HEADER, [_row(STATUS_PASS, command=command)]) == []


@pytest.mark.parametrize(
    "command",
    [
        "uv run aiconfigurator cli default --database-mode --top-n 1",
        "uv run aiconfigurator cli default --database-mode",
    ],
)
def test_csv_sanity_rejects_missing_database_mode_value(command):
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command=command)])

    assert any("exactly one effective --database-mode SILICON" in err for err in errors)


def test_csv_sanity_rejects_malformed_replay_command():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, command="uv run 'unterminated")])

    assert any("not valid shell syntax" in err for err in errors)


def test_csv_sanity_requires_silicon_replay_for_nonpass_status():
    command = "uv run aiconfigurator cli default --database-mode HYBRID"
    errors = check_csv_sanity(HEADER, [_row(STATUS_FAIL, command=command)])

    assert any("exactly one effective --database-mode SILICON" in err for err in errors)


def test_csv_sanity_rejects_unknown_or_nonpass_source():
    hybrid_errors = check_csv_sanity(
        HEADER,
        [
            _row(
                STATUS_HYBRID_PASS,
                command=("uv run aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --database-mode HYBRID"),
                source="made-up-tier",
            )
        ],
    )
    fail_errors = check_csv_sanity(HEADER, [_row(STATUS_FAIL, source="xshape")])

    assert any("Invalid Source" in err for err in hybrid_errors)
    assert any("must not include Source" in err for err in fail_errors)


def test_csv_sanity_rejects_duplicate_configuration_key():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS), _row(STATUS_HYBRID_PASS, source="xshape")])

    assert any("duplicate support-matrix key" in err for err in errors)


def test_csv_sanity_rejects_legacy_pass_with_empirical_source():
    errors = check_csv_sanity(HEADER, [_row(STATUS_PASS, source="empirical")])

    assert any("PASS is reserved for SILICON support" in err for err in errors)


def test_metadata_diff_detects_source_and_replay_command_changes():
    old = _row(STATUS_HYBRID_PASS, command="old command", source="xshape")
    new = _row(STATUS_HYBRID_PASS, command="new command", source="xop")

    changes = find_metadata_changes([old], [new])

    assert len(changes) == 1
    assert changes[0][-4:] == ("old command", "new command", "xshape", "xop")


def test_pass_to_hardware_incompatible_is_blocking_transition():
    errors = find_blocking_status_transitions([_changed(STATUS_PASS, STATUS_HW_INCOMPATIBLE)])

    assert len(errors) == 1
    assert "PASS -> HW_INCOMPATIBLE" in errors[0]


def test_pass_to_hybrid_pass_is_a_blocking_silicon_regression():
    errors = find_blocking_status_transitions([_changed(STATUS_PASS, STATUS_HYBRID_PASS)])

    assert len(errors) == 1
    assert "PASS -> HYBRID_PASS" in errors[0]


def test_hybrid_pass_to_pass_is_a_non_blocking_silicon_fix():
    errors = find_blocking_status_transitions([_changed(STATUS_HYBRID_PASS, STATUS_PASS)])
    pr_description = generate_pr_description([], [], [_changed(STATUS_HYBRID_PASS, STATUS_PASS)])

    assert errors == []
    assert "Silicon fixes" in pr_description


def test_hybrid_pass_to_fail_is_blocking():
    errors = find_blocking_status_transitions([_changed(STATUS_HYBRID_PASS, STATUS_FAIL)])

    assert len(errors) == 1
    assert "HYBRID_PASS -> FAIL" in errors[0]


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
