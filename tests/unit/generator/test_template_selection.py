# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for backend versioned template selection."""

from pathlib import Path

import pytest

from aiconfigurator.generator.rendering.engine import _log_versioned_template_selection, _select_versioned_template

pytestmark = pytest.mark.unit


def _paths(*names: str) -> list[Path]:
    return [Path(name) for name in names]


def test_selects_exact_cli_template_version():
    selected = _select_versioned_template(
        _paths("cli_args.j2", "cli_args.0.5.10.post1.j2", "cli_args.0.5.11.j2"),
        "cli_args",
        ".j2",
        "0.5.11",
    )

    assert selected is not None
    assert selected.name == "cli_args.0.5.11.j2"


def test_selects_closest_prior_cli_template_version():
    selected = _select_versioned_template(
        _paths("cli_args.j2", "cli_args.0.5.10.post1.j2", "cli_args.0.5.11.j2"),
        "cli_args",
        ".j2",
        "0.5.12",
    )

    assert selected is not None
    assert selected.name == "cli_args.0.5.11.j2"


def test_selects_closest_prior_engine_template_version():
    selected = _select_versioned_template(
        _paths(
            "extra_engine_args.yaml.j2",
            "extra_engine_args.1.3.0rc11.yaml.j2",
            "extra_engine_args.1.3.0rc14.yaml.j2",
        ),
        "extra_engine_args",
        ".yaml.j2",
        "1.3.0rc12",
    )

    assert selected is not None
    assert selected.name == "extra_engine_args.1.3.0rc11.yaml.j2"


def test_selects_default_template_without_requested_version():
    selected = _select_versioned_template(
        _paths("cli_args.j2", "cli_args.0.5.11.j2"),
        "cli_args",
        ".j2",
        None,
    )

    assert selected is not None
    assert selected.name == "cli_args.j2"


def test_logs_closest_prior_template_selection(caplog):
    selected = _paths("cli_args.0.20.1.j2")[0]

    _log_versioned_template_selection("CLI args", selected, "cli_args", ".j2", "9.9.9")

    assert "No exact CLI args template for 9.9.9" in caplog.text
    assert "cli_args.0.20.1.j2" in caplog.text


def test_logs_default_template_selection_for_invalid_requested_version(caplog):
    selected = _paths("cli_args.j2")[0]

    _log_versioned_template_selection("CLI args", selected, "cli_args", ".j2", "9.9.9-does-not-exist")

    assert "No version-specific CLI args template for 9.9.9-does-not-exist" in caplog.text
    assert "cli_args.j2" in caplog.text
