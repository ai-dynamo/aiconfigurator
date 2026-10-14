# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Migration-warning contracts for legacy AIC CLI and Sweeper surfaces."""

import warnings

import pytest

from aiconfigurator import deprecation
from aiconfigurator import main as root_main
from aiconfigurator.sdk import sweep

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_warning_state():
    before = set(deprecation._warned_entry_points)
    deprecation._warned_entry_points.clear()
    try:
        yield
    finally:
        deprecation._warned_entry_points.clear()
        deprecation._warned_entry_points.update(before)


def _stub_cli(monkeypatch) -> None:
    monkeypatch.setattr(root_main, "generator_cli_helper", lambda _args: False)
    monkeypatch.setattr(
        root_main,
        "configure_cli_parser",
        lambda parser: parser.add_argument("mode"),
    )
    monkeypatch.setattr(root_main, "cli_main", lambda _args: None)


@pytest.mark.parametrize(
    "mode",
    ["default", "estimate", "recommend", "exp", "generate", "support"],
)
def test_each_legacy_cli_mode_warns_without_renaming_the_command(monkeypatch, mode) -> None:
    _stub_cli(monkeypatch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        root_main.main(["cli", mode])

    assert len(captured) == 1
    warning = captured[0]
    assert warning.category is DeprecationWarning
    assert f"`aiconfigurator cli {mode}`" in str(warning.message)
    assert f"`aiconfigurator cli {mode} ...`" in str(warning.message)
    assert "`aisimulate cli" not in str(warning.message)
    assert "AIConfigurator 0.13.0" in str(warning.message)
    assert "deprecated AIConfigurator distribution" in str(warning.message)
    assert "preserves the established command name" in str(warning.message)
    assert warning.filename == __file__


def test_cli_warning_fires_once_per_mode(monkeypatch) -> None:
    _stub_cli(monkeypatch)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        root_main.main(["cli", "default"])
        root_main.main(["cli", "default"])

    assert len(captured) == 1


def test_version_command_remains_unaffected() -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        root_main.main(["version"])

    assert captured == []


@pytest.mark.parametrize("entry_point", [sweep.sweep_agg, sweep.sweep_disagg, sweep.sweep_afd])
def test_each_legacy_sweeper_entry_point_warns_once_at_caller(entry_point) -> None:
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            entry_point()
        with pytest.raises(TypeError):
            entry_point()

    assert len(captured) == 1
    warning = captured[0]
    assert warning.category is DeprecationWarning
    assert entry_point.__name__ in str(warning.message)
    assert "aisimulate.sweeper.Sweeper" in str(warning.message)
    assert "AIConfigurator 0.13.0" in str(warning.message)
    assert warning.filename == __file__
