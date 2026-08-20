# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted migration warnings for legacy AIConfigurator entry points."""

from __future__ import annotations

import functools
import logging
import warnings
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

_MIGRATION_GUIDE = "https://github.com/ai-dynamo/aiconfigurator/blob/main/docs/aisimulate_migration.md"
_warned_entry_points: set[str] = set()

_P = ParamSpec("_P")
_R = TypeVar("_R")


def warn_legacy_cli(mode: str | None) -> None:
    """Warn once when the CLI is run from the legacy AIC distribution."""

    command = "aiconfigurator cli" + (f" {mode}" if mode else "")
    key = f"cli:{mode or '<root>'}"
    if key in _warned_entry_points:
        return
    _warned_entry_points.add(key)
    message = (
        f"`{command}` is running from the deprecated AIConfigurator "
        "distribution, which is planned for removal in AIConfigurator 0.13.0. "
        f"Install `aisimulate==0.12.0` and keep running `{command} ...`; "
        "AISimulate preserves the established command name and arguments. "
        f"Migration guide: {_MIGRATION_GUIDE}"
    )
    warnings.warn(message, DeprecationWarning, stacklevel=4)
    logger.warning("%s", message)


def deprecated_sweeper_entry_point(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Mark one legacy sweep function with a warn-once migration message."""

    entry_point = f"{function.__module__}.{function.__name__}"

    @functools.wraps(function)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if entry_point not in _warned_entry_points:
            _warned_entry_points.add(entry_point)
            warnings.warn(
                f"`{entry_point}()` is deprecated and will be removed in "
                "AIConfigurator 0.13.0. Migrate to "
                "`aisimulate.sweeper.Sweeper(...).run(config)`. Installing "
                "`aisimulate==0.12.0` temporarily retains the legacy "
                f"`aiconfigurator` import namespace. Migration guide: {_MIGRATION_GUIDE}",
                DeprecationWarning,
                stacklevel=2,
            )
        return function(*args, **kwargs)

    return wrapper
