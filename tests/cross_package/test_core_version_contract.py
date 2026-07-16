# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version and wire-schema contracts shared by the core wheel and crate."""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

import aiconfigurator_core
from aiconfigurator_core.sdk import engine

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = REPO_ROOT / "aic-core"
RUST_CONFIG = CORE_ROOT / "rust" / "aiconfigurator-core" / "src" / "config.rs"


def _project_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text())["project"]["version"])


def _crate_version(path: Path) -> str:
    return str(tomllib.loads(path.read_text())["package"]["version"])


def _rust_u32_constant(path: Path, name: str) -> int:
    source = path.read_text()
    match = re.search(rf"^pub const {re.escape(name)}: u32 = (\d+);$", source, re.MULTILINE)
    assert match is not None, f"missing public Rust constant {name}"
    return int(match.group(1))


def test_core_wheel_and_crate_versions_match() -> None:
    assert _project_version(CORE_ROOT / "pyproject.toml") == _crate_version(
        CORE_ROOT / "rust" / "aiconfigurator-core" / "Cargo.toml"
    )


def test_engine_schema_versions_match_across_python_and_rust() -> None:
    assert _rust_u32_constant(RUST_CONFIG, "ENGINE_CONFIG_SCHEMA_VERSION") == engine.ENGINE_CONFIG_SCHEMA_VERSION
    assert _rust_u32_constant(RUST_CONFIG, "ENGINE_SPEC_SCHEMA_VERSION") == engine.ENGINE_SPEC_SCHEMA_VERSION
    assert aiconfigurator_core._build_smoke() == engine.ENGINE_CONFIG_SCHEMA_VERSION
