# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integrity-guard coverage for ``regenerate_goldens.py``.

The golden fixtures are Gate 2's frozen deletion anchor, so the regeneration
script must (a) refuse a dirty starting tree before any evaluation or write,
and (b) never leave a partially rewritten fixture set behind a mid-capture
failure. These tests pin both guards without running a real capture.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_PARITY_DIR = Path(__file__).resolve().parent


@pytest.fixture()
def regen():
    """Import ``regenerate_goldens`` with its import-time env pinning
    contained: the module sets thread caps and
    ``AICONFIGURATOR_ENGINE_STEP_BACKEND=python`` for a real capture run,
    which must not leak into the rest of this pytest process (the parity
    suites construct rust-backed configs)."""
    saved = os.environ.copy()
    if str(_PARITY_DIR) not in sys.path:
        sys.path.insert(0, str(_PARITY_DIR))
    try:
        import regenerate_goldens as module

        yield module
    finally:
        for key in set(os.environ) - set(saved):
            del os.environ[key]
        os.environ.update(saved)


def test_dirty_paths_ignores_goldens_and_untracked(regen) -> None:
    goldens = regen._GOLDEN_REL_PREFIX
    porcelain = "\n".join(
        [
            f" M {goldens}engine_step.json",  # own output of a previous run
            f"M  {goldens}per_op.json",  # staged own output
            "?? scratch/notes.txt",  # untracked: no describe/capture impact
            "",
        ]
    )
    assert regen._dirty_paths(porcelain) == []

    porcelain = "\n".join(
        [
            f" M {goldens}engine_step.json",
            " M aic-core/rust/aiconfigurator-core/src/engine/runtime.rs",
            "R  old_name.py -> new_name.py",
        ]
    )
    assert regen._dirty_paths(porcelain) == [
        "aic-core/rust/aiconfigurator-core/src/engine/runtime.rs",
        "old_name.py -> new_name.py",
    ]


def test_dirty_start_aborts_before_any_capture_or_write(regen, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(regen, "GOLDEN_DIR", tmp_path / "goldens")

    def _fail(*_args, **_kwargs):  # pragma: no cover — reaching this is the failure
        raise AssertionError("capture ran despite a dirty starting tree")

    for name in ("capture_engine_step", "capture_compile_engine", "capture_per_op"):
        monkeypatch.setattr(regen, name, _fail)
    monkeypatch.setattr(
        regen.subprocess,
        "check_output",
        lambda *a, **k: " M aic-core/src/aiconfigurator_core/sdk/perf_database.py\n",
    )

    with pytest.raises(RuntimeError, match="clean tree"):
        regen.main()
    assert not (tmp_path / "goldens").exists()


def test_late_skip_leaves_goldens_untouched(regen, monkeypatch, tmp_path) -> None:
    """A skip in the LAST capture must not have written the earlier
    payloads: all three are captured before the first write."""
    golden_dir = tmp_path / "goldens"
    monkeypatch.setattr(regen, "GOLDEN_DIR", golden_dir)
    monkeypatch.setattr(regen, "_require_clean_tree", lambda: None)
    monkeypatch.setattr(regen, "capture_engine_step", lambda: {"header": {}, "cases": {}})
    monkeypatch.setattr(regen, "capture_compile_engine", lambda: {"header": {}, "references": {}})

    def _skip():
        raise regen._PytestSkipped("wideep data set missing")

    monkeypatch.setattr(regen, "capture_per_op", _skip)

    with pytest.raises(RuntimeError, match="BEFORE any write"):
        regen.main()
    assert list(golden_dir.glob("*")) == []
