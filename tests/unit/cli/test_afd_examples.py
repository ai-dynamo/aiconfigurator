# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The AFD entries in ``example.yaml`` must stay loadable as real ``Task``s.

Documentation that does not parse is worse than no documentation: a user copies
it, hits an error, and has no way to tell whether the example or their edit is
at fault. A plain ``yaml.safe_load`` is not enough either -- a file can be
valid YAML and still name a field ``Task`` has never had.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

_EXAMPLE = Path(__file__).resolve().parents[3] / "src" / "aiconfigurator" / "cli" / "example.yaml"

_AFD_EXPERIMENTS = ("afd_search", "afd_pinned", "afd_combined_with_pd")


@pytest.fixture(scope="module")
def experiments() -> dict:
    if not _EXAMPLE.is_file():
        pytest.skip(f"{_EXAMPLE} not found")
    return yaml.safe_load(_EXAMPLE.read_text(encoding="utf-8"))


class TestAfdExamplesParse:
    @pytest.mark.parametrize("name", _AFD_EXPERIMENTS)
    def test_experiment_is_present(self, experiments, name):
        assert name in experiments, f"example.yaml lost the {name} experiment"

    @pytest.mark.parametrize("name", _AFD_EXPERIMENTS)
    def test_experiment_builds_a_task(self, experiments, name):
        """Construct the Task the same way ``cli exp`` would.

        ``Task.__post_init__`` is where AFD budget resolution and pool
        validation live, so a successful construction is what proves the
        example is usable rather than merely well-formed.
        """
        task = Task(**experiments[name])
        assert task.serving_mode == "afd"

    def test_pinned_topology_pins_all_three_fields(self, experiments):
        """A partial pin silently falls back to search, which would make the
        'pinned' example misleading rather than wrong."""
        pinned = experiments["afd_pinned"]
        for field in ("afd_n_a_nodes", "afd_n_f_nodes", "afd_tp_a"):
            assert pinned.get(field) is not None, f"afd_pinned must set {field}"

    def test_topology_respects_the_two_node_minimum(self, experiments):
        """Both pools are node-granular and need at least one node each."""
        pinned = experiments["afd_pinned"]
        assert pinned["afd_n_a_nodes"] >= 1
        assert pinned["afd_n_f_nodes"] >= 1
        assert pinned["afd_n_a_nodes"] + pinned["afd_n_f_nodes"] >= 2


class TestCommentedFieldsAreReal:
    def test_every_commented_afd_key_is_a_task_field(self):
        """A typo inside a commented-out example still teaches the wrong name.

        Only ``afd_``-prefixed keys are checked: the file also carries prose
        comments, and restricting the scan to the documented field prefix keeps
        this from tripping over English text that happens to contain a colon.
        """
        if not _EXAMPLE.is_file():
            pytest.skip(f"{_EXAMPLE} not found")
        known = {f.name for f in dataclasses.fields(Task)}

        unknown = []
        for raw in _EXAMPLE.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped.startswith("# afd_"):
                continue
            key = stripped[2:].split(":")[0].strip()
            if key and key not in known:
                unknown.append(key)

        assert unknown == [], f"example.yaml documents non-existent Task fields: {unknown}"
