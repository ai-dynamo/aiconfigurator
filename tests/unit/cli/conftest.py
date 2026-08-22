# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _allow_unlisted_versions_for_cli_fixtures(monkeypatch):
    """CLI suites exercise orchestration with synthetic/mocked version labels
    (e.g. 'test-version') against the real systems root. Transition escape
    until the fixture-discipline follow-up re-pins them onto version slots."""
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
