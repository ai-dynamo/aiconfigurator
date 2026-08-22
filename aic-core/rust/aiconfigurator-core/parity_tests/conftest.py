# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(autouse=True)
def _allow_unlisted_versions_for_frozen_goldens(monkeypatch):
    """Golden cases pin the exact data coordinates they were frozen against
    (old framework versions included) — they exercise the loader, not the
    user-facing version surface. Bypass the queryable-version slot gate."""
    monkeypatch.setenv("AIC_ALLOW_UNLISTED_VERSIONS", "1")
