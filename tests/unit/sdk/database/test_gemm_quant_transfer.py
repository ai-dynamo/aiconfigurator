# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GEMM quant-transfer admission metadata.

The quant-transfer LADDER this file used to exercise on the synthetic fixture
(xquant same-profile borrowing, xprofile weight-only -> bf16 reference
ordering, policy gating, HYBRID invariance for covered quants, SILICON's
never-borrow contrast) retired to the compiled engine with #1357 PR-5; it is
anchored by the frozen
parity goldens. What stays Python-owned is the util-LEVEL admission table
(``xprofile_util_level_known`` metadata) below.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations import util_empirical
from aiconfigurator.sdk.operations.gemm import _GEMM_QUANT_UTIL_LEVEL

pytestmark = pytest.mark.unit


def test_level_table_covers_every_gemm_quant_profile():
    """Every GEMMQuantMode profile must have a calibrated level line — the
    gate refuses XPROFILE admission for unlisted profiles by design, so a
    profile silently missing here would strand its quants in HYBRID."""
    for quant in common.GEMMQuantMode:
        assert util_empirical.quant_profile(quant) in _GEMM_QUANT_UTIL_LEVEL, quant
