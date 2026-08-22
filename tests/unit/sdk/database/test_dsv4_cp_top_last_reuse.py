# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CSA topk-calib CP top_last value pins (issue #1498 repro numbers).

Historical context: this file used to pin the reuse.yaml donor-resolution
behavior for the reuse-dependent 0.5.12 (issue #1498, defect 1). The
queryable-version slot model retired that mechanism — 0.5.12 is a data
coordinate, not a version — so the donor-behavior test is gone and only the
adjudicated repro VALUES remain pinned, on the current-slot primary rows
they always came from (0.5.12's declaration pointed at 0.5.14 verbatim).
"""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

pytestmark = pytest.mark.unit

_FLASH_NATIVE_HEADS = 64


def test_csa_cp_top_last_rows_pin_adjudicated_repro_values():
    # The exact sparse-gate row of the issue #1498 repro
    # (DeepSeek-V4-Flash | tp1 ep8 cp8 | b=1 isl=8192): top_last 0.048698.
    # Keyed [native][step][isl][bs][score_mode].
    db = get_database("b200_sxm", "sglang", "0.5.14")
    rows = fetch_table_view(db, "_dsv4_csa_topk_calib_data")
    assert rows and _FLASH_NATIVE_HEADS in rows
    leaf = rows[_FLASH_NATIVE_HEADS][0][8192][1]
    assert leaf["v1_top_last"]["latency"] == pytest.approx(0.048698, rel=1e-9)
    # The percolated 1024-isl point of the repro: flat == top_last == 0.0, so
    # the DELTA the engine derives there is zero (the repro's tl_perc 0.0).
    perc = rows[_FLASH_NATIVE_HEADS][0][1024][1]
    assert perc["v1_top_last"]["latency"] == 0.0
    assert perc["v1_flat"]["latency"] == 0.0
