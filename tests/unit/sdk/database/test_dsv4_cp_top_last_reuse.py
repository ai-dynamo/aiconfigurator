# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse-aware CP top_last loader pins (issue #1498, defect 1).

``ContextDeepSeekV4AttentionModule._load_csa_topk_top_last`` historically read
the version-pinned primary path only — it predated the ``reuse.yaml``
machinery, so every CSA CP config on a reuse-dependent version raised
``PerfDataNotAvailableError`` while its sibling DELTA consumer
(``_get_dsv4_topk_calib``) happily used the approved donor. The loader now
projects the same ``_build_op_sources`` + ``load_dsv4_sparse_op_data``
resolution as the DELTA path.

The pins below freeze the fixed behavior on the shipped b200_sxm/sglang data:
``sparse_attention/sglang/0.5.12/reuse.yaml`` declares
``dsv4_csa_topk_calib_perf from_version 0.5.14 (approved_by yimingl)``, so
0.5.12 must serve the donor's grid verbatim. If any of these move (reuse.yaml
edit, new 0.5.12 primary drop, loader change), that shift must be a conscious
act.
"""

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.operations.dsv4 import ContextDeepSeekV4AttentionModule

pytestmark = pytest.mark.unit

_FLASH_NATIVE_HEADS = 64


def test_csa_cp_top_last_loads_via_approved_reuse_donor():
    db = get_database("b200_sxm", "sglang", "0.5.12")
    grid = ContextDeepSeekV4AttentionModule._load_csa_topk_top_last(db, _FLASH_NATIVE_HEADS)
    # The reuse-dependent version resolves the donor's rows — the loader
    # previously returned {} here and the CP composition raised.
    assert grid, "0.5.12 must serve dsv4_csa_topk_calib top_last rows via the approved donor"
    # Donor identity: 0.5.12 carries no primary table, so the grid is the
    # 0.5.14 donor's grid verbatim (same reason the adjudicated CP repro
    # produces identical latencies on both versions).
    donor = get_database("b200_sxm", "sglang", "0.5.14")
    assert grid == ContextDeepSeekV4AttentionModule._load_csa_topk_top_last(donor, _FLASH_NATIVE_HEADS)


def test_csa_cp_top_last_lookup_pins_adjudicated_repro_values():
    # The exact sparse-gate values of the issue #1498 repro
    # (DeepSeek-V4-Flash | tp1 ep8 cp8 | b=1 isl=8192): tl_full 0.048698 /
    # tl_perc 0.0 — now resolvable on the reuse-dependent 0.5.12.
    db = get_database("b200_sxm", "sglang", "0.5.12")
    tl_full = ContextDeepSeekV4AttentionModule._csa_topk_top_last(db, 8192, 0, _FLASH_NATIVE_HEADS, 1)
    tl_perc = ContextDeepSeekV4AttentionModule._csa_topk_top_last(db, 1024, 0, _FLASH_NATIVE_HEADS, 1)
    assert tl_full == pytest.approx(0.048698, rel=1e-9)
    assert tl_perc == 0.0
