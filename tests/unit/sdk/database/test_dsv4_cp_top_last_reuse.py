# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse-aware CP top_last loading pins (issue #1498, defect 1).

The CSA topk-calib loader historically read the version-pinned primary path
only — it predated the ``reuse.yaml`` machinery, so every CSA CP config on a
reuse-dependent version raised ``PerfDataNotAvailableError`` while the DELTA
consumer happily used the approved donor. The fix projects the standard
``_build_op_sources`` + ``load_dsv4_sparse_op_data`` resolution; the pins
below freeze that behavior on the shipped b200_sxm/sglang data:
``sparse_attention/sglang/0.5.12/reuse.yaml`` declares
``dsv4_csa_topk_calib_perf from_version 0.5.14 (approved_by yimingl)``, so
0.5.12 must serve the donor's grid verbatim. If any of these move (reuse.yaml
edit, new 0.5.12 primary drop, loader change), that shift must be a conscious
act.

Observed on the RAW loaded rows: the per-op ``_load_csa_topk_top_last`` /
``_csa_topk_top_last`` lookup helpers (and their strict-provenance-keyed
cache) retired with the CP query math in #1357 PR-5 — the compiled engine
builds its calib from the same source chain (parity goldens); generic
strict-provenance source gating is pinned in test_strict_mode.py.
"""

import os

import pytest

from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator_core.sdk.common import PerfDataFilename
from aiconfigurator_core.sdk.operations.base import resolve_op_data_path
from aiconfigurator_core.sdk.operations.dsv4 import _TOPK_CALIB_KEYS, load_dsv4_sparse_op_data

pytestmark = pytest.mark.unit

_FLASH_NATIVE_HEADS = 64


def _load_topk_calib_rows(db):
    """The same source projection the engine consumes: shared-layer + reuse
    resolution via ``_build_op_sources``, parsed by the generic sparse-op
    loader under the calib key order."""
    system_data_root = os.path.join(db.systems_root, db.system_spec["data_dir"])
    primary = resolve_op_data_path(system_data_root, db.backend, db.version, PerfDataFilename.dsv4_csa_topk_calib.value)
    sources = db._build_op_sources(PerfDataFilename.dsv4_csa_topk_calib, primary, system_data_root)
    return load_dsv4_sparse_op_data(sources, _TOPK_CALIB_KEYS)


def test_csa_cp_top_last_loads_via_approved_reuse_donor():
    db = get_database("b200_sxm", "sglang", "0.5.12")
    rows = _load_topk_calib_rows(db)
    # The reuse-dependent version resolves the donor's rows — the pre-#1498
    # primary-only read returned nothing here and the CP composition raised.
    assert rows and _FLASH_NATIVE_HEADS in rows, "0.5.12 must serve dsv4_csa_topk_calib rows via the approved donor"
    # Donor identity: 0.5.12 carries no primary table, so the grid is the
    # 0.5.14 donor's grid verbatim (same reason the adjudicated CP repro
    # produces identical latencies on both versions).
    donor = get_database("b200_sxm", "sglang", "0.5.14")
    assert rows == _load_topk_calib_rows(donor)


def test_csa_cp_top_last_rows_pin_adjudicated_repro_values():
    # The exact sparse-gate row of the issue #1498 repro
    # (DeepSeek-V4-Flash | tp1 ep8 cp8 | b=1 isl=8192): top_last 0.048698 —
    # now resolvable on the reuse-dependent 0.5.12. Keyed
    # [native][step][isl][bs][score_mode].
    db = get_database("b200_sxm", "sglang", "0.5.12")
    rows = _load_topk_calib_rows(db)
    leaf = rows[_FLASH_NATIVE_HEADS][0][8192][1]
    assert leaf["v1_top_last"]["latency"] == pytest.approx(0.048698, rel=1e-9)
    # The percolated 1024-isl point of the repro: flat == top_last == 0.0, so
    # the DELTA the engine derives there is zero (the repro's tl_perc 0.0).
    perc = rows[_FLASH_NATIVE_HEADS][0][1024][1]
    assert perc["v1_top_last"]["latency"] == 0.0
    assert perc["v1_flat"]["latency"] == 0.0
