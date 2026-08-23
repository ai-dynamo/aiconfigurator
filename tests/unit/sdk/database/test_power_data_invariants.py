# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power/energy data invariants — version-agnostic by construction.

Policy (2026-08): power/energy tests pin no values and bind to no specific
backend version. They assert query-surface invariants over WHATEVER power
data is currently shipped: every parquet that carries power columns must
satisfy the energy model's input contract (finite, non-negative power;
positive power limit). The energy MATH is anchored by the rust synthetic
oracles on power-carrying fixtures (``energy_test_fixtures`` tests in
``operators/{gemm,attention}.rs``); this test guards the shipped data plane
those models consume. If no power-carrying parquet is shipped at all, the
suite records that state explicitly instead of passing vacuously.
"""

from pathlib import Path

import pyarrow.parquet as pq
import pytest

import aiconfigurator_core

pytestmark = pytest.mark.unit

_DATA_ROOT = Path(aiconfigurator_core.__file__).parent / "systems" / "data"
_POWER_COLUMNS = ("power", "power_limit")


def _power_carrying_files() -> list[Path]:
    files = []
    for path in sorted(_DATA_ROOT.rglob("*_perf.parquet")):
        schema = pq.read_schema(path)
        if any(col in schema.names for col in _POWER_COLUMNS):
            files.append(path)
    return files


def test_power_columns_satisfy_energy_model_input_contract():
    files = _power_carrying_files()
    if not files:
        pytest.skip("no power-carrying parquet shipped (energy path idle)")
    problems = []
    for path in files:
        rel = path.relative_to(_DATA_ROOT)
        table = pq.read_table(
            path, columns=[c for c in _POWER_COLUMNS if c in pq.read_schema(path).names]
        )
        frame = table.to_pandas()
        if "power" in frame:
            bad = frame["power"].isna() | (frame["power"] < 0)
            if bad.any():
                problems.append(f"{rel}: {int(bad.sum())} rows with NaN/negative power")
        if "power_limit" in frame:
            bad = frame["power_limit"].isna() | (frame["power_limit"] <= 0)
            if bad.any():
                problems.append(f"{rel}: {int(bad.sum())} rows with NaN/non-positive power_limit")
    assert not problems, "power data violates the energy-model input contract:\n" + "\n".join(
        problems
    )
