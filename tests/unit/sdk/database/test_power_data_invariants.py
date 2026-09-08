# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Power/energy data invariants — version-agnostic by construction.

Policy (2026-08): power/energy tests pin no values and bind to no specific
backend version. They assert query-surface invariants over WHATEVER power
data is currently shipped: every parquet that carries power columns must
store either a finite positive ``power`` / ``power_limit`` measurement pair
or the exact ``0.0`` / ``0.0`` unavailable-measurement sentinel. The energy
MATH is anchored by the rust synthetic oracles on power-carrying fixtures
(``energy_test_fixtures`` tests in ``operators/{gemm,attention}.rs``); this
test guards the shipped data plane those models consume. If no power-carrying
parquet is shipped at all, the suite records that state explicitly instead
of passing vacuously.
"""

from pathlib import Path

import numpy as np
import pandas as pd
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


def _invalid_power_pairs(frame: pd.DataFrame) -> pd.Series:
    power = frame["power"]
    power_limit = frame["power_limit"]
    finite = np.isfinite(power) & np.isfinite(power_limit)
    measured = (power > 0.0) & (power_limit > 0.0)
    unmeasured = (power == 0.0) & (power_limit == 0.0)
    return ~(finite & (measured | unmeasured))


def test_power_columns_satisfy_energy_model_input_contract():
    files = _power_carrying_files()
    if not files:
        pytest.skip("no power-carrying parquet shipped (energy path idle)")
    problems = []
    for path in files:
        rel = path.relative_to(_DATA_ROOT)
        schema = pq.read_schema(path)
        missing = [column for column in _POWER_COLUMNS if column not in schema.names]
        if missing:
            problems.append(f"{rel}: missing paired columns {missing}")
            continue
        table = pq.read_table(path, columns=list(_POWER_COLUMNS))
        frame = table.to_pandas()
        bad = _invalid_power_pairs(frame)
        if bad.any():
            problems.append(
                f"{rel}: {int(bad.sum())} invalid power pairs "
                "(expected positive/positive measurement or 0.0/0.0 sentinel)"
            )
    assert not problems, "power data violates the energy-model input contract:\n" + "\n".join(problems)


def test_power_pair_contract_accepts_only_measurements_or_paired_zero_sentinel():
    frame = pd.DataFrame(
        {
            "power": [100.0, 0.0, 0.0, 100.0, -1.0, np.nan, 100.0],
            "power_limit": [1000.0, 0.0, 1000.0, 0.0, 1000.0, 1000.0, np.inf],
        }
    )

    assert _invalid_power_pairs(frame).tolist() == [False, False, True, True, True, True, True]
