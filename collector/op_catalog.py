# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Table→family identity map from the op-backend catalog (Collector V3 §2).

The catalog file comes from the op-backend-facts registry (PR #1345, on main).
``load_family_map`` returns ``None`` when the catalog is absent (e.g. a
stripped or partial checkout); callers must then reject configurations that
need family identity (fail-closed, spec §4).
"""

from __future__ import annotations

from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).with_name("op_backend_catalog.yaml")


def load_family_map(path: str | Path = CATALOG_PATH) -> dict[str, str] | None:
    """Return {perf-table stem -> family}, or None when the catalog is absent."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        return None
    with catalog_path.open(encoding="utf-8") as catalog_file:
        catalog = yaml.safe_load(catalog_file) or {}
    if not isinstance(catalog, dict):
        raise ValueError("op catalog must be a mapping at the top level")  # noqa: TRY004 (caught alongside ValueError by validate_resolution)
    families = catalog.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("op catalog must define a non-empty families list")
    family_map: dict[str, str] = {}
    for entry in families:
        family = entry.get("family") if isinstance(entry, dict) else None
        op_files = entry.get("op_files") if isinstance(entry, dict) else None
        if not isinstance(family, str) or not isinstance(op_files, list) or not op_files:
            raise ValueError(f"invalid catalog family entry: {entry!r}")
        for op_file in op_files:
            if op_file in family_map:
                raise ValueError(f"table {op_file} is mapped to two families: {family_map[op_file]}, {family}")
            family_map[op_file] = family
    return family_map


def family_for_perf_file(perf_filename: str, family_map: dict[str, str]) -> str | None:
    """Map a perf filename ('gemm_perf.txt'/'.parquet' or a PerfFile) to its family."""
    return family_map.get(Path(str(perf_filename)).stem)
