# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shipped-data guard for per-dtype tc_flops resolution (issue #1398).

The op-level SOL/eager-resolution tests that used to live here (divergent
per-dtype ratios feeding sol_math, ``MissingSystemFlopsError`` raised at
query entry in every mode, the sm<89 fp8-KV bf16-pipeline derivation)
verified the retired Python query stack on monkeypatched in-memory system
specs; that math and its eager ``?``-style flops resolution now live solely
in the compiled engine (#1357 PR-5), anchored by the frozen parity goldens
and tests/cross_package/test_query_shim_baseline.py. What remains here is
the shipped-data/CI invariant, which reads disk directly.
"""

from __future__ import annotations

import math
import typing

from aiconfigurator.sdk import common


class TestShippedDataImpliesYamlKey:
    """Every dtype demanded by SHIPPED silicon data must have its YAML entry.

    Under strict per-dtype resolution a missing ``*_tc_flops`` entry hard-fails
    every query of that quant at entry — correct for unsupported hardware, but
    a data-collection campaign that lands rows for a dtype whose key the
    system YAML forgot would brick that precision for users at query time
    (the pre-#1418 b60 shipped exactly that: fp8 gemm/moe data with no
    fp8_tc_flops). This sweep moves the failure to CI: data implies key.

    Labels are resolved through the quant enums' own ``compute_dtype`` (so new
    modes are covered automatically); memory-only labels (kv/comm) carry
    ``None`` and are skipped, as are loader-remapped raw labels listed in
    ``_EXTRA_LABELS``.
    """

    _QUANT_COLUMNS = frozenset(
        {
            "gemm_dtype",
            "moe_dtype",
            "attn_dtype",  # context/generation attention tables -> FMHAQuantMode
            "bmm_dtype",  # MLA-BMM tables -> GEMMQuantMode
            "mla_dtype",
            "quant_mode",
            "gemm_quant_mode",
            "fmha_quant_mode",
        }
    )
    # Raw collector labels that predate / bypass the enum names.
    _EXTRA_LABELS: typing.ClassVar[dict[str, str]] = {"fp8_e4m3": "fp8", "float16": "bfloat16"}

    def test_every_shipped_quant_label_has_its_flops_key(self):
        import pathlib

        import pyarrow.parquet as pq
        import yaml

        from aiconfigurator.sdk import perf_database

        label_dtype: dict[str, str | None] = dict(self._EXTRA_LABELS)
        for enum in (common.GEMMQuantMode, common.MoEQuantMode, common.FMHAQuantMode):
            for mode in enum:
                label_dtype.setdefault(mode.value.name, mode.value.compute_dtype)

        key_for = {
            "bfloat16": "bfloat16_tc_flops",
            "int8": "int8_tc_flops",
            "fp8": "fp8_tc_flops",
            "fp4": "fp4_tc_flops",
        }

        systems_root = pathlib.Path(perf_database.get_systems_paths()[0])
        problems: list[str] = []
        for spec_path in sorted(systems_root.glob("*.yaml")):
            system = spec_path.stem
            data_dir = systems_root / "data" / system
            if not data_dir.is_dir():
                continue
            gpu = (yaml.safe_load(spec_path.read_text()) or {}).get("gpu") or {}
            needed: dict[str, set[str]] = {}
            for parquet_path in data_dir.rglob("*.parquet"):
                try:
                    schema_names = set(pq.read_schema(parquet_path).names)
                except Exception as exc:
                    # An unreadable shipped file must fail the invariant, not
                    # silently drop out of it — it could be hiding exactly the
                    # data-without-key mismatch this test exists to catch.
                    problems.append(f"{system}: unreadable shipped parquet {parquet_path}: {exc}")
                    continue
                cols = sorted(self._QUANT_COLUMNS & schema_names)
                if not cols:
                    continue
                table = pq.read_table(parquet_path, columns=cols)
                for col in cols:
                    for label in table.column(col).unique().to_pylist():
                        dtype = label_dtype.get(str(label))
                        if dtype is not None:
                            needed.setdefault(dtype, set()).add(str(label))
            for dtype, labels in sorted(needed.items()):
                key = key_for[dtype]
                value = gpu.get(key)
                if value is None or not (value > 0 and math.isfinite(value)):
                    problems.append(
                        f"{system}: shipped data uses {sorted(labels)} (compute dtype '{dtype}') "
                        f"but {system}.yaml has no usable '{key}' — every query of those quants "
                        f"would raise MissingSystemFlopsError"
                    )
        assert not problems, "\n".join(problems)
