# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Op-level guards for per-dtype tc_flops resolution (issue #1398).

The shared fixtures deliberately pin ``int8/fp8/fp4_tc_flops`` at the
historical 1/2/4 ratios of bf16 so pre-existing numeric assertions stay
valid — which also means those tests cannot distinguish the new per-dtype
resolution from the old ``bfloat16_tc_flops * compute_factor`` scaling.
The tests here break the ratio on purpose: they monkeypatch a divergent
per-dtype entry (mirroring b300, where ``fp4_tc_flops`` is 14 PFLOPS versus
``bf16*4`` = 9) and assert the SOL math follows the YAML entry, so any
regression back to compute-factor scaling fails loudly at the op level.

Also covers the eager-resolution contract: a missing ``*_tc_flops`` entry
rejects the query at entry in every database mode — including SILICON, where
an exact table hit never invokes the SOL closure — keeping the Python engine
aligned with Rust, which resolves flops with ``?`` before the table lookup.
"""

from __future__ import annotations

import math
import typing

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.errors import MissingSystemFlopsError


class TestDivergentRatioSol:
    """SOL math must read the per-dtype YAML entry, not scale bf16."""

    def test_moe_sol_uses_fp4_entry_not_bf16_ratio(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        # 10x bf16 — far from the fixture's 4x — so a compute-factor
        # regression produces a 2.5x-wrong sol_math.
        monkeypatch.setitem(gpu, "fp4_tc_flops", gpu["bfloat16_tc_flops"] * 10)

        num_tokens, hidden_size, inter_size = 512, 4096, 16384
        topk, num_experts = 2, 8
        # Compute-bound shape (sol_math >> sol_mem on the fixture spec even at
        # 10x fp4 flops), so the SOL scalar IS sol_math — the FLOPs term the
        # retired SOL_FULL tuple used to expose directly.
        sol_scalar = float(
            comprehensive_perf_db.query_moe(
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                1,
                1,
                common.MoEQuantMode.nvfp4,
                "uniform",
                database_mode=common.DatabaseMode.SOL,
            )
        )

        ops = num_tokens * topk * hidden_size * inter_size * 3 * 2
        assert math.isclose(sol_scalar, ops / gpu["fp4_tc_flops"] * 1000, rel_tol=1e-9)
        legacy_math = ops / (gpu["bfloat16_tc_flops"] * common.MoEQuantMode.nvfp4.value.compute) * 1000
        assert not math.isclose(sol_scalar, legacy_math, rel_tol=1e-2)

    def test_generation_attention_sol_uses_fp8_entry(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "fp8_tc_flops", gpu["bfloat16_tc_flops"] * 5)

        b, s, n, n_kv = 4, 4096, 32, 8
        # Compute-bound on the fixture spec (1 TFLOPS : 1 TB/s), so the SOL
        # scalar IS sol_math (the retired SOL_FULL tuple exposed it directly).
        sol_scalar = float(
            comprehensive_perf_db.query_generation_attention(
                b,
                s,
                n,
                n_kv,
                common.KVCacheQuantMode.fp8,
                database_mode=common.DatabaseMode.SOL,
            )
        )

        ops = 2 * b * n * 128 * 2 * (s - 1)
        assert math.isclose(sol_scalar, ops / gpu["fp8_tc_flops"] * 1000, rel_tol=1e-9)


class TestEagerResolution:
    """A missing entry rejects the query at entry in EVERY mode (Rust parity)."""

    @pytest.mark.parametrize(
        "database_mode",
        [common.DatabaseMode.SOL, common.DatabaseMode.HYBRID, common.DatabaseMode.SILICON],
    )
    def test_moe_missing_fp4_entry_rejected_in_all_modes(self, comprehensive_perf_db, monkeypatch, database_mode):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp4_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp4_tc_flops"):
            comprehensive_perf_db.query_moe(
                8,
                1024,
                4096,
                2,
                8,
                1,
                1,
                common.MoEQuantMode.nvfp4,
                "uniform",
                database_mode=database_mode,
            )

    def test_wideep_generation_mla_validates_derived_dtype_not_label(self, comprehensive_perf_db, monkeypatch):
        """The generation SOL dtype is re-derived from the kv-cache dtype; the
        fmha label is inert. The eager check must follow the derivation: an
        fp8 KV with a bf16 label needs fp8_tc_flops (no exact-hit bypass),
        while a bf16 KV with an fp8 label must not be falsely rejected."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            comprehensive_perf_db.query_wideep_generation_mla(
                4,
                1024,
                8,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.bfloat16,
                database_mode=common.DatabaseMode.SOL,
            )

        result = comprehensive_perf_db.query_wideep_generation_mla(
            4,
            1024,
            8,
            common.KVCacheQuantMode.bfloat16,
            common.FMHAQuantMode.fp8,
            database_mode=common.DatabaseMode.SOL,
        )
        assert float(result) > 0

    def test_pre_sm89_fp8_kv_derives_bf16_pipeline(self, comprehensive_perf_db, monkeypatch):
        """Ampere-class hardware has no fp8 tensor cores: the decode kernel
        dequantizes fp8 KV and runs the MMA on the bf16 pipeline, so fp8-kv
        generation queries must neither demand fp8_tc_flops nor price the
        fp8 rate (a100 ships 2,534 measured fp8-kv generation-MLA rows
        collected exactly that way)."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "sm_version", 80)
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        # Distinct shape from the fp8-entry test above: these PerfDatabase
        # query methods are lru_cached, and identical args would replay the
        # earlier fp8-pipeline result instead of re-deriving under sm=80.
        # Compute-bound shape -> the SOL scalar IS sol_math.
        sol_scalar = float(
            comprehensive_perf_db.query_generation_attention(
                2,
                2048,
                32,
                8,
                common.KVCacheQuantMode.fp8,
                database_mode=common.DatabaseMode.SOL,
            )
        )
        ops = 2 * 2 * 32 * 128 * 2 * (2048 - 1)
        assert math.isclose(sol_scalar, ops / gpu["bfloat16_tc_flops"] * 1000, rel_tol=1e-9)

    def test_sm89_plus_fp8_kv_still_requires_fp8_entry(self, comprehensive_perf_db, monkeypatch):
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.setitem(gpu, "sm_version", 90)
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            comprehensive_perf_db.query_generation_attention(
                2, 1024, 32, 8, common.KVCacheQuantMode.fp8, database_mode=common.DatabaseMode.SOL
            )

    def test_context_attention_missing_fp8_entry_rejected(self, comprehensive_perf_db, monkeypatch):
        """attn_dtype family: context attention resolves the fmha label
        strictly (no kv-derivation gate on the prefill path)."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            comprehensive_perf_db.query_context_attention(
                2,
                512,
                0,
                32,
                8,
                common.KVCacheQuantMode.fp8,
                common.FMHAQuantMode.fp8,
                database_mode=common.DatabaseMode.SOL,
            )

    def test_mla_bmm_missing_fp4_entry_rejected(self, comprehensive_perf_db, monkeypatch):
        """bmm_dtype family: MLA-BMM resolves its GEMM quant strictly."""
        gpu = comprehensive_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp4_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp4_tc_flops"):
            comprehensive_perf_db.query_mla_bmm(
                16, 16, common.GEMMQuantMode.nvfp4, database_mode=common.DatabaseMode.SOL
            )

    def test_gemm_missing_fp8_entry_rejected_in_silicon_mode(self, stub_perf_db, monkeypatch):
        """SILICON specifically: an exact table hit must not bypass the check."""
        gpu = stub_perf_db.system_spec["gpu"]
        monkeypatch.delitem(gpu, "fp8_tc_flops", raising=False)

        with pytest.raises(MissingSystemFlopsError, match="fp8_tc_flops"):
            stub_perf_db.query_gemm(16, 1024, 1024, common.GEMMQuantMode.fp8, database_mode=common.DatabaseMode.SILICON)


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
