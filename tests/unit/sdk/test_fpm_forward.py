# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for forward_model="fpm": loader, FPMForwardOp, centralized
model rewrite, and the explicit mixed-step branch.

Synthetic parquet/metadata pairs are written directly from the documented
``aic_fpm_forward_perf`` schema (v5) — deliberately NOT via collector code, so
this suite doubles as the producer/consumer contract test on the modeling
side of the module boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aiconfigurator.sdk import common, models
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from aiconfigurator.sdk.operations import FPMForwardOp
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator_core.sdk.operations.fpm_forward import _CELL_MATCH_COLUMNS

pytestmark = pytest.mark.unit

SYSTEM = "h200_sxm"
BACKEND = "vllm"
VERSION = "test-fpm-version"
MODEL_PATH = "test-org/test-model"

import aiconfigurator_core

_CORE_SYSTEMS = os.path.join(os.path.dirname(aiconfigurator_core.__file__), "systems")


def _row(
    workload_kind: str,
    batch_size: int,
    total_prefill_tokens: int,
    total_kv_read_tokens: int,
    latency_ms: float,
    *,
    model_path: str = MODEL_PATH,
    identity: dict | None = None,
) -> dict:
    base_identity = {
        "gemm_quant_mode": "bfloat16",
        "moe_quant_mode": "",
        "fmha_quant_mode": "",
        "comm_quant_mode": "half",
        "kv_cache_dtype": "bfloat16",
        "tp": "1",
        "pp": "1",
        "dp": "1",
        "moe_tp": "1",
        "moe_ep": "1",
        "cp": "1",
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    if identity:
        base_identity.update(identity)
    # Identity dicts built from an op's _match_identity carry str(bool)
    # ("True"/"False"); the parquet columns are REAL booleans.
    for knob in ("enable_wideep", "enable_eplb"):
        if isinstance(base_identity[knob], str):
            base_identity[knob] = base_identity[knob] == "True"
    return {
        "cell_id": f"fpm-test-{workload_kind}",
        "model_path": model_path,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
        "weight_quantization": base_identity["gemm_quant_mode"],
        "gemm_quant_mode": base_identity["gemm_quant_mode"],
        "moe_quant_mode": base_identity["moe_quant_mode"] or None,
        "fmha_quant_mode": base_identity["fmha_quant_mode"] or None,
        "comm_quant_mode": base_identity["comm_quant_mode"] or None,
        "kv_cache_dtype": base_identity["kv_cache_dtype"],
        "tp": int(base_identity["tp"]),
        "pp": int(base_identity["pp"]),
        "dp": int(base_identity["dp"]),
        "moe_tp": int(base_identity["moe_tp"]),
        "moe_ep": int(base_identity["moe_ep"]),
        "cp": int(base_identity["cp"]),
        "moe_backend": base_identity["moe_backend"],
        "attention_backend": base_identity["attention_backend"],
        "enable_wideep": base_identity["enable_wideep"],
        "enable_eplb": base_identity["enable_eplb"],
        "workload_kind": workload_kind,
        "batch_size": batch_size,
        "total_prefill_tokens": total_prefill_tokens,
        "total_kv_read_tokens": total_kv_read_tokens,
        "partition_policy": "balanced_v1",
        "latency_ms": latency_ms,
    }


def _default_rows() -> list[dict]:
    return [
        # prefill, P=0
        _row("prefill", 1, 512, 0, 10.0),
        _row("prefill", 1, 1024, 0, 20.0),
        _row("prefill", 2, 1024, 0, 25.0),
        # prefill with past-KV
        _row("prefill", 1, 512, 512, 15.0),
        _row("prefill", 1, 1024, 512, 27.0),
        # decode
        _row("decode", 1, 0, 1024, 5.0),
        _row("decode", 2, 0, 2048, 8.0),
        _row("decode", 2, 0, 4096, 12.0),
        _row("decode", 4, 0, 4096, 14.0),
    ]


def _write_pair(data_dir: str, rows: list[dict], *, sidecar_overrides: dict | None = None) -> str:
    os.makedirs(data_dir, exist_ok=True)
    parquet_path = os.path.join(data_dir, "fpm_forward_perf.parquet")
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    with open(parquet_path, "rb") as handle:
        parquet_sha = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "schema_name": "aic_fpm_forward_perf",
        "schema_version": 6,
        "coordinate_system": "iteration_totals_balanced_v1",
        "measurement_policy": "dynamo_native_single_sample_v1",
        "row_count": len(rows),
        "parquet_sha256": parquet_sha,
        "system": SYSTEM,
        "backend": BACKEND,
        "backend_version": VERSION,
    }
    metadata.update(sidecar_overrides or {})
    with open(os.path.join(data_dir, "fpm_forward_perf.metadata.json"), "w") as handle:
        json.dump(metadata, handle)
    return parquet_path


class _FakeDatabase:
    """Minimal stand-in exposing exactly what FPMForwardOp.load_data reads."""

    def __init__(self, systems_root: str):
        self.systems_root = systems_root
        self.system = SYSTEM
        self.backend = BACKEND
        self.version = VERSION
        self.system_spec = {"data_dir": "data"}


@pytest.fixture()
def fake_db(tmp_path):
    def _build(rows: list[dict] | None = None, *, write: bool = True, sidecar_overrides: dict | None = None):
        root = tmp_path / "systems"
        data_dir = root / "data" / BACKEND / VERSION
        if write:
            _write_pair(
                str(data_dir), rows if rows is not None else _default_rows(), sidecar_overrides=sidecar_overrides
            )
        else:
            os.makedirs(data_dir, exist_ok=True)
        return _FakeDatabase(str(root))

    yield _build
    FPMForwardOp.clear_cache()


def _model_config(**overrides) -> sdk_config.ModelConfig:
    defaults = dict(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
    )
    defaults.update(overrides)
    return sdk_config.ModelConfig(**defaults)


def _make_op(phase: str, model_config=None, model_path: str = MODEL_PATH) -> FPMForwardOp:
    if phase == "prefill":
        sol = lambda b, tp, tk: tp * (1.0 + (tp + tk) / max(b, 1.0))
    else:
        sol = lambda b, tk: 1e9 + 100.0 * tk
    return FPMForwardOp(phase, model_config or _model_config(), model_path, sol_fn=sol, weight_bytes=1e9)


# ---------------------------------------------------------------------------
# Loader + op query
# ---------------------------------------------------------------------------


class TestFPMForwardOpQuery:
    def test_exact_match_prefill_p0(self, fake_db):
        result = _make_op("prefill").query(fake_db(), batch_size=1, s=512, prefix=0)
        assert float(result) == pytest.approx(10.0)
        assert result.energy == 0.0
        assert result.source == "silicon"

    def test_exact_match_prefill_past_kv(self, fake_db):
        result = _make_op("prefill").query(fake_db(), batch_size=1, s=512, prefix=512)
        assert float(result) == pytest.approx(15.0)

    def test_exact_match_decode(self, fake_db):
        # batch=2, per-request KV length 1024 -> total_kv 2048
        result = _make_op("decode").query(fake_db(), batch_size=2, s=1024)
        assert float(result) == pytest.approx(8.0)

    def test_decode_interpolation_on_collected_site(self, fake_db):
        # batch=2 is a collected site; TK=3072 sits between 2048 (8ms) and
        # 4096 (12ms) -> RAW linear blend = 10ms exactly.
        result = _make_op("decode").query(fake_db(), batch_size=2, s=1536)
        assert float(result) == pytest.approx(10.0)

    def test_prefill_interpolation_on_collected_site(self, fake_db):
        # site (batch=1, kv=0); new-token curve between 512 (10ms) and 1024 (20ms).
        result = _make_op("prefill").query(fake_db(), batch_size=1, s=768, prefix=0)
        assert float(result) == pytest.approx(15.0)

    def test_unknown_batch_site_transfer_is_finite(self, fake_db):
        # batch=3 is not collected; in-domain query must resolve via
        # cross-site util transfer to something finite and positive.
        result = _make_op("decode").query(fake_db(), batch_size=3, s=1024)
        assert math.isfinite(float(result)) and float(result) > 0

    def test_out_of_domain_raises(self, fake_db):
        op = _make_op("decode")
        db = fake_db()
        with pytest.raises(PerfDataNotAvailableError, match="outside the collected domain"):
            op.query(db, batch_size=2, s=999999)
        with pytest.raises(PerfDataNotAvailableError, match="outside the collected domain"):
            op.query(db, batch_size=64, s=64)

    def test_beam_width_rejected(self, fake_db):
        with pytest.raises(PerfDataNotAvailableError, match="beam"):
            _make_op("decode").query(fake_db(), batch_size=2, s=1024, beam_width=2)

    def test_missing_parquet_raises_perf_data_error(self, fake_db):
        with pytest.raises(PerfDataNotAvailableError):
            _make_op("decode").query(fake_db(write=False), batch_size=2, s=1024)

    def test_identity_mismatch_raises(self, fake_db):
        op = _make_op("decode", model_config=_model_config(tp_size=2))
        with pytest.raises(PerfDataNotAvailableError, match="No FPM cell matches"):
            op.query(fake_db(), batch_size=2, s=1024)

    def test_prefill_batch_flatness_certificate(self):
        from aiconfigurator_core.sdk.operations.fpm_forward import _prefill_batch_axis_is_flat

        flat = {
            1: {1024: {0: 10.0}, 2048: {0: 20.0}, 4096: {0: 40.0}},
            2: {1024: {0: 10.2}, 2048: {0: 20.4}, 4096: {0: 40.8}},
            4: {1024: {0: 10.4}, 2048: {0: 20.8}, 4096: {0: 41.6}},
        }
        assert _prefill_batch_axis_is_flat(flat)
        # A real batch effect (>5% median) fails the certificate.
        bumpy = {1: flat[1], 2: {t: {0: v[0] * 1.2} for t, v in flat[1].items()}}
        assert not _prefill_batch_axis_is_flat(bumpy)
        # A single collected batch offers no evidence.
        assert not _prefill_batch_axis_is_flat({1: flat[1]})
        # Fewer than 3 overlapping points: insufficient evidence.
        assert not _prefill_batch_axis_is_flat({1: {1024: {0: 10.0}, 2048: {0: 20.0}}, 2: {2048: {0: 20.2}}})

    def _flat_prefill_rows(self):
        rows = []
        for b, bump in ((1, 1.0), (2, 1.02), (4, 1.04)):
            for total, lat in ((1024, 10.0), (2048, 20.0), (4096, 40.0)):
                rows.append(_row("prefill", b, total, 0, lat * bump))
        return rows

    def test_certified_prefill_batch_clamp_answers_above_the_ceiling(self, fake_db):
        # Real steps can schedule more whole prefills than the collected
        # batch ceiling (short sequences). With the flatness certificate the
        # query clamps its batch coordinate to the ceiling, keeping the TRUE
        # token total — the regime coordinate — untouched.
        db = fake_db(self._flat_prefill_rows() + [r for r in _default_rows() if r["workload_kind"] == "decode"])
        op = _make_op("prefill")
        assert op._load_cell(db)["prefill_batch_clamp_max"] == 4
        clamped = float(op.query_totals(db, batch_size=16, total_prefill_tokens=4096, total_kv_read_tokens=0))
        ceiling = float(op.query_totals(db, batch_size=4, total_prefill_tokens=4096, total_kv_read_tokens=0))
        assert clamped == pytest.approx(ceiling) == pytest.approx(41.6)
        # The other axes stay honestly gated: totals beyond the box still miss.
        with pytest.raises(PerfDataNotAvailableError, match="outside the collected domain"):
            op.query_totals(db, batch_size=16, total_prefill_tokens=16384, total_kv_read_tokens=0)

    def test_uncertified_prefill_batch_stays_hard_gated(self, fake_db):
        # Default fixture rows carry a single sparse batch pair -> no
        # certificate -> the domain gate rejects exactly as before.
        db = fake_db()
        op = _make_op("prefill")
        assert op._load_cell(db)["prefill_batch_clamp_max"] is None
        with pytest.raises(PerfDataNotAvailableError, match="outside the collected domain"):
            op.query_totals(db, batch_size=16, total_prefill_tokens=512, total_kv_read_tokens=0)

    def test_decode_batch_is_never_clamped(self, fake_db):
        # The decode batch axis carries a REAL regime cliff (the Task B
        # partition); a flat-prefill certificate must not leak into it.
        db = fake_db(self._flat_prefill_rows() + [r for r in _default_rows() if r["workload_kind"] == "decode"])
        op = _make_op("decode")
        with pytest.raises(PerfDataNotAvailableError, match="outside the collected domain"):
            op.query_totals(db, batch_size=4096, total_kv_read_tokens=4096)

    def test_decode_regime_boundary_detection(self):
        from aiconfigurator_core.sdk.operations.fpm_forward import _detect_decode_regime_boundary

        graph = {kv: 10.0 + i for i, kv in enumerate((1024, 2048, 4096))}
        eager = {kv: 3.0 * (10.0 + i) for i, kv in enumerate((1024, 2048, 4096))}
        # Cliff pair 512/513 (~3x on 3 overlapping KV points) -> boundary 512.
        assert _detect_decode_regime_boundary({512: graph, 513: eager}) == 512
        # Pad-up neighbours (~10%) never mint a cliff.
        pad = {kv: v * 1.1 for kv, v in graph.items()}
        assert _detect_decode_regime_boundary({8: graph, 9: pad}) is None
        # Fewer than 3 overlapping points: one bad row cannot mint a cliff.
        assert _detect_decode_regime_boundary({512: {1024: 10.0, 2048: 11.0}, 513: {1024: 30.0, 2048: 33.0}}) is None
        # Two cliffs do not look like one capture surface: loud error.
        with pytest.raises(ValueError, match="ambiguous decode regime cliffs"):
            _detect_decode_regime_boundary(
                {512: graph, 513: eager, 1024: eager, 1025: {k: v * 3 for k, v in eager.items()}}
            )

    def _cliff_rows(self):
        rows = []
        for b, base in ((496, 9.5), (497, 10.0), (512, 10.5)):
            for i, kv in enumerate((1024, 2048, 4096)):
                rows.append(_row("decode", b, 0, kv, base + i))
        for b, base in ((513, 31.0), (1024, 62.0)):
            for i, kv in enumerate((1024, 2048, 4096)):
                rows.append(_row("decode", b, 0, kv, base + 3 * i))
        return rows

    def test_decode_regime_routing_separates_the_cliff(self, fake_db):
        # The b=600 pathology: pre-split, k-NN mixes one eager vote (513)
        # with three graph votes (496/497/512) and answers ~44% of truth.
        # Post-split it interpolates among eager batches only.
        db = fake_db(self._cliff_rows())
        op = _make_op("decode")
        cell = op._load_cell(db)
        assert cell["decode_regime_boundary"] == 512
        assert set(cell["tables"]["decode_graph"]) == {496, 497, 512}
        assert set(cell["tables"]["decode_eager"]) == {513, 1024}
        # Exact hits on either side are untouched by the split.
        assert float(op.query_totals(db, batch_size=512, total_kv_read_tokens=2048)) == pytest.approx(11.5)
        assert float(op.query_totals(db, batch_size=513, total_kv_read_tokens=2048)) == pytest.approx(34.0)
        # Off-site queries route by regime: eager-level above, graph-level at
        # or below the boundary (coarse bands — transfer math is interp's).
        eager_q = float(op.query_totals(db, batch_size=600, total_kv_read_tokens=2048))
        graph_q = float(op.query_totals(db, batch_size=500, total_kv_read_tokens=2048))
        assert eager_q > 25.0
        assert graph_q < 15.0
        # In-between eager batches keep working (frontier semantics live
        # inside the eager sub-table).
        mid_q = float(op.query_totals(db, batch_size=800, total_kv_read_tokens=2048))
        assert mid_q > 25.0

    def test_decode_without_cliff_is_unsplit(self, fake_db):
        db = fake_db()
        op = _make_op("decode")
        cell = op._load_cell(db)
        assert cell["decode_regime_boundary"] is None
        assert "decode_graph" not in cell["tables"]

    def test_query_totals_addresses_raw_coordinates(self, fake_db):
        # Totals a mixed step schedules (chunk + riders) are generally not
        # divisible into a per-request (b, s, prefix) shape; query_totals
        # addresses the collected coordinates directly.
        db = fake_db()
        pre = _make_op("prefill").query_totals(db, batch_size=1, total_prefill_tokens=512, total_kv_read_tokens=0)
        assert float(pre) == pytest.approx(10.0)
        dec = _make_op("decode").query_totals(db, batch_size=2, total_kv_read_tokens=2048)
        assert float(dec) == pytest.approx(8.0)

    def test_query_totals_validates_phase_coordinates(self, fake_db):
        db = fake_db()
        with pytest.raises(ValueError, match="total_prefill_tokens"):
            _make_op("prefill").query_totals(db, batch_size=1, total_prefill_tokens=0, total_kv_read_tokens=0)
        with pytest.raises(ValueError, match="no prefill tokens"):
            _make_op("decode").query_totals(db, batch_size=2, total_prefill_tokens=8, total_kv_read_tokens=2048)
        with pytest.raises(ValueError, match="invalid FPM totals query"):
            _make_op("decode").query_totals(db, batch_size=0, total_kv_read_tokens=2048)

    @pytest.mark.parametrize(
        "identity,config_overrides",
        [
            ({"moe_backend": "flashinfer_cutlass"}, {"moe_backend": "flashinfer_cutlass"}),
            ({"attention_backend": "fa3"}, {"attention_backend": "fa3"}),
            ({"enable_eplb": True}, {"enable_eplb": True}),
        ],
    )
    def test_unemittable_vllm_identity_is_rejected_before_cell_selection(self, fake_db, identity, config_overrides):
        # Exercise the producer-to-consumer boundary with a row that exactly
        # matches a direct SDK request. The standard Task-to-generator path
        # cannot emit these vLLM settings yet, so selection must fail closed
        # instead of pricing a deployment AIC would not reproduce.
        rows = [_row("decode", 2, 0, 2048, 8.0, identity=identity)]
        op = _make_op("decode", model_config=_model_config(**config_overrides))
        with pytest.raises(PerfDataNotAvailableError, match="Task-to-generator"):
            op.query(fake_db(rows), batch_size=2, s=1024)

    def test_model_path_mismatch_never_falls_back(self, fake_db):
        # The identity match is unique, but the identity carries no model
        # fingerprint: borrowing the sole collected path would silently
        # answer for a different model (D1 resolved to exact-only).
        with pytest.raises(PerfDataNotAvailableError, match="never substitutes"):
            _make_op("decode", model_path="local/checkout/of/model").query(fake_db(), batch_size=2, s=1024)

    def test_exact_model_path_selects_among_collected(self, fake_db):
        rows = _default_rows() + [_row("decode", 2, 0, 2048, 9.0, model_path="other-org/other-model")]
        db = fake_db(rows)
        with pytest.raises(PerfDataNotAvailableError, match="No FPM cell matches"):
            _make_op("decode", model_path="unrelated/path").query(db, batch_size=2, s=1024)
        result = _make_op("decode", model_path="other-org/other-model").query(db, batch_size=2, s=1024)
        assert float(result) == pytest.approx(9.0)

    def test_pinned_backend_knob_cell_not_matched(self, fake_db):
        # A cell collected under a pinned MoE backend must not answer an
        # "auto" request even when path and quant/parallel identity match.
        rows = _default_rows()
        for row in rows:
            row["moe_backend"] = "deepep_moe"
        with pytest.raises(PerfDataNotAvailableError, match="No FPM cell matches"):
            _make_op("decode").query(fake_db(rows), batch_size=2, s=1024)

    def test_backend_knob_data_answers_matching_config(self, fake_db):
        # v6 backend knobs are ordinary identity columns: wideep-collected
        # rows answer a wideep config — and only that config.
        rows = [_row("decode", 2, 0, 2048, 8.0, identity={"enable_wideep": True})]
        db = fake_db(rows)
        cfg = _model_config(enable_wideep=True, moe_tp_size=1, moe_ep_size=1)
        result = _make_op("decode", model_config=cfg).query(db, batch_size=2, s=1024)
        assert float(result) == pytest.approx(8.0)
        with pytest.raises(PerfDataNotAvailableError, match="No FPM cell matches"):
            _make_op("decode").query(db, batch_size=2, s=1024)

    @pytest.mark.parametrize(
        "overrides,message",
        [
            # wideep is generator-emittable, so it reaches cell selection and
            # fails as an ordinary identity miss...
            ({"enable_wideep": True, "moe_tp_size": 1, "moe_ep_size": 1}, "No FPM cell matches"),
            # ...while pinned backends / EPLB are rejected earlier by the
            # deployment-identity gate (the standard Task-to-generator path
            # cannot emit them yet).
            ({"enable_eplb": True}, "Task-to-generator"),
            ({"moe_backend": "megamoe"}, "Task-to-generator"),
            ({"attention_backend": "fa3"}, "Task-to-generator"),
        ],
    )
    def test_off_baseline_config_is_a_data_miss(self, fake_db, overrides, message):
        # Backend knobs are identity columns: a config whose knob identity
        # was never collected fails as a data miss (sweeps skip the point) —
        # never silently rides on auto-collected curves.
        op = _make_op("decode", model_config=_model_config(**overrides))
        with pytest.raises(PerfDataNotAvailableError, match=message):
            op.query(fake_db(), batch_size=2, s=1024)

    def test_dp_identity_uses_local_batch(self, fake_db):
        rows = [
            _row("decode", 2, 0, 2048, 8.0, identity={"dp": "2"}),
        ]
        op = _make_op("decode", model_config=_model_config(attention_dp_size=2))
        # Local per-rank batch of 2 (global 4 across dp=2) hits the dp=2 row.
        result = op.query(fake_db(rows), batch_size=2, s=1024)
        assert float(result) == pytest.approx(8.0)

    def test_query_pass_baseline_samples_kv_floor(self, fake_db):
        # Decode KV-domain floor is 1024; baseline for batch=1 is the exact
        # (1, 1024) row.
        result = _make_op("decode").query_pass_baseline(fake_db(), batch_size=1)
        assert float(result) == pytest.approx(5.0)

    def test_query_pass_baseline_is_decode_only(self, fake_db):
        with pytest.raises(ValueError, match="decode-only"):
            _make_op("prefill").query_pass_baseline(fake_db(), batch_size=1)


class TestDegenerateSiteCoverageFallback:
    """The runtime grid emits orphan coordinates (e.g. batch=3 exists only at
    total_prefill_tokens=3, a max-batch straggler). A collected-but-degenerate
    site must answer only inside its own curve coverage; outside it, the
    query falls through to covering neighbour sites."""

    def _rows_with_degenerate_site(self):
        return _default_rows() + [_row("prefill", 3, 3, 0, 1.0)]

    def test_out_of_coverage_query_uses_neighbours(self, fake_db):
        db = fake_db(self._rows_with_degenerate_site())
        # (B=3, s=256 -> totals (3, 768, 0)): site (3,0) covers only TP=3.
        # Must transfer from the (1,0)/(2,0) neighbour curves (10-27ms range),
        # not extrapolate 256x from the 1.0ms orphan point.
        result = _make_op("prefill").query(db, batch_size=3, s=256, prefix=0)
        assert 5.0 < float(result) < 60.0

    def test_in_coverage_degenerate_site_still_answers_exactly(self, fake_db):
        db = fake_db(self._rows_with_degenerate_site())
        result = _make_op("prefill").query(db, batch_size=3, s=1, prefix=0)
        assert float(result) == pytest.approx(1.0)

    def test_default_engine_behavior_unchanged(self):
        from aiconfigurator_core.sdk.perf_interp import ScatteredSites

        assert ScatteredSites(site_axes=("n", "k"), curve_axis="m").own_curve_coverage_fallback is False


class TestFPMForwardLoaderValidation:
    def _query(self, db):
        return _make_op("decode").query(db, batch_size=2, s=1024)

    def test_missing_sidecar(self, fake_db, tmp_path):
        db = fake_db()
        os.remove(os.path.join(db.systems_root, "data", BACKEND, VERSION, "fpm_forward_perf.metadata.json"))
        with pytest.raises(ValueError, match="metadata sidecar"):
            self._query(db)

    def test_parquet_digest_mismatch(self, fake_db):
        db = fake_db(sidecar_overrides={"parquet_sha256": "0" * 64})
        with pytest.raises(ValueError, match="digest mismatch"):
            self._query(db)

    def test_unsupported_schema_version(self, fake_db):
        db = fake_db(sidecar_overrides={"schema_version": 4})
        with pytest.raises(ValueError, match="schema_version"):
            self._query(db)

    def test_row_count_mismatch(self, fake_db):
        db = fake_db(sidecar_overrides={"row_count": 3})
        with pytest.raises(ValueError, match="row_count"):
            self._query(db)

    def test_duplicate_row_key(self, fake_db):
        rows = _default_rows()
        rows.append(dict(rows[-1]))
        with pytest.raises(ValueError, match="duplicate"):
            self._query(fake_db(rows))

    def test_conflicting_backend_version(self, fake_db):
        rows = _default_rows()
        rows[0]["backend_version"] = "some-other-version"
        with pytest.raises(ValueError, match="backend_version"):
            self._query(fake_db(rows))

    @pytest.mark.parametrize(
        "key,value",
        [
            ("system", "gb200_nvl72"),
            ("backend", "sglang"),
            ("backend_version", "some-other-version"),
            ("measurement_policy", "multi_sample_v2"),
        ],
    )
    def test_contradictory_sidecar_identity_fails_loudly(self, fake_db, key, value):
        # The sidecar commit record names the identity it was published for;
        # a pair whose rows match the tree but whose sidecar names foreign
        # values is inconsistent and must be rejected before any row is read.
        db = fake_db(sidecar_overrides={key: value})
        with pytest.raises(ValueError, match=key):
            self._query(db)

    def test_misplaced_system_or_backend_fails_loudly(self, fake_db):
        # system/backend are in the physical row key but NOT the cell key: a
        # pair copied into the wrong tree would merge into the same cells and
        # silently serve wrong latencies. Pin them like backend_version.
        rows = _default_rows()
        rows[0]["system"] = "gb200_nvl72"
        with pytest.raises(ValueError, match="does not match the database system"):
            self._query(fake_db(rows))
        rows = _default_rows()
        rows[0]["backend"] = "sglang"
        with pytest.raises(ValueError, match="does not match the database backend"):
            self._query(fake_db(rows))

    def test_coordinate_collision_within_cell_fails_loudly(self, fake_db):
        # Same cell identity + same (phase, batch, tokens) but a different
        # cell_id passes the physical-row-key duplicate check yet targets the
        # SAME table slot. Silent last-wins would serve arbitrary latencies;
        # the loader must refuse (producing such rows is a collector bug).
        rows = _default_rows()
        clash = dict(rows[-1])
        clash["cell_id"] = "fpm-test-another-attempt"
        clash["latency_ms"] = clash["latency_ms"] * 2.0
        rows.append(clash)
        with pytest.raises(ValueError, match="collides with an earlier row"):
            self._query(fake_db(rows))

    @pytest.mark.parametrize("bad_latency", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
    def test_non_finite_or_non_positive_latency(self, fake_db, bad_latency):
        rows = _default_rows()
        rows[0]["latency_ms"] = bad_latency
        with pytest.raises(ValueError, match="latency"):
            self._query(fake_db(rows))

    def test_unknown_workload_kind(self, fake_db):
        rows = _default_rows()
        rows[0]["workload_kind"] = "mixed"
        with pytest.raises(ValueError, match="workload_kind"):
            self._query(fake_db(rows))

    def test_decode_row_with_prefill_tokens(self, fake_db):
        rows = _default_rows()
        rows[-1]["total_prefill_tokens"] = 64
        with pytest.raises(ValueError, match="decode point carrying prefill"):
            self._query(fake_db(rows))


# ---------------------------------------------------------------------------
# Centralized model rewrite
# ---------------------------------------------------------------------------


class TestForwardModelRewrite:
    def test_default_keeps_op_level_lists(self):
        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        assert model.forward_model == "op_level"
        assert len(model.context_ops) > 1
        assert not any(isinstance(op, FPMForwardOp) for op in model.context_ops)

    def test_fpm_rewrite_yields_exactly_one_op_per_phase(self):
        baseline = models.get_model("Qwen/Qwen3-0.6B", _model_config(), "vllm")
        expected_weights = float(sum(op.get_weights() for op in baseline.context_ops))

        model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), "vllm")
        assert model.forward_model == "fpm"
        assert [op._name for op in model.context_ops] == ["fpm_forward_prefill"]
        assert [op._name for op in model.generation_ops] == ["fpm_forward_decode"]
        assert all(isinstance(op, FPMForwardOp) for op in (*model.context_ops, *model.generation_ops))
        # Weight bytes captured from the original lists keep memory estimation intact.
        assert model.context_ops[0].get_weights() == pytest.approx(expected_weights)

    def test_unknown_forward_model_rejected(self):
        with pytest.raises(ValueError, match="Unknown forward_model"):
            models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="banana"), "vllm")

    def test_encoder_model_rejected(self):
        cfg = _model_config(forward_model="fpm")
        with pytest.raises(NotImplementedError, match="encoder"):
            models.get_model("Qwen/Qwen3-VL-2B-Instruct", cfg, "vllm")

    def test_mtp_rejected(self):
        cfg = _model_config(forward_model="fpm", nextn=1)
        with pytest.raises(NotImplementedError, match="MTP"):
            models.get_model("Qwen/Qwen3-0.6B", cfg, "vllm")


# ---------------------------------------------------------------------------
# Static + mixed-step integration through a real PerfDatabase/backend
# ---------------------------------------------------------------------------


@pytest.fixture()
def fpm_session(tmp_path):
    """A real PerfDatabase over a temp systems root holding ONLY fpm data,
    plus an fpm-mode model whose identity the rows are written to match."""
    systems_root = tmp_path / "systems"
    os.makedirs(systems_root, exist_ok=True)
    shutil.copy(os.path.join(_CORE_SYSTEMS, f"{SYSTEM}.yaml"), systems_root / f"{SYSTEM}.yaml")

    model = models.get_model("Qwen/Qwen3-0.6B", _model_config(forward_model="fpm"), BACKEND)
    identity = dict(zip(_CELL_MATCH_COLUMNS, model.context_ops[0]._match_identity, strict=True))

    isl, osl = 512, 2
    rows = [
        # static_ctx: batch=2 x isl new tokens, and the mixed ctx component (batch=1 x isl)
        _row("prefill", 2, 2 * isl, 0, 40.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, isl, 0, 22.0, model_path=model.model_path, identity=identity),
        # static_gen with osl=2 runs one decode step at s = isl+1.
        _row("decode", 2, 0, 2 * (isl + 1), 6.0, model_path=model.model_path, identity=identity),
        # mixed/genonly route through run_static(isl=isl+osl//2, osl=2), whose
        # decode step lands at s = isl + osl//2 + 1.
        _row("decode", 2, 0, 2 * (isl + osl // 2 + 1), 7.0, model_path=model.model_path, identity=identity),
        # Totals-coordinate rows for the mixed-step composition: the prefill
        # component queries (batch, chunk + gen_tokens, past_kv).
        _row("prefill", 1, isl + 2, 0, 23.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 0, 11.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 258, 256, 13.0, model_path=model.model_path, identity=identity),
        # CUDA-graph cliff pair at capture=2048 plus the eager plateau: the
        # regime is encoded in the data, the formula only addresses it.
        _row("prefill", 1, 2048, 0, 47.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 2049, 0, 99.0, model_path=model.model_path, identity=identity),
        _row("prefill", 1, 4096, 0, 99.0, model_path=model.model_path, identity=identity),
        # Decode coverage for the cliff test (gen=8 at isl=2048, osl=2).
        _row("decode", 8, 0, 1026, 6.5, model_path=model.model_path, identity=identity),
        _row("decode", 8, 0, 16400, 9.5, model_path=model.model_path, identity=identity),
    ]
    # data_dir comes from the system yaml ("data/h200_sxm").
    data_dir = os.path.join(systems_root, "data", SYSTEM, BACKEND, VERSION)
    _write_pair(data_dir, rows)

    database = PerfDatabase(SYSTEM, BACKEND, VERSION, systems_root=str(systems_root))
    backend = get_backend(BACKEND)
    yield model, database, backend, isl, osl
    FPMForwardOp.clear_cache()


class TestFPMStaticAndMixed:
    def test_static_ctx_uses_fpm_row(self, fpm_session):
        from aiconfigurator.sdk.config import RuntimeConfig
        from aiconfigurator.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python"),
            mode="static_ctx",
        )
        latency_dict = summary.get_context_latency_dict()
        assert list(latency_dict) == ["fpm_forward_prefill"]
        assert latency_dict["fpm_forward_prefill"] == pytest.approx(40.0)

    def test_static_gen_uses_fpm_row(self, fpm_session):
        from aiconfigurator.sdk.config import RuntimeConfig
        from aiconfigurator.sdk.inference_session import InferenceSession

        model, database, backend, isl, osl = fpm_session
        session = InferenceSession(model, database, backend)
        summary = session.run_static(
            runtime_config=RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python"),
            mode="static_gen",
        )
        latency_dict = summary.get_generation_latency_dict()
        assert list(latency_dict) == ["fpm_forward_decode"]
        # osl=2 -> one decode step at s=isl+1, repeat_count 1.
        assert latency_dict["fpm_forward_decode"] == pytest.approx(6.0)

    def test_mixed_step_is_prefill_plus_marginal_decode(self, fpm_session):
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python")
        total, energy, per_op, per_src = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        # ctx component prices the step's SCHEDULED TOTAL: one whole prefill
        # (isl tokens) plus 2 decode riders -> totals (1, isl+2, 0) = 23.0.
        # gen component rides the prefill pass, so only its marginal counts:
        # full decode at s=isl+osl//2+1 (7.0) minus the pass baseline at the
        # KV-domain floor (the 2*(isl+1)=1026 row, 6.0) -> 1.0.
        assert per_op["fpm_forward_prefill"] == pytest.approx(23.0)
        assert per_op["fpm_forward_decode"] == pytest.approx(1.0)
        assert total == pytest.approx(24.0)
        assert energy == 0.0
        assert set(per_src.values()) == {"silicon"}

    def test_mixed_step_total_crosses_the_graph_cliff(self, fpm_session):
        # Spec tests 1+2: the engine picks its regime from the step's TOTAL
        # scheduled tokens. ctx=2048 alone sits ON the capture boundary
        # (graph side, 47 ms); the same chunk with 8 decode riders crosses
        # it and must price on the eager plateau (99 ms).
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=2048, osl=osl, engine_step_backend="python")
        _, _, graph_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=0, isl=2048, osl=osl, prefix=0
        )
        assert graph_ops["fpm_forward_prefill"] == pytest.approx(47.0)
        _, _, eager_ops, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=2048, gen_tokens=8, isl=2048, osl=osl, prefix=0
        )
        assert eager_ops["fpm_forward_prefill"] == pytest.approx(99.0)
        assert eager_ops["fpm_forward_prefill"] > 2 * graph_ops["fpm_forward_prefill"]

    def test_mixed_step_chunks_average_exact_coordinates(self, fpm_session):
        # Spec tests 3+4: a chunked request prices each chunk at its own
        # (chunk + gen, past_kv) coordinates — (1, 258, 0)=11.0 and
        # (1, 258, 256)=13.0 for ctx=256 of isl=512 — and the component is
        # their per-iteration average, identical to querying the chunks
        # independently (no double billing, no averaging artifacts).
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python")
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=256, gen_tokens=2, isl=isl, osl=osl, prefix=0
        )
        prefill_op = model.context_ops[0]
        chunk1 = float(
            prefill_op.query_totals(database, batch_size=1, total_prefill_tokens=258, total_kv_read_tokens=0)
        )
        chunk2 = float(
            prefill_op.query_totals(database, batch_size=1, total_prefill_tokens=258, total_kv_read_tokens=256)
        )
        assert per_op["fpm_forward_prefill"] == pytest.approx((chunk1 + chunk2) / 2) == pytest.approx(12.0)
        assert total == pytest.approx(12.0 + 1.0)

    def test_mixed_step_gen_zero_prices_pure_chunk(self, fpm_session):
        # Spec test 5 (gen=0 degenerate): a pure-prefill step prices its own
        # totals with no decode marginal term.
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python")
        total, _, per_op, _ = backend._get_mix_step_latency(
            model, database, runtime_config, ctx_tokens=isl, gen_tokens=0, isl=isl, osl=osl, prefix=0
        )
        assert per_op == {"fpm_forward_prefill": pytest.approx(22.0)}
        assert total == pytest.approx(22.0)

    def test_genonly_step_keeps_full_decode_pass(self, fpm_session):
        # With no prefill work in the step there is no pass to ride on: the
        # decode component must keep its full standalone latency. A step
        # without prefill work is a GENONLY step now — `MixedStepInput`
        # requires context_tokens > 0, so the gen-only contract lives behind
        # `_get_genonly_step_latency` (the mixed entry raises instead of
        # silently rerouting).
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python")
        total, _energy, per_op, _src = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)
        with pytest.raises(ValueError, match="context_tokens must be positive"):
            backend._get_mix_step_latency(
                model, database, runtime_config, ctx_tokens=0, gen_tokens=2, isl=isl, osl=osl, prefix=0
            )

    def test_genonly_step_works_with_single_op(self, fpm_session):
        from aiconfigurator.sdk.config import RuntimeConfig

        model, database, backend, isl, osl = fpm_session
        runtime_config = RuntimeConfig(batch_size=2, beam_width=1, isl=isl, osl=osl, engine_step_backend="python")
        total, energy, per_op, _ = backend._get_genonly_step_latency(
            model, database, runtime_config, gen_tokens=2, isl=isl, osl=osl
        )
        assert per_op["fpm_forward_decode"] == pytest.approx(7.0)
        assert total == pytest.approx(7.0)
