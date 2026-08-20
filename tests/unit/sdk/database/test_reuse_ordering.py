# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the design-§6 reuse-ordering contract in ``_build_op_sources``
(Collector V3 PR 4, AIC-1503).

New source order per op file: (1) primary, (2) declared donors from the
REQUESTED version dir's ``reuse.yaml`` (any direction, channel
``declared_reuse``), (3) same-backend siblings STRICTLY EARLIER than
requested, nearest first (channel ``fallback`` — never admits a version newer
than requested implicitly), (4) cross-backend kernel-source-gated fill
(channel ``cross_backend``, mechanism unchanged from before this PR). The
``comm`` family is hard-excluded from every non-primary channel. Every
admitted source is recorded into ``PerfDatabase.data_provenance``.

These tests call ``PerfDatabase._build_op_sources`` directly against
synthetic on-disk trees — no CSV/parquet content is ever read by that
function, only path existence, so stub file contents are fine (mirrors
``tests/unit/sdk/database/test_dual_layout_discovery.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.base import resolve_op_data_path
from aiconfigurator.sdk.perf_database import PerfDatabase

pytestmark = pytest.mark.unit

PARQUET_STUB = b"PAR1stub"  # _build_op_sources only checks existence, never parses


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(root: Path, rel: str, data: bytes = PARQUET_STUB) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _write_yaml(root: Path, rel: str, doc: dict) -> None:
    _write(root, rel, yaml.safe_dump(doc).encode("utf-8"))


def _reuse_entry(table: str, from_version: str, reason: str = "test donor") -> dict:
    return {"table": table, "from_version": from_version, "reason": reason, "approved_by": "yimingl"}


def _write_manifest(systems_root: Path, entries: list[tuple[str, str, str, list[str]]]) -> None:
    """Write perf_data_reuse_manifest.yaml. Each entry is (op_file, kernel_source, tier, frameworks)."""
    lines = ["groups:"]
    for op_file, ks, tier, frameworks in entries:
        lines.extend(
            [
                f"  - op_file: {op_file}",
                f"    kernel_source: '{ks}'",
                f"    tier: {tier}",
                f"    frameworks: [{', '.join(frameworks)}]",
            ]
        )
    (systems_root / "perf_data_reuse_manifest.yaml").write_text("\n".join(lines) + "\n")


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
    """A ``h100_sxm`` systems tree with just a system YAML. Each test adds
    whatever data/reuse.yaml/manifest it needs under ``data/h100_sxm/``."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "h100_sxm.yaml").write_text("data_dir: data/h100_sxm\n", encoding="utf-8")
    # (manifest parsing moved into the engine resolver — no Python cache to clear)
    return root


def _build_db(systems_root: Path, *, backend: str, version: str, database_mode: str | None = "HYBRID") -> PerfDatabase:
    # Synthetic source-ordering trees intentionally omit Collector V3 sidecars.
    return PerfDatabase(
        system="h100_sxm",
        backend=backend,
        version=version,
        systems_root=str(systems_root),
        database_mode=database_mode,
        strict_provenance=False,
    )


def _sources_for(db: PerfDatabase, systems_root: Path, op: common.PerfDataFilename):
    system_data_root = str(systems_root / "data" / "h100_sxm")
    primary_path = resolve_op_data_path(system_data_root, db.backend, db.version, op.value)
    return db._build_op_sources(op, primary_path, system_data_root)


def _channels(db: PerfDatabase, op_file_basename: str) -> list[str]:
    return [entry["channel"] for entry in db.data_provenance[op_file_basename]]


def _versions(db: PerfDatabase, op_file_basename: str) -> list[str]:
    return [entry["version"] for entry in db.data_provenance[op_file_basename]]


# ---------------------------------------------------------------------------
# Channel 1 (primary) sanity
# ---------------------------------------------------------------------------


def test_primary_only_when_no_siblings(systems_root: Path) -> None:
    _write(systems_root, "data/h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet")

    db = _build_db(systems_root, backend="trtllm", version="1.0.0")
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert len(sources) == 1
    assert sources[0][1] is None
    assert _channels(db, "gemm_perf.parquet") == ["primary"]
    assert db.data_provenance["gemm_perf.parquet"][0]["exists"] is True


# ---------------------------------------------------------------------------
# Channel 2 — declared reuse (reuse.yaml, same backend, any direction)
# ---------------------------------------------------------------------------


def test_declared_reuse_channel_admits_newer_donor_in_isolation(systems_root: Path) -> None:
    """Declared reuse is the only channel that may borrow FORWARD (a version
    newer than requested) — proven here with no older siblings at all."""
    backend, requested, donor = "sglang", "0.5.12", "0.5.14"
    _write(systems_root, f"data/h100_sxm/moe/{backend}/{requested}/moe_perf.parquet")
    _write(systems_root, f"data/h100_sxm/moe/{backend}/{donor}/moe_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/moe/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("moe_perf", donor)]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.moe)

    assert len(sources) == 2
    assert sources[1][0].endswith(f"moe/{backend}/{donor}/moe_perf.parquet")
    assert sources[1][1] is None  # declared donors are unfiltered, same as primary
    assert _channels(db, "moe_perf.parquet") == ["primary", "declared_reuse"]


def test_declared_reuse_rejects_legacy_incomplete_donor(systems_root: Path) -> None:
    """A declared legacy-layout donor carrying INCOMPLETE.txt is unusable."""
    backend, requested, donor = "sglang", "0.5.12", "0.5.14"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", donor)]},
    )
    _write(systems_root, f"data/h100_sxm/{backend}/{donor}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{backend}/{donor}/INCOMPLETE.txt", b"partial collection\n")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert _channels(db, "gemm_perf.parquet") == ["primary"]
    assert all(donor not in path for path, _ in sources)


def test_declared_reuse_works_with_no_primary_data_at_all(systems_root: Path) -> None:
    """Mirrors the real l40s/quantize/vllm/0.22.0 case: the requested version
    dir holds ONLY a reuse.yaml, no parquet of its own."""
    backend, requested, donor = "vllm", "0.22.0", "0.24.0"
    _write(systems_root, f"data/h100_sxm/quantize/{backend}/{donor}/computescale_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/quantize/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("computescale_perf", donor)]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.compute_scale)

    assert len(sources) == 2
    assert sources[1][0].endswith(f"quantize/{backend}/{donor}/computescale_perf.parquet")
    provenance = db.data_provenance["computescale_perf.parquet"]
    assert provenance[0]["channel"] == "primary"
    assert provenance[0]["exists"] is False  # no parquet at the requested dir
    assert provenance[1]["channel"] == "declared_reuse"
    assert provenance[1]["exists"] is True


# ---------------------------------------------------------------------------
# Channel 3 — nearest-earlier same-backend fallback
# ---------------------------------------------------------------------------


def test_fallback_nearest_earlier_descending_no_manifest_needed(systems_root: Path) -> None:
    """Free/always-on channel: no perf_data_reuse_manifest.yaml entry needed
    at all, unlike today's behavior."""
    backend = "trtllm"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.0.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/0.9.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/0.8.0/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version="1.0.0")
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert _versions(db, "gemm_perf.parquet") == ["1.0.0", "0.9.0", "0.8.0"]
    assert _channels(db, "gemm_perf.parquet") == ["primary", "fallback", "fallback"]


def test_fallback_rejects_legacy_incomplete_donor(systems_root: Path) -> None:
    """Implicit same-backend fallback must honor the legacy whole-dir veto."""
    backend, requested, donor = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{backend}/{donor}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{backend}/{donor}/INCOMPLETE.txt", b"partial collection\n")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert _channels(db, "gemm_perf.parquet") == ["primary"]
    assert all(donor not in path for path, _ in sources)


def test_fallback_excludes_newer_than_requested(systems_root: Path) -> None:
    backend = "trtllm"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.0.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.1.0/gemm_perf.parquet")  # newer, must be excluded
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/0.9.0/gemm_perf.parquet")  # older, admitted

    db = _build_db(systems_root, backend=backend, version="1.0.0")
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    versions = _versions(db, "gemm_perf.parquet")
    assert versions == ["1.0.0", "0.9.0"]
    assert "1.1.0" not in versions


def test_unparseable_sibling_version_excluded_and_warns_once(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    backend = "trtllm"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.0.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/nightly-build/gemm_perf.parquet")  # not PEP 440

    db = _build_db(systems_root, backend=backend, version="1.0.0")
    with caplog.at_level(logging.WARNING):
        _sources_for(db, systems_root, common.PerfDataFilename.gemm)
        _sources_for(db, systems_root, common.PerfDataFilename.gemm)  # second call: still only 1 warning

    assert _versions(db, "gemm_perf.parquet") == ["1.0.0"]
    warnings = [r for r in caplog.records if "not PEP 440-parseable" in r.getMessage()]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Declared donor dedup against fallback (AIC-1503 PR4 task 1, FIX 1)
# ---------------------------------------------------------------------------


def test_declared_donor_not_duplicated_in_fallback(systems_root: Path) -> None:
    """The dominant real pattern (180 of 479 committed reuse.yaml entries):
    reuse.yaml declares a donor that points BACKWARD at an earlier sibling
    version which also physically exists on disk. That donor must be
    admitted exactly once, via ``declared_reuse`` — not a second time via
    the fallback nearest-earlier scan, which would list the same physical
    source under two channels (doubling I/O and corrupting
    data_provenance)."""
    backend, requested, donor = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{donor}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", donor)]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    paths = [e["path"] for e in provenance]
    assert len(paths) == len(set(paths)), f"donor path listed more than once: {paths}"
    donor_entries = [e for e in provenance if e["version"] == donor]
    assert len(donor_entries) == 1
    assert donor_entries[0]["channel"] == "declared_reuse"
    assert [path for path, _ in sources] == paths


# ---------------------------------------------------------------------------
# Duplicate declared-reuse entries within one reuse.yaml (AIC-1503 PR4 task
# 5, FIX 2)
# ---------------------------------------------------------------------------


def test_duplicate_declared_reuse_entry_admitted_once(systems_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Two identical (table, from_version) entries in one reuse.yaml (author
    copy-paste) must not admit the same donor twice -- first occurrence wins,
    logged at debug level."""
    backend, requested, donor = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{donor}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", donor), _reuse_entry("gemm_perf", donor)]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    with caplog.at_level(logging.DEBUG):
        sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    donor_entries = [e for e in provenance if e["version"] == donor]
    assert len(donor_entries) == 1
    assert donor_entries[0]["channel"] == "declared_reuse"
    assert [path for path, _ in sources] == [e["path"] for e in provenance]
    assert any("Duplicate declared-reuse entry" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Newer-only-via-declaration + full channel order
# ---------------------------------------------------------------------------


def test_newer_sibling_only_admitted_when_declared(systems_root: Path) -> None:
    backend, requested = "sglang", "0.5.12"
    declared_newer, undeclared_newer = "0.5.14", "0.5.15"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{declared_newer}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{undeclared_newer}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", declared_newer)]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    versions = _versions(db, "gemm_perf.parquet")
    assert versions == [requested, declared_newer]
    assert undeclared_newer not in versions


def test_full_channel_order_declared_then_fallback_nearest_then_cross_backend(systems_root: Path) -> None:
    backend = "trtllm"
    requested = "1.0.0"
    declared_donor = "1.2.0"  # newer, only admitted because declared
    nearest_earlier = "0.9.0"
    further_earlier = "0.5.0"
    newer_undeclared = "1.1.0"  # must never appear
    cross_backend_version = "0.5.0"  # sibling framework (sglang)

    for v in (requested, declared_donor, nearest_earlier, further_earlier, newer_undeclared):
        _write(systems_root, f"data/h100_sxm/gemm/{backend}/{v}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/sglang/{cross_backend_version}/gemm_perf.parquet")

    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", declared_donor)]},
    )
    _write_manifest(systems_root, [("gemm_perf.parquet", "shared_kernel", "shared", [backend, "sglang"])])

    db = _build_db(systems_root, backend=backend, version=requested)
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [(e["version"], e["channel"]) for e in provenance] == [
        (requested, "primary"),
        (declared_donor, "declared_reuse"),
        (nearest_earlier, "fallback"),
        (further_earlier, "fallback"),
        (cross_backend_version, "cross_backend"),
    ]
    assert newer_undeclared not in [e["version"] for e in provenance]
    # cross_backend rows keep the kernel_source filter; same-backend channels don't.
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)
    assert sources[-1][1] == {"shared_kernel"}
    assert all(ks is None for _, ks in sources[:-1])


# ---------------------------------------------------------------------------
# Self-overlap (the l40s case): primary already owns the table
# ---------------------------------------------------------------------------


def test_cross_backend_rejects_legacy_incomplete_donor(systems_root: Path) -> None:
    """Cross-backend fill must not admit an INCOMPLETE legacy directory."""
    backend, requested = "trtllm", "1.0.0"
    donor_backend, donor_version = "sglang", "0.5.14"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{donor_backend}/{donor_version}/gemm_perf.parquet")
    _write(
        systems_root,
        f"data/h100_sxm/{donor_backend}/{donor_version}/INCOMPLETE.txt",
        b"partial collection\n",
    )
    _write_manifest(
        systems_root,
        [("gemm_perf.parquet", "shared_kernel", "shared", [backend, donor_backend])],
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert _channels(db, "gemm_perf.parquet") == ["primary"]
    assert all(donor_backend not in path for path, _ in sources)


def test_loaded_rows_keep_primary_and_fill_only_missing_shapes(systems_root: Path) -> None:
    """Exercise the LOADED table (the engine view): overlap is first-wins;
    fallback fills only gaps."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    def _write_gemm_parquet(rel: str, rows: list[tuple[str, str, int, int, int, float]]) -> None:
        path = systems_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "framework": [r[0] for r in rows],
                    "version": [r[1] for r in rows],
                    "device": ["h100"] * len(rows),
                    "op_name": ["gemm"] * len(rows),
                    "gemm_dtype": ["bfloat16"] * len(rows),
                    "m": [r[2] for r in rows],
                    "n": [r[3] for r in rows],
                    "k": [r[4] for r in rows],
                    "latency": [r[5] for r in rows],
                }
            ),
            path,
        )

    # The engine view resolves through the probe handle; unlike the
    # sources-only tests sharing this fixture, it needs the full gpu/node
    # spec shape.
    (systems_root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )

    backend, requested, donor = "trtllm", "1.0.0", "0.9.0"
    _write_gemm_parquet(
        f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet",
        [("trtllm", "1.0.0", 128, 256, 512, 1.25)],
    )
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/collection_meta.yaml",
        {"tables": {"gemm_perf": {"status": "partial"}}},
    )
    _write_gemm_parquet(
        f"data/h100_sxm/gemm/{backend}/{donor}/gemm_perf.parquet",
        [("trtllm", "0.9.0", 128, 256, 512, 9.50), ("trtllm", "0.9.0", 256, 256, 512, 2.50)],
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    loaded = fetch_table_view(db, "_gemm_data")

    quant = common.GEMMQuantMode.bfloat16
    assert loaded[quant][128][256][512]["latency"] == pytest.approx(1.25)
    assert loaded[quant][256][256][512]["latency"] == pytest.approx(2.50)


def test_self_overlap_declared_donor_admitted_after_primary(systems_root: Path) -> None:
    """Requested dir owns SOME shapes of gemm_perf AND declares a donor for
    the SAME table (matches data/l40s/gemm/sglang/0.5.12/reuse.yaml in the
    real tree). Declared donor must land right after primary — first-wins
    merge then keeps the requested dir's own shapes authoritative."""
    backend, requested, donor = "sglang", "0.5.12", "0.5.14"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{donor}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("gemm_perf", donor, reason="self-overlap; mechanically derived")]},
    )

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert len(sources) == 2
    assert sources[0][0].endswith(f"gemm/{backend}/{requested}/gemm_perf.parquet")
    assert sources[1][0].endswith(f"gemm/{backend}/{donor}/gemm_perf.parquet")
    assert _channels(db, "gemm_perf.parquet") == ["primary", "declared_reuse"]


# ---------------------------------------------------------------------------
# comm family hard exclusion (design §6.5 rule 5)
# ---------------------------------------------------------------------------


def test_comm_family_custom_allreduce_gets_primary_only(systems_root: Path) -> None:
    """custom_allreduce lives under the comm family. Today it inherits
    newest-first sibling rows via the kernel-source manifest (same mechanism
    as any other op); this PR tightens that to primary-only, regardless of a
    stray declaration or a matching cross-backend manifest entry."""
    backend, requested, older = "trtllm", "1.3.0", "1.2.0"
    _write(systems_root, f"data/h100_sxm/comm/{backend}/{requested}/custom_allreduce_perf.parquet")
    _write(systems_root, f"data/h100_sxm/comm/{backend}/{older}/custom_allreduce_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/comm/{backend}/{requested}/reuse.yaml",
        {"reuse": [_reuse_entry("custom_allreduce_perf", older)]},
    )
    _write_manifest(systems_root, [("custom_allreduce_perf.parquet", "TRTLLM", "shared", [backend])])

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.custom_allreduce)

    assert len(sources) == 1
    assert sources[0][0].endswith(f"comm/{backend}/{requested}/custom_allreduce_perf.parquet")
    assert _channels(db, "custom_allreduce_perf.parquet") == ["primary"]


def test_legacy_layout_comm_op_keeps_pre_v3_siblings(systems_root: Path) -> None:
    """Pins the documented AIC-1503 PR4 task-1 FIX-2 exception
    (``_op_file_family_from_path`` docstring): design §6.5 rule 5's comm
    hard-exclusion is detected structurally off the primary path's family
    component, which only exists in the family-first layout. A LEGACY-shaped
    comm op (3-component path, no ``comm/`` family dir) therefore does NOT
    get the exclusion applied — it keeps pre-V3 sibling-reuse behavior for
    as long as its tree stays legacy-shaped. Contrast with
    ``test_comm_family_custom_allreduce_gets_primary_only`` above, which
    pins the family-shaped case (exclusion DOES apply, primary-only)."""
    backend, requested, older = "trtllm", "1.3.0", "1.2.0"
    _write(systems_root, f"data/h100_sxm/{backend}/{requested}/custom_allreduce_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{backend}/{older}/custom_allreduce_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.custom_allreduce)

    assert len(sources) == 2
    assert _channels(db, "custom_allreduce_perf.parquet") == ["primary", "fallback"]
    assert _versions(db, "custom_allreduce_perf.parquet") == [requested, older]


def test_nccl_op_name_early_exit_still_applies(systems_root: Path) -> None:
    """NCCL is comm-family too, but this proves the pre-existing op-name
    early-exit (kept per the PR3 carry-in) independently still short-circuits,
    not just the new family-based check."""
    _write(systems_root, "data/h100_sxm/comm/nccl/2.26.2/nccl_perf.parquet")
    _write(systems_root, "data/h100_sxm/comm/nccl/2.20.0/nccl_perf.parquet")  # older sibling, must be ignored

    db = _build_db(systems_root, backend="trtllm", version="2.26.2")
    system_data_root = str(systems_root / "data" / "h100_sxm")
    primary_path = resolve_op_data_path(system_data_root, "nccl", "2.26.2", common.PerfDataFilename.nccl.value)
    sources = db._build_op_sources(common.PerfDataFilename.nccl, primary_path, system_data_root)

    assert len(sources) == 1
    assert _channels(db, "nccl_perf.parquet") == ["primary"]


# ---------------------------------------------------------------------------
# Partial collection semantics
# ---------------------------------------------------------------------------


def test_legacy_incomplete_primary_not_admitted_donors_still_fill(
    systems_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The unstructured legacy INCOMPLETE marker remains a whole-dir veto."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/{backend}/{requested}/INCOMPLETE.txt", b"partial collection\n")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    with caplog.at_level(logging.WARNING):
        sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["fallback"]
    assert provenance[0]["version"] == earlier
    assert [path for path, _ in sources] == [e["path"] for e in provenance]
    assert not any(f"{backend}/{requested}/gemm_perf.parquet" in path for path, _ in sources)
    assert any("INCOMPLETE.txt" in r.getMessage() and requested in r.getMessage() for r in caplog.records)


def test_structured_partial_legacy_primary_is_admitted_before_fallback(systems_root: Path) -> None:
    """Structured partial means missing coverage, not invalid successful rows."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/{backend}/{requested}/gemm_perf.parquet")
    _write_yaml(
        systems_root,
        f"data/h100_sxm/{backend}/{requested}/collection_meta.yaml",
        {
            "schema_version": 1,
            "runtime": {"framework": backend, "version": requested},
            "tables": {"gemm_perf": {"status": "partial"}},
        },
    )
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["primary", "fallback"]
    assert [e["version"] for e in provenance] == [requested, earlier]
    assert provenance[0]["exists"] is True
    assert [path for path, _ in sources] == [e["path"] for e in provenance]


def test_structured_partial_family_primary_is_resolved_and_admitted(systems_root: Path) -> None:
    """Family-layout partial data stays primary; older rows fill gaps only."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    primary_rel = f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet"
    _write(systems_root, primary_rel)
    _write_yaml(
        systems_root,
        f"data/h100_sxm/gemm/{backend}/{requested}/collection_meta.yaml",
        {"tables": {"gemm_perf": {"status": "partial"}}},
    )
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["primary", "fallback"]
    assert provenance[0]["path"] == str(systems_root / primary_rel)
    assert provenance[0]["exists"] is True
    assert [path for path, _ in sources] == [e["path"] for e in provenance]


def test_legacy_incomplete_family_dir_is_skipped_by_resolver(systems_root: Path) -> None:
    """Legacy INCOMPLETE still skips the family path because it has no coverage detail."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/INCOMPLETE.txt", b"partial collection\n")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["primary", "fallback"]
    assert provenance[0]["exists"] is False
    assert f"gemm/{backend}/{requested}" not in provenance[0]["path"]
    assert provenance[1]["version"] == earlier
    assert [path for path, _ in sources] == [e["path"] for e in provenance]


# ---------------------------------------------------------------------------
# data_provenance shape + content
# ---------------------------------------------------------------------------


def test_data_provenance_shape_and_content(systems_root: Path) -> None:
    backend = "trtllm"
    requested, older = "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{older}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert isinstance(provenance, list)
    for entry in provenance:
        assert set(entry.keys()) == {"version", "path", "channel", "exists"}
        assert isinstance(entry["exists"], bool)

    assert provenance[0]["channel"] == "primary"
    assert provenance[0]["version"] == requested
    assert provenance[0]["exists"] is False  # requested dir never populated

    assert provenance[1]["channel"] == "fallback"
    assert provenance[1]["version"] == older
    assert provenance[1]["exists"] is True
    assert provenance[1]["path"].endswith(f"gemm/{backend}/{older}/gemm_perf.parquet")

    # data_provenance mirrors the returned sources list exactly (paths + order).
    assert [path for path, _ in sources] == [entry["path"] for entry in provenance]


def test_data_provenance_populated_per_op_file(systems_root: Path) -> None:
    """Two different op files get independent data_provenance entries."""
    backend = "trtllm"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.0.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/moe/{backend}/1.0.0/moe_perf.parquet")

    db = _build_db(systems_root, backend=backend, version="1.0.0")
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)
    _sources_for(db, systems_root, common.PerfDataFilename.moe)

    assert set(db.data_provenance.keys()) == {"gemm_perf.parquet", "moe_perf.parquet"}


def test_vetoed_primary_with_no_donor_loads_nothing_through_the_engine_view(systems_root: Path) -> None:
    """An explicitly EMPTY source list must stay empty end to end (review
    #1555 P1): a legacy INCOMPLETE.txt vetoes the primary, no donor is
    admissible, ``_build_op_sources`` returns ``[]`` — and the engine view
    must NOT fall back to re-resolving the vetoed primary file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    (systems_root / "h100_sxm.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data/h100_sxm",
                "gpu": {
                    "sm_version": 90,
                    "mem_bw": 4_800_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 989_000_000_000_000.0,
                    "fp8_tc_flops": 1_978_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 8,
                    "inter_node_bw": 50_000_000_000.0,
                    "intra_node_bw": 450_000_000_000.0,
                    "p2p_latency": 0.00001,
                },
                "misc": {"nccl_version": "2.26.2"},
            }
        ),
        encoding="utf-8",
    )
    # Legacy layout: the primary carries rows AND a legacy INCOMPLETE.txt
    # (whole-dir veto, no collection_meta.yaml); no sibling/donor exists.
    version_dir = systems_root / "data" / "h100_sxm" / "trtllm" / "1.0.0"
    version_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "framework": ["trtllm"],
                "version": ["1.0.0"],
                "device": ["h100"],
                "op_name": ["gemm"],
                "gemm_dtype": ["bfloat16"],
                "m": [128],
                "n": [256],
                "k": [512],
                "latency": [7.25],
            }
        ),
        version_dir / "gemm_perf.parquet",
    )
    (version_dir / "INCOMPLETE.txt").write_text("partial collection\n")

    db = _build_db(systems_root, backend="trtllm", version="1.0.0")
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)
    assert sources == [], "the veto must yield an explicitly empty source list"

    view = fetch_table_view(db, "_gemm_data")
    assert view is None or not view, f"the vetoed primary leaked into the engine view: {view!r}"
