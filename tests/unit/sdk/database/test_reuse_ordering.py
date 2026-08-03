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
from aiconfigurator.sdk.perf_database import PerfDatabase, _load_op_kernel_source_manifest_entries

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
    """Write op_kernel_source_manifest.yaml. Each entry is (op_file, kernel_source, tier, frameworks)."""
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
    (systems_root / "op_kernel_source_manifest.yaml").write_text("\n".join(lines) + "\n")


@pytest.fixture
def systems_root(tmp_path: Path) -> Path:
    """A ``h100_sxm`` systems tree with just a system YAML. Each test adds
    whatever data/reuse.yaml/manifest it needs under ``data/h100_sxm/``."""
    root = tmp_path / "systems"
    root.mkdir()
    (root / "h100_sxm.yaml").write_text("data_dir: data/h100_sxm\n", encoding="utf-8")
    _load_op_kernel_source_manifest_entries.cache_clear()
    return root


def _build_db(systems_root: Path, *, backend: str, version: str, database_mode: str | None = "HYBRID") -> PerfDatabase:
    return PerfDatabase(
        system="h100_sxm",
        backend=backend,
        version=version,
        systems_root=str(systems_root),
        database_mode=database_mode,
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
    """Free/always-on channel: no op_kernel_source_manifest.yaml entry needed
    at all, unlike today's behavior."""
    backend = "trtllm"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/1.0.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/0.9.0/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/0.8.0/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version="1.0.0")
    _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    assert _versions(db, "gemm_perf.parquet") == ["1.0.0", "0.9.0", "0.8.0"]
    assert _channels(db, "gemm_perf.parquet") == ["primary", "fallback", "fallback"]


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
# Partial version dirs are never admitted as the primary source.
# resolve_op_data_path skips partial FAMILY dirs, but its final legacy-layout
# fallback returns an existing file with no partial check — the admission
# chokepoint (_build_op_sources) must refuse that primary itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", ["incomplete_txt", "meta_yaml_partial"])
def test_partial_legacy_primary_not_admitted_donors_still_fill(
    systems_root: Path, caplog: pytest.LogCaptureFixture, marker: str
) -> None:
    """A legacy-layout dir holding the requested version's table but marked
    partial (INCOMPLETE.txt legacy marker, or collection_meta.yaml with any
    ``status: partial`` table) must NOT be admitted as the primary source.
    Channels 2-4 still fill, and data_provenance lists admitted sources
    only — no primary record for the refused file."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/{backend}/{requested}/gemm_perf.parquet")
    if marker == "incomplete_txt":
        _write(systems_root, f"data/h100_sxm/{backend}/{requested}/INCOMPLETE.txt", b"partial collection\n")
    else:
        _write_yaml(
            systems_root,
            f"data/h100_sxm/{backend}/{requested}/collection_meta.yaml",
            {
                "schema_version": 1,
                "runtime": {"framework": backend, "version": requested},
                "tables": {"gemm_perf": {"status": "partial"}},
            },
        )
    # A complete family-layout earlier sibling fills via channel 3 (fallback).
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    with caplog.at_level(logging.WARNING):
        sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["fallback"]
    assert provenance[0]["version"] == earlier
    assert [path for path, _ in sources] == [e["path"] for e in provenance]
    assert not any(f"{backend}/{requested}/gemm_perf.parquet" in path for path, _ in sources)
    assert any("partial" in r.getMessage() and requested in r.getMessage() for r in caplog.records)


def test_partial_family_dir_is_skipped_by_resolver_not_the_admission_guard(systems_root: Path) -> None:
    """Scope guard: a family-layout primary can never be partial, because
    resolve_op_data_path already skips partial family dirs — the admission
    guard in _build_op_sources only ever fires for the legacy-layout
    fallback path. Here the partial family dir is skipped upstream, so the
    primary resolves to the (nonexistent) legacy-shaped path and keeps its
    provenance record with exists=False, per the usual missing-primary
    semantics."""
    backend, requested, earlier = "trtllm", "1.0.0", "0.9.0"
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/gemm_perf.parquet")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{requested}/INCOMPLETE.txt", b"partial collection\n")
    _write(systems_root, f"data/h100_sxm/gemm/{backend}/{earlier}/gemm_perf.parquet")

    db = _build_db(systems_root, backend=backend, version=requested)
    sources = _sources_for(db, systems_root, common.PerfDataFilename.gemm)

    provenance = db.data_provenance["gemm_perf.parquet"]
    assert [e["channel"] for e in provenance] == ["primary", "fallback"]
    assert provenance[0]["exists"] is False  # legacy-shaped path, not the partial family file
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
