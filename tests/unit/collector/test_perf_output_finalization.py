# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import multiprocessing as mp
import stat
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pc_csv
import pyarrow.parquet as pq
import pytest

import collector.helper as helper_mod
from collector.helper import (
    PerfFinalizationInfo,
    convert_perf_csv_to_parquet,
    finalize_perf_files,
    finalize_perf_outputs,
    find_perf_csv_outputs,
)

pytestmark = pytest.mark.unit


def _write_perf_csv(path: Path, latency: float = 1.25) -> None:
    path.write_text(f"op,latency\nmatmul,{latency}\n")


def _write_keyed_perf_csv(path: Path, rows) -> None:
    """rows: iterable of (shape, latency). Identity key is (op, shape)."""
    lines = ["op,shape,latency"] + [f"matmul,{shape},{latency}" for shape, latency in rows]
    path.write_text("\n".join(lines) + "\n")


def _finalize_case_aliases_in_child(paths: tuple[str, str]) -> None:
    finalization_info = {}
    converted = finalize_perf_files(
        [Path(path) for path in paths],
        finalization_info=finalization_info,
    )
    assert len(converted) == 1
    assert len(finalization_info) == 1


def test_find_perf_csv_outputs_is_non_recursive_by_default(tmp_path):
    top_level = tmp_path / "gemm_perf.txt"
    nested = tmp_path / "src" / "aiconfigurator" / "systems" / "data" / "gemm_perf.txt"
    incomplete = tmp_path / "INCOMPLETE.txt"

    _write_perf_csv(top_level)
    nested.parent.mkdir(parents=True)
    _write_perf_csv(nested)
    incomplete.write_text("incomplete\n")

    assert find_perf_csv_outputs(tmp_path) == [top_level]
    assert find_perf_csv_outputs(tmp_path, recursive=True) == [top_level, nested]


def test_find_perf_csv_outputs_ignores_structured_provenance_markers(tmp_path):
    """collection_meta.yaml / reuse.yaml (Collector V3 design §5/§6.3) sit flat
    beside the staged CSV outputs at finalize time but are not CSVs; the
    `*_perf.txt` glob must not pick them up (they don't match the pattern, so
    this is a regression guard, not new filtering logic)."""
    top_level = tmp_path / "gemm_perf.txt"
    _write_perf_csv(top_level)
    (tmp_path / "collection_meta.yaml").write_text("schema_version: 1\n")
    (tmp_path / "reuse.yaml").write_text("schema_version: 1\n")

    assert find_perf_csv_outputs(tmp_path) == [top_level]


def test_finalize_perf_outputs_does_not_recurse_into_checked_in_assets(tmp_path):
    top_level = tmp_path / "gemm_perf.txt"
    nested = tmp_path / "src" / "aiconfigurator" / "systems" / "data" / "gemm_perf.txt"

    _write_perf_csv(top_level)
    nested.parent.mkdir(parents=True)
    _write_perf_csv(nested)

    converted = finalize_perf_outputs(tmp_path)

    assert converted == [top_level.with_suffix(".parquet")]
    assert top_level.with_suffix(".parquet").exists()
    assert not top_level.exists()
    assert nested.exists()
    assert not nested.with_suffix(".parquet").exists()


def test_finalize_perf_files_converts_only_explicit_outputs(tmp_path):
    touched = tmp_path / "gemm_perf.txt"
    untouched = tmp_path / "allreduce_perf.txt"
    nested = tmp_path / "nested" / "moe_perf.txt"

    _write_perf_csv(touched, latency=1.0)
    _write_perf_csv(untouched, latency=2.0)
    nested.parent.mkdir()
    _write_perf_csv(nested, latency=3.0)

    converted = finalize_perf_files([touched, touched, nested])

    assert converted == [touched.with_suffix(".parquet"), nested.with_suffix(".parquet")]
    assert pq.read_table(touched.with_suffix(".parquet")).to_pylist() == [{"op": "matmul", "latency": 1.0}]
    assert pq.read_table(nested.with_suffix(".parquet")).to_pylist() == [{"op": "matmul", "latency": 3.0}]
    assert untouched.exists()
    assert not untouched.with_suffix(".parquet").exists()


@pytest.mark.parametrize(
    "invalid_second",
    [
        "malformed-csv",
        "invalid-power",
        "corrupt-existing-parquet",
    ],
)
def test_finalize_perf_files_prepares_every_input_before_publishing_any(tmp_path, invalid_second):
    first = tmp_path / "gemm_perf.txt"
    second = tmp_path / "moe_perf.txt"
    first_parquet = first.with_suffix(".parquet")
    second_parquet = second.with_suffix(".parquet")
    first.write_text("shape,latency\nnew-gemm,1.0\n", encoding="utf-8")
    second.write_text("shape,latency\nnew-moe,2.0\n", encoding="utf-8")
    pq.write_table(pa.table({"shape": ["old-gemm"], "latency": [9.0]}), first_parquet)
    pq.write_table(pa.table({"shape": ["old-moe"], "latency": [8.0]}), second_parquet)

    if invalid_second == "malformed-csv":
        second.write_text("shape,latency\nnew-moe,2.0,unexpected\n", encoding="utf-8")
    elif invalid_second == "invalid-power":
        second.write_text("shape,latency,power\nnew-moe,2.0,-1.0\n", encoding="utf-8")
    else:
        second_parquet.write_bytes(b"not parquet")

    staging_before = {path: path.read_bytes() for path in (first, second)}
    parquet_before = {path: path.read_bytes() for path in (first_parquet, second_parquet)}
    sentinel_info = PerfFinalizationInfo(
        new_rows=1,
        merged_existing=False,
        source_digest="sha256:" + "0" * 64,
        source_device=0,
        source_inode=0,
    )
    finalization_info = {Path("sentinel"): sentinel_info}

    with pytest.raises((ValueError, OSError)):
        finalize_perf_files([first, second], delete_source=False, finalization_info=finalization_info)

    assert {path: path.read_bytes() for path in (first, second)} == staging_before
    assert {path: path.read_bytes() for path in (first_parquet, second_parquet)} == parquet_before
    assert finalization_info == {Path("sentinel"): sentinel_info}
    assert list(tmp_path.glob(".*.tmp")) == []


def test_finalize_perf_files_rolls_back_every_target_when_second_publish_fails(tmp_path, monkeypatch):
    first = tmp_path / "gemm_perf.txt"
    second = tmp_path / "moe_perf.txt"
    first_parquet = first.with_suffix(".parquet")
    second_parquet = second.with_suffix(".parquet")
    first.write_text("shape,latency\nnew-gemm,1.0\n", encoding="utf-8")
    second.write_text("shape,latency\nnew-moe,2.0\n", encoding="utf-8")
    pq.write_table(pa.table({"shape": ["old-gemm"], "latency": [9.0]}), first_parquet)
    pq.write_table(pa.table({"shape": ["old-moe"], "latency": [8.0]}), second_parquet)
    parquet_before = {path: path.read_bytes() for path in (first_parquet, second_parquet)}
    staging_before = {path: path.read_bytes() for path in (first, second)}
    real_replace = helper_mod.os.replace
    failed = False

    def fail_second_publish(source, target):
        nonlocal failed
        if Path(target) == second_parquet and not failed:
            failed = True
            raise OSError("simulated second parquet publish failure")
        return real_replace(source, target)

    monkeypatch.setattr(helper_mod.os, "replace", fail_second_publish)
    finalization_info = {}

    with pytest.raises(OSError, match="simulated second parquet publish failure"):
        finalize_perf_files([first, second], delete_source=False, finalization_info=finalization_info)

    assert {path: path.read_bytes() for path in (first_parquet, second_parquet)} == parquet_before
    assert {path: path.read_bytes() for path in (first, second)} == staging_before
    assert finalization_info == {}


def test_finalize_perf_files_without_transaction_never_uses_private_claims(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    perf.write_text("shape,latency\nnew,1.0\n", encoding="utf-8")
    pq.write_table(pa.table({"shape": ["old"], "latency": [9.0]}), parquet)

    def reject_unjournaled_claim(*_args, **_kwargs):
        raise AssertionError("non-transactional publication must stay a single atomic replace")

    monkeypatch.setattr(helper_mod, "_rename_noreplace", reject_unjournaled_claim)

    finalize_perf_files([perf], delete_source=False)

    assert {row["shape"] for row in pq.read_table(parquet).to_pylist()} == {"old", "new"}
    assert list(tmp_path.glob(".*.claim")) == []


def test_finalize_perf_files_rejects_target_appearing_after_merge_observed_absence(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    perf.write_text("shape,latency\nnew,1.0\n", encoding="utf-8")
    real_prepare_publications = helper_mod._prepare_perf_publications
    concurrent_bytes = None

    def create_target_before_publication(prepared):
        nonlocal concurrent_bytes
        pq.write_table(pa.table({"shape": ["concurrent"], "latency": [9.0]}), parquet)
        concurrent_bytes = parquet.read_bytes()
        return real_prepare_publications(prepared)

    monkeypatch.setattr(helper_mod, "_prepare_perf_publications", create_target_before_publication)
    finalization_info = {}

    with pytest.raises(RuntimeError, match="changed after merge preparation"):
        finalize_perf_files([perf], delete_source=False, finalization_info=finalization_info)

    assert concurrent_bytes is not None
    assert parquet.read_bytes() == concurrent_bytes
    assert perf.exists()
    assert finalization_info == {}
    assert list(tmp_path.glob(".*.rollback")) == []


def test_finalize_perf_files_cleans_rollback_backup_when_transaction_prepare_fails(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    perf.write_text("shape,latency\nnew,1.0\n", encoding="utf-8")
    pq.write_table(pa.table({"shape": ["old"], "latency": [9.0]}), parquet)
    parquet_before = parquet.read_bytes()

    class RejectingPublicationTransaction:
        def prepare(self, _publications):
            raise RuntimeError("simulated transaction prepare failure")

        def rollback_complete(self, _publications):
            raise AssertionError("publication never started")

        def has_durable_journal(self):
            return False

    with pytest.raises(RuntimeError, match="simulated transaction prepare failure"):
        finalize_perf_files(
            [perf],
            delete_source=False,
            publication_transaction=RejectingPublicationTransaction(),
        )

    assert parquet.read_bytes() == parquet_before
    assert perf.exists()
    assert list(tmp_path.glob(".*.rollback")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_finalize_perf_files_never_reopens_or_publishes_replaced_temporary(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    outside_victim = tmp_path / "outside-victim"
    outside_victim.write_bytes(b"victim must remain unchanged")
    victim_before = outside_victim.read_bytes()
    real_write_table = pq.write_table

    def replace_temporary_before_write(table, destination, *args, **kwargs):
        temporary = perf.with_suffix(".parquet").with_name(".gemm_perf.parquet.tmp")
        temporary.unlink()
        try:
            temporary.symlink_to(outside_victim)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        return real_write_table(table, destination, *args, **kwargs)

    monkeypatch.setattr(pq, "write_table", replace_temporary_before_write)
    finalization_info = {}

    with pytest.raises(RuntimeError, match="temporary"):
        finalize_perf_files([perf], delete_source=False, finalization_info=finalization_info)

    assert outside_victim.read_bytes() == victim_before
    assert not perf.with_suffix(".parquet").exists()
    assert finalization_info == {}


def test_finalize_closes_and_reclaims_reservation_when_directory_fsync_fails(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    reserved = parquet.with_name(f".{parquet.name}.tmp")
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    real_fsync_directory = helper_mod._fsync_directory

    def fail_after_reservation(directory):
        if reserved.exists():
            raise OSError("simulated reservation directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(helper_mod, "_fsync_directory", fail_after_reservation)

    with pytest.raises(OSError, match="reservation directory fsync failure"):
        finalize_perf_files([perf], delete_source=False)

    assert not reserved.exists()
    assert not parquet.exists()


def test_finalize_direct_retry_reclaims_stale_deterministic_reservation(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    reserved = parquet.with_name(f".{parquet.name}.tmp")
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    reserved.write_bytes(b"partial parquet from interrupted direct finalization")

    assert finalize_perf_files([perf], delete_source=False) == [parquet]

    assert pq.read_table(parquet).to_pylist() == [{"shape": "s1", "latency": 1.0}]
    assert not reserved.exists()


def test_reservation_cleanup_restores_object_changed_during_claim(tmp_path, monkeypatch):
    parquet = tmp_path / "gemm_perf.parquet"
    reserved = helper_mod.perf_preparation_path(parquet)
    cleanup_claim = helper_mod.perf_preparation_cleanup_path(parquet)
    reserved.write_bytes(b"expected partial parquet")
    real_rename_noreplace = helper_mod._rename_noreplace

    def rename_then_mutate(source, target):
        real_rename_noreplace(source, target)
        Path(target).write_bytes(b"unknown raced object")

    monkeypatch.setattr(helper_mod, "_rename_noreplace", rename_then_mutate)

    with pytest.raises(RuntimeError, match="changed"):
        helper_mod.cleanup_unjournaled_perf_preparations([parquet], transaction_paths=())

    assert reserved.read_bytes() == b"unknown raced object"
    assert not cleanup_claim.exists()


def test_perf_attestation_rejects_execute_bit_change(tmp_path):
    artifact = tmp_path / "artifact.parquet"
    artifact.write_bytes(b"parquet bytes")
    artifact.chmod(0o600)
    attestation = helper_mod._attest_regular_file(artifact)
    artifact.chmod(0o700)

    with pytest.raises(RuntimeError, match="changed"):
        helper_mod._attest_expected_perf_file(artifact, attestation)


def test_perf_attestation_rejects_chmod_during_digest(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.parquet"
    artifact.write_bytes(b"parquet bytes")
    artifact.chmod(0o600)
    real_stream_digest = helper_mod._stream_digest
    changed = False

    def digest_then_chmod(file, *args, **kwargs):
        nonlocal changed
        digest = real_stream_digest(file, *args, **kwargs)
        if not changed:
            changed = True
            artifact.chmod(0o400)
        return digest

    monkeypatch.setattr(helper_mod, "_stream_digest", digest_then_chmod)

    with pytest.raises(RuntimeError, match="changed"):
        helper_mod._attest_regular_file(artifact)

    assert changed


def test_finalize_perf_files_rejects_source_mode_change_after_preparation(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\nnew,1.0\n", encoding="utf-8")
    perf.chmod(0o600)
    real_prepare_publications = helper_mod._prepare_perf_publications

    def chmod_source_before_publication(prepared):
        perf.chmod(0o400)
        return real_prepare_publications(prepared)

    class UnreachedPublicationTransaction:
        def prepare(self, _publications):
            raise AssertionError("journal preparation must not be reached")

        def rollback_complete(self, _publications):
            raise AssertionError("publication never started")

        def has_durable_journal(self):
            return False

    monkeypatch.setattr(helper_mod, "_prepare_perf_publications", chmod_source_before_publication)

    with pytest.raises(RuntimeError, match=r"staging.*changed|artifact changed"):
        finalize_perf_files(
            [perf],
            delete_source=False,
            publication_transaction=UnreachedPublicationTransaction(),
        )

    assert not perf.with_suffix(".parquet").exists()


def test_finalization_info_records_the_exact_staging_digest(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    contents = b"shape,latency\ns1,1.0\n"
    perf.write_bytes(contents)
    finalization_info = {}

    [parquet] = finalize_perf_files([perf], delete_source=False, finalization_info=finalization_info)

    assert finalization_info[parquet.resolve()].source_digest == "sha256:" + hashlib.sha256(contents).hexdigest()


def test_finalization_digest_and_parquet_use_the_same_open_staging_file(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    original = b"shape,latency\noriginal,1.0\n"
    replacement = b"shape,latency\nreplacement,9.0\n"
    perf.write_bytes(original)
    replacement_path = tmp_path / "replacement.txt"
    replacement_path.write_bytes(replacement)
    backup_path = tmp_path / "original.txt"
    real_read_csv = pc_csv.read_csv

    def swap_path_while_parsing(source, *args, **kwargs):
        perf.replace(backup_path)
        replacement_path.replace(perf)
        try:
            return real_read_csv(source, *args, **kwargs)
        finally:
            perf.replace(replacement_path)
            backup_path.replace(perf)

    monkeypatch.setattr(pc_csv, "read_csv", swap_path_while_parsing)
    finalization_info = {}

    [parquet] = finalize_perf_files([perf], delete_source=False, finalization_info=finalization_info)

    assert pq.read_table(parquet).to_pylist() == [{"shape": "original", "latency": 1.0}]
    assert finalization_info[parquet.resolve()].source_digest == "sha256:" + hashlib.sha256(original).hexdigest()


def test_finalization_parses_the_immutable_staging_snapshot(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    original = b"shape,latency\noriginal,1.0\n"
    replacement = b"shape,latency\nreplacement,9.0\n"
    perf.write_bytes(original)
    real_read_csv = pc_csv.read_csv

    def rewrite_source_while_parsing(source, *args, **kwargs):
        perf.write_bytes(replacement)
        try:
            return real_read_csv(source, *args, **kwargs)
        finally:
            perf.write_bytes(original)

    monkeypatch.setattr(pc_csv, "read_csv", rewrite_source_while_parsing)
    finalization_info = {}

    [parquet] = finalize_perf_files([perf], delete_source=False, finalization_info=finalization_info)

    assert pq.read_table(parquet).to_pylist() == [{"shape": "original", "latency": 1.0}]
    assert finalization_info[parquet.resolve()].source_digest == "sha256:" + hashlib.sha256(original).hexdigest()


def test_finalize_perf_files_canonicalizes_aliases_before_locking(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    alias = nested / ".." / perf.name
    acquired: set[tuple[int, int]] = set()
    real_acquire = helper_mod._acquire_merge_lock

    def reject_duplicate_lock(lock_fd):
        opened = helper_mod.os.fstat(lock_fd)
        identity = (opened.st_dev, opened.st_ino)
        if identity in acquired:
            raise AssertionError(f"duplicate lock acquisition: {identity}")
        acquired.add(identity)
        return real_acquire(lock_fd)

    monkeypatch.setattr(helper_mod, "_acquire_merge_lock", reject_duplicate_lock)

    converted = finalize_perf_files([perf, alias], delete_source=False)

    assert [path.resolve() for path in converted] == [perf.with_suffix(".parquet").resolve()]
    assert len(acquired) == 1


def test_finalize_perf_files_case_aliases_do_not_self_deadlock(tmp_path):
    perf = tmp_path / "Case_perf.txt"
    alias = tmp_path / "case_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    if not alias.exists() or not perf.samefile(alias):
        pytest.skip("case-sensitive filesystem")

    process = mp.get_context("spawn").Process(
        target=_finalize_case_aliases_in_child,
        args=((str(perf), str(alias)),),
    )
    process.start()
    process.join(timeout=3)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("case-alias finalization self-deadlocked")

    assert process.exitcode == 0
    assert perf.with_suffix(".parquet").exists()
    assert not perf.exists()
    assert not alias.exists()


def test_finalize_perf_files_retains_distinct_inputs_with_hardlinked_merge_locks(tmp_path):
    first = tmp_path / "a_perf.txt"
    second = tmp_path / "b_perf.txt"
    _write_perf_csv(first, latency=1.0)
    _write_perf_csv(second, latency=2.0)
    first_lock = Path(f"{first.with_suffix('.parquet')}.mergelock")
    second_lock = Path(f"{second.with_suffix('.parquet')}.mergelock")
    first_lock.touch()
    helper_mod.os.link(first_lock, second_lock)
    finalization_info = {}

    converted = finalize_perf_files(
        [first, second],
        delete_source=False,
        finalization_info=finalization_info,
    )

    expected = [first.with_suffix(".parquet"), second.with_suffix(".parquet")]
    assert converted == expected
    assert all(path.exists() for path in expected)
    assert set(finalization_info) == {path.resolve() for path in expected}


def test_finalize_perf_files_retains_hardlinked_sources_with_distinct_targets(tmp_path):
    first = tmp_path / "a_perf.txt"
    second = tmp_path / "b_perf.txt"
    _write_perf_csv(first, latency=1.0)
    helper_mod.os.link(first, second)
    finalization_info = {}

    converted = finalize_perf_files(
        [first, second],
        delete_source=False,
        finalization_info=finalization_info,
    )

    expected = [first.with_suffix(".parquet"), second.with_suffix(".parquet")]
    assert converted == expected
    assert all(path.exists() for path in expected)
    assert set(finalization_info) == {path.resolve() for path in expected}


def test_finalize_perf_files_retains_distinct_entries_when_sources_and_locks_are_hardlinked(tmp_path):
    first = tmp_path / "a_perf.txt"
    second = tmp_path / "b_perf.txt"
    _write_perf_csv(first, latency=1.0)
    helper_mod.os.link(first, second)
    first_lock = Path(f"{first.with_suffix('.parquet')}.mergelock")
    second_lock = Path(f"{second.with_suffix('.parquet')}.mergelock")
    first_lock.touch()
    helper_mod.os.link(first_lock, second_lock)
    source_stat = first.stat()
    source_identity = (
        "sha256:" + hashlib.sha256(first.read_bytes()).hexdigest(),
        source_stat.st_dev,
        source_stat.st_ino,
    )
    finalization_info = {}

    converted = finalize_perf_files(
        [first, second],
        delete_source=False,
        finalization_info=finalization_info,
        expected_source_identities={first.resolve(): source_identity, second.resolve(): source_identity},
    )

    expected = [first.with_suffix(".parquet"), second.with_suffix(".parquet")]
    assert converted == expected
    assert all(path.exists() for path in expected)
    assert set(finalization_info) == {path.resolve() for path in expected}


def test_finalize_perf_files_orders_locks_by_canonical_target(tmp_path, monkeypatch):
    first = tmp_path / "a_perf.txt"
    second = tmp_path / "b_perf.txt"
    for path in (first, second):
        path.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    late_alias_dir = tmp_path / "z"
    early_alias_dir = tmp_path / "0"
    late_alias_dir.mkdir()
    early_alias_dir.mkdir()
    aliases = [late_alias_dir / ".." / first.name, early_alias_dir / ".." / second.name]
    orders: list[list[tuple[int, int]]] = []
    current_order: list[tuple[int, int]] = []
    real_acquire = helper_mod._acquire_merge_lock

    def record_lock(lock_fd):
        opened = helper_mod.os.fstat(lock_fd)
        current_order.append((opened.st_dev, opened.st_ino))
        return real_acquire(lock_fd)

    monkeypatch.setattr(helper_mod, "_acquire_merge_lock", record_lock)

    finalize_perf_files([first, second], delete_source=False)
    orders.append(list(current_order))
    current_order.clear()
    finalize_perf_files(aliases, delete_source=False)
    orders.append(list(current_order))

    assert orders[0] == orders[1]


@pytest.mark.parametrize("source_mode", [0o600, 0o644, 0o750])
def test_finalize_perf_files_preserves_staging_permissions(tmp_path, source_mode):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    perf.chmod(source_mode)

    [parquet] = finalize_perf_files([perf])

    assert stat.S_IMODE(parquet.stat().st_mode) == source_mode


def test_finalize_perf_files_records_info_only_after_source_cleanup_succeeds(tmp_path, monkeypatch):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    real_unlink = Path.unlink

    def fail_source_cleanup(path, *args, **kwargs):
        if path == perf:
            raise OSError("simulated source cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_source_cleanup)
    finalization_info = {}

    with pytest.raises(OSError, match="simulated source cleanup failure"):
        finalize_perf_files([perf], finalization_info=finalization_info)

    assert perf.exists()
    assert perf.with_suffix(".parquet").exists()
    assert finalization_info == {}


def _rows_by_key(parquet_path):
    """Return {shape: latency} for a keyed perf parquet."""
    return {r["shape"]: r["latency"] for r in pq.read_table(parquet_path).to_pylist()}


def test_finalize_merges_disjoint_keys_into_existing_parquet(tmp_path):
    # Regression for the resume/retry-failed data-loss footgun: finalize deletes
    # the source .txt, so a second (partial) finalize must NOT clobber the
    # already-finalized rows — the disjoint keys from both runs must survive.
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])  # first (full) run
    finalize_perf_files([perf])
    assert not perf.exists()  # source consumed
    assert _rows_by_key(parquet) == {"s1": 1.0}

    _write_keyed_perf_csv(perf, [("s2", 2.0)])  # retry-failed run: a new key only
    finalize_perf_files([perf])
    assert _rows_by_key(parquet) == {"s1": 1.0, "s2": 2.0}  # s1 NOT lost


def test_finalize_merge_replaces_same_key_with_newest_measurement(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0), ("s2", 2.0)])
    finalize_perf_files([perf])

    # Re-measure s1 (new value) and add s3; s2 untouched must persist.
    _write_keyed_perf_csv(perf, [("s1", 9.0), ("s3", 3.0)])
    finalize_perf_files([perf])

    assert _rows_by_key(parquet) == {"s1": 9.0, "s2": 2.0, "s3": 3.0}
    # exactly one row per identity key (no duplicate keys)
    shapes = [r["shape"] for r in pq.read_table(parquet).to_pylist()]
    assert sorted(shapes) == ["s1", "s2", "s3"]


def test_finalize_merge_attests_deduplicated_current_identity_count(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])
    finalize_perf_files([perf])

    _write_keyed_perf_csv(perf, [("s1", 2.0), ("s1", 3.0)])
    source_digest = "sha256:" + hashlib.sha256(perf.read_bytes()).hexdigest()
    source_stat = perf.stat()
    finalization_info = {}
    finalize_perf_files([perf], finalization_info=finalization_info)

    assert _rows_by_key(parquet) == {"s1": 3.0}
    assert finalization_info[parquet.resolve()] == PerfFinalizationInfo(
        new_rows=1,
        merged_existing=True,
        source_digest=source_digest,
        source_device=source_stat.st_dev,
        source_inode=source_stat.st_ino,
    )


def test_convert_without_merge_overwrites(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])
    convert_perf_csv_to_parquet(perf, merge_existing=False)
    assert _rows_by_key(parquet) == {"s1": 1.0}

    _write_keyed_perf_csv(perf, [("s2", 2.0)])
    convert_perf_csv_to_parquet(perf, merge_existing=False)
    assert _rows_by_key(parquet) == {"s2": 2.0}  # legacy overwrite preserved when opted out


def test_finalize_merge_falls_back_to_overwrite_on_schema_mismatch(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    _write_keyed_perf_csv(perf, [("s1", 1.0)])
    finalize_perf_files([perf])

    # A run whose columns differ (no 'shape') cannot be safely merged; the
    # finalize must not raise and must not silently corrupt — it overwrites.
    perf.write_text("op,latency\nmatmul,5.0\n")
    finalize_perf_files([perf])
    assert pq.read_table(parquet).to_pylist() == [{"op": "matmul", "latency": 5.0}]


def test_finalize_merge_tolerates_metric_column_type_drift(tmp_path):
    """An all-empty optional metric column (e.g. power on a power-off run) is
    inferred as `null` by pyarrow.csv while populated runs infer `double`.
    That drift must NOT be treated as a schema mismatch — the old behavior
    silently overwrote the accumulated parquet with the new run's subset."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency,power\ns1,1.0,5.0\ns2,2.0,6.0\n")
    finalize_perf_files([perf])

    # New run with power entirely empty -> pyarrow infers `null` for it.
    perf.write_text("shape,latency,power\ns3,3.0,\n")
    finalize_perf_files([perf])

    rows = {r["shape"]: r for r in pq.read_table(parquet).to_pylist()}
    assert sorted(rows) == ["s1", "s2", "s3"]  # accumulated, not overwritten
    assert rows["s1"]["power"] == 5.0
    assert rows["s3"]["power"] == 0.0
    # metric column keeps the existing (double) type
    assert str(pq.read_table(parquet).schema.field("power").type) == "double"


def test_finalize_merge_still_overwrites_on_identity_type_mismatch(tmp_path):
    """Identity-column type drift stays a hard mismatch (overwrite path)."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency\n1,1.0\n")  # shape inferred int64
    finalize_perf_files([perf])
    perf.write_text("shape,latency\ns2,2.0\n")  # shape inferred string
    finalize_perf_files([perf])
    assert [r["shape"] for r in pq.read_table(parquet).to_pylist()] == ["s2"]


def test_finalize_merge_lock_does_not_block_sequential_runs(tmp_path):
    """The flock-based merge lock must release on close: a second finalize of
    the same target must proceed (and the lock file itself stays, by design —
    unlinking would reopen the create/steal race)."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    perf.write_text("shape,latency\ns1,1.0\n")
    finalize_perf_files([perf])
    perf.write_text("shape,latency\ns2,2.0\n")
    finalize_perf_files([perf])  # would deadlock if the first run kept the flock
    assert sorted(r["shape"] for r in pq.read_table(parquet).to_pylist()) == ["s1", "s2"]


def test_finalize_rejects_merge_lock_path_replacement_after_flock(tmp_path, monkeypatch):
    import fcntl

    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")
    merge_lock = parquet.with_name(f"{parquet.name}.mergelock")
    displaced_lock = tmp_path / "displaced.mergelock"
    perf.write_text("shape,latency\ns1,1.0\n", encoding="utf-8")
    real_flock = fcntl.flock
    replaced = False

    def flock_then_replace(lock_fd, operation):
        nonlocal replaced
        result = real_flock(lock_fd, operation)
        if operation & fcntl.LOCK_EX and merge_lock.exists() and not replaced:
            replaced = True
            merge_lock.replace(displaced_lock)
            merge_lock.write_bytes(b"independent lock inode")
        return result

    monkeypatch.setattr(fcntl, "flock", flock_then_replace)

    with pytest.raises(RuntimeError, match=r"merge lock.*changed|Invalid collector parquet merge lock"):
        finalize_perf_files([perf], delete_source=False)

    assert replaced
    assert perf.exists()
    assert not parquet.exists()
    assert merge_lock.read_bytes() == b"independent lock inode"
    assert displaced_lock.exists()


def test_finalize_merge_tolerates_reverse_metric_type_drift(tmp_path):
    """Old parquet has an all-null metric column, the NEW run has real values:
    the null side must be cast toward double — the naive cast direction
    (double -> null) raises ArrowNotImplementedError and aborted finalize."""
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    pq.write_table(
        pa.table(
            {
                "shape": ["s1"],
                "latency": [1.0],
                "power": pa.nulls(1),
            }
        ),
        parquet,
    )
    perf.write_text("shape,latency,power\ns2,2.0,7.5\n")
    finalize_perf_files([perf])

    rows = {r["shape"]: r for r in pq.read_table(parquet).to_pylist()}
    assert sorted(rows) == ["s1", "s2"]
    assert rows["s1"]["power"] == 0.0
    assert rows["s2"]["power"] == 7.5
    assert str(pq.read_table(parquet).schema.field("power").type) == "double"


def test_finalize_normalizes_all_empty_power_metrics_to_typed_zero(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    parquet = perf.with_suffix(".parquet")

    perf.write_text("shape,latency,power,power_limit\ns1,1.0,,\ns2,2.0,,\n")
    finalize_perf_files([perf])

    table = pq.read_table(parquet)
    assert table.select(["power", "power_limit"]).to_pylist() == [
        {"power": 0.0, "power_limit": 0.0},
        {"power": 0.0, "power_limit": 0.0},
    ]
    assert str(table.schema.field("power").type) == "double"
    assert str(table.schema.field("power_limit").type) == "double"
    assert table.column("power").null_count == 0
    assert table.column("power_limit").null_count == 0


@pytest.mark.parametrize("value", ["-1.0", "inf"])
def test_finalize_rejects_invalid_power_measurements(tmp_path, value):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text(f"shape,latency,power,power_limit\ns1,1.0,{value},1000.0\n")

    with pytest.raises(ValueError, match="power must contain finite non-negative values"):
        finalize_perf_files([perf])


def test_finalize_treats_csv_nan_power_as_unavailable(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency,power,power_limit\ns1,1.0,nan,1000.0\n")

    [parquet] = finalize_perf_files([perf])

    assert pq.read_table(parquet).column("power").to_pylist() == [0.0]


def test_finalize_keeps_power_disabled_schema_column_free(tmp_path):
    perf = tmp_path / "gemm_perf.txt"
    perf.write_text("shape,latency\ns1,1.0\n")

    [parquet] = finalize_perf_files([perf])

    assert pq.read_table(parquet).column_names == ["shape", "latency"]
