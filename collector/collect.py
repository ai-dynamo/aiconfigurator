# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level collector entrypoint.

This script resolves the requested backend, framework version, model/SM case
plan, and op registry entry, then runs the selected collector functions and
writes perf files. It is the orchestration layer for collector v2; individual
modules own benchmark setup, while `model_cases.py` and YAML own case selection.
"""

import contextlib
import functools
import os
import warnings

from helper import get_device_module, get_device_str


def setup_warning_filters():
    """Configure warning filters to suppress known non-critical warnings"""

    # Suppress the modelopt transformers version warning
    warnings.filterwarnings(
        "ignore",
        message="transformers version .* is incompatible with nvidia-modelopt",
        category=UserWarning,
        module="modelopt",
    )

    # Suppress the cuda.cudart deprecation warning
    warnings.filterwarnings("ignore", message="The cuda.cudart module is deprecated", category=FutureWarning)

    warnings.filterwarnings("ignore", message="The cuda.cuda module is deprecated", category=FutureWarning)

    # Suppress TensorRT-LLM specific warnings if needed
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorrt_llm")

    # Suppress flashinfer warnings
    warnings.filterwarnings("ignore", message="Prebuilt kernels not found", module="flashinfer")

    # Suppress torch operator override warnings (flash_attn kernel re-registration)
    warnings.filterwarnings(
        "ignore",
        message="Warning only once for all operators.*",
        category=UserWarning,
    )

    # Suppress pynvml deprecation warning from torch.cuda
    warnings.filterwarnings(
        "ignore",
        message="The pynvml package is deprecated",
        category=FutureWarning,
    )


import random
import resource

from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError:
    torch = None

setup_warning_filters()

import argparse
import cProfile
import hashlib
import importlib
import importlib.util
import io
import json
import multiprocessing as mp
import pstats
import signal
import stat
import subprocess
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from inspect import Parameter, signature
from pathlib import Path

from helper import (
    EXIT_CODE_RESTART,
    PERF_TRANSACTION_FILENAME,
    SIDECAR_STAGING_FILENAME,
    SIDECAR_TRANSACTION_FILENAME,
    LockedOutputRoot,
    PerfFileAttestation,
    PerfFinalizationInfo,
    PerfPublication,
    WorkerRestartSignal,
    _rename_noreplace,
    _rename_noreplace_at,
    atomic_write_reservation_path,
    cleanup_atomic_write_reservations,
    cleanup_perf_publication_artifacts,
    cleanup_unjournaled_perf_preparations,
    collector_retained_path,
    create_test_case_id,
    finalize_perf_files,
    find_perf_csv_outputs,
    perf_finalization_lifecycle,
    perf_merge_locks,
    perf_preparation_cleanup_path,
    perf_preparation_path,
    perf_transaction_artifact_paths,
    restore_perf_publications,
    save_error_report,
    setup_logging,
    setup_signal_handlers,
    validate_restored_perf_publications,
)

logger = None
RESUME_SCHEMA_VERSION = "collector-resume-v2"
STALL_THRESHOLD = 30  # iterations (x 0.5 s sleep = 15 s) before logging a stall warning
# Failures of one (model, dtype) group within an op before the summary flags
# it as systemic (a fix-me warning; nothing is skipped).
SYSTEMIC_GROUP_THRESHOLD = 5
_SIDECAR_TRANSACTION_SCHEMA = "collector-sidecar-transaction-v4"
_SIDECAR_TRANSACTION_FIELD = "sidecar_transaction"
_SIDECAR_TRANSACTION_FILENAME = SIDECAR_TRANSACTION_FILENAME
_SIDECAR_STAGING_FILENAME = SIDECAR_STAGING_FILENAME
_PERF_TRANSACTION_SCHEMA = "collector-perf-publication-v2"
_PERF_TRANSACTION_FILENAME = PERF_TRANSACTION_FILENAME
_CHECKPOINT_IDENTITY_FIELDS = ("schema", "backend", "module", "run_func", "framework_version", "sm_version")
_SIDECAR_TRANSACTION_FIELDS = {
    "schema",
    "transaction_id",
    "output_root",
    "backend",
    "checkpoint_root",
    "sidecar_path",
    "sidecar_digest",
    "pending_sidecar",
    "previous_sidecar",
    "checkpoints",
    "staging_paths",
    "perf_publications",
}
_PERF_TRANSACTION_FIELDS = {
    "schema",
    "transaction_id",
    "output_root",
    "backend",
    "checkpoint_root",
    "previous_sidecar",
    "sidecar_staging_path",
    "checkpoints",
    "publications",
}

FPM_INPUT_ERRORS = (TypeError, ValueError, subprocess.CalledProcessError, FileNotFoundError, RuntimeError)


@dataclass(frozen=True)
class _ProducerCheckpointPlan:
    attestation: "_FileAttestation"
    table: str
    identity: tuple[tuple[str, object], ...]
    done: frozenset[str]
    attempted: frozenset[str]
    failed: frozenset[str]

    @property
    def path(self) -> Path:
        return self.attestation.path

    def identity_dict(self) -> dict:
        return dict(self.identity)


@dataclass(frozen=True)
class _FileAttestation:
    path: Path
    digest: str
    device: int
    inode: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode


@dataclass(frozen=True)
class _ValidatedCheckpoint:
    attestation: _FileAttestation
    table: str
    attempted: frozenset[str]
    document: dict

    @property
    def path(self) -> Path:
        return self.attestation.path


@dataclass(frozen=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    digest: str
    contents: bytes | None = None

    @property
    def identity(self) -> tuple[int, int]:
        return self.device, self.inode

    def attest(self, path: Path) -> _FileAttestation:
        return _FileAttestation(path=path, digest=self.digest, device=self.device, inode=self.inode)


@dataclass(frozen=True)
class _ClaimedTransactionFile:
    original: _FileAttestation
    claimed_path: Path


def _fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _digest_open_fd(file_descriptor: int) -> tuple[str, os.stat_result]:
    position = os.lseek(file_descriptor, 0, os.SEEK_CUR)
    digest = hashlib.sha256()
    try:
        os.lseek(file_descriptor, 0, os.SEEK_SET)
        for chunk in iter(lambda: os.read(file_descriptor, 1024 * 1024), b""):
            digest.update(chunk)
        opened = os.fstat(file_descriptor)
    finally:
        os.lseek(file_descriptor, position, os.SEEK_SET)
    return "sha256:" + digest.hexdigest(), opened


@dataclass(frozen=True)
class _AtomicPathState:
    identity: tuple[int, int]
    digest: str
    size: int
    mode: int
    nlink: int


def _atomic_write_previous_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp.previous")


def _require_locked_directory(
    directory_fd: int,
    directory: Path,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[int, int]:
    opened = os.fstat(directory_fd)
    try:
        current = directory.lstat()
    except OSError as error:
        raise RuntimeError(f"Checkpoint directory changed while locked: {directory}") from error
    identity = (opened.st_dev, opened.st_ino)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or identity != (current.st_dev, current.st_ino)
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise RuntimeError(f"Checkpoint directory changed while locked: {directory}")
    return identity


def _entry_state_at(directory_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_regular_entry_at(directory_fd: int, name: str) -> tuple[int, os.stat_result]:
    open_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    file_descriptor = os.open(name, open_flags, dir_fd=directory_fd)
    try:
        opened = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"Entry is not a regular file: {name}")
    except BaseException:
        os.close(file_descriptor)
        raise
    return file_descriptor, opened


def _atomic_path_state_at(directory_fd: int, name: str, display_path: Path) -> _AtomicPathState:
    try:
        artifact_fd, _opened = _open_regular_entry_at(directory_fd, name)
        with os.fdopen(artifact_fd, "rb") as artifact:
            digest, opened = _digest_open_fd(artifact.fileno())
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except Exception as error:
        raise RuntimeError(f"Atomic write artifact is not a regular file: {display_path}") from error
    identity = (opened.st_dev, opened.st_ino)
    mode = stat.S_IMODE(opened.st_mode)
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != identity
        or current.st_size != opened.st_size
        or stat.S_IMODE(current.st_mode) != mode
        or current.st_nlink != opened.st_nlink
    ):
        raise RuntimeError(f"Atomic write artifact changed: {display_path}")
    return _AtomicPathState(
        identity=identity,
        digest=digest,
        size=opened.st_size,
        mode=mode,
        nlink=opened.st_nlink,
    )


def _require_atomic_path_state_at(
    directory_fd: int,
    name: str,
    display_path: Path,
    expected: _AtomicPathState,
) -> None:
    if _atomic_path_state_at(directory_fd, name, display_path) != expected:
        raise RuntimeError(f"Atomic write artifact changed: {display_path}")


def _attest_atomic_publication_at(
    file_descriptor: int,
    directory_fd: int,
    name: str,
    display_path: Path,
    *,
    expected_identity: tuple[int, int],
    expected_digest: str,
    expected_size: int,
    expected_mode: int,
) -> None:
    descriptor_digest, descriptor_state = _digest_open_fd(file_descriptor)
    try:
        published_fd, _opened = _open_regular_entry_at(directory_fd, name)
        with os.fdopen(published_fd, "rb") as published_file:
            published_digest, published_state = _digest_open_fd(published_file.fileno())
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except Exception as error:
        raise RuntimeError(f"Atomic write publication changed before attestation: {display_path}") from error
    if (
        not stat.S_ISREG(descriptor_state.st_mode)
        or not stat.S_ISREG(published_state.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (descriptor_state.st_dev, descriptor_state.st_ino) != expected_identity
        or (published_state.st_dev, published_state.st_ino) != expected_identity
        or (current.st_dev, current.st_ino) != expected_identity
        or descriptor_digest != expected_digest
        or published_digest != expected_digest
        or descriptor_state.st_size != expected_size
        or published_state.st_size != expected_size
        or current.st_size != expected_size
        or stat.S_IMODE(descriptor_state.st_mode) != expected_mode
        or stat.S_IMODE(published_state.st_mode) != expected_mode
        or stat.S_IMODE(current.st_mode) != expected_mode
        or descriptor_state.st_nlink != 1
        or published_state.st_nlink != 1
        or current.st_nlink != 1
    ):
        raise RuntimeError(f"Atomic write publication changed before attestation: {display_path}")


def _validated_regular_file_at(
    directory_fd: int,
    name: str,
    display_path: Path,
    expected_digest: str | None,
    *,
    context_path: Path,
    expected_identity: tuple[int, int] | None = None,
    capture_contents: bool = False,
    kind: str,
) -> _FileSnapshot:
    try:
        owned_fd, opened = _open_regular_entry_at(directory_fd, name)
        with os.fdopen(owned_fd, "rb") as owned_file:
            digest = hashlib.sha256()
            captured_chunks: list[bytes] | None = [] if capture_contents else None
            for chunk in iter(lambda: owned_file.read(1024 * 1024), b""):
                digest.update(chunk)
                if captured_chunks is not None:
                    captured_chunks.append(chunk)
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except Exception as error:
        raise RuntimeError(f"Invalid collector {kind} in {context_path}: {display_path}") from error
    snapshot = _FileSnapshot(
        device=opened.st_dev,
        inode=opened.st_ino,
        mode=stat.S_IMODE(opened.st_mode),
        digest="sha256:" + digest.hexdigest(),
        contents=b"".join(captured_chunks) if captured_chunks is not None else None,
    )
    if (
        not stat.S_ISREG(current.st_mode)
        or snapshot.identity != (current.st_dev, current.st_ino)
        or snapshot.mode != stat.S_IMODE(current.st_mode)
        or opened.st_nlink != current.st_nlink
        or (expected_identity is not None and snapshot.identity != expected_identity)
        or (expected_digest is not None and snapshot.digest != expected_digest)
    ):
        raise RuntimeError(f"Collector {kind} changed in {context_path}: {display_path}")
    return snapshot


def _normalize_atomic_replace_state_at(directory_fd: int, path: Path) -> None:
    name = path.name
    reservation_name = atomic_write_reservation_path(Path(name)).name
    previous_name = _atomic_write_previous_path(Path(name)).name
    if _entry_state_at(directory_fd, previous_name) is None:
        return
    if _entry_state_at(directory_fd, reservation_name) is not None:
        if _entry_state_at(directory_fd, name) is not None:
            raise RuntimeError(f"Conflicting atomic write artifacts for {path}")
        previous_state = _atomic_path_state_at(directory_fd, previous_name, path.with_name(previous_name))
        _rename_noreplace_at(previous_name, name, directory_fd)
        _require_atomic_path_state_at(directory_fd, name, path, previous_state)
        return
    previous_state = _atomic_path_state_at(directory_fd, previous_name, path.with_name(previous_name))
    _rename_noreplace_at(previous_name, reservation_name, directory_fd)
    _require_atomic_path_state_at(
        directory_fd,
        reservation_name,
        path.with_name(reservation_name),
        previous_state,
    )


def _atomic_replace_bytes_at(
    directory_fd: int,
    path: Path,
    contents: bytes,
    mode: int,
    *,
    expected_replaced_state: _AtomicPathState | None,
) -> None:
    name = path.name
    reservation_name = atomic_write_reservation_path(Path(name)).name
    reservation_path = path.with_name(reservation_name)
    expected_digest = "sha256:" + hashlib.sha256(contents).hexdigest()
    expected_size = len(contents)
    expected_mode = stat.S_IMODE(mode)
    open_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    reservation_created = False
    try:
        temp_fd = os.open(
            reservation_name,
            open_flags | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory_fd,
        )
        reservation_created = True
    except FileExistsError:
        temp_fd = os.open(reservation_name, open_flags, dir_fd=directory_fd)
    temp_file = os.fdopen(temp_fd, "r+b")
    with temp_file:
        if reservation_created:
            os.fsync(directory_fd)
        opened = os.fstat(temp_file.fileno())
        temp_identity = (opened.st_dev, opened.st_ino)
        current = os.stat(reservation_name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != temp_identity
            or opened.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise RuntimeError(f"Atomic write temporary changed before reuse: {reservation_path}")
        temp_file.seek(0)
        temp_file.truncate()
        temp_file.write(contents)
        temp_file.flush()
        os.fchmod(temp_file.fileno(), expected_mode)
        os.fsync(temp_file.fileno())
        descriptor_digest, descriptor_state = _digest_open_fd(temp_file.fileno())
        if (
            not stat.S_ISREG(descriptor_state.st_mode)
            or (descriptor_state.st_dev, descriptor_state.st_ino) != temp_identity
            or descriptor_digest != expected_digest
            or descriptor_state.st_size != expected_size
            or stat.S_IMODE(descriptor_state.st_mode) != expected_mode
            or descriptor_state.st_nlink != 1
        ):
            raise RuntimeError(f"Atomic write temporary changed before publication: {reservation_path}")
        _require_atomic_path_state_at(
            directory_fd,
            reservation_name,
            reservation_path,
            _AtomicPathState(
                identity=temp_identity,
                digest=expected_digest,
                size=expected_size,
                mode=expected_mode,
                nlink=1,
            ),
        )

        previous_name = _atomic_write_previous_path(Path(name)).name
        current_present = _entry_state_at(directory_fd, name) is not None
        if expected_replaced_state is None:
            if current_present:
                raise RuntimeError(f"Atomic write target changed before replacement: {path}")
        else:
            if not current_present:
                raise RuntimeError(f"Atomic write target changed before replacement: {path}")
            _require_atomic_path_state_at(directory_fd, name, path, expected_replaced_state)

        if current_present:
            if _entry_state_at(directory_fd, previous_name) is not None:
                raise RuntimeError(f"Conflicting atomic write artifacts for {path}")
            _rename_noreplace_at(name, previous_name, directory_fd)
            try:
                _require_atomic_path_state_at(
                    directory_fd,
                    previous_name,
                    path.with_name(previous_name),
                    expected_replaced_state,
                )
            except Exception as error:
                if _entry_state_at(directory_fd, name) is None:
                    try:
                        _rename_noreplace_at(previous_name, name, directory_fd)
                    except Exception as restore_error:
                        raise RuntimeError(f"Atomic write target changed during replacement: {path}") from restore_error
                raise RuntimeError(f"Atomic write target changed during replacement: {path}") from error
            _rename_noreplace_at(reservation_name, name, directory_fd)
            _attest_atomic_publication_at(
                temp_file.fileno(),
                directory_fd,
                name,
                path,
                expected_identity=temp_identity,
                expected_digest=expected_digest,
                expected_size=expected_size,
                expected_mode=expected_mode,
            )
            _rename_noreplace_at(previous_name, reservation_name, directory_fd)
            _require_atomic_path_state_at(
                directory_fd,
                reservation_name,
                reservation_path,
                expected_replaced_state,
            )
            _attest_atomic_publication_at(
                temp_file.fileno(),
                directory_fd,
                name,
                path,
                expected_identity=temp_identity,
                expected_digest=expected_digest,
                expected_size=expected_size,
                expected_mode=expected_mode,
            )
            return

        _rename_noreplace_at(reservation_name, name, directory_fd)
        _attest_atomic_publication_at(
            temp_file.fileno(),
            directory_fd,
            name,
            path,
            expected_identity=temp_identity,
            expected_digest=expected_digest,
            expected_size=expected_size,
            expected_mode=expected_mode,
        )


@dataclass(frozen=True)
class _LockedCheckpointDirectory:
    path: Path
    file_descriptor: int
    identity: tuple[int, int]


@dataclass(frozen=True)
class _EffectiveCheckpointState:
    canonical: _AtomicPathState | None
    reservation: _AtomicPathState | None
    previous: _AtomicPathState | None
    effective_name: str
    snapshot: _FileSnapshot


def _effective_checkpoint_state_at(
    directory_fd: int,
    path: Path,
    *,
    context_path: Path,
    kind: str,
) -> _EffectiveCheckpointState:
    name = path.name
    reservation_name = atomic_write_reservation_path(Path(name)).name
    previous_name = _atomic_write_previous_path(Path(name)).name

    def state(entry_name: str) -> _AtomicPathState | None:
        return (
            _atomic_path_state_at(directory_fd, entry_name, path.with_name(entry_name))
            if _entry_state_at(directory_fd, entry_name) is not None
            else None
        )

    canonical = state(name)
    reservation = state(reservation_name)
    previous = state(previous_name)
    if previous is None and canonical is not None:
        effective_name = name
    elif previous is not None and canonical is None and reservation is not None:
        effective_name = previous_name
    elif previous is not None and canonical is not None and reservation is None:
        effective_name = name
    else:
        raise RuntimeError(f"Conflicting atomic write artifacts for {path}")
    snapshot = _validated_regular_file_at(
        directory_fd,
        effective_name,
        path,
        None,
        context_path=context_path,
        capture_contents=True,
        kind=kind,
    )
    return _EffectiveCheckpointState(
        canonical=canonical,
        reservation=reservation,
        previous=previous,
        effective_name=effective_name,
        snapshot=snapshot,
    )


def _require_effective_checkpoint_state_at(
    directory_fd: int,
    path: Path,
    expected: _EffectiveCheckpointState,
) -> None:
    entries = (
        (path.name, expected.canonical),
        (atomic_write_reservation_path(Path(path.name)).name, expected.reservation),
        (_atomic_write_previous_path(Path(path.name)).name, expected.previous),
    )
    for name, expected_state in entries:
        if expected_state is None:
            if _entry_state_at(directory_fd, name) is not None:
                raise RuntimeError(f"Checkpoint atomic write state changed: {path}")
        else:
            _require_atomic_path_state_at(directory_fd, name, path.with_name(name), expected_state)


@contextlib.contextmanager
def _locked_checkpoint_root(directory: Path):
    directory = directory.absolute()
    with perf_finalization_lifecycle(directory, resolve_path=False) as locked_root:
        directory_fd = locked_root.file_descriptor
        identity = _require_locked_directory(directory_fd, directory)
        locked = _LockedCheckpointDirectory(
            path=directory,
            file_descriptor=directory_fd,
            identity=identity,
        )
        yield locked
        _require_locked_directory(directory_fd, directory, identity)


@contextlib.contextmanager
def _locked_checkpoint_directory(path: Path):
    with _locked_checkpoint_root(path.parent) as locked:
        yield locked


def _checkpoint_path_in_locked_directory(path: Path, locked: _LockedCheckpointDirectory) -> Path:
    name = path.name
    if not name or name in (".", "..") or "\0" in name or Path(name).name != name:
        raise RuntimeError(f"Checkpoint path is not a direct entry: {path}")
    canonical_path = locked.path / name
    if path.parent.absolute() != locked.path:
        raise RuntimeError(f"Checkpoint path is outside its locked directory: {path}")
    _require_locked_directory(locked.file_descriptor, locked.path, locked.identity)
    return canonical_path


def _checkpoint_snapshot_at(
    path: Path,
    expected_digest: str | None,
    *,
    locked: _LockedCheckpointDirectory,
    context_path: Path,
    expected_identity: tuple[int, int] | None = None,
    capture_contents: bool = False,
    kind: str = "checkpoint",
) -> _FileSnapshot:
    canonical_path = _checkpoint_path_in_locked_directory(path, locked)
    _normalize_atomic_replace_state_at(locked.file_descriptor, canonical_path)
    snapshot = _validated_regular_file_at(
        locked.file_descriptor,
        canonical_path.name,
        canonical_path,
        expected_digest,
        context_path=context_path,
        expected_identity=expected_identity,
        capture_contents=capture_contents,
        kind=kind,
    )
    _require_locked_directory(locked.file_descriptor, locked.path, locked.identity)
    return snapshot


def _checkpoint_snapshot(
    path: Path,
    expected_digest: str | None,
    *,
    context_path: Path,
    expected_identity: tuple[int, int] | None = None,
    capture_contents: bool = False,
    kind: str = "checkpoint",
) -> _FileSnapshot:
    with _locked_checkpoint_directory(path) as locked:
        return _checkpoint_snapshot_at(
            path,
            expected_digest,
            locked=locked,
            context_path=context_path,
            expected_identity=expected_identity,
            capture_contents=capture_contents,
            kind=kind,
        )


def _atomic_write_bytes(
    path: Path,
    contents: bytes,
    *,
    mode: int = 0o600,
    _locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Publish new bytes without replacement under a stable output-root lock."""
    if _locked_output_root is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path = path.parent.resolve() / path.name
        with perf_finalization_lifecycle(canonical_path.parent) as locked_root:
            return _atomic_write_bytes(
                canonical_path,
                contents,
                mode=mode,
                _locked_output_root=locked_root,
            )
    locked_root = _locked_output_root
    path_name = locked_root.entry(path)
    reservation = atomic_write_reservation_path(path)
    reservation_name = locked_root.entry(reservation)
    retained = collector_retained_path(path)
    retained_name = locked_root.entry(retained)
    if _entry_state_at(locked_root.file_descriptor, retained_name) is not None:
        if any(
            _entry_state_at(locked_root.file_descriptor, name) is not None for name in (reservation_name, path_name)
        ):
            raise RuntimeError(f"Conflicting atomic write artifacts for {path}")
        retained_state = _atomic_path_state_at(locked_root.file_descriptor, retained_name, retained)
        if retained_state.nlink != 1:
            raise RuntimeError(f"Unowned retained atomic write artifact: {retained}")
        locked_root.rename_noreplace(retained, reservation)
        _require_atomic_path_state_at(
            locked_root.file_descriptor,
            reservation_name,
            reservation,
            retained_state,
        )
    expected_digest = "sha256:" + hashlib.sha256(contents).hexdigest()
    expected_size = len(contents)
    expected_mode = stat.S_IMODE(mode)
    temp_identity: tuple[int, int] | None = None
    temp_path = atomic_write_reservation_path(path)
    open_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    reservation_created = False
    try:
        temp_fd = locked_root.open(temp_path, open_flags | os.O_CREAT | os.O_EXCL, 0o600)
        reservation_created = True
    except FileExistsError:
        temp_fd = locked_root.open(temp_path, open_flags)
    temp_file = os.fdopen(temp_fd, "r+b")
    with temp_file:
        if reservation_created:
            locked_root.fsync()
        opened = os.fstat(temp_file.fileno())
        temp_identity = (opened.st_dev, opened.st_ino)
        current = locked_root.stat(temp_path)
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != temp_identity
            or opened.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise RuntimeError(f"Atomic write temporary changed before reuse: {temp_path}")
        temp_file.seek(0)
        temp_file.truncate()
        temp_file.write(contents)
        temp_file.flush()
        os.fchmod(temp_file.fileno(), expected_mode)
        os.fsync(temp_file.fileno())
        descriptor_digest, descriptor_state = _digest_open_fd(temp_file.fileno())
        if (
            not stat.S_ISREG(descriptor_state.st_mode)
            or (descriptor_state.st_dev, descriptor_state.st_ino) != temp_identity
            or descriptor_digest != expected_digest
            or descriptor_state.st_size != expected_size
            or stat.S_IMODE(descriptor_state.st_mode) != expected_mode
            or descriptor_state.st_nlink != 1
        ):
            raise RuntimeError(f"Atomic write temporary changed before publication: {temp_path}")
        current = locked_root.stat(temp_path)
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != temp_identity
            or current.st_size != expected_size
            or stat.S_IMODE(current.st_mode) != expected_mode
            or current.st_nlink != 1
        ):
            raise RuntimeError(f"Atomic write temporary changed before publication: {temp_path}")
        locked_root.rename_noreplace(temp_path, path)
        _attest_atomic_publication_at(
            temp_file.fileno(),
            locked_root.file_descriptor,
            path_name,
            path,
            expected_identity=temp_identity,
            expected_digest=expected_digest,
            expected_size=expected_size,
            expected_mode=expected_mode,
        )


def _atomic_write_json(
    path: Path,
    data: dict,
    *,
    expected_attestation: _FileAttestation | None = None,
    _locked_directory: _LockedCheckpointDirectory | None = None,
) -> None:
    """Atomically replace one JSON document without exposing a partial file."""
    if _locked_directory is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _locked_checkpoint_directory(path) as locked:
            _atomic_write_json(
                path,
                data,
                expected_attestation=expected_attestation,
                _locked_directory=locked,
            )
        return
    canonical_path = _checkpoint_path_in_locked_directory(path, _locked_directory)
    _normalize_atomic_replace_state_at(_locked_directory.file_descriptor, canonical_path)
    canonical_state = _entry_state_at(_locked_directory.file_descriptor, canonical_path.name)
    expected_replaced_state = (
        _atomic_path_state_at(
            _locked_directory.file_descriptor,
            canonical_path.name,
            canonical_path,
        )
        if canonical_state is not None
        else None
    )
    if expected_attestation is not None:
        if expected_attestation.path != canonical_path:
            raise RuntimeError(f"Checkpoint attested path mismatch: {canonical_path}")
        _validated_regular_file_at(
            _locked_directory.file_descriptor,
            canonical_path.name,
            canonical_path,
            expected_attestation.digest,
            context_path=canonical_path,
            expected_identity=expected_attestation.identity,
            kind="checkpoint",
        )
        if (
            expected_replaced_state is None
            or expected_replaced_state.identity != expected_attestation.identity
            or expected_replaced_state.digest != expected_attestation.digest
        ):
            raise RuntimeError(f"Checkpoint changed before replacement: {canonical_path}")
    _atomic_replace_bytes_at(
        _locked_directory.file_descriptor,
        canonical_path,
        json.dumps(data, indent=2).encode(),
        0o600,
        expected_replaced_state=expected_replaced_state,
    )
    _require_locked_directory(
        _locked_directory.file_descriptor,
        _locked_directory.path,
        _locked_directory.identity,
    )


def _checkpoint_case_ids(checkpoint: object, field: str) -> set[str]:
    """Return one strictly formed checkpoint case-ID ledger."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint document must be an object")
    values = checkpoint.get(field)
    if (
        not isinstance(values, list)
        or any(not isinstance(case_id, str) or not case_id for case_id in values)
        or len(set(values)) != len(values)
    ):
        raise TypeError(f"checkpoint {field} must be a list of unique non-empty case IDs")
    return set(values)


def _load_checkpoint_for_sidecar_mutation(
    checkpoint_path: Path,
    expected_attestation: _FileAttestation | None,
    *,
    action: str,
    _locked_directory: _LockedCheckpointDirectory | None = None,
) -> tuple[dict, _FileAttestation]:
    if expected_attestation is not None and expected_attestation.path != checkpoint_path:
        raise RuntimeError(f"Failed to {action} checkpoint event {checkpoint_path}: attested path mismatch")
    if _locked_directory is None:
        with _locked_checkpoint_directory(checkpoint_path) as locked:
            return _load_checkpoint_for_sidecar_mutation(
                checkpoint_path,
                expected_attestation,
                action=action,
                _locked_directory=locked,
            )
    try:
        snapshot = _checkpoint_snapshot_at(
            checkpoint_path,
            expected_attestation.digest if expected_attestation is not None else None,
            locked=_locked_directory,
            context_path=checkpoint_path,
            expected_identity=expected_attestation.identity if expected_attestation is not None else None,
            capture_contents=True,
            kind="checkpoint",
        )
        data = json.loads(snapshot.contents)
    except Exception as error:
        raise RuntimeError(f"Failed to {action} checkpoint event {checkpoint_path}: {error}") from error
    return data, snapshot.attest(checkpoint_path)


def _close_checkpoint_attempts(
    checkpoint_path: Path,
    attempted_case_ids: set[str],
    *,
    transaction_id: str | None = None,
    expected_attestation: _FileAttestation | None = None,
    journal_attestation: _FileAttestation | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileAttestation:
    """Close only the pending attempts already committed to a sidecar event."""
    if journal_attestation is not None:
        _revalidate_journal_attestation(journal_attestation, locked_output_root)
    with _locked_checkpoint_directory(checkpoint_path) as locked:
        data, live_attestation = _load_checkpoint_for_sidecar_mutation(
            checkpoint_path,
            expected_attestation,
            action="close finalized",
            _locked_directory=locked,
        )
        if transaction_id is not None and data.get(_SIDECAR_TRANSACTION_FIELD) != transaction_id:
            raise RuntimeError(
                f"Failed to close finalized checkpoint event {checkpoint_path}: "
                f"sidecar transaction {data.get(_SIDECAR_TRANSACTION_FIELD)!r} != {transaction_id!r}"
            )

        current_attempted = _checkpoint_case_ids(data, "attempted")
        if transaction_id is not None and current_attempted != attempted_case_ids:
            raise RuntimeError(
                f"Failed to close finalized checkpoint event {checkpoint_path}: live attempts "
                f"{sorted(current_attempted)} != transaction attempts {sorted(attempted_case_ids)}"
            )
        data["attempted"] = sorted(current_attempted - attempted_case_ids)
        if transaction_id is not None:
            data.pop(_SIDECAR_TRANSACTION_FIELD, None)
        data["updated_at"] = datetime.now().isoformat()
        _checkpoint_snapshot_at(
            checkpoint_path,
            live_attestation.digest,
            locked=locked,
            context_path=checkpoint_path,
            expected_identity=live_attestation.identity,
            kind="checkpoint",
        )
        if journal_attestation is not None:
            _revalidate_journal_attestation(journal_attestation, locked_output_root)
        _atomic_write_json(
            checkpoint_path,
            data,
            expected_attestation=live_attestation,
            _locked_directory=locked,
        )
        return _checkpoint_snapshot_at(
            checkpoint_path,
            None,
            locked=locked,
            context_path=checkpoint_path,
            kind="checkpoint",
        ).attest(checkpoint_path)


def _tag_checkpoint_sidecar_transaction(
    checkpoint_path: Path,
    attempted_case_ids: set[str],
    transaction_id: str,
    *,
    expected_attestation: _FileAttestation | None = None,
    journal_attestation: _FileAttestation | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileAttestation:
    """Durably bind pending attempts to one prepared sidecar transaction."""
    if journal_attestation is not None:
        _revalidate_journal_attestation(journal_attestation, locked_output_root)
    with _locked_checkpoint_directory(checkpoint_path) as locked:
        data, live_attestation = _load_checkpoint_for_sidecar_mutation(
            checkpoint_path,
            expected_attestation,
            action="prepare finalized",
            _locked_directory=locked,
        )

        current_attempted = _checkpoint_case_ids(data, "attempted")
        if current_attempted != attempted_case_ids:
            raise RuntimeError(
                f"Failed to prepare finalized checkpoint event {checkpoint_path}: "
                f"live attempts {sorted(current_attempted)} != transaction attempts {sorted(attempted_case_ids)}"
            )
        existing_transaction = data.get(_SIDECAR_TRANSACTION_FIELD)
        if existing_transaction not in (None, transaction_id):
            raise RuntimeError(
                f"Failed to prepare finalized checkpoint event {checkpoint_path}: "
                f"unresolved sidecar transaction {existing_transaction!r}"
            )
        data[_SIDECAR_TRANSACTION_FIELD] = transaction_id
        data["updated_at"] = datetime.now().isoformat()
        _checkpoint_snapshot_at(
            checkpoint_path,
            live_attestation.digest,
            locked=locked,
            context_path=checkpoint_path,
            expected_identity=live_attestation.identity,
            kind="checkpoint",
        )
        if journal_attestation is not None:
            _revalidate_journal_attestation(journal_attestation, locked_output_root)
        _atomic_write_json(
            checkpoint_path,
            data,
            expected_attestation=live_attestation,
            _locked_directory=locked,
        )
        return _checkpoint_snapshot_at(
            checkpoint_path,
            None,
            locked=locked,
            context_path=checkpoint_path,
            kind="checkpoint",
        ).attest(checkpoint_path)


def _resolve_fpm_cli_inputs(parser, resolver):
    """Render expected FPM input-resolution failures through argparse."""

    try:
        return resolver()
    except FPM_INPUT_ERRORS as error:
        parser.error(str(error))


@contextlib.contextmanager
def _collector_model_path(model_path: str | None):
    previous = os.environ.get("COLLECTOR_MODEL_PATH")
    if model_path:
        os.environ["COLLECTOR_MODEL_PATH"] = model_path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)
        else:
            os.environ["COLLECTOR_MODEL_PATH"] = previous


def _get_test_cases_for_model(get_func, model_path: str | None):
    if not model_path:
        return get_func()
    with _collector_model_path(model_path):
        sig = signature(get_func)
        params = sig.parameters
        if "model_path" in params or any(param.kind == Parameter.VAR_KEYWORD for param in params.values()):
            return get_func(model_path=model_path)
        return get_func()


def _requested_ops(ops: list[str] | None, case_plan=None) -> set[str]:
    if ops is not None:
        return set(ops)
    if case_plan is not None:
        return set(case_plan.ops)
    return set()


def _require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required to run collectors. Use --plan-only to inspect collector v2 YAML plans.")
    return torch


def _cuda_available() -> bool:
    return torch is not None and torch.cuda.is_available()


def _xpu_available() -> bool:
    return torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available()


def _wideep_registry_for_backend(backend: str) -> list:
    module_name = f"collector.wideep.{backend}.registry"
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        return []
    if spec is None:
        return []
    return list(importlib.import_module(module_name).REGISTRY)


def _registry_with_requested_wideep(registry: list, backend: str, ops: list[str] | None, case_plan=None) -> list:
    wideep_registry = _wideep_registry_for_backend(backend)
    if not wideep_registry:
        return registry

    requested_ops = _requested_ops(ops, case_plan)
    requested_wideep_ops = requested_ops & {entry.op for entry in wideep_registry}
    if not requested_wideep_ops:
        return registry

    if logger is not None:
        logger.info(f"WideEP registry active for {backend}: {sorted(requested_wideep_ops)}")
    return [*registry, *wideep_registry]


def _checkpoint_backend_root(checkpoint_dir: str | Path, backend: str) -> Path:
    """Resolve one direct backend checkpoint root without following its symlink."""
    base = Path(checkpoint_dir).expanduser().resolve()
    if not backend or Path(backend).name != backend:
        raise RuntimeError(f"Invalid checkpoint backend name: {backend!r}")
    backend_root = base / backend
    if backend_root.is_symlink():
        raise RuntimeError(f"Checkpoint backend directory must not be a symlink: {backend_root}")
    return backend_root.resolve()


def _require_separate_checkpoint_root(output_root: Path, checkpoint_dir: str | Path, backend: str) -> None:
    checkpoint_root = _checkpoint_backend_root(checkpoint_dir, backend)
    if checkpoint_root == output_root.resolve():
        raise RuntimeError(f"Collector output and checkpoint roots must be different: {output_root}")


def _checkpoint_path(checkpoint_root: Path, module_name: str) -> Path:
    safe_name = module_name.replace("/", "_").replace(":", "_")
    return checkpoint_root / f"{safe_name}.json"


class ResumeCheckpoint:
    """Tracks which tasks are done so a collection run can be resumed.

    Always writes checkpoint files.  When ``--resume`` is passed the existing
    checkpoint is loaded and done tasks are skipped; otherwise the checkpoint
    is overwritten from scratch (so a future ``--resume`` can pick up).
    """

    FLUSH_INTERVAL_SEC = 2.0

    def __init__(
        self,
        backend: str,
        module_name: str,
        run_func_name: str,
        checkpoint_dir: str,
        framework_version: str | None = None,
        sm_version: int | None = None,
    ):
        self.module_name = module_name
        self._dirty = False
        self._last_flush = 0.0
        # framework_version/sm_version bind the checkpoint to the runtime it
        # was collected under: resuming a plan across a version bump or on a
        # different GPU generation silently mislabels data, so it must fail.
        self._metadata = {
            "schema": RESUME_SCHEMA_VERSION,
            "backend": backend,
            "module": module_name,
            "run_func": run_func_name,
            "framework_version": framework_version,
            "sm_version": sm_version,
        }
        self._done: set[str] = set()
        self._failed: set[str] = set()
        # Pending-event scoped, unlike the cumulative resume sets above. It is
        # preserved across interrupted resumes and cleared only after the
        # common finalizer successfully writes the matching sidecar event.
        self._attempted: set[str] = set()
        self._source_digest: str | None = None
        self._source_device: int | None = None
        self._source_inode: int | None = None

        self._path = _checkpoint_path(_checkpoint_backend_root(checkpoint_dir, backend), module_name)

    def load_existing(self):
        """Load an existing checkpoint for resume.  Raises on mismatch."""
        if not self._path.parent.exists() and not self._path.parent.is_symlink():
            logger.info(f"{self.module_name}: no checkpoint found, starting fresh")
            return
        with _locked_checkpoint_directory(self._path) as locked:
            canonical_path = _checkpoint_path_in_locked_directory(self._path, locked)
            _normalize_atomic_replace_state_at(locked.file_descriptor, canonical_path)
            if _entry_state_at(locked.file_descriptor, canonical_path.name) is None:
                logger.info(f"{self.module_name}: no checkpoint found, starting fresh")
                return

            try:
                snapshot = _validated_regular_file_at(
                    locked.file_descriptor,
                    canonical_path.name,
                    canonical_path,
                    None,
                    context_path=canonical_path,
                    capture_contents=True,
                    kind="checkpoint",
                )
                data = json.loads(snapshot.contents)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load checkpoint {canonical_path}: {e}. Run without --resume to start fresh."
                ) from e
            _require_locked_directory(locked.file_descriptor, locked.path, locked.identity)

        for key in ("schema", "backend", "module", "run_func", "framework_version", "sm_version"):
            if data.get(key) != self._metadata[key]:
                raise RuntimeError(
                    f"{self.module_name}: checkpoint mismatch "
                    f"({key}: {data.get(key)} != {self._metadata[key]}). "
                    "Run without --resume to start fresh."
                )

        try:
            ledgers = {field: _checkpoint_case_ids(data, field) for field in ("done", "failed", "attempted")}
        except TypeError as error:
            raise RuntimeError(f"{self.module_name}: {error}") from error
        if data.get(_SIDECAR_TRANSACTION_FIELD) is not None:
            raise RuntimeError(f"{self.module_name}: checkpoint has an unresolved sidecar transaction")

        self._done = ledgers["done"]
        self._failed = ledgers["failed"]
        self._attempted = ledgers["attempted"]
        self._source_digest = snapshot.digest
        self._source_device = snapshot.device
        self._source_inode = snapshot.inode
        logger.info(f"{self.module_name}: loaded checkpoint — {len(self._done)} passed, {len(self._failed)} failed")

    # -- public API -------------------------------------------------------

    def start_fresh(self) -> None:
        """Persist an empty lifecycle before a non-resume run consumes work."""
        self._done.clear()
        self._failed.clear()
        self._attempted.clear()
        self._dirty = True
        self.flush(force=True)

    def filter_done(self, task_infos: list[dict], retry_failed: bool = False) -> list[dict]:
        """Return only tasks that need to run.

        By default, skips both passed and failed tasks. With retry_failed=True,
        previously failed tasks are retried.
        """
        skip_set = self._done if retry_failed else (self._done | self._failed)
        runnable = [t for t in task_infos if t["id"] not in skip_set]
        skipped_done = sum(1 for t in task_infos if t["id"] in self._done)
        skipped_failed = sum(1 for t in task_infos if t["id"] in self._failed)
        retrying = sum(1 for t in runnable if t["id"] in self._failed) if retry_failed else 0
        if skipped_done or skipped_failed or retrying:
            parts = [f"skipping {skipped_done} passed"]
            if retry_failed:
                parts.append(f"retrying {retrying} previously failed")
            else:
                parts.append(f"skipping {skipped_failed} failed")
            parts.append(f"running {len(runnable)}")
            logger.info(f"{self.module_name}: {', '.join(parts)}")
        return runnable

    def mark_passed(self, task_id: str):
        """Mark a task as successfully completed. Skipped on resume."""
        self._done.add(task_id)
        self._failed.discard(task_id)  # if it was previously failed, it passed now
        self._dirty = True
        self.flush()

    def mark_attempted(self, task_id: str):
        """Record that the pending provenance event consumed one task."""
        self._attempted.add(task_id)
        self._dirty = True
        self.flush()

    def mark_attempted_many(self, task_ids: Iterable[str]) -> None:
        """Atomically persist a batch that one full-node runner will consume."""
        self._attempted.update(task_ids)
        self._dirty = True
        self.flush(force=True)

    def mark_failed(self, task_id: str):
        """Mark a task as failed in cumulative resume state."""
        self._failed.add(task_id)
        self._dirty = True
        self.flush()

    def unresolved_failed_count(self) -> int:
        """Number of tasks the checkpoint holds as failed and unresolved."""
        return len(self._failed)

    # Keep mark_done as alias for backwards compat
    mark_done = mark_passed

    def flush(self, force: bool = False):
        if not self._dirty:
            return
        now = time.time()
        if not force and (now - self._last_flush) < self.FLUSH_INTERVAL_SEC:
            return

        data = {
            **self._metadata,
            "updated_at": datetime.now().isoformat(),
            "done": sorted(self._done),
            "failed": sorted(self._failed),
            "attempted": sorted(self._attempted),
        }
        _atomic_write_json(self._path, data)
        self._dirty = False
        self._last_flush = now


class ProfilerContext:
    """Context manager for profiling collector execution"""

    def __init__(self, backend: str, enabled: bool = False):
        self.enabled = enabled
        self.backend = backend
        self.profiler = None
        self.start_time = None
        self.log_dir = None

    def __enter__(self):
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            self.start_time = time.perf_counter()
            self.log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
            if not self.log_dir:
                self.log_dir = "."
            logger.info("Profiling enabled - running sequentially in main process (no parallel workers)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled or self.profiler is None:
            return

        self.profiler.disable()
        profile_file = os.path.join(self.log_dir, f"collector_profile_{self.backend}.prof")
        self.profiler.dump_stats(profile_file)

        # Calculate elapsed time
        end_time = time.perf_counter()
        elapsed_time = end_time - self.start_time if self.start_time else 0

        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total elapsed time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")
        logger.info(f"Profile file: {profile_file}")
        logger.info("=" * 80)

        # Print slow operations ranked by tottime and cumtime
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()

        # Get top functions by tottime (time spent in the function itself, excluding subcalls)
        logger.info("Top 20 functions by tottime (time in function excluding subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        import sys

        old_stdout = sys.stdout
        sys.stdout = stream
        try:
            stats.sort_stats("tottime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        # Get top functions by cumtime (cumulative time including subcalls)
        logger.info("=" * 80)
        logger.info("Top 20 functions by cumtime (cumulative time including subcalls):")
        logger.info("=" * 80)
        stream = io.StringIO()
        sys.stdout = stream
        try:
            stats.sort_stats("cumtime")
            stats.print_stats(20)
        finally:
            sys.stdout = old_stdout
        for line in stream.getvalue().split("\n"):
            if line.strip():
                logger.info(line)

        logger.info("=" * 80)
        logger.info(f"Full profile saved to: {profile_file}")


def _failure_group(task) -> str | None:
    """Group label for failure aggregation: one (model, dtype) family within an op.

    A whole group failing is the signal that something needs FIXING (collector
    bug, unverified combo, framework gap) — the summary aggregates failures by
    this label so systemic groups are visible at a glance. Returns None when
    the task carries neither a model nor a dtype attribute (e.g. positional
    tuple cases).
    """
    from collector.capabilities import case_dtypes

    model = getattr(task, "model_name", None) or getattr(task, "model_path", None) or ""
    dtypes = ",".join(case_dtypes(task))
    if not model and not dtypes:
        return None
    return f"{model}|{dtypes}"


def _is_cuda_fatal_exception(exc, torch_mod) -> bool:
    fatal_error_types = tuple(
        error_type
        for error_type in (
            getattr(torch_mod, "AcceleratorError", None),
            getattr(torch_mod, "OutOfMemoryError", None),
        )
        if isinstance(error_type, type)
    )
    is_cuda_fatal = isinstance(exc, fatal_error_types)
    if not is_cuda_fatal:
        error_text = str(exc).lower()
        fatal_markers = (
            "illegal memory access",
            "unspecified launch failure",
            "cuda_error_launch_failed",
            "cublas_status_execution_failed",
            "cublas_status_internal_error",
            "cublas_status_alloc_failed",
        )
        is_cuda_fatal = any(marker in error_text for marker in fatal_markers)
    if not is_cuda_fatal:
        # DSLCudaRuntimeError from CUTLASS DSL also corrupts CUDA context but
        # is not a torch.AcceleratorError subclass.
        is_cuda_fatal = type(exc).__name__ == "DSLCudaRuntimeError"
    return is_cuda_fatal


def collect_module_safe(
    module_name,
    test_type,
    get_test_cases_func,
    run_func,
    num_processes,
    resume_options=None,
):
    """
    Safely collect module with comprehensive error handling

    Args:
        num_processes: Number of parallel processes to use. If 0, runs sequentially in main process.
    """
    full_name = f"{module_name}.{test_type}"
    logger.info(f"Starting collection: {full_name}")

    try:
        # Get test cases
        test_cases = get_test_cases_func()
        logger.info(f"Generated {len(test_cases)} test cases for {full_name}")

        # Run collection
        errors = parallel_run(
            test_cases,
            run_func,
            num_processes,
            full_name,
            resume_options=resume_options,
        )

        return errors

    except Exception as e:
        logger.exception(f"Failed to collect {full_name}")
        return [
            {
                "module": full_name,
                "error_type": "ModuleCollectionFailure",
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }
        ]


def worker(
    queue,
    device_id: int,
    func,
    progress_value,
    lock,
    error_queue=None,
    done_tasks=None,
    failed_tasks=None,
    attempted_tasks=None,
    module_name="unknown",
    current_task_ids=None,
    consumed_sentinel=None,
):
    """worker with automatic logging setup"""

    # Disable core dumps — GPU crashes are expected and handled via error_queue;
    # without this, each SIGSEGV/SIGABRT writes a multi-GB core file to disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    setup_warning_filters()  # Must run in each spawned process

    # Setup logging for this worker - reads config from environment automatically
    worker_logger = setup_logging(worker_id=device_id)

    # Setup signal handlers
    setup_signal_handlers(device_id)

    # Setup device
    torch_mod = _require_torch()
    device = torch_mod.device(f"{get_device_str()}:{device_id}")
    get_device_module().set_device(device)
    worker_logger.info(f"Worker {device_id} initialized for {module_name}")

    # Process tasks
    while True:
        task_info = queue.get()
        if task_info is None:
            if current_task_ids is not None:
                current_task_ids[device_id] = None
            if consumed_sentinel is not None:
                consumed_sentinel[device_id] = True
            worker_logger.debug("Received termination signal")
            break

        # Handle both old format (tuple) and new format (dict)
        if isinstance(task_info, dict):
            task_id = task_info.get("id", "unknown")
            task = task_info.get("params", task_info)
        else:
            task = task_info
            task_id = create_test_case_id(task, "unknown", module_name)

        if attempted_tasks is not None:
            # Record consumption before publishing the active-task pointer so
            # a hard kill between manager RPCs cannot classify a consumed case
            # as failed while omitting it from the pending event's attestation.
            attempted_tasks[task_id] = True
        if current_task_ids is not None:
            current_task_ids[device_id] = task_id

        try:
            worker_logger.debug(f"Starting task {task_id}")
            result = func(*task, device=device)
            # Only the dedicated sentinel requests a recycle: entrypoints also
            # return plain ints (row counts), which must never be mistaken for
            # EXIT_CODE_RESTART.
            if isinstance(result, WorkerRestartSignal):
                raise SystemExit(EXIT_CODE_RESTART)
            worker_logger.debug(f"Completed task {task_id}")

            # Mark done ONLY on success — failed tasks should be retried on resume
            if done_tasks is not None:
                try:
                    done_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID on success so crash handler knows it completed
            if current_task_ids is not None:
                current_task_ids[device_id] = None
        except SystemExit as e:
            # EXIT_CODE_RESTART: task completed successfully, worker exits to free GPU memory
            # (e.g., MOE collectors call sys.exit(EXIT_CODE_RESTART) after finishing)
            if e.code == EXIT_CODE_RESTART:
                if done_tasks is not None:
                    try:
                        done_tasks[task_id] = True
                    except Exception:
                        pass
                if current_task_ids is not None:
                    current_task_ids[device_id] = None
            raise  # re-raise so the worker actually exits
        except Exception as e:
            # Build comprehensive error info
            error_info = {
                "module": module_name,
                "device_id": device_id,
                "task_id": task_id,
                "task_params": str(task),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "classification": "unexpected",
                "group": _failure_group(task),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            }

            # Report error to queue BEFORE any exit
            if error_queue:
                error_queue.put(error_info)

            worker_logger.exception(f"Task {task_id} failed")

            # Track failed task for checkpoint
            if failed_tasks is not None:
                try:
                    failed_tasks[task_id] = True
                except Exception:
                    pass
            # Clear task ID so crash handler knows it was handled
            if current_task_ids is not None:
                current_task_ids[device_id] = None

            # Force flush logs before any potential exit
            for handler in worker_logger.handlers:
                handler.flush()

            if _is_cuda_fatal_exception(e, torch_mod):
                worker_logger.warning(
                    f"Fatal {type(e).__name__} encountered on task {task_id}. "
                    f"Worker {device_id} exiting to reset GPU context. "
                    f"Progress: {progress_value.value}"
                )
                # Flush logs again after warning
                for handler in worker_logger.handlers:
                    handler.flush()
                # Exiting with non-zero code will add an additional error to the summary,
                # which we don't want (error already reported above).
                exit(0)
        finally:
            with lock:
                progress_value.value += 1

            # Periodic memory cleanup to reduce fragmentation
            if progress_value.value % 100 == 0:
                import gc

                gc.collect()
                get_device_module().empty_cache()


def parallel_run(tasks, func, num_processes, module_name="unknown", resume_options=None):
    """parallel runner with error collection

    Args:
        num_processes: Number of parallel processes. If 0, runs sequentially in main process.
    """
    # func may be a functools.partial (perf_filename bound by collect_ops),
    # which lacks __name__. Fall back to partial.func to get the wrapped function.
    func_name = getattr(func, "__name__", None) or getattr(func, "func", func).__name__
    raw_task_infos = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict) and "id" in task and "params" in task:
            task_id = task["id"]
            task_params = task["params"]
        else:
            task_id = create_test_case_id(task, func_name, module_name)
            task_params = task
        raw_task_infos.append({"id": task_id, "params": task_params, "index": i})

    checkpoint_dir = (
        resume_options.get("checkpoint_dir", ".collector_checkpoint") if resume_options else ".collector_checkpoint"
    )
    resume_tracker = ResumeCheckpoint(
        backend=resume_options.get("backend", "unknown") if resume_options else "unknown",
        module_name=module_name,
        run_func_name=func_name,
        checkpoint_dir=checkpoint_dir,
        framework_version=resume_options.get("framework_version") if resume_options else None,
        sm_version=resume_options.get("sm_version") if resume_options else None,
    )

    if resume_options and resume_options.get("resume"):
        resume_tracker.load_existing()
        retry_failed = resume_options.get("retry_failed", False)
        task_infos = resume_tracker.filter_done(raw_task_infos, retry_failed=retry_failed)
    else:
        resume_tracker.start_fresh()
        task_infos = raw_task_infos

    def _unresolved_failure_errors():
        # A resumed run must not look clean while its checkpoint still holds
        # unresolved failures: completion and acceptance are distinct.
        if not (resume_options and resume_options.get("resume")):
            return []
        unresolved = resume_tracker.unresolved_failed_count()
        if not unresolved:
            return []
        logger.warning(
            f"{module_name}: checkpoint holds {unresolved} unresolved failed tasks "
            "(skipped on resume; rerun with --resume-retry-failed to retry)"
        )
        return [
            {
                "module": module_name,
                "task_id": "resume_unresolved",
                "error_type": "UnresolvedFailures",
                "error_message": f"checkpoint holds {unresolved} unresolved failed tasks",
                "classification": "unresolved_from_checkpoint",
                "timestamp": datetime.now().isoformat(),
            }
        ]

    if not task_infos:
        logger.info(f"{module_name}: no tasks to run")
        return _unresolved_failure_errors()

    queue = mp.Queue()
    error_queue = mp.Queue()
    processes = []

    manager = mp.Manager()
    progress_value = manager.Value("i", 0)
    lock = manager.Lock()

    # Track process health
    process_stats = {i: {"restarts": 0, "errors": []} for i in range(num_processes)}

    # Per-worker flag: True once a worker has consumed its None sentinel.
    # Used to decide whether a replacement sentinel is needed on restart.
    consumed_sentinel = manager.dict(dict.fromkeys(range(num_processes), False))
    current_task_ids = manager.dict(dict.fromkeys(range(num_processes), None))
    # Synchronous record of completed task IDs.  Workers write here via
    # manager RPC in their finally block — same mechanism as progress_value,
    # so it is guaranteed to be visible before the worker touches the next
    # task.  Unlike mp.Queue (async feeder thread) this cannot be lost when
    # a worker is killed by a signal on a subsequent task.
    done_tasks = manager.dict()
    failed_tasks = manager.dict()
    attempted_tasks = manager.dict()

    def start_process(device_id):
        p = mp.Process(
            target=worker,
            args=(
                queue,
                device_id,
                func,
                progress_value,
                lock,
                error_queue,
                done_tasks,
                failed_tasks,
                attempted_tasks,
                module_name,
                current_task_ids,
                consumed_sentinel,
            ),
        )
        p.start()
        logger.info(f"Started worker process {p.pid} on device {device_id}")
        return p

    def create_process_exit_error(device_id, exit_code):
        if exit_code in (None, 0, EXIT_CODE_RESTART):
            return None

        if exit_code < 0:
            signum = -exit_code
            try:
                signame = signal.Signals(signum).name
            except Exception:
                signame = f"SIG{signum}"
            reason = f"terminated by signal {signum} ({signame})"
            error_type = "WorkerSignalCrash"
        else:
            reason = f"exited with status {exit_code}"
            error_type = "WorkerAbnormalExit"

        logger.error(f"Process {device_id} ({module_name}) {reason}")

        return {
            "module": module_name,
            "device_id": device_id,
            "task_id": "process_exit",
            "task_params": None,
            "error_type": error_type,
            "error_message": reason,
            "traceback": "",
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
        }

    # Parent-side exactly-once ledger of finished task IDs. The monitoring
    # loop's exit condition MUST NOT depend on worker-side progress_value
    # ticks: a worker records done_tasks/failed_tasks and bumps
    # progress_value as separate manager RPCs, so a hard kill between them
    # (hardware-observed on H20 2026-07-19: TRT-LLM C++ teardown SIGABRT
    # right after the task completed) loses the tick forever — the run then
    # sat at 99/100 with a complete checkpoint and every GPU idle. Every
    # task that reaches done_tasks or failed_tasks lands here exactly once
    # via sync_done_to_checkpoint, so `len(accounted)` is the authoritative
    # completion count; progress_value remains for worker-side GC cadence.
    accounted = set()

    def sync_done_to_checkpoint():
        for task_id in list(attempted_tasks.keys()):
            resume_tracker.mark_attempted(task_id)
            try:
                del attempted_tasks[task_id]
            except KeyError:
                pass
        for task_id in list(done_tasks.keys()):
            resume_tracker.mark_passed(task_id)
            accounted.add(task_id)
            try:
                del done_tasks[task_id]
            except KeyError:
                pass
        for task_id in list(failed_tasks.keys()):
            resume_tracker.mark_failed(task_id)
            accounted.add(task_id)
            try:
                del failed_tasks[task_id]
            except KeyError:
                pass

    # Start processes
    for device_id in range(num_processes):
        processes.append(start_process(device_id))

    # Queue tasks with IDs
    for task_info in task_infos:
        queue.put(task_info)

    # Add termination signals
    for _ in range(len(processes)):
        queue.put(None)

    # Monitor progress with error collection
    errors = []

    with tqdm(total=len(task_infos), desc=f"{module_name}", dynamic_ncols=True, leave=True) as pbar:
        last_progress = 0
        stall_count = 0
        last_error_count = 0

        if num_processes == 0:
            # Special handling for --profile
            # Run tasks sequentially in main process
            torch_mod = _require_torch()
            device = torch_mod.device(f"{get_device_str()}:0")
            get_device_module().set_device(device)

            for task_info in task_infos:
                task_id = task_info["id"]
                task_params = task_info["params"]
                resume_tracker.mark_attempted(task_id)

                try:
                    func(*task_params, device=device)
                    resume_tracker.mark_passed(task_id)
                    accounted.add(task_id)
                except Exception as e:
                    resume_tracker.mark_failed(task_id)
                    accounted.add(task_id)
                    error_info = {
                        "module": module_name,
                        "device_id": 0,
                        "task_id": task_id,
                        "task_params": str(task_params),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "classification": "unexpected",
                        "group": _failure_group(task_params),
                        "traceback": traceback.format_exc(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    errors.append(error_info)
                    logger.exception(f"Task {task_id} failed")

                pbar.update(1)
                progress_value.value += 1
                if len(errors) > 0:
                    pbar.set_postfix({"errors": len(errors)})
                resume_tracker.flush()
            resume_tracker.flush(force=True)

        while len(accounted) < len(task_infos):
            # Drain errors
            while not error_queue.empty():
                error = error_queue.get()
                errors.append(error)
                process_stats[error["device_id"]]["errors"].append(error["task_id"])
            sync_done_to_checkpoint()

            # Update postfix only if count changed
            if len(errors) != last_error_count:
                pbar.set_postfix({"errors": len(errors)})
                last_error_count = len(errors)

            if len(accounted) == last_progress:
                stall_count += 1
                if stall_count > STALL_THRESHOLD:
                    logger.warning(f"Progress stalled at {len(accounted)}/{len(task_infos)}")
                    stall_count = 0
            else:
                stall_count = 0
                last_progress = len(accounted)

            # Check process health — only restart if there is still work
            # remaining.  Workers that consumed a None sentinel or finished
            # via sys.exit(EXIT_CODE_RESTART) should not be restarted once
            # all tasks are dispatched, otherwise the new worker blocks
            # forever on queue.get().
            for i, p in enumerate(processes):
                if p is None:
                    continue

                if not p.is_alive():
                    exit_code = p.exitcode
                    active_task_id = current_task_ids.get(i)
                    process_stats[i]["restarts"] += 1
                    if exit_code == EXIT_CODE_RESTART:
                        logger.debug(
                            f"Process {i} completed task and exited normally for release gpu memory"
                            f"(completed tasks: {process_stats[i]['restarts']})"
                        )
                    else:
                        logger.warning(
                            f"Process {i} died (exit code: {exit_code}, "
                            f"restarts: {process_stats[i]['restarts']}, "
                            f"errors: {len(process_stats[i]['errors'])})"
                        )

                    # Mark active task as failed if the process died while
                    # running it. The `accounted` guard covers the window
                    # where the worker recorded done but died before its
                    # progress tick and the sync already drained done_tasks:
                    # without it the parent would re-mark a passed task as
                    # failed (mislabel) on top of double-counting it.
                    if (
                        active_task_id is not None
                        and active_task_id not in done_tasks
                        and active_task_id not in accounted
                    ):
                        try:
                            failed_tasks[active_task_id] = True
                        except Exception:
                            pass
                        current_task_ids[i] = None
                        with lock:
                            progress_value.value += 1

                    crash_error = create_process_exit_error(i, exit_code)
                    if crash_error:
                        errors.append(crash_error)
                        process_stats[i]["errors"].append("process_exit")
                        pbar.set_postfix({"errors": len(errors)})
                        last_error_count = len(errors)

                    if process_stats[i]["restarts"] > 8192:
                        logger.error(f"Process {i} exceeded restart limit, not restarting")
                        processes[i] = None
                        continue

                    if consumed_sentinel.get(i, False):
                        processes[i] = None
                        continue

                    remaining = len(task_infos) - len(accounted)
                    if remaining > 0:
                        processes[i] = start_process(i)
                    else:
                        processes[i] = None

            # Escape hatch: every worker slot is permanently gone (restart
            # limit, consumed sentinel, or no remaining work to justify a
            # restart) while tasks are still unaccounted — e.g. a worker was
            # SIGKILLed after queue.get() but before it recorded anything in
            # done_tasks/current_task_ids. Without this the monitor loop
            # would spin at 0.5s forever with nothing able to make progress.
            # Record the orphaned ids as failures so the summary and resume
            # checkpoint stay complete, then stop the loop.
            if all(p is None for p in processes) and len(accounted) < len(task_infos):
                while not error_queue.empty():
                    error = error_queue.get()
                    errors.append(error)
                    process_stats[error["device_id"]]["errors"].append(error["task_id"])
                sync_done_to_checkpoint()
                orphaned = [info["id"] for info in task_infos if info["id"] not in accounted]
                if orphaned:
                    logger.error(
                        f"All workers exited with {len(orphaned)} task(s) unaccounted; "
                        f"recording them as failed (orphaned by worker death) and stopping the monitor loop."
                    )
                    for task_id in orphaned:
                        errors.append(
                            {
                                "module": module_name,
                                "device_id": None,
                                "task_id": task_id,
                                "task_params": None,
                                "error_type": "WorkerOrphanedTask",
                                "error_message": "all workers exited before this task was accounted",
                                "classification": "unexpected",
                                "group": None,
                                "traceback": "",
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        failed_tasks[task_id] = True
                    sync_done_to_checkpoint()
                break

            current = len(accounted)
            if current > pbar.n:
                pbar.update(current - pbar.n)

            resume_tracker.flush()
            time.sleep(0.5)
        sync_done_to_checkpoint()

    # Collect remaining errors
    while not error_queue.empty():
        errors.append(error_queue.get())
    sync_done_to_checkpoint()
    resume_tracker.flush(force=True)

    # Wait for processes
    for p in processes:
        if p is None:
            continue
        p.join(timeout=42)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, forcing...")
            p.terminate()

    # Shutdown manager to clean up resources (semaphores, etc.)
    manager.shutdown()

    # Surface systemic failure groups — a whole (model, dtype) family failing
    # is a fix-me signal, not something to tolerate.
    group_counts = Counter(error["group"] for error in errors if error.get("group"))
    for group, count in sorted(group_counts.items()):
        if count >= SYSTEMIC_GROUP_THRESHOLD:
            logger.warning(f"{module_name}: failure group {group!r} failed {count} times — needs fixing")

    errors.extend(_unresolved_failure_errors())

    # Log summary
    if errors:
        log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")
        logger.error(f"{module_name}: Completed with {len(errors)} errors")
        error_file = f"{log_dir}/errors_{module_name}.json"
        save_error_report(errors, error_file)
        logger.error(f"Error details saved to {error_file}")
    else:
        logger.info(f"{module_name}: Completed successfully with no errors")

    return errors


def collect_ops(
    num_processes: int,
    collections: list[dict],
    runtime_version: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    shuffle_seed: int = 42,
    backend: str = "unknown",
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
) -> list[dict]:
    """Run collection for a list of resolved collection entries.

    Each entry must have: name, type, module, get_func, run_func.
    Version resolution and op filtering are handled upstream by
    version_resolver.build_collections(). If runtime_version is provided,
    per-module __compat__ is validated and incompatible ops fail explicitly.
    If limit is provided, the number of test cases is limited to the limit.
    If shuffle is True, the test cases are shuffled with the given seed.
    """

    class CompatibilityError(RuntimeError):
        """Raised when a resolved collector module is incompatible."""

    check_compat = None
    if runtime_version:
        from collector.version_resolver import _check_compat as check_compat

    all_errors = []

    for collection in collections:
        try:
            if case_plan is not None and not case_plan.has_op(collection["type"]):
                logger.info(f"Skipping {collection['name']}.{collection['type']} — not in collector v2 case plan")
                continue
            unverified_here = collection.get("unverified") or (
                sm_version is not None and sm_version in (collection.get("unverified_sms") or ())
            )
            if unverified_here:
                scope = "this backend" if collection.get("unverified") else f"SM{sm_version}"
                logger.warning(
                    f"Skipping {collection['name']}.{collection['type']} — registry marks it unverified "
                    f"on {scope}; remove the OpEntry marker once the collector is debugged there"
                )
                all_errors.append(
                    {
                        "module": f"{collection['name']}.{collection['type']}",
                        "error_type": "UnverifiedCollector",
                        "error_message": f"registry marks this op unverified on {scope}; collection skipped",
                        "classification": "unverified_skipped",
                    }
                )
                continue
            module_name = collection["module"]
            get_module = __import__(module_name, fromlist=[collection["get_func"]])
            run_module = __import__(module_name, fromlist=[collection["run_func"]])

            # Fail this op explicitly if declared compatibility doesn't match runtime.
            if check_compat:
                declared = getattr(get_module, "__compat__", None)
                if declared:
                    try:
                        if not check_compat(declared, runtime_version):
                            if _xpu_available():
                                # Disable vllm xpu runtime version check for now
                                logger.warning(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                    runtime is v{runtime_version}"
                                )
                            else:
                                raise CompatibilityError(
                                    f"module {module_name} declares __compat__={declared!r}, \
                                        runtime is v{runtime_version}"
                                )
                    except ValueError as e:
                        raise CompatibilityError(f"invalid __compat__ {declared!r}: {e}") from e

            get_func = getattr(get_module, collection["get_func"])
            run_func = getattr(run_module, collection["run_func"])
            run_func = functools.partial(run_func, perf_filename=collection["perf_filename"])

            def get_func_with_limit(get_func=get_func, op=collection["type"]):
                from collector.capabilities import filter_cases

                cases = _get_test_cases_for_model(get_func, model_path)
                cases, _dropped = filter_cases(cases, op=op, sm_version=sm_version)
                if case_filters:
                    before_count = len(cases)
                    cases = [case for case in cases if any(fragment in str(case) for fragment in case_filters)]
                    logger.info(f"{op}: --case-filter kept {len(cases)}/{before_count} cases")
                if shuffle:
                    rng = random.Random(shuffle_seed)
                    rng.shuffle(cases)
                if limit is not None:
                    cases = cases[:limit]
                return cases

            merged_resume = {
                **(resume_options or {}),
                "backend": backend,
                "framework_version": runtime_version,
                "sm_version": sm_version,
            }
            errors = collect_module_safe(
                collection["name"],
                collection["type"],
                get_func_with_limit,
                run_func,
                num_processes,
                resume_options=merged_resume,
            )
            all_errors.extend(errors)

        except Exception as e:
            logger.exception(f"Failed to process {collection['name']}.{collection['type']}")
            all_errors.append(
                {
                    "module": f"{collection['name']}.{collection['type']}",
                    "error_type": "CompatibilityError" if isinstance(e, CompatibilityError) else type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return all_errors


def collect_sglang(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for SGLang with enhanced error tracking"""
    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    # DSV4-Pro mhc-pre fast path: the DeepGEMM tf32 prenorm + TileLang fused
    # kernels must have these names present in os.environ at worker-spawn time,
    # otherwise mhc-pre collects ~53% slow and diverges from the reference
    # dataset. Both default to True in environ.py, but the effect is triggered
    # by presence in the process env (consumed downstream in the JIT path), not
    # just by .get(). setdefault so an explicit caller value still wins.
    os.environ.setdefault("SGLANG_OPT_DEEPGEMM_HC_PRENORM", "1")
    os.environ.setdefault("SGLANG_OPT_USE_TILELANG_MHC_PRE", "1")

    try:
        from importlib.metadata import version as get_version

        version = get_version("sglang")
        logger.info(f"SGLang version: {version}")
    except Exception:
        logger.exception("SGLang is not installed")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = _requested_ops(ops, case_plan)
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("sglang")}
    runtime = require_collector_runtime("sglang", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    from collector.fullnode import SGLANG_FULLNODE_OPS, collect_sglang_fullnode_op
    from collector.sglang.registry import REGISTRY
    from collector.version_resolver import build_collections

    all_errors = []
    registry = _registry_with_requested_wideep(REGISTRY, "sglang", ops, case_plan)
    ops_filter = ops if ops is not None else (case_plan.ops if case_plan is not None else None)
    collections = build_collections(registry, "sglang", version, ops_filter, logger=logger)
    requested_fullnode_ops = requested_ops & set(SGLANG_FULLNODE_OPS)
    fullnode_collections = [collection for collection in collections if collection["type"] in requested_fullnode_ops]
    pool_collections = [collection for collection in collections if collection["type"] not in SGLANG_FULLNODE_OPS]

    if pool_collections:
        all_errors = collect_ops(
            num_processes,
            pool_collections,
            version,
            limit=limit,
            shuffle=shuffle,
            backend="sglang",
            resume_options=resume_options,
            model_path=model_path,
            case_plan=case_plan,
            sm_version=sm_version,
            case_filters=case_filters,
        )

    for collection in fullnode_collections:
        all_errors.extend(
            collect_sglang_fullnode_op(
                collection,
                runtime_version=version,
                limit=limit,
                shuffle=shuffle,
                shuffle_seed=42,
                backend="sglang",
                resume_options=resume_options,
                model_path=model_path,
                case_plan=case_plan,
                sm_version=sm_version,
                case_filters=case_filters,
                get_test_cases_for_model=_get_test_cases_for_model,
                resume_checkpoint_cls=ResumeCheckpoint,
                logger=logger,
            )
        )

    generate_collection_summary(all_errors, "sglang", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "sm_version": sm_version,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def collect_vllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for vLLM"""
    from collector.version_resolver import build_collections

    is_xpu_backend = False
    if _cuda_available():
        from collector.vllm.registry import REGISTRY
    elif _xpu_available():
        from collector.vllm.registry import REGISTRY_XPU as REGISTRY

        is_xpu_backend = True
    else:
        raise RuntimeError("No supported hardware detected. Neither CUDA nor XPU is available.")

    try:
        from vllm.version import __version__ as vllm_version

        version = vllm_version
    except Exception:
        logger.exception("vLLM is not installed. Please install it from https://github.com/vllm-project/vllm")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("vllm")}
    if is_xpu_backend:
        runtime = require_collector_runtime("vllm_xpu", version, requested_ops=requested_ops, wideep_ops=set())
    else:
        runtime = require_collector_runtime("vllm", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    registry = _registry_with_requested_wideep(REGISTRY, "vllm", ops, case_plan)
    collections = build_collections(registry, "vllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="vllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
        sm_version=sm_version,
        case_filters=case_filters,
    )

    generate_collection_summary(all_errors, "vllm", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "sm_version": sm_version,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def collect_trtllm(
    num_processes: int,
    ops: list[str] | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    resume_options: dict | None = None,
    model_path: str | None = None,
    case_plan=None,
    sm_version: int | None = None,
    case_filters: list[str] | None = None,
):
    """Collect performance data for TensorRT LLM with enhanced error tracking"""
    from collector.trtllm.registry import REGISTRY
    from collector.version_resolver import build_collections

    os.environ["TLLM_LOG_LEVEL"] = "ERROR"
    os.environ["TRTLLM_DG_ENABLED"] = "1"
    os.environ["FLASHINFER_LOG_LEVEL"] = "ERROR"

    try:
        with (
            open(os.devnull, "w") as _null,
            contextlib.redirect_stdout(_null),
            contextlib.redirect_stderr(_null),
        ):
            import tensorrt_llm
        version = tensorrt_llm.__version__
        logger.info(f"TensorRT LLM version: {version}")
    except Exception:
        logger.exception("TensorRT LLM is not installed")
        return None, None

    from collector.framework_manifest import require_collector_runtime

    requested_ops = set(ops if ops is not None else (case_plan.ops if case_plan is not None else []))
    wideep_ops = {entry.op for entry in _wideep_registry_for_backend("trtllm")}
    runtime = require_collector_runtime("trtllm", version, requested_ops=requested_ops, wideep_ops=wideep_ops)

    registry = _registry_with_requested_wideep(REGISTRY, "trtllm", ops, case_plan)
    collections = build_collections(registry, "trtllm", version, ops, logger=logger)
    all_errors = collect_ops(
        num_processes,
        collections,
        version,
        limit=limit,
        shuffle=shuffle,
        backend="trtllm",
        resume_options=resume_options,
        model_path=model_path,
        case_plan=case_plan,
        sm_version=sm_version,
        case_filters=case_filters,
    )

    generate_collection_summary(all_errors, "trtllm", version)
    provenance_ctx = {
        "framework": runtime.framework,
        "installed_version": version,
        "runtime": runtime,
        "sm_version": sm_version,
        "collections": collections,
    }
    return all_errors, provenance_ctx


def generate_collection_summary(all_errors, backend, version):
    """Generate comprehensive collection summary"""
    summary = {
        "backend": backend,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "total_errors": len(all_errors),
        "errors_by_module": {},
        "errors_by_type": {},
        "errors_by_group": {},
    }

    for error in all_errors:
        module = error.get("module", "unknown")
        error_type = error.get("error_type", "unknown")

        summary["errors_by_module"][module] = summary["errors_by_module"].get(module, 0) + 1
        summary["errors_by_type"][error_type] = summary["errors_by_type"].get(error_type, 0) + 1
        group = error.get("group")
        if group:
            group_key = f"{module}:{group}"
            summary["errors_by_group"][group_key] = summary["errors_by_group"].get(group_key, 0) + 1

    log_dir = os.environ.get("COLLECTOR_LOG_DIR", "")

    # Save summary
    summary_file = f"{log_dir}/collection_summary_{backend}.json"
    with open(summary_file, "w") as f:
        json.dump({"summary": summary, "errors": all_errors}, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info(f"COLLECTION SUMMARY - {backend} v{version}")
    logger.info("=" * 60)
    logger.info(f"Total errors: {summary['total_errors']}")

    if summary["errors_by_module"]:
        logger.info("\nErrors by module:")
        for module, count in sorted(summary["errors_by_module"].items()):
            logger.info(f"  {module}: {count}")

    if summary["errors_by_type"]:
        logger.info("\nErrors by type:")
        for error_type, count in sorted(summary["errors_by_type"].items()):
            logger.info(f"  {error_type}: {count}")

    if summary["errors_by_group"]:
        logger.info("\nErrors by (model, dtype) group — whole groups failing need fixing:")
        for group, count in sorted(summary["errors_by_group"].items(), key=lambda item: -item[1]):
            logger.info(f"  {group}: {count}")

    logger.info(f"\nDetailed error report saved to: {summary_file}")


def _all_op_names() -> list[str]:
    """Collect all unique op names across normal and WideEP registries."""
    from collector.sglang.registry import REGISTRY as SGLANG_REG
    from collector.trtllm.registry import REGISTRY as TRTLLM_REG
    from collector.vllm.registry import REGISTRY as VLLM_REG

    seen = set()
    ops = []
    registries = [
        TRTLLM_REG,
        VLLM_REG,
        SGLANG_REG,
        _wideep_registry_for_backend("trtllm"),
        _wideep_registry_for_backend("vllm"),
        _wideep_registry_for_backend("sglang"),
    ]
    for reg in registries:
        for entry in reg:
            if entry.op not in seen:
                seen.add(entry.op)
                ops.append(entry.op)
    # Whole-forward FPM collection is an explicit campaign runner rather than
    # a normal per-device OpEntry. Keep it in the existing --ops interface,
    # but do not add it to backend registries consumed by collect_ops().
    from collector.fpm_forward import FPM_FORWARD_OP

    ops.append(FPM_FORWARD_OP)
    return ops


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_collector_ref(repo_root: Path) -> str:
    """The repo SHA the collector ran from (design §5), "unknown" outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        logger.warning(f"collector_ref: `git rev-parse HEAD` failed ({error}); recording 'unknown'")
        return "unknown"


def _split_image_digest(image_ref: str) -> tuple[str, str | None]:
    """Split "repo/image:tag@sha256:<hex>" into (image, digest); digest is None for bare internal images."""
    image, sep, digest = image_ref.partition("@")
    return image, (digest if sep else None)


def _runtime_metadata(provenance_ctx: dict) -> dict[str, str]:
    runtime = provenance_ctx["runtime"]
    image, image_digest = _split_image_digest(runtime.image())
    runtime_meta = {"framework": runtime.framework, "version": runtime.version, "image": image}
    if image_digest:
        runtime_meta["image_digest"] = image_digest
    return runtime_meta


def _load_clean_collector_sidecar(
    output_root: Path,
    *,
    tables_to_update: set[str] | frozenset[str] = frozenset(),
    locked_output_root: LockedOutputRoot | None = None,
) -> tuple[dict | None, _FileSnapshot | None]:
    """Load one regular sidecar only when no transaction artifacts remain."""
    import yaml

    for transaction_path in (
        output_root / _SIDECAR_STAGING_FILENAME,
        output_root / _SIDECAR_TRANSACTION_FILENAME,
    ):
        transaction_absent = (
            locked_output_root.absent(transaction_path)
            if locked_output_root is not None
            else not transaction_path.exists()
        )
        if not transaction_absent:
            raise RuntimeError(f"Unexpected collector sidecar transaction document: {transaction_path}")

    existing_meta = output_root / "collection_meta.yaml"
    if locked_output_root is not None:
        try:
            meta_state = locked_output_root.stat(existing_meta)
        except FileNotFoundError:
            meta_state = None
        if meta_state is not None and not stat.S_ISREG(meta_state.st_mode):
            raise RuntimeError(f"Invalid collector sidecar document: {existing_meta}")
    else:
        if existing_meta.is_symlink() or (existing_meta.exists() and not existing_meta.is_file()):
            raise RuntimeError(f"Invalid collector sidecar document: {existing_meta}")
        meta_state = existing_meta.stat() if existing_meta.is_file() else None
    if meta_state is None:
        return None, None
    try:
        snapshot = _validated_regular_file(
            existing_meta,
            None,
            context_path=existing_meta,
            capture_contents=True,
            kind="sidecar document",
            locked_output_root=locked_output_root,
        )
        existing_doc = yaml.safe_load(snapshot.contents) or {}
    except Exception as error:
        raise RuntimeError(f"Invalid collector sidecar document {existing_meta}: {error}") from error
    if not isinstance(existing_doc, dict):
        raise RuntimeError(  # noqa: TRY004 - malformed persisted state, not a caller argument
            f"Invalid collector sidecar document {existing_meta}: expected an object"
        )
    try:
        from collector import provenance

        provenance.validate_collection_meta_for_update(existing_doc, tables_to_update=tables_to_update)
    except Exception as error:
        raise RuntimeError(f"Invalid collector sidecar document {existing_meta}: {error}") from error
    return existing_doc, snapshot


def _preflight_collector_provenance(
    output_root: Path,
    provenance_ctx: dict,
    *,
    tables_to_update: set[str] | frozenset[str] = frozenset(),
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileSnapshot | None:
    """Reject unsafe sidecar state before perf finalization mutates files."""
    existing_doc, existing_snapshot = _load_clean_collector_sidecar(
        output_root,
        tables_to_update=tables_to_update,
        locked_output_root=locked_output_root,
    )
    if existing_doc is None:
        return existing_snapshot

    existing_meta = output_root / "collection_meta.yaml"
    existing_runtime = existing_doc.get("runtime")
    runtime_meta = _runtime_metadata(provenance_ctx)
    if existing_runtime != runtime_meta:
        raise RuntimeError(
            f"{existing_meta}: cannot finalize collector outputs in place with a different runtime identity "
            f"(existing={existing_runtime!r}, current={runtime_meta!r}). Use a clean output directory for the "
            "new runtime."
        )
    return existing_snapshot


def _revalidate_collector_sidecar_preflight(
    output_root: Path,
    expected_snapshot: _FileSnapshot | None,
    *,
    tables_to_update: set[str] | frozenset[str],
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Require the same clean sidecar state immediately before parquet publish."""
    _document, current_snapshot = _load_clean_collector_sidecar(
        output_root,
        tables_to_update=tables_to_update,
        locked_output_root=locked_output_root,
    )
    if not _same_file_snapshot(current_snapshot, expected_snapshot):
        raise RuntimeError(
            f"Collector sidecar document changed after preflight: {output_root / 'collection_meta.yaml'}"
        )


def _resume_tracker_for_collection(
    collection: dict,
    provenance_ctx: dict,
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
) -> ResumeCheckpoint:
    """Build the same runtime-bound checkpoint contract used by collection."""
    full_name = f"{collection['name']}.{collection['type']}"
    runtime_version = provenance_ctx.get("installed_version")
    if runtime_version is None:
        runtime_version = provenance_ctx["runtime"].version
    return ResumeCheckpoint(
        backend=backend,
        module_name=full_name,
        run_func_name=collection["run_func"],
        checkpoint_dir=checkpoint_dir,
        framework_version=runtime_version,
        sm_version=sm_version,
    )


def _registered_checkpoint_table(identity: dict, *, backend: str) -> str:
    """Resolve a checkpoint producer to its registry-owned table."""
    from collector.version_resolver import resolve_module

    if set(identity) != set(_CHECKPOINT_IDENTITY_FIELDS) or identity.get("backend") != backend:
        raise RuntimeError(f"checkpoint producer identity is not bound to {backend}: {identity!r}")
    framework_version = identity.get("framework_version")
    sm_version = identity.get("sm_version")
    if (
        identity.get("schema") != RESUME_SCHEMA_VERSION
        or not isinstance(identity.get("module"), str)
        or not identity["module"]
        or not isinstance(identity.get("run_func"), str)
        or not identity["run_func"]
        or not isinstance(framework_version, str)
        or not framework_version
        or (sm_version is not None and (not isinstance(sm_version, int) or isinstance(sm_version, bool)))
    ):
        raise RuntimeError(f"invalid checkpoint producer identity: {identity!r}")

    registry_module = importlib.import_module(f"collector.{backend}.registry")
    if backend == "vllm" and sm_version is None:
        registry = list(registry_module.REGISTRY_XPU)
    else:
        registry = [*registry_module.REGISTRY, *_wideep_registry_for_backend(backend)]
    owned_tables = {
        Path(str(entry.perf_filename)).stem
        for entry in registry
        if identity["module"] == f"{backend}.{entry.op}"
        and identity["run_func"] == entry.run_func
        and resolve_module(entry, framework_version) is not None
    }
    if len(owned_tables) != 1:
        raise RuntimeError(f"checkpoint producer has no unambiguous registered table: {identity!r}")
    return owned_tables.pop()


def _load_selected_producer_checkpoint(
    collection: dict,
    provenance_ctx: dict,
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
    staging_path: Path,
    required: bool,
    context: str,
) -> ResumeCheckpoint | None:
    """Load and bind one selected producer checkpoint without mutating it."""
    resume_tracker = _resume_tracker_for_collection(
        collection,
        provenance_ctx,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        sm_version=sm_version,
    )
    checkpoint_path = resume_tracker._path
    resume_tracker.load_existing()
    if resume_tracker._source_digest is None:
        if not required:
            return None
        raise RuntimeError(
            f"{context}: staged table {staging_path} has no checkpoint for selected producer "
            f"{resume_tracker.module_name} at {checkpoint_path}"
        )
    if _registered_checkpoint_table(resume_tracker._metadata, backend=backend) != staging_path.stem:
        raise RuntimeError(
            f"{context}: checkpoint producer {resume_tracker.module_name} does not own staged table {staging_path}"
        )
    return resume_tracker


def _producer_checkpoint_plan(resume_tracker: ResumeCheckpoint, table: str) -> _ProducerCheckpointPlan:
    if (
        resume_tracker._source_digest is None
        or resume_tracker._source_device is None
        or resume_tracker._source_inode is None
    ):
        raise RuntimeError(f"Checkpoint producer was not loaded from a regular file: {resume_tracker._path}")
    return _ProducerCheckpointPlan(
        attestation=_FileAttestation(
            path=resume_tracker._path,
            digest=resume_tracker._source_digest,
            device=resume_tracker._source_device,
            inode=resume_tracker._source_inode,
        ),
        table=table,
        identity=tuple((field, resume_tracker._metadata[field]) for field in _CHECKPOINT_IDENTITY_FIELDS),
        done=frozenset(resume_tracker._done),
        attempted=frozenset(resume_tracker._attempted),
        failed=frozenset(resume_tracker._failed),
    )


def _revalidate_producer_plan(producer_plan: dict[Path, _ProducerCheckpointPlan]) -> None:
    """Require every selected producer to remain the exact preflight object."""
    if not producer_plan:
        return
    with _locked_checkpoint_directory(next(iter(producer_plan))) as locked:
        for checkpoint_path, plan in producer_plan.items():
            try:
                if checkpoint_path != plan.path:
                    raise RuntimeError("checkpoint path is not a regular selected producer")
                snapshot = _checkpoint_snapshot_at(
                    checkpoint_path,
                    plan.attestation.digest,
                    locked=locked,
                    context_path=checkpoint_path,
                    expected_identity=plan.attestation.identity,
                    capture_contents=True,
                    kind="selected producer checkpoint",
                )
                checkpoint = json.loads(snapshot.contents)
                identity = {field: checkpoint.get(field) for field in _CHECKPOINT_IDENTITY_FIELDS}
                if (
                    identity != plan.identity_dict()
                    or _registered_checkpoint_table(identity, backend=identity["backend"]) != plan.table
                ):
                    raise RuntimeError("checkpoint producer identity changed after preflight")
                if (
                    _checkpoint_case_ids(checkpoint, "done") != set(plan.done)
                    or _checkpoint_case_ids(checkpoint, "attempted") != set(plan.attempted)
                    or _checkpoint_case_ids(checkpoint, "failed") != set(plan.failed)
                    or checkpoint.get(_SIDECAR_TRANSACTION_FIELD) is not None
                ):
                    raise RuntimeError("checkpoint ledgers changed after preflight")
            except Exception as error:
                raise RuntimeError(
                    f"Selected producer checkpoint changed after preflight: {checkpoint_path}: {error}"
                ) from error


def _pending_resume_perf_outputs(
    output_root: Path,
    provenance_ctx: dict,
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
) -> list[Path]:
    """Return staged tables whose loaded checkpoint still has an open event.

    This is the zero-work half of resume: a prior process may have completed
    its cases (or deliberately used ``--keep-csv``) and stopped before common
    finalization. The staging file is unchanged during the resumed process, so
    mtime-based selection alone cannot see it; the pending checkpoint ledger is
    the evidence that makes selecting this explicit registry-owned file safe.
    """
    producers_by_output: dict[Path, list[dict]] = {}
    for collection in provenance_ctx.get("collections") or []:
        perf_path = Path(str(collection["perf_filename"]))
        if not perf_path.is_absolute():
            perf_path = output_root / perf_path
        if perf_path.name.endswith("_perf.txt"):
            producers_by_output.setdefault(perf_path, []).append(collection)

    pending_outputs: set[Path] = set()
    for perf_path, producers in producers_by_output.items():
        staging_present = perf_path.exists() or perf_path.is_symlink()
        has_pending_attempts = False
        for collection in producers:
            resume_tracker = _load_selected_producer_checkpoint(
                collection,
                provenance_ctx,
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                sm_version=sm_version,
                staging_path=perf_path,
                required=staging_present,
                context="resume finalization",
            )
            has_pending_attempts = has_pending_attempts or bool(resume_tracker and resume_tracker._attempted)
        if has_pending_attempts:
            if perf_path.is_symlink() or not perf_path.is_file():
                raise RuntimeError(
                    f"resume finalization: open checkpoint event has no regular staging table {perf_path}"
                )
            pending_outputs.add(perf_path)
    return sorted(pending_outputs)


def _same_file_snapshot(first: _FileSnapshot | None, second: _FileSnapshot | None) -> bool:
    return first is second is None or (
        first is not None and second is not None and first.identity == second.identity and first.digest == second.digest
    )


def _validated_regular_file(
    path: Path,
    expected_digest: str | None,
    *,
    context_path: Path,
    expected_identity: tuple[int, int] | None = None,
    expected_mode: int | None = None,
    capture_contents: bool = False,
    kind: str,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileSnapshot:
    """Read one regular file once and bind its path, descriptor, and bytes."""
    try:
        if locked_output_root is None:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Invalid collector {kind} in {context_path}: {path}")
            file_descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
        else:
            file_descriptor = locked_output_root.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            )
        with os.fdopen(file_descriptor, "rb") as owned_file:
            opened = os.fstat(owned_file.fileno())
            if not stat.S_ISREG(opened.st_mode):
                raise RuntimeError(f"{kind} is not a regular file: {path}")
            digest = hashlib.sha256()
            captured_chunks: list[bytes] | None = [] if capture_contents else None
            for chunk in iter(lambda: owned_file.read(1024 * 1024), b""):
                digest.update(chunk)
                if captured_chunks is not None:
                    captured_chunks.append(chunk)
        current = locked_output_root.stat(path) if locked_output_root is not None else path.lstat()
    except Exception as error:
        raise RuntimeError(f"Invalid collector {kind} in {context_path}: {path}") from error

    opened_mode = stat.S_IMODE(opened.st_mode)
    current_mode = stat.S_IMODE(current.st_mode)
    snapshot = _FileSnapshot(
        device=opened.st_dev,
        inode=opened.st_ino,
        mode=opened_mode,
        digest="sha256:" + digest.hexdigest(),
        contents=b"".join(captured_chunks) if captured_chunks is not None else None,
    )
    if (
        not stat.S_ISREG(current.st_mode)
        or snapshot.identity != (current.st_dev, current.st_ino)
        or opened_mode != current_mode
        or (expected_identity is not None and snapshot.identity != expected_identity)
        or (expected_digest is not None and snapshot.digest != expected_digest)
        or (expected_mode is not None and snapshot.mode != expected_mode)
    ):
        raise RuntimeError(f"Collector {kind} changed in {context_path}: {path}")
    return snapshot


def _validated_sidecar_document(
    path: Path,
    expected_digest: str,
    *,
    context_path: Path,
    expected_identity: tuple[int, int] | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileSnapshot:
    return _validated_regular_file(
        path,
        expected_digest,
        context_path=context_path,
        expected_identity=expected_identity,
        capture_contents=True,
        kind="sidecar transaction document",
        locked_output_root=locked_output_root,
    )


def _revalidate_journal_attestation(
    attestation: _FileAttestation,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    _validated_regular_file(
        attestation.path,
        attestation.digest,
        context_path=attestation.path,
        expected_identity=attestation.identity,
        kind="sidecar transaction journal",
        locked_output_root=locked_output_root,
    )


def _preflight_collector_finalization_inputs(
    output_root: Path,
    staging_paths: list[Path],
    provenance_ctx: dict,
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
    locked_output_root: LockedOutputRoot | None = None,
) -> tuple[dict[Path, _ProducerCheckpointPlan], dict[Path, _FileAttestation]]:
    """Reject unsafe collector inputs before parquet finalization mutates data."""
    output_root = locked_output_root.path if locked_output_root is not None else output_root.resolve()
    validated_paths = _validated_collector_staging_paths(
        [str(path) for path in staging_paths],
        output_root=output_root,
        context_path=output_root / "collection_meta.yaml",
        require_present=True,
        locked_output_root=locked_output_root,
    )
    staging_attestations = {
        path: _validated_regular_file(
            path,
            None,
            context_path=output_root,
            kind="staging file",
            locked_output_root=locked_output_root,
        ).attest(path)
        for path in validated_paths
    }
    producers_by_path: dict[Path, list[dict]] = {}
    for collection in provenance_ctx.get("collections") or []:
        staging_path = Path(str(collection["perf_filename"]))
        if not staging_path.is_absolute():
            staging_path = output_root / staging_path
        producers_by_path.setdefault(staging_path, []).append(collection)

    producer_plan: dict[Path, _ProducerCheckpointPlan] = {}
    seen_attempted_case_ids: set[str] = set()
    for staging_path in validated_paths:
        producers = producers_by_path.get(staging_path)
        if not producers:
            raise RuntimeError(f"collector finalization: staged table {staging_path} has no selected producer")
        attempted_case_ids: set[str] = set()
        for collection in producers:
            resume_tracker = _load_selected_producer_checkpoint(
                collection,
                provenance_ctx,
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                sm_version=sm_version,
                staging_path=staging_path,
                required=True,
                context="collector finalization",
            )
            if resume_tracker is not None:
                checkpoint_path = resume_tracker._path
                checkpoint_plan = _producer_checkpoint_plan(resume_tracker, staging_path.stem)
                prior_plan = producer_plan.get(checkpoint_path)
                if prior_plan is None:
                    duplicate_case_ids = set(checkpoint_plan.attempted) & seen_attempted_case_ids
                    if duplicate_case_ids:
                        raise RuntimeError(
                            "collector finalization: attempted case IDs must be unique across selected "
                            f"checkpoint producers: {sorted(duplicate_case_ids)}"
                        )
                    producer_plan[checkpoint_path] = checkpoint_plan
                    seen_attempted_case_ids.update(checkpoint_plan.attempted)
                elif prior_plan != checkpoint_plan:
                    raise RuntimeError(
                        f"collector finalization: checkpoint changed during preflight: {checkpoint_path}"
                    )
                attempted_case_ids.update(resume_tracker._attempted)
        if not attempted_case_ids:
            raise RuntimeError(
                f"collector finalization: staged table {staging_path} has no attempted checkpoint case IDs"
            )
    return producer_plan, staging_attestations


def _perf_file_record(attestation: PerfFileAttestation, *, include_path: bool = True) -> dict:
    record = {
        "digest": attestation.digest,
        "device": attestation.device,
        "inode": attestation.inode,
        "mode": attestation.mode,
    }
    if include_path:
        record = {"path": str(attestation.path), **record}
    return record


def _perf_file_from_record(
    record: object,
    *,
    context_path: Path,
    expected_path: Path | None = None,
) -> PerfFileAttestation:
    expected_fields = {"path", "digest", "device", "inode", "mode"}
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise RuntimeError(f"Invalid perf publication file in {context_path}: {record!r}")
    path_text = record["path"]
    if not isinstance(path_text, str) or path_text != str(Path(path_text)):
        raise RuntimeError(f"Invalid perf publication path in {context_path}: {path_text!r}")
    path = Path(path_text)
    if expected_path is not None and path != expected_path:
        raise RuntimeError(f"Unexpected perf publication path in {context_path}: {path}")
    digest = record["digest"]
    device = record["device"]
    inode = record["inode"]
    mode = record["mode"]
    if (
        not _valid_sha256_digest(digest)
        or not isinstance(device, int)
        or isinstance(device, bool)
        or not isinstance(inode, int)
        or isinstance(inode, bool)
        or not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode < 0
        or mode > 0o7777
    ):
        raise RuntimeError(f"Invalid perf publication attestation in {context_path}: {record!r}")
    return PerfFileAttestation(path=path, digest=digest, device=device, inode=inode, mode=mode)


def _perf_publication_record(publication: PerfPublication) -> dict:
    return {
        "source": _perf_file_record(publication.source),
        "target": str(publication.target),
        "prepared": _perf_file_record(publication.prepared),
        "previous_target": (
            _perf_file_record(publication.previous_target) if publication.previous_target is not None else None
        ),
        "target_claim": str(publication.target_claim) if publication.target_claim is not None else None,
        "new_rows": publication.info.new_rows,
        "merged_existing": publication.info.merged_existing,
    }


def _perf_publication_from_record(
    record: object,
    *,
    output_root: Path,
    context_path: Path,
) -> PerfPublication:
    from collector.registry_types import PerfFile

    expected_fields = {
        "source",
        "target",
        "prepared",
        "previous_target",
        "target_claim",
        "new_rows",
        "merged_existing",
    }
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise RuntimeError(f"Invalid perf publication in {context_path}: {record!r}")
    target_text = record["target"]
    if not isinstance(target_text, str) or target_text != str(Path(target_text)):
        raise RuntimeError(f"Invalid perf publication target in {context_path}: {target_text!r}")
    target = Path(target_text)
    allowed_sources = {output_root / str(perf_file) for perf_file in PerfFile}
    source = _perf_file_from_record(record["source"], context_path=context_path)
    if source.path not in allowed_sources or target != source.path.with_suffix(".parquet"):
        raise RuntimeError(f"Unowned perf publication path in {context_path}: {source.path} -> {target}")
    prepared = _perf_file_from_record(record["prepared"], context_path=context_path)
    if (
        prepared.path.parent != output_root
        or not prepared.path.name.startswith(f".{target.name}.")
        or not prepared.path.name.endswith(".tmp")
    ):
        raise RuntimeError(f"Invalid prepared perf publication path in {context_path}: {prepared.path}")
    previous_record = record["previous_target"]
    claim_text = record["target_claim"]
    if (previous_record is None) != (claim_text is None):
        raise RuntimeError(f"Incomplete perf target claim state in {context_path}: {target}")
    previous = (
        None
        if previous_record is None
        else _perf_file_from_record(previous_record, context_path=context_path, expected_path=target)
    )
    claim = None if claim_text is None else Path(claim_text)
    if claim is not None and (
        claim_text != str(claim)
        or claim.parent != output_root
        or not claim.name.startswith(f".{target.name}.")
        or not claim.name.endswith(".claim")
        or previous is None
    ):
        raise RuntimeError(f"Invalid perf target claim in {context_path}: {target}")
    new_rows = record["new_rows"]
    merged_existing = record["merged_existing"]
    if (
        not isinstance(new_rows, int)
        or isinstance(new_rows, bool)
        or new_rows < 0
        or not isinstance(merged_existing, bool)
    ):
        raise RuntimeError(f"Invalid perf finalization facts in {context_path}: {target}")
    info = PerfFinalizationInfo(
        new_rows=new_rows,
        merged_existing=merged_existing,
        source_digest=source.digest,
        source_device=source.device,
        source_inode=source.inode,
    )
    return PerfPublication(
        source=source,
        target=target,
        prepared=prepared,
        previous_target=previous,
        target_claim=claim,
        info=info,
    )


def _perf_checkpoint_record(record: _ProducerCheckpointPlan) -> dict:
    return {
        "path": str(record.path),
        "digest": record.attestation.digest,
        "device": record.attestation.device,
        "inode": record.attestation.inode,
        "table": record.table,
        "done": sorted(record.done),
        "failed": sorted(record.failed),
        "attempted": sorted(record.attempted),
        "identity": record.identity_dict(),
    }


def _validate_perf_transaction_document(
    transaction: object,
    *,
    output_root: Path,
    backend: str,
    checkpoint_root: Path,
    journal_path: Path,
) -> tuple[list[PerfPublication], list[dict], _FileAttestation | None]:
    if not isinstance(transaction, dict) or set(transaction) != _PERF_TRANSACTION_FIELDS:
        raise RuntimeError(f"Invalid collector perf transaction fields in {journal_path}")
    if transaction["schema"] != _PERF_TRANSACTION_SCHEMA:
        raise RuntimeError(f"Unsupported collector perf transaction schema in {journal_path}")
    transaction_id = transaction["transaction_id"]
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError(f"Invalid collector perf transaction ID in {journal_path}")
    expected_context = {
        "output_root": str(output_root),
        "backend": backend,
        "checkpoint_root": str(checkpoint_root),
        "sidecar_staging_path": str(output_root / _SIDECAR_STAGING_FILENAME),
    }
    if {field: transaction[field] for field in expected_context} != expected_context:
        raise RuntimeError(f"Collector perf transaction context changed in {journal_path}")
    publication_records = transaction["publications"]
    if not isinstance(publication_records, list) or not publication_records:
        raise RuntimeError(f"Invalid collector perf publications in {journal_path}")
    publications = [
        _perf_publication_from_record(record, output_root=output_root, context_path=journal_path)
        for record in publication_records
    ]
    if len({publication.target for publication in publications}) != len(publications):
        raise RuntimeError(f"Duplicate collector perf publication targets in {journal_path}")

    checkpoint_records = transaction["checkpoints"]
    if not isinstance(checkpoint_records, list) or not checkpoint_records:
        raise RuntimeError(f"Invalid collector perf checkpoint participants in {journal_path}")
    tables = {publication.target.stem for publication in publications}
    seen_checkpoint_paths: set[Path] = set()
    seen_attempted: set[str] = set()
    for checkpoint in checkpoint_records:
        expected_fields = {
            "path",
            "digest",
            "device",
            "inode",
            "table",
            "done",
            "failed",
            "attempted",
            "identity",
        }
        if not isinstance(checkpoint, dict) or set(checkpoint) != expected_fields:
            raise RuntimeError(f"Invalid collector perf checkpoint record in {journal_path}")
        attestation = _attestation_from_record(
            {field: checkpoint[field] for field in ("path", "digest", "device", "inode")},
            None,
            context_path=journal_path,
            kind="perf checkpoint participant",
        )
        identity = checkpoint["identity"]
        if not isinstance(identity, dict) or set(identity) != set(_CHECKPOINT_IDENTITY_FIELDS):
            raise RuntimeError(f"Invalid collector perf checkpoint identity in {journal_path}")
        expected_path = _checkpoint_path(checkpoint_root, identity["module"])
        table = _registered_checkpoint_table(identity, backend=backend)
        if attestation.path != expected_path or checkpoint["table"] != table or table not in tables:
            raise RuntimeError(f"Unowned collector perf checkpoint in {journal_path}: {attestation.path}")
        if attestation.path in seen_checkpoint_paths:
            raise RuntimeError(f"Duplicate collector perf checkpoint in {journal_path}: {attestation.path}")
        for field in ("done", "failed", "attempted"):
            values = checkpoint[field]
            if (
                not isinstance(values, list)
                or any(not isinstance(case_id, str) or not case_id for case_id in values)
                or len(values) != len(set(values))
            ):
                raise RuntimeError(f"Invalid collector perf checkpoint {field} in {journal_path}")
        attempted = set(checkpoint["attempted"])
        if not attempted or attempted & seen_attempted:
            raise RuntimeError(f"Invalid collector perf checkpoint attempts in {journal_path}")
        seen_checkpoint_paths.add(attestation.path)
        seen_attempted.update(attempted)
    if {checkpoint["table"] for checkpoint in checkpoint_records} != tables:
        raise RuntimeError(f"Collector perf transaction tables lack checkpoint owners in {journal_path}")

    previous_record = transaction["previous_sidecar"]
    previous_sidecar = (
        None
        if previous_record is None
        else _attestation_from_record(
            previous_record,
            output_root / "collection_meta.yaml",
            context_path=journal_path,
            kind="perf transaction sidecar snapshot",
        )
    )
    return publications, checkpoint_records, previous_sidecar


def _revalidate_perf_transaction_rollback_inputs(
    publications: list[PerfPublication],
    checkpoint_records: list[dict],
    previous_sidecar: _FileAttestation | None,
    *,
    output_root: Path,
    journal_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    for publication in publications:
        _validated_regular_file(
            publication.source.path,
            publication.source.digest,
            context_path=journal_path,
            expected_identity=publication.source.identity,
            expected_mode=publication.source.mode,
            kind="perf transaction staging file",
            locked_output_root=locked_output_root,
        )
    if checkpoint_records:
        with _locked_checkpoint_directory(Path(checkpoint_records[0]["path"])) as locked:
            checkpoint_states = []
            for checkpoint in checkpoint_records:
                checkpoint_path = Path(checkpoint["path"])
                canonical_path = _checkpoint_path_in_locked_directory(checkpoint_path, locked)
                effective_state = _effective_checkpoint_state_at(
                    locked.file_descriptor,
                    canonical_path,
                    context_path=journal_path,
                    kind="perf transaction checkpoint",
                )
                snapshot = effective_state.snapshot
                if snapshot.digest != checkpoint["digest"] or snapshot.identity != (
                    checkpoint["device"],
                    checkpoint["inode"],
                ):
                    raise RuntimeError(f"Collector perf transaction checkpoint changed in {journal_path}")
                document = json.loads(snapshot.contents)
                if {field: document.get(field) for field in _CHECKPOINT_IDENTITY_FIELDS} != checkpoint["identity"]:
                    raise RuntimeError(f"Collector perf transaction checkpoint identity changed in {journal_path}")
                if any(
                    set(document.get(field, [])) != set(checkpoint[field]) for field in ("done", "failed", "attempted")
                ):
                    raise RuntimeError(f"Collector perf transaction checkpoint ledgers changed in {journal_path}")
                if document.get(_SIDECAR_TRANSACTION_FIELD) is not None:
                    raise RuntimeError(f"Collector perf transaction checkpoint was already tagged in {journal_path}")
                checkpoint_states.append((canonical_path, effective_state))
            for canonical_path, effective_state in checkpoint_states:
                _require_effective_checkpoint_state_at(
                    locked.file_descriptor,
                    canonical_path,
                    effective_state,
                )
                _normalize_atomic_replace_state_at(locked.file_descriptor, canonical_path)
                _validated_regular_file_at(
                    locked.file_descriptor,
                    canonical_path.name,
                    canonical_path,
                    effective_state.snapshot.digest,
                    context_path=journal_path,
                    expected_identity=effective_state.snapshot.identity,
                    kind="perf transaction checkpoint",
                )
    meta_path = output_root / "collection_meta.yaml"
    if previous_sidecar is None:
        if not (locked_output_root.absent(meta_path) if locked_output_root is not None else not meta_path.exists()):
            raise RuntimeError(f"Collector perf transaction sidecar target changed in {journal_path}")
    else:
        _validated_regular_file(
            meta_path,
            previous_sidecar.digest,
            context_path=journal_path,
            expected_identity=previous_sidecar.identity,
            kind="perf transaction sidecar target",
            locked_output_root=locked_output_root,
        )


def _revalidate_published_perf_publications(
    publications: Iterable[PerfPublication],
    *,
    context_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    for publication in publications:
        snapshot = _validated_regular_file(
            publication.target,
            publication.prepared.digest,
            context_path=context_path,
            expected_identity=publication.prepared.identity,
            kind="published perf target",
            locked_output_root=locked_output_root,
        )
        if snapshot.mode != publication.prepared.mode:
            raise RuntimeError(f"Collector published perf target mode changed in {context_path}: {publication.target}")


class _CollectorPerfPublicationTransaction:
    def __init__(
        self,
        output_root: Path,
        *,
        backend: str,
        checkpoint_dir: str,
        producer_plan: dict[Path, _ProducerCheckpointPlan],
        sidecar_snapshot: _FileSnapshot | None,
        locked_output_root: LockedOutputRoot,
    ) -> None:
        self.output_root = locked_output_root.path
        self.locked_output_root = locked_output_root
        self.backend = backend
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_root = _checkpoint_backend_root(checkpoint_dir, backend)
        self.producer_plan = producer_plan
        self.sidecar_snapshot = sidecar_snapshot
        self.journal_path = self.output_root / _PERF_TRANSACTION_FILENAME
        self.publications: tuple[PerfPublication, ...] = ()
        self.journal_attestation: _FileAttestation | None = None
        self.sidecar_journal_attestation: _FileAttestation | None = None
        self._journal_published = False

    def has_durable_journal(self) -> bool:
        return self._journal_published

    def prepare(self, publications: tuple[PerfPublication, ...]) -> None:
        if not self.locked_output_root.absent(self.journal_path):
            raise RuntimeError(f"Unexpected collector perf transaction journal: {self.journal_path}")
        if not self.locked_output_root.absent(
            self.output_root / _SIDECAR_TRANSACTION_FILENAME
        ) or not self.locked_output_root.absent(self.output_root / _SIDECAR_STAGING_FILENAME):
            raise RuntimeError(f"Unexpected collector sidecar transaction before perf publication: {self.output_root}")
        self.publications = publications
        _revalidate_producer_plan(self.producer_plan)
        transaction = {
            "schema": _PERF_TRANSACTION_SCHEMA,
            "transaction_id": uuid.uuid4().hex,
            "output_root": str(self.output_root),
            "backend": self.backend,
            "checkpoint_root": str(self.checkpoint_root),
            "previous_sidecar": _attestation_record(
                self.sidecar_snapshot.attest(self.output_root / "collection_meta.yaml")
                if self.sidecar_snapshot is not None
                else None
            ),
            "sidecar_staging_path": str(self.output_root / _SIDECAR_STAGING_FILENAME),
            "checkpoints": [_perf_checkpoint_record(record) for _path, record in sorted(self.producer_plan.items())],
            "publications": [_perf_publication_record(publication) for publication in publications],
        }
        parsed_publications, checkpoint_records, previous_sidecar = _validate_perf_transaction_document(
            transaction,
            output_root=self.output_root,
            backend=self.backend,
            checkpoint_root=self.checkpoint_root,
            journal_path=self.journal_path,
        )
        _revalidate_perf_transaction_rollback_inputs(
            parsed_publications,
            checkpoint_records,
            previous_sidecar,
            output_root=self.output_root,
            journal_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        _atomic_write_bytes(
            self.journal_path,
            json.dumps(transaction, indent=2).encode(),
            _locked_output_root=self.locked_output_root,
        )
        self._journal_published = True
        snapshot = _validated_regular_file(
            self.journal_path,
            None,
            context_path=self.journal_path,
            kind="perf transaction journal",
            locked_output_root=self.locked_output_root,
        )
        self.journal_attestation = snapshot.attest(self.journal_path)

    def _validated_state(
        self,
    ) -> tuple[_FileAttestation, list[PerfPublication], list[dict], _FileAttestation | None]:
        if self.journal_attestation is None:
            raise RuntimeError("Collector perf transaction has no durable journal")
        state = _load_perf_publication_transaction(
            self.output_root,
            backend=self.backend,
            checkpoint_dir=self.checkpoint_dir,
            locked_output_root=self.locked_output_root,
        )
        if state is None or state[0] != self.journal_attestation or tuple(state[1]) != self.publications:
            raise RuntimeError("Collector perf transaction journal changed after publication preparation")
        return state

    def rollback_complete(self, publications: tuple[PerfPublication, ...]) -> None:
        if publications != self.publications or self.journal_attestation is None:
            raise RuntimeError("Collector perf transaction rollback does not match its prepared batch")
        state = self._validated_state()
        _revalidate_perf_transaction_rollback_inputs(
            state[1],
            state[2],
            state[3],
            output_root=self.output_root,
            journal_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        locked_roots = {self.output_root: self.locked_output_root}
        validate_restored_perf_publications(state[1], locked_roots)
        cleanup_perf_publication_artifacts(publications, locked_roots)
        _revalidate_journal_attestation(state[0], self.locked_output_root)
        _revalidate_perf_transaction_rollback_inputs(
            state[1],
            state[2],
            state[3],
            output_root=self.output_root,
            journal_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        validate_restored_perf_publications(state[1], locked_roots)
        _cleanup_transaction_files(
            [self.journal_attestation],
            context_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        self.journal_attestation = None
        self._journal_published = False

    def rollback(self) -> None:
        state = self._validated_state()
        _rollback_perf_publication_transaction(
            state,
            output_root=self.output_root,
            locked_output_root=self.locked_output_root,
        )
        self.journal_attestation = None
        self._journal_published = False

    def handoff_to_sidecar(self, sidecar_journal_attestation: _FileAttestation) -> None:
        self._validated_state()
        _revalidate_journal_attestation(sidecar_journal_attestation, self.locked_output_root)
        _revalidate_published_perf_publications(
            self.publications,
            context_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        self.sidecar_journal_attestation = sidecar_journal_attestation

    def revalidate_published(self, sidecar_journal_attestation: _FileAttestation) -> None:
        if sidecar_journal_attestation != self.sidecar_journal_attestation:
            raise RuntimeError("Collector perf transaction sidecar handoff changed before commit")
        self._validated_state()
        _revalidate_journal_attestation(sidecar_journal_attestation, self.locked_output_root)
        _revalidate_published_perf_publications(
            self.publications,
            context_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )

    def commit_complete(self, sidecar_journal_attestation: _FileAttestation) -> None:
        self.revalidate_published(sidecar_journal_attestation)
        cleanup_perf_publication_artifacts(self.publications, {self.output_root: self.locked_output_root})
        self.revalidate_published(sidecar_journal_attestation)
        _cleanup_transaction_files(
            [self.journal_attestation],
            context_path=self.journal_path,
            locked_output_root=self.locked_output_root,
        )
        self.journal_attestation = None
        self.sidecar_journal_attestation = None
        self._journal_published = False


def _sidecar_perf_publications(
    transaction: dict,
    *,
    output_root: Path,
    journal_path: Path,
) -> list[PerfPublication] | None:
    records = transaction.get("perf_publications")
    if records is None:
        return None
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"Invalid collector sidecar perf publications in {journal_path}")
    publications = [
        _perf_publication_from_record(record, output_root=output_root, context_path=journal_path) for record in records
    ]
    if len({publication.target for publication in publications}) != len(publications):
        raise RuntimeError(f"Duplicate collector sidecar perf publication targets in {journal_path}")
    return publications


def _validate_transaction_envelope(
    transaction: dict,
    *,
    output_root: Path,
    backend: str,
    checkpoint_root: Path,
    journal_path: Path,
) -> tuple[str, str]:
    """Validate immutable journal context before inspecting any mutation target."""
    if not isinstance(transaction, dict) or set(transaction) != _SIDECAR_TRANSACTION_FIELDS:
        raise RuntimeError(f"Invalid collector sidecar transaction fields in {journal_path}")
    if transaction["schema"] != _SIDECAR_TRANSACTION_SCHEMA:
        raise RuntimeError(
            f"Unsupported collector sidecar transaction schema in {journal_path}: {transaction['schema']!r}"
        )

    transaction_id = transaction["transaction_id"]
    expected_digest = transaction["sidecar_digest"]
    if (
        not isinstance(transaction_id, str)
        or len(transaction_id) != 32
        or any(character not in "0123456789abcdef" for character in transaction_id)
    ):
        raise RuntimeError(f"Invalid collector sidecar transaction ID in {journal_path}: {transaction_id!r}")
    if not _valid_sha256_digest(expected_digest):
        raise RuntimeError(f"Invalid collector sidecar digest in {journal_path}: {expected_digest!r}")
    _sidecar_perf_publications(
        transaction,
        output_root=output_root,
        journal_path=journal_path,
    )

    expected_context = {
        "output_root": str(output_root),
        "backend": backend,
        "checkpoint_root": str(checkpoint_root),
        "sidecar_path": str(output_root / "collection_meta.yaml"),
    }
    recorded_context = {field: transaction[field] for field in expected_context}
    if recorded_context != expected_context:
        raise RuntimeError(
            f"Collector sidecar transaction {journal_path} context {recorded_context!r} != {expected_context!r}"
        )
    return transaction_id, expected_digest


def _validated_collector_staging_paths(
    recorded_paths: object,
    *,
    output_root: Path,
    context_path: Path,
    require_present: bool = False,
    locked_output_root: LockedOutputRoot | None = None,
) -> list[Path]:
    """Resolve collector staging targets to direct registry-owned files."""
    from collector.registry_types import PerfFile

    if not isinstance(recorded_paths, list) or not recorded_paths:
        raise RuntimeError(f"Invalid collector staging paths for {context_path}")

    root = locked_output_root.path if locked_output_root is not None else output_root.resolve()
    allowed_paths = {root / str(perf_file) for perf_file in PerfFile}
    staging_paths: list[Path] = []
    seen: set[Path] = set()
    for path_text in recorded_paths:
        if not isinstance(path_text, str):
            raise TypeError(f"Invalid collector staging path for {context_path}: {path_text!r}")
        staging_path = Path(path_text)
        if path_text != str(staging_path) or staging_path not in allowed_paths or staging_path in seen:
            raise RuntimeError(f"Invalid collector staging path for {context_path}: {staging_path}")
        if locked_output_root is not None:
            try:
                staging_state = locked_output_root.stat(staging_path)
            except FileNotFoundError:
                staging_state = None
        else:
            staging_state = staging_path.lstat() if staging_path.exists() or staging_path.is_symlink() else None
        if (staging_state is not None and not stat.S_ISREG(staging_state.st_mode)) or (
            require_present and staging_state is None
        ):
            raise RuntimeError(f"Invalid collector staging path for {context_path}: {staging_path}")
        if staging_state is not None:
            lock_path = Path(f"{staging_path}.lock")
            lock_present = (
                not locked_output_root.absent(lock_path)
                if locked_output_root is not None
                else lock_path.exists() or lock_path.is_symlink()
            )
            if lock_present:
                raise RuntimeError(f"Collector staging path has an active writer lock for {context_path}: {lock_path}")
            try:
                if staging_state.st_mode & 0o444 == 0:
                    raise PermissionError("file has no read permission bits")
                file_descriptor = (
                    locked_output_root.open(
                        staging_path,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                    )
                    if locked_output_root is not None
                    else os.open(
                        staging_path,
                        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                    )
                )
                with os.fdopen(file_descriptor, "rb") as staging_file:
                    if not staging_file.read(1):
                        raise ValueError("file is empty")
            except Exception as error:
                raise RuntimeError(
                    f"Collector staging path is not readable for {context_path}: {staging_path}: {error}"
                ) from error
        seen.add(staging_path)
        staging_paths.append(staging_path)
    return staging_paths


def _valid_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _attestation_record(
    attestation: _FileAttestation | None,
    *,
    include_path: bool = False,
) -> dict | None:
    if attestation is None:
        return None
    record = {
        "digest": attestation.digest,
        "device": attestation.device,
        "inode": attestation.inode,
    }
    if include_path:
        record = {"path": str(attestation.path), **record}
    return record


def _attestation_from_record(
    record: object,
    path: Path | None,
    *,
    context_path: Path,
    kind: str,
) -> _FileAttestation:
    expected_fields = {"digest", "device", "inode"} | ({"path"} if path is None else set())
    if not isinstance(record, dict) or set(record) != expected_fields:
        raise RuntimeError(f"Invalid {kind} in {context_path}: {record!r}")
    if path is None:
        path_text = record["path"]
        if not isinstance(path_text, str) or path_text != str(Path(path_text)):
            raise RuntimeError(f"Invalid {kind} in {context_path}: {record!r}")
        path = Path(path_text)
    digest = record["digest"]
    device = record["device"]
    inode = record["inode"]
    if (
        not _valid_sha256_digest(digest)
        or not isinstance(device, int)
        or isinstance(device, bool)
        or not isinstance(inode, int)
        or isinstance(inode, bool)
    ):
        raise RuntimeError(f"Invalid {kind} in {context_path}: {record!r}")
    return _FileAttestation(path=path, digest=digest, device=device, inode=inode)


def _validated_sidecar_target(
    record: object,
    meta_path: Path,
    *,
    context_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileAttestation | None:
    """Bind publication to the exact pre-transaction sidecar target, or its absence."""
    if record is None:
        if not (locked_output_root.absent(meta_path) if locked_output_root is not None else not meta_path.exists()):
            raise RuntimeError(f"Collector sidecar target changed in {context_path}: {meta_path}")
        return None
    attestation = _attestation_from_record(
        record,
        meta_path,
        context_path=context_path,
        kind="previous sidecar target",
    )
    snapshot = _validated_regular_file(
        meta_path,
        attestation.digest,
        context_path=context_path,
        expected_identity=attestation.identity,
        kind="sidecar target",
        locked_output_root=locked_output_root,
    )
    return snapshot.attest(meta_path)


def _revalidate_transaction_staging_files(
    staging_files: list[_FileAttestation],
    *,
    context_path: Path,
    require_present: bool,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Verify all present staging bytes before any transaction mutation."""
    for staging_file in staging_files:
        staging_path = staging_file.path
        if locked_output_root is not None:
            try:
                staging_state = locked_output_root.stat(staging_path)
            except FileNotFoundError:
                staging_state = None
        else:
            staging_state = staging_path.lstat() if staging_path.exists() or staging_path.is_symlink() else None
        if staging_state is not None and not stat.S_ISREG(staging_state.st_mode):
            raise RuntimeError(f"Invalid collector staging path for {context_path}: {staging_path}")
        if staging_state is None:
            if require_present:
                raise RuntimeError(f"Missing collector staging path for {context_path}: {staging_path}")
            continue
        lock_path = Path(f"{staging_path}.lock")
        lock_present = (
            not locked_output_root.absent(lock_path)
            if locked_output_root is not None
            else lock_path.exists() or lock_path.is_symlink()
        )
        if lock_present:
            raise RuntimeError(f"Collector staging path has an active writer lock for {context_path}: {lock_path}")
        _validated_regular_file(
            staging_path,
            staging_file.digest,
            context_path=context_path,
            expected_identity=staging_file.identity,
            kind="staging path",
            locked_output_root=locked_output_root,
        )


def _validated_transaction_staging_files(
    recorded_files: object,
    *,
    output_root: Path,
    context_path: Path,
    require_present: bool,
    locked_output_root: LockedOutputRoot | None = None,
) -> list[_FileAttestation]:
    """Validate canonical paths and immutable bytes from one transaction."""
    if not isinstance(recorded_files, list) or not recorded_files:
        raise RuntimeError(f"Invalid collector staging files for {context_path}")
    staging_files = [
        _attestation_from_record(
            record,
            None,
            context_path=context_path,
            kind="collector staging file",
        )
        for record in recorded_files
    ]
    if len({staging_file.path for staging_file in staging_files}) != len(staging_files):
        raise RuntimeError(f"Duplicate collector staging files for {context_path}")

    staging_paths = _validated_collector_staging_paths(
        [str(staging_file.path) for staging_file in staging_files],
        output_root=output_root,
        context_path=context_path,
        require_present=require_present,
        locked_output_root=locked_output_root,
    )
    record_by_path = {staging_file.path: staging_file for staging_file in staging_files}
    staging_files = [record_by_path[path] for path in staging_paths]
    _revalidate_transaction_staging_files(
        staging_files,
        context_path=context_path,
        require_present=require_present,
        locked_output_root=locked_output_root,
    )
    return staging_files


def _validated_transaction_checkpoints(
    transaction: dict,
    *,
    backend: str,
    checkpoint_root: Path,
    journal_path: Path,
) -> list[_ValidatedCheckpoint]:
    """Load every journal checkpoint without trusting its path or identity."""
    participants = transaction.get("checkpoints")
    if not isinstance(participants, list) or not participants:
        raise RuntimeError(f"Invalid checkpoint participants in collector sidecar transaction {journal_path}")

    parsed_participants = []
    seen_checkpoint_paths: set[Path] = set()
    seen_attempted_case_ids: set[str] = set()
    for participant in participants:
        try:
            if not isinstance(participant, dict) or set(participant) != {
                "path",
                "done",
                "failed",
                "attempted",
                "identity",
            }:
                raise TypeError("participant has invalid fields")
            path_text = participant["path"]
            identity = participant["identity"]
            if not isinstance(path_text, str):
                raise TypeError("checkpoint path must be a string")
            recorded_ledgers = {}
            for field in ("done", "failed", "attempted"):
                case_ids = participant[field]
                if (
                    not isinstance(case_ids, list)
                    or any(not isinstance(case_id, str) or not case_id for case_id in case_ids)
                    or len(set(case_ids)) != len(case_ids)
                ):
                    raise TypeError(f"{field} case IDs must be a list of unique strings")
                recorded_ledgers[field] = set(case_ids)
            attempted_case_ids = recorded_ledgers["attempted"]
            if attempted_case_ids & seen_attempted_case_ids:
                raise ValueError("attempted case IDs must be unique across checkpoint participants")
            if not isinstance(identity, dict) or set(identity) != set(_CHECKPOINT_IDENTITY_FIELDS):
                raise TypeError(f"checkpoint identity must contain exactly {_CHECKPOINT_IDENTITY_FIELDS!r}")
            table = _registered_checkpoint_table(identity, backend=backend)
            checkpoint_path = Path(path_text)
            expected_path = _checkpoint_path(checkpoint_root, identity["module"])
            if path_text != str(expected_path) or checkpoint_path in seen_checkpoint_paths:
                raise ValueError(f"checkpoint path is not a direct owned file: {checkpoint_path}")
        except Exception as error:
            raise RuntimeError(f"Invalid checkpoint participant in {journal_path}: {error}") from error
        seen_checkpoint_paths.add(checkpoint_path)
        seen_attempted_case_ids.update(attempted_case_ids)
        parsed_participants.append((checkpoint_path, identity, recorded_ledgers, attempted_case_ids, table))

    validated_participants: list[_ValidatedCheckpoint] = []
    with _locked_checkpoint_root(checkpoint_root) as locked:
        preflight_participants = []
        for checkpoint_path, identity, recorded_ledgers, attempted_case_ids, table in parsed_participants:
            try:
                canonical_path = _checkpoint_path_in_locked_directory(checkpoint_path, locked)
                effective_state = _effective_checkpoint_state_at(
                    locked.file_descriptor,
                    canonical_path,
                    context_path=journal_path,
                    kind="checkpoint participant",
                )
                snapshot = effective_state.snapshot
                checkpoint = json.loads(snapshot.contents)
                if not isinstance(checkpoint, dict):
                    raise TypeError("checkpoint document must be an object")
                actual_identity = {field: checkpoint.get(field) for field in _CHECKPOINT_IDENTITY_FIELDS}
                if actual_identity != identity:
                    raise ValueError(f"checkpoint identity {actual_identity!r} != recorded {identity!r}")
                live_ledgers = {field: _checkpoint_case_ids(checkpoint, field) for field in recorded_ledgers}
                if any(live_ledgers[field] != recorded_ledgers[field] for field in ("done", "failed")):
                    raise ValueError("checkpoint done/failed ledgers do not match the transaction")
                checkpoint_transaction = checkpoint.get(_SIDECAR_TRANSACTION_FIELD)
                if checkpoint_transaction not in (None, transaction["transaction_id"]):
                    raise ValueError(f"checkpoint is bound to another transaction: {checkpoint_transaction!r}")
            except Exception as error:
                raise RuntimeError(f"Invalid checkpoint participant in {journal_path}: {error}") from error
            preflight_participants.append(
                (
                    checkpoint_path,
                    table,
                    attempted_case_ids,
                    checkpoint,
                    effective_state,
                )
            )

        for checkpoint_path, table, attempted_case_ids, checkpoint, effective_state in preflight_participants:
            canonical_path = _checkpoint_path_in_locked_directory(checkpoint_path, locked)
            _require_effective_checkpoint_state_at(
                locked.file_descriptor,
                canonical_path,
                effective_state,
            )
            _normalize_atomic_replace_state_at(locked.file_descriptor, canonical_path)
            snapshot = _validated_regular_file_at(
                locked.file_descriptor,
                canonical_path.name,
                canonical_path,
                effective_state.snapshot.digest,
                context_path=journal_path,
                expected_identity=effective_state.snapshot.identity,
                kind="checkpoint participant",
            )
            validated_participants.append(
                _ValidatedCheckpoint(
                    attestation=snapshot.attest(checkpoint_path),
                    table=table,
                    attempted=frozenset(attempted_case_ids),
                    document=checkpoint,
                )
            )
    return validated_participants


def _validate_transaction_table_ownership(
    sidecar: object,
    staging_paths: list[Path],
    checkpoints: list[_ValidatedCheckpoint],
    *,
    output_root: Path,
    journal_path: Path,
) -> None:
    """Bind every journal mutation target to one attested sidecar table event."""
    from collector import provenance
    from collector.registry_types import PerfFile

    try:
        provenance.validate_collection_meta_for_update(sidecar)
    except Exception as error:
        raise RuntimeError(
            f"Invalid sidecar document in collector sidecar transaction {journal_path}: {error}"
        ) from error
    if not isinstance(sidecar, dict) or not isinstance(sidecar.get("tables"), dict):
        raise TypeError(f"Invalid sidecar document in collector sidecar transaction {journal_path}")

    allowed_staging_by_table = {Path(str(perf_file)).stem: output_root / str(perf_file) for perf_file in PerfFile}
    checkpoints_by_table: dict[str, list[_ValidatedCheckpoint]] = {}
    for checkpoint in checkpoints:
        checkpoints_by_table.setdefault(checkpoint.table, []).append(checkpoint)
    if any(table not in allowed_staging_by_table for table in checkpoints_by_table):
        raise RuntimeError(f"Invalid table in collector sidecar transaction {journal_path}")
    expected_staging_paths = {allowed_staging_by_table[table] for table in checkpoints_by_table}
    if set(staging_paths) != expected_staging_paths:
        raise RuntimeError(f"Collector sidecar transaction {journal_path} staging paths do not match its owned tables")

    sidecar_tables = sidecar["tables"]
    for table, table_checkpoint_records in checkpoints_by_table.items():
        table_entry = sidecar_tables.get(table)
        if not isinstance(table_entry, dict):
            raise TypeError(f"Owned table {table!r} is absent from sidecar transaction {journal_path}")
        history = table_entry.get("collections")
        if history is None:
            current_event = table_entry
        elif isinstance(history, list) and history and isinstance(history[-1], dict):
            current_event = history[-1]
        else:
            raise RuntimeError(f"Invalid sidecar history for owned table {table!r} in {journal_path}")
        expected_case_plan_hash = current_event.get("case_plan_hash")
        attempted_case_ids = set().union(*(checkpoint.attempted for checkpoint in table_checkpoint_records))
        actual_case_plan_hash = provenance.case_plan_hash(sorted(attempted_case_ids))
        if not attempted_case_ids or expected_case_plan_hash != actual_case_plan_hash:
            raise RuntimeError(
                f"Owned table {table!r} case plan does not match checkpoint participants in {journal_path}"
            )


def _validated_checkpoint_participants_for_phase(
    checkpoints: list[_ValidatedCheckpoint],
    transaction_id: str,
    journal_path: Path,
    *,
    phase: str,
) -> list[_ValidatedCheckpoint]:
    """Prevalidate every live ledger before the transaction mutates anything."""
    participants: list[_ValidatedCheckpoint] = []
    for checkpoint in checkpoints:
        checkpoint_path = checkpoint.path
        attempted_case_ids = set(checkpoint.attempted)
        live_attempts = set(checkpoint.document["attempted"])
        checkpoint_transaction = checkpoint.document.get(_SIDECAR_TRANSACTION_FIELD)
        tagged = checkpoint_transaction == transaction_id and bool(attempted_case_ids)
        if phase == "committed" and checkpoint_transaction is None and not live_attempts:
            continue
        tag_matches = (
            tagged if phase == "committed" else checkpoint_transaction is None or (phase == "preparing" and tagged)
        )
        if live_attempts == attempted_case_ids and tag_matches:
            if attempted_case_ids:
                participants.append(checkpoint)
            continue
        raise RuntimeError(
            f"Invalid {phase} checkpoint participant in {journal_path}: live attempts {sorted(live_attempts)} "
            f"or transaction tag is inconsistent for {checkpoint_path}"
        )
    return participants


def _revalidate_checkpoint_attestations(attestations: Iterable[_FileAttestation]) -> None:
    attestations = list(attestations)
    if not attestations:
        return
    with _locked_checkpoint_directory(attestations[0].path) as locked:
        for attestation in attestations:
            _checkpoint_snapshot_at(
                attestation.path,
                attestation.digest,
                locked=locked,
                context_path=attestation.path,
                expected_identity=attestation.identity,
                kind="checkpoint",
            )


def _restore_claimed_transaction_files(
    claimed_files: list[_ClaimedTransactionFile],
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    for claimed_file in reversed(claimed_files):
        try:
            _validated_regular_file(
                claimed_file.claimed_path,
                claimed_file.original.digest,
                context_path=claimed_file.claimed_path,
                expected_identity=claimed_file.original.identity,
                kind="sidecar transaction claim",
                locked_output_root=locked_output_root,
            )
        except RuntimeError:
            continue
        if not (
            locked_output_root.absent(claimed_file.original.path)
            if locked_output_root is not None
            else not claimed_file.original.path.exists()
        ):
            try:
                _validated_regular_file(
                    claimed_file.original.path,
                    claimed_file.original.digest,
                    context_path=claimed_file.claimed_path,
                    expected_identity=claimed_file.original.identity,
                    kind="restored sidecar transaction file",
                    locked_output_root=locked_output_root,
                )
                _validated_regular_file(
                    claimed_file.claimed_path,
                    claimed_file.original.digest,
                    context_path=claimed_file.claimed_path,
                    expected_identity=claimed_file.original.identity,
                    kind="sidecar transaction claim",
                    locked_output_root=locked_output_root,
                )
            except RuntimeError:
                pass
            continue
        try:
            if locked_output_root is not None:
                locked_output_root.rename_noreplace(claimed_file.claimed_path, claimed_file.original.path)
            else:
                _rename_noreplace(claimed_file.claimed_path, claimed_file.original.path)
        except OSError:
            continue
        try:
            _validated_regular_file(
                claimed_file.original.path,
                claimed_file.original.digest,
                context_path=claimed_file.original.path,
                expected_identity=claimed_file.original.identity,
                kind="restored sidecar transaction file",
                locked_output_root=locked_output_root,
            )
        except RuntimeError:
            continue


def _transaction_claim_path(path: Path, transaction_id: str) -> Path:
    return path.with_name(f".{path.name}.{transaction_id}.transaction-claim")


def _recover_transaction_claim(
    attestation: _FileAttestation,
    transaction_id: str,
    *,
    context_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> _FileAttestation | None:
    """Inspect an exact durable claim without mutating transaction state."""
    claim_path = _transaction_claim_path(attestation.path, transaction_id)
    if locked_output_root.absent(claim_path) if locked_output_root is not None else not claim_path.exists():
        return None
    claim_snapshot = _validated_regular_file(
        claim_path,
        attestation.digest,
        context_path=context_path,
        expected_identity=attestation.identity,
        kind="sidecar transaction claim",
        locked_output_root=locked_output_root,
    )
    return claim_snapshot.attest(claim_path)


def _restore_transaction_claim(
    attestation: _FileAttestation,
    claim_attestation: _FileAttestation,
    *,
    context_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Restore one prevalidated durable claim without overwriting a competitor."""
    _validated_regular_file(
        claim_attestation.path,
        claim_attestation.digest,
        context_path=context_path,
        expected_identity=claim_attestation.identity,
        kind="sidecar transaction claim",
        locked_output_root=locked_output_root,
    )
    if not (
        locked_output_root.absent(attestation.path) if locked_output_root is not None else not attestation.path.exists()
    ):
        _validated_regular_file(
            attestation.path,
            attestation.digest,
            context_path=context_path,
            expected_identity=attestation.identity,
            kind="restored sidecar transaction file",
            locked_output_root=locked_output_root,
        )
        raise RuntimeError(f"Collector transaction claim already has a restored target in {context_path}")
    else:
        try:
            if locked_output_root is not None:
                locked_output_root.rename_noreplace(claim_attestation.path, attestation.path)
            else:
                _rename_noreplace(claim_attestation.path, attestation.path)
        except FileExistsError as error:
            raise RuntimeError(f"Collector transaction target changed in {context_path}: {attestation.path}") from error
        _validated_regular_file(
            attestation.path,
            attestation.digest,
            context_path=context_path,
            expected_identity=attestation.identity,
            kind="restored sidecar transaction file",
            locked_output_root=locked_output_root,
        )


def _claim_transaction_files(
    staging_files: list[_FileAttestation],
    *,
    context_path: Path,
    require_present: bool,
    transaction_id: str,
    locked_output_root: LockedOutputRoot | None = None,
) -> list[_ClaimedTransactionFile]:
    """Exclusively claim exact transaction-owned objects under durable names."""
    claim_plan: list[tuple[_FileAttestation, Path, bool, bool]] = []
    for staging_file in staging_files:
        claimed_path = _transaction_claim_path(staging_file.path, transaction_id)
        source_present = not (
            locked_output_root.absent(staging_file.path)
            if locked_output_root is not None
            else not staging_file.path.exists()
        )
        claim_present = not (
            locked_output_root.absent(claimed_path) if locked_output_root is not None else not claimed_path.exists()
        )
        if source_present:
            _validated_regular_file(
                staging_file.path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="staging file",
                locked_output_root=locked_output_root,
            )
        if claim_present:
            _validated_regular_file(
                claimed_path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="sidecar transaction claim",
                locked_output_root=locked_output_root,
            )
        if source_present and claim_present:
            raise RuntimeError(
                f"Legacy collector transaction source/claim hardlink requires manual recovery: {staging_file.path}"
            )
        if require_present and not source_present and not claim_present:
            raise RuntimeError(f"Missing collector staging file in {context_path}: {staging_file.path}")
        claim_plan.append((staging_file, claimed_path, source_present, claim_present))

    claimed_files: list[_ClaimedTransactionFile] = []
    try:
        for staging_file, claimed_path, source_present, claim_present in claim_plan:
            if not source_present and not claim_present:
                continue
            if not claim_present:
                _validated_regular_file(
                    staging_file.path,
                    staging_file.digest,
                    context_path=context_path,
                    expected_identity=staging_file.identity,
                    kind="staging file",
                    locked_output_root=locked_output_root,
                )
                try:
                    if locked_output_root is not None:
                        locked_output_root.rename_noreplace(staging_file.path, claimed_path)
                    else:
                        _rename_noreplace(staging_file.path, claimed_path)
                except FileExistsError as error:
                    raise RuntimeError(
                        f"Collector transaction claim changed in {context_path}: {claimed_path}"
                    ) from error
            _validated_regular_file(
                claimed_path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="claimed staging file",
                locked_output_root=locked_output_root,
            )
            claimed_file = _ClaimedTransactionFile(
                original=staging_file,
                claimed_path=claimed_path,
            )
            claimed_files.append(claimed_file)
            _validated_regular_file(
                claimed_path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="claimed staging file",
                locked_output_root=locked_output_root,
            )
        for claimed_file in claimed_files:
            _validated_regular_file(
                claimed_file.claimed_path,
                claimed_file.original.digest,
                context_path=context_path,
                expected_identity=claimed_file.original.identity,
                kind="claimed staging file",
                locked_output_root=locked_output_root,
            )
    except Exception:
        _restore_claimed_transaction_files(claimed_files, locked_output_root)
        raise
    return claimed_files


def _retain_claimed_transaction_files(
    claimed_files: list[_ClaimedTransactionFile],
    *,
    context_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Retain only the private exact objects already claimed by a transaction."""
    for claimed_file in claimed_files:
        _validated_regular_file(
            claimed_file.claimed_path,
            claimed_file.original.digest,
            context_path=context_path,
            expected_identity=claimed_file.original.identity,
            kind="claimed transaction file",
            locked_output_root=locked_output_root,
        )
    for claimed_file in claimed_files:
        _validated_regular_file(
            claimed_file.claimed_path,
            claimed_file.original.digest,
            context_path=context_path,
            expected_identity=claimed_file.original.identity,
            kind="claimed transaction file",
            locked_output_root=locked_output_root,
        )
        retained_path = collector_retained_path(claimed_file.original.path)
        if not (
            locked_output_root.absent(retained_path) if locked_output_root is not None else not retained_path.exists()
        ):
            raise RuntimeError(f"Conflicting retained collector transaction file in {context_path}: {retained_path}")
        if locked_output_root is not None:
            locked_output_root.rename_noreplace(claimed_file.claimed_path, retained_path)
        else:
            _rename_noreplace(claimed_file.claimed_path, retained_path)
        _validated_regular_file(
            retained_path,
            claimed_file.original.digest,
            context_path=context_path,
            expected_identity=claimed_file.original.identity,
            kind="retained transaction file",
            locked_output_root=locked_output_root,
        )


def _cleanup_transaction_files(
    staging_files: list[_FileAttestation],
    *,
    context_path: Path,
    transaction_id: str | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Retain only exact transaction-owned files, using durable claims when requested."""
    if transaction_id is None:
        _revalidate_transaction_staging_files(
            staging_files,
            context_path=context_path,
            require_present=False,
            locked_output_root=locked_output_root,
        )
        for staging_file in staging_files:
            staging_absent = (
                locked_output_root.absent(staging_file.path)
                if locked_output_root is not None
                else not staging_file.path.exists()
            )
            if staging_absent:
                if locked_output_root is not None:
                    locked_output_root.fsync()
                else:
                    _fsync_directory(staging_file.path.parent)
                continue
            _validated_regular_file(
                staging_file.path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="transaction cleanup file",
                locked_output_root=locked_output_root,
            )
            retained_path = collector_retained_path(staging_file.path)
            if not (
                locked_output_root.absent(retained_path)
                if locked_output_root is not None
                else not retained_path.exists()
            ):
                raise RuntimeError(
                    f"Conflicting retained collector transaction file in {context_path}: {retained_path}"
                )
            if locked_output_root is not None:
                locked_output_root.rename_noreplace(staging_file.path, retained_path)
            else:
                _rename_noreplace(staging_file.path, retained_path)
            _validated_regular_file(
                retained_path,
                staging_file.digest,
                context_path=context_path,
                expected_identity=staging_file.identity,
                kind="retained transaction file",
                locked_output_root=locked_output_root,
            )
        return

    claimed_files = _claim_transaction_files(
        staging_files,
        context_path=context_path,
        require_present=False,
        transaction_id=transaction_id,
        locked_output_root=locked_output_root,
    )
    _retain_claimed_transaction_files(
        claimed_files,
        context_path=context_path,
        locked_output_root=locked_output_root,
    )


def _publish_transaction_sidecar(
    source_path: Path,
    source_snapshot: _FileSnapshot,
    meta_path: Path,
    target_attestation: _FileAttestation | None,
    transaction_id: str,
    *,
    context_path: Path,
    journal_attestation: _FileAttestation | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    """Publish captured bytes after claiming the exact source and prior target."""
    if journal_attestation is not None:
        _revalidate_journal_attestation(journal_attestation, locked_output_root)
    if source_snapshot.contents is None:
        raise RuntimeError(f"Collector sidecar bytes were not captured for {context_path}")
    source_record = source_snapshot.attest(source_path)
    claim_records = [source_record]
    if source_path == meta_path:
        if target_attestation != source_record:
            raise RuntimeError(f"Collector sidecar target changed in {context_path}: {meta_path}")
    elif target_attestation is None:
        if not (locked_output_root.absent(meta_path) if locked_output_root is not None else not meta_path.exists()):
            raise RuntimeError(f"Collector sidecar target changed in {context_path}: {meta_path}")
    else:
        claim_records.append(target_attestation)
    claimed_files = _claim_transaction_files(
        claim_records,
        context_path=context_path,
        require_present=True,
        transaction_id=transaction_id,
        locked_output_root=locked_output_root,
    )
    try:
        if not (locked_output_root.absent(meta_path) if locked_output_root is not None else not meta_path.exists()):
            raise RuntimeError(f"Collector sidecar target changed in {context_path}: {meta_path}")
        if journal_attestation is not None:
            _revalidate_journal_attestation(journal_attestation, locked_output_root)
        _atomic_write_bytes(
            meta_path,
            source_snapshot.contents,
            mode=source_snapshot.mode,
            _locked_output_root=locked_output_root,
        )
    except Exception:
        _restore_claimed_transaction_files(claimed_files, locked_output_root)
        raise
    _retain_claimed_transaction_files(
        claimed_files,
        context_path=context_path,
        locked_output_root=locked_output_root,
    )


def _load_perf_publication_transaction(
    output_root: Path,
    *,
    backend: str,
    checkpoint_dir: str,
    locked_output_root: LockedOutputRoot | None = None,
) -> tuple[_FileAttestation, list[PerfPublication], list[dict], _FileAttestation | None] | None:
    output_root = locked_output_root.path if locked_output_root is not None else output_root.resolve()
    journal_path = output_root / _PERF_TRANSACTION_FILENAME
    if locked_output_root is not None:
        try:
            journal_state = locked_output_root.stat(journal_path)
        except FileNotFoundError:
            journal_state = None
    else:
        journal_state = journal_path.lstat() if journal_path.exists() or journal_path.is_symlink() else None
    if journal_state is None:
        return None
    if not stat.S_ISREG(journal_state.st_mode):
        raise RuntimeError(f"Invalid collector perf transaction journal: {journal_path}")
    try:
        snapshot = _validated_regular_file(
            journal_path,
            None,
            context_path=journal_path,
            capture_contents=True,
            kind="perf transaction journal",
            locked_output_root=locked_output_root,
        )
        transaction = json.loads(snapshot.contents)
        publications, checkpoint_records, previous_sidecar = _validate_perf_transaction_document(
            transaction,
            output_root=output_root,
            backend=backend,
            checkpoint_root=_checkpoint_backend_root(checkpoint_dir, backend),
            journal_path=journal_path,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load collector perf transaction {journal_path}: {error}") from error
    return snapshot.attest(journal_path), publications, checkpoint_records, previous_sidecar


def _rollback_perf_publication_transaction(
    state: tuple[_FileAttestation, list[PerfPublication], list[dict], _FileAttestation | None],
    *,
    output_root: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    journal_attestation, publications, checkpoint_records, previous_sidecar = state
    journal_path = journal_attestation.path
    staged_meta_path = output_root / _SIDECAR_STAGING_FILENAME
    staged_meta_attestation: _FileAttestation | None = None
    if locked_output_root is not None:
        try:
            staged_state = locked_output_root.stat(staged_meta_path)
        except FileNotFoundError:
            staged_state = None
    else:
        staged_state = staged_meta_path.lstat() if staged_meta_path.exists() or staged_meta_path.is_symlink() else None
    if staged_state is not None and not stat.S_ISREG(staged_state.st_mode):
        raise RuntimeError(f"Invalid sidecar staging document during perf rollback: {journal_path}")
    if staged_state is not None:
        staged_meta_attestation = _validated_regular_file(
            staged_meta_path,
            None,
            context_path=journal_path,
            kind="uncommitted sidecar staging document",
            locked_output_root=locked_output_root,
        ).attest(staged_meta_path)
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _revalidate_perf_transaction_rollback_inputs(
        publications,
        checkpoint_records,
        previous_sidecar,
        output_root=output_root,
        journal_path=journal_path,
        locked_output_root=locked_output_root,
    )
    locked_roots = {output_root: locked_output_root} if locked_output_root is not None else None
    restore_perf_publications(publications, locked_roots)
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _revalidate_perf_transaction_rollback_inputs(
        publications,
        checkpoint_records,
        previous_sidecar,
        output_root=output_root,
        journal_path=journal_path,
        locked_output_root=locked_output_root,
    )
    cleanup_perf_publication_artifacts(publications, locked_roots)
    if staged_meta_attestation is not None:
        _cleanup_transaction_files(
            [staged_meta_attestation],
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _revalidate_perf_transaction_rollback_inputs(
        publications,
        checkpoint_records,
        previous_sidecar,
        output_root=output_root,
        journal_path=journal_path,
        locked_output_root=locked_output_root,
    )
    validate_restored_perf_publications(publications, locked_roots)
    _cleanup_transaction_files(
        [journal_attestation],
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )


def _validate_perf_to_sidecar_handoff(
    perf_state: tuple[_FileAttestation, list[PerfPublication], list[dict], _FileAttestation | None],
    sidecar_transaction: dict,
    *,
    journal_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    perf_journal, publications, checkpoint_records, previous_sidecar = perf_state
    _revalidate_journal_attestation(perf_journal, locked_output_root)
    if sidecar_transaction.get("perf_publications") != [
        _perf_publication_record(publication) for publication in publications
    ]:
        raise RuntimeError(f"Collector perf and sidecar transactions disagree on publications in {journal_path}")
    expected_previous = _attestation_record(previous_sidecar)
    if sidecar_transaction.get("previous_sidecar") != expected_previous:
        raise RuntimeError(f"Collector perf and sidecar transactions disagree on prior sidecar in {journal_path}")
    expected_staging = [
        {
            "path": str(publication.source.path),
            "digest": publication.source.digest,
            "device": publication.source.device,
            "inode": publication.source.inode,
        }
        for publication in publications
    ]
    if sidecar_transaction.get("staging_paths") != expected_staging:
        raise RuntimeError(f"Collector perf and sidecar transactions disagree on staging files in {journal_path}")
    expected_checkpoints = [
        {
            "path": checkpoint["path"],
            "done": checkpoint["done"],
            "failed": checkpoint["failed"],
            "attempted": checkpoint["attempted"],
            "identity": checkpoint["identity"],
        }
        for checkpoint in checkpoint_records
    ]
    if sidecar_transaction.get("checkpoints") != expected_checkpoints:
        raise RuntimeError(f"Collector perf and sidecar transactions disagree on checkpoints in {journal_path}")
    _revalidate_published_perf_publications(
        publications,
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )


def _revalidate_perf_sidecar_commit(
    perf_transaction: tuple[_FileAttestation, list[PerfPublication], list[dict], _FileAttestation | None] | None,
    sidecar_publications: list[PerfPublication] | None,
    transaction: dict,
    *,
    journal_path: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    if perf_transaction is not None:
        _validate_perf_to_sidecar_handoff(
            perf_transaction,
            transaction,
            journal_path=journal_path,
            locked_output_root=locked_output_root,
        )
    elif sidecar_publications is not None:
        _revalidate_published_perf_publications(
            sidecar_publications,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )


def _recover_collector_provenance_transaction(
    output_root: Path,
    *,
    backend: str,
    checkpoint_dir: str,
) -> Path | None:
    _require_separate_checkpoint_root(output_root, checkpoint_dir, backend)
    with perf_finalization_lifecycle(output_root) as locked_output_root:
        return _recover_collector_provenance_transaction_locked(
            locked_output_root.path,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            locked_output_root=locked_output_root,
        )


def _cleanup_unjournaled_perf_preparations(
    output_root: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    from collector.registry_types import PerfFile

    parquet_paths = [output_root / Path(str(perf_file)).with_suffix(".parquet") for perf_file in PerfFile]
    candidates = [
        parquet_path
        for parquet_path in parquet_paths
        if any(
            not (locked_output_root.absent(path) if locked_output_root is not None else not path.exists())
            for path in (
                perf_preparation_path(parquet_path),
                perf_preparation_cleanup_path(parquet_path),
            )
        )
    ]
    if not candidates:
        return
    locked_roots = {output_root: locked_output_root} if locked_output_root is not None else None
    with perf_merge_locks(candidates, locked_roots):
        cleanup_unjournaled_perf_preparations(
            candidates,
            transaction_paths=perf_transaction_artifact_paths(output_root),
            locked_roots=locked_roots,
        )


def _cleanup_atomic_write_reservations(
    output_root: Path,
    locked_output_root: LockedOutputRoot | None = None,
) -> None:
    cleanup_atomic_write_reservations(
        (
            output_root / _PERF_TRANSACTION_FILENAME,
            output_root / _SIDECAR_STAGING_FILENAME,
            output_root / _SIDECAR_TRANSACTION_FILENAME,
            output_root / "collection_meta.yaml",
        ),
        {output_root: locked_output_root} if locked_output_root is not None else None,
    )


def _recover_collector_provenance_transaction_locked(
    output_root: Path,
    *,
    backend: str,
    checkpoint_dir: str,
    locked_output_root: LockedOutputRoot,
) -> Path | None:
    """Finish an interrupted sidecar commit without appending its event twice."""
    output_root = locked_output_root.path
    _cleanup_atomic_write_reservations(output_root, locked_output_root)
    _cleanup_unjournaled_perf_preparations(output_root, locked_output_root)
    checkpoint_root = _checkpoint_backend_root(checkpoint_dir, backend)
    perf_transaction = _load_perf_publication_transaction(
        output_root,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        locked_output_root=locked_output_root,
    )
    journal_path = output_root / _SIDECAR_TRANSACTION_FILENAME
    try:
        journal_state = locked_output_root.stat(journal_path)
    except FileNotFoundError:
        journal_state = None
    if journal_state is None:
        if perf_transaction is not None:
            _rollback_perf_publication_transaction(
                perf_transaction,
                output_root=output_root,
                locked_output_root=locked_output_root,
            )
        return None
    if not stat.S_ISREG(journal_state.st_mode):
        raise RuntimeError(f"Invalid collector sidecar transaction journal: {journal_path}")

    try:
        journal_snapshot = _validated_regular_file(
            journal_path,
            None,
            context_path=journal_path,
            capture_contents=True,
            kind="sidecar transaction journal",
            locked_output_root=locked_output_root,
        )
        transaction = json.loads(journal_snapshot.contents)
        transaction_id, expected_digest = _validate_transaction_envelope(
            transaction,
            output_root=output_root,
            backend=backend,
            checkpoint_root=checkpoint_root,
            journal_path=journal_path,
        )
        sidecar_publications = _sidecar_perf_publications(
            transaction,
            output_root=output_root,
            journal_path=journal_path,
        )
        _revalidate_perf_sidecar_commit(
            perf_transaction,
            sidecar_publications,
            transaction,
            journal_path=journal_path,
            locked_output_root=locked_output_root,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load collector sidecar transaction {journal_path}: {error}") from error

    meta_path = output_root / "collection_meta.yaml"
    staged_meta_path = output_root / _SIDECAR_STAGING_FILENAME
    pending_attestation = _attestation_from_record(
        transaction.get("pending_sidecar"),
        staged_meta_path,
        context_path=journal_path,
        kind="pending sidecar",
    )
    previous_record = transaction.get("previous_sidecar")
    previous_attestation = (
        None
        if previous_record is None
        else _attestation_from_record(
            previous_record,
            meta_path,
            context_path=journal_path,
            kind="previous sidecar target",
        )
    )
    journal_attestation = journal_snapshot.attest(journal_path)
    for document_path in (meta_path, staged_meta_path):
        try:
            document_state = locked_output_root.stat(document_path)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(document_state.st_mode):
            raise RuntimeError(f"Invalid collector sidecar transaction document in {journal_path}: {document_path}")

    pending_claim = _recover_transaction_claim(
        pending_attestation,
        transaction_id,
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )
    previous_claim = (
        _recover_transaction_claim(
            previous_attestation,
            transaction_id,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
        if previous_attestation is not None
        else None
    )
    if previous_attestation is None:
        unexpected_claim = _transaction_claim_path(meta_path, transaction_id)
        if not locked_output_root.absent(unexpected_claim):
            raise RuntimeError(f"Unexpected collector transaction claim in {journal_path}: {unexpected_claim}")

    committed_document_snapshot: _FileSnapshot | None = None
    if not locked_output_root.absent(meta_path):
        candidate_snapshot = _validated_regular_file(
            meta_path,
            None,
            context_path=journal_path,
            capture_contents=True,
            kind="committed sidecar document",
            locked_output_root=locked_output_root,
        )
        if candidate_snapshot.digest == expected_digest:
            committed_document_snapshot = _validated_sidecar_document(
                meta_path,
                expected_digest,
                context_path=journal_path,
                expected_identity=candidate_snapshot.identity,
                locked_output_root=locked_output_root,
            )

    staged_document_snapshot: _FileSnapshot | None = None
    if not locked_output_root.absent(staged_meta_path):
        staged_document_snapshot = _validated_sidecar_document(
            staged_meta_path,
            expected_digest,
            context_path=journal_path,
            expected_identity=pending_attestation.identity,
            locked_output_root=locked_output_root,
        )
    claimed_document_snapshot = (
        _validated_sidecar_document(
            pending_claim.path,
            expected_digest,
            context_path=journal_path,
            expected_identity=pending_attestation.identity,
            locked_output_root=locked_output_root,
        )
        if pending_claim is not None
        else None
    )
    phase = "committed" if committed_document_snapshot is not None else "preparing"
    sidecar_snapshot = committed_document_snapshot or staged_document_snapshot or claimed_document_snapshot
    if sidecar_snapshot is None:
        raise RuntimeError(
            f"Collector sidecar transaction {journal_path} has neither its committed nor staged document"
        )
    try:
        import yaml

        if sidecar_snapshot.contents is None:
            raise RuntimeError("sidecar bytes were not captured")
        sidecar = yaml.safe_load(sidecar_snapshot.contents)
    except Exception as error:
        raise RuntimeError(f"Invalid sidecar document in collector transaction {journal_path}: {error}") from error

    validated_staging_files = _validated_transaction_staging_files(
        transaction.get("staging_paths"),
        output_root=output_root,
        context_path=journal_path,
        require_present=phase == "preparing",
        locked_output_root=locked_output_root,
    )
    validated_staging_paths = [staging_file.path for staging_file in validated_staging_files]
    staging_claims = [
        claim
        for staging_file in validated_staging_files
        if (
            claim := _recover_transaction_claim(
                staging_file,
                transaction_id,
                context_path=journal_path,
                locked_output_root=locked_output_root,
            )
        )
        is not None
    ]
    if phase == "preparing" and staging_claims:
        raise RuntimeError(f"Unexpected collector staging cleanup claim in {journal_path}")

    validated_checkpoints = _validated_transaction_checkpoints(
        transaction,
        backend=backend,
        checkpoint_root=checkpoint_root,
        journal_path=journal_path,
    )
    _validate_transaction_table_ownership(
        sidecar,
        validated_staging_paths,
        validated_checkpoints,
        output_root=output_root,
        journal_path=journal_path,
    )

    if phase == "preparing":
        if previous_attestation is None:
            if not locked_output_root.absent(meta_path):
                raise RuntimeError(f"Collector sidecar target changed in {journal_path}: {meta_path}")
        elif not locked_output_root.absent(meta_path):
            _validated_regular_file(
                meta_path,
                previous_attestation.digest,
                context_path=journal_path,
                expected_identity=previous_attestation.identity,
                kind="sidecar target",
                locked_output_root=locked_output_root,
            )
        elif previous_claim is None:
            raise RuntimeError(f"Missing collector sidecar target in {journal_path}: {meta_path}")

    tagged_participants = _validated_checkpoint_participants_for_phase(
        validated_checkpoints,
        transaction_id,
        journal_path,
        phase=phase,
    )
    participant_attestations = {checkpoint.path: checkpoint.attestation for checkpoint in tagged_participants}
    _revalidate_checkpoint_attestations(participant_attestations.values())
    _revalidate_journal_attestation(journal_attestation, locked_output_root)

    if phase == "preparing":
        if pending_claim is not None:
            _restore_transaction_claim(
                pending_attestation,
                pending_claim,
                context_path=journal_path,
                locked_output_root=locked_output_root,
            )
        if previous_attestation is not None and previous_claim is not None:
            _restore_transaction_claim(
                previous_attestation,
                previous_claim,
                context_path=journal_path,
                locked_output_root=locked_output_root,
            )
        source_snapshot = _validated_sidecar_document(
            staged_meta_path,
            expected_digest,
            context_path=journal_path,
            expected_identity=pending_attestation.identity,
            locked_output_root=locked_output_root,
        )
        sidecar_target_attestation = _validated_sidecar_target(
            transaction.get("previous_sidecar"),
            meta_path,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
        # The journal is durable before any checkpoint is tagged. Resume a
        # prepare-phase interruption by tagging every participant, then make
        # the already-rendered exact document visible atomically.
        _validated_sidecar_document(
            staged_meta_path,
            expected_digest,
            context_path=journal_path,
            expected_identity=source_snapshot.identity,
            locked_output_root=locked_output_root,
        )
        for checkpoint in tagged_participants:
            participant_attestations[checkpoint.path] = _tag_checkpoint_sidecar_transaction(
                checkpoint.path,
                set(checkpoint.attempted),
                transaction_id,
                expected_attestation=participant_attestations[checkpoint.path],
                journal_attestation=journal_attestation,
                locked_output_root=locked_output_root,
            )
        publish_snapshot = _validated_sidecar_document(
            staged_meta_path,
            expected_digest,
            context_path=journal_path,
            expected_identity=source_snapshot.identity,
            locked_output_root=locked_output_root,
        )
        _revalidate_checkpoint_attestations(participant_attestations.values())
        _revalidate_perf_sidecar_commit(
            perf_transaction,
            sidecar_publications,
            transaction,
            journal_path=journal_path,
            locked_output_root=locked_output_root,
        )
        _publish_transaction_sidecar(
            staged_meta_path,
            publish_snapshot,
            meta_path,
            sidecar_target_attestation,
            transaction_id,
            context_path=journal_path,
            journal_attestation=journal_attestation,
            locked_output_root=locked_output_root,
        )
    else:
        _validated_sidecar_document(
            meta_path,
            expected_digest,
            context_path=journal_path,
            expected_identity=committed_document_snapshot.identity,
            locked_output_root=locked_output_root,
        )

    # A matching tag identifies work left between publish and checkpoint close;
    # an untagged participant is accepted only when its live ledger is empty.
    recovered_open_event = bool(tagged_participants)
    for checkpoint in tagged_participants:
        _close_checkpoint_attempts(
            checkpoint.path,
            set(checkpoint.attempted),
            transaction_id=transaction_id,
            expected_attestation=participant_attestations[checkpoint.path],
            journal_attestation=journal_attestation,
            locked_output_root=locked_output_root,
        )

    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _revalidate_perf_sidecar_commit(
        perf_transaction,
        sidecar_publications,
        transaction,
        journal_path=journal_path,
        locked_output_root=locked_output_root,
    )
    _cleanup_transaction_files(
        validated_staging_files,
        context_path=journal_path,
        transaction_id=transaction_id,
        locked_output_root=locked_output_root,
    )
    if phase == "committed":
        recovered_claims = [
            _ClaimedTransactionFile(original=original, claimed_path=claim.path)
            for original, claim in (
                (pending_attestation, pending_claim),
                (previous_attestation, previous_claim),
            )
            if original is not None and claim is not None
        ]
        _retain_claimed_transaction_files(
            recovered_claims,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
    if perf_transaction is not None:
        perf_journal, publications, _checkpoint_records, _previous_sidecar = perf_transaction
        _revalidate_perf_sidecar_commit(
            perf_transaction,
            sidecar_publications,
            transaction,
            journal_path=journal_path,
            locked_output_root=locked_output_root,
        )
        cleanup_perf_publication_artifacts(publications, {output_root: locked_output_root})
        _revalidate_perf_sidecar_commit(
            perf_transaction,
            sidecar_publications,
            transaction,
            journal_path=journal_path,
            locked_output_root=locked_output_root,
        )
        _revalidate_journal_attestation(journal_attestation, locked_output_root)
        _cleanup_transaction_files(
            [perf_journal],
            context_path=perf_journal.path,
            locked_output_root=locked_output_root,
        )
    if sidecar_publications is not None:
        _revalidate_published_perf_publications(
            sidecar_publications,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
        if perf_transaction is None:
            # A prior process may have retained the perf journal without
            # persisting that rename. Make its canonical absence durable before
            # the sidecar journal, now the sole recovery record, is retained.
            locked_output_root.fsync()
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _cleanup_transaction_files(
        [journal_attestation],
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )
    if recovered_open_event:
        logger.info(f"Recovered collector provenance sidecar transaction: {meta_path}")
        return meta_path
    return None


def _commit_collector_provenance_transaction(
    output_root: Path,
    runtime_meta: dict,
    tables: dict[str, dict],
    *,
    provenance_tier: str | None,
    checkpoint_records: dict[Path, _ProducerCheckpointPlan],
    staging_files: dict[Path, PerfFinalizationInfo],
    existing_sidecar_snapshot: _FileSnapshot | None,
    backend: str,
    checkpoint_dir: str,
    perf_publication_transaction: _CollectorPerfPublicationTransaction | None = None,
    locked_output_root: LockedOutputRoot,
) -> Path:
    """Atomically publish a sidecar and close its checkpoint participants."""
    from collector import provenance

    _revalidate_producer_plan(checkpoint_records)
    output_root = locked_output_root.path
    checkpoint_root = _checkpoint_backend_root(checkpoint_dir, backend)
    staged_meta_path = output_root / _SIDECAR_STAGING_FILENAME
    transaction_id = uuid.uuid4().hex
    meta_path = output_root / "collection_meta.yaml"
    journal_path = output_root / _SIDECAR_TRANSACTION_FILENAME
    rendered_contents = provenance.render_collection_meta(
        runtime_meta,
        tables,
        provenance_tier=provenance_tier,
    )
    rendered_digest = "sha256:" + hashlib.sha256(rendered_contents).hexdigest()
    transaction = {
        "schema": _SIDECAR_TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "output_root": str(output_root),
        "backend": backend,
        "checkpoint_root": str(checkpoint_root),
        "sidecar_path": str(meta_path),
        "sidecar_digest": rendered_digest,
        "pending_sidecar": None,
        "previous_sidecar": _attestation_record(
            existing_sidecar_snapshot.attest(meta_path) if existing_sidecar_snapshot is not None else None
        ),
        "checkpoints": [
            {
                "path": str(checkpoint_path),
                "done": sorted(record.done),
                "failed": sorted(record.failed),
                "attempted": sorted(record.attempted),
                "identity": record.identity_dict(),
            }
            for checkpoint_path, record in sorted(checkpoint_records.items())
        ],
        "staging_paths": [
            _attestation_record(
                _FileAttestation(
                    path=path,
                    digest=info.source_digest,
                    device=info.source_device,
                    inode=info.source_inode,
                ),
                include_path=True,
            )
            for path, info in sorted(staging_files.items())
        ],
        "perf_publications": (
            [_perf_publication_record(publication) for publication in perf_publication_transaction.publications]
            if perf_publication_transaction is not None
            else None
        ),
    }
    _validate_transaction_envelope(
        transaction,
        output_root=output_root,
        backend=backend,
        checkpoint_root=checkpoint_root,
        journal_path=journal_path,
    )
    sidecar_publications = _sidecar_perf_publications(
        transaction,
        output_root=output_root,
        journal_path=journal_path,
    )
    validated_staging_files = _validated_transaction_staging_files(
        transaction.get("staging_paths"),
        output_root=output_root,
        context_path=journal_path,
        require_present=True,
        locked_output_root=locked_output_root,
    )
    validated_staging_paths = [staging_file.path for staging_file in validated_staging_files]
    validated_checkpoints = _validated_transaction_checkpoints(
        transaction,
        backend=backend,
        checkpoint_root=checkpoint_root,
        journal_path=journal_path,
    )
    import yaml

    rendered_sidecar = yaml.safe_load(rendered_contents)
    _validate_transaction_table_ownership(
        rendered_sidecar,
        validated_staging_paths,
        validated_checkpoints,
        output_root=output_root,
        journal_path=journal_path,
    )
    participants = _validated_checkpoint_participants_for_phase(
        validated_checkpoints,
        transaction_id,
        journal_path,
        phase="new",
    )
    _existing_sidecar, live_sidecar_snapshot = _load_clean_collector_sidecar(
        output_root,
        locked_output_root=locked_output_root,
    )
    if not _same_file_snapshot(live_sidecar_snapshot, existing_sidecar_snapshot):
        raise RuntimeError(f"Collector sidecar document changed before transaction commit: {meta_path}")
    sidecar_target_attestation = _validated_sidecar_target(
        transaction["previous_sidecar"],
        meta_path,
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )
    _atomic_write_bytes(
        staged_meta_path,
        rendered_contents,
        mode=0o644,
        _locked_output_root=locked_output_root,
    )
    staged_document_snapshot = _validated_sidecar_document(
        staged_meta_path,
        transaction["sidecar_digest"],
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )
    transaction["pending_sidecar"] = _attestation_record(staged_document_snapshot.attest(staged_meta_path))

    _atomic_write_bytes(
        journal_path,
        json.dumps(transaction, indent=2).encode(),
        _locked_output_root=locked_output_root,
    )
    journal_snapshot = _validated_regular_file(
        journal_path,
        None,
        context_path=journal_path,
        kind="sidecar transaction journal",
        locked_output_root=locked_output_root,
    )
    journal_attestation = journal_snapshot.attest(journal_path)
    if perf_publication_transaction is not None:
        perf_publication_transaction.handoff_to_sidecar(journal_attestation)
    _validated_sidecar_document(
        staged_meta_path,
        transaction["sidecar_digest"],
        context_path=journal_path,
        expected_identity=staged_document_snapshot.identity,
        locked_output_root=locked_output_root,
    )
    participant_attestations = {checkpoint.path: checkpoint.attestation for checkpoint in participants}
    _revalidate_checkpoint_attestations(participant_attestations.values())
    for checkpoint in participants:
        participant_attestations[checkpoint.path] = _tag_checkpoint_sidecar_transaction(
            checkpoint.path,
            set(checkpoint.attempted),
            transaction_id,
            expected_attestation=participant_attestations[checkpoint.path],
            journal_attestation=journal_attestation,
            locked_output_root=locked_output_root,
        )
    publish_snapshot = _validated_sidecar_document(
        staged_meta_path,
        transaction["sidecar_digest"],
        context_path=journal_path,
        expected_identity=staged_document_snapshot.identity,
        locked_output_root=locked_output_root,
    )
    _revalidate_checkpoint_attestations(participant_attestations.values())
    if perf_publication_transaction is not None:
        perf_publication_transaction.revalidate_published(journal_attestation)

    _publish_transaction_sidecar(
        staged_meta_path,
        publish_snapshot,
        meta_path,
        sidecar_target_attestation,
        transaction_id,
        context_path=journal_path,
        journal_attestation=journal_attestation,
        locked_output_root=locked_output_root,
    )
    logger.info(f"Wrote collector provenance sidecar: {meta_path}")
    for checkpoint in participants:
        _close_checkpoint_attempts(
            checkpoint.path,
            set(checkpoint.attempted),
            transaction_id=transaction_id,
            expected_attestation=participant_attestations[checkpoint.path],
            journal_attestation=journal_attestation,
            locked_output_root=locked_output_root,
        )
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    if perf_publication_transaction is not None:
        perf_publication_transaction.revalidate_published(journal_attestation)
    _cleanup_transaction_files(
        validated_staging_files,
        context_path=journal_path,
        transaction_id=transaction_id,
        locked_output_root=locked_output_root,
    )
    if perf_publication_transaction is not None:
        perf_publication_transaction.commit_complete(journal_attestation)
    if sidecar_publications is not None:
        _revalidate_published_perf_publications(
            sidecar_publications,
            context_path=journal_path,
            locked_output_root=locked_output_root,
        )
    _revalidate_journal_attestation(journal_attestation, locked_output_root)
    _cleanup_transaction_files(
        [journal_attestation],
        context_path=journal_path,
        locked_output_root=locked_output_root,
    )
    return meta_path


def _write_collector_provenance(
    output_root: Path,
    converted: list[Path],
    provenance_ctx: dict,
    run_errors: list[dict],
    *,
    backend: str,
    checkpoint_dir: str,
    finalization_info: dict[Path, PerfFinalizationInfo],
    producer_plan: dict[Path, _ProducerCheckpointPlan] | None = None,
    perf_publication_transaction: _CollectorPerfPublicationTransaction | None = None,
    locked_output_root: LockedOutputRoot | None = None,
) -> Path | None:
    """Write collection_meta.yaml (design §5) flat beside the just-finalized parquet.

    Reuses the checkpoint files ResumeCheckpoint already persisted per op.
    ``done``/``failed`` remain cumulative resume state; ``attempted`` is the
    pending event's case-id set, which can span interrupted invocations.
    """
    if locked_output_root is None:
        _require_separate_checkpoint_root(output_root, checkpoint_dir, backend)
        with perf_finalization_lifecycle(output_root) as acquired_root:
            return _write_collector_provenance(
                acquired_root.path,
                converted,
                provenance_ctx,
                run_errors,
                backend=backend,
                checkpoint_dir=checkpoint_dir,
                finalization_info=finalization_info,
                producer_plan=producer_plan,
                perf_publication_transaction=perf_publication_transaction,
                locked_output_root=acquired_root,
            )
    output_root = locked_output_root.path
    if perf_publication_transaction is None:
        recovered_meta_path = _recover_collector_provenance_transaction_locked(
            output_root,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            locked_output_root=locked_output_root,
        )
        if recovered_meta_path is not None:
            return recovered_meta_path
    if producer_plan is not None:
        _revalidate_producer_plan(producer_plan)

    import pyarrow.parquet as pq

    from collector import provenance

    collections = provenance_ctx.get("collections") or []
    ops_by_table: dict[str, list[str]] = {}
    collection_by_full_name: dict[str, dict] = {}
    module_by_table: dict[str, str] = {}
    staging_by_table: dict[str, Path] = {}
    for collection in collections:
        table = Path(str(collection["perf_filename"])).stem
        full_name = f"{collection['name']}.{collection['type']}"
        ops_by_table.setdefault(table, []).append(full_name)
        collection_by_full_name[full_name] = collection
        module_by_table.setdefault(table, collection["module"])
        staging_path = Path(str(collection["perf_filename"]))
        if not staging_path.is_absolute():
            staging_path = output_root / staging_path
        staging_by_table.setdefault(table, staging_path)

    module_failure_names = {e["module"] for e in run_errors if e.get("error_type") == "ModuleCollectionFailure"}
    checkpoint_root = _checkpoint_backend_root(checkpoint_dir, backend)
    closures = provenance.load_closures(_REPO_ROOT / "collector" / "hash_closures.yaml")
    collector_ref = _git_collector_ref(_REPO_ROOT)

    runtime_meta = _runtime_metadata(provenance_ctx)
    collected_at = datetime.now().strftime("%Y-%m-%d")

    tables: dict[str, dict] = {}
    new_rows_by_table: dict[str, int] = {}
    merged_existing_by_table: dict[str, bool] = {}
    finalization_by_table: dict[str, PerfFinalizationInfo] = {}
    checkpoint_records: dict[Path, _ProducerCheckpointPlan] = {}
    for parquet_path in converted:
        table = parquet_path.stem
        full_names = ops_by_table.get(table)
        module = module_by_table.get(table)
        if not full_names or module is None:
            logger.warning(f"collection_meta: {table} has no registry mapping this run; skipping its provenance entry")
            continue

        if parquet_path not in finalization_info:
            raise RuntimeError(
                f"collection_meta: table '{table}' has no finalization facts for {parquet_path}; "
                "cannot attest the current collection event"
            )
        info = finalization_info[parquet_path]
        if (
            not _valid_sha256_digest(info.source_digest)
            or not isinstance(info.source_device, int)
            or isinstance(info.source_device, bool)
            or not isinstance(info.source_inode, int)
            or isinstance(info.source_inode, bool)
        ):
            raise RuntimeError(f"collection_meta: table '{table}' has no valid staging file identity")
        new_rows_by_table[table] = info.new_rows
        merged_existing_by_table[table] = info.merged_existing
        finalization_by_table[table] = info

        case_ids: set[str] = set()
        unresolved_failed = 0
        for full_name in full_names:
            collection = collection_by_full_name[full_name]
            if producer_plan is None:
                resume_tracker = _load_selected_producer_checkpoint(
                    collection,
                    provenance_ctx,
                    backend=backend,
                    checkpoint_dir=checkpoint_dir,
                    sm_version=provenance_ctx.get("sm_version"),
                    staging_path=staging_by_table[table],
                    required=True,
                    context="collection_meta",
                )
                if resume_tracker is None:
                    continue
                checkpoint_plan = _producer_checkpoint_plan(resume_tracker, table)
            else:
                resume_tracker = _resume_tracker_for_collection(
                    collection,
                    provenance_ctx,
                    backend=backend,
                    checkpoint_dir=checkpoint_dir,
                    sm_version=provenance_ctx.get("sm_version"),
                )
                checkpoint_plan = producer_plan.get(resume_tracker._path)
                if (
                    checkpoint_plan is None
                    or checkpoint_plan.table != table
                    or checkpoint_plan.identity_dict() != resume_tracker._metadata
                ):
                    raise RuntimeError(
                        f"collection_meta: selected producer plan does not own table '{table}': {resume_tracker._path}"
                    )
            checkpoint_path = checkpoint_plan.path
            attempted = set(checkpoint_plan.attempted)
            failed = set(checkpoint_plan.failed)
            if checkpoint_path in checkpoint_records:
                checkpoint_record = checkpoint_records[checkpoint_path]
                if checkpoint_record != checkpoint_plan:
                    raise RuntimeError(f"collection_meta: checkpoint producer {checkpoint_path} changed identity")
            else:
                checkpoint_records[checkpoint_path] = checkpoint_plan
            case_ids.update(attempted)
            unresolved_failed += len(failed)

        if not case_ids:
            # Empty pending-event evidence means no case explains these rows:
            # every op's checkpoint is missing/unreadable, predates the
            # attempted ledger, or was already closed by a successful sidecar.
            # Finalized parquet with zero attempted cases is unattestable —
            # fail closed instead of writing a fabricated 'complete' sidecar
            # whose case_plan_hash covers an empty case set.
            raise RuntimeError(
                f"collection_meta: table '{table}' has finalized parquet ({parquet_path}) but no "
                f"readable checkpoint evidence for any of its ops ({', '.join(full_names)}) under "
                f"{checkpoint_root}. Zero attempted cases cannot explain produced parquet; writing "
                "a sidecar here would attest provenance that was never observed. Verify the "
                "checkpoint dir matches the one this collection ran with."
            )

        with os.fdopen(
            locked_output_root.open(
                parquet_path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
            ),
            "rb",
        ) as parquet_file:
            parquet_rows = pq.read_metadata(parquet_file).num_rows
        tables[table] = {
            "collector_ref": collector_ref,
            "collector_hash": provenance.collector_hash(module, _REPO_ROOT, closures),
            "case_plan_hash": provenance.case_plan_hash(sorted(case_ids)),
            "collected_at": collected_at,
            "rows": parquet_rows,
            "status": provenance.derive_table_status(
                unresolved_failed_count=unresolved_failed,
                had_module_failure=any(full_name in module_failure_names for full_name in full_names),
            ),
        }

    if not tables:
        return None
    staging_files = {
        staging_by_table[table]: finalization_by_table[table]
        for table in tables
        if table in staging_by_table and table in finalization_by_table
    }

    # A prior invocation against the same scratch dir (e.g. --ops split across
    # runs) may have already written a sidecar for other tables; preserve them.
    existing_meta = output_root / "collection_meta.yaml"
    existing_doc, existing_sidecar_snapshot = _load_clean_collector_sidecar(
        output_root,
        tables_to_update=set(tables),
        locked_output_root=locked_output_root,
    )
    provenance_tier: str | None = None
    if existing_doc is not None:
        provenance_tier = existing_doc.get("provenance")
        existing_tables = existing_doc["tables"]
        existing_runtime = existing_doc.get("runtime")
        if existing_runtime != runtime_meta:
            raise RuntimeError(
                f"{existing_meta}: cannot write collector provenance in place with a different runtime identity "
                f"(existing={existing_runtime!r}, current={runtime_meta!r}). Use a clean output directory for "
                "the new runtime."
            )
        surviving_tables = [
            table
            for table in existing_tables
            if table not in tables or merged_existing_by_table.get(table) is not False
        ]
        if provenance_tier == "local" and not surviving_tables:
            provenance_tier = None
        merged_tables = dict(existing_tables)
        for table, current_entry in tables.items():
            existing_entry = existing_tables.get(table)
            if isinstance(existing_entry, dict) and merged_existing_by_table[table]:
                merged_tables[table] = provenance.append_collection_event(
                    existing_entry,
                    {**current_entry, "rows": new_rows_by_table[table]},
                    table=table,
                    merged_rows=current_entry["rows"],
                )
            else:
                merged_tables[table] = current_entry
        tables = merged_tables

    meta_path = _commit_collector_provenance_transaction(
        output_root,
        runtime_meta,
        tables,
        provenance_tier=provenance_tier,
        checkpoint_records=checkpoint_records,
        staging_files=staging_files,
        existing_sidecar_snapshot=existing_sidecar_snapshot,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        perf_publication_transaction=perf_publication_transaction,
        locked_output_root=locked_output_root,
    )
    return meta_path


def _finalize_collector_outputs_transaction(
    output_root: Path,
    staging_paths: list[Path],
    provenance_ctx: dict,
    run_errors: list[dict],
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
) -> list[Path]:
    _require_separate_checkpoint_root(output_root, checkpoint_dir, backend)
    with perf_finalization_lifecycle(output_root) as locked_output_root:
        return _finalize_collector_outputs_transaction_locked(
            locked_output_root.path,
            staging_paths,
            provenance_ctx,
            run_errors,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            sm_version=sm_version,
            locked_output_root=locked_output_root,
        )


def _finalize_collector_outputs_transaction_locked(
    output_root: Path,
    staging_paths: list[Path],
    provenance_ctx: dict,
    run_errors: list[dict],
    *,
    backend: str,
    checkpoint_dir: str,
    sm_version: int | None,
    locked_output_root: LockedOutputRoot,
) -> list[Path]:
    """Commit parquets, provenance, staging cleanup, and checkpoints coherently."""
    if not staging_paths:
        return []
    output_root = locked_output_root.path
    tables_to_update = {path.stem for path in staging_paths}
    sidecar_snapshot = _preflight_collector_provenance(
        output_root,
        provenance_ctx,
        tables_to_update=tables_to_update,
        locked_output_root=locked_output_root,
    )
    producer_plan, staging_attestations = _preflight_collector_finalization_inputs(
        output_root,
        staging_paths,
        provenance_ctx,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        sm_version=sm_version,
        locked_output_root=locked_output_root,
    )

    def prepublish_validate():
        for transaction_path in (
            output_root / _PERF_TRANSACTION_FILENAME,
            output_root / _SIDECAR_TRANSACTION_FILENAME,
            output_root / _SIDECAR_STAGING_FILENAME,
        ):
            if not locked_output_root.absent(transaction_path):
                raise RuntimeError(f"Unexpected collector finalization transaction artifact: {transaction_path}")
        _revalidate_producer_plan(producer_plan)
        _revalidate_transaction_staging_files(
            staging_attestations.values(),
            context_path=output_root,
            require_present=True,
            locked_output_root=locked_output_root,
        )
        _revalidate_collector_sidecar_preflight(
            output_root,
            sidecar_snapshot,
            tables_to_update=tables_to_update,
            locked_output_root=locked_output_root,
        )

    publication_transaction = _CollectorPerfPublicationTransaction(
        output_root,
        backend=backend,
        checkpoint_dir=checkpoint_dir,
        producer_plan=producer_plan,
        sidecar_snapshot=sidecar_snapshot,
        locked_output_root=locked_output_root,
    )
    finalization_info: dict[Path, PerfFinalizationInfo] = {}
    converted = finalize_perf_files(
        staging_paths,
        delete_source=False,
        finalization_info=finalization_info,
        prepublish_validate=prepublish_validate,
        expected_source_identities={
            path: (attestation.digest, attestation.device, attestation.inode)
            for path, attestation in staging_attestations.items()
        },
        publication_transaction=publication_transaction,
        _locked_output_roots={output_root: locked_output_root},
    )
    try:
        meta_path = _write_collector_provenance(
            output_root,
            converted,
            provenance_ctx,
            run_errors,
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            finalization_info=finalization_info,
            producer_plan=producer_plan,
            perf_publication_transaction=publication_transaction,
            locked_output_root=locked_output_root,
        )
        if meta_path is None:
            raise RuntimeError("Collector finalization produced parquets without a provenance sidecar")
    except Exception:
        if publication_transaction.journal_attestation is not None and locked_output_root.absent(
            output_root / _SIDECAR_TRANSACTION_FILENAME
        ):
            publication_transaction.rollback()
        raise
    return converted


def main():
    global logger
    parser = argparse.ArgumentParser(description="Collect performance data for backends")
    parser.add_argument("--backend", type=str, choices=["trtllm", "sglang", "vllm"], default="trtllm")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--ops",
        nargs="*",
        type=str,
        choices=_all_op_names(),
        help="Run only specified collection items. Leave empty to run all. "
        "Available ops vary by backend — see backend-specific registry.py for details.",
        default=None,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: randomly sample 4 test cases per op to verify the collector runs end-to-end",
    )
    parser.add_argument(
        "--measure_power",
        action="store_true",
        help="Enable power monitoring during kernel execution (samples at 100ms intervals)",
    )
    parser.add_argument(
        "--power_test_duration_sec",
        type=float,
        default=1.0,
        help="Minimum duration for kernel runs when power measurement is enabled (default: 1.0s)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume collection from checkpoint, skipping passed and failed tasks",
    )
    parser.add_argument(
        "--resume-retry-failed",
        action="store_true",
        help="When resuming, retry previously failed tasks instead of skipping them. Requires --resume.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=".collector_checkpoint",
        help="Directory for per-module resume checkpoints (default: .collector_checkpoint)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of test cases per collection (useful for debugging)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle test cases before applying --limit (uses seed 42 for reproducibility)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Collector v2 model path (for example 'MiniMaxAI/MiniMax-M2.5'). "
        "When set, collect.py resolves collector/cases/models/<architecture>_cases.yaml by model alias, "
        "then runs only the planned ops/cases.",
    )
    parser.add_argument(
        "--model-architecture",
        type=str,
        default=None,
        help="Collector v2 model architecture (for example 'Qwen3MoeForCausalLM'). "
        "Defaults to resolving the architecture case file from --model-path aliases.",
    )
    parser.add_argument(
        "--model-cases",
        type=str,
        default=None,
        help="Optional path to a model cases YAML file. Defaults to collector/cases/models/<architecture>_cases.yaml.",
    )
    parser.add_argument(
        "--model-cases-full",
        action="store_true",
        help="Collector v2 full mode: aggregate base op cases plus every model cases YAML file.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU type for resolving hardware capability floors, for example b200_sxm. "
        "The SM version is read from src/aiconfigurator/systems/<gpu>.yaml unless --sm is provided.",
    )
    parser.add_argument(
        "--sm",
        type=int,
        default=None,
        help="Explicit SM version for hardware capability floors, for example 100. "
        "Overrides --gpu SM resolution; defaults to the local device capability.",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        dest="case_filters",
        default=None,
        metavar="SUBSTR",
        help="Run only cases whose string form contains SUBSTR (repeatable, OR semantics). "
        "Ephemeral healing filter — never persisted to YAML.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the collector v2 case plan and exit without running collectors.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the collector run and save output ",
    )
    parser.add_argument(
        "--keep-csv",
        action="store_true",
        help="Keep collector CSV staging files instead of finalizing *_perf.txt outputs to parquet.",
    )
    from collector.fpm_forward import FPM_FORWARD_OP, add_fpm_arguments
    from collector.fpm_forward.config import add_fpm_generator_arguments

    add_fpm_arguments(parser)
    # Registering these flags imports no Generator code (they live in
    # collector.fpm_forward.config, already imported above), so they are added
    # unconditionally: an argv pre-scan cannot reproduce argparse's own
    # abbreviation semantics (--op=fpm_forward) and crashed when it disagreed.
    # reject_fpm_arguments_without_fpm keeps them explicit-only.
    add_fpm_generator_arguments(parser)
    args = parser.parse_args()
    from collector.fpm_forward.config import reject_fpm_arguments_without_fpm

    try:
        reject_fpm_arguments_without_fpm(args)
    except ValueError as error:
        parser.error(str(error))
    fpm_requested = FPM_FORWARD_OP in (args.ops or ())
    if fpm_requested and set(args.ops or ()) != {FPM_FORWARD_OP}:
        parser.error("fpm_forward must be collected alone; do not mix campaign and local op entries")
    if fpm_requested and not (args.model_path or args.model_architecture or args.model_cases):
        parser.error("fpm_forward requires --model-path, --model-architecture, or --model-cases")
    ops = args.ops
    case_plan = None
    logger_message = None
    if args.plan_only and not (args.model_path or args.model_architecture or args.model_cases or args.model_cases_full):
        parser.error("--plan-only requires --model-path, --model-architecture, --model-cases, or --model-cases-full")
    if args.model_path or args.model_architecture or args.model_cases or args.model_cases_full:
        from collector.model_cases import build_collection_case_plan

        if args.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = args.model_path
        else:
            os.environ.pop("COLLECTOR_MODEL_PATH", None)

        case_plan = build_collection_case_plan(
            backend=args.backend,
            model_path=args.model_path,
            model_architecture=args.model_architecture,
            gpu_type=args.gpu,
            sm_version=args.sm,
            model_cases_path=args.model_cases,
            full=args.model_cases_full,
        )
        if case_plan.model_path:
            os.environ["COLLECTOR_MODEL_PATH"] = case_plan.model_path

        planned_ops = case_plan.ops
        if args.ops is None:
            ops = planned_ops
        else:
            requested_ops = set(args.ops) - {FPM_FORWARD_OP}
            ops = [op for op in planned_ops if op in requested_ops]
            missing_ops = requested_ops - set(ops)
            if missing_ops:
                parser.error(
                    "Requested ops are not present in the collector v2 case plan: " + ", ".join(sorted(missing_ops))
                )

        if (args.model_path or args.model_architecture) and not case_plan.model_cases_paths:
            logger_message = (
                "No collector v2 model cases YAML found for "
                f"model_path={args.model_path!r}, model_architecture={args.model_architecture!r}; "
                "using base op cases only plus legacy model filtering."
            )

        if args.plan_only and fpm_requested:
            from collector.fpm_forward.entry import resolve_inputs

            fpm_plan, _generator_overrides = _resolve_fpm_cli_inputs(parser, lambda: resolve_inputs(args, case_plan))
            print(json.dumps(fpm_plan.to_dict(), indent=2, sort_keys=True))
            return
        if args.plan_only:
            log_dict = case_plan.to_log_dict()
            log_dict["ops"] = ops
            print(json.dumps(log_dict, indent=2))
            return
    else:
        os.environ.pop("COLLECTOR_MODEL_PATH", None)

    # Setup logging - debug flag is handled inside setup_logging
    if logger is None:
        if args.model_cases_full:
            log_scope = ["model_cases_full"]
        else:
            log_scope = ops if ops else ["all"]
        logger = setup_logging(scope=log_scope, debug=args.debug)
    elif args.debug:
        # Update log level if debug flag changed
        setup_logging(debug=args.debug)

    if logger_message:
        logger.warning(logger_message)
    if case_plan is not None:
        logger.info("Collector v2 case plan active:")
        for key, value in case_plan.to_log_dict().items():
            logger.info(f"  {key}: {value}")
        if ops and args.ops is None:
            logger.info(f"  expanded to model-specific ops: {ops}")
    elif args.model_path:
        logger.info(f"Legacy model filter active: collecting only for '{args.model_path}'")

    # Hardware capability floor target: explicit --sm / --gpu wins, otherwise
    # detect from the local device (None on XPU -> filter is permissive).
    from collector.capabilities import detect_sm_version
    from collector.model_cases import resolve_sm_version

    sm_version = (
        case_plan.sm_version
        if case_plan is not None and case_plan.sm_version is not None
        else resolve_sm_version(gpu_type=args.gpu, sm_version=args.sm)
    )
    if sm_version is None:
        sm_version = detect_sm_version()
    logger.info(f"Hardware capability floors target SM version: {sm_version}")

    resume_options = {
        "resume": args.resume,
        "checkpoint_dir": args.checkpoint_dir,
        "retry_failed": args.resume_retry_failed,
    }
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")
    if args.resume:
        logger.info(
            f"Resume enabled: dir={Path(args.checkpoint_dir).expanduser()}"
            + (" (retrying previously failed tasks)" if args.resume_retry_failed else "")
        )

    if fpm_requested:
        from collector.fpm_forward.entry import resolve_run_inputs, run_resolved

        resolved_inputs = _resolve_fpm_cli_inputs(parser, lambda: resolve_run_inputs(args, case_plan))
        run_errors = run_resolved(args, resolved_inputs)
        generate_collection_summary(run_errors, args.backend, "generator-resolved")
        if run_errors:
            raise SystemExit(1)
        return

    _require_torch()

    # Determine number of processes (0 = sequential mode for profiling)
    if args.profile:
        num_processes = 0
        logger.info("Starting collection in sequential mode (profiling enabled)")
    else:
        num_processes = get_device_module().device_count()
        logger.info(f"Starting collection with {num_processes} GPU processes")

    # Set environment variables for worker processes
    if args.measure_power:
        os.environ["COLLECTOR_MEASURE_POWER"] = "true"
        os.environ["COLLECTOR_POWER_MIN_DURATION"] = str(args.power_test_duration_sec)
        logger.info(f"Power monitoring enabled (min duration: {args.power_test_duration_sec}s)")
    else:
        os.environ["COLLECTOR_MEASURE_POWER"] = "false"

    # Suppress torch operator override warnings in spawned workers
    # (env var takes effect at interpreter startup, before any module imports)
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:torch.library"

    shuffle = args.shuffle
    limit = args.limit
    if args.smoke:
        shuffle = True
        limit = args.limit if args.limit is not None else 4
        logger.info(f"Smoke test mode enabled — sampling {limit} random test cases per op")

    # Warn if profiling without limit (profiling can be very slow)
    if args.profile and limit is None:
        logger.warning(
            "Profiling is enabled but --limit is not set. "
            "Profiling all test cases can be very slow. "
            "Consider using --limit to restrict the number of test cases."
        )

    # Disable core dumps — GPU crashes are expected and handled; core files waste disk.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    # Only set multiprocessing start method if not profiling (profiling uses sequential mode via num_processes=0)
    if not args.profile:
        mp.set_start_method("spawn")

    output_root = Path.cwd()
    _recover_collector_provenance_transaction(
        output_root,
        backend=args.backend,
        checkpoint_dir=args.checkpoint_dir,
    )
    existing_perf_outputs = {path.resolve(): path.stat().st_mtime_ns for path in find_perf_csv_outputs(output_root)}

    def was_touched_by_run(path: Path) -> bool:
        resolved = path.resolve()
        return resolved not in existing_perf_outputs or path.stat().st_mtime_ns != existing_perf_outputs[resolved]

    # Use profiling context manager
    with ProfilerContext(args.backend, enabled=args.profile):
        collect_backend = {"trtllm": collect_trtllm, "sglang": collect_sglang, "vllm": collect_vllm}[args.backend]
        run_errors, provenance_ctx = collect_backend(
            num_processes,
            ops,
            limit=limit,
            shuffle=shuffle,
            resume_options=resume_options,
            model_path=case_plan.model_path if case_plan is not None else None,
            case_plan=case_plan,
            sm_version=sm_version,
            case_filters=args.case_filters,
        )

    converted: list[Path] = []
    if args.keep_csv:
        logger.info("Keeping collector CSV staging files because --keep-csv was passed")
    else:
        touched_perf_outputs = {path for path in find_perf_csv_outputs(output_root) if was_touched_by_run(path)}
        pending_resume_outputs: set[Path] = set()
        if resume_options.get("resume") and provenance_ctx is not None:
            pending_resume_outputs.update(
                _pending_resume_perf_outputs(
                    output_root,
                    provenance_ctx,
                    backend=args.backend,
                    checkpoint_dir=args.checkpoint_dir,
                    sm_version=sm_version,
                )
            )
        touched_perf_outputs = sorted(touched_perf_outputs | pending_resume_outputs)
        if touched_perf_outputs:
            logger.info(
                "Finalizing collector CSV staging files as parquet:\n  "
                + "\n  ".join(str(path) for path in touched_perf_outputs)
            )
        if touched_perf_outputs and provenance_ctx is not None:
            converted = _finalize_collector_outputs_transaction(
                output_root,
                touched_perf_outputs,
                provenance_ctx,
                run_errors or [],
                backend=args.backend,
                checkpoint_dir=args.checkpoint_dir,
                sm_version=sm_version,
            )
        else:
            converted = finalize_perf_files(touched_perf_outputs, delete_source=False)
        if converted:
            logger.info(f"Finalized {len(converted)} collector perf files as parquet")

    # A ModuleCollectionFailure means an op failed before running a single case
    # (population raised, or the run infrastructure crashed) — the op collected
    # nothing. Exit non-zero AFTER finalization so partial data from other ops
    # is still packaged, but the job is not reported as a clean success.
    module_failures = sorted(
        {e["module"] for e in (run_errors or []) if e.get("error_type") == "ModuleCollectionFailure"}
    )
    if module_failures:
        logger.error("Module-level collection failures (no cases ran): " + ", ".join(module_failures))
        raise SystemExit(1)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    main()
