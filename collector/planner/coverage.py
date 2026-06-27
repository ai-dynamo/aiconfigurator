# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic compatibility manifests for collector physical coverage."""

from __future__ import annotations

import gzip
import io
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from collector.planner.physical_keys import (
    PHYSICAL_KEY_REGISTRY,
    PHYSICAL_KEY_SCHEMA_VERSION,
    PhysicalRowKey,
)

COVERAGE_CANONICAL_VERSION = 1


class CoverageManifestError(ValueError):
    """A coverage manifest is malformed or cannot be compared safely."""


class CoverageMismatchError(AssertionError):
    """Generated coverage is missing protected legacy physical keys."""


def canonical_perf_table(perf_file: str | os.PathLike[str]) -> str:
    """Return the registered parquet table name or fail for an unknown table."""

    name = Path(perf_file).name
    table = f"{name[:-4]}.parquet" if name.endswith(".txt") else name
    if table not in PHYSICAL_KEY_REGISTRY:
        raise CoverageManifestError(f"unknown perf table: {name}")
    return table


@dataclass(frozen=True, slots=True)
class CoverageHeader:
    """Scope and serialization contract for one physical-key manifest."""

    source_git_ref: str
    backend_variant: str
    framework_version: str
    gpu_type: str
    sm_version: int
    perf_table: str
    key_schema_version: int = PHYSICAL_KEY_SCHEMA_VERSION
    canonical_version: int = COVERAGE_CANONICAL_VERSION

    def __post_init__(self) -> None:
        for name in ("source_git_ref", "backend_variant", "framework_version", "gpu_type", "perf_table"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CoverageManifestError(f"{name} must be a non-empty string")
        if isinstance(self.sm_version, bool) or not isinstance(self.sm_version, int) or self.sm_version < 0:
            raise CoverageManifestError("sm_version must be a non-negative integer")
        if isinstance(self.key_schema_version, bool) or not isinstance(self.key_schema_version, int):
            raise CoverageManifestError("key_schema_version must be an integer")
        if isinstance(self.canonical_version, bool) or not isinstance(self.canonical_version, int):
            raise CoverageManifestError("canonical_version must be an integer")

        table = canonical_perf_table(self.perf_table)
        object.__setattr__(self, "perf_table", table)
        schema_version = PHYSICAL_KEY_REGISTRY[table].version
        if self.key_schema_version != schema_version:
            raise CoverageManifestError(
                f"key schema version mismatch for {table}: header={self.key_schema_version}, registry={schema_version}"
            )
        if self.canonical_version != COVERAGE_CANONICAL_VERSION:
            raise CoverageManifestError(
                f"unsupported coverage canonical version {self.canonical_version}; "
                f"expected {COVERAGE_CANONICAL_VERSION}"
            )


@dataclass(frozen=True, slots=True)
class CoverageManifest:
    """One header and its exact set of consumer-visible physical keys."""

    header: CoverageHeader
    keys: frozenset[PhysicalRowKey]

    def __post_init__(self) -> None:
        key_set = frozenset(self.keys)
        schema = PHYSICAL_KEY_REGISTRY[self.header.perf_table]
        for key in key_set:
            if key.table != self.header.perf_table:
                raise CoverageManifestError(
                    f"physical key table {key.table!r} does not match manifest table {self.header.perf_table!r}"
                )
            if key.schema_version != self.header.key_schema_version:
                raise CoverageManifestError(
                    f"physical key schema version {key.schema_version} does not match "
                    f"manifest version {self.header.key_schema_version}"
                )
            if key.fields != schema.fields:
                raise CoverageManifestError(f"physical key fields do not match registry for {key.table}")
        object.__setattr__(self, "keys", key_set)


@dataclass(frozen=True, slots=True)
class CoverageDiff:
    """Set difference from protected legacy coverage to generated coverage."""

    legacy_count: int
    generated_count: int
    missing: tuple[PhysicalRowKey, ...]
    added: tuple[PhysicalRowKey, ...]

    @property
    def retained_count(self) -> int:
        return self.legacy_count - len(self.missing)

    @property
    def legacy_is_subset(self) -> bool:
        return not self.missing


_HEADER_FIELDS = tuple(item.name for item in fields(CoverageHeader))
_COMPARISON_FIELDS = tuple(name for name in _HEADER_FIELDS if name != "source_git_ref")


def _json_line(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise CoverageManifestError(f"coverage value is not canonical JSON: {exc}") from exc


def _key_sort_token(key: PhysicalRowKey) -> str:
    return _json_line(list(key.values))


def _manifest_bytes(manifest: CoverageManifest) -> bytes:
    lines = [_json_line(asdict(manifest.header))]
    lines.extend(_key_sort_token(key) for key in sorted(manifest.keys, key=_key_sort_token))
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
        compressed.write(payload)
    return output.getvalue()


def write(
    path: str | os.PathLike[str],
    manifest_or_header: CoverageManifest | CoverageHeader,
    keys: Iterable[PhysicalRowKey] | None = None,
) -> CoverageManifest:
    """Write a deterministic gzip JSONL manifest and return its normalized value."""

    if isinstance(manifest_or_header, CoverageManifest):
        if keys is not None:
            raise TypeError("keys must not be provided when writing a CoverageManifest")
        manifest = manifest_or_header
    else:
        if keys is None:
            raise TypeError("keys are required when writing a CoverageHeader")
        manifest = CoverageManifest(header=manifest_or_header, keys=frozenset(keys))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_manifest_bytes(manifest))
    return manifest


def _parse_header(value: Any) -> CoverageHeader:
    if not isinstance(value, Mapping):
        raise CoverageManifestError("manifest line 1 must be a JSON object header")
    actual_fields = set(value)
    expected_fields = set(_HEADER_FIELDS)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise CoverageManifestError(f"invalid coverage header fields: missing={missing}, extra={extra}")
    try:
        return CoverageHeader(**{name: value[name] for name in _HEADER_FIELDS})
    except TypeError as exc:
        raise CoverageManifestError(f"invalid coverage header: {exc}") from exc


def _freeze_json_value(value: Any) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    raise CoverageManifestError(f"physical key contains unsupported JSON value {value!r}")


def load(path: str | os.PathLike[str]) -> CoverageManifest:
    """Load and validate a deterministic gzip JSONL coverage manifest."""

    input_path = Path(path)
    try:
        payload = gzip.decompress(input_path.read_bytes()).decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise CoverageManifestError(f"cannot read coverage manifest {input_path}: {exc}") from exc

    lines = payload.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise CoverageManifestError(f"coverage manifest is empty: {input_path}")
    try:
        header_value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise CoverageManifestError(f"invalid JSON on manifest line 1: {exc.msg}") from exc
    header = _parse_header(header_value)
    schema = PHYSICAL_KEY_REGISTRY[header.perf_table]

    keys: set[PhysicalRowKey] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        if not line:
            raise CoverageManifestError(f"blank manifest line {line_number}")
        try:
            values = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CoverageManifestError(f"invalid JSON on manifest line {line_number}: {exc.msg}") from exc
        if not isinstance(values, list):
            raise CoverageManifestError(f"manifest line {line_number} must be a physical-key values array")
        if len(values) != len(schema.fields):
            raise CoverageManifestError(
                f"manifest line {line_number} has {len(values)} values; expected {len(schema.fields)}"
            )
        _json_line(values)  # Reject non-finite or otherwise non-canonical JSON values.
        key = PhysicalRowKey(
            schema_version=header.key_schema_version,
            table=header.perf_table,
            fields=schema.fields,
            values=tuple(_freeze_json_value(item) for item in values),
        )
        keys.add(key)

    return CoverageManifest(header=header, keys=frozenset(keys))


def _assert_compatible_headers(legacy: CoverageHeader, generated: CoverageHeader) -> None:
    mismatches = [
        f"{name}: legacy={getattr(legacy, name)!r}, generated={getattr(generated, name)!r}"
        for name in _COMPARISON_FIELDS
        if getattr(legacy, name) != getattr(generated, name)
    ]
    if mismatches:
        raise CoverageManifestError("incompatible coverage manifests: " + "; ".join(mismatches))


def diff(legacy: CoverageManifest, generated: CoverageManifest) -> CoverageDiff:
    """Return deterministic missing and added key sets for compatible manifests."""

    _assert_compatible_headers(legacy.header, generated.header)
    missing = tuple(sorted(legacy.keys - generated.keys, key=_key_sort_token))
    added = tuple(sorted(generated.keys - legacy.keys, key=_key_sort_token))
    return CoverageDiff(
        legacy_count=len(legacy.keys),
        generated_count=len(generated.keys),
        missing=missing,
        added=added,
    )


def assert_legacy_subset(
    legacy: CoverageManifest,
    generated: CoverageManifest,
    *,
    sample_size: int = 5,
) -> CoverageDiff:
    """Assert that generated coverage retains every protected legacy key."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    result = diff(legacy, generated)
    if result.missing:
        sample = ", ".join(_key_sort_token(key) for key in result.missing[:sample_size])
        raise CoverageMismatchError(
            f"generated coverage is missing {len(result.missing)} legacy physical key(s) "
            f"for {legacy.header.perf_table}; missing key sample: {sample}"
        )
    return result


write_coverage_manifest = write
load_coverage_manifest = load
diff_coverage_manifests = diff


__all__ = [
    "COVERAGE_CANONICAL_VERSION",
    "CoverageDiff",
    "CoverageHeader",
    "CoverageManifest",
    "CoverageManifestError",
    "CoverageMismatchError",
    "assert_legacy_subset",
    "canonical_perf_table",
    "diff",
    "diff_coverage_manifests",
    "load",
    "load_coverage_manifest",
    "write",
    "write_coverage_manifest",
]
