#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compare old and new support matrix CSVs and validate consistency.

This script performs the following checks:
1. CSV sanity check - validates structure and data
2. Range matches database - ensures CSV has expected combinations

Exit codes:
    0: All checks pass, no changes detected
    1: Changes detected (added, removed, or changed rows)
    2: Validation errors (sanity or range check failures)

Usage:
    python compare_support_matrix.py --old <old_matrix> --new <new_matrix> [--output-diff <diff_file>]
"""

import argparse
import csv
import json
import os
import shlex
import sys
from pathlib import Path

# Ensure local repo paths are importable when running as a standalone script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from tools.support_matrix.support_matrix import (
    STATUS_FAIL,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    STATUS_HYBRID_PASS,
    STATUS_PASS,
    SUPPORT_MATRIX_BASE_HEADER,
    SUPPORT_MATRIX_HEADER,
    VALID_PROVENANCE_SOURCES,
    VALID_STATUSES,
    SupportMatrix,
)

# Accept the transitional 9-col header (base + Command, pre-Source) alongside the
# current 10-col (base + Command + Source) and the legacy 8-col base. Some committed
# per-system CSVs were generated at the 9-col stage; rejecting them breaks compare.
# Mirrors support_matrix.py:_row_values, which already reads 8/9/10-col rows.
_SUPPORT_MATRIX_HEADER_WITH_COMMAND = SUPPORT_MATRIX_BASE_HEADER + ["Command"]
SUPPORTED_HEADERS = (
    SUPPORT_MATRIX_HEADER,
    _SUPPORT_MATRIX_HEADER_WITH_COMMAND,
    SUPPORT_MATRIX_BASE_HEADER,
)


def _read_single_csv(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """
    Read a CSV file and return header and data rows.

    Args:
        csv_path: Path to the CSV file

    Returns:
        tuple: (header, data_rows)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError("CSV file is empty")

    header = rows[0]
    data_rows = rows[1:]

    return header, data_rows


def read_csv(matrix_path: str) -> tuple[list[str], list[list[str]]]:
    """
    Read either a legacy support matrix CSV or a split support matrix directory.

    Split support matrix directories are expected to contain one CSV per system,
    named ``<system>.csv``.
    """
    path = Path(matrix_path)
    if not path.is_dir():
        return _read_single_csv(path)

    csv_paths = sorted(path.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No support matrix CSV files found in directory: {matrix_path}")

    combined_header: list[str] | None = None
    combined_rows: list[list[str]] = []
    for csv_path in csv_paths:
        header, data_rows = _read_single_csv(csv_path)
        if combined_header is None:
            combined_header = header
        elif header != combined_header:
            raise ValueError(f"Inconsistent CSV header in {csv_path}: expected {combined_header}, got {header}")

        systems = {row[2] for row in data_rows if len(row) >= 3}
        if len(systems) != 1:
            raise ValueError(f"{csv_path} must contain rows for exactly one system, found {sorted(systems)}")

        [system] = systems
        if csv_path.stem != system:
            raise ValueError(f"{csv_path} filename must match its System value '{system}'")

        combined_rows.extend(data_rows)

    return combined_header or [], combined_rows


def check_csv_sanity(header: list[str], data_rows: list[list[str]]) -> list[str]:
    """
    Validate CSV structure and data.

    Args:
        header: CSV header row
        data_rows: CSV data rows

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    if header not in SUPPORTED_HEADERS:
        errors.append(f"Invalid header: expected {SUPPORT_MATRIX_HEADER}, got {header}")
        return errors  # Can't continue without valid header

    if len(data_rows) == 0:
        errors.append("CSV file has header but no data rows")
        return errors

    seen_keys: dict[tuple[str, ...], int] = {}
    for i, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            errors.append(f"Row {i} has {len(row)} columns, expected {len(header)}")
            continue

        key = tuple(row[:6])
        if key in seen_keys:
            errors.append(f"Row {i}: duplicate support-matrix key; first seen on row {seen_keys[key]}: {key}")
        else:
            seen_keys[key] = i

        mode = row[5]
        if mode not in ["agg", "disagg"]:
            errors.append(f"Row {i}: Invalid mode '{mode}', expected 'agg' or 'disagg'")

        status = row[6]
        if status not in VALID_STATUSES:
            errors.append(f"Row {i}: Invalid status '{status}', expected one of {sorted(VALID_STATUSES)}")

        err_msg = row[7].strip() if len(row) > 7 else ""
        if status == STATUS_HW_INCOMPATIBLE and not err_msg:
            errors.append(f"Row {i}: {STATUS_HW_INCOMPATIBLE} rows must include a hardware incompatibility reason")
        if status == STATUS_FRAMEWORK_INCOMPATIBLE and not err_msg:
            errors.append(
                f"Row {i}: {STATUS_FRAMEWORK_INCOMPATIBLE} rows must include a framework incompatibility reason"
            )
        if status == STATUS_HYBRID_PASS and header != SUPPORT_MATRIX_HEADER:
            errors.append(f"Row {i}: HYBRID_PASS requires the current header with Command and Source columns")
        if header == SUPPORT_MATRIX_HEADER and not row[8].strip():
            errors.append(f"Row {i}: Command column must include the support-matrix rerun command")
        if header == SUPPORT_MATRIX_HEADER:
            command = row[8]
            source = row[9].strip()
            if source and source not in VALID_PROVENANCE_SOURCES:
                errors.append(
                    f"Row {i}: Invalid Source={source!r}, expected one of {sorted(VALID_PROVENANCE_SOURCES)} or empty"
                )
            if status == STATUS_PASS and source != "silicon":
                errors.append(f"Row {i}: PASS is reserved for SILICON support; found Source={source!r}")
            if status == STATUS_HYBRID_PASS and source not in VALID_PROVENANCE_SOURCES - {"silicon"}:
                errors.append(f"Row {i}: HYBRID_PASS must include an empirical transfer Source")
            if status not in {STATUS_PASS, STATUS_HYBRID_PASS} and source:
                errors.append(f"Row {i}: non-pass status {status} must not include Source={source!r}")

            expected_database_mode = "HYBRID" if status == STATUS_HYBRID_PASS else "SILICON"
            try:
                command_parts = shlex.split(command)
            except ValueError as exc:
                errors.append(f"Row {i}: Command is not valid shell syntax: {exc}")
            else:
                database_modes: list[str] = []
                missing_database_mode_value = False
                for part_index, part in enumerate(command_parts):
                    if part == "--database-mode":
                        if part_index + 1 >= len(command_parts) or command_parts[part_index + 1].startswith("-"):
                            missing_database_mode_value = True
                        else:
                            database_modes.append(command_parts[part_index + 1].upper())
                    elif part.startswith("--database-mode="):
                        value = part.partition("=")[2]
                        if value:
                            database_modes.append(value.upper())
                        else:
                            missing_database_mode_value = True
                if missing_database_mode_value or database_modes != [expected_database_mode]:
                    errors.append(
                        f"Row {i}: replay command must include exactly one effective "
                        f"--database-mode {expected_database_mode}; found {database_modes or 'none'}"
                    )

    return errors


def check_range_matches_database(data_rows: list[list[str]]) -> list[str]:
    """
    Verify CSV contains exactly the combinations expected from the database.

    Args:
        data_rows: CSV data rows

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []

    support_matrix = SupportMatrix()
    expected_base_combinations = set(support_matrix.generate_combinations())

    # Each base combination should have both agg and disagg entries
    # Note: generate_combinations returns (huggingface_id, system, backend, version)
    # Models are identified by HuggingFace IDs from DefaultHFModels
    expected_combinations = set()
    for huggingface_id, system, backend, version in expected_base_combinations:
        architecture = support_matrix.get_architecture(huggingface_id)
        expected_combinations.add((huggingface_id, architecture, system, backend, version, "agg"))
        expected_combinations.add((huggingface_id, architecture, system, backend, version, "disagg"))

    # Extract actual combinations from CSV (huggingface_id, architecture, system, backend, version, mode)
    actual_combinations = {(row[0], row[1], row[2], row[3], row[4], row[5]) for row in data_rows}

    missing = expected_combinations - actual_combinations
    extra = actual_combinations - expected_combinations

    if missing:
        errors.append(f"Missing in CSV: {len(missing)} combinations")
        for combo in sorted(missing)[:10]:  # Limit output
            errors.append(f"  - {combo}")
        if len(missing) > 10:
            errors.append(f"  ... and {len(missing) - 10} more")

    if extra:
        errors.append(f"Extra in CSV: {len(extra)} combinations")
        for combo in sorted(extra)[:10]:
            errors.append(f"  - {combo}")
        if len(extra) > 10:
            errors.append(f"  ... and {len(extra) - 10} more")

    return errors


def compare_csv_files(
    old_data_rows: list[list[str]], new_data_rows: list[list[str]]
) -> tuple[list[tuple], list[tuple], list[tuple]]:
    """
    Compare old and new CSV data to find added, removed, and changed rows.

    Args:
        old_data_rows: Data rows from old CSV
        new_data_rows: Data rows from new CSV

    Returns:
        Tuple of (added_rows, removed_rows, changed_rows)
        - added_rows: List of (huggingface_id, architecture, system, backend, version, mode, status) tuples
        - removed_rows: List of (huggingface_id, architecture, system, backend, version, mode, status) tuples
        - changed_rows: List of (huggingface_id, architecture, system, backend, version,
            mode, old_status, new_status) tuples
    """
    # Build dicts: key = (huggingface_id, architecture, system, backend, version, mode) -> status
    old_status_map = {(row[0], row[1], row[2], row[3], row[4], row[5]): row[6] for row in old_data_rows}
    new_status_map = {(row[0], row[1], row[2], row[3], row[4], row[5]): row[6] for row in new_data_rows}

    old_keys = set(old_status_map.keys())
    new_keys = set(new_status_map.keys())

    # Find added rows (in new but not in old)
    added_rows = []
    for key in sorted(new_keys - old_keys):
        huggingface_id, architecture, system, backend, version, mode = key
        status = new_status_map[key]
        added_rows.append((huggingface_id, architecture, system, backend, version, mode, status))

    # Find removed rows (in old but not in new)
    removed_rows = []
    for key in sorted(old_keys - new_keys):
        huggingface_id, architecture, system, backend, version, mode = key
        status = old_status_map[key]
        removed_rows.append((huggingface_id, architecture, system, backend, version, mode, status))

    # Find changed rows (in both, but status changed)
    changed_rows = []
    for key in sorted(old_keys & new_keys):
        old_status = old_status_map[key]
        new_status = new_status_map[key]
        if old_status != new_status:
            huggingface_id, architecture, system, backend, version, mode = key
            changed_rows.append((huggingface_id, architecture, system, backend, version, mode, old_status, new_status))

    return added_rows, removed_rows, changed_rows


def find_metadata_changes(old_data_rows: list[list[str]], new_data_rows: list[list[str]]) -> list[tuple]:
    """Return Command/Source changes for rows present in both matrices."""

    def _metadata(row: list[str]) -> tuple[str, str]:
        return (row[8] if len(row) > 8 else "", row[9] if len(row) > 9 else "")

    old_rows = {tuple(row[:6]): row for row in old_data_rows}
    new_rows = {tuple(row[:6]): row for row in new_data_rows}
    changes = []
    for key in sorted(old_rows.keys() & new_rows.keys()):
        old_command, old_source = _metadata(old_rows[key])
        new_command, new_source = _metadata(new_rows[key])
        if (old_command, old_source) != (new_command, new_source):
            changes.append((*key, old_command, new_command, old_source, new_source))
    return changes


def find_blocking_status_transitions(changed_rows: list[tuple]) -> list[str]:
    """
    Return status transitions that should block an automated support-matrix PR.

    Hardware-incompatible rows are produced by deterministic preflight, and
    framework-incompatible rows are produced by deterministic runtime/data gap
    classification. Losing SILICON support (including PASS -> HYBRID_PASS), losing
    HYBRID estimability, or turning an incompatibility into a normal FAIL should be
    investigated explicitly.
    """
    errors = []
    for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in changed_rows:
        if old_status == STATUS_PASS and new_status != STATUS_PASS:
            errors.append(
                f"Unexpected PASS -> {new_status} transition: "
                f"{huggingface_id} ({architecture}) {system}/{backend} v{version} {mode}"
            )
        elif old_status == STATUS_HYBRID_PASS and new_status in {
            STATUS_FAIL,
            STATUS_HW_INCOMPATIBLE,
            STATUS_FRAMEWORK_INCOMPATIBLE,
        }:
            errors.append(
                f"Unexpected {STATUS_HYBRID_PASS} -> {new_status} transition: "
                f"{huggingface_id} ({architecture}) {system}/{backend} v{version} {mode}"
            )
        elif old_status in {STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE} and new_status == STATUS_FAIL:
            errors.append(
                f"Unexpected {old_status} -> FAIL transition: "
                f"{huggingface_id} ({architecture}) {system}/{backend} v{version} {mode}"
            )
    return errors


def generate_pr_description(
    added_rows: list[tuple],
    removed_rows: list[tuple],
    changed_rows: list[tuple],
    *,
    metadata_changes: list[tuple] | None = None,
    header_changed: bool = False,
) -> str:
    """
    Generate PR description markdown with tables of changes.

    Sections are ordered by importance for readability when GitHub truncates
    long output: regressions first, then fixed, removed, and added.

    Args:
        added_rows: List of added row tuples
        removed_rows: List of removed row tuples
        changed_rows: List of changed row tuples (each has old_status, new_status)

    Returns:
        Markdown formatted PR description
    """
    metadata_changes = metadata_changes or []
    regressions = [r for r in changed_rows if r[6] == STATUS_PASS and r[7] != STATUS_PASS]
    hybrid_regressions = [
        r
        for r in changed_rows
        if r[6] == STATUS_HYBRID_PASS and r[7] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    fixed = [
        r
        for r in changed_rows
        if r[7] == STATUS_PASS
        and r[6] in {STATUS_HYBRID_PASS, STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    hybrid_coverage = [
        r
        for r in changed_rows
        if r[7] == STATUS_HYBRID_PASS and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    reclassified_hw = [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_HW_INCOMPATIBLE]
    reclassified_framework = [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_FRAMEWORK_INCOMPATIBLE]

    lines = [
        "This PR updates the split support matrix CSV files with the following changes:",
        "",
        "### Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Silicon regressions (PASS -> non-PASS) | {len(regressions)} |",
        f"| Hybrid regressions ({STATUS_HYBRID_PASS} -> non-pass) | {len(hybrid_regressions)} |",
        f"| Silicon fixes (non-PASS -> PASS) | {len(fixed)} |",
        f"| New hybrid coverage (non-pass -> {STATUS_HYBRID_PASS}) | {len(hybrid_coverage)} |",
        f"| Reclassified as hardware-incompatible ({STATUS_FAIL} -> {STATUS_HW_INCOMPATIBLE}) "
        f"| {len(reclassified_hw)} |",
        f"| Reclassified as framework-incompatible ({STATUS_FAIL} -> {STATUS_FRAMEWORK_INCOMPATIBLE}) "
        f"| {len(reclassified_framework)} |",
        f"| Removed rows | {len(removed_rows)} |",
        f"| Added rows | {len(added_rows)} |",
        f"| Command/Source metadata changes | {len(metadata_changes)} |",
        f"| Header changed | {'yes' if header_changed else 'no'} |",
        "",
    ]

    section = 1

    # Regressions from measured-silicon support.
    lines.append(f"### {section}. Silicon regressions (PASS -> non-PASS): {len(regressions)} rows")
    section += 1
    if regressions:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in regressions:
            row = (
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No regressions*")
    lines.append("")

    lines.append(
        f"### {section}. Hybrid regressions ({STATUS_HYBRID_PASS} -> non-pass): {len(hybrid_regressions)} rows"
    )
    section += 1
    if hybrid_regressions:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in hybrid_regressions:
            lines.append(
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
    else:
        lines.extend(["", "*No hybrid regressions*"])
    lines.append("")

    # Fixed to measured-silicon support.
    lines.append(f"### {section}. Silicon fixes (non-PASS -> PASS): {len(fixed)} rows")
    section += 1
    if fixed:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in fixed:
            row = (
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No fixes*")
    lines.append("")

    lines.append(f"### {section}. New hybrid coverage (non-pass -> {STATUS_HYBRID_PASS}): {len(hybrid_coverage)} rows")
    section += 1
    if hybrid_coverage:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in hybrid_coverage:
            lines.append(
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
    else:
        lines.extend(["", "*No new hybrid coverage*"])
    lines.append("")

    # Reclassified hardware incompatibilities
    lines.append(
        f"### {section}. Reclassified as hardware-incompatible ({STATUS_FAIL} -> {STATUS_HW_INCOMPATIBLE}): "
        f"{len(reclassified_hw)} rows"
    )
    section += 1
    if reclassified_hw:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in reclassified_hw:
            row = (
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No hardware-incompatible reclassifications*")
    lines.append("")

    lines.append(
        f"### {section}. Reclassified as framework-incompatible "
        f"({STATUS_FAIL} -> {STATUS_FRAMEWORK_INCOMPATIBLE}): {len(reclassified_framework)} rows"
    )
    section += 1
    if reclassified_framework:
        lines.append("")
        lines.append(
            "| HuggingFaceID | Architecture | System | Backend | Version | Mode | Previous Status | New Status |"
        )
        lines.append(
            "|---------------|--------------|--------|---------|---------|------|-----------------|------------|"
        )
        for (
            huggingface_id,
            architecture,
            system,
            backend,
            version,
            mode,
            old_status,
            new_status,
        ) in reclassified_framework:
            row = (
                f"| {huggingface_id} | {architecture} | {system} | {backend} "
                f"| {version} | {mode} | {old_status} | {new_status} |"
            )
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No framework-incompatible reclassifications*")
    lines.append("")

    # Removed rows
    lines.append(f"### {section}. Removed rows: {len(removed_rows)}")
    section += 1
    if removed_rows:
        lines.append("")
        lines.append("| HuggingFaceID | Architecture | System | Backend | Version | Mode | Status |")
        lines.append("|---------------|--------------|--------|---------|---------|------|--------|")
        for huggingface_id, architecture, system, backend, version, mode, status in removed_rows:
            row = f"| {huggingface_id} | {architecture} | {system} | {backend} | {version} | {mode} | {status} |"
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No rows removed*")
    lines.append("")

    # Added rows
    lines.append(f"### {section}. Added rows: {len(added_rows)}")
    if added_rows:
        lines.append("")
        lines.append("| HuggingFaceID | Architecture | System | Backend | Version | Mode | Status |")
        lines.append("|---------------|--------------|--------|---------|---------|------|--------|")
        for huggingface_id, architecture, system, backend, version, mode, status in added_rows:
            row = f"| {huggingface_id} | {architecture} | {system} | {backend} | {version} | {mode} | {status} |"
            lines.append(row)
    else:
        lines.append("")
        lines.append("*No rows added*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare old and new support matrix CSVs and validate consistency")
    parser.add_argument(
        "--old",
        type=str,
        required=True,
        help="Path to the old support matrix CSV or split support matrix directory",
    )
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path to the new support matrix CSV or split support matrix directory",
    )
    parser.add_argument(
        "--output-diff",
        type=str,
        help="Output file path to save diff results as JSON",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Support Matrix Comparison")
    print("=" * 80)
    print(f"Old support matrix: {args.old}")
    print(f"New support matrix: {args.new}")
    print()

    # Read both CSVs
    try:
        old_header, old_data_rows = read_csv(args.old)
        print(f"✓ Old support matrix loaded: {len(old_data_rows)} rows")
    except Exception as e:
        print(f"✗ Failed to read old support matrix: {e}")
        sys.exit(2)

    try:
        new_header, new_data_rows = read_csv(args.new)
        print(f"✓ New support matrix loaded: {len(new_data_rows)} rows")
    except Exception as e:
        print(f"✗ Failed to read new support matrix: {e}")
        sys.exit(2)

    print()

    # Run validation checks on new CSV
    validation_errors = []

    print("Running validation checks on new support matrix...")
    print("-" * 40)

    # 1. CSV sanity check
    sanity_errors = check_csv_sanity(new_header, new_data_rows)
    if sanity_errors:
        print("✗ CSV sanity check failed:")
        for err in sanity_errors:
            print(f"  - {err}")
        validation_errors.extend(sanity_errors)
    else:
        print("✓ CSV sanity check passed")

    # 2. Range matches database
    range_errors = check_range_matches_database(new_data_rows)
    if range_errors:
        print("✗ Range check failed:")
        for err in range_errors:
            print(f"  - {err}")
        validation_errors.extend(range_errors)
    else:
        print("✓ Range matches database")

    print()

    # Compare CSVs
    print("Comparing old and new support matrices...")
    print("-" * 40)

    added_rows, removed_rows, changed_rows = compare_csv_files(old_data_rows, new_data_rows)
    metadata_changes = find_metadata_changes(old_data_rows, new_data_rows)
    header_changed = old_header != new_header
    transition_errors = find_blocking_status_transitions(changed_rows)
    validation_errors.extend(transition_errors)

    regression_count = len([r for r in changed_rows if r[6] == STATUS_PASS and r[7] != STATUS_PASS])
    hybrid_regression_count = len(
        [
            r
            for r in changed_rows
            if r[6] == STATUS_HYBRID_PASS
            and r[7] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
        ]
    )
    fixed_count = len(
        [
            r
            for r in changed_rows
            if r[7] == STATUS_PASS
            and r[6] in {STATUS_HYBRID_PASS, STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
        ]
    )
    hybrid_coverage_count = len(
        [
            r
            for r in changed_rows
            if r[7] == STATUS_HYBRID_PASS
            and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
        ]
    )
    reclassified_hw_count = len([r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_HW_INCOMPATIBLE])
    reclassified_framework_count = len(
        [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_FRAMEWORK_INCOMPATIBLE]
    )

    print(f"Added rows: {len(added_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Changed rows: {len(changed_rows)}")
    print(f"Command/Source metadata changes: {len(metadata_changes)}")
    print(f"Header changed: {header_changed}")
    print(f"  - Silicon regressions (PASS -> non-PASS): {regression_count}")
    print(f"  - Hybrid regressions ({STATUS_HYBRID_PASS} -> non-pass): {hybrid_regression_count}")
    print(f"  - Silicon fixes (non-PASS -> PASS): {fixed_count}")
    print(f"  - New hybrid coverage (non-pass -> {STATUS_HYBRID_PASS}): {hybrid_coverage_count}")
    print(
        f"  - Reclassified as hardware-incompatible ({STATUS_FAIL} -> {STATUS_HW_INCOMPATIBLE}): "
        f"{reclassified_hw_count}"
    )
    print(
        f"  - Reclassified as framework-incompatible ({STATUS_FAIL} -> {STATUS_FRAMEWORK_INCOMPATIBLE}): "
        f"{reclassified_framework_count}"
    )
    if transition_errors:
        print("Blocking status transitions:")
        for err in transition_errors:
            print(f"  - {err}")

    has_changes = bool(added_rows or removed_rows or changed_rows or metadata_changes or header_changed)

    regressions = [r for r in changed_rows if r[6] == STATUS_PASS and r[7] != STATUS_PASS]
    hybrid_regressions = [
        r
        for r in changed_rows
        if r[6] == STATUS_HYBRID_PASS and r[7] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    fixed = [
        r
        for r in changed_rows
        if r[7] == STATUS_PASS
        and r[6] in {STATUS_HYBRID_PASS, STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    hybrid_coverage = [
        r
        for r in changed_rows
        if r[7] == STATUS_HYBRID_PASS and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
    ]
    reclassified_hw = [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_HW_INCOMPATIBLE]
    reclassified_framework = [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_FRAMEWORK_INCOMPATIBLE]

    # Generate output
    if args.output_diff:
        diff_data = {
            "has_changes": has_changes,
            "validation_errors": validation_errors,
            "added_count": len(added_rows),
            "removed_count": len(removed_rows),
            "changed_count": len(changed_rows),
            "metadata_changed_count": len(metadata_changes),
            "header_changed": header_changed,
            "regression_count": len(regressions),
            "hybrid_regression_count": len(hybrid_regressions),
            "fixed_count": len(fixed),
            "hybrid_coverage_count": len(hybrid_coverage),
            "reclassified_hw_incompatible_count": len(reclassified_hw),
            "reclassified_framework_incompatible_count": len(reclassified_framework),
            "added_rows": added_rows,
            "removed_rows": removed_rows,
            "changed_rows": changed_rows,
            "metadata_changes": metadata_changes,
            "blocking_status_transition_errors": transition_errors,
            "pr_description": generate_pr_description(
                added_rows,
                removed_rows,
                changed_rows,
                metadata_changes=metadata_changes,
                header_changed=header_changed,
            ),
        }
        with open(args.output_diff, "w") as f:
            json.dump(diff_data, f, indent=2)
        print(f"\nDiff results saved to: {args.output_diff}")

    print()
    print("=" * 80)

    # Exit with appropriate code
    if validation_errors:
        print("RESULT: Validation errors detected")
        sys.exit(2)
    elif has_changes:
        print("RESULT: Changes detected - PR required")
        # Print PR description preview
        print()
        print("PR Description Preview:")
        print("-" * 40)
        print(
            generate_pr_description(
                added_rows,
                removed_rows,
                changed_rows,
                metadata_changes=metadata_changes,
                header_changed=header_changed,
            )
        )
        sys.exit(1)
    else:
        print("RESULT: No changes detected - support matrix is up to date")
        sys.exit(0)


if __name__ == "__main__":
    main()
