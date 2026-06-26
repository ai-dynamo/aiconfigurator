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
    STATUS_PASS,
    SUPPORT_MATRIX_BASE_HEADER,
    SUPPORT_MATRIX_HEADER,
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

    for i, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            errors.append(f"Row {i} has {len(row)} columns, expected {len(header)}")
            continue

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
        if header == SUPPORT_MATRIX_HEADER and not row[8].strip():
            errors.append(f"Row {i}: Command column must include the support-matrix rerun command")

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


def find_blocking_status_transitions(changed_rows: list[tuple]) -> list[str]:
    """
    Return status transitions that should block an automated support-matrix PR.

    Hardware-incompatible rows are produced by deterministic preflight, and
    framework-incompatible rows are produced by deterministic runtime/data gap
    classification. A previous PASS becoming either terminal incompatibility, or
    an incompatibility becoming a normal FAIL, should be investigated explicitly.
    """
    errors = []
    for huggingface_id, architecture, system, backend, version, mode, old_status, new_status in changed_rows:
        if old_status == STATUS_PASS and new_status in {STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}:
            errors.append(
                f"Unexpected PASS -> {new_status} transition: "
                f"{huggingface_id} ({architecture}) {system}/{backend} v{version} {mode}"
            )
        elif old_status in {STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE} and new_status == STATUS_FAIL:
            errors.append(
                f"Unexpected {old_status} -> FAIL transition: "
                f"{huggingface_id} ({architecture}) {system}/{backend} v{version} {mode}"
            )
    return errors


def generate_pr_description(added_rows: list[tuple], removed_rows: list[tuple], changed_rows: list[tuple]) -> str:
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
    regressions = [r for r in changed_rows if r[6] == STATUS_PASS and r[7] == STATUS_FAIL]
    fixed = [
        r
        for r in changed_rows
        if r[7] == STATUS_PASS and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
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
        f"| Regressions (PASS -> FAIL) | {len(regressions)} |",
        f"| Fixed ({STATUS_FAIL}/{STATUS_HW_INCOMPATIBLE}/{STATUS_FRAMEWORK_INCOMPATIBLE} -> PASS) | {len(fixed)} |",
        f"| Reclassified as hardware-incompatible ({STATUS_FAIL} -> {STATUS_HW_INCOMPATIBLE}) "
        f"| {len(reclassified_hw)} |",
        f"| Reclassified as framework-incompatible ({STATUS_FAIL} -> {STATUS_FRAMEWORK_INCOMPATIBLE}) "
        f"| {len(reclassified_framework)} |",
        f"| Removed rows | {len(removed_rows)} |",
        f"| Added rows | {len(added_rows)} |",
        "",
    ]

    section = 1

    # Regressions (PASS -> FAIL)
    lines.append(f"### {section}. Regressions (PASS -> FAIL): {len(regressions)} rows")
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

    # Fixed (FAIL -> PASS)
    lines.append(
        f"### {section}. Fixed "
        f"({STATUS_FAIL}/{STATUS_HW_INCOMPATIBLE}/{STATUS_FRAMEWORK_INCOMPATIBLE} -> PASS): {len(fixed)} rows"
    )
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
    transition_errors = find_blocking_status_transitions(changed_rows)
    validation_errors.extend(transition_errors)

    regression_count = len([r for r in changed_rows if r[6] == STATUS_PASS and r[7] == STATUS_FAIL])
    fixed_count = len(
        [
            r
            for r in changed_rows
            if r[7] == STATUS_PASS and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
        ]
    )
    reclassified_hw_count = len([r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_HW_INCOMPATIBLE])
    reclassified_framework_count = len(
        [r for r in changed_rows if r[6] == STATUS_FAIL and r[7] == STATUS_FRAMEWORK_INCOMPATIBLE]
    )

    print(f"Added rows: {len(added_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Changed rows: {len(changed_rows)}")
    print(f"  - Regressions (PASS -> FAIL): {regression_count}")
    print(f"  - Fixed ({STATUS_FAIL}/{STATUS_HW_INCOMPATIBLE}/{STATUS_FRAMEWORK_INCOMPATIBLE} -> PASS): {fixed_count}")
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

    has_changes = len(added_rows) > 0 or len(removed_rows) > 0 or len(changed_rows) > 0

    regressions = [r for r in changed_rows if r[6] == STATUS_PASS and r[7] == STATUS_FAIL]
    fixed = [
        r
        for r in changed_rows
        if r[7] == STATUS_PASS and r[6] in {STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE}
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
            "regression_count": len(regressions),
            "fixed_count": len(fixed),
            "reclassified_hw_incompatible_count": len(reclassified_hw),
            "reclassified_framework_incompatible_count": len(reclassified_framework),
            "added_rows": added_rows,
            "removed_rows": removed_rows,
            "changed_rows": changed_rows,
            "blocking_status_transition_errors": transition_errors,
            "pr_description": generate_pr_description(added_rows, removed_rows, changed_rows),
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
        print(generate_pr_description(added_rows, removed_rows, changed_rows))
        sys.exit(1)
    else:
        print("RESULT: No changes detected - support matrix is up to date")
        sys.exit(0)


if __name__ == "__main__":
    main()
