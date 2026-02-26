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
    2: Validation errors (sanity check failures)

Usage:
    python compare_support_matrix.py --old <old_csv> --new <new_csv> [--output-diff <diff_file>]
"""

import argparse
import csv
import json
import os
import sys

# Ensure local repo paths are importable when running as a standalone script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from tools.support_matrix.support_matrix import SupportMatrix


def read_csv(csv_path: str) -> tuple[list[str], list[list[str]]]:
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
    expected_header = ["HuggingFaceID", "Architecture", "System", "Backend", "Version", "Mode", "Status", "ErrMsg"]

    if header != expected_header:
        errors.append(f"Invalid header: expected {expected_header}, got {header}")
        return errors  # Can't continue without valid header

    if len(data_rows) == 0:
        errors.append("CSV file has header but no data rows")
        return errors

    for i, row in enumerate(data_rows, start=2):
        if len(row) != len(expected_header):
            errors.append(f"Row {i} has {len(row)} columns, expected {len(expected_header)}")
            continue

        mode = row[5]
        if mode not in ["agg", "disagg"]:
            errors.append(f"Row {i}: Invalid mode '{mode}', expected 'agg' or 'disagg'")

        status = row[6]
        if status not in ["PASS", "FAIL"]:
            errors.append(f"Row {i}: Invalid status '{status}', expected 'PASS' or 'FAIL'")

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
    regressions = [r for r in changed_rows if r[6] == "PASS" and r[7] == "FAIL"]
    fixed = [r for r in changed_rows if r[6] == "FAIL" and r[7] == "PASS"]

    lines = [
        "This PR updates aiconfigurator/systems/support_matrix.csv with the following changes:",
        "",
        "### Summary",
        "",
        "| Category | Count |",
        "|----------|-------|",
        f"| Regressions (PASS -> FAIL) | {len(regressions)} |",
        f"| Fixed (FAIL -> PASS) | {len(fixed)} |",
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
    lines.append(f"### {section}. Fixed (FAIL -> PASS): {len(fixed)} rows")
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
        help="Path to the old support matrix CSV",
    )
    parser.add_argument(
        "--new",
        type=str,
        required=True,
        help="Path to the new support matrix CSV",
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
    print(f"Old CSV: {args.old}")
    print(f"New CSV: {args.new}")
    print()

    # Read both CSVs
    try:
        old_header, old_data_rows = read_csv(args.old)
        print(f"✓ Old CSV loaded: {len(old_data_rows)} rows")
    except Exception as e:
        print(f"✗ Failed to read old CSV: {e}")
        sys.exit(2)

    try:
        new_header, new_data_rows = read_csv(args.new)
        print(f"✓ New CSV loaded: {len(new_data_rows)} rows")
    except Exception as e:
        print(f"✗ Failed to read new CSV: {e}")
        sys.exit(2)

    print()

    # Run validation checks on new CSV
    validation_errors = []

    print("Running validation checks on new CSV...")
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
    print("Comparing old and new CSVs...")
    print("-" * 40)

    added_rows, removed_rows, changed_rows = compare_csv_files(old_data_rows, new_data_rows)

    print(f"Added rows: {len(added_rows)}")
    print(f"Removed rows: {len(removed_rows)}")
    print(f"Changed rows: {len(changed_rows)}")
    print(f"  - Regressions (PASS -> FAIL): {len([r for r in changed_rows if r[6] == 'PASS' and r[7] == 'FAIL'])}")
    print(f"  - Fixed (FAIL -> PASS): {len([r for r in changed_rows if r[6] == 'FAIL' and r[7] == 'PASS'])}")

    has_changes = len(added_rows) > 0 or len(removed_rows) > 0 or len(changed_rows) > 0

    regressions = [r for r in changed_rows if r[6] == "PASS" and r[7] == "FAIL"]
    fixed = [r for r in changed_rows if r[6] == "FAIL" and r[7] == "PASS"]

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
            "added_rows": added_rows,
            "removed_rows": removed_rows,
            "changed_rows": changed_rows,
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
