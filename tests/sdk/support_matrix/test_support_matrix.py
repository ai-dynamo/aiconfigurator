# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import subprocess
import sys
from collections import defaultdict

import pytest
from packaging.version import Version

# Add tests directory to path for support_matrix module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sdk.support_matrix.suppport_matrix import SupportMatrix

SUPPORT_MATRIX_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "src",
    "aiconfigurator",
    "systems",
    "support_matrix.csv",
)

GENERATE_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "tools",
    "support_matrix",
    "generate_support_matrix.py",
)


@pytest.fixture
def csv_data():
    """
    Fixture that reads the support matrix CSV and returns header and data rows.

    Returns:
        tuple: (header, data_rows) where:
            - header: list of column names
            - data_rows: list of lists containing the data
    """
    assert os.path.exists(SUPPORT_MATRIX_CSV), f"Support matrix CSV not found at {SUPPORT_MATRIX_CSV}"

    with open(SUPPORT_MATRIX_CSV, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) > 0, "CSV file is empty"

    header = rows[0]
    data_rows = rows[1:]

    return header, data_rows


@pytest.fixture(scope="session")
def fresh_csv_data(tmp_path_factory):
    """
    Fixture that generates a fresh support matrix CSV and returns header and data rows.
    Uses session scope to ensure the CSV is only generated once per test session (results are cached by pytest).

    Returns:
        tuple: (header, data_rows) where:
            - header: list of column names
            - data_rows: list of lists containing the data
    """
    # Create a temporary directory for this session
    tmp_dir = tmp_path_factory.mktemp("support_matrix")
    output_csv = tmp_dir / "support_matrix_fresh.csv"

    # Run the generate script
    result = subprocess.run(
        ["python", GENERATE_SCRIPT, "--output", str(output_csv)],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        pytest.fail(f"Failed to generate support matrix: {result.stderr}")

    assert output_csv.exists(), f"Generated CSV not found at {output_csv}"

    # Read the generated CSV
    with open(output_csv, newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) > 0, "Generated CSV file is empty"

    header = rows[0]
    data_rows = rows[1:]

    return header, data_rows


@pytest.mark.skipif(
    os.environ.get("TEST_SUPPORT_MATRIX", "").lower() != "true",
    reason="TEST_SUPPORT_MATRIX environment variable must be set to 'true' to run these tests",
)
class TestSupportMatrix:
    """
    Test suite for the support matrix.
    Set TEST_SUPPORT_MATRIX=true to enable these tests.
    """

    def test_system_and_backend_matches_database(self):
        """
        Test that the system and backend defined in the support matrix matches the database.
        """
        support_matrix = SupportMatrix()
        systems_in_database = set(support_matrix.databases.keys())
        backends_in_database = {
            backend for system in systems_in_database for backend in support_matrix.databases[system]
        }
        assert systems_in_database == support_matrix.get_systems()
        assert backends_in_database == support_matrix.get_backends()

    def test_csv_sanity_check(self, csv_data):
        """
        Test that the CSV file exists, has correct structure, and contains data.
        """
        header, data_rows = csv_data

        # Check header row
        expected_header = ["Model", "System", "Backend", "Version", "Mode", "Status", "ErrMsg"]
        assert header == expected_header, f"Expected header {expected_header}, got {header}"

        # Check that there are data rows
        assert len(data_rows) > 0, "CSV file has header but no data rows"

        # Validate data rows have correct number of columns
        for i, row in enumerate(data_rows, start=2):
            assert len(row) == len(expected_header), f"Row {i} has {len(row)} columns, expected {len(expected_header)}"

            # Check that Mode column has valid values
            mode = row[4]
            assert mode in ["agg", "disagg"], f"Row {i}: Invalid mode '{mode}', expected 'agg' or 'disagg'"

            # Check that Status column has valid values
            status = row[5]
            assert status in ["PASS", "FAIL"], f"Row {i}: Invalid status '{status}', expected 'PASS' or 'FAIL'"

    def test_range_matches_database(self, csv_data):
        """
        Test that the CSV contains exactly the combinations expected from the database.
        Each combination should appear twice (once for agg, once for disagg mode).
        """
        header, data_rows = csv_data

        # Get expected combinations from the support matrix
        support_matrix = SupportMatrix()
        expected_base_combinations = set(support_matrix.generate_combinations())

        # Each base combination should have both agg and disagg entries
        expected_combinations = set()
        for model, system, backend, version in expected_base_combinations:
            expected_combinations.add((model, system, backend, version, "agg"))
            expected_combinations.add((model, system, backend, version, "disagg"))

        # Extract actual combinations from CSV (convert to dict format for easy access)
        # Header indices: Model=0, System=1, Backend=2, Version=3, Mode=4
        actual_combinations = {(row[0], row[1], row[2], row[3], row[4]) for row in data_rows}

        # Compare sets
        assert expected_combinations == actual_combinations, (
            f"CSV combinations don't match expected combinations.\n"
            f"Missing in CSV: {expected_combinations - actual_combinations}\n"
            f"Extra in CSV: {actual_combinations - expected_combinations}"
            + "\n\nIf these are intentional improvements, update the committed support_matrix.csv"
            " with the `tools/support_matrix/generate_support_matrix.py` script and commit the changes."
        )

    def test_no_deprecated_support(self, csv_data, fresh_csv_data):
        """
        Test that all previously supported items are still supported.
        This ensures no regression - configurations that used to work should still work.
        """
        _, old_data_rows = csv_data
        _, new_data_rows = fresh_csv_data

        # Build a dict of (model, system, backend, version, mode) -> status for old CSV
        # Status indices: Model=0, System=1, Backend=2, Version=3, Mode=4, Status=5
        old_status_map = {(row[0], row[1], row[2], row[3], row[4]): row[5] for row in old_data_rows}

        # Build a dict for new CSV
        new_status_map = {(row[0], row[1], row[2], row[3], row[4]): row[5] for row in new_data_rows}

        # Find configurations that were PASS in old but are now FAIL in new
        deprecated_support = []
        for config, old_status in old_status_map.items():
            if old_status == "PASS":
                new_status = new_status_map.get(config)
                if new_status == "FAIL":
                    model, system, backend, version, mode = config
                    deprecated_support.append(f"{model} on {system} with {backend} v{version} ({mode})")

        # Assert no regressions
        assert len(deprecated_support) == 0, (
            f"Found {len(deprecated_support)} previously supported configurations that are now failing:\n"
            + "\n".join(f"  - {item}" for item in deprecated_support)
            + "\n\nIf these are intentional improvements, update the committed support_matrix.csv"
            " with the `tools/support_matrix/generate_support_matrix.py` script and commit the changes."
        )

    def test_no_undocumented_support(self, csv_data, fresh_csv_data):
        """
        Test that all previously unsupported items are still unsupported.
        This ensures no accidental new support - configurations that used to fail should still fail
        unless there was an intentional change.
        """
        _, old_data_rows = csv_data
        _, new_data_rows = fresh_csv_data

        # Build a dict of (model, system, backend, version, mode) -> status for old CSV
        old_status_map = {(row[0], row[1], row[2], row[3], row[4]): row[5] for row in old_data_rows}

        # Build a dict for new CSV
        new_status_map = {(row[0], row[1], row[2], row[3], row[4]): row[5] for row in new_data_rows}

        # Find configurations that were FAIL in old but are now PASS in new
        undocumented_support = []
        for config, old_status in old_status_map.items():
            if old_status == "FAIL":
                new_status = new_status_map.get(config)
                if new_status == "PASS":
                    model, system, backend, version, mode = config
                    undocumented_support.append(f"{model} on {system} with {backend} v{version} ({mode})")

        # If there are newly supported configurations, list them
        # This is not necessarily a failure - it might be intentional
        # But we want to be explicit about it
        if len(undocumented_support) > 0:
            message = (
                f"Found {len(undocumented_support)} newly supported configurations:\n"
                + "\n".join(f"  - {item}" for item in undocumented_support)
                + "\n\nIf these are intentional improvements, update the committed support_matrix.csv"
                " with the `tools/support_matrix/generate_support_matrix.py` script and commit the changes."
            )
            pytest.fail(message)

    def test_newer_versions_have_no_narrower_support(self, csv_data):
        """
        For each (model, system, backend), get its latest version and second latest version.
        If the latest version is not supported, the second latest version should also not be supported.
        """
        _, data_rows = csv_data

        # Group data by (model, system, backend, mode)
        # Key: (model, system, backend, mode) -> List of (version, status)
        grouped_data = defaultdict(list)

        for row in data_rows:
            model, system, backend, version, mode, status = row[0], row[1], row[2], row[3], row[4], row[5]
            key = (model, system, backend, mode)
            grouped_data[key].append((version, status))

        violations = []

        # For each group, check the version constraint
        for (model, system, backend, mode), version_status_list in grouped_data.items():
            # Sort by version (descending - latest first)
            sorted_versions = sorted(version_status_list, key=lambda x: Version(x[0]), reverse=True)

            # Need at least 2 versions to compare
            if len(sorted_versions) < 2:
                continue

            latest_version, latest_status = sorted_versions[0]
            second_latest_version, second_latest_status = sorted_versions[1]

            # If latest is not supported (FAIL), second latest should also not be supported (FAIL)
            # Using set subtraction: if latest is FAIL, we expect second_latest to be FAIL
            # Violation: latest is FAIL but second_latest is PASS
            if latest_status == "FAIL" and second_latest_status == "PASS":
                violations.append(
                    f"{model} on {system} with {backend} ({mode}): "
                    f"Latest version {latest_version} is FAIL, but older version {second_latest_version} is PASS"
                )

        # Assert no violations found
        assert len(violations) == 0, (
            f"Found {len(violations)} cases where newer versions have narrower support than older versions:\n"
            + "\n".join(f"  - {item}" for item in violations)
            + "\n\nNewer versions should not have narrower support than older versions."
        )
