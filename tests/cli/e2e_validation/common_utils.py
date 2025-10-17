#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for E2E validation tests.
Shared functions for infrastructure checking, command execution, and result analysis.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def find_project_root():
    """
    Find the project root directory by looking for key files/directories.

    Returns:
        Path: Path to the project root directory

    Raises:
        RuntimeError: If project root cannot be found
    """
    current_path = Path(__file__).resolve()

    # Look for project indicators (pyproject.toml, src/ directory, etc.)
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent

    # If not found, raise an error
    raise RuntimeError(
        "Cannot find project root directory. Make sure you're running this script from within the project."
    )


def ensure_project_root():
    """
    Ensure we're running from the project root directory.
    Change to project root if necessary.

    Returns:
        Path: Path to the project root directory
    """
    project_root = find_project_root()
    current_dir = Path.cwd()

    if current_dir != project_root:
        print(f"📁 Changing working directory to project root: {project_root}")
        os.chdir(project_root)

    return project_root


def check_test_infrastructure():
    """
    Check if the test infrastructure is properly set up.

    Returns:
        tuple: (success: bool, message: str, test_count: int)
    """
    print("🔍 Checking test infrastructure...")

    # Ensure we're in the project root
    try:
        project_root = ensure_project_root()
    except RuntimeError as e:
        return False, f"❌ {e}", 0

    # Check if test files exist
    test_file = Path("tests/cli/e2e_validation/test_e2e_sweep.py")

    if not test_file.exists():
        return False, f"❌ Main test file not found: {test_file}", 0

    print("✅ Test infrastructure exists")
    print(f"✅ Main test file: {test_file}")
    print(f"✅ Working directory: {project_root}")

    # Try to collect tests without running them
    try:
        cmd = ["python3", "-m", "pytest", str(test_file), "--collect-only", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            test_count = len([line for line in lines if "test_e2e_configuration_sweep" in line])
            print(f"✅ Found {test_count} test combinations")
            return True, "Infrastructure ready", test_count
        else:
            return False, f"❌ Test collection failed: {result.stderr}", 0
    except Exception as e:
        return False, f"❌ Error checking tests: {e}", 0


def run_pytest_command(cmd, description):
    """
    Run a pytest command, streaming its output in real-time and capturing it for summary.

    Args:
        cmd (list): Command to execute
        description (str): Description of what this command does

    Returns:
        tuple: (success: bool, result: subprocess.CompletedProcess)
    """
    # Ensure we're in the project root before running pytest
    try:
        ensure_project_root()
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        return False, None

    print(f"\n{'=' * 60}")
    print(f"📋 {description}")
    print(f"💻 Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    output_lines = []
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
        output_lines.append(line)

    return_code = process.wait()

    if return_code == 0:
        print("✅ Success!")
    else:
        print("❌ Failed!")
        # The full output has already been streamed.
        # A summary could be printed here if needed, but the user will see the error in the stream.

    # Create a CompletedProcess-like object for compatibility.
    result = subprocess.CompletedProcess(
        cmd,
        return_code,
        stdout="".join(output_lines),
        stderr=None,  # stderr is redirected to stdout
    )

    return return_code == 0, result


def generate_test_filter(models=None, systems=None, gpu_configs=None, isl_osl_combinations=None, tpot_values=None):
    """
    Generate pytest filter string based on test parameters.

    The actual test names follow this format:
    test_e2e_configuration_sweep[0.20.0-5000-10-4000-1000-8-h200_sxm-model_QWEN3_32B]

    Args:
        models (list): List of model names to include
        systems (list): List of systems to include
        gpu_configs (list): List of GPU configurations to include
        isl_osl_combinations (list): List of (isl, osl) tuples to include
        tpot_values (list): List of TPOT values to include

    Returns:
        str: pytest filter string (-k argument)
    """
    filters = []

    if models:
        # Model names in test IDs have "model_" prefix
        model_filter = " or ".join([f"model_{model}" for model in models])
        filters.append(f"({model_filter})")

    if systems:
        # Systems appear directly in the test ID
        system_filter = " or ".join(systems)
        filters.append(f"({system_filter})")

    if gpu_configs:
        # GPU configs appear as "-{gpu}-" in test ID
        gpu_filter = " or ".join([f"-{gpu}-" for gpu in gpu_configs])
        filters.append(f"({gpu_filter})")

    if isl_osl_combinations:
        # ISL values appear as "-{isl}-" in test ID
        isl_filter = " or ".join([f"-{isl}-" for isl, _ in isl_osl_combinations])
        filters.append(f"({isl_filter})")

    if tpot_values:
        # TPOT values appear as "-{tpot}-" in test ID
        tpot_filter = " or ".join([f"-{tpot}-" for tpot in tpot_values])
        filters.append(f"({tpot_filter})")

    return " and ".join(filters) if filters else ""


def create_temp_results_dir(prefix="e2e_test_"):
    """
    Create a temporary directory for test results.

    Args:
        prefix (str): Prefix for the temporary directory name

    Returns:
        Path: Path to the created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return Path(temp_dir)


def cleanup_temp_dir(temp_dir):
    """
    Clean up a temporary directory.

    Args:
        temp_dir (Path): Path to the temporary directory to clean up
    """
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"🧹 Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"⚠️ Warning: Could not clean up {temp_dir}: {e}")


def print_cicd_summary(test_type, success, total_tests=None, failed_tests=None, execution_time=None):
    """
    Print a CI/CD friendly summary of test results.

    Args:
        test_type (str): Type of test executed
        success (bool): Whether the overall test was successful
        total_tests (int): Total number of tests run
        failed_tests (int): Number of failed tests
        execution_time (float): Total execution time in seconds
    """
    print(f"\n{'=' * 80}")
    print(f"🎯 {test_type.upper()} TEST RESULTS")
    print(f"{'=' * 80}")

    if total_tests is not None:
        success_rate = ((total_tests - (failed_tests or 0)) / total_tests * 100) if total_tests > 0 else 0
        print(f"📊 Total tests: {total_tests}")
        if failed_tests is not None:
            print(f"❌ Failed: {failed_tests}")
            print(f"✅ Success rate: {success_rate:.1f}%")

    if execution_time is not None:
        print(f"⏱️ Execution time: {execution_time:.1f}s")

    print(f"🎯 Overall result: {'✅ PASSED' if success else '❌ FAILED'}")
    print(f"{'=' * 80}")


def get_cicd_exit_code(success):
    """
    Get appropriate exit code for CI/CD systems.

    Args:
        success (bool): Whether the test was successful

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    return 0 if success else 1


def print_cicd_recommendations(test_type, success):
    """
    Print CI/CD specific recommendations based on test results.

    Args:
        test_type (str): Type of test that was executed
        success (bool): Whether the test was successful
    """
    print(f"\n💡 CI/CD Recommendations for {test_type}:")

    if success:
        if test_type == "smoke":
            print("   ✅ Smoke test passed - safe to proceed with deployment")
            print("   📝 Consider running selective tests for changed components")
        elif test_type == "selective":
            print("   ✅ Selective tests passed - targeted validation successful")
            print("   📝 Consider full sweep before major releases")
        elif test_type == "full_sweep":
            print("   ✅ Full sweep passed - comprehensive validation complete")
            print("   📝 All configurations are working correctly")
    else:
        if test_type == "smoke":
            print("   ❌ Smoke test failed - DO NOT deploy")
            print("   🔧 Fix critical issues before proceeding")
            print("   📝 Run selective tests to identify specific problems")
        elif test_type == "selective":
            print("   ❌ Selective tests failed - targeted issues detected")
            print("   🔧 Review specific configurations that failed")
            print("   📝 Fix issues before broader testing")
        elif test_type == "full_sweep":
            print("   ❌ Full sweep failed - comprehensive issues detected")
            print("   🔧 Systematic review of failed configurations needed")
            print("   📝 Consider rollback if in production environment")
