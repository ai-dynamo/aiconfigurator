# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end sweep tests for CLI functionality.

Tests all combinations of models, systems, total GPUs, ISL/OSL combinations,
TTFT/TPOT configurations across all supported versions to detect errors
and record detailed error information.
"""

import json
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiconfigurator.cli.main import main as cli_main
from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import get_latest_database_version

_SUPPORTED_MODELS = [model for model in common.SupportedModels if model not in ["QWEN3_480B", "KIMI_K2"]]
_SUPPORTED_SYSTEMS = ["h100_sxm", "h200_sxm", "b200_sxm", "gb200_sxm", "a100_sxm", "l40s"]
_SUPPORTED_GPU_CONFIGS = [32, 512]
_SUPPORTED_ISL_OSL_PREFIX_COMBINATIONS = [(4000, 1000, 0), (4000, 1000, 2000), (1000, 2, 0), (32, 1000, 0)]
_SUPPORTED_TTFT_TPOT_COMBINATIONS = [(5000, 10), (5000, 100)]
_EXCLUDED_MODEL_SYSTEM_PAIRS = {
    ("DEEPSEEK_V3", "a100_sxm"),
    ("DEEPSEEK_V3", "l40s"),
    ("QWEN3_235B", "l40s"),
}
_SUPPORTED_BACKENDS = ["trtllm", "sglang", "vllm"]


def _iter_valid_parameter_combinations():
    for model_name in _SUPPORTED_MODELS:
        for system in _SUPPORTED_SYSTEMS:
            if (model_name, system) in _EXCLUDED_MODEL_SYSTEM_PAIRS:
                continue
            for total_gpus in _SUPPORTED_GPU_CONFIGS:
                for isl, osl, prefix in _SUPPORTED_ISL_OSL_PREFIX_COMBINATIONS:
                    for ttft, tpot in _SUPPORTED_TTFT_TPOT_COMBINATIONS:
                        for backend in _SUPPORTED_BACKENDS:
                            yield pytest.param(
                                model_name,
                                system,
                                total_gpus,
                                isl,
                                osl,
                                prefix,
                                ttft,
                                tpot,
                                backend,
                                id=(
                                    f"MODEL_{model_name}"
                                    f"__SYSTEM_{system}"
                                    f"__GPU_{total_gpus}"
                                    f"__ISL_{isl}__OSL_{osl}__PREFIX_{prefix}"
                                    f"__TTFT_{int(ttft)}__TPOT_{int(tpot)}"
                                    f"__BACKEND_{backend}"
                                ),
                            )


_VALID_PARAMETER_COMBINATIONS = list(_iter_valid_parameter_combinations())


def _value_or_placeholder(value):
    return value or "N/A"


@pytest.fixture(scope="session")
def test_results_dir():
    """Create and return a persistent, timestamped test results directory for the session."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"e2e_test_results/e2e_sweep_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = Path("e2e_test_results/latest")
    try:
        if latest_dir.is_symlink():
            latest_dir.unlink(missing_ok=True)
        elif latest_dir.exists():
            if latest_dir.is_dir():
                shutil.rmtree(latest_dir, ignore_errors=True)
            else:
                latest_dir.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError:
        if latest_dir.exists() and latest_dir.is_dir():
            shutil.rmtree(latest_dir, ignore_errors=True)

    try:
        latest_dir.symlink_to(results_dir.resolve(), target_is_directory=True)
    except Exception:
        pass

    print(f"\n📝 Test results will be saved to: {results_dir.resolve()}")
    print(f"🔗 Latest results available at: {latest_dir.resolve()}")

    yield results_dir

    print(f"\n✅ Test run finished. Results are available in: {results_dir.resolve()}")


@pytest.fixture(scope="session")
def error_log(test_results_dir):
    """Initialize a session-wide error log."""
    error_log_file = test_results_dir / "error_log.json"
    return {
        "file": error_log_file,
        "log_file_path": str(error_log_file),
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "test_summary": {
            "total_combinations": 0,
            "successful": 0,
            "failed": 0,
            "errors_by_type": {},
        },
    }


def _analyze_errors(errors):
    """Analyze error patterns and categorize them."""
    error_analysis = {
        "total_errors": len(errors),
        "unique_error_types": len(set(e["error_type"] for e in errors)),
        "error_patterns": {},
        "most_common_errors": {},
        "model_specific_errors": {},
        "system_specific_errors": {},
    }

    for error in errors:
        error_type = error["error_type"]
        test_case_id = error["test_case_id"]

        parts = test_case_id.split("_")
        model = parts[0]
        system = parts[1]

        if error_type not in error_analysis["most_common_errors"]:
            error_analysis["most_common_errors"][error_type] = 0
        error_analysis["most_common_errors"][error_type] += 1

        if model not in error_analysis["model_specific_errors"]:
            error_analysis["model_specific_errors"][model] = {}
        if error_type not in error_analysis["model_specific_errors"][model]:
            error_analysis["model_specific_errors"][model][error_type] = 0
        error_analysis["model_specific_errors"][model][error_type] += 1

        if system not in error_analysis["system_specific_errors"]:
            error_analysis["system_specific_errors"][system] = {}
        if error_type not in error_analysis["system_specific_errors"][system]:
            error_analysis["system_specific_errors"][system][error_type] = 0
        error_analysis["system_specific_errors"][system][error_type] += 1

    return error_analysis


def _generate_recommendations(error_log):
    """Generate recommendations based on error analysis."""
    recommendations = []

    summary = error_log["test_summary"]
    if summary["total_combinations"] == 0:
        return ["No tests were run."]

    failure_rate = summary["failed"] / summary["total_combinations"]

    if failure_rate > 0.5:
        recommendations.append("High failure rate detected. Review database availability and system configurations.")
    elif failure_rate > 0.1:
        recommendations.append("Significant number of failures. Recommended to investigate common error patterns.")

    errors_by_type = summary["errors_by_type"]
    if errors_by_type:
        most_common = max(errors_by_type.items(), key=lambda x: x[1])
        recommendations.append(
            f"Most common error is '{most_common[0]}' ({most_common[1]} occurrences). "
            "Prioritize fixing this error type."
        )

    if summary["failed"] == 0:
        recommendations.append("All tests passed successfully!")

    return recommendations


def _generate_markdown_report(summary_report, test_results_dir):
    """Generate a human-readable markdown report."""
    report_content = f"""# E2E Sweep Test Results Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Test Execution Summary

- **Total combinations tested**: {summary_report["test_execution_summary"]["total_combinations"]}
- **Successful**: {summary_report["test_execution_summary"]["successful"]}
- **Failed**: {summary_report["test_execution_summary"]["failed"]}
"""
    if summary_report["test_execution_summary"]["total_combinations"] > 0:
        success_rate = (
            summary_report["test_execution_summary"]["successful"]
            / summary_report["test_execution_summary"]["total_combinations"]
            * 100
        )
        report_content += f"- **Success rate**: {success_rate:.1f}%\n"

    report_content += """
## Test Configuration

### Models Tested
"""
    report_content += ", ".join(summary_report["test_configuration"]["models_tested"]) or "N/A"

    report_content += """

### Systems Tested
"""
    report_content += ", ".join(summary_report["test_configuration"]["systems_tested"]) or "N/A"

    report_content += """

### GPU Configurations
"""
    report_content += ", ".join(map(str, summary_report["test_configuration"]["gpu_configs_tested"])) or "N/A"

    report_content += """

### ISL/OSL/PREFIX Combinations
"""
    report_content += (
        ", ".join(
            [
                f"({isl}, {osl}, {prefix})"
                for isl, osl, prefix in summary_report["test_configuration"]["isl_osl_prefix_combinations"]
            ]
        )
        or "N/A"
    )

    report_content += """

### TTFT/TPOT Combinations
"""
    report_content += (
        ", ".join(
            [f"({ttft}, {tpot})" for ttft, tpot in summary_report["test_configuration"]["ttft_tpot_combinations"]]
        )
        or "N/A"
    )

    report_content += """

### Backend Versions Tested
"""
    backend_versions = summary_report["test_configuration"]["versions_tested"]
    if backend_versions:
        for system, backends in backend_versions.items():
            if backends:
                report_content += f"- **{system}**:\n"
                for backend, version in backends.items():
                    report_content += f"  - {backend}: {version} (latest)\n"
    else:
        report_content += "N/A"

    report_content += """

## Error Analysis
"""
    if summary_report["error_analysis"]["total_errors"] > 0:
        report_content += f"""
### Error Summary
- Total errors: {summary_report["error_analysis"]["total_errors"]}
- Unique error types: {summary_report["error_analysis"]["unique_error_types"]}

### Most Common Errors
"""
        for error_type, count in summary_report["error_analysis"]["most_common_errors"].items():
            report_content += f"- **{error_type}**: {count} occurrences\n"
    else:
        report_content += "No errors detected in any test cases.\n"

    report_content += """
## Recommendations
"""
    for recommendation in summary_report["recommendations"]:
        report_content += f"- {recommendation}\n"

    report_content += """
## Files Generated

- `error_log.json`: Detailed error log with stack traces.
- `sweep_test_summary.json`: Machine-readable test summary.
- Individual test case results: `*_result.json` files.
- This report: `test_report.md`
"""
    report_file = test_results_dir / "test_report.md"
    with open(report_file, "w") as f:
        f.write(report_content)


@pytest.fixture(scope="session", autouse=True)
def session_summary_report(request, test_results_dir, error_log):
    """
    A session-scoped autouse fixture to generate a summary report
    after all tests have completed.
    """
    # This code runs before any tests
    yield
    # This code runs after all tests in the session are finished

    print("\n--- Generating Final Summary Report ---")

    error_log["end_time"] = datetime.now().isoformat()

    # Collect configuration details from the TestE2ESweep class

    summary_report = {
        "test_execution_summary": error_log["test_summary"],
        "test_configuration": {
            "models_tested": _SUPPORTED_MODELS,
            "systems_tested": _SUPPORTED_SYSTEMS,
            "gpu_configs_tested": _SUPPORTED_GPU_CONFIGS,
            "isl_osl_prefix_combinations": _SUPPORTED_ISL_OSL_PREFIX_COMBINATIONS,
            "ttft_tpot_combinations": _SUPPORTED_TTFT_TPOT_COMBINATIONS,
            "versions_tested": {
                system: {
                    backend: _value_or_placeholder(get_latest_database_version(system=system, backend=backend))
                    for backend in _SUPPORTED_BACKENDS
                }
                for system in _SUPPORTED_SYSTEMS
            },
        },
        "error_analysis": _analyze_errors(error_log["errors"]),
        "recommendations": _generate_recommendations(error_log),
    }

    summary_file = test_results_dir / "sweep_test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    _generate_markdown_report(summary_report, test_results_dir)

    summary = error_log["test_summary"]
    print(f"\n{'=' * 80}")
    print("E2E Test Run Summary")
    print(f"{'=' * 80}")
    print(f"Total combinations tested: {summary['total_combinations']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    if summary["total_combinations"] > 0:
        success_rate = summary["successful"] / summary["total_combinations"] * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Results saved to: {test_results_dir}")
    print(f"{'=' * 80}\n")


class TestE2ESweep:
    """End-to-end sweep tests for all configuration combinations."""

    @staticmethod
    def get_all_models():
        """Get all supported models from common.SupportedModels."""
        return [model for model in common.SupportedModels if model not in ["QWEN3_480B", "KIMI_K2"]]

    @staticmethod
    def get_all_systems():
        """Get all supported systems."""
        return ["h100_sxm", "h200_sxm", "b200_sxm", "gb200_sxm", "a100_sxm"]

    @staticmethod
    def get_all_gpu_configs():
        """Get all GPU configurations to test."""
        return [8, 512]

    @staticmethod
    def get_all_isl_osl_prefix_combinations():
        """Get all ISL/OSL/PREFIX combinations to test."""
        return [(4000, 1000, 0), (4000, 1000, 2000), (1000, 2, 0), (32, 1000, 0)]

    @staticmethod
    def get_all_ttft_tpot_combinations():
        """Get all TTFT/TPOT combinations to test."""
        return [(5000, 10), (5000, 100)]

    @staticmethod
    def get_excluded_model_system_pairs():
        """Return model/system combinations that should be skipped."""
        return {
            ("DEEPSEEK_V3", "h100_sxm"),
            ("DEEPSEEK_V3", "a100_sxm"),
        }

    @pytest.mark.parametrize(
        "model_name,system,total_gpus,isl,osl,prefix,ttft,tpot,backend",
        _VALID_PARAMETER_COMBINATIONS,
    )
    def test_e2e_configuration_sweep(
        self,
        model_name,
        system,
        total_gpus,
        isl,
        osl,
        prefix,
        ttft,
        tpot,
        backend,
        test_results_dir,
        error_log,
    ):
        """
        Test end-to-end configuration for each combination of parameters.

        This test executes the full aiconfigurator pipeline and captures any errors
        that occur during execution, logging them for analysis.
        """

        version = get_latest_database_version(system=system, backend=backend)
        if not version:
            pytest.skip(f"No latest version found for {system=}, {backend=}")

        error_log["test_summary"]["total_combinations"] += 1

        # Create unique test case identifier
        test_case_id = (
            f"{model_name}_{system}_{total_gpus}gpu_isl{isl}_osl{osl}_prefix{prefix}_ttft{int(ttft)}"
            f"_tpot{int(tpot)}_{backend}_{version}"
        )

        # Prepare test case metadata
        test_case = {
            "test_case_id": test_case_id,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "model_name": model_name,
                "system": system,
                "total_gpus": total_gpus,
                "isl": isl,
                "osl": osl,
                "prefix": prefix,
                "ttft": ttft,
                "tpot": tpot,
                "backend": backend,
                "version": version,
            },
            "status": "unknown",
            "error_details": None,
            "execution_time": None,
            "result_summary": None,
        }

        start_time = time.time()
        error_occurred = False
        error_details = None

        try:
            # Create temporary directory for this test case
            with tempfile.TemporaryDirectory() as temp_dir:
                # Mock CLI arguments
                args = MagicMock()
                args.mode = "default"
                args.model = model_name
                args.hf_id = None
                args.system = system
                args.decode_system = None
                args.total_gpus = total_gpus
                args.isl = isl
                args.osl = osl
                args.prefix = prefix
                args.ttft = float(ttft)
                args.tpot = float(tpot)
                args.request_latency = None
                args.backend = backend
                args.backend_version = version
                args.database_mode = None  # Use default (SILICON)
                args.save_dir = temp_dir
                args.debug = False
                args.generated_config_version = None
                args.config_output_dir = None
                args.deploy_output_dir = None
                args._dyn_meta = {}

                # Execute the aiconfigurator via the main CLI entrypoint
                cli_main(args)

                # For the purpose of this test, we assume success if no exception is raised.
                # A more thorough check could inspect the contents of `temp_dir`.
                test_case["status"] = "success"
                error_log["test_summary"]["successful"] += 1

        except Exception as e:
            error_occurred = True
            error_type = type(e).__name__
            error_details = {
                "error_type": error_type,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
                "test_case_id": test_case_id,
            }

            test_case["status"] = "failed"
            test_case["error_details"] = error_details

            # Update error summary
            error_log["test_summary"]["failed"] += 1
            if error_type not in error_log["test_summary"]["errors_by_type"]:
                error_log["test_summary"]["errors_by_type"][error_type] = 0
            error_log["test_summary"]["errors_by_type"][error_type] += 1

            # Add to error log
            error_log["errors"].append(error_details)

        finally:
            execution_time = time.time() - start_time
            test_case["execution_time"] = execution_time

            # Save individual test case result (optional, for debugging)
            test_case_file = test_results_dir / f"{test_case_id}_result.json"
            with open(test_case_file, "w") as f:
                json.dump(test_case, f, indent=2)

            # Update error log file (exclude the 'file' key which contains PosixPath)
            error_log_for_json = {k: v for k, v in error_log.items() if k != "file"}
            with open(error_log["file"], "w") as f:
                json.dump(error_log_for_json, f, indent=2)

        # If error occurred, fail the test with detailed information
        if error_occurred:
            pytest.fail(f"Configuration failed for {test_case_id}:\n{error_details['traceback']}")

    # The summary report generation is now handled by the session_summary_report fixture.
    # The test_generate_sweep_summary_report method has been removed.
