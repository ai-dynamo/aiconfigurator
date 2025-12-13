# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import os
import traceback

from packaging.version import Version
from tqdm import tqdm

from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

logger = logging.getLogger(__name__)

# Test configuration constants
TOTAL_GPUS = 128
ISL = 4000
OSL = 500
PREFIX = 500
TTFT = 2000.0
TPOT = 50.0


class SupportMatrix:
    def __init__(self):
        self.models: set[str] = self.get_models()
        # database structure: {system: {backend: {version}}}
        self.databases: dict[str, dict[str, dict[str, str]]] = self.load_databases()

    def get_models(self):
        return set[str](common.SupportedModels.keys())

    def get_systems(self):
        return set(common.SupportedSystems)

    def get_backends(self):
        return set(x.value for x in common.BackendName)

    def load_databases(self):
        return perf_database.get_all_databases()

    def __get_hardware_and_backend_combinations(self) -> list[tuple[str, str, str]]:
        """
        Iterate over all combinations of hardware, and inference backend, version.
        """
        for hardware in self.get_systems():
            for backend in self.get_backends():
                for version in self.databases[hardware][backend]:
                    yield hardware, backend, version

    def __get_model_and_hardware_and_backend_combinations(self) -> list[tuple[str, str, str, str]]:
        """
        Iterate over all combinations of models, hardware, and inference backend, version.
        """
        for hardware, backend, version in self.__get_hardware_and_backend_combinations():
            for model in self.models:
                yield model, hardware, backend, version

    def generate_combinations(self):
        """
        Generate all combinations of models, hardware, and inference backend, version.
        """
        # get all combinations of hardware, and inference backend, version
        combinations = list(self.__get_model_and_hardware_and_backend_combinations())
        return combinations

    def run_single_test(
        self,
        model: str,
        system: str,
        backend: str,
        version: str,
    ) -> tuple[dict[str, bool], dict[str, str | None]]:
        """
        Run a single configuration test for both agg and disagg modes.

        Args:
            model: Model name
            system: System/hardware name
            backend: Backend name
            version: Backend version

        Returns:
            Tuple of (dict with results, dict with error messages)
            Both dicts have keys "agg" and "disagg"
        """
        modes_to_test = ["agg", "disagg"]
        results = {}
        error_messages = {}

        for mode in modes_to_test:
            try:
                # Create TaskConfig for the test
                task_config_kwargs = {
                    "serving_mode": mode,
                    "model_name": model,
                    "system_name": system,
                    "backend_name": backend,
                    "backend_version": version,
                    "total_gpus": TOTAL_GPUS,
                    "isl": ISL,
                    "osl": OSL,
                    "prefix": PREFIX,
                    "ttft": TTFT,
                    "tpot": TPOT,
                }

                # For disagg mode, set decode_system_name
                if mode == "disagg":
                    task_config_kwargs["decode_system_name"] = system

                task_config = TaskConfig(**task_config_kwargs)

                # Run the configuration
                runner = TaskRunner()
                result = runner.run(task_config)

                # Check if we got valid results
                # Note that we do not use pareto_frontier_df here because for the pareto_df
                # if is not None and not empty, it means the pareto_frontier_df is also not None and not empty.
                pareto_df = result.get("pareto_df")
                if pareto_df is not None and not pareto_df.empty:
                    results[mode] = True
                    error_messages[mode] = None
                else:  # pragma: no cover
                    logger.debug(
                        "Configuration returned no results: %s, %s, %s, %s, mode=%s",
                        model,
                        system,
                        backend,
                        version,
                        mode,
                    )
                    results[mode] = False
                    error_messages[mode] = "Configuration returned no results, failed to catch traceback"

            except Exception as e:
                logger.debug(
                    "Configuration failed: %s, %s, %s, %s, mode=%s - Error: %s",
                    model,
                    system,
                    backend,
                    version,
                    mode,
                    str(e),
                )
                results[mode] = False
                error_messages[mode] = traceback.format_exc()
            finally:
                # format error messages to one line with "\n" as separator
                # remove absolute path prefix to avoid PII exposure
                if error_messages[mode]:
                    cwd = os.getcwd() + os.sep
                    error_messages[mode] = error_messages[mode].replace(cwd, "")
                    error_messages[mode] = error_messages[mode].replace("\n", "\\n")
                else:
                    error_messages[mode] = None
        return results, error_messages

    def test_support_matrix(self) -> list[tuple[str, str, str, str, str, bool, str | None]]:
        """
        Test whether each combination is supported by AIC.
        Tests both agg and disagg modes for each combination and captures error messages.

        Returns:
            List of tuples (model, system, backend, version, mode, success, err_msg)
            Returns separate entries for agg and disagg modes
        """
        # Print configuration
        print("\n" + "=" * 80)
        print("AIConfigurator Support Matrix Test")
        print("=" * 80)
        print("Testing both agg and disagg modes for all combinations")
        print(f"Total GPUs: {TOTAL_GPUS}")
        print(f"Input Sequence Length (ISL): {ISL}")
        print(f"Output Sequence Length (OSL): {OSL}")
        print(f"Prefix: {PREFIX}")
        print(f"Target TTFT: {TTFT}ms")
        print(f"Target TPOT: {TPOT}ms")
        print("=" * 80 + "\n")

        combinations = self.generate_combinations()
        results = []

        # Use tqdm for progress tracking
        for model, system, backend, version in tqdm(
            combinations,
            desc="Testing support matrix",
            unit="config",
        ):
            success_dict, error_dict = self.run_single_test(
                model=model,
                system=system,
                backend=backend,
                version=version,
            )

            # Add separate entries for agg and disagg modes
            for mode in success_dict:
                results.append((model, system, backend, version, mode, success_dict[mode], error_dict[mode]))

        # Sort results by (model, system, backend, version, mode)
        results.sort(key=lambda x: (x[0], x[1], x[2], Version(x[3]), x[4]))

        # Print results summary
        self._print_results_summary(results)

        return results

    def _print_results_summary(self, results: list[tuple[str, str, str, str, str, bool, str | None]]) -> None:
        """Print summary of test results."""
        total_tests = len(results)
        passed = sum(1 for _, _, _, _, _, success, _ in results if success)
        failed = total_tests - passed

        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        print(f"Total configurations tested: {total_tests}")
        print(f"✓ Passed: {passed} ({100 * passed / total_tests:.1f}%)")
        print(f"✗ Failed: {failed} ({100 * failed / total_tests:.1f}%)")
        print("=" * 80)

        # Group results by status
        passed_configs = []
        failed_configs = []

        for model, system, backend, version, mode, success, err_msg in results:
            config = (model, system, backend, version, mode)
            if success:
                passed_configs.append(config)
            else:
                failed_configs.append(config)

        # Print passed configurations
        if passed_configs:
            print(f"\n✓ Passed Configurations ({len(passed_configs)}):")
            for model, system, backend, version, mode in sorted(passed_configs):
                print(f"  • {model} on {system} with {backend} v{version} ({mode})")

        # Print failed configurations
        if failed_configs:
            print(f"\n✗ Failed Configurations ({len(failed_configs)}):")
            for model, system, backend, version, mode in sorted(failed_configs):
                print(f"  • {model} on {system} with {backend} v{version} ({mode})")

    def save_results_to_csv(
        self, results: list[tuple[str, str, str, str, str, bool, str | None]], output_file: str
    ) -> None:
        """
        Save test results to a CSV file.

        Args:
            results: List of tuples (model, system, backend, version, mode, success, err_msg)
            output_file: Path to the output CSV file
        """

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "System", "Backend", "Version", "Mode", "Status", "ErrMsg"])
            for model, system, backend, version, mode, success, err_msg in results:
                status = "PASS" if success else "FAIL"
                writer.writerow([model, system, backend, version, mode, status, err_msg or ""])
        print(f"\nResults saved to: {output_file}")
