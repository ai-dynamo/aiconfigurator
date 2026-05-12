#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Support matrix generation and validation utilities.

This module provides the SupportMatrix class for generating and validating
the model/system/backend/version support matrix for AIConfigurator.
"""

import csv
import logging
import os
import traceback
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass

import pandas as pd
from packaging.version import Version
from tqdm import tqdm

from aiconfigurator.generator.naive import _estimate_model_weight_bytes
from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.models import _get_model_info
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

logger = logging.getLogger(__name__)

_BYTES_PER_PARAM = 2
DEFAULT_ENGINE_STEP_COMPARISON_RTOL = 0.05
DEFAULT_ENGINE_STEP_COMPARISON_ATOL = 1e-3
DEFAULT_ENGINE_STEP_FRONTIER_RTOL = 0.75
DEFAULT_ENGINE_STEP_FRONTIER_ATOL = 1e-3
_RUST_CORE_AUTOBUILD_ENV = "AICONFIGURATOR_RUST_CORE_AUTOBUILD"
_APPROXIMATE_ENGINE_STEP_COLUMNS = frozenset(
    {
        "request_rate",
        "ttft",
        "tpot",
        "request_latency",
        "seq/s",
        "seq/s/gpu",
        "tokens/s",
        "tokens/s/gpu",
        "tokens/s/user",
        "(p)seq/s/worker",
        "(d)seq/s/worker",
        "balance_score",
        "power_w",
    }
)
_FRONTIER_ENVELOPE_COLUMNS = {
    "tokens/s/user": "max",
    "tpot": "min",
    "request_latency": "min",
}


@dataclass(frozen=True)
class TestConstraints:
    total_gpus: int
    isl: int
    osl: int
    prefix: int
    ttft: float
    tpot: float


# Tiered constraints by model size (parameter count)
_SMALL = TestConstraints(total_gpus=4, isl=256, osl=256, prefix=128, ttft=1500.0, tpot=50.0)
_MEDIUM = TestConstraints(total_gpus=32, isl=256, osl=256, prefix=128, ttft=2000.0, tpot=50.0)
_LARGE = TestConstraints(total_gpus=128, isl=256, osl=256, prefix=128, ttft=2000000.0, tpot=50000.0)

_SIZE_TIERS: list[tuple[float, TestConstraints]] = [
    (10e9, _SMALL),  # < 10B params
    (100e9, _MEDIUM),  # 10B - 100B params
]
_DEFAULT_TIER = _LARGE  # > 100B params


def _get_test_constraints(model_path: str) -> TestConstraints:
    """Return the appropriate test constraints based on estimated model size."""
    weight_bytes = _estimate_model_weight_bytes(model_path)
    num_params = weight_bytes / _BYTES_PER_PARAM
    for threshold, constraints in _SIZE_TIERS:
        if num_params < threshold:
            logger.info(
                "Model %s: ~%.1fB params → %s",
                model_path,
                num_params / 1e9,
                constraints,
            )
            return constraints
    logger.info(
        "Model %s: ~%.1fB params → %s",
        model_path,
        num_params / 1e9,
        _DEFAULT_TIER,
    )
    return _DEFAULT_TIER


@contextmanager
def _rust_core_autobuild_enabled():
    previous_value = os.environ.get(_RUST_CORE_AUTOBUILD_ENV)
    os.environ[_RUST_CORE_AUTOBUILD_ENV] = "1"
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(_RUST_CORE_AUTOBUILD_ENV, None)
        else:
            os.environ[_RUST_CORE_AUTOBUILD_ENV] = previous_value


def _format_exception_for_csv(error_message: str | None) -> str | None:
    if not error_message:
        return None
    cwd = os.getcwd() + os.sep
    return error_message.replace(cwd, "").replace("\n", "\\n")


def _shorten_error(error_message: str, max_chars: int = 600) -> str:
    if len(error_message) <= max_chars:
        return error_message
    return error_message[: max_chars - 3] + "..."


def _normalize_pareto_df_for_comparison(df: pd.DataFrame, sort_columns: list[str]) -> pd.DataFrame:
    normalized = df.copy().reset_index(drop=True)
    if not sort_columns:
        return normalized
    return normalized.sort_values(
        by=sort_columns,
        kind="mergesort",
        key=lambda col: col.astype(str),
    ).reset_index(drop=True)


def _values_are_close(python_value: float, rust_value: float, *, rtol: float, atol: float) -> bool:
    denominator = max(abs(python_value), abs(rust_value), atol)
    return abs(python_value - rust_value) <= atol + rtol * denominator


def _compare_frontier_envelope(
    python_df: pd.DataFrame,
    rust_df: pd.DataFrame,
    *,
    rtol: float,
    atol: float,
) -> str | None:
    """Compare user-facing Pareto frontier envelope metrics when exact rows differ."""
    mismatches: list[str] = []
    comparable_columns = [
        col for col in _FRONTIER_ENVELOPE_COLUMNS if col in python_df.columns and col in rust_df.columns
    ]
    for column in comparable_columns:
        aggregate = _FRONTIER_ENVELOPE_COLUMNS[column]
        python_values = pd.to_numeric(python_df[column], errors="coerce").dropna()
        rust_values = pd.to_numeric(rust_df[column], errors="coerce").dropna()
        if python_values.empty or rust_values.empty:
            continue

        if aggregate == "max":
            python_value = float(python_values.max())
            rust_value = float(rust_values.max())
        else:
            python_value = float(python_values.min())
            rust_value = float(rust_values.min())

        if _values_are_close(python_value, rust_value, rtol=rtol, atol=atol):
            continue
        denominator = max(abs(python_value), abs(rust_value), atol)
        relative_diff = abs(python_value - rust_value) / denominator
        mismatches.append(
            f"{aggregate}({column}) python={python_value:.6g} rust={rust_value:.6g} rel_diff={relative_diff:.3%}"
        )

    if not comparable_columns:
        return "no common frontier envelope metrics were available"
    if mismatches:
        return f"Rust frontier envelope differs beyond relaxed tolerance rtol={rtol:g}, atol={atol:g}: " + "; ".join(
            mismatches[:5]
        )
    return None


def _compare_pareto_dfs(
    python_df: pd.DataFrame,
    rust_df: pd.DataFrame,
    *,
    rtol: float = DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
    atol: float = DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
    frontier_rtol: float = DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
    frontier_atol: float = DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
) -> str | None:
    """Return a mismatch description when Rust and Python Pareto results drift."""
    python_columns = list(python_df.columns)
    rust_columns = list(rust_df.columns)
    if python_columns != rust_columns:
        return f"Rust pareto_df columns differ from Python: python={python_columns}, rust={rust_columns}"

    if python_df.empty and rust_df.empty:
        return None

    approximate_columns = [col for col in python_columns if col in _APPROXIMATE_ENGINE_STEP_COLUMNS]
    identity_columns = [col for col in python_columns if col not in _APPROXIMATE_ENGINE_STEP_COLUMNS]

    def _compare_relaxed_frontier(reason: str) -> str | None:
        mismatch = _compare_frontier_envelope(
            python_df,
            rust_df,
            rtol=frontier_rtol,
            atol=frontier_atol,
        )
        if mismatch:
            return f"{reason}; {mismatch}"
        return None

    if len(python_df) != len(rust_df):
        return _compare_relaxed_frontier(
            f"Rust pareto_df row count differs from Python: python={len(python_df)}, rust={len(rust_df)}"
        )

    python_normalized = _normalize_pareto_df_for_comparison(python_df, identity_columns)
    rust_normalized = _normalize_pareto_df_for_comparison(rust_df, identity_columns)

    try:
        pd.testing.assert_frame_equal(
            python_normalized[identity_columns],
            rust_normalized[identity_columns],
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        return _compare_relaxed_frontier(
            f"Rust pareto_df selected different configurations: {_shorten_error(str(exc))}"
        )

    mismatches: list[str] = []
    for column in approximate_columns:
        python_values = pd.to_numeric(python_normalized[column], errors="coerce")
        rust_values = pd.to_numeric(rust_normalized[column], errors="coerce")
        absolute_diff = (python_values - rust_values).abs()
        tolerance = atol + rtol * rust_values.abs()
        both_missing = python_values.isna() & rust_values.isna()
        within_tolerance = both_missing | (absolute_diff <= tolerance)
        if within_tolerance.all():
            continue

        bad_indexes = within_tolerance[~within_tolerance].index
        first_bad_index = int(bad_indexes[0])
        denominator = max(
            abs(float(python_values.iloc[first_bad_index])), abs(float(rust_values.iloc[first_bad_index])), atol
        )
        relative_diff = float(absolute_diff.iloc[first_bad_index]) / denominator
        mismatches.append(
            f"{column}[{first_bad_index}] python={python_values.iloc[first_bad_index]} "
            f"rust={rust_values.iloc[first_bad_index]} abs_diff={absolute_diff.iloc[first_bad_index]:.6g} "
            f"rel_diff={relative_diff:.3%}"
        )

    if mismatches:
        return f"Rust pareto_df differs beyond tolerance rtol={rtol:g}, atol={atol:g}: " + "; ".join(mismatches[:5])
    return None


# Per-process SupportMatrix instance for ProcessPoolExecutor workers.
# Set in the parent before forking; children inherit it via copy-on-write.
_worker_matrix: "SupportMatrix | None" = None


def _process_combination_worker(
    combo: tuple[str, str, str, str],
) -> list[tuple[str, str, str, str, str, str, bool, str | None]]:
    """
    Run a single combination in a worker process. Uses the process-local SupportMatrix.
    Must be a module-level function for pickling by ProcessPoolExecutor.
    """
    assert _worker_matrix is not None  # this only works in linux, not in windows/macos
    model, system, backend, version = combo
    success_dict, error_dict = _worker_matrix.run_single_test(
        model=model,
        system=system,
        backend=backend,
        version=version,
        compare_engine_step_backends=_worker_matrix.compare_engine_step_backends,
        engine_step_comparison_rtol=_worker_matrix.engine_step_comparison_rtol,
        engine_step_comparison_atol=_worker_matrix.engine_step_comparison_atol,
        engine_step_frontier_rtol=_worker_matrix.engine_step_frontier_rtol,
        engine_step_frontier_atol=_worker_matrix.engine_step_frontier_atol,
    )
    architecture = _worker_matrix.get_architecture(model)
    return [
        (model, architecture, system, backend, version, mode, success_dict[mode], error_dict[mode])
        for mode in success_dict
    ]


class SupportMatrix:
    """
    Helper to generate and validate the model/system/backend/version support matrix.
    """

    def __init__(
        self,
        *,
        compare_engine_step_backends: bool = False,
        engine_step_comparison_rtol: float = DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
        engine_step_comparison_atol: float = DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
        engine_step_frontier_rtol: float = DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
        engine_step_frontier_atol: float = DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
    ):
        self.compare_engine_step_backends = compare_engine_step_backends
        self.engine_step_comparison_rtol = engine_step_comparison_rtol
        self.engine_step_comparison_atol = engine_step_comparison_atol
        self.engine_step_frontier_rtol = engine_step_frontier_rtol
        self.engine_step_frontier_atol = engine_step_frontier_atol
        logger.info("Loading models...")
        self.models: set[str] = self.get_models()
        logger.info("Found %d models", len(self.models))
        # database structure: {system: {backend: {version}}}
        logger.info("Loading perf databases...")
        self.databases: dict[str, dict[str, dict[str, str]]] = self.load_databases()
        logger.info("Databases loaded for %d systems", len(self.databases))

    def get_models(self):
        """Get the set of models to test - uses DefaultHFModels (models with cached configs)."""
        return set[str](common.DefaultHFModels)

    def get_architecture(self, huggingface_id: str) -> str:
        """Get the HuggingFace architecture for a model."""
        return _get_model_info(huggingface_id)["architecture"]

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
        combinations = list(self.__get_model_and_hardware_and_backend_combinations())
        return combinations

    @staticmethod
    def _create_task_config(
        *,
        mode: str,
        model: str,
        system: str,
        backend: str,
        version: str,
        constraints: TestConstraints,
        engine_step_backend: str | None,
    ) -> TaskConfig:
        task_config_kwargs = {
            "serving_mode": mode,
            "model_path": model,
            "system_name": system,
            "backend_name": backend,
            "backend_version": version,
            "total_gpus": constraints.total_gpus,
            "isl": constraints.isl,
            "osl": constraints.osl,
            "prefix": constraints.prefix,
            "ttft": constraints.ttft,
            "tpot": constraints.tpot,
            "engine_step_backend": engine_step_backend,
        }
        if mode == "disagg":
            task_config_kwargs["decode_system_name"] = system
        return TaskConfig(**task_config_kwargs)

    @staticmethod
    def _run_mode(
        *,
        mode: str,
        model: str,
        system: str,
        backend: str,
        version: str,
        constraints: TestConstraints,
        engine_step_backend: str | None,
    ) -> pd.DataFrame | None:
        task_config = SupportMatrix._create_task_config(
            mode=mode,
            model=model,
            system=system,
            backend=backend,
            version=version,
            constraints=constraints,
            engine_step_backend=engine_step_backend,
        )
        result = TaskRunner().run(task_config)
        return result.get("pareto_df")

    @staticmethod
    def run_single_test(
        model: str,
        system: str,
        backend: str,
        version: str,
        *,
        compare_engine_step_backends: bool = False,
        engine_step_comparison_rtol: float = DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
        engine_step_comparison_atol: float = DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
        engine_step_frontier_rtol: float = DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
        engine_step_frontier_atol: float = DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
    ) -> tuple[dict[str, bool], dict[str, str | None]]:
        """
        Run a single configuration test for both agg and disagg modes.

        Args:
            model: Model name
            system: System/hardware name
            backend: Backend name
            version: Backend version
            compare_engine_step_backends: When True, run both Python and Rust engine-step backends.
            engine_step_comparison_rtol: Relative tolerance for Python-vs-Rust Pareto metrics.
            engine_step_comparison_atol: Absolute tolerance for Python-vs-Rust Pareto metrics.
            engine_step_frontier_rtol: Loose relative tolerance when frontiers choose different rows.
            engine_step_frontier_atol: Loose absolute tolerance when frontiers choose different rows.

        Returns:
            Tuple of (dict with results, dict with error messages)
            Both dicts have keys "agg" and "disagg"
        """
        constraints = _get_test_constraints(model)
        modes_to_test = ["agg", "disagg"]
        results = {}
        error_messages = {}

        for mode in modes_to_test:
            try:
                python_pareto_df = SupportMatrix._run_mode(
                    mode=mode,
                    model=model,
                    system=system,
                    backend=backend,
                    version=version,
                    constraints=constraints,
                    engine_step_backend="python" if compare_engine_step_backends else None,
                )

                # Note that we do not use pareto_frontier_df here because for the pareto_df
                # if is not None and not empty, it means the pareto_frontier_df is also not None and not empty.
                if python_pareto_df is None or python_pareto_df.empty:
                    logger.warning(
                        "Configuration returned no results: %s, %s, %s, %s, mode=%s",
                        model,
                        system,
                        backend,
                        version,
                        mode,
                    )
                    results[mode] = False
                    error_messages[mode] = "Configuration returned no results, failed to catch traceback"
                    continue

                if compare_engine_step_backends:
                    with _rust_core_autobuild_enabled():
                        rust_pareto_df = SupportMatrix._run_mode(
                            mode=mode,
                            model=model,
                            system=system,
                            backend=backend,
                            version=version,
                            constraints=constraints,
                            engine_step_backend="rust",
                        )
                    if rust_pareto_df is None or rust_pareto_df.empty:
                        results[mode] = False
                        error_messages[mode] = "Rust engine-step backend returned no results"
                        continue

                    mismatch = _compare_pareto_dfs(
                        python_pareto_df,
                        rust_pareto_df,
                        rtol=engine_step_comparison_rtol,
                        atol=engine_step_comparison_atol,
                        frontier_rtol=engine_step_frontier_rtol,
                        frontier_atol=engine_step_frontier_atol,
                    )
                    if mismatch:
                        results[mode] = False
                        error_messages[mode] = mismatch
                        continue

                results[mode] = True
                error_messages[mode] = None

            except Exception as e:
                logger.warning(
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
                error_messages[mode] = _format_exception_for_csv(error_messages[mode])
        return results, error_messages

    def test_support_matrix(
        self, max_workers: int | None = None
    ) -> list[tuple[str, str, str, str, str, str, bool, str | None]]:
        """
        Test whether each combination is supported by AIC.
        Tests both agg and disagg modes for each combination and captures error messages.

        Runs in two phases:
        1. Parallel execution with ProcessPoolExecutor.
        2. Sequential single-process retry of every combination that failed in phase 1
           (including combos that never ran due to a broken process pool).

        Args:
            max_workers: Maximum number of worker processes for parallel execution.
                         Defaults to None, which uses os.cpu_count() or 1.

        Returns:
            List of tuples (huggingface_id, architecture, system, backend, version, mode, success, err_msg)
            Returns separate entries for agg and disagg modes
        """
        # Print configuration
        print("\n" + "=" * 80)
        print("AIConfigurator Support Matrix Test")
        print("=" * 80)
        print("Testing both agg and disagg modes for all combinations")
        if self.compare_engine_step_backends:
            print(
                "Comparing Python and Rust engine-step backends "
                f"(rtol={self.engine_step_comparison_rtol:g}, atol={self.engine_step_comparison_atol:g}, "
                f"frontier_rtol={self.engine_step_frontier_rtol:g}, frontier_atol={self.engine_step_frontier_atol:g})"
            )
        print("Tiered constraints by model size:")
        print(
            f"  <10B:      GPUs={_SMALL.total_gpus}, ISL={_SMALL.isl}, OSL={_SMALL.osl}, "
            f"PREFIX={_SMALL.prefix}, TTFT={_SMALL.ttft}ms, TPOT={_SMALL.tpot}ms"
        )
        print(
            f"  10B-100B:  GPUs={_MEDIUM.total_gpus}, ISL={_MEDIUM.isl}, OSL={_MEDIUM.osl}, "
            f"PREFIX={_MEDIUM.prefix}, TTFT={_MEDIUM.ttft}ms, TPOT={_MEDIUM.tpot}ms"
        )
        print(
            f"  >100B:     GPUs={_LARGE.total_gpus}, ISL={_LARGE.isl}, OSL={_LARGE.osl}, "
            f"PREFIX={_LARGE.prefix}, TTFT={_LARGE.ttft}ms, TPOT={_LARGE.tpot}ms"
        )
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        print(f"Max workers: {max_workers}")
        print("=" * 80 + "\n")

        combinations = self.generate_combinations()
        print(f"Total combinations to test: {len(combinations)}")
        results: list[tuple[str, str, str, str, str, str, bool, str | None]] = []
        retry_combos: set[tuple[str, str, str, str]] = set()

        global _worker_matrix
        _worker_matrix = self

        # -- Phase 1: parallel execution --
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_combination_worker, combo): combo for combo in combinations}
            pbar = tqdm(total=len(combinations), desc="Phase 1: parallel testing", unit="config")
            for future in as_completed(futures):
                combo = futures[future]
                model, system, backend, version = combo
                try:
                    results.extend(future.result())
                except BrokenExecutor:
                    logger.warning(
                        "Process pool broken while running %s/%s/%s/%s. "
                        "A worker was likely killed (OOM). "
                        "Queuing this and remaining combos for sequential retry.",
                        model,
                        system,
                        backend,
                        version,
                    )
                    retry_combos.add(combo)
                    for remaining in futures:
                        if remaining is not future and not remaining.done():
                            remaining.cancel()
                            retry_combos.add(futures[remaining])
                    pbar.update(len(combinations) - pbar.n)
                    break
                except Exception:
                    logger.exception(
                        "Unexpected error retrieving result for %s/%s/%s/%s",
                        model,
                        system,
                        backend,
                        version,
                    )
                    retry_combos.add(combo)
                pbar.update(1)
            pbar.close()

        # Also collect combos whose Phase 1 results had any failure
        for model, _arch, system, backend, version, _mode, success, _err in results:
            if not success:
                retry_combos.add((model, system, backend, version))

        # -- Phase 2: sequential single-process retry of all failures --
        if retry_combos:
            results = [r for r in results if (r[0], r[2], r[3], r[4]) not in retry_combos]

            print(f"\n{'=' * 80}")
            print(f"Phase 2: retrying {len(retry_combos)} failed combination(s) sequentially")
            print(f"{'=' * 80}\n")

            for combo in tqdm(sorted(retry_combos), desc="Phase 2: sequential retry", unit="config"):
                model, system, backend, version = combo
                try:
                    success_dict, error_dict = self.run_single_test(
                        model=model,
                        system=system,
                        backend=backend,
                        version=version,
                        compare_engine_step_backends=self.compare_engine_step_backends,
                        engine_step_comparison_rtol=self.engine_step_comparison_rtol,
                        engine_step_comparison_atol=self.engine_step_comparison_atol,
                        engine_step_frontier_rtol=self.engine_step_frontier_rtol,
                        engine_step_frontier_atol=self.engine_step_frontier_atol,
                    )
                    architecture = self.get_architecture(model)
                    for mode in success_dict:
                        results.append(
                            (model, architecture, system, backend, version, mode, success_dict[mode], error_dict[mode])
                        )
                except Exception:
                    logger.exception(
                        "Sequential retry also failed for %s/%s/%s/%s",
                        model,
                        system,
                        backend,
                        version,
                    )
                    architecture = self.get_architecture(model)
                    for mode in ("agg", "disagg"):
                        results.append(
                            (
                                model,
                                architecture,
                                system,
                                backend,
                                version,
                                mode,
                                False,
                                traceback.format_exc().replace("\n", "\\n"),
                            )
                        )

        # Sort results by (huggingface_id, architecture, system, backend, version, mode)
        results.sort(key=lambda x: (x[0], x[1], x[2], x[3], Version(x[4]), x[5]))

        # Print results summary
        self._print_results_summary(results)

        return results

    def _print_results_summary(self, results: list[tuple[str, str, str, str, str, str, bool, str | None]]) -> None:
        """Print summary of test results."""
        total_tests = len(results)
        passed = sum(1 for _, _, _, _, _, _, success, _ in results if success)
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

        for huggingface_id, architecture, system, backend, version, mode, success, _ in results:
            config = (huggingface_id, architecture, system, backend, version, mode)
            if success:
                passed_configs.append(config)
            else:
                failed_configs.append(config)

        # Print passed configurations
        if passed_configs:
            print(f"\n✓ Passed Configurations ({len(passed_configs)}):")
            for huggingface_id, architecture, system, backend, version, mode in sorted(passed_configs):
                print(f"  • {huggingface_id} ({architecture}) on {system} with {backend} v{version} ({mode})")

        # Print failed configurations
        if failed_configs:
            print(f"\n✗ Failed Configurations ({len(failed_configs)}):")
            for huggingface_id, architecture, system, backend, version, mode in sorted(failed_configs):
                print(f"  • {huggingface_id} ({architecture}) on {system} with {backend} v{version} ({mode})")

    def save_results_to_csv(
        self, results: list[tuple[str, str, str, str, str, str, bool, str | None]], output_file: str
    ) -> None:
        """
        Save test results to a CSV file.

        Args:
            results: List of tuples (huggingface_id, architecture, system, backend, version, mode, success, err_msg)
            output_file: Path to the output CSV file
        """

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["HuggingFaceID", "Architecture", "System", "Backend", "Version", "Mode", "Status", "ErrMsg"]
            writer.writerow(header)
            for huggingface_id, architecture, system, backend, version, mode, success, err_msg in results:
                status = "PASS" if success else "FAIL"
                writer.writerow([huggingface_id, architecture, system, backend, version, mode, status, err_msg or ""])
        print(f"\nResults saved to: {output_file}")
