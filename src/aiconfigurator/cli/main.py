# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any

import pandas as pd
import yaml

from aiconfigurator import __version__
from aiconfigurator.cli.report_and_save import log_final_summary, save_results
from aiconfigurator.generator.api import (
    add_generator_override_arguments,
    generate_naive_config,
    generator_cli_helper,
)
from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.pareto_analysis import (
    get_best_configs_under_request_latency_constraint,
    get_best_configs_under_tpot_constraint,
    get_pareto_front,
)
from aiconfigurator.sdk.perf_database import get_latest_database_version
from aiconfigurator.sdk.task import TaskConfig, TaskRunner
from aiconfigurator.sdk.utils import get_model_config_from_model_path

logger = logging.getLogger(__name__)


def _build_common_cli_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the results.")
    common_parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    common_parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top configurations to save for each mode (agg/disagg). Default: 5.",
    )
    add_generator_override_arguments(common_parser)
    return common_parser


def _validate_model_path(model_path: str) -> str:
    """
    Validate model_path which can be:
    1. A HuggingFace model path (e.g., "Qwen/Qwen3-32B")
    2. A local path containing a config.json file
    """
    import os

    # Check if it's a local path with config.json
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            return model_path
        raise argparse.ArgumentTypeError(f"Directory '{model_path}' does not contain a config.json file.")

    # Check if it's a file path to config.json directly
    if os.path.isfile(model_path) and model_path.endswith("config.json"):
        return os.path.dirname(model_path) or model_path

    # Otherwise treat as HuggingFace model path
    if model_path in common.DefaultHFModels:
        return model_path

    # Try to fetch from HuggingFace
    try:
        get_model_config_from_model_path(model_path)
        return model_path
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"'{model_path}' is not a valid HuggingFace model path or local path with config.json. Error: {e}"
        ) from e


def _add_default_mode_arguments(parser):
    parser.add_argument(
        "--model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument("--total_gpus", type=int, required=True, help="Total GPUs for deployment.")
    parser.add_argument(
        "--system", choices=common.SupportedSystems, type=str, required=True, help="Default system name (GPU type)."
    )
    parser.add_argument(
        "--decode_system",
        type=str,
        default=None,
        help="System name for disagg decode workers. Defaults to --system if omitted.",
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in common.BackendName],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name. Use 'any' to check all backends (trtllm/sglang/vllm) and find the "
        "best performing configuration. For agg mode, checks 3 backends. For disagg mode, "
        "checks all 9 prefill/decode backend combinations (3x3). Total: 12 combinations.",
    )
    parser.add_argument(
        "--backend_version",
        type=str,
        default=None,
        help="Backend database version. Default is latest",
    )
    parser.add_argument(
        "--database_mode",
        choices=[mode.name for mode in common.DatabaseMode if mode != common.DatabaseMode.SOL_FULL],
        type=str,
        default=common.DatabaseMode.SILICON.name,
        help="Database mode for performance estimation. Options: SILICON (default, uses silicon data), "
        "HYBRID (uses silicon data when available, otherwise SOL+empirical factor), "
        "EMPIRICAL (SOL+empirical factor), SOL (provide SOL time only).",
    )
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    parser.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    parser.add_argument("--ttft", type=float, default=2000.0, help="Time to first token in ms.")
    parser.add_argument("--tpot", type=float, default=30.0, help="Time per output token in ms.")
    parser.add_argument(
        "--request_latency",
        type=float,
        default=None,
        help="Optional end-to-end request latency target (ms). Enables request-latency optimization mode.",
    )
    parser.add_argument("--prefix", type=int, default=0, help="Prefix cache length. Default to 0.")


def _add_experiments_mode_arguments(parser):
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to a YAML file containing experiment definitions.",
    )


def _add_generate_mode_arguments(parser):
    """Add arguments for the generate mode (naive config generation)."""
    parser.add_argument(
        "--model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument(
        "--total_gpus",
        type=int,
        required=True,
        help="Total GPUs for deployment.",
    )
    parser.add_argument(
        "--system",
        choices=common.SupportedSystems,
        type=str,
        required=True,
        help="System name (GPU type).",
    )
    parser.add_argument(
        "--backend",
        choices=[b.value for b in common.CONCRETE_BACKENDS],  # 'any' not supported for generate mode
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name (default: trtllm). Note: 'any' is not supported in generate mode.",
    )


def configure_parser(parser):
    common_cli_parser = _build_common_cli_parser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    default_parser = subparsers.add_parser(
        "default", parents=[common_cli_parser], help="Run the default agg vs disagg comparison."
    )
    _add_default_mode_arguments(default_parser)

    help_text = "Run one or more experiments defined in a YAML file. Example: example.yaml"
    # an example yaml for demonstration
    example_yaml_path = os.path.join(os.path.dirname(__file__), "example.yaml")
    with open(example_yaml_path) as f:
        example_yaml = yaml.safe_load(f)
    description = help_text + "\n\nExample:\n" + json.dumps(example_yaml, indent=2)

    experiments_parser = subparsers.add_parser(
        "exp",
        parents=[common_cli_parser],
        help=help_text,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_experiments_mode_arguments(experiments_parser)

    # Generate mode - naive config without sweeping
    generate_parser = subparsers.add_parser(
        "generate",
        parents=[common_cli_parser],
        help="Generate naive agg config without SLA optimization (no sweeping).",
        description=(
            "Generate a working agg configuration without running the parameter sweep. "
            "Calculates the smallest TP that fits the model in memory "
            "(TP * VRAM/GPU > 1.5 * model_weight). No SLA optimization is performed."
        ),
    )
    _add_generate_mode_arguments(generate_parser)


def _get_backend_data_path(system_name: str, backend_name: str, backend_version: str) -> str | None:
    systems_dir = perf_database.get_system_config_path()
    system_yaml = os.path.join(systems_dir, f"{system_name}.yaml")
    if not os.path.exists(system_yaml):
        return None
    with open(system_yaml, encoding="utf-8") as fh:
        system_spec = yaml.safe_load(fh) or {}
    data_dir = system_spec.get("data_dir")
    if not data_dir:
        return None
    return os.path.join(systems_dir, data_dir, backend_name, backend_version)


def _ensure_backend_version_available(system_name: str, backend_name: str, backend_version: str) -> None:
    supported = perf_database.get_supported_databases()
    versions = supported.get(system_name, {}).get(backend_name, [])
    if backend_version in versions:
        return

    logger.error(
        "No perf database for system=%s backend=%s version=%s.",
        system_name,
        backend_name,
        backend_version,
    )
    data_path = _get_backend_data_path(system_name, backend_name, backend_version)
    if data_path:
        logger.error("Searched: %s", data_path)
    if versions:
        logger.error("Available versions: %s", ", ".join(versions))
    else:
        logger.error("Available versions: none")
    logger.error("Fix: remove --backend_version to use the latest, or provide one of the available versions.")
    raise SystemExit(1)


def _get_backends_to_check(backend: str) -> list[str]:
    """
    Get the list of concrete backends to check based on the backend parameter.
    If 'any', returns all concrete backends; otherwise returns a single-element list.
    """
    if backend == common.BackendName.any.value:
        return [b.value for b in common.CONCRETE_BACKENDS]
    return [backend]


def build_default_task_configs(
    model_path: str,
    total_gpus: int,
    system: str,
    decode_system: str | None = None,
    backend: str = "trtllm",
    backend_version: str | None = None,
    database_mode: str = "SILICON",
    isl: int = 4000,
    osl: int = 1000,
    ttft: float = 2000.0,
    tpot: float = 30.0,
    request_latency: float | None = None,
    prefix: int = 0,
) -> dict[str, TaskConfig]:
    """Build agg and disagg task configs for default mode comparison.

    Args:
        model_path: HuggingFace model path or local path.
        total_gpus: Total number of GPUs for deployment.
        system: System name (GPU type).
        decode_system: System for disagg decode workers. Defaults to `system`.
        backend: Backend name ('trtllm', 'sglang', 'vllm', 'any').
            If 'any', creates task configs for all backend combinations:
            - For agg: 3 configs (one per backend)
            - For disagg: 9 configs (3 prefill backends x 3 decode backends)
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation.
        isl: Input sequence length.
        osl: Output sequence length.
        ttft: Time to first token target in ms.
        tpot: Time per output token target in ms.
        request_latency: Optional end-to-end request latency target (ms).
        prefix: Prefix cache length.

    Returns:
        Dict with task configs. Keys are 'agg' and 'disagg' for single backend,
        or 'agg_{backend}' and 'disagg_{prefill}_{decode}' for 'any' backend.
    """
    decode_system = decode_system or system

    # Get list of backends to check
    backends_to_check = _get_backends_to_check(backend)
    is_any_backend = backend == common.BackendName.any.value

    # When using 'any' backend, ignore backend_version since each backend has different versions
    # Each backend will use its own latest version
    if is_any_backend:
        if backend_version:
            logger.warning(
                "Ignoring --backend_version=%s when using --backend=any. Each backend will use its own latest version.",
                backend_version,
            )
        effective_backend_version = None
    else:
        effective_backend_version = backend_version
        # Validate backend version for single backend
        if effective_backend_version:
            for b in backends_to_check:
                _ensure_backend_version_available(system, b, effective_backend_version)
                if decode_system != system:
                    _ensure_backend_version_available(decode_system, b, effective_backend_version)

    task_configs: dict[str, TaskConfig] = {}

    # Build agg task configs
    for agg_backend in backends_to_check:
        common_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "system_name": system,
            "backend_name": agg_backend,
            "backend_version": effective_backend_version,
            "total_gpus": total_gpus,
            "isl": isl,
            "osl": osl,
            "ttft": ttft,
            "tpot": tpot,
            "request_latency": request_latency,
            "prefix": prefix,
            "database_mode": database_mode,
        }
        agg_task = TaskConfig(serving_mode="agg", **common_kwargs)
        # Use descriptive name when checking multiple backends
        if is_any_backend:
            task_configs[f"agg_{agg_backend}"] = agg_task
        else:
            task_configs["agg"] = agg_task

    # Build disagg task configs
    # For 'any' backend, check all prefill/decode combinations (3x3 = 9)
    for prefill_backend in backends_to_check:
        for decode_backend in backends_to_check:
            disagg_kwargs: dict[str, Any] = {
                "model_path": model_path,
                "system_name": system,
                "backend_name": prefill_backend,
                "backend_version": effective_backend_version,
                "total_gpus": total_gpus,
                "isl": isl,
                "osl": osl,
                "ttft": ttft,
                "tpot": tpot,
                "request_latency": request_latency,
                "prefix": prefix,
                "database_mode": database_mode,
                "decode_system_name": decode_system,
            }

            # For disagg mode, we can specify different backends for prefill and decode
            # The TaskConfig uses backend_name for prefill, and we need to override decode
            # This requires YAML config to set decode_worker_config.backend_name and backend_version
            if prefill_backend != decode_backend:
                # Get the latest version for the decode backend
                decode_backend_version = get_latest_database_version(system=decode_system, backend=decode_backend)
                disagg_kwargs["yaml_config"] = {
                    "mode": "patch",
                    "config": {
                        "decode_worker_config": {
                            "backend_name": decode_backend,
                            "backend_version": decode_backend_version,
                        }
                    },
                }

            disagg_task = TaskConfig(serving_mode="disagg", **disagg_kwargs)

            # Use descriptive name when checking multiple backends
            if is_any_backend:
                task_configs[f"disagg_{prefill_backend}_{decode_backend}"] = disagg_task
            else:
                task_configs["disagg"] = disagg_task
                break  # Only one disagg config for single backend
        if not is_any_backend:
            break  # Only one disagg config for single backend

    return task_configs


_EXPERIMENT_RESERVED_KEYS = {
    "mode",
    "serving_mode",
    "model_path",
    "system_name",
    "decode_system_name",
    "backend_name",
    "backend_version",
    "profiles",
    "isl",
    "osl",
    "ttft",
    "tpot",
    "request_latency",
    "enable_wideep",
    "total_gpus",
    "use_specific_quant_mode",
    "database_mode",
}


def _build_yaml_config(exp_config: dict, config_section: dict) -> dict | None:
    if not config_section:
        config_section = {
            key: copy.deepcopy(value) for key, value in exp_config.items() if key not in _EXPERIMENT_RESERVED_KEYS
        }
    if not config_section:
        return None

    yaml_config = {
        "mode": exp_config.get("mode", "patch"),
        "config": config_section,
    }

    return yaml_config


def build_experiment_task_configs(
    yaml_path: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, TaskConfig]:
    """Build task configs from YAML file or config dict.

    Args:
        yaml_path: Path to a YAML file containing experiment definitions.
        config: Dict containing experiment definitions (alternative to yaml_path).
            Keys are experiment names, values are experiment configs.

    Returns:
        Dict mapping experiment names to TaskConfig objects.

    Raises:
        ValueError: If both or neither of yaml_path/config provided, or YAML load fails.
        TypeError: If experiment data is not a dict.
    """
    if yaml_path is not None and config is not None:
        raise ValueError("Provide either yaml_path or config, not both.")
    if yaml_path is None and config is None:
        raise ValueError("Must provide either yaml_path or config.")

    # Load experiment data
    if yaml_path is not None:
        try:
            with open(yaml_path, encoding="utf-8") as fh:
                experiment_data = yaml.safe_load(fh) or {}
        except Exception as exc:
            raise ValueError(f"Error loading experiment YAML file '{yaml_path}'") from exc
    else:
        experiment_data = config

    if not isinstance(experiment_data, dict):
        raise TypeError("Experiment data must be a mapping (dict).")

    order = experiment_data.get("exps")
    if isinstance(order, list):
        experiment_names = [name for name in order if name in experiment_data]
    else:
        experiment_names = [name for name in experiment_data if name != "exps"]

    task_configs: dict[str, TaskConfig] = {}

    for exp_name in experiment_names:
        exp_config = experiment_data[exp_name]
        if not isinstance(exp_config, dict):
            logger.warning("Skipping experiment '%s': configuration is not a mapping.", exp_name)
            continue

        config_section = exp_config.get("config")
        if not isinstance(config_section, dict):
            config_section = {}
        else:
            config_section = copy.deepcopy(config_section)

        serving_mode = exp_config.get("serving_mode")
        model_path = exp_config.get("model_path")
        if serving_mode not in {"agg", "disagg"} or not model_path:
            logger.warning("Skipping experiment '%s': missing serving_mode or model_path.", exp_name)
            continue

        # system
        if serving_mode == "agg":
            inferred_system = exp_config.get("system_name")
            inferred_decode_system = None
        else:
            inferred_system = exp_config.get("system_name")
            inferred_decode_system = exp_config.get("decode_system_name") or inferred_system
        system_name = inferred_system
        if not system_name:
            logger.warning(
                "Skipping experiment '%s': no system name provided "
                "(provide system_name at the top level or inside worker config).",
                exp_name,
            )
            continue

        # backend, default to trtllm
        backend_name = exp_config.get("backend_name") or common.BackendName.trtllm.value
        backend_version = exp_config.get("backend_version")

        total_gpus = exp_config.get("total_gpus")
        if total_gpus is None:
            logger.warning("Skipping experiment '%s': total_gpus not provided.", exp_name)
            continue

        # Get list of backends to check (expand 'any' into all concrete backends)
        backends_to_check = _get_backends_to_check(backend_name)
        is_any_backend = backend_name == common.BackendName.any.value

        # When using 'any' backend, ignore backend_version since each backend has different versions
        # Each backend will use its own latest version
        if is_any_backend:
            if backend_version is not None:
                logger.warning(
                    "Ignoring backend_version=%s in experiment '%s' when using backend_name=any. "
                    "Each backend will use its own latest version.",
                    backend_version,
                    exp_name,
                )
            effective_backend_version = None
        else:
            effective_backend_version = backend_version
            # Validate backend version for single backend
            if effective_backend_version is not None:
                for b in backends_to_check:
                    _ensure_backend_version_available(system_name, b, effective_backend_version)
                    if serving_mode == "disagg" and inferred_decode_system and inferred_decode_system != system_name:
                        _ensure_backend_version_available(inferred_decode_system, b, effective_backend_version)

        # Build base task kwargs (without backend-specific settings)
        base_task_kwargs: dict[str, Any] = {
            "serving_mode": serving_mode,
            "model_path": model_path,
            "system_name": system_name,
            "total_gpus": total_gpus,
            "profiles": exp_config.get("profiles", []),
        }

        if effective_backend_version is not None:
            base_task_kwargs["backend_version"] = effective_backend_version

        if serving_mode == "disagg":
            base_task_kwargs["decode_system_name"] = inferred_decode_system or system_name

        # Per-experiment overrides for runtime numeric parameters if provided at top level
        for numeric_key in ("isl", "osl", "ttft", "tpot", "request_latency"):
            if numeric_key in exp_config:
                base_task_kwargs[numeric_key] = exp_config[numeric_key]

        if "enable_wideep" in exp_config:
            base_task_kwargs["enable_wideep"] = exp_config["enable_wideep"]
        if "use_specific_quant_mode" in exp_config:
            base_task_kwargs["use_specific_quant_mode"] = exp_config["use_specific_quant_mode"]
        if "database_mode" in exp_config:
            base_task_kwargs["database_mode"] = exp_config["database_mode"]

        base_yaml_config = _build_yaml_config(exp_config, config_section)

        # Create task configs for each backend combination
        if serving_mode == "agg":
            # For agg mode: create one config per backend
            for agg_backend in backends_to_check:
                task_kwargs = copy.deepcopy(base_task_kwargs)
                task_kwargs["backend_name"] = agg_backend
                if base_yaml_config:
                    task_kwargs["yaml_config"] = copy.deepcopy(base_yaml_config)

                config_name = f"{exp_name}_{agg_backend}" if is_any_backend else exp_name
                try:
                    task_configs[config_name] = TaskConfig(**task_kwargs)
                except Exception:
                    logger.exception("Failed to build TaskConfig for experiment '%s'", config_name)
        else:
            # For disagg mode: create configs for all prefill/decode backend combinations
            for prefill_backend in backends_to_check:
                for decode_backend in backends_to_check:
                    task_kwargs = copy.deepcopy(base_task_kwargs)
                    task_kwargs["backend_name"] = prefill_backend

                    # Merge base YAML config with decode backend override if needed
                    if prefill_backend != decode_backend:
                        # Get the latest version for the decode backend
                        decode_system_for_version = inferred_decode_system or system_name
                        decode_backend_version = get_latest_database_version(
                            system=decode_system_for_version, backend=decode_backend
                        )
                        if base_yaml_config:
                            merged_yaml = copy.deepcopy(base_yaml_config)
                            if "config" not in merged_yaml:
                                merged_yaml["config"] = {}
                            if "decode_worker_config" not in merged_yaml["config"]:
                                merged_yaml["config"]["decode_worker_config"] = {}
                            merged_yaml["config"]["decode_worker_config"]["backend_name"] = decode_backend
                            merged_yaml["config"]["decode_worker_config"]["backend_version"] = decode_backend_version
                            task_kwargs["yaml_config"] = merged_yaml
                        else:
                            task_kwargs["yaml_config"] = {
                                "mode": "patch",
                                "config": {
                                    "decode_worker_config": {
                                        "backend_name": decode_backend,
                                        "backend_version": decode_backend_version,
                                    }
                                },
                            }
                    elif base_yaml_config:
                        task_kwargs["yaml_config"] = copy.deepcopy(base_yaml_config)

                    config_name = f"{exp_name}_{prefill_backend}_{decode_backend}" if is_any_backend else exp_name
                    try:
                        task_configs[config_name] = TaskConfig(**task_kwargs)
                    except Exception:
                        logger.exception("Failed to build TaskConfig for experiment '%s'", config_name)

                    if not is_any_backend:
                        break  # Only one config for single backend
                if not is_any_backend:
                    break  # Only one config for single backend

    return task_configs


def _is_any_backend_mode(task_configs: dict[str, TaskConfig]) -> bool:
    """Check if we're running in 'any' backend mode (multiple agg/disagg backend combinations)."""
    # If we have multiple configs starting with agg_ or disagg_, we're in 'any' mode
    agg_count = sum(1 for name in task_configs if name.startswith("agg_"))
    disagg_count = sum(1 for name in task_configs if name.startswith("disagg_") and "_" in name[7:])
    return agg_count > 1 or disagg_count > 1


def _aggregate_results_by_mode(
    results: dict[str, dict],
    task_configs: dict[str, TaskConfig],
) -> tuple[dict[str, pd.DataFrame], dict[str, TaskConfig]]:
    """
    Aggregate pareto results by serving mode (agg vs disagg) for 'any' backend mode.

    Returns:
        Tuple of (aggregated_results, representative_task_configs)
        - aggregated_results: {"agg": combined_pareto_df, "disagg": combined_pareto_df}
        - representative_task_configs: {"agg": first_agg_task, "disagg": first_disagg_task}
    """
    agg_dfs = []
    disagg_dfs = []
    agg_task = None
    disagg_task = None

    for name, task_result in results.items():
        pareto_df = task_result["pareto_df"]
        if pareto_df is None or pareto_df.empty:
            continue

        task_config = task_configs[name]
        serving_mode = task_config.serving_mode

        # Add source experiment column to track which backend combination this came from
        pareto_df = pareto_df.copy()
        pareto_df["source_experiment"] = name

        # Add backend and version info for config generation
        if serving_mode == "agg":
            pareto_df["backend"] = task_config.config.worker_config.backend_name
            pareto_df["backend_version"] = task_config.config.worker_config.backend_version
            agg_dfs.append(pareto_df)
            if agg_task is None:
                agg_task = task_config
        else:
            # For disagg, track both prefill and decode backends
            pareto_df["(p)backend"] = task_config.config.prefill_worker_config.backend_name
            pareto_df["(p)backend_version"] = task_config.config.prefill_worker_config.backend_version
            pareto_df["(d)backend"] = task_config.config.decode_worker_config.backend_name
            pareto_df["(d)backend_version"] = task_config.config.decode_worker_config.backend_version
            disagg_dfs.append(pareto_df)
            if disagg_task is None:
                disagg_task = task_config

    aggregated_results = {}
    representative_configs = {}

    if agg_dfs:
        aggregated_results["agg"] = {"pareto_df": pd.concat(agg_dfs, ignore_index=True)}
        representative_configs["agg"] = agg_task

    if disagg_dfs:
        aggregated_results["disagg"] = {"pareto_df": pd.concat(disagg_dfs, ignore_index=True)}
        representative_configs["disagg"] = disagg_task

    return aggregated_results, representative_configs


def _execute_task_configs(
    task_configs: dict[str, TaskConfig],
    mode: str,
    top_n: int = 5,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, float]]:
    """Execute the task configs and return the chosen experiment, best configs, results, and best
    throughputs."""
    results: dict[str, dict[str, pd.DataFrame]] = {}
    start_time = time.time()
    runner = TaskRunner()

    # TODO, can run in parallel
    for exp_name, task_config in task_configs.items():
        try:
            logger.info("Starting experiment: %s", exp_name)
            logger.info("Task config: %s", task_config.pretty())
            task_result = runner.run(task_config)
            pareto_df = task_result["pareto_df"]
            if pareto_df is not None and not pareto_df.empty:
                results[exp_name] = task_result
                logger.info("Experiment %s completed with %d results.", exp_name, len(pareto_df))
            else:
                logger.warning(
                    "Experiment %s returned no results. The TTFT and TPOT constraints may need to be relaxed.", exp_name
                )
        except Exception:
            logger.exception("Error running experiment %s", exp_name)

    if len(results) < 1:
        logger.error("No successful experiment runs to compare.")
        raise SystemExit(1)

    # Check if we should aggregate by serving mode (for 'any' backend)
    is_any_mode = _is_any_backend_mode(task_configs)

    if is_any_mode:
        # Aggregate results by serving mode (agg vs disagg)
        aggregated_results, representative_configs = _aggregate_results_by_mode(results, task_configs)
        results_to_process = aggregated_results
        configs_to_use = representative_configs
    else:
        results_to_process = results
        configs_to_use = task_configs

    best_configs: dict[str, pd.DataFrame] = {}
    best_throughputs: dict[str, float] = {}
    pareto_fronts: dict[str, pd.DataFrame | None] = {}
    pareto_x_axis: dict[str, str] = {}

    for name, task_result in results_to_process.items():
        pareto_df = task_result["pareto_df"]
        task_config = configs_to_use[name]
        runtime_cfg = task_config.config.runtime_config
        target_tpot = runtime_cfg.tpot
        target_request_latency = runtime_cfg.request_latency
        use_request_latency = target_request_latency is not None and target_request_latency > 0
        total_gpus = getattr(task_config, "total_gpus", None) or 0

        # Add backend columns for single-backend mode (already added in 'any' mode by aggregator)
        if pareto_df is not None and not pareto_df.empty and not is_any_mode:
            pareto_df = pareto_df.copy()
            if task_config.serving_mode == "agg":
                pareto_df["backend"] = task_config.config.worker_config.backend_name
                pareto_df["backend_version"] = task_config.config.worker_config.backend_version
            else:
                pareto_df["(p)backend"] = task_config.config.prefill_worker_config.backend_name
                pareto_df["(p)backend_version"] = task_config.config.prefill_worker_config.backend_version
                pareto_df["(d)backend"] = task_config.config.decode_worker_config.backend_name
                pareto_df["(d)backend_version"] = task_config.config.decode_worker_config.backend_version
            # Update the result dict so downstream uses the updated pareto_df
            task_result["pareto_df"] = pareto_df

        # Compute tokens/s/gpu_cluster for pareto_df
        if pareto_df is not None and not pareto_df.empty:
            pareto_df["tokens/s/gpu_cluster"] = (
                pareto_df["tokens/s/gpu"]
                * (total_gpus // pareto_df["num_total_gpus"])
                * pareto_df["num_total_gpus"]
                / total_gpus
            )
            x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"
            pareto_frontier_df = get_pareto_front(
                pareto_df,
                x_axis_col,
                "tokens/s/gpu_cluster",
                maximize_x=not use_request_latency,
                maximize_y=True,
            )
        else:
            pareto_frontier_df = pd.DataFrame()
            x_axis_col = "request_latency" if use_request_latency else "tokens/s/user"

        # For 'any' mode, skip group_by to get absolute top 5 across all backends
        # For single backend mode, group_by deduplicates by parallelism config
        if is_any_mode:
            group_by_key = None
        else:
            group_by_key = "(d)parallel" if task_config.serving_mode == "disagg" else "parallel"

        if use_request_latency:
            best_config_df = get_best_configs_under_request_latency_constraint(
                total_gpus=total_gpus,
                pareto_df=pareto_df,
                target_request_latency=target_request_latency,
                top_n=top_n,
                group_by=group_by_key,
            )
        else:
            best_config_df = get_best_configs_under_tpot_constraint(  # based on all data points
                total_gpus=total_gpus,
                pareto_df=pareto_df,
                target_tpot=target_tpot,
                top_n=top_n,
                group_by=group_by_key,
            )
        best_configs[name] = best_config_df
        pareto_fronts[name] = pareto_frontier_df
        pareto_x_axis[name] = x_axis_col
        if not best_config_df.empty:
            best_throughputs[name] = best_config_df["tokens/s/gpu_cluster"].values[0]
        else:
            best_throughputs[name] = 0.0

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"

    # For 'any' mode, use representative configs for summary/saving
    configs_for_summary = configs_to_use if is_any_mode else task_configs

    log_final_summary(
        chosen_exp=chosen_exp,  # for summary
        best_throughputs=best_throughputs,  # for summary
        best_configs=best_configs,  # for table
        pareto_fronts=pareto_fronts,  # for plotting
        task_configs=configs_for_summary,  # for info in summary
        mode=mode,
        pareto_x_axis=pareto_x_axis,
        top_n=top_n,
    )

    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds", end_time - start_time)

    # Return both aggregated configs (for directory structure) and original task_configs
    # (for config generation in 'any' mode)
    return chosen_exp, best_configs, pareto_fronts, best_throughputs, configs_for_summary, task_configs


def _run_generate_mode(args):
    """Run the generate mode to create a naive agg config without sweeping."""
    model_path = args.model_path
    logger.info("Generating naive agg configuration for %s on %d GPUs", model_path, args.total_gpus)

    # Use the public API function
    result = generate_naive_config(
        model_path=model_path,
        total_gpus=args.total_gpus,
        system=args.system,
        backend=args.backend,
        output_dir=args.save_dir or "./output",
    )

    # Extract result data for CLI output
    generator_params = result["generator_params"]
    backend_version = result["backend_version"]
    output_dir = result["output_dir"]
    parallelism = result["parallelism"]
    tp = parallelism["tp"]
    pp = parallelism["pp"]
    replicas = parallelism["replicas"]
    gpus_used = parallelism["gpus_used"]

    # Print summary
    print("\n" + "=" * 60)
    print("  Naive Configuration Generated Successfully")
    print("=" * 60)
    print(f"  Model:           {model_path}")
    print(f"  System:          {args.system}")
    print(f"  Backend:         {args.backend} ({backend_version})")
    print(f"  Total GPUs:      {args.total_gpus} (using {gpus_used})")
    print(f"  Parallelism:     TP={tp}, PP={pp}")
    print(f"  Replicas:        {replicas} (each using {tp * pp} GPUs)")
    print(f"  Max Batch Size:  {generator_params['params']['agg']['max_batch_size']}")
    print(f"  Output:          {output_dir}")
    print("=" * 60)
    print("\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        filepath = os.path.join(output_dir, filename)
        if os.path.isfile(filepath):
            print(f"  - {filename}")
        elif os.path.isdir(filepath):
            print(f"  - {filename}/")
            for subfile in sorted(os.listdir(filepath)):
                print(f"      - {subfile}")
    print("\n" + "-" * 60)
    print("  WARNING: This is a NAIVE configuration generated without")
    print("  memory validation or performance optimization. It may NOT")
    print("  work if the model is too large for the available GPU memory.")
    print("")
    print("  For production deployments, use 'aiconfigurator cli default'")
    print("  to run the full parameter sweep with SLA optimization.")
    print("-" * 60)
    print("\nTo deploy, run the generated shell script or apply the k8s manifest.")
    print("=" * 60 + "\n")


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")

    # Handle generate mode separately (no sweeping)
    if args.mode == "generate":
        _run_generate_mode(args)
        return

    if args.mode == "default":
        task_configs = build_default_task_configs(
            model_path=args.model_path,
            total_gpus=args.total_gpus,
            system=args.system,
            decode_system=args.decode_system,
            backend=args.backend,
            backend_version=args.backend_version,
            database_mode=args.database_mode,
            isl=args.isl,
            osl=args.osl,
            ttft=args.ttft,
            tpot=args.tpot,
            request_latency=args.request_latency,
            prefix=args.prefix,
        )
    elif args.mode == "exp":
        try:
            task_configs = build_experiment_task_configs(yaml_path=args.yaml_path)
        except (ValueError, TypeError) as exc:
            logger.exception("Failed to build experiment task configs")
            raise SystemExit(1) from exc
        if not task_configs:
            logger.error("No valid experiments found in '%s'.", args.yaml_path)
            raise SystemExit(1)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    (
        chosen_exp,
        best_configs,
        pareto_fronts,
        best_throughputs,
        effective_task_configs,
        all_task_configs,
    ) = _execute_task_configs(
        task_configs,
        args.mode,
        top_n=args.top_n,
    )

    if args.save_dir:
        save_results(
            args=args,
            best_configs=best_configs,
            pareto_fronts=pareto_fronts,
            task_configs=effective_task_configs,  # Aggregated ("agg"/"disagg") for 'any' mode
            save_dir=args.save_dir,
            generated_backend_version=args.generated_config_version,
        )


if __name__ == "__main__":
    if generator_cli_helper(sys.argv[1:]):
        sys.exit(0)
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for Disaggregated Serving Deployment")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
