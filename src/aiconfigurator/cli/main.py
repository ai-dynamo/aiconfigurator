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
from aiconfigurator.cli.utils import merge_experiment_results_by_mode, process_experiment_result
from aiconfigurator.generator.api import (
    add_generator_override_arguments,
    generate_naive_config,
    generator_cli_helper,
)
from aiconfigurator.sdk import common, perf_database
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
        help="Number of top configurations to output for each experiment (in exp mode) "
        "or for each mode (agg/disagg) in default mode. Default: 5.",
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
        choices=[backend.value for backend in common.BackendName] + ["any"],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name. Use 'any' to sweep across all backends (trtllm, vllm, sglang) and compare results.",
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
        choices=[backend.value for backend in common.BackendName],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name (default: trtllm).",
    )


def _add_support_mode_arguments(parser):
    """Add arguments for the support mode (support matrix check)."""
    parser.add_argument(
        "--model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
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
        choices=[backend.value for backend in common.BackendName],
        type=str,
        default="trtllm",
        help="Backend name to filter by. Defaults to 'trtllm'.",
    )
    parser.add_argument(
        "--backend_version",
        type=str,
        default=None,
        help="Optional backend version to filter by.",
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

    # Support mode - support matrix check
    support_parser = subparsers.add_parser(
        "support",
        parents=[common_cli_parser],
        help="Check if AIC supports the model/hardware combo for (agg, disagg).",
        description="Verify support for a specific model and system combination using the support matrix.",
    )
    _add_support_mode_arguments(support_parser)


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
            Use 'any' to sweep across all backends.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation.
        isl: Input sequence length.
        osl: Output sequence length.
        ttft: Time to first token target in ms.
        tpot: Time per output token target in ms.
        request_latency: Optional end-to-end request latency target (ms).
        prefix: Prefix cache length.

    Returns:
        Dict with TaskConfig objects. When backend='any', returns 6 configs
        (agg_trtllm, agg_vllm, agg_sglang, disagg_trtllm, disagg_vllm, disagg_sglang).
        Otherwise returns 2 configs ('agg' and 'disagg').
    """
    decode_system = decode_system or system
    # Expand "any" backend to all available backends
    backends_to_sweep = [b.value for b in common.BackendName] if backend == "any" else [backend]

    if backend_version:
        for backend_name in backends_to_sweep:
            _ensure_backend_version_available(system, backend_name, backend_version)
            if decode_system != system:
                _ensure_backend_version_available(decode_system, backend_name, backend_version)

    common_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "system_name": system,
        "backend_version": backend_version,
        "total_gpus": total_gpus,
        "isl": isl,
        "osl": osl,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "prefix": prefix,
        "database_mode": database_mode,
    }

    task_configs: dict[str, TaskConfig] = {}

    for backend_name in backends_to_sweep:
        # Create agg task for this backend
        agg_kwargs = dict(common_kwargs)
        agg_kwargs["backend_name"] = backend_name
        agg_task = TaskConfig(serving_mode="agg", **agg_kwargs)
        exp_name = f"agg_{backend_name}" if backend == "any" else "agg"
        task_configs[exp_name] = agg_task

        # Create disagg task for this backend
        disagg_kwargs = dict(common_kwargs)
        disagg_kwargs["backend_name"] = backend_name
        disagg_kwargs["decode_system_name"] = decode_system
        disagg_task = TaskConfig(serving_mode="disagg", **disagg_kwargs)
        exp_name = f"disagg_{backend_name}" if backend == "any" else "disagg"
        task_configs[exp_name] = disagg_task

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

        task_kwargs: dict[str, Any] = {
            "serving_mode": serving_mode,
            "model_path": model_path,
            "system_name": system_name,
            "backend_name": backend_name,
            "total_gpus": total_gpus,
            "profiles": exp_config.get("profiles", []),
        }

        if backend_version is not None:
            _ensure_backend_version_available(system_name, backend_name, backend_version)
            if serving_mode == "disagg" and inferred_decode_system and inferred_decode_system != system_name:
                _ensure_backend_version_available(inferred_decode_system, backend_name, backend_version)
            task_kwargs["backend_version"] = backend_version

        if serving_mode == "disagg":
            task_kwargs["decode_system_name"] = inferred_decode_system or system_name

        # Per-experiment overrides for runtime numeric parameters if provided at top level
        for numeric_key in ("isl", "osl", "ttft", "tpot", "request_latency"):
            if numeric_key in exp_config:
                task_kwargs[numeric_key] = exp_config[numeric_key]

        if "enable_wideep" in exp_config:
            task_kwargs["enable_wideep"] = exp_config["enable_wideep"]
        if "use_specific_quant_mode" in exp_config:
            task_kwargs["use_specific_quant_mode"] = exp_config["use_specific_quant_mode"]
        if "database_mode" in exp_config:
            task_kwargs["database_mode"] = exp_config["database_mode"]

        yaml_config = _build_yaml_config(exp_config, config_section)
        if yaml_config:
            task_kwargs["yaml_config"] = yaml_config

        try:
            task_configs[exp_name] = TaskConfig(**task_kwargs)
        except Exception:
            logger.exception("Failed to build TaskConfig for experiment '%s'", exp_name)

    return task_configs


def _execute_task_configs(
    task_configs: dict[str, TaskConfig],
    mode: str,
    top_n: int = 5,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, float]]:
    """
    Execute task configs and return the chosen experiment, best configs, results, and best
    throughputs.

    Args:
        task_configs: Dictionary mapping experiment names to TaskConfig objects to execute.
        mode: Execution mode ('default' or 'exp').
        top_n: Number of top configurations to return for each experiment.

    Returns:
        tuple:
            - The experiment name with the overall best throughput ("chosen experiment").
            - Dictionary of best config DataFrames per experiment (or per serving mode if merged).
            - Dictionary of Pareto frontier DataFrames per experiment (or mode).
            - Dictionary of best throughput values per experiment (or mode).
    """
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

    best_configs: dict[str, pd.DataFrame] = {}
    best_throughputs: dict[str, float] = {}
    pareto_fronts: dict[str, pd.DataFrame | None] = {}
    pareto_x_axis: dict[str, str] = {}
    for name, task_result in results.items():
        task_config = task_configs[name]
        best_config_df, best_throughput, pareto_frontier_df, x_axis_col = process_experiment_result(
            task_config, task_result, top_n
        )
        best_configs[name] = best_config_df
        best_throughputs[name] = best_throughput
        pareto_fronts[name] = pareto_frontier_df
        pareto_x_axis[name] = x_axis_col

    if mode == "default" and len(task_configs) > 2:
        best_configs, best_throughputs, pareto_fronts, pareto_x_axis = merge_experiment_results_by_mode(
            task_configs, best_configs, pareto_fronts, pareto_x_axis, top_n
        )

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"

    log_final_summary(
        chosen_exp=chosen_exp,  # for summary
        best_throughputs=best_throughputs,  # for summary
        best_configs=best_configs,  # for table
        pareto_fronts=pareto_fronts,  # for plotting
        task_configs=task_configs,  # for info in summary
        mode=mode,
        pareto_x_axis=pareto_x_axis,
        top_n=top_n,
    )

    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds", end_time - start_time)

    return chosen_exp, best_configs, pareto_fronts, best_throughputs


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


def _run_support_mode(args):
    """Run the support mode to see if a model/hardware combo is supported."""
    model = args.model_path
    system = args.system
    backend = args.backend
    version = args.backend_version

    # If no version specified, find the latest version in the support matrix
    if not version:
        matrix = common.get_support_matrix()
        versions_for_combo = [row["Version"] for row in matrix if row["System"] == system and row["Backend"] == backend]
        if versions_for_combo:
            # Sort versions and take the latest (assumes semantic versioning or lexicographic order)
            version = sorted(set(versions_for_combo), reverse=True)[0]

    logger.info("Checking support for model=%s, system=%s, backend=%s, version=%s", model, system, backend, version)

    # Resolve architecture for better check
    try:
        model_info = get_model_config_from_model_path(model)
        architecture = model_info[0]
    except Exception:
        architecture = None

    result = common.check_support(
        model=model, system=system, backend=backend, version=version, architecture=architecture
    )

    print("\n" + "=" * 60)
    print("  AIC Support Check Results")
    print("=" * 60)
    print(f"  Model:           {model}")
    print(f"  System:          {system}")
    print(f"  Backend:         {backend}")
    print(f"  Version:         {version}")
    print("-" * 60)
    print(f"  Aggregated Support:    {'YES' if result.agg_supported else 'NO'}")
    print(f"  Disaggregated Support: {'YES' if result.disagg_supported else 'NO'}")

    # Show explanation if support was inferred from architecture majority vote
    if not result.exact_match and result.architecture:
        print("-" * 60)
        print(f"  Note: Model '{model}' not found in support matrix.")
        print(f"  Support inferred from architecture '{result.architecture}' majority vote:")
        if result.agg_total_count:
            p, t = result.agg_pass_count, result.agg_total_count
            print(f"    Aggregated:    {p}/{t} passed (>{t // 2} required)")
        if result.disagg_total_count:
            p, t = result.disagg_pass_count, result.disagg_total_count
            print(f"    Disaggregated: {p}/{t} passed (>{t // 2} required)")

    print("=" * 60 + "\n")


def main(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    )

    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")
    logger.info(f"Number of top configurations to output: {args.top_n} (change with --top_n)")

    # Handle generate mode separately (no sweeping)
    if args.mode == "generate":
        _run_generate_mode(args)
        return

    # Handle support mode separately (no sweeping)
    if args.mode == "support":
        _run_support_mode(args)
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

    _, best_configs, pareto_fronts, _ = _execute_task_configs(
        task_configs,
        args.mode,
        top_n=args.top_n,
    )

    if args.save_dir:
        save_results(
            args=args,
            best_configs=best_configs,
            pareto_fronts=pareto_fronts,
            task_configs=task_configs,
            save_dir=args.save_dir,
            generated_backend_version=args.generated_config_version,
            backend=args.backend if args.mode == "default" else None,
        )


if __name__ == "__main__":
    if generator_cli_helper(sys.argv[1:]):
        sys.exit(0)
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for Disaggregated Serving Deployment")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
