# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python API for calling CLI workflows programmatically.

This module provides simple function interfaces to the CLI's "default", "exp",
"generate", and "support" modes, making it easy to use from Python code without going through argparse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from aiconfigurator.cli.main import (
    _execute_task_configs as _execute_task_configs_internal,
)
from aiconfigurator.cli.main import (
    build_default_task_configs,
    build_experiment_task_configs,
)
from aiconfigurator.cli.report_and_save import save_results
from aiconfigurator.sdk.task import TaskConfig


def cli_support(
    model_path: str,
    system: str,
    *,
    backend: str = "trtllm",
    backend_version: str | None = None,
) -> tuple[bool, bool]:
    """
    Check if AIC supports the model/hardware combo for (agg, disagg).
    Support is determined by a majority vote of PASS status for the given
    architecture, system, backend, and version in the support matrix.

    This is the programmatic equivalent of:
        aiconfigurator cli support --model_path ... --system ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        system: System name (GPU type), e.g., 'h200_sxm', 'b200_sxm'.
        backend: Optional backend name to filter by ('trtllm', 'sglang', 'vllm').
        backend_version: Optional backend database version.

    Returns:
        tuple[bool, bool]: (agg_supported, disagg_supported)
    """
    from aiconfigurator.sdk.common import check_support
    from aiconfigurator.sdk.utils import get_model_config_from_model_path

    try:
        model_info = get_model_config_from_model_path(model_path)
        architecture = model_info[0]
    except Exception:
        architecture = None

    return check_support(model_path, system, backend, backend_version, architecture=architecture)


logger = logging.getLogger(__name__)


@dataclass
class CLIResult:
    """Result from running CLI default or exp mode."""

    chosen_exp: str
    """Name of the experiment with the best throughput."""

    best_configs: dict[str, pd.DataFrame]
    """Best configurations per experiment, filtered by latency constraints."""

    pareto_fronts: dict[str, pd.DataFrame]
    """Pareto frontier data per experiment."""

    best_throughputs: dict[str, float]
    """Best throughput (tokens/s/gpu_cluster) per experiment."""

    task_configs: dict[str, TaskConfig]
    """TaskConfig objects used for each experiment."""

    raw_results: dict[str, dict[str, pd.DataFrame | None]] = field(default_factory=dict)
    """Raw pareto_df results from TaskRunner, keyed by experiment name."""

    def __repr__(self) -> str:
        return (
            f"CLIResult(chosen_exp={self.chosen_exp!r}, "
            f"experiments={list(self.task_configs.keys())}, "
            f"best_throughputs={self.best_throughputs})"
        )


def _execute_and_wrap_result(
    task_configs: dict[str, TaskConfig],
    mode: str,
    top_n: int = 5,
) -> CLIResult:
    """Execute task configs using main.py's function and wrap result in CLIResult."""
    chosen_exp, best_configs, pareto_fronts, best_throughputs = _execute_task_configs_internal(
        task_configs, mode, top_n=top_n
    )

    return CLIResult(
        chosen_exp=chosen_exp,
        best_configs=best_configs,
        pareto_fronts=pareto_fronts,
        best_throughputs=best_throughputs,
        task_configs=task_configs,
        raw_results={},
    )


def cli_default(
    model_path: str,
    total_gpus: int,
    system: str,
    *,
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
    top_n: int = 5,
    save_dir: str | None = None,
) -> CLIResult:
    """
    Run the default CLI mode: compare aggregated vs disaggregated serving.

    This is the programmatic equivalent of:
        aiconfigurator cli default --model_path ... --total_gpus ... --system ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        total_gpus: Total number of GPUs for deployment.
        system: System name (GPU type), e.g., 'h200_sxm', 'b200_sxm'.
        decode_system: System name for disagg decode workers. Defaults to `system`.
        backend: Backend name ('trtllm', 'sglang', 'vllm', 'any'). Default is 'trtllm'.
            Use 'any' to sweep across all three backends and compare results.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation
            ('SILICON', 'HYBRID', 'EMPIRICAL', 'SOL'). Default is 'SILICON'.
        isl: Input sequence length. Default is 4000.
        osl: Output sequence length. Default is 1000.
        ttft: Time to first token target in ms. Default is 2000.
        tpot: Time per output token target in ms. Default is 30.
        request_latency: Optional end-to-end request latency target (ms).
            Enables request-latency optimization mode.
        prefix: Prefix cache length. Default is 0.
        top_n: Number of top configurations to return for each mode (agg/disagg). Default is 5.
        save_dir: Directory to save results. If None, results are not saved to disk.

    Returns:
        CLIResult with chosen experiment, best configs, pareto fronts, and throughputs.

    Example:
        >>> result = cli_default(
        ...     model_path="Qwen/Qwen3-32B",
        ...     total_gpus=8,
        ...     system="h200_sxm",
        ...     ttft=2000,
        ...     tpot=30,
        ... )
        >>> print(result.chosen_exp)  # 'agg' or 'disagg'
        >>> print(result.best_throughputs)

        >>> # Compare all backends
        >>> result = cli_default(
        ...     model_path="Qwen/Qwen3-32B",
        ...     total_gpus=8,
        ...     system="h200_sxm",
        ...     backend="any",
        ...     ttft=2000,
        ...     tpot=30,
        ... )
        >>> print(result.chosen_exp)  # e.g., 'agg_trtllm' or 'disagg_vllm'
        >>> print(result.best_throughputs)  # Shows all 6 backend/mode combinations
    """
    # Reuse build_default_task_configs from main.py
    task_configs = build_default_task_configs(
        model_path=model_path,
        total_gpus=total_gpus,
        system=system,
        decode_system=decode_system,
        backend=backend,
        backend_version=backend_version,
        database_mode=database_mode,
        isl=isl,
        osl=osl,
        ttft=ttft,
        tpot=tpot,
        request_latency=request_latency,
        prefix=prefix,
    )

    result = _execute_and_wrap_result(task_configs, mode="default", top_n=top_n)

    if save_dir:
        # Create a mock args object for save_results compatibility
        class _MockArgs:
            pass

        mock_args = _MockArgs()
        mock_args.save_dir = save_dir
        mock_args.mode = "default"
        mock_args.model_path = model_path
        mock_args.total_gpus = total_gpus
        mock_args.system = system
        mock_args.backend = backend
        mock_args.isl = isl
        mock_args.osl = osl
        mock_args.ttft = ttft
        mock_args.tpot = tpot
        mock_args.request_latency = request_latency
        mock_args.top_n = top_n
        mock_args.generated_config_version = None

        save_results(
            args=mock_args,
            best_configs=result.best_configs,
            pareto_fronts=result.pareto_fronts,
            task_configs=result.task_configs,
            save_dir=save_dir,
            generated_backend_version=None,
        )

    return result


def cli_exp(
    *,
    yaml_path: str | None = None,
    config: dict[str, dict] | None = None,
    top_n: int = 5,
    save_dir: str | None = None,
) -> CLIResult:
    """
    Run multiple experiments defined by YAML file or dict config.

    This is the programmatic equivalent of:
        aiconfigurator cli exp --yaml_path experiments.yaml

    You must provide either `yaml_path` or `config`, but not both.

    Args:
        yaml_path: Path to a YAML file containing experiment definitions.
        config: Dict containing experiment definitions (alternative to yaml_path).
            Keys are experiment names, values are experiment configs.
        top_n: Number of top configurations to return for each experiment. Default is 5.
        save_dir: Directory to save results. If None, results are not saved to disk.

    Returns:
        CLIResult with chosen experiment, best configs, pareto fronts, and throughputs.

    Example (from YAML file):
        >>> result = cli_exp(yaml_path="experiments.yaml")

    Example (from dict config):
        >>> result = cli_exp(config={
        ...     "agg_qwen3": {
        ...         "serving_mode": "agg",
        ...         "model_path": "Qwen/Qwen3-32B",
        ...         "system_name": "h200_sxm",
        ...         "backend_name": "trtllm",
        ...         "total_gpus": 8,
        ...         "isl": 4000,
        ...         "osl": 1000,
        ...         "ttft": 2000,
        ...         "tpot": 30,
        ...     },
        ...     "disagg_qwen3": {
        ...         "serving_mode": "disagg",
        ...         "model_path": "Qwen/Qwen3-32B",
        ...         "system_name": "h200_sxm",
        ...         "backend_name": "trtllm",
        ...         "total_gpus": 16,
        ...         "isl": 4000,
        ...         "osl": 1000,
        ...         "ttft": 2000,
        ...         "tpot": 30,
        ...     },
        ... })
        >>> print(result.chosen_exp)
        >>> print(result.best_throughputs)

    YAML file format example:
        exps:  # Optional: defines execution order
          - agg_qwen3
          - disagg_qwen3

        agg_qwen3:
          serving_mode: agg
          model_path: Qwen/Qwen3-32B
          system_name: h200_sxm
          backend_name: trtllm
          total_gpus: 8
          isl: 4000
          osl: 1000

        disagg_qwen3:
          serving_mode: disagg
          model_path: Qwen/Qwen3-32B
          system_name: h200_sxm
          backend_name: trtllm
          total_gpus: 16
    """
    task_configs = build_experiment_task_configs(
        yaml_path=yaml_path,
        config=config,
    )

    if not task_configs:
        raise ValueError("No valid experiments found in configuration.")

    result = _execute_and_wrap_result(task_configs, mode="exp", top_n=top_n)

    if save_dir:
        # Create a mock args object for save_results compatibility
        class _MockArgs:
            pass

        mock_args = _MockArgs()
        mock_args.save_dir = save_dir
        mock_args.mode = "exp"
        mock_args.yaml_path = yaml_path
        mock_args.top_n = top_n
        mock_args.generated_config_version = None

        save_results(
            args=mock_args,
            best_configs=result.best_configs,
            pareto_fronts=result.pareto_fronts,
            task_configs=result.task_configs,
            save_dir=save_dir,
            generated_backend_version=None,
        )

    return result


# Re-export generate_naive_config as cli_generate for consistency
# This is already a clean Python function in generator.api
from aiconfigurator.generator.api import generate_naive_config as cli_generate

__all__ = [
    "CLIResult",
    "cli_default",
    "cli_exp",
    "cli_generate",
    "cli_support",
]
