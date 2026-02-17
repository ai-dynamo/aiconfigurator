# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python API for calling CLI workflows programmatically.

This module provides simple function interfaces to the CLI's "default", "exp",
"generate", "estimate", and "support" modes, making it easy to use from Python code without going through argparse.
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
    It's a light-weight check, need to verify under the CLI default or exp mode.

    This is the programmatic equivalent of:
        aiconfigurator cli support --model-path ... --system ...

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
        architecture = model_info["architecture"]
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
        aiconfigurator cli default --model-path ... --total-gpus ... --system ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        total_gpus: Total number of GPUs for deployment.
        system: System name (GPU type), e.g., 'h200_sxm', 'b200_sxm'.
        decode_system: System name for disagg decode workers. Defaults to `system`.
        backend: Backend name ('trtllm', 'sglang', 'vllm', 'auto'). Default is 'trtllm'.
            Use 'auto' to sweep across all three backends and compare results.
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
        ...     backend="auto",
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
        aiconfigurator cli exp --yaml-path experiments.yaml

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


@dataclass
class EstimateResult:
    """Result from running a single-point performance estimate."""

    ttft: float
    """Time to first token (ms)."""

    tpot: float
    """Time per output token (ms)."""

    power_w: float
    """End-to-end weighted average power per GPU (watts)."""

    isl: int
    """Input sequence length used."""

    osl: int
    """Output sequence length used."""

    batch_size: int
    """Batch size used."""

    ctx_tokens: int
    """Context tokens budget for IFB scheduling."""

    tp_size: int
    """Tensor parallelism degree."""

    pp_size: int
    """Pipeline parallelism degree."""

    model_path: str
    """Model path used."""

    system_name: str
    """System name used."""

    backend_name: str
    """Backend name used."""

    backend_version: str
    """Backend version used."""

    raw: dict
    """Full result dict from the InferenceSummary."""

    per_ops_data: dict | None = None
    """Per-operation latency breakdown (populated when available)."""

    def __repr__(self) -> str:
        return (
            f"EstimateResult(ttft={self.ttft:.3f}ms, tpot={self.tpot:.3f}ms, "
            f"power={self.power_w:.1f}W, model={self.model_path}, "
            f"system={self.system_name}, backend={self.backend_name})"
        )


def cli_estimate(
    model_path: str,
    system_name: str,
    *,
    backend_name: str = "trtllm",
    backend_version: str | None = None,
    database_mode: str = "SILICON",
    isl: int = 1024,
    osl: int = 1024,
    batch_size: int = 128,
    ctx_tokens: int | None = None,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    systems_paths: str | None = None,
) -> EstimateResult:
    """
    Estimate TTFT, TPOT, and power for a single model/system/config combination.

    This runs the SDK's aggregated (IFB) inference estimation and returns
    the predicted latency and power metrics without performing any parameter
    sweep or SLA optimization.

    This is the programmatic equivalent of:
        aiconfigurator cli estimate --model-path ... --system ... --batch-size ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        system_name: System name (GPU type), e.g., 'h200_sxm', 'h100_sxm'.
        backend_name: Backend name ('trtllm', 'sglang', 'vllm'). Default is 'trtllm'.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation
            ('SILICON', 'HYBRID', 'EMPIRICAL', 'SOL'). Default is 'SILICON'.
        isl: Input sequence length. Default is 1024.
        osl: Output sequence length. Default is 1024.
        batch_size: Batch size (max concurrent requests). Default is 128.
        ctx_tokens: Context tokens budget for IFB scheduling.
            Default is None, which uses ``isl`` as the budget.
        tp_size: Tensor parallelism size. Default is 1.
        pp_size: Pipeline parallelism size. Default is 1.
        attention_dp_size: Attention data parallelism size. Default is 1.
        moe_tp_size: MoE tensor parallelism size. Default is None (auto).
        moe_ep_size: MoE expert parallelism size. Default is None (auto).
        gemm_quant_mode: GEMM quantization mode (e.g., 'fp8', 'float16', 'int8_wo').
            Default is None (auto-inferred from model config).
        kvcache_quant_mode: KV cache quantization mode (e.g., 'fp8', 'float16').
            Default is None (auto-inferred from model config).
        fmha_quant_mode: FMHA quantization mode (e.g., 'fp8', 'float16').
            Default is None (auto-inferred from model config).
        moe_quant_mode: MoE quantization mode (e.g., 'fp8', 'float16', 'fp8_block').
            Default is None (auto-inferred from model config).
        comm_quant_mode: Communication quantization mode (e.g., 'fp8', 'half').
            Default is None (auto-inferred, defaults to 'half').
        systems_paths: Comma-separated systems search paths. Use 'default' for built-in.

    Returns:
        EstimateResult with ttft, tpot, power_w, and the full raw result dict.

    Example:
        >>> result = cli_estimate(
        ...     model_path="Qwen/Qwen3-32B",
        ...     system_name="h100_sxm",
        ...     batch_size=64,
        ...     isl=2048,
        ...     osl=512,
        ...     tp_size=2,
        ... )
        >>> print(f"TTFT: {result.ttft:.2f}ms, TPOT: {result.tpot:.2f}ms, Power: {result.power_w:.1f}W")
    """
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
    from aiconfigurator.sdk.inference_session import InferenceSession
    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.perf_database import (
        get_database,
        get_latest_database_version,
        set_systems_paths,
    )

    if systems_paths is not None:
        set_systems_paths(systems_paths)

    if ctx_tokens is None:
        ctx_tokens = isl

    # Resolve backend version
    resolved_version = backend_version
    if resolved_version is None:
        resolved_version = get_latest_database_version(system=system_name, backend=backend_name)
        if resolved_version is None:
            raise ValueError(
                f"No database found for system={system_name}, backend={backend_name}. "
                "Check --systems-paths or available databases."
            )

    # Default moe_tp_size/moe_ep_size to match attention parallelism width
    if moe_tp_size is None and moe_ep_size is None:
        moe_tp_size = tp_size
        moe_ep_size = attention_dp_size
    elif moe_tp_size is None:
        moe_tp_size = tp_size * attention_dp_size // moe_ep_size
    elif moe_ep_size is None:
        moe_ep_size = tp_size * attention_dp_size // moe_tp_size

    # Validate MoE parallelism width matches attention parallelism width
    attn_width = tp_size * attention_dp_size
    moe_width = moe_tp_size * moe_ep_size
    if attn_width != moe_width:
        raise ValueError(
            f"Parallelism width mismatch: tp_size({tp_size}) * attention_dp_size({attention_dp_size}) = {attn_width}, "
            f"but moe_tp_size({moe_tp_size}) * moe_ep_size({moe_ep_size}) = {moe_width}. "
            f"These must be equal. Adjust --moe-tp-size/--moe-ep-size or --tp-size/--attention-dp-size."
        )

    # Build model config — quant defaults are auto-applied inside get_model for any None fields
    from aiconfigurator.sdk.common import (
        CommQuantMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )

    model_config = ModelConfig(
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        gemm_quant_mode=GEMMQuantMode[gemm_quant_mode] if gemm_quant_mode else None,
        kvcache_quant_mode=KVCacheQuantMode[kvcache_quant_mode] if kvcache_quant_mode else None,
        fmha_quant_mode=FMHAQuantMode[fmha_quant_mode] if fmha_quant_mode else None,
        moe_quant_mode=MoEQuantMode[moe_quant_mode] if moe_quant_mode else None,
        comm_quant_mode=CommQuantMode[comm_quant_mode] if comm_quant_mode else None,
    )

    runtime_config = RuntimeConfig(
        isl=isl,
        osl=osl,
        batch_size=batch_size,
    )

    model = get_model(model_path, model_config, backend_name)
    database = get_database(system_name, backend_name, resolved_version)
    if database is None:
        raise ValueError(
            f"Failed to load perf database for system={system_name}, "
            f"backend={backend_name}, version={resolved_version}."
        )
    if database_mode != "SILICON":
        from aiconfigurator.sdk.common import DatabaseMode

        database.set_default_database_mode(DatabaseMode[database_mode])
    backend = get_backend(backend_name)
    session = InferenceSession(model, database, backend)
    summary = session.run_agg(runtime_config, ctx_tokens=ctx_tokens)

    result_dict = summary.get_result_dict()
    if result_dict is None:
        raise RuntimeError("Estimation produced no results. The configuration may be invalid or OOM.")

    return EstimateResult(
        ttft=result_dict["ttft"],
        tpot=result_dict["tpot"],
        power_w=result_dict.get("power_w", 0.0),
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        ctx_tokens=ctx_tokens,
        tp_size=tp_size,
        pp_size=pp_size,
        model_path=model_path,
        system_name=system_name,
        backend_name=backend_name,
        backend_version=resolved_version,
        raw=result_dict,
        per_ops_data=summary.get_per_ops_data(),
    )


# Re-export generate_naive_config as cli_generate for consistency
# This is already a clean Python function in generator.api
from aiconfigurator.generator.api import generate_naive_config as cli_generate

__all__ = [
    "CLIResult",
    "EstimateResult",
    "cli_default",
    "cli_estimate",
    "cli_exp",
    "cli_generate",
    "cli_support",
]
