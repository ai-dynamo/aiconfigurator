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
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.task import (
    DEFAULT_DECODE_LATENCY_CORRECTION_SCALE,
    DEFAULT_PREFILL_LATENCY_CORRECTION_SCALE,
    TaskConfig,
)


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

    best_latencies: dict[str, dict[str, float]] = field(default_factory=dict)
    """Estimated latencies (ttft, tpot, request_latency) from the rank-1 config per experiment."""

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
    chosen_exp, best_configs, pareto_fronts, best_throughputs, best_latencies = _execute_task_configs_internal(
        task_configs, mode, top_n=top_n
    )

    return CLIResult(
        chosen_exp=chosen_exp,
        best_configs=best_configs,
        pareto_fronts=pareto_fronts,
        best_throughputs=best_throughputs,
        best_latencies=best_latencies,
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
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    top_n: int = 5,
    save_dir: str | None = None,
    generator_set: list[str] | None = None,
    generator_config: str | None = None,
    generator_dynamo_version: str | None = None,
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
        free_gpu_memory_fraction: Fraction of free GPU memory allocated for KV cache
            (default ``None``, meaning the backend default is used). Must be > 0 and <= 1.
            Used to filter batch sizes that would exceed KV cache capacity.
        max_seq_len: TRT-LLM ``--max_seq_len`` setting. Controls how many KV blocks are
            pre-allocated per sequence. Defaults to ``isl + osl`` when ``None``.
        top_n: Number of top configurations to return for each mode (agg/disagg). Default is 5.
        save_dir: Directory to save results. If None, results are not saved to disk.
        generator_set: List of inline generator overrides in KEY=VALUE format (e.g.,
            ``["rule=benchmark", "ServiceConfig.model_path=Qwen/Qwen3-32B-FP8"]``).
            Equivalent to repeating ``--generator-set`` on the CLI.
        generator_config: Path to a unified generator YAML config file.
        generator_dynamo_version: Override Dynamo version used by the generator.

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

        >>> # Use benchmark rule plugin for generator
        >>> result = cli_default(
        ...     model_path="Qwen/Qwen3-32B-FP8",
        ...     total_gpus=8,
        ...     system="h200_sxm",
        ...     save_dir="./results",
        ...     generator_set=["rule=benchmark"],
        ... )

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
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        max_seq_len=max_seq_len,
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
        mock_args.generator_set = generator_set
        mock_args.generator_config = generator_config
        mock_args.generator_dynamo_version = generator_dynamo_version

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

    mode: str = "agg"
    """Estimation mode: 'agg' or 'disagg'."""

    per_ops_data: dict | None = None
    """Per-operation latency breakdown (populated when available)."""

    kv_cache_warning: str | None = None
    """Warning message when batch_size exceeds KV cache capacity."""

    @property
    def request_latency(self) -> float:
        """End-to-end request latency (ms)."""
        return self.raw.get("request_latency", 0.0)

    @property
    def tokens_per_second(self) -> float:
        """Total output throughput (tokens/s)."""
        return self.raw.get("tokens/s", 0.0)

    @property
    def tokens_per_second_per_gpu(self) -> float:
        """Per-GPU output throughput (tokens/s/gpu)."""
        return self.raw.get("tokens/s/gpu", 0.0)

    @property
    def tokens_per_second_per_user(self) -> float:
        """Per-user output throughput (tokens/s/user)."""
        return self.raw.get("tokens/s/user", 0.0)

    @property
    def concurrency(self) -> float:
        """Effective concurrency (requests in flight)."""
        return self.raw.get("concurrency", 0.0)

    @property
    def seq_per_second(self) -> float:
        """Sequence throughput (seq/s)."""
        return self.raw.get("seq/s", 0.0)

    @property
    def num_total_gpus(self) -> int:
        """Total GPUs used by the parallelism config."""
        return int(self.raw.get("num_total_gpus", 0))

    @property
    def memory(self) -> float:
        """Estimated GPU memory usage (GB).

        For disagg mode there is no single memory value; use
        ``raw["(p)memory"]`` and ``raw["(d)memory"]`` instead.
        """
        if self.mode == "disagg":
            return 0.0
        return self.raw.get("memory", 0.0)

    def get(self) -> dict:
        """
        Return all metrics as a dict matching the CSV column schema.

        Includes ``tokens/s/gpu_cluster`` (equals ``tokens/s/gpu`` for
        single-point estimates where only one replica group is evaluated).

        Verified by ``tests/e2e/cli/test_cli_estimate_vs_default.py`` which
        asserts that the returned keys and values match the
        ``best_config_topn.csv`` output from ``cli_default``.
        """
        result = dict(self.raw)
        if "tokens/s/gpu_cluster" not in result:
            result["tokens/s/gpu_cluster"] = result.get("tokens/s/gpu", 0.0)
        return result

    def __repr__(self) -> str:
        return (
            f"EstimateResult(mode={self.mode!r}, ttft={self.ttft:.3f}ms, tpot={self.tpot:.3f}ms, "
            f"power={self.power_w:.1f}W, model={self.model_path}, "
            f"system={self.system_name}, backend={self.backend_name})"
        )


def _resolve_moe_parallelism(
    tp_size: int,
    attention_dp_size: int,
    moe_tp_size: int | None,
    moe_ep_size: int | None,
    model_path: str | None = None,
) -> tuple[int, int]:
    """Resolve and validate MoE parallelism widths, returning (moe_tp_size, moe_ep_size).

    For dense (non-MoE) models the width constraint is not enforced because
    MoE parallelism has no effect on the computation.
    """
    if moe_tp_size is None and moe_ep_size is None:
        moe_tp_size = tp_size
        moe_ep_size = attention_dp_size
    elif moe_tp_size is None:
        moe_tp_size = tp_size * attention_dp_size // moe_ep_size
    elif moe_ep_size is None:
        moe_ep_size = tp_size * attention_dp_size // moe_tp_size

    attn_width = tp_size * attention_dp_size
    moe_width = moe_tp_size * moe_ep_size
    if attn_width != moe_width and (model_path is None or check_is_moe(model_path)):
        raise ValueError(
            f"Parallelism width mismatch: tp_size({tp_size}) * attention_dp_size({attention_dp_size}) = {attn_width}, "
            f"but moe_tp_size({moe_tp_size}) * moe_ep_size({moe_ep_size}) = {moe_width}. "
            f"These must be equal."
        )
    return moe_tp_size, moe_ep_size


def _build_model_config(
    tp_size: int,
    pp_size: int,
    attention_dp_size: int,
    moe_tp_size: int,
    moe_ep_size: int,
    gemm_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
):
    """Build a ModelConfig with optional quant mode overrides."""
    from aiconfigurator.sdk.common import (
        CommQuantMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aiconfigurator.sdk.config import ModelConfig

    return ModelConfig(
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


def cli_estimate(
    model_path: str,
    system_name: str,
    *,
    mode: str = "agg",
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
    # Disagg-specific parameters (ignored when mode='agg')
    decode_system_name: str | None = None,
    prefill_tp_size: int | None = None,
    prefill_pp_size: int | None = None,
    prefill_attention_dp_size: int | None = None,
    prefill_moe_tp_size: int | None = None,
    prefill_moe_ep_size: int | None = None,
    prefill_batch_size: int | None = None,
    prefill_num_workers: int | None = None,
    decode_tp_size: int | None = None,
    decode_pp_size: int | None = None,
    decode_attention_dp_size: int | None = None,
    decode_moe_tp_size: int | None = None,
    decode_moe_ep_size: int | None = None,
    decode_batch_size: int | None = None,
    decode_num_workers: int | None = None,
    systems_paths: str | None = None,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
) -> EstimateResult:
    """
    Estimate TTFT, TPOT, and power for a single model/system/config combination.

    Supports both aggregated (IFB) and disaggregated serving estimation.

    This is the programmatic equivalent of:
        aiconfigurator cli estimate --model-path ... --system ... --batch-size ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        system_name: System name (GPU type), e.g., 'h200_sxm', 'h100_sxm'.
        mode: Estimation mode — 'agg' (default) or 'disagg'.
        backend_name: Backend name ('trtllm', 'sglang', 'vllm'). Default is 'trtllm'.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation
            ('SILICON', 'HYBRID', 'EMPIRICAL', 'SOL'). Default is 'SILICON'.
        isl: Input sequence length. Default is 1024.
        osl: Output sequence length. Default is 1024.
        batch_size: Batch size (max concurrent requests, used for agg mode). Default is 128.
        ctx_tokens: Context tokens budget for IFB scheduling (agg mode only).
            Default is None, which uses ``isl`` as the budget.
        tp_size: Tensor parallelism size. Default is 1. Also serves as fallback for
            prefill/decode TP when their specific args are omitted.
        pp_size: Pipeline parallelism size. Default is 1.
        attention_dp_size: Attention data parallelism size. Default is 1.
        moe_tp_size: MoE tensor parallelism size. Default is None (auto).
        moe_ep_size: MoE expert parallelism size. Default is None (auto).
        gemm_quant_mode: GEMM quantization mode. Default is None (auto-inferred).
        kvcache_quant_mode: KV cache quantization mode. Default is None (auto-inferred).
        fmha_quant_mode: FMHA quantization mode. Default is None (auto-inferred).
        moe_quant_mode: MoE quantization mode. Default is None (auto-inferred).
        comm_quant_mode: Communication quantization mode. Default is None (auto-inferred).
        decode_system_name: System for disagg decode workers. Defaults to ``system_name``.
        prefill_tp_size: Prefill TP size (disagg). Defaults to ``tp_size``.
        prefill_pp_size: Prefill PP size (disagg). Defaults to ``pp_size``.
        prefill_attention_dp_size: Prefill attention DP size (disagg). Defaults to ``attention_dp_size``.
        prefill_moe_tp_size: Prefill MoE TP size (disagg). Defaults to ``moe_tp_size``.
        prefill_moe_ep_size: Prefill MoE EP size (disagg). Defaults to ``moe_ep_size``.
        prefill_batch_size: Prefill batch size (disagg). Required when mode='disagg'.
        prefill_num_workers: Number of prefill workers (disagg). Required when mode='disagg'.
        decode_tp_size: Decode TP size (disagg). Defaults to ``tp_size``.
        decode_pp_size: Decode PP size (disagg). Defaults to ``pp_size``.
        decode_attention_dp_size: Decode attention DP size (disagg). Defaults to ``attention_dp_size``.
        decode_moe_tp_size: Decode MoE TP size (disagg). Defaults to ``moe_tp_size``.
        decode_moe_ep_size: Decode MoE EP size (disagg). Defaults to ``moe_ep_size``.
        decode_batch_size: Decode batch size (disagg). Required when mode='disagg'.
        decode_num_workers: Number of decode workers (disagg). Required when mode='disagg'.
        systems_paths: Comma-separated systems search paths. Use 'default' for built-in.
        free_gpu_memory_fraction: Fraction of free GPU memory TRT-LLM allocates for
            KV cache (default 0.9 for TRTLLM, 1.0 for other backends). Used to check whether the requested batch_size
            exceeds KV cache capacity.
        max_seq_len: The TRT-LLM ``--max_seq_len`` setting used at serving time.
            Controls how many KV blocks TRT-LLM pre-allocates per sequence. Defaults
            to ``isl + osl`` when ``None``.

    Returns:
        EstimateResult with ttft, tpot, power_w, mode, and the full raw result dict.

    Example (agg):
        >>> result = cli_estimate("Qwen/Qwen3-32B", "h100_sxm", batch_size=64, isl=2048, osl=512, tp_size=2)

    Example (disagg):
        >>> result = cli_estimate(
        ...     "Qwen/Qwen3-32B", "h100_sxm", mode="disagg",
        ...     isl=2048, osl=512, tp_size=2,
        ...     prefill_batch_size=4, prefill_num_workers=2,
        ...     decode_batch_size=64, decode_num_workers=2,
        ... )
    """
    from aiconfigurator.sdk.backends.factory import get_backend
    from aiconfigurator.sdk.models import get_model
    from aiconfigurator.sdk.perf_database import (
        get_database,
        get_latest_database_version,
        set_systems_paths,
    )

    if systems_paths is not None:
        set_systems_paths(systems_paths)

    # Resolve backend version
    resolved_version = backend_version
    if resolved_version is None:
        resolved_version = get_latest_database_version(system=system_name, backend=backend_name)
        if resolved_version is None:
            raise ValueError(
                f"No database found for system={system_name}, backend={backend_name}. "
                "Check --systems-paths or available databases."
            )

    def _load_database(sys_name: str):
        db = get_database(sys_name, backend_name, resolved_version)
        if db is None:
            raise ValueError(
                f"Failed to load perf database for system={sys_name}, "
                f"backend={backend_name}, version={resolved_version}."
            )
        if database_mode != "SILICON":
            from aiconfigurator.sdk.common import DatabaseMode

            db.set_default_database_mode(DatabaseMode[database_mode])
        return db

    if mode == "agg":
        return _run_agg_estimate(
            model_path=model_path,
            system_name=system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            batch_size=batch_size,
            ctx_tokens=ctx_tokens if ctx_tokens is not None else isl,
            tp_size=tp_size,
            pp_size=pp_size,
            attention_dp_size=attention_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            gemm_quant_mode=gemm_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            moe_quant_mode=moe_quant_mode,
            comm_quant_mode=comm_quant_mode,
            load_database=_load_database,
            get_backend=get_backend,
            get_model=get_model,
            free_gpu_memory_fraction=free_gpu_memory_fraction,
            max_seq_len=max_seq_len,
        )
    elif mode == "disagg":
        # Validate required disagg params
        for name, val in [
            ("prefill_batch_size", prefill_batch_size),
            ("prefill_num_workers", prefill_num_workers),
            ("decode_batch_size", decode_batch_size),
            ("decode_num_workers", decode_num_workers),
        ]:
            if val is None:
                raise ValueError(f"{name} is required for disagg mode.")

        return _run_disagg_estimate(
            model_path=model_path,
            system_name=system_name,
            decode_system_name=decode_system_name or system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            # Prefill config (fall back to shared args)
            prefill_tp_size=prefill_tp_size if prefill_tp_size is not None else tp_size,
            prefill_pp_size=prefill_pp_size if prefill_pp_size is not None else pp_size,
            prefill_attention_dp_size=prefill_attention_dp_size
            if prefill_attention_dp_size is not None
            else attention_dp_size,
            prefill_moe_tp_size=prefill_moe_tp_size if prefill_moe_tp_size is not None else moe_tp_size,
            prefill_moe_ep_size=prefill_moe_ep_size if prefill_moe_ep_size is not None else moe_ep_size,
            prefill_batch_size=prefill_batch_size,
            prefill_num_workers=prefill_num_workers,
            # Decode config (fall back to shared args)
            decode_tp_size=decode_tp_size if decode_tp_size is not None else tp_size,
            decode_pp_size=decode_pp_size if decode_pp_size is not None else pp_size,
            decode_attention_dp_size=decode_attention_dp_size
            if decode_attention_dp_size is not None
            else attention_dp_size,
            decode_moe_tp_size=decode_moe_tp_size if decode_moe_tp_size is not None else moe_tp_size,
            decode_moe_ep_size=decode_moe_ep_size if decode_moe_ep_size is not None else moe_ep_size,
            decode_batch_size=decode_batch_size,
            decode_num_workers=decode_num_workers,
            # Shared quant
            gemm_quant_mode=gemm_quant_mode,
            kvcache_quant_mode=kvcache_quant_mode,
            fmha_quant_mode=fmha_quant_mode,
            moe_quant_mode=moe_quant_mode,
            comm_quant_mode=comm_quant_mode,
            load_database=_load_database,
            get_backend=get_backend,
            get_model=get_model,
        )
    else:
        raise ValueError(f"Unsupported estimate mode: {mode!r}. Use 'agg' or 'disagg'.")


def _run_agg_estimate(
    *,
    model_path,
    system_name,
    backend_name,
    resolved_version,
    isl,
    osl,
    batch_size,
    ctx_tokens,
    tp_size,
    pp_size,
    attention_dp_size,
    moe_tp_size,
    moe_ep_size,
    gemm_quant_mode,
    kvcache_quant_mode,
    fmha_quant_mode,
    moe_quant_mode,
    comm_quant_mode,
    load_database,
    get_backend,
    get_model,
    free_gpu_memory_fraction=None,
    max_seq_len=None,
) -> EstimateResult:
    """Run aggregated (IFB) estimation."""
    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.inference_session import InferenceSession

    moe_tp_size, moe_ep_size = _resolve_moe_parallelism(
        tp_size, attention_dp_size, moe_tp_size, moe_ep_size, model_path=model_path
    )

    model_config = _build_model_config(
        tp_size,
        pp_size,
        attention_dp_size,
        moe_tp_size,
        moe_ep_size,
        gemm_quant_mode,
        kvcache_quant_mode,
        fmha_quant_mode,
        moe_quant_mode,
        comm_quant_mode,
    )
    runtime_config = RuntimeConfig(isl=isl, osl=osl, batch_size=batch_size)

    model = get_model(model_path, model_config, backend_name)
    database = load_database(system_name)
    backend = get_backend(backend_name)
    session = InferenceSession(model, database, backend)
    summary = session.run_agg(
        runtime_config,
        ctx_tokens=ctx_tokens,
        max_seq_len=max_seq_len if max_seq_len is not None else isl + osl,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
    )

    if summary.check_oom():
        raise RuntimeError(
            f"OOM: the model '{model_path}' does not fit in GPU memory on system '{system_name}' "
            f"with the given parallelism (tp={tp_size}, pp={pp_size}, dp={attention_dp_size}). "
            "Try increasing tp_size/pp_size, using a quantized model, or "
            "using a system with more VRAM per GPU."
        )

    result_dict = summary.get_result_dict()
    if result_dict is None:
        raise RuntimeError("Estimation produced no results. The configuration may be invalid.")

    kv_warning = None
    if summary.check_kv_cache_oom():
        frac_str = str(free_gpu_memory_fraction) if free_gpu_memory_fraction is not None else "backend default"
        kv_warning = (
            f"Requested batch_size ({batch_size}) exceeds estimated KV cache capacity "
            f"(free_gpu_memory_fraction={frac_str}). "
            "The serving runtime will queue excess requests, causing significantly higher TTFT and inaccurate TPOT."
        )

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
        mode="agg",
        per_ops_data=summary.get_per_ops_data(),
        kv_cache_warning=kv_warning,
    )


def _run_disagg_estimate(
    *,
    model_path,
    system_name,
    decode_system_name,
    backend_name,
    resolved_version,
    isl,
    osl,
    prefill_tp_size,
    prefill_pp_size,
    prefill_attention_dp_size,
    prefill_moe_tp_size,
    prefill_moe_ep_size,
    prefill_batch_size,
    prefill_num_workers,
    decode_tp_size,
    decode_pp_size,
    decode_attention_dp_size,
    decode_moe_tp_size,
    decode_moe_ep_size,
    decode_batch_size,
    decode_num_workers,
    gemm_quant_mode,
    kvcache_quant_mode,
    fmha_quant_mode,
    moe_quant_mode,
    comm_quant_mode,
    load_database,
    get_backend,
    get_model,
) -> EstimateResult:
    """Run disaggregated estimation."""
    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.inference_session import DisaggInferenceSession

    # Resolve MoE parallelism for prefill and decode separately
    p_moe_tp, p_moe_ep = _resolve_moe_parallelism(
        prefill_tp_size,
        prefill_attention_dp_size,
        prefill_moe_tp_size,
        prefill_moe_ep_size,
        model_path=model_path,
    )
    d_moe_tp, d_moe_ep = _resolve_moe_parallelism(
        decode_tp_size,
        decode_attention_dp_size,
        decode_moe_tp_size,
        decode_moe_ep_size,
        model_path=model_path,
    )

    prefill_model_config = _build_model_config(
        prefill_tp_size,
        prefill_pp_size,
        prefill_attention_dp_size,
        p_moe_tp,
        p_moe_ep,
        gemm_quant_mode,
        kvcache_quant_mode,
        fmha_quant_mode,
        moe_quant_mode,
        comm_quant_mode,
    )
    decode_model_config = _build_model_config(
        decode_tp_size,
        decode_pp_size,
        decode_attention_dp_size,
        d_moe_tp,
        d_moe_ep,
        gemm_quant_mode,
        kvcache_quant_mode,
        fmha_quant_mode,
        moe_quant_mode,
        comm_quant_mode,
    )

    runtime_config = RuntimeConfig(isl=isl, osl=osl)

    prefill_database = load_database(system_name)
    decode_database = load_database(decode_system_name)
    prefill_backend = get_backend(backend_name)
    decode_backend = get_backend(backend_name)

    session = DisaggInferenceSession(
        prefill_database=prefill_database,
        prefill_backend=prefill_backend,
        decode_database=decode_database,
        decode_backend=decode_backend,
    )
    session.set_latency_correction_scales(
        DEFAULT_PREFILL_LATENCY_CORRECTION_SCALE,
        DEFAULT_DECODE_LATENCY_CORRECTION_SCALE,
    )

    summary = session.run_disagg(
        model_path=model_path,
        runtime_config=runtime_config,
        prefill_model_config=prefill_model_config,
        prefill_batch_size=prefill_batch_size,
        prefill_num_worker=prefill_num_workers,
        decode_model_config=decode_model_config,
        decode_batch_size=decode_batch_size,
        decode_num_worker=decode_num_workers,
    )

    if summary.check_oom():
        oom_details = []
        if decode_system_name != system_name:
            oom_details.append(f"prefill system '{system_name}' or decode system '{decode_system_name}'")
        else:
            oom_details.append(f"system '{system_name}'")
        raise RuntimeError(
            f"OOM: the model '{model_path}' does not fit in GPU memory on {oom_details[0]} "
            f"with the given parallelism. "
            "Try increasing tp_size/pp_size, using a quantized model, or "
            "using a system with more VRAM per GPU."
        )

    result_dict = summary.get_result_dict()
    if result_dict is None:
        raise RuntimeError("Disagg estimation produced no results. The configuration may be invalid.")

    return EstimateResult(
        ttft=result_dict["ttft"],
        tpot=result_dict["tpot"],
        power_w=result_dict.get("power_w", 0.0),
        isl=isl,
        osl=osl,
        batch_size=prefill_batch_size,
        ctx_tokens=0,
        tp_size=prefill_tp_size,
        pp_size=prefill_pp_size,
        model_path=model_path,
        system_name=system_name,
        backend_name=backend_name,
        backend_version=resolved_version,
        raw=result_dict,
        mode="disagg",
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
