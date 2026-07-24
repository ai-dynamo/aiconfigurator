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
    _execute_tasks as _execute_tasks_internal,
)
from aiconfigurator.cli.main import (
    build_default_tasks,
    build_experiment_tasks,
)
from aiconfigurator.cli.report_and_save import save_results
from aiconfigurator.sdk.config import ModelConfig
from aiconfigurator.sdk.config_builders import apply_nextn as _apply_nextn
from aiconfigurator.sdk.config_builders import build_model_config as _build_model_config
from aiconfigurator.sdk.config_builders import resolve_nextn_auto as _resolve_nextn_auto
from aiconfigurator.sdk.models import check_is_moe, resolve_context_fmha_by_data, resolve_dsv4_moe_arch
from aiconfigurator.sdk.speculative import (
    SpeculativeDecodingProfile,
)
from aiconfigurator.sdk.speculative import (
    normalize_speculative_decoding as _normalize_nextn,
)
from aiconfigurator.sdk.task_v2 import Task

# Default per-phase latency-correction scales for single-point disagg estimates.
# (Migrated from the legacy V1 task module; same values as Task's field defaults.)
DEFAULT_PREFILL_LATENCY_CORRECTION_SCALE = 1.1
DEFAULT_DECODE_LATENCY_CORRECTION_SCALE = 1.08


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

    tasks: dict[str, Task]
    """Task objects used for each experiment."""

    best_latencies: dict[str, dict[str, float]] = field(default_factory=dict)
    """Estimated latencies (ttft, tpot, request_latency) from the rank-1 config per experiment."""

    raw_results: dict[str, dict[str, pd.DataFrame | None]] = field(default_factory=dict)
    """Raw pareto_df results from TaskRunner, keyed by experiment name."""

    def __repr__(self) -> str:
        return (
            f"CLIResult(chosen_exp={self.chosen_exp!r}, "
            f"experiments={list(self.tasks.keys())}, "
            f"best_throughputs={self.best_throughputs})"
        )


def _execute_and_wrap_result(
    tasks: dict[str, Task],
    mode: str,
    top_n: int = 5,
    strict_sla: bool = False,
) -> CLIResult:
    """Execute task configs using main.py's function and wrap result in CLIResult."""
    chosen_exp, best_configs, pareto_fronts, best_throughputs, best_latencies = _execute_tasks_internal(
        tasks, mode, top_n=top_n, strict_sla=strict_sla
    )

    return CLIResult(
        chosen_exp=chosen_exp,
        best_configs=best_configs,
        pareto_fronts=pareto_fronts,
        best_throughputs=best_throughputs,
        best_latencies=best_latencies,
        tasks=tasks,
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
    transfer_policy: str | list | None = None,
    isl: int = 4000,
    osl: int = 1000,
    image_height: int = 0,
    image_width: int = 0,
    num_images: int = 1,
    ttft: float = 2000.0,
    tpot: float = 30.0,
    request_latency: float | None = None,
    prefix: int = 0,
    nextn: int | str = 0,
    nextn_accepted: float | None = None,
    strict_sla: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    top_n: int = 5,
    save_dir: str | None = None,
    generator_set: list[str] | None = None,
    generator_config: str | None = None,
    generator_dynamo_version: str | None = None,
    engine_step_backend: str | None = None,
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
        nextn: MTP draft length, or ``"auto"`` to use the checkpoint's
            ``num_nextn_predict_layers`` (absent/0 keeps MTP disabled).
            Default 0 (disabled); never enabled implicitly.
        nextn_accepted: Average accepted draft tokens per decode step
            (0 <= nextn_accepted <= nextn). Required when the draft depth
            resolves to > 0; never inferred.
        strict_sla: When True, ``pareto_df`` is filtered to only
            SLA-compliant data points (TPOT or request-latency) *before*
            the Pareto frontier is computed.  TTFT is already enforced at
            sweep time.  Default is False (full Pareto frontier, TPOT-only
            constraint at picking time).
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
        engine_step_backend: Experimental static latency backend ("python" or "rust").

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
    # Fail fast on inconsistent MTP inputs (same early check as the CLI path).
    # nextn="auto" resolves the draft depth from the checkpoint first.
    if nextn == "auto":
        nextn = _resolve_nextn_auto(model_path)
    nextn, nextn_accepted = _normalize_nextn(nextn, nextn_accepted)

    # Reuse build_default_tasks from main.py
    tasks = build_default_tasks(
        model_path=model_path,
        total_gpus=total_gpus,
        system=system,
        decode_system=decode_system,
        backend=backend,
        backend_version=backend_version,
        database_mode=database_mode,
        transfer_policy=transfer_policy,
        isl=isl,
        osl=osl,
        image_height=image_height,
        image_width=image_width,
        num_images=num_images,
        ttft=ttft,
        tpot=tpot,
        request_latency=request_latency,
        prefix=prefix,
        nextn=nextn,
        nextn_accepted=nextn_accepted,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        max_seq_len=max_seq_len,
        engine_step_backend=engine_step_backend,
    )

    result = _execute_and_wrap_result(tasks, mode="default", top_n=top_n, strict_sla=strict_sla)

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
        mock_args.image_height = image_height
        mock_args.image_width = image_width
        mock_args.num_images = num_images
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
            tasks=result.tasks,
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
    tasks = build_experiment_tasks(
        yaml_path=yaml_path,
        config=config,
    )

    if not tasks:
        raise ValueError("No valid experiments found in configuration.")

    result = _execute_and_wrap_result(tasks, mode="exp", top_n=top_n)

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
            tasks=result.tasks,
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
    """Estimation mode: 'agg', 'disagg', 'static', 'static_ctx', or 'static_gen'."""

    summary: object | None = None
    """The underlying :class:`InferenceSummary`.

    Populated for static and agg modes. ``None`` for disagg (which uses its
    own DisaggInferenceSession.run_disagg pipeline and does not expose a
    single summary object with per-op breakdowns at the API boundary).
    """

    per_ops_data: dict | None = None
    """Per-operation latency breakdown (populated when available)."""

    per_ops_source: dict | None = None
    """Per-operation data source breakdown, parallel to ``per_ops_data``.

    Same nested shape as ``per_ops_data`` (one entry per op_name in each
    step), but values are source tags rather than latencies. Example::

        {
            "mix_step": {
                "context_attention (scaled)": "mixed",
                "context_full_moe":           "empirical",
                "context_qkv_gemm":           "silicon",
                ...
            },
            "genonly_step": {
                "generation_attention": "empirical",
                "generation_qkv_gemm":  "silicon",
                ...
            },
        }

    Values are ``"silicon"`` (table data), ``"empirical"`` (formula fallback),
    ``"sol"`` (explicit SOL estimate), or ``"mixed"`` (a sum of values from
    different sources). The ``scheduling`` section of ``per_ops_data`` is
    intentionally omitted here -- those entries are scheduling math / aggregate
    sums, not DB queries.
    """

    kv_cache_warning: str | None = None
    """Warning message for non-fatal memory capacity issues."""

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

    For dense (non-MoE) models, MoE parallelism has no effect on the
    computation, so leave the fields as provided.
    """
    if model_path is not None and not check_is_moe(model_path):
        return moe_tp_size, moe_ep_size

    cfg = ModelConfig(
        tp_size=tp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
    )
    return cfg.resolve_moe_parallelism()


def cli_estimate(
    model_path: str,
    system_name: str,
    *,
    mode: str = "agg",
    backend_name: str = "trtllm",
    backend_version: str | None = None,
    database_mode: str = "SILICON",
    transfer_policy: str | list | None = None,
    isl: int = 1024,
    osl: int = 1024,
    image_height: int = 0,
    image_width: int = 0,
    num_images: int = 1,
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
    engine_step_backend: str | None = None,
    # Static-mode (and shared) extras
    prefix: int = 0,
    nextn: int | str = 0,
    nextn_accepted: float | None = None,
    stride: int = 32,
    # AFD-specific parameters (ignored when mode != 'afd')
    n_a_nodes: int | None = None,
    n_f_nodes: int | None = None,
    a_tp_size: int = 1,
    a_batch_size: int = 128,
    f_moe_ep_size: int = 1,
    num_microbatches: int = 3,
    pipeline_model: str = "optimistic",
    comm_overhead_factor: float = 1.0,
    afd_phase: str = "decode",
    afd_combined_with_pd: bool = True,
    afd_boundary_on_attn: bool = True,
) -> EstimateResult:
    """
    Estimate TTFT, TPOT, and power for a single model/system/config combination.

    Supports aggregated (IFB), disaggregated, and static-batching estimation.

    This is the programmatic equivalent of:
        aiconfigurator cli estimate --model-path ... --system ... --batch-size ...

    Args:
        model_path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or local path.
        system_name: System name (GPU type), e.g., 'h200_sxm', 'h100_sxm'.
        mode: Estimation mode — 'agg' (default), 'disagg', 'afd', or one of the
            static modes ``'static'`` / ``'static_ctx'`` / ``'static_gen'``.
            Static modes run a single-pass ``InferenceSession.run_static`` (no
            IFB scheduling, no rate matching) and are useful for first-order
            latency/memory breakdowns. ``'afd'`` runs Attention-FFN
            Disaggregated estimation (see the afd-only args below).
        backend_name: Backend name ('trtllm', 'sglang', 'vllm'). Default is 'trtllm'.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation
            ('SILICON', 'HYBRID', 'EMPIRICAL', 'SOL'). Default is 'SILICON'.
        isl: Input sequence length. Default is 1024.
        osl: Output sequence length. Default is 1024.
        image_height: Image height in pixels for VL models. Default 0 disables encoder modeling.
        image_width: Image width in pixels for VL models. Default 0 disables encoder modeling.
        num_images: Number of images per request for VL models. Default 1.
        batch_size: Batch size (max concurrent requests, used for agg mode). Default is 128.
        ctx_tokens: Context tokens budget for IFB scheduling (agg mode only).
            Default is None, which uses ``isl`` as the budget.
        tp_size: Tensor parallelism size. Default is 1. Also serves as fallback for
            prefill/decode TP when their specific args are omitted.
        pp_size: Pipeline parallelism size. Default is 1.
        attention_dp_size: Attention data parallelism size. Default is 1.
        moe_tp_size: MoE tensor parallelism size. At least one of ``moe_tp_size``
            or ``moe_ep_size`` is required for MoE models; the missing dimension
            is inferred when possible.
        moe_ep_size: MoE expert parallelism size. At least one of ``moe_tp_size``
            or ``moe_ep_size`` is required for MoE models; the missing dimension
            is inferred when possible.
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
        engine_step_backend: Experimental static latency backend ("python" or "rust").
        prefix: (common) Prefix cache length (subset of ``isl`` already cached).
            Applied to agg, disagg, and all static modes. Default 0.
        nextn: (common) MTP draft length, or ``"auto"`` to use the checkpoint's
            ``num_nextn_predict_layers``. Applied to agg, disagg, and all
            static modes. Default 0 (disabled); MTP is never enabled implicitly.
        nextn_accepted: (common) Average accepted draft tokens per decode step
            (0 <= nextn_accepted <= nextn). Required when the draft depth
            resolves to > 0; never inferred.
        stride: (static-only) Stride used by ``run_static`` to accelerate the
            OSL sweep. Ignored by agg / disagg. Default 32.
        n_a_nodes: (afd-only) Number of A-Worker (attention) nodes. Required
            when ``mode='afd'``.
        n_f_nodes: (afd-only) Number of F-Worker (FFN/MoE) nodes. Required
            when ``mode='afd'``.
        a_tp_size: (afd-only) Attention-side tensor parallelism size. Default 1.
        a_batch_size: (afd-only) Total in-flight batch size per A-Worker before
            micro-batch splitting. Default 128.
        f_moe_ep_size: (afd-only) FFN-side MoE expert parallelism size. Default 1.
        num_microbatches: (afd-only) Number of micro-batches for the ping-pong
            A/F pipeline. Default 3.
        pipeline_model: (afd-only) Ping-pong overlap model — ``'optimistic'``
            (K=3, communication hidden) or ``'conservative'`` (K=2). Default
            ``'optimistic'``.
        comm_overhead_factor: (afd-only) Multiplier applied to cross-pool
            communication latency. Default 1.0.
        afd_phase: (afd-only) Phase AFD is applied to — ``'prefill'``,
            ``'decode'`` (default), or ``'both'``. AFD is orthogonal to P/D
            disaggregation: ``'decode'`` models AFD on decode only, ``'prefill'``
            on the context phase (reports TTFT), and ``'both'`` reports TTFT+TPOT
            with AFD on both phases.
        afd_combined_with_pd: (afd-only) When True (default), combine the
            single-phase AFD estimate with a regular static estimate for the
            other phase (merging TTFT/TPOT, rate-matched throughput, and GPU
            budget). Must be False when ``afd_phase='both'`` (AFD already covers
            both phases). Default True.
        afd_boundary_on_attn: (afd-only) Assign boundary ops (``add_norm_2``,
            ``logits_gemm``) to the A-Worker when True (default); set False to
            place them on the F-Worker. Inverse of the CLI ``--boundary-on-ffn``
            flag.

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
        get_database_view,
        get_latest_database_version,
        get_systems_paths,
        set_systems_paths,
    )

    # Resolve nextn="auto" against the checkpoint before mode dispatch so every
    # estimate path (agg/disagg/static/afd) sees a plain int.
    if nextn == "auto":
        nextn = _resolve_nextn_auto(model_path)
    nextn, nextn_accepted = _normalize_nextn(nextn, nextn_accepted)

    active_systems_paths = None
    if systems_paths is not None:
        previous_systems_paths = get_systems_paths()
        try:
            set_systems_paths(systems_paths)
            active_systems_paths = get_systems_paths()
        finally:
            set_systems_paths(previous_systems_paths)

    def _resolve_version_for(sys_name: str) -> str:
        resolved_version = backend_version
        if resolved_version is None:
            if active_systems_paths is None:
                resolved_version = get_latest_database_version(system=sys_name, backend=backend_name)
            else:
                resolved_version = get_latest_database_version(
                    system=sys_name,
                    backend=backend_name,
                    systems_paths=active_systems_paths,
                )
        if resolved_version is None:
            if database_mode == "SILICON":
                raise ValueError(
                    f"No database found for system={sys_name}, backend={backend_name}. "
                    "Check --systems-paths or available databases."
                )
            resolved_version = "estimate"
        return resolved_version

    def _load_database(sys_name: str):
        resolved_version = _resolve_version_for(sys_name)
        # database_mode is needed at construction, not just at query time: SILICON and
        # HYBRID load declared shared-layer silicon rows, while formula-only modes do not.
        database_kwargs = {
            "allow_missing_data": database_mode != "SILICON",
            "database_mode": database_mode,
            "transfer_policy": transfer_policy,
        }
        if active_systems_paths is not None:
            database_kwargs["systems_paths"] = active_systems_paths
        db = get_database_view(
            sys_name,
            backend_name,
            resolved_version,
            **database_kwargs,
        )
        if db is None:
            raise ValueError(
                f"Failed to load perf database for system={sys_name}, "
                f"backend={backend_name}, version={resolved_version}."
            )
        return db

    if mode in ("static", "static_ctx", "static_gen"):
        resolved_version = _resolve_version_for(system_name)
        return _run_static_estimate(
            static_mode=mode,
            model_path=model_path,
            system_name=system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            image_height=image_height,
            image_width=image_width,
            num_images=num_images,
            batch_size=batch_size,
            prefix=prefix,
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
            nextn=nextn,
            nextn_accepted=nextn_accepted,
            stride=stride,
            engine_step_backend=engine_step_backend,
            load_database=_load_database,
            get_backend=get_backend,
            get_model=get_model,
        )

    if mode == "agg":
        resolved_version = _resolve_version_for(system_name)
        return _run_agg_estimate(
            model_path=model_path,
            system_name=system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            image_height=image_height,
            image_width=image_width,
            num_images=num_images,
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
            engine_step_backend=engine_step_backend,
            prefix=prefix,
            nextn=nextn,
            nextn_accepted=nextn_accepted,
        )
    elif mode == "disagg":
        prefill_resolved_version = _resolve_version_for(system_name)
        decode_system = decode_system_name or system_name
        decode_resolved_version = _resolve_version_for(decode_system)
        resolved_version = (
            prefill_resolved_version
            if prefill_resolved_version == decode_resolved_version
            else f"{prefill_resolved_version}-{decode_resolved_version}"
        )
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
            decode_system_name=decode_system,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            image_height=image_height,
            image_width=image_width,
            num_images=num_images,
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
            engine_step_backend=engine_step_backend,
            prefix=prefix,
            nextn=nextn,
            nextn_accepted=nextn_accepted,
        )
    elif mode == "afd":
        for name, val in [
            ("n_a_nodes", n_a_nodes),
            ("n_f_nodes", n_f_nodes),
        ]:
            if val is None:
                raise ValueError(f"{name} is required for afd mode.")
        if afd_phase not in ("prefill", "decode", "both"):
            raise ValueError(f"afd_phase must be 'prefill', 'decode', or 'both'; got {afd_phase!r}.")
        # ``combined_with_pd`` only makes sense for single-phase AFD: when
        # AFD covers prefill+decode internally there is no separate static
        # pool to combine with. ``AFDConfig.__post_init__`` enforces the
        # same invariant, but checking up-front gives a clearer CLI-level
        # error before any database/model loading happens.
        if afd_phase == "both" and afd_combined_with_pd:
            raise ValueError(
                "afd_combined_with_pd=True is incompatible with afd_phase='both': "
                "'both' means AFD covers prefill+decode internally; there is no "
                "separate static pool to combine with. Pass --no-afd-combined-with-pd, "
                "or pick afd_phase in {'prefill','decode'}."
            )

        resolved_version = _resolve_version_for(system_name)
        afd_result = _run_afd_estimate(
            model_path=model_path,
            system_name=system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            tp_size=tp_size,
            a_tp_size=a_tp_size,
            n_a_nodes=n_a_nodes,
            n_f_nodes=n_f_nodes,
            a_batch_size=a_batch_size,
            f_moe_ep_size=f_moe_ep_size,
            num_microbatches=num_microbatches,
            pipeline_model=pipeline_model,
            comm_overhead_factor=comm_overhead_factor,
            afd_phase=afd_phase,
            afd_combined_with_pd=afd_combined_with_pd,
            afd_boundary_on_attn=afd_boundary_on_attn,
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
            prefix=prefix,
            nextn=nextn,
            nextn_accepted=nextn_accepted,
        )
        # ``phase == "both"`` covers prefill+decode inside AFD; no static
        # complement is needed. When ``combined_with_pd`` is False the
        # caller explicitly asked for the AFD-only estimate of a single
        # phase (the other phase is left unmodeled and the result's GPU
        # budget only counts the AFD pool). In both cases we short-circuit
        # before running the static estimate.
        if afd_phase == "both" or not afd_combined_with_pd:
            return afd_result

        static_mode = "static_gen" if afd_phase == "prefill" else "static_ctx"
        static_result = _run_static_estimate(
            static_mode=static_mode,
            model_path=model_path,
            system_name=system_name,
            backend_name=backend_name,
            resolved_version=resolved_version,
            isl=isl,
            osl=osl,
            image_height=image_height,
            image_width=image_width,
            num_images=num_images,
            batch_size=batch_size,
            prefix=prefix,
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
            nextn=nextn,
            nextn_accepted=nextn_accepted,
            stride=stride,
            engine_step_backend=engine_step_backend,
            load_database=_load_database,
            get_backend=get_backend,
            get_model=get_model,
        )
        if static_result.summary is not None and static_result.summary.check_oom():
            phase_label = "decode" if static_mode == "static_gen" else "prefill"
            raise RuntimeError(
                f"OOM: the model '{model_path}' does not fit in GPU memory on system "
                f"'{system_name}' for the regular {phase_label} phase."
            )
        return _combine_afd_static_estimate_results(
            afd_result=afd_result,
            static_result=static_result,
            afd_phase=afd_phase,
        )
    else:
        raise ValueError(
            f"Unsupported estimate mode: {mode!r}. Use 'agg', 'disagg', 'afd', 'static', 'static_ctx', or 'static_gen'."
        )


def _run_agg_estimate(
    *,
    model_path,
    system_name,
    backend_name,
    resolved_version,
    isl,
    osl,
    image_height,
    image_width,
    num_images,
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
    engine_step_backend=None,
    # Common (also accepted by disagg / static)
    prefix: int = 0,
    nextn: int = 0,
    nextn_accepted: float | None = None,
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
    _apply_nextn(model_config, nextn)
    # Agg workers run context attention → resolve fmha against the perf data
    # before building the model (mirrors the sweep/task_v2 path).
    resolve_context_fmha_by_data(
        model_config, model_path, load_database(system_name), backend_name, is_context_role=True
    )
    resolve_dsv4_moe_arch(model_config, model_path, system_name=system_name, backend_name=backend_name)
    runtime_config = RuntimeConfig(
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        image_height=image_height,
        image_width=image_width,
        num_images_per_request=num_images,
        prefix=prefix,
        engine_step_backend=engine_step_backend,
    )

    model = get_model(model_path, model_config, backend_name)
    database = load_database(system_name)
    backend = get_backend(backend_name)
    session = InferenceSession(model, database, backend)
    speculative_profile = SpeculativeDecodingProfile.from_inputs(nextn, nextn_accepted)
    summary = session.run_agg(
        runtime_config,
        ctx_tokens=ctx_tokens,
        max_seq_len=max_seq_len,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        decode_tokens_per_iteration=speculative_profile.tokens_per_iteration,
    )
    summary = speculative_profile.project_summary(summary, role="agg")

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
        summary=summary,
        per_ops_data=summary.get_per_ops_data(),
        per_ops_source=summary.get_per_ops_source(),
        kv_cache_warning=kv_warning,
    )


def _run_static_estimate(
    *,
    static_mode: str,
    model_path,
    system_name,
    backend_name,
    resolved_version,
    isl,
    osl,
    image_height,
    image_width,
    num_images,
    batch_size,
    prefix,
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
    nextn,
    nextn_accepted,
    stride,
    engine_step_backend,
    load_database,
    get_backend,
    get_model,
) -> EstimateResult:
    """Run a single-pass static-batching estimation.

    Wraps :meth:`InferenceSession.run_static` and produces an ``EstimateResult``
    whose ``raw`` dict follows :data:`aiconfigurator.sdk.common.ColumnsStatic`.
    """
    from aiconfigurator.sdk.config import RuntimeConfig
    from aiconfigurator.sdk.inference_session import InferenceSession

    if static_mode not in ("static", "static_ctx", "static_gen"):
        raise ValueError(
            f"Unsupported static mode: {static_mode!r}. Expected one of 'static', 'static_ctx', 'static_gen'."
        )

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
    _apply_nextn(model_config, nextn)
    # static / static_ctx run context attention; static_gen is generation-only
    # and legitimately keeps fp8 FMHA. Resolve fmha against the perf data accordingly.
    resolve_context_fmha_by_data(
        model_config,
        model_path,
        load_database(system_name),
        backend_name,
        is_context_role=static_mode != "static_gen",
    )
    resolve_dsv4_moe_arch(model_config, model_path, system_name=system_name, backend_name=backend_name)

    runtime_config = RuntimeConfig(
        batch_size=batch_size,
        isl=isl,
        osl=osl,
        image_height=image_height,
        image_width=image_width,
        num_images_per_request=num_images,
        prefix=prefix,
        engine_step_backend=engine_step_backend,
    )

    model = get_model(model_path, model_config, backend_name)
    database = load_database(system_name)
    backend = get_backend(backend_name)
    session = InferenceSession(model, database, backend)
    summary = session.run_static(
        runtime_config=runtime_config,
        mode=static_mode,
        stride=stride,
    )
    projection_role = (
        "prefill" if static_mode == "static_ctx" else ("decode" if static_mode == "static_gen" else "static")
    )
    summary = SpeculativeDecodingProfile.from_inputs(nextn, nextn_accepted).project_summary(
        summary, role=projection_role
    )

    static_warning = None
    if summary.check_oom():
        static_warning = (
            f"OOM: the model '{model_path}' does not fit in GPU memory on system "
            f"'{system_name}' with the given parallelism (tp={tp_size}, pp={pp_size}, "
            f"dp={attention_dp_size}) and batch_size={batch_size}. Reduce batch_size, "
            "increase tp/pp, use quantization, or pick a system with more VRAM per GPU."
        )

    result_dict = summary.get_result_dict()
    if result_dict is None:
        raise RuntimeError("Static estimation produced no results. The configuration may be invalid.")

    return EstimateResult(
        ttft=float(result_dict.get("ttft", 0.0) or 0.0),
        tpot=float(result_dict.get("tpot", 0.0) or 0.0),
        power_w=float(result_dict.get("power_w", 0.0) or 0.0),
        isl=isl,
        osl=osl,
        batch_size=batch_size,
        ctx_tokens=isl,  # static has no IFB budget; expose isl for convenience.
        tp_size=tp_size,
        pp_size=pp_size,
        model_path=model_path,
        system_name=system_name,
        backend_name=backend_name,
        backend_version=resolved_version,
        raw=result_dict,
        mode=static_mode,
        summary=summary,
        per_ops_data=None,
        per_ops_source=None,
        kv_cache_warning=static_warning,
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
    image_height,
    image_width,
    num_images,
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
    engine_step_backend=None,
    # Common (also accepted by agg / static)
    prefix: int = 0,
    nextn: int = 0,
    nextn_accepted: float | None = None,
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
    # Apply common nextn/MTP overrides to *both* prefill and decode worker
    # configs so a single ``--nextn N`` reaches each side of the disagg pair.
    _apply_nextn(prefill_model_config, nextn)
    _apply_nextn(decode_model_config, nextn)
    # Prefill runs context attention → resolve fmha against the perf data. Decode
    # is generation-only and keeps fp8, so it needs no adjustment.
    resolve_context_fmha_by_data(
        prefill_model_config, model_path, load_database(system_name), backend_name, is_context_role=True
    )
    resolve_dsv4_moe_arch(prefill_model_config, model_path, system_name=system_name, backend_name=backend_name)
    resolve_dsv4_moe_arch(
        decode_model_config,
        model_path,
        system_name=decode_system_name or system_name,
        backend_name=backend_name,
    )

    runtime_config = RuntimeConfig(
        isl=isl,
        osl=osl,
        image_height=image_height,
        image_width=image_width,
        num_images_per_request=num_images,
        prefix=prefix,
        engine_step_backend=engine_step_backend,
    )

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
        speculative_profile=SpeculativeDecodingProfile.from_inputs(nextn, nextn_accepted),
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
        per_ops_source=summary.get_per_ops_source(),
    )


def _combine_afd_static_estimate_results(
    *,
    afd_result: EstimateResult,
    static_result: EstimateResult,
    afd_phase: str,
) -> EstimateResult:
    """Merge an AFD phase with the regular static phase into one estimate result."""
    if afd_phase not in ("prefill", "decode"):
        raise ValueError(f"afd_phase must be 'prefill' or 'decode', got {afd_phase!r}.")

    afd_raw = afd_result.raw
    afd_gpus = afd_result.num_total_gpus
    static_gpus = static_result.num_total_gpus
    total_gpus = afd_gpus + static_gpus

    if afd_phase == "decode":
        ttft = static_result.ttft
        tpot = afd_result.tpot
        prefill_seq_s = static_result.seq_per_second
        decode_seq_s = afd_result.seq_per_second
        concurrency = afd_result.concurrency
        prefill_impl = "static_ctx"
        decode_impl = "afd"
        prefill_power = static_result.power_w
        decode_power = afd_result.power_w
    else:
        ttft = afd_result.ttft
        tpot = static_result.tpot
        afd_batch = float(afd_raw.get("b_total", afd_result.concurrency) or 0.0)
        prefill_seq_s = afd_batch / (ttft / 1000.0) if ttft > 0.0 else 0.0
        decode_seq_s = static_result.seq_per_second
        concurrency = static_result.concurrency
        prefill_impl = "afd"
        decode_impl = "static_gen"
        prefill_power = afd_result.power_w
        decode_power = static_result.power_w

    seq_candidates = [v for v in (prefill_seq_s, decode_seq_s) if v > 0.0]
    seq_s = min(seq_candidates) if seq_candidates else 0.0
    tokens_s = seq_s * afd_result.osl
    decode_time = tpot * max(afd_result.osl - 1, 0)
    request_latency = ttft + decode_time
    power_w = (prefill_power * ttft + decode_power * decode_time) / request_latency if request_latency > 0.0 else 0.0

    result_dict = dict(afd_raw)
    result_dict.update(
        {
            "ttft": round(ttft, 3),
            "tpot": round(tpot, 3),
            "request_latency": round(request_latency, 3),
            "seq/s": round(seq_s, 3),
            "tokens/s": round(tokens_s, 2),
            "tokens/s/gpu": round(tokens_s / total_gpus, 2) if total_gpus > 0 else 0.0,
            "tokens/s/user": round(1000.0 / tpot, 2) if tpot > 0.0 else 0.0,
            "concurrency": concurrency,
            "num_total_gpus": total_gpus,
            "memory": round(max(afd_result.memory, static_result.memory), 2),
            "power_w": round(power_w, 3),
            "(p)impl": prefill_impl,
            "(d)impl": decode_impl,
        }
    )

    per_ops_data = dict(afd_result.per_ops_data or {})
    static_summary = static_result.summary
    if static_summary is not None:
        if static_result.mode == "static_ctx":
            prefill_ctx_latency = static_summary.get_context_latency_dict()
            if prefill_ctx_latency:
                per_ops_data["prefill"] = dict(prefill_ctx_latency)
        elif static_result.mode == "static_gen":
            decode_gen_latency = static_summary.get_generation_latency_dict()
            if decode_gen_latency:
                per_ops_data["decode"] = dict(decode_gen_latency)

    per_ops_source = dict(afd_result.per_ops_source or {})

    return EstimateResult(
        ttft=result_dict.get("ttft", 0.0),
        tpot=result_dict.get("tpot", 0.0),
        power_w=result_dict.get("power_w", 0.0),
        isl=afd_result.isl,
        osl=afd_result.osl,
        batch_size=afd_result.batch_size,
        ctx_tokens=afd_result.ctx_tokens,
        tp_size=afd_result.tp_size,
        pp_size=afd_result.pp_size,
        model_path=afd_result.model_path,
        system_name=afd_result.system_name,
        backend_name=afd_result.backend_name,
        backend_version=afd_result.backend_version,
        raw=result_dict,
        mode="afd",
        per_ops_data=per_ops_data,
        per_ops_source=per_ops_source,
        kv_cache_warning=static_result.kv_cache_warning,
    )


def _run_afd_estimate(
    *,
    model_path,
    system_name,
    backend_name,
    resolved_version,
    isl,
    osl,
    tp_size,
    a_tp_size,
    n_a_nodes,
    n_f_nodes,
    a_batch_size,
    f_moe_ep_size,
    num_microbatches,
    pipeline_model,
    comm_overhead_factor,
    afd_phase,
    afd_combined_with_pd,
    afd_boundary_on_attn,
    gemm_quant_mode,
    kvcache_quant_mode,
    fmha_quant_mode,
    moe_quant_mode,
    comm_quant_mode,
    load_database,
    get_backend,
    get_model,
    free_gpu_memory_fraction,
    max_seq_len,
    prefix: int = 0,
    nextn: int = 0,
    nextn_accepted: float | None = None,
) -> EstimateResult:
    """Run AFD (Attention-FFN Disaggregated) estimation.

    AFD is orthogonal to P/D disagg: ``afd_phase`` selects whether this
    single-point estimate covers the prefill phase (TTFT), the decode
    phase (TPOT), or both.  Memory (HBM) bound for each pool is surfaced
    via ``summary.check_oom()``; the raw result dict also carries
    per-pool ``(a)is_oom`` / ``(f)is_oom`` booleans.

    ``gpus_per_node`` is pulled from ``database.system_spec`` and is
    therefore not a parameter; ``f_tp_size`` is derived (Phase 1:
    F-DP=1) inside ``AFDConfig.__post_init__``.
    """
    from aiconfigurator.sdk.config import AFDConfig, RuntimeConfig
    from aiconfigurator.sdk.inference_session import AFDInferenceSession

    # Load the database first so we can read gpus_per_node from the
    # system_spec — the single source of truth that drives BW selection
    # in perf_database / AFDTransfer. Doing this before building the
    # model configs lets us derive f_tp_size = n_f_nodes * gpus_per_node
    # under the Phase 1 F-DP=1 assumption.
    database = load_database(system_name)
    backend = get_backend(backend_name)
    gpus_per_node = int(database.system_spec["node"]["num_gpus_per_node"])

    f_tp_size = n_f_nodes * gpus_per_node

    # Build model configs for A-Worker and F-Worker.
    # A-Worker: attention-only pool; MoE dims are irrelevant but must satisfy
    #   tp_size * attention_dp_size == moe_tp_size * moe_ep_size.
    # F-Worker: FFN/MoE pool; moe_tp_size = tp_f / f_moe_ep_size so the
    #   product constraint holds with attention_dp_size = 1.
    if f_moe_ep_size <= 0 or f_tp_size % f_moe_ep_size != 0:
        raise ValueError(
            f"f_moe_ep_size ({f_moe_ep_size}) must be a positive divisor of "
            f"f_tp_size ({f_tp_size}) (= n_f_nodes * gpus_per_node, "
            f"n_f_nodes={n_f_nodes}, gpus_per_node={gpus_per_node}) so that "
            "f_moe_tp = f_tp / f_moe_ep is an integer."
        )
    f_moe_tp_size = f_tp_size // f_moe_ep_size

    a_model_config = _build_model_config(
        a_tp_size,
        1,
        1,
        a_tp_size,
        1,
        gemm_quant_mode,
        kvcache_quant_mode,
        fmha_quant_mode,
        moe_quant_mode,
        comm_quant_mode,
    )
    f_model_config = _build_model_config(
        f_tp_size,
        1,
        1,
        f_moe_tp_size,
        f_moe_ep_size,
        gemm_quant_mode,
        kvcache_quant_mode,
        fmha_quant_mode,
        moe_quant_mode,
        comm_quant_mode,
    )
    # Pass speculative decode knobs through to A/F model configs. TODO:
    # AFDTransfer still models committed decode-token volume only; recalibrate
    # MTP transfer amplification once the serving semantics are finalized.
    _apply_nextn(a_model_config, nextn)
    _apply_nextn(f_model_config, nextn)
    # The A-worker runs context attention whenever the phase covers prefill
    # ("prefill" or "both"); resolve fmha against the perf data then. The
    # F-worker is FFN/MoE only and never touches FMHA. The decode-phase static
    # complement is a static_ctx call handled in _run_static_estimate.
    resolve_context_fmha_by_data(
        a_model_config, model_path, database, backend_name, is_context_role=afd_phase in ("prefill", "both")
    )
    resolve_dsv4_moe_arch(a_model_config, model_path, system_name=system_name, backend_name=backend_name)
    resolve_dsv4_moe_arch(f_model_config, model_path, system_name=system_name, backend_name=backend_name)

    afd_config = AFDConfig(
        n_a_nodes=n_a_nodes,
        n_f_nodes=n_f_nodes,
        gpus_per_node=gpus_per_node,
        tp_a=a_tp_size,
        # tp_f is derived inside AFDConfig (Phase 1: F-DP=1).
        f_moe_ep_size=f_moe_ep_size,
        a_batch_size=a_batch_size,
        num_microbatches=num_microbatches,
        pipeline_model=pipeline_model,
        comm_overhead_factor=comm_overhead_factor,
        phase=afd_phase,
        combined_with_pd=bool(afd_combined_with_pd),
        boundary_on_attn=bool(afd_boundary_on_attn),
    )
    runtime_config = RuntimeConfig(
        isl=isl,
        osl=osl,
        batch_size=afd_config.n_a_workers * a_batch_size,
        prefix=prefix,
    )

    session = AFDInferenceSession(
        model_path=model_path,
        a_model_config=a_model_config,
        f_model_config=f_model_config,
        database=database,
        backend=backend,
        afd_config=afd_config,
    )
    summary = session.run_afd(
        runtime_config,
        phase=afd_phase,
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        max_seq_len=max_seq_len,
        speculative_profile=SpeculativeDecodingProfile.from_inputs(nextn, nextn_accepted),
    )

    if summary.check_oom():
        raise RuntimeError(
            f"OOM: the model '{model_path}' does not fit in GPU memory on system '{system_name}' "
            f"with the given AFD configuration (phase={afd_phase}). "
            "Try increasing a_tp_size or n_f_nodes (which widens the F-replica "
            "under Phase 1 F-DP=1), reducing batch size, or using a system "
            "with more VRAM per GPU."
        )

    result_dict = summary.get_result_dict()
    if result_dict is None:
        raise RuntimeError("AFD estimation produced no results. The configuration may be invalid.")
    result_dict = dict(result_dict)
    if afd_phase == "both":
        result_dict.setdefault("(p)impl", "afd")
        result_dict.setdefault("(d)impl", "afd")

    return EstimateResult(
        ttft=result_dict.get("ttft", 0.0),
        tpot=result_dict.get("tpot", 0.0),
        power_w=result_dict.get("power_w", 0.0),
        isl=isl,
        osl=osl,
        batch_size=a_batch_size,
        ctx_tokens=0,
        tp_size=a_tp_size,
        pp_size=1,
        model_path=model_path,
        system_name=system_name,
        backend_name=backend_name,
        backend_version=resolved_version,
        raw=result_dict,
        mode="afd",
        per_ops_data=summary.get_per_ops_data(),
        per_ops_source=summary.get_per_ops_source(),
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
