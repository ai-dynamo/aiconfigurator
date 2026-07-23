# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
import os
import sys
import time
from typing import Any

import pandas as pd
import yaml

from aiconfigurator import __version__
from aiconfigurator.cli.estimate_detail_report import detail_requests_time, format_estimate_detail_report
from aiconfigurator.cli.report_and_save import log_final_summary, save_results
from aiconfigurator.cli.utils import merge_experiment_results_by_mode, process_experiment_result
from aiconfigurator.generator.api import (
    add_generator_override_arguments,
    generate_naive_config,
    generator_cli_helper,
    load_generator_overrides_from_args,
)
from aiconfigurator.logging_utils import setup_logging
from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk.config_builders import resolve_nextn_auto
from aiconfigurator.sdk.errors import (
    NoFeasibleConfigError,
    UnsupportedWideepConfigError,
    is_expected_cli_error,
)
from aiconfigurator.sdk.models import check_is_moe
from aiconfigurator.sdk.operations.base import resolve_op_data_path
from aiconfigurator.sdk.speculative import normalize_speculative_decoding
from aiconfigurator.sdk.task_v2 import Task
from aiconfigurator.sdk.utils import ListFlowDumper, get_model_config_from_model_path

logger = logging.getLogger(__name__)


def _latest_support_matrix_version(
    matrix: list[dict[str, str]],
    system: str,
    backend: str,
    model: str | None = None,
    architecture: str | None = None,
) -> str | None:
    """Pick the highest PEP 440 version for the relevant support-matrix rows.

    Matches system and backend case-insensitively. When a model is provided,
    exact-model rows win, then architecture rows. If neither model nor
    architecture matches, return None instead of selecting an unrelated row.
    """
    rows = [
        row for row in matrix if row["System"].lower() == system.lower() and row["Backend"].lower() == backend.lower()
    ]

    if model:
        exact_rows = [row for row in rows if row["HuggingFaceID"].lower() == model.lower()]
        if exact_rows:
            rows = exact_rows
        elif architecture:
            architecture_rows = [row for row in rows if row["Architecture"] == architecture]
            if architecture_rows:
                rows = architecture_rows
            else:
                logger.debug(
                    "No support-matrix rows match model=%s or architecture=%s for system=%s backend=%s",
                    model,
                    architecture,
                    system,
                    backend,
                )
                return None
        else:
            logger.debug(
                "No exact support-matrix rows match model=%s for system=%s backend=%s and no architecture was provided",
                model,
                system,
                backend,
            )
            return None

    versions = [
        (version, parsed)
        for version in {row["Version"] for row in rows}
        if (parsed := common.parse_support_matrix_version(version))
    ]
    if not versions:
        return None
    return max(versions, key=lambda version: version[1])[0]


def _build_common_cli_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--log-level",
        type=str.upper,
        default=None,
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help=("Set the minimum log level. Priority: --log-level > AICONFIGURATOR_LOG_LEVEL > --debug > INFO."),
    )
    common_parser.add_argument(
        "--debug",
        action="store_true",
        help="Deprecated alias for --log-level DEBUG, kept for backward compatibility.",
    )
    common_parser.add_argument(
        "--no-color",
        dest="no_color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    # TODO: maybe move --systems-path here?
    return common_parser


def _build_common_cli_experiments_parser() -> argparse.ArgumentParser:
    # TODO: some arguments might be unused in some modes.
    # Example: --top-n not used for generate mode.
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--save-dir", type=str, default=None, help="Directory to save the results.")
    common_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top configurations to output for each experiment (in exp mode) "
        "or for each mode (agg/disagg) in default mode. Default: 5.",
    )
    common_parser.add_argument(
        "--systems-paths",
        type=str,
        default=None,
        help=(
            "Systems search paths (comma-separated). Use 'default' for the built-in systems path. "
            "Example: default,/opt/aic/systems,/data/aic/systems."
        ),
    )
    common_parser.add_argument(
        "--deployment-target",
        type=str,
        choices=["dynamo-j2", "dynamo-python", "llm-d-helm", "llm-d-kustomize", "fpm"],
        default="dynamo-j2",
        help="Deployment target platform. Options: dynamo-j2 (default, typed Dynamo manifests), "
        "dynamo-python (Dynamo Python config modifiers), llm-d-helm (llm-d Helm values), "
        "llm-d-kustomize (llm-d Kustomize overlays), fpm (reusable resource Pod + run.sh).",
    )
    common_parser.add_argument(
        "--engine-step-backend",
        choices=["python", "rust"],
        default=None,
        help="Experimental static latency backend. Default keeps the existing Python SDK path; "
        "use 'rust' to route static step estimates through the Rust FPM estimator.",
    )
    add_generator_override_arguments(common_parser)
    return common_parser


def _parse_nextn(value: str) -> int | str:
    """argparse type for --nextn: a non-negative integer draft length, or 'auto'
    to use the checkpoint's num_nextn_predict_layers."""
    if value.strip().lower() == "auto":
        return "auto"
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--nextn must be a non-negative integer or 'auto', got {value!r}") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"--nextn must be >= 0 or 'auto', got {parsed}")
    return parsed


def _resolve_and_validate_nextn(args) -> None:
    """Fail fast on inconsistent MTP input; resolve --nextn auto to the checkpoint depth.

    Mutates ``args.nextn`` in place so everything downstream sees a plain int.
    """
    if args.nextn == "auto":
        try:
            resolved = resolve_nextn_auto(args.model_path)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if resolved > 0:
            logger.info(
                "--nextn auto: modeling MTP with nextn=%d from the checkpoint's num_nextn_predict_layers.",
                resolved,
            )
        else:
            logger.info(
                "--nextn auto: checkpoint ships no MTP layers (num_nextn_predict_layers absent or 0); "
                "MTP stays disabled."
            )
        try:
            resolved, args.nextn_accepted = normalize_speculative_decoding(resolved, args.nextn_accepted)
        except ValueError as exc:
            raise SystemExit(
                f"--nextn auto resolved to nextn={resolved} from the checkpoint's num_nextn_predict_layers: {exc}"
            ) from exc
        args.nextn = resolved
        return
    try:
        args.nextn, args.nextn_accepted = normalize_speculative_decoding(args.nextn, args.nextn_accepted)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


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
        "--model-path",
        "--model",
        dest="model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument("--total-gpus", type=int, required=True, help="Total GPUs for deployment.")
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help=(
            "System name (GPU type). Example: "
            "h200_sxm,h100_sxm,h100_pcie,b200_sxm,b300_sxm,gb200,a100_sxm,a100_pcie,l40s,l4,a30,gb300."
        ),
    )
    parser.add_argument(
        "--decode-system",
        type=str,
        default=None,
        help="System name for disagg decode workers. Defaults to --system if omitted.",
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in common.BackendName] + ["auto"],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name. Use a specific backend (trtllm, vllm, sglang) or 'auto' to sweep "
        "across all supported backends for the given system and compare results side by side. "
        "When 'auto' is used, both agg and disagg results are merged across backends and the "
        "globally optimal configuration is selected. Default: trtllm.",
    )
    parser.add_argument(
        "--perf-db-version",
        "--backend-version",
        dest="backend_version",
        type=str,
        default=None,
        help="[expert] Performance-database version used for the simulation/search "
        "(search fidelity). Default: latest measured version; marker-only shared-layer versions "
        "require an explicit value. Alias: --backend-version.",
    )
    parser.add_argument(
        "--database-mode",
        choices=[mode.name for mode in common.DatabaseMode if mode != common.DatabaseMode.SOL_FULL],
        type=str,
        default=common.DatabaseMode.SILICON.name,
        help="Database mode for performance estimation. "
        "SILICON (default): uses silicon-collected data; results are fully reproducible. "
        "HYBRID (recommended for frontier/new models): extends SILICON coverage with "
        "SOL+empirical estimates for configurations not yet in the database — use this "
        "for models released after the last silicon data collection. "
        "EMPIRICAL: SOL+empirical factor only. SOL: theoretical Speed-of-Light only.",
    )
    parser.add_argument(
        "--transfer-policy",
        type=str,
        default=None,
        help="Fine-grained HYBRID/EMPIRICAL transfer control: which empirical transfer kinds "
        "may fill missing data. A preset (off|conservative|balanced|aggressive) or a "
        "comma-separated list of kinds (xshape,xquant,xprofile,xop). "
        "Default: all kinds enabled. Ignored in SILICON mode.",
    )
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length. Default: 4000.")
    parser.add_argument("--osl", type=int, default=1000, help="Output sequence length. Default: 1000.")
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Image height in pixels for vision-language models. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Image width in pixels for vision-language models. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images per request for vision-language models. Default: 1."
    )
    parser.add_argument(
        "--ttft",
        type=float,
        default=2000.0,
        help="Time to first token SLA target in ms. Configurations exceeding this value are "
        "filtered from the pareto and topN configs. (Default: 2000)",
    )
    parser.add_argument(
        "--tpot",
        type=float,
        default=30.0,
        help="Time per output token SLA target in ms. Configurations exceeding this value are "
        "filtered from the topN configs. (Default: 30) \n"
        "**Note: the pareto may still include configs exceeding the given tpot. "
        "Pass --strict-sla to only keep configs that meet the given tpot constraint.**",
    )
    parser.add_argument(
        "--strict-sla",
        action="store_true",
        default=False,
        help="Filter the Pareto frontier and best configs to only SLA-compliant data points "
        "(--ttft + --tpot, or --request-latency). Without this flag, the Pareto frontier "
        "may include configs that exceed --tpot.",
    )
    parser.add_argument(
        "--request-latency",
        type=float,
        default=None,
        help="Optional end-to-end request latency target (ms). Enables request-latency optimization mode.",
    )
    parser.add_argument(
        "--inclusive-tpot",
        action="store_true",
        default=False,
        help=(
            "Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT cost across all output tokens. "
            "Affects terminal output and saved CSV only; SLA filtering always uses inter-token latency."
        ),
    )
    parser.add_argument("--prefix", type=int, default=0, help="Prefix cache length. Default to 0.")
    parser.add_argument(
        "--nextn",
        type=_parse_nextn,
        default=0,
        help="MTP (Multi-Token Prediction) draft length, or 'auto' to use the checkpoint's "
        "num_nextn_predict_layers (absent/0 keeps MTP disabled). When the depth is > 0, enables "
        "speculative decoding in the configuration search and requires --nextn-accepted. "
        "Default: 0 (disabled); MTP is never enabled implicitly when the flag is omitted.",
    )
    parser.add_argument(
        "--nextn-accepted",
        type=float,
        default=None,
        help="Average accepted draft tokens per decode step (0 <= nextn_accepted <= nextn). "
        "Required when --nextn resolves to > 0; there is no built-in acceptance "
        "assumption — use a measured value from your deployment (e.g. the engine's "
        "reported average acceptance length minus 1).",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        default=False,
        help="Enable chunked prefill for finer-grained context token sweep during optimization. "
        "When off (default), context token stride is aligned to ISL for faster sweeping.",
    )
    parser.add_argument(
        "--free-gpu-memory-fraction",
        type=float,
        default=1.0,
        help="Fraction of free GPU memory TRT-LLM allocates for KV cache (default: 1.0). "
        "Used to filter batch sizes that would exceed KV cache capacity.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="TRT-LLM --max_seq_len setting (default: isl + osl). "
        "Controls how many KV blocks TRT-LLM pre-allocates per sequence. "
        "Set this to match your actual deployment for accurate KV cache capacity filtering.",
    )
    parser.add_argument(
        "--enable-wideep",
        action="store_true",
        default=False,
        help="Enable Wide Expert Parallelism (WideEP) for MoE models. "
        "When set, MoE models use EP-only parallelism with deepep_moe backend. "
        "Applies to both DeepSeek and Qwen3-235B on SGLang.",
    )
    parser.add_argument(
        "--moe-backend",
        type=str,
        choices=["deepep_moe", "megamoe"],
        default=None,
        help="Explicit SGLang MoE backend. Use 'megamoe' to model DeepSeek-V4 MegaMoE on Blackwell.",
    )


def _add_experiments_mode_arguments(parser):
    parser.add_argument(
        "--yaml-path",
        type=str,
        required=True,
        help="Path to a YAML file containing experiment definitions.",
    )
    parser.add_argument(
        "--inclusive-tpot",
        action="store_true",
        default=False,
        help=(
            "Report TPOT as (ttft + tpot * (osl - 1)) / osl, spreading TTFT cost across all output tokens. "
            "Affects terminal output and saved CSV only; SLA filtering always uses inter-token latency."
        ),
    )


def _add_generate_mode_arguments(parser):
    """Add arguments for the generate mode (naive config generation)."""
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument(
        "--total-gpus",
        type=int,
        required=True,
        help="Total GPUs for deployment.",
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help=(
            "System name (GPU type). Example: "
            "h200_sxm,h100_sxm,h100_pcie,b200_sxm,b300_sxm,gb200,a100_sxm,a100_pcie,l40s,l4,a30,gb300."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in common.BackendName],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name (default: trtllm).",
    )


def _add_estimate_mode_arguments(parser):
    """Add arguments for the estimate mode (single-point TTFT/TPOT/power estimation)."""
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument(
        "--estimate-mode",
        choices=["agg", "disagg", "afd", "static", "static_ctx", "static_gen"],
        type=str,
        default="agg",
        help="Estimation mode: 'agg' (default, IFB), 'disagg' (separate prefill/decode workers), "
        "'afd' (attention-FFN disaggregated), or one of the static modes "
        "'static' / 'static_ctx' / 'static_gen' for a single-pass, no-IFB latency/memory "
        "breakdown.",
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help=(
            "System name (GPU type). Example: "
            "h200_sxm,h100_sxm,h100_pcie,b200_sxm,b300_sxm,gb200,a100_sxm,a100_pcie,l40s,l4,a30,gb300."
        ),
    )
    parser.add_argument(
        "--decode-system",
        type=str,
        default=None,
        help="System name for disagg decode workers. Defaults to --system if omitted.",
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in common.BackendName],
        type=str,
        default=common.BackendName.trtllm.value,
        help="Backend name (default: trtllm).",
    )
    parser.add_argument(
        "--perf-db-version",
        "--backend-version",
        dest="backend_version",
        type=str,
        default=None,
        help="[expert] Performance-database version used for the simulation/search "
        "(search fidelity). Default: latest measured version; marker-only shared-layer versions "
        "require an explicit value. Alias: --backend-version.",
    )
    parser.add_argument("--isl", type=int, default=1024, help="Input sequence length. Default: 1024.")
    parser.add_argument("--osl", type=int, default=1024, help="Output sequence length. Default: 1024.")
    parser.add_argument(
        "--image-height",
        type=int,
        default=0,
        help="Image height in pixels for vision-language models. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=0,
        help="Image width in pixels for vision-language models. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images per request for vision-language models. Default: 1."
    )
    parser.add_argument(
        "--batch-size",
        "--bs",
        dest="batch_size",
        type=int,
        default=128,
        help="Batch size (max concurrent requests, used for agg/static). Default: 128. Alias: --bs.",
    )
    parser.add_argument(
        "--ctx-tokens",
        type=int,
        default=None,
        help="Context tokens budget for IFB scheduling (agg only). Default: same as ISL.",
    )

    # Shared parallelism defaults (also used as fallback for prefill/decode-specific args)
    parser.add_argument(
        "--tp-size",
        "--tp",
        dest="tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size. Default: 1. Alias: --tp.",
    )
    parser.add_argument(
        "--pp-size",
        "--pp",
        dest="pp_size",
        type=int,
        default=1,
        help="Pipeline parallelism size. Default: 1. Alias: --pp.",
    )
    parser.add_argument(
        "--attention-dp-size",
        "--dp",
        dest="attention_dp_size",
        type=int,
        default=1,
        help="Attention data parallelism size. Default: 1. Alias: --dp.",
    )
    parser.add_argument(
        "--moe-tp-size",
        "--etp",
        dest="moe_tp_size",
        type=int,
        default=None,
        help="MoE tensor parallelism size. Alias: --etp.",
    )
    parser.add_argument(
        "--moe-ep-size",
        "--ep",
        dest="moe_ep_size",
        type=int,
        default=None,
        help="MoE expert parallelism size. Alias: --ep.",
    )

    # Disagg: prefill-specific overrides (fall back to shared args when None)
    parser.add_argument(
        "--prefill-tp-size",
        "--p-tp",
        dest="prefill_tp_size",
        type=int,
        default=None,
        help="Prefill TP size (disagg). Defaults to --tp-size. Alias: --p-tp.",
    )
    parser.add_argument(
        "--prefill-pp-size",
        "--p-pp",
        dest="prefill_pp_size",
        type=int,
        default=None,
        help="Prefill PP size (disagg). Defaults to --pp-size. Alias: --p-pp.",
    )
    parser.add_argument(
        "--prefill-attention-dp-size",
        "--p-dp",
        dest="prefill_attention_dp_size",
        type=int,
        default=None,
        help="Prefill attention DP size (disagg). Defaults to --attention-dp-size. Alias: --p-dp.",
    )
    parser.add_argument(
        "--prefill-moe-tp-size",
        "--p-etp",
        dest="prefill_moe_tp_size",
        type=int,
        default=None,
        help="Prefill MoE TP size (disagg). Defaults to --moe-tp-size. Alias: --p-etp.",
    )
    parser.add_argument(
        "--prefill-moe-ep-size",
        "--p-ep",
        dest="prefill_moe_ep_size",
        type=int,
        default=None,
        help="Prefill MoE EP size (disagg). Defaults to --moe-ep-size. Alias: --p-ep.",
    )
    parser.add_argument(
        "--prefill-batch-size",
        "--p-bs",
        dest="prefill_batch_size",
        type=int,
        default=None,
        help="Prefill batch size (disagg). Required for disagg mode. Alias: --p-bs.",
    )
    parser.add_argument(
        "--prefill-num-workers",
        "--p-workers",
        dest="prefill_num_workers",
        type=int,
        default=None,
        help="Number of prefill workers (disagg). Required for disagg mode. Alias: --p-workers.",
    )

    # Disagg: decode-specific overrides (fall back to shared args when None)
    parser.add_argument(
        "--decode-tp-size",
        "--d-tp",
        dest="decode_tp_size",
        type=int,
        default=None,
        help="Decode TP size (disagg). Defaults to --tp-size. Alias: --d-tp.",
    )
    parser.add_argument(
        "--decode-pp-size",
        "--d-pp",
        dest="decode_pp_size",
        type=int,
        default=None,
        help="Decode PP size (disagg). Defaults to --pp-size. Alias: --d-pp.",
    )
    parser.add_argument(
        "--decode-attention-dp-size",
        "--d-dp",
        dest="decode_attention_dp_size",
        type=int,
        default=None,
        help="Decode attention DP size (disagg). Defaults to --attention-dp-size. Alias: --d-dp.",
    )
    parser.add_argument(
        "--decode-moe-tp-size",
        "--d-etp",
        dest="decode_moe_tp_size",
        type=int,
        default=None,
        help="Decode MoE TP size (disagg). Defaults to --moe-tp-size. Alias: --d-etp.",
    )
    parser.add_argument(
        "--decode-moe-ep-size",
        "--d-ep",
        dest="decode_moe_ep_size",
        type=int,
        default=None,
        help="Decode MoE EP size (disagg). Defaults to --moe-ep-size. Alias: --d-ep.",
    )
    parser.add_argument(
        "--decode-batch-size",
        "--d-bs",
        dest="decode_batch_size",
        type=int,
        default=None,
        help="Decode batch size (disagg). Required for disagg mode. Alias: --d-bs.",
    )
    parser.add_argument(
        "--decode-num-workers",
        "--d-workers",
        dest="decode_num_workers",
        type=int,
        default=None,
        help="Number of decode workers (disagg). Required for disagg mode. Alias: --d-workers.",
    )

    # AFD (Attention-FFN Disaggregation) specific parameters
    parser.add_argument(
        "--n-a-nodes",
        type=int,
        default=None,
        help="Number of A-Worker (attention) nodes (AFD mode). Required for afd mode.",
    )
    parser.add_argument(
        "--n-f-nodes",
        type=int,
        default=None,
        help="Number of F-Worker (FFN/MoE) nodes (AFD mode). Required for afd mode.",
    )
    parser.add_argument(
        "--a-tp-size",
        type=int,
        default=1,
        help="Attention-side tensor parallelism (AFD mode). Default: 1.",
    )
    parser.add_argument(
        "--a-batch-size",
        type=int,
        default=128,
        help=("Total in-flight batch size per A-Worker before microbatch splitting (AFD mode). Default: 128."),
    )
    parser.add_argument(
        "--f-moe-ep-size",
        type=int,
        default=1,
        help="FFN-side MoE expert parallelism (AFD mode). Default: 1.",
    )
    parser.add_argument(
        "--num-microbatches",
        type=int,
        default=3,
        help="Number of micro-batches for ping-pong pipeline (AFD mode). Default: 3.",
    )
    parser.add_argument(
        "--pipeline-model",
        choices=["optimistic", "conservative"],
        type=str,
        default="optimistic",
        help="Pipeline model for AFD: 'optimistic' (K=3, comm hidden) or 'conservative' (K=2). Default: optimistic.",
    )
    parser.add_argument(
        "--comm-overhead-factor",
        type=float,
        default=1.0,
        help="Communication overhead multiplier (AFD mode). Default: 1.0.",
    )
    parser.add_argument(
        "--afd-phase",
        choices=["prefill", "decode", "both"],
        type=str,
        default="decode",
        help="Which phase AFD is applied to. AFD is orthogonal to P/D disaggregation: "
        "'decode' (default) models AFD on decode only (existing behavior), 'prefill' "
        "models AFD on the context phase and reports TTFT, and 'both' reports TTFT+TPOT "
        "for a deployment where AFD is used on both phases.",
    )
    parser.add_argument(
        "--afd-combined-with-pd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Combine the single-phase AFD estimate with a regular static "
        "estimate for the other phase. When enabled (default), --afd-phase=decode "
        "also runs a static prefill estimate (and vice versa), merging TTFT/TPOT, "
        "throughput (rate-matched on min seq/s), and GPU budget into one result. "
        "Pass --no-afd-combined-with-pd to report only the AFD phase. Required to "
        "be off when --afd-phase=both (AFD covers both phases internally).",
    )
    parser.add_argument(
        "--boundary-on-ffn",
        action="store_true",
        default=False,
        help="Assign boundary ops (add_norm_2, logits_gemm) to F-Worker. Default is A-Worker; pass this flag to flip.",
    )

    # Quantization
    parser.add_argument(
        "--gemm-quant-mode",
        choices=[m.name for m in common.GEMMQuantMode],
        type=str,
        default=None,
        help="GEMM quantization mode. Auto-inferred from model config if omitted.",
    )
    parser.add_argument(
        "--kvcache-quant-mode",
        choices=[m.name for m in common.KVCacheQuantMode],
        type=str,
        default=None,
        help="KV cache quantization mode. Auto-inferred from model config if omitted.",
    )
    parser.add_argument(
        "--fmha-quant-mode",
        choices=[m.name for m in common.FMHAQuantMode],
        type=str,
        default=None,
        help="FMHA quantization mode. Auto-inferred from model config if omitted.",
    )
    parser.add_argument(
        "--moe-quant-mode",
        choices=[m.name for m in common.MoEQuantMode],
        type=str,
        default=None,
        help="MoE quantization mode. Auto-inferred from model config if omitted.",
    )
    parser.add_argument(
        "--comm-quant-mode",
        choices=[m.name for m in common.CommQuantMode],
        type=str,
        default=None,
        help="Communication quantization mode. Auto-inferred (default: half) if omitted.",
    )
    parser.add_argument(
        "--database-mode",
        choices=[mode.name for mode in common.DatabaseMode if mode != common.DatabaseMode.SOL_FULL],
        type=str,
        default=common.DatabaseMode.SILICON.name,
        help="Database mode for performance estimation. "
        "SILICON (default): uses silicon-collected data; results are fully reproducible. "
        "HYBRID (recommended for frontier/new models): extends SILICON coverage with "
        "SOL+empirical estimates for configurations not yet in the database — use this "
        "for models released after the last silicon data collection. "
        "EMPIRICAL: SOL+empirical factor only. SOL: theoretical Speed-of-Light only.",
    )
    parser.add_argument(
        "--transfer-policy",
        type=str,
        default=None,
        help="Fine-grained HYBRID/EMPIRICAL transfer control: which empirical transfer kinds "
        "may fill missing data. A preset (off|conservative|balanced|aggressive) or a "
        "comma-separated list of kinds (xshape,xquant,xprofile,xop). "
        "Default: all kinds enabled. Ignored in SILICON mode.",
    )
    parser.add_argument(
        "--detail",
        type=str,
        default=None,
        help="Comma-separated breakdown sections to print after the summary box. "
        "Choices: summary, memory, time, energy, source, all. "
        "Example: --detail memory,time. Default: no extra detail. "
        "Use 'all' to print every section.",
    )
    # Common workload extras — apply to agg / disagg / static / static_ctx / static_gen.
    parser.add_argument(
        "--prefix",
        type=int,
        default=0,
        help="(common) Prefix cache length (subset of ISL already cached per request). Default: 0. "
        "Applied to agg, disagg, and all static modes.",
    )
    parser.add_argument(
        "--nextn",
        type=_parse_nextn,
        default=0,
        help="(common) MTP draft length (compute cost side), or 'auto' to use the checkpoint's "
        "num_nextn_predict_layers. Default: 0 (disabled); MTP is never enabled implicitly. "
        "Applied to agg, disagg, and all static modes. Requires --nextn-accepted when the "
        "resolved depth is > 0.",
    )
    parser.add_argument(
        "--nextn-accepted",
        type=float,
        default=None,
        help="(common) Average accepted draft tokens per decode step "
        "(0 <= nextn_accepted <= nextn). Required when --nextn resolves to > 0; "
        "there is no built-in acceptance assumption — use a measured value from "
        "your deployment.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="(static-only) Stride used by run_static to accelerate the OSL sweep. "
        "Ignored for agg / disagg modes. Default: 32.",
    )
    parser.add_argument(
        "--free-gpu-memory-fraction",
        type=float,
        default=0.9,
        help="Fraction of free GPU memory available for KV cache (default: 0.9). "
        "Used to estimate max concurrent sequences and warn when batch_size "
        "exceeds KV cache capacity.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="TRT-LLM --max_seq_len setting (default: isl + osl). "
        "Controls how many KV blocks TRT-LLM pre-allocates per sequence. "
        "Set this to match your actual deployment to get an accurate KV cache capacity warning.",
    )


def _add_support_mode_arguments(parser):
    """Add arguments for the support mode (support matrix check)."""
    parser.add_argument(
        "--model-path",
        "--model",
        dest="model_path",
        type=_validate_model_path,
        required=True,
        help="Model path: HuggingFace model path (e.g., 'Qwen/Qwen3-32B') or "
        "local path to directory containing config.json.",
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="System name (GPU type) or 'all' for a matrix view across every system. "
        "Example: h200_sxm, h100_sxm, h100_pcie, b200_sxm, b300_sxm, gb200, a100_sxm, a100_pcie, l40s, l4, a30, gb300.",
    )
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in common.BackendName] + ["all"],
        type=str,
        default=None,
        help="Backend name to filter by, or 'all' for all backends. "
        "Defaults to 'all' when --system is 'all', otherwise 'trtllm'.",
    )
    parser.add_argument(
        "--perf-db-version",
        "--backend-version",
        dest="backend_version",
        type=str,
        default=None,
        help="Optional backend / perf-db version to filter by. Alias: --backend-version.",
    )


_USAGE_EXAMPLES = """
Examples:
# Sweep across all backends for Dynamo 1.0.0
aiconfigurator cli default --model Qwen/Qwen3-32B-FP8 \\
    --backend auto \\
    --top-n 3 \\
    --total-gpus 8 --system h200_sxm \\
    --ttft 600 --tpot 50 --isl 4000 --osl 500 \\
    --dynamo-version 0.7.1 \\
    --generator-set K8sConfig.k8s_pvc_name=$YOUR_PVC_NAME \\
    --generator-set K8sConfig.k8s_namespace=$YOUR_NAMESPACE \\
    --save-dir results

# Sweep against trtllm 1.2.0rc5 perf data but generate config matching trtllm 1.2.0rc6
aiconfigurator cli default --model Qwen/Qwen3-32B-FP8 \\
    --backend trtllm \\
    --total-gpus 8 --system h200_sxm \\
    --perf-db-version 1.2.0rc5 \\
    --config-template-version 1.2.0rc6 \\
    --save-dir results
"""


def configure_parser(parser):
    common_cli_parser = _build_common_cli_parser()
    common_cli_experiments_parser = _build_common_cli_experiments_parser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    default_parser = subparsers.add_parser(
        "default",
        parents=[common_cli_parser, common_cli_experiments_parser],
        help="Run the default agg vs disagg comparison.",
        description="Run the default agg vs disagg comparison.",
        epilog=_USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_default_mode_arguments(default_parser)

    help_text = "Run one or more experiments defined in a YAML file. Example: example.yaml"
    # an example yaml for demonstration
    example_yaml_path = os.path.join(os.path.dirname(__file__), "example.yaml")
    with open(example_yaml_path) as f:
        example_yaml = yaml.safe_load(f)
    description = help_text + "\n\nExample:\n\n" + yaml.dump(example_yaml, indent=2, Dumper=ListFlowDumper)

    experiments_parser = subparsers.add_parser(
        "exp",
        parents=[common_cli_parser, common_cli_experiments_parser],
        help=help_text,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_experiments_mode_arguments(experiments_parser)

    # Generate mode - naive config without sweeping
    generate_parser = subparsers.add_parser(
        "generate",
        parents=[common_cli_parser, common_cli_experiments_parser],
        help="Generate naive agg config without SLA optimization (no sweeping).",
        description=(
            "Generate a working agg configuration without running the parameter sweep. "
            "Calculates the smallest TP that fits the model in memory "
            "(TP * VRAM/GPU > 1.5 * model_weight). No SLA optimization is performed."
        ),
    )
    _add_generate_mode_arguments(generate_parser)

    # Estimate mode - single-point performance estimation
    estimate_parser = subparsers.add_parser(
        "estimate",
        parents=[common_cli_parser, common_cli_experiments_parser],
        help="Estimate TTFT, TPOT, and power for a single model/system/config.",
        description=(
            "Run a single-point performance estimation to predict TTFT (time to first token), "
            "TPOT (time per output token), and power consumption for a given model, system, "
            "and configuration. No parameter sweep or SLA optimization is performed."
        ),
    )
    _add_estimate_mode_arguments(estimate_parser)

    # Support mode - support matrix check
    support_parser = subparsers.add_parser(
        "support",
        parents=[common_cli_parser],
        help="(Optional) Check if AIC supports the model/hardware combo for (agg, disagg).",
        description="Optional pre-flight check to verify support for a specific model and system "
        "combination using the support matrix. You can skip this and run 'cli default' directly. "
        "Use --system all for a consolidated matrix view across all systems and backends.",
    )
    _add_support_mode_arguments(support_parser)


def _get_system_data_root(system_name: str) -> str | None:
    """Resolve a system's perf-data root (the dir holding either
    <family>/<backend>/<version> or legacy <backend>/<version> subtrees)."""
    for systems_root in perf_database.get_systems_paths():
        system_yaml = os.path.join(systems_root, f"{system_name}.yaml")
        if not os.path.isfile(system_yaml):
            continue
        with open(system_yaml, encoding="utf-8") as fh:
            system_spec = yaml.safe_load(fh) or {}
        data_dir = system_spec.get("data_dir")
        if not data_dir:
            return None
        return os.path.join(systems_root, data_dir)
    return None


def _get_backend_data_path(system_name: str, backend_name: str, backend_version: str, op_filename: str) -> str | None:
    """Resolve one perf-data file's on-disk path for (system, backend, version),
    across both the family-first and legacy tree layouts (see resolve_op_data_path)."""
    system_data_root = _get_system_data_root(system_name)
    if system_data_root is None:
        return None
    return resolve_op_data_path(system_data_root, backend_name, backend_version, op_filename)


_SGLANG_DEEPEP_REQUIRED_FILES = (
    common.PerfDataFilename.wideep_deepep_normal.value,
    common.PerfDataFilename.wideep_deepep_ll.value,
)


def _database_mode_requires_declared_perf_database(database_mode: str | None) -> bool:
    return (database_mode or "").upper() in {
        common.DatabaseMode.SILICON.name,
        common.DatabaseMode.HYBRID.name,
    }


def _sglang_deepep_perf_data_skip_reason(
    system_name: str,
    decode_system_name: str | None,
    backend_version: str | None,
) -> str | None:
    """Return a concise skip reason when optional SGLang DeepEP data is absent."""
    missing_paths: list[str] = []
    missing_versions: list[str] = []

    systems_to_check = [system_name]
    if decode_system_name and decode_system_name != system_name:
        systems_to_check.append(decode_system_name)

    for system_to_check in systems_to_check:
        resolved_version = backend_version or perf_database.get_latest_database_version(
            system=system_to_check,
            backend=common.BackendName.sglang.value,
        )
        if resolved_version is None:
            missing_versions.append(f"{system_to_check}/{common.BackendName.sglang.value}")
            continue

        for filename in _SGLANG_DEEPEP_REQUIRED_FILES:
            resolved_path = _get_backend_data_path(
                system_to_check, common.BackendName.sglang.value, resolved_version, filename
            )
            if resolved_path is None or not os.path.isfile(resolved_path):
                missing_paths.append(
                    resolved_path
                    or f"{system_to_check}/{common.BackendName.sglang.value}/{resolved_version}/{filename}"
                )

    if missing_versions:
        return "no database version available for " + ", ".join(missing_versions)
    if missing_paths:
        return "missing required DeepEP perf data: " + ", ".join(missing_paths)
    return None


def _ensure_backend_version_available(
    system_name: str,
    backend_name: str,
    backend_version: str | None = None,
) -> None:
    """
    Validate that the backend is supported for the given system and version.

    Args:
        system_name: System name (e.g., 'gb200_sxm')
        backend_name: Backend name (e.g., 'vllm')
        backend_version: Backend database version. Default is None, which means latest version.
    Raises:
        SystemExit: If the backend is not supported for the given system and version.
    """
    supported = perf_database.get_supported_databases()
    backends = supported.get(system_name, {}).keys()
    if backend_name not in backends:
        logger.error(
            "Backend %s is not supported for system %s. Supported backends: %s",
            backend_name,
            system_name,
            ", ".join(sorted(backends)),
        )
        raise SystemExit(1)

    versions = supported.get(system_name, {}).get(backend_name, [])
    if backend_version is None or backend_version in versions:
        return

    systems_paths = perf_database.get_systems_paths()
    systems_paths_display = ", ".join(systems_paths) if systems_paths else "<none>"

    logger.error(
        "No perf database for system=%s backend=%s version=%s.",
        system_name,
        backend_name,
        backend_version,
    )
    system_data_root = _get_system_data_root(system_name)
    if system_data_root:
        logger.error(
            "Searched: %s (backend=%s, version=%s; both family-first <family>/<backend>/<version> "
            "and legacy <backend>/<version> layouts)",
            system_data_root,
            backend_name,
            backend_version,
        )
    logger.error("Configured systems paths: %s", systems_paths_display)
    if versions:
        logger.error("Available versions: %s", ", ".join(versions))
        logger.error(
            "Fix: switch --backend-version to one of the available versions, "
            "remove --backend-version to use latest, "
            "or add a declared version directory with %s (legacy: %s) when this version "
            "intentionally reuses shared-layer data.",
            perf_database.REUSE_YAML_MARKER,
            perf_database.SHARED_LAYER_REUSE_MARKER,
        )
    else:
        logger.error("Available versions: none")
        logger.error(
            "Fix: no database was found for system=%s backend=%s in current --systems-paths.",
            system_name,
            backend_name,
        )
        logger.error(
            "Try adding a path that contains this database via --systems-paths. "
            'Example: --systems-paths "default,/path/to/extra/systems".'
        )
    raise SystemExit(1)


def build_default_tasks(
    model_path: str,
    total_gpus: int,
    system: str,
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
    enable_chunked_prefill: bool = False,
    free_gpu_memory_fraction: float | None = None,
    max_seq_len: int | None = None,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    engine_step_backend: str | None = None,
) -> dict[str, Task]:
    """Build agg and disagg task configs for default mode comparison.

    Args:
        model_path: HuggingFace model path or local path.
        total_gpus: Total number of GPUs for deployment.
        system: System name (GPU type).
        decode_system: System for disagg decode workers. Defaults to `system`.
        backend: Backend name ('trtllm', 'sglang', 'vllm', 'auto').
            Use 'auto' to sweep across all backends.
        backend_version: Backend database version. Default is latest.
        database_mode: Database mode for performance estimation.
        isl: Input sequence length.
        osl: Output sequence length.
        ttft: Time to first token target in ms.
        tpot: Time per output token target in ms.
        request_latency: Optional end-to-end request latency target (ms).
        prefix: Prefix cache length.
        nextn: MTP draft length, or ``"auto"`` to use the checkpoint's
            ``num_nextn_predict_layers`` (absent/0 keeps MTP disabled).
            Default 0 (disabled); never enabled implicitly.
        nextn_accepted: Average accepted draft tokens per decode step
            (0 <= nextn_accepted <= nextn). Required when the draft depth
            resolves to > 0; never inferred.
        enable_chunked_prefill: Whether to enable chunked prefill for finer context token sweep.
        enable_wideep: Whether to enable Wide Expert Parallelism (WideEP) for MoE models.
        moe_backend: Explicit SGLang MoE backend override.
        engine_step_backend: Experimental static latency backend ("python" or "rust").

    Returns:
        Dict with Task objects. When backend='auto', returns 6 configs
        (agg_trtllm, agg_vllm, agg_sglang, disagg_trtllm, disagg_vllm, disagg_sglang).
        Otherwise returns 2 configs ('agg' and 'disagg').
    """
    decode_system = decode_system or system
    # Expand "auto" backend to all available backends
    backends_to_sweep = [b.value for b in common.BackendName] if backend == "auto" else [backend]
    if backend == "auto" and moe_backend == "megamoe":
        backends_to_sweep = [common.BackendName.sglang.value]

    if backend == "auto":
        supported = perf_database.get_supported_databases()
        available = []
        requires_declared_perf_database = _database_mode_requires_declared_perf_database(database_mode)
        for backend_name in backends_to_sweep:
            sys_backends = supported.get(system, {})
            decode_backends = supported.get(decode_system, {}) if decode_system != system else sys_backends
            if not requires_declared_perf_database:
                sys_versions = sys_backends.get(backend_name, [])
                decode_versions = decode_backends.get(backend_name, [])
                if not sys_versions or (decode_system != system and not decode_versions):
                    logger.warning(
                        "No measured database for backend %s on system=%s%s; including it for %s estimates.",
                        backend_name,
                        system,
                        f", decode_system={decode_system}" if decode_system != system else "",
                        database_mode,
                    )
                elif backend_version is not None and (
                    backend_version not in sys_versions
                    or (decode_system != system and backend_version not in decode_versions)
                ):
                    logger.warning(
                        "No measured database version %s for backend %s on system=%s%s; including it for %s estimates.",
                        backend_version,
                        backend_name,
                        system,
                        f", decode_system={decode_system}" if decode_system != system else "",
                        database_mode,
                    )
                available.append(backend_name)
                continue
            if backend_name not in sys_backends:
                logger.warning("Skipping backend %s: not supported for system %s.", backend_name, system)
                continue
            if decode_system != system and backend_name not in decode_backends:
                logger.warning("Skipping backend %s: not supported for decode system %s.", backend_name, decode_system)
                continue
            if backend_version is not None:
                if backend_version not in sys_backends.get(backend_name, []):
                    logger.warning(
                        "Skipping backend %s: version %s not available for system %s.",
                        backend_name,
                        backend_version,
                        system,
                    )
                    continue
                if decode_system != system and backend_version not in decode_backends.get(backend_name, []):
                    logger.warning(
                        "Skipping backend %s: version %s not available for decode system %s.",
                        backend_name,
                        backend_version,
                        decode_system,
                    )
                    continue
            available.append(backend_name)
        if not available:
            logger.error(
                "No backends available for system %s. Supported backends: %s",
                system,
                ", ".join(sorted(supported.get(system, {}).keys())),
            )
            raise SystemExit(1)
        backends_to_sweep = available
    elif _database_mode_requires_declared_perf_database(database_mode):
        _ensure_backend_version_available(system, backend, backend_version)
        if decode_system != system:
            _ensure_backend_version_available(decode_system, backend, backend_version)
    else:
        supported = perf_database.get_supported_databases()
        for role, sys_name in (("prefill", system), ("decode", decode_system)):
            versions = supported.get(sys_name, {}).get(backend, [])
            if backend_version is not None and versions and backend_version not in versions:
                logger.warning(
                    "No measured database version %s for %s system=%s backend=%s; using %s estimates.",
                    backend_version,
                    role,
                    sys_name,
                    backend,
                    database_mode,
                )
            elif not versions:
                logger.warning(
                    "No measured database for %s system=%s backend=%s; using estimate-only version with %s mode.",
                    role,
                    sys_name,
                    backend,
                    database_mode,
                )

    # v2 Task uses a flat schema. Global (both-mode) fields stay top-level;
    # worker-spec fields are top-level for agg but must be fanned out to
    # prefill_* / decode_* for disagg (v2 forbids shared top-level worker fields).
    global_kwargs: dict[str, Any] = {
        "isl": isl,
        "osl": osl,
        "prefix": prefix,
        "ttft": ttft,
        "tpot": tpot,
        "request_latency": request_latency,
        "total_gpus": total_gpus,
        "database_mode": database_mode,
        "transfer_policy": transfer_policy,
        "free_gpu_memory_fraction": free_gpu_memory_fraction,
        "max_seq_len": max_seq_len,
        "engine_step_backend": engine_step_backend,
    }
    if nextn == "auto" or (isinstance(nextn, int) and nextn > 0):
        global_kwargs["nextn"] = nextn
        global_kwargs["nextn_accepted"] = nextn_accepted

    if image_height or image_width or (num_images and num_images != 1):
        global_kwargs["image_height"] = image_height
        global_kwargs["image_width"] = image_width
        global_kwargs["num_images_per_request"] = num_images

    def _sglang_moe_backend_override(backend_name: str) -> str | None:
        if backend_name != common.BackendName.sglang.value:
            return None
        # Auto-set moe_backend for SGLang wideep to preserve existing UI parity.
        return moe_backend or ("deepep_moe" if enable_wideep else None)

    def _make_agg(backend_name: str, moe_backend_value: str | None) -> Task:
        return Task(
            serving_mode="agg",
            model_path=model_path,
            system_name=system,
            backend_name=backend_name,
            backend_version=backend_version,
            enable_wideep=enable_wideep,
            enable_chunked_prefill=enable_chunked_prefill,
            moe_backend=moe_backend_value,
            **global_kwargs,
        )

    def _make_disagg(backend_name: str, moe_backend_value: str | None) -> Task:
        # Fan out the shared worker spec to both roles (v2 disagg forbids top-level worker fields).
        return Task(
            serving_mode="disagg",
            prefill_model_path=model_path,
            decode_model_path=model_path,
            prefill_system_name=system,
            decode_system_name=decode_system,
            prefill_backend_name=backend_name,
            decode_backend_name=backend_name,
            prefill_backend_version=backend_version,
            decode_backend_version=backend_version,
            prefill_enable_wideep=enable_wideep,
            decode_enable_wideep=enable_wideep,
            prefill_enable_chunked_prefill=enable_chunked_prefill,
            moe_backend=moe_backend_value,
            **global_kwargs,
        )

    tasks: dict[str, Task] = {}
    is_moe_model = check_is_moe(model_path)

    for backend_name in backends_to_sweep:
        backend_moe = _sglang_moe_backend_override(backend_name)
        exp_name = f"agg_{backend_name}" if backend == "auto" else "agg"
        tasks[exp_name] = _make_agg(backend_name, backend_moe)

        # For SGLang MoE without --enable-wideep, also sweep DeepEP intra-node
        if backend_name == "sglang" and not enable_wideep and moe_backend is None and is_moe_model:
            skip_reason = _sglang_deepep_perf_data_skip_reason(system, None, backend_version)
            if skip_reason:
                logger.info("Skipping SGLang DeepEP agg sweep: %s", skip_reason)
            else:
                try:
                    deepep_task = _make_agg(backend_name, "deepep_moe")
                except UnsupportedWideepConfigError as exc:
                    logger.info("Skipping SGLang DeepEP agg sweep: %s", exc)
                else:
                    deepep_name = f"agg_{backend_name}_deepep" if backend == "auto" else "agg_deepep"
                    tasks[deepep_name] = deepep_task

        if total_gpus < 2:
            logger.warning("Skipping disagg since it requires at least 2 GPUs.")
            continue

        exp_name = f"disagg_{backend_name}" if backend == "auto" else "disagg"
        tasks[exp_name] = _make_disagg(backend_name, backend_moe)

        # For SGLang MoE without --enable-wideep, also sweep DeepEP intra-node
        if backend_name == "sglang" and not enable_wideep and moe_backend is None and is_moe_model:
            skip_reason = _sglang_deepep_perf_data_skip_reason(system, decode_system, backend_version)
            if skip_reason:
                logger.info("Skipping SGLang DeepEP disagg sweep: %s", skip_reason)
            else:
                try:
                    deepep_disagg_task = _make_disagg(backend_name, "deepep_moe")
                except UnsupportedWideepConfigError as exc:
                    logger.info("Skipping SGLang DeepEP disagg sweep: %s", exc)
                else:
                    deepep_name = f"disagg_{backend_name}_deepep" if backend == "auto" else "disagg_deepep"
                    tasks[deepep_name] = deepep_disagg_task
    return tasks


def build_experiment_tasks(
    yaml_path: str | None = None,
    config: dict[str, Any] | None = None,
    engine_step_backend: str | None = None,
) -> dict[str, Task]:
    """Build task configs from YAML file or config dict.

    Args:
        yaml_path: Path to a YAML file containing experiment definitions.
        config: Dict containing experiment definitions (alternative to yaml_path).
            Keys are experiment names, values are experiment configs.
        engine_step_backend: Optional global experimental static-latency backend.
            Per-experiment ``engine_step_backend`` entries take precedence.

    Returns:
        Dict mapping experiment names to Task objects.

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

    tasks: dict[str, Task] = {}

    for exp_name in experiment_names:
        exp_config = experiment_data[exp_name]
        if not isinstance(exp_config, dict):
            logger.warning("Skipping experiment '%s': configuration is not a mapping.", exp_name)
            continue

        serving_mode = exp_config.get("serving_mode")
        # model_path / system_name are top-level for agg and legacy-V1 disagg, but
        # flat-V2 disagg carries them only under prefill_* / decode_*.  Accept either.
        model_path = (
            exp_config.get("model_path") or exp_config.get("prefill_model_path") or exp_config.get("decode_model_path")
        )
        if serving_mode not in {"agg", "disagg"} or not model_path:
            logger.warning("Skipping experiment '%s': missing serving_mode or model_path.", exp_name)
            continue

        system_name = (
            exp_config.get("system_name")
            or exp_config.get("prefill_system_name")
            or exp_config.get("decode_system_name")
        )
        if not system_name:
            logger.warning("Skipping experiment '%s': no system_name provided.", exp_name)
            continue

        database_mode = exp_config.get("database_mode", common.DatabaseMode.SILICON.name)

        if exp_config.get("total_gpus") is None:
            logger.warning("Skipping experiment '%s': total_gpus not provided.", exp_name)
            continue

        if _database_mode_requires_declared_perf_database(database_mode):
            # Early-fail on an unavailable backend version (clearer than failing deep in the sweep).
            # Role-aware: agg and legacy-v1 disagg carry backend/version/system at the top level;
            # flat-v2 disagg carries them per role (prefill_*/decode_*), falling back to top-level so
            # v1 configs still validate.
            if serving_mode == "disagg":
                role_checks = [
                    (
                        exp_config.get("prefill_system_name") or system_name,
                        exp_config.get("prefill_backend_name") or exp_config.get("backend_name"),
                        exp_config.get("prefill_backend_version") or exp_config.get("backend_version"),
                    ),
                    (
                        exp_config.get("decode_system_name") or system_name,
                        exp_config.get("decode_backend_name") or exp_config.get("backend_name"),
                        exp_config.get("decode_backend_version") or exp_config.get("backend_version"),
                    ),
                ]
            else:
                role_checks = [(system_name, exp_config.get("backend_name"), exp_config.get("backend_version"))]
            seen_combos: set[tuple[str, str, str]] = set()
            for sys_name, bname, bver in role_checks:
                bname = bname or common.BackendName.trtllm.value
                if bver is not None and sys_name and (sys_name, bname, bver) not in seen_combos:
                    seen_combos.add((sys_name, bname, bver))
                    _ensure_backend_version_available(sys_name, bname, bver)

        # Per-experiment engine_step_backend wins over the global default.
        overrides: dict[str, Any] = {}
        if engine_step_backend is not None and "engine_step_backend" not in exp_config:
            overrides["engine_step_backend"] = engine_step_backend

        # exp_config is a legacy V1 experiment dict (top-level fields + nested config /
        # mode / profiles).  Task.from_yaml auto-detects and converts it to the flat V2
        # schema (emitting a DeprecationWarning); a native V2 flat dict also works.
        try:
            task_config = {**exp_config, "database_mode": database_mode}
            tasks[exp_name] = Task.from_yaml(task_config, **overrides)
        except Exception as exc:
            if is_expected_cli_error(exc):
                # Expected config/compatibility rejection (e.g. MegaMoE requires a
                # Blackwell system): report cleanly and keep the traceback at DEBUG.
                logger.log(logging.ERROR, "Failed to build Task for experiment '%s': %s", exp_name, exc)
                logger.debug("Traceback for experiment %s", exp_name, exc_info=True)
            else:
                logger.exception("Failed to build Task for experiment '%s'", exp_name)

    return tasks


def _execute_tasks(
    tasks: dict[str, Task],
    mode: str,
    top_n: int = 5,
    target_request_rate: float | None = None,
    target_concurrency: float | None = None,
    max_total_gpus: int | None = None,
    strict_sla: bool = False,
    inclusive_tpot: bool = False,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, float], dict[str, dict[str, float]]]:
    """
    Execute task configs and return the chosen experiment, best configs, results, best
    throughputs, and estimated latencies.

    Args:
        tasks: Dictionary mapping experiment names to Task objects to execute.
        mode: Execution mode ('default' or 'exp').
        top_n: Number of top configurations to return for each experiment.
        target_request_rate: If set, activates load-match picking (minimize
            GPUs for the given request rate in req/s).
        target_concurrency: If set, activates load-match picking (minimize
            GPUs for the given number of concurrent requests).
        max_total_gpus: Optional upper bound on total GPUs for load-match.
        strict_sla: When True, enforce both TTFT and TPOT SLA constraints
            during picking.
        inclusive_tpot: When True, replace TPOT in terminal and CSV output
            with (ttft + tpot * (osl - 1)) / osl.

    Returns:
        tuple:
            - The experiment name with the overall best throughput ("chosen experiment").
            - Dictionary of best config DataFrames per experiment (or per serving mode if merged).
            - Dictionary of Pareto frontier DataFrames per experiment (or mode).
            - Dictionary of best throughput values per experiment (or mode).
            - Dictionary of estimated latencies per experiment. Each value is a
              dict with keys ``"ttft"``, ``"tpot"``, ``"request_latency"``
              extracted from the rank-1 config.
    """
    results: dict[str, dict[str, pd.DataFrame]] = {}
    failure_messages: list[str] = []
    start_time = time.time()
    # TODO, can run in parallel
    for exp_name, task in tasks.items():
        try:
            logger.info("Starting experiment: %s", exp_name)
            logger.debug("Task config: \n%s", task.to_yaml())
            pareto_df = task.run()
            task_result = {"pareto_df": pareto_df}
            if pareto_df is not None and not pareto_df.empty:
                results[exp_name] = task_result
                logger.info("Experiment %s completed with %d results.", exp_name, len(pareto_df))
            else:
                db_mode = getattr(task, "database_mode", None)
                hybrid_hint = (
                    " For frontier/new models without silicon data, try --database-mode HYBRID."
                    if db_mode == common.DatabaseMode.SILICON.name
                    else ""
                )
                msg = (
                    f"Experiment {exp_name} returned no results. Possible causes: "
                    "(1) TTFT/TPOT constraints are too tight — try relaxing --ttft or --tpot; "
                    "(2) the model does not fit on the available GPUs — try increasing --total-gpus; "
                    f"(3) no perf data in the database for this configuration.{hybrid_hint}"
                )
                logger.warning(msg)
                failure_messages.append(msg)
        except NoFeasibleConfigError as exc:
            msg = f"Experiment {exp_name} found no SLA-feasible configuration: {exc}"
            logger.warning(msg)
            failure_messages.append(msg)
        except Exception as exc:
            if is_expected_cli_error(exc):
                # Expected failure (no feasible config / OOM / KV-cache capacity,
                # a per-op perf-data miss, or an unsupported quant/compatibility
                # config): report cleanly. Keep the traceback at DEBUG for
                # diagnosis via --log-level DEBUG.
                logger.log(logging.ERROR, "Error running experiment %s: %s", exp_name, exc)
                logger.debug("Traceback for experiment %s", exp_name, exc_info=True)
            else:
                logger.exception("Error running experiment %s", exp_name)
            failure_messages.append(f"Experiment {exp_name} failed: {exc}")

    if len(results) < 1:
        first_config = next(iter(tasks.values()), None)
        db_mode = getattr(first_config, "database_mode", None) if first_config else None
        if db_mode == common.DatabaseMode.SILICON.name:
            logger.error(
                "No successful experiment runs to compare. "
                "If this is a frontier or newly-released model, retry with --database-mode HYBRID "
                "to extend coverage beyond the silicon database."
            )
        else:
            logger.error("No successful experiment runs to compare.")
        for msg in failure_messages:
            logger.error("  -> %s", msg)
        raise SystemExit(1)

    best_configs: dict[str, pd.DataFrame] = {}
    best_throughputs: dict[str, float] = {}
    best_latencies: dict[str, dict[str, float]] = {}
    pareto_fronts: dict[str, pd.DataFrame | None] = {}
    pareto_x_axis: dict[str, str] = {}
    for name, task_result in results.items():
        task = tasks[name]
        best_config_df, best_throughput, pareto_frontier_df, x_axis_col, latencies = process_experiment_result(
            task,
            task_result,
            top_n,
            target_request_rate=target_request_rate,
            target_concurrency=target_concurrency,
            max_total_gpus=max_total_gpus,
            strict_sla=strict_sla,
        )
        best_configs[name] = best_config_df
        best_throughputs[name] = best_throughput
        best_latencies[name] = latencies
        pareto_fronts[name] = pareto_frontier_df
        pareto_x_axis[name] = x_axis_col

    if mode == "default" and len(tasks) > 2:
        best_configs, best_throughputs, pareto_fronts, pareto_x_axis = merge_experiment_results_by_mode(
            tasks, best_configs, pareto_fronts, pareto_x_axis, top_n
        )

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"

    log_final_summary(
        chosen_exp=chosen_exp,  # for summary
        best_throughputs=best_throughputs,  # for summary
        best_configs=best_configs,  # for table
        pareto_fronts=pareto_fronts,  # for plotting
        tasks=tasks,  # for info in summary
        mode=mode,
        pareto_x_axis=pareto_x_axis,
        top_n=top_n,
        target_request_rate=target_request_rate,
        target_concurrency=target_concurrency,
        inclusive_tpot=inclusive_tpot,
    )

    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds", end_time - start_time)

    return chosen_exp, best_configs, pareto_fronts, best_throughputs, best_latencies


def _run_generate_mode(args):
    """Run the generate mode to create a naive agg config without sweeping."""
    model_path = args.model_path
    logger.info("Generating naive agg configuration for %s on %d GPUs", model_path, args.total_gpus)

    generator_overrides = load_generator_overrides_from_args(args)

    # Use the public API function
    result = generate_naive_config(
        model_path=model_path,
        total_gpus=args.total_gpus,
        system=args.system,
        backend=args.backend,
        output_dir=args.save_dir or "./output",
        generated_config_version=getattr(args, "generated_config_version", None),
        generator_dynamo_version=getattr(args, "generator_dynamo_version", None),
        generator_overrides=generator_overrides,
        deployment_target=getattr(args, "deployment_target", "dynamo-j2"),
    )

    # Extract result data for CLI output
    generator_params = result["generator_params"]
    backend_version = result["backend_version"]
    output_dir = result["output_dir"]
    parallelism = result["parallelism"]
    tp = parallelism["tp"]
    pp = parallelism["pp"]
    gpus_per_worker = parallelism.get("gpus_per_worker", tp * pp)
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
    print(f"  Replicas:        {replicas} (each using {gpus_per_worker} GPUs)")
    print(f"  Max Batch Size:  {_get_naive_summary_max_batch_size(generator_params)}")
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


def _get_naive_summary_max_batch_size(generator_params: dict[str, Any]) -> Any:
    params = generator_params.get("params", {})
    if not isinstance(params, dict):
        return "n/a"
    for role in ("agg", "prefill", "decode"):
        role_params = params.get(role)
        if isinstance(role_params, dict) and role_params.get("max_batch_size") is not None:
            return role_params["max_batch_size"]
    for role_params in params.values():
        if isinstance(role_params, dict) and role_params.get("max_batch_size") is not None:
            return role_params["max_batch_size"]
    return "n/a"


def _run_support_matrix_mode(args):
    """Run support check across system/backend combinations and display a matrix."""
    model = args.model_path
    version_filter = args.backend_version

    try:
        architecture = get_model_config_from_model_path(model)["architecture"]
    except Exception:
        architecture = None

    matrix = common.get_support_matrix()
    systems = common.sort_support_matrix_systems(common.SupportedSystems) if args.system == "all" else [args.system]
    backends = [b.value for b in common.BackendName] if args.backend == "all" else [args.backend]
    existing_combos = {(row["System"].lower(), row["Backend"].lower()) for row in matrix}

    results: dict[tuple[str, str], common.SupportResult | None] = {}
    has_inferred = False

    for system in systems:
        for be in backends:
            if (system.lower(), be.lower()) not in existing_combos:
                results[(system, be)] = None
                continue

            if version_filter:
                version = version_filter
            else:
                version = _latest_support_matrix_version(matrix, system, be, model=model, architecture=architecture)
                if version is None:
                    results[(system, be)] = None
                    continue

            result = common.check_support(
                model=model, system=system, backend=be, version=version, architecture=architecture
            )
            if not result.exact_match and result.agg_total_count == 0 and result.disagg_total_count == 0:
                result = None
            elif not result.exact_match and result.architecture:
                has_inferred = True
            results[(system, be)] = result

    # ── render matrix ────────────────────────────────────────────
    col_w = 6  # sub-column width (fits "disagg", "YES*", etc.)
    gap = "   "
    sys_w = max(len("System"), *(len(s) for s in systems)) + 2
    block_w = col_w * 2 + 1  # "agg" + space + "disagg"

    def _cell(supported: bool, inferred: bool) -> str:
        return f"{('YES' if supported else 'NO') + ('*' if inferred else ''):>{col_w}}"

    sub_col = f"{'agg':>{col_w}} {'disagg':>{col_w}}"
    header1 = " " * sys_w + gap.join(be.center(block_w) for be in backends)
    header2 = f"{'System':<{sys_w}}" + gap.join(sub_col for _ in backends)
    sep = "-" * sys_w + gap.join("-" * block_w for _ in backends)
    total_w = max(60, len(sep) + 4)

    lines = ["", "=" * total_w, "  AIC Support Matrix", "=" * total_w, f"  Model:  {model}"]
    if architecture:
        lines.append(f"  Arch:   {architecture}")
    if version_filter:
        lines.append(f"  Version filter: {version_filter}")
    lines += ["-" * total_w, f"  {header1}", f"  {header2}", f"  {sep}"]

    for system in systems:
        cells = []
        for be in backends:
            r = results.get((system, be))
            if r is None:
                cells.append(f"{'-':>{col_w}} {'-':>{col_w}}")
            else:
                inf = not r.exact_match and r.architecture is not None
                cells.append(f"{_cell(r.agg_supported, inf)} {_cell(r.disagg_supported, inf)}")
        lines.append(f"  {system:<{sys_w}}" + gap.join(cells))

    lines.append("=" * total_w)
    lines.append("  YES = supported  NO = not supported  - = no data")
    if has_inferred:
        lines.append("  * = inferred from architecture majority vote")
    if not version_filter:
        lines.append("  Using latest available version per system/backend combination.")
    lines += ["=" * total_w, ""]
    print("\n".join(lines))


def _run_support_mode(args):
    """Run the support mode to see if a model/hardware combo is supported."""
    if args.backend is None:
        args.backend = "all" if args.system == "all" else "trtllm"

    if args.system == "all" or args.backend == "all":
        _run_support_matrix_mode(args)
        return

    model = args.model_path
    system = args.system
    backend = args.backend
    version = args.backend_version

    # Resolve architecture for better check
    try:
        model_info = get_model_config_from_model_path(model)
        architecture = model_info["architecture"]
    except Exception:
        architecture = None

    # If no version specified, find the latest model-relevant version in the support matrix
    if not version:
        matrix = common.get_support_matrix()
        version = _latest_support_matrix_version(matrix, system, backend, model=model, architecture=architecture)
        if version is None:
            logger.info(
                "No valid support-matrix backend version found for model=%s system=%s backend=%s",
                model,
                system,
                backend,
            )
            print("\nNo valid support-matrix backend version found for this model/system/backend combination.\n")
            return

    logger.info("Checking support for model=%s, system=%s, backend=%s, version=%s", model, system, backend, version)

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
    if not result.exact_match and result.architecture and (result.agg_total_count > 0 or result.disagg_total_count > 0):
        print("-" * 60)
        print(
            f"  Note: Model '{model}' not found in support matrix cache,\n\
    but the model matches architecture '{result.architecture}'."
        )
        print(f"  Support inferred from architecture '{result.architecture}' majority vote:")
        if result.agg_total_count:
            p, t = result.agg_pass_count, result.agg_total_count
            print(f"    Aggregated:    {p}/{t} passed (>{t // 2} required)")
        if result.disagg_total_count:
            p, t = result.disagg_pass_count, result.disagg_total_count
            print(f"    Disaggregated: {p}/{t} passed (>{t // 2} required)")

    print("=" * 60 + "\n")


def _print_per_ops_section(title: str, ops: dict) -> None:
    """Print a single section of per-op latency breakdown."""
    total = sum(ops.values())
    print(f"  {title} (total: {total:.3f} ms)")
    for op_name, latency in sorted(ops.items(), key=lambda x: -x[1]):
        pct = latency / total * 100 if total > 0 else 0
        print(f"    {op_name:<40s} {latency:>10.3f} ms  ({pct:>5.1f}%)")


def _print_per_ops_latency(per_ops_data: dict) -> None:
    """Print per-operation latency breakdown from run_agg / run_disagg / run_afd.

    NOTE: ``cli estimate`` now surfaces per-op breakdowns through
    ``format_estimate_detail_report`` (driven by ``--detail``). These helpers
    are kept available for the AFD path / future callers that still want the
    standalone summary print.
    """
    print("\n" + "-" * 60)
    print("  Per-Operation Latency Breakdown")
    print("-" * 60)

    # Agg mode: mix_step + genonly_step + scheduling
    scheduling = per_ops_data.get("scheduling")
    if scheduling:
        num_mix = scheduling.get("num_mix_steps", 0)
        num_genonly = scheduling.get("num_genonly_steps", 0)
        print(f"  Scheduling: {num_mix:.0f} mix steps + {num_genonly:.0f} gen-only steps")
        print()

    mix_ops = per_ops_data.get("mix_step", {})
    if mix_ops:
        _print_per_ops_section("Mix Step", mix_ops)

    genonly_ops = per_ops_data.get("genonly_step", {})
    if genonly_ops:
        print()
        _print_per_ops_section("Gen-Only Step", genonly_ops)

    # Disagg mode: prefill + decode
    prefill_ops = per_ops_data.get("prefill", {})
    if prefill_ops:
        _print_per_ops_section("Prefill (static_ctx)", prefill_ops)

    decode_ops = per_ops_data.get("decode", {})
    if decode_ops:
        print()
        _print_per_ops_section("Decode (static_gen)", decode_ops)

    afd_sections = [
        ("Prefill A-Worker", per_ops_data.get("prefill_a_worker", {})),
        ("Prefill F-Worker", per_ops_data.get("prefill_f_worker", {})),
        ("Decode A-Worker", per_ops_data.get("decode_a_worker", {})),
        ("Decode F-Worker", per_ops_data.get("decode_f_worker", {})),
    ]
    afd_emitted = False
    for title, ops in afd_sections:
        if not ops:
            continue
        if not afd_emitted:
            afd_emitted = True
        print()
        _print_per_ops_section(title, ops)

    comm = per_ops_data.get("comm", {})
    if comm:
        directional = {k: v for k, v in comm.items() if k.endswith("_a2f") or k.endswith("_f2a")}
        if directional:
            print()
            _print_per_ops_section("AFD Transfer (per layer, a2f + f2a)", directional)


def _run_estimate_mode(args):
    """Run the estimate mode to predict TTFT, TPOT, and power for a single config."""
    from aiconfigurator.cli.api import cli_estimate

    estimate_mode = args.estimate_mode

    logger.info(
        "Estimating performance (%s) for %s on %s (backend=%s, tp=%d, bs=%d)",
        estimate_mode,
        args.model_path,
        args.system,
        args.backend,
        args.tp_size,
        args.batch_size,
    )

    _resolve_and_validate_nextn(args)

    # Resolve --detail before running the estimate so time detail can compare
    # against a second SOL-mode result.
    detail_arg = (args.detail or "").strip()
    try:
        needs_sol_detail = bool(detail_arg) and detail_requests_time(detail_arg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # Build kwargs shared between agg, disagg, and static
    estimate_kwargs = dict(
        model_path=args.model_path,
        system_name=args.system,
        mode=estimate_mode,
        backend_name=args.backend,
        backend_version=args.backend_version,
        database_mode=args.database_mode,
        transfer_policy=args.transfer_policy,
        isl=args.isl,
        osl=args.osl,
        image_height=args.image_height,
        image_width=args.image_width,
        num_images=args.num_images,
        batch_size=args.batch_size,
        ctx_tokens=args.ctx_tokens,
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        attention_dp_size=args.attention_dp_size,
        moe_tp_size=args.moe_tp_size,
        moe_ep_size=args.moe_ep_size,
        gemm_quant_mode=args.gemm_quant_mode,
        kvcache_quant_mode=args.kvcache_quant_mode,
        fmha_quant_mode=args.fmha_quant_mode,
        moe_quant_mode=args.moe_quant_mode,
        comm_quant_mode=args.comm_quant_mode,
        free_gpu_memory_fraction=args.free_gpu_memory_fraction,
        max_seq_len=args.max_seq_len,
        engine_step_backend=args.engine_step_backend,
        prefix=args.prefix,
        nextn=args.nextn,
        nextn_accepted=args.nextn_accepted,
        stride=args.stride,
    )

    if estimate_mode == "disagg":
        estimate_kwargs.update(
            decode_system_name=args.decode_system,
            prefill_tp_size=args.prefill_tp_size,
            prefill_pp_size=args.prefill_pp_size,
            prefill_attention_dp_size=args.prefill_attention_dp_size,
            prefill_moe_tp_size=args.prefill_moe_tp_size,
            prefill_moe_ep_size=args.prefill_moe_ep_size,
            prefill_batch_size=args.prefill_batch_size,
            prefill_num_workers=args.prefill_num_workers,
            decode_tp_size=args.decode_tp_size,
            decode_pp_size=args.decode_pp_size,
            decode_attention_dp_size=args.decode_attention_dp_size,
            decode_moe_tp_size=args.decode_moe_tp_size,
            decode_moe_ep_size=args.decode_moe_ep_size,
            decode_batch_size=args.decode_batch_size,
            decode_num_workers=args.decode_num_workers,
        )
    elif estimate_mode == "afd":
        # gpus_per_node and f_tp_size are intentionally derived from the
        # system_spec / topology by cli_estimate -> _run_afd_estimate;
        # they are no longer exposed as CLI flags to prevent silent
        # mis-shaping (e.g. gb200 has 4 GPUs/node, not the historical 8).
        estimate_kwargs.update(
            n_a_nodes=args.n_a_nodes,
            n_f_nodes=args.n_f_nodes,
            a_tp_size=args.a_tp_size,
            a_batch_size=args.a_batch_size,
            f_moe_ep_size=args.f_moe_ep_size,
            num_microbatches=args.num_microbatches,
            pipeline_model=args.pipeline_model,
            comm_overhead_factor=args.comm_overhead_factor,
            afd_phase=args.afd_phase,
            afd_combined_with_pd=getattr(args, "afd_combined_with_pd", True),
            afd_boundary_on_attn=not getattr(args, "boundary_on_ffn", False),
        )

    result = cli_estimate(**estimate_kwargs)
    sol_result = None
    if needs_sol_detail:
        if args.database_mode == common.DatabaseMode.SOL.name:
            sol_result = result
        else:
            sol_estimate_kwargs = dict(estimate_kwargs)
            sol_estimate_kwargs["database_mode"] = common.DatabaseMode.SOL.name
            sol_result = cli_estimate(**sol_estimate_kwargs)

    print("\n" + "=" * 60)
    print(f"  Performance Estimate ({result.mode})")
    print("=" * 60)
    print(f"  Model:            {result.model_path}")
    print(f"  System:           {result.system_name}")
    print(f"  Backend:          {result.backend_name} ({result.backend_version})")
    print("-" * 60)
    print(f"  ISL:              {result.isl}")
    print(f"  OSL:              {result.osl}")
    if args.image_height > 0 and args.image_width > 0 and args.num_images > 0:
        print(f"  Images:           {args.num_images} x {args.image_height}x{args.image_width}")

    # ``--prefix`` and ``--nextn`` are common parameters applied to every
    # mode (agg / disagg / afd / static*), so surface them in the summary box
    # for all modes rather than gating on mode.
    if args.prefix:
        print(f"  Prefix:           {args.prefix}")
    if args.nextn:
        print(f"  MTP nextn:        {args.nextn} (nextn_accepted={args.nextn_accepted})")

    if result.mode == "disagg":
        raw = result.raw
        print(f"  (p) TP:           {raw.get('(p)tp', 'N/A')}")
        print(f"  (p) PP:           {raw.get('(p)pp', 'N/A')}")
        print(f"  (p) BS:           {raw.get('(p)bs', 'N/A')}")
        print(f"  (p) Workers:      {raw.get('(p)workers', 'N/A')}")
        print(f"  (d) TP:           {raw.get('(d)tp', 'N/A')}")
        print(f"  (d) PP:           {raw.get('(d)pp', 'N/A')}")
        print(f"  (d) BS:           {raw.get('(d)bs', 'N/A')}")
        print(f"  (d) Workers:      {raw.get('(d)workers', 'N/A')}")
        print(f"  Total GPUs:       {raw.get('num_total_gpus', 'N/A')}")
    elif result.mode == "afd":
        raw = result.raw
        print(f"  AFD Phase:        {raw.get('phase', 'decode')}")
        if raw.get("combined_with_pd"):
            print("  Combined w/ P/D:  yes")
        print(f"  GPUs/Node:        {raw.get('gpus_per_node', 'N/A')}")
        print(f"  (a) Nodes:        {raw.get('(a)nodes', 'N/A')}")
        print(f"  (a) TP:           {raw.get('(a)tp', 'N/A')}")
        print(f"  (a) BS:           {raw.get('(a)bs', 'N/A')}")
        print(f"  (a) Workers(DP):  {raw.get('(a)workers', 'N/A')}")
        print(f"  (f) Nodes:        {raw.get('(f)nodes', 'N/A')}")
        print(f"  (f) TP:           {raw.get('(f)tp', 'N/A')}")
        print(f"  (f) EP:           {raw.get('(f)ep', 'N/A')}")
        print(f"  (f) Workers:      {raw.get('(f)workers', 'N/A')}")
        print(f"  B_total:          {raw.get('b_total', 'N/A')}")
        print(f"  Total GPUs:       {raw.get('num_total_gpus', 'N/A')}")
        print(f"  Pipeline Model:   {raw.get('pipeline_model', 'N/A')}")
        print(f"  Micro-batches:    {raw.get('num_microbatches', 'N/A')}")
        boundary_side = "A-Worker" if raw.get("boundary_on_attn", True) else "F-Worker"
        print(f"  Boundary on:      {boundary_side}")
    else:
        # agg / static / static_ctx / static_gen share the same single-replica shape.
        print(f"  Batch Size:       {result.batch_size}")
        if result.mode == "agg":
            print(f"  Context Tokens:   {result.ctx_tokens}")
        print(f"  TP Size:          {result.tp_size}")
        print(f"  PP Size:          {result.pp_size}")
        if args.attention_dp_size and args.attention_dp_size != 1:
            print(f"  Attention DP:     {args.attention_dp_size}")

    print("-" * 60)
    if result.mode == "static_gen":
        # ttft is zero by construction; surface generation_latency instead.
        gen_lat = float(result.raw.get("generation_latency", 0.0) or 0.0)
        print(f"  Generation lat.:  {gen_lat:.3f} ms")
        print(f"  TPOT:             {result.tpot:.3f} ms")
    elif result.mode == "static_ctx":
        print(f"  TTFT:             {result.ttft:.3f} ms")
    elif result.mode == "afd":
        raw = result.raw
        afd_phase = raw.get("phase")
        if afd_phase == "both":
            # phase="both" runs prefill + decode through AFD; un-prefixed
            # layer scalars are deliberately NaN to keep the two estimates
            # distinguishable. Render the paired ``prefill_*`` / ``decode_*``
            # blocks instead so users can compare A/F balance per phase.
            print("  -- Prefill (AFD) --")
            print(f"  T_a_layer:        {raw.get('prefill_t_a_layer', 0):.3f} ms")
            print(f"  T_f_layer:        {raw.get('prefill_t_f_layer', 0):.3f} ms")
            print(f"  T_a2f_layer:      {raw.get('prefill_t_a2f_layer', 0):.3f} ms")
            print(f"  T_f2a_layer:      {raw.get('prefill_t_f2a_layer', 0):.3f} ms")
            print(f"  T_c_layer:        {raw.get('prefill_t_c_layer', 0):.3f} ms  (round-trip = a2f + f2a)")
            print(f"  T_step:           {raw.get('prefill_t_step', 0):.3f} ms")
            print(f"  Balance Ratio:    {raw.get('prefill_balance_ratio', 0):.3f}")
            print("  -- Decode (AFD) --")
            print(f"  T_a_layer:        {raw.get('decode_t_a_layer', 0):.3f} ms")
            print(f"  T_f_layer:        {raw.get('decode_t_f_layer', 0):.3f} ms")
            print(f"  T_a2f_layer:      {raw.get('decode_t_a2f_layer', 0):.3f} ms")
            print(f"  T_f2a_layer:      {raw.get('decode_t_f2a_layer', 0):.3f} ms")
            print(f"  T_c_layer:        {raw.get('decode_t_c_layer', 0):.3f} ms  (round-trip = a2f + f2a)")
            print(f"  T_step:           {raw.get('decode_t_step', 0):.3f} ms")
            print(f"  Balance Ratio:    {raw.get('decode_balance_ratio', 0):.3f}")
        else:
            print(f"  T_a_layer:        {raw.get('t_a_layer', 0):.3f} ms")
            print(f"  T_f_layer:        {raw.get('t_f_layer', 0):.3f} ms")
            print(f"  T_a2f_layer:      {raw.get('t_a2f_layer', 0):.3f} ms")
            print(f"  T_f2a_layer:      {raw.get('t_f2a_layer', 0):.3f} ms")
            print(f"  T_c_layer:        {raw.get('t_c_layer', 0):.3f} ms  (round-trip = a2f + f2a)")
            print(f"  T_step:           {raw.get('t_step', 0):.3f} ms")
            print(f"  Balance Ratio:    {raw.get('balance_ratio', 0):.3f}")
        # Composition row: shown when the combined-with-PD merge has
        # written (p)impl/(d)impl markers, i.e. when the AFD result was
        # merged with a static estimate of the other phase. Lets the user
        # see at a glance which phase is AFD vs static.
        p_impl = raw.get("(p)impl")
        d_impl = raw.get("(d)impl")
        if p_impl or d_impl:
            print(f"  Composition:      (p)={p_impl or 'unmodeled'}  (d)={d_impl or 'unmodeled'}")
        print(f"  TTFT:             {result.ttft:.3f} ms")
        print(f"  TPOT:             {result.tpot:.3f} ms")
        print(f"  Request Latency:  {result.request_latency:.3f} ms")
    else:
        print(f"  TTFT:             {result.ttft:.3f} ms")
        print(f"  TPOT:             {result.tpot:.3f} ms")
        print(f"  Request Latency:  {result.request_latency:.3f} ms")
    encoder_latency = float(result.raw.get("encoder_latency", 0.0) or 0.0)
    if encoder_latency > 0.0:
        print(f"  Encoder lat.:     {encoder_latency:.3f} ms")
    print(f"  Power (per GPU):  {result.power_w:.1f} W")
    print("-" * 60)
    print(f"  tokens/s:         {result.tokens_per_second:,.2f}")
    print(f"  tokens/s/gpu:     {result.tokens_per_second_per_gpu:,.2f}")
    print(f"  tokens/s/user:    {result.tokens_per_second_per_user:,.2f}")
    print(f"  seq/s:            {result.seq_per_second:,.3f}")
    print(f"  Concurrency:      {result.concurrency:.0f}")
    if result.mode == "disagg":
        raw = result.raw
        print(f"  (p) Memory:       {raw.get('(p)memory', 'N/A')} GB")
        print(f"  (d) Memory:       {raw.get('(d)memory', 'N/A')} GB")
        encoder_memory = float(raw.get("(e)memory", 0.0) or 0.0)
        if encoder_memory > 0.0:
            print(f"  Encoder memory:   {encoder_memory:.3f} GB (included in prefill)")
    elif result.mode == "afd":
        raw = result.raw
        a_oom = " (OOM!)" if raw.get("(a)is_oom") else ""
        f_oom = " (OOM!)" if raw.get("(f)is_oom") else ""
        print(f"  (a) Memory:       {raw.get('(a)memory', 'N/A')} GB{a_oom}")
        print(f"  (f) Memory:       {raw.get('(f)memory', 'N/A')} GB{f_oom}")
    else:
        print(f"  Memory (GPU):     {result.memory:.2f} GB")
        encoder_memory = float(result.raw.get("encoder_memory", 0.0) or 0.0)
        if encoder_memory > 0.0:
            print(f"  Encoder memory:   {encoder_memory:.3f} GB (included)")
    print("=" * 60)

    if result.kv_cache_warning:
        logger.warning(result.kv_cache_warning)

    if detail_arg:
        try:
            report = format_estimate_detail_report(result, sol_result, detail=detail_arg)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if report:
            print("\n" + "-" * 60)
            print(f"  Detailed Breakdown ({detail_arg})")
            print("-" * 60)
            print(report)
        else:
            logger.warning("--detail requested but no breakdown data is available for this mode.")

    print()


def _resolve_cli_log_level(args) -> int:
    """Pick the log level with priority: --log-level > AICONFIGURATOR_LOG_LEVEL > --debug > INFO."""
    cli_level = getattr(args, "log_level", None)
    if cli_level:
        return getattr(logging, cli_level)
    env_level = os.environ.get("AICONFIGURATOR_LOG_LEVEL")
    if env_level:
        resolved = env_level.strip().upper()
        if resolved in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            return getattr(logging, resolved)
    if getattr(args, "debug", False):
        return logging.DEBUG
    return logging.INFO


def _validate_fpm_sweep_tasks(args, tasks: dict[str, Task]) -> None:
    """Reject task shapes that can only fail after an expensive FPM sweep."""
    if getattr(args, "deployment_target", "dynamo-j2") != "fpm":
        return

    unsupported: list[str] = []
    for name, task in tasks.items():
        serving_mode = getattr(task, "serving_mode", None)
        backend = getattr(task, "primary_backend_name", None)
        if serving_mode != "agg" or backend != common.BackendName.vllm.value:
            unsupported.append(f"{name} ({serving_mode or 'unknown'}/{backend or 'unknown'})")

    if unsupported:
        raise SystemExit(
            "--deployment-target fpm supports only vLLM aggregated tasks; "
            "unsupported task(s): " + ", ".join(unsupported)
        )


def main(args):
    setup_logging(
        level=_resolve_cli_log_level(args),
        no_color=getattr(args, "no_color", False),
    )

    # Handle support mode early — it doesn't need systems_paths or top_n
    if args.mode == "support":
        _run_support_mode(args)
        return

    try:
        perf_database.set_systems_paths(args.systems_paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    logger.info(f"Loading AIConfigurator version: {__version__}")
    logger.info(f"Number of top configurations to output: {args.top_n} (change with --top-n)")

    # Handle generate mode separately (no sweeping)
    if args.mode == "generate":
        _run_generate_mode(args)
        return

    # Handle estimate mode separately (single-point estimation)
    if args.mode == "estimate":
        # Expected user errors surface from the SDK as ValueError (invalid
        # sizing/parameters, unsupported quant/compatibility) or as a perf-data
        # coverage miss (PerfDataNotAvailableError / EmpiricalNotImplementedError).
        # Convert them to a concise CLI error instead of a full traceback; keep
        # the traceback at DEBUG (--log-level DEBUG) and let genuine bugs
        # (KeyError, OOM RuntimeError, …) propagate unchanged.
        try:
            _run_estimate_mode(args)
        except Exception as exc:
            if is_expected_cli_error(exc):
                logger.debug("Traceback for estimate mode", exc_info=True)
                raise SystemExit("Error: " + str(exc)) from exc
            raise
        return

    if args.mode == "default":
        _resolve_and_validate_nextn(args)

        # Warn when SLA/workload parameters are implicitly defaulted
        _default_params = {"isl": 4000, "osl": 1000, "ttft": 2000.0, "tpot": 30.0}
        _implicit = [
            f"{k.upper()}={getattr(args, k)}"
            for k, v in _default_params.items()
            if f"--{k}" not in sys.argv and getattr(args, k) == v
        ]
        if _implicit:
            logger.warning(
                "Using default SLA/workload parameters: %s. "
                "These act as filters — configurations exceeding these thresholds are excluded. "
                "Set them explicitly (e.g. --ttft, --tpot, --isl, --osl) to avoid unexpected filtering.",
                ", ".join(_implicit),
            )
        logger.info(
            "Effective parameters: ISL=%d, OSL=%d, TTFT=%.1fms, TPOT=%.1fms, backend=%s",
            args.isl,
            args.osl,
            args.ttft,
            args.tpot,
            args.backend,
        )
        tasks = build_default_tasks(
            model_path=args.model_path,
            total_gpus=args.total_gpus,
            system=args.system,
            decode_system=args.decode_system,
            backend=args.backend,
            backend_version=args.backend_version,
            database_mode=args.database_mode,
            transfer_policy=args.transfer_policy,
            isl=args.isl,
            osl=args.osl,
            image_height=args.image_height,
            image_width=args.image_width,
            num_images=args.num_images,
            ttft=args.ttft,
            tpot=args.tpot,
            request_latency=args.request_latency,
            prefix=args.prefix,
            nextn=args.nextn,
            nextn_accepted=args.nextn_accepted,
            enable_chunked_prefill=args.enable_chunked_prefill,
            free_gpu_memory_fraction=args.free_gpu_memory_fraction,
            max_seq_len=args.max_seq_len,
            engine_step_backend=args.engine_step_backend,
            enable_wideep=getattr(args, "enable_wideep", False),
            moe_backend=getattr(args, "moe_backend", None),
        )
    elif args.mode == "exp":
        try:
            build_kwargs: dict[str, Any] = {"yaml_path": args.yaml_path}
            if args.engine_step_backend is not None:
                build_kwargs["engine_step_backend"] = args.engine_step_backend
            tasks = build_experiment_tasks(**build_kwargs)
        except (ValueError, TypeError) as exc:
            logger.exception("Failed to build experiment task configs")
            raise SystemExit(1) from exc
        if not tasks:
            logger.error("No valid experiments found in '%s'.", args.yaml_path)
            raise SystemExit(1)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    _validate_fpm_sweep_tasks(args, tasks)

    execute_kwargs: dict = {}
    if getattr(args, "strict_sla", False):
        execute_kwargs["strict_sla"] = True
    if getattr(args, "inclusive_tpot", False):
        execute_kwargs["inclusive_tpot"] = True
    _, best_configs, pareto_fronts, _, _ = _execute_tasks(
        tasks,
        args.mode,
        top_n=args.top_n,
        **execute_kwargs,
    )

    if args.save_dir:
        save_results(
            args=args,
            best_configs=best_configs,
            pareto_fronts=pareto_fronts,
            tasks=tasks,
            save_dir=args.save_dir,
            generated_backend_version=args.generated_config_version,
            backend=args.backend if args.mode == "default" else None,
        )


if __name__ == "__main__":
    if generator_cli_helper(sys.argv[1:]):
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description="AIConfigurator for Disaggregated Serving Deployment",
        epilog=_USAGE_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
