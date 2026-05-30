#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Support matrix generation and validation utilities.

This module provides the SupportMatrix class for generating and validating
the model/system/backend/version support matrix for AIConfigurator.
"""

import csv
import json
import logging
import os
import shlex
import traceback
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from aiconfigurator.generator.naive import _estimate_model_weight_bytes
from aiconfigurator.sdk import common, perf_database
from aiconfigurator.sdk import config as sdk_config
from aiconfigurator.sdk.models import _get_model_info
from aiconfigurator.sdk.models.helpers import _apply_model_quant_defaults
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

logger = logging.getLogger(__name__)

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_HW_INCOMPATIBLE = "HW_INCOMPATIBLE"
STATUS_FRAMEWORK_INCOMPATIBLE = "FRAMEWORK_INCOMPATIBLE"
VALID_STATUSES = frozenset({STATUS_PASS, STATUS_FAIL, STATUS_HW_INCOMPATIBLE, STATUS_FRAMEWORK_INCOMPATIBLE})
SUPPORT_MATRIX_BASE_HEADER = [
    "HuggingFaceID",
    "Architecture",
    "System",
    "Backend",
    "Version",
    "Mode",
    "Status",
    "ErrMsg",
]
SUPPORT_MATRIX_HEADER = SUPPORT_MATRIX_BASE_HEADER + ["Command"]
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
_FP8_QUANT_MODE_NAMES = frozenset({"fp8", "fp8_static", "fp8_block", "w4afp8"})
_NATIVE_FP4_QUANT_MODE_NAMES = frozenset({"nvfp4"})
_FP8_SOFTWARE_FALLBACK_SYSTEMS = frozenset({"b60"})


def _combination_sort_key(combo: tuple[str, str, str, str]) -> tuple[tuple[int, str], str, str, str]:
    model, system, backend, version = combo
    return common.get_support_matrix_system_sort_key(system), backend, version, model


def _combination_group_key(combo: tuple[str, str, str, str]) -> tuple[str, str, str]:
    _model, system, backend, version = combo
    return system, backend, version


@dataclass(frozen=True)
class TestConstraints:
    total_gpus: int
    isl: int
    osl: int
    prefix: int
    ttft: float
    tpot: float


@dataclass(frozen=True)
class HardwareIncompatibility:
    missing_datatypes: tuple[str, ...]
    reason: str


def _support_matrix_row_command(
    *,
    model: str,
    system: str,
    backend: str,
    version: str,
    mode: str,
    constraints: TestConstraints | None = None,
    compare_engine_step_backends: bool = False,
    engine_step_comparison_rtol: float = DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
    engine_step_comparison_atol: float = DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
    engine_step_frontier_rtol: float = DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
    engine_step_frontier_atol: float = DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
) -> str:
    """Return the repo-local CLI command that checks this model/system/backend path."""
    if constraints is None:
        constraints = _get_test_constraints(model)
    parts = [
        "uv",
        "run",
        "aiconfigurator",
        "cli",
        "default",
        "--model-path",
        model,
        "--total-gpus",
        str(constraints.total_gpus),
        "--system",
        system,
        "--backend",
        backend,
        "--backend-version",
        version,
        "--database-mode",
        "SILICON",
        "--isl",
        str(constraints.isl),
        "--osl",
        str(constraints.osl),
        "--prefix",
        str(constraints.prefix),
        "--ttft",
        str(constraints.ttft),
        "--tpot",
        str(constraints.tpot),
        "--top-n",
        "1",
        "--no-color",
    ]
    if compare_engine_step_backends:
        # ``cli default`` does not expose the support-matrix Python/Rust
        # comparator; use the default Python engine-step path for the public
        # replay command.
        parts.extend(["--engine-step-backend", "python"])
    _ = (
        mode,
        engine_step_comparison_rtol,
        engine_step_comparison_atol,
        engine_step_frontier_rtol,
        engine_step_frontier_atol,
    )
    return " ".join(shlex.quote(str(part)) for part in parts)


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


def _large_pipeline_parallel_attempt(
    *,
    mode: str,
    backend: str,
    constraints: TestConstraints,
) -> TaskAttempt | None:
    """Return a support-matrix retry that shards one large-model worker across nodes."""
    if backend not in {
        common.BackendName.sglang.value,
        common.BackendName.trtllm.value,
        common.BackendName.vllm.value,
    }:
        return None
    if constraints.total_gpus < 16:
        return None

    # Some very large checkpoints are too large for the default one-node search.
    # Keep attention TP at 8 and add larger PP splits so a single serving worker
    # can span enough GPUs to fit checkpoints with large active/resident weights.
    # This also constrains MoE splits away from uncollected smaller shapes when
    # the viable data was collected for larger model-parallel workers.
    worker_config = {
        "num_gpu_per_worker": [16, 32, 64],
        "tp_list": [8],
        "pp_list": [2, 4, 8],
        "dp_list": [1],
        "moe_tp_list": [1, 2, 4, 8],
        "moe_ep_list": [1, 2, 4, 8],
    }
    config = {"worker_config": worker_config}
    if mode == "disagg":
        config = {
            "prefill_worker_config": worker_config,
            "decode_worker_config": worker_config,
            "replica_config": {
                "max_gpu_per_replica": 128,
                "num_gpu_per_replica": [32, 64, 128],
            },
        }
    return TaskAttempt(
        name="large-pipeline-parallel-worker",
        yaml_config={"mode": "patch", "config": config},
    )


def _should_retry_with_large_worker(error_message: str | None) -> bool:
    if not error_message:
        return False
    normalized = error_message.lower()
    return (
        "does not fit in gpu memory" in normalized
        or "try increasing --total-gpus" in normalized
        or "try increasing total gpus" in normalized
        or "failed to query moe data" in normalized
        or "missing silicon data for the requested lookup" in normalized
    )


def _is_known_framework_incompatible_gap(
    *,
    model: str,
    system: str,
    backend: str,
    version: str,
    error_message: str | None,
) -> bool:
    """Return True for deterministic framework/data gaps that should not stay plain FAIL."""
    if not error_message:
        return False

    normalized = error_message.lower()
    if (
        backend == common.BackendName.vllm.value
        and version == "0.19.0"
        and "DeepSeek-V4" in model
        and (
            "unsupported moe quant mode 'w4a8_mxfp4_mxfp8'" in normalized
            or "deepseek-v4 mhc module data not loaded" in normalized
        )
    ):
        return True

    if "unsupported gemm quant mode 'fp8_static'" in normalized:
        return True

    if system == "rtx_pro_6000_server":
        if (
            "rtx_pro_6000_server/nccl/" in normalized
            or "failed to query context attention data" in normalized
            or "failed to query moe data" in normalized
        ):
            return True
        if backend == common.BackendName.sglang.value and version == "0.5.10":
            return (
                "unsupported gemm quant mode 'fp8_block'" in normalized
                or "unsupported moe quant mode 'nvfp4'" in normalized
                or (
                    model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
                    and "unsupported moe quant mode 'w4a16_mxfp4'" in normalized
                )
                or "dsa_context_module_perf.txt" in normalized
            )
        if backend == common.BackendName.trtllm.value and version == "1.3.0rc10":
            return (
                "unsupported moe quant mode 'fp8_block'" in normalized
                or ("DeepSeek-V4" in model and "unsupported moe quant mode 'w4a8_mxfp4_mxfp8'" in normalized)
                or (
                    model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}
                    and "unsupported moe quant mode 'w4a16_mxfp4'" in normalized
                )
                or (model == "moonshotai/Kimi-K2.5" and "unsupported moe quant mode 'int4_wo'" in normalized)
                or "dsa_context_module_perf.txt" in normalized
            )
        if backend == common.BackendName.vllm.value and version == "0.19.0":
            return "unsupported moe quant mode 'nvfp4'" in normalized or "dsa_context_module_perf.txt" in normalized

    return (
        model == "moonshotai/Kimi-K2.5"
        and system == "b200_sxm"
        and backend == common.BackendName.trtllm.value
        and version == "1.3.0rc10"
        and "unsupported moe quant mode 'int4_wo'" in normalized
    )


def _is_known_hw_incompatible_gap(
    *,
    system: str,
    error_message: str | None,
) -> bool:
    """Return True for deterministic runtime errors caused by GPU capability gaps."""
    if not error_message:
        return False

    normalized = error_message.lower()
    return system == "l40s" and (
        "unsupported gemm quant mode 'fp8_block'" in normalized
        or "unsupported moe quant mode 'fp8_block'" in normalized
        or "quant_mode=<gemmquantmode.fp8_block" in normalized
        or "quant_mode=<moequantmode.fp8_block" in normalized
    )


def _enum_name(value: object | None) -> str | None:
    if value is None:
        return None
    return value.name if hasattr(value, "name") else str(value)


def _format_datatype_list(datatypes: tuple[str, ...]) -> str:
    if len(datatypes) == 1:
        return datatypes[0]
    return ", ".join(datatypes[:-1]) + f" or {datatypes[-1]}"


def _gpu_label(system: str, system_spec: dict) -> str:
    sm_version = (system_spec.get("gpu") or {}).get("sm_version")
    if sm_version is None:
        return system
    return f"{system} (SM{sm_version})"


def _gpu_supports_datatype(system: str, system_spec: dict, datatype: str) -> bool:
    gpu_spec = system_spec.get("gpu") or {}
    sm_version = gpu_spec.get("sm_version")
    if datatype == "FP8":
        # Intel Arc Pro B60 can run FP8 PyTorch/vLLM model paths by converting
        # FP8 to BF16. Do not model it as native FP8 throughput in b60.yaml.
        if system.lower() in _FP8_SOFTWARE_FALLBACK_SYSTEMS:
            return True
        return "fp8_tc_flops" in gpu_spec or (sm_version is not None and sm_version >= 89)
    if datatype == "FP4":
        return "fp4_tc_flops" in gpu_spec or (sm_version is not None and sm_version >= 100)
    return True


def _required_datatypes_for_model(model: str, backend: str) -> tuple[str, ...]:
    """Infer hardware datatypes required by a model's quantization metadata."""
    model_info = dict(_get_model_info(model))
    raw_config = model_info.get("raw_config", {}) or {}
    architecture = model_info["architecture"]
    model_config = sdk_config.ModelConfig(tp_size=1, moe_tp_size=1, moe_ep_size=1)
    _apply_model_quant_defaults(model_config, raw_config, architecture, backend)

    quant_mode_names = {
        _enum_name(model_config.gemm_quant_mode),
        _enum_name(model_config.moe_quant_mode),
        _enum_name(model_config.kvcache_quant_mode),
        _enum_name(model_config.fmha_quant_mode),
    }
    quant_mode_names.discard(None)

    raw_quant_algo = str(raw_config.get("quant_algo") or "").lower()
    raw_kv_cache_algo = str(raw_config.get("kv_cache_quant_algo") or "").lower()
    expert_dtype = str(raw_config.get("expert_dtype") or "").lower()

    required: set[str] = set()
    if quant_mode_names & _NATIVE_FP4_QUANT_MODE_NAMES or raw_quant_algo == "nvfp4" or expert_dtype == "fp4":
        required.add("FP4")
    if quant_mode_names & _FP8_QUANT_MODE_NAMES or raw_quant_algo in {"fp8", "fp8_block"}:
        required.add("FP8")
    if raw_kv_cache_algo == "fp8":
        required.add("FP8")

    # Keep FP4 first so FP4 model failures on older GPUs read as FP4 incompatibility
    # even when the model also uses FP8 scales or KV cache.
    return tuple(datatype for datatype in ("FP4", "FP8") if datatype in required)


def get_hardware_incompatibility(
    *,
    model: str,
    system: str,
    backend: str,
    system_spec: dict,
) -> HardwareIncompatibility | None:
    """Return a deterministic hardware/model datatype incompatibility, if any."""
    required_datatypes = _required_datatypes_for_model(model, backend)
    missing = tuple(dt for dt in required_datatypes if not _gpu_supports_datatype(system, system_spec, dt))
    if not missing:
        return None

    datatype_text = _format_datatype_list(missing)
    return HardwareIncompatibility(
        missing_datatypes=missing,
        reason=f"{_gpu_label(system, system_spec)} does not support {datatype_text} required by {model}",
    )


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
_worker_modes_to_test: tuple[str, ...] | None = None


def _process_combination_worker(
    combo: tuple[str, str, str, str],
) -> list[tuple[str, str, str, str, str, str, str, str | None, str]]:
    """
    Run a single combination in a worker process. Uses the process-local SupportMatrix.
    Must be a module-level function for pickling by ProcessPoolExecutor.
    """
    assert _worker_matrix is not None  # this only works in linux, not in windows/macos
    model, system, backend, version = combo
    status_dict, error_dict, command_dict = _worker_matrix.run_single_test(
        model=model,
        system=system,
        backend=backend,
        version=version,
        modes_to_test=_worker_modes_to_test,
        compare_engine_step_backends=_worker_matrix.compare_engine_step_backends,
        engine_step_comparison_rtol=_worker_matrix.engine_step_comparison_rtol,
        engine_step_comparison_atol=_worker_matrix.engine_step_comparison_atol,
        engine_step_frontier_rtol=_worker_matrix.engine_step_frontier_rtol,
        engine_step_frontier_atol=_worker_matrix.engine_step_frontier_atol,
        include_commands=True,
    )
    architecture = _worker_matrix.get_architecture(model)
    return [
        (model, architecture, system, backend, version, mode, status_dict[mode], error_dict[mode], command_dict[mode])
        for mode in status_dict
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
        # database structure: {system: {backend: [version]}}
        logger.info("Discovering perf databases...")
        self.databases: dict[str, dict[str, list[str]]] = self.load_databases()
        logger.info("Discovered perf databases for %d systems", len(self.databases))

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
        return perf_database.get_supported_databases()

    def __get_hardware_and_backend_combinations(self) -> list[tuple[str, str, str]]:
        """
        Iterate over all combinations of hardware, and inference backend, version.
        """
        for hardware in common.sort_support_matrix_systems(self.get_systems()):
            hardware_databases = self.databases.get(hardware, {})
            for backend in sorted(self.get_backends()):
                for version in sorted(hardware_databases.get(backend, [])):
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
        combinations = sorted(self.__get_model_and_hardware_and_backend_combinations(), key=_combination_sort_key)
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
        system_spec: dict | None = None,
        modes_to_test: tuple[str, ...] | list[str] | None = None,
        compare_engine_step_backends: bool = False,
        engine_step_comparison_rtol: float = DEFAULT_ENGINE_STEP_COMPARISON_RTOL,
        engine_step_comparison_atol: float = DEFAULT_ENGINE_STEP_COMPARISON_ATOL,
        engine_step_frontier_rtol: float = DEFAULT_ENGINE_STEP_FRONTIER_RTOL,
        engine_step_frontier_atol: float = DEFAULT_ENGINE_STEP_FRONTIER_ATOL,
        include_commands: bool = False,
    ) -> tuple[dict[str, str], dict[str, str | None]] | tuple[dict[str, str], dict[str, str | None], dict[str, str]]:
        """
        Run a single configuration test for both agg and disagg modes.

        Args:
            model: Model name
            system: System/hardware name
            backend: Backend name
            version: Backend version
            system_spec: Optional system spec to avoid reloading the database for hardware preflight.
            compare_engine_step_backends: When True, run both Python and Rust engine-step backends.
            engine_step_comparison_rtol: Relative tolerance for Python-vs-Rust Pareto metrics.
            engine_step_comparison_atol: Absolute tolerance for Python-vs-Rust Pareto metrics.
            engine_step_frontier_rtol: Loose relative tolerance when frontiers choose different rows.
            engine_step_frontier_atol: Loose absolute tolerance when frontiers choose different rows.

        Returns:
            Tuple of (dict with statuses, dict with error messages).
            Status values are PASS, FAIL, or HW_INCOMPATIBLE.
            Both dicts have keys "agg" and "disagg".
        """
        if modes_to_test is None:
            modes_to_test = ("agg", "disagg")
        else:
            modes_to_test = tuple(modes_to_test)
            unsupported_modes = set(modes_to_test) - {"agg", "disagg"}
            if unsupported_modes:
                raise ValueError(f"Unsupported support-matrix mode(s): {sorted(unsupported_modes)}")
        constraints = _get_test_constraints(model)
        statuses: dict[str, str] = {}
        error_messages = {}
        commands = {
            mode: _support_matrix_row_command(
                model=model,
                system=system,
                backend=backend,
                version=version,
                mode=mode,
                constraints=constraints,
                compare_engine_step_backends=compare_engine_step_backends,
                engine_step_comparison_rtol=engine_step_comparison_rtol,
                engine_step_comparison_atol=engine_step_comparison_atol,
                engine_step_frontier_rtol=engine_step_frontier_rtol,
                engine_step_frontier_atol=engine_step_frontier_atol,
            )
            for mode in modes_to_test
        }

        if system_spec is None:
            database = perf_database.get_database(system, backend, version)
            system_spec = database.system_spec if database is not None else None

        if system_spec is not None:
            try:
                incompatibility = get_hardware_incompatibility(
                    model=model,
                    system=system,
                    backend=backend,
                    system_spec=system_spec,
                )
            except Exception:
                logger.exception("Hardware compatibility preflight failed for %s on %s/%s", model, system, backend)
                incompatibility = None
            if incompatibility is not None:
                reason = _format_exception_for_csv(incompatibility.reason)
                statuses = dict.fromkeys(modes_to_test, STATUS_HW_INCOMPATIBLE)
                error_messages = dict.fromkeys(modes_to_test, reason)
                if include_commands:
                    return statuses, error_messages, commands
                return statuses, error_messages

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
                    raise RuntimeError("Configuration returned no results, failed to catch traceback")

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
                        raise RuntimeError("Rust engine-step backend returned no results")

                    mismatch = _compare_pareto_dfs(
                        python_pareto_df,
                        rust_pareto_df,
                        rtol=engine_step_comparison_rtol,
                        atol=engine_step_comparison_atol,
                        frontier_rtol=engine_step_frontier_rtol,
                        frontier_atol=engine_step_frontier_atol,
                    )
                    if mismatch:
                        raise RuntimeError(mismatch)

                statuses[mode] = STATUS_PASS
                error_messages[mode] = None

            except Exception as e:
                raw_error = traceback.format_exc()
                logger.warning(
                    "Configuration failed: %s, %s, %s, %s, mode=%s - Error: %s",
                    model,
                    system,
                    backend,
                    version,
                    mode,
                    str(e),
                )

                if _is_known_hw_incompatible_gap(
                    system=system,
                    error_message=raw_error,
                ):
                    statuses[mode] = STATUS_HW_INCOMPATIBLE
                elif _is_known_framework_incompatible_gap(
                    model=model,
                    system=system,
                    backend=backend,
                    version=version,
                    error_message=raw_error,
                ):
                    statuses[mode] = STATUS_FRAMEWORK_INCOMPATIBLE
                else:
                    statuses[mode] = STATUS_FAIL
                error_messages[mode] = raw_error

            finally:
                perf_database.clear_database_runtime_caches(system, backend, version)

            error_messages[mode] = _format_exception_for_csv(error_messages.get(mode))
        if include_commands:
            return statuses, error_messages, commands
        return statuses, error_messages

    def _run_parallel_combinations(
        self,
        combinations: list[tuple[str, str, str, str]],
        *,
        max_workers: int,
        pbar: tqdm,
    ) -> tuple[list[tuple[str, str, str, str, str, str, str, str | None, str]], set[tuple[str, str, str, str]]]:
        group_results: list[tuple[str, str, str, str, str, str, str, str | None, str]] = []
        retry_combos: set[tuple[str, str, str, str]] = set()
        processed_futures = set()

        with ProcessPoolExecutor(max_workers=min(max_workers, len(combinations))) as executor:
            futures = {executor.submit(_process_combination_worker, combo): combo for combo in combinations}
            for future in as_completed(futures):
                combo = futures[future]
                model, system, backend, version = combo
                try:
                    group_results.extend(future.result())
                    processed_futures.add(future)
                    pbar.update(1)
                except BrokenExecutor:
                    logger.warning(
                        "Process pool broken while running %s/%s/%s/%s. "
                        "A worker was likely killed (OOM). "
                        "Queuing this and remaining group combos for sequential retry.",
                        model,
                        system,
                        backend,
                        version,
                    )
                    unprocessed_futures = [remaining for remaining in futures if remaining not in processed_futures]
                    for remaining in unprocessed_futures:
                        remaining.cancel()
                        retry_combos.add(futures[remaining])
                    pbar.update(len(unprocessed_futures))
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
                    processed_futures.add(future)
                    pbar.update(1)

        return group_results, retry_combos

    def test_support_matrix(
        self,
        max_workers: int | None = None,
        *,
        combinations: list[tuple[str, str, str, str]] | None = None,
        modes_to_test: tuple[str, ...] | list[str] | None = None,
    ) -> list[tuple[str, str, str, str, str, str, str, str | None, str]]:
        """
        Test whether each combination is supported by AIC.
        Tests both agg and disagg modes for each combination and captures error messages.

        Runs in two phases:
        1. Parallel execution with one ProcessPoolExecutor per (system, backend, version).
        2. Sequential single-process retry of every combination that failed in phase 1
           (including combos that never ran due to a broken process pool).

        Args:
            max_workers: Maximum number of worker processes for parallel execution.
                         Defaults to None, which uses os.cpu_count() or 1.

        Returns:
            List of tuples (huggingface_id, architecture, system, backend, version, mode, status, err_msg, command).
            Returns separate entries for each requested mode.
        """
        # Print configuration
        print("\n" + "=" * 80)
        print("AIConfigurator Support Matrix Test")
        print("=" * 80)
        print("Testing requested support-matrix modes for selected combinations")
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

        if modes_to_test is None:
            modes_to_test = ("agg", "disagg")
        else:
            modes_to_test = tuple(modes_to_test)
        combinations = (
            self.generate_combinations() if combinations is None else sorted(combinations, key=_combination_sort_key)
        )
        print(f"Total combinations to test: {len(combinations)}")
        print(f"Modes: {', '.join(modes_to_test)}")
        results: list[tuple[str, str, str, str, str, str, str, str | None, str]] = []
        retry_combos: set[tuple[str, str, str, str]] = set()

        global _worker_matrix, _worker_modes_to_test
        _worker_matrix = self
        _worker_modes_to_test = tuple(modes_to_test)

        # -- Phase 1: parallel execution, one short-lived pool per database group --
        with tqdm(total=len(combinations), desc="Phase 1: parallel testing", unit="config") as pbar:
            for (system, backend, version), group_iter in groupby(combinations, key=_combination_group_key):
                group_combinations = list(group_iter)
                tqdm.write(
                    f"Phase 1 group: {system}/{backend}/{version} ({len(group_combinations)} model combination(s))"
                )
                try:
                    group_results, group_retry_combos = self._run_parallel_combinations(
                        group_combinations,
                        max_workers=max_workers,
                        pbar=pbar,
                    )
                    results.extend(group_results)
                    retry_combos.update(group_retry_combos)
                finally:
                    perf_database.unload_database(system, backend, version)

        # Also collect combos whose Phase 1 results had any failure
        for model, _arch, system, backend, version, _mode, status, _err, _command in results:
            if status == STATUS_FAIL:
                retry_combos.add((model, system, backend, version))

        # -- Phase 2: sequential single-process retry of all failures --
        if retry_combos:
            results = [r for r in results if (r[0], r[2], r[3], r[4]) not in retry_combos]

            print(f"\n{'=' * 80}")
            print(f"Phase 2: retrying {len(retry_combos)} failed combination(s) sequentially")
            print(f"{'=' * 80}\n")

            sorted_retry_combos = sorted(retry_combos, key=_combination_sort_key)
            with tqdm(total=len(sorted_retry_combos), desc="Phase 2: sequential retry", unit="config") as pbar:
                for (system, backend, version), group_iter in groupby(sorted_retry_combos, key=_combination_group_key):
                    try:
                        for combo in group_iter:
                            model, system, backend, version = combo
                            try:
                                status_dict, error_dict, command_dict = self.run_single_test(
                                    model=model,
                                    system=system,
                                    backend=backend,
                                    version=version,
                                    modes_to_test=modes_to_test,
                                    compare_engine_step_backends=self.compare_engine_step_backends,
                                    engine_step_comparison_rtol=self.engine_step_comparison_rtol,
                                    engine_step_comparison_atol=self.engine_step_comparison_atol,
                                    engine_step_frontier_rtol=self.engine_step_frontier_rtol,
                                    engine_step_frontier_atol=self.engine_step_frontier_atol,
                                    include_commands=True,
                                )
                                architecture = self.get_architecture(model)
                                for mode in status_dict:
                                    results.append(
                                        (
                                            model,
                                            architecture,
                                            system,
                                            backend,
                                            version,
                                            mode,
                                            status_dict[mode],
                                            error_dict[mode],
                                            command_dict[mode],
                                        )
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
                                for mode in modes_to_test:
                                    command = _support_matrix_row_command(
                                        model=model,
                                        system=system,
                                        backend=backend,
                                        version=version,
                                        mode=mode,
                                        compare_engine_step_backends=self.compare_engine_step_backends,
                                        engine_step_comparison_rtol=self.engine_step_comparison_rtol,
                                        engine_step_comparison_atol=self.engine_step_comparison_atol,
                                        engine_step_frontier_rtol=self.engine_step_frontier_rtol,
                                        engine_step_frontier_atol=self.engine_step_frontier_atol,
                                    )
                                    results.append(
                                        (
                                            model,
                                            architecture,
                                            system,
                                            backend,
                                            version,
                                            mode,
                                            STATUS_FAIL,
                                            traceback.format_exc().replace("\n", "\\n"),
                                            command,
                                        )
                                    )
                            finally:
                                pbar.update(1)
                    finally:
                        perf_database.unload_database(system, backend, version)

        # Sort results by (huggingface_id, architecture, system, backend, version, mode)
        results.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]))

        # Print results summary
        self._print_results_summary(results)

        return results

    def _print_results_summary(self, results: list[tuple[str, str, str, str, str, str, str, str | None, str]]) -> None:
        """Print summary of test results."""
        total_tests = len(results)
        passed = sum(1 for _, _, _, _, _, _, status, _, _ in results if status == STATUS_PASS)
        failed = sum(1 for _, _, _, _, _, _, status, _, _ in results if status == STATUS_FAIL)
        hw_incompatible = sum(1 for _, _, _, _, _, _, status, _, _ in results if status == STATUS_HW_INCOMPATIBLE)
        framework_incompatible = sum(
            1 for _, _, _, _, _, _, status, _, _ in results if status == STATUS_FRAMEWORK_INCOMPATIBLE
        )

        print("\n" + "=" * 80)
        print("Test Results Summary")
        print("=" * 80)
        print(f"Total configurations tested: {total_tests}")
        print(f"✓ Passed: {passed} ({100 * passed / total_tests:.1f}%)")
        print(f"✗ Failed: {failed} ({100 * failed / total_tests:.1f}%)")
        print(f"⚪ Hardware incompatible: {hw_incompatible} ({100 * hw_incompatible / total_tests:.1f}%)")
        print(
            f"⚪ Framework incompatible: {framework_incompatible} ({100 * framework_incompatible / total_tests:.1f}%)"
        )
        print("=" * 80)

        # Group results by status
        passed_configs = []
        failed_configs = []
        hw_incompatible_configs = []
        framework_incompatible_configs = []

        for huggingface_id, architecture, system, backend, version, mode, status, _err, _command in results:
            config = (huggingface_id, architecture, system, backend, version, mode)
            if status == STATUS_PASS:
                passed_configs.append(config)
            elif status == STATUS_FAIL:
                failed_configs.append(config)
            elif status == STATUS_HW_INCOMPATIBLE:
                hw_incompatible_configs.append(config)
            elif status == STATUS_FRAMEWORK_INCOMPATIBLE:
                framework_incompatible_configs.append(config)

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

        if hw_incompatible_configs:
            print(f"\n⚪ Hardware-Incompatible Configurations ({len(hw_incompatible_configs)}):")
            for huggingface_id, architecture, system, backend, version, mode in sorted(hw_incompatible_configs):
                print(f"  • {huggingface_id} ({architecture}) on {system} with {backend} v{version} ({mode})")

        if framework_incompatible_configs:
            print(f"\n⚪ Framework-Incompatible Configurations ({len(framework_incompatible_configs)}):")
            for huggingface_id, architecture, system, backend, version, mode in sorted(framework_incompatible_configs):
                print(f"  • {huggingface_id} ({architecture}) on {system} with {backend} v{version} ({mode})")

    def save_results_to_csv(self, results: list[tuple[str, ...]], output_file: str) -> None:
        """
        Save test results to split CSV files, one per system.

        Passing a path ending in ``.csv`` preserves the legacy single-file output
        for ad hoc comparisons.

        Args:
            results: List of tuples
                (huggingface_id, architecture, system, backend, version, mode, status, err_msg, command)
            output_file: Path to the output directory, or a legacy output CSV file
        """
        output_path = Path(output_file)

        def _row_values(row: tuple[str, ...]) -> tuple[str, str, str, str, str, str, str, str, str]:
            if len(row) == 9:
                huggingface_id, architecture, system, backend, version, mode, status, err_msg, command = row
            elif len(row) == 8:
                huggingface_id, architecture, system, backend, version, mode, status, err_msg = row
                command = _support_matrix_row_command(
                    model=huggingface_id,
                    system=system,
                    backend=backend,
                    version=version,
                    mode=mode,
                    constraints=_DEFAULT_TIER,
                    compare_engine_step_backends=getattr(self, "compare_engine_step_backends", False),
                    engine_step_comparison_rtol=getattr(
                        self, "engine_step_comparison_rtol", DEFAULT_ENGINE_STEP_COMPARISON_RTOL
                    ),
                    engine_step_comparison_atol=getattr(
                        self, "engine_step_comparison_atol", DEFAULT_ENGINE_STEP_COMPARISON_ATOL
                    ),
                    engine_step_frontier_rtol=getattr(
                        self, "engine_step_frontier_rtol", DEFAULT_ENGINE_STEP_FRONTIER_RTOL
                    ),
                    engine_step_frontier_atol=getattr(
                        self, "engine_step_frontier_atol", DEFAULT_ENGINE_STEP_FRONTIER_ATOL
                    ),
                )
            else:
                raise ValueError(f"Invalid support-matrix result row length: {len(row)}")
            return (
                huggingface_id,
                architecture,
                system,
                backend,
                version,
                mode,
                status,
                err_msg or "",
                command,
            )

        if output_path.suffix == ".csv":
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(SUPPORT_MATRIX_HEADER)
                for row in results:
                    huggingface_id, architecture, system, backend, version, mode, status, err_msg, command = (
                        _row_values(row)
                    )
                    if status not in VALID_STATUSES:
                        raise ValueError(f"Invalid support-matrix status: {status}")
                    writer.writerow(
                        [huggingface_id, architecture, system, backend, version, mode, status, err_msg, command]
                    )
            print(f"\nResults saved to: {output_file}")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        for stale_csv in output_path.glob("*.csv"):
            stale_csv.unlink()

        sorted_results = sorted(
            (_row_values(row) for row in results),
            key=lambda x: (common.get_support_matrix_system_sort_key(x[2]), x[0], x[1], x[3], x[4], x[5]),
        )
        grouped_results = {
            system: list(system_results) for system, system_results in groupby(sorted_results, key=lambda x: x[2])
        }

        manifest = {"files": []}
        for system, system_results in grouped_results.items():
            csv_path = output_path / f"{system}.csv"
            manifest["files"].append(csv_path.name)
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(SUPPORT_MATRIX_HEADER)
                for (
                    huggingface_id,
                    architecture,
                    system,
                    backend,
                    version,
                    mode,
                    status,
                    err_msg,
                    command,
                ) in system_results:
                    if status not in VALID_STATUSES:
                        raise ValueError(f"Invalid support-matrix status: {status}")
                    writer.writerow(
                        [huggingface_id, architecture, system, backend, version, mode, status, err_msg, command]
                    )

        with open(output_path / "index.json", "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")

        print(f"\nResults saved to: {output_file}")
        print(f"Split support matrix files: {len(manifest['files'])}")
