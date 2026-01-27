# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib import resources as pkg_resources


@dataclass(frozen=True)
class BlockConfig:
    """
    Configuration for a single transformer block in NemotronNas.

    Attributes:
        attn_n_heads_in_group (int): Number of attention heads in the group for this block
        attn_no_op (bool): If True, skip attention operations for this block
        ffn_ffn_mult (float): Multiplier for FFN intermediate size relative to hidden size
        ffn_no_op (bool): If True, skip FFN operations for this block
        num_inst (int): number of ocurrances of the given block
    """

    attn_n_heads_in_group: int = 8
    attn_no_op: bool = False
    ffn_ffn_mult: float = 3.5
    ffn_no_op: bool = False
    num_inst: int = 0


def _get_support_matrix_resource():
    """Get the support_matrix.csv as a Traversable resource."""
    return pkg_resources.files("aiconfigurator") / "systems" / "support_matrix.csv"


@cache
def get_support_matrix() -> list[dict[str, str]]:
    """
    Get the support matrix as a list of dictionaries.

    Returns:
        list[dict[str, str]]: List of rows from support_matrix.csv.
    """
    csv_resource = _get_support_matrix_resource()
    results = []
    # Use as_file() context manager for proper package resource access
    with pkg_resources.as_file(csv_resource) as csv_path, open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


@dataclass
class SupportResult:
    """Result of a support check with explanation details."""

    agg_supported: bool
    disagg_supported: bool
    exact_match: bool  # True if model was found in matrix, False if inferred from architecture
    architecture: str | None = None  # Architecture used for inference (if not exact match)
    agg_pass_count: int = 0  # Number of passing agg tests (for majority vote)
    agg_total_count: int = 0  # Total agg tests (for majority vote)
    disagg_pass_count: int = 0  # Number of passing disagg tests (for majority vote)
    disagg_total_count: int = 0  # Total disagg tests (for majority vote)

    def __iter__(self):
        """Support tuple unpacking: agg, disagg = check_support(...)"""
        return iter((self.agg_supported, self.disagg_supported))


def check_support(
    model: str,
    system: str,
    backend: str | None = None,
    version: str | None = None,
    architecture: str | None = None,
) -> SupportResult:
    """
    Check if a model/system combination is supported for agg and disagg modes.
    If the model exists in the support matrix, support is determined by the
    matrix entries for that specific model. Otherwise, support is determined
    by a majority vote of PASS status for models sharing the same architecture.

    Args:
        model: HuggingFace model ID or local path.
        system: System/hardware name.
        backend: Optional backend name to filter by.
        version: Optional backend version to filter by.
        architecture: Optional architecture name. If not provided and model is
            not in matrix, it will be resolved if possible.

    Returns:
        SupportResult: Contains (agg_supported, disagg_supported) plus explanation details.
            Supports tuple unpacking for backward compatibility.
    """
    matrix = get_support_matrix()

    # 1. Check for exact model+system matches and resolve architecture from matrix if needed
    agg_exact = []
    disagg_exact = []
    matrix_arch = None

    for row in matrix:
        if row["HuggingFaceID"] == model:
            matrix_arch = row["Architecture"]
            if row["System"] == system:
                if backend and row["Backend"] != backend:
                    continue
                if version and row["Version"] != version:
                    continue

                is_pass = row["Status"] == "PASS"
                if row["Mode"] == "agg":
                    agg_exact.append(is_pass)
                elif row["Mode"] == "disagg":
                    disagg_exact.append(is_pass)

    # If we found any entries for this specific model on this specific system, use them
    if agg_exact or disagg_exact:
        return SupportResult(
            agg_supported=any(agg_exact),
            disagg_supported=any(disagg_exact),
            exact_match=True,
        )

    # 2. Fallback to architecture-based majority vote
    # Use provided architecture or the one found in the matrix
    architecture = architecture or matrix_arch
    if not architecture:
        return SupportResult(
            agg_supported=False,
            disagg_supported=False,
            exact_match=False,
        )

    agg_arch_results = []
    disagg_arch_results = []

    for row in matrix:
        if row["Architecture"] == architecture and row["System"] == system:
            if backend and row["Backend"] != backend:
                continue
            if version and row["Version"] != version:
                continue

            is_pass = row["Status"] == "PASS"
            if row["Mode"] == "agg":
                agg_arch_results.append(is_pass)
            elif row["Mode"] == "disagg":
                disagg_arch_results.append(is_pass)

    def is_majority_pass(results):
        return sum(results) > len(results) / 2 if results else False

    return SupportResult(
        agg_supported=is_majority_pass(agg_arch_results),
        disagg_supported=is_majority_pass(disagg_arch_results),
        exact_match=False,
        architecture=architecture,
        agg_pass_count=sum(agg_arch_results),
        agg_total_count=len(agg_arch_results),
        disagg_pass_count=sum(disagg_arch_results),
        disagg_total_count=len(disagg_arch_results),
    )


@cache
def get_supported_architectures() -> set[str]:
    """
    Get the set of supported architectures from support_matrix.csv.

    Returns:
        set[str]: Set of architecture names that have at least one PASSing configuration.
    """
    matrix = get_support_matrix()
    return {row["Architecture"] for row in matrix if row["Status"] == "PASS"}


@cache
def get_default_models() -> set[str]:
    """
    Get the set of supported HuggingFace model IDs from support_matrix.csv.

    Returns:
        set[str]: Set of unique HuggingFace model IDs that are supported.
    """
    csv_resource = _get_support_matrix_resource()
    models = set()
    # Use as_file() context manager for proper package resource access
    with pkg_resources.as_file(csv_resource) as csv_path, open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.add(row["HuggingFaceID"])
    return models


"""
Cached HuggingFace model configs - these are pre-downloaded and stored in model_configs/
Model parameters are parsed from these configs via get_model_config_from_model_path() in utils.py
The list of default models for testing is derived from support_matrix.csv via get_default_models()
"""
DefaultHFModels = {
    # Llama 2 Models
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    # Llama 3.1 Models
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    # Mixtral Models
    "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/Mixtral-8x22B-v0.1",
    # DeepSeek Models
    "deepseek-ai/DeepSeek-V3",
    # Qwen 2.5 Models
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    # Qwen 3 Models
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    # GPT-OSS Models
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # NVIDIA Nemotron
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
}

"""
Supported systems (GPU types)
"""
SupportedSystems = {
    "h100_sxm",
    "h200_sxm",
    "b200_sxm",
    "gb200_sxm",
    "a100_sxm",
    "l40s",
}

"""
Model family for model definition
"""
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "NEMOTRONNAS"}
ARCHITECTURE_TO_MODEL_FAMILY = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "LLAMA",
    "Qwen3ForCausalLM": "LLAMA",
    "DeepSeekForCausalLM": "DEEPSEEK",
    "DeepseekV3ForCausalLM": "DEEPSEEK",
    "NemotronForCausalLM": "NEMOTRONNAS",
    "DeciLMForCausalLM": "NEMOTRONNAS",
    "MixtralForCausalLM": "MOE",
    "GptOssForCausalLM": "MOE",
    "Qwen3MoeForCausalLM": "MOE",
}

"""
All reduce strategy for trtllm custom allreduce
"""
AllReduceStrategy = {"NCCL", "ONESHOT", "TWOSHOT", "AUTO"}

"""
Columns for static inference summary dataframe
"""
ColumnsStatic = [
    "model",
    "isl",
    "osl",
    "prefix",
    "concurrency",
    "request_rate",
    "bs",
    "global_bs",
    "ttft",
    "tpot",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "request_latency",
    "context_latency",
    "generation_latency",
    "num_total_gpus",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "parallel",
    "gemm",
    "kvcache",
    "fmha",
    "moe",
    "comm",
    "memory",
    "backend",
    "version",
    "system",
    "power_w",  # NEW: E2E weighted average power in watts
]

"""
Columns for Agg inference summary dataframe
"""
ColumnsAgg = [
    "model",
    "isl",
    "osl",
    "prefix",
    "concurrency",
    "request_rate",
    "bs",
    "global_bs",
    "ttft",
    "tpot",
    "request_latency",
    "seq/s",
    "seq/s/gpu",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "num_total_gpus",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "parallel",
    "gemm",
    "kvcache",
    "fmha",
    "moe",
    "comm",
    "memory",
    "balance_score",
    "num_ctx_reqs",
    "num_gen_reqs",
    "num_tokens",
    "ctx_tokens",
    "gen_tokens",  # agg specific
    "backend",
    "version",
    "system",
    "power_w",  # NEW: E2E weighted average power in watts
]

"""
Columns for disaggregated inference summary dataframe
"""
ColumnsDisagg = [
    "model",
    "isl",
    "osl",
    "prefix",
    "concurrency",
    "request_rate",
    "(p)bs",
    "(p)global_bs",
    "(p)workers",
    "(d)bs",
    "(d)global_bs",
    "(d)workers",
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
    "num_total_gpus",
    "(p)tp",
    "(p)pp",
    "(p)dp",
    "(p)moe_tp",
    "(p)moe_ep",
    "(p)parallel",
    "(p)gemm",
    "(p)kvcache",
    "(p)fmha",
    "(p)moe",
    "(p)comm",
    "(p)memory",
    "(p)backend",
    "(p)version",
    "(p)system",
    "(d)tp",
    "(d)pp",
    "(d)dp",
    "(d)moe_tp",
    "(d)moe_ep",
    "(d)parallel",
    "(d)gemm",
    "(d)kvcache",
    "(d)fmha",
    "(d)moe",
    "(d)comm",
    "(d)memory",
    "(d)backend",
    "(d)version",
    "(d)system",
    "power_w",  # NEW: E2E weighted average power in watts
]


class DatabaseMode(Enum):
    """
    Database mode.
    """

    SILICON = 0  # default mode using silicon data
    HYBRID = 1  # use silicon data when available, otherwise use SOL+empirical factor
    EMPIRICAL = 2  # SOL+empirical factor
    SOL = 3  # Provide SOL time only
    SOL_FULL = 4  # Provide SOL time and details


class BackendName(Enum):
    """
    Backend name for inference.
    """

    trtllm = "trtllm"
    sglang = "sglang"
    vllm = "vllm"


class PerfDataFilename(Enum):
    """
    Perf data filename for database to load.
    """

    gemm = "gemm_perf.txt"
    nccl = "nccl_perf.txt"
    generation_attention = "generation_attention_perf.txt"
    context_attention = "context_attention_perf.txt"
    context_mla = "context_mla_perf.txt"
    generation_mla = "generation_mla_perf.txt"
    mla_bmm = "mla_bmm_perf.txt"
    moe = "moe_perf.txt"
    custom_allreduce = "custom_allreduce_perf.txt"
    wideep_context_mla = "wideep_context_mla_perf.txt"
    wideep_generation_mla = "wideep_generation_mla_perf.txt"
    wideep_context_moe = "wideep_context_moe_perf.txt"
    wideep_generation_moe = "wideep_generation_moe_perf.txt"
    wideep_deepep_normal = "wideep_deepep_normal_perf.txt"
    wideep_deepep_ll = "wideep_deepep_ll_perf.txt"


QuantMapping = namedtuple("QuantMapping", ["memory", "compute", "name"])


class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """

    float16 = QuantMapping(2, 1, "float16")  # w16a16
    int8_wo = QuantMapping(1, 1, "int8_wo")  # w8a16
    int4_wo = QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8 = QuantMapping(1, 2, "fp8")  # w8fp8
    sq = QuantMapping(1, 2, "sq")  # w8int8
    fp8_block = QuantMapping(1, 2, "fp8_block")  # specific for trtllm torch ds fp8
    fp8_ootb = QuantMapping(
        1, 2, "fp8_ootb"
    )  # in future, should deprecate this mode as it's specific for trtllm trt backend
    nvfp4 = QuantMapping(0.5, 4, "nvfp4")  # nvfp4 on blackwell


class MoEQuantMode(Enum):
    """
    MoE quant mode.
    """

    float16 = QuantMapping(2, 1, "float16")  # w16a16
    fp8 = QuantMapping(1, 2, "fp8")  # w8fp8
    int4_wo = QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8_block = QuantMapping(1, 2, "fp8_block")  # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, "w4afp8")  # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(0.5, 4, "nvfp4")  # nvfp4 on blackwell
    w4a16_mxfp4 = QuantMapping(0.5, 1, "w4a16_mxfp4")  # native data format for gpt oss


class FMHAQuantMode(Enum):
    """
    FMHA quant mode.
    """

    float16 = QuantMapping(2, 1, "float16")
    fp8 = QuantMapping(1, 2, "fp8")
    fp8_block = QuantMapping(1, 2, "fp8_block")  # FIXME: specific for sglang wideep


class KVCacheQuantMode(Enum):
    """
    KVCache quant mode.
    """

    float16 = QuantMapping(2, 0, "float16")
    int8 = QuantMapping(1, 0, "int8")
    fp8 = QuantMapping(1, 0, "fp8")


class CommQuantMode(Enum):
    """
    Comm quant mode.
    """

    half = QuantMapping(2, 0, "half")
    int8 = QuantMapping(1, 0, "int8")
    fp8 = QuantMapping(1, 0, "fp8")
