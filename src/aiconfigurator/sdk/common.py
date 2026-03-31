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


@dataclass(frozen=True)
class NemotronHConfig:
    """
    Configuration for NemotronH hybrid model (Mamba + MoE + Transformer).

    Only includes fields unique to NemotronH that are not in standard model parameters.
    Standard fields (num_attention_heads, num_key_value_heads, n_routed_experts,
    num_experts_per_tok, moe_intermediate_size) are already in the base model config.

    Attributes:
        hybrid_override_pattern (str): Pattern string defining layer types.
            'M' = Mamba layer, 'E' = MoE layer, '*' = Transformer layer, '-' = MLP layer
        mamba_num_heads (int): Number of heads in Mamba2 layers
        mamba_head_dim (int): Head dimension for Mamba2 layers
        ssm_state_size (int): SSM state size (d_state) for Mamba2
        conv_kernel (int): Convolution kernel size for Mamba2
        n_groups (int): Number of groups for Mamba2
        chunk_size (int): Chunk size for Mamba2 chunked scan
        moe_shared_expert_intermediate_size (int): Intermediate size for shared expert
    """

    hybrid_override_pattern: str
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    chunk_size: int
    moe_shared_expert_intermediate_size: int = 0  # Optional: 0 for non-MoE NemotronH models


@dataclass(frozen=True)
class HybridMoEConfig:
    """
    Unified config for hybrid attention (SWA/local + global) + mixed FFN (MoE + dense) models.
    Covers MiMo-V2-Flash, Llama 4 Scout/Maverick, and similar architectures.

    Both patterns are stored as normalized per-layer tuples of length num_layers:
        attn_layer_pattern: 0 = SWA/local attention, 1 = global (full) attention
        moe_layer_freq:     0 = dense SwiGLU FFN,    1 = MoE FFN

    SWA/local attention dims — set to 0 to fall back to model-level defaults
    (head_dim / num_kv_heads). MiMo-V2-Flash has different dims per attention type;
    Llama 4 uses the same dims for all layers so all four fields are 0.
        swa_num_kv_heads: KV heads for SWA/local layers  (0 → num_kv_heads)
        swa_head_dim:     Q/K head dim for SWA layers     (0 → head_dim)
        swa_v_head_dim:   V head dim for SWA layers       (0 → head_dim)
        global_v_head_dim: V head dim for global layers   (0 → head_dim)

    sliding_window_size: token window for SWA/local attention layers
    dense_inter_size: intermediate size for dense FFN layers (0 → use inter_size)
    """

    attn_layer_pattern: tuple[int, ...]  # per-layer: 0=SWA/local, 1=global
    moe_layer_freq: tuple[int, ...]  # per-layer: 0=dense, 1=MoE
    swa_num_kv_heads: int = 0
    swa_head_dim: int = 0
    swa_v_head_dim: int = 0
    global_v_head_dim: int = 0
    sliding_window_size: int = 0
    dense_inter_size: int = 0


@dataclass(frozen=True)
class Qwen35Config:
    """Config for Qwen3.5 hybrid GDN + full-attention model (dense and MoE).

    layer_types: per-layer tuple of "linear_attention" (GDN) or "full_attention" (standard GQA)
    linear_*: GDN layer dimensions (linear_key_head_dim=128, linear_value_head_dim=128,
              linear_conv_kernel_dim=4, linear_num_key_heads=16 across all current models)
    MoE fields default to 0 for the dense 27B; populated for 35B-A3B and 397B-A17B.
    """

    layer_types: tuple[str, ...]  # per-layer: "linear_attention" (GDN) or "full_attention"
    linear_num_key_heads: int  # K heads for GDN layers
    linear_key_head_dim: int  # K/Q head dim for GDN layers
    linear_num_value_heads: int  # V heads for GDN layers
    linear_value_head_dim: int  # V head dim for GDN layers
    linear_conv_kernel_dim: int  # Conv1D kernel size for GDN layers
    # MoE fields (0 for dense models)
    topk: int = 0
    num_experts: int = 0
    moe_inter_size: int = 0
    shared_expert_inter_size: int = 0


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

    def _matches_filters(row: dict, backend: str | None, version: str | None) -> bool:
        if backend and row["Backend"] != backend:
            return False
        return not (version and row["Version"] != version)

    # 1. Check for exact model+system matches
    exact_matches = [
        row
        for row in matrix
        if row["HuggingFaceID"].lower() == model.lower()
        and row["System"].lower() == system.lower()
        and _matches_filters(row, backend, version)
    ]

    # Resolve architecture from matrix if model is found anywhere
    matrix_arch = next((row["Architecture"] for row in matrix if row["HuggingFaceID"] == model), None)

    if exact_matches:
        return SupportResult(
            agg_supported=any(row["Status"] == "PASS" for row in exact_matches if row["Mode"] == "agg"),
            disagg_supported=any(row["Status"] == "PASS" for row in exact_matches if row["Mode"] == "disagg"),
            exact_match=True,
        )

    # 2. Fallback to architecture-based inference
    # Use provided architecture or the one found in the matrix
    architecture = architecture or matrix_arch
    if not architecture:
        return SupportResult(agg_supported=False, disagg_supported=False, exact_match=False)

    arch_matches = [
        row
        for row in matrix
        if row["Architecture"] == architecture and row["System"] == system and _matches_filters(row, backend, version)
    ]

    agg_results = [row["Status"] == "PASS" for row in arch_matches if row["Mode"] == "agg"]
    disagg_results = [row["Status"] == "PASS" for row in arch_matches if row["Mode"] == "disagg"]

    def is_majority_pass(results: list[bool]) -> bool:
        # We use majority vote to infer support for an untested model of a known architecture.
        # This provides a balanced estimate: not too optimistic (any) nor too pessimistic (all).
        return sum(results) > len(results) / 2 if results else False

    return SupportResult(
        agg_supported=is_majority_pass(agg_results),
        disagg_supported=is_majority_pass(disagg_results),
        exact_match=False,
        architecture=architecture,
        agg_pass_count=sum(agg_results),
        agg_total_count=len(agg_results),
        disagg_pass_count=sum(disagg_results),
        disagg_total_count=len(disagg_results),
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
    # Llama 3.1 Models
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "nvidia/Llama-3.1-70B-Instruct-FP8",
    # DeepSeek V3/V3.1 Models
    "deepseek-ai/DeepSeek-V3",
    "nvidia/DeepSeek-V3.1-NVFP4",
    # DeepSeek V3.2 / GLM-5 (DEEPSEEKV32 family)
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-5",
    # Qwen 3 Models
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-32B-FP8",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-30B-A3B-FP8",
    "Qwen/Qwen3-235B-A22B",
    "Qwen/Qwen3-235B-A22B-FP8",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "nvidia/Qwen3-235B-A22B-NVFP4",
    "Qwen/Qwen3-32B-FP8-Static-PerTensor",
    # MiniMax Models
    "MiniMaxAI/MiniMax-M2.5",
    "nvidia/MiniMax-M2.5-NVFP4",
    # GPT-OSS Models
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    # Llama 4 Models
    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    # Qwen3.5 Models
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-397B-A17B",
    # MiMo Models
    "XiaomiMiMo/MiMo-V2-Flash",
    "XiaomiMiMo/MiMo-7B-Base",
    # NVIDIA Nemotron
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "nvidia/Nemotron-H-56B-Base-8K",
}

"""
Supported systems (GPU types)
"""
SupportedSystems = {
    "h100_sxm",
    "h200_sxm",
    "b200_sxm",
    "gb200",
    "gb300",
    "a100_sxm",
    "l40s",
    "b60",
}

"""
Model family for model definition
"""
ModelFamily = {"GPT", "LLAMA", "MOE", "DEEPSEEK", "DEEPSEEKV32", "NEMOTRONNAS", "NEMOTRONH", "HYBRIDMOE", "QWEN35"}
ARCHITECTURE_TO_MODEL_FAMILY = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "LLAMA",
    "Qwen3ForCausalLM": "LLAMA",
    "MiMoForCausalLM": "LLAMA",
    "DeepSeekForCausalLM": "DEEPSEEK",
    "DeepseekV3ForCausalLM": "DEEPSEEK",
    "DeepseekV32ForCausalLM": "DEEPSEEKV32",
    "GlmMoeDsaForCausalLM": "DEEPSEEKV32",
    "KimiK25ForConditionalGeneration": "DEEPSEEK",
    "NemotronForCausalLM": "NEMOTRONNAS",
    "DeciLMForCausalLM": "NEMOTRONNAS",
    "NemotronHForCausalLM": "NEMOTRONH",
    "MixtralForCausalLM": "MOE",
    "GptOssForCausalLM": "MOE",
    "Qwen2MoeForCausalLM": "MOE",
    "Qwen3MoeForCausalLM": "MOE",
    "MiniMaxM2ForCausalLM": "MOE",
    "MiMoV2FlashForCausalLM": "HYBRIDMOE",
    "Llama4ForConditionalGeneration": "HYBRIDMOE",
    "Qwen3_5ForConditionalGeneration": "QWEN35",
    "Qwen3_5MoeForConditionalGeneration": "QWEN35",
}

# Multimodal architectures whose LLM config lives under a nested key (e.g. "text_config").
# _parse_hf_config_json will flatten these before parsing.
MULTIMODAL_TEXT_CONFIG_KEY = {
    "KimiK25ForConditionalGeneration": "text_config",
    "Llama4ForConditionalGeneration": "text_config",
    "Qwen3_5ForConditionalGeneration": "text_config",
    "Qwen3_5MoeForConditionalGeneration": "text_config",
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
    # TensorRT-LLM WideEP specific
    wideep_moe_compute = "wideep_moe_perf.txt"
    # TensorRT-LLM AlltoAll (covers WideEP NVLinkTwoSided + CutlassFusedMoE NVLinkOneSided)
    trtllm_alltoall = "trtllm_alltoall_perf.txt"
    compute_scale = "computescale_perf.txt"
    scale_matrix = "scale_matrix_perf.txt"
    mamba2 = "mamba2_perf.txt"
    gdn = "gdn_perf.txt"
    # Module-level attention profiling (complete self_attn forward)
    mla_context_module = "mla_context_module_perf.txt"
    mla_generation_module = "mla_generation_module_perf.txt"
    dsa_context_module = "dsa_context_module_perf.txt"
    dsa_generation_module = "dsa_generation_module_perf.txt"


QuantMapping = namedtuple("QuantMapping", ["memory", "compute", "name"])


class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """

    float16 = QuantMapping(2, 1, "float16")  # w16a16
    int8_wo = QuantMapping(1, 1, "int8_wo")  # w8a16
    int4_wo = QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8 = QuantMapping(1, 2, "fp8")  # w8fp8
    fp8_static = QuantMapping(1, 2, "fp8_static")  # fp8 with static quantization (compute_scale/scale_matrix modeled)
    sq = QuantMapping(1, 2, "sq")  # w8int8
    fp8_block = QuantMapping(1, 2, "fp8_block")  # specific for trtllm torch ds fp8
    fp8_ootb = QuantMapping(
        1, 2, "fp8_ootb"
    )  # in future, should deprecate this mode as it's specific for trtllm trt backend
    nvfp4 = QuantMapping(9 / 16, 4, "nvfp4")  # nvfp4 on blackwell. 1 fp8 scale per 16 nvfp4 weights.


class MoEQuantMode(Enum):
    """
    MoE quant mode.
    """

    float16 = QuantMapping(2, 1, "float16")  # w16a16
    fp8 = QuantMapping(1, 2, "fp8")  # w8fp8
    int4_wo = QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8_block = QuantMapping(1, 2, "fp8_block")  # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, "w4afp8")  # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(9 / 16, 4, "nvfp4")  # nvfp4 on blackwell. 1 fp8 scale per 16 nvfp4 weights.
    w4a16_mxfp4 = QuantMapping(0.5, 1, "w4a16_mxfp4")  # native data format for gpt oss
    w4a8_mxfp4_mxfp8 = QuantMapping(0.5, 2, "w4a8_mxfp4_mxfp8")
    # mxfp4 weights, mxfp8 activations (recommended for Blackwell)


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
