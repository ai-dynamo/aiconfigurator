# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import cache
from importlib import resources as pkg_resources

from packaging.version import InvalidVersion, Version


def parse_support_matrix_version(version: str | None) -> Version | None:
    """Parse a support-matrix backend version as PEP 440, or return None."""
    if not version:
        return None
    try:
        return Version(version)
    except InvalidVersion:
        return None


SupportMatrixSystemOrder = (
    "b200",
    "gb200",
    "b300",
    "gb300",
    "rtx_pro_6000",
    "h200",
    "h100",
    "l40s",
    "a100",
    "b60",
)


def get_support_matrix_system_sort_key(system: str) -> tuple[int, str]:
    """Sort support-matrix systems by product priority, then by name."""
    normalized_system = system.lower()
    for index, prefix in enumerate(SupportMatrixSystemOrder):
        if normalized_system.startswith(prefix):
            return index, normalized_system
    return len(SupportMatrixSystemOrder), normalized_system


def sort_support_matrix_systems(systems):
    """Return systems in the preferred support-matrix display order."""
    return sorted(systems, key=get_support_matrix_system_sort_key)


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
        moe_latent_size (int): Latent dim for routed-expert projections (0 disables
            latent compression and routed experts run on hidden_size directly).
            Used by latent-MoE variants like Nemotron-3-Super.
    """

    hybrid_override_pattern: str
    mamba_num_heads: int
    mamba_head_dim: int
    ssm_state_size: int
    conv_kernel: int
    n_groups: int
    chunk_size: int
    moe_shared_expert_intermediate_size: int = 0  # Optional: 0 for non-MoE NemotronH models
    moe_latent_size: int = 0  # Optional: 0 means routed experts use hidden_size directly


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
class VisionEncoderConfig:
    """
    Configuration for the vision encoder (ViT) component of multimodal VL models.

    Covers Qwen3-VL and similar vision-language architectures where the visual
    encoder is a separate ViT that runs before the LLM backbone.

    Attributes:
        depth (int): Number of ViT transformer layers
        hidden_size (int): Hidden dimension of the ViT
        num_heads (int): Number of attention heads in the ViT
        intermediate_size (int): FFN intermediate size in the ViT
        patch_size (int): Spatial patch size in pixels (applied to H and W)
        temporal_patch_size (int): Temporal patch size for video inputs (1 for image-only)
        spatial_merge_size (int): Pixel-shuffle reduction factor applied after ViT
            (e.g., 2 means 2x2 patches are merged, dividing token count by 4)
        out_hidden_size (int): Output projection dimension (must match LLM hidden size)
        projector_dims (tuple[tuple[int, int], ...]): Per-layer (in_dim, out_dim) pairs
            for the vision-to-LLM projector MLP. Empty tuple means no projector.
            Dimensions are absolute (before TP sharding); build_encoder_ops applies TP.
        projector_n_instances (int): Number of projector instances to model (e.g.,
            1 + len(deepstack_visual_indexes) for Qwen3VL deepstack variants).
        partial_rotary_factor (float): Fraction of head_dim that RoPE rotates on Q/K
            in each ViT attention block. 0.0 means no RoPE.
    """

    depth: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    deepstack_visual_indexes: tuple[int, ...] = ()
    projector_dims: tuple[tuple[int, int], ...] = ()
    projector_n_instances: int = 1
    partial_rotary_factor: float = 0.0


@dataclass(frozen=True)
class Gemma4MixConfig:
    """Config for Google Gemma 4 (gemma4_text) hybrid attention + dense-MLP-plus-MoE FFN.

    Every layer runs both a shared dense MLP (intermediate_size, ``Gemma4TextMLP``) and a
    routed top-k MoE branch in parallel, summed at the end of the block. Attention shape
    differs per layer type:
      - sliding_attention (SWA): num_key_value_heads x head_dim, separate K and V projections,
        token window = sliding_window_size.
      - full_attention (global): num_global_key_value_heads x global_head_dim, K=V at the
        projection (no v_proj) when attention_k_eq_v is set, no window cap.

    Shared dense MLP intermediate is the model-level ``inter_size`` (HF ``intermediate_size``).
    Routed-expert intermediate is the model-level ``moe_inter_size`` (HF ``moe_intermediate_size``).
    """

    layer_types: tuple[str, ...]  # per-layer: "sliding_attention" or "full_attention"
    swa_num_kv_heads: int  # KV heads on sliding_attention layers
    swa_head_dim: int  # Q/K/V head dim on sliding_attention layers
    global_num_kv_heads: int  # KV heads on full_attention layers
    global_head_dim: int  # Q/K/V head dim on full_attention layers
    sliding_window_size: int  # token window for sliding_attention layers
    attention_k_eq_v: bool = False  # true means global layers reuse K as V (no v_proj)


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


@dataclass(frozen=True)
class DeepSeekV4Config:
    """Config fields unique to DeepSeek-V4 compressed attention + mHC models."""

    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    head_dim: int
    qk_rope_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    compress_rope_theta: int
    num_hash_layers: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    n_shared_experts: int = 1


def indexer_cache_entry_bytes(index_head_dim: int) -> int:
    """Bytes per token in the FP8 indexer KV cache, including one scale per 128 values."""
    return index_head_dim + ((index_head_dim + 127) // 128) * 4


def deepseek_v4_indexer_cache_entry_bytes(index_head_dim: int) -> float:
    """Bytes per compressed token in DeepSeek-V4's FP4 indexer KV cache."""
    return index_head_dim * 0.5


DEEPSEEK_V4_HF_MODELS = frozenset(
    {
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
        "sgl-project/DeepSeek-V4-Flash-FP8",
        "sgl-project/DeepSeek-V4-Pro-FP8",
    }
)


def _iter_support_matrix_resources():
    """Yield support matrix CSV resources in deterministic order."""
    systems_resource = pkg_resources.files("aiconfigurator_core") / "systems"
    split_matrix_resource = systems_resource / "support_matrix"

    if split_matrix_resource.is_dir():
        csv_resources = {
            resource.name: resource for resource in split_matrix_resource.iterdir() if resource.name.endswith(".csv")
        }
        index_resource = split_matrix_resource / "index.json"
        if index_resource.is_file():
            try:
                index_data = json.loads(index_resource.read_text())
                ordered_files = index_data.get("files", []) if isinstance(index_data, dict) else []
            except (OSError, json.JSONDecodeError):
                ordered_files = []

            for file_name in ordered_files:
                resource = csv_resources.pop(file_name, None)
                if resource is not None:
                    yield resource

        yield from sorted(
            csv_resources.values(),
            key=lambda resource: get_support_matrix_system_sort_key(resource.name.removesuffix(".csv")),
        )
        return

    legacy_matrix_resource = systems_resource / "support_matrix.csv"
    if legacy_matrix_resource.is_file():
        yield legacy_matrix_resource


@cache
def get_support_matrix() -> list[dict[str, str]]:
    """
    Get the support matrix as a list of dictionaries.

    Returns:
        list[dict[str, str]]: List of rows from the support matrix CSV files.
    """
    results = []
    for csv_resource in _iter_support_matrix_resources():
        # Use as_file() context manager for proper package resource access.
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


def _is_silicon_support_row(row: dict[str, str]) -> bool:
    """Return whether a matrix row proves support from collected SILICON data.

    ``Source`` is optional for compatibility with legacy matrices.  During the
    transition to the explicit ``HYBRID_PASS`` status, reject old rows that used
    ``PASS`` together with an empirical/transfer source.
    """
    if row.get("Status") != "PASS":
        return False
    # Legacy 8/9-column matrices have no Source key. Current 10-column rows do,
    # and must identify collected silicon explicitly rather than with an empty cell.
    return "Source" not in row or row.get("Source") == "silicon"


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
        if backend and row["Backend"].lower() != backend.lower():
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
    matrix_arch = next((row["Architecture"] for row in matrix if row["HuggingFaceID"].lower() == model.lower()), None)

    if exact_matches:
        return SupportResult(
            agg_supported=any(_is_silicon_support_row(row) for row in exact_matches if row["Mode"] == "agg"),
            disagg_supported=any(_is_silicon_support_row(row) for row in exact_matches if row["Mode"] == "disagg"),
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
        if row["Architecture"] == architecture
        and row["System"].lower() == system.lower()
        and _matches_filters(row, backend, version)
        and row["Status"] != "HW_INCOMPATIBLE"
    ]

    agg_results = [_is_silicon_support_row(row) for row in arch_matches if row["Mode"] == "agg"]
    disagg_results = [_is_silicon_support_row(row) for row in arch_matches if row["Mode"] == "disagg"]

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
    Get the set of supported architectures from the support matrix CSV files.

    Returns:
        set[str]: Set of architecture names that have at least one PASSing configuration.
    """
    matrix = get_support_matrix()
    return {row["Architecture"] for row in matrix if _is_silicon_support_row(row)}


@cache
def get_default_models() -> set[str]:
    """
    Get the set of default HuggingFace model IDs.

    Returns:
        set[str]: Set of unique HuggingFace model IDs from the support matrix
            plus locally cached default model configs.
    """
    models = {row["HuggingFaceID"] for row in get_support_matrix()}
    models.update(DefaultHFModels)
    return models


"""
Cached HuggingFace model configs - these are pre-downloaded and stored in model_configs/
Model parameters are parsed from these configs via get_model_config_from_model_path() in utils.py
The list of default models for testing is derived from the support matrix CSV files
and this set via get_default_models()
"""
DefaultHFModels = {
    # Llama 3.1 Models
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-70B",
    "meta-llama/Meta-Llama-3.1-405B",
    "nvidia/Llama-3.1-70B-Instruct-FP8",
    # DeepSeek R1
    "deepseek-ai/DeepSeek-R1",
    # DeepSeek V3/V3.1 Models
    "deepseek-ai/DeepSeek-V3",
    "nvidia/DeepSeek-V3.1-NVFP4",
    # Kimi K2.5 Models
    "moonshotai/Kimi-K2.5",
    "nvidia/Kimi-K2.5-NVFP4",
    # DeepSeek V3.2 / GLM-5 (DEEPSEEKV32 family)
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-5",
    "zai-org/GLM-5-FP8",
    "nvidia/GLM-5-NVFP4",
    "zai-org/GLM-5.1",
    "zai-org/GLM-5.1-FP8",
    "nvidia/GLM-5.1-NVFP4",
    "zai-org/GLM-5.2",
    "zai-org/GLM-5.2-FP8",
    "nvidia/GLM-5.2-NVFP4",
    # DeepSeek V4
    *DEEPSEEK_V4_HF_MODELS,
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
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    # MiniMax Models
    "MiniMaxAI/MiniMax-M2.5",
    "nvidia/MiniMax-M2.5-NVFP4",
    "MiniMaxAI/MiniMax-M2.7",
    "nvidia/MiniMax-M2.7-NVFP4",
    "MiniMaxAI/MiniMax-M3",
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
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8",
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
    "nvidia/Nemotron-H-56B-Base-8K",
    # Google Gemma 4 Models
    "google/gemma-4-26B-A4B",
}

"""
Supported systems (GPU types)
"""
SupportedSystems = {
    "h100_sxm",
    "h100_pcie",
    "h200_sxm",
    "b200_sxm",
    "b300_sxm",
    "gb200",
    "gb300",
    "a100_sxm",
    "a100_pcie",
    "a30",
    "l4",
    "l40s",
    "b60",
    "rtx_pro_6000_server",
}

"""
Model family for model definition
"""
ModelFamily = {
    "GPT",
    "LLAMA",
    "MOE",
    "DEEPSEEK",
    "DEEPSEEKV32",
    "DEEPSEEKV4",
    "KIMIK25",
    "NEMOTRONNAS",
    "NEMOTRONH",
    "HYBRIDMOE",
    "QWEN35",
    "QWEN3VL",
    "QWEN3VL_MOE",
    "GEMMA4MIX",
    "MINIMAXM3",
}
ARCHITECTURE_TO_MODEL_FAMILY = {
    "LlamaForCausalLM": "LLAMA",
    "Qwen2ForCausalLM": "LLAMA",
    "Qwen3ForCausalLM": "LLAMA",
    "Qwen3VLForConditionalGeneration": "QWEN3VL",
    "Qwen3VLMoeForConditionalGeneration": "QWEN3VL_MOE",
    "MiMoForCausalLM": "LLAMA",
    "DeepSeekForCausalLM": "DEEPSEEK",
    "DeepseekV3ForCausalLM": "DEEPSEEK",
    "DeepseekV32ForCausalLM": "DEEPSEEKV32",
    "GlmMoeDsaForCausalLM": "DEEPSEEKV32",
    "DeepseekV4ForCausalLM": "DEEPSEEKV4",
    "KimiK25ForConditionalGeneration": "KIMIK25",
    "NemotronForCausalLM": "NEMOTRONNAS",
    "DeciLMForCausalLM": "NEMOTRONNAS",
    "NemotronHForCausalLM": "NEMOTRONH",
    "MixtralForCausalLM": "MOE",
    "GptOssForCausalLM": "MOE",
    "Qwen2MoeForCausalLM": "MOE",
    "Qwen3MoeForCausalLM": "MOE",
    "MiniMaxM2ForCausalLM": "MOE",
    "MiniMaxM3ForCausalLM": "MINIMAXM3",
    "MiniMaxM3SparseForConditionalGeneration": "MINIMAXM3",
    "MiMoV2FlashForCausalLM": "HYBRIDMOE",
    "Llama4ForConditionalGeneration": "HYBRIDMOE",
    "Qwen3_5ForConditionalGeneration": "QWEN35",
    "Qwen3_5MoeForConditionalGeneration": "QWEN35",
    "Gemma4ForConditionalGeneration": "GEMMA4MIX",
}

# Multimodal architectures whose LLM config lives under a nested key (e.g. "text_config").
# _parse_hf_config_json will flatten these before parsing.
MULTIMODAL_TEXT_CONFIG_KEY = {
    "KimiK25ForConditionalGeneration": "text_config",
    "Llama4ForConditionalGeneration": "text_config",
    "Qwen3_5ForConditionalGeneration": "text_config",
    "Qwen3_5MoeForConditionalGeneration": "text_config",
    "Gemma4ForConditionalGeneration": "text_config",
    "Qwen3VLForConditionalGeneration": "text_config",
    "Qwen3VLMoeForConditionalGeneration": "text_config",
    "MiniMaxM3SparseForConditionalGeneration": "text_config",
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
    "encoder_latency",
    "encoder_memory",
    "context_latency",
    "generation_latency",
    "num_total_gpus",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
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
    "encoder_latency",
    "encoder_memory",
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
    "cp",
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
    "encoder_latency",
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
    "(p)cp",
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
    "(e)workers",
    "(e)tp",
    "(e)pp",
    "(e)parallel",
    "(e)memory",
    "power_w",  # NEW: E2E weighted average power in watts
]

"""
Columns for AFD (Attention-FFN Disaggregated) inference summary dataframe

AFD is orthogonal to P/D disaggregation: the same schema is used whether
AFD is applied to the prefill phase, the decode phase, or both.

Per-phase layer scalars (``t_a_layer`` / ``t_f_layer`` / ``t_a2f_layer`` /
``t_f2a_layer`` / ``t_c_layer`` / ``t_step`` / ``balance_ratio`` /
``comm_hidden``) appear in three forms:

* ``<scalar>``                       -- un-prefixed "headline" value.
* ``prefill_<scalar>`` / ``decode_<scalar>``  -- per-phase paired values.

Filling rules:

* ``phase="prefill"`` -- un-prefixed and ``prefill_*`` reflect the prefill
  estimate; ``decode_*`` are NaN/None.
* ``phase="decode"``  -- mirror of the above.
* ``phase="both"``    -- ``prefill_*`` and ``decode_*`` carry the two
  estimates; the un-prefixed scalars are NaN/None (refusing to pick a single
  "headline" value when two phases run and may diverge).
* AFD-with-PD combined runs always have ``phase`` set to the AFD side
  (``"prefill"`` or ``"decode"``); the static side's scalars are NaN/None
  in the corresponding ``prefill_*``/``decode_*`` slot to flag "this side
  was not estimated under AFD".
"""
ColumnsAFD = [
    "model",
    "phase",
    "isl",
    "osl",
    "gpus_per_node",
    "(a)nodes",
    "(a)tp",
    "(a)bs",
    "(a)micro_bs",
    "(a)workers",
    "(a)memory",
    "(a)is_oom",
    "(f)nodes",
    "(f)tp",
    "(f)ep",
    "(f)workers",
    "(f)memory",
    "(f)is_oom",
    "t_a_layer",
    "t_f_layer",
    "t_a2f_layer",
    "t_f2a_layer",
    "t_c_layer",
    "t_step",
    "balance_ratio",
    "comm_hidden",
    "prefill_t_a_layer",
    "prefill_t_f_layer",
    "prefill_t_a2f_layer",
    "prefill_t_f2a_layer",
    "prefill_t_c_layer",
    "prefill_t_step",
    "prefill_balance_ratio",
    "prefill_comm_hidden",
    "decode_t_a_layer",
    "decode_t_f_layer",
    "decode_t_a2f_layer",
    "decode_t_f2a_layer",
    "decode_t_c_layer",
    "decode_t_step",
    "decode_balance_ratio",
    "decode_comm_hidden",
    "ttft",
    "tpot",
    "request_latency",
    "b_total",
    "b_micro_total",
    "tokens/s",
    "tokens/s/gpu",
    "tokens/s/user",
    "seq/s",
    "concurrency",
    "pipeline_model",
    "num_microbatches",
    "combined_with_pd",
    "boundary_on_attn",
    "num_total_gpus",
    "memory",
    "backend",
    "version",
    "system",
    "power_w",
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


class TransferKind(Enum):
    """A way the empirical path may borrow utilisation when an op's own slice has no
    data. Each kind is independently enabled/disabled by a transfer policy, so HYBRID/
    EMPIRICAL coverage can be tuned (and the kind that fired is the result's provenance).
    Roughly ordered by decreasing confidence."""

    XSHAPE = "xshape"  # cross-shape within the SAME quant (nearest collected config)
    XQUANT = "xquant"  # cross-quant within the SAME (memory, compute) profile
    XPROFILE = "xprofile"  # cross-quant across profiles (rescaled by a util-level ratio)
    XOP = "xop"  # cross-op (borrow a related op's util, e.g. MSA<-DSA, via util_scale)


# All transfer kinds enabled — the default HYBRID/EMPIRICAL behaviour (backward compatible).
ALL_TRANSFERS: frozenset[TransferKind] = frozenset(TransferKind)

# Named presets along the confidence ladder, for a simple external surface.
TRANSFER_PRESETS: dict[str, frozenset[TransferKind]] = {
    "off": frozenset(),
    "conservative": frozenset({TransferKind.XSHAPE}),
    "balanced": frozenset({TransferKind.XSHAPE, TransferKind.XQUANT}),
    "aggressive": ALL_TRANSFERS,
}


def resolve_transfer_policy(spec) -> frozenset[TransferKind]:
    """Normalise an external transfer-policy spec into a frozenset of TransferKind.

    Accepts: None (-> ALL_TRANSFERS), a preset name, a TransferKind, or any iterable of
    those (names or TransferKind). Unknown names raise ValueError so a typo surfaces."""
    if spec is None:
        return ALL_TRANSFERS
    if isinstance(spec, TransferKind):
        return frozenset({spec})
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in TRANSFER_PRESETS:
            return TRANSFER_PRESETS[key]
        if "," in key:  # comma-separated kinds/presets from CLI/YAML, e.g. "xshape,xquant"
            return resolve_transfer_policy([t for t in (p.strip() for p in key.split(",")) if t])
        return frozenset({_transfer_kind_from_token(key)})
    out: set[TransferKind] = set()
    for item in spec:
        if isinstance(item, TransferKind):
            out.add(item)
        else:
            token = str(item).strip().lower()
            if token in TRANSFER_PRESETS:
                out |= TRANSFER_PRESETS[token]
            else:
                out.add(_transfer_kind_from_token(token))
    return frozenset(out)


def _transfer_kind_from_token(token: str) -> TransferKind:
    for kind in TransferKind:
        if kind.value == token:
            return kind
    valid = ", ".join(sorted(k.value for k in TransferKind))
    presets = ", ".join(sorted(TRANSFER_PRESETS))
    raise ValueError(f"unknown transfer kind/preset {token!r}; valid kinds: {valid}; presets: {presets}")


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

    gemm = "gemm_perf.parquet"
    nccl = "nccl_perf.parquet"
    oneccl = "oneccl_perf.parquet"
    generation_attention = "generation_attention_perf.parquet"
    context_attention = "context_attention_perf.parquet"
    encoder_attention = "encoder_attention_perf.parquet"
    context_mla = "context_mla_perf.parquet"
    generation_mla = "generation_mla_perf.parquet"
    mla_bmm = "mla_bmm_perf.parquet"
    moe = "moe_perf.parquet"
    custom_allreduce = "custom_allreduce_perf.parquet"
    wideep_context_mla = "wideep_context_mla_perf.parquet"
    wideep_generation_mla = "wideep_generation_mla_perf.parquet"
    wideep_context_moe = "wideep_context_moe_perf.parquet"
    wideep_generation_moe = "wideep_generation_moe_perf.parquet"
    wideep_deepep_normal = "wideep_deepep_normal_perf.parquet"
    wideep_deepep_ll = "wideep_deepep_ll_perf.parquet"
    # TensorRT-LLM WideEP specific
    wideep_moe_compute = "wideep_moe_perf.parquet"
    # TensorRT-LLM AlltoAll (covers WideEP NVLinkTwoSided + CutlassFusedMoE NVLinkOneSided)
    trtllm_alltoall = "trtllm_alltoall_perf.parquet"
    compute_scale = "computescale_perf.parquet"
    scale_matrix = "scale_matrix_perf.parquet"
    mamba2 = "mamba2_perf.parquet"
    gdn = "gdn_perf.parquet"
    # Module-level attention profiling (complete self_attn forward)
    mla_context_module = "mla_context_module_perf.parquet"
    mla_generation_module = "mla_generation_module_perf.parquet"
    dsa_context_module = "dsa_context_module_perf.parquet"
    dsa_generation_module = "dsa_generation_module_perf.parquet"
    # NOTE: GLM-5.2 skip-indexer (reuse-layer) rows live in the SAME
    # dsa_*_module file, tagged by the op_name column; the loader splits them
    # via op_kind="full"/"skip" — no separate filename needed here.
    mhc_module = "mhc_module_perf.parquet"
    # DeepSeek-V4 module-level data — one file per (attn_kind ∈ {csa, hca},
    # mode ∈ {context, generation}) = 4 files. Each file contains all
    # (tp_size, gemm_type, b, s) rows for that kind+mode.  SWA layers are
    # folded into HCA at the model layer (see models.py:_attention_ops),
    # so no separate SWA collector / data is needed.
    dsv4_csa_context_module = "dsv4_csa_context_module_perf.parquet"
    dsv4_hca_context_module = "dsv4_hca_context_module_perf.parquet"
    dsv4_csa_generation_module = "dsv4_csa_generation_module_perf.parquet"
    dsv4_hca_generation_module = "dsv4_hca_generation_module_perf.parquet"
    # DeepSeek-V4 sparse-op family — all share one column schema and load
    # through ``operations.dsv4.load_dsv4_sparse_op_data``:
    #   csa_attn / hca_attn / paged_mqa_logits : FMLA & indexer kernel latency,
    #     keyed ``num_heads -> tp -> past_kv -> isl -> bs`` (kernel-level Δ data,
    #     queried by ``_lookup_sparse_kernel``).
    #   csa_topk_calib : two rows/shape (score_mode=flat|top_last); the topK
    #     DELTA (flat-top_last) correction applied to CSA module latency.
    dsv4_paged_mqa_logits_module = "dsv4_paged_mqa_logits_module_perf.parquet"
    dsv4_hca_attn_module = "dsv4_hca_attn_module_perf.parquet"
    dsv4_csa_attn_module = "dsv4_csa_attn_module_perf.parquet"
    dsv4_csa_topk_calib = "dsv4_csa_topk_calib_perf.parquet"
    dsv4_megamoe_module = "dsv4_megamoe_module_perf.parquet"


QuantMapping = namedtuple("QuantMapping", ["memory", "compute", "name"])


class GEMMQuantMode(Enum):
    """
    GEMM quant mode.
    """

    bfloat16 = QuantMapping(2, 1, "bfloat16")  # w16a16
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

    bfloat16 = QuantMapping(2, 1, "bfloat16")  # w16a16
    fp8 = QuantMapping(1, 2, "fp8")  # w8fp8
    int4_wo = QuantMapping(0.5, 1, "int4_wo")  # w4a16
    fp8_block = QuantMapping(1, 2, "fp8_block")  # specific for trtllm torch ds fp8
    w4afp8 = QuantMapping(0.5, 2, "w4afp8")  # specific for trtllm torch ds w4a8
    nvfp4 = QuantMapping(9 / 16, 4, "nvfp4")  # nvfp4 on blackwell. 1 fp8 scale per 16 nvfp4 weights.
    w4a16_mxfp4 = QuantMapping(0.5, 1, "w4a16_mxfp4")  # native data format for gpt oss
    w4a8_mxfp4_mxfp8 = QuantMapping(0.5, 2, "w4a8_mxfp4_mxfp8")
    # mxfp4 weights, mxfp8 activations (recommended for Blackwell)
    w4a8_mxfp4_mxfp8_trtllm = QuantMapping(0.5, 2, "w4a8_mxfp4_mxfp8_trtllm")
    # Blackwell trtllm-gen fused MoE: MXFP4 (E2M1, block-32) weights x MXFP8 (E4M3)
    # activations -- the kernel DeepSeek-V4-Pro actually runs in prefill on sm100
    # (bmm_MxE4m3_MxE2m1MxE4m3 ... sm100f, flashinfer trtllm_fp4_block_scale_moe).
    # Distinct backend from w4a8_mxfp4_mxfp8 above (flashinfer cutedsl). DSV4 MoE
    # weights are stored MXFP4 (I8-packed E2M1 + E8M0 scales), so sglang dispatches
    # by GPU: sm100 -> this (trtllm-gen); sm90 -> w4a16_mxfp4_cutlass below.
    w4a16_mxfp4_cutlass = QuantMapping(0.5, 1, "w4a16_mxfp4_cutlass")
    # Hopper (sm90) DeepSeek-V4-Pro MoE: flashinfer cutlass SM90 mixed GEMM
    # (cutlass_fused_moe(use_w4_group_scaling=True)) -- MXFP4 weights x BF16
    # activations (weight-only). Distinct backend from w4a16_mxfp4 above, which is
    # GPT-OSS's triton_kernels mxfp4 path. (DSV4 Hopper silicon data pending.)


class FMHAQuantMode(Enum):
    """
    FMHA quant mode.
    """

    bfloat16 = QuantMapping(2, 1, "bfloat16")
    fp8 = QuantMapping(1, 2, "fp8")
    fp8_block = QuantMapping(1, 2, "fp8_block")  # FIXME: specific for sglang wideep


class KVCacheQuantMode(Enum):
    """
    KVCache quant mode.
    """

    bfloat16 = QuantMapping(2, 0, "bfloat16")
    int8 = QuantMapping(1, 0, "int8")
    fp8 = QuantMapping(1, 0, "fp8")


class CommQuantMode(Enum):
    """
    Comm quant mode.
    """

    half = QuantMapping(2, 0, "half")
    int8 = QuantMapping(1, 0, "int8")
    fp8 = QuantMapping(1, 0, "fp8")
