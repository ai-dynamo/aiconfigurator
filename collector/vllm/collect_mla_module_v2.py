# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm>=0.17.0"

"""
MLA Module Collector for vLLM — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass (projections + attention +
output), not just the bare attention kernel. Uses vLLM's own modeling code to
construct a single `DeepseekV2MLAAttention` module with dummy weights, then
benchmarks its forward.

MLA vs DSA is determined by the presence of `index_topk` in the HF config.
Op names and data schema are aligned with TRT-LLM's collect_mla_module.py
so that queries can be reused across frameworks.

Supported models and their attention types are defined in SUPPORTED_MODELS.
The collector reads a real HF config, overrides the layer-local shape fields
in-memory, and then instantiates just the attention module.

Usage:
    # MLA context phase (DeepSeek-V3 style)
    python collect_mla_module.py --mode context --model mla

    # DSA generation phase (DeepSeek-V3.2 style)
    python collect_mla_module.py --mode generation --model dsa

    # All models, context phase
    python collect_mla_module.py --mode context

    # Quick single-point test
    python collect_mla_module.py --mode context --model mla --quick --batch-size 4 --seq-len 2048
"""

import argparse
import gc
import json
import math
import os
import tempfile
import traceback
from pathlib import Path

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.platforms import current_platform

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — vLLM 0.16.0 registers the GlmMoeDsaForCausalLM
# model class but omits the config-type mapping for "glm_moe_dsa", so
# AutoConfig.from_pretrained() fails.  The config layout is identical to
# DeepSeek-V3 (GlmMoeDsaForCausalLM inherits DeepseekV2ForCausalLM), so
# reusing DeepseekV3Config is safe.
# ═══════════════════════════════════════════════════════════════════════
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.version import __version__ as vllm_version

from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_vllm_config,
    setup_distributed,
    with_exit_stack,
)

if "glm_moe_dsa" not in _CONFIG_REGISTRY:
    _CONFIG_REGISTRY["glm_moe_dsa"] = "DeepseekV3Config"


# ═══════════════════════════════════════════════════════════════════════
# Local model config resolution — avoid HuggingFace Hub downloads
# ═══════════════════════════════════════════════════════════════════════

# Pre-cached HF configs live in src/aiconfigurator/model_configs/ as
# "<org>--<model>_config.json".  vLLM's ModelConfig accepts a local
# directory containing config.json, so we create a temp dir with a
# symlink when the cached file exists.
_MODEL_CONFIGS_DIR = Path(__file__).resolve().parents[2] / "src" / "aiconfigurator" / "model_configs"

# Cache of model_name -> temp dir path (created once per process).
_local_config_cache: dict[str, str] = {}


def _resolve_model_path(model_name: str) -> str:
    """Return a local directory path for *model_name* if a cached config exists, else return model_name as-is."""
    if model_name in _local_config_cache:
        return _local_config_cache[model_name]

    config_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_config.json"
    if not config_file.exists():
        return model_name

    # Create a temp directory with config.json so vLLM's ModelConfig
    # loads from disk instead of downloading from HuggingFace Hub.
    tmp_dir = tempfile.mkdtemp(prefix=f"aic_model_{model_name.replace('/', '_')}_")
    os.symlink(config_file, os.path.join(tmp_dir, "config.json"))
    # Strip auto_map if present.  Some models (e.g. DeepSeek-V3) ship
    # config.json with auto_map pointing to a custom Python config class
    # (configuration_deepseek.py).  HuggingFace's AutoConfig.from_pretrained()
    # — called by vLLM's ModelConfig — unconditionally tries to import that
    # module from the model directory where it doesn't exist; vLLM natively
    # supports these architectures and only needs the JSON fields.
    with open(config_file) as f:
        config_data = json.load(f)
    if "auto_map" in config_data:
        config_data.pop("auto_map")
        os.remove(os.path.join(tmp_dir, "config.json"))
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_data, f)

    # Also symlink hf_quant_config.json if present (used by quantized models).
    quant_file = _MODEL_CONFIGS_DIR / f"{model_name.replace('/', '--')}_hf_quant_config.json"
    if quant_file.exists():
        os.symlink(quant_file, os.path.join(tmp_dir, "hf_quant_config.json"))

    _local_config_cache[model_name] = tmp_dir
    return tmp_dir


# ═══════════════════════════════════════════════════════════════════════
# Supported Models — model_path → attention type
# ═══════════════════════════════════════════════════════════════════════

SUPPORTED_MODELS: dict[str, str] = {
    "deepseek-ai/DeepSeek-V3": "mla",
    "deepseek-ai/DeepSeek-V3.2": "dsa",
    "zai-org/GLM-5": "dsa",
}


# ═══════════════════════════════════════════════════════════════════════
# Test Cases — aligned with TRT-LLM's collect_mla_module.py
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str):
    """Return (compute_dtype, kv_cache_dtype, gemm_type) triples for a phase.

    Each triple describes the full quantisation configuration for one
    benchmark sweep.  GPU capability (SM version) determines which
    combos are available.

    Precision axes:
      gemm_type    — linear-layer GEMMs (projections inside the module)
        bfloat16:  always
        fp8_block: SM >= 89 (Ada / Hopper / Blackwell)
        nvfp4:     SM >= 100 (Blackwell)

      (compute_dtype, kv_cache_dtype) — attention compute + KV cache
        context:    (bf16, bf16) always;  (fp8, fp8) SM >= 100
        generation: (bf16, bf16) always;  (bf16, fp8) SM >= 90
    """
    sm = get_sm_version()

    gemm_types = ["bfloat16"]
    if sm >= 89:
        gemm_types.append("fp8_block")
    if sm >= 100:
        gemm_types.append("nvfp4")

    attn_combos = [("bfloat16", "bfloat16")]
    if phase == "context":
        if sm >= 100:
            attn_combos.append(("fp8", "fp8"))
    else:
        if sm >= 90:
            attn_combos.append(("bfloat16", "fp8"))

    return [(c, kv, g) for g in gemm_types for c, kv in attn_combos]


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
    base_fname = f"{attn_type}_context_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context"):
        for num_heads in [128, 64, 32, 16, 8, 4, 2, 1]:
            for b in b_list:
                for s in s_list:
                    if b * s > 131072:
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, perf_filename].
    """
    cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    s_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    base_fname = f"{attn_type}_generation_module_perf.txt"
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation"):
        for num_heads in [128, 64, 32, 16, 8, 4, 2, 1]:
            for b in b_list:
                for s in s_list:
                    if b * s > 1024 * 4096 * 2 * 2 * 2:
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, base_fname])
    return cases


def _build_module_test_cases(attn_type: str, mode: str):
    """Build module-level test cases for a specific attention type and phase.

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, perf_filename, model_path, attn_type]
    """
    base_cases = get_context_test_cases(attn_type) if mode == "context" else get_generation_test_cases(attn_type)
    model_paths = [m for m, t in SUPPORTED_MODELS.items() if t == attn_type]
    cases = []
    for model_path in model_paths:
        for s, b, h, kv_dtype, compute_dtype, gemm_type, fname in base_cases:
            cases.append([s, b, h, kv_dtype, compute_dtype, gemm_type, fname, model_path, attn_type])
    return cases


def get_mla_context_module_test_cases():
    """collect.py entrypoint for MLA context module collection."""
    return _build_module_test_cases(attn_type="mla", mode="context")


def get_mla_generation_module_test_cases():
    """collect.py entrypoint for MLA generation module collection."""
    return _build_module_test_cases(attn_type="mla", mode="generation")


def get_dsa_context_module_test_cases():
    """collect.py entrypoint for DSA context module collection."""
    return _build_module_test_cases(attn_type="dsa", mode="context")


def get_dsa_generation_module_test_cases():
    """collect.py entrypoint for DSA generation module collection."""
    return _build_module_test_cases(attn_type="dsa", mode="generation")


# ═══════════════════════════════════════════════════════════════════════
# Module Construction
# ═══════════════════════════════════════════════════════════════════════


def _create_gemm_quant_config(gemm_type: str):
    """Create the vLLM QuantizationConfig for a given gemm_type.

    Returns None for bfloat16 (unquantised GEMMs).
    For fp8_block / nvfp4, returns an online-quantisation config so that
    dummy BF16 weights are dynamically quantised during
    ``process_weights_after_loading``.
    """
    if gemm_type == "bfloat16":
        return None
    if gemm_type == "fp8_block":
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        # vLLM requires is_checkpoint_fp8_serialized=True for block-scaled
        # FP8 (fp8.py raises ValueError otherwise).  This routes through
        # Fp8LinearMethod (block_quant=True) → W8A8BlockFp8LinearOp →
        # DeepGEMM on SM≥89.
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    if gemm_type == "nvfp4":
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
        )

        return ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
    raise ValueError(f"Unknown gemm_type: {gemm_type!r}")


def _create_attention_module(
    model_path: str,
    attn_type: str,
    num_heads: int,
    use_fp8_kv_cache: bool,
    use_prefill_fp8: bool,
    max_seq_len: int,
    max_batch_size: int,
    gemm_type: str = "bfloat16",
    device: str = "cuda:0",
    is_context: bool = True,
):
    """
    Create a DeepseekV2MLAAttention module from vLLM's own modeling code.

    Loads a real HF config from model_path, overrides the layer-local attention
    dimensions we want to benchmark in-memory, and then constructs the module
    with dummy weights. The module includes all projections + attention +
    output.

    Args:
        model_path: HuggingFace model path (e.g. "deepseek-ai/DeepSeek-V3.2").
        attn_type: Attention type ("mla" or "dsa").
        use_prefill_fp8: When True and on SM100+, enable FP8 prefill
            attention via ``attention_config.use_prefill_query_quantization``.
        gemm_type: Precision for linear-layer GEMMs — "bfloat16",
            "fp8_block", or "nvfp4".
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

    local_model_path = _resolve_model_path(model_path)

    block_size = 64
    max_model_len = max(max_seq_len + 1, 4096)
    num_kv_cache_blocks = max(
        1 + math.ceil((max_seq_len + 1) / block_size) * max_batch_size,
        8192,
    )

    # Determine kv cache dtype string for sparse MLA.
    # For DSA (DeepSeekV3.2), fp8 uses the custom ``fp8_ds_mla`` 656-byte
    # cache format (512B quantized NoPE + 16B scales + 128B RoPE).
    # For dense MLA, standard fp8 (fp8_e4m3) is used.
    is_dsa = attn_type == "dsa"

    vllm_config = create_vllm_config(
        model_name=local_model_path,
        max_model_len=max_model_len,
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=max_batch_size,
        max_num_batched_tokens=max(max_batch_size * max_seq_len, 131072) if is_context else max_batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
        trust_remote_code=True,
    )

    # Override quant_config to control linear-layer GEMM precision.
    # DeepSeek-V3.2 ships with FP8 quantisation by default, so we
    # must always set quant_config explicitly: None for bf16,
    # Fp8Config (blockwise) for fp8_block, ModelOptNvFp4Config for nvfp4.
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    # For DSA, mirror the DeepseekV32ForCausalLM.verify_and_update_config()
    # logic: fp8 cache must use ``fp8_ds_mla`` format.
    if is_dsa and use_fp8_kv_cache:
        vllm_config.cache_config.cache_dtype = "fp8_ds_mla"

    # Enable FP8 prefill attention on SM100+ (Blackwell).
    # This quantizes Q/K/V to FP8 before sending to the prefill kernel.
    if use_prefill_fp8:
        vllm_config.attention_config.use_prefill_query_quantization = True

    # Override just the layer-local dimensions we sweep in the collector.
    hf_config = vllm_config.model_config.hf_config
    hf_config.num_hidden_layers = 1
    hf_config.num_attention_heads = num_heads
    hf_config.num_key_value_heads = num_heads

    # Create topk_indices_buffer for DSA
    topk_indices_buffer = None
    if is_dsa and hasattr(hf_config, "index_topk"):
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_indices_buffer = torch.empty(
            max_tokens,
            hf_config.index_topk,
            dtype=torch.int32,
            device=device,
        )

    # Build the attention module inside set_current_vllm_config() context.
    # FP8 quantized Linear layers (QuantFP8 / CustomOp) call
    # get_current_vllm_config() during __init__, so the config must be set.
    # set_default_torch_dtype is required because MLAAttention.__init__
    # calls torch.get_default_dtype() to select the attention backend
    # (MLA backends only support bfloat16, not float32).
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vllm_config), set_default_torch_dtype(vllm_config.model_config.dtype):
        attn_module = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_config,
            hidden_size=hf_config.hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=hf_config.qk_nope_head_dim,
            qk_rope_head_dim=hf_config.qk_rope_head_dim,
            v_head_dim=hf_config.v_head_dim,
            q_lora_rank=hf_config.q_lora_rank if hasattr(hf_config, "q_lora_rank") else None,
            kv_lora_rank=hf_config.kv_lora_rank,
            max_position_embeddings=hf_config.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix="model.layers.0.self_attn",
            topk_indices_buffer=topk_indices_buffer,
        )

    # Serialized block-scaled FP8 creates weight params on meta device;
    # to() cannot copy meta tensors, so use to_empty() when needed.
    if any(p.is_meta for p in attn_module.parameters()):
        attn_module = attn_module.to_empty(device=torch.device(device))
    else:
        attn_module = attn_module.to(device)
    attn_module.eval()
    attn_module.requires_grad_(False)

    # Initialize with random weights.
    # FP8 weights → zero (safe dummy value).
    # Scale params → 1.0 (avoid NaN during process_weights_after_loading).
    # Everything else → small constant.
    #
    # Deterministic init — vLLM 0.17.0 DSA modules leave CUDA graph RNG
    # offset tracking active after construction (likely from FlashInfer
    # sparse MLA backend, vllm-project/vllm#33451 / vllm-project/vllm#34457).
    # Any RNG call (normal_, uniform_, randn) crashes with "Offset increment
    # outside graph capture".  Using fill_() is safe because kernel latency
    # depends on shapes/dtypes, not values, and dummy weights are overwritten
    # by process_weights_after_loading() anyway.
    # See: https://github.com/vllm-project/vllm/issues/39371
    with torch.no_grad():
        for name, tensor in list(attn_module.named_parameters()) + list(attn_module.named_buffers()):
            if tensor.is_meta:
                continue
            if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8):
                tensor.data.zero_()
            elif tensor.dtype == torch.float32 and "scale" in name:
                tensor.data.fill_(0.5)
            else:
                tensor.data.fill_(0.01)

    return attn_module, vllm_config


def _process_module_weights(attn_module, vllm_config, device):
    """Process weights after loading, mimicking vLLM's model loader.

    This must be called after module construction to:
      1. Run FP8 quantization on linear layer weights.
      2. Create W_UK_T and W_UV matrices in MLAAttention that are
         required for the forward pass.
    """
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

    with set_current_vllm_config(vllm_config):
        # 1. Process quantized linear layers (FP8 weight conversion).
        for _, module in attn_module.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module)

        # 2. Process MLAAttention layers (creates W_UK_T, W_UV).
        for _, module in attn_module.named_modules():
            if isinstance(module, MLAAttention) and hasattr(module, "process_weights_after_loading"):
                module.process_weights_after_loading(vllm_config.model_config.dtype)


def _create_context_kv_inputs(batch_spec: BatchSpec, kv_lora_rank: int, qk_rope_head_dim: int, device: str):
    """
    Create cached KV tensors for the tokens that already exist in the paged KV
    cache before the current forward() call.

    Context phase: cache is empty because all tokens are processed in the same
    prefill step.
    Generation phase: cache holds seq_len - 1 historical tokens, and the
    current forward processes exactly 1 new token per request.
    """
    kv_c_contexts = []
    k_pe_contexts = []
    for seq_len, query_len in zip(batch_spec.seq_lens, batch_spec.query_lens, strict=True):
        context_len = max(0, int(seq_len) - int(query_len))
        kv_c_contexts.append(torch.full((context_len, kv_lora_rank), 0.01, dtype=torch.bfloat16, device=device))
        k_pe_contexts.append(torch.full((context_len, 1, qk_rope_head_dim), 0.01, dtype=torch.bfloat16, device=device))
    return kv_c_contexts, k_pe_contexts


def _populate_indexer_kv_cache(
    indexer_kv_cache: torch.Tensor,
    common_attn_metadata,
    context_lens: list[int],
) -> None:
    """
    Populate the DSA indexer cache so generation benchmarks see a realistic
    historical K cache instead of an all-zero buffer.
    """
    block_table = common_attn_metadata.block_table_tensor
    block_size = indexer_kv_cache.shape[1]
    entry_dim = indexer_kv_cache.shape[2]
    device = indexer_kv_cache.device

    for i, context_len in enumerate(context_lens):
        if context_len <= 0:
            continue
        token_offsets = torch.arange(context_len, dtype=torch.long, device=device)
        block_indices = token_offsets // block_size
        intra_block_offsets = token_offsets % block_size
        block_ids = block_table[i, block_indices]
        dummy_cache = torch.full((context_len, entry_dim), 42, dtype=torch.uint8, device=device)
        indexer_kv_cache[block_ids, intra_block_offsets, :] = dummy_cache


# ═══════════════════════════════════════════════════════════════════════
# KV Cache + Metadata
# ═══════════════════════════════════════════════════════════════════════


def _create_kv_cache_and_metadata(
    vllm_config,
    attn_type: str,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    is_context: bool,
    use_fp8_kv_cache: bool,
    device: str = "cuda:0",
):
    """Create KV cache and attention metadata for benchmarking."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    hf_config = vllm_config.model_config.hf_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    head_dim = kv_lora_rank + qk_rope_head_dim
    block_size = vllm_config.cache_config.block_size
    is_dsa = attn_type == "dsa"

    if is_context:
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * batch_size,
            query_lens=[seq_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    num_kv_cache_blocks = max(
        1 + math.ceil((seq_len + 1) / block_size) * batch_size,
        8192,
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, torch.device(device), arange_block_indices=True
    )

    # Select the correct dtype for cache.
    # DSA fp8 uses a custom 656-byte ``fp8_ds_mla`` cache format that
    # stores quantised NoPE + per-128-element scales + BF16 RoPE.
    # Dense MLA fp8 uses standard fp8_e4m3.
    if is_dsa and use_fp8_kv_cache:
        cache_dtype = current_platform.fp8_dtype()
        kv_cache_dtype_str = "fp8_ds_mla"
    elif use_fp8_kv_cache:
        cache_dtype = current_platform.fp8_dtype()
        kv_cache_dtype_str = "fp8"
    else:
        cache_dtype = torch.bfloat16
        kv_cache_dtype_str = None

    # Populate KV cache with the tokens that exist before this forward.
    kv_c_contexts, k_pe_contexts = _create_context_kv_inputs(
        batch_spec=batch_spec,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        device=device,
    )

    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=kv_c_contexts,
        k_pe_contexts=k_pe_contexts,
        block_size=block_size,
        head_size=head_dim,
        dtype=cache_dtype,
        device=torch.device(device),
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype_str,
    )

    # Build attention metadata via backend builder
    backend_cls = _get_attention_backend(vllm_config, head_dim, use_fp8_kv_cache, is_dsa)
    builder_cls = backend_cls.get_builder_cls()

    kv_cache_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,  # MLA uses 1 KV head
        head_size=head_dim,
        dtype=cache_dtype,
        sliding_window=None,
        cache_dtype_str=kv_cache_dtype_str,
    )

    attn_layer_name = "model.layers.0.self_attn.attn"
    layer_names = [attn_layer_name]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, torch.device(device))
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # For DSA, the Indexer has its own KV cache and metadata builder.
    indexer_kv_cache = None
    indexer_metadata = None
    if is_dsa:
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerBackend,
        )

        index_head_dim = hf_config.index_head_dim
        quant_block_size = 128
        indexer_head_dim = index_head_dim + index_head_dim // quant_block_size * 4

        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=indexer_head_dim,
            dtype=torch.uint8,
        )
        indexer_kv_cache = torch.zeros(
            num_kv_cache_blocks,
            block_size,
            indexer_head_dim,
            dtype=torch.uint8,
            device=device,
        )
        indexer_builder_cls = DeepseekV32IndexerBackend.get_builder_cls()
        indexer_builder = indexer_builder_cls(indexer_spec, [indexer_layer_name], vllm_config, torch.device(device))
        indexer_metadata = indexer_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        _populate_indexer_kv_cache(
            indexer_kv_cache=indexer_kv_cache,
            common_attn_metadata=common_attn_metadata,
            context_lens=[tensor.shape[0] for tensor in kv_c_contexts],
        )

    return kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata


def _get_attention_backend(vllm_config, head_dim, use_fp8_kv_cache, is_dsa):
    """Select attention backend based on GPU capability and config.

    The backend selector uses kv_cache_dtype to pick the right implementation:
      - DSA fp8 → ``fp8_ds_mla`` (FlashMLA Sparse custom format)
      - MLA fp8 → ``fp8`` (standard fp8_e4m3)
      - BF16   → None / "auto"
    """
    dtype = torch.bfloat16

    # Compute the kv_cache_dtype token the selector expects.
    if is_dsa and use_fp8_kv_cache:
        kv_cache_dtype_val = "fp8_ds_mla"
    elif use_fp8_kv_cache:
        kv_cache_dtype_val = "fp8"
    else:
        kv_cache_dtype_val = None

    from vllm.utils.import_utils import resolve_obj_by_qualname
    from vllm.v1.attention.selector import AttentionSelectorConfig

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_dim,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype_val,
        block_size=vllm_config.cache_config.block_size,
        use_mla=True,
        has_sink=False,
        use_sparse=is_dsa,
    )
    backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    return resolve_obj_by_qualname(backend)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


@with_exit_stack
def run_mla_module(
    exit_stack,
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single MLA / DSA module-level benchmark point."""
    setup_distributed(device)
    torch.cuda.set_device(device)

    # DSA's sparse_attn_indexer requires a WorkspaceManager.
    try:
        from vllm.v1.worker.workspace import init_workspace_manager

        init_workspace_manager(torch.device(device))
    except (ImportError, RuntimeWarning):
        pass

    use_fp8_kv_cache = kv_cache_dtype == "fp8"
    use_prefill_fp8 = compute_dtype == "fp8"
    is_context = "context" in perf_filename
    phase = "context" if is_context else "generation"
    variant = attn_type.upper()
    print(
        f"\n[{variant} module] {phase} b={batch_size}, s={seq_len}, "
        f"heads={num_heads}, gemm={gemm_type}, compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention module
    attn_module, vllm_config = _create_attention_module(
        model_path=model_path,
        attn_type=attn_type,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        use_prefill_fp8=use_prefill_fp8,
        max_seq_len=seq_len,
        max_batch_size=batch_size,
        gemm_type=gemm_type,
        device=device,
        is_context=is_context,
    )

    # 1b. Process weights (FP8 quantization + create W_UK_T / W_UV for MLA)
    _process_module_weights(attn_module, vllm_config, device)

    # 2. Create KV cache + metadata
    with set_current_vllm_config(vllm_config):
        kv_cache, attn_metadata, _, indexer_kv_cache, indexer_metadata = _create_kv_cache_and_metadata(
            vllm_config=vllm_config,
            attn_type=attn_type,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_heads,
            is_context=is_context,
            use_fp8_kv_cache=use_fp8_kv_cache,
            device=device,
        )

    # 2b. Bind KV cache to the attention layer so forward() can access it.
    #     MLAAttention registers itself in static_forward_context during
    #     __init__, and reads self.kv_cache[virtual_engine] during forward.
    attn_layer_name = "model.layers.0.self_attn.attn"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx[attn_layer_name].kv_cache = [kv_cache]

    # For DSA, also bind the indexer's KV cache.
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    if indexer_kv_cache is not None and indexer_layer_name in forward_ctx:
        forward_ctx[indexer_layer_name].kv_cache = [indexer_kv_cache]

    # 3. Input tensors
    hidden_size = vllm_config.model_config.hf_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(seq_len, device=device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .reshape(-1)
            .contiguous()
        )
    else:
        num_tokens = batch_size
        positions = torch.full(
            (batch_size,),
            seq_len - 1,
            device=device,
            dtype=torch.long,
        )

    hidden_states = torch.full(
        (num_tokens, hidden_size),
        0.01,
        dtype=torch.bfloat16,
        device=device,
    )

    # 4. Dry run
    #    set_current_vllm_config — needed by quantised layers and RoPE.
    #    set_forward_context — provides attn_metadata + kv_cache to the
    #    MLAAttention.forward() path (it calls get_forward_context()).
    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    attn_metadata_dict = {attn_layer_name: attn_metadata}
    if indexer_metadata is not None:
        attn_metadata_dict[indexer_layer_name] = indexer_metadata
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))
    try:
        with torch.inference_mode():
            attn_module.forward(positions, hidden_states, None)
    except Exception as e:
        print(f"  Dry run failed: {e}")
        traceback.print_exc()
        _cleanup()
        return None

    # 5. Benchmark
    def kernel_func():
        attn_module.forward(positions, hidden_states, None)

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
        allow_graph_fail=True,
    ) as results:
        pass

    latency = results["latency_ms"]

    # 6. Log results — schema aligned with TRT-LLM
    if is_context:
        isl = seq_len
        step = 0
    else:
        isl = 1
        step = seq_len

    op_name = f"{attn_type}_{phase}_module"

    # Record architecture to distinguish different DSA models in the perf CSV.
    # perf_database uses this as a dict key when loading data.
    # Aligns with sdk/models.py which uses architectures[0] throughout.
    hf_cfg = vllm_config.model_config.hf_config
    architecture = getattr(hf_cfg, "architectures", [getattr(hf_cfg, "model_type", "unknown")])[0]

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "float16" if compute_dtype == "bfloat16" else compute_dtype,
                "kv_cache_dtype": "float16" if kv_cache_dtype == "bfloat16" else kv_cache_dtype,
                "gemm_type": "float16" if gemm_type == "bfloat16" else gemm_type,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": 1,
                "step": step,
                "latency": f"{latency:.4f}",
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
        f"gemm={gemm_type}, compute={compute_dtype}, kv={kv_cache_dtype}: {latency:.4f} ms"
    )

    _cleanup()
    return latency


def run_mla_module_worker(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    kv_cache_dtype: str,
    compute_dtype: str,
    gemm_type: str,
    perf_filename: str,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
):
    """Worker-compatible positional wrapper used by collector/collect.py."""
    return run_mla_module(
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        kv_cache_dtype=kv_cache_dtype,
        compute_dtype=compute_dtype,
        gemm_type=gemm_type,
        perf_filename=perf_filename,
        model_path=model_path,
        attn_type=attn_type,
        device=device,
    )


def _cleanup():
    torch.cuda.empty_cache()
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    model_names = list(SUPPORTED_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="MLA/DSA module-level collector for vLLM",
    )
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=model_names,
        help=f"Model to benchmark. If not specified, runs all: {model_names}",
    )
    parser.add_argument("--num-heads", type=int, default=None, help="Filter by number of heads")
    parser.add_argument("--batch-size", type=int, default=None, help="Single batch size (for --quick)")
    parser.add_argument("--seq-len", type=int, default=None, help="Single seq len (for --quick)")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="KV cache dtype (default: run both bfloat16 and fp8 when GPU supports it)",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        choices=["bfloat16", "fp8"],
        default=None,
        help="Compute dtype for attention (default: auto based on phase and GPU)",
    )
    parser.add_argument(
        "--gemm-type",
        type=str,
        choices=["bfloat16", "fp8_block", "nvfp4"],
        default=None,
        help="GEMM quantisation type for linear layers (default: run all supported by GPU)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true", help="Quick single-point test")
    args = parser.parse_args()

    # Select models to run
    if args.model:
        models_to_run = {args.model: SUPPORTED_MODELS[args.model]}
    else:
        models_to_run = SUPPORTED_MODELS

    for model_path, attn_type in models_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path}  |  Attention: {attn_type.upper()}")
        print(f"{'=' * 60}")

        if args.quick:
            b = args.batch_size or 4
            s = args.seq_len or 2048
            h = args.num_heads or 128
            kv_dtype = args.kv_cache_dtype or "bfloat16"
            compute = args.compute_dtype or "bfloat16"
            gemm = args.gemm_type or "bfloat16"
            fname = f"{attn_type}_{args.mode}_module_perf.txt"
            run_mla_module(
                seq_len=s,
                batch_size=b,
                num_heads=h,
                kv_cache_dtype=kv_dtype,
                compute_dtype=compute,
                gemm_type=gemm,
                perf_filename=fname,
                model_path=model_path,
                attn_type=attn_type,
                device=args.device,
            )
            continue

        if args.mode == "context":
            test_cases = get_context_test_cases(attn_type=attn_type)
        else:
            test_cases = get_generation_test_cases(attn_type=attn_type)

        if args.num_heads is not None:
            test_cases = [tc for tc in test_cases if tc[2] == args.num_heads]

        if args.kv_cache_dtype is not None:
            test_cases = [tc for tc in test_cases if tc[3] == args.kv_cache_dtype]

        if args.compute_dtype is not None:
            test_cases = [tc for tc in test_cases if tc[4] == args.compute_dtype]

        if args.gemm_type is not None:
            test_cases = [tc for tc in test_cases if tc[5] == args.gemm_type]

        print(f"Running {len(test_cases)} {args.mode} {attn_type.upper()} module test cases...")

        for i, (s, b, h, kv_dtype, compute, gemm, fname) in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}]", end="")
            try:
                run_mla_module(
                    seq_len=s,
                    batch_size=b,
                    num_heads=h,
                    kv_cache_dtype=kv_dtype,
                    compute_dtype=compute,
                    gemm_type=gemm,
                    perf_filename=fname,
                    model_path=model_path,
                    attn_type=attn_type,
                    device=args.device,
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM: b={b}, s={s}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}")
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  FAILED: b={b}, s={s}, heads={h}, gemm={gemm}, compute={compute}, kv={kv_dtype}: {e}")
                traceback.print_exc()
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    main()
