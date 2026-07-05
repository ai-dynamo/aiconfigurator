# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__compat__ = "vllm==0.24.0"

"""
MLA Module Collector for vLLM — unified MLA and DSA benchmarking.

Profiles the complete attention module forward pass (projections + attention +
output), not just the bare attention kernel. Uses vLLM's own modeling code to
construct a single `DeepseekV2MLAAttention` module with dummy weights, then
benchmarks its forward.

MLA vs DSA is determined by the presence of `index_topk` in the HF config.
Op names and data schema are aligned with TRT-LLM's collect_mla_module.py
so that queries can be reused across frameworks.

Supported models, attention types, and micro-sweeps are defined in collector v2
YAML and loaded through collector.case_generator. The collector reads a real HF
config, overrides the layer-local shape fields in-memory, and then instantiates
just the attention module.

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

# ═══════════════════════════════════════════════════════════════════════
# Config registry patch — vLLM 0.24.0 registers the GlmMoeDsaForCausalLM
# model class but omits the config-type mapping for "glm_moe_dsa", so
# AutoConfig.from_pretrained() fails.  The config layout is identical to
# DeepSeek-V3 (GlmMoeDsaForCausalLM inherits DeepseekV2ForCausalLM), so
# reusing DeepseekV3Config is safe.
# ═══════════════════════════════════════════════════════════════════════
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.v1.worker.workspace import init_workspace_manager
from vllm.version import __version__ as vllm_version

from collector.case_generator import (
    get_mla_module_model_specs,
    get_mla_module_precision_specs,
    get_mla_module_sweep_spec,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile
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
# Test Cases — aligned with TRT-LLM's collect_mla_module.py
# ═══════════════════════════════════════════════════════════════════════


def _get_precision_combos(phase: str, attn_type: str):
    """Return YAML-backed (compute_dtype, kv_cache_dtype, gemm_type) triples."""

    return [
        (spec.compute_dtype, spec.kv_cache_dtype, spec.gemm_type)
        for spec in get_mla_module_precision_specs(
            "vllm",
            phase=phase,
            sm_version=get_sm_version(),
            attention_type=attn_type,
        )
    ]


def get_context_test_cases(attn_type: str):
    """Context-phase test cases.

    Returns list of [seq_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type, prefix_len].
    """
    cases = []
    sweep = get_mla_module_sweep_spec("vllm")
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("context", attn_type):
        for num_heads in sweep.inner_sweep_head_counts:
            for b in sweep.context_batch_sizes:
                for s in sweep.context_sequence_lengths:
                    if b * s > sweep.context_max_tokens:
                        continue
                    if attn_type == "dsa":
                        for prefix_len in sweep.context_prefix_lengths:
                            cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type, prefix_len])
                    else:
                        cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def get_generation_test_cases(attn_type: str):
    """Generation-phase test cases.

    Returns list of [kv_cache_len, batch_size, num_heads, kv_cache_dtype,
                     compute_dtype, gemm_type].
    """
    cases = []
    sweep = get_mla_module_sweep_spec("vllm")
    for compute_dtype, kv_dtype, gemm_type in _get_precision_combos("generation", attn_type):
        for num_heads in sweep.inner_sweep_head_counts:
            for b in sweep.generation_batch_sizes:
                for s in sweep.generation_sequence_lengths:
                    if b * s > sweep.generation_max_tokens:
                        continue
                    cases.append([s, b, num_heads, kv_dtype, compute_dtype, gemm_type])
    return cases


def _build_module_test_cases(attn_type: str, mode: str):
    """Build module-level test cases for a specific attention type and phase.

    Output format: [seq_len, batch_size, num_heads, kv_cache_dtype,
                    compute_dtype, gemm_type, model_path, attn_type]
    """
    base_cases = get_context_test_cases(attn_type) if mode == "context" else get_generation_test_cases(attn_type)
    cases = []
    for model_spec in get_mla_module_model_specs(attention_type=attn_type, backend="vllm"):
        for base_case in base_cases:
            s, b, h, kv_dtype, compute_dtype, gemm_type, *rest = base_case
            case = [s, b, h, kv_dtype, compute_dtype, gemm_type, model_spec.model_path, attn_type]
            if rest:
                case.append(rest[0])
            cases.append(case)
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


def _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata):
    """Ground-truth backend for the perf row.

    MLA prefill runs mla_layer.prefill_backend; everything else — DSA, decode,
    and "context" batches that vLLM classified entirely as decodes
    (num_prefills == 0, e.g. s=1) — runs mla_layer.attn_backend.

    ``num_prefills`` must only be read on the MLA-context branch: sparse DSA
    metadata (FlashMLASparseMetadata / FlashInferMLASparseMetadata @0.24.0)
    does not carry that attribute at the top level, so an eager read breaks
    every DSA row.
    """
    if attn_type == "dsa" or not is_context or attn_metadata.num_prefills == 0:
        return mla_layer.attn_backend.get_name()
    return mla_layer.prefill_backend.get_name()


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
    max_seq_len: int,
    max_batch_size: int,
    use_prefill_fp8: bool = False,
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
        use_prefill_fp8: When True, opt in to FP8 prefill query compute via
            ``attention_config.use_prefill_query_quantization`` plus an
            explicit ``attention_config.mla_prefill_backend`` pin. vLLM honors
            the quantization flag only with an FP8 KV cache on the SM100
            family with a FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA prefill
            backend (determine_prefill_query_data_type +
            backend_supports_prefill_query_quantization,
            mla_attention.py:1371-1396,1453-1493 @0.24.0), and the SM100 auto
            selector always picks FLASH_ATTN instead
            (prefill/selector.py:60-66); the metadata probe in
            _create_kv_cache_and_metadata fails closed if the decision
            does not match the case label.
        gemm_type: Precision for linear-layer GEMMs — "bfloat16",
            "fp8_block", or "nvfp4".
    """
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

    local_model_path = _resolve_model_path(model_path)

    block_size = 64
    # seq_len includes the current token; generation caches only seq_len - 1.
    # Keep exact-limit models such as Kimi-K2-Instruct at their declared limit.
    max_model_len = max(max_seq_len, 4096)
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
        # Forward the test sweep's per-case head counts so that
        # ModelConfig.model_arch_config (built once at __init__ from hf_config) stays in
        # sync. Without this, the V1 FA3 builder reads the model's natural head count
        # via get_num_attention_heads(), the AOT scheduler precomputes
        # scheduler_metadata for that shape, and impl.forward then runs with the test's
        # actual head count — _vllm_fa3_C.fwd's shape check rejects the mismatch.
        # MLA collapses KV to 1 head via get_num_kv_heads (use_mla=True), so the
        # num_kv_heads override is a no-op in that path; we pass it for parity with
        # the attention collector.
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )

    # Override quant_config to control linear-layer GEMM precision.
    # DeepSeek-V3.2 ships with FP8 quantisation by default, so we
    # must always set quant_config explicitly: None for bf16,
    # Fp8Config (blockwise) for fp8_block, ModelOptNvFp4Config for nvfp4.
    vllm_config.quant_config = _create_gemm_quant_config(gemm_type)

    # Opt in to FP8 prefill query compute before the module (and later the
    # metadata builder) reads the attention config.
    if use_prefill_fp8:
        from vllm.platforms import current_platform

        vllm_config.attention_config.use_prefill_query_quantization = True
        if current_platform.is_device_capability_family(100):
            from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

            # The quantization flag alone never engages on SM100: the auto
            # selector ranks FLASH_ATTN first (prefill/selector.py:60-66
            # @0.24.0) and FLASH_ATTN is always eligible there, but FP8 query
            # compute requires a FLASHINFER/TRTLLM_RAGGED/TOKENSPEED_MLA
            # prefill backend (backend_supports_prefill_query_quantization,
            # mla_attention.py:1371-1396 @0.24.0). The FP8-prefill serving
            # configuration therefore also pins attention_config
            # .mla_prefill_backend; TRTLLM_RAGGED is the framework's own
            # highest-priority SM100 backend among the supporting set
            # (prefill/selector.py:60-66). get_mla_prefill_backend validates
            # the explicit selection and raises when it is invalid, and the
            # metadata probe below still fails closed on the resolved
            # q_data_type.
            vllm_config.attention_config.mla_prefill_backend = MLAPrefillBackendEnum.TRTLLM_RAGGED
        # Outside the SM100 family the pin has no source proof: vLLM gates
        # FP8 prefill query quantization to that family outright
        # (backend_supports_prefill_query_quantization returns False when
        # not is_device_capability_family(100), mla_attention.py:1385-1386
        # @0.24.0), and TRTLLM_RAGGED fails its own capability validation
        # (e.g. "compute capability 12.0 not supported" on SM120). Leave the
        # auto selector in charge so the q_data_type probe below raises the
        # accurate classified error for the unsupported fp8 label.

    # backend_supports_prefill_query_quantization() is a zero-argument
    # functools.cache that reads the *current* vllm config on first call
    # (mla_attention.py:1371 @0.24.0). Serving processes hold one config, but
    # collector workers build many per process, so a value cached from a
    # previous case (e.g. bf16-prefill, FLASH_ATTN) would silently redirect
    # this case's q_data_type decision. Clear it so vLLM re-decides from this
    # case's config, exactly as a fresh serving process would.
    from vllm.model_executor.layers.attention.mla_attention import (
        backend_supports_prefill_query_quantization,
    )

    backend_supports_prefill_query_quantization.cache_clear()

    # Override just the layer-local dimensions we sweep in the collector.
    hf_config = vllm_config.model_config.hf_text_config
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
    # Deterministic init — vLLM 0.24.0 DSA modules leave CUDA graph RNG
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
            if isinstance(module, MLAAttention):
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
    is_context: bool,
    prefix_len: int = 0,
    compute_dtype: str = "bfloat16",
    device: str = "cuda:0",
):
    """Create KV cache and attention metadata for benchmarking."""
    hf_config = vllm_config.model_config.hf_text_config
    kv_lora_rank = hf_config.kv_lora_rank
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    block_size = vllm_config.cache_config.block_size
    is_dsa = attn_type == "dsa"

    prefix_len = int(prefix_len) if is_context else 0

    if is_context:
        batch_spec = BatchSpec(
            seq_lens=[prefix_len + seq_len] * batch_size,
            query_lens=[seq_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[seq_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    num_kv_cache_blocks = max(
        1 + math.ceil((prefix_len + seq_len + 1) / block_size) * batch_size,
        8192,
    )

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, block_size, torch.device(device), arange_block_indices=True
    )

    # TRT-LLM Gen MLA (and cutlass_mla) require the page table's block
    # dimension to be a multiple of ``128 / block_size``; for block_size=64
    # that means at least two blocks even when a request only fills one.
    # Short sequences (seq_len <= block_size) otherwise produce block_num=1
    # and the kernel raises ``Expected block_num % (128 / block_size) == 0``.
    # Pad with zero-indexed blocks (they are not read for absent tokens).
    required_divisor = max(1, 128 // block_size)
    current_block_num = common_attn_metadata.block_table_tensor.shape[1]
    if current_block_num % required_divisor != 0:
        padded_block_num = ((current_block_num + required_divisor - 1) // required_divisor) * required_divisor
        padding = torch.zeros(
            (common_attn_metadata.block_table_tensor.shape[0], padded_block_num - current_block_num),
            dtype=common_attn_metadata.block_table_tensor.dtype,
            device=common_attn_metadata.block_table_tensor.device,
        )
        common_attn_metadata.block_table_tensor = torch.cat([common_attn_metadata.block_table_tensor, padding], dim=1)

    # Use the cache format chosen by the concrete production layer. In 0.24.0
    # this matters on both SM10x (FlashMLA vs FlashInfer sparse) and SM120,
    # where sparse MLA canonicalizes auto/fp8 to the packed fp8_ds_mla layout.
    attn_layer_name = "model.layers.0.self_attn.attn"
    attn_layer = vllm_config.compilation_config.static_forward_context[attn_layer_name]
    backend_cls = attn_layer.get_attn_backend()
    kv_cache_spec = attn_layer.get_kv_cache_spec(vllm_config)
    cache_dtype = kv_cache_spec.dtype
    kv_cache_dtype_str = kv_cache_spec.cache_dtype_str

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
        head_size=kv_cache_spec.head_size,
        dtype=cache_dtype,
        device=torch.device(device),
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=False,
        kv_cache_dtype=kv_cache_dtype_str,
        scale=attn_layer._k_scale,
    )

    builder_cls = backend_cls.get_builder_cls()

    layer_names = [attn_layer_name]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, torch.device(device))

    # Fail closed on the framework's own prefill query dtype decision:
    # vLLM silently falls back to the model dtype when FP8 prefill query
    # quantization is requested but unsupported (fp8 KV cache + SM100 family
    # + FLASHINFER/TRTLLM_RAGGED prefill backend required —
    # determine_prefill_query_data_type, mla_attention.py:1453-1493 @0.24.0).
    # A silently-bf16 run must not be recorded as an mla_dtype=fp8 row.
    if is_context:
        from vllm.platforms import current_platform

        model_dtype = vllm_config.model_config.dtype
        expected_q_dtype = current_platform.fp8_dtype() if compute_dtype == "fp8" else model_dtype
        actual_q_dtype = getattr(builder, "q_data_type", model_dtype)
        if actual_q_dtype != expected_q_dtype:
            raise RuntimeError(
                f"vLLM selected prefill query dtype {actual_q_dtype} but the case is labeled "
                f"compute_dtype={compute_dtype!r}; refusing to record a mislabeled row "
                "(see determine_prefill_query_data_type @0.24.0 for the support conditions)"
            )

    attn_metadata = builder.build(
        common_prefix_len=prefix_len,
        common_attn_metadata=common_attn_metadata,
    )

    # For DSA, the Indexer has its own KV cache and metadata builder.
    indexer_kv_cache = None
    indexer_metadata = None
    if is_dsa:
        indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
        indexer_layer = vllm_config.compilation_config.static_forward_context[indexer_layer_name]
        indexer_spec = indexer_layer.get_kv_cache_spec(vllm_config)
        indexer_kv_cache = torch.zeros(
            num_kv_cache_blocks,
            block_size,
            indexer_spec.head_size,
            dtype=indexer_spec.dtype,
            device=device,
        )
        indexer_builder_cls = indexer_layer.get_attn_backend().get_builder_cls()
        indexer_builder = indexer_builder_cls(indexer_spec, [indexer_layer_name], vllm_config, torch.device(device))
        indexer_metadata = indexer_builder.build(
            common_prefix_len=prefix_len,
            common_attn_metadata=common_attn_metadata,
        )
        _populate_indexer_kv_cache(
            indexer_kv_cache=indexer_kv_cache,
            common_attn_metadata=common_attn_metadata,
            context_lens=[tensor.shape[0] for tensor in kv_c_contexts],
        )

    return kv_cache, attn_metadata, common_attn_metadata, indexer_kv_cache, indexer_metadata


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
    prefix_len: int = 0,
    *,
    model_path: str,
    attn_type: str,
    device: str = "cuda:0",
    warming_up: int = 10,
    test_ite: int = 6,
):
    """Run a single MLA / DSA module-level benchmark point."""
    if attn_type not in {"mla", "dsa"}:
        raise ValueError(f"unsupported vLLM attention type: {attn_type!r}")
    if kv_cache_dtype not in {"bfloat16", "fp8"}:
        raise ValueError(f"unsupported vLLM MLA KV-cache dtype: {kv_cache_dtype!r}")
    if compute_dtype not in {"bfloat16", "fp8"}:
        raise ValueError(f"unsupported vLLM MLA query compute dtype: {compute_dtype!r}")
    if compute_dtype == "fp8" and kv_cache_dtype != "fp8":
        raise ValueError(
            "vLLM FP8 MLA prefill query compute requires an FP8 KV cache "
            "(determine_prefill_query_data_type, mla_attention.py:1462-1466 "
            f"@0.24.0); got kv_cache_dtype={kv_cache_dtype!r}"
        )

    setup_distributed(device)
    torch.cuda.set_device(device)

    # DSA's sparse_attn_indexer requires a WorkspaceManager.
    init_workspace_manager(torch.device(device))

    use_fp8_kv_cache = kv_cache_dtype == "fp8"
    use_prefill_fp8 = compute_dtype == "fp8"
    is_context = "context" in perf_filename
    prefix_len = int(prefix_len) if is_context else 0
    phase = "context" if is_context else "generation"
    variant = attn_type.upper()
    print(
        f"\n[{variant} module] {phase} b={batch_size}, s={seq_len}, "
        f"prefix={prefix_len}, heads={num_heads}, gemm={gemm_type}, "
        f"compute={compute_dtype}, kv={kv_cache_dtype}, model={model_path}"
    )

    # 1. Create attention module
    attn_module, vllm_config = _create_attention_module(
        model_path=model_path,
        attn_type=attn_type,
        num_heads=num_heads,
        use_fp8_kv_cache=use_fp8_kv_cache,
        use_prefill_fp8=use_prefill_fp8,
        max_seq_len=prefix_len + seq_len,
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
            is_context=is_context,
            prefix_len=prefix_len,
            compute_dtype=compute_dtype,
            device=device,
        )

    # 2b. Bind KV cache to the 0.24.0 attention layer.
    attn_layer_name = "model.layers.0.self_attn.attn"
    forward_ctx = vllm_config.compilation_config.static_forward_context
    forward_ctx[attn_layer_name].kv_cache = kv_cache

    # For DSA, also bind the indexer's KV cache.
    indexer_layer_name = "model.layers.0.self_attn.indexer.k_cache"
    if indexer_kv_cache is not None and indexer_layer_name in forward_ctx:
        forward_ctx[indexer_layer_name].kv_cache = indexer_kv_cache

    # 3. Input tensors
    hidden_size = vllm_config.model_config.hf_text_config.hidden_size
    if is_context:
        num_tokens = seq_len * batch_size
        positions = (
            torch.arange(prefix_len, prefix_len + seq_len, device=device, dtype=torch.long)
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
    # FIXME(kernel-limit): on SM100, FlashInfer's trtllm-gen sparse-MLA decode
    # ships kernels only for tileSizeQ >= 8. DSA cases whose per-rank head
    # count is 1/2/4 (tileSizeQ = heads for q_len 1) raise "Missing TRTLLM-GEN
    # kernel (decode)" (flashinfer csrc/trtllm_fmha_kernel_launcher.cu:272,
    # flashinfer 0.6.12) whenever vLLM selects FLASHINFER_MLA_SPARSE — always
    # for FP8 KV, and for BF16 KV at heads <= 16 (platforms/cuda.py:98-116
    # @0.24.0). Heads 8..128 pass; boundary measured on B200. Serving fails
    # identically, so the affected cases stay observed runtime failures.
    # Re-verify on the next vLLM/FlashInfer bump.
    # FIXME(kernel-limit): on SM120, vLLM's dense-MLA decode backend is
    # TRITON_MLA (platforms/cuda.py:130-134 @0.24.0) and its grouped decode
    # kernel requests 102400B of shared memory with an FP8 KV cache — above
    # SM120's 101376B/SM limit, so Triton raises OutOfResources for every
    # fp8-KV MLA decode case (bf16-KV passes). vLLM's overflow guard only
    # drops num_stages when BLOCK_DMODEL >= 1024, which the MLA Lk=576 path
    # (BLOCK_DMODEL=512) never reaches (triton_decode_attention.py:490-532
    # @0.24.0). Measured on RTX PRO 6000 Blackwell; serving fails
    # identically. Re-verify on the next vLLM bump.
    # FIXME(kernel-limit): on SM120, every DSA case fails: the CUDA sparse
    # attention indexer hard-requires DeepGEMM (sparse_attn_indexer.py:468-472
    # @0.24.0), whose fp8 MQA-logits kernels ship for SM90/SM100 only
    # (vllm/third_party/deep_gemm .../impls/sm{90,100}_fp8_mqa_logits.cuh) and
    # assert "Unsupported architecture" (deepgemm csrc/apis/attention.hpp:184);
    # cases with fp8_block linears die even earlier in DeepGEMM's scale-factor
    # layout transform ("Unknown SF transformation", layout.hpp:59). Serving
    # fails identically. Re-verify on the next vLLM/DeepGEMM bump.
    exit_stack.enter_context(set_current_vllm_config(vllm_config))
    attn_metadata_dict = {attn_layer_name: attn_metadata}
    if indexer_metadata is not None:
        attn_metadata_dict[indexer_layer_name] = indexer_metadata
    exit_stack.enter_context(set_forward_context(attn_metadata_dict, vllm_config))
    try:
        with torch.inference_mode():
            attn_module.forward(positions, hidden_states, None)
    except torch.cuda.OutOfMemoryError as e:
        print(f"  Dry run OOM: {e}")
        _cleanup()
        # Let collect.py record the capacity failure. Returning normally would
        # mark a task done even though it emitted no performance row.
        raise
    except Exception as e:
        print(f"  Dry run failed: {e}")
        traceback.print_exc()
        _cleanup()
        # Propagate to collect.py's worker so the failure is recorded in the
        # error queue. Swallowing here lets a whole op complete with zero
        # rows while the pipeline reports success.
        raise

    # 5. Benchmark
    def kernel_func():
        attn_module.forward(positions, hidden_states, None)

    # DSA context captures a ~18 GiB flashmla-sparse scratch into the CUDA
    # graph's private pool on big shapes. PyTorch doesn't reclaim that pool
    # aggressively enough between tasks, so after a handful of big captures
    # the per-worker private-pool retention saturates at ~146 GiB and every
    # subsequent task — regardless of size — OOMs at
    # ``WorkspaceManager._ensure_workspace_size``. Running eagerly avoids the
    # private-pool retention entirely; the bias is <0.5% for DSA context
    # because per-forward latency is tens of ms while graph launch overhead
    # is tens of μs. Other ops (MLA context, both generation phases) keep
    # graph capture because they don't hit this retention pattern and the
    # overhead matters more for their faster kernels.
    use_cuda_graph = not (attn_type == "dsa" and is_context)

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
        use_cuda_graph=use_cuda_graph,
    ) as results:
        pass

    latency = results["latency_ms"]

    # 6. Log results — schema aligned with TRT-LLM
    if is_context:
        isl = seq_len
        step = prefix_len
    else:
        isl = 1
        step = seq_len

    op_name = f"{attn_type}_{phase}_module"

    # Record architecture to distinguish different DSA models in the perf CSV.
    # perf_database uses this as a dict key when loading data.
    # Aligns with sdk/models.py which uses architectures[0] throughout.
    hf_cfg = vllm_config.model_config.hf_config
    architecture = getattr(hf_cfg, "architectures", [getattr(hf_cfg, "model_type", "unknown")])[0]
    mla_layer = attn_module.mla_attn.mla_attn
    backend_name = _mla_backend_name(mla_layer, attn_type, is_context, attn_metadata)
    actual_kv_cache_dtype = "fp8" if mla_layer.kv_cache_dtype.startswith("fp8") else "bfloat16"

    log_perf(
        item_list=[
            {
                "model": model_path,
                "architecture": architecture,
                "mla_dtype": "bfloat16" if compute_dtype == "bfloat16" else compute_dtype,
                "kv_cache_dtype": actual_kv_cache_dtype,
                "gemm_type": "bfloat16" if gemm_type == "bfloat16" else gemm_type,
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
        kernel_source=backend_name,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    print(
        f"  [{phase}] b={batch_size}, s={seq_len}, heads={num_heads}, "
        f"prefix={prefix_len}, gemm={gemm_type}, compute={compute_dtype}, "
        f"kv={kv_cache_dtype}, backend={backend_name}: {latency:.4f} ms"
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
    model_path: str,
    attn_type: str,
    prefix_len: int = 0,
    *,
    perf_filename: str,
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
        prefix_len=prefix_len,
        perf_filename=perf_filename,
        model_path=model_path,
        attn_type=attn_type,
        device=device,
    )


def _cleanup():
    # Release vLLM's WorkspaceManager singleton scratch buffers before
    # returning to the worker loop. ``_ensure_workspace_size`` only grows —
    # once a task demands N bytes, the manager pins N bytes for the worker's
    # lifetime, so a single large task turns into a permanent allocator
    # reservation that starves subsequent small tasks into OOM.
    #
    # The module-level ``_manager`` is private; vLLM offers no public
    # teardown API. Nulling it drops the Python reference to the old
    # manager (and its workspace tensors), and the next task's
    # ``init_workspace_manager()`` call creates a fresh instance on demand.
    import vllm.v1.worker.workspace as _ws_mod

    _ws_mod._manager = None
    # gc.collect() must run BEFORE empty_cache() — ``empty_cache`` only
    # releases blocks with zero live allocations, so any Python-reachable
    # tensors need to be dropped first or the pass is a no-op.
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def _supported_model_map() -> dict[str, str]:
    return {
        spec.model_path: spec.attention_type
        for spec in get_mla_module_model_specs(backend="vllm", apply_model_filter=False)
    }


def main():
    supported_models = _supported_model_map()
    model_names = list(supported_models.keys())

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
        models_to_run = {args.model: supported_models[args.model]}
    else:
        models_to_run = supported_models

    for model_path, attn_type in models_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"Model: {model_path}  |  Attention: {attn_type.upper()}")
        print(f"{'=' * 60}")

        if args.mode == "context":
            perf_filename = PerfFile.MLA_CONTEXT_MODULE if attn_type == "mla" else PerfFile.DSA_CONTEXT_MODULE
        else:
            perf_filename = PerfFile.MLA_GENERATION_MODULE if attn_type == "mla" else PerfFile.DSA_GENERATION_MODULE

        if args.quick:
            b = args.batch_size or 4
            s = args.seq_len or 2048
            h = args.num_heads or 128
            kv_dtype = args.kv_cache_dtype or "bfloat16"
            compute = args.compute_dtype or "bfloat16"
            gemm = args.gemm_type or "bfloat16"
            run_mla_module(
                seq_len=s,
                batch_size=b,
                num_heads=h,
                kv_cache_dtype=kv_dtype,
                compute_dtype=compute,
                gemm_type=gemm,
                perf_filename=perf_filename,
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
        for i, (s, b, h, kv_dtype, compute, gemm) in enumerate(test_cases):
            print(f"[{i + 1}/{len(test_cases)}]", end="")
            try:
                run_mla_module(
                    seq_len=s,
                    batch_size=b,
                    num_heads=h,
                    kv_cache_dtype=kv_dtype,
                    compute_dtype=compute,
                    gemm_type=gemm,
                    perf_filename=perf_filename,
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
