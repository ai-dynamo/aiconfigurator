# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications copyright (C) 2025 NVIDIA CORPORATION & AFFILIATES.

# Modified from https://github.com/vllm-project/vllm/blob/v0.11.0/tests/v1/attention/utils.py

"""Shared vLLM 0.24.0 collector test-harness utilities."""

import functools
import os
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from typing import Optional, Union

import torch
from vllm import _custom_ops as ops
from vllm.config import (
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.model import ModelDType
from vllm.distributed import init_distributed_environment
from vllm.distributed.parallel_state import ensure_model_parallel_initialized
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE, kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, get_kv_quant_mode


class MockAttentionLayer:
    """A mock attention layer for testing."""

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        # Add float versions for flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


@dataclass
class BatchSpec:
    """Specification for a batch configuration (workload shape only)."""

    seq_lens: list[int]
    query_lens: list[int]

    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)

    def __post_init__(self):
        assert len(self.seq_lens) == len(self.query_lens)

    def compute_num_tokens(self):
        return sum(self.query_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device,
    max_block_idx: int = 1000,
    arange_block_indices: bool = False,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata from a BatchSpec and ModelParams."""
    # Create query start locations
    query_start_loc = torch.zeros(batch_spec.batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(batch_spec.query_lens, dtype=torch.int32, device=device).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = batch_spec.compute_num_tokens()

    # Create sequence lengths
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())

    # Create computed tokens (context length for each sequence)
    context_lens = [batch_spec.seq_lens[i] - batch_spec.query_lens[i] for i in range(batch_spec.batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)

    # Create block table and slot mapping
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    if arange_block_indices:
        num_blocks = batch_spec.batch_size * max_blocks
        block_table_tensor = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
            batch_spec.batch_size, max_blocks
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device).view(num_tokens)
    else:
        block_table_tensor = torch.randint(
            0, max_block_idx, (batch_spec.batch_size, max_blocks), dtype=torch.int32, device=device
        )
        slot_mapping = torch.randint(0, max_block_idx, (num_tokens,), dtype=torch.int64, device=device)

    # Calculate max query length
    max_query_len = max(batch_spec.query_lens)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def get_attention_backend(backend_name: AttentionBackendEnum):
    """Return vLLM 0.24.0 metadata-builder and implementation classes."""
    backend_class = backend_name.get_class()
    return backend_class.get_builder_cls(), backend_class.get_impl_cls()


def create_standard_kv_cache_spec(
    vllm_config: VllmConfig, use_fp8_kv_cache: bool = False
) -> FullAttentionSpec | SlidingWindowSpec:
    """Create the KV-cache spec used by vLLM's production attention layer."""
    spec_kwargs = dict(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config),
        head_size=vllm_config.model_config.get_head_size(),
        head_size_v=vllm_config.model_config.get_head_size(),
        dtype=kv_cache_dtype_str_to_dtype(vllm_config.cache_config.cache_dtype, vllm_config.model_config),
        kv_quant_mode=get_kv_quant_mode(vllm_config.cache_config.cache_dtype),
    )
    sliding_window = vllm_config.model_config.get_sliding_window()
    if sliding_window is not None:
        return SlidingWindowSpec(sliding_window=sliding_window, **spec_kwargs)
    return FullAttentionSpec(**spec_kwargs)


def create_vllm_config(
    model_name: str = "meta-llama/Meta-Llama-3-8B",
    tensor_parallel_size: int = 1,
    distributed_executor_backend: str | None = None,
    max_model_len: int = 1024,
    dtype: Union[ModelDType, torch.dtype] = "auto",
    num_gpu_blocks: int = 1000,
    block_size: int = 16,
    max_num_seqs: int = 256,
    max_num_batched_tokens: int = 8192,
    enable_chunked_prefill: bool = True,
    add_mock_model_methods: bool = True,
    hf_config_override: dict | None = None,
    use_fp8_kv_cache: bool = False,
    trust_remote_code: bool = False,
    sliding_window: int | None = None,
    head_dim: int | None = None,
    num_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> VllmConfig:
    """Create a VllmConfig for testing with reasonable defaults."""

    model_config = ModelConfig(
        model=model_name,
        tokenizer=model_name,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
    )

    cache_config = CacheConfig(
        block_size=block_size,
        cache_dtype="fp8" if use_fp8_kv_cache else "auto",
    )
    # Set cache blocks for testing
    #   (these may be set during initialization normally)
    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = 0

    parallel_kwargs: dict[str, object] = {"tensor_parallel_size": tensor_parallel_size}
    if distributed_executor_backend is not None:
        parallel_kwargs["distributed_executor_backend"] = distributed_executor_backend
    parallel_config = ParallelConfig(**parallel_kwargs)

    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )

    device_config = DeviceConfig()
    load_config = LoadConfig()
    compilation_config = CompilationConfig()

    if add_mock_model_methods:
        # Add mock methods to satisfy backends that need them
        # This is a workaround because tests don't build full, real models,
        # but some backends expect to query the model for layer-specific
        # parameters
        import types

        model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
        _sw = sliding_window
        model_config.get_sliding_window_for_layer = types.MethodType(lambda self, i: _sw, model_config)
        model_config.get_logits_soft_cap_for_layer = types.MethodType(lambda self, i: 0.0, model_config)
        model_config.get_sm_scale_for_layer = types.MethodType(
            lambda self, i: 1.0 / model_config.get_head_size() ** 0.5, model_config
        )

    if sliding_window is not None:
        model_config.hf_text_config.sliding_window = sliding_window

    if hf_config_override:
        model_config.hf_config.update(hf_config_override)
    if head_dim is not None:
        model_config.hf_config.head_dim = head_dim
        model_config.model_arch_config.head_size = head_dim
    # ModelConfig.model_arch_config is built once in __init__, so keep its
    # cached head counts aligned with the fake HF config used by this harness.
    arch_cfg = model_config.model_arch_config
    if num_heads is not None:
        model_config.hf_config.num_attention_heads = num_heads
        arch_cfg.total_num_attention_heads = num_heads
    if num_kv_heads is not None:
        model_config.hf_config.num_key_value_heads = num_kv_heads
        arch_cfg.total_num_kv_heads = num_kv_heads

    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )


def create_dummy_kv_cache(
    block_size: int, num_kv_heads: int, head_size: int, dtype: torch.dtype, device: torch.device, num_blocks: int = 100
) -> torch.Tensor:
    """Create a dummy KV cache tensor for testing."""
    kv_cache = torch.randn(
        num_blocks,
        2,  # K and V
        block_size,
        num_kv_heads,
        head_size,
        dtype=dtype,
        device=device,
    )
    return kv_cache


def convert_dtype_to_torch(dtype):
    """Convert ModelDType to torch.dtype."""
    if isinstance(dtype, str):
        if dtype == "auto":
            return torch.bfloat16  # Default dtype for testing
        elif dtype in STR_DTYPE_TO_TORCH_DTYPE:
            return STR_DTYPE_TO_TORCH_DTYPE[dtype]
        else:
            raise TypeError(f"Unknown dtype: {dtype}")
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(f"Unknown dtype: {dtype}")


def create_and_prepopulate_kv_cache_mla(
    kv_c_contexts: list[torch.Tensor],
    k_pe_contexts: list[torch.Tensor],
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: CommonAttentionMetadata,
    randomize_blocks: bool = True,
    kv_cache_dtype: Optional[str] = None,
    scale: Union[float, torch.Tensor] = 1.0,
) -> torch.Tensor:
    """Create and prepopulate an MLA KV cache with context data.

    Args:
        kv_c_contexts: List of latent KV context tensors for each sequence
        k_pe_contexts: List of key positional embedding context tensors
                       for each sequence
        block_size: Size of each block
        head_size: Size of each head (latent dimension)
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        common_attn_metadata: Common attention metadata
        randomize_blocks: Whether to randomly permute blocks
                          or use sequential order
        kv_cache_dtype: Optional kv cache dtype string. When set to
                        "fp8_ds_mla" the cache is populated using the
                        fp8 DeepSeek MLA layout via concat_and_cache_mla.
        scale: Scaling factor forwarded to concat_and_cache_mla when the
               fp8 cache layout is requested.

    Returns:
        MLA KV cache tensor
    """
    batch_size = len(kv_c_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    use_fp8_ds_mla = kv_cache_dtype == "fp8_ds_mla"

    if use_fp8_ds_mla:
        if not kv_c_contexts:
            raise ValueError("kv_c_contexts cannot be empty when using fp8_ds_mla cache dtype")
        kv_lora_rank = kv_c_contexts[0].shape[-1]
        rope_dim = k_pe_contexts[0].shape[-1]
        entry_size = kv_lora_rank + 4 * 4 + 2 * rope_dim
        kv_cache = torch.zeros(num_blocks, block_size, entry_size, dtype=torch.uint8, device=device)
        scale_tensor = (
            scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, dtype=torch.float32, device=device)
        )
        scale_tensor = scale_tensor.to(device=device, dtype=torch.float32)
    else:
        # Create MLA KV cache: (num_blocks, block_size, head_size)
        kv_cache = torch.empty(num_blocks, block_size, head_size, dtype=dtype, device=device)
        kv_cache_flat = kv_cache.view(-1, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        kv_c_context, k_pe_context = kv_c_contexts[i], k_pe_contexts[i]
        context_len = kv_c_context.shape[0]
        if context_len == 0:
            start_block_idx += cdiv(int(seq_lens[i]), block_size)
            continue

        start = start_block_idx * block_size

        if use_fp8_ds_mla:
            slots = torch.arange(context_len, device=device, dtype=torch.long) + start
            ops.concat_and_cache_mla(
                kv_c_context,
                k_pe_context.squeeze(1),
                kv_cache,
                slots,
                kv_cache_dtype="fp8_ds_mla",
                scale=scale_tensor,
            )
        else:
            kv_context = torch.cat([kv_c_context, k_pe_context.squeeze(1)], dim=-1)
            end = start + kv_context.shape[0]
            kv_cache_flat[start:end, ...] = kv_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null). Avoid an
    # identity advanced-index copy when the caller requests sequential blocks;
    # that copy can temporarily duplicate tens of GiB for long MLA sequences.
    if randomize_blocks:
        perm = torch.randperm(blocks_end - 1) + 1  # Random permutation starting from block 1
        inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
        inv_perm[1:] = torch.argsort(perm) + 1  # Add 1 to account for starting from block 1
        kv_cache[1:blocks_end, ...] = kv_cache[perm, ...]
    else:
        inv_perm = torch.arange(blocks_end, dtype=torch.long, device=device)

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[i, block_indices] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


def create_and_prepopulate_kv_cache(
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_blocks: int,
    common_attn_metadata: CommonAttentionMetadata,
    randomize_blocks: bool = True,
) -> torch.Tensor:
    """Create and prepopulate a KV cache with context data.

    Args:
        k_contexts: List of key context tensors for each sequence
        v_contexts: List of value context tensors for each sequence
        seq_lens: List of sequence lengths
        block_size: Size of each block
        num_kv_heads: Number of KV heads
        head_size: Size of each head
        dtype: Data type for the cache
        device: Device to create the cache on
        num_blocks: Total number of blocks in the cache
        block_table: Block table tensor to populate
        randomize_blocks: Whether to randomly permute blocks
                          or use sequential order

    Returns:
        Tuple of (kv_cache, updated_block_table)
    """
    batch_size = len(k_contexts)
    seq_lens = common_attn_metadata.seq_lens_cpu
    query_lens = common_attn_metadata.query_start_loc_cpu[1:] - common_attn_metadata.query_start_loc_cpu[:-1]
    context_lens = common_attn_metadata.num_computed_tokens_cpu
    block_table = common_attn_metadata.block_table_tensor
    slot_mapping = common_attn_metadata.slot_mapping

    # Create KV cache
    kv_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device)
    kv_cache_flat = kv_cache.view(2, -1, num_kv_heads, head_size)

    # Populate the cache with the context tokens
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        k_context, v_context = k_contexts[i], v_contexts[i]
        start = start_block_idx * block_size
        end = start + k_context.shape[0]
        kv_cache_flat[0, start:end, ...] = k_context
        kv_cache_flat[1, start:end, ...] = v_context

        # Stay block aligned and allocate enough blocks for the new tokens
        start_block_idx += cdiv(int(seq_lens[i]), block_size)

    blocks_end = start_block_idx

    # Permute the context blocks (excluding block 0 which is null)
    if randomize_blocks:
        # Random permutation starting from block 1
        perm = torch.randperm(blocks_end - 1) + 1
    else:
        # Sequential order starting from block 1
        perm = torch.arange(1, blocks_end)

    inv_perm = torch.zeros(blocks_end, dtype=torch.long, device=device)
    # Add 1 to account for starting from block 1
    inv_perm[1:] = torch.argsort(perm) + 1

    # Workaround for XPU FP8 indexing not implemented:
    # Intel Extension for PyTorch (IPEX) currently lacks support for advanced
    # indexing (slicing via LongTensor) on Float8 tensors ("index_xpu" not implemented).
    # To bypass this, we temporarily cast the KV cache to bfloat16, perform the
    # permutation, and then cast it back to the original FP8 format.
    if "xpu" in str(device) and kv_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        temp_cache = kv_cache.to(torch.bfloat16)
        temp_cache_sliced = temp_cache[:, perm, ...]
        kv_cache[:, 1:blocks_end, ...] = temp_cache_sliced.to(kv_cache.dtype)
    else:
        kv_cache[:, 1:blocks_end, ...] = kv_cache[:, perm, ...]

    # Construct the right block table
    # Start from block_id=1 since block_id=0 is considered the null block
    start_block_idx = 1
    for i in range(batch_size):
        num_blocks_for_seq = cdiv(int(seq_lens[i]), block_size)
        start = start_block_idx
        end = start + num_blocks_for_seq
        block_table[i, :num_blocks_for_seq] = inv_perm[start:end]
        start_block_idx += num_blocks_for_seq

        # Create a realistic slot mapping that corresponds to the block table
    for i in range(batch_size):
        token_offsets = torch.arange(int(query_lens[i])) + int(context_lens[i])
        block_indices = token_offsets // block_size
        token_inter_block_offsets = token_offsets % block_size
        start = common_attn_metadata.query_start_loc_cpu[i]
        end = common_attn_metadata.query_start_loc_cpu[i + 1]
        slot_mapping[start:end] = block_table[i, block_indices] * block_size + token_inter_block_offsets.to(device)

    return kv_cache


@functools.cache  # only run once per process
def setup_distributed(device):
    # Each process needs to use a different port.
    device_idx = torch.device(device).index
    port = 8889 + device_idx
    print(device, device_idx, port)

    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment()
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(1, 1)


def with_exit_stack(func):
    """
    Decorator that creates an ExitStack, passes it as the first argument
    to the function, and closes it when the function returns.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # The wrapper handles the safety indentation for you
        with ExitStack() as stack:
            # We inject 'stack' as the first argument to your function
            return func(stack, *args, **kwargs)

    return wrapper
