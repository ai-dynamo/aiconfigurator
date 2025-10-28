# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for v1 attention backends without GPUModelRunner dependency."""
from functools import partial
from typing import Optional, Union

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from utils import (BatchSpec, _Backend,
                                      create_common_attn_metadata,
                                      create_standard_kv_cache_spec,
                                      create_vllm_config,
                                      get_attention_backend)
from vllm.config import ModelConfig
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, cdiv, is_torch_equal_or_newer
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              set_kv_cache_layout)
from vllm.v1.kv_cache_interface import FullAttentionSpec


# Define common batch configurations
BATCH_SPECS = {
    "small_decode":
    BatchSpec(seq_lens=[32, 40], query_lens=[1, 1]),
    "small_prefill":
    BatchSpec(seq_lens=[32, 40], query_lens=[8, 8]),
    "mixed_small":
    BatchSpec(seq_lens=[32, 40, 48, 56], query_lens=[1, 1, 5, 5]),
    "medium_decode":
    BatchSpec(seq_lens=[128, 256, 512, 1024, 128, 256, 512, 1024],
              query_lens=[1, 1, 1, 1, 1, 1, 1, 1]),
    "medium_prefill":
    BatchSpec(seq_lens=[256, 512, 1024, 2048], query_lens=[16, 16, 16, 16]),
    "mixed_medium":
    BatchSpec(seq_lens=[512, 1024, 2048, 512, 1024, 2048],
              query_lens=[1, 1, 1, 7, 7, 7]),
    "large_decode":
    BatchSpec(seq_lens=[2048] * 32, query_lens=[1] * 32),
    "large_prefill":
    BatchSpec(seq_lens=[4096] * 8, query_lens=[32] * 8),
    "single_decode":
    BatchSpec(seq_lens=[1024], query_lens=[1]),
    "single_prefill":
    BatchSpec(seq_lens=[1024], query_lens=[64]),
}




def run_attention_backend(
    backend: _Backend,
    kv_cache_spec: FullAttentionSpec,
    layer_names: list[str],
    vllm_config,
    device: torch.device,
    common_attn_metadata: CommonAttentionMetadata,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    sliding_window: Optional[int] = None,
) -> torch.Tensor:
    """Run attention computation using the specified backend's AttentionImpl."""

    # Handle special case for FLEX_ATTENTION_SLOW
    actual_backend = backend

    use_direct_block_mask = is_torch_equal_or_newer("2.9.0.dev0")
    if backend == "FLEX_ATTENTION_SLOW":
        actual_backend = _Backend.FLEX_ATTENTION
        use_direct_block_mask = False

    builder_cls, impl_cls = get_attention_backend(actual_backend)

    # Mock flashinfer's get_per_layer_parameters if needed
    if actual_backend == _Backend.FLASHINFER:
        import unittest.mock

        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            # Return mock parameters for a single layer
            head_size = vllm_config.model_config.get_head_size()
            return {
                layer_name:
                PerLayerParameters(
                    window_left=-1,  # No sliding window
                    logits_soft_cap=0.0,  # No soft cap
                    sm_scale=1.0 / (head_size**0.5)  # Standard scale
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
                'vllm.v1.attention.backends.flashinfer.get_per_layer_parameters',
                mock_get_per_layer_parameters):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config,
                                  device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        if actual_backend == _Backend.FLEX_ATTENTION:
            builder.direct_build = use_direct_block_mask
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    scale = 1.0 / (head_size**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="auto",
    )

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty_like(query)

    # Run forward pass
    # NOTE: The query, key, and value are already shaped correctly
    # in the calling test function.

    test_ite = 6
    warm_up = 3

    torch.cuda.synchronize()

    def run():
        impl.forward(
            mock_layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output=output
        )

    # Warmup
    for i in range(warm_up):
        run()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for i in range(test_ite):
        run()
    end_event.record()
    torch.cuda.synchronize()

    latency = start_event.elapsed_time(end_event) / test_ite
    print(f"latency: {latency}")

    # warmup
    # for i in range(warming_up):
    #     # DEBUG
    #     # print(
    #     #     f"q: {q.shape}, k: {k.shape}, v: {v.shape} kv_cache: {kv_cache.shape} "
    #     #     f"attn_metadata: {attn_metadata.num_actual_tokens} output: {output.shape}"
    #     # )
    #     # print(f"all attn_metadata: {attn_metadata}")
    #     g.replay()

    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)
    # start_event.record()
    # for i in range(test_ite):
    #     g.replay()
    # end_event.record()

    return output


def _test_backend_correctness(
    batch_spec: BatchSpec,
    model: str,
    backend_name: Union[_Backend, str],
    mask_mod,
    *,
    block_size: int = 16,
    atol: float = 1e-2,
    rtol: float = 1e-2,
):
    """
    Test that all backends produce similar outputs to a reference implementation
    using torch.nn.functional.scaled_dot_product_attention.

    This test works by:
    1. Generating a batch of sequences with specified context and query lengths.
    2. Computing a ground-truth attention output using torch.sdpa on
       contiguous Q, K, and V tensors.
    3. Simulating vLLM's paged KV cache: It takes the context portion of the
       K/V tensors and manually places them into a paged buffer according to
       the test's (randomly generated) block table.
    4. Running each vLLM attention backend with the new queries and the
       simulated paged KV cache.
    5. Comparing the vLLM backend's output to the ground-truth SDPA output.
    """

    ################
    # batch_size,
    # input_len,
    # num_heads,
    # num_key_value_heads,  # keep same as num_heads for MHA
    # head_dim,
    # use_fp8_kv_cache,
    # use_fp8_context_fmha,
    # is_context_phase,
    # perf_filename,
    # device="cuda:0",
    ################

    current_platform.seed_everything(42)
    vllm_config = create_vllm_config(model_name=model,
                                     max_model_len=max(batch_spec.seq_lens),
                                     block_size=block_size,
                                     num_gpu_blocks=8192)
    device = torch.device("cuda:0")

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    # 1. Setup
    batch_size = batch_spec.batch_size
    seq_lens = batch_spec.seq_lens
    query_lens = batch_spec.query_lens
    num_q_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config)
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config)
    head_size = vllm_config.model_config.get_head_size()
    sliding_window = vllm_config.model_config.get_sliding_window()
    dtype = convert_dtype_to_torch(vllm_config.model_config.dtype)
    block_size = vllm_config.cache_config.block_size
    scale = 1.0 / (head_size**0.5)

    # 2. Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    all_sdpa_outputs = []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = seq_lens[i]
        q_len = query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence to be used in SDPA
        q = torch.randn(q_len,
                        num_q_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
        k_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)
        v_full = torch.randn(s_len,
                             num_kv_heads,
                             head_size,
                             dtype=dtype,
                             device=device)

        # SDPA expects (N, H, L, D), so unsqueeze batch and permute
        q_sdpa_in = q.unsqueeze(0).transpose(1, 2)
        k_sdpa_in = k_full.unsqueeze(0).transpose(1, 2)
        v_sdpa_in = v_full.unsqueeze(0).transpose(1, 2)

        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0, (
                f"num_q_heads ({num_q_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})")
            repeats = num_q_heads // num_kv_heads
            k_sdpa_in = k_sdpa_in.repeat_interleave(repeats, dim=1)
            v_sdpa_in = v_sdpa_in.repeat_interleave(repeats, dim=1)

        # Create causal mask: query token i attends to positions 0 to
        #  (context_len + i)
        kv_len = s_len

        final_mask_mod = partial(mask_mod, context_len=context_len)
        block_mask = create_block_mask(final_mask_mod,
                                       B=None,
                                       H=None,
                                       Q_LEN=q_len,
                                       KV_LEN=kv_len,
                                       device=device)
        sdpa_out_i = flex_attention(q_sdpa_in,
                                    k_sdpa_in,
                                    v_sdpa_in,
                                    block_mask=block_mask,
                                    scale=scale,
                                    enable_gqa=True)

        all_sdpa_outputs.append(sdpa_out_i.transpose(1, 2).squeeze(0))

        # Inputs for vLLM backends are just the new tokens
        all_q_vllm.append(q)
        all_k_vllm.append(k_full[context_len:])
        all_v_vllm.append(v_full[context_len:])

        # Contextual K/V data used to populate the paged cache
        k_contexts.append(k_full[:context_len])
        v_contexts.append(v_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    key_vllm = torch.cat(all_k_vllm, dim=0)
    value_vllm = torch.cat(all_v_vllm, dim=0)
    sdpa_output = torch.cat(all_sdpa_outputs, dim=0)

    common_attn_metadata = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1000,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True)

    # 4. Run vLLM backends and compare
    # Note: flex_attention has known Triton kernel compatibility issues
    # with test infrastructures
    # for backend_name in backend_to_test:
        # FlashAttentionm + FlexAttention:
        #   [2, num_blocks, block_size, num_kv_heads, head_size]
        # FlashInfer:
        #   [num_blocks, 2, block_size, num_kv_heads, head_size]
        # Select the appropriate KV cache format for each backend
    kv_cache_for_backend = kv_cache
    if backend_name == _Backend.FLASHINFER:
        kv_cache_for_backend = kv_cache.transpose(0, 1)

        # For FlashInfer default to HND layout and
        kv_cache_for_backend = kv_cache_for_backend.transpose(
            2, 3).contiguous().transpose(2, 3)
        set_kv_cache_layout("HND")

    backend_output = run_attention_backend(
        backend_name,
        kv_cache_spec,
        ["placeholder"],
        vllm_config,
        device,
        common_attn_metadata,
        query_vllm,
        key_vllm,
        value_vllm,
        kv_cache_for_backend,
        sliding_window=sliding_window,
    )

    # Check shape and dtype consistency
    assert backend_output.shape == sdpa_output.shape, (
        f"[{backend_name}] shape {backend_output.shape} != "
        f"SDPA shape {sdpa_output.shape}")
    assert backend_output.dtype == sdpa_output.dtype, (
        f"[{backend_name}] dtype {backend_output.dtype} != "
        f"SDPA dtype {sdpa_output.dtype}")

    assert torch.isfinite(backend_output).all(), (
        f"[{backend_name}] produced non-finite values")

    # Check numerical similarity
    def error_msg(msg: str, backend_name: str):
        return (f"[{backend_name}] output differs from SDPA baseline. "
                f"{msg}")

    torch.testing.assert_close(backend_output,
                                sdpa_output,
                                rtol=rtol,
                                atol=atol,
                                msg=partial(error_msg,
                                            backend_name=backend_name))


@pytest.mark.parametrize("batch_spec_name", [
    "small_decode", "small_prefill", "mixed_small", "medium_decode",
    "medium_prefill", "mixed_medium", "large_decode", "large_prefill",
    "single_decode", "single_prefill"
])
@pytest.mark.parametrize("model", ["qwen/qwen2.5-0.5B-instruct"])
@pytest.mark.parametrize("backend_name", [
    _Backend.FLASH_ATTN,
    _Backend.FLASHINFER,
    _Backend.FLEX_ATTENTION,
])
# @pytest.mark.parametrize("model", ["meta-llama/Meta-Llama-3-8B"])
def test_causal_backend_correctness(batch_spec_name: str, model: str, backend_name: str):
    """Test backend's correctness with causal attention."""

    def causal_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
        *,
        context_len: int,
    ):
        return (q_idx + context_len) >= kv_idx

    batch_spec = BATCH_SPECS[batch_spec_name]

    # Fast FlexAttention needs to run with block_size=128
    LARGE_BLOCK_BACKENDS = ([_Backend.FLEX_ATTENTION]
                            if is_torch_equal_or_newer("2.9.0.dev0") else [])

    block_size = 128 if backend_name in LARGE_BLOCK_BACKENDS else 16

    _test_backend_correctness(batch_spec,
                                model,
                                backend_name,
                                causal_mask_mod,
                                block_size=block_size)

