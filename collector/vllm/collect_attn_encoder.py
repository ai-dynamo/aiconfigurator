# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder (non-causal) attention collector for multimodal / omni-modal models.

Covers ViT-style vision encoders, audio encoders, and any other bidirectional
encoder path: full N^2, MHA, no KV cache. Uses vLLM's official
``AttentionType.ENCODER_ONLY`` path, which:
  - sets ``causal=False`` (vit_attn_wrappers.py)
  - skips ``reshape_and_cache_flash`` in ``FlashAttentionImpl.do_kv_cache_update``
so the measured latency matches the kernel actually invoked by encoder inference.
"""

__compat__ = "vllm>=0.21.0"

import os

import torch
import vllm
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.version import __version__ as vllm_version

try:
    from vllm.utils import resolve_obj_by_qualname
except ImportError:
    from vllm.utils.import_utils import resolve_obj_by_qualname  # type: ignore

from collector.helper import benchmark_with_power, log_perf
from collector.vllm.collect_attn import MockAttentionLayer
from collector.vllm.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    with_exit_stack,
)


@with_exit_stack
def run_encoder_attention_torch(
    exit_stack,
    batch_size,
    seq_len,
    num_heads,
    head_dim,
    *,
    perf_filename,
    device="cuda:0",
):
    from vllm.v1.attention.backend import AttentionType

    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    block_size = 64

    # All seqs are full-prefill (encoder is single-pass)
    batch_spec = BatchSpec(
        seq_lens=[seq_len] * batch_size,
        query_lens=[seq_len] * batch_size,
    )

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    # MHA: num_kv_heads == num_heads
    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=seq_len,
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=False,
        sliding_window=None,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_heads,
    )
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    # vLLM >=0.21: signature is (selected_backend, AttentionSelectorConfig)
    from vllm.v1.attention.selector import AttentionSelectorConfig
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_dim, dtype=dtype, kv_cache_dtype=None,
        block_size=block_size, use_mla=False, has_sink=False, use_sparse=False,
    )
    backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    backend_name_obj = resolve_obj_by_qualname(backend)
    backend_name_str = backend_name_obj.get_name()

    common_attn_metadata = create_common_attn_metadata(batch_spec, block_size, device)
    # Non-causal: encoder attention does not use causal mask
    if hasattr(common_attn_metadata, "causal"):
        common_attn_metadata.causal = False

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, False)
    # ``backend`` from get_attn_backend_cls is a qualname string in vLLM >=0.21;
    # the resolved class exposes get_builder_cls/get_impl_cls directly, so skip
    # the Enum-based ``get_attention_backend()`` helper.
    builder_cls = backend_name_obj.get_builder_cls()
    impl_cls = backend_name_obj.get_impl_cls()
    layer_names = ["placeholder"]
    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    scale = 1.0 / (head_dim**0.5)
    # ENCODER_ONLY tells FlashAttentionImpl: causal=False + skip KV cache write
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=AttentionType.ENCODER_ONLY,
    )

    # Generate raw Q/K/V (no paged KV cache prepopulation needed for encoder)
    total_tokens = batch_size * seq_len
    query = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    key = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    output = torch.empty_like(query)
    # ENCODER_ONLY path uses Q/K/V directly; pass an empty KV cache tensor.
    kv_cache = torch.empty(0, dtype=dtype, device=device)

    mock_layer = MockAttentionLayer(device)

    def run():
        impl.forward(mock_layer, query, key, value, kv_cache, attn_metadata, output=output)

    # Use benchmark_with_power context manager
    with benchmark_with_power(
        device=device, kernel_func=run, num_warmups=3, num_runs=6, repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"encoder attn latency: {latency}")
    kernel_source = f"vllm_{backend_name_str}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "seqlen": seq_len,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "attn_dtype": "bfloat16",
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name="encoder_attention",
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def get_encoder_attention_test_cases(if_unit_test=False):
    if if_unit_test:
        return [[1, 256, 16, 72]]

    b_list = [1, 2, 4, 8, 16, 32, 64]
    s_list = [256, 400, 576, 1024, 1296, 2304, 3136, 4096, 5184, 6400,
              7744, 8192, 9216, 10816, 12544, 14400, 16384]
    n_list = [12, 16, 25]
    head_dim_list = [64, 72, 80, 88, 112, 128]

    test_cases = []
    for head_dim in head_dim_list:
        for n in sorted(n_list, reverse=True):
            for s in sorted(s_list, reverse=True):
                for b in sorted(b_list, reverse=True):
                    # Memory guard (single-pass, no paged cache): Q+K+V+O ~ 4*b*s*n*h*2B < 2GB
                    if 4 * b * s * n * head_dim * 2 >= 2**31:
                        continue
                    test_cases.append([b, s, n, head_dim])
    return test_cases


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    for test_case in get_encoder_attention_test_cases()[:5]:
        print(f"Running encoder attention test case: {test_case}")
        run_encoder_attention_torch(*test_case, perf_filename=PerfFile.ENCODER_ATTENTION)
