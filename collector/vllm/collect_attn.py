# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM 0.24.0 dense-attention collector for CUDA backends."""

__compat__ = "vllm==0.24.0"

import os

import torch
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.version import __version__ as vllm_version

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.vllm.utils import (
    BatchSpec,
    create_and_prepopulate_kv_cache,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    get_attention_backend,
    with_exit_stack,
)


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


# https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends
# support MHA GQA MQA bfloat16 tensor and bfloat16/fp8 kv cache


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    num_kv_heads,  # keep same as num_heads for MHA
    head_dim,
    use_fp8_kv_cache,
    is_context_phase,
    window_size=0,
    *,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_hf_model")
    block_size = 64

    if is_context_phase:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[input_len] * batch_size,
        )
    else:
        batch_spec = BatchSpec(
            seq_lens=[input_len] * batch_size,
            query_lens=[1] * batch_size,
        )

    set_random_seed(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=8192,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
        sliding_window=window_size if window_size > 0 else None,
        head_dim=head_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )

    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_dim,
        dtype=dtype,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
        block_size=block_size,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
    )
    backend_path = current_platform.get_attn_backend_cls(
        None,
        attn_selector_config,
        num_heads=num_heads,
    )
    backend_cls = resolve_obj_by_qualname(backend_path)
    backend_name_str = backend_cls.get_name()
    backend_name = AttentionBackendEnum[backend_name_str]

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Ensure the KV cache has enough blocks for all sequences.
    required_blocks = 1 + sum((s_len + block_size - 1) // block_size for s_len in batch_spec.seq_lens)
    num_blocks = vllm_config.cache_config.num_gpu_blocks or required_blocks
    num_blocks = max(num_blocks, required_blocks)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_k_vllm, all_v_vllm = [], [], []
    k_contexts, v_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate Q, K, V for the whole sequence
        q = torch.randn(q_len, num_heads, head_dim, dtype=dtype, device=device)
        k_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_full = torch.randn(s_len, num_kv_heads, head_dim, dtype=dtype, device=device)

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

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=current_platform.fp8_dtype() if use_fp8_kv_cache else dtype,
        device=device,
        num_blocks=num_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
    )

    # The helper populates [2, num_blocks, ...]; vLLM 0.24 uses logical
    # [num_blocks, 2, ...] and may impose a backend-specific physical stride.
    kv_cache = kv_cache.transpose(0, 1).contiguous()
    set_kv_cache_layout(backend_cls.get_required_kv_cache_layout())
    exit_stack.callback(set_kv_cache_layout, None)
    stride_order = backend_cls.get_kv_cache_stride_order()
    if stride_order != tuple(range(kv_cache.ndim)):
        inverse_order = [stride_order.index(i) for i in range(kv_cache.ndim)]
        kv_cache = kv_cache.permute(*stride_order).contiguous().permute(*inverse_order)

    builder_cls, impl_cls = get_attention_backend(backend_name)
    layer_names = ["placeholder"]

    # Mock flashinfer's get_per_layer_parameters if needed
    if backend_name_str == "FLASHINFER":
        import unittest.mock

        from vllm.v1.attention.backends.utils import PerLayerParameters

        def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
            window_left = window_size - 1 if window_size > 0 else -1
            return {
                layer_name: PerLayerParameters(
                    window_left=window_left,
                    logits_soft_cap=0.0,
                    sm_scale=1.0 / (head_dim**0.5),
                )
                for layer_name in layer_names
            }

        with unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters", mock_get_per_layer_parameters
        ):
            builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
            attn_metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
    else:
        # Build metadata
        builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
        if backend_name_str == "FLEX_ATTENTION":
            builder.direct_build = True
        # FA3's metadata builder auto-detects sliding_window by walking registered
        # Attention layers; the collector doesn't register any, so the auto-detect
        # finds nothing and aot_sliding_window stays at (-1, -1). Then FA3's
        # AOT scheduler computes scheduler_metadata for "no sliding window", but
        # impl.forward later passes the actual (window_size-1, 0) tuple, and FA3's
        # shape check rejects the mismatched metadata_size. Pre-populate the
        # builder's aot_sliding_window to match what impl.sliding_window will be.
        if window_size > 0 and hasattr(builder, "aot_sliding_window"):
            builder.aot_sliding_window = (window_size - 1, 0)
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )

    # Instantiate implementation
    sliding_window = vllm_config.model_config.get_sliding_window()
    scale = 1.0 / (head_dim**0.5)
    impl = impl_cls(
        num_heads=num_heads,
        head_size=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=sliding_window,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
    )

    # Create mock layer
    mock_layer = MockAttentionLayer(device)

    # Run forward pass

    test_ite = 6
    warm_up = 3

    needs_fp8_query = use_fp8_kv_cache and impl.supports_quant_query_input
    query_fwd = query_vllm.to(current_platform.fp8_dtype()) if needs_fp8_query else query_vllm
    # Output buffer is always BF16 — the impl dequantises internally.
    output = torch.empty_like(query_vllm)

    def run():
        impl.forward(
            mock_layer,
            query_fwd,
            key_vllm,
            value_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )

    # Use benchmark_with_power context manager
    with benchmark_with_power(
        device=device,
        kernel_func=run,
        num_warmups=warm_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]
    print(f"attn latency: {latency}")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    kv_cache_dtype_str = "bfloat16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "bfloat16"
    kernel_source = f"vllm_{backend_name_str}".lower()

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "window_size": window_size,
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )


def get_context_attention_test_cases(if_unit_test=False):
    test_cases = []

    if if_unit_test:
        shape_sweeps = [
            {
                "batch_sizes": [1],
                "sequence_lengths": [64],
                "query_head_counts": [4],
                "kv_head_options": [0],
                "head_dims": [128],
                "window_sizes": [0, 128],
                "max_tokens_self_attention": 65536,
                "max_tokens_grouped_query_attention": 131072,
                "max_kv_elements": 2147483647,
            }
        ]
    else:
        shape_sweeps = get_attention_context_shape_sweeps("vllm")

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    for shape_sweep in shape_sweeps:
        batch_sizes = [int(value) for value in shape_sweep["batch_sizes"]]
        sequence_lengths = [int(value) for value in shape_sweep["sequence_lengths"]]
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_config in get_attention_head_configs(shape_sweep, phase="context"):
            n = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            for s in sorted(sequence_lengths, reverse=True):
                for b in sorted(batch_sizes, reverse=True):
                    if num_kv_heads == n:
                        if b * s > max_tokens_self_attention or b > 128:
                            continue
                    elif b * s > max_tokens_grouped_query_attention:
                        continue
                    if b * s * num_kv_heads * head_dim * 2 >= max_kv_elements:
                        continue
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                head_dim,
                                is_fp8_kv_cache,
                                True,
                                window_size,
                            ]
                        )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, head_dim, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens * 128 // head_dim // s // num_heads
        for b in batch_sizes:
            if b > max_b:
                break
            if s not in s_b_dict:
                s_b_dict[s] = {b}
            else:
                s_b_dict[s].add(b)
    for s, b_set in s_b_dict.items():
        if len(b_set) < int(shape_sweep["min_batch_options_per_sequence"]):
            continue
        for b in b_set:
            if b not in b_s_dict:
                b_s_dict[b] = {s - 1}
            b_s_dict[b].add(s - 1)
    return b_s_dict


def get_generation_attention_test_cases():
    test_cases = []

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    for shape_sweep in get_attention_generation_shape_sweeps("vllm"):
        batch_sizes = [int(value) for value in shape_sweep["batch_sizes"]]
        sequence_lengths = [int(value) for value in shape_sweep["sequence_lengths"]]
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])

        for head_config in get_attention_head_configs(shape_sweep, phase="generation"):
            n = head_config.num_heads
            n_kv = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            # The generation schema has separate caps because MHA and GQA have
            # different memory/throughput limits. The old loop used the MHA cap
            # for every case even though it enumerated GQA shapes as well.
            max_tokens_key = "max_mha_tokens_per_step" if n == n_kv else "max_xqa_tokens_per_step"
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                head_dim,
                int(shape_sweep[max_tokens_key]),
                shape_sweep,
            )
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= min_drop_batch:
                    target_s_list = target_s_list[:-1]
                for s in target_s_list:
                    for is_fp8_kv_cache in kv_cache_dtype_list:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                head_dim,
                                is_fp8_kv_cache,
                                False,
                                window_size,
                            ]
                        )
    return test_cases


if __name__ == "__main__":
    from collector.registry_types import PerfFile

    test_cases = get_context_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
