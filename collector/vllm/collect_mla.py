# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
import os

import torch
import vllm
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.version import __version__ as vllm_version

from collector.common_test_cases import get_context_mla_common_test_cases, get_generation_mla_common_test_cases
from collector.helper import get_sm_version, log_perf
from collector.vllm.utils import (
    BatchSpec,
    MockAttentionLayer,
    _Backend,
    convert_dtype_to_torch,
    create_and_prepopulate_kv_cache_mla,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
    get_attention_backend,
    resolve_obj_by_qualname,
    setup_distributed,
    with_exit_stack,
)


@with_exit_stack
def run_attention_torch(
    exit_stack,
    batch_size,
    input_len,
    num_heads,
    tp_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    block_size,
    model_name,
    use_fp8_kv_cache,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    setup_distributed(device)
    torch.cuda.set_device(device)

    assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
    num_heads = num_heads // tp_size

    dtype = torch.bfloat16
    model = os.path.join(os.path.dirname(__file__), "fake_mla_hf_model")
    head_dim = kv_lora_rank + qk_rope_head_dim
    num_kv_heads = num_heads

    num_kv_cache_blocks = max(
        # Number of kv cache blocks needed for number of tokens in the entire KV cache.
        # Add +1 because VLLM considers the 1st block to be the "null" block.
        1 + math.ceil((input_len + 1) / block_size) * batch_size,
        # set a reasonable minimum
        8192,
    )
    try:
        # Let vllm choose the backend.
        # defautl for vllm 0.11.0
        backend = current_platform.get_attn_backend_cls(
            None,
            head_dim,
            dtype,
            kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
            block_size=block_size,
            use_v1=True,
            use_mla=True,
            has_sink=False,
            use_sparse=False,
        )
    except TypeError:
        try:
            # in the case of vllm 0.12.0 use_v1 is removed
            backend = current_platform.get_attn_backend_cls(
                None,
                head_dim,
                dtype,
                kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
        except TypeError:
            # vllm 0.14.0
            from vllm.v1.attention.selector import AttentionSelectorConfig

            attn_selector_config = AttentionSelectorConfig(
                head_size=head_dim,
                dtype=dtype,
                kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
                block_size=block_size,
                use_mla=True,
                has_sink=False,
                use_sparse=False,
            )
            backend = current_platform.get_attn_backend_cls(None, attn_selector_config)

    if _Backend is not None:
        backend_name = _Backend[resolve_obj_by_qualname(backend).get_name()]
        print(f"VLLM chose MLA backend: {backend_name}")
        builder_cls, impl_cls = get_attention_backend(backend_name)
    else:
        backend_cls = resolve_obj_by_qualname(backend)
        backend_name = backend_cls.get_name()
        print(f"VLLM chose MLA backend: {backend_name}")
        builder_cls = backend_cls.get_builder_cls()
        impl_cls = backend_cls.get_impl_cls()

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

    try:
        vllm.utils.torch_utils.set_random_seed(42)
    except AttributeError:
        current_platform.seed_everything(42)

    vllm_config = create_vllm_config(
        model_name=model,
        max_model_len=max(batch_spec.seq_lens),
        block_size=block_size,
        num_gpu_blocks=num_kv_cache_blocks,
        max_num_seqs=batch_size,
        use_fp8_kv_cache=use_fp8_kv_cache,
    )
    assert convert_dtype_to_torch(vllm_config.model_config.dtype) == torch.bfloat16

    kv_cache_spec = create_standard_kv_cache_spec(vllm_config, use_fp8_kv_cache)

    # Generate data and compute SDPA reference output
    all_q_vllm, all_kv_c_vllm, all_k_pe_vllm = [], [], []
    kv_c_contexts, k_pe_contexts = [], []

    for i in range(batch_size):
        s_len = batch_spec.seq_lens[i]
        q_len = batch_spec.query_lens[i]
        context_len = s_len - q_len

        # Generate MLA tensors
        # Q has both nope and rope components:
        # [q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim]
        q_c = torch.randn(q_len, num_heads, qk_nope_head_dim + qk_rope_head_dim, dtype=dtype, device=device)

        # KV_C (latent K/V): [s_len, kv_lora_rank]
        kv_c_full = torch.randn(s_len, kv_lora_rank, dtype=dtype, device=device)

        # K_PE (rope component): [s_len, 1, qk_rope_head_dim]
        k_pe_full = torch.randn(s_len, 1, qk_rope_head_dim, dtype=dtype, device=device)

        # Inputs for vLLM MLA backends are just the new tokens
        all_q_vllm.append(q_c)
        all_kv_c_vllm.append(kv_c_full[context_len:])  # New kv_c tokens
        all_k_pe_vllm.append(k_pe_full[context_len:])  # New k_pe tokens

        # Contextual K/V data used to populate the paged cache (MLA format)
        kv_c_contexts.append(kv_c_full[:context_len])
        k_pe_contexts.append(k_pe_full[:context_len])

    query_vllm = torch.cat(all_q_vllm, dim=0)
    kv_c_vllm = torch.cat(all_kv_c_vllm, dim=0)
    k_pe_vllm = torch.cat(all_k_pe_vllm, dim=0)

    common_attn_metadata = create_common_attn_metadata(batch_spec, vllm_config.cache_config.block_size, device)

    # 3. Simulate Paged KV Cache and a realistic slot_mapping
    kv_cache = create_and_prepopulate_kv_cache_mla(
        kv_c_contexts=all_kv_c_vllm,
        k_pe_contexts=all_k_pe_vllm,
        block_size=block_size,
        head_size=head_dim,
        dtype=current_platform.fp8_dtype() if use_fp8_kv_cache else dtype,
        device=device,
        num_blocks=num_kv_cache_blocks,
        common_attn_metadata=common_attn_metadata,
        randomize_blocks=True,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else None,
    )

    # Build metadata
    layer_names = ["placeholder"]
    exit_stack.enter_context(set_current_vllm_config(vllm_config))

    builder = builder_cls(kv_cache_spec, layer_names, vllm_config, device)
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )

    # Create mock kv_b_proj using the same weights as reference implementation
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    mock_kv_b_proj = ColumnParallelLinear(
        input_size=kv_lora_rank, output_size=num_heads * (qk_nope_head_dim + v_head_dim), bias=False
    ).to(device=device, dtype=dtype)

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
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_b_proj=mock_kv_b_proj,
    )

    # Process weights to create W_UK_T and W_UV attributes needed by MLA
    impl.process_weights_after_loading(dtype)

    # Create mock layer and output buffer
    mock_layer = MockAttentionLayer(device)
    output = torch.empty(
        query_vllm.shape[0],
        num_heads * v_head_dim,
        dtype=query_vllm.dtype,
        device=query_vllm.device,
    )

    # Run forward pass

    test_ite = 6
    warm_up = 3

    def run():
        impl.forward(
            mock_layer,
            query_vllm,
            kv_c_vllm,
            k_pe_vllm,
            kv_cache,
            attn_metadata,
            output=output,
        )

    # Warmup
    for i in range(warm_up):
        run()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for i in range(test_ite):
        run()
    end_event.record()
    torch.cuda.synchronize()

    latency = start_event.elapsed_time(end_event) / test_ite
    print(f"MLA latency: {latency}")

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_mla"
    else:
        isl = 1
        step = input_len
        op_name = "generation_mla"

    kv_cache_dtype_str = "float16" if not use_fp8_kv_cache else "fp8"
    dtype_str = "float16"
    kernel_source = f"vllm_{backend_name}".lower()

    log_perf(
        item_list=[
            {
                "mla_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="VLLM",
        version=vllm_version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=kernel_source,
        perf_filename=perf_filename,
    )


def _get_mla_test_cases(is_context: bool):
    test_cases = []

    kv_cache_dtype_list = [False]
    if get_sm_version() > 86:
        kv_cache_dtype_list.append(True)

    if is_context:
        common_test_cases = get_context_mla_common_test_cases()
    else:
        common_test_cases = get_generation_mla_common_test_cases()

    tp_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    for common_mla_testcase in common_test_cases:
        for tp_size in tp_sizes:
            if common_mla_testcase.num_heads % tp_size != 0:
                continue

            for is_fp8_kv_cache in kv_cache_dtype_list:
                test_cases.append(
                    [
                        common_mla_testcase.batch_size,
                        common_mla_testcase.input_len,
                        common_mla_testcase.num_heads,
                        tp_size,
                        common_mla_testcase.q_lora_rank,
                        common_mla_testcase.kv_lora_rank,
                        common_mla_testcase.qk_rope_head_dim,
                        common_mla_testcase.qk_nope_head_dim,
                        common_mla_testcase.v_head_dim,
                        common_mla_testcase.kv_cache_block_size,
                        common_mla_testcase.model_name,
                        is_fp8_kv_cache,
                        is_context,
                        "context_mla_perf.txt" if is_context else "generation_mla_perf.txt",
                    ]
                )

    return test_cases


def get_context_mla_test_cases():
    return _get_mla_test_cases(is_context=True)


def get_generation_mla_test_cases():
    return _get_mla_test_cases(is_context=False)


if __name__ == "__main__":
    test_cases = get_context_mla_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running context attention test case: {test_case}")
        run_attention_torch(*test_case)

    test_cases = get_generation_mla_test_cases()
    test_cases = test_cases[:10]
    for test_case in test_cases:
        print(f"Running generation attention test case: {test_case}")
        run_attention_torch(*test_case)
