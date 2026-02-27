# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import AttentionInputType, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from helper import benchmark_with_power, log_perf


# TODO: refactor to use common_test_cases.py
def get_context_mla_test_cases():
    dtype_list = [tensorrt_llm.bindings.DataType.BF16]  # not support f8 for trt < v1.1
    test_cases = []
    n_list = [128]
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
    for n in n_list:
        for b in b_list:
            for s in s_list:  # [2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072]:
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
                        if b * s > 65536:
                            continue
                        # (input_len, batch_size, output_len, kv_cache_dtype, num_heads, world_size,
                        #  tp_size, tokens_per_block, warming_up, test_ite, is_context_phase)
                        test_cases.append(
                            [
                                s,
                                b,
                                1,
                                dtype,
                                n,
                                tp_size,
                                tp_size,
                                64,
                                10,
                                6,
                                True,
                                "context_mla_perf.txt",
                            ]
                        )
    return test_cases


# TODO: refactor to use common_test_cases.py
def get_generation_mla_test_cases():
    dtype_list = [tensorrt_llm.bindings.DataType.BF16]  # not support f8 for trt < v1.1
    test_cases = []
    n_list = [128]
    for n in n_list:
        for b in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for s in [
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
                65536,
                131072,
            ]:  # [target token s] is equivalent to [in: s-1, step=1]
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
                        if b * s > 1024 * 4096 * 2 * 2:
                            continue
                        # (input_len, batch_size, output_len, kv_cache_dtype, num_heads, world_size,
                        #  tp_size, tokens_per_block, warming_up, test_ite, is_context_phase)
                        test_cases.append(
                            [
                                s - 1,
                                b,
                                1,
                                dtype,
                                n,
                                tp_size,
                                tp_size,
                                64,
                                10,
                                6,
                                False,
                                "generation_mla_perf.txt",
                            ]
                        )
    return test_cases


def run_mla(
    input_len,
    batch_size,
    output_len,
    kv_cache_dtype,
    num_heads,
    world_size,
    tp_size,
    tokens_per_block,
    warming_up,
    test_ite,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    device = torch.device(device)
    torch.cuda.set_device(device)
    backend_name = "TRTLLM"
    layer_idx = 0

    assert kv_cache_dtype == tensorrt_llm.bindings.DataType.BF16, "only support bfloat16 for trtllm"
    assert num_heads % tp_size == 0, "num_heads != N * tp_size"
    num_heads = num_heads // tp_size
    num_key_value_heads = num_heads

    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        embedder=None,
        rope=RopeParams(
            dim=56,
            theta=10000,
            scale_type=RotaryScalingType.yarn,
            scale=40,
            low_freq_factor=1.0,
            high_freq_factor=4.0,
            short_m_scale=1.0,
            long_m_scale=1.0,
            max_positions=163840,
            original_max_positions=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        ),
    )

    quant_config = QuantConfig(
        kv_cache_quant_algo=None,
        group_size=None,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    predicted_tokens_per_seq = 1

    attn = create_attention(
        backend_name=backend_name,
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=(qk_nope_head_dim if is_context_phase else kv_lora_rank) + qk_rope_head_dim,
        num_kv_heads=num_key_value_heads if is_context_phase else 1,
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=True,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim if is_context_phase else kv_lora_rank,
        predicted_tokens_per_seq=predicted_tokens_per_seq,
    )

    total_num_tokens = (input_len + output_len) * batch_size
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp_size)

    num_hidden_layers = 1

    kv_cache_config = KvCacheConfig(
        max_tokens=int((input_len + output_len - 1) / tokens_per_block + 1)
        * tokens_per_block
        * batch_size
        * 2,  # num_bloacks * block_size
        enable_block_reuse=False,
    )

    kv_cache_manager = KVCacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_hidden_layers,
        num_kv_heads=1,
        head_dim=kv_lora_rank + qk_rope_head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=input_len + output_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    input_seq_lens = [input_len for _ in range(batch_size)]
    total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    if is_context_phase:
        num_cached_tokens_per_seq = [0 for _ in range(batch_size)]
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=total_num_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            enable_flash_mla=True,
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32),
            position_ids=None,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            cross=None,
            request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
        )
    else:
        gen_seq_lens = [1 for _ in range(batch_size)]
        attn_metadata = TrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=total_num_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            enable_flash_mla=True,
            seq_lens=torch.tensor(gen_seq_lens, dtype=torch.int32),
            position_ids=None,
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=input_seq_lens,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            cross=None,
            request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
        )

    attn_metadata.prepare()

    if is_context_phase:
        num_tokens = input_len * batch_size
    else:
        num_tokens = batch_size

    compressed_kv = torch.randn([num_tokens, kv_lora_rank], dtype=torch.bfloat16, device=device)
    k_pe = torch.randn([num_tokens, qk_rope_head_dim], dtype=torch.bfloat16, device=device)

    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

    # self.qk_nope_head_dim + self.qk_rope_head_dim
    if is_context_phase:
        fused_q = torch.randn(
            [
                num_tokens,
                num_heads
                * (
                    qk_nope_head_dim + qk_rope_head_dim + qk_nope_head_dim + qk_rope_head_dim + v_head_dim
                ),  # 128 128 64
            ],
            device=device,
            dtype=torch.bfloat16,
        )
        q_pe = None
    else:
        fused_q = torch.randn(
            [num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)],
            device=device,
            dtype=torch.bfloat16,
        )
        q_pe = torch.randn([num_tokens, num_heads, qk_rope_head_dim], dtype=torch.bfloat16, device=device)

    # dry run
    if is_context_phase:
        attn.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            out_scale=torch.tensor([]).float().to(torch.device(device)),
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
        )
    else:
        attn.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            out_scale=torch.tensor([]).float().to(torch.device(device)),
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
        )

    # Use benchmark_with_power context manager
    def kernel_func():
        if is_context_phase:
            attn.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                out_scale=torch.tensor([]).float().to(torch.device(device)),
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
            )
        else:
            attn.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                out_scale=torch.tensor([]).float().to(torch.device(device)),
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
            )

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    # write result
    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    log_perf(
        item_list=[
            {
                "mla_dtype": "float16",
                "kv_cache_dtype": "float16",
                "num_heads": num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
