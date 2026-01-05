# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

from helper import benchmark_with_power, get_sm_version, log_perf


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,  # keep same as num_heads for MHA
    head_dim,
    attention_window_size,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    device = torch.device(device)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    # if XQA JIT is enabled, the context phase will also trigger XQA prepare which causes the error
    # with specifc q/kv head and seq setting.
    if is_context_phase:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "0"
    else:
        os.environ["TRTLLM_ENABLE_XQA_JIT"] = "1"

    backend_name = "TRTLLM"
    layer_idx = 0
    world_size = 1
    tp_size = 1
    tokens_per_block = 64
    warming_up = 10
    test_ite = 6
    output_len = 1
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache
        quant_algo = QuantAlgo.FP8
        out_scale = torch.tensor(
            [1.0],
            dtype=torch.float32,
            device=device,
        )  # fp8 fmha
    else:
        quant_algo = None
        out_scale = None

    if use_fp8_kv_cache:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16

    pos_embd_params = PositionalEmbeddingParams(type=PositionEmbeddingType.rope_gpt_neox, rope=RopeParams(dim=128))

    quant_config = QuantConfig(
        quant_algo=quant_algo,  # fp8 fmha
        kv_cache_quant_algo=QuantAlgo.FP8 if use_fp8_kv_cache else None,  # fp8 kv,
        group_size=128,
        smoothquant_val=0.5,
        clamp_val=None,
        use_meta_recipe=False,
        has_zero_point=False,
        pre_quant_scale=False,
        exclude_modules=None,
    )

    attn = create_attention(
        backend_name=backend_name,
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_key_value_heads,
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=False,
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
        kv_cache_config=kv_cache_config,
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_hidden_layers,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=input_len + output_len + 1,  # +1 for the magic fixme mentioned in trtllm xqa JIT path impl.
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
            enable_flash_mla=False,
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32, device="cpu"),
            num_contexts=batch_size,
            position_ids=None,
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
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False, cache_reuse=False, has_speculative_draft_tokens=False
            ),
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
            enable_flash_mla=False,
            seq_lens=torch.tensor(gen_seq_lens, dtype=torch.int32, device="cpu"),
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

    sinks = torch.randn(num_heads, dtype=torch.float32) if head_dim == 64 else None
    q = torch.randn([num_tokens, num_heads * head_dim]).bfloat16().to(torch.device(device))
    kv = torch.randn([num_tokens, 2 * num_key_value_heads * head_dim]).bfloat16().to(torch.device(device))
    input_qkv = torch.concat([q, kv], dim=-1)
    attn.forward(
        input_qkv,
        None,
        None,
        attn_metadata,
        attention_window_size=attention_window_size if attention_window_size > 0 else None,
        attention_sinks=sinks,
        out_scale=out_scale,
    )

    # Use benchmark_with_power context manager
    def kernel_func():
        attn.forward(
            input_qkv,
            None,
            None,
            attn_metadata,
            attention_window_size=attention_window_size if attention_window_size > 0 else None,
            attention_sinks=sinks,
            out_scale=out_scale,
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
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"
    kv_cache_dtype_str = "float16"
    if use_fp8_kv_cache:
        kv_cache_dtype_str = "fp8"
    if use_fp8_context_fmha:
        dtype_str = "fp8"
    else:
        dtype_str = "float16"

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "window_size": attention_window_size,
                "beam_width": 1,
                "attn_dtype": dtype_str,
                "kv_cache_dtype": kv_cache_dtype_str,
                "step": step,
                "latency": latency,
            }
        ],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="torch_flow",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )
    kv_cache_manager.shutdown()


def get_context_attention_test_cases():
    has_fp8 = get_sm_version() > 86
    test_cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [
        1,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        10240,
        12288,
        16384,
        262144,
    ]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96]
    n_kv_list = [0, 1, 2, 4, 8]
    head_dim = [64, 128]

    for h in head_dim:
        for n in sorted(n_list, reverse=True):
            for s in sorted(s_list, reverse=True):
                for b in sorted(b_list, reverse=True):
                    for n_kv in n_kv_list:
                        if n_kv != 0 and (n_kv >= n or n % n_kv != 0):
                            continue
                        num_kv_heads = n_kv if n_kv != 0 else n

                        if num_kv_heads == n:
                            if b * s > 65536 or b > 128:
                                continue
                        else:
                            if b * s > 131072:
                                continue
                        if b * s * num_kv_heads * 128 * 2 >= 2147483647:
                            continue
                        if get_sm_version() >= 100:
                            # though it's a precheck of gen kernels during the attention op init,
                            # this cannot be skipped for now
                            # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                            m_num_heads_q_per_kv = 1 if n_kv == 0 else n // n_kv
                            max_num_heads_q_per_kv_in_cta = 32
                            if (
                                m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                                and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                            ):
                                continue

                        # print(
                        #     f"collecting heads: {n} kv_heads: {num_kv_heads} seq: {s} "
                        #     f"batchsize: {b}"
                        # )
                        # use fp8 kv cache, fp8 context fmha, is_context_phase. in torch flow,
                        # int8 kvcache is not supported yet.
                        #
                        # fp16 kv cache, fp16 context fmha, is_context_phase
                        if h == 64:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    num_kv_heads,
                                    h,
                                    128,
                                    False,
                                    False,
                                    True,
                                    "context_attention_perf.txt",
                                ]
                            )
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    num_kv_heads,
                                    h,
                                    0,
                                    False,
                                    False,
                                    True,
                                    "context_attention_perf.txt",
                                ]
                            )
                            if has_fp8:
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        128,
                                        True,
                                        False,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        128,
                                        True,
                                        True,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        0,
                                        True,
                                        False,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        0,
                                        True,
                                        True,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )
                        else:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    num_kv_heads,
                                    h,
                                    0,
                                    False,
                                    False,
                                    True,
                                    "context_attention_perf.txt",
                                ]
                            )
                            if has_fp8:
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        0,
                                        True,
                                        False,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        0,
                                        True,
                                        True,
                                        True,
                                        "context_attention_perf.txt",
                                    ]
                                )

    return test_cases


def get_generation_attention_test_cases():
    has_fp8 = get_sm_version() > 86
    test_cases = []

    # generation
    b_list = [
        1,
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
    ]
    # the i-th token to record. 1 for context phase. mapping to osl definition
    s_list = [
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
    ]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
    n_list_xqa = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    n_kv_list = [1, 2, 4, 8]
    head_dim = [64, 128]

    # MHA
    max_bsn = 8192 * 1024  # 2*1024*1024*1024/128/2 INT32MAX/128/2
    for n in sorted(n_list, reverse=True):
        b_s_dict = {}
        s_b_dict = {}
        for s in s_list:
            max_b = max_bsn // s // n  # b*s*n*byte <= max_bsn
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict:
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict:
                    b_s_dict[b] = {s - 1}
                b_s_dict[b].add(s - 1)
        for h in head_dim:
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= 256:
                    target_s_list = target_s_list[:-1]
                # print(f'collecting MHA heads: {n} batchsize: {b}  steps: {s_list_limited}')
                # fp8 kv cache, fp8 context fmha, is_context_phase
                for s in target_s_list:
                    test_cases.append([b, s, n, n, h, 0, False, False, False, "generation_attention_perf.txt"])

                    if has_fp8:
                        test_cases.append([b, s, n, n, h, 0, True, False, False, "generation_attention_perf.txt"])
                        # currently, fp8 is not for generation compute
                        # test_cases.append(
                        #     [b, s, n, n, 128, True, True, False, "generation_attention_perf.txt"]
                        # )

    # XQA
    max_bsn = 8192 * 1024 * 2  # 2*1024*1024*1024/128/2
    for n in sorted(n_list_xqa, reverse=True):
        b_s_dict = {}
        s_b_dict = {}
        for s in s_list:
            max_b = max_bsn // s // n
            for b in b_list:
                if b > max_b:
                    break
                if s not in s_b_dict:
                    s_b_dict[s] = {b}
                else:
                    s_b_dict[s].add(b)
        for s, b_set in s_b_dict.items():
            if len(b_set) < 4:
                continue
            for b in b_set:
                if b not in b_s_dict:
                    b_s_dict[b] = {s - 1}
                b_s_dict[b].add(s - 1)
        for h in head_dim:
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= 256:
                    target_s_list = target_s_list[:-1]
                for n_kv in n_kv_list:
                    if n_kv >= n:
                        continue

                    # fp8 kv cache, fp8 context fmha, is_context_phase
                    for s in target_s_list:
                        if get_sm_version() >= 100:
                            # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                            m_num_heads_q_per_kv = 1 if n_kv == 0 else n // n_kv
                            max_num_heads_q_per_kv_in_cta = 32
                            if (
                                m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                                and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                            ):
                                continue
                        if h == 64:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    128,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    0,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            if has_fp8:
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        n_kv,
                                        h,
                                        128,
                                        True,
                                        False,
                                        False,
                                        "generation_attention_perf.txt",
                                    ]
                                )
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        n_kv,
                                        h,
                                        0,
                                        True,
                                        False,
                                        False,
                                        "generation_attention_perf.txt",
                                    ]
                                )
                                # currently, fp8 is not for generation compute
                                # test_cases.append(
                                #     [
                                #         b,
                                #         s,
                                #         n,
                                #         n_kv,
                                #         128,
                                #         True,
                                #         True,
                                #         False,
                                #         "generation_attention_perf.txt",
                                #     ]
                                # )
                        else:
                            test_cases.append(
                                [
                                    b,
                                    s,
                                    n,
                                    n_kv,
                                    h,
                                    0,
                                    False,
                                    False,
                                    False,
                                    "generation_attention_perf.txt",
                                ]
                            )
                            if has_fp8:
                                test_cases.append(
                                    [
                                        b,
                                        s,
                                        n,
                                        n_kv,
                                        h,
                                        0,
                                        True,
                                        False,
                                        False,
                                        "generation_attention_perf.txt",
                                    ]
                                )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case)
