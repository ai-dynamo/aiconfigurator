# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from vllm.attention.backends.abstract import (
    AttentionType,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl, FlashAttentionMetadata
from vllm.v1.attention.backends.flashinfer import FlashInferImpl, FlashInferMetadata
from vllm.version import __version__ as vllm_version

from helper import get_sm_version, log_perf

# https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/backends
# support MHA GQA MQA fp16 tensor and float16/fp8 kv cache
# flashatten for prefill and flashinfer for decode (TODO) actual model support status
# TODO to support cross-attention


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,  # keep same as num_heads for MHA
    head_dim,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    perf_filename,
    device="cuda:0",
):
    torch.cuda.set_device(device)
    warming_up = 10
    test_ite = 6

    if is_context_phase:
        # Context phase: always use FlashAttentionImpl
        dtype = torch.float16
        num_tokens = input_len * batch_size
        # Create query/key/value tensors
        # Note: For MHA, num_key_value_heads must equal num_heads
        # For MQA/GQA, num_key_value_heads can be smaller (but must divide num_heads)
        q = torch.randn([num_tokens, num_heads, head_dim], dtype=dtype, device=device)
        k = torch.randn([num_tokens, num_key_value_heads, head_dim], dtype=dtype, device=device)
        v = torch.randn([num_tokens, num_key_value_heads, head_dim], dtype=dtype, device=device)
        # Calculate required number of blocks
        # Each block holds 64 tokens (block_size), and we need enough blocks to store input_len
        # tokens per sequence
        block_size = 64  # Fixed block size (tokens per block)
        num_blocks_per_seq = (input_len + block_size - 1) // block_size  # Ceiling division
        # Initialize KV cache
        # Shape: [num_layers, total_blocks, block_size, num_key_value_heads, head_dim]
        # Total blocks needed = batch_size * blocks_per_sequence
        total_blocks = batch_size * num_blocks_per_seq
        if use_fp8_kv_cache:
            kv_cache_dtype = torch.float8_e4m3fn
        else:
            kv_cache_dtype = torch.float16
        kv_cache = torch.zeros(
            (2, total_blocks, block_size, num_key_value_heads, head_dim),
            dtype=kv_cache_dtype,
            device=device,
        )
        # query_start_loc marks the starting position of each sequence in the flattened token array
        # Format: [0, input_len, 2*input_len, ..., batch_size*input_len]
        query_start_loc = torch.arange(0, batch_size + 1, 1, dtype=torch.int32, device=device) * input_len
        # seq_lens stores the length of each sequence (all equal to input_len here)
        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=device)
        # block_table maps each sequence to its allocated physical blocks
        # Linear allocation: first sequence gets blocks 0...N-1, second gets N...2N-1, etc.
        block_table = torch.arange(0, total_blocks, dtype=torch.int32, device=device).reshape(
            batch_size, num_blocks_per_seq
        )
        # slot_mapping maps each token to its physical storage location
        # Linear mapping: token i goes to slot i in the KV cache
        slot_mapping = torch.arange(0, num_tokens, dtype=torch.long, device=device)
        # Safety checks
        assert block_table.max() < total_blocks, "block_table references non-existent blocks"
        assert slot_mapping.max() < total_blocks * block_size, "slot_mapping exceeds physical storage capacity"

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=input_len,
            query_start_loc=query_start_loc,
            max_seq_len=input_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            scheduler_metadata=None,
            prefix_scheduler_metadata=None,
            local_attn_metadata=None,
        )
        attn = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_key_value_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
            blocksparse_params=None,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            use_irope=False,
        )

        class DummyLayer(torch.nn.Module):
            def __init__(self, num_heads, num_key_value_heads, head_dim, device):
                super().__init__()
                assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"

                self.num_heads = num_heads
                self.num_key_value_heads = num_key_value_heads

                # orignal scale tensor
                self.register_buffer("_q_scale_base", torch.ones(num_heads, dtype=torch.float32, device=device))
                self.register_buffer(
                    "_k_scale_base",
                    torch.ones(num_key_value_heads, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "_v_scale_base",
                    torch.ones(num_key_value_heads, dtype=torch.float32, device=device),
                )

                # FlashAttention demanded API
                self.register_buffer("_k_scale", self._k_scale_base)
                self.register_buffer("_v_scale", self._v_scale_base)

                # adapt reshape_and_cache_flash API
                self._k_scale_float = self._k_scale_base
                self._v_scale_float = self._v_scale_base

            @property
            def _q_scale(self):
                # return scalar mean, avoid shape problem
                return self._q_scale_base.mean().unsqueeze(0)

        layer = DummyLayer(num_heads, num_key_value_heads, head_dim, device)

        if use_fp8_kv_cache:
            # FP8 input requires BF16 output
            output = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device)
        else:
            # FP16 input requires FP16 output
            output = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.float16, device=device)

        # cudagraph capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            # for i in range(test_ite):
            attn.forward(layer, q, k, v, kv_cache, attn_metadata, output)
        # warmup
        for i in range(warming_up):
            # DEBUG
            # print(
            #     f"q: {q.shape}, k: {k.shape}, v: {v.shape} kv_cache: {kv_cache.shape} "
            #     f"attn_metadata: {attn_metadata.num_actual_tokens} output: {output.shape}"
            # )
            # print(f"all attn_metadata: {attn_metadata}")
            g.replay()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(test_ite):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()
        # latency = start_event.elapsed_time(end_event)/(test_ite*test_ite)
        latency = start_event.elapsed_time(end_event) / test_ite

        isl = input_len
        step = 0
        op_name = "context_attention"
        kv_cache_dtype_str = "float16" if not use_fp8_kv_cache else "fp8"
        dtype_str = "float16"

        log_perf(
            item_list=[
                {
                    "batch_size": batch_size,
                    "isl": isl,
                    "num_heads": num_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "head_dim": head_dim,
                    "beam_width": 1,
                    "attn_dtype": dtype_str,
                    "kv_cache_dtype": kv_cache_dtype_str,
                    "step": step,
                    "latency": latency,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source="vllm_flashattention",
            perf_filename=perf_filename,
        )
    else:
        # Generation phase: always use FlashInferImpl
        dtype = torch.float16
        num_tokens = batch_size
        block_size = 64
        q = torch.randn([num_tokens, num_heads, head_dim], dtype=dtype, device=device)
        k = torch.randn([num_tokens, num_key_value_heads, head_dim], dtype=dtype, device=device)
        v = torch.randn([num_tokens, num_key_value_heads, head_dim], dtype=dtype, device=device)
        num_blocks = batch_size
        # Initialize KV cache - note FlashInfer shape requirements
        if use_fp8_kv_cache:
            kv_cache_dtype = torch.float8_e4m3fn  # or torch.float8_e5m2
        else:
            kv_cache_dtype = torch.float16
        kv_cache = torch.zeros(
            (num_blocks, 2, block_size, num_key_value_heads, head_dim),
            dtype=kv_cache_dtype,
            device=device,
        )

        # Initialize metadata - key correction section
        qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        paged_kv_indices = torch.arange(0, batch_size, dtype=torch.int32, device=device)  # 1 block per sequence
        paged_kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)  # last page length = 1
        slot_mapping = torch.arange(0, batch_size, dtype=torch.long, device=device)

        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_tokens,
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            page_size=64,
            data_type=dtype,
            q_data_type=dtype,
            slot_mapping=slot_mapping,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=batch_size,
            num_prefill_tokens=num_tokens,
            use_cascade=False,
        )

        attn = FlashInferImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=1.0 / (head_dim**0.5),
            num_kv_heads=num_key_value_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
            blocksparse_params=None,
            logits_soft_cap=None,
        )

        class DummyLayer(torch.nn.Module):
            def __init__(self, num_heads, num_key_value_heads, head_dim, device):
                super().__init__()
                assert num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads"
                self.num_heads = num_heads
                self.num_key_value_heads = num_key_value_heads

                # Original scale tensors
                self.register_buffer("_q_scale", torch.ones(num_heads, dtype=torch.float32, device=device))
                self.register_buffer("_k_scale", torch.ones(num_key_value_heads, dtype=torch.float32, device=device))
                self.register_buffer("_v_scale", torch.ones(num_key_value_heads, dtype=torch.float32, device=device))

                # FlashInfer required additional interfaces
                self._k_scale_float = self._k_scale
                self._v_scale_float = self._v_scale

        layer = DummyLayer(num_heads, num_key_value_heads, head_dim, device)
        output = torch.empty((num_tokens, num_heads, head_dim), dtype=dtype, device=device)
        # cudagraph capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(test_ite):
                attn.forward(layer, q, k, v, kv_cache, attn_metadata, output)
        # warmup
        for i in range(warming_up):
            # DEBUG
            # print(
            #     f"q: {q.shape}, k: {k.shape}, v: {v.shape} kv_cache: {kv_cache.shape} "
            #     f"attn_metadata: {attn_metadata.num_actual_tokens} output: {output.shape}"
            # )
            # print(f"all attn_metadata: {attn_metadata}")
            g.replay()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(test_ite):
            g.replay()
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event) / (test_ite * test_ite)

        isl = 1
        step = input_len
        op_name = "generation_attention"
        kv_cache_dtype_str = "float16" if not use_fp8_kv_cache else "fp8"
        dtype_str = "float16"

        log_perf(
            item_list=[
                {
                    "batch_size": batch_size,
                    "isl": isl,
                    "num_heads": num_heads,
                    "num_key_value_heads": num_key_value_heads,
                    "head_dim": head_dim,
                    "beam_width": 1,
                    "attn_dtype": dtype_str,
                    "kv_cache_dtype": kv_cache_dtype_str,
                    "step": step,
                    "latency": latency,
                }
            ],
            framework="VLLM",
            version=vllm_version,
            device_name=torch.cuda.get_device_name(device),
            op_name=op_name,
            kernel_source="vllm_flashinfer",
            perf_filename=perf_filename,
        )


def get_context_attention_test_cases(if_unit_test=False):
    has_fp8_kv_cache = get_sm_version() > 86
    test_cases = []

    if not if_unit_test:
        b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        s_list = [
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
        n_list = [4, 8, 12, 16, 24, 32, 40, 48, 64]
        n_kv_list = [0, 1, 2, 4, 8]
        # n_kv_list = [64]
    else:
        b_list = [1]
        s_list = [64]
        n_list = [4]
        n_kv_list = [0]

    # DEBUG
    # print(f"b_list: {b_list}, s_list: {s_list}, n_list: {n_list}, n_kv_list: {n_kv_list}")
    for n in sorted(n_list, reverse=True):
        for s in sorted(s_list, reverse=True):
            for b in sorted(b_list, reverse=True):
                for n_kv in n_kv_list:
                    if n_kv != 0 and (n_kv > n or n % n_kv != 0):
                        continue
                    num_kv_heads = n_kv if n_kv != 0 else n
                    # Only keep self-attention case
                    # if n != num_kv_heads:
                    #    continue
                    if num_kv_heads == n:
                        if b * s > 65536 or b > 128:
                            continue
                    else:
                        if b * s > 131072:
                            continue
                    if b * s * num_kv_heads * 128 * 2 >= 2147483647:
                        continue
                    test_cases.append(
                        [
                            b,
                            s,
                            n,
                            num_kv_heads,
                            128,
                            False,
                            False,
                            True,
                            "context_attention_perf.txt",
                        ]
                    )
                    if has_fp8_kv_cache:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                num_kv_heads,
                                128,
                                True,
                                False,
                                True,
                                "context_attention_perf.txt",
                            ]
                        )
                        # flashattention impl does not support fp8 context fmha
                        # test_cases.append(
                        #     [
                        #         b,
                        #         s,
                        #         n,
                        #         num_kv_heads,
                        #         128,
                        #         True,
                        #         True,
                        #         True,
                        #         "context_attention_perf.txt",
                        #     ]
                        # )

    return test_cases


def get_generation_attention_test_cases():
    has_fp8_kv_cache = get_sm_version() > 86
    test_cases = []

    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # b_list_xqa = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    n_list = [4, 8, 12, 16, 24, 32, 40, 48, 64]
    # n_list_xqa = [4,8,16,32,64,128]
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
    n_kv_list = [1, 2, 4, 8]

    max_bsn = 8192 * 1024
    for n in sorted(n_list, reverse=True):
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
        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            for n_kv in n_kv_list:
                if n_kv > n or n % n_kv != 0:
                    continue
                for s in target_s_list:
                    test_cases.append([b, s, n, n_kv, 128, False, False, False, "generation_attention_perf.txt"])
                    if has_fp8_kv_cache:
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                128,
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
