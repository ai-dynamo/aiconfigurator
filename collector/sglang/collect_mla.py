# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
import random

import pkg_resources
import torch
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from helper import benchmark_with_power, get_sm_version, log_perf

compatible_version = ["0.5.5.post3", "0.5.6.post2"]

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
MLA_PAGE_SIZE = 64
MLA_SCALING = 1 / math.sqrt(QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)
INT32_MAX = 2**31 - 1
# Largest kv slot index we can safely touch before the old flashmla kernels overflow
MAX_KV_LOC = (INT32_MAX // (KV_LORA_RANK + QK_ROPE_HEAD_DIM)) - MLA_PAGE_SIZE


class MockModelConfig:
    def __init__(self, context_len: int = 32768):
        self.is_encoder_decoder = False
        self.context_len = context_len
        self.attention_arch = AttentionArch.MLA
        self.is_hybrid = False
        self.attention_chunk_size = None
        # Provide compatibility with newer sglang versions that expect hybrid-SWA metadata
        self.is_hybrid_swa = None
        self.swa_attention_layer_ids = None
        self.full_attention_layer_ids = None


class MockServerArgs:
    def __init__(self, kv_cache_dtype: torch.dtype, page_size: int):
        self.enable_lora = False
        self.enable_deterministic_inference = False
        self.kv_cache_dtype = "fp8" if kv_cache_dtype == torch.float8_e4m3fn else "float16"
        self.speculative_eagle_topk = 0
        self.speculative_num_draft_tokens = 0
        self.speculative_attention_mode = "prefill"
        self.prefill_attention_backend = "fa3"
        self.decode_attention_backend = "fa3"
        self.page_size = page_size
        self.device = "cuda"


class MockModelRunner:
    def __init__(self, device: torch.device, kv_cache_dtype: torch.dtype, page_size: int):
        self.device = device
        self.kv_cache_dtype = kv_cache_dtype
        self.page_size = page_size
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.sliding_window_size = None
        self.is_hybrid = False
        self.model_config = MockModelConfig()
        # Keep attribute for compatibility across sglang versions (older code ignores it)
        self.is_hybrid_swa = self.model_config.is_hybrid_swa
        self.server_args = MockServerArgs(kv_cache_dtype, page_size)


def create_req_to_token_pool(
    batch_size: int,
    total_len: int,
    page_size: int,
    torch_device: torch.device,
    device_str: str,
) -> tuple[ReqToTokenPool, torch.Tensor]:
    pool = ReqToTokenPool(
        size=batch_size,
        max_context_len=total_len,
        device=device_str,
        enable_memory_saver=False,
    )
    req_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device).view(batch_size, 1)
    token_offsets = torch.arange(total_len, dtype=torch.int32, device=torch_device).view(1, total_len)
    token_matrix = (req_indices * total_len) + token_offsets + page_size
    pool.req_to_token[:batch_size, :total_len] = token_matrix
    return pool, token_matrix.contiguous()


def benchmark_layer(layer, forward_batch, q, k, v, q_rope, k_rope):
    # Use benchmark_with_power context manager
    device = q.device

    def kernel_func():
        layer(q, k, v, forward_batch, q_rope=q_rope, k_rope=k_rope)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass

    return results["latency_ms"], results["power_stats"]


def get_context_mla_test_cases():
    # MLA requires SM90+ (Hopper) due to asymmetric head dimensions
    # (Q/K headdim != V headdim requires Hopper-specific FlashAttention kernels)
    sm_version = get_sm_version()
    if sm_version < 90:
        return []

    dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
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
    ]
    for n in n_list:
        for b in b_list:
            for s in s_list:
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64]:
                        if b * s > 32768:
                            continue
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


def get_generation_mla_test_cases():
    # MLA requires SM90+ (Hopper) due to asymmetric head dimensions
    # (Q/K headdim != V headdim requires Hopper-specific FlashAttention kernels)
    sm_version = get_sm_version()
    if sm_version < 90:
        return []

    dtype_list = [torch.bfloat16, torch.float8_e4m3fn]
    test_cases = []
    n_list = [64, 128]
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
            ]:
                for dtype in dtype_list:
                    for tp_size in [1, 2, 4, 8, 16, 32, 64]:
                        if b * s > 1024 * 4096 * 4:
                            continue
                        total_len = s
                        # Guard against hitting int32 limits in the legacy flashmla kernel path.
                        if (b * total_len) + MLA_PAGE_SIZE > MAX_KV_LOC:
                            continue
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
    torch.cuda.set_device(device)
    torch_device = torch.device(device)
    random.seed(0)
    torch.manual_seed(0)
    del world_size, tokens_per_block, warming_up, test_ite, output_len

    assert kv_cache_dtype in [torch.bfloat16, torch.float8_e4m3fn], "Unsupported kv cache dtype"
    assert num_heads % tp_size == 0, "num_heads must be divisible by tp_size"
    local_num_heads = num_heads // tp_size

    model_runner = MockModelRunner(torch_device, kv_cache_dtype, MLA_PAGE_SIZE)
    total_len = input_len if is_context_phase else input_len + 1
    req_to_token_pool, token_matrix = create_req_to_token_pool(
        batch_size=batch_size,
        total_len=total_len,
        page_size=MLA_PAGE_SIZE,
        torch_device=torch_device,
        device_str=str(torch_device),
    )
    model_runner.req_to_token_pool = req_to_token_pool

    total_tokens = max(1, batch_size * total_len)
    kv_cache_size = max(MLA_PAGE_SIZE, math.ceil(total_tokens / MLA_PAGE_SIZE) * MLA_PAGE_SIZE)
    kv_pool = MLATokenToKVPool(
        size=kv_cache_size,
        page_size=MLA_PAGE_SIZE,
        dtype=kv_cache_dtype,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=1,
        device=str(torch_device),
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool

    attn_backend = FlashAttentionBackend(model_runner)
    layer = RadixAttention(
        num_heads=local_num_heads,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        scaling=MLA_SCALING,
        num_kv_heads=1,
        layer_id=0,
        v_head_dim=KV_LORA_RANK,
    ).to(torch_device)

    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device)

    if is_context_phase:
        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=torch_device)
        prefix_lens = torch.zeros_like(seq_lens)
        q_shape = (batch_size * input_len, local_num_heads, KV_LORA_RANK)
        q = torch.randn(q_shape, device=torch_device, dtype=torch.bfloat16)
        q_rope = torch.randn(
            batch_size * input_len,
            local_num_heads,
            QK_ROPE_HEAD_DIM,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        k_shape = (batch_size * input_len, 1, KV_LORA_RANK)
        k = torch.randn(k_shape, device=torch_device, dtype=torch.bfloat16)
        k_rope = torch.randn(
            batch_size * input_len,
            1,
            QK_ROPE_HEAD_DIM,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        v = k  # MLA stores latent cache, so v matches k_nope

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, input_len, dtype=torch.long, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix.reshape(-1).to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
            extend_seq_lens=seq_lens,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=seq_lens.cpu(),
            extend_prefix_lens_cpu=prefix_lens.cpu(),
            extend_num_tokens=int(seq_lens.sum().item()),
        )
    else:
        history_len = input_len
        seq_lens = torch.full((batch_size,), history_len + 1, dtype=torch.int32, device=torch_device)
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, 1, dtype=torch.long, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=token_matrix[:, history_len:].reshape(-1).to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
        )

        if history_len > 0:
            history_loc = token_matrix[:, :history_len].reshape(-1).contiguous()
            cache_k = torch.randn(
                history_loc.numel(),
                1,
                KV_LORA_RANK,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            cache_k_rope = torch.randn(
                history_loc.numel(),
                1,
                QK_ROPE_HEAD_DIM,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            kv_pool.set_mla_kv_buffer(
                layer,
                history_loc.to(torch.int64),
                cache_k,
                cache_k_rope,
            )

        q = torch.randn(batch_size, local_num_heads, KV_LORA_RANK, device=torch_device, dtype=torch.bfloat16)
        q_rope = torch.randn(batch_size, local_num_heads, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.bfloat16)
        k = torch.randn(batch_size, 1, KV_LORA_RANK, device=torch_device, dtype=torch.bfloat16)
        k_rope = torch.randn(batch_size, 1, QK_ROPE_HEAD_DIM, device=torch_device, dtype=torch.bfloat16)
        v = k
        q = q.view(batch_size * 1, local_num_heads, KV_LORA_RANK)
        q_rope = q_rope.view(batch_size * 1, local_num_heads, QK_ROPE_HEAD_DIM)

    forward_batch.req_to_token_pool = req_to_token_pool
    forward_batch.token_to_kv_pool = kv_pool
    forward_batch.attn_backend = attn_backend
    attn_backend.init_forward_metadata(forward_batch)

    latency, power_stats = benchmark_layer(layer, forward_batch, q, k, v, q_rope, k_rope)

    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    str_type = "float16" if kv_cache_dtype == torch.bfloat16 else "fp8"
    log_perf(
        item_list=[
            {
                "mla_dtype": "float16",
                "kv_cache_dtype": str_type,
                "num_heads": local_num_heads,
                "batch_size": batch_size,
                "isl": isl,
                "tp_size": tp_size,
                "step": step,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"mla_{'context' if is_context_phase else 'generation'}",
        kernel_source="flash_attention",
        perf_filename=perf_filename,
        power_stats=power_stats,
    )
