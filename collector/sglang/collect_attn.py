# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import NamedTuple

import pkg_resources
import torch
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from helper import benchmark_with_power, get_sm_version, log_perf

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"

compatible_version = ["0.5.5.post3", "0.5.6.post2"]


class Timing(NamedTuple):
    mean: float


# Mock objects to satisfy RadixAttention dependencies
class MockModelConfig:
    def __init__(self):
        self.is_encoder_decoder = False
        self.context_len = 32768
        self.attention_arch = AttentionArch.MHA
        # Align with newer sglang ModelConfig while remaining harmless on older versions
        self.is_hybrid_swa = None
        self.swa_attention_layer_ids = None
        self.full_attention_layer_ids = None


class MockServerArgs:
    def __init__(self, page_size: int):
        self.enable_lora = False
        self.enable_deterministic_inference = False
        self.kv_cache_dtype = "auto"
        self.speculative_eagle_topk = 0
        self.speculative_num_draft_tokens = 0
        self.page_size = page_size


class MockModelRunner:
    def __init__(self, device, kv_cache_dtype="auto", page_size: int = 64):
        self.device = device
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.server_args = MockServerArgs(page_size=page_size)
        self.is_draft_worker = False
        self.model_is_mrope = False
        self.sliding_window_size = None
        self.attention_chunk_size = None
        self.model_config = MockModelConfig()
        self.kv_cache_dtype = kv_cache_dtype  # Default
        self.page_size = page_size
        self.is_hybrid = False
        # Provide compatibility across sglang versions that expect this flag
        self.is_hybrid_swa = self.model_config.is_hybrid_swa
        self.server_args.kv_cache_dtype = kv_cache_dtype
        self.server_args.page_size = page_size


def create_req_to_token_pool(batch_size, total_len, page_size, torch_device, device_str):
    """Create req_to_token mapping consistent with test_flashattn_backend.py."""
    assert total_len > 0, "Total sequence length must be positive"
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


def get_context_attention_test_cases():
    test_cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 262144]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96]
    n_kv_list = [0, 1, 2, 4, 8]

    # FP8 attention requires SM90+ (Hopper)
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 90

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

                    # BF16 attention - works on all GPUs
                    test_cases.append([b, s, n, num_kv_heads, 128, False, False, True, "context_attention_perf.txt"])

                    # FP8 attention - requires SM90+ (Hopper)
                    if not skip_fp8:
                        test_cases.append([b, s, n, num_kv_heads, 128, True, False, True, "context_attention_perf.txt"])
                        test_cases.append([b, s, n, num_kv_heads, 128, True, True, True, "context_attention_perf.txt"])

    return test_cases


def get_generation_attention_test_cases():
    test_cases = []

    # FP8 attention requires SM90+ (Hopper)
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 90

    # generation
    b_list = [1, 2, 4, 64, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    s_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    n_list = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 64]
    n_list_xqa = [1, 2, 4, 8, 16, 32, 64, 96, 128]
    n_kv_list = [1, 2, 4, 8]

    # MHA
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
            for s in target_s_list:
                # BF16 attention - works on all GPUs
                test_cases.append([b, s, n, n, 128, False, False, False, "generation_attention_perf.txt"])
                # FP8 attention - requires SM90+ (Hopper)
                if not skip_fp8:
                    test_cases.append([b, s, n, n, 128, True, False, False, "generation_attention_perf.txt"])

    # XQA
    max_bsn = 8192 * 1024 * 2
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

        for b, s_list_limited in b_s_dict.items():
            target_s_list = sorted(s_list_limited)
            if b >= 256:
                target_s_list = target_s_list[:-1]
            for n_kv in n_kv_list:
                if n_kv >= n:
                    continue
                for s in target_s_list:
                    # BF16 attention - works on all GPUs
                    test_cases.append([b, s, n, n_kv, 128, False, False, False, "generation_attention_perf.txt"])
                    # FP8 attention - requires SM90+ (Hopper)
                    if not skip_fp8:
                        test_cases.append([b, s, n, n_kv, 128, True, False, False, "generation_attention_perf.txt"])
    return test_cases


def run_attention_torch(
    batch_size,
    input_len,
    num_heads,
    num_key_value_heads,
    head_dim,
    use_fp8_kv_cache,
    use_fp8_context_fmha,
    is_context_phase,
    perf_filename,
    device="cuda:0",
    *,
    page_size: int = 64,
):
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache, "If you want to use fp8 context fmha, kv cache must be fp8"
    kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16

    torch_device = torch.device(device)
    device_str = str(torch_device)
    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype="fp8" if use_fp8_kv_cache else "auto",
        page_size=page_size,
    )
    model_runner.kv_cache_dtype = kvtype

    total_len = input_len if is_context_phase else input_len + 1
    req_to_token_pool, token_matrix = create_req_to_token_pool(
        batch_size=batch_size,
        total_len=total_len,
        page_size=model_runner.page_size,
        torch_device=torch_device,
        device_str=device_str,
    )
    model_runner.req_to_token_pool = req_to_token_pool

    total_tokens = batch_size * total_len
    kv_cache_size = max(
        model_runner.page_size,
        math.ceil(total_tokens / model_runner.page_size) * model_runner.page_size,
    )
    kv_pool = MHATokenToKVPool(
        size=kv_cache_size,
        page_size=model_runner.page_size,
        dtype=kvtype,
        head_num=num_key_value_heads,
        head_dim=head_dim,
        layer_num=1,
        device=device_str,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool

    attn_backend = FlashAttentionBackend(model_runner)
    model_runner.attn_backend = attn_backend

    layer = RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        num_kv_heads=num_key_value_heads,
        layer_id=0,
    ).to(torch_device)

    seqlen_q = input_len if is_context_phase else 1
    q = torch.randn(
        batch_size * seqlen_q,
        num_heads,
        head_dim,
        device=torch_device,
        dtype=torch.bfloat16,
    )

    req_pool_indices = torch.arange(batch_size, dtype=torch.int32, device=torch_device)

    if is_context_phase:
        forward_mode = ForwardMode.EXTEND
        k = torch.randn(
            batch_size * input_len,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch_size * input_len,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )

        seq_lens = torch.full((batch_size,), input_len, dtype=torch.int32, device=torch_device)
        seq_lens_cpu = seq_lens.cpu()
        prefix_lens = torch.zeros((batch_size,), dtype=torch.int32, device=torch_device)
        out_cache_loc = token_matrix.reshape(-1).to(torch.int32)

        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, input_len, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens_cpu,
            extend_seq_lens=seq_lens,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=seq_lens_cpu,
            extend_prefix_lens_cpu=prefix_lens.cpu(),
            extend_num_tokens=int(seq_lens.sum().item()),
            attn_backend=attn_backend,
        )
    else:
        forward_mode = ForwardMode.DECODE
        history_len = input_len
        new_token_loc = token_matrix[:, history_len:].reshape(-1).contiguous()
        history_loc = token_matrix[:, :history_len].reshape(-1).contiguous() if history_len > 0 else None

        k = torch.randn(
            batch_size,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            batch_size,
            num_key_value_heads,
            head_dim,
            device=torch_device,
            dtype=torch.bfloat16,
        )

        seq_lens = torch.full((batch_size,), total_len, dtype=torch.int32, device=torch_device)
        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=batch_size,
            input_ids=torch.zeros(batch_size, 1, dtype=torch.int64, device=torch_device),
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=new_token_loc.to(torch.int32),
            seq_lens_sum=int(seq_lens.sum().item()),
            seq_lens_cpu=seq_lens.cpu(),
            attn_backend=attn_backend,
        )

        if history_loc is not None and history_loc.numel() > 0:
            cache_k = torch.randn(
                history_loc.numel(),
                num_key_value_heads,
                head_dim,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            cache_v = torch.randn(
                history_loc.numel(),
                num_key_value_heads,
                head_dim,
                device=torch_device,
                dtype=torch.bfloat16,
            )
            cache_k = cache_k.to(kvtype)
            cache_v = cache_v.to(kvtype)
            kv_pool.set_kv_buffer(
                layer,
                history_loc.to(torch.int64),
                cache_k,
                cache_v,
                layer.k_scale,
                layer.v_scale,
            )

    forward_batch.req_to_token_pool = req_to_token_pool
    forward_batch.token_to_kv_pool = kv_pool

    attn_backend.init_forward_metadata(forward_batch)

    if use_fp8_context_fmha or use_fp8_kv_cache:
        q = q.to(kvtype)
        k = k.to(kvtype)
        v = v.to(kvtype)

    def run_iter():
        layer(q, k, v, forward_batch)

    warmup = 3
    # Use benchmark_with_power context manager
    with benchmark_with_power(
        device=torch_device,
        kernel_func=run_iter,
        num_warmups=warmup,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    if is_context_phase:
        isl = input_len
        step = 0
        op_name = "context_attention"
    else:
        isl = 1
        step = input_len
        op_name = "generation_attention"

    log_perf(
        item_list=[
            {
                "batch_size": batch_size,
                "isl": isl,
                "num_heads": num_heads,
                "num_key_value_heads": num_key_value_heads,
                "head_dim": head_dim,
                "beam_width": 1,
                "attn_dtype": "fp8" if use_fp8_context_fmha else "float16",
                "kv_cache_dtype": "fp8" if use_fp8_kv_cache else "float16",
                "step": step,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source="flash_attention",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    return Timing(latency * 1e-3)
