# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang dense attention collector.

Builds lightweight RadixAttention/ForwardBatch mocks to benchmark context and
generation attention without launching a full SGLang server. Shared attention
shape intent should live in YAML; this file owns SGLang backend construction,
KV-cache setup, SM-specific skips, and perf logging for the SGLang runtime.
"""

__compat__ = "sglang>=0.5.10rc0"

import math
import os
from typing import NamedTuple

import pkg_resources
import sglang.srt.layers.dp_attention as dp_attention
import torch
from sglang.srt.configs.model_config import AttentionArch

# Initialize DP attention variables globally for FlashInfer backend compatibility
dp_attention.get_attention_tp_size = lambda: 1
dp_attention.get_attention_tp_rank = lambda: 0
dp_attention.get_attention_dp_size = lambda: 1
dp_attention.get_attention_dp_rank = lambda: 0
dp_attention.get_local_attention_dp_size = lambda: 1
dp_attention.get_local_attention_dp_rank = lambda: 0
dp_attention.is_dp_attention_enabled = lambda: False
dp_attention._ENABLE_DP_ATTENTION_FLAG = False

# Also set the private variables if they exist
if hasattr(dp_attention, "_ATTN_TP_SIZE"):
    dp_attention._ATTN_TP_SIZE = 1
    dp_attention._ATTN_TP_RANK = 0
dp_attention._ATTN_DP_SIZE = 1
dp_attention._ATTN_DP_RANK = 0
dp_attention._LOCAL_ATTN_DP_SIZE = 1
dp_attention._LOCAL_ATTN_DP_RANK = 0

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from collector.case_generator import get_attention_context_shape_sweeps, get_attention_generation_shape_sweeps
from collector.helper import benchmark_with_power, get_sm_version, log_perf

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


class Timing(NamedTuple):
    mean: float


# Mock objects to satisfy RadixAttention dependencies
class MockModelConfig:
    def __init__(self, num_attention_heads, num_key_value_heads, head_dim):
        self.is_encoder_decoder = False
        self.context_len = 32768
        self.attention_arch = AttentionArch.MHA
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        # Align with newer sglang ModelConfig while remaining harmless on older versions
        self.is_hybrid_swa = False
        self.swa_attention_layer_ids = []
        self.full_attention_layer_ids = []
        self.is_multimodal = False
        self.hidden_size = num_attention_heads * head_dim
        self.is_local_attention_model = False

        class MockHFConfig:
            def __init__(self, *, num_attention_heads, num_key_value_heads, head_dim):
                self.architectures = ["LlamaForCausalLM"]
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                self.head_dim = head_dim
                self.hidden_size = num_attention_heads * head_dim
                self.attn_logit_softcapping = None

        self.hf_config = MockHFConfig(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        self.hf_text_config = self.hf_config
        self.dtype = torch.bfloat16

    def get_num_kv_heads(self, tp_size):
        return self.num_key_value_heads // tp_size


class MockServerArgs:
    def __init__(self, page_size: int):
        self.enable_lora = False
        self.enable_deterministic_inference = False
        self.kv_cache_dtype = "auto"
        self.speculative_eagle_topk = 0
        self.speculative_num_draft_tokens = 0
        self.speculative_num_steps = None
        self.page_size = page_size
        self.multi_item_scoring_delimiter = None
        self.dllm_algorithm = None
        self.dllm_algorithm_config = None
        self.enable_piecewise_cuda_graph = False  # sglang <=0.5.9
        self.disable_piecewise_cuda_graph = True  # sglang >=0.5.10
        self.is_embedding = False
        self.disable_radix_cache = False
        self.enable_dp_attention = False
        self.model_path = None
        self.revision = None
        # Required by TritonAttnBackend
        self.triton_attention_num_kv_splits = 8
        self.triton_attention_split_tile_size = None
        self.disable_cuda_graph = False
        self.chunked_prefill_size = -1


class MockModelRunner:
    def __init__(
        self,
        device,
        kv_cache_dtype="auto",
        page_size: int = 64,
        num_heads=None,
        num_kv_heads=None,
        head_dim=None,
    ):
        self.device = device
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.server_args = MockServerArgs(page_size=page_size)
        self.attn_cp_size = 1  # Context parallelism size; required by FlashAttentionBackend in sglang >=0.5.10
        self.is_draft_worker = False
        self.model_is_mrope = False
        self.sliding_window_size = 0
        self.attention_chunk_size = None
        # FlashInferIndicesUpdaterPrefill expects an allocator; keep None for mock.
        self.token_to_kv_pool_allocator = None
        self.model_config = MockModelConfig(num_heads, num_kv_heads, head_dim)
        self.kv_cache_dtype = kv_cache_dtype  # Default
        self.page_size = page_size
        self.tp_size = 1
        self.is_hybrid = False
        self.dtype = torch.bfloat16
        # Provide compatibility across sglang versions that expect this flag
        self.is_hybrid_swa = self.model_config.is_hybrid_swa
        self.server_args.kv_cache_dtype = kv_cache_dtype
        self.server_args.page_size = page_size
        # Required by TritonAttnBackend
        self.gpu_id = 0
        self.hybrid_gdn_config = None
        self.kimi_linear_config = None


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


def _int_list(values):
    return [int(value) for value in values]


def _kv_head_options(values, num_heads):
    return [num_heads if value == "self" else int(value) for value in values]


def get_context_attention_test_cases():
    test_cases = []

    # FP8 attention requires SM90+ (Hopper)
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 90

    for shape_sweep in get_attention_context_shape_sweeps("sglang"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        query_head_counts = _int_list(shape_sweep["query_head_counts"])
        kv_head_options = shape_sweep["kv_head_options"]
        head_dims = _int_list(shape_sweep["head_dims"])
        window_sizes = _int_list(shape_sweep.get("window_sizes", [0]))
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep["max_batch_size_self_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_dim in head_dims:
            for n in sorted(query_head_counts, reverse=True):
                for s in sorted(sequence_lengths, reverse=True):
                    for b in sorted(batch_sizes, reverse=True):
                        for num_kv_heads in _kv_head_options(kv_head_options, n):
                            if num_kv_heads != n and (num_kv_heads >= n or n % num_kv_heads != 0):
                                continue

                            if num_kv_heads == n:
                                if b * s > max_tokens_self_attention or b > max_batch_size_self_attention:
                                    continue
                            else:
                                if b * s > max_tokens_grouped_query_attention:
                                    continue
                            if b * s * num_kv_heads * head_dim * 2 >= max_kv_elements:
                                continue
                            # SGLang's SM120 Triton context attention path uses
                            # 32-bit indexing for large Q/O tensors.  Shapes at or
                            # above this element boundary crash with an illegal
                            # memory access and poison the worker CUDA context.
                            if sm_version >= 120 and b * s * n * head_dim >= max_kv_elements:
                                continue

                            for window_size in window_sizes:
                                for precision_case in shape_sweep["precision_cases"]:
                                    use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                    use_fp8_context_fmha = bool(precision_case["fp8_context_fmha"])
                                    if skip_fp8 and use_fp8_kv_cache:
                                        continue
                                    test_cases.append(
                                        [
                                            b,
                                            s,
                                            n,
                                            num_kv_heads,
                                            head_dim,
                                            use_fp8_kv_cache,
                                            use_fp8_context_fmha,
                                            True,
                                            window_size,
                                        ]
                                    )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, head_dim, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens // s // num_heads * 128 // head_dim
        for b in sorted(batch_sizes):
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

    # FP8 attention requires SM90+ (Hopper)
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 90

    for shape_sweep in get_attention_generation_shape_sweeps("sglang"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        head_dims = _int_list(shape_sweep["head_dims"])
        window_sizes = _int_list(shape_sweep.get("window_sizes", [0]))
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])

        for head_dim in head_dims:
            for n in sorted(_int_list(shape_sweep["mha_query_head_counts"]), reverse=True):
                b_s_dict = _generation_target_sequence_lengths(
                    batch_sizes,
                    sequence_lengths,
                    n,
                    head_dim,
                    int(shape_sweep["max_mha_tokens_per_step"]),
                    shape_sweep,
                )

                for b, s_list_limited in b_s_dict.items():
                    target_s_list = sorted(s_list_limited)
                    if b >= min_drop_batch:
                        target_s_list = target_s_list[:-1]
                    for s in target_s_list:
                        for window_size in window_sizes:
                            for precision_case in shape_sweep["precision_cases"]:
                                use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                if skip_fp8 and use_fp8_kv_cache:
                                    continue
                                test_cases.append([b, s, n, n, head_dim, use_fp8_kv_cache, False, False, window_size])

        for head_dim in head_dims:
            for n in sorted(_int_list(shape_sweep["xqa_query_head_counts"]), reverse=True):
                b_s_dict = _generation_target_sequence_lengths(
                    batch_sizes,
                    sequence_lengths,
                    n,
                    head_dim,
                    int(shape_sweep["max_xqa_tokens_per_step"]),
                    shape_sweep,
                )

                for b, s_list_limited in b_s_dict.items():
                    target_s_list = sorted(s_list_limited)
                    if b >= min_drop_batch:
                        target_s_list = target_s_list[:-1]
                    for n_kv in _int_list(shape_sweep["kv_head_counts"]):
                        if n_kv >= n:
                            continue
                        for s in target_s_list:
                            for window_size in window_sizes:
                                for precision_case in shape_sweep["precision_cases"]:
                                    use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                    if skip_fp8 and use_fp8_kv_cache:
                                        continue
                                    test_cases.append(
                                        [b, s, n, n_kv, head_dim, use_fp8_kv_cache, False, False, window_size]
                                    )
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
    window_size=0,
    *,
    perf_filename,
    device="cuda:0",
    page_size: int = 64,
):
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache, "If you want to use fp8 context fmha, kv cache must be fp8"
    kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16

    torch_device = torch.device(device)
    device_str = str(torch_device)

    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype="fp8_e4m3" if use_fp8_kv_cache else "auto",
        page_size=page_size,
        num_heads=num_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
    )
    model_runner.kv_cache_dtype = kvtype
    model_runner.sliding_window_size = window_size

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

    sm_version = get_sm_version()
    use_triton_attention = sm_version >= 110 or (sm_version >= 100 and head_dim == 192)
    if use_triton_attention:
        # SM120+ (workstation Blackwell): TRTLLM prefill (TllmGenFmhaRunner) is unsupported;
        # FA3 is not compiled for SM120. B200/SM100 TRTLLM also lacks head-dim 192
        # kernels, which MiMo-style models need. Use Triton JIT-compiled backend instead.
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        attn_backend = TritonAttnBackend(model_runner)
        attn_backend_name = "triton"
    elif sm_version >= 100:
        try:
            from sglang.srt.layers.attention.trtllm_mha_backend import (
                TRTLLMHAAttnBackend,
            )

            attn_backend = TRTLLMHAAttnBackend(model_runner)
            attn_backend_name = "trtllm_mha"
        except ImportError:
            attn_backend = FlashAttentionBackend(model_runner)
            attn_backend_name = "flash_attention"
    else:
        attn_backend = FlashAttentionBackend(model_runner)
        attn_backend_name = "flash_attention"

    model_runner.attn_backend = attn_backend

    layer = RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
        num_kv_heads=num_key_value_heads,
        layer_id=0,
    ).to(torch_device)
    layer.sliding_window_size = window_size

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

    # FP8 KV cache controls cache storage.  Live q/k/v activations remain
    # BF16 in the normal decode path; only explicit FP8 context FMHA casts them.
    if is_context_phase and use_fp8_context_fmha:
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
                "attn_dtype": "fp8" if use_fp8_context_fmha else "bfloat16",
                "kv_cache_dtype": "fp8" if use_fp8_kv_cache else "bfloat16",
                "step": step,
                "window_size": window_size,
                "latency": latency,
            }
        ],
        framework="SGLang",
        version=pkg_resources.get_distribution("sglang").version,
        device_name=torch.cuda.get_device_name(device),
        op_name=op_name,
        kernel_source=attn_backend_name,
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    return Timing(latency * 1e-3)
