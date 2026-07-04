# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang dense attention collector.

Builds lightweight RadixAttention/ForwardBatch mocks to benchmark context and
generation attention without launching a full SGLang server. Shared attention
shape intent should live in YAML; this file owns SGLang backend construction,
KV-cache setup, SM-specific skips, and perf logging for the SGLang runtime.
"""

__compat__ = "sglang==0.5.14"

import math
import os
from types import SimpleNamespace
from typing import NamedTuple

import pkg_resources
import torch
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
    get_attention_head_configs,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf

DISABLE_BACKWARD = os.getenv("FLASH_ATTENTION_DISABLE_BACKWARD", "FALSE") == "TRUE"


class Timing(NamedTuple):
    mean: float


# Mock objects to satisfy RadixAttention dependencies
class MockModelConfig:
    def __init__(
        self,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        v_head_dim,
        architecture,
        runtime_window_size,
        attention_chunk_size,
    ):
        self.is_encoder_decoder = False
        self.context_len = 32768
        self.attention_arch = AttentionArch.MHA
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.swa_head_dim = head_dim
        self.swa_v_head_dim = v_head_dim
        self.sliding_window_size = runtime_window_size if runtime_window_size >= 0 else None
        self.attention_chunk_size = attention_chunk_size
        self.is_hybrid_swa = runtime_window_size >= 0 or attention_chunk_size is not None
        self.swa_attention_layer_ids = [0] if self.is_hybrid_swa else []
        self.full_attention_layer_ids = [] if self.is_hybrid_swa else [0]
        self.is_multimodal = False
        self.hidden_size = num_attention_heads * head_dim
        self.is_local_attention_model = attention_chunk_size is not None

        class MockHFConfig:
            def __init__(self, *, num_attention_heads, num_key_value_heads, head_dim, v_head_dim, architecture):
                self.architectures = [architecture or "LlamaForCausalLM"]
                self.num_attention_heads = num_attention_heads
                self.num_key_value_heads = num_key_value_heads
                self.head_dim = head_dim
                self.v_head_dim = v_head_dim
                self.swa_head_dim = head_dim
                self.swa_v_head_dim = v_head_dim
                self.hidden_size = num_attention_heads * head_dim
                self.attn_logit_softcapping = None

        self.hf_config = MockHFConfig(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            architecture=architecture,
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
        self.enable_mis = False
        self.enable_two_batch_overlap = False
        self.disable_attn_tp_gather = False
        self.moe_dense_tp_size = None


class MockModelRunner:
    def __init__(
        self,
        device,
        kv_cache_dtype="auto",
        page_size: int = 64,
        num_heads=None,
        num_kv_heads=None,
        head_dim=None,
        v_head_dim=None,
        architecture=None,
        runtime_window_size=-1,
        attention_chunk_size=None,
    ):
        self.device = device
        self.req_to_token_pool = None
        self.token_to_kv_pool = None
        self.attn_backend = None
        self.server_args = MockServerArgs(page_size=page_size)
        self.attn_cp_size = 1  # Context parallelism size; required by FlashAttentionBackend in sglang >=0.5.10
        self.is_draft_worker = False
        self.model_is_mrope = False
        self.sliding_window_size = attention_chunk_size
        if self.sliding_window_size is None and runtime_window_size >= 0:
            self.sliding_window_size = runtime_window_size
        self.attention_chunk_size = attention_chunk_size
        # Initialized after the KV pool is created.
        self.token_to_kv_pool_allocator = None
        self.model_config = MockModelConfig(
            num_heads,
            num_kv_heads,
            head_dim,
            v_head_dim,
            architecture,
            runtime_window_size,
            attention_chunk_size,
        )
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
        self.linear_attn_model_spec = None


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


def get_context_attention_test_cases():
    test_cases = []

    # FP8 attention requires SM90+ (Hopper)
    sm_version = get_sm_version()
    skip_fp8 = sm_version < 90

    for shape_sweep in get_attention_context_shape_sweeps("sglang"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep["max_batch_size_self_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for head_config in get_attention_head_configs(
            shape_sweep,
            phase="context",
            backend="sglang",
            sm_version=sm_version,
        ):
            n = head_config.num_heads
            num_kv_heads = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            for s in sorted(sequence_lengths, reverse=True):
                for b in sorted(batch_sizes, reverse=True):
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

                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        use_fp8_context_fmha = bool(precision_case["fp8_context_fmha"])
                        if skip_fp8 and use_fp8_kv_cache:
                            continue
                        # Blackwell accepts an FP8 KV pool but plans live Q/K/V
                        # from BF16; do not persist a fake FP8-attn label.
                        if sm_version in {100, 103, 120} and use_fp8_context_fmha:
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
                                head_config.v_head_dim,
                                head_config.runtime_window_size,
                                head_config.attention_chunk_size,
                                head_config.has_attention_sink,
                                head_config.scaling,
                                head_config.kernel_source,
                                head_config.architecture,
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
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])
        for head_config in get_attention_head_configs(
            shape_sweep,
            phase="generation",
            backend="sglang",
            sm_version=sm_version,
        ):
            n = head_config.num_heads
            n_kv = head_config.num_kv_heads
            head_dim = head_config.head_dim
            window_size = head_config.window_size
            max_tokens = (
                int(shape_sweep["max_mha_tokens_per_step"])
                if n == n_kv
                else int(shape_sweep["max_xqa_tokens_per_step"])
            )
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                head_dim,
                max_tokens,
                shape_sweep,
            )
            for b, s_list_limited in b_s_dict.items():
                target_s_list = sorted(s_list_limited)
                if b >= min_drop_batch:
                    target_s_list = target_s_list[:-1]
                for s in target_s_list:
                    for precision_case in shape_sweep["precision_cases"]:
                        use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                        if skip_fp8 and use_fp8_kv_cache:
                            continue
                        test_cases.append(
                            [
                                b,
                                s,
                                n,
                                n_kv,
                                head_dim,
                                use_fp8_kv_cache,
                                False,
                                False,
                                window_size,
                                head_config.v_head_dim,
                                head_config.runtime_window_size,
                                head_config.attention_chunk_size,
                                head_config.has_attention_sink,
                                head_config.scaling,
                                head_config.kernel_source,
                                head_config.architecture,
                            ]
                        )
    return test_cases


@get_parallel().override(
    attn_tp_size=1,
    attn_tp_rank=0,
    attn_cp_size=1,
    attn_cp_rank=0,
    attn_dp_size=1,
    attn_dp_rank=0,
)
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
    v_head_dim=None,
    runtime_window_size=None,
    attention_chunk_size=None,
    has_attention_sink=False,
    scaling=None,
    attn_backend_name=None,
    architecture=None,
    *,
    perf_filename,
    device="cuda:0",
    page_size: int | None = None,
):
    if use_fp8_context_fmha:
        assert use_fp8_kv_cache, "If you want to use fp8 context fmha, kv cache must be fp8"
    kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16

    torch_device = torch.device(device)
    device_str = str(torch_device)

    explicit_v_head_dim = v_head_dim is not None
    if head_dim == 192 and (
        not explicit_v_head_dim or architecture != "MiMoV2FlashForCausalLM" or attn_backend_name is None
    ):
        raise ValueError("head_dim=192 requires the explicit MiMo-V2 SGLang attention contract")

    v_head_dim = head_dim if v_head_dim is None else v_head_dim
    if runtime_window_size is None:
        runtime_window_size = window_size or -1

    if attn_backend_name is None:
        sm_version = get_sm_version()
        attn_backend_name = {
            80: "triton",
            86: "triton",
            89: "triton",
            90: "fa3",
            100: "trtllm_mha",
            103: "trtllm_mha",
            120: "flashinfer",
        }.get(sm_version)
        if attn_backend_name is None:
            raise ValueError(f"No SGLang 0.5.14 attention backend mapping for SM{sm_version}")
    if page_size is None:
        page_size = 64 if attn_backend_name == "trtllm_mha" else 1

    model_runner = MockModelRunner(
        torch_device,
        kv_cache_dtype="fp8_e4m3" if use_fp8_kv_cache else "auto",
        page_size=page_size,
        num_heads=num_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        architecture=architecture,
        runtime_window_size=runtime_window_size,
        attention_chunk_size=attention_chunk_size,
    )
    model_runner.kv_cache_dtype = kvtype

    total_len = input_len if is_context_phase else input_len + 1
    # TRTLLM MHA sizes its page table from context_len.
    model_runner.model_config.context_len = max(model_runner.model_config.context_len, total_len)
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
        v_head_dim=v_head_dim,
        layer_num=1,
        device=device_str,
        enable_memory_saver=False,
    )
    model_runner.token_to_kv_pool = kv_pool
    model_runner.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=model_runner.page_size,
        get_kvcache=lambda: model_runner.token_to_kv_pool,
    )

    if attn_backend_name == "flashinfer":
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        attn_backend = FlashInferAttnBackend(model_runner)
    elif attn_backend_name == "trtllm_mha":
        from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

        attn_backend = TRTLLMHAAttnBackend(model_runner)
    elif attn_backend_name == "triton":
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        attn_backend = TritonAttnBackend(model_runner)
    elif attn_backend_name == "fa3":
        attn_backend = FlashAttentionBackend(model_runner)
    else:
        raise ValueError(f"Unsupported SGLang attention backend: {attn_backend_name}")

    model_runner.attn_backend = attn_backend

    layer = RadixAttention(
        num_heads=num_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5 if scaling is None else scaling,
        num_kv_heads=num_key_value_heads,
        layer_id=0,
        v_head_dim=v_head_dim,
        sliding_window_size=runtime_window_size,
        use_irope=attention_chunk_size is not None and window_size > 0,
    ).to(torch_device)

    sinks = None
    if has_attention_sink:
        sinks_dtype = torch.float32 if attn_backend_name == "trtllm_mha" else torch.bfloat16
        sinks = torch.randn(num_heads, device=torch_device, dtype=sinks_dtype)

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
            v_head_dim,
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
            v_head_dim,
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
                v_head_dim,
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

    with forward_context(ForwardContext(attn_backend=attn_backend)):
        attn_backend.init_forward_metadata(forward_batch)

        # FP8 KV cache controls storage. TRTLLM MHA takes BF16 live Q/K/V
        # for both logical FMHA modes and converts inside the backend.
        if is_context_phase and use_fp8_context_fmha and attn_backend_name != "trtllm_mha":
            if attn_backend_name == "flashinfer":
                raise ValueError("SGLang 0.5.14 FlashInfer has no FP8 live-activation prefill path")
            q = q.to(kvtype)
            k = k.to(kvtype)
            v = v.to(kvtype)

        def run_iter():
            layer(q, k, v, forward_batch, sinks=sinks)

        with benchmark_with_power(
            device=torch_device,
            kernel_func=run_iter,
            num_warmups=3,
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
