# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM dense attention collector.

Constructs a single TRT-LLM torch-flow attention layer and synthetic metadata to
benchmark context and generation attention. This file owns TRT-LLM cache manager
setup, quantization flags, SM/version-specific skips, and perf-row formatting.
"""

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

from collector.case_generator import (
    get_attention_context_shape_sweeps,
    get_attention_generation_shape_sweeps,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.planner.schemas.attention import build_attention_context_cases, build_attention_generation_cases
from collector.registry_types import PerfFile


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
    *,
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

    synthetic_max_seq_len = input_len + output_len + 1
    if attention_window_size > 0:
        synthetic_max_seq_len = max(synthetic_max_seq_len, attention_window_size + output_len + 1)

    kv_cache_config = KvCacheConfig(
        max_tokens=int((synthetic_max_seq_len - 1) / tokens_per_block + 1)
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
        max_seq_len=synthetic_max_seq_len,  # +1 for the magic fixme mentioned in trtllm xqa JIT path impl.
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
    kv_cache_dtype_str = "bfloat16"
    if use_fp8_kv_cache:
        kv_cache_dtype_str = "fp8"
    if use_fp8_context_fmha:
        dtype_str = "fp8"
    else:
        dtype_str = "bfloat16"

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
    return build_attention_context_cases(
        "trtllm",
        get_attention_context_shape_sweeps("trtllm"),
        sm_version=get_sm_version(),
        framework_version=tensorrt_llm.__version__,
    )


def get_generation_attention_test_cases():
    return build_attention_generation_cases(
        "trtllm",
        get_attention_generation_shape_sweeps("trtllm"),
        sm_version=get_sm_version(),
        framework_version=tensorrt_llm.__version__,
    )


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
