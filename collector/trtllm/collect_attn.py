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
    windows_for_head_dim,
)
from collector.helper import benchmark_with_power, get_sm_version, log_perf
from collector.registry_types import PerfFile


def _skip_trtllm_sm120_fp8_context_fmha(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not (tensorrt_llm.__version__.startswith(("1.3.0rc5", "1.3.0rc10")) and get_sm_version() >= 120):
        return False

    num_tokens = batch_size * input_len
    if tensorrt_llm.__version__.startswith("1.3.0rc5"):
        return (
            # MHA h=128 max-token cases crash with an illegal memory access in
            # the SM120 FP8 context FMHA kernel.
            (num_heads == num_key_value_heads == 96 and head_dim == 128 and num_tokens == 65536)
            # h=256 cases fail in the qkv_256 SM120 FP8 FMHA kernel.
            or head_dim == 256
        )

    if head_dim != 256:
        return False

    return (
        # TRT-LLM 1.3.0rc10 SM120 qkv_256 FP8 context FMHA crashes with
        # cudaErrorIllegalAddress for these verified high-token regions.
        (num_heads == 96 and num_key_value_heads == 8 and num_tokens >= 81920)
        or (num_heads == 48 and num_key_value_heads == 8 and num_tokens >= 131072)
        or (num_heads == num_key_value_heads == 96 and batch_size >= 2 and input_len >= 16384)
    )


def _skip_trtllm_sm89_rc15_long_context_gqa(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> bool:
    if not (tensorrt_llm.__version__.startswith("1.3.0rc15") and get_sm_version() == 89):
        return False

    if num_key_value_heads not in {1, 2, 4, 8}:
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 98304
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 81920
        return head_dim >= 192 and num_tokens >= 98304
    if num_heads == 48:
        if head_dim >= 256:
            return num_tokens >= 98304
        return head_dim >= 192 and num_tokens >= 131072
    if num_heads == 40:
        return head_dim >= 256 and num_tokens >= 131072
    return False


def _skip_trtllm_sm89_rc15_fp8_context_mha(
    batch_size: int,
    input_len: int,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    use_fp8_kv_cache: bool,
    use_fp8_context_fmha: bool,
) -> bool:
    if not (tensorrt_llm.__version__.startswith("1.3.0rc15") and get_sm_version() == 89):
        return False

    if not (use_fp8_kv_cache and use_fp8_context_fmha and num_heads == num_key_value_heads):
        return False

    num_tokens = batch_size * input_len
    if num_heads == 96:
        if head_dim == 128:
            return num_tokens >= 65536
        if head_dim >= 256:
            return num_tokens >= 32768
        return head_dim >= 192 and num_tokens >= 40960
    if num_heads == 64:
        if head_dim >= 256:
            return num_tokens >= 49152
        return head_dim >= 192 and num_tokens >= 65536
    if num_heads == 48:
        return head_dim >= 256 and num_tokens >= 65536
    return False


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


def _int_list(values):
    return [int(value) for value in values]


def _kv_head_options(values, num_heads):
    return [num_heads if value == "self" else int(value) for value in values]


def _default_attention_window_options(head_dim: int) -> list[int]:
    return [128, 0] if head_dim == 64 else [0]


def _attention_window_options(shape_sweep: dict, head_dim: int) -> list[int]:
    windows = _default_attention_window_options(head_dim)
    if "window_sizes" in shape_sweep:
        windows = [*windows, *windows_for_head_dim(_int_list(shape_sweep["window_sizes"]), head_dim)]
    return sorted(set(windows), reverse=True)


def get_context_attention_test_cases():
    has_fp8 = get_sm_version() > 86
    test_cases = []

    for shape_sweep in get_attention_context_shape_sweeps("trtllm"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        query_head_counts = _int_list(shape_sweep["query_head_counts"])
        kv_head_options = shape_sweep["kv_head_options"]
        head_dims = _int_list(shape_sweep["head_dims"])
        max_tokens_self_attention = int(shape_sweep["max_tokens_self_attention"])
        max_tokens_grouped_query_attention = int(shape_sweep["max_tokens_grouped_query_attention"])
        max_batch_size_self_attention = int(shape_sweep["max_batch_size_self_attention"])
        max_kv_elements = int(shape_sweep["max_kv_elements"])

        for h in head_dims:
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
                            if b * s * num_kv_heads * h * 2 >= max_kv_elements:
                                continue
                            if get_sm_version() >= 100:
                                # though it's a precheck of gen kernels during the attention op init,
                                # this cannot be skipped for now
                                # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                                m_num_heads_q_per_kv = n // num_kv_heads
                                max_num_heads_q_per_kv_in_cta = 32
                                if (
                                    m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                                    and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                                ):
                                    continue

                            skip_fp8_context_fmha = _skip_trtllm_sm120_fp8_context_fmha(
                                b,
                                s,
                                n,
                                num_kv_heads,
                                h,
                            )
                            if _skip_trtllm_sm89_rc15_long_context_gqa(b, s, n, num_kv_heads, h):
                                continue
                            for attention_window_size in _attention_window_options(shape_sweep, h):
                                for precision_case in shape_sweep["precision_cases"]:
                                    use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                    use_fp8_context_fmha = bool(precision_case["fp8_context_fmha"])
                                    if not has_fp8 and use_fp8_kv_cache:
                                        continue
                                    if skip_fp8_context_fmha and use_fp8_context_fmha:
                                        continue
                                    if _skip_trtllm_sm89_rc15_fp8_context_mha(
                                        b,
                                        s,
                                        n,
                                        num_kv_heads,
                                        h,
                                        use_fp8_kv_cache,
                                        use_fp8_context_fmha,
                                    ):
                                        continue
                                    test_cases.append(
                                        [
                                            b,
                                            s,
                                            n,
                                            num_kv_heads,
                                            h,
                                            attention_window_size,
                                            use_fp8_kv_cache,
                                            use_fp8_context_fmha,
                                            True,
                                        ]
                                    )

    return test_cases


def _generation_target_sequence_lengths(batch_sizes, sequence_lengths, num_heads, max_tokens, shape_sweep):
    b_s_dict = {}
    s_b_dict = {}
    for s in sequence_lengths:
        max_b = max_tokens // s // num_heads
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
    has_fp8 = get_sm_version() > 86
    test_cases = []

    for shape_sweep in get_attention_generation_shape_sweeps("trtllm"):
        batch_sizes = _int_list(shape_sweep["batch_sizes"])
        sequence_lengths = _int_list(shape_sweep["sequence_lengths"])
        head_dims = _int_list(shape_sweep["head_dims"])
        min_drop_batch = int(shape_sweep["drop_largest_sequence_for_batch_at_least"])

        # MHA
        for n in sorted(_int_list(shape_sweep["mha_query_head_counts"]), reverse=True):
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                int(shape_sweep["max_mha_tokens_per_step"]),
                shape_sweep,
            )
            for h in head_dims:
                for b, s_list_limited in b_s_dict.items():
                    target_s_list = sorted(s_list_limited)
                    if b >= min_drop_batch:
                        target_s_list = target_s_list[:-1]
                    for s in target_s_list:
                        for attention_window_size in _attention_window_options(shape_sweep, h):
                            for precision_case in shape_sweep["precision_cases"]:
                                use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                if not has_fp8 and use_fp8_kv_cache:
                                    continue
                                test_cases.append(
                                    [b, s, n, n, h, attention_window_size, use_fp8_kv_cache, False, False]
                                )

        # XQA
        for n in sorted(_int_list(shape_sweep["xqa_query_head_counts"]), reverse=True):
            b_s_dict = _generation_target_sequence_lengths(
                batch_sizes,
                sequence_lengths,
                n,
                int(shape_sweep["max_xqa_tokens_per_step"]),
                shape_sweep,
            )
            for h in head_dims:
                for b, s_list_limited in b_s_dict.items():
                    target_s_list = sorted(s_list_limited)
                    if b >= min_drop_batch:
                        target_s_list = target_s_list[:-1]
                    for n_kv in _int_list(shape_sweep["kv_head_counts"]):
                        if n_kv >= n:
                            continue
                        if get_sm_version() >= 100:
                            # TLLM_CHECK_WITH_INFO((params.m_num_heads_q_per_kv < max_num_heads_q_per_kv_in_cta || params.m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta == 0), # noqa: E501
                            m_num_heads_q_per_kv = n // n_kv
                            max_num_heads_q_per_kv_in_cta = 32
                            if (
                                m_num_heads_q_per_kv >= max_num_heads_q_per_kv_in_cta
                                and m_num_heads_q_per_kv % max_num_heads_q_per_kv_in_cta != 0
                            ):
                                continue
                        for s in target_s_list:
                            for attention_window_size in _attention_window_options(shape_sweep, h):
                                for precision_case in shape_sweep["precision_cases"]:
                                    use_fp8_kv_cache = bool(precision_case["fp8_kv_cache"])
                                    if not has_fp8 and use_fp8_kv_cache:
                                        continue
                                    test_cases.append(
                                        [b, s, n, n_kv, h, attention_window_size, use_fp8_kv_cache, False, False]
                                    )
    return test_cases


if __name__ == "__main__":
    test_cases = get_context_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.CONTEXT_ATTENTION)

    test_cases = get_generation_attention_test_cases()
    for test_case in test_cases:
        run_attention_torch(*test_case, perf_filename=PerfFile.GENERATION_ATTENTION)
