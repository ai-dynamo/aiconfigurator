# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collect DSA (DeepSeek Sparse Attention) performance data for TRT-LLM.

Profiles the full DSA attention block (kv_a_proj + indexer + sparse MLA)
for DeepSeek-V3.2 on Hopper GPUs (SM90).

Usage:
    python3 collect_dsa.py --mode context
    python3 collect_dsa.py --mode generation
"""

import argparse
import gc
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

# Add collector root to path for helper imports (same pattern as other collectors)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import AttentionInputType
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.attention import MLA
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

from helper import benchmark_with_power, log_perf

# ═══════════════════════════════════════════════════════════════════════
# DeepSeek-V3.2 Model Constants (from config.json)
# ═══════════════════════════════════════════════════════════════════════
HIDDEN_SIZE = 7168
NUM_ATTENTION_HEADS = 128
NUM_KEY_VALUE_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
MAX_POSITION_EMBEDDINGS = 163840
VOCAB_SIZE = 129280


# ═══════════════════════════════════════════════════════════════════════
# Test Case Generation
# ═══════════════════════════════════════════════════════════════════════

def get_context_dsa_test_cases():
    """Context phase test cases, same format as MLA: each case is a positional arg list for run_dsa."""
    test_cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    s_list = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048,
              3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]

    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for b in b_list:
            for s in s_list:
                if b * s > 65536:
                    continue
                # Positional args: (seq_len, batch_size, tp_size, is_context, perf_filename)
                test_cases.append([s, b, tp_size, True, "dsa_context_perf.txt"])
    return test_cases


def get_generation_dsa_test_cases():
    """Generation phase test cases, same format as MLA."""
    test_cases = []
    b_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    s_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
              2048, 4096, 8192, 16384, 32768, 65536, 131072]

    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for b in b_list:
            for s in s_list:
                if b * s > 1024 * 4096 * 2 * 2:
                    continue
                # seq_len here is kv_cache_len for generation
                test_cases.append([s, b, tp_size, False, "dsa_generation_perf.txt"])
    return test_cases


# ═══════════════════════════════════════════════════════════════════════
# DSA Layer Factory
# ═══════════════════════════════════════════════════════════════════════

def create_dsa_layer(tp_size=1, device="cuda:0"):
    """
    Create a standalone DSA attention layer using DeepseekV32Attention.

    TP handling: Same approach as collect_mla.py — compute local heads
    (num_heads // tp_size) and run on a single GPU with Mapping(tp_size=1).
    This simulates one TP rank's computation without needing multi-GPU.
    """
    from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention

    assert NUM_ATTENTION_HEADS % tp_size == 0, f"num_heads {NUM_ATTENTION_HEADS} not divisible by tp_size {tp_size}"
    local_num_heads = NUM_ATTENTION_HEADS // tp_size

    # Always use single-GPU mapping; TP is simulated via local_num_heads
    mapping = Mapping(world_size=1, rank=0, tp_size=1)

    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        indexer_max_chunk_size=32768,
    )

    # Pass local (post-TP-split) num_heads to pretrained_config.
    # DeepseekV32Attention reads num_attention_heads from config and divides
    # by mapping.tp_size internally. Since mapping.tp_size=1 here, the
    # local_num_heads will be used directly — matching one TP rank's workload.
    pretrained_config = SimpleNamespace(
        rms_norm_eps=1e-6,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=local_num_heads,  # local heads for this TP rank
        num_key_value_heads=local_num_heads,  # MLA: num_kv_heads == num_heads per rank
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        torch_dtype=torch.bfloat16,
        rope_theta=10000,
        rope_scaling={
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    )

    model_config = ModelConfig(
        mapping=mapping,
        sparse_attention_config=sparse_config,
        pretrained_config=pretrained_config,
    )

    attn_layer = DeepseekV32Attention(
        model_config=model_config,
        layer_idx=0,
        aux_stream=None,
        reduce_output=False,  # No TP all-reduce for single-GPU benchmarking
    ).to(device)

    return attn_layer, model_config, mapping, sparse_config, local_num_heads


# ═══════════════════════════════════════════════════════════════════════
# DSA Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════

def run_dsa(
    seq_len,
    batch_size,
    tp_size,
    is_context,
    perf_filename,
    device="cuda:0",
    warming_up=10,
    test_ite=6,
):
    """Run a single DSA benchmark test case. Args match test case list order."""
    from tensorrt_llm._torch.attention_backend.sparse.dsa import (
        DSACacheManager,
        DSAtrtllmAttentionMetadata,
    )

    device = torch.device(device)
    torch.cuda.set_device(device)

    head_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576

    # ── Create DSA layer (single GPU, local heads computed inside) ──
    attn_layer, model_config, mapping, sparse_config, local_num_heads = create_dsa_layer(tp_size, device)
    num_heads = local_num_heads  # already divided by tp_size

    # ── KV Cache Setup ──
    tokens_per_block = 64
    if is_context:
        max_seq = seq_len + 1
        total_tokens = seq_len * batch_size
    else:
        max_seq = seq_len + 1  # kv_cache_len + 1 for generation token
        total_tokens = batch_size

    kv_cache_config = KvCacheConfig(
        max_tokens=int((max_seq) / tokens_per_block + 1)
        * tokens_per_block * batch_size * 2,
        enable_block_reuse=False,
    )

    kv_cache_manager = DSACacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=tensorrt_llm.bindings.DataType.BF16,
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )

    # ── Add dummy requests ──
    if is_context:
        input_seq_lens = [seq_len] * batch_size
        total_seq_lens = [seq_len + 1] * batch_size
    else:
        input_seq_lens = [seq_len] * batch_size  # kv_cache already filled
        total_seq_lens = [seq_len + 1] * batch_size

    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    # ── Create Attention Metadata ──
    if is_context:
        num_cached_tokens = [0] * batch_size
        attn_metadata = DSAtrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=total_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            enable_flash_mla=True,
            seq_lens=torch.tensor(input_seq_lens, dtype=torch.int32),
            position_ids=None,
            num_contexts=batch_size,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens,
                block_ids_per_seq=None,
                host_max_attention_window_sizes=None,
                host_sink_token_length=None,
            ),
            cross=None,
            request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False, cache_reuse=False
            ),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            sparse_attention_config=sparse_config,
        )
    else:
        gen_seq_lens = [1] * batch_size
        attn_metadata = DSAtrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=total_tokens,
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
            runtime_features=AttentionRuntimeFeatures(
                chunked_prefill=False, cache_reuse=False
            ),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            sparse_attention_config=sparse_config,
        )

    # Set indexer reference (DeepseekV32Attention stores it on self.indexer)
    attn_metadata.indexer = attn_layer.indexer
    attn_metadata.prepare()

    # ── Create Input Tensors ──
    if is_context:
        num_tokens = seq_len * batch_size
    else:
        num_tokens = batch_size

    hidden_states = torch.randn(
        [num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16, device=device
    )
    position_ids = torch.arange(
        seq_len if is_context else seq_len,
        dtype=torch.long, device=device
    ).unsqueeze(0).expand(batch_size, -1).contiguous()

    if is_context:
        position_ids = position_ids.reshape(-1)
    else:
        position_ids = torch.tensor(
            [seq_len - 1] * batch_size, dtype=torch.long, device=device
        )

    # ── Dry run ──
    # DeepseekV32Attention.forward inherits MLA.forward:
    #   forward(position_ids, hidden_states, attn_metadata, ...)
    try:
        attn_layer.forward(position_ids, hidden_states, attn_metadata)
    except Exception as e:
        print(f"  Dry run failed: {e}")
        return

    # ── Benchmark ──
    def kernel_func():
        attn_layer.forward(position_ids, hidden_states, attn_metadata)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
        allow_graph_fail=True,
    ) as results:
        pass

    latency = results["latency_ms"]

    # ── Log Results ──
    if is_context:
        isl = seq_len
        step = 0
    else:
        isl = 1
        step = seq_len  # kv_cache_len

    log_perf(
        item_list=[{
            "dsa_dtype": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "num_heads": num_heads,
            "index_n_heads": INDEX_N_HEADS,
            "index_topk": INDEX_TOPK,
            "batch_size": batch_size,
            "isl": isl,
            "tp_size": tp_size,
            "step": step,
            "latency": f"{latency:.4f}",
        }],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"dsa_{'context' if is_context else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    phase = "context" if is_context else "generation"
    print(f"  [{phase}] b={batch_size}, s={seq_len}, tp={tp_size}: {latency:.4f} ms")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Standalone entry point. For use with collect.py, import get_*_test_cases + run_dsa directly."""
    parser = argparse.ArgumentParser(description="Collect DSA performance data")
    parser.add_argument("--mode", choices=["context", "generation"], required=True)
    parser.add_argument("--tp", type=int, default=None, help="Filter by tp_size")
    args = parser.parse_args()

    if args.mode == "context":
        test_cases = get_context_dsa_test_cases()
    else:
        test_cases = get_generation_dsa_test_cases()

    if args.tp is not None:
        test_cases = [tc for tc in test_cases if tc[2] == args.tp]  # tc[2] = tp_size

    print(f"Running {len(test_cases)} {args.mode} DSA test cases...")

    for i, tc in enumerate(test_cases):
        s, b, tp = tc[0], tc[1], tc[2]
        print(f"[{i+1}/{len(test_cases)}]", end="")
        try:
            run_dsa(*tc)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM: b={b}, s={s}, tp={tp}")
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"  FAILED: b={b}, s={s}, tp={tp}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
