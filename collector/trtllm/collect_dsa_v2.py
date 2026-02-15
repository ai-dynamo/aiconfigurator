# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3.2 DSA (DeepSeek Sparse Attention) Performance Collector

This collector benchmarks complete DSA attention kernels using TensorRT-LLM's native
DSATrtllmAttention class.

DSA is only available on Hopper (SM90) and Blackwell GPUs.

Usage:
    # Run context benchmark
    python collect_dsa_v2.py --mode context --num_heads 128 --batch_sizes 1 --seq_lens 4096

    # Run generation benchmark
    python collect_dsa_v2.py --mode generation --num_heads 128 --batch_sizes 1 --seq_lens 4096
"""

import tensorrt_llm
import torch
import argparse
import os
import sys
from pathlib import Path

# Add parent directory for helper import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from collector.helper import benchmark_with_power, log_perf, get_sm_version

# Check SM version (DSA requires SM90+)
sm_version = get_sm_version()
if sm_version < 90:
    print(f"DSA requires SM90+ (Hopper/Blackwell), but got SM{sm_version}")
    sys.exit(1)

from tensorrt_llm._torch.attention_backend import AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata,
    DSACacheManager,
)
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionRuntimeFeatures,
    PositionalEmbeddingParams,
    RopeParams,
    MLAParams,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


# DSA Configuration (DeepSeek V3.2 defaults)
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048

# MLA Configuration
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128


def run_dsa(
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
    """Run DSA benchmark using DSATrtllmAttention."""
    device = torch.device(device)
    torch.cuda.set_device(device)
    layer_idx = 0

    # FlashMLA sparse requires num_heads to be multiple of 64
    assert num_heads % 64 == 0, f"num_heads ({num_heads}) must be multiple of 64 for FlashMLA sparse"
    assert num_heads % tp_size == 0, "num_heads != N * tp_size"
    
    num_heads = num_heads // tp_size
    num_key_value_heads = num_heads

    print(f"\n[DSA] batch={batch_size}, seq_len={input_len}, num_heads={num_heads * tp_size}")
    print(f"      device={device}, is_context={is_context_phase}")

    # Positional embedding params
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        embedder=None,
        rope=RopeParams(
            dim=QK_ROPE_HEAD_DIM,
            theta=10000,
            scale_type=RotaryScalingType.yarn,
            scale=40,
            max_positions=163840,
            original_max_positions=4096,
            beta_fast=32,
            beta_slow=1,
        ),
    )

    # Quant config
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
        print(f"      Using FP8 KV cache")
    else:
        quant_config = QuantConfig(kv_cache_quant_algo=None)
        print(f"      Using BF16 KV cache")

    # MLA params
    mla_params = MLAParams(
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        predicted_tokens_per_seq=1,
        hidden_size=HIDDEN_SIZE,
    )

    # DSA config
    sparse_attention_config = DeepSeekSparseAttentionConfig(
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
    )

    # Create DSATrtllmAttention
    attn = create_attention(
        backend_name="TRTLLM",
        layer_idx=layer_idx,
        num_heads=num_heads,
        head_dim=(QK_NOPE_HEAD_DIM if is_context_phase else KV_LORA_RANK) + QK_ROPE_HEAD_DIM,
        num_kv_heads=num_key_value_heads if is_context_phase else 1,
        pos_embd_params=pos_embd_params,
        quant_config=quant_config,
        is_mla_enable=True,
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM if is_context_phase else KV_LORA_RANK,
        predicted_tokens_per_seq=1,
        sparse_attention_config=sparse_attention_config,
        hidden_size=HIDDEN_SIZE,  # Required for Indexer
    )
    
    # Move indexer weights to CUDA
    attn.indexer.to(device)

    total_num_tokens = (input_len + output_len) * batch_size
    mapping = Mapping(world_size=world_size, rank=0, tp_size=tp_size)

    # KV cache config
    max_seq_len = input_len + output_len + 1
    kv_cache_config = KvCacheConfig(
        max_tokens=int((max_seq_len - 1) / tokens_per_block + 1)
        * tokens_per_block * batch_size * 2,
        enable_block_reuse=False,
    )

    # Use DSACacheManager
    kv_cache_manager = DSACacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        sparse_attn_config=sparse_attention_config,
    )

    input_seq_lens = [input_len for _ in range(batch_size)]
    total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    # Create metadata
    if is_context_phase:
        num_cached_tokens_per_seq = [0 for _ in range(batch_size)]
        attn_metadata = DSAtrtllmAttentionMetadata(
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
            ),
            cross=None,
            request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            sparse_attention_config=sparse_attention_config,
        )
        num_tokens = input_len * batch_size
    else:
        gen_seq_lens = [1 for _ in range(batch_size)]
        attn_metadata = DSAtrtllmAttentionMetadata(
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
            ),
            cross=None,
            request_ids=request_ids,
            prompt_lens=input_seq_lens,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=False, cache_reuse=False),
            all_rank_num_tokens=None,
            workspace=torch.tensor([], device=device, dtype=torch.int8),
            sparse_attention_config=sparse_attention_config,
        )
        num_tokens = batch_size

    attn_metadata.prepare()

    # Create input tensors
    # For DSA, we need:
    # 1. q: [num_tokens, num_heads * (kv_lora_rank + qk_rope_head_dim)] for generation
    # 2. hidden_states: [num_tokens, hidden_size] for indexer
    # 3. latent_cache: [num_tokens, kv_lora_rank + qk_rope_head_dim]
    # 4. qr: [num_tokens, q_lora_rank] for indexer
    # 5. indexer_k: [num_tokens, indexer.head_dim] for indexer
    
    hidden_states = torch.randn([num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16, device=device)
    
    if is_context_phase:
        # Context: fused_q contains q_nope + q_rope + k_nope + k_rope + v
        fused_q = torch.randn(
            [num_tokens, num_heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM + QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM + V_HEAD_DIM)],
            device=device,
            dtype=torch.bfloat16,
        )
        q_pe = None
    else:
        # Generation: fused_q contains q_latent + q_rope
        fused_q = torch.randn(
            [num_tokens, num_heads * (KV_LORA_RANK + QK_ROPE_HEAD_DIM)],
            device=device,
            dtype=torch.bfloat16,
        )
        q_pe = torch.randn([num_tokens, num_heads, QK_ROPE_HEAD_DIM], dtype=torch.bfloat16, device=device)

    compressed_kv = torch.randn([num_tokens, KV_LORA_RANK], dtype=torch.bfloat16, device=device)
    k_pe = torch.randn([num_tokens, QK_ROPE_HEAD_DIM], dtype=torch.bfloat16, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)

    # Indexer inputs
    qr = torch.randn([num_tokens, Q_LORA_RANK], dtype=torch.bfloat16, device=device)
    indexer_k = torch.randn([num_tokens, attn.indexer.head_dim], dtype=torch.bfloat16, device=device)

    if is_context_phase:
        position_ids = torch.arange(input_len, device=device, dtype=torch.int32)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    else:
        position_ids = torch.tensor([[input_len]] * batch_size, device=device, dtype=torch.int32)

    # Compute topk_indices from indexer
    topk_indices = attn.indexer(qr, hidden_states, attn_metadata, position_ids, indexer_k=indexer_k)

    # Dry run
    if is_context_phase:
        attn.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            out_scale=torch.tensor([]).float().to(device),
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
        )
    else:
        attn.forward(
            fused_q,
            None,
            None,
            attn_metadata,
            out_scale=torch.tensor([]).float().to(device),
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
            topk_indices=topk_indices,
        )

    torch.cuda.synchronize()

    # Benchmark
    def kernel_func():
        if is_context_phase:
            attn.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                out_scale=torch.tensor([]).float().to(device),
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                topk_indices=topk_indices,
            )
        else:
            attn.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                out_scale=torch.tensor([]).float().to(device),
                attention_input_type=AttentionInputType.generation_only,
                latent_cache=latent_cache,
                q_pe=q_pe,
                topk_indices=topk_indices,
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

    # Log result
    if is_context_phase:
        isl = input_len
        step = 0
    else:
        isl = 1
        step = input_len

    log_perf(
        item_list=[{
            "dsa_dtype": "bfloat16",
            "kv_cache_dtype": "fp8" if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8 else "bf16",
            "num_heads": num_heads * tp_size,
            "index_n_heads": INDEX_N_HEADS,
            "index_topk": INDEX_TOPK,
            "batch_size": batch_size,
            "isl": isl,
            "tp_size": tp_size,
            "step": step,
            "latency": latency,
        }],
        framework="TRTLLM",
        version=tensorrt_llm.__version__,
        device_name=torch.cuda.get_device_name(device),
        op_name=f"dsa_{'context' if is_context_phase else 'generation'}",
        kernel_source="default",
        perf_filename=perf_filename,
        power_stats=results["power_stats"],
    )

    return latency


def main():
    parser = argparse.ArgumentParser(description="Collect DSA performance data (v2 - using DSATrtllmAttention)")
    parser.add_argument("--mode", choices=["context", "generation"], default="context")
    parser.add_argument("--output_dir", type=str, default="./dsa_perf_data")
    parser.add_argument("--num_heads", type=int, default=128,
                        help="Number of heads (must be multiple of 64)")
    parser.add_argument("--kv_cache_dtype", choices=["bf16", "fp8"], default="bf16",
                        help="KV cache data type")
    parser.add_argument("--batch_sizes", type=str, default="1")
    parser.add_argument("--seq_lens", type=str, default="4096")
    parser.add_argument("--num_warmup", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--quick", action="store_true")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # FlashMLA sparse requires num_heads % 64 == 0
    if args.num_heads % 64 != 0:
        print(f"ERROR: num_heads ({args.num_heads}) must be multiple of 64 for FlashMLA sparse")
        sys.exit(1)
    
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    
    # Convert kv_cache_dtype string to DataType
    kv_dtype = tensorrt_llm.bindings.DataType.FP8 if args.kv_cache_dtype == "fp8" else tensorrt_llm.bindings.DataType.BF16
    
    print("=" * 60)
    print("DSA Performance Collector v2 (DSATrtllmAttention)")
    print("=" * 60)
    print(f"Mode: {args.mode}, num_heads: {args.num_heads}")
    print(f"KV cache dtype: {args.kv_cache_dtype}")
    print(f"Batch sizes: {batch_sizes}, Seq lens: {seq_lens}")
    print()
    
    results = []
    output_file = os.path.join(args.output_dir, f"dsa_{args.mode}_{args.kv_cache_dtype}_perf.txt")
    
    for b in batch_sizes:
        for s in seq_lens:
            if b * s > 65536:
                continue
            
            try:
                latency = run_dsa(
                    input_len=s,
                    batch_size=b,
                    output_len=1,
                    kv_cache_dtype=kv_dtype,
                    num_heads=args.num_heads,
                    world_size=1,
                    tp_size=1,
                    tokens_per_block=64,
                    warming_up=args.num_warmup,
                    test_ite=args.num_iterations,
                    is_context_phase=(args.mode == "context"),
                    perf_filename=output_file,
                    device=args.device,
                )
                
                if latency is not None:
                    results.append({"batch": b, "seq": s, "latency_ms": latency})
                    print(f"      ✅ Latency: {latency:.2f} ms")
                
                if args.quick:
                    break
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
        if args.quick:
            break
    
    print(f"\nTotal tests: {len(results)}")
    for r in results:
        print(f"  batch={r['batch']}, seq={r['seq']}, latency={r['latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
