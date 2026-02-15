# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeepSeek V3.2 DSA (DeepSeek Sparse Attention) Performance Collector

This collector benchmarks complete DSA attention kernels using TensorRT-LLM's native
DSA backend on single GPU.

DSA is only available on Hopper (SM90) and Blackwell GPUs.

DSA Flow:
1. Indexer: Compute topk_indices (Q/K projection + TopK via DeepGEMM)
2. FlashMLA Sparse: Sparse attention using topk_indices

Key Requirements:
- FlashMLA sparse requires num_heads to be multiple of 64 on SM90
- For TP=8: use 128 heads total (16 per GPU after TP)
- For TP=1: use 128 heads total

Usage:
    # Run benchmark
    python collect_dsa.py --mode context --tp_size 8

    # Profile with nsys (cuda-graph mode)
    nsys profile -o dsa_profile --cuda-graph-trace=node -t cuda,nvtx \
        python collect_dsa.py --mode context --batch_size 1 --seq_len 4096 --profile
"""

import tensorrt_llm
import torch
import torch.nn as nn
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

from tensorrt_llm._torch.attention_backend.interface import (
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata,
    DSACacheManager,
    Indexer,
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm._utils import get_sm_version

# Import FlashMLA sparse kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
    HAS_FLASH_MLA_SPARSE = True
except ImportError:
    flash_mla_sparse_fwd = None
    HAS_FLASH_MLA_SPARSE = False


# DSA Configuration (DeepSeek V3.2 defaults)
# FlashMLA sparse requires num_heads to be multiple of 64 on SM90
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
NUM_HEADS = 128  # Must be multiple of 64 for FlashMLA sparse

# MLA Configuration
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
SOFTMAX_SCALE = (KV_LORA_RANK + QK_ROPE_HEAD_DIM) ** -0.5


class DSAOp(nn.Module):
    """Complete DSA operation: Indexer + FlashMLA Sparse"""
    
    def __init__(
        self,
        num_heads: int,
        quant_config: QuantConfig,
        pos_embd_params: PositionalEmbeddingParams,
        mla_params: MLAParams,
        sparse_attention_config: DeepSeekSparseAttentionConfig,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.qk_head_dim = mla_params.qk_nope_head_dim + mla_params.qk_rope_head_dim
        self.qk_nope_head_dim = mla_params.qk_nope_head_dim
        self.qk_rope_head_dim = mla_params.qk_rope_head_dim
        self.kv_lora_rank = mla_params.kv_lora_rank
        self.v_head_dim = mla_params.v_head_dim
        self.layer_idx = layer_idx
        self.softmax_scale = SOFTMAX_SCALE
        
        # Indexer
        self.indexer = Indexer(
            quant_config=quant_config,
            pos_embd_params=pos_embd_params,
            mla_params=mla_params,
            skip_create_weights_in_init=False,
            sparse_attention_config=sparse_attention_config,
            dtype=dtype,
            layer_idx=layer_idx,
        )
        
        # DSATrtllmAttention for MLA operations
        self.mqa = create_attention(
            backend_name="TRTLLM",
            layer_idx=layer_idx,
            num_heads=num_heads,
            head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            is_mla_enable=True,
            q_lora_rank=mla_params.q_lora_rank,
            kv_lora_rank=mla_params.kv_lora_rank,
            qk_nope_head_dim=mla_params.qk_nope_head_dim,
            qk_rope_head_dim=mla_params.qk_rope_head_dim,
            v_head_dim=mla_params.kv_lora_rank,
            hidden_size=mla_params.hidden_size,
            predicted_tokens_per_seq=1,
            sparse_attention_config=sparse_attention_config,
        )
        
        # K_b projection: [num_heads, kv_lora_rank, qk_nope_head_dim]
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(num_heads, mla_params.kv_lora_rank, mla_params.qk_nope_head_dim, dtype=dtype),
            requires_grad=False,
        )
        
        # V_b projection: [num_heads, v_head_dim, kv_lora_rank]
        self.v_b_proj = nn.Parameter(
            torch.empty(num_heads, mla_params.v_head_dim, mla_params.kv_lora_rank, dtype=dtype),
            requires_grad=False,
        )
        
        # Initialize weights
        nn.init.normal_(self.k_b_proj_trans, mean=0.0, std=0.02)
        nn.init.normal_(self.v_b_proj, mean=0.0, std=0.02)
    
    def forward(
        self,
        qr: torch.Tensor,
        hidden_states: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        position_ids: torch.Tensor,
        indexer_k: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        output: torch.Tensor,
        is_generation: bool = False,
    ) -> torch.Tensor:
        """Full DSA forward"""
        num_tokens = q.shape[0]
        
        # 1. Indexer: compute topk_indices
        topk_indices = self.indexer(qr, hidden_states, attn_metadata, position_ids, indexer_k=indexer_k)
        
        # 2. MLA RoPE + KV append
        self.mqa.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=is_generation
        )
        
        # 3. Q nope projection
        q_nope, q_rope = q.view(-1, self.num_heads, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        q_nope_out = torch.empty(
            [num_tokens, self.num_heads, self.kv_lora_rank],
            dtype=q.dtype,
            device=q.device,
        )
        
        # BMM for q_nope projection
        q_nope_t = q_nope.transpose(0, 1)
        q_nope_out_t = q_nope_out.transpose(0, 1)
        torch.ops.trtllm.bmm_out(q_nope_t, self.k_b_proj_trans.transpose(1, 2), q_nope_out_t)
        q_nope_out = q_nope_out_t.transpose(0, 1)
        
        # Concat q_nope_out + q_rope
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)
        
        # Pad num_heads to multiple of 64 for FlashMLA sparse (SM90)
        sm_ver = get_sm_version()
        if sm_ver >= 100:
            padding = 128
        else:
            padding = ((self.num_heads + 63) // 64) * 64
        
        if self.num_heads != padding:
            q_padded = q_concat.new_zeros((num_tokens, padding, q_concat.shape[2]))
            q_padded[:, :self.num_heads, :] = q_concat
            q_concat = q_padded
        
        # 4. Convert topk_indices and get KV cache pool
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            attn_metadata,
            layer_idx=self.layer_idx,
            is_generation=is_generation,
        )
        topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
        
        # 5. FlashMLA sparse attention
        if flash_mla_sparse_fwd is not None:
            attn_out_latent = flash_mla_sparse_fwd(
                q_concat, kv_cache_pool, topk_indices_pool, self.softmax_scale
            )[0]
        else:
            raise RuntimeError("flash_mla_sparse_fwd not available")
        
        # Remove padding
        attn_out_latent = attn_out_latent[:, :self.num_heads, :]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads, self.kv_lora_rank])
        if self.num_heads != padding:
            attn_out_latent = attn_out_latent.contiguous()
        
        # 6. V projection
        attn_output = output.view([num_tokens, self.num_heads, self.v_head_dim])
        torch.ops.trtllm.bmm_out(
            attn_out_latent.transpose(0, 1),
            self.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )
        
        return output


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
    profile=False,
):
    """Run DSA benchmark on single GPU."""
    device = torch.device(device)
    torch.cuda.set_device(device)
    layer_idx = 0

    # FlashMLA sparse requires num_heads to be multiple of 64
    assert num_heads % 64 == 0, f"num_heads ({num_heads}) must be multiple of 64 for FlashMLA sparse"

    print(f"\n[DSA] batch={batch_size}, seq_len={input_len}, num_heads={num_heads}")
    print(f"      device={device}")
    print(f"      kv_cache_dtype={kv_cache_dtype}, is_context={is_context_phase}")

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

    # FP8 KV cache quantization
    if kv_cache_dtype == tensorrt_llm.bindings.DataType.FP8:
        quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
        print(f"      Using FP8 KV cache")
    else:
        quant_config = QuantConfig(kv_cache_quant_algo=None)
        print(f"      Using BF16 KV cache")

    mla_params = MLAParams(
        q_lora_rank=Q_LORA_RANK,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        predicted_tokens_per_seq=1,
        hidden_size=HIDDEN_SIZE,
    )

    sparse_attention_config = DeepSeekSparseAttentionConfig(
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
    )

    # Create DSA Op
    dsa_op = DSAOp(
        num_heads=num_heads,
        quant_config=quant_config,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        sparse_attention_config=sparse_attention_config,
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
    )
    dsa_op.cuda()
    
    print(f"      Indexer: n_heads={dsa_op.indexer.n_heads}, topk={dsa_op.indexer.index_topk}")

    total_num_tokens = (input_len + output_len) * batch_size
    mapping = Mapping(world_size=1, rank=0, tp_size=1)

    # Use DSACacheManager
    kv_cache_config = KvCacheConfig(
        max_tokens=int((input_len + output_len - 1) / tokens_per_block + 1)
        * tokens_per_block * batch_size * 2,
        enable_block_reuse=False,
    )

    kv_cache_manager = DSACacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=tokens_per_block,
        max_seq_len=input_len + output_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
        sparse_attn_config=sparse_attention_config,
    )

    input_seq_lens = [input_len for _ in range(batch_size)]
    total_seq_lens = [input_len + output_len for _ in range(batch_size)]
    request_ids = list(range(batch_size))
    kv_cache_manager.add_dummy_requests(request_ids, total_seq_lens)

    # Create DSA metadata
    if is_context_phase:
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
                num_cached_tokens_per_seq=[0] * batch_size,
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
        attn_metadata = DSAtrtllmAttentionMetadata(
            max_num_requests=batch_size,
            max_num_tokens=total_num_tokens,
            kv_cache_manager=kv_cache_manager,
            mapping=mapping,
            enable_flash_mla=True,
            seq_lens=torch.tensor([1] * batch_size, dtype=torch.int32),
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
    hidden_states = torch.randn([num_tokens, HIDDEN_SIZE], dtype=torch.bfloat16, device=device)
    
    if is_context_phase:
        position_ids = torch.arange(input_len, device=device, dtype=torch.int32)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    else:
        position_ids = torch.tensor([[input_len]] * batch_size, device=device, dtype=torch.int32)

    qr = torch.randn([num_tokens, Q_LORA_RANK], dtype=torch.bfloat16, device=device)
    indexer_k = torch.randn([num_tokens, INDEX_HEAD_DIM], dtype=torch.bfloat16, device=device)
    q = torch.randn([num_tokens, num_heads * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM)], dtype=torch.bfloat16, device=device)
    compressed_kv = torch.randn([num_tokens, KV_LORA_RANK], dtype=torch.bfloat16, device=device)
    k_pe = torch.randn([num_tokens, QK_ROPE_HEAD_DIM], dtype=torch.bfloat16, device=device)
    latent_cache = torch.cat([compressed_kv, k_pe], dim=-1)
    output = torch.empty([num_tokens, num_heads * V_HEAD_DIM], dtype=torch.bfloat16, device=device)

    # Dry run
    dsa_op(qr, hidden_states, attn_metadata, position_ids, indexer_k, q, latent_cache, output)
    torch.cuda.synchronize()

    # Profile mode
    if profile:
        print(f"      Running {test_ite} iterations for profiling...")
        for i in range(test_ite):
            dsa_op(qr, hidden_states, attn_metadata, position_ids, indexer_k, q, latent_cache, output)
            if (i + 1) % 5 == 0:
                print(f"        Iteration {i+1}/{test_ite}")
        torch.cuda.synchronize()
        print(f"      Done.")
        return None

    # Benchmark
    def kernel_func():
        dsa_op(qr, hidden_states, attn_metadata, position_ids, indexer_k, q, latent_cache, output)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=warming_up,
        num_runs=test_ite,
        repeat_n=1,
    ) as results:
        pass

    latency = results["latency_ms"]

    log_perf(
        item_list=[{
            "dsa_dtype": "bfloat16",
            "num_heads": num_heads,
            "index_n_heads": INDEX_N_HEADS,
            "index_topk": INDEX_TOPK,
            "batch_size": batch_size,
            "isl": input_len if is_context_phase else 1,
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
    parser = argparse.ArgumentParser(description="Collect DSA performance data")
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
    parser.add_argument("--profile", action="store_true")
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
    print("DSA Performance Collector")
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
                    profile=args.profile,
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
