# Phase 2: Mock Model Layer

## Objective

Create a standalone DSA operation for testing and profiling without full model overhead.

## Two Approaches

### Approach A: Official Class (Recommended)

Use TensorRT-LLM's built-in `DeepseekV32Attention`:

```python
import torch
import tensorrt_llm
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV32Attention
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

# Constants for DeepSeek V3.2
HIDDEN_SIZE = 7168
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 128
V_HEAD_DIM = 128
INDEX_N_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048

def create_mla_module(num_heads, device="cuda:0"):
    """Create MLA module with DSA enabled."""
    
    # Create DSA config
    sparse_attention_config = DeepSeekSparseAttentionConfig(
        index_n_heads=INDEX_N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
    )
    
    # Create PretrainedConfig with all required fields
    pretrained_config = PretrainedConfig()
    pretrained_config.hidden_size = HIDDEN_SIZE
    pretrained_config.num_attention_heads = num_heads
    pretrained_config.q_lora_rank = Q_LORA_RANK
    pretrained_config.kv_lora_rank = KV_LORA_RANK
    pretrained_config.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    pretrained_config.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    pretrained_config.v_head_dim = V_HEAD_DIM
    
    # Create ModelConfig
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        sparse_attention_config=sparse_attention_config,
    )
    
    # Create DeepseekV32Attention
    mla = DeepseekV32Attention(
        model_config=model_config,
        layer_idx=0,
    )
    
    return mla, sparse_attention_config

def create_kv_cache(batch_size, input_len, sparse_attention_config, device="cuda:0"):
    """Create KV cache manager for DSA."""
    
    max_seq_len = input_len + 10
    
    kv_cache_config = KvCacheConfig(
        max_tokens=batch_size * max_seq_len,
        enable_block_reuse=False,
    )
    
    kv_cache_manager = DSACacheManager(
        kv_cache_config,
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        tokens_per_block=64,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=Mapping(world_size=1, rank=0, tp_size=1),
        dtype=tensorrt_llm.bindings.DataType.BF16,
        sparse_attn_config=sparse_attention_config,  # Required!
    )
    
    return kv_cache_manager
```

### Approach B: Custom DSAOp (For Learning)

Build from scratch to understand each component:

```python
class DSAOp(nn.Module):
    """
    Complete DSA operation: Indexer + FlashMLA Sparse
    
    Flow:
    1. Indexer: Select top-k relevant tokens
    2. MLA RoPE: Apply rotary embeddings
    3. FlashMLA Sparse: Sparse attention
    """
    
    def __init__(
        self,
        num_heads: int,
        index_n_heads: int = 64,
        index_head_dim: int = 128,
        index_topk: int = 2048,
        device: str = "cuda:0",
    ):
        super().__init__()
        
        # Validate num_heads for SM90
        assert num_heads % 64 == 0, f"num_heads ({num_heads}) must be multiple of 64"
        
        self.num_heads = num_heads
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        
        # Create indexer (MQA for K selection)
        from tensorrt_llm._torch.attention_backend.sparse.dsa import Indexer
        self.indexer = Indexer(...)
        
        # Create MLA attention
        from tensorrt_llm._torch.attention_backend.utils import create_attention
        self.mqa = create_attention(
            ...,  # MLA params
            is_mla_enable=True,
            sparse_attention_config=...,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata,
        position_ids: torch.Tensor,
    ):
        # 1. Project to q, k, v
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 2. Indexer: compute topk_indices
        qr = q.view(-1, self.num_heads, -1).split([128, 64], dim=-1)[0]  # q_nope
        topk_indices = self.indexer(
            qr,
            hidden_states,
            attn_metadata,
            position_ids,
        )
        
        # 3. MLA RoPE + KV append
        self.mqa.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata
        )
        
        # 4. FlashMLA sparse attention
        from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
        
        # Load KV from cache
        kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices, attn_metadata, layer_idx=0
        )
        
        attn_out = flash_mla_sparse_fwd(
            q_concat,      # [num_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]
            kv_cache_pool,  # [total_kv, 1, head_dim]
            topk_indices,   # [num_tokens, 1, topk]
            self.softmax_scale,
        )
        
        return attn_out
```

## Critical Requirements

### 1. num_heads Must Be Multiple of 64

FlashMLA sparse on SM90 requires:
```python
# SM90: num_heads % 64 == 0
# SM100: num_heads == 128 exactly
```

### 2. BF16 Only for FlashMLA Sparse

```python
# ❌ Wrong - will fail
kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8

# ✅ Correct
kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
```

### 3. Use DSACacheManager

```python
# ❌ Wrong - missing indexer K cache
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

# ✅ Correct - has index_head_dim, quant_block_size
from tensorrt_llm._torch.attention_backend.sparse.dsa import DSACacheManager
```

### 4. Position IDs Must Be Contiguous

```python
# ❌ Wrong - non-contiguous after expand
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).reshape(-1)

# ✅ Correct
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).reshape(-1).contiguous()
```

## Testing the Mock Layer

```python
def test_dsa_op():
    # Create module
    mla, dsa_config = create_mla_module(num_heads=128)
    
    # Create KV cache
    kv_cache = create_kv_cache(
        batch_size=1,
        input_len=4096,
        sparse_attention_config=dsa_config,
    )
    
    # Create metadata
    attn_metadata = create_metadata(
        batch_size=1,
        input_len=4096,
        kv_cache_manager=kv_cache,
        sparse_attention_config=dsa_config,
    )
    
    # Run forward
    hidden_states = torch.randn(4096, 7168, dtype=torch.bfloat16, device="cuda:0")
    position_ids = torch.arange(4096, device="cuda:0", dtype=torch.int32)
    
    output = mla.forward(
        position_ids=position_ids,
        hidden_states=hidden_states,
        attn_metadata=attn_metadata,
    )
    
    print(f"Output shape: {output.shape}")  # [4096, 128 * 128]
```
