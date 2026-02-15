# Phase 4: aiconfigurator Integration

## Objective

Add DSA operations to the aiconfigurator SDK for performance estimation.

## File Structure

```
src/aiconfigurator/sdk/
├── common.py           # Config dataclasses, architecture mapping
├── utils.py            # Config parsing
├── operations.py       # Operation classes
├── perf_database.py    # Database queries
└── models.py           # Model integration
```

## Step-by-Step Integration

### 4.1 Add Operation Classes

**File**: `src/aiconfigurator/sdk/operations.py`

```python
from dataclasses import dataclass
from typing import Optional
from .common import KVCacheQuantMode, FMHAQuantMode

@dataclass
class ContextDSA(Operation):
    """
    DSA context phase operation.
    
    Complexity: O(n × k) where k = index_topk
    Per-token compute: index_n_heads × index_head_dim × index_topk
    """
    name: str = "context_dsa"
    scale_factor: float = 1.0
    
    num_heads: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.float16
    fmha_quant_mode: FMHAQuantMode = FMHAQuantMode.float16
    
    def query(
        self,
        db: "PerfDatabase",
        batch_size: int,
        s: int,
        **kwargs,
    ) -> float:
        """Query performance from database or use SOL estimation."""
        
        # Try database first
        try:
            result = db.query_context_dsa(
                b=batch_size,
                s=s,
                num_heads=self.num_heads,
                index_n_heads=self.index_n_heads,
                index_topk=self.index_topk,
                kvcache_quant_mode=self.kvcache_quant_mode,
                fmha_quant_mode=self.fmha_quant_mode,
            )
            return float(result) * self.scale_factor
        except KeyError:
            # Fall back to SOL estimation
            return self._sol_estimate(batch_size, s)
    
    def _sol_estimate(self, batch_size: int, s: int) -> float:
        """
        SOL (Speed-of-light) estimation.
        
        DSA complexity: O(n × k) vs MLA's O(n²)
        """
        # Assume H200 SXM peak: ~2.0 TFLOPS for sparse attention
        # This is a rough estimate; use real data when available
        peak_tflops = 2.0
        
        # FLOPs: 2 × batch_size × s × index_n_heads × index_head_dim × index_topk
        flops = 2 * batch_size * s * self.index_n_heads * self.index_head_dim * self.index_topk
        
        # Latency in ms
        latency_ms = (flops / (peak_tflops * 1e12)) * 1000
        
        return latency_ms * self.scale_factor


@dataclass
class GenerationDSA(Operation):
    """
    DSA generation phase operation.
    
    Generation processes 1 token, referencing cached context.
    Latency is nearly constant regardless of seq_len.
    """
    name: str = "generation_dsa"
    scale_factor: float = 1.0
    
    num_heads: int = 128
    index_n_heads: int = 64
    index_topk: int = 2048
    kvcache_quant_mode: KVCacheQuantMode = KVCacheQuantMode.float16
    fmha_quant_mode: FMHAQuantMode = FMHAQuantMode.float16
    
    def query(
        self,
        db: "PerfDatabase",
        batch_size: int,
        s: int,  # Cached context length (affects latency slightly)
        **kwargs,
    ) -> float:
        try:
            result = db.query_generation_dsa(
                b=batch_size,
                s=s,
                num_heads=self.num_heads,
                index_n_heads=self.index_n_heads,
                index_topk=self.index_topk,
                kvcache_quant_mode=self.kvcache_quant_mode,
                fmha_quant_mode=self.fmha_quant_mode,
            )
            return float(result) * self.scale_factor
        except KeyError:
            # Generation is fast, rough estimate
            return 0.5 * self.scale_factor  # ~0.5ms for generation
```

### 4.2 Add Database Queries

**File**: `src/aiconfigurator/sdk/perf_database.py`

```python
class PerfDatabase:
    # ... existing methods ...
    
    def query_context_dsa(
        self,
        b: int,
        s: int,
        num_heads: int,
        index_n_heads: int,
        index_topk: int,
        kvcache_quant_mode: KVCacheQuantMode,
        fmha_quant_mode: FMHAQuantMode,
    ) -> float:
        """Query DSA context phase performance."""
        
        # Load from data file
        data_file = self._get_data_file("dsa_context_perf.txt")
        
        # Find matching entry
        for line in data_file:
            if (line["batch_size"] == b and
                line["isl"] == s and
                line["num_heads"] == num_heads and
                line["index_n_heads"] == index_n_heads and
                line["index_topk"] == index_topk):
                return float(line["latency"])
        
        raise KeyError(f"No DSA context data for b={b}, s={s}, heads={num_heads}")
    
    def query_generation_dsa(
        self,
        b: int,
        s: int,
        num_heads: int,
        index_n_heads: int,
        index_topk: int,
        kvcache_quant_mode: KVCacheQuantMode,
        fmha_quant_mode: FMHAQuantMode,
    ) -> float:
        """Query DSA generation phase performance."""
        
        data_file = self._get_data_file("dsa_generation_perf.txt")
        
        for line in data_file:
            if (line["batch_size"] == b and
                line["isl"] == s and
                line["num_heads"] == num_heads):
                return float(line["latency"])
        
        raise KeyError(f"No DSA generation data for b={b}, s={s}")
```

### 4.3 Add Model Integration

**File**: `src/aiconfigurator/sdk/models.py`

```python
class DeepSeekModel:
    def __init__(
        self,
        topk: int,
        num_experts: int,
        moe_inter_size: int,
        model_path: str,
        model_family: str,
        architecture: str,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
        hidden_size: int,
        inter_size: int,
        vocab_size: int,
        context_length: int,
        config: "ModelConfig",
        extra_params: Optional[dict] = None,
    ):
        # ... existing init ...
        
        # Parse DSA config
        self._use_dsa = extra_params is not None and "index_n_heads" in extra_params
        
        if self._use_dsa:
            self._dsa_config = DeepSeekV32Config(
                index_n_heads=extra_params.get("index_n_heads", 64),
                index_head_dim=extra_params.get("index_head_dim", 128),
                index_topk=extra_params.get("index_topk", 2048),
                first_k_dense_replace=extra_params.get("first_k_dense_replace", 3),
            )
        else:
            self._dsa_config = None
        
        self._setup_operations()
    
    def _setup_operations(self):
        """Setup context and generation operations."""
        
        self.context_ops = []
        self.generation_ops = []
        
        # Add attention ops for each layer
        for i in range(self.num_layers):
            if self._use_dsa and i >= self._dsa_config.first_k_dense_replace:
                # DSA layers
                self.context_ops.append(ContextDSA(
                    name=f"context_dsa_layer_{i}",
                    scale_factor=1.0,
                    num_heads=self.num_heads,
                    index_n_heads=self._dsa_config.index_n_heads,
                    index_topk=self._dsa_config.index_topk,
                ))
                self.generation_ops.append(GenerationDSA(
                    name=f"generation_dsa_layer_{i}",
                    scale_factor=1.0,
                    num_heads=self.num_heads,
                    index_n_heads=self._dsa_config.index_n_heads,
                    index_topk=self._dsa_config.index_topk,
                ))
            else:
                # MLA layers (or all layers if no DSA)
                self.context_ops.append(ContextMLA(
                    name=f"context_mla_layer_{i}",
                    scale_factor=1.0,
                    num_heads=self.num_heads,
                ))
                self.generation_ops.append(GenerationMLA(
                    name=f"generation_mla_layer_{i}",
                    scale_factor=1.0,
                    num_heads=self.num_heads,
                ))
        
        # Add MoE ops, etc.
        # ...
```

## Testing

```python
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk import config

# Test V3.2 model
model = get_model(
    model_path="deepseek-ai/DeepSeek-V3.2",
    system="h200_sxm",
    backend="trtllm",
    config=config.ModelConfig(tp_size=8),
)

# Check DSA operations
dsa_ops = [op for op in model.context_ops if "dsa" in op.name.lower()]
print(f"DSA context ops: {len(dsa_ops)}")  # Should be 58 for V3.2

# Query performance
from aiconfigurator.sdk.perf_database import PerfDatabase
db = PerfDatabase(system="h200_sxm", backend="trtllm")

latency = dsa_ops[0].query(db, batch_size=1, s=4096)
print(f"DSA context @ ISL=4096: {latency:.2f} ms")
```

## CLI Usage

```bash
aiconfigurator cli support \
  --model-path deepseek-ai/DeepSeek-V3.2 \
  --system h200_sxm \
  --backend trtllm
```

Output:
```
Model: deepseek-ai/DeepSeek-V3.2
Architecture: DeepseekV32ForCausalLM
Family: DEEPSEEK
DSA config: index_n_heads=64, index_topk=2048, first_k_dense=3
Layers: 61 (MLA: 3, DSA: 58)
```
