# Phase 2: Mock Layer 构建

## 目标

创建一个独立可运行的算子层（mock layer），用于：
1. 验证算子行为正确
2. 后续 nsys profiling 对齐
3. 作为 Phase 5 数据采集的基础

## 方式 A: 使用框架官方类（推荐）

优先使用 TensorRT-LLM / vLLM / SGLang 中已有的模型层类。

参考 `collector/trtllm/collect_mla.py` 的做法：

```python
import tensorrt_llm
import torch
from tensorrt_llm._torch.attention_backend import AttentionInputType, TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.utils import create_attention
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.mapping import Mapping

# 1. 构建 pretrained config（从模型 config 推导）
pretrained_config = ...

# 2. 构建 model config
model_config = ModelConfig(
    pretrained_config=pretrained_config,
    mapping=Mapping(world_size=1, rank=0, tp_size=1),
)

# 3. 实例化算子
layer = XxxAttention(model_config=model_config, layer_idx=0)
# 或
layer = create_attention(...)
```

## 方式 B: 自定义实现

当官方类无法直接使用（依赖过重、需要特殊配置等）时，自行实现。

```python
import torch
import torch.nn as nn

class XxxOp(nn.Module):
    """Mock implementation of the Xxx operation."""
    
    def __init__(self, param1: int, param2: int, dtype=torch.bfloat16):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.dtype = dtype
        # 初始化权重和子模块
        self.weight = nn.Parameter(
            torch.randn(param1, param2, dtype=dtype, device="cuda")
        )
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        # 实现算子核心计算逻辑
        output = ...
        return output
```

## Context vs Generation 的区别

大部分算子在 context（prefill）和 generation（decode）阶段有不同的行为：

| 维度 | Context Phase | Generation Phase |
|------|--------------|------------------|
| **输入 tokens** | batch_size * seq_len（多 token） | batch_size * 1（单 token per request） |
| **KV Cache** | 不读取/少量 prefix 读取 | 读取全部历史 KV |
| **计算模式** | 通常 compute-bound | 通常 memory-bound |
| **典型参数范围** | batch 1~256, seq 1~32768 | batch 1~1024, kv_len 1~131072 |

**mock layer 需要同时覆盖这两种模式。**

## 关键检查项

| 检查项 | 说明 | 验证方法 |
|--------|------|----------|
| **输入格式** | shape, dtype, device 是否正确 | `print(input.shape, input.dtype, input.device)` |
| **输出格式** | shape 应与输入 hidden_size 对齐 | `assert output.shape[-1] == hidden_size` |
| **KV Cache Manager** | Attention 类 op 通常需要 cache manager | 检查是否有 `KVCacheManager` 初始化 |
| **边界条件** | batch=1, seq=1, 大 batch 等 | 多种参数组合测试 |
| **Dtype** | BF16/FP16/FP8 限制 | 确认目标 dtype 下能正确运行 |
| **CUDA Graph 兼容** | 采集时会用 CUDA Graph 加速 | 测试 graph capture 是否成功 |

## 独立测试脚本模板

```python
#!/usr/bin/env python3
"""Test script for Xxx mock layer."""

import torch

def create_xxx_layer(param1, param2, dtype=torch.bfloat16):
    """Create and return the mock layer."""
    # ... 构建逻辑 ...
    return layer

def test_context_phase():
    """Test context (prefill) phase."""
    layer = create_xxx_layer(param1=64, param2=2048)
    
    batch_size, seq_len = 4, 1024
    hidden_size = 4096
    hidden = torch.randn(
        [batch_size * seq_len, hidden_size],
        dtype=torch.bfloat16, device="cuda"
    )
    
    # Forward pass
    output = layer(hidden)
    print(f"Context - Input: {hidden.shape}, Output: {output.shape}")
    assert output.shape == hidden.shape, f"Shape mismatch: {output.shape} != {hidden.shape}"
    
    # Verify no NaN
    assert not torch.isnan(output).any(), "Output contains NaN!"

def test_generation_phase():
    """Test generation (decode) phase."""
    layer = create_xxx_layer(param1=64, param2=2048)
    
    batch_size = 32
    hidden_size = 4096
    hidden = torch.randn(
        [batch_size, hidden_size],  # 单 token per request
        dtype=torch.bfloat16, device="cuda"
    )
    
    output = layer(hidden)
    print(f"Generation - Input: {hidden.shape}, Output: {output.shape}")
    assert output.shape == hidden.shape

def test_cuda_graph_compatibility():
    """Test that the layer works with CUDA Graph capture."""
    layer = create_xxx_layer(param1=64, param2=2048)
    hidden = torch.randn([32, 4096], dtype=torch.bfloat16, device="cuda")
    
    # Warmup
    for _ in range(3):
        layer(hidden)
    torch.cuda.synchronize()
    
    # Graph capture
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            output = layer(hidden)
        torch.cuda.synchronize()
        print("CUDA Graph capture: SUCCESS")
    except Exception as e:
        print(f"CUDA Graph capture: FAILED - {e}")
        print("Note: benchmark_with_power has allow_graph_fail fallback")

if __name__ == "__main__":
    test_context_phase()
    test_generation_phase()
    test_cuda_graph_compatibility()
    print("\nAll tests passed!")
```

## 输出

Phase 2 完成后应产出：
1. **可运行的 mock layer 代码** — 可以是独立 .py 文件或集成到 collector 中
2. **测试脚本** — 验证 context/generation 两种模式都正常
3. **参数文档** — 记录 mock layer 需要的所有参数及其含义

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `ImportError` 缺少模块 | TRT-LLM 版本不匹配 | 检查所需的 TRT-LLM 版本，参考 `frameworks/` |
| CUDA OOM | 输入太大 | 减小 batch_size 或 seq_len |
| CUDA Graph 捕获失败 | 算子内有动态控制流 | 使用 `allow_graph_fail=True` 走 eager fallback |
| 输出全零或全 NaN | 权重未正确初始化 | 检查权重初始化和 forward 逻辑 |
| KV Cache 未初始化 | Attention 类 op 需要 cache manager | 参考 `collect_mla.py` 中 KVCacheManager 的用法 |
| MPI 初始化失败 | TRT-LLM import 需要 MPI | 设置 `export OPAL_PREFIX=/opt/hpcx/ompi` |

## TP 模拟（关键）

参考 `collect_mla.py` 的做法：**不需要多卡**。单卡计算 `local_num_heads = global_heads // tp_size`，然后用 `Mapping(world_size=1, tp_size=1)` 创建 layer。数据中 `num_heads` 列记录 local heads。
