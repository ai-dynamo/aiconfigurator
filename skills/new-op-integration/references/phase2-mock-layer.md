# Phase 2: Mock Layer 模板

## 方式 A: 使用官方类 (推荐)

```python
from tensorrt_llm._torch.models.modeling_xxx import XxxAttention

# 创建 config
model_config = ModelConfig(
    pretrained_config=pretrained_config,
    mapping=Mapping(world_size=1, rank=0, tp_size=1),
)

# 创建算子
layer = XxxAttention(model_config=model_config, layer_idx=0)
```

## 方式 B: 自定义实现

```python
class XxxOp(nn.Module):
    def __init__(self, param1, param2):
        super().__init__()
        self.param1 = param1
        # 初始化子模块
        
    def forward(self, hidden_states, ...):
        # 算子逻辑
        return output
```

## 关键检查项

| 检查项 | 说明 |
|--------|------|
| **输入格式** | shape, dtype, device |
| **Cache Manager** | 是否需要特殊的 cache manager |
| **边界条件** | batch=1, seq=1, 大 batch 等 |
| **Dtype 约束** | BF16/FP16/FP8 限制 |

## 测试

```python
def test_op():
    layer = create_xxx_layer()
    hidden = torch.randn([num_tokens, hidden_size], dtype=torch.bfloat16, device="cuda")
    output = layer(hidden, ...)
    print(f"Output shape: {output.shape}")
```
