# Phase 4: SDK 集成模板

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `operations.py` | 添加 Operation 类 |
| `perf_database.py` | 添加查询方法 |
| `models.py` | 添加模型集成逻辑 |

## Operation 类模板

```python
@dataclass
class ContextXxx(Operation):
    """Context phase operation"""
    name: str = "context_xxx"
    scale_factor: float = 1.0
    
    # 算子参数
    param1: int = 64
    param2: int = 2048
    
    def query(self, db, batch_size, s, **kwargs) -> float:
        # 1. 尝试从数据库查询
        try:
            return db.query_context_xxx(b=batch_size, s=s, ...)
        except KeyError:
            # 2. 无数据则用 SOL 估算
            return self._sol_estimate(batch_size, s)
    
    def _sol_estimate(self, batch_size, s) -> float:
        # 理论性能估算
        flops = ...  # 根据算子复杂度计算
        peak_tflops = 2.0  # GPU 峰值
        return (flops / (peak_tflops * 1e12)) * 1000
```

## 数据库查询模板

```python
# perf_database.py
def query_context_xxx(self, b, s, **params) -> float:
    data = self._load_perf_data("xxx_context_perf.txt")
    
    for row in data:
        if row["batch_size"] == b and row["isl"] == s:
            return float(row["latency"])
    
    raise KeyError(f"No data for b={b}, s={s}")
```

## 模型集成模板

```python
# models.py
class NewModel:
    def _setup_operations(self):
        for i in range(self.num_layers):
            if self._use_new_op(i):
                self.context_ops.append(ContextXxx(...))
            else:
                self.context_ops.append(ContextExisting(...))
```

## CLI 验证

```bash
aiconfigurator cli support --model-path <model> --system h200_sxm --backend trtllm
```
