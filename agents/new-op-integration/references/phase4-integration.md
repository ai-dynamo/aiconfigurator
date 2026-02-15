# Phase 4: SDK 集成

## 目标

将新算子添加到 aiconfigurator 的建模体系中，使其能参与端到端延迟估算。

## 修改清单

需要修改以下文件（按顺序）：

| # | 文件 | 修改内容 |
|---|------|----------|
| 1 | `src/aiconfigurator/sdk/common.py` | 添加 `PerfDataFilename` 枚举值 |
| 2 | `src/aiconfigurator/sdk/perf_database.py` | 添加数据加载函数 + 查询方法 |
| 3 | `src/aiconfigurator/sdk/operations.py` | 添加 Operation 子类 |
| 4 | `src/aiconfigurator/sdk/models.py` | 在 Model 类中使用新 Operation |

## Step 1: 添加 PerfDataFilename

**文件**: `src/aiconfigurator/sdk/common.py`

```python
# common.py — PerfDataFilename 枚举 (约第 469 行)
class PerfDataFilename(Enum):
    gemm = "gemm_perf.txt"
    # ... 已有文件名 ...
    mamba2 = "mamba2_perf.txt"
    new_op_context = "new_op_context_perf.txt"      # ← 添加
    new_op_generation = "new_op_generation_perf.txt" # ← 添加 (如果 context/generation 数据分开)
```

## Step 2: 添加数据加载和查询

**文件**: `src/aiconfigurator/sdk/perf_database.py`

### 2a. 数据加载函数

参考 `load_context_mla_data()` 或 `load_mamba2_data()` 的模式：

```python
def load_new_op_data(new_op_file):
    """
    Load new op performance data from new_op_perf.txt.
    
    CSV columns: framework, version, device, op_name, kernel_source,
    param1, param2, batch_size, seq_len, latency [, power]
    """
    if not os.path.exists(new_op_file):
        logger.warning(f"New op data file {new_op_file} not found.")
        return None
    
    # 使用嵌套 defaultdict 构建查询索引
    # 索引结构: data[param1][param2][batch_size][seq_len] = {"latency": ..., "power": ..., "energy": ...}
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    with open(new_op_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 检查是否有 power 列（向后兼容）
    has_power = len(rows) > 0 and "power" in rows[0]
    if not has_power:
        logger.debug(f"Legacy database format detected in {new_op_file} - power will default to 0.0")
    
    for row in rows:
        param1 = int(row["param1"])
        param2 = int(row["param2"])
        b = int(row["batch_size"])
        s = int(row["seq_len"])
        latency = float(row["latency"])
        power = float(row.get("power", 0.0)) if has_power else 0.0
        energy = power * latency  # W·ms
        
        data[param1][param2][b][s] = {
            "latency": latency,
            "power": power,
            "energy": energy,
        }
    
    return data
```

### 2b. 在 PerfDatabase.__init__ 中加载数据

```python
# perf_database.py — PerfDatabase.__init__() 中
# 在对应 backend 的加载块中添加：

self._new_op_data = load_new_op_data(
    os.path.join(data_dir, common.PerfDataFilename.new_op_context.value)
)
```

### 2c. 查询方法

```python
def query_new_op(
    self,
    b: int,
    s: int,
    param1: int,
    param2: int,
    database_mode: common.DatabaseMode | None = None,
) -> PerformanceResult:
    """
    Query new op latency and energy.
    
    Args:
        b: Batch size
        s: Sequence length
        param1, param2: Op-specific parameters
        database_mode: Database mode override
    
    Returns:
        PerformanceResult: Acts as float (latency in ms).
                          Energy accessible via .energy attribute (W·ms).
    """
    mode = database_mode or self._database_mode
    
    # SOL fallback
    def get_sol(b, s, param1, param2):
        # 理论性能估算 (Speed of Light)
        # 基于 FLOPs 和 GPU 峰值算力 / 或基于 memory bandwidth
        flops = b * s * param1 * param2 * 2  # 示例
        peak_tflops = self._system_config.peak_tflops
        sol_latency = (flops / (peak_tflops * 1e12)) * 1000  # ms
        return sol_latency, 0.0, 0.0  # latency, power, energy
    
    if mode == common.DatabaseMode.SOL:
        lat, pwr, eng = get_sol(b, s, param1, param2)
        return PerformanceResult(latency=lat, energy=eng)
    
    # 查询实际数据（带插值）
    new_op_data = self._new_op_data
    if new_op_data is None:
        if mode == common.DatabaseMode.HYBRID:
            lat, pwr, eng = get_sol(b, s, param1, param2)
            return PerformanceResult(latency=lat, energy=eng)
        raise RuntimeError("New op data not loaded. Check data file path.")
    
    # 使用插值查询
    # 参考 query_context_mla 中的 interp_3d / interp_2d 用法
    try:
        result = new_op_data[param1][param2][b][s]
        return PerformanceResult(
            latency=result["latency"],
            energy=result["energy"],
        )
    except KeyError:
        # 插值逻辑（使用 self._interpolated_xxx_data 如果在 __init__ 中做了预插值）
        # 或者用 SOL fallback
        lat, pwr, eng = get_sol(b, s, param1, param2)
        return PerformanceResult(latency=lat, energy=eng)
```

### 2d. 插值支持（重要）

aiconfigurator 的核心能力之一是对未采集的参数组合做插值。在 `PerfDatabase.__init__()` 末尾，
为新 op 的数据做预插值：

```python
# perf_database.py — PerfDatabase.__init__() 末尾的插值块中
# 参考已有 attention/MLA 的插值代码模式

if self._new_op_data is not None:
    for param1 in self._new_op_data:
        data_dict = self._new_op_data[param1]
        target_x_list = [...]  # param2 的插值目标点
        target_y_list = [...]  # batch_size 的插值目标点
        target_z_list = [...]  # seq_len 的插值目标点
        
        self._new_op_data[param1] = interp_3d(
            data_dict, target_x_list, target_y_list, target_z_list
        )
```

## Step 3: 添加 Operation 子类

**文件**: `src/aiconfigurator/sdk/operations.py`

**关键**: 不要使用 `@dataclass`。Operation 子类使用普通的类继承模式。

```python
class ContextNewOp(Operation):
    """
    Context phase new operation.
    
    Models the Xxx computation during prefill.
    """
    
    def __init__(
        self,
        name: str,
        scale_factor: float,
        param1: int,
        param2: int,
    ) -> None:
        super().__init__(name, scale_factor)
        self._param1 = param1
        self._param2 = param2
        # 权重大小计算 (用于 memory estimation)
        self._weights = param1 * param2 * 2  # 假设 FP16, 2 bytes per element
    
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query context new op latency with energy data.
        
        Expected kwargs:
            x: total number of tokens (batch_size * seq_len)
        """
        x = kwargs.get("x")
        # 注意: 某些 op 需要分解 x 为 batch_size 和 seq_len
        # 此时需要从 kwargs 中获取更多参数
        
        result = database.query_new_op(
            b=kwargs.get("batch_size", 1),
            s=x,  # 或从 kwargs 获取
            param1=self._param1,
            param2=self._param2,
        )
        
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
        )
    
    def get_weights(self, **kwargs):
        """Get weight memory size for this operation (in bytes)."""
        return self._weights * self._scale_factor


class GenerationNewOp(Operation):
    """
    Generation phase new operation.
    
    Models the Xxx computation during decode.
    """
    
    def __init__(
        self,
        name: str,
        scale_factor: float,
        param1: int,
        param2: int,
    ) -> None:
        super().__init__(name, scale_factor)
        self._param1 = param1
        self._param2 = param2
        self._weights = param1 * param2 * 2
    
    def query(self, database: PerfDatabase, **kwargs) -> PerformanceResult:
        """
        Query generation new op latency with energy data.
        
        Expected kwargs:
            x: total number of tokens (= batch_size for generation)
            s: KV cache length (sequence length so far)
        """
        x = kwargs.get("x")  # batch_size
        s = kwargs.get("s")  # kv cache length
        
        result = database.query_new_op(
            b=x,
            s=s,
            param1=self._param1,
            param2=self._param2,
        )
        
        return PerformanceResult(
            latency=float(result) * self._scale_factor,
            energy=result.energy * self._scale_factor,
        )
    
    def get_weights(self, **kwargs):
        return self._weights * self._scale_factor
```

### Operation 命名约定

观察已有代码的命名规律：

| Context Op | Generation Op | 对应数据 |
|-----------|--------------|---------|
| `ContextAttention` | `GenerationAttention` | `context_attention_perf.txt` / `generation_attention_perf.txt` |
| `ContextMLA` | `GenerationMLA` | `context_mla_perf.txt` / `generation_mla_perf.txt` |
| `Mamba2` (含 phase 参数) | 同一个类 | `mamba2_perf.txt` |

**关于 `scale_factor`**:
- `scale_factor` 通常等于 `num_layers`（每层一个 op，总延迟 = 单层 * 层数）
- 对于只出现一次的 op（如 embedding），`scale_factor = 1`

### Operation 与 models.py 中 kwargs 的对应

`models.py` 中调用 `op.query(db, **kwargs)` 时传入的 kwargs 因 phase 不同而异：

```python
# models.py 中的调用模式（参考 BaseModel._evaluate_context_latency）

# Context phase:
for op in self.context_ops:
    latency += op.query(database, x=num_tokens, batch_size=batch_size, ...)

# Generation phase:
for op in self.generation_ops:
    latency += op.query(database, x=batch_size, s=kv_cache_length, ...)
```

## Step 4: 在 Model 类中使用新 Operation

**文件**: `src/aiconfigurator/sdk/models.py`

参考 `LLAMAModel` 或 `DeepSeekModel` 的 `__init__()` 方法，将新 Operation 添加到 `self.context_ops` 和 `self.generation_ops` 列表中。

```python
class NewModel(BaseModel):
    """New model implementation."""
    
    def __init__(self, param1, param2, *args) -> None:
        super().__init__(*args)
        
        self._param1 = param1
        self._param2 = param2
        
        h = self._hidden_size
        tp_size = self.config.tp_size
        pp_size = self.config.pp_size
        
        # ══════════════════════════════
        # Context Operations
        # ══════════════════════════════
        self.context_ops.extend([
            ops.Embedding("context_embedding", 1, self._vocab_size, h, 0.3),
            ops.ElementWise("context_add_norm_1", self._num_layers, 2 * h, 2 * h, 0.8),
            
            # 标准 GEMM (QKV projection 等)
            ops.GEMM("context_qkv_gemm", self._num_layers, ..., h, ...),
            
            # ← 新算子
            ops.ContextNewOp(
                "context_new_op",
                self._num_layers,  # scale_factor = num_layers
                param1=self._param1,
                param2=self._param2,
            ),
            
            # 其他标准 ops...
            ops.GEMM("context_o_gemm", self._num_layers, h, ..., ...),
        ])
        
        # ══════════════════════════════
        # Generation Operations
        # ══════════════════════════════
        self.generation_ops.extend([
            ops.Embedding("generation_embedding", 1, self._vocab_size, h, 0.3),
            
            # ← 新算子
            ops.GenerationNewOp(
                "generation_new_op",
                self._num_layers,
                param1=self._param1,
                param2=self._param2,
            ),
            
            # 其他标准 ops...
        ])
```

### 混合模型的层级处理

对于如 NemotronH 这样不同层有不同 Op 的模型，需要逐层判断：

```python
# 参考 NemotronHModel._build_ops()
for i, layer_type in enumerate(self._hybrid_config.hybrid_override_pattern):
    if layer_type == "M":  # Mamba layer
        self.context_ops.append(ops.Mamba2(...))
    elif layer_type == "*":  # Transformer layer
        self.context_ops.append(ops.ContextAttention(...))
    elif layer_type == "E":  # MoE layer
        self.context_ops.append(ops.MoE(...))
```

## Step 5: 放置性能数据文件

将 Phase 5 采集的 `*_perf.txt` 文件放到正确的目录：

```
src/aiconfigurator/systems/data/{system_name}/{backend}/{version}/
  ├── gemm_perf.txt                  # 已有
  ├── context_attention_perf.txt     # 已有
  ├── ...
  └── new_op_context_perf.txt        # ← 新增
```

## CLI 验证

```bash
# 完整的功能验证
aiconfigurator cli support --model-path <model> --system h200_sxm --backend trtllm

# 端到端建模测试
aiconfigurator cli default --model-path <model> --total-gpus 8 --system h200_sxm \
  --isl 4000 --osl 1000 --ttft 2000 --tpot 30
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `query` 返回 0 | 数据文件未加载或路径错误 | 检查 `PerfDataFilename` 和 `PerfDatabase.__init__` |
| `KeyError` in query | 查询参数组合不在数据中 | 添加插值支持或扩展数据采集范围 |
| 建模延迟与实际差距大 | Operation 分解不正确或遗漏了某些 Op | 回到 Phase 3 重新对齐 |
| `PerformanceResult` 相关错误 | 返回了 float 而非 PerformanceResult | 确保 `query()` 返回 `PerformanceResult(latency=..., energy=...)` |
| `_interp_3d` 返回 dict 不是 float | 插值返回 `{"latency":..., "energy":...}` | 需要手动 `PerformanceResult(result["latency"], energy=result.get("energy", 0.0))` |
| load 后 leaf 全是空 dict `{}` | `defaultdict(dict)` 的 `data[key]` 不抛 KeyError | 最内层用 `defaultdict()` 不是 `defaultdict(dict)`，或用 `if key in data` 判断 |
| 插值/query 维度错误 | dict 嵌套顺序与 `_interp_3d(x,y,z)` 不匹配 | 确认 x=第1层key, y=第2层key, z=第3层key |
| 模型推断 FP8 但 Op 不支持 | HF config 自动推断量化模式 | 在 `models.py` 强制 override 到支持的 dtype，检查框架的实际支持 |

## perf_database 关键规则

**严格复制最相似的已有 Op**。以 MLA 为参照：

1. **load 函数**: dict 结构必须层级一致（context 5层, generation 4层），leaf 用 `defaultdict()` 不是 `defaultdict(dict)`
2. **插值**: 在 `__init__` 的插值块中添加，用 `_extrapolate_data_grid()`，维度顺序跟 dict 嵌套一致
3. **query**: 用 `_interp_3d()` + `PerformanceResult` 包装，不要 fallback 到粒度不同的 Op
4. **quant key**: 如果 Op 不支持某些量化模式（如 FP8 KV cache），只注册支持的 key，在 model 层面强制 override
