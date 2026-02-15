---
name: dsa-op-integration
description: |
  Integrate new DSA (DeepSeek Sparse Attention) operations into aiconfigurator. Use when adding support for DeepSeek V3.2 or similar models with sparse attention. Covers: (1) Model architecture analysis, (2) Mock model layer creation, (3) nsys profiling and alignment, (4) aiconfigurator integration, (5) Performance data collection.
---

# New Operation Integration for aiconfigurator

通用流程：为新模型/新算子添加 aiconfigurator 支持。

## 五阶段流程

```
Phase 1: 分析 → Phase 2: Mock → Phase 3: 验证 → Phase 4: 集成 → Phase 5: 采集
```

---

## Phase 1: 模型架构分析

**目标**: 理解新模型/新算子的配置参数

**步骤**:
1. 获取 `config.json` (HuggingFace 或本地)
2. 识别新增的配置字段
3. 添加架构映射 (`common.py`)
4. 添加配置解析 (`utils.py`)

**输出**:
- Config dataclass
- 架构映射规则
- 解析函数

---

## Phase 2: Mock Model Layer

**目标**: 创建独立的算子层用于测试和 profiling

**两种方式**:

| 方式 | 优点 | 缺点 |
|------|------|------|
| **官方类** | 准确、维护简单 | 需要找到正确的类 |
| **自定义** | 完全可控 | 需要理解内部实现 |

**关键点**:
- 确认输入/输出格式
- 确认依赖的 cache manager
- 处理边界条件 (dtype, shape)

**验证**: 能独立 forward 不报错

---

## Phase 3: nsys Profiling 对齐

**目标**: 验证 mock layer 与 E2E 模型行为一致

**对比项**:
- [ ] Kernel 名称匹配
- [ ] Latency 在合理范围内
- [ ] 无缺失/多余的 kernel

**Profiling 方法**:

| 场景 | 命令 |
|------|------|
| 单进程 | `nsys profile -t cuda,nvtx python test.py` |
| MPI spawn | `nsys profile --sample=system-wide ...` |

**输出**: kernel 时间分布对比表

---

## Phase 4: SDK 集成

**目标**: 将算子添加到 aiconfigurator

**修改文件**:

| 文件 | 内容 |
|------|------|
| `operations.py` | Operation 类 |
| `perf_database.py` | 数据库查询方法 |
| `models.py` | 模型集成逻辑 |

**Operation 类模板**:
```python
@dataclass
class ContextXxx(Operation):
    name: str = "context_xxx"
    scale_factor: float = 1.0
    # ... 参数字段
    
    def query(self, db, batch_size, s, **kwargs) -> float:
        # 先查数据库，无数据则估算
        pass
```

---

## Phase 5: 性能数据采集

**目标**: 收集 benchmark 数据填充数据库

**Collector 要素**:
1. 算子初始化 (使用 Phase 2 的 mock layer)
2. 标准测试用例 (参考 MLA collector)
3. 输出格式 (CSV，aiconfigurator 兼容)

**标准测试范围**:

| Phase | Batch Sizes | Seq Lens | Limit |
|-------|-------------|----------|-------|
| Context | 1,2,4,8,...,256 | 1-32768 | b×s ≤ 65536 |
| Generation | 1,2,4,...,1024 | 1-131071 | b×s ≤ 8M |

**输出文件**: `xxx_perf.txt` (aiconfigurator 格式)

---

## 检查清单

**Phase 1 完成标志**:
- [ ] Config 解析正确
- [ ] 架构映射生效

**Phase 2 完成标志**:
- [ ] Mock layer 可独立运行
- [ ] 输出格式正确

**Phase 3 完成标志**:
- [ ] Kernel 对齐验证通过
- [ ] Latency 合理

**Phase 4 完成标志**:
- [ ] CLI 识别新模型
- [ ] 性能查询返回结果

**Phase 5 完成标志**:
- [ ] 数据点数量达标 (参考 MLA)
- [ ] 数据格式正确

---

## 注意事项

1. **先读官方实现** - 确认 tensorrt_llm 是否已支持
2. **使用 dummy weights** - `load_format='dummy'` 跳过权重加载
3. **Profiling 需要正确的工具** - MPI 用 system-wide sampling
4. **数据格式要兼容** - 参考 MLA collector 的输出格式
