---
name: new-op-integration
description: |
  Integrate new operations into aiconfigurator. Use when adding support for new models or attention mechanisms. 5-phase workflow: Analysis → Mock → Profile → Integrate → Collect.
---

# New Operation Integration

## 流程

```
Phase 1: 分析     → 理解配置参数
Phase 2: Mock     → 创建独立测试层
Phase 3: Profile  → 验证 kernel 对齐
Phase 4: 集成     → 添加 Operation 类
Phase 5: 采集     → 收集性能数据
```

---

## Phase 1: 分析

**目标**: 理解新算子的配置参数

**步骤**:
1. 获取 `config.json`
2. 识别新增字段
3. 添加架构映射 (`common.py`)
4. 添加解析逻辑 (`utils.py`)

**完成标志**: CLI 能识别新模型

---

## Phase 2: Mock

**目标**: 创建独立的算子层

**方式**:
- 优先用官方类 (`from tensorrt_llm._torch.models...`)
- 必要时自定义

**完成标志**: 能独立 forward

---

## Phase 3: Profile

**目标**: 验证 mock layer 与 E2E 行为一致

**方法**:
- 单进程: `nsys profile -t cuda,nvtx python test.py`
- MPI: `nsys profile --sample=system-wide ...`

**完成标志**: Kernel 名称匹配，latency 合理

---

## Phase 4: 集成

**目标**: 添加到 aiconfigurator SDK

**修改**:
- `operations.py` - Operation 类
- `perf_database.py` - 查询方法
- `models.py` - 模型逻辑

**完成标志**: `op.query(db, b, s)` 返回结果

---

## Phase 5: 采集

**目标**: 收集 benchmark 数据

**范围**:
- Context: batch 1-256, seq 1-32768
- Generation: batch 1-1024, seq 1-131071

**完成标志**: ~110 context + ~181 generation 数据点
