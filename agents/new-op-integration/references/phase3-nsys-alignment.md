# Phase 3: nsys Profile 对齐

## 目标

验证 Phase 2 构建的 mock layer 的 GPU kernel 行为与完整模型 E2E 推理一致。
这是确保采集数据有效的关键步骤 — 如果 mock layer 启动的 kernel 与真实推理不同，采集的数据将不可靠。

## 基本原理

aiconfigurator 通过将完整推理分解为 Operation 来建模。每个 Operation（如 attention、MoE）对应一组 GPU kernel。
对齐的目标是：**mock layer 启动的 kernel 集合 ≈ E2E 推理中该 Operation 对应的 kernel 集合**。

## Step 1: Profile Mock Layer

```bash
# 对 Phase 2 的测试脚本做 profiling
nsys profile -o mock_layer_report \
  -t cuda,nvtx \
  --force-overwrite true \
  python3 test_xxx_op.py
```

> **注意**: 确保测试脚本中有 warmup 步骤，profiling 只采集稳定态。

## Step 2: Profile E2E Model

根据模型的部署方式选择不同的 profiling 策略：

### 方式 A: 单进程模型

```bash
nsys profile -o e2e_report \
  -t cuda,nvtx \
  --force-overwrite true \
  python3 run_full_model.py
```

### 方式 B: MPI/多进程模型（如 trtllm-serve）

多进程模型需要使用系统级采样，因为 worker 进程由框架内部 spawn：

```bash
# 1. 先启动服务
trtllm-serve /path/to/model --tp_size 8 &

# 2. 等待模型加载和 warmup 完成
sleep 300

# 3. 系统级 profiling
nsys profile -o e2e_report \
  -y 60 -d 20 \
  --sample=system-wide \
  --cpuctxsw=system-wide \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --force-overwrite true \
  trtllm-serve /path/to/model --tp_size 8
```

参数说明：
- `-y 60`: 延迟 60 秒后开始采集（等待 warmup）
- `-d 20`: 采集 20 秒
- `--sample=system-wide`: 系统级采样，捕获所有进程
- `--cuda-graph-trace=node`: 展开 CUDA Graph 节点

## Step 3: 提取 Kernel 统计

```bash
# Mock layer kernel 统计
nsys stats --report cuda_gpu_kern_sum mock_layer_report.nsys-rep > mock_kernels.txt

# E2E model kernel 统计
nsys stats --report cuda_gpu_kern_sum e2e_report.nsys-rep > e2e_kernels.txt
```

也可以在 Nsight Systems GUI 中打开 `.nsys-rep` 文件做可视化对比。

## Step 4: 对齐分析

### 对齐检查清单

| 检查项 | 通过条件 | 失败处理 |
|--------|----------|----------|
| **Kernel 名称匹配** | 主要 kernel 名称在两边都出现 | 检查 mock layer 是否调用了正确的底层实现 |
| **Kernel 数量** | 每个 forward 的 kernel 数量接近 | 如果 mock 多了 kernel，检查是否有不必要的初始化 |
| **Latency 比例** | 单次 forward 的总 latency 差距 < 2x | 检查是否有额外开销或缺失的计算 |
| **无缺失关键 kernel** | E2E 中的关键 kernel 在 mock 中都出现 | 可能缺少某个子模块初始化（如 cache manager） |
| **无多余 kernel** | mock 没有 E2E 中不存在的 kernel | 可能有不必要的计算或 debug 代码 |

### 如何识别哪些 kernel 属于目标 Op

E2E profile 中包含模型所有层的所有 Op 的 kernel。需要从中识别出目标 Op 的 kernel：

1. **利用 NVTX 标记**: 很多框架会用 NVTX 标记不同的 Op，在 GUI 中按 NVTX range 过滤
2. **利用 kernel 名称规律**: 比如 attention kernel 通常包含 `fmha`、`flash` 等关键词
3. **时序分析**: 对比 E2E 中一层的 kernel 序列和 mock 的 kernel 序列

### 对齐度量示例

```
Mock Layer:
  flash_fwd_kernel          : 1 次调用, 0.42ms
  void gemm_kernel<...>     : 2 次调用, 0.15ms + 0.15ms
  
E2E Model (单层的目标 Op):
  flash_fwd_kernel          : 1 次调用, 0.40ms  ← 匹配 ✓
  void gemm_kernel<...>     : 2 次调用, 0.16ms + 0.14ms  ← 匹配 ✓
  
对齐结论: 通过 (kernel 名称和数量匹配，latency 差距 < 10%)
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| nsys 输出无 GPU 数据 | MPI worker 进程未被捕获 | 使用 `--sample=system-wide` |
| Mock latency 远大于 E2E | Mock 缺少优化（如未用 CUDA Graph） | 确保 mock 测试中也用了 graph replay |
| Mock 缺少某些 kernel | 依赖模块未初始化 | 检查 KV Cache Manager、quantization 模块等 |
| E2E 中找不到目标 Op | NVTX 标记缺失或 kernel 融合 | 用 kernel 名称模式匹配；考虑框架可能做了 kernel fusion |
| Latency 差距很大但 kernel 名称正确 | 输入参数不同 | 确保 mock 的输入 shape 与 E2E 一致 |

## 何时可以跳过 Phase 3

- 如果 mock layer 直接使用了框架的官方实现类（方式 A），且参数配置完全一致，可以**简化**对齐验证
- 但建议至少做一次快速的 kernel 名称对比确认

## 正确的对齐做法（实战总结）

**不够的**: 只对比 kernel name 列表。
**正确的**: 创建完整 decoder layer，取出新 Op 对应的子模块（如 `decoder.self_attn`），跟 collector 的 mock layer 用相同输入做 benchmark：

```python
# 从 decoder layer 取出 attention 子模块
decoder = DecoderLayer(model_config=mc, layer_idx=0, ...)
decoder_attn = decoder.self_attn

# collector 独立创建的 attention
collector_attn = create_xxx_layer(tp_size=8)

# 相同输入，对比 latency
for name, attn in [("collector", collector_attn), ("decoder", decoder_attn)]:
    # ... 创建相同的 metadata, hidden_states, position_ids ...
    with benchmark_with_power(...) as res:
        pass
    print(f"{name}: {res['latency_ms']:.4f}ms")
```

**验证标准**: 两者 latency 差异 < 1%，CUDA graph 行为一致。

## 输出

Phase 3 完成后应有：
1. **对齐分析结论** — 记录 kernel 匹配情况和 latency 比较
2. **已知差异说明** — 如果有已知的差异，说明原因和对建模的影响
3. **对 mock layer 的修正**（如需要） — 基于对齐发现修正 Phase 2 的实现
