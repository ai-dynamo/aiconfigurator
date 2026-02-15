# Phase 3: nsys Profiling 对齐模板

## Mock Layer Profiling

```bash
nsys profile -o mock_layer \
  -t cuda,nvtx \
  python3 test_op.py
```

## E2E Model Profiling

### 单进程模型

```bash
nsys profile -o e2e_model \
  -t cuda,nvtx \
  python3 run_model.py
```

### MPI Spawn 模型 (如 trtllm-serve)

```bash
# 先启动服务
trtllm-serve /path/to/model --tp_size 8 &

# 等待 warmup
sleep 300

# 系统级采样
nsys profile -o e2e_model \
  -y 60 -d 20 \
  --sample=system-wide \
  --cpuctxsw=system-wide \
  -t cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  trtllm-serve ...
```

## 对比分析

```bash
# 提取 kernel 统计
nsys stats --report cuda_gpu_kern_sum mock_layer.nsys-rep
nsys stats --report cuda_gpu_kern_sum e2e_model.nsys-rep
```

## 对齐检查清单

| 检查项 | 通过条件 |
|--------|----------|
| Kernel 名称 | 主要 kernel 名称匹配 |
| Latency | 差距 < 2x |
| 缺失 kernel | 无关键 kernel 缺失 |
| 额外 kernel | 无不必要的 kernel |

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| nsys 无 GPU 数据 | MPI worker 未被捕获 | 用 `--sample=system-wide` |
| Latency 差距过大 | 实现不一致 | 检查算法逻辑 |
| 缺少 kernel | 依赖未初始化 | 检查 cache manager |
