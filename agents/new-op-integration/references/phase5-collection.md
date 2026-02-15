# Phase 5: 性能数据采集

## 目标

收集新算子在目标 GPU 上各种参数组合下的 latency 数据，生成 `*_perf.txt` 文件供 aiconfigurator 查询。

## 架构概览

```
collector/
├── collect.py                  # 统一采集入口（多进程调度）
├── helper.py                   # benchmark_with_power(), log_perf()
├── common_test_cases.py        # 标准测试用例定义
├── trtllm/
│   ├── collect_mla.py          # MLA 采集（参考实现）
│   ├── collect_dsa.py          # DSA 采集（V3.2，实战示例）
│   ├── collect_moe.py
│   └── collect_xxx.py          # ← 新增你的采集脚本
└── ...
```

**关键**: 新 Op 的采集脚本必须适配 `collect.py` 的调度模式，不要写独立的并行脚本。

## Step 1: 编写采集脚本

**文件**: `collector/trtllm/collect_new_op.py`

需要导出 3 个东西：
1. `get_context_xxx_test_cases()` — 返回 test case 列表
2. `get_generation_xxx_test_cases()` — 同上
3. `run_xxx()` — 运行单个 test case

### Test case 格式

每个 test case 是一个 **positional arg list**，`run_xxx(*test_case)` 展开调用。参考 MLA 的格式：

```python
# MLA: [input_len, batch_size, output_len, dtype, num_heads, world_size, tp_size, ...]
# DSA: [seq_len, batch_size, tp_size, is_context, perf_filename]
```

关键原则：
- list 的顺序必须跟 `run_xxx()` 的参数顺序一致
- `perf_filename` 放在 list 里（不是硬编码在 run 函数中）

```python
def get_context_xxx_test_cases():
    test_cases = []
    for tp_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        for b in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            for s in [1, 16, 32, 64, 128, 256, 512, 1024, ...]:
                if b * s > 65536:
                    continue
                test_cases.append([s, b, tp_size, True, "xxx_context_perf.txt"])
    return test_cases
```

### run 函数

```python
def run_xxx(seq_len, batch_size, tp_size, is_context, perf_filename, device="cuda:0"):
    """Run a single benchmark. Args match test case list order."""
    # 1. 创建 mock layer（单卡模拟 TP）
    layer = create_xxx_layer(tp_size, device)  # local_heads = global / tp
    
    # 2. 准备输入 + metadata
    # ...
    
    # 3. benchmark_with_power
    def kernel_func():
        layer.forward(...)
    with benchmark_with_power(device, kernel_func, num_warmups=5, num_runs=6,
                              repeat_n=1, allow_graph_fail=True) as res:
        pass
    
    # 4. log_perf
    log_perf(item_list=[{...}], framework="TRTLLM", version=...,
             device_name=..., op_name="xxx_context", kernel_source="default",
             perf_filename=perf_filename, power_stats=res["power_stats"])
```

### TP 模拟（关键）

**不需要多卡**。跟 MLA 一致：
```python
local_num_heads = GLOBAL_HEADS // tp_size
mapping = Mapping(world_size=1, rank=0, tp_size=1)  # 始终单卡
# 把 local_num_heads 传给 layer 构造函数
```

## Step 2: 注册到 collect.py

在 `collect.py` 的 `collect_trtllm()` 函数中的 `collections` 列表添加：

```python
{
    "name": "trtllm",
    "type": "xxx_context",
    "module": "collector.trtllm.collect_xxx",
    "get_func": "get_context_xxx_test_cases",
    "run_func": "run_xxx",
},
{
    "name": "trtllm",
    "type": "xxx_generation",
    "module": "collector.trtllm.collect_xxx",
    "get_func": "get_generation_xxx_test_cases",
    "run_func": "run_xxx",
},
```

同时在 `--ops` choices 中添加 `"xxx_context"`, `"xxx_generation"`。

## Step 3: 运行采集

```bash
# 通过 collect.py 统一入口（推荐，自动多 GPU 并行）
cd collector/
python3 collect.py --backend trtllm --ops xxx_context xxx_generation

# 或单独运行（调试用）
cd collector/trtllm/
python3 collect_xxx.py --mode context --tp 8
```

## Step 4: 验证

采集完成后验证数据：
1. 检查数据行数是否符合预期（~110 context + ~181 generation per tp_size）
2. 用相同参数重新跑几个点，对比 latency 偏差应 < 2%
3. 安装后用 `aiconfigurator cli default` 确认端到端可运行

```python
# 快速验证：重跑几个点对比
run_xxx(4096, 1, 8, True, "/tmp/verify.txt")
# 对比 /tmp/verify.txt 最后一行 vs 已有数据
```

## Step 5: 放置数据文件

```bash
cp xxx_context_perf.txt src/aiconfigurator/systems/data/{system}/{backend}/{version}/
cp xxx_generation_perf.txt src/aiconfigurator/systems/data/{system}/{backend}/{version}/
```

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| CUDA OOM | batch*seq 太大 | test case 里加 `if b*s > limit: continue` |
| latency 波动大 | GPU 节流或后台进程 | 检查 `results["throttled"]`；确保 GPU 空闲 |
| `log_perf` 写入错乱 | 多进程写同一文件 | `log_perf` 用 `fcntl.flock` 文件锁，NFS 上不安全 |
| collect.py 找不到模块 | import path | 脚本顶部加 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` |
| MPI 初始化失败 | TRT-LLM 需要 MPI | `export OPAL_PREFIX=/opt/hpcx/ompi` |
