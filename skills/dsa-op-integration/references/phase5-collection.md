# Phase 5: 性能数据采集模板

## Collector 框架

```python
def run_op(input_len, batch_size, is_context, ...):
    """运行算子 benchmark"""
    
    # 1. 创建算子 (使用 Phase 2 的 mock layer)
    layer = create_xxx_layer(...)
    
    # 2. 创建 cache manager (如需要)
    cache = create_cache_manager(...)
    
    # 3. 准备输入
    hidden = torch.randn([num_tokens, hidden_size], ...)
    
    # 4. Warmup
    _ = layer(hidden, ...)
    torch.cuda.synchronize()
    
    # 5. Benchmark
    with benchmark(layer, num_iterations=6) as results:
        pass
    
    # 6. 记录结果
    log_perf({
        "batch_size": batch_size,
        "isl": input_len,
        "step": 0 if is_context else 1,
        "latency": results["median_ms"],
    }, filename="xxx_perf.txt")
```

## 标准测试用例

### Context Phase

```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
seq_lens = [1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 
            3072, 4096, 6144, 8192, 10240, 12288, 16384, 32768]
limit = batch_size * seq_len <= 65536
```

### Generation Phase

```python
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
seq_lens = [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023,
            2047, 4095, 8191, 16383, 32767, 65535, 131071]
limit = batch_size * seq_len <= 8388608
```

## 输出格式

```
framework,version,device,op_name,kernel_source,dtype,param1,param2,batch_size,isl,tp_size,step,latency
TRTLLM,1.2.0rc5,NVIDIA H20-3e,xxx_context,default,bfloat16,64,2048,1,4096,1,0,40.77
```

## 运行命令

```bash
# Context phase
python3 collect_xxx.py --mode context \
  --batch_sizes 1,2,4,8,16,32,64,128,256 \
  --seq_lens 1,16,32,...,32768

# Generation phase
python3 collect_xxx.py --mode generation \
  --batch_sizes 1,2,4,...,1024 \
  --seq_lens 1,3,7,...,131071
```

## 数据验证

```python
import csv

with open("xxx_perf.txt") as f:
    rows = list(csv.DictReader(f))

# Context: 约 110 个数据点
# Generation: 约 181 个数据点
```
