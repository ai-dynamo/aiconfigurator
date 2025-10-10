# SGLang Operator Performance Collection Tools

This directory contains three scripts for collecting performance data of DeepSeek model operators for the SGLang framework.

## Overview

- **collect_attn.py**: Collects performance data for DeepSeek Attention (MLA) operators
- **collect_deepep_moe.py**: Collects performance data for DeepSeek MoE operators
- **collect_mlp.py**: Collects performance data for MLP operators

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- SGLang framework
- DeepSeek model weights (or use dummy weights)

## 1. Attention Operator Collection (collect_attn.py)

### Features
- Tests different attention backends (flashinfer, fa3)
- Tests various batch sizes, sequence lengths, and head numbers
- Supports both prefill and decode phases
- Optional dummy weights mode for fast testing

### Usage

#### Basic Run
```bash
cd /home/scratch.aichenf_wwfo/aiconfigurator/collector/sglang
python collect_attn.py
```

#### Quick Test with Dummy Weights (Recommended)
```bash
# Use dummy weights, load only 2 layers, test layer 0
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 SGLANG_TEST_LAYER=0 python collect_attn.py
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model (default: `/home/scratch.aichenf_wwfo/scripts/deepseek-v3`)
- `SGLANG_LOAD_FORMAT`: Load format, set to `dummy` to skip weight loading
- `SGLANG_TEST_NUM_LAYERS`: Load only specified number of layers (with dummy mode)
- `SGLANG_TEST_LAYER`: Layer index to test (default: 0)

### Test Parameters
The script automatically tests the following configuration combinations:
- Attention backends: `flashinfer`, `fa3`
- Head numbers: 128, 64, 32, 16
- Batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
- Sequence lengths: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384

### Output
Results are saved to:
- `context_mla_perf.txt`: Prefill phase performance data
- `generation_mla_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency
```

## 2. MoE Operator Collection (collect_deepep_moe.py)

### Features
- Tests DeepEP MoE operator performance
- Supports different expert number configurations
- Tests both prefill and decode phases
- Supports power-law and uniform distribution modes

### Usage

#### Basic Run
```bash
python collect_deepep_moe.py
```

#### Modify Configuration
Edit the configuration at the bottom of the script:
```python
server_args = ServerArgs(
    model_path=model_path,
    dtype="auto",
    device="cuda",
    load_format="dummy",        # Use dummy weights
    tp_size=2,                   # Tensor parallel size
    ep_size=2,                   # Expert parallel size
    # ... other parameters
)

bench_args = MoEBenchArgs(
    num_warmup=3,
    num_iterations=10,
    test_layer=3,                # Test layer 3
    num_experts=16,              # Number of experts: 16, 32, 64, 128, 256
)
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model

### Test Parameters
- Number of experts: Configurable (suggested: 16, 32, 64, 128, 256)
- Number of tokens: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Distribution mode: `power_law` (default) or `uniform`

### Output
Results are saved to:
- `context_moe_perf.txt`: Prefill phase performance data
- `generation_moe_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency
```

## 3. MLP Operator Collection (collect_mlp.py)

### Features
- Tests DeepSeek V2/V3 MLP operator performance
- Supports FP8 quantization
- Tests prefill (direct execution) and decode (CUDA Graph) modes

### Usage

#### Basic Run
```bash
python collect_mlp.py
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model (default: `/home/scratch.aichenf_wwfo/scripts/deepseek-v3`)

### Test Parameters
The script automatically tests the following configurations:
- Quantization: FP8
- Number of tokens: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Hidden size: 7168
- Intermediate size: 2048

### Output
Results are saved to:
- `prefill_mlp_perf.txt`: Prefill phase performance data
- `generation_mlp_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,quant_type,num_token,hidden_size,intermediate_size,avg_ms
```

## General Configuration

### Modify Output Path
Find the `output_path` variable in each script and modify it:

**collect_attn.py (line 606):**
```python
output_path = "/home/scratch.aichenf_wwfo/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
```

**collect_deepep_moe.py (line 616):**
```python
output_path = "path/to/aiconfigurator/src/aiconfigurator/systems/data/h200_sxm/sglang/0.5.0/"
```

**collect_mlp.py (line 409):**
```python
output_path = "/home/scratch.aichenf_wwfo/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
```

### Adjust Test Parameters
The `get_*_test_cases()` function in each script defines the test cases and can be modified as needed:
- `get_attention_test_cases()` in collect_attn.py
- `get_moe_prefill_test_cases()` / `get_moe_decode_test_cases()` in collect_deepep_moe.py
- `get_mlp_test_cases()` in collect_mlp.py

### Warmup and Iterations
Adjust in the configuration section of each script:
```python
num_warmup = 3      # Number of warmup iterations
num_iterations = 10 # Number of test iterations
```

## Recommended Workflow

1. **Quick Verification**: Use dummy weights for fast testing
   ```bash
   SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 python collect_attn.py
   python collect_mlp.py
   python collect_deepep_moe.py
   ```

2. **Full Testing**: Use real weights for complete performance testing
   ```bash
   export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
   python collect_attn.py
   python collect_mlp.py
   python collect_deepep_moe.py
   ```

3. **Result Analysis**: Check the performance data files in the output directory

## Notes

1. **Memory Limits**: Some configuration combinations may exceed GPU memory; the scripts will automatically skip these configurations
2. **Parallel Execution**: The MoE script supports multi-GPU parallelism via `tp_size` and `ep_size` configuration
3. **CUDA Graph**: Decode phase uses CUDA Graph by default for more accurate performance data
4. **Result Appending**: Output files support append mode; multiple runs will not overwrite previous results

## Troubleshooting

### Issue: CUDA Out of Memory
- Solution: Reduce the test range for batch size or sequence length
- Use dummy weights to reduce model memory usage

### Issue: Distributed Initialization Failure
- Solution: Scripts automatically clean up distributed environment; if issues persist, restart the Python process

### Issue: Slow Model Loading
- Solution: Use `SGLANG_LOAD_FORMAT=dummy` to skip weight loading

## Contact

For questions or suggestions, please contact the development team.
