<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Introduction
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, you can try belowing commands. However, you need to prepare the env by yourself such as installing a specific trtllm version.
This process is not well verified, you need to debug sometimes.

# Preparation
Before collecting the data, make sure you own the whole node and no interfierence happens.
Next, please enable persistent-mode and lock frequency of the node. Make sure the cooling system of the node is working well.
```bash
sudo nvidia-smi -pm 1
```
```bash
sudo nvidia-smi -ac yyy,xxx
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
A script to set frequency:
```
#!/bin/bash

# Run nvidia-smi query and extract SM and Memory frequencies from Max Clocks
sm_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "SM " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")
mem_freq=$(nvidia-smi -q -i 0 | grep -A 4 "Max Clocks" | grep "Memory " | grep -o "[0-9]\+ MHz" | grep -o "[0-9]\+")

# Check if frequencies were successfully extracted
if [ -z "$sm_freq" ] || [ -z "$mem_freq" ]; then
    echo "Error: Could not extract SM or Memory frequency from Max Clocks."
    exit 1
fi

# Generate the command
echo "sudo nvidia-smi -ac $mem_freq,$sm_freq"
```
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
collect_comm.sh #all_reduce data will be collected using default trtllm backend
collect_comm.sh --all_reduce_backend vllm #all_reduce data will be collected using vllm backend
```
Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreudce, all_gather, all2all, reduce_scatter using nccl.
The generated file is comm_perf.txt and custom_all_reduce.txt.

# Collect gemm/attention/moe data/etc.

## Power Monitoring (Optional)

The collector supports GPU power monitoring during kernel execution using NVML. This feature is optional and disabled by default.

### Enable Power Monitoring
```bash
# Basic power monitoring
python3 collect.py --backend trtllm --measure_power

# With custom minimum duration (default: 1.0s)
python3 collect.py --backend trtllm --measure_power --power_test_duration_sec 2.0
```

### Options
- `--measure_power`: Enable NVML-based power monitoring (samples at 100ms intervals)
- `--power_test_duration_sec`: Minimum test duration for accurate power readings (default: 1.0s)

### Output
When power monitoring is enabled, performance CSV files will include additional columns:
- `power`: Average power consumption during kernel execution (Watts)
- `power_limit`: GPU power management limit (Watts)

**Example output:**
```csv
framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency,power,power_limit
TRTLLM,1.2.0,NVIDIA H200 SXM,gemm,torch_flow,float16,1024,4096,4096,0.234,523.4,700.0
```

### Requirements
Power monitoring requires:
- `pynvml` Python package: `pip install pynvml`
- NVML support (NVIDIA drivers)

If unavailable, a warning is logged and execution continues without power data.

### Notes
- Power monitoring adds minimal overhead (<1%)
- Kernel iterations are automatically adjusted to meet minimum duration for accurate measurements
- Backward compatible: without `--measure_power`, CSVs remain unchanged

## CUDA Graph Fallback Support

The `benchmark_with_power` helper function now supports graceful fallback to eager execution when CUDA graph capture fails. This is particularly useful for complex operations like MOE (Mixture of Experts) with large batch sizes.

### Features
- **Automatic fallback**: When `allow_graph_fail=True`, CUDA graph capture failures trigger eager execution instead of raising exceptions
- **Power measurement in both paths**: Power monitoring works correctly in both graph replay and eager execution modes
- **Memory safety**: Automatic `torch.cuda.empty_cache()` call on graph capture failure to prevent memory fragmentation
- **Transparency**: Results include `used_cuda_graph` flag to indicate which execution path was used

### Usage Example
```python
from helper import benchmark_with_power

def my_kernel():
    # Your kernel code here
    moe.forward(hidden_states, logits)

# Use benchmark_with_power with fallback support
with benchmark_with_power(
    device=device,
    kernel_func=my_kernel,
    num_warmups=3,
    num_runs=6,
    repeat_n=1,
    allow_graph_fail=True,  # Enable graceful fallback
) as results:
    latency = results["latency_ms"]
    power_stats = results["power_stats"]  # Available in both paths

    # Check which execution path was used
    if not results["used_cuda_graph"]:
        print("CUDA graph capture failed, used eager execution")
```

### When to Use
- **Complex operations**: MOE, dynamic memory patterns, or operations that may not be graph-compatible
- **Large batch sizes**: When graph capture may fail due to memory constraints
- **Development/debugging**: To ensure collection continues even if graph capture fails

### Backward Compatibility
- Default behavior unchanged: `allow_graph_fail=False` maintains existing behavior
- Existing collectors work without modifications
- Only opt-in when needed for specific use cases

## For TensorRT-LLM
```bash
python3 collect.py --backend trtllm
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aiconfigurator/systems/ folder to prepare the database including
how many files are needed accordingly.

## For SGLang

SGLang requires a **hybrid collection approach**:

### 1. Run unified collectors (GEMM, MLA, MoE, Normal Attention)
Suggest to start from lmsysorg docker image. Say, for 0.5.5.post3, we can use lmsysorg/sglang:v0.5.5.post3-cu126
```bash
python3 collect.py --backend sglang
```
This collects data for:
- GEMM operations (FP8, FP16, INT8, INT4)
- MLA (Multi-head Latent Attention) for context and generation
- MLA BMM (Batch Matrix Multiplication) operations
- MoE (Mixture of Experts) operations
- Normal attention operations

### 2. Run DeepSeek-specific collectors independently
Some SGLang collectors are **DeepSeek model-specific** and must be run separately:
```bash
cd sglang/
# Set model and output paths
export MODEL_PATH=/path/to/deepseek-v3
export OUTPUT_PATH=/path/to/output

# Run DeepSeek-specific attention collector
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
  python collect_wideep_attn.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH

# Run DeepSeek MLP collector
python collect_wideep_mlp.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH

# Run DeepSeek DeepEP MoE collector (requires 2+ GPUs)
python collect_wideep_deepep_moe.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH \
  --tp_size 2 --ep_size 2 --num_experts 256
```
See `sglang/README.md` for detailed documentation on these collectors.

### 3. Run DeepEP collector for distributed MoE data
For **DeepSeek V3** models with DeepEP MoE, collect distributed performance data:
```bash
# Follow instructions in deep_collector/README.md
# This requires multi-node setup for inter-node communication profiling
```
See `deep_collector/README.md` for complete multi-node setup instructions.

**Note**: SGLang collection requires more manual steps than TensorRT-LLM due to DeepSeek-specific operators and distributed MoE configurations.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aiconfigurator/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding
of the GPU system and kernel optimization.

# Known Issues

## NFS File Locking (Worker Deadlock)

**Symptom**: Collection stalls after a few test cases with no error messages.

**Cause**: `fcntl.flock()` doesn't work reliably on NFS. Workers deadlock when writing to shared output files.

**Solution**: Use `/tmp/` for output files, then copy results after collection.

# Support Matrix
aiconfigurator 0.1.0
trtllm: 0.20.0, 1.0.0rc3 on Hopper GPUs
vllm: NA
sglang: 0.5.5.post2, 0.5.5.post3 on Hopper GPUs
