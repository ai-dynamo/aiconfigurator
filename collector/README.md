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
  python collect_attn.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH

# Run DeepSeek MLP collector
python collect_mlp.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH

# Run DeepSeek DeepEP MoE collector (requires 2+ GPUs)
python collect_deepep_moe.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH \
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

## Kernel Power Profiling

The collector can measure kernel power consumption during benchmarking, which is used for power-aware configuration generation.

```bash
python collector/collect.py \
  --backend trtllm \
  --ops gemm attention_context attention_generation \
  --measure_power \
  --power_limits 700 500 300 \
  --kernel_power_measurement_duration 3

bash collector/collect_comm.sh --measure_power
```

### Current Support

- TensorRT-LLM backend: GEMM and attention kernels
- NCCL kernels (`collect_NCCL.py`)

### How It Works

- Calculate arithmetic intensity for each kernel configuration
- **Memory-bound kernels**: Measures actual power over a default 3-second window (duration configured with `--kernel_power_measurement_duration`)
- **Compute-bound kernels**: Power assumed to be equal power limit (power not measured)

NVML power/energy counters only update once 100 ms, which is significantly longer than most kernels. Thus, kernels are run repeatedly for `--kernel_power_measurement_duration` seconds for power measurement.

### Requirements

- pynvml (`pip install nvidia-ml-py`, already included in aiconfigurator dependencies) for measurement & GPU power limit control
- Setting power limits requires root access. Easiest way is to run in a Docker container run with `--cap-add SYS_ADMIN`.

### Relevant Columns in Result CSV

- `power_limit`: Current GPU power limit (W)
- `power`: The kernel's power consumption (W)
- `compute_bound`: Whether kernel is compute-bound (1) or memory-bound (0)

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aiconfigurator/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding 
of the GPU system and kernel optimization.

# Support Matrix
aiconfigurator 0.1.0  
trtllm: 0.20.0, 1.0.0rc3 on Hopper GPUs  
vllm: NA  
sglang: 0.5.1.post1 on Hopper GPUs
