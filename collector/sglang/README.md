# SGLang Operator Performance Collection Tools

This directory contains scripts for collecting performance data of **Prefill-Decode (PD) disaggregated** DeepSeek model operators for the SGLang framework.

## Purpose

These scripts are designed to collect operator-level performance data for DeepSeek models in a PD-disaggregated serving architecture. They focus on the three largest modules in DeepSeek models:

1. **Attention (MLA)**: Multi-head Latent Attention mechanism
2. **MoE**: Mixture of Experts layers
3. **Shared Expert (MLP)**: Shared Multi-Layer Perceptron layers

The collected performance data can be used for performance modeling, scheduling optimization, and resource allocation in disaggregated serving systems.

## Overview

- **collect_wideep_attn.py**: Collects performance data for DeepSeek Attention (MLA) operators
- **collect_wideep_deepep_moe.py**: Collects performance data for DeepSeek MoE operators
- **collect_wideep_mlp.py**: Collects performance data for Shared Expert (MLP) operators

## Requirements

- SGLang framework: v0.5.0rc0 
```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host --name sglang lmsysorg/sglang:v0.5.0rc0-cu126
```
- DeepSeek model config (or use dummy weights)

## Execution Modes

The wideep collectors support two execution modes:

### Mode 1: Direct Execution

Run scripts directly with command-line arguments for single GPU collection:

```bash
# MLP
python collect_wideep_mlp.py --device cuda:0 --output-path /path/to/output/

# Attention (MLA)
python collect_wideep_attn.py --device cuda:0 --output-path /path/to/output/

# MoE
python collect_wideep_deepep_moe.py --device cuda:0 --output-path /path/to/output/
```

**Arguments:**
- `--device`: CUDA device (e.g., `cuda:0`, `cuda:1`)
- `--output-path`: Directory to save performance data files

### Mode 2: Framework Execution (collect.py)

Use the `collect.py` framework for integrated collection with other operators:

```bash
cd /path/to/collector/

# Run ALL sglang operators (13 total: 8 non-wideep + 5 wideep)
python collect.py --backend sglang

# Run wideep collectors only
python collect.py --backend sglang --ops wideep_mlp_context wideep_mlp_generation

# Run all wideep operators
python collect.py --backend sglang --ops wideep_mlp_context wideep_mlp_generation \
    wideep_mla_context wideep_mla_generation wideep_moe

# Mixed: non-wideep + wideep (all run in parallel across GPUs)
python collect.py --backend sglang --ops mla_bmm_gen_pre wideep_mlp_context
```

**All available operators (when no `--ops` specified):**

| Category | Operator | Description |
|----------|----------|-------------|
| Non-wideep | `gemm` | GEMM matrix multiplication |
| Non-wideep | `mla_context` | MLA prefill phase |
| Non-wideep | `mla_generation` | MLA decode phase |
| Non-wideep | `mla_bmm_gen_pre` | MLA BMM gen pre |
| Non-wideep | `mla_bmm_gen_post` | MLA BMM gen post |
| Non-wideep | `moe` | MOE operator |
| Non-wideep | `attention_context` | Standard Attention prefill |
| Non-wideep | `attention_generation` | Standard Attention decode |
| Wideep | `wideep_mla_context` | Wideep MLA prefill |
| Wideep | `wideep_mla_generation` | Wideep MLA decode |
| Wideep | `wideep_mlp_context` | Wideep MLP prefill |
| Wideep | `wideep_mlp_generation` | Wideep MLP decode |
| Wideep | `wideep_moe` | Wideep MOE |

**Note:** Both non-wideep and wideep operators run in parallel across multiple GPUs. Wideep operators use subprocess-based GPU isolation (via `CUDA_VISIBLE_DEVICES`) to prevent NCCL/distributed initialization conflicts while maintaining parallel execution.

## General Configuration

All scripts save results to the same output directory. Modify `output_path` in each script to your desired location:
```python
output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
```


## 1. Attention Operator Collection (collect_wideep_attn.py)

### Features
- Tests different attention backends (flashinfer, fa3)
- Tests various batch sizes, sequence lengths, and head numbers
- Supports both prefill and decode phases
- Optional dummy weights mode for fast testing

### Usage

#### Direct Mode
```bash
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_wideep_attn.py --device cuda:0 --output-path /path/to/output/
```

#### Framework Mode
```bash
python collect.py --backend sglang --ops wideep_mla_context wideep_mla_generation
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model 
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
- `wideep_context_mla_perf.txt`: Prefill phase performance data
- `wideep_generation_mla_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency
```

## 2. MoE Operator Collection (collect_wideep_deepep_moe.py)

### Features
- Tests DeepEP MoE operator performance
- Supports different expert number configurations
- Tests both prefill and decode phases
- Supports power-law and uniform distribution modes

### Usage

#### Direct Mode
```bash
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_wideep_deepep_moe.py --device cuda:0 --output-path /path/to/output/
```

#### Framework Mode
```bash
python collect.py --backend sglang --ops wideep_moe
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model

#### Modify Configuration

**Multi-GPU Parallel Mode**: The script supports parallel execution across multiple GPUs using subprocess isolation. Each GPU runs a different EP configuration simultaneously.

Edit the configuration at the bottom of the script:
```python
# Configuration variables (modify as needed)
num_experts_list = [128, 64, 32, 16, 8, 4, 2, 1]  # List of expert counts to simulate different EP sizes

# Server arguments (per-GPU subprocess)
server_args = ServerArgs(
    tp_size=1,                   # Each subprocess uses 1 GPU
    ep_size=1,                   # Each subprocess uses 1 GPU
)
```

**Simulating Different EP Configurations**:

The `num_experts` parameter is used to simulate different expert parallel (EP) sizes. With `tp_size=1` and `ep_size=1` (single GPU):

- `num_experts=128` → simulates **EP 2** (256 / 128 = 2)
- `num_experts=64` → simulates **EP 4** (256 / 64 = 4)
- `num_experts=32` → simulates **EP 8** (256 / 32 = 8)
- `num_experts=16` → simulates **EP 16** (256 / 16 = 16)
- `num_experts=8` → simulates **EP 32** (256 / 8 = 32)
- `num_experts=4` → simulates **EP 64** (256 / 4 = 64)
- `num_experts=2` → simulates **EP 128** (256 / 2 = 128)
- `num_experts=1` → simulates **EP 256** (256 / 1 = 256)

The simulated EP size is calculated as: `simulated_ep_size = 256 / num_experts * ep_size`

### Test Parameters
- Number of experts: Configurable (suggested: 16, 32, 64, 128, 256)
- Number of tokens: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Distribution mode: `power_law`  or `uniform`

### Output
Results are saved to:
- `wideep_context_moe_perf.txt`: Prefill phase performance data
- `wideep_generation_moe_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency
```

## 3. MLP Operator Collection (collect_wideep_mlp.py)

### Features
- Tests DeepSeek V2/V3 MLP operator performance
- Supports FP8 quantization
- Separately tests prefill (context, direct execution) and decode (generation, CUDA Graph) phases

### Usage

#### Direct Mode
```bash
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_wideep_mlp.py --device cuda:0 --output-path /path/to/output/
```

#### Framework Mode
```bash
python collect.py --backend sglang --ops wideep_mlp_context wideep_mlp_generation
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model (default: `/deepseek-v3`)

### Test Parameters
The script automatically tests the following configurations for both prefill and decode phases:
- Quantization: FP8 block quantization
- Number of tokens: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
- Hidden size: 7168
- Intermediate size: 2048

### Test Phases
1. **Prefill Phase**: Direct execution without CUDA Graph
2. **Decode Phase**: CUDA Graph enabled for optimized performance

### Output
Results are saved to:
- `wideep_context_mlp_perf.txt`: Prefill phase performance data
- `wideep_generation_mlp_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,quant_type,num_token,hidden_size,intermediate_size,avg_ms
```

