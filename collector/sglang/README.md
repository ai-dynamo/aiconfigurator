# SGLang Operator Performance Collection Tools

This directory contains scripts for collecting performance data of **Prefill-Decode (PD) disaggregated** DeepSeek model operators for the SGLang framework.

## Purpose

These scripts are designed to collect operator-level performance data for DeepSeek models in a PD-disaggregated serving architecture. They focus on the three largest modules in DeepSeek models:

1. **Attention (MLA)**: Multi-head Latent Attention mechanism
2. **MoE**: Mixture of Experts layers
3. **Shared Expert (MLP)**: Shared Multi-Layer Perceptron layers

The collected performance data can be used for performance modeling, scheduling optimization, and resource allocation in disaggregated serving systems.

## Overview

- **collect_attn.py**: Collects performance data for DeepSeek Attention (MLA) operators
- **collect_deepep_moe.py**: Collects performance data for DeepSeek MoE operators
- **collect_mlp.py**: Collects performance data for Shared Expert (MLP) operators

## Requirements

- SGLang framework: v0.5.0rc0 
```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host --name sglang lmsysorg/sglang:v0.5.0rc0-cu126
```
- DeepSeek model config (or use dummy weights)

## General Configuration

All scripts save results to the same output directory. Modify `output_path` in each script to your desired location:
```python
output_path = "/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.0/"
```


## 1. Attention Operator Collection (collect_attn.py)

### Features
- Tests different attention backends (flashinfer, fa3)
- Tests various batch sizes, sequence lengths, and head numbers
- Supports both prefill and decode phases
- Optional dummy weights mode for fast testing

### Usage

#### Basic Run with dummy weight
```bash
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_attn.py
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
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_deepep_moe.py
```

#### Environment Variables
- `DEEPSEEK_MODEL_PATH`: Path to DeepSeek model

#### Modify Configuration

**Important**: DeepEP MoE collection requires **at least 2 GPUs** for distributed execution.

Edit the configuration at the bottom of the script:
```python
# Configuration variables (modify as needed)
num_experts=256,             # Number of experts to simulate different EP configurations

# Server arguments
server_args = ServerArgs(
    tp_size=2,                   # Tensor parallel size
    ep_size=2,                   # Expert parallel size
)


```

**Simulating Different EP Configurations**:

The `num_experts` parameter in `MoEBenchArgs` is used to simulate different expert parallel (EP) sizes. For example, when using 2 GPUs with `tp_size=2` and `ep_size=2`:

- `num_experts=256` → simulates **EP 2** (256 experts / 2 = 128 experts per GPU)
- `num_experts=128` → simulates **EP 4** (128 experts / 4 = 32 experts per GPU)
- `num_experts=64` → simulates **EP 8** (64 experts / 8 = 8 experts per GPU)
- `num_experts=32` → simulates **EP 16** (32 experts / 16 = 2 experts per GPU)
- `num_experts=16` → simulates **EP 32**
- `num_experts=8` → simulates **EP 64**
- `num_experts=4` → simulates **EP 128**
- `num_experts=2` → simulates **EP 256** (2 experts / 2 = 1 experts per GPU)

The actual `moe_ep_size` is automatically calculated based on the relationship between `num_experts` and the base EP size (256).

### Test Parameters
- Number of experts: Configurable (suggested: 16, 32, 64, 128, 256)
- Number of tokens: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
- Distribution mode: `power_law`  or `uniform`

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
- Separately tests prefill (context, direct execution) and decode (generation, CUDA Graph) phases

### Usage

#### Basic Run
```bash
export DEEPSEEK_MODEL_PATH=/path/to/deepseek-v3
python collect_mlp.py
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
- `context_mlp_perf.txt`: Prefill phase performance data
- `generation_mlp_perf.txt`: Decode phase performance data

Output format:
```
framework,version,device,op_name,kernel_source,quant_type,num_token,hidden_size,intermediate_size,avg_ms
```

