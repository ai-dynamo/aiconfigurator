# SGLang Operator Performance Collection Tools

This directory contains scripts for collecting performance data of **Prefill-Decode (PD) disaggregated** DeepSeek model operators for the SGLang framework.

## Purpose

These scripts are designed to collect operator-level performance data for DeepSeek models in a PD-disaggregated serving architecture. They focus on the three largest modules in DeepSeek models:

1. **Attention (MLA)**: Multi-head Latent Attention mechanism
2. **MoE**: Mixture of Experts layers
3. **Shared Expert (MLP)**: Shared Multi-Layer Perceptron layers

The collected performance data can be used for performance modeling, scheduling optimization, and resource allocation in disaggregated serving systems.

## Overview

- **collect_mla_module.py**: Collects performance data for MLA and DSA attention module operators
- Large-EP expert compute is modeled from stock `moe_perf`; measured
  communication is collected by `../wideep/sglang/collect_moe_a2a.py`.

## Requirements

- Stock SGLang collectors: v0.5.14

```bash
docker run -itd --shm-size 32g --gpus all --ipc=host --network=host --name sglang lmsysorg/sglang:v0.5.14-cu130
```

- WideEP MoE collector: the independent v0.5.10 image declared in
  `../framework_manifest.yaml`; WideEP MLA is not registered for stock v0.5.14.
- DeepSeek model config (or use dummy weights)

## Execution Modes

Stock diagnostics and the independent WideEP MoE collector use these modes:

### Mode 1: Direct Diagnostic Execution

Run scripts directly with command-line arguments for single GPU diagnostics.
The MLA command exercises the stock 0.5.14 module implementation; it is not a
registered or accepted WideEP MLA collector:

```bash
# Attention (MLA/DSA Module)
python collect_mla_module.py --mode context --attn-type mla

```

**Arguments:**
- `--device`: CUDA device (e.g., `cuda:0`, `cuda:1`)
- `--output-path`: Directory to save performance data files

### Mode 2: Framework Execution (collect.py)

Use the `collect.py` framework for integrated collection with other operators:

```bash
cd /path/to/collector/

# Run all stock SGLang operators
python collect.py --backend sglang

# Run stock DSA module operators in the v0.5.14 container
python collect.py --backend sglang --ops dsa_context_module dsa_generation_module

# Run the large-EP MoE compute operator separately in the v0.5.10 container
python collect.py --backend sglang --ops moe_ep

# Mixed: operations run in order; cases within each operation use the GPU pool
python collect.py --backend sglang --ops mla_bmm_gen_pre dsa_context_module
```

**Selected operators (`moe_ep` requires its separate v0.5.10 run):**

| Category | Operator | Description |
|----------|----------|-------------|
| Kernel | `gemm` | GEMM matrix multiplication |
| Kernel | `mla_context` | MLA prefill phase |
| Kernel | `mla_generation` | MLA decode phase |
| Kernel | `mla_bmm_gen_pre` | MLA BMM gen pre |
| Kernel | `mla_bmm_gen_post` | MLA BMM gen post |
| Kernel | `moe` | MOE operator |
| Kernel | `attention_context` | Standard Attention prefill |
| Kernel | `attention_generation` | Standard Attention decode |
| Module | `dsa_context_module` | DSA module prefill (DeepSeek-V3.2, GLM-5) |
| Module | `dsa_generation_module` | DSA module decode (DeepSeek-V3.2, GLM-5) |
| Wideep | `moe_a2a` | Large-EP MoE dispatch/combine communication |

**Note:** Requested operators run sequentially. Cases within one operator are
distributed across the selected GPU workers. Module-level operators use
subprocess-based GPU isolation (via `CUDA_VISIBLE_DEVICES`) to prevent
NCCL/distributed initialization conflicts.

## General Configuration

Direct scripts save staging results to one output directory. Keep that staging
directory outside the packaged perf-data tree; `collect.py` finalizes accepted
files as parquet, and each file is then published to the family selected by
`collector/op_backend_catalog.yaml`.

Modify `output_path` in each direct script to your desired staging location:
```python
output_path = "/aiconfigurator/collector_output/h100_sxm/sglang/0.5.14/"
```


## 1. Stock Attention Module Diagnostics (`collect_mla_module.py`)

This direct script retains historical `wideep_*` output filenames for local
diagnostics, but those files are not supported WideEP 0.5.10 artifacts. Use the
registered stock MLA/DSA ops through `collect.py` for accepted collection.

### Features
- Unified MLA (DeepSeek-V3) and DSA (DeepSeek-V3.2, GLM-5) benchmarking
- SM-gated precision sweep (bfloat16 + fp8 on Hopper+)
- Tests various batch sizes, sequence lengths, and head numbers
- Supports both prefill and decode phases
- Optional dummy weights mode for fast testing

### Usage

#### Direct Mode
```bash
# MLA context phase
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
    python collect_mla_module.py --mode context --attn-type mla

# DSA generation phase
SGLANG_LOAD_FORMAT=dummy SGLANG_TEST_NUM_LAYERS=2 \
    python collect_mla_module.py --mode generation --attn-type dsa
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
Direct diagnostic results use the historical filenames:
- `wideep_context_mla_perf.txt` / `dsa_context_module_perf.txt`: Prefill phase performance data
- `wideep_generation_mla_perf.txt` / `dsa_generation_module_perf.txt`: Decode phase performance data

Output format:
```csv
framework,version,device,op_name,kernel_source,model,architecture,mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency
```

## 2. Large-EP communication collection

Use `collector/wideep/sglang/collect_moe_a2a.py` for measured dispatch/combine communication. Large-EP local expert compute has no dedicated collector or table; it is estimated from stock `moe_perf`.
