<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# aiconfigurator

In disaggregated serving, configuring an effective deployment is challenging: you need to decide how many prefill and decode
workers to run, and the parallelism for each worker. Combined with SLA targets for TTFT (Time to First Token) and
TPOT (Time per Output Token), optimizing throughput at a given latency becomes even more complex.

`aiconfigurator` helps you find a strong starting configuration for disaggregated serving. Given your model, GPU
count, and GPU type, it searches the configuration space and generates configuration files you can use for deployment with Dynamo.

The tool models LLM inference using collected data for a target machine and framework. It evaluates thousands of
configurations and runs anywhere via the CLI and the web app.

Let's get started.

## Build and Install

### Install from PyPI

```bash
pip3 install aiconfigurator
```

### Build and Install from Source

```bash
# 1. Install Git LFS
apt-get install git-lfs  # (Linux)
# brew install git-lfs   # (macOS)

# 2. Clone the repo
git clone https://github.com/ai-dynamo/aiconfigurator.git

# 3. Create and activate a virtual environment
python3 -m venv myenv && source myenv/bin/activate # (requires Python 3.9 or later)

# 4. Install aiconfigurator
pip3 install .
```

### Build with Docker

```bash
# This will create a ./dist/ folder containing the wheel file
docker build -f docker/Dockerfile --no-cache --target build -t aiconfigurator:latest .
docker create --name aic aiconfigurator:latest && docker cp aic:/workspace/dist dist/ && docker rm aic
```

## Run

### CLI

```bash
aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm
```

- With **three basic arguments**, it prints the estimated best deployment and the deployment details.
- Use `--save_dir DIR` to generate framework configuration files for Dynamo.
- Use `-h` for more options and customization.

```text
********************************************************************************
*                      Dynamo AIConfigurator Final Results                     *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: QWEN3_32B (is_moe: False)
    Total GPUs: 512
    I/O Length (tokens): Input=4000, Output=500
    SLA Target: TTFT <= 300.0ms, TPOT <= 10.0ms
  ----------------------------------------------------------------------------
  Overall best system chosen: disagg at 812.48 tokens/s/gpu (2.39x better)
    - Agg Actual Best: 340.48 tokens/s/gpu  100.83 tokens/s/user | TTFT: 188.91ms TPOT: 9.92ms
    - Disagg Actual Best: 812.48 tokens/s/gpu  109.12 tokens/s/user | TTFT: 276.94ms TPOT: 9.16ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
               QWEN3_32B Pareto Frontier: tokens/s/gpu vs tokens/s/user
      ┌────────────────────────────────────────────────────────────────────────┐
1600.0┤ dd Disagg                                                              │
      │ aa Agg                                                                 │
      │ XX Best                                                                │
      │                                                                        │
1333.3┤    a                                                                   │
      │     a                                                                  │
      │      aaaa    d                                                         │
      │          a    ddddddddd                                                │
1066.7┤           a           dd                                               │
      │            aa           dddddddd                                       │
      │              aaa                dd                                     │
      │                 a                 d                                    │
 800.0┤                 a                  dddddddXdd                          │
      │                  aaaa                        d                         │
      │                      aaa                      d                        │
      │                         aa                     d                       │
 533.3┤                           aaaaaa                dd                     │
      │                                 aa                dd                   │
      │                                   aa                dd                 │
      │                                     aaaaaa            ddd              │
 266.7┤                                          aaaaa           d             │
      │                                              aaaaaaa                   │
      │                                                     aaaaaaa            │
      │                                                                        │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0                45                90               135              180
tokens/s/gpu                         tokens/s/user

  ----------------------------------------------------------------------------
  Worker Setup:
    Model: QWEN3_32B (is_moe: False)
    Disagg Prefill: h200_sxm (trtllm)
    Disagg Decode:  h200_sxm (trtllm)
    Prefill Quantization: GEMM: fp8_block, KVCache: fp8, FMHA: fp8
    Decode Quantization: GEMM: fp8_block, KVCache: fp8, FMHA: fp8
    Agg: h200_sxm (trtllm)
    Quantization: GEMM: fp8_block, KVCache: fp8, FMHA: fp8
  ----------------------------------------------------------------------------
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

Disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user | concurrency | total_gpus(used) | replicas |  gpus/replica  | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    812.48    |     109.12    |      60     |  512 (512=64x8)  |    64    |  8 (=4x1+1x4)  |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   60  |
|  2   |    802.97    |     100.56    |     204     | 512 (500=20x25)  |    20    | 25 (=13x1+3x4) |     13     |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    4 (=4x1)    |    tp4pp1   |   68  |
|  3   |    802.09    |     106.73    |     192     | 512 (500=20x25)  |    20    | 25 (=13x1+3x4) |     13     |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    4 (=4x1)    |    tp4pp1   |   64  |
|  4   |    767.19    |     114.22    |     156     | 512 (506=22x23)  |    22    | 23 (=11x1+3x4) |     11     |    1 (=1x1)    |    tp1pp1   |   1   |     3      |    4 (=4x1)    |    tp4pp1   |   52  |
|  5   |    761.70    |     111.61    |     224     | 512 (496=16x31)  |    16    | 31 (=15x1+4x4) |     15     |    1 (=1x1)    |    tp1pp1   |   1   |     4      |    4 (=4x1)    |    tp4pp1   |   56  |
+------+--------------+---------------+-------------+------------------+----------+----------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
Agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+-------------+------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user | concurrency | total_gpus(used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+-------------+------------------+----------+--------------+-------------+----------+----+
|  1   |    340.48    |     100.83    |      15     | 512 (512=128x4)  |   128    |      4       |   4 (=4x1)  |  tp4pp1  | 15 |
|  2   |    326.78    |     104.48    |      14     | 512 (512=128x4)  |   128    |      4       |   4 (=4x1)  |  tp4pp1  | 14 |
|  3   |    307.50    |     105.57    |      13     | 512 (512=128x4)  |   128    |      4       |   4 (=4x1)  |  tp4pp1  | 13 |
|  4   |    296.61    |     107.15    |      24     |  512 (512=64x8)  |    64    |      8       |   8 (=8x1)  |  tp8pp1  | 24 |
|  5   |    265.44    |     115.81    |      20     |  512 (512=64x8)  |    64    |      8       |   8 (=8x1)  |  tp8pp1  | 20 |
+------+--------------+---------------+-------------+------------------+----------+--------------+-------------+----------+----+
********************************************************************************
INFO 2025-07-28 17:23:10,701 main.py:1035] Configuration completed in 48.18 seconds
```

These results indicate that deploying Qwen3-32B on `h200_sxm` in FP8 can achieve **2.39x** higher tokens/s/gpu for disaggregated versus aggregated deployment under the SLA targets TTFT ≤ 300 ms and TPOT ≤ 10 ms, with ISL:OSL of 4000:500.
Try different ISL:OSL values and SLA limits to fit your use case, for example:

```bash
aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200
```

You will get different results.

### Customized Configuration for aiconfigurator

To customize further (including the search space and per-component quantization) parameters are defined in a YAML file.
Built-in YAML files are under `src/aiconfigurator/cli/templates/trtllm/xxx_default.yaml` (in the future, `trtllm` can be replaced by other backend names).
Refer to the YAML file and modify as needed. Pass your customized YAML file with `--yaml_path`:

```bash
aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200 --yaml_path customized_config.yaml
```

For guidance on tuning these parameters, refer to [Advanced Tuning](docs/advanced_tuning.md).

### Generate Configurations for Dynamo

In the `aiconfigurator` CLI, if you specify `--save_dir`, the tool generates configuration files for deploying with Dynamo.
This feature bridges the gap between configuration and Dynamo deployment.
The folder structure looks like this:

```text
backend_configs/
├── agg/
│   ├── agg_config.yaml
│   └── node_0_run.sh
└── disagg/
│   ├── decode_config.yaml
│   ├── prefill_config.yaml
│   ├── node_0_run.sh
│   ├── node_1_run.sh
│   └── ...
└──
```

Refer to the [Deployment Guide](docs/dynamo_deployment_guide.md) for details.

### All-in-one automation

To further simpify the end-to-end user experience, we're now supporting automate everything in one script, starting from configuring the deployment, generating the configs, preparing docker image and container, pulling model checkpoints, deploying the service, benchmarking and summarizing. 
```bash
  bash launch_eval.sh config.env
```
Everything is in one command! We're trying to integrate our expertise to make the deployment smarter. Refer to [Automation](tools/automation/README.md) for more details.

## Webapp

```bash
aiconfigurator webapp
```

Visit `127.0.0.1:7860`.
Refer to [Advanced Tuning](docs/advanced_tuning.md) and the webapp README tab before running experiments.

## Tuning with Advanced Features

There are many features, such as different quantizations and parallelism strategies, to tune performance beyond the default configurations.
These apply to both the CLI and the webapp. Refer to [Advanced Tuning](docs/advanced_tuning.md) for details.

## How It Works

### Modeling and Mechanism

LLM inference performance is dominated by:

1. Compute cost (such as GEMM and attention).
2. Communication cost (such as all-reduce for tensor parallel and P2P for pipeline parallel).

To estimate performance, we take the following steps:

1. Break down LLM inference into operations: GEMM, attention, communication, embedding, element-wise operations, and others.
2. Collect operation execution times on the target hardware.
3. Estimate end-to-end execution time for a configuration by composing operation times using interpolation and extrapolation.
4. Model in-flight batching (aggregated) and disaggregated serving on top of that.
5. Search thousands of combinations to find strong configurations and generate Dynamo configuration files based on the results.

### Supported Features

- Models:
  - GPT
  - LLAMA (2, 3)
  - MOE
  - QWEN
  - DEEPSEEK_V3
- Operations:
  - Attention
    - MHA/GQA (FP8, FP16)
    - MLA (FP8, FP16)
  - KV Cache (FP16, FP8, INT8)
  - GEMM (FP16, FP8, FP8-Block, FP8-OOTB, SQ, INT8 WO, INT4 WO, NVFP4)
  - AllReduce (FP16)
  - Embedding
  - P2P
  - ElementWise
  - NCCL (all_reduce, all_gather, all-to-all, reduce_scatter)
  - MoE (FP16, FP8, FP8-Block, W4A-FP8, INT4 WO, NVFP4)
  - MLA BMM (FP16, FP8)
- TRTLLM Versions
  - 0.20.0
  - 1.0.0rc3
- Parallel modes:
  - Tensor-parallel
  - Pipeline-parallel
  - Expert Tensor-parallel/Expert-parallel
  - Attention DP (for DEEPSEEK and MoE)
- Scheduling:
  - Static
  - IFB (continuous batching)
  - Disaggregated serving
  - MTP (for DEEPSEEK)

### Data Collection

Data collection is a standalone process for building the database used by aiconfigurator. By default, you do not need to collect data yourself.
Small changes to the database may not materially change performance estimates. For example, you can use 1.0.0rc3 data of `trtllm` on `h200_sxm` and deploy the generated configuration with Dynamo and a `trtllm` 1.0.0rc4 worker.

To go through the process, refer to the [guidance](collector/README.md) under the `collector` folder.

### System Data Support Matrix

| System | Framework(Version) | Status |
|--------|-------------------|--------|
| h100_sxm | TRTLLM(0.20.0, 1.0.0rc3) | ✅ |
| h200_sxm | TRTLLM(0.20.0, 1.0.0rc3) | ✅ |
| b200_sxm | TRTLLM(NA) | 🚧 |

## Known Issues

1. MoE memory estimation for the `trtllm` backend needs to consider workspace.
2. Results can be overly optimistic in the low-speed, high-throughput region.

> **Note**: The results are not final or absolute. They can be inaccurate due to modeling gaps or indicate performance improvement opportunities. The tool aims to align with the framework's current implementation and to provide configuration suggestions. Verify results in real benchmarks with the generated configurations and perform follow-up tuning.
