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
git lfs pull

# 3. Create and activate a virtual environment
python3 -m venv myenv && source myenv/bin/activate # (requires Python 3.9 or later)

# 4. Install aiconfigurator
pip3 install .

# 5. Install aiconfigurator with webapp support
pip3 install .[webapp]
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
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm
aiconfigurator cli exp --yaml_path exp.yaml
```
- We have two modes, `default` and `exp`.
- Use `default`, followed with **three basic arguments** (model, total_gpus, system), it prints the estimated best deployment and the deployment details.
- Use `--backend` to specify the inference backend: `trtllm` (default), `vllm`, or `sglang`.
- Use `exp`, pass in exp.yaml by `--yaml_path` to customize your experiments and even a heterogenous one.
- Use `--save_dir DIR` to generate framework configuration files for Dynamo.
- Use `--database_mode` to control performance estimation mode: `SILICON` (default, uses collected silicon data), `HYBRID` (uses silicon data when available, otherwise SOL+empirical), `EMPIRICAL` (SOL+empirical for all), or `SOL` (speed-of-light only).
- Use `-h` for more options and customization.
- SLA constraints:
  - `--ttft` and `--tpot` filter configurations that exceed either bound; omit a flag to leave that constraint unset.
  - `--request_latency` applies an end-to-end per-request limit. The CLI searches for all configurations whose estimated 
  latency stays within that budget, optionally honoring a provided `--ttft`. 
  When this flag is set, `--tpot` becomes implicit and is ignored.

Refer to [CLI User Guide](docs/cli_user_guide.md)

An example here, 
```bash
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --isl 4000 --osl 500 --prefix 500 --ttft 300 --tpot 10
```

```text
********************************************************************************
*                     Dynamo aiconfigurator Final Results                      *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: QWEN3_32B (is_moe: False)
    Total GPUs: 32
    Best Experiment Chosen: disagg at 804.83 tokens/s/gpu (disagg 1.64x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 25,754.50 tokens/s
    - Per-GPU Throughput: 804.83 tokens/s/gpu
    - Per-User Throughput: 109.13 tokens/s/user
    - TTFT: 486.53ms
    - TPOT: 9.16ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
               QWEN3_32B Pareto Frontier: tokens/s/gpu vs tokens/s/user         
      ┌────────────────────────────────────────────────────────────────────────┐
1400.0┤ •• disagg                                                              │
      │ ff agg                                                                 │
      │ xx disagg best                                                         │
      │                                                                        │
1166.7┤          •                                                             │
      │           •••••••                                                      │
      │                  •                                                     │
      │                   ••••••                                               │
 933.3┤                         •                                              │
      │                         •                                              │
      │                         •••••••x                                       │
      │                                 •                                      │
 700.0┤                                 •                                      │
      │        f                         •                                     │
      │         ffff                      •                                    │
      │             ffffffffffffffff       ••                                  │
 466.7┤                             fffffff  •••                               │
      │                                    fff  •••                            │
      │                                       ffff•                            │
      │                                           ffffffff                     │
 233.3┤                                                  fffffff               │
      │                                                      •••fffff          │
      │                                                         ••• fff        │
      │                                                               f        │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0                60                120              180              240 
tokens/s/gpu                         tokens/s/user                              

  ----------------------------------------------------------------------------
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency  | total_gpus (used) | replicas |  gpus/replica | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+--------------+---------------+--------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |    804.83    |     109.13    | 486.53 | 256 (=64x4)  |    32 (32=4x8)    |    4     |  8 (=4x1+1x4) |     4      |    1 (=1x1)    |    tp1pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   64  |
|  2   |    416.25    |     102.85    | 486.53 | 144 (=144x1) |    32 (24=1x24)   |    1     | 24 (=8x1+8x2) |     8      |    1 (=1x1)    |    tp1pp1   |   1   |     8      |    2 (=2x1)    |    tp2pp1   |   18  |
|  3   |    416.25    |     118.53    | 486.53 | 128 (=128x1) |    32 (24=1x24)   |    1     | 24 (=8x1+2x8) |     8      |    1 (=1x1)    |    tp1pp1   |   1   |     2      |    8 (=8x1)    |    tp8pp1   |   64  |
+------+--------------+---------------+--------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+--------------+---------------+--------+-------------+-------------------+----------+--------------+-------------+----------+----+
| Rank | tokens/s/gpu | tokens/s/user |  TTFT  | concurrency | total_gpus (used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+--------------+---------------+--------+-------------+-------------------+----------+--------------+-------------+----------+----+
|  1   |    491.03    |     103.66    | 272.50 | 160 (=20x8) |    32 (32=8x4)    |    8     |      4       |  4 (=4x1x1) |  tp4pp1  | 20 |
|  2   |    359.06    |     106.77    | 226.94 | 112 (=28x4) |    32 (32=4x8)    |    4     |      8       |  8 (=8x1x1) |  tp8pp1  | 28 |
|  3   |    180.64    |     129.41    | 295.45 |  48 (=3x16) |    32 (32=16x2)   |    16    |      2       |  2 (=2x1x1) |  tp2pp1  | 3  |
+------+--------------+---------------+--------+-------------+-------------------+----------+--------------+-------------+----------+----+
********************************************************************************
INFO 2025-10-25 23:40:24,396 main.py:340] All experiments completed in 13.90 seconds
```

These results indicate that deploying Qwen3-32B on h200_sxm in FP8 can achieve **1.64x** higher tokens/s/gpu for disaggregated versus aggregated deployment **under the SLA targets TTFT ≤ 300 ms and TPOT ≤ 10 ms**, with ISL:OSL of 4000:500 (with prefix len: 500).
Try different ISL:OSL values and SLA limits to fit your use case, for example:

```bash
aiconfigurator cli default --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200 --prefix 500
```

You will get different results.

### Customized Configuration for aiconfigurator

The `default` mode will create two experiments, one is `agg` and another one is `disagg` and then compare the results.  
To further customize (including the search space and per-component quantization), parameters are defined in a YAML file.
Built-in YAML files are under `src/aiconfigurator/cli/example.yaml` and `src/aiconfigurator/cli/exps/*.yaml`
Refer to the YAML file and modify as needed. Pass your customized YAML file to `exp` mode:

```bash
aiconfigurator cli exp --yaml_path customized_config.yaml
```
We can use `exp` mode to compare multiple results, including disagg vs. agg, homegenous vs. heterogenous, and more than 2 experiments. 
We've crafted several examples in `src/aiconfigurator/cli/exps/*.yaml`  
For the full guide, refer to [CLI User Guide](docs/cli_user_guide.md).

### Generate Configurations for Dynamo and Reproduce the results

Please refer to the [Deployment Guide](docs/dynamo_deployment_guide.md) for details about deployment and reproduction especially about the benchmark methodology.

To simplify the deployment and reproduction, in the `aiconfigurator` CLI, if you specify `--save_dir`, the tool generates configuration files for deploying with Dynamo.
This feature bridges the gap between configuration and Dynamo deployment.
The folder structure looks like this:

```text
results/QWEN3_32B_isl4000_osl1000_ttft1000_tpot20_904495
├── agg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── agg
│   │   │   ├── agg_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   └── node_0_run.sh 
│   │   └── generator_config.yaml
│   ...
├── disagg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── disagg
│   │   │   ├── decode_config.yaml
│   │   │   ├── k8s_deploy.yaml
│   │   │   ├── node_0_run.sh
│   │   │   └── prefill_config.yaml
│   │   └── generator_config.yaml
│   ...
└── pareto_frontier.png
```

Use `--generator-config path/to/file.yaml` to load a YAML payload with `ServiceConfig`, `K8sConfig`, and `Workers.*` sections, or specify inline overrides with `--generator-set KEY=VALUE` (repeatable). Examples:

- `--generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8`
- `--generator-set K8sConfig.k8s_namespace=dynamo \`

Run `aiconfigurator cli default --generator-help` to print information that is sourced directly from `src/aiconfigurator/generator/config/deployment_config.yaml` and `backend_config_mapping.yaml`. 

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

- **Models**:
  - GPT
  - LLAMA (2, 3)
  - MOE
  - QWEN
  - DEEPSEEK_V3
  - Support using huggingface model id if falls into these model family and not MoE models.

- **Operations**:
  - Attention
    - MHA/GQA (FP8, FP16)
    - MLA (FP8, FP16)
  - KV Cache (FP16, FP8, INT8)
  - GEMM (FP16, FP8, FP8-Block, FP8-OOTB, SQ, INT8 WO, INT4 WO, NVFP4)
  - CustomAllReduce (FP16)
  - Embedding
  - P2P
  - ElementWise
  - NCCL (all_reduce, all_gather, all-to-all, reduce_scatter)
  - MoE (FP16, FP8, FP8-Block, W4A-FP8, INT4 WO, NVFP4)
  - MLA BMM (FP16, FP8)

- **Parallel modes**:
  - Tensor-parallel
  - Pipeline-parallel
  - Expert Tensor-parallel/Expert-parallel
  - Attention DP (for DEEPSEEK and MoE)

- **Scheduling**:
  - Static
  - Aggregated serving (continuous batching)
  - Disaggregated serving
  - MTP (for DEEPSEEK)

- **Inference Backends**:
  - TensorRT-LLM (trtllm)
  - vLLM
  - SGLang

### Data Collection

Data collection is a standalone process for building the database used by aiconfigurator. By default, you do not need to collect data yourself.
Small changes to the database may not materially change performance estimates. For example, you can use 1.0.0rc3 data of `trtllm` on `h200_sxm` and deploy the generated configuration with Dynamo and a `trtllm` 1.0.0rc4 worker.

To go through the process, refer to the [guidance](collector/README.md) under the `collector` folder.

**New:** The collector now supports optional GPU power monitoring during kernel execution. Use the `--measure_power` flag to collect power consumption data alongside performance metrics. See the [collector README](collector/README.md#power-monitoring-optional) for details.

### System Data Support Matrix

| System | Framework(Version) | Status |
|--------|-------------------|--------|
| h100_sxm | TRTLLM(0.20.0, 1.0.0rc3), SGLang(0.5.5.post2, 0.5.5.post3), vLLM(0.11.0) | ✅ |
| h200_sxm | TRTLLM(0.20.0, 1.0.0rc3), SGLang(0.5.5.post2, 0.5.5.post3), vLLM(0.11.0) | ✅ |
| b200_sxm | TRTLLM(1.0.0rc6) | ✅ |
| gb200_sxm | TRTLLM(1.0.0rc6) | ✅ |
| a100_sxm | TRTLLM(1.0.0) | ✅ |

> **Note**: b200 and gb200 are under dev. Results are to be aligned. For preview now.

#### Detailed Support Matrix

For a comprehensive breakdown of which model/system/backend/version combinations are supported in both aggregated and disaggregated modes, refer to the [**support matrix CSV**](src/aiconfigurator/systems/support_matrix.csv). This file is automatically generated and tested to ensure accuracy across all supported configurations.

## Contributing and Development

We welcome contributions from the community! Check out the below resources to get started:

- [DEVELOPMENT.md](DEVELOPMENT.md) - Set up your development environment, run tests, and follow our coding standards
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and requirements

### How To Add A New Model
Adding a new model will require modifying the source code and perhaps collecting new data for the model. Please refer to [How to Add a New Model](docs/add_a_new_model.md).

## Known Issues

1. Memory estimation for the backends needs to be studied more.
2. Results can be overly optimistic in the low-speed, high-throughput region.
3. **vLLM and SGLang support is currently being evaluated**. While both backends are functional and available for use, we are still completing comprehensive performance evaluations and alignment testing. We recommend validating results with real benchmarks for production use.

> **Note**: The results are not final or absolute. They can be inaccurate due to modeling gaps or indicate performance improvement opportunities. The tool aims to align with the framework's current implementation and to provide configuration suggestions. Verify results in real benchmarks with the generated configurations and perform follow-up tuning.
