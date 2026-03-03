<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# aiconfigurator

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ai-dynamo/aiconfigurator)

In disaggregated serving, configuring an effective deployment is challenging: you need to decide how many prefill and decode
workers to run, and the parallelism for each worker. Combined with SLA targets for TTFT (Time to First Token) and
TPOT (Time per Output Token), optimizing throughput at a given latency becomes even more complex.

`aiconfigurator` helps you find a strong starting configuration for disaggregated serving. Given your model, GPU
count, and GPU type, it searches the configuration space and generates configuration files you can use for deployment with Dynamo.

For a technical deep dive into the design and methodology of AIConfigurator, please refer to our paper:  
[**AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving**](https://arxiv.org/abs/2601.06288).

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
aiconfigurator cli default --model Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm
aiconfigurator cli exp --yaml-path exp.yaml
aiconfigurator cli generate --model-path Qwen/Qwen3-32B-FP8 --total-gpus 8 --system h200_sxm
aiconfigurator cli support --model-path Qwen/Qwen3-32B-FP8 --system h200_sxm
```
- We have four modes: `default`, `exp`, `generate`, and `support`.
- Use `default` to find the estimated best deployment by searching the configuration space.
- Use `exp` to run customized experiments defined in a YAML file.
- Use `generate` to quickly create a naive configuration without a parameter sweep.
- Use `support` to verify if AIC supports a model/hardware combination for agg and disagg modes.
- `--model` is an alias for `--model-path` in the CLI.
- Use `--backend` to specify the inference backend: `trtllm` (default), `vllm`, or `sglang`.
- Use `exp`, pass in exp.yaml by `--yaml-path` to customize your experiments and even a heterogenous one.
- Use `--save-dir DIR` to generate framework configuration files for Dynamo.
- Use `--database-mode` to control performance estimation mode: `SILICON` (default, uses collected silicon data), `HYBRID` (uses silicon data when available, otherwise SOL+empirical), `EMPIRICAL` (SOL+empirical for all), or `SOL` (speed-of-light only). Please be careful, only `SILICON` mode's result is reproducible. Other modes are for research purpose
- Use `--systems-paths` to override where system YAMLs and data are loaded from (comma-separated; `default` maps to the built-in systems path). First match wins for identical system/backend/version.
- Use `-h` for more options and customization.
- SLA constraints:
  - `--ttft` and `--tpot` filter configurations that exceed either bound; omit a flag to leave that constraint unset.
  - `--request-latency` applies an end-to-end per-request limit. The CLI searches for all configurations whose estimated 
  latency stays within that budget, optionally honoring a provided `--ttft`. 
  When this flag is set, `--tpot` becomes implicit and is ignored.

Quantization defaults are inferred from the Hugging Face model config (`config.json` plus optional `hf_quant_config.json`).  
For low-precision models, use a quantized HF ID (for example, `Qwen/Qwen3-32B-FP8`) or a local model directory containing those files.  
Any quantization set via `profiles` or YAML `config` overrides the HF defaults.

For a full end-to-end walkthrough (support check, sweep, deploy, benchmark), see the [CLI User Guide -- End-to-End Workflow](docs/cli_user_guide.md#end-to-end-workflow).

Refer to [CLI User Guide](docs/cli_user_guide.md)

### Python API

You can also use `aiconfigurator` programmatically in Python:

```python
from aiconfigurator.cli import cli_default, cli_exp, cli_generate, cli_support

# 1. Run default agg vs disagg comparison
result = cli_default(model_path="Qwen/Qwen3-32B-FP8", total_gpus=32, system="h200_sxm")
print(result.best_configs["disagg"].head())

# 2. Run experiments from a YAML file or a dictionary config
result = cli_exp(yaml_path="my_experiments.yaml")
# Or use a dictionary config directly
result = cli_exp(config={
    "my_exp": {
        "serving_mode": "disagg",
        "model_path": "Qwen/Qwen3-32B-FP8",
        "total_gpus": 32,
        "system_name": "h200_sxm",
        "isl": 4000,
        "osl": 1000,
    }
})

# 3. Generate a naive configuration
result = cli_generate(model_path="Qwen/Qwen3-32B-FP8", total_gpus=8, system="h200_sxm")
print(result["parallelism"]) # {'tp': 1, 'pp': 1, 'replicas': 8, 'gpus_used': 8}

# 4. Check support for a model/system combination
agg, disagg = cli_support(model_path="Qwen/Qwen3-32B-FP8", system="h200_sxm")
print(f"Agg supported: {agg}, Disagg supported: {disagg}")
```

An example here, 
```bash
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --isl 4000 --osl 500 --prefix 500 --ttft 300 --tpot 10
```

```text
********************************************************************************
*                     Dynamo aiconfigurator Final Results                      *
********************************************************************************
  ----------------------------------------------------------------------------
  Input Configuration & SLA Target:
    Model: Qwen/Qwen3-32B-FP8 (is_moe: False)
    Total GPUs: 32
    Best Experiment Chosen: disagg at 684.79 tokens/s/gpu (disagg 1.67x better)
  ----------------------------------------------------------------------------
  Overall Best Configuration:
    - Best Throughput: 21,913.22 tokens/s
    - Per-GPU Throughput: 684.79 tokens/s/gpu
    - Per-User Throughput: 100.31 tokens/s/user
    - TTFT: 295.71ms
    - TPOT: 9.97ms
    - Request Latency: 5270.24ms
  ----------------------------------------------------------------------------
  Pareto Frontier:
       Qwen/Qwen3-32B-FP8 Pareto Frontier: tokens/s/gpu_cluster vs tokens/s/user
      ┌────────────────────────────────────────────────────────────────────────┐
1250.0┤ •• agg                                                                 │
      │ ff disagg                                                              │
      │ xx disagg best                                                         │
      │                                                                        │
1041.7┤                                                                        │
      │          f                                                             │
      │          fffffffff                                                     │
      │                   fff                                                  │
 833.3┤                      ffff                                              │
      │                          f                                             │
      │       •                   ff                                           │
      │       ••                    fxfff                                      │
 625.0┤         •••••                   f                                      │
      │              •                  f                                      │
      │               ••••••••••••      f                                      │
      │                           •••   f                                      │
 416.7┤                              ••••ff                                    │
      │                                  ••ff                                  │
      │                                     •fffffffffffff                     │
      │                                           ••••••••ff•                  │
 208.3┤                                                     ff•••              │
      │                                                       ff ••••          │
      │                                                         fff •••        │
      │                                                                •       │
   0.0┤                                                                        │
      └┬─────────────────┬─────────────────┬────────────────┬─────────────────┬┘
       0                60                120              180              240 
tokens/s/gpu_cluster                 tokens/s/user                              

  ----------------------------------------------------------------------------
  Deployment Details:
    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system
    Some math: total gpus used = replicas * gpus/replica
               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker
               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined numbers are the actual values in math)

agg Top Configurations: (Sorted by tokens/s/gpu)
+------+---------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+--------------+-------------+----------+----+
| Rank | backend | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency | total_gpus (used) | replicas | gpus/replica | gpus/worker | parallel | bs |
+------+---------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+--------------+-------------+----------+----+
|  1   |  trtllm |    410.22    |     108.48    | 251.10 |     4850.91     | 128 (=16x8) |    32 (32=8x4)    |    8     |      4       |  4 (=4x1x1) |  tp4pp1  | 16 |
|  2   |  trtllm |    361.33    |     107.43    | 224.48 |     4869.40     | 112 (=28x4) |    32 (32=4x8)    |    4     |      8       |  8 (=8x1x1) |  tp8pp1  | 28 |
|  3   |  trtllm |    117.92    |     122.25    | 292.72 |     4374.38     |  32 (=2x16) |    32 (32=16x2)   |    16    |      2       |  2 (=2x1x1) |  tp2pp1  | 2  |
+------+---------+--------------+---------------+--------+-----------------+-------------+-------------------+----------+--------------+-------------+----------+----+

disagg Top Configurations: (Sorted by tokens/s/gpu)
+------+---------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
| Rank | backend | tokens/s/gpu | tokens/s/user |  TTFT  | request_latency | concurrency  | total_gpus (used) | replicas |  gpus/replica | (p)workers | (p)gpus/worker | (p)parallel | (p)bs | (d)workers | (d)gpus/worker | (d)parallel | (d)bs |
+------+---------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
|  1   |  trtllm |    684.79    |     100.31    | 295.71 |     5270.24     | 272 (=68x4)  |    32 (32=4x8)    |    4     |  8 (=2x2+1x4) |     2      |    2 (=2x1)    |    tp2pp1   |   1   |     1      |    4 (=4x1)    |    tp4pp1   |   68  |
|  2   |  trtllm |    684.79    |     100.16    | 295.71 |     5277.73     | 240 (=120x2) |    32 (32=2x16)   |    2     | 16 (=4x2+1x8) |     4      |    2 (=2x1)    |    tp2pp1   |   1   |     1      |    8 (=8x1)    |    tp8pp1   |  120  |
|  3   |  trtllm |    404.71    |     100.35    | 295.71 |     5268.25     | 140 (=140x1) |    32 (24=1x24)   |    1     | 24 (=5x2+7x2) |     5      |    2 (=2x1)    |    tp2pp1   |   1   |     7      |    2 (=2x1)    |    tp2pp1   |   20  |
+------+---------+--------------+---------------+--------+-----------------+--------------+-------------------+----------+---------------+------------+----------------+-------------+-------+------------+----------------+-------------+-------+
********************************************************************************
2026-02-08 23:10:21,413 - aiconfigurator.cli.main - INFO - All experiments completed in 6.50 seconds
```

These results indicate that deploying Qwen3-32B-FP8 on h200_sxm in FP8 can achieve **1.67x** higher tokens/s/gpu for disaggregated versus aggregated deployment **under the SLA targets TTFT ≤ 300 ms and TPOT ≤ 10 ms**, with ISL:OSL of 4000:500 (with prefix len: 500).
Try different ISL:OSL values and SLA limits to fit your use case, for example:

```bash
aiconfigurator cli default --model-path Qwen/Qwen3-32B-FP8 --total-gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200 --prefix 500
```

You will get different results.

### Customized Configuration for aiconfigurator

The `default` mode will create two experiments, one is `agg` and another one is `disagg` and then compare the results.  
To further customize (including the search space and per-component quantization), parameters are defined in a YAML file.
Built-in YAML files are under `src/aiconfigurator/cli/example.yaml` and `src/aiconfigurator/cli/exps/*.yaml`
Refer to the YAML file and modify as needed. Pass your customized YAML file to `exp` mode:

```bash
aiconfigurator cli exp --yaml-path customized_config.yaml
```
We can use `exp` mode to compare multiple results, including disagg vs. agg, homegenous vs. heterogenous, and more than 2 experiments. 
We've crafted several examples in `src/aiconfigurator/cli/exps/*.yaml`  
For the full guide, refer to [CLI User Guide](docs/cli_user_guide.md).

### Generate Configurations for Dynamo and Reproduce the results

Please refer to the [Deployment Guide](docs/dynamo_deployment_guide.md) for details about deployment and reproduction especially about the benchmark methodology.

To simplify the deployment and reproduction, in the `aiconfigurator` CLI, if you specify `--save-dir`, the tool generates configuration files for deploying with Dynamo.
This feature bridges the gap between configuration and Dynamo deployment.
The folder structure looks like this:

```text
results/QWEN3_32B_FP8_h200_sxm_trtllm_isl4000_osl1000_ttft1000_tpot20_904495
├── agg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── agg
│   │   │   ├── agg_config.yaml
│   │   │   ├── bench_run.sh          # aiperf benchmark sweep script (bare-metal)
│   │   │   ├── k8s_bench.yaml        # aiperf benchmark sweep Job (Kubernetes)
│   │   │   ├── k8s_deploy.yaml
│   │   │   └── node_0_run.sh 
│   │   └── generator_config.yaml
│   ...
├── disagg
│   ├── best_config_topn.csv
│   ├── config.yaml
│   ├── pareto.csv
│   ├── top1
│   │   ├── disagg
│   │   │   ├── bench_run.sh          # aiperf benchmark sweep script (bare-metal)
│   │   │   ├── decode_config.yaml
│   │   │   ├── k8s_bench.yaml        # aiperf benchmark sweep Job (Kubernetes)
│   │   │   ├── k8s_deploy.yaml
│   │   │   ├── node_0_run.sh
│   │   │   └── prefill_config.yaml
│   │   └── generator_config.yaml
│   ...
└── pareto_frontier.png
```

Use `--generator-config path/to/file.yaml` to load a YAML payload with `ServiceConfig`, `K8sConfig`, `DynConfig`, `WorkerConfig`, and `Workers.<role>` sections, or specify inline overrides with `--generator-set KEY=VALUE` (repeatable). Examples:

- `--generator-set ServiceConfig.model_path=Qwen/Qwen3-32B-FP8`
- `--generator-set K8sConfig.k8s_namespace=dynamo \`

Run `aiconfigurator cli default --generator-help` to print information that is sourced directly from `src/aiconfigurator/generator/config/deployment_config.yaml` and `backend_config_mapping.yaml`. 

### All-in-one automation (To be refactored)

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
| h100_sxm | TRTLLM(1.0.0rc3, 1.2.0rc5), SGLang(0.5.6.post2), vLLM(0.12.0) | ✅ |
| h200_sxm | TRTLLM(1.0.0rc3, 1.2.0rc5), SGLang(0.5.6.post2), vLLM(0.12.0) | ✅ |
| b200_sxm | TRTLLM(1.0.0rc3, 1.2.0rc5), SGLang(0.5.6.post2) | ✅ |
| gb200 | TRTLLM(1.0.0rc3, 1.2.0rc5) | ✅ |
| a100_sxm | TRTLLM(1.0.0), vLLM(0.12.0) | ✅ |
(last updated: 2026/02/02)

> **Note**: b200 and gb200 are under dev. Results are to be aligned. For preview now.

#### Detailed Support Matrix

For a comprehensive breakdown of which model/system/backend/version combinations are supported in both aggregated and disaggregated modes, refer to the [**support matrix CSV**](src/aiconfigurator/systems/support_matrix.csv). This file is automatically generated and tested to ensure accuracy across all supported configurations.

You can also check if a system / framework version is supported via the `aiconfigurator cli support` command. For example:
```bash
aiconfigurator cli support --model-path Qwen/Qwen3-32B-FP8 --system h100_sxm --backend-version 1.2.0rc5
```

## Contributing and Development

We welcome contributions from the community! Check out the below resources to get started:

- [DEVELOPMENT.md](DEVELOPMENT.md) - Set up your development environment, run tests, and follow our coding standards
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines and requirements

### How To Add A New Model
Adding a new model will require modifying the source code and perhaps collecting new data for the model. Please refer to [How to Add a New Model](docs/add_a_new_model.md).

## Citation

If you use AIConfigurator for your research, please cite our paper:

```bibtex
@article{xu2026aiconfigurator,
  title={AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving},
  author={Tianhao Xu and Yiming Liu and Xianglong Lu and Yijia Zhao and Xuting Zhou and Aichen Feng and Yiyi Chen and Yi Shen and Qin Zhou and Xumeng Chen and Ilya Sherstyuk and Haorui Li and Rishi Thakkar and Ben Hamm and Yuanzhe Li and Xue Huang and Wenpeng Wu and Anish Shanbhag and Harry Kim and Chuan Chen and Junjie Lai},
  journal={arXiv preprint arXiv:2601.06288},
  year={2026}
}
```

## Known Issues

1. Memory estimation for the backends needs to be studied more.
2. Results can be overly optimistic in the low-speed, high-throughput region.
3. **vLLM and SGLang support is currently being evaluated**. While both backends are functional and available for use, we are still completing comprehensive performance evaluations and alignment testing. We recommend validating results with real benchmarks for production use.

> **Note**: The results are not final or absolute. They can be inaccurate due to modeling gaps or indicate performance improvement opportunities. The tool aims to align with the framework's current implementation and to provide configuration suggestions. Verify results in real benchmarks with the generated configurations and perform follow-up tuning.
