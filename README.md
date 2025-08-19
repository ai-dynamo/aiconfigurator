<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIConfigurator
Today, in disaggregated serving, it's quite difficult to find a proper config
to get benefits from disaggregation such as how many prefill workers and decode workers 
do I need and what about the parallelism for each worker. Combined with SLA: 
TTFT(Time-To-First-Token) and TPOT(Time-Per-Output-Token), it becomes even more complicated 
to solve the throughput @ latency problem.

We're introducing AIConfigurator to help you find a good reference to start with in your 
disaggregated serving journey. The tool will try to search the space to get a good deployment config 
based on your requirement including which model you want to serve, how many GPUs you have and what's 
the GPU. Automatically generate the config files for you to deploy with Dynamo.

It's based on modeling the LLM inference with collected data on a target machine with a specific framework.
It searches thousands of different configurations in the background in tens of seconds and runs on any machine 
with a CLI tool and a webapp provided.

Let's get started.


# Build and Install
## Install from PyPI
```bash
pip3 install aiconfigurator
```
## Build and install from source
1. apt-get install git-lfs (linux) or brew install git-lfs (macos)
2. clone the repo
3. (optional) python3 -m venv myenv && source myenv/bin/activate  (need to have python >= 3.9)
4. (optional) pip3 install --upgrade pip (if you encounter issue that didn't find setup.py)
5. pip3 install "."

## Build with Dockerfile
```bash
  # This will create a ./dist/ folder containing the wheel file
  docker build -f docker/Dockerfile --no-cache --target build -t aiconfigurator:latest .
  docker create --name aic aiconfigurator:latest && docker cp aic:/workspace/dist dist/ && docker rm aic
```

# Run
## CLI
```bash
  aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm
```
With **3 basic args**, it will report out the estimated best deployment result and the deployment details  
With **--save_dir DIR**, you can output the framework configs automatically to deploy with Dynamo  
With **-h**, you can have more information about optional args to customize your deployment target  

```
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
The results indicate that, when you want to deploy Qwen3-32B on h200_sxm in fp8, you can get **2.39x** of disagg over agg deployment under SLA TTFT<=300ms and TPOT<=10ms with ISL:OSL as 4000:500  
Try different ISL:OSL for differnt TTFT and TPOT limit with, say, 
```bash
  aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200
```
You will get different answers.  

### Customized config for aiconfigurator
If you want to even customize more, including the search space, quantization for each component, we define all these parameters in a yaml file.  
The built-in yaml files are under src/aiconigurator/cli/templates/trtllm/xxx_default.yaml (in future, trtllm can be other backend names)  
Please refer to the yaml file and modify what you want. Pass your customized yaml file by **--yaml_path**, 
```bash
  aiconfigurator cli --model QWEN3_32B --total_gpus 32 --system h200_sxm --ttft 200 --tpot 10 --isl 8000 --osl 200 --yaml_path customized_config.yaml
```
About how to tune these parameters, please refer to [Advanced Tuning](docs/advanced_tuning.md) for details

### Generate configs for Dynamo
In aiconfigurator cli, if you specify --save_dir, we'll generate configs for deploying with Dynamo.
This is an **important** feature to bridge the gap between configuration and Dynamo deployment.  
The folder structure will be like this,
````
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
````
Please refer to [Deployment Guide](docs/dynamo_deployment_guide.md) for details

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
Visit 127.0.0.1:7860  
Make sure to read [Advanced Tuning](docs/advanced_tuning.md) and the readme tab of webapp before you do experiments.


## Tuning with advanced features
There're a lot of features like different quantizations, different parallel strategies for you to tune the performance 
beyond the default configurations. This is common for both CLI and Webapp. Please refer to [Advanced Tuning](docs/advanced_tuning.md) for details


# How it works
## Modeling and mechanism

If we want to estimate the inference perf for a LLM, below should be considered,
1. compute cost, gemm, attention, others
2. communication cost, all-reduce for tensor-parallel, p2p for pipeline-parallel

Based on breaking down the LLM inference process into operations, i.e., gemm, attention, communication, embedding, elementwise operations, others.  
Collect operation execution time on a given hardware  
Estimate the given config execution time composed of operation execution time based on interpolation/extrapolation.  
We then model the inflight-batching (aggregated) and disaggregated serving on top of that.  
Search for the best config among those thousands of possible combinations and generate configs for Dynamo based on the results.

## Support list 
Models: GPT, LLAMA(2,3), MOE, QWEN, DEEPSEEK_V3  
OPs: MHA/GQA/MLA(FP8,FP16,FP32 fmha), 8bit kvcache, GEMM(FP16, 8/4bit WO, SQ, FP8), AllReduce(FP16), Embedding, P2P, ElementWise, NCCL(all2all, allgather, reducescatter), MoE(FP16, FP8, W4AFP8)  
TRTLLM Versions: 0.20.0, 1.0.0rc3  
Parallel modes: Tensor-parallel; Pipeline-parallel; Expert Tensor-parallel/Expert-parallell; Attention DP for DEEPSEEK and MoE  
Scheduling: Static; IFB(continuous batching); Disaggregated serving; MTP for DEEPSEEK

### System Data Support Matrix

| System | Framework(Version) | Status |
|--------|-------------------|--------|
| h100_sxm | TRTLLM(0.20.0, 1.0.0rc3) | ✅ |
| h200_sxm | TRTLLM(0.20.0, 1.0.0rc3) | ✅ |
| b200_sxm | TRTLLM(NA) | 🚧 |


## Data Collection
Data collection is a standalone process for collecting the database for aiconfigurator. By default, you don't have to collect the data by yourself.
Small versions of database will not introduce huge perf difference. Say, you can use 1.0.0rc3 data of trtllm on h200_sxm and deploy the generated 
configs with Dynamo + trtllm 1.0.0rc4 worker.

If you want to go through the process, please refer to this [guidance](collector/README.md) under collector folder


# Known issues
1. moe memory estimation of trtllm backend needs to consider workspace  
2. result is relatively too optimisitc in low-speed high-throughput region.  
> **Note**: the result is not final absolute one. It can be inaccurate due to modeling gap or indicate performance improvement opportunity. It's trying to align with framework's current implementation and aming to provide configuration suggestion. Please verify it in real benchmark with our generated configs and do follow-up tuning.
