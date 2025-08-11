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
sudo nvidia-smi -lgc xxx,yyy
```
xxx, yyy frequency can be queried by nvidia-smi -q -i 0, refer to the Max Clocks part, xxx is SM frequency, yyy is Memory frequency.
Prepare a clean env with the target framework and nccl lib installed.

# Collect comm data
```bash
sh collect_comm.sh
```
Today we only collect intra-node comm. This script will collect custom allreduce data for trtllm within a node.
It will also collect nccl allreudce, all_gather, all2all, reduce_scatter using nccl.
The generated file is comm_perf.txt and custom_all_reduce.txt.

# Collect gemm/attention/moe data/etc.
```bash
python3 collect.py
```
For trtllm, the whole collecting process takes about 30 gpu-hours. On 8-gpu, it takes 3-4 hours.
Please note that the whole process will report a lot of missing datapoints with errors. But it's okay. Our system is kindof robust to fair amount of missing data.
Once everything is done, you might see mutliple xxx.txt files under the same folder. Refer to src/aiconfigurator/systems/ folder to prepare the database including 
how many files are needed accordingly.

# Test
Rebuild and install the new aiconfigurator. Please make sure you have your new system definition file prepared. It's src/aiconfigurator/systems/xxx.yaml

# Validate the correctness
Today, we have limited method to validate the database. You can try tools/sanity_check to validate the database a little bit. But it highly depends on your understanding 
of the GPU system and kernel optimization.

# Support Matrix
aiconfigurator 0.1.0  
trtllm: 0.20.0, 1.0.0rc3 on Hopper GPUs  
vllm: NA  
sglang: NA  
