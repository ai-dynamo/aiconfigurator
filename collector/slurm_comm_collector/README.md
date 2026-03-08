<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 1.nccl-test collection step by step 

## 1.1 Replace the "/path/to" in the sctipt with actual path

## 1.2 Build nccl test
```
git clone https://github.com/nvidia/nccl-tests
cd nccl-tests
make MPI=1 MPI_HOME=/usr/local/mpi
```

## 1.3 Run nccl test with slurm
```
sbatch -N 1 ./slurm_nccl_test_1node2gpu.sh
sbatch -N 1 ./slurm_nccl_test_1node4gpu.sh
sbatch -N 2 ./slurm_nccl_test_2node8gpu.sh
sbatch -N 4 ./slurm_nccl_test_4node16gpu.sh
sbatch -N 8 ./slurm_nccl_test_8node32gpu.sh
sbatch -N 16 ./slurm_nccl_test_16node64gpu.sh
```
*If the nodes don't work for some reason, give the specific node id like: sbatch --nodelist s03-p1-dgx-01-c06,s03-p1-dgx-01-c07 slurm_nccl_test_16node64gpu.sh

## 1.4 Extract the nccl data
```
cat log_nccl/1node2gpu.out | grep 0:\ \  > 1u2g
```

## 1.5 Get the result by cvt_log_to_perf_txt.py
```
python3 cvt_log_to_perf_txt.py
```

# 2.Custom allreduce collection in one node

## 2.1 Replace the "/path/to" in the sctipt with actual path

## 2.2 Run the collector
```
sbatch -N 1 ./slurm_custom_ar_2gpu.sh
sbatch -N 1 ./slurm_custom_ar_4gpu.sh
```

# 3. TensorRT-LLM MoE AlltoAll collection (NVLink)

Benchmarks MoE alltoall **dispatch** and **combine** over NVLink. Supports two kernel sources: **NVLinkTwoSided** (WideEP/MNNVL) and **NVLinkOneSided** (CutlassFusedMoE). Results are written to `results/trtllm_alltoall_perf.txt`. Configure `SCRIPT_DIR`, `CONTAINER_IMAGE`, `CONTAINER_MOUNTS`, `ACCOUNT`, `PARTITION`, `GPU_COUNTS`, and `GPUS_PER_NODE` in `submit_trtllm_alltoall.sh` before running.

## 3.1 Parameters (edit in `submit_trtllm_alltoall.sh`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `SCRIPT_DIR` | Directory of this script | `/path/to/slurm_comm_collector` |
| `CONTAINER_IMAGE` | TensorRT-LLM container image (.sqsh) | `/path/to/tensorrt-llm.sqsh` |
| `CONTAINER_MOUNTS` | Container mount paths (src:dst) | `/yourdata:/yourdata` |
| `ACCOUNT` | Slurm account name | `your account` |
| `PARTITION` | Slurm partition name | `your partition` |
| `GPU_COUNTS` | Array of GPU counts to test | `(2 4 8 16 32 48 64 72)` |
| `GPUS_PER_NODE` | Number of GPUs per node | `4` (e.g. GB200 NVL72) |

## 3.2 Run the collector

```bash
# Default: NVLinkTwoSided, GPU counts 2,4,8,16,32,48,64,72
bash submit_trtllm_alltoall.sh

# NVLinkOneSided, only 2 and 4 GPUs
bash submit_trtllm_alltoall.sh --kernel-source NVLinkOneSided --gpu-list 2,4

# NVLinkTwoSided, custom GPU counts
bash submit_trtllm_alltoall.sh --gpu-list 4,8,16
```

## 3.3 Check results
```bash
squeue -u $USER
cat results/trtllm_alltoall_perf.txt
```
