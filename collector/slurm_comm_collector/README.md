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
