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

# 2. Custom allreduce collection

Single-node scripts cover TP=2/4. On GB200/GB300 (4 GPUs per node) serving-relevant TP sizes span nodes; use the multi-node scripts so the TRT-LLM MNNVL path is exercised (issue #1416). `collect_allreduce.py` must construct `AllReduce(..., dtype=...)` — without dtype the collector silently measures the fallback kernel.

The TP8/TP16 launchers are reviewed templates and have not been exercised as-is on a Slurm cluster; the published GB300 rows were collected through `torchrun`/`mpirun`. Before use, replace the rc20 image and repository-mount placeholders, and verify the site supports `--gpus-per-task=1 --gpu-bind=single:1`; the worker intentionally fails unless each task sees exactly one CUDA device.

## 2.1 Replace the "/path/to" in the script with actual path

## 2.2 Run the collector
```bash
sbatch -N 1 ./slurm_custom_ar_2gpu.sh
sbatch -N 1 ./slurm_custom_ar_4gpu.sh
sbatch -N 2 ./slurm_custom_ar_8gpu.sh
sbatch -N 4 ./slurm_custom_ar_16gpu.sh
```

# 3. TensorRT-LLM MoE AlltoAll collection (NVLink)

Benchmarks MoE alltoall **prepare**, **dispatch** and **combine** over NVLink. Supports two kernel sources: **NVLinkTwoSided** (WideEP/MNNVL) and **NVLinkOneSided** (CutlassFusedMoE). Results are written per job in the unified `moe_a2a` schema to `results/moe_a2a_<kernel-source>.<N>gpu/job_<jobid>/moe_a2a_perf.parquet` with a `collection_meta.yaml` provenance sidecar. The default container image is the manifest `trtllm` pin. Run from `collector/network/slurm/` and configure `CONTAINER_IMAGE`, `CONTAINER_MOUNTS`, `ACCOUNT`, `PARTITION`, `GPU_LIST`, and `GPUS_PER_NODE` in `submit_trtllm_alltoall.sh` before running.

## 3.1 Parameters (edit in `submit_trtllm_alltoall.sh`)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `CONTAINER_IMAGE` | TensorRT-LLM container image (.sqsh) | `/path/to/tensorrt-llm.sqsh` |
| `CONTAINER_MOUNTS` | Container mount paths (src:dst) | `/yourdata:/yourdata` |
| `ACCOUNT` | Slurm account name | `your account` |
| `PARTITION` | Slurm partition name | `your partition` |
| `GPU_LIST` | Comma-separated GPU counts to test | `2,4,8,16,32,64` |
| `GPUS_PER_NODE` | Number of GPUs per node | `4` (e.g. GB200 NVL72) |

## 3.2 Run the collector

```bash
# Default: NVLinkTwoSided, GPU counts 2,4,8,16,32,64. The 48- and 72-GPU
# worlds are absent because the declared 256-expert shape does not shard
# evenly into either world, so both expand to zero cases.
bash submit_trtllm_alltoall.sh

# NVLinkOneSided, only 2 and 4 GPUs
bash submit_trtllm_alltoall.sh --kernel-source NVLinkOneSided --gpu-list 2,4

# NVLinkTwoSided, custom GPU counts
bash submit_trtllm_alltoall.sh --gpu-list 4,8,16
```

## 3.3 Check results
```bash
squeue -u $USER
ls results/moe_a2a_NVLinkTwoSided.*gpu/job_*/
```

# 4. MoE all-to-all (DeepEP HT + LL) collection

Launches `collector/wideep/sglang/collect_moe_a2a.py` (unified `moe_a2a`
schema, sglang DeepEP) across nodes; one Slurm job per world size, each with
its own output directory and provenance sidecar. The default container image
is the manifest `wideep_sglang` pin.

```bash
# Default: 8,16,32,48,64 GPUs at 4 GPUs/node (GB200). The 72-GPU world is
# absent because no declared wideep shape has an expert count divisible by
# 72, so it expands to zero cases.
bash submit_moe_a2a.sh

# Custom worlds / kernel families
bash submit_moe_a2a.sh --gpu-list 8,16 --modes deepep_ht
```

Configure `CONTAINER_IMAGE`, `CONTAINER_MOUNTS`, `ACCOUNT` and `PARTITION`
via environment variables; `--gpus-per-node` must divide every entry of
`--gpu-list` (the collector derives `node_num = WORLD_SIZE // gpus_per_node`
and raises on a non-integral node count).

## 4.1 Per-job output directories

Each world size runs as its own Slurm job writing to its own
`results/moe_a2a_<N>gpu/job_<jobid>/` directory: the collector finalizes
`moe_a2a_perf.parquet` and attests a world-specific `collection_meta.yaml`
per run, so jobs must not share one output file across worlds (within a job,
all cases append to that job's single staging CSV via `helper.log_perf`'s
lockfile). The trtllm alltoall launcher (section 3) uses the same per-job
layout under `results/moe_a2a_<kernel-source>.<N>gpu/job_<jobid>/`.

## 4.2 MASTER_PORT collision on co-scheduled jobs

Every job exports a fixed `MASTER_PORT=29500` on its own head node. Worlds
that occupy whole nodes cannot collide, but if two jobs are packed onto a
shared node (e.g. `--gpus-per-node` smaller than the physical GPU count, or
other torch-distributed jobs on the same machine) and both elect that node as
rank-0 host, the second rendezvous fails to bind. Symptoms: a job stuck in
`init_process_group` or dying with "address already in use". Workaround:
serialize the submissions or edit the exported port per job in
`submit_moe_a2a.sh`.

## 4.3 Publishing across jobs

The launcher deliberately produces one `(parquet, sidecar)` pair per world
size, but the SDK consumes ONE `moe_a2a_perf.parquet` per
`(system, backend, version)` directory in the family tree:

```
aic-core/src/aiconfigurator_core/systems/data/<system>/comm/sglang/<version>/moe_a2a_perf.parquet
aic-core/src/aiconfigurator_core/systems/data/<system>/comm/sglang/<version>/collection_meta.yaml
```

(`moe_a2a_perf` maps to the `comm` family in
`collector/op_backend_catalog.yaml`; `wideep_sglang` publishes under its
`data_backend`, `sglang`, at the manifest-pinned version. The trtllm
alltoall results publish the same way under `comm/trtllm/<version>/`.)

**There is no automated cross-job merge tool today** — be honest about this
gap when publishing. `tools/perf_database/migrate_family_layout.py` only
relocates existing tables into the family layout, and `collect.py`'s sidecar
handling merges *different* table stems written into one directory: for the
same stem (`moe_a2a_perf` from every world) a later entry would replace the
earlier one, not combine them. The working procedure is manual:

1. Verify every per-job sidecar first: identical `runtime` block, identical
   `collector_ref`/`collector_hash` (same commit, same image). Jobs collected
   from different commits or images must not be merged into one attestation.
2. Concatenate the per-job parquets row-wise into one `moe_a2a_perf.parquet`
   (worlds are disjoint on the `ep_size`/`node_num` key axes, so plain
   concatenation cannot collide).
3. Regenerate ONE sidecar entry for the merged table: `rows` = the merged row
   count; `status` = `complete` only if every per-job sidecar was complete
   (otherwise `partial`); `case_plan_hash` =
   `provenance.case_plan_hash(sorted(union of per-world case-id lists))`,
   where each world's ids are reproduced GPU-free from the collector commit
   the sidecars pin (`build_case_plan` + `case_plan_ids` with that world's
   `ep_size`/`node_num` — the same functions the run used).
4. Never copy a parquet without its sidecar, and never merge fresh provenance
   into a `provenance: legacy` directory — publish to a fresh runtime
   directory or migrate every table in that directory together.
