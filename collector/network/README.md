# Network Collectors

This folder contains collective communication and network-facing collector
scripts. These collectors are intentionally separate from framework op
registries because their output feeds shared communication tables instead of a
single framework backend registry.

- `collect_comm.sh`: local NCCL/oneCCL plus custom allreduce collection driver.
- `collect_nccl.py`: local NCCL collective benchmark wrapper.
- `collect_oneccl_xpu.py`: local oneCCL XPU collective benchmark wrapper.
- `collect_all_reduce.py`: local custom allreduce benchmark wrapper.
- `slurm/`: multi-node Slurm communication collectors and post-processing.

Perf output names stay unchanged (`nccl_perf.txt`, `oneccl_perf.txt`,
`custom_allreduce_perf.txt`, and `trtllm_alltoall_perf.txt`) so existing perf
database organization does not change with this source layout split.
