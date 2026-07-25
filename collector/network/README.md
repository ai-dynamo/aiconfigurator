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

The standalone scripts keep their staging output names (`nccl_perf.txt`,
`oneccl_perf.txt`, `custom_allreduce_perf.txt`, and
`trtllm_alltoall_perf.txt`). Collector finalization converts accepted output
to parquet under
`aic-core/src/aiconfigurator_core/systems/data/<system>/comm/<backend>/<version>/`.
