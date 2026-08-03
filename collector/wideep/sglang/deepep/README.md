> **DEPRECATED — superseded by `collector/wideep/sglang/collect_moe_a2a.py`.** The standalone `moe_a2a` collector runs the same DeepEP HT and low-latency benchmarks as a Collector-V3 citizen: it emits the unified `moe_a2a_perf` table directly at measurement time with live identity columns (`node_num` from `WORLD_SIZE // --gpus-per-node`, device from torch, version cross-checked against the `wideep_sglang` manifest pin) and writes a real `collection_meta.yaml` sidecar. The scripts below and `extract_data.py` derive those same columns from module-level constants and log filenames (`extract_data.py:14-21`), which is why they are being retired. They remain here only until the new collector is hardware-validated; **do not add new data to the shipped tables via `extract_data.py`.**
>
> Use instead: `bash collector/network/slurm/submit_moe_a2a.sh` (one Slurm job
> per world size; launcher usage, per-job output layout and the cross-job
> publish procedure are in section 4 of `collector/network/slurm/README.md`),
> or launch the collector directly with
> `torchrun --nnodes <N> --nproc-per-node <G> collector/wideep/sglang/collect_moe_a2a.py --gpus-per-node <G>`.
> The new table publishes under `<system>/comm/sglang/<version>/moe_a2a_perf.parquet`
> in the family layout; the SDK loads it alongside the legacy
> `wideep_deepep_{normal,ll}_perf` tables through
> `load_moe_a2a_data`'s legacy adapters, with new-schema rows taking
> precedence at identical keys. Note the adapters assume the legacy fleets
> were 8 GPUs/node (`ep_size = node_num * 8`), so new worlds collected at a
> different GPUs-per-node **coexist** with rather than replace the legacy
> leaves (see "MoE table units and caveats" in `collector/README.md`).
> Already-shipped legacy tables stay loadable indefinitely — this retires the
> pipeline, not the data. Once the new collector is hardware-validated
> (the two flagged port risks: LL max-buffer reuse and `bench_kineto` name
> matching at the pinned DeepEP), this directory's scripts and
> `extract_data.py` are deleted.

Guidance for collecting deepep data in normal and low-latency modes.

Notes:
- MASTER_ADDR: IP address of the node with RANK=0.
- WORLD_SIZE: total number of nodes.
- RANK: 0-based index for this node.
- {num_node}: total number of nodes (e.g., 2 or 4).
- xxx: GPU type/model (e.g., A100, H100).

# Build Docker

Note: The test files under `collector/wideep/sglang/deepep/` are sourced from [DeepEP](https://github.com/deepseek-ai/DeepEP/tree/main/tests) with some modifications applied.

```bash
docker build -t deepep:latest -f docker/Dockerfile.deepep .
docker run -it --network host --gpus all -v aiconfigurator/collector/wideep/sglang/deepep:/new_workspace --privileged deepep:latest bash
```

# Two-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.20
export WORLD_SIZE=2
export MASTER_PORT=40303
export RANK=1
```

# Four-node configuration

Server:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=0
```

Client:
```bash
export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=1

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=2

export MASTER_ADDR=10.6.131.13
export WORLD_SIZE=4
export MASTER_PORT=40303
export RANK=3
```

# Test intra-node mode

Run the following command on a single node:
```bash
python /new_workspace/test_intranode.py |& tee deepep_node_1_mode_normal.log
```

# Test inter-node normal mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py |& tee deepep_node_{num_node}_mode_normal.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py
```

# Test low-latency mode

On the first node:
Note: Replace {num_node} with the total number of nodes (e.g., 2 or 4).
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility |& tee deepep_node_{num_node}_mode_ll.log
```
On the other node(s):
```bash
python /new_workspace/test_internode.py  --test-ll-compatibility
```

# Post-process log files
Keep raw DeepEP logs in a staging directory outside the packaged perf-data
tree. Replace `xxx` with the GPU type (for example, A100) and point `--log-dir`
to that directory.
```bash
python collector/wideep/sglang/deepep/extract_data.py \
  --log-dir path/to/deepep-logs/xxx/sglang/<sglang_version>/
```

This legacy publish contract applies only to data already collected through
this pipeline (new collections go through `collect_moe_a2a.py` — see the
deprecation notice at the top). After validation and parquet finalization,
publish `wideep_deepep_{normal,ll}_perf.parquet` under
`aic-core/src/aiconfigurator_core/systems/data/<system>/comm/sglang/<sglang_version>/`.
Publishing must also write or refresh the corresponding table entry in
`collection_meta.yaml`; never copy a parquet table without its provenance.
Do not merge fresh provenance into a `provenance: legacy` directory—publish
to a fresh runtime directory or migrate every table in that directory
together.
