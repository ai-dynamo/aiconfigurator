# WideEP collectors

WideEP collectors use runtime images independently from the normal framework
collectors. The authoritative versions, source commits, ABI pins, and images
are in `collector/framework_manifest.yaml`.

Layout:

- `sglang/collect_deepep_moe.py`: SGLang DeepEP expert compute (`moe_ep`).
- `sglang/collect_moe_a2a.py`: standalone multi-node SGLang DeepEP
  communication (`moe_a2a`).
- `trtllm/collect_moe_compute.py`: TensorRT-LLM expert compute (`moe_ep`).
- `trtllm/collect_moe_a2a.py`: TensorRT-LLM serving-path communication.
- `vllm/collect_moe_a2a.py`: vLLM 0.24.0 serving-path communication for
  `deepep_ht`, `deepep_ll`, and `deepep_v2`.
- `vllm/collect_moe_ep.py`: dormant vLLM expert-compute implementation; it is
  not registered until its serving dispatch identity is hardware-verified.
- `vllm/slurm/`: the fail-closed six-system canary/full campaign launcher.
- `vllm/finalize_campaign.py`: validates six independent backend/topology
  jobs for one system and atomically merges the publishable parquet/sidecar.
- `sglang/deepep/`: deprecated manual log collection and extraction scripts.

## vLLM 0.24.0 multi-node campaign

The formal matrix is:

| System | GPUs/node | 2-node identity | 4-node identity |
| --- | ---: | --- | --- |
| `gb200` | 4 | EP8 | EP16 |
| `gb300` | 4 | EP8 | EP16 |
| `b200_sxm` | 8 | EP16 | EP32 |
| `b300_sxm` | 8 | EP16 | EP32 |
| `h100_sxm` | 8 | EP16 | EP32 |
| `h200_sxm` | 8 | EP16 | EP32 |

Each topology runs three independent jobs, one per backend. A short 2-node
canary must succeed for every backend before full jobs may be submitted.
Non-default LL transport flags are diagnostic: the collectors keep staging
rows but do not finalize parquet or a sidecar under the default identity.

`run_vllm_moe_a2a_job.sh` resolves and checks every repository, source,
image, cache, log, staging, and output path. `/mnt/cifs` and `/mnt/nvdl` are
always rejected. Artifacts finalize in job-unique `/tmp`, receive parquet and
sidecar checksums, and are copied to the campaign root only after validation.

GB200, GB300, H100, and H200 allocations must be contained in exactly one
authoritative Slurm leaf switch or NVL block. ComputeLab exposes B200/B300
through `topology/flat`; consequently B200/B300 submission additionally
requires an infrastructure-approved exact nodelist and approval ID. Without
both, the launcher exits before `sbatch`.

## Deprecated SGLang manual pipeline

The old `sglang/deepep/` scripts derived identity columns from constants and
parsed `node_num` from log filenames. Do not use them to add shipped data.
The legacy `wideep_deepep_{normal,ll}_perf` tables remain readable through SDK
adapters; deprecation retires the producer pipeline, not existing data.

## Activating vLLM `moe_ep`

The vLLM WideEP runtime and empty registry now exist for standalone
`moe_a2a`, but `collect_moe_ep.py` remains deliberately dormant. Enrollment
is one coordinated change:

1. Prove on the pinned vLLM 0.24.0 runtime that the benchmarked
   `fused_experts` call is the serving kernel selected for the large-EP DeepEP
   path and confirm its global-token accounting.
2. Add only the verified `moe_ep` `OpEntry` to
   `collector/wideep/vllm/registry.py` and set the module's `__compat__` pin.
3. In the same commit, add its collector hash closure and the cited vLLM
   `deepep_moe` kernel-source mapping.
4. Replace the dormant contract tests with positive registry, runtime,
   kernel-identity, and hardware validation tests.
