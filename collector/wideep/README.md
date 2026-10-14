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
- `vllm/finalize_campaign.py`: validates three independent backend
  jobs for one system and atomically merges the publishable parquet/sidecar.
- `trtllm/slurm/`: the fail-closed TensorRT-LLM source-wheel stage and
  single-node canary/full campaign launcher.
- `trtllm/finalize_campaign.py`: validates independent HT and LL jobs and
  merges them without relabeling the pinned TensorRT-LLM runtime.
- `sglang/deepep/`: deprecated manual log collection and extraction scripts.

## vLLM 0.24.0 single-node campaign

The formal matrix is:

| System | Nodes | GPUs/node | Formal identity |
| --- | ---: | ---: | --- |
| `gb200` | 1 | 4 | EP4 |
| `gb300` | 1 | 4 | EP4 |
| `b200_sxm` | 1 | 8 | EP8 |
| `b300_sxm` | 1 | 8 | EP8 |
| `h100_sxm` | 1 | 8 | EP8 |
| `h200_sxm` | 1 | 8 | EP8 |

Each system runs three independent jobs, one per backend. A short single-node
canary must succeed for every backend before its full job may be submitted.
Non-default LL transport flags are diagnostic: the collectors keep staging
rows but do not finalize parquet or a sidecar under the default identity.

`run_vllm_moe_a2a_job.sh` resolves and checks every repository, source,
image, cache, log, staging, and output path. `/mnt/cifs` and `/mnt/nvdl` are
always rejected. Artifacts finalize in job-unique `/tmp`, receive parquet and
sidecar checksums, and are copied to the campaign root only after validation.

The formal launcher rejects every node count other than one. ComputeLab exposes
B200/B300 through `topology/flat`; consequently B200/B300 submission also
requires an infrastructure-approved exact nodelist and approval ID. Without
both, the launcher exits before `sbatch`.

## TensorRT-LLM 1.3.0rc11 single-node campaign

TensorRT-LLM uses the same six-system one-node layout as vLLM: GB200/GB300
run EP4 on four GPUs, while B200/B300/H100/H200 run EP8 on eight GPUs. Each
system has two independent chains: `trtllm_deepep_ht` and
`trtllm_deepep_ll`, with every full job gated by its own successful canary.

The configured runtime identity is the multi-architecture TensorRT-LLM rc20
container index, used only as the immutable build base. Image staging resolves
and records the platform child, checks out source commit `14efb6ac`, verifies
the vendored DeepEP and NVSHMEM pins, builds package version `1.3.0rc11`, and
records the source-wheel SHA256. The runner installs that wheel in a job-local
overlay before entering the MPI collector.

Image staging may reuse an already attested same-CPU-architecture squashfs.
Supplying its wheel directory additionally reuses the complete runtime only
when the source and target CUDA architecture sets are identical. Seed image,
wheel, dependencies, pins, and metadata are checksum-validated, and the seed
provenance is carried through job evidence and the final sidecar. Cross-SM
wheel reuse fails closed.

Case failures are never allowlisted. A failed full job preserves its partial
parquet, sidecar, and rank failure records in campaign failure evidence. The
finalizer rejects those inputs by default; `--allow-partial-evidence` may be
used only to assemble an explicitly `partial` evidence table, never a complete
formal publication.

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
