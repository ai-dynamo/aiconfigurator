# PR #1542 single-node delta campaign

This is a preparation artifact, not a submission script. Replace
`FINAL_COLLECTOR_HASH` only after the final squash is available, sync a clean
checkout at that exact hash to each cluster, and then instantiate the command
templates below. Never submit data collected from a different hash.

| System | Login / partition | GPUs | vLLM HT/LL | vLLM V2 | TRT-LLM HT/LL |
|---|---|---:|---|---|---|
| `gb200` | `ocics001 / batch` | 4 | canary → backend-bound full | capability-failed | canary → backend-bound full |
| `gb300` | `dlcluster / gb300nvl72_preprod` | 4 | canary → backend-bound full | capability-failed | canary → backend-bound full |
| `b200_sxm` | `awscmh / b200@cr+mp-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb` | 8 | canary → backend-bound full | capability-failed | canary → backend-bound full |
| `b300_sxm` | `awscmh / b300@ts5/b300-nvl8@ts5/8gpu-224cpu-2048gb` | 8 | canary → backend-bound full | capability-failed | canary → backend-bound full |
| `h100_sxm` | `dlcluster / dgxh100` | 8 | canary → backend-bound full | canary after final hash | canary → backend-bound full |
| `h200_sxm` | `dlcluster / dgxh200` | 8 | canary → backend-bound full | capability-failed | canary → backend-bound full |

V2 capability-failed means no job is submitted and no formal artifact is
finalized. Re-open those cells only after external transport capability changes
and a fresh non-publishing probe succeeds. It is not a case-generation skip.

For each supported vLLM backend, submit its canary separately, capture the job
ID, then bind the corresponding full job to that backend's canary:

```bash
collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh \
  --system SYSTEM --run-kind canary --backends BACKEND \
  --campaign-root CAMPAIGN --repo-dir REPO_AT_FINAL_COLLECTOR_HASH \
  --vllm-source-root VLLM_SOURCE --container-image IMAGE \
  --legacy-overlay-dir LEGACY_OVERLAY --v2-overlay-dir V2_OVERLAY

collector/wideep/vllm/slurm/submit_vllm_moe_a2a.sh \
  --system SYSTEM --run-kind full --backends BACKEND \
  --afterok-job BACKEND=CANARY_JOB_ID \
  --campaign-root CAMPAIGN --repo-dir REPO_AT_FINAL_COLLECTOR_HASH \
  --vllm-source-root VLLM_SOURCE --container-image IMAGE \
  --legacy-overlay-dir LEGACY_OVERLAY --v2-overlay-dir V2_OVERLAY
```

For TRT-LLM, submit one backend at a time. Stage jobs may seed a validated
same-architecture runtime; canaries use `--afterok-stage-job`, and each full
uses only its same-backend canary through `--afterok-job`.

```bash
collector/wideep/trtllm/slurm/submit_trtllm_moe_a2a.sh \
  --system SYSTEM --run-kind canary --backend BACKEND \
  --afterok-stage-job STAGE_JOB_ID --campaign-root CAMPAIGN \
  --repo-dir REPO_AT_FINAL_COLLECTOR_HASH --container-image IMAGE --wheel-dir WHEEL_DIR

collector/wideep/trtllm/slurm/submit_trtllm_moe_a2a.sh \
  --system SYSTEM --run-kind full --backend BACKEND \
  --afterok-job CANARY_JOB_ID --campaign-root CAMPAIGN \
  --repo-dir REPO_AT_FINAL_COLLECTOR_HASH --container-image IMAGE --wheel-dir WHEEL_DIR
```
