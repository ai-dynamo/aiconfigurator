# DeepSeek-V4 MegaMoE Collector

This directory contains the cross-rank DSv4 MegaMoE collection entrypoints.

## Measurement Boundary

The collector measures:

```text
prepared hidden_states + prepared topk_ids/topk_weights
  -> SGLang-style cached symmetric buffer
  -> SGLang/DeepGEMM pre-dispatch into the symmetric buffer
  -> deep_gemm.fp8_fp4_mega_moe
  -> SGLang routed output scaling
```

It does not measure gate GEMM, top-k selection, routing generation, routing
validation, or distributed setup.  The perf rows set
`includes_gate_topk=false` and `includes_routed_scale=true`.
It also does not include the unfused shared-expert MLP when SGLang executes it
outside `deep_gemm.fp8_fp4_mega_moe`; this collector is scoped to the routed
MegaMoE module.
The cold symmetric buffer allocation/rendezvous path is outside per-module
latency, matching SGLang's cached-buffer steady state.  Perf rows record
`buffer_policy=cached_sglang` and `includes_buffer_init=false`.

CUDA graph capture is mandatory.  The collector calls AIC
`benchmark_with_power(..., use_cuda_graph=True, allow_graph_fail=False)` and
raises if any rank does not use CUDA graph.

This collector is specifically for MegaMoE.  The module-level path directly
calls `deep_gemm.fp8_fp4_mega_moe`; it does not benchmark the ordinary DeepEP
MoE module or the `flashinfer_mxfp4` runner path.  The launch wrapper also sets
the SGLang cookbook MegaMoE environment:

```bash
SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
SGLANG_OPT_FIX_HASH_MEGA_MOE=1
SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=<selected max local tokens>
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0
```

For end-to-end SGLang alignment, the serving command must also use
`--moe-a2a-backend deepep`; otherwise the run is not a MegaMoE comparison.

The perf row separates the AIC quant-mode label from the actual kernel path:
`moe_dtype=w4a8_mxfp4_mxfp8` and `kernel_dtype=fp8_fp4`.  In the DeepGEMM
MegaMoE kernel this means FP8 activation operands and FP4 weight operands.
The collector also follows the DSv4 model configs: `norm_topk_prob=true`, with
`routed_scaling_factor=1.5` for Flash and `2.5` for Pro unless explicitly
overridden.

The routed MegaMoE path uses the same SGLang/DeepGEMM kernel for context and
generation.  The Kubernetes and Slurm runners can still be split with
`PHASES=context` and `PHASES=generation` to reduce collection scope, but both
phases write standard rows to the same `dsv4_megamoe_module_perf.txt` table.
The default split is:

- context/prefill tokens: `1024,2048,4096,8192,16384,32768`
- generation/decode tokens: `1,2,4,8,16,32,64,128,256,512`

Together these cover the standard MoE token sweep.  For GB200 full collection,
run both context and generation for EP4, EP8, EP16, and EP32 when all EP sizes are
needed by the AIC query path.  Use `run_k8s_full_collection.sh` for Kubernetes
or `run_slurm_full_collection.sh` for Slurm.  The Kubernetes runner points
every job at the same output directory and `dsv4_megamoe_module_perf.txt`; the
Slurm runner writes per-job perf files and merges them locally after all jobs
complete.  Both paths validate the same logical row count and reject duplicate
loader keys.

Each row records both `num_max_tokens_per_rank` and
`effective_num_max_tokens_per_rank`.  The effective value is read from the
DeepGEMM symmetric buffer after DeepGEMM applies its internal token alignment.
The launch runners/renderers set `NUM_MAX_TOKENS_PER_RANK`; `run_torchrun.sh`
requires it to be a positive integer.  The default cap policy is `case_tokens`,
matching legacy baselines; set `CAP_POLICY=fixed` when you want one fixed buffer
cap across the sweep.

The committed perf schema follows the existing SGLang WideEP MoE tables and
keeps only query keys plus required boundary/topology fields:

```text
framework,version,device,op_name,kernel_source,
phase,moe_dtype,kernel_dtype,num_tokens,global_num_tokens,
hidden_size,inter_size,topk,num_experts,num_fused_shared_experts,
moe_tp_size,moe_ep_size,distribution,source_policy,pre_dispatch,
num_max_tokens_per_rank,effective_num_max_tokens_per_rank,
routed_scaling_factor,includes_routed_scale,includes_gate_topk,
buffer_policy,includes_buffer_init,used_cuda_graph,latency
```

`num_tokens` is the local-rank token count used by AIC's MegaMoE query path;
`global_num_tokens` is recorded only to make cross-rank collection explicit.

## Routing Distributions

The collector supports:

- `balanced`
- `power_law_1.01`
- `power_law_1.2`
- `power_law_sampled_1.9`

The distribution helpers are AIC's existing `collector.helper.balanced_logits`
and `collector.helper.power_law_logits_v3`.  `power_law_sampled_1.9` keeps
the same inverse-CDF power-law weight family, samples the curve with
`xmax=1024`, then samples each token's top-k experts without replacement.
The default MegaMoE collection includes
`power_law_sampled_1.9`.  When this sampled distribution is requested without
an explicit `--routing-seeds` list, collection expands it to ten consecutive
routing seeds starting at `--routing-seed` and writes one perf row whose latency
is the mean of those seed samples. The standard perf schema intentionally omits
the routing seed column, so repeated samples are collapsed during collection.
Other synthetic distributions use the single `--routing-seed` unless
`--routing-seeds` is provided.
`balanced` is already a deterministic per-token top-k assignment and therefore
does not have a separate sampled variant.

`distribution` controls destination expert load.  `source_policy=random`
shuffles complete token rows before assigning them to source ranks, preserving
expert counts and per-token top-k structure while making the source/destination
traffic explicit.  Source placement is prepared before timing, so it does not
add overhead to the measured MegaMoE module.

## Hardware Mapping

The scripts parameterize GPU topology:

- B200: default `GPUS_PER_NODE=8`
- GB200: default `GPUS_PER_NODE=4`
- B300: default `GPUS_PER_NODE=8`
- GB300: default `GPUS_PER_NODE=4`

Override `GPUS_PER_NODE` for any cluster-specific layout.

The Kubernetes renderer also follows the DeepSeek-V4 SGLang cookbook image
mapping when `--image` is omitted:

- B200: `lmsysorg/sglang:deepseek-v4-blackwell`
- GB200: `lmsysorg/sglang:deepseek-v4-grace-blackwell`
- B300: `lmsysorg/sglang:deepseek-v4-b300`
- GB300: `lmsysorg/sglang:deepseek-v4-grace-blackwell`

The GB200 image was smoke-tested on `dynamo-gcp-dev-02`: it reports
`NVIDIA GB200`, `sglang 0.5.10rc0`, imports `deep_gemm`, and exposes
`deep_gemm.fp8_fp4_mega_moe`.

For DSv4 cookbook images, do not mount a PVC at `/workspace`: the image keeps
its DSv4 SGLang source tree under `/workspace/sglang`, and shadowing that path
causes imports such as `sglang.jit_kernel.deepseek_v4` to fail.  The renderer
defaults the PVC mount to `/mnt/shared` for this reason.

For GB200 runs spanning more than one node, the renderer follows the cookbook
and emits `NCCL_MNNVL_ENABLE=1` and `NCCL_CUMEM_ENABLE=1`.  Add cluster-local
network overrides with repeated `--env KEY=VALUE` arguments when needed.  Later
`--env` entries override renderer defaults; for example, use
`--env NCCL_MNNVL_ENABLE=0` on clusters where IMEX/MNNVL channels are not
configured.

## Single Or Manual Multi-Node Run

Run one process per GPU with `torchrun`:

```bash
SYSTEM_NAME=b200_sxm \
EP_SIZE=8 \
GPUS_PER_NODE=8 \
SOURCE_POLICY=random \
NUM_MAX_TOKENS_PER_RANK=32768 \
OUTPUT_PATH=/workspace/results/dsv4_megamoe_b200_ep8 \
bash collector/sglang/dsv4_megamoe/run_torchrun.sh
```

For manual multi-node runs, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and
`MASTER_PORT` on each node.  Example for GB200 EP16:

```bash
SYSTEM_NAME=gb200 \
EP_SIZE=16 \
GPUS_PER_NODE=4 \
NNODES=4 \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=29500 \
NUM_MAX_TOKENS_PER_RANK=32768 \
OUTPUT_PATH=/workspace/results/dsv4_megamoe_gb200_ep16 \
bash collector/sglang/dsv4_megamoe/run_torchrun.sh
```

## Kubernetes

Render an Indexed Job plus headless Service:

```bash
python3 collector/sglang/dsv4_megamoe/render_k8s_indexed_job.py \
  --job-name aic-dsv4-megamoe-gb200-ep16 \
  --namespace ${NAMESPACE} \
  --system-name gb200 \
  --ep-size 16 \
  --source-policy random \
  --pvc-name ${PVC_NAME} \
  --pvc-mount /mnt/shared \
  --node-selector kubernetes.io/arch=arm64,nvidia.com/gpu.product=NVIDIA-GB200 \
  --toleration-key kubernetes.io/arch \
  --toleration-key nvidia.com/gpu \
  --working-dir /mnt/shared/aiconfigurator \
  --output-path /mnt/shared/aiconfigurator/collector/sglang/dsv4_megamoe/results \
  --num-max-tokens-per-rank 32768 \
  > /tmp/aic-dsv4-megamoe-gb200-ep16.yaml

kubectl apply -f /tmp/aic-dsv4-megamoe-gb200-ep16.yaml
```

Cleanup path:

```bash
bash collector/sglang/dsv4_megamoe/cleanup_k8s_job.sh \
  aic-dsv4-megamoe-gb200-ep16 ${NAMESPACE}
```

Always run cleanup after completion, failure, timeout, or interruption.

For the standard GB200 full matrix, run:

```bash
NAMESPACE=${NAMESPACE} \
PVC_NAME=${PVC_NAME} \
GPU_CLIQUE=${GPU_CLIQUE} \
LOCAL_REPO=/path/to/aiconfigurator \
bash collector/sglang/dsv4_megamoe/run_k8s_full_collection.sh
```

Defaults are:

- `PREFILL_EP_SIZES=4,8,16,32`
- `DECODE_EP_SIZES=4,8,16,32`
- `PREFILL_TOKENS=1024,2048,4096,8192,16384,32768`
- `DECODE_TOKENS=1,2,4,8,16,32,64,128,256,512`
- `DISTRIBUTIONS=balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9`

The final local result is:

```text
${LOCAL_RESULT_DIR}/remote_results/dsv4_megamoe_module_perf.txt
```

`validation_summary.txt` must report `perf_files=1` and `VALIDATION=PASS`.

## Slurm

Render and submit the standard full matrix through Slurm:

```bash
SSH_TARGET=${SSH_TARGET} \
SLURM_ACCOUNT=${SLURM_ACCOUNT} \
SLURM_PARTITION=${SLURM_PARTITION} \
SYSTEM_NAME=gb300 \
LOCAL_REPO=/path/to/aiconfigurator \
bash collector/sglang/dsv4_megamoe/run_slurm_full_collection.sh
```

`SSH_TARGET` is required for non-dry-run Slurm execution.
`DRY_RUN=1` renders the `.sbatch` files locally without staging or submitting.
`TEST_ONLY=1` stages the repo and runs `sbatch --test-only` for every rendered
job.  If a submitted Slurm run fails while `WAIT=1`, the runner writes
`cancel_jobs.sh` and cancels the submitted jobs unless `KEEP_JOBS=1`.
Set `COPY_VALIDATED=1` to copy the merged perf file into
`src/aiconfigurator/systems/data/${SYSTEM_NAME,,}/sglang/${TARGET_SGLANG_VERSION}`;
set `ALLOW_VERSION_MISMATCH=0` when that copy should require exact collected
SGLang version matching.

The final local result is:

```text
${LOCAL_RESULT_DIR}/merged/dsv4_megamoe_module_perf.txt
```

The Slurm validation follows the Kubernetes validation semantics: sampled seed
measurements must already be averaged into one row per logical case, and
`validation_summary.txt` must report `duplicate_loader_keys=0` and
`VALIDATION=PASS`.

## Alignment Cases

The alignment gate should run these collector cases and compare them with the
corresponding SGLang end-to-end MegaMoE timeline:

- Prefill: `isl=1024,8192`
  - GB200: `EP=4`
  - B200/B300: `EP=8`
- Decode: `bs=16,64,256`
  - GB200: `EP=8,16,32`

The comparison must check kernel names, CUDA graph coverage, and latency gap.
