# Layerwise Collector

vLLM layerwise/FPM collection tools for comparing one-layer GPU attribution
against Dynamo forward-pass metrics.

Included paths:

- `vllm/`: vLLM layerwise collector, layer-skip patch, and step marker.
- `fpm/`: simple public FPM collection CLI.
- `fpm_ground_truth/`: lower-level Dynamo FPM ground-truth collection helpers.
- `common/`: shared config patching, nsys parsers, and random prompt-token helpers.

Public entrypoints:

```bash
python -m collector.layerwise.vllm.collect
python -m collector.layerwise.fpm.collect --model Qwen/Qwen3-32B
```

## Quick Smoke Tests

The layerwise collector requires vLLM and Nsight Systems. Run it inside the
Dynamo vLLM runtime and mount host Nsight:

```bash
export AIC_REPO="${AIC_REPO:-$PWD}"
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\n' < ~/hf.token)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-$HOME/.cache/aic-vllm}"
export NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
export RUN_DIR="$AIC_REPO/.tmp/smoke-layerwise-$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" "$HF_HOME" "$VLLM_CACHE_HOST/tilelang/tmp"

docker run --rm --ipc=host --network=host \
  --gpus '"device=0"' \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$AIC_REPO:/workspace" \
  -v "$RUN_DIR:/results" \
  -v "$HF_HOME:/hf-cache" \
  -v "$VLLM_CACHE_HOST:/home/dynamo/.cache/vllm" \
  -v "$VLLM_CACHE_HOST:/root/.cache/vllm" \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
  -e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
  -e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
  -e HF_TOKEN="$HF_TOKEN" \
  -w /workspace \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; python3 -m collector.layerwise.vllm.collect --run-dir /results --models Qwen/Qwen3-32B --tp-sizes 1 --phases both --run-preset smoke --gpus 0 --max-workers 1'
```

Expected layerwise smoke output is `layerwise.csv` with four rows: two `ctx`
rows and two `gen` rows. Nsight artifacts are under `profiles/nsys/`.

The layerwise CLI defaults to full Nsight capture. This is intentional: vLLM
captures CUDA graphs during engine setup/warmup, and the parser needs those
graph-node records to attribute decode replay kernels back to the measured
layer. `--nsys-capture cuda_profiler_api` is a diagnostic speed mode and should
not be used for AIC-quality layerwise data.

Production decode batch sizes default up to `256`. Use
`--max-decode-batch-size auto` to follow vLLM's hardware-dependent
`max_num_seqs` default, or use `--gen-batch-sizes` to request an exact decode
grid.

The full preset uses `--ctx-batch-sizes auto` by default. Auto keeps
single-request context rows and adds a bounded batched-context grid whose
aggregate scheduled tokens fit the resolved `max_num_batched_tokens` budget.
Use an explicit `--ctx-batch-sizes` list only for diagnostic shapes.

## Full Collection Run

Run the full default registry (Qwen3-32B, Qwen3.6 MoE, DeepSeek-V4-Flash) on all
GPUs:

```bash
export AIC_REPO="$PWD"
export RUN_DIR="$AIC_REPO/runs/layerwise_full_vllm0201_$(date -u +%Y%m%d_%H%M%S)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-$HOME/.cache/aic-vllm}"
export NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
mkdir -p "$RUN_DIR" "$HF_HOME" "$VLLM_CACHE_HOST/tilelang/tmp"

docker run --rm --gpus all --ipc=host --network=host --entrypoint bash \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$AIC_REPO:/workspace" \
  -v "$RUN_DIR:/results" \
  -v "$HF_HOME:/hf-cache" \
  -v "$VLLM_CACHE_HOST:/home/dynamo/.cache/vllm" \
  -v "$VLLM_CACHE_HOST:/root/.cache/vllm" \
  -v "$HOME/hf.token:/run/secrets/hf.token:ro" \
  -e HF_HOME=/hf-cache -e HF_HUB_CACHE=/hf-cache/hub \
  -e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
  -e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
  -e NSYS_VERSION="$NSYS_VERSION" \
  -w /workspace \
  vllm/vllm-openai:v0.20.1 \
  -lc 'set -euo pipefail
       export PATH="/opt/nvidia/nsight-systems/${NSYS_VERSION}/target-linux-x64:$PATH"
       export HF_TOKEN="$(tr -d "\n" < /run/secrets/hf.token)"
       python3 -m collector.layerwise.vllm.collect --run-dir /results --max-workers 8'
```

Progress from another shell:

```bash
wc -l "$RUN_DIR/layerwise.csv"
tail -f "$RUN_DIR/profiles/status.jsonl"
```

Then compare against the FPM golden runs (see the status section below for the
current numbers and the chart tool):

```bash
uv run python collector/layerwise/diagnostics/compare_aic_layerwise_fpm_summary.py \
  --layerwise "$RUN_DIR/layerwise.csv"
```

Decode timing now uses the `execute_model_gpu` source (do not re-enable the old
`--live-step-driver` path, which produced the corrupt MoE decode curves).

## Scheduler Sizing

Do not hardcode `max_num_seqs` or `max_num_batched_tokens` in layerwise or FPM
commands. They are part of the deployment shape and must either be inferred
from the collector defaults/vLLM effective config or set intentionally for a
specific deployment being measured. Do not copy values from old FPM artifacts.

For decode, `batch_size` is the number of active sequences in one vLLM
iteration, so the requested decode grid must satisfy:

```text
max(gen_batch_sizes) <= max_num_seqs
```

If a comparison FPM run used a smaller `max_num_seqs`, compare only rows inside
that deployment envelope or recollect FPM with the intended scheduler config.
For context and mixed rows, `max_num_batched_tokens` controls the per-iteration
token budget and must match the deployment being modeled; otherwise chunking,
scheduler behavior, and CUDA graph selection are not comparable.

If a work unit crashes repeatedly with the same error and no parsed successes,
the scheduler records a `work_unit_omitted` event in `profiles/status.jsonl`
with the stderr tail and marks the remaining datapoints `skipped_same_error`.

FPM uses the public Python wrapper around the lower-level shell collector.
By default it sends a real-workload mixed request stream with deterministic
random token IDs for model-agnostic reliability. OpenAssistant/oasst1 supplies
the request ordering, then the ISL/OSL values are scaled to a larger serving
regime: ISL `100..16384` with mean roughly `4096`, and OSL `100..4096` with
mean roughly `1024`. Use `--no-real-workload` with explicit shape overrides for
static debug runs.

```bash
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\n' < ~/hf.token)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0
export RUN_DIR="$PWD/.tmp/fpm-$(date -u +%Y%m%d_%H%M%S)"

python -m collector.layerwise.fpm.collect \
  --model Qwen/Qwen3-32B \
  --run-dir "$RUN_DIR" \
  --tp-sizes 1 \
  --ep-sizes 1 \
  --real-workload-requests 128 \
  --real-workload-concurrency 32 \
  --warmup-requests 1 \
  --gpus '"device=0"'
```

Expected FPM output includes `fpm_metrics.csv`, `fpm_metrics_phase.csv`,
`request_workload.csv`, `warmup_workload.csv`, `vllm_metadata.json`, and
`effective_vllm_config.json`.

## Code Structure

- `vllm/collect.py`: public layerwise CLI and worker subcommand dispatch.
- `vllm/registry.py`: default model registry and model filters.
- `vllm/datapoint_generator.py`: context/decode shape grids, TP/EP expansion,
  and patched-model work-unit generation.
- `vllm/scheduler.py`: one-GPU-slot scheduler, retry logic, status log, and
  nsys export/parse ingestion.
- `vllm/worker.py`: vLLM engine setup and measurement drivers executed under
  `nsys profile`.
- `vllm/engine.py`: vLLM engine argument defaults and deployment-parity setup.
- `vllm/nsys.py`: parsed nsys row aggregation and latency-source selection.
- `vllm/results.py`: layerwise CSV schema and row writing.
- `vllm/data.py`: dataclasses shared by scheduler and worker.
- `vllm/runtime.py`: small filesystem, hashing, GPU, and version helpers.
- `fpm/collect.py`: public FPM ground-truth CLI.
- `fpm/datapoint_generator.py`: FPM default workload shapes and TP/EP case expansion.
- `fpm/docker.py`: command construction for the existing shell collector.
- `fpm/artifacts.py`: FPM run-directory layout helpers.
- `common/paths.py`: shared default artifact/run-directory helpers.

The SGLang and TensorRT-LLM prototype collectors are intentionally not included
in this AIC branch.

Current validation data is stored under
`src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`.
The public vLLM collector defaults to phase-specific representative depth and
latency source. Context rows use a scheduler/worker timing envelope so the
backend can consume them as full-step data. **Decode rows now use the
GPU-isolated `execute_model_gpu` source** (one uniform timing method across all
batch sizes), which replaced the earlier scheduler-envelope sources
(`schedule_to_update`/`live_step_wall`) that produced non-physical MoE decode
curves — see "FPM-vs-AIC MoE modeling: status & handoff" below. Critically, the
timing method must never switch with batch size or shape. Diagnostic
`--latency-source span`, `gpu`, and `gpu_capped` rows are still available for
module-level analysis, but they should not be promoted as backend-ready decode
data without metadata that prevents representative layer scaling.
`--target-layer-count` overrides both phases; `--ctx-target-layer-count` and
`--gen-target-layer-count` override them independently.

MoE layerwise collection is phase-specific by default. Context rows keep dummy
MoE weights active, which captures the vLLM MoE module/envelope cost without
downloading real checkpoint weights. Decode rows use a no-op MoE block and AIC
composes those rows with op-level routed MoE, router, shared-expert, and comm
data. This split avoids real-weight dependency while keeping decode collection
fast. New layerwise CSVs include `moe_weight_mode` (`dummy`, `noop`,
`real_router`, or `dense`) so runs can be audited without reconstructing the
launch command. `--moe-noop` forces no-op MoE for all selected phases,
`--moe-dummy-router` forces dummy MoE for all selected phases, and
`--moe-real-router` is only a diagnostic mode for inspecting routing-specific
behavior on a local snapshot.

The full shape grid is intentionally sparse at very small token counts.
Context new-token rows use `1,16,128,...`; context and decode share the same
nonzero past-KV grid up to 32k tokens, with context adding `0` for no-prefix
measurements.

## FPM-vs-AIC MoE modeling: status & handoff

### Decode timing: RESOLVED
The "Fix MoE decode GPU timing collection" change added the uniform GPU-isolated
latency source `execute_model_gpu` (one method for every batch size — no more
`schedule_to_update`/`live_step_wall` mixing). New decode runs are clean and
monotonic; decode MAPE is **qwen36 ~8.9%, dsv4 ~11.7%** (was catastrophic, 90%+;
e.g. the old `dsv4_tp4_ep4` 192% decode outlier is gone). Latest MoE decode run:
`runs/layerwise_moe_decode_execgpu_stablep25_compare_20260616_212000/`.

**Most important rule for any re-collection:** never switch the timing method /
`latency_source` by batch size or shape. The corruption came from mixing
`schedule_to_update` (small batch) with `live_step_wall` (batch>=8) in one sweep
(a step-change discontinuity), plus `live_step_wall` capturing per-step host/wall
overhead (a fixed ~7-8 ms floor that even exceeded the real full step) instead of
the GPU step. Use one GPU-isolated method for every batch, KV, model, and phase.

### Open: mixed step is context/prefill-limited (not decode)
With clean decode, mixed is still ~47% MAPE for qwen36 because (a) the golden-set
mixed steps are context/prefill-dominated (`decode_delta ~= 0` — decode barely
contributes) and (b) the headline number is polluted by FPM measurement outliers
(e.g. `ctx_tokens=928` reported at ~1.6-2.8 ms, physically impossible; AIC's
~31 ms is right). Legitimate large-prefill steps are ~+8 to +41%. Next focus is
the context/prefill model and FPM-outlier handling — NOT decode.

### Charts (FPM vs AIC)
`tools/plot_fpm_vs_aic.py` writes ctx/gen/mixed log-log charts plus an all-reduce
comparison. The new decode runs are decode-only (`phase=gen`, `max_num_seqs=64`,
`past_kv in {4096,8192,16384}`), so they refresh only the gen charts:
```bash
.venv/bin/python tools/plot_fpm_vs_aic.py \
  --layerwise runs/layerwise_moe_decode_execgpu_stablep25_compare_20260616_212000/layerwise.csv \
  --model "Qwen/Qwen3.6-35B-A3B" \
  --moe-perf-file collector/layerwise/wip/moe_perf.txt \
  --out-dir fpm_vs_aic_charts_qwen36 --phases gen --vllm-max-num-seqs 64
```
For dense Qwen3-32B drop `--moe-perf-file`. A clean **mixed** chart needs a full
(ctx+gen) run with `execute_model_gpu`; the committed mixed charts are partial
(old-run context merged with new clean decode). The MoE overlay `moe_perf.txt`
(real measured fused-experts kernel timings) lives in `collector/layerwise/wip/`.

### What is NOT broken — do not change
- **MoE op overlay is correct.** AIC decode = backbone (`generation_layerwise`,
  `includes_moe=False`) + MoE overlay (`generation_moe` etc.) from `moe_perf` at
  `num_tokens=batch`, scaled per layer. Overlay values are real kernel timings,
  queried correctly; at batch=1 (clean backbone) `backbone + moe ~= FPM`.
- **Decode compute calibration is dense-only.** `_DECODE_COMPUTE_BATCH_CAL = 0.0066`
  in `vllm_backend.py` (gated behind `is_moe_model`), validated on dense Qwen3-32B
  (MAPE 10.6% -> 3.4%); intentionally not applied to MoE.
- Other in-place modeling fixes (do not redo): fused all-reduce for decode comm
  (`_LAYERWISE_USE_FUSED_ALLREDUCE_RMS`, decode only); generation comm
  un-suppression for single-GPU data (`_LAYERWISE_GEN_SINGLE_GPU_COMM`); high-KV
  dense decode repair (`_repair_decode_high_kv`); mixed model
  `= context_total + decode attention` (`_get_mix_step_latency`).

### Verify a re-collection
1. Group `layerwise.csv` by (model, attn_tp, ep, past_kv): `latency_ms` must be
   non-decreasing in `batch_size`, with a single `latency_source`.
2. Regenerate charts (set `--vllm-max-num-seqs` to the collected value); the gen
   AIC line should track FPM and the blue "layerwise collected" dots should rise
   smoothly (no jump/plateau). A full (ctx+gen) run enables clean mixed charts.
3. The summary command also reports per-case MAPE:
   ```bash
   uv run python collector/layerwise/diagnostics/compare_aic_layerwise_fpm_summary.py \
     --layerwise <RUN_DIR>/layerwise.csv
   ```
   (append `--shape-breakdown aggregate` to see error by phase/token/batch bucket).

## TODO

- (DONE) DeepSeek-V4-Flash `tp4_ep4` decode is fixed by the GPU-isolated
  `execute_model_gpu` timing — the old `192.02%` decode outlier was corrupt
  collection, not modeling. See the status section above.
- Improve mixed-step / context modeling. Mixed error is now dominated by
  context/prefill and FPM measurement outliers, not decode. Add FPM-outlier
  filtering (drop points where FPM << prefill-only SOL) and verify the one-GPU
  patched config, context batch shape, prefix length, and `max_num_batched_tokens`
  match the FPM deployment envelope. Keep workload-mode accuracy separate from
  pathology-filtered diagnostics.
- Reduce Qwen3.6 MoE context error. Prioritize batched context and TP/EP
  scheduler-envelope parity for the cases above `35%` context MAPE before
  broadening the grid. In particular, verify that the one-GPU patched config,
  context batch shape, prefix length, and `max_num_batched_tokens` match the
  FPM deployment envelope.
- Replace the current empirical MoE EP communication addend with a
  backend-aware model. The current AIC path treats context as one EP exchange
  and decode/op fallback as two EP exchanges; this matched a small validation
  set but is not what vLLM implements. vLLM's default EP backend is
  `allgather_reducescatter`, where MoE dispatch is an all-gather/all-gatherv
  phase and combine is a reduce-scatter/reduce-scatterv phase. A better model
  should compute `dispatch_ms + combine_ms` from the selected vLLM
  `--all2all-backend`, using NCCL `all_gather` and `reduce_scatter` payloads
  for the default backend and backend-specific perf data/fallbacks for
  DeepEP/FlashInfer/NIXL/MoRI.
- Extend dataset-shaped workloads into MoE layerwise routing validation. FPM
  already defaults to OpenAssistant/oasst1-ordered, large-scaled ISL/OSL shapes
  with deterministic random token IDs; layerwise still needs a comparable
  model-agnostic routing strategy for MoE blocks.
- Keep prefix-cache decode as the normal MoE decode path. `live_decode`
  remains only a diagnostic path because it is substantially slower and makes
  the collector harder to scale.
