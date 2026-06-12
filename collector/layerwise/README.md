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
export NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
export RUN_DIR="$AIC_REPO/.tmp/smoke-layerwise-$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR" "$HF_HOME"

docker run --rm --ipc=host --network=host \
  --gpus '"device=0"' \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$AIC_REPO:/workspace" \
  -v "$RUN_DIR:/results" \
  -v "$HF_HOME:/hf-cache" \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
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

Production decode batch sizes default up to `512`. Use
`--max-decode-batch-size auto` to follow vLLM's hardware-dependent
`max_num_seqs` default, or use `--gen-batch-sizes` to request an exact decode
grid.

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
latency source. Context and dense decode use kernel-span data. High-batch MoE
decode uses GPU-sum because wrapper span overcounts scheduler/replay gaps there.
The diagnostic `--latency-source schedule_to_update` path is still available
when comparing context kernel timing against the full scheduler/update envelope.
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

TODO:

- Extend dataset-shaped workloads into MoE layerwise routing validation.
  FPM already defaults to OpenAssistant/oasst1-ordered, large-scaled ISL/OSL
  shapes with deterministic random token IDs; layerwise still needs a
  comparable model-agnostic routing strategy for MoE blocks.
- Validate prefix-cache decode as the normal MoE decode path. `live_decode`
  remains only a diagnostic path because it is substantially slower and makes
  the collector harder to scale.
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
