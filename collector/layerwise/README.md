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

The FPM smoke uses the public Python wrapper around the lower-level shell
collector:

```bash
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\n' < ~/hf.token)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0
export RUN_DIR="$PWD/.tmp/smoke-fpm-$(date -u +%Y%m%d_%H%M%S)"

python -m collector.layerwise.fpm.collect \
  --model Qwen/Qwen3-32B \
  --run-dir "$RUN_DIR" \
  --tp-sizes 1 \
  --ep-sizes 1 \
  --run-preset smoke \
  --warmup-requests 1 \
  --gpus '"device=0"'
```

Expected FPM smoke output includes `fpm_metrics.csv`,
`fpm_metrics_phase.csv`, `request_workload.csv`, `warmup_workload.csv`,
`vllm_metadata.json`, and `effective_vllm_config.json`.

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
- `fpm/datapoint_generator.py`: FPM smoke/production workload presets.
- `fpm/docker.py`: command construction for the existing shell collector.
- `fpm/artifacts.py`: FPM run-directory layout helpers.
- `common/paths.py`: shared default artifact/run-directory helpers.

The SGLang and TensorRT-LLM prototype collectors are intentionally not included
in this AIC branch.

Current validation data is stored under
`src/aiconfigurator/systems/data/b300_sxm/vllm/0.20.1/layerwise_perf.csv`.
Decode uses one-layer span data. Context currently uses 16-layer measurements
normalized per layer because one-layer context underpredicts standard-random
FPM at long sequence lengths.
