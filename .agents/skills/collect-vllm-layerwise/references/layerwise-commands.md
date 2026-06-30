# vLLM Layerwise Commands

## Docker Wrapper

Use the Dynamo vLLM runtime plus host Nsight:

```bash
export AIC_REPO="${AIC_REPO:-$PWD}"
export AIC_LAYERWISE_ARTIFACTS="${AIC_LAYERWISE_ARTIFACTS:-$AIC_REPO/.tmp/layerwise-artifacts}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-$HOME/.cache/aic-vllm}"
export NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
# export HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
mkdir -p "$AIC_LAYERWISE_ARTIFACTS" "$HF_HOME" "$VLLM_CACHE_HOST/tilelang/tmp"

docker run --rm --ipc=host --network=host \
  --gpus '"device=0"' \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$AIC_REPO:/workspace" \
  -v "$AIC_LAYERWISE_ARTIFACTS:/results" \
  -v "$HF_HOME:/hf-cache" \
  -v "$VLLM_CACHE_HOST:/home/dynamo/.cache/vllm" \
  -v "$VLLM_CACHE_HOST:/root/.cache/vllm" \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
  -e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
  -e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
  -e HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN directly or export it from HF_TOKEN_FILE}" \
  -w /workspace \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; <collector command>'
```

Smoke check:

```bash
export NSYS_VERSION="${NSYS_VERSION:-2025.6.3}"
docker run --rm --network none \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; python3 -c "import vllm; print(vllm.__version__)"; nsys --version'
```

## Smoke Collection

Run this before a production sweep. It measures Qwen3-32B TP1 with two context
points and two decode points:

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
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; python3 -m collector.layerwise.vllm.collect --run-dir /results --models Qwen/Qwen3-32B --tp-sizes 1 --phases both --run-preset smoke --gpus 0 --max-workers 1 --live-step-driver'
```

Expected output:

- `layerwise.csv` has four rows: two `ctx`, two `gen`.
- `profiles/status.jsonl` contains one `nsys_parse_succeeded` event and four
  `success` events.
- `profiles/nsys/` contains one `.nsys-rep` and one exported `.sqlite`.

For the default `--latency-source span`, `meta.attribution_source=nvtx_span` is
acceptable as long as the parser reports nonzero `attributed_kernels` and each
success row has nonzero `kernel_count`. For `--latency-source gpu` or
`gpu_capped`, require CUPTI-backed kernel attribution.

## Public CLI

The default public entrypoint collects the registered production model set, currently dense Qwen3-32B and the Qwen MoE model. `--run-dir` is optional; when omitted the collector writes to a timestamped directory under `.tmp/layerwise-artifacts/runs/`.

```bash
python3 -m collector.layerwise.vllm.collect \
  --live-step-driver
```

Collect one model over TP 1/2/4/8:

```bash
python3 -m collector.layerwise.vllm.collect \
  --models Qwen/Qwen3-32B \
  --tp-sizes 1,2,4,8 \
  --live-step-driver
```

Collect MoE TP/EP points. `--ep-sizes auto` keeps EP at 1 for dense models and uses supported registry EP sizes for MoE models:

```bash
python3 -m collector.layerwise.vllm.collect \
  --models Qwen/Qwen3.6-35B-A3B \
  --tp-sizes 1,2,4,8 \
  --ep-sizes auto \
  --live-step-driver
```

Run a small smoke sweep before committing to the full grid:

```bash
python3 -m collector.layerwise.vllm.collect \
  --models Qwen/Qwen3-32B \
  --tp-sizes 2 \
  --run-preset smoke \
  --live-step-driver
```

Use optional shape overrides only when narrowing or expanding a sweep:

```bash
python3 -m collector.layerwise.vllm.collect \
  --models Qwen/Qwen3-32B \
  --tp-sizes 2 \
  --phases ctx \
  --ctx-new-tokens 8192 \
  --ctx-past-kv 0,8192 \
  --live-step-driver
```

## AIC Comparison Notes

- The collector output is per simulated rank. For TP>1, AIC must add TP allreduce separately.
- The AIC layerwise backend should use CTX lookup with `(seq_len_q, seq_len_kv_cache)`, so `8192,8192` is distinct from `8192,0`.
- For Qwen3/LLAMA-style dense models, add two custom allreduces per transformer layer for TP>1.
- For decode TP>1, do not add generic allreduce on top of deployment-parity GEN rows. Use `rms_latency_ms` plus `allreduce_rms_perf.parquet` for fused allreduce+rms attribution.
- The vLLM layerwise GEN lookup should use the KV length at the start of the decode iteration. For FPM `past_kv=1024`, compare with AIC `isl=1024, osl=2`, not an off-by-one workaround.
- Clear caches after editing data or lookup code if comparing in the same Python process.
