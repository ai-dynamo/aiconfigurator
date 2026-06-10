# vLLM Layerwise Commands

## Docker Wrapper

Use the Dynamo vLLM runtime plus host Nsight:

```bash
export AIC_REPO="${AIC_REPO:-$PWD}"
export AIC_LAYERWISE_ARTIFACTS="${AIC_LAYERWISE_ARTIFACTS:-$AIC_REPO/.tmp/layerwise-artifacts}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export NSYS_VERSION="${NSYS_VERSION:-2025.3.2}"
# export HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
mkdir -p "$AIC_LAYERWISE_ARTIFACTS" "$HF_HOME"

docker run --rm --ipc=host --network=host \
  --gpus '"device=0"' \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  -v "$AIC_REPO:/workspace" \
  -v "$AIC_LAYERWISE_ARTIFACTS:/results" \
  -v "$HF_HOME:/hf-cache" \
  -e HF_HOME=/hf-cache \
  -e HF_HUB_CACHE=/hf-cache/hub \
  -e HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN directly or export it from HF_TOKEN_FILE}" \
  -w /workspace \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; <collector command>'
```

Smoke check:

```bash
export NSYS_VERSION="${NSYS_VERSION:-2025.3.2}"
docker run --rm --network none \
  -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
  bash -lc 'export PATH=/opt/nvidia/nsight-systems/'"$NSYS_VERSION"'/target-linux-x64:$PATH; python3 -c "import vllm; print(vllm.__version__)"; nsys --version'
```

## TP=2 Decode

Collect batch sizes `1,2,4,8,16,32,64` at KV 1024. Use deployment parity/default compile for FPM comparisons:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_decode_b1_64_past1024_span.csv \
  --work-dir /results/profiles/vllm_decode_qwen32b_tp2_b1_64_past1024_span \
  --config-cache-dir /results/profiles/config_cache \
  --tp-sizes 2 \
  --phases gen \
  --gen-batch-sizes 1,2,4,8,16,32,64 \
  --gen-past-kv 1024 \
  --target-layer-count 1 \
  --rank-reduce max \
  --latency-source span \
  --gpus 0 \
  --max-workers 1
```

## TP=2 Context

Use 16 layers for the production AIC context table. Collect both axes and let the collector skip points beyond model max length:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_context_b300_gpu_capped_w2m6_trimmed_16layers.csv \
  --work-dir /results/profiles/vllm_context_qwen32b_tp2_gpu_capped_w2m6_trimmed_16layers \
  --config-cache-dir /results/profiles/config_cache \
  --tp-sizes 2 \
  --phases ctx \
  --ctx-new-tokens 1,64,256,1024,2048,4096,8192 \
  --ctx-past-kv 0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536 \
  --target-layer-count 16 \
  --rank-reduce max \
  --latency-source gpu_capped \
  --ctx-warmup-runs 2 \
  --ctx-measured-runs 6 \
  --ctx-repeat-aggregation trimmed_mean \
  --gpus 0 \
  --max-workers 1
```

If running a small sanity check instead of the full grid, include at least `8192,0` and `8192,8192`:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_context_8192past8192_gpu_capped_w2m6_trimmed_16layers.csv \
  --work-dir /results/profiles/vllm_context_qwen32b_tp2_8192past8192_gpu_capped_w2m6_trimmed_16layers \
  --config-cache-dir /results/profiles/config_cache \
  --tp-sizes 2 \
  --phases ctx \
  --ctx-new-tokens 8192 \
  --ctx-past-kv 8192 \
  --target-layer-count 16 \
  --rank-reduce max \
  --latency-source gpu_capped \
  --ctx-warmup-runs 2 \
  --ctx-measured-runs 6 \
  --ctx-repeat-aggregation trimmed_mean \
  --gpus 0 \
  --max-workers 1
```

## AIC Comparison Notes

- The collector output is per simulated rank. For TP>1, AIC must add TP allreduce separately.
- The AIC layerwise backend should use CTX lookup with `(seq_len_q, seq_len_kv_cache)`, so `8192,8192` is distinct from `8192,0`.
- For Qwen3/LLAMA-style dense models, add two custom allreduces per transformer layer for TP>1.
- For decode TP>1, do not add generic allreduce on top of deployment-parity GEN rows. Use `rms_latency_ms` plus `allreduce_rms_perf.parquet` for fused allreduce+rms attribution.
- The vLLM layerwise GEN lookup should use the KV length at the start of the decode iteration. For FPM `past_kv=1024`, compare with AIC `isl=1024, osl=2`, not an off-by-one workaround.
- Clear caches after editing data or lookup code if comparing in the same Python process.
