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

Collect batch sizes `1,2,4,8,16,32,64` at KV 1024:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_decode_b1_64_past1024_span.csv \
  --work-dir /results/profiles/vllm_decode_qwen32b_tp2_b1_64_past1024_span \
  --config-cache-dir /results/profiles/config_cache \
  --system "NVIDIA B300 SXM6 AC" \
  --framework-version 0.20.1 \
  --tp-sizes 2 \
  --phases gen \
  --gen-batch-sizes 1,2,4,8,16,32,64 \
  --gen-past-kv 1024 \
  --target-layer-count 1 \
  --rank-reduce max \
  --latency-source span \
  --min-max-num-batched-tokens 64 \
  --gpus 0 \
  --max-workers 1
```

## TP=2 Context

Use 16 layers for the production AIC context table:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_context_b300_gpu_capped_w2m6_trimmed_16layers.csv \
  --work-dir /results/profiles/vllm_context_qwen32b_tp2_gpu_capped_w2m6_trimmed_16layers \
  --config-cache-dir /results/profiles/config_cache \
  --system "NVIDIA B300 SXM6 AC" \
  --framework-version 0.20.1 \
  --tp-sizes 2 \
  --phases ctx \
  --ctx-new-tokens 1,64,256,1024,2048,4096,8192 \
  --ctx-past-kv 0 \
  --target-layer-count 16 \
  --rank-reduce max \
  --latency-source gpu_capped \
  --ctx-warmup-runs 2 \
  --ctx-measured-runs 6 \
  --ctx-repeat-aggregation trimmed_mean \
  --min-max-num-batched-tokens 8192 \
  --gpus 0 \
  --max-workers 1
```

Collect chunked-prefill 16k second chunk separately:

```bash
python3 collector/layerwise/vllm/collect_layerwise.py \
  --model Qwen/Qwen3-32B \
  --output /results/qwen3_32b_tp2_vllm_context_8192past8192_gpu_capped_w2m6_trimmed_16layers.csv \
  --work-dir /results/profiles/vllm_context_qwen32b_tp2_8192past8192_gpu_capped_w2m6_trimmed_16layers \
  --config-cache-dir /results/profiles/config_cache \
  --system "NVIDIA B300 SXM6 AC" \
  --framework-version 0.20.1 \
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
  --min-max-num-batched-tokens 8192 \
  --gpus 0 \
  --max-workers 1
```

## AIC Comparison Notes

- The collector output is per simulated rank. For TP>1, AIC must add TP allreduce separately.
- The AIC layerwise backend should use CTX lookup with `(seq_len_q, seq_len_kv_cache)`, so `8192,8192` is distinct from `8192,0`.
- For Qwen3/LLAMA-style dense models, add two custom allreduces per transformer layer for TP>1.
- Clear caches after editing data or lookup code if comparing in the same Python process.
