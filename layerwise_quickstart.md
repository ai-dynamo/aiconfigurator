# Layerwise  cd /home/shadeform/aiconfigurator

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
-e HF_HOME=/hf-cache \
-e HF_HUB_CACHE=/hf-cache/hub \
-e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
-e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
-e NSYS_VERSION="$NSYS_VERSION" \
-w /workspace \
vllm/vllm-openai:v0.20.1 \
-lc 'set -euo pipefail
        export PATH="/opt/nvidia/nsight-systems/${NSYS_VERSION}/target-linux-x64:$PATH"
        export HF_TOKEN="$(tr -d "\n" < /run/secrets/hf.token)"
        python3 -m collector.layerwise.vllm.collect \
        --run-dir /results \
        --max-workers 8 \
        --live-step-driver'

That runs the full default registry: Qwen3-32B, Qwen3.6 MoE, and DeepSeek-V4-Flash.

# Useful progress checks from another shell:

wc -l "$RUN_DIR/layerwise.csv"
tail -f "$RUN_DIR/profiles/status.jsonl"


# Compare results
uv run python collector/layerwise/diagnostics/compare_aic_layerwise_fpm_summary.py \
--layerwise runs/layerwise_full_vllm0201_20260615_045248/layerwise.csv

# shadeform@brev-rt3rd2j9g:~/aiconfigurator$ uv run python collector/layerwise/diagnostics/compare_aic_layerwise_fpm_summary.py --layerwise runs/layerwise_full_vllm0201_20260615_045248/layerwise.csv
# case                      rows      all      ctx      gen    mixed status
# qwen32_tp1                  58   20.92%    8.58%    3.90%   21.92% ok
# qwen32_tp2                  55   12.46%    4.34%    7.40%   13.04% ok
# qwen32_tp4                  41   11.98%   10.64%    6.13%   12.41% ok
# qwen32_tp8                  39   22.01%   12.46%    1.09%   24.09% ok
# qwen36_tp1_ep1              75   59.06%   27.33%   12.21%   60.59% ok
# qwen36_tp2_ep2              73   28.26%   22.08%   14.69%   28.73% ok
# qwen36_tp4_ep4              75   29.97%   38.18%   12.92%   29.87% ok
# qwen36_tp8_ep8              63   26.24%   27.45%   14.12%   26.40% ok
# dsv4_tp1_ep4                 6   18.66%   14.77%    8.32%   29.68% ok
# dsv4_tp2_ep4                13   29.71%   27.35%   15.56%   32.07% ok
# dsv4_tp4_ep4                33   36.78%   30.79%  192.02%   26.33% ok
# qwen36_tp2_ep1              77   20.84%   18.39%   21.39%   20.93% ok
# qwen36_tp4_ep1              73   30.06%   46.07%   15.19%   29.58% ok
# qwen36_tp1_ep2              42   38.63%   22.50%    2.96%   40.85% ok
# qwen36_tp1_ep4              24   40.39%   22.92%    8.76%   44.59% ok
# qwen36_tp1_ep8              16   40.71%   20.15%    8.52%   48.53% ok
# qwen36_tp2_ep4              30   27.93%   24.61%    9.22%   29.04% ok
# qwen36_tp2_ep8              23   36.78%   26.85%    5.02%   40.02% ok
# qwen36_tp4_ep8              49   32.74%   17.79%   12.67%   34.18% ok

# aggregate                 rows      all      ctx      gen    mixed
# all_cases                  865   29.69%   22.09%   25.97%   30.33%
