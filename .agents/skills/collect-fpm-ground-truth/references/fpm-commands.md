# FPM Ground Truth Commands

## Canonical TP=2 Sweep

Run from the `aiconfigurator` repo. Set `HF_TOKEN` directly, or set `HF_TOKEN_FILE` and export from it before running:

```bash
export AIC_REPO="${AIC_REPO:-$PWD}"
export AIC_LAYERWISE_ARTIFACTS="${AIC_LAYERWISE_ARTIFACTS:-$AIC_REPO/.tmp/layerwise-artifacts}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# export HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"
mkdir -p "$AIC_LAYERWISE_ARTIFACTS" "$HF_HOME"

HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN directly or export it from HF_TOKEN_FILE}" \
HF_HOME="$HF_HOME" \
DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0 \
POST_REQUEST_COLLECT_SECONDS=5 \
bash collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh \
  --model Qwen/Qwen3-32B \
  --max-model-len 16385 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --gpus '"device=0,1"' \
  --output "$AIC_LAYERWISE_ARTIFACTS/qwen3_32b_tp2_fpm_sweep_b300_vllm0201.csv" \
  --warmup-requests 8 \
  --warmup-concurrency 8 \
  --warmup-isl-values 1024 \
  --warmup-osl-values 32 \
  --post-warmup-seconds 1 \
  --context-isl-values 1,64,256,1024,2048,4096,8192,16384 \
  --context-repeats 6 \
  --decode-batch-sizes 1,2,4,8,16,32,64 \
  --decode-past-kv 1024 \
  --decode-osl 32 \
  --mixed-requests 128 \
  --mixed-concurrency 32 \
  --mixed-isl-values 1024,2048,4096,8192 \
  --mixed-osl-values 32 \
  --file-discovery-touch-seconds 2 \
  -- --tensor-parallel-size 2
```

Expected outputs:

- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_detail.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_phase.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_workload.csv`
- `qwen3_32b_tp2_fpm_sweep_b300_vllm0201_warmup_workload.csv`

## Summarize Rows

Use this after collection:

```bash
python3 - <<'PY'
import csv, os, statistics
from collections import Counter, defaultdict
from pathlib import Path

root = Path(os.environ["AIC_LAYERWISE_ARTIFACTS"])
p = root / "qwen3_32b_tp2_fpm_sweep_b300_vllm0201_phase.csv"
rows = []
with open(p, newline="") as f:
    for r in csv.DictReader(f):
        r = {k: (v if k == "phase" else float(v)) for k, v in r.items()}
        rows.append(r)

def trimmed(vals):
    vals = sorted(vals)
    if len(vals) >= 3:
        vals = vals[1:-1]
    return statistics.mean(vals)

print("rows", len(rows), Counter(r["phase"] for r in rows))

ctx = defaultdict(list)
for r in rows:
    if r["phase"] == "context":
        ctx[(int(r["ctx_tokens"]), int(r["ctx_kv_tokens"]))].append(r["latency_ms"])
print("context")
for key in sorted(ctx):
    print(key, len(ctx[key]), round(trimmed(ctx[key]), 3))

dec = defaultdict(list)
for r in rows:
    if r["phase"] == "decode" and 1000 <= r["mean_decode_kv_tokens"] <= 1100:
        dec[int(r["decode_requests"])].append(r["latency_ms"])
print("decode")
for key in sorted(dec):
    print(key, len(dec[key]), round(trimmed(dec[key]), 3))
PY
```

## Failure Handling

- If HTTP 503 appears after warmup, check local file-discovery expiry; keep the heartbeat enabled.
- If a mixed request returns HTTP 500 after retries, preserve outputs and inspect `*_phase.csv`; context/decode rows can still be valid.
- If outputs are absent after a failure, patch the script so it stops the collector and copies partial CSVs before exiting.
- Confirm image vLLM with `docker run --rm --network none "$DYNAMO_VLLM_IMAGE" python3 -c 'import vllm; print(vllm.__version__)'`.
