# FPM Ground Truth Commands

## Smoke Collection

Run this before a larger FPM sweep. It uses one GPU, Qwen3-32B TP1, one 128-token
context shape, and one decode batch at `past_kv=1024`:

```bash
export HF_TOKEN="${HF_TOKEN:-$(tr -d '\n' < ~/hf.token)}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DYNAMO_VLLM_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0
export RUN_DIR="$PWD/.tmp/smoke-fpm-$(date -u +%Y%m%d_%H%M%S)"

python3 -m collector.layerwise.fpm.collect \
  --model Qwen/Qwen3-32B \
  --run-dir "$RUN_DIR" \
  --tp-sizes 1 \
  --ep-sizes 1 \
  --run-preset smoke \
  --warmup-requests 1 \
  --gpus '"device=0"'
```

Expected output:

- `fpm_metrics.csv`
- `fpm_metrics_detail.csv`
- `fpm_metrics_phase.csv`
- `request_workload.csv`
- `warmup_workload.csv`
- `vllm_metadata.json`
- `effective_vllm_config.json`

The smoke should produce context and decode phase rows. A first cold context row
can be much slower than the subsequent measured row; preserve it, but summarize
with medians or trimmed means for comparisons.

## Canonical TP=2 Context/Decode Sweep

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
python3 -m collector.layerwise.fpm.collect \
  --model Qwen/Qwen3-32B \
  --tp-sizes 2 \
  --warmup-requests 4
```

Expected outputs under `$RUN_DIR`:

- `fpm_metrics.csv`
- `fpm_metrics_detail.csv`
- `fpm_metrics_phase.csv`
- `request_workload.csv`
- `warmup_workload.csv`
- `vllm_metadata.json`
- `effective_vllm_config.json`

For FP8, set `--model Qwen/Qwen3-32B-FP8`. Keep vLLM defaults unless the experiment explicitly requires an override. `--run-dir` is optional; when omitted the CLI writes to a timestamped directory under `.tmp/layerwise-artifacts/runs/`.

Default measured phases are `context,decode`. Default context shapes are `128,1024,4096`; default decode batches are `1,4,16` at `past_kv=4096`, `osl=8`. Docker GPUs are inferred from `--tp-sizes`/`--ep-sizes` (`--tp-sizes 2` uses `device=0,1`). Use `--phases`, `--contexts`, `--decode-batches`, `--decode-past-kv`, and `--decode-osl` only when narrowing or expanding the default sweep. Add `mixed` to `--phases` only when you specifically need scheduler mixed-step rows.

For `openai/gpt-oss-*`, the wrapper automatically applies the vLLM synthetic-benchmark defaults used by the GPT-OSS recipe: `--kv-cache-dtype fp8` if unset, `--no-enable-prefix-caching` unless explicitly overridden, `--max-cudagraph-capture-size 2048`, and `--stream-interval 20`.

## Summarize Rows

Use this after collection:

```bash
python3 - <<'PY'
import csv, os, statistics
from collections import Counter, defaultdict
from pathlib import Path

p = Path(os.environ["RUN_DIR"]) / "fpm_metrics_phase.csv"
rows = []
with open(p, newline="") as f:
    for r in csv.DictReader(f):
        r = {k: (v if k == "phase" else float(v)) for k, v in r.items()}
        rows.append(r)

def summarize(vals):
    vals = sorted(vals)
    if len(vals) >= 3:
        return statistics.median(vals)
    return statistics.mean(vals)

print("rows", len(rows), Counter(r["phase"] for r in rows))

ctx = defaultdict(list)
for r in rows:
    if r["phase"] == "context":
        ctx[(int(r["ctx_tokens"]), int(r["ctx_kv_tokens"]))].append(r["latency_ms"])
print("context")
for key in sorted(ctx):
    print(key, len(ctx[key]), round(summarize(ctx[key]), 3))

dec = defaultdict(list)
for r in rows:
    if (
        r["phase"] == "decode"
        and r["decode_tokens"] == r["decode_requests"]
        and 1000 <= r["mean_decode_kv_tokens"] <= 1100
    ):
        dec[int(r["decode_requests"])].append(r["latency_ms"])
print("decode")
for key in sorted(dec):
    print(key, len(dec[key]), round(summarize(dec[key]), 3))
PY
```

## Failure Handling

- If HTTP 503 appears after warmup, check local file-discovery expiry; keep the heartbeat enabled.
- If a mixed request returns HTTP 500 after retries, preserve outputs and inspect `*_phase.csv`; context/decode rows can still be valid.
- If outputs are absent after a failure, patch the script so it stops the collector and copies partial CSVs before exiting.
- Confirm image vLLM with `docker run --rm --network none "$DYNAMO_VLLM_IMAGE" python3 -c 'import vllm; print(vllm.__version__)'`.
- If the container emits `.qdstrm` rather than `.nsys-rep`, import on the host with `QdstrmImporter` before exporting SQLite.
