#!/bin/bash
set -e
cd /workspace/repo/aiconfigurator
export HF_HOME=/workspace/models/hf_home
export VLLM_LOGGING_LEVEL=WARNING
MODEL=/workspace/models/hf_home/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0
PORT=8765
# launch vllm serve under nsys (capture whole process; children included)
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node \
  --force-overwrite=true -o runs/serve_nsys_trace \
  vllm serve "$MODEL" --served-model-name qwen36 \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-num-batched-tokens 2048 --max-num-seqs 128 --gpu-memory-utilization 0.9 \
  --skip-mm-profiling --limit-mm-per-prompt '{"image":0,"video":0}' \
  --no-enable-prefix-caching --trust-remote-code --port $PORT > runs/serve_nsys_serve.out 2>&1 &
SERVE_PID=$!
echo "serve pid=$SERVE_PID, waiting for /health ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then echo "READY after ${i}0s"; break; fi
  sleep 10
done
# probe
uv run --active python runs/serve_probe.py $PORT 2>&1 | tee runs/serve_probe.out || true
# stop serve so nsys finalizes
kill -INT $SERVE_PID 2>/dev/null || true
sleep 20
kill -TERM $SERVE_PID 2>/dev/null || true
wait $SERVE_PID 2>/dev/null || true
echo "=== SERVE_NSYS DONE ==="
