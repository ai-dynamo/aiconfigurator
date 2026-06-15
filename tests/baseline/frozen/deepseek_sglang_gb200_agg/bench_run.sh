#!/bin/bash
set -euo pipefail

concurrency_array=(1 2 8 16 32 64 128 243 256 268)

if [[ -n "${AICONFIGURATOR_BENCH_CONCURRENCY:-}" ]]; then
  concurrency_array=(${AICONFIGURATOR_BENCH_CONCURRENCY})
fi

ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-/tmp/bench_artifacts}"
BENCH_MODEL="${AICONFIGURATOR_BENCH_MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
BENCH_ENDPOINT_TYPE="${AICONFIGURATOR_BENCH_ENDPOINT_TYPE:-chat}"
BENCH_ENDPOINT_URL="${AICONFIGURATOR_BENCH_ENDPOINT_URL:-http://0.0.0.0:8000}"
BENCH_TOKENIZER="${AICONFIGURATOR_BENCH_TOKENIZER:-deepseek-ai/DeepSeek-V4-Pro}"
BENCH_ISL="${AICONFIGURATOR_BENCH_ISL:-4000}"
BENCH_ISL_STDDEV="${AICONFIGURATOR_BENCH_ISL_STDDEV:-0}"
BENCH_OSL="${AICONFIGURATOR_BENCH_OSL:-1000}"
BENCH_OSL_STDDEV="${AICONFIGURATOR_BENCH_OSL_STDDEV:-0}"
BENCH_PREFIX="${AICONFIGURATOR_BENCH_PREFIX:-0}"
BENCH_PREFIX_PROMPTS="${AICONFIGURATOR_BENCH_PREFIX_PROMPTS:-1}"
BENCH_UI="simple"
BENCH_MULTI_ROUND="${AICONFIGURATOR_BENCH_MULTI_ROUND:-20}"

prefix_args=()
if [[ "${BENCH_PREFIX}" =~ ^[0-9]+$ && "${BENCH_PREFIX}" -gt 0 \
    && "${BENCH_PREFIX_PROMPTS}" =~ ^[0-9]+$ && "${BENCH_PREFIX_PROMPTS}" -gt 0 ]]; then
  prefix_args+=(--prefix-prompt-length "${BENCH_PREFIX}" --num-prefix-prompts "${BENCH_PREFIX_PROMPTS}")
fi

for concurrency in "${concurrency_array[@]}"; do
  echo "Run concurrency: $concurrency"
  aiperf profile \
    --artifact-dir "${ARTIFACT_DIR}/concurrency_${concurrency}" \
    -m "${BENCH_MODEL}" \
    --endpoint-type "${BENCH_ENDPOINT_TYPE}" \
    -u "${BENCH_ENDPOINT_URL}" \
    --tokenizer "${BENCH_TOKENIZER}" \
    --isl "${BENCH_ISL}" --isl-stddev "${BENCH_ISL_STDDEV}" \
    --osl "${BENCH_OSL}" --osl-stddev "${BENCH_OSL_STDDEV}" \
    "${prefix_args[@]}" \
    --extra-inputs ignore_eos:true \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --concurrency ${concurrency} \
    --num-requests $(($concurrency*${BENCH_MULTI_ROUND})) \
    --warmup-request-count $(($concurrency*2)) \
    --random-seed 100 \
    --ui "${BENCH_UI}" \
    --streaming
done