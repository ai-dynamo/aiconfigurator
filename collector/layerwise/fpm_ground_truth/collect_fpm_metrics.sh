#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Launch a local Dynamo vLLM container stack, drive sample requests, and collect
# raw vLLM ForwardPassMetrics as:
#
#   num_context_tokens,num_decode_tokens,latency_ms
#
# A second detail CSV is also written with the full scheduled/queued FPM
# fields needed to interpret mixed prefill/decode batches.
#
# Requirements:
#   - Docker with NVIDIA Container Toolkit.
#   - A Dynamo vLLM runtime image built from a branch that contains
#     dynamo.vllm.instrumented_scheduler and vLLM 0.20.1.
#
# Typical use:
#   DYNAMO_VLLM_IMAGE=dynamo:latest-vllm bash collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh
#
# If the image is not local, build it first, for example:
#   python3 container/render.py --framework vllm --output-short-filename
#   docker build -f container/rendered.Dockerfile -t dynamo:latest-vllm .

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMON_DIR="$(cd "${SCRIPT_DIR}/../common" && pwd)"

IMAGE="${DYNAMO_VLLM_IMAGE:-dynamo:latest-vllm}"
EXPECTED_VLLM_VERSION="${EXPECTED_VLLM_VERSION:-0.20.1}"
ALLOW_VERSION_MISMATCH="${ALLOW_VERSION_MISMATCH:-0}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
FPM_PORT="${DYN_FORWARDPASS_METRIC_PORT:-20380}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
GPUS="${GPUS:-}"
TP_SIZE="${TP_SIZE:-}"
EP_SIZE="${EP_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-}"
DATA_PARALLEL_SIZE_EXPLICIT=0
if [[ -n "${DATA_PARALLEL_SIZE}" ]]; then
    DATA_PARALLEL_SIZE_EXPLICIT=1
fi
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
if [[ -v DISABLE_PREFIX_CACHING ]]; then
    PREFIX_CACHING_EXPLICIT=1
else
    PREFIX_CACHING_EXPLICIT=0
fi
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-0}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
FILE_DISCOVERY_TOUCH_SECONDS="${FILE_DISCOVERY_TOUCH_SECONDS:-2}"

REQUESTS="${REQUESTS:-16}"
CONCURRENCY="${CONCURRENCY:-32}"
WORKLOAD_PLAN="${WORKLOAD_PLAN:-sweep}"
MEASURED_PHASES="${MEASURED_PHASES:-context,decode}"
REAL_WORKLOAD="${REAL_WORKLOAD:-1}"
INCLUDE_SWEEP="${INCLUDE_SWEEP:-0}"
REAL_WORKLOAD_REQUESTS="${REAL_WORKLOAD_REQUESTS:-128}"
REAL_WORKLOAD_CONCURRENCY="${REAL_WORKLOAD_CONCURRENCY:-32}"
REAL_WORKLOAD_DATASET="${REAL_WORKLOAD_DATASET:-OpenAssistant/oasst1}"
REAL_WORKLOAD_MAX_ROWS="${REAL_WORKLOAD_MAX_ROWS:-5000}"
REAL_WORKLOAD_SHAPE_SOURCE="${REAL_WORKLOAD_SHAPE_SOURCE:-scaled_dataset}"
REAL_WORKLOAD_ISL_MIN="${REAL_WORKLOAD_ISL_MIN:-100}"
REAL_WORKLOAD_ISL_MAX="${REAL_WORKLOAD_ISL_MAX:-16384}"
REAL_WORKLOAD_ISL_MEAN="${REAL_WORKLOAD_ISL_MEAN:-4096}"
REAL_WORKLOAD_OSL_MIN="${REAL_WORKLOAD_OSL_MIN:-100}"
REAL_WORKLOAD_OSL_MAX="${REAL_WORKLOAD_OSL_MAX:-4096}"
REAL_WORKLOAD_OSL_MEAN="${REAL_WORKLOAD_OSL_MEAN:-1024}"
CONTEXT_ISL_VALUES="${CONTEXT_ISL_VALUES:-128,1024,4096}"
CONTEXT_OSL="${CONTEXT_OSL:-1}"
CONTEXT_REPEATS="${CONTEXT_REPEATS:-6}"
CONTEXT_CONCURRENCY="${CONTEXT_CONCURRENCY:-1}"
DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1,4,16}"
DECODE_PAST_KV="${DECODE_PAST_KV:-4096}"
DECODE_OSL="${DECODE_OSL:-8}"
DECODE_REPEATS="${DECODE_REPEATS:-6}"
DECODE_PREFIX_WARMUP="${DECODE_PREFIX_WARMUP:-1}"
MIX_REQUESTS="${MIX_REQUESTS:-64}"
MIX_CONCURRENCY="${MIX_CONCURRENCY:-32}"
MIX_ISL_VALUES="${MIX_ISL_VALUES:-1024,2048,4096}"
MIX_OSL_VALUES="${MIX_OSL_VALUES:-32}"
MIX_REPEATS="${MIX_REPEATS:-1}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-0}"
WARMUP_CONCURRENCY="${WARMUP_CONCURRENCY:-}"
WARMUP_ISL_VALUES="${WARMUP_ISL_VALUES:-}"
WARMUP_OSL_VALUES="${WARMUP_OSL_VALUES:-}"
POST_WARMUP_SECONDS="${POST_WARMUP_SECONDS:-1}"
MAX_TOKENS="${MAX_TOKENS:-64}"
PROMPT_TOKEN_SEED="${PROMPT_TOKEN_SEED:-}"
PROMPT_TOKEN_MODE="${PROMPT_TOKEN_MODE:-safe_ascii}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-900}"
REQUEST_RETRIES="${REQUEST_RETRIES:-3}"
REQUEST_RETRY_BACKOFF_SECONDS="${REQUEST_RETRY_BACKOFF_SECONDS:-2}"
REQUEST_ALLOW_FAILURES="${REQUEST_ALLOW_FAILURES:-0}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-900}"
POST_REQUEST_COLLECT_SECONDS="${POST_REQUEST_COLLECT_SECONDS:-3}"
VARY_ISL_OSL="${VARY_ISL_OSL:-1}"
REQUEST_ENDPOINT="${REQUEST_ENDPOINT:-completions}"
ISL_MIN="${ISL_MIN:-1}"
ISL_MAX="${ISL_MAX:-4096}"
OSL_MIN="${OSL_MIN:-1}"
OSL_MAX="${OSL_MAX:-1024}"
ISL_VALUES="${ISL_VALUES:-}"
OSL_VALUES="${OSL_VALUES:-}"
IGNORE_EOS="${IGNORE_EOS:-1}"
MEASUREMENT_MODE="${MEASUREMENT_MODE:-deployment-parity}"
NSYS_PROFILE_WORKER="${NSYS_PROFILE_WORKER:-0}"
NSYS_BIN="${NSYS_BIN:-nsys}"
NSYS_HOST_DIR="${NSYS_HOST_DIR:-}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-node}"
NSYS_PROFILE_TRAFFIC_ONLY="${NSYS_PROFILE_TRAFFIC_ONLY:-1}"
NSYS_SESSION_NAME="${NSYS_SESSION_NAME:-fpm_worker}"

RUN_ID="${RUN_ID:-dynamo-fpm-$(date +%Y%m%d-%H%M%S)-$$}"
NAME_PREFIX="${NAME_PREFIX:-${RUN_ID}}"
RUN_DIR="${RUN_DIR:-/tmp/${RUN_ID}}"
OUTPUT_CSV="${OUTPUT_CSV:-}"
DETAIL_OUTPUT_CSV="${DETAIL_OUTPUT_CSV:-}"
PHASE_OUTPUT_CSV="${PHASE_OUTPUT_CSV:-}"
WORKLOAD_OUTPUT_CSV="${WORKLOAD_OUTPUT_CSV:-}"
WARMUP_WORKLOAD_OUTPUT_CSV="${WARMUP_WORKLOAD_OUTPUT_CSV:-}"
METADATA_OUTPUT_JSON="${METADATA_OUTPUT_JSON:-}"
EFFECTIVE_CONFIG_OUTPUT_JSON="${EFFECTIVE_CONFIG_OUTPUT_JSON:-}"
KEEP_RUNNING="${KEEP_RUNNING:-0}"
SKIP_REQUESTS="${SKIP_REQUESTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

HF_HOME_HOST="${HF_HOME:-}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-${VLLM_CACHE:-}}"
TILELANG_CACHE_DIR_CONTAINER="${TILELANG_CACHE_DIR_CONTAINER:-/home/dynamo/.cache/vllm/tilelang}"
TILELANG_TMP_DIR_CONTAINER="${TILELANG_TMP_DIR_CONTAINER:-${TILELANG_CACHE_DIR_CONTAINER}/tmp}"

WORKER_EXTRA_ARGS=()
CLEANUP_ENABLED=0
DISCOVERY_TOUCH_PID=""

usage() {
    cat <<EOF
Usage: $0 [options] [-- extra vLLM worker args...]

Options:
  --image IMAGE                 Dynamo vLLM runtime image (default: ${IMAGE})
  --model MODEL                 Model to serve (default: ${MODEL})
  --expected-vllm-version VER   Required vLLM version (default: ${EXPECTED_VLLM_VERSION})
  --allow-version-mismatch      Warn instead of failing if image vLLM version differs
  --http-port PORT              Dynamo frontend port (default: ${HTTP_PORT})
  --system-port PORT            Worker health/metrics port (default: ${SYSTEM_PORT})
  --fpm-port PORT               Raw FPM ZMQ PUB port (default: ${FPM_PORT})
  --max-model-len N             vLLM --max-model-len override (default: vLLM model default)
  --max-num-seqs N              vLLM --max-num-seqs override (default: vLLM scheduler default)
  --max-num-batched-tokens N    vLLM --max-num-batched-tokens override (default: vLLM scheduler default)
  --gpu-memory-utilization X    vLLM --gpu-memory-utilization (default: ${GPU_MEMORY_UTILIZATION})
  --gpus SPEC                   Docker --gpus value for worker (default: inferred from TP/EP, else device=0)
  --tp-size N                   Tensor parallel size for this deployment
  --tp-sizes N                  Alias for --tp-size. Comma lists are not supported by one invocation.
  --data-parallel-size N        vLLM data parallel size for this deployment
  --ep-size N                   vLLM expert parallel size. Values >1 enable vLLM expert parallel.
                                vLLM computes EP as TP * DP, so DP is inferred as EP/TP when omitted.
  --ep-sizes N                  Alias for --ep-size. Comma lists are not supported by one invocation.
  --enable-expert-parallel      Pass vLLM --enable-expert-parallel
  --enforce-eager               Force vLLM eager mode instead of standard compile/graph behavior
  --kv-cache-dtype DTYPE        vLLM --kv-cache-dtype. GPT-OSS defaults to fp8 on Blackwell if unset.
  --disable-prefix-caching      Pass --no-enable-prefix-caching instead of standard vLLM behavior
  --file-discovery-touch-seconds N
                                Host-side mtime refresh interval for local file discovery (default: ${FILE_DISCOVERY_TOUCH_SECONDS})
  --phases CSV                  Phases to send: context,decode,mixed (default: ${MEASURED_PHASES})
  --real-workload               Send dataset-shaped mixed request traffic (default)
  --no-real-workload            Use the static context/decode/mixed sweep instead
  --include-sweep               With --real-workload, also send the static sweep first
  --real-workload-requests N    Dataset-shaped request count (default: ${REAL_WORKLOAD_REQUESTS})
  --real-workload-concurrency N Dataset-shaped request concurrency (default: ${REAL_WORKLOAD_CONCURRENCY})
  --real-workload-dataset NAME  HF dataset for shape sampling (default: ${REAL_WORKLOAD_DATASET})
  --real-workload-max-rows N    Max dataset rows to scan for shape sampling (default: ${REAL_WORKLOAD_MAX_ROWS})
  --real-workload-shape-source scaled_dataset|synthetic
                                Shape source mode (default: ${REAL_WORKLOAD_SHAPE_SOURCE})
  --real-workload-isl-min N     Real workload min ISL (default: ${REAL_WORKLOAD_ISL_MIN})
  --real-workload-isl-max N     Real workload max ISL (default: ${REAL_WORKLOAD_ISL_MAX})
  --real-workload-isl-mean N    Real workload approximate mean ISL (default: ${REAL_WORKLOAD_ISL_MEAN})
  --real-workload-osl-min N     Real workload min OSL (default: ${REAL_WORKLOAD_OSL_MIN})
  --real-workload-osl-max N     Real workload max OSL (default: ${REAL_WORKLOAD_OSL_MAX})
  --real-workload-osl-mean N    Real workload approximate mean OSL (default: ${REAL_WORKLOAD_OSL_MEAN})
  --contexts CSV                Context target ISLs (default: ${CONTEXT_ISL_VALUES})
  --decode-batches CSV          Decode batch sizes/concurrency values (default: ${DECODE_BATCH_SIZES})
  --workload-plan sweep|legacy  Advanced: measured request plan (default: ${WORKLOAD_PLAN})
  --measured-phases CSV         Advanced alias for --phases
  --context-isl-values CSV      Context sweep target ISLs (default: ${CONTEXT_ISL_VALUES})
  --context-osl N               Context sweep max_tokens (default: ${CONTEXT_OSL})
  --context-repeats N           Context sweep repetitions (default: ${CONTEXT_REPEATS})
  --context-concurrency N       Context sweep concurrency (default: ${CONTEXT_CONCURRENCY})
  --decode-batch-sizes CSV      Decode sweep batch sizes/concurrency values (default: ${DECODE_BATCH_SIZES})
  --decode-past-kv N            Decode sweep prompt length / target KV (default: ${DECODE_PAST_KV})
  --decode-osl N                Decode sweep max_tokens (default: ${DECODE_OSL})
  --decode-repeats N            Decode sweep repetitions per batch size (default: ${DECODE_REPEATS})
  --disable-decode-prefix-warmup
                                Do not prefill decode prompts before decode-only FPM collection
  --mixed-requests N            Mixed workload request count (default: ${MIX_REQUESTS})
  --mixed-concurrency N         Mixed workload concurrency (default: ${MIX_CONCURRENCY})
  --mixed-isl-values CSV        Mixed workload target ISLs (default: ${MIX_ISL_VALUES})
  --mixed-osl-values CSV        Mixed workload target OSLs (default: ${MIX_OSL_VALUES})
  --mixed-repeats N             Mixed workload repetitions (default: ${MIX_REPEATS})
  --requests N                  Number of sample requests (default: ${REQUESTS})
  --concurrency N               Sample request concurrency (default: ${CONCURRENCY})
  --warmup-requests N           Requests to send before FPM collection starts (default: ${WARMUP_REQUESTS})
  --warmup-concurrency N        Warmup request concurrency (default: CONCURRENCY)
  --warmup-isl-values CSV       Explicit warmup ISL list (default: generated from measured range)
  --warmup-osl-values CSV       Explicit warmup OSL list (default: generated from measured range)
  --post-warmup-seconds N       Delay after warmup before starting collector (default: ${POST_WARMUP_SECONDS})
  --vary-isl-osl                Use variable ISL/OSL synthetic requests (default)
  --fixed-workload              Use one fixed prompt and --max-tokens for all requests
  --endpoint completions        Request endpoint for generated random-token workload (default: ${REQUEST_ENDPOINT})
  --isl-min N                   Minimum target input tokens (default: ${ISL_MIN})
  --isl-max N                   Maximum target input tokens (default: ${ISL_MAX})
  --osl-min N                   Minimum max_tokens/output budget (default: ${OSL_MIN})
  --osl-max N                   Maximum max_tokens/output budget (default: ${OSL_MAX})
  --isl-values CSV              Explicit target ISL list, e.g. 1,64,256,1024,4096
  --osl-values CSV              Explicit target OSL list, e.g. 1,16,64,256,1024
  --disable-ignore-eos          Do not request ignore_eos=true; OSL becomes a cap
  --measurement-mode MODE       Metadata tag; FPM should normally use deployment-parity (default: ${MEASUREMENT_MODE})
  --nsys-profile-worker         Wrap the vLLM worker in nsys profile and write reports under RUN_DIR/nsys
  --nsys-bin PATH               nsys binary path inside the worker container (default: ${NSYS_BIN})
  --nsys-host-dir DIR           Optional host directory to mount read-only at the same path for --nsys-bin
  --nsys-trace CSV              nsys --trace value when profiling worker (default: ${NSYS_TRACE})
  --nsys-cuda-graph-trace MODE  nsys --cuda-graph-trace value when profiling worker (default: ${NSYS_CUDA_GRAPH_TRACE})
  --nsys-full-worker            Profile from worker start instead of only measured traffic
  --max-tokens N                Fixed-workload max_tokens (default: ${MAX_TOKENS})
  --prompt-token-seed N         Seed for reproducible random prompt token IDs (default: random)
  --prompt-token-mode MODE      random_vocab_excluding_special or safe_ascii (default: ${PROMPT_TOKEN_MODE})
  --request-retries N           Retries per request for transient HTTP errors (default: ${REQUEST_RETRIES})
  --request-retry-backoff N     Base seconds between request retries (default: ${REQUEST_RETRY_BACKOFF_SECONDS})
  --request-allow-failures N    Continue if at most N requests fail after retries (default: ${REQUEST_ALLOW_FAILURES})
  --output PATH                 Advanced: compact CSV output path (default: RUN_DIR/fpm_metrics.csv)
  --detail-output PATH          Advanced: full FPM detail CSV path (default: RUN_DIR/fpm_metrics_detail.csv)
  --phase-output PATH           Advanced: classified step CSV path (default: RUN_DIR/fpm_metrics_phase.csv)
  --workload-output PATH        Advanced: request workload CSV path (default: RUN_DIR/request_workload.csv)
  --warmup-workload-output PATH Advanced: warmup workload CSV path (default: RUN_DIR/warmup_workload.csv)
  --metadata-output PATH        Advanced: requested/effective vLLM metadata JSON (default: RUN_DIR/vllm_metadata.json)
  --effective-config-output PATH
                                Advanced: effective vLLM config JSON (default: RUN_DIR/effective_vllm_config.json)
  --run-dir DIR                 Shared host run dir (default: ${RUN_DIR})
  --name-prefix NAME            Docker container name prefix (default: ${NAME_PREFIX})
  --skip-requests               Start stack and collector, but do not send sample requests
  --keep-running                Leave containers running after collection
  --dry-run                     Print Docker commands without running them
  -h, --help                    Show this help

Environment aliases:
  DYNAMO_VLLM_IMAGE, MODEL, REQUESTS, CONCURRENCY, MAX_TOKENS, GPUS,
  WARMUP_REQUESTS, WARMUP_CONCURRENCY, WARMUP_ISL_VALUES,
  WARMUP_OSL_VALUES, POST_WARMUP_SECONDS,
  TP_SIZE, DATA_PARALLEL_SIZE, EP_SIZE, ENABLE_EXPERT_PARALLEL, WORKLOAD_PLAN, MEASURED_PHASES,
  REAL_WORKLOAD, INCLUDE_SWEEP, REAL_WORKLOAD_REQUESTS, REAL_WORKLOAD_CONCURRENCY,
  REAL_WORKLOAD_DATASET,
  CONTEXT_ISL_VALUES, CONTEXT_REPEATS, DECODE_BATCH_SIZES, DECODE_PAST_KV, DECODE_OSL, MIX_ISL_VALUES,
  MIX_OSL_VALUES, REQUEST_RETRIES, REQUEST_RETRY_BACKOFF_SECONDS,
  REQUEST_ALLOW_FAILURES,
  VARY_ISL_OSL, REQUEST_ENDPOINT, ISL_MIN, ISL_MAX, OSL_MIN, OSL_MAX,
  PROMPT_TOKEN_SEED, PROMPT_TOKEN_MODE, ISL_VALUES, OSL_VALUES, DYN_HTTP_PORT, DYN_SYSTEM_PORT,
  DYN_FORWARDPASS_METRIC_PORT, HF_HOME, VLLM_CACHE_HOST.

Output:
  CSV columns are exactly:
    num_context_tokens,num_decode_tokens,latency_ms
  The detail CSV is line-aligned with the compact CSV and includes the full
  scheduled/queued token counts from ForwardPassMetrics. In particular,
  num_context_tokens is scheduled sum_prefill_tokens: freshly computed
  prefill tokens for that scheduler iteration, not the full prompt/context.
  The phase CSV classifies each detail row as context, decode, mixed, or idle
  and preserves scheduled ctx/decode token counts for AIC validation.
  The request workload CSV records per-request target_isl, target_osl, and
  prompt_tokens so the aggregate FPM rows can be interpreted against the mix.
  Generated requests use random prompt token IDs through the completions API;
  pass --prompt-token-seed for reproducible token IDs.

Notes:
  The script uses file discovery across containers by mounting RUN_DIR at /work
  and setting DYN_FILE_KV=/work/discovery. It subscribes directly to the raw
  InstrumentedScheduler ZMQ publisher on tcp://127.0.0.1:FPM_PORT.
EOF
}

log() {
    printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

csv_count() {
    local raw="$1"
    local count=0
    local part
    IFS=',' read -ra parts <<< "${raw}"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        if [[ -n "${part}" ]]; then
            count=$((count + 1))
        fi
    done
    echo "${count}"
}

csv_max() {
    local raw="$1"
    local max_value=0
    local part
    IFS=',' read -ra parts <<< "${raw}"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        if [[ -z "${part}" ]]; then
            continue
        fi
        if ! [[ "${part}" =~ ^[0-9]+$ ]]; then
            die "invalid integer list '${raw}'"
        fi
        if (( part > max_value )); then
            max_value="${part}"
        fi
    done
    echo "${max_value}"
}

single_parallel_size() {
    local flag="$1"
    local raw="$2"
    raw="${raw//[[:space:]]/}"
    if [[ -z "${raw}" ]]; then
        die "${flag} must not be empty"
    fi
    if [[ "${raw}" == *,* ]]; then
        die "${flag} runs one deployment at a time; use one value, not '${raw}'"
    fi
    if ! [[ "${raw}" =~ ^[0-9]+$ ]] || (( raw < 1 )); then
        die "${flag} must be a positive integer, got '${raw}'"
    fi
    echo "${raw}"
}

infer_docker_gpus() {
    local count="$1"
    local idx
    local devices=()
    if (( count < 1 )); then
        die "cannot infer GPUs for non-positive parallel size: ${count}"
    fi
    for ((idx = 0; idx < count; idx++)); do
        devices+=("${idx}")
    done
    local joined
    joined="$(IFS=,; echo "${devices[*]}")"
    echo "\"device=${joined}\""
}

is_gpt_oss_model() {
    local model_lower="${MODEL,,}"
    [[ "${model_lower}" == *"gpt-oss"* || "${model_lower}" == *"gpt_oss"* ]]
}

worker_extra_has_flag() {
    local flag="$1"
    local extra
    for extra in "${WORKER_EXTRA_ARGS[@]}"; do
        if [[ "${extra}" == "${flag}" || "${extra}" == "${flag}="* ]]; then
            return 0
        fi
    done
    return 1
}

phase_enabled() {
    local phase="$1"
    [[ ",${MEASURED_PHASES}," == *",${phase},"* ]]
}

detect_system_hint() {
    if [[ -n "${AIC_SYSTEM:-}" ]]; then
        printf '%s\n' "${AIC_SYSTEM}"
        return
    fi
    local gpu_name=""
    local compute_cap=""
    gpu_name="$(nvidia-smi -i 0 --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    compute_cap="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)"
    gpu_name="${gpu_name//[$'\t\r\n']}"
    compute_cap="${compute_cap//[[:space:]]/}"
    if [[ -n "${gpu_name}" || -n "${compute_cap}" ]]; then
        printf '%s compute-cap=%s\n' "${gpu_name}" "${compute_cap}"
    fi
}

apply_vllm_runtime_defaults() {
    local disable_prefix_requested="${DISABLE_PREFIX_CACHING}"
    if is_gpt_oss_model && [[ "${PREFIX_CACHING_EXPLICIT}" != "1" ]] && ! worker_extra_has_flag "--enable-prefix-caching"; then
        disable_prefix_requested=1
    fi

    local helper_args=(
        runtime-defaults
        --model "${MODEL}"
        --system "$(detect_system_hint)"
    )
    if [[ -n "${KV_CACHE_DTYPE}" ]]; then
        helper_args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi
    if [[ "${disable_prefix_requested}" == "1" ]]; then
        helper_args+=(--disable-prefix-caching)
    fi
    local extra
    for extra in "${WORKER_EXTRA_ARGS[@]}"; do
        helper_args+=("--extra-arg=${extra}")
    done

    local defaults=()
    local defaults_output
    if ! defaults_output="$(python3 "${COMMON_DIR}/vllm_deployment.py" "${helper_args[@]}" --format lines)"; then
        die "failed to resolve shared vLLM runtime defaults"
    fi
    mapfile -t defaults <<< "${defaults_output}"
    WORKER_EXTRA_ARGS=()
    local key value
    for line in "${defaults[@]}"; do
        IFS=$'\t' read -r key value <<< "${line}"
        case "${key}" in
            KV_CACHE_DTYPE) KV_CACHE_DTYPE="${value}" ;;
            DISABLE_PREFIX_CACHING) DISABLE_PREFIX_CACHING="${value}" ;;
            EXTRA_ARG) WORKER_EXTRA_ARGS+=("${value}") ;;
        esac
    done
}

cleanup() {
    local rc=$?
    if [[ -n "${DISCOVERY_TOUCH_PID}" ]]; then
        kill "${DISCOVERY_TOUCH_PID}" >/dev/null 2>&1 || true
        wait "${DISCOVERY_TOUCH_PID}" >/dev/null 2>&1 || true
    fi
    if [[ "${CLEANUP_ENABLED}" != "1" ]]; then
        exit "${rc}"
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "Dry run complete; no containers were started."
        exit "${rc}"
    fi
    if [[ "${KEEP_RUNNING}" == "1" || "${DRY_RUN}" == "1" ]]; then
        log "Leaving containers running:"
        log "  ${NAME_PREFIX}-frontend"
        log "  ${NAME_PREFIX}-worker"
        log "  ${NAME_PREFIX}-collector"
        log "Stop them with: docker rm -f ${NAME_PREFIX}-frontend ${NAME_PREFIX}-worker ${NAME_PREFIX}-collector"
        exit "${rc}"
    fi

    for name in "${NAME_PREFIX}-collector" "${NAME_PREFIX}-worker" "${NAME_PREFIX}-frontend"; do
        if container_exists "${name}"; then
            docker rm -f "${name}" >/dev/null 2>&1 || true
        fi
    done
    exit "${rc}"
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --image) IMAGE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --expected-vllm-version) EXPECTED_VLLM_VERSION="$2"; shift 2 ;;
        --allow-version-mismatch) ALLOW_VERSION_MISMATCH=1; shift ;;
        --http-port) HTTP_PORT="$2"; shift 2 ;;
        --system-port) SYSTEM_PORT="$2"; shift 2 ;;
        --fpm-port) FPM_PORT="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
        --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --tp-size) TP_SIZE="$(single_parallel_size "$1" "$2")"; shift 2 ;;
        --tp-sizes) TP_SIZE="$(single_parallel_size "$1" "$2")"; shift 2 ;;
        --data-parallel-size) DATA_PARALLEL_SIZE="$(single_parallel_size "$1" "$2")"; DATA_PARALLEL_SIZE_EXPLICIT=1; shift 2 ;;
        --ep-size) EP_SIZE="$(single_parallel_size "$1" "$2")"; shift 2 ;;
        --ep-sizes) EP_SIZE="$(single_parallel_size "$1" "$2")"; shift 2 ;;
        --enable-expert-parallel) ENABLE_EXPERT_PARALLEL=1; shift ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --no-enforce-eager) ENFORCE_EAGER=0; shift ;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --disable-prefix-caching) DISABLE_PREFIX_CACHING=1; PREFIX_CACHING_EXPLICIT=1; shift ;;
        --enable-prefix-caching) DISABLE_PREFIX_CACHING=0; PREFIX_CACHING_EXPLICIT=1; shift ;;
        --file-discovery-touch-seconds) FILE_DISCOVERY_TOUCH_SECONDS="$2"; shift 2 ;;
        --workload-plan) WORKLOAD_PLAN="$2"; shift 2 ;;
        --phases) MEASURED_PHASES="$2"; shift 2 ;;
        --measured-phases) MEASURED_PHASES="$2"; shift 2 ;;
        --real-workload) REAL_WORKLOAD=1; shift ;;
        --no-real-workload) REAL_WORKLOAD=0; shift ;;
        --include-sweep) INCLUDE_SWEEP=1; shift ;;
        --real-workload-requests) REAL_WORKLOAD_REQUESTS="$2"; shift 2 ;;
        --real-workload-concurrency) REAL_WORKLOAD_CONCURRENCY="$2"; shift 2 ;;
        --real-workload-dataset) REAL_WORKLOAD_DATASET="$2"; shift 2 ;;
        --real-workload-max-rows) REAL_WORKLOAD_MAX_ROWS="$2"; shift 2 ;;
        --real-workload-shape-source) REAL_WORKLOAD_SHAPE_SOURCE="$2"; shift 2 ;;
        --real-workload-isl-min) REAL_WORKLOAD_ISL_MIN="$2"; shift 2 ;;
        --real-workload-isl-max) REAL_WORKLOAD_ISL_MAX="$2"; shift 2 ;;
        --real-workload-isl-mean) REAL_WORKLOAD_ISL_MEAN="$2"; shift 2 ;;
        --real-workload-osl-min) REAL_WORKLOAD_OSL_MIN="$2"; shift 2 ;;
        --real-workload-osl-max) REAL_WORKLOAD_OSL_MAX="$2"; shift 2 ;;
        --real-workload-osl-mean) REAL_WORKLOAD_OSL_MEAN="$2"; shift 2 ;;
        --contexts) CONTEXT_ISL_VALUES="$2"; shift 2 ;;
        --context-values) CONTEXT_ISL_VALUES="$2"; shift 2 ;;
        --ctx-values) CONTEXT_ISL_VALUES="$2"; shift 2 ;;
        --context-isl-values) CONTEXT_ISL_VALUES="$2"; shift 2 ;;
        --context-osl) CONTEXT_OSL="$2"; shift 2 ;;
        --context-repeats) CONTEXT_REPEATS="$2"; shift 2 ;;
        --context-concurrency) CONTEXT_CONCURRENCY="$2"; shift 2 ;;
        --decode-batches) DECODE_BATCH_SIZES="$2"; shift 2 ;;
        --decode-batch-sizes) DECODE_BATCH_SIZES="$2"; shift 2 ;;
        --decode-past-kv) DECODE_PAST_KV="$2"; shift 2 ;;
        --decode-osl) DECODE_OSL="$2"; shift 2 ;;
        --decode-repeats) DECODE_REPEATS="$2"; shift 2 ;;
        --disable-decode-prefix-warmup) DECODE_PREFIX_WARMUP=0; shift ;;
        --mixed-requests) MIX_REQUESTS="$2"; shift 2 ;;
        --mixed-concurrency) MIX_CONCURRENCY="$2"; shift 2 ;;
        --mixed-isl-values) MIX_ISL_VALUES="$2"; shift 2 ;;
        --mixed-osl-values) MIX_OSL_VALUES="$2"; shift 2 ;;
        --mixed-repeats) MIX_REPEATS="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS="$2"; shift 2 ;;
        --warmup-concurrency) WARMUP_CONCURRENCY="$2"; shift 2 ;;
        --warmup-isl-values) WARMUP_ISL_VALUES="$2"; shift 2 ;;
        --warmup-osl-values) WARMUP_OSL_VALUES="$2"; shift 2 ;;
        --post-warmup-seconds) POST_WARMUP_SECONDS="$2"; shift 2 ;;
        --vary-isl-osl) VARY_ISL_OSL=1; shift ;;
        --fixed-workload) VARY_ISL_OSL=0; shift ;;
        --endpoint) REQUEST_ENDPOINT="$2"; shift 2 ;;
        --isl-min) ISL_MIN="$2"; shift 2 ;;
        --isl-max) ISL_MAX="$2"; shift 2 ;;
        --osl-min) OSL_MIN="$2"; shift 2 ;;
        --osl-max) OSL_MAX="$2"; shift 2 ;;
        --isl-values) ISL_VALUES="$2"; shift 2 ;;
        --osl-values) OSL_VALUES="$2"; shift 2 ;;
        --disable-ignore-eos) IGNORE_EOS=0; shift ;;
        --measurement-mode) MEASUREMENT_MODE="$2"; shift 2 ;;
        --nsys-profile-worker) NSYS_PROFILE_WORKER=1; shift ;;
        --nsys-bin) NSYS_BIN="$2"; shift 2 ;;
        --nsys-host-dir) NSYS_HOST_DIR="$2"; shift 2 ;;
        --nsys-trace) NSYS_TRACE="$2"; shift 2 ;;
        --nsys-cuda-graph-trace) NSYS_CUDA_GRAPH_TRACE="$2"; shift 2 ;;
        --nsys-full-worker) NSYS_PROFILE_TRAFFIC_ONLY=0; shift ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --prompt-token-seed) PROMPT_TOKEN_SEED="$2"; shift 2 ;;
        --prompt-token-mode) PROMPT_TOKEN_MODE="$2"; shift 2 ;;
        --request-retries) REQUEST_RETRIES="$2"; shift 2 ;;
        --request-retry-backoff) REQUEST_RETRY_BACKOFF_SECONDS="$2"; shift 2 ;;
        --request-allow-failures) REQUEST_ALLOW_FAILURES="$2"; shift 2 ;;
        --output) OUTPUT_CSV="$2"; shift 2 ;;
        --detail-output) DETAIL_OUTPUT_CSV="$2"; shift 2 ;;
        --phase-output) PHASE_OUTPUT_CSV="$2"; shift 2 ;;
        --workload-output) WORKLOAD_OUTPUT_CSV="$2"; shift 2 ;;
        --warmup-workload-output) WARMUP_WORKLOAD_OUTPUT_CSV="$2"; shift 2 ;;
        --metadata-output) METADATA_OUTPUT_JSON="$2"; shift 2 ;;
        --effective-config-output) EFFECTIVE_CONFIG_OUTPUT_JSON="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --name-prefix) NAME_PREFIX="$2"; shift 2 ;;
        --skip-requests) SKIP_REQUESTS=1; shift ;;
        --keep-running) KEEP_RUNNING=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            WORKER_EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

if [[ "${RUN_DIR}" != /* ]]; then
    RUN_DIR="${PWD}/${RUN_DIR}"
fi
if [[ -z "${OUTPUT_CSV}" ]]; then
    OUTPUT_CSV="${RUN_DIR}/fpm_metrics.csv"
elif [[ "${OUTPUT_CSV}" != /* ]]; then
    OUTPUT_CSV="${PWD}/${OUTPUT_CSV}"
fi
if [[ -z "${DETAIL_OUTPUT_CSV}" ]]; then
    DETAIL_OUTPUT_CSV="${RUN_DIR}/fpm_metrics_detail.csv"
elif [[ "${DETAIL_OUTPUT_CSV}" != /* ]]; then
    DETAIL_OUTPUT_CSV="${PWD}/${DETAIL_OUTPUT_CSV}"
fi
if [[ -z "${PHASE_OUTPUT_CSV}" ]]; then
    PHASE_OUTPUT_CSV="${RUN_DIR}/fpm_metrics_phase.csv"
elif [[ "${PHASE_OUTPUT_CSV}" != /* ]]; then
    PHASE_OUTPUT_CSV="${PWD}/${PHASE_OUTPUT_CSV}"
fi
if [[ -z "${WORKLOAD_OUTPUT_CSV}" ]]; then
    WORKLOAD_OUTPUT_CSV="${RUN_DIR}/request_workload.csv"
elif [[ "${WORKLOAD_OUTPUT_CSV}" != /* ]]; then
    WORKLOAD_OUTPUT_CSV="${PWD}/${WORKLOAD_OUTPUT_CSV}"
fi
if [[ -z "${WARMUP_WORKLOAD_OUTPUT_CSV}" ]]; then
    WARMUP_WORKLOAD_OUTPUT_CSV="${RUN_DIR}/warmup_workload.csv"
elif [[ "${WARMUP_WORKLOAD_OUTPUT_CSV}" != /* ]]; then
    WARMUP_WORKLOAD_OUTPUT_CSV="${PWD}/${WARMUP_WORKLOAD_OUTPUT_CSV}"
fi
if [[ -z "${METADATA_OUTPUT_JSON}" ]]; then
    METADATA_OUTPUT_JSON="${RUN_DIR}/vllm_metadata.json"
elif [[ "${METADATA_OUTPUT_JSON}" != /* ]]; then
    METADATA_OUTPUT_JSON="${PWD}/${METADATA_OUTPUT_JSON}"
fi
if [[ -z "${EFFECTIVE_CONFIG_OUTPUT_JSON}" ]]; then
    EFFECTIVE_CONFIG_OUTPUT_JSON="${RUN_DIR}/effective_vllm_config.json"
elif [[ "${EFFECTIVE_CONFIG_OUTPUT_JSON}" != /* ]]; then
    EFFECTIVE_CONFIG_OUTPUT_JSON="${PWD}/${EFFECTIVE_CONFIG_OUTPUT_JSON}"
fi
HF_HOME_HOST_IS_RUN_LOCAL=0
if [[ -z "${HF_HOME_HOST}" ]]; then
    if [[ -d "${HOME}/.cache/huggingface" ]]; then
        HF_HOME_HOST="${HOME}/.cache/huggingface"
    else
        HF_HOME_HOST="${RUN_DIR}/hf-home"
        HF_HOME_HOST_IS_RUN_LOCAL=1
    fi
fi
if [[ "${HF_HOME_HOST}" != /* ]]; then
    HF_HOME_HOST="${PWD}/${HF_HOME_HOST}"
fi
if [[ -z "${VLLM_CACHE_HOST}" ]]; then
    VLLM_CACHE_HOST="${HOME}/.cache/aic-vllm"
fi
if [[ "${VLLM_CACHE_HOST}" != /* ]]; then
    VLLM_CACHE_HOST="${PWD}/${VLLM_CACHE_HOST}"
fi
OUTPUT_DIR="$(dirname "${OUTPUT_CSV}")"
DETAIL_OUTPUT_DIR="$(dirname "${DETAIL_OUTPUT_CSV}")"
PHASE_OUTPUT_DIR="$(dirname "${PHASE_OUTPUT_CSV}")"
WORKLOAD_OUTPUT_DIR="$(dirname "${WORKLOAD_OUTPUT_CSV}")"
WARMUP_WORKLOAD_OUTPUT_DIR="$(dirname "${WARMUP_WORKLOAD_OUTPUT_CSV}")"
METADATA_OUTPUT_DIR="$(dirname "${METADATA_OUTPUT_JSON}")"
EFFECTIVE_CONFIG_OUTPUT_DIR="$(dirname "${EFFECTIVE_CONFIG_OUTPUT_JSON}")"
COLLECTOR_OUTPUT_CSV="${RUN_DIR}/fpm_metrics.csv"
COLLECTOR_OUTPUT_IN_CONTAINER="/work/fpm_metrics.csv"
COLLECTOR_DETAIL_CSV="${RUN_DIR}/fpm_metrics_detail.csv"
COLLECTOR_DETAIL_IN_CONTAINER="/work/fpm_metrics_detail.csv"
COLLECTOR_PHASE_CSV="${RUN_DIR}/fpm_metrics_phase.csv"
REQUEST_WORKLOAD_CSV="${RUN_DIR}/request_workload.csv"
REQUEST_WORKLOAD_IN_CONTAINER="/work/request_workload.csv"
WARMUP_WORKLOAD_CSV="${RUN_DIR}/warmup_workload.csv"
WARMUP_WORKLOAD_IN_CONTAINER="/work/warmup_workload.csv"
RUN_METADATA_JSON="${RUN_DIR}/vllm_metadata.json"
RUN_EFFECTIVE_CONFIG_JSON="${RUN_DIR}/effective_vllm_config.json"
RUN_EFFECTIVE_CONFIG_IN_CONTAINER="/work/effective_vllm_config.json"
SEGMENT_FILE="${RUN_DIR}/fpm_segment.txt"
SEGMENT_IN_CONTAINER="/work/fpm_segment.txt"
NSYS_WORKER_OUTPUT_BASE="${RUN_DIR}/nsys/fpm_worker"
NSYS_WORKER_OUTPUT_IN_CONTAINER="/work/nsys/fpm_worker"
MODEL_IN_CONTAINER="${MODEL}"
MODEL_REQUEST_NAME="${MODEL}"

FRONTEND_NAME="${NAME_PREFIX}-frontend"
WORKER_NAME="${NAME_PREFIX}-worker"
COLLECTOR_NAME="${NAME_PREFIX}-collector"

DOCKER_ENV=(
    -e "DYN_DISCOVERY_BACKEND=file"
    -e "DYN_REQUEST_PLANE=tcp"
    -e "DYN_EVENT_PLANE=zmq"
    -e "DYN_FILE_KV=/work/discovery"
    -e "DYN_NAMESPACE=dynamo"
)
WORKER_DOCKER_ENV=("${DOCKER_ENV[@]}")
WORKER_DOCKER_ENV+=(
    -e "TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR_CONTAINER}"
    -e "TILELANG_TMP_DIR=${TILELANG_TMP_DIR_CONTAINER}"
)
if [[ -n "${VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8:-}" ]]; then
    WORKER_DOCKER_ENV+=(
        -e "VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=${VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8}"
    )
fi
NSYS_DOCKER_MOUNTS=()
HF_TOKEN_FILE_HOST="${HF_TOKEN_FILE:-/home/shadeform/hf.token}"
HF_TOKEN_DOCKER_MOUNTS=()
HF_TOKEN_CONTAINER_PREFIX=()
if [[ -f "${HF_TOKEN_FILE_HOST}" ]]; then
    HF_TOKEN_DOCKER_MOUNTS=(-v "${HF_TOKEN_FILE_HOST}:/run/secrets/hf.token:ro")
    HF_TOKEN_CONTAINER_PREFIX=(
        bash
        -lc
        'if [[ -f /run/secrets/hf.token ]]; then export HF_TOKEN="$(tr -d "\r\n" < /run/secrets/hf.token)"; fi; exec "$@"'
        bash
    )
fi

mkdir -p \
    "${RUN_DIR}/discovery" \
    "${OUTPUT_DIR}" \
    "${DETAIL_OUTPUT_DIR}" \
    "${PHASE_OUTPUT_DIR}" \
    "${WORKLOAD_OUTPUT_DIR}" \
    "${WARMUP_WORKLOAD_OUTPUT_DIR}" \
    "${METADATA_OUTPUT_DIR}" \
    "${EFFECTIVE_CONFIG_OUTPUT_DIR}" \
    "${RUN_DIR}/nsys" \
    "${HF_HOME_HOST}" \
    "${VLLM_CACHE_HOST}" \
    "${VLLM_CACHE_HOST}/tilelang/tmp"
chmod a+rwx "${RUN_DIR}" "${RUN_DIR}/discovery" "${VLLM_CACHE_HOST}" "${VLLM_CACHE_HOST}/tilelang" "${VLLM_CACHE_HOST}/tilelang/tmp"
if [[ "${HF_HOME_HOST_IS_RUN_LOCAL}" == "1" ]]; then
    chmod a+rwx "${HF_HOME_HOST}"
fi

if [[ "${NSYS_PROFILE_WORKER}" == "1" ]]; then
    if [[ -z "${NSYS_HOST_DIR}" && "${NSYS_BIN}" == /* && -x "${NSYS_BIN}" ]]; then
        NSYS_HOST_DIR="$(dirname "${NSYS_BIN}")"
    fi
    if [[ -n "${NSYS_HOST_DIR}" ]]; then
        [[ -d "${NSYS_HOST_DIR}" ]] || die "--nsys-host-dir does not exist: ${NSYS_HOST_DIR}"
        NSYS_DOCKER_MOUNTS=(-v "${NSYS_HOST_DIR}:${NSYS_HOST_DIR}:ro")
    fi
fi

case "${REQUEST_ENDPOINT}" in
    completions) ;;
    *) die "--endpoint must be 'completions' because generated workloads use random prompt token IDs" ;;
esac

if [[ -z "${WARMUP_CONCURRENCY}" ]]; then
    WARMUP_CONCURRENCY="${CONCURRENCY}"
fi
if (( WARMUP_REQUESTS < 0 )); then
    die "invalid warmup request count: ${WARMUP_REQUESTS}"
fi
if (( WARMUP_CONCURRENCY < 1 )); then
    die "invalid warmup concurrency: ${WARMUP_CONCURRENCY}"
fi
if (( REQUEST_RETRIES < 0 )); then
    die "invalid request retry count: ${REQUEST_RETRIES}"
fi
if (( REQUEST_ALLOW_FAILURES < 0 )); then
    die "invalid allowed request failure count: ${REQUEST_ALLOW_FAILURES}"
fi
if (( FILE_DISCOVERY_TOUCH_SECONDS < 0 )); then
    die "invalid file discovery touch interval: ${FILE_DISCOVERY_TOUCH_SECONDS}"
fi
if [[ -n "${TP_SIZE}" ]]; then
    TP_SIZE="$(single_parallel_size "TP_SIZE" "${TP_SIZE}")"
fi
if [[ -n "${DATA_PARALLEL_SIZE}" ]]; then
    DATA_PARALLEL_SIZE="$(single_parallel_size "DATA_PARALLEL_SIZE" "${DATA_PARALLEL_SIZE}")"
fi
EP_SIZE="$(single_parallel_size "EP_SIZE" "${EP_SIZE}")"
if (( EP_SIZE > 1 )); then
    ENABLE_EXPERT_PARALLEL=1
    if [[ -z "${TP_SIZE}" ]]; then
        TP_SIZE=1
    fi
    if [[ -z "${DATA_PARALLEL_SIZE}" ]]; then
        if (( EP_SIZE % TP_SIZE != 0 )); then
            die "--ep-size ${EP_SIZE} is not divisible by --tp-size ${TP_SIZE}; vLLM EP size is TP_SIZE * DATA_PARALLEL_SIZE"
        fi
        DATA_PARALLEL_SIZE="$((EP_SIZE / TP_SIZE))"
    elif (( EP_SIZE != TP_SIZE * DATA_PARALLEL_SIZE )); then
        die "--ep-size ${EP_SIZE} conflicts with --tp-size ${TP_SIZE} and --data-parallel-size ${DATA_PARALLEL_SIZE}; vLLM EP size is TP_SIZE * DATA_PARALLEL_SIZE"
    fi
fi
if [[ -z "${DATA_PARALLEL_SIZE}" ]]; then
    DATA_PARALLEL_SIZE=1
fi
if [[ -z "${TP_SIZE}" && "${DATA_PARALLEL_SIZE}" != "1" ]]; then
    TP_SIZE=1
fi
if [[ -z "${GPUS}" ]]; then
    gpu_count="$(( ${TP_SIZE:-1} * DATA_PARALLEL_SIZE ))"
    GPUS="$(infer_docker_gpus "${gpu_count}")"
fi
apply_vllm_runtime_defaults
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    WORKER_EXTRA_ARGS+=(--enable-expert-parallel)
fi

case "${WORKLOAD_PLAN}" in
    sweep|legacy) ;;
    *) die "--workload-plan must be 'sweep' or 'legacy'" ;;
esac
case "${PROMPT_TOKEN_MODE}" in
    random_vocab_excluding_special|safe_ascii) ;;
    *) die "--prompt-token-mode must be 'random_vocab_excluding_special' or 'safe_ascii'" ;;
esac
MEASURED_PHASES="${MEASURED_PHASES//[[:space:]]/}"
if [[ -z "${MEASURED_PHASES}" ]]; then
    die "--measured-phases must not be empty"
fi
for phase in ${MEASURED_PHASES//,/ }; do
    case "${phase}" in
        context|decode|mixed) ;;
        *) die "unknown measured phase '${phase}' in '${MEASURED_PHASES}'" ;;
    esac
done

RUN_SWEEP=0
RUN_REAL_WORKLOAD=0
if [[ "${WORKLOAD_PLAN}" == "sweep" ]]; then
    if [[ "${REAL_WORKLOAD}" == "1" ]]; then
        RUN_REAL_WORKLOAD=1
        if [[ "${INCLUDE_SWEEP}" == "1" ]]; then
            RUN_SWEEP=1
        fi
    else
        RUN_SWEEP=1
    fi
fi

if [[ "${WORKLOAD_PLAN}" == "sweep" ]]; then
    if [[ "${RUN_REAL_WORKLOAD}" == "1" ]]; then
        if ! [[ "${REAL_WORKLOAD_REQUESTS}" =~ ^[0-9]+$ ]] || (( REAL_WORKLOAD_REQUESTS < 1 )); then
            die "real workload requests must be >= 1"
        fi
        if ! [[ "${REAL_WORKLOAD_CONCURRENCY}" =~ ^[0-9]+$ ]] || (( REAL_WORKLOAD_CONCURRENCY < 1 )); then
            die "real workload concurrency must be >= 1"
        fi
        if [[ -n "${MAX_NUM_SEQS}" ]] && (( REAL_WORKLOAD_CONCURRENCY > MAX_NUM_SEQS )); then
            die "real workload concurrency ${REAL_WORKLOAD_CONCURRENCY} exceeds --max-num-seqs ${MAX_NUM_SEQS}"
        fi
        case "${REAL_WORKLOAD_SHAPE_SOURCE}" in
            scaled_dataset|synthetic) ;;
            *) die "invalid real workload shape source: ${REAL_WORKLOAD_SHAPE_SOURCE}" ;;
        esac
        if (( REAL_WORKLOAD_ISL_MIN < 1 || REAL_WORKLOAD_ISL_MAX < REAL_WORKLOAD_ISL_MIN )); then
            die "invalid real workload ISL range: ${REAL_WORKLOAD_ISL_MIN}..${REAL_WORKLOAD_ISL_MAX}"
        fi
        if (( REAL_WORKLOAD_OSL_MIN < 1 || REAL_WORKLOAD_OSL_MAX < REAL_WORKLOAD_OSL_MIN )); then
            die "invalid real workload OSL range: ${REAL_WORKLOAD_OSL_MIN}..${REAL_WORKLOAD_OSL_MAX}"
        fi
    fi
    if [[ "${RUN_SWEEP}" == "1" ]]; then
        if (( CONTEXT_REPEATS < 1 || CONTEXT_CONCURRENCY < 1 )); then
            die "context repeats/concurrency must be >= 1"
        fi
        if (( DECODE_REPEATS < 1 || DECODE_PAST_KV < 1 || DECODE_OSL < 1 )); then
            die "decode repeats, past_kv, and OSL must be >= 1"
        fi
        if (( MIX_REPEATS < 1 || MIX_REQUESTS < 1 || MIX_CONCURRENCY < 1 )); then
            die "mixed repeats, requests, and concurrency must be >= 1"
        fi
        if phase_enabled context; then
            context_max_isl=$(csv_max "${CONTEXT_ISL_VALUES}")
            if (( context_max_isl < 1 )); then
                die "context ISL list must not be empty"
            fi
            if [[ -n "${MAX_MODEL_LEN}" ]] && (( context_max_isl + CONTEXT_OSL > MAX_MODEL_LEN )); then
                die "context max ISL + OSL (${context_max_isl} + ${CONTEXT_OSL}) exceeds --max-model-len ${MAX_MODEL_LEN}"
            fi
        fi
        if phase_enabled decode; then
            decode_max_batch=$(csv_max "${DECODE_BATCH_SIZES}")
            if (( decode_max_batch < 1 )); then
                die "decode batch-size list must not be empty"
            fi
            if [[ -n "${MAX_NUM_SEQS}" ]] && (( decode_max_batch > MAX_NUM_SEQS )); then
                die "decode max batch size ${decode_max_batch} exceeds --max-num-seqs ${MAX_NUM_SEQS}"
            fi
            if [[ -n "${MAX_MODEL_LEN}" ]] && (( DECODE_PAST_KV + DECODE_OSL > MAX_MODEL_LEN )); then
                die "decode past_kv + OSL (${DECODE_PAST_KV} + ${DECODE_OSL}) exceeds --max-model-len ${MAX_MODEL_LEN}"
            fi
        fi
        if phase_enabled mixed; then
            mixed_max_isl=$(csv_max "${MIX_ISL_VALUES}")
            mixed_max_osl=$(csv_max "${MIX_OSL_VALUES}")
            if (( mixed_max_isl < 1 || mixed_max_osl < 1 )); then
                die "mixed ISL/OSL lists must not be empty"
            fi
            if [[ -n "${MAX_NUM_SEQS}" ]] && (( MIX_CONCURRENCY > MAX_NUM_SEQS )); then
                die "mixed concurrency ${MIX_CONCURRENCY} exceeds --max-num-seqs ${MAX_NUM_SEQS}"
            fi
            if [[ -n "${MAX_MODEL_LEN}" ]] && (( mixed_max_isl + mixed_max_osl > MAX_MODEL_LEN )); then
                die "mixed max ISL + OSL (${mixed_max_isl} + ${mixed_max_osl}) exceeds --max-model-len ${MAX_MODEL_LEN}"
            fi
        fi
    fi
fi

if [[ "${VARY_ISL_OSL}" == "1" && "${WORKLOAD_PLAN}" == "legacy" ]]; then
    if (( ISL_MIN < 1 || ISL_MAX < ISL_MIN )); then
        die "invalid ISL range: ${ISL_MIN}..${ISL_MAX}"
    fi
    if (( OSL_MIN < 1 || OSL_MAX < OSL_MIN )); then
        die "invalid OSL range: ${OSL_MIN}..${OSL_MAX}"
    fi
    if [[ -n "${MAX_MODEL_LEN}" ]] && (( ISL_MAX + OSL_MAX > MAX_MODEL_LEN )); then
        die "ISL_MAX + OSL_MAX (${ISL_MAX} + ${OSL_MAX}) exceeds --max-model-len ${MAX_MODEL_LEN}. Increase --max-model-len."
    fi
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    command -v docker >/dev/null 2>&1 || die "docker is not on PATH"
    docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "Docker image '${IMAGE}' was not found locally. Build it or pass --image / DYNAMO_VLLM_IMAGE."
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    log "Verifying image '${IMAGE}' uses vLLM ${EXPECTED_VLLM_VERSION}"
    actual_version="$(
        docker run --rm --network none "${IMAGE}" \
            python3 -c 'import vllm; print(vllm.__version__)' |
        sed -nE 's/^[[:space:]]*([0-9]+[.][0-9]+[.][0-9]+).*$/\1/p' \
        | tail -n 1
    )"
    if [[ "${actual_version}" != "${EXPECTED_VLLM_VERSION}" ]]; then
        msg="image '${IMAGE}' has vLLM ${actual_version}, expected ${EXPECTED_VLLM_VERSION}"
        if [[ "${ALLOW_VERSION_MISMATCH}" == "1" ]]; then
            log "WARNING: ${msg}"
        else
            die "${msg}. Rebuild/pass a vLLM 0.20.1 Dynamo image or use --allow-version-mismatch."
        fi
    fi

    docker run --rm --network none "${IMAGE}" \
        python3 -c 'import dynamo.common.forward_pass_metrics; import dynamo.vllm.instrumented_scheduler' \
        >/dev/null
fi

for name in "${FRONTEND_NAME}" "${WORKER_NAME}" "${COLLECTOR_NAME}"; do
    if [[ "${DRY_RUN}" != "1" ]] && container_exists "${name}"; then
        die "container '${name}' already exists. Remove it or choose --name-prefix."
    fi
done

stage_helper_scripts() {
    local helper
    for helper in fpm_collect.py send_requests.py wait_http.py summarize_fpm.py; do
        cp "${SCRIPT_DIR}/${helper}" "${RUN_DIR}/${helper}"
        chmod a+r "${RUN_DIR}/${helper}"
    done
    mkdir -p "${RUN_DIR}/common"
    cp "${COMMON_DIR}/random_prompt_tokens.py" "${RUN_DIR}/common/random_prompt_tokens.py"
    cp "${COMMON_DIR}/vllm_deployment.py" "${RUN_DIR}/vllm_deployment.py"
    chmod a+r "${RUN_DIR}/common/random_prompt_tokens.py"
    chmod a+r "${RUN_DIR}/vllm_deployment.py"
}

stage_helper_scripts

stage_local_model_dir() {
    if [[ ! -d "${MODEL}" ]]; then
        return
    fi
    rm -rf "${RUN_DIR}/model"
    mkdir -p "${RUN_DIR}/model"
    cp -a "${MODEL}/." "${RUN_DIR}/model/"
    chmod -R a+rX "${RUN_DIR}/model"
    MODEL_IN_CONTAINER="/work/model"
    MODEL_REQUEST_NAME="/work/model"
}

stage_local_model_dir

REQUEST_INDEX_OFFSET=0

start_file_discovery_touch_loop() {
    if [[ "${FILE_DISCOVERY_TOUCH_SECONDS}" == "0" || "${DRY_RUN}" == "1" ]]; then
        return
    fi
    (
        while true; do
            if [[ -d "${RUN_DIR}/discovery" ]]; then
                find "${RUN_DIR}/discovery" -type f -exec touch {} + 2>/dev/null || true
            fi
            sleep "${FILE_DISCOVERY_TOUCH_SECONDS}"
        done
    ) &
    DISCOVERY_TOUCH_PID=$!
    log "Refreshing local file-discovery mtimes every ${FILE_DISCOVERY_TOUCH_SECONDS}s (pid ${DISCOVERY_TOUCH_PID})"
}

deployment_helper_args() {
    local args=(
        --model "${MODEL_IN_CONTAINER}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    )
    if [[ -n "${MAX_NUM_SEQS}" ]]; then
        args+=(--max-num-seqs "${MAX_NUM_SEQS}")
    fi
    if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
        args+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
    fi
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        args+=(--max-model-len "${MAX_MODEL_LEN}")
    fi
    if [[ -n "${TP_SIZE}" ]]; then
        args+=(--tensor-parallel-size "${TP_SIZE}")
    fi
    if [[ "${DATA_PARALLEL_SIZE}" != "1" || "${DATA_PARALLEL_SIZE_EXPLICIT}" == "1" ]]; then
        args+=(--data-parallel-size "${DATA_PARALLEL_SIZE}")
    fi
    if [[ -n "${KV_CACHE_DTYPE}" ]]; then
        args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
        args+=(--enforce-eager)
    fi
    if [[ "${DISABLE_PREFIX_CACHING}" == "1" ]]; then
        args+=(--disable-prefix-caching)
    fi
    local extra
    for extra in "${WORKER_EXTRA_ARGS[@]}"; do
        args+=("--extra-arg=${extra}")
    done
    printf '%s\n' "${args[@]}"
}

build_vllm_deployment_args() {
    local helper_args=()
    mapfile -t helper_args < <(deployment_helper_args)
    mapfile -t VLLM_DEPLOYMENT_ARGS < <(
        python3 "${COMMON_DIR}/vllm_deployment.py" build-args "${helper_args[@]}" --format lines
    )
    VLLM_DEPLOYMENT_ARGS_JSON="$(
        python3 "${COMMON_DIR}/vllm_deployment.py" build-args "${helper_args[@]}" --format json
    )"
}

write_vllm_run_metadata() {
    local helper_args=()
    mapfile -t helper_args < <(deployment_helper_args)
    local metadata_args=(
        "${COMMON_DIR}/vllm_deployment.py"
        write-metadata
        "${helper_args[@]}"
        --artifact-kind fpm
        --measurement-mode "${MEASUREMENT_MODE}"
        --output "${RUN_METADATA_JSON}"
    )
    if [[ -f "${RUN_EFFECTIVE_CONFIG_JSON}" ]]; then
        metadata_args+=(--effective-config "${RUN_EFFECTIVE_CONFIG_JSON}")
    fi
    python3 "${metadata_args[@]}"
}

snapshot_effective_vllm_config() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi
    local snapshot_rc=0
    run docker run --rm \
        --network host \
        -v "${RUN_DIR}:/work" \
        -v "${HF_HOME_HOST}:/work/hf-home" \
        "${HF_TOKEN_DOCKER_MOUNTS[@]}" \
        -e "HF_HOME=/work/hf-home" \
        -e "HF_HUB_CACHE=/work/hf-home/hub" \
        -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
        "${IMAGE}" \
        "${HF_TOKEN_CONTAINER_PREFIX[@]}" \
        python3 /work/vllm_deployment.py snapshot-effective \
            --args-json "${VLLM_DEPLOYMENT_ARGS_JSON}" \
            --output "${RUN_EFFECTIVE_CONFIG_IN_CONTAINER}" || snapshot_rc=$?
    if [[ "${snapshot_rc}" != "0" ]]; then
        log "WARNING: failed to snapshot effective vLLM config; metadata will contain requested args only"
    fi
}

resolved_max_model_len() {
    if [[ -n "${MAX_MODEL_LEN}" ]]; then
        printf '%s\n' "${MAX_MODEL_LEN}"
        return
    fi
    if [[ ! -f "${RUN_EFFECTIVE_CONFIG_JSON}" ]]; then
        return
    fi
    python3 - "${RUN_EFFECTIVE_CONFIG_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
value = data.get("model_config.max_model_len")
if value is not None:
    print(value)
PY
}

send_request_workload() {
    local phase="$1"
    local container_suffix="$2"
    local request_count="$3"
    local concurrency_count="$4"
    local workload_in_container="$5"
    local isl_values="$6"
    local osl_values="$7"
    local append_workload="${8:-0}"
    local seed_offset="${9:-0}"
    local real_workload="${10:-0}"
    local workload_segment="${11:-${phase}}"
    local request_rc=0

    printf '%s\n' "${workload_segment}" > "${SEGMENT_FILE}"

    local request_driver_cmd=(
        python3 /work/send_requests.py
        --url "http://127.0.0.1:${HTTP_PORT}"
        --model "${MODEL_REQUEST_NAME}"
        --requests "${request_count}"
        --concurrency "${concurrency_count}"
        --max-tokens "${MAX_TOKENS}"
        --prompt-token-mode "${PROMPT_TOKEN_MODE}"
        --endpoint "${REQUEST_ENDPOINT}"
        --isl-min "${ISL_MIN}"
        --isl-max "${ISL_MAX}"
        --osl-min "${OSL_MIN}"
        --osl-max "${OSL_MAX}"
        --isl-values "${isl_values}"
        --osl-values "${osl_values}"
        --workload-output "${workload_in_container}"
        --workload-label "${phase}"
        --request-index-offset "${REQUEST_INDEX_OFFSET}"
        --timeout "${REQUEST_TIMEOUT_SECONDS}"
        --retries "${REQUEST_RETRIES}"
        --retry-backoff "${REQUEST_RETRY_BACKOFF_SECONDS}"
        --allow-failures "${REQUEST_ALLOW_FAILURES}"
    )
    if [[ -n "${PROMPT_TOKEN_SEED}" ]]; then
        local prompt_token_seed="$((PROMPT_TOKEN_SEED + seed_offset))"
        request_driver_cmd+=(--prompt-token-seed "${prompt_token_seed}")
    fi
    if [[ "${real_workload}" == "1" ]]; then
        request_driver_cmd+=(
            --real-workload
            --real-workload-dataset "${REAL_WORKLOAD_DATASET}"
            --real-workload-max-rows "${REAL_WORKLOAD_MAX_ROWS}"
            --real-workload-shape-source "${REAL_WORKLOAD_SHAPE_SOURCE}"
            --real-workload-isl-min "${REAL_WORKLOAD_ISL_MIN}"
            --real-workload-isl-max "${REAL_WORKLOAD_ISL_MAX}"
            --real-workload-isl-mean "${REAL_WORKLOAD_ISL_MEAN}"
            --real-workload-osl-min "${REAL_WORKLOAD_OSL_MIN}"
            --real-workload-osl-max "${REAL_WORKLOAD_OSL_MAX}"
            --real-workload-osl-mean "${REAL_WORKLOAD_OSL_MEAN}"
        )
        local request_max_model_len
        request_max_model_len="$(resolved_max_model_len)"
        if [[ -n "${request_max_model_len}" ]]; then
            request_driver_cmd+=(--max-model-len "${request_max_model_len}")
        fi
    fi
    if [[ "${append_workload}" == "1" ]]; then
        request_driver_cmd+=(--append-workload)
    fi
    if [[ "${VARY_ISL_OSL}" == "1" || "${WORKLOAD_PLAN}" == "sweep" ]]; then
        request_driver_cmd+=(--vary-isl-osl)
        local isl_desc="${ISL_MIN}..${ISL_MAX}"
        local osl_desc="${OSL_MIN}..${OSL_MAX}"
        if [[ -n "${isl_values}" ]]; then
            isl_desc="${isl_values}"
        fi
        if [[ -n "${osl_values}" ]]; then
            osl_desc="${osl_values}"
        fi
        if [[ "${real_workload}" == "1" ]]; then
            log "Sending ${request_count} real-workload requests at concurrency ${concurrency_count} (dataset ${REAL_WORKLOAD_DATASET})"
        else
            log "Sending ${request_count} ${phase} variable ISL/OSL requests at concurrency ${concurrency_count} (ISL ${isl_desc}, OSL ${osl_desc})"
        fi
    else
        log "Sending ${request_count} ${phase} fixed requests at concurrency ${concurrency_count}"
    fi
    if [[ "${IGNORE_EOS}" == "1" ]]; then
        request_driver_cmd+=(--ignore-eos)
    fi

    run docker run --rm \
        --name "${NAME_PREFIX}-${container_suffix}" \
        --network host \
        -v "${RUN_DIR}:/work" \
        -v "${HF_HOME_HOST}:/work/hf-home" \
        "${HF_TOKEN_DOCKER_MOUNTS[@]}" \
        -e "HF_HOME=/work/hf-home" \
        -e "HF_HUB_CACHE=/work/hf-home/hub" \
        -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
        "${IMAGE}" \
        "${HF_TOKEN_CONTAINER_PREFIX[@]}" \
        "${request_driver_cmd[@]}" || request_rc=$?

    REQUEST_INDEX_OFFSET=$((REQUEST_INDEX_OFFSET + request_count))
    return "${request_rc}"
}

decode_prefix_warmup_enabled() {
    [[ "${WORKLOAD_PLAN}" == "sweep" ]] || return 1
    [[ "${DECODE_PREFIX_WARMUP}" == "1" ]] || return 1
    [[ "${MEASURED_PHASES}" == "decode" ]] || return 1
    phase_enabled decode
}

ensure_decode_prefix_warmup_seed() {
    if [[ -n "${PROMPT_TOKEN_SEED}" ]]; then
        return
    fi
    PROMPT_TOKEN_SEED=0
    log "Using PROMPT_TOKEN_SEED=${PROMPT_TOKEN_SEED} so decode prefix warmup and measured decode prompts match"
}

warm_decode_prefix_cache() {
    local repeat batch_size
    local warmup_rc=0
    local saved_request_index_offset="${REQUEST_INDEX_OFFSET}"
    rm -f "${WARMUP_WORKLOAD_CSV}"
    log "Priming decode prefix cache for decode-only sweep before FPM collection"
    for repeat in $(seq 1 "${DECODE_REPEATS}"); do
        IFS=',' read -ra decode_batches <<< "${DECODE_BATCH_SIZES}"
        for batch_size in "${decode_batches[@]}"; do
            batch_size="${batch_size//[[:space:]]/}"
            if [[ -z "${batch_size}" ]]; then
                continue
            fi
            send_request_workload \
                "decode_prefix_warmup_b${batch_size}" \
                "request-driver-decode-prefix-warmup-b${batch_size}-r${repeat}" \
                "${batch_size}" \
                "${batch_size}" \
                "${WARMUP_WORKLOAD_IN_CONTAINER}" \
                "${DECODE_PAST_KV}" \
                "1" \
                1 \
                0 \
                0 \
                "warmup" || warmup_rc=$?
            if [[ "${warmup_rc}" != "0" ]]; then
                break 2
            fi
        done
    done
    REQUEST_INDEX_OFFSET="${saved_request_index_offset}"
    if [[ "${DRY_RUN}" != "1" ]]; then
        log "Waiting ${POST_WARMUP_SECONDS}s after decode prefix warmup before starting FPM collector"
        sleep "${POST_WARMUP_SECONDS}"
    fi
    return "${warmup_rc}"
}

start_nsys_worker_collection() {
    if [[ "${NSYS_PROFILE_WORKER}" != "1" || "${NSYS_PROFILE_TRAFFIC_ONLY}" != "1" ]]; then
        return
    fi
    log "Starting Nsight worker collection session ${NSYS_SESSION_NAME}"
    run docker exec "${WORKER_NAME}" "${NSYS_BIN}" start --session "${NSYS_SESSION_NAME}" || \
        log "WARNING: failed to start Nsight session ${NSYS_SESSION_NAME}"
}

stop_nsys_worker_collection() {
    if [[ "${NSYS_PROFILE_WORKER}" != "1" || "${NSYS_PROFILE_TRAFFIC_ONLY}" != "1" ]]; then
        return
    fi
    log "Stopping Nsight worker collection session ${NSYS_SESSION_NAME}"
    run docker exec "${WORKER_NAME}" "${NSYS_BIN}" stop --session "${NSYS_SESSION_NAME}" || \
        log "WARNING: failed to stop Nsight session ${NSYS_SESSION_NAME}"
}

log "Run directory: ${RUN_DIR}"
log "CSV output: ${OUTPUT_CSV}"
log "FPM detail CSV: ${DETAIL_OUTPUT_CSV}"
log "FPM phase CSV: ${PHASE_OUTPUT_CSV}"
log "Request workload CSV: ${WORKLOAD_OUTPUT_CSV}"
log "vLLM metadata JSON: ${METADATA_OUTPUT_JSON}"
log "vLLM effective config JSON: ${EFFECTIVE_CONFIG_OUTPUT_JSON}"
if [[ "${NSYS_PROFILE_WORKER}" == "1" ]]; then
    log "Nsight worker profile: ${NSYS_WORKER_OUTPUT_BASE}.nsys-rep"
fi
log "Measured phases: ${MEASURED_PHASES}"
if [[ "${REAL_WORKLOAD}" == "1" ]]; then
    log "Real workload: dataset=${REAL_WORKLOAD_DATASET}, shape_source=${REAL_WORKLOAD_SHAPE_SOURCE}, requests=${REAL_WORKLOAD_REQUESTS}, concurrency=${REAL_WORKLOAD_CONCURRENCY}, ISL=${REAL_WORKLOAD_ISL_MIN}..${REAL_WORKLOAD_ISL_MAX} mean~${REAL_WORKLOAD_ISL_MEAN}, OSL=${REAL_WORKLOAD_OSL_MIN}..${REAL_WORKLOAD_OSL_MAX} mean~${REAL_WORKLOAD_OSL_MEAN}"
fi
if [[ "${WORKLOAD_PLAN}" != "sweep" ]]; then
    log "Advanced workload plan: ${WORKLOAD_PLAN}"
fi
if [[ "${WARMUP_REQUESTS}" != "0" ]]; then
    log "Warmup workload CSV: ${WARMUP_WORKLOAD_OUTPUT_CSV}"
fi

CLEANUP_ENABLED=1
start_file_discovery_touch_loop
build_vllm_deployment_args
snapshot_effective_vllm_config
write_vllm_run_metadata

log "Starting frontend container ${FRONTEND_NAME}"
run docker run -d \
    --name "${FRONTEND_NAME}" \
    --network host \
    -v "${RUN_DIR}:/work" \
    "${DOCKER_ENV[@]}" \
    -e "DYN_HTTP_PORT=${HTTP_PORT}" \
    "${IMAGE}" \
    python3 -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --discovery-backend file \
        --request-plane tcp \
        --event-plane zmq

WORKER_CMD=(
    python3 -m dynamo.vllm
    "${VLLM_DEPLOYMENT_ARGS[@]}"
    --discovery-backend file
    --request-plane tcp
    --event-plane zmq
    --kv-events-config '{"enable_kv_cache_events": false}'
)
WORKER_CONTAINER_CMD=("${WORKER_CMD[@]}")
if [[ "${NSYS_PROFILE_WORKER}" == "1" ]]; then
    WORKER_CONTAINER_CMD=(
        "${NSYS_BIN}" profile
        "--trace=${NSYS_TRACE}"
        "--cuda-graph-trace=${NSYS_CUDA_GRAPH_TRACE}"
        --trace-fork-before-exec=false
        --sample=none
        --cpuctxsw=none
        --force-overwrite=true
        --output "${NSYS_WORKER_OUTPUT_IN_CONTAINER}"
    )
    if [[ "${NSYS_PROFILE_TRAFFIC_ONLY}" == "1" ]]; then
        WORKER_CONTAINER_CMD+=(
            --session-new "${NSYS_SESSION_NAME}"
            --start-later=true
        )
    fi
    WORKER_CONTAINER_CMD+=("${WORKER_CMD[@]}")
fi

log "Starting vLLM worker container ${WORKER_NAME}"
run docker run -d \
    --name "${WORKER_NAME}" \
    --network host \
    --gpus "${GPUS}" \
    -v "${RUN_DIR}:/work" \
    -v "${HF_HOME_HOST}:/work/hf-home" \
    -v "${VLLM_CACHE_HOST}:/home/dynamo/.cache/vllm" \
    -v "${VLLM_CACHE_HOST}:/root/.cache/vllm" \
    "${HF_TOKEN_DOCKER_MOUNTS[@]}" \
    "${NSYS_DOCKER_MOUNTS[@]}" \
    "${WORKER_DOCKER_ENV[@]}" \
    -e "DYN_FORWARDPASS_METRIC_PORT=${FPM_PORT}" \
    -e "DYN_SYSTEM_PORT=${SYSTEM_PORT}" \
    -e "HF_HOME=/work/hf-home" \
    -e "HF_HUB_CACHE=/work/hf-home/hub" \
    -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
    "${IMAGE}" \
    "${HF_TOKEN_CONTAINER_PREFIX[@]}" \
    "${WORKER_CONTAINER_CMD[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
    log "Waiting for frontend health"
    docker run --rm --network host -v "${RUN_DIR}:/work" "${IMAGE}" \
        python3 /work/wait_http.py \
            --url "http://127.0.0.1:${HTTP_PORT}/health" \
            --timeout "${START_TIMEOUT_SECONDS}" >/dev/null

    log "Waiting for model registration (${MODEL_REQUEST_NAME})"
    docker run --rm --network host -v "${RUN_DIR}:/work" "${IMAGE}" \
        python3 /work/wait_http.py \
            --url "http://127.0.0.1:${HTTP_PORT}/v1/models" \
            --contains "${MODEL_REQUEST_NAME}" \
            --timeout "${START_TIMEOUT_SECONDS}" >/dev/null
fi

if [[ "${SKIP_REQUESTS}" != "1" && "${WARMUP_REQUESTS}" != "0" ]]; then
    send_request_workload \
        "warmup" \
        "warmup-driver" \
        "${WARMUP_REQUESTS}" \
        "${WARMUP_CONCURRENCY}" \
        "${WARMUP_WORKLOAD_IN_CONTAINER}" \
        "${WARMUP_ISL_VALUES}" \
        "${WARMUP_OSL_VALUES}" \
        0 \
        1000000000 \
        0 \
        "warmup"
    if [[ "${DRY_RUN}" != "1" ]]; then
        log "Waiting ${POST_WARMUP_SECONDS}s after warmup before starting FPM collector"
        sleep "${POST_WARMUP_SECONDS}"
    fi
fi

if [[ "${SKIP_REQUESTS}" != "1" ]] && decode_prefix_warmup_enabled; then
    ensure_decode_prefix_warmup_seed
    warm_decode_prefix_cache || die "decode prefix warmup failed"
fi

log "Starting FPM collector container ${COLLECTOR_NAME}"
run docker run -d \
    --name "${COLLECTOR_NAME}" \
    --network host \
    -v "${RUN_DIR}:/work" \
    "${IMAGE}" \
    python3 /work/fpm_collect.py \
        --port "${FPM_PORT}" \
        --output "${COLLECTOR_OUTPUT_IN_CONTAINER}" \
        --detail-output "${COLLECTOR_DETAIL_IN_CONTAINER}" \
        --segment-file "${SEGMENT_IN_CONTAINER}"

if [[ "${DRY_RUN}" != "1" ]]; then
    # Avoid ZMQ slow-joiner loss on the first prefill iteration.
    sleep 1
fi

start_nsys_worker_collection

send_sweep_workloads() {
    local context_count context_requests
    local repeat batch_size
    local sweep_rc=0
    rm -f "${REQUEST_WORKLOAD_CSV}"

    if phase_enabled context; then
        context_count="$(csv_count "${CONTEXT_ISL_VALUES}")"
        context_requests=$((context_count * CONTEXT_REPEATS))
        send_request_workload \
            "context" \
            "request-driver-context" \
            "${context_requests}" \
            "${CONTEXT_CONCURRENCY}" \
            "${REQUEST_WORKLOAD_IN_CONTAINER}" \
            "${CONTEXT_ISL_VALUES}" \
            "${CONTEXT_OSL}" \
            1 \
            0 \
            0 \
            "sweep" || sweep_rc=$?
    fi

    if phase_enabled decode; then
        for repeat in $(seq 1 "${DECODE_REPEATS}"); do
            IFS=',' read -ra decode_batches <<< "${DECODE_BATCH_SIZES}"
            for batch_size in "${decode_batches[@]}"; do
                batch_size="${batch_size//[[:space:]]/}"
                if [[ -z "${batch_size}" ]]; then
                    continue
                fi
                send_request_workload \
                    "decode_b${batch_size}" \
                    "request-driver-decode-b${batch_size}-r${repeat}" \
                    "${batch_size}" \
                    "${batch_size}" \
                    "${REQUEST_WORKLOAD_IN_CONTAINER}" \
                    "${DECODE_PAST_KV}" \
                    "${DECODE_OSL}" \
                    1 \
                    0 \
                    0 \
                    "sweep" || sweep_rc=$?
            done
        done
    fi

    if phase_enabled mixed; then
        for repeat in $(seq 1 "${MIX_REPEATS}"); do
            send_request_workload \
                "mixed" \
                "request-driver-mixed-r${repeat}" \
                "${MIX_REQUESTS}" \
                "${MIX_CONCURRENCY}" \
                "${REQUEST_WORKLOAD_IN_CONTAINER}" \
                "${MIX_ISL_VALUES}" \
                "${MIX_OSL_VALUES}" \
                1 \
                0 \
                0 \
                "sweep" || sweep_rc=$?
        done
    fi

    return "${sweep_rc}"
}

send_real_workload() {
    local append_workload="${1:-0}"
    if [[ "${append_workload}" != "1" ]]; then
        rm -f "${REQUEST_WORKLOAD_CSV}"
    fi
    send_request_workload \
        "real" \
        "request-driver-real" \
        "${REAL_WORKLOAD_REQUESTS}" \
        "${REAL_WORKLOAD_CONCURRENCY}" \
        "${REQUEST_WORKLOAD_IN_CONTAINER}" \
        "" \
        "" \
        "${append_workload}" \
        2000000000 \
        1 \
        "real"
}

REQUEST_SEND_RC=0
if [[ "${SKIP_REQUESTS}" == "1" ]]; then
    log "Skipping sample requests. Collector is running; send traffic to http://127.0.0.1:${HTTP_PORT}."
elif [[ "${WORKLOAD_PLAN}" == "sweep" ]]; then
    if [[ "${RUN_SWEEP}" == "1" ]]; then
        send_sweep_workloads || REQUEST_SEND_RC=$?
    fi
    if [[ "${RUN_REAL_WORKLOAD}" == "1" && "${REQUEST_SEND_RC}" == "0" ]]; then
        send_real_workload "${RUN_SWEEP}" || REQUEST_SEND_RC=$?
    fi
else
    send_request_workload \
        "measured" \
        "request-driver" \
        "${REQUESTS}" \
        "${CONCURRENCY}" \
        "${REQUEST_WORKLOAD_IN_CONTAINER}" \
        "${ISL_VALUES}" \
        "${OSL_VALUES}" \
        0 \
        0 \
        0 \
        "legacy" || REQUEST_SEND_RC=$?
fi

stop_nsys_worker_collection

if [[ "${REQUEST_SEND_RC}" != "0" ]]; then
    log "Request driver exited with ${REQUEST_SEND_RC}; preserving collector output before exiting."
fi

if [[ "${DRY_RUN}" != "1" ]]; then
    log "Collecting for ${POST_REQUEST_COLLECT_SECONDS}s after requests"
    sleep "${POST_REQUEST_COLLECT_SECONDS}"

    if [[ "${KEEP_RUNNING}" != "1" ]] && container_exists "${COLLECTOR_NAME}"; then
        docker stop -t 2 "${COLLECTOR_NAME}" >/dev/null || true
    fi
    if [[ "${NSYS_PROFILE_WORKER}" == "1" && "${KEEP_RUNNING}" != "1" ]] && container_exists "${WORKER_NAME}"; then
        log "Stopping profiled worker container to flush Nsight report"
        docker stop -t 60 "${WORKER_NAME}" >/dev/null || true
        if compgen -G "${RUN_DIR}/nsys/fpm_worker*.nsys-rep" >/dev/null; then
            log "Nsight worker report(s):"
            find "${RUN_DIR}/nsys" -maxdepth 1 -type f -name 'fpm_worker*.nsys-rep' -print >&2
        else
            log "WARNING: no Nsight worker report found under ${RUN_DIR}/nsys"
        fi
    fi

    if [[ -f "${COLLECTOR_OUTPUT_CSV}" ]]; then
        if [[ "${COLLECTOR_OUTPUT_CSV}" != "${OUTPUT_CSV}" ]]; then
            cp "${COLLECTOR_OUTPUT_CSV}" "${OUTPUT_CSV}"
        fi
        if [[ -f "${COLLECTOR_DETAIL_CSV}" && "${COLLECTOR_DETAIL_CSV}" != "${DETAIL_OUTPUT_CSV}" ]]; then
            cp "${COLLECTOR_DETAIL_CSV}" "${DETAIL_OUTPUT_CSV}"
            log "FPM detail written to ${DETAIL_OUTPUT_CSV}"
        fi
        if [[ -f "${COLLECTOR_DETAIL_CSV}" ]]; then
            python3 "${SCRIPT_DIR}/summarize_fpm.py" \
                --detail "${COLLECTOR_DETAIL_CSV}" \
                --output "${COLLECTOR_PHASE_CSV}" >/dev/null
            if [[ "${COLLECTOR_PHASE_CSV}" != "${PHASE_OUTPUT_CSV}" ]]; then
                cp "${COLLECTOR_PHASE_CSV}" "${PHASE_OUTPUT_CSV}"
            fi
            log "FPM phase rows written to ${PHASE_OUTPUT_CSV}"
        fi
        if [[ -f "${REQUEST_WORKLOAD_CSV}" && "${REQUEST_WORKLOAD_CSV}" != "${WORKLOAD_OUTPUT_CSV}" ]]; then
            cp "${REQUEST_WORKLOAD_CSV}" "${WORKLOAD_OUTPUT_CSV}"
            log "Request workload written to ${WORKLOAD_OUTPUT_CSV}"
        fi
        if [[ -f "${WARMUP_WORKLOAD_CSV}" && "${WARMUP_WORKLOAD_CSV}" != "${WARMUP_WORKLOAD_OUTPUT_CSV}" ]]; then
            cp "${WARMUP_WORKLOAD_CSV}" "${WARMUP_WORKLOAD_OUTPUT_CSV}"
            log "Warmup workload written to ${WARMUP_WORKLOAD_OUTPUT_CSV}"
        fi
        if [[ -f "${RUN_METADATA_JSON}" && "${RUN_METADATA_JSON}" != "${METADATA_OUTPUT_JSON}" ]]; then
            cp "${RUN_METADATA_JSON}" "${METADATA_OUTPUT_JSON}"
            log "vLLM metadata written to ${METADATA_OUTPUT_JSON}"
        fi
        if [[ -f "${RUN_EFFECTIVE_CONFIG_JSON}" && "${RUN_EFFECTIVE_CONFIG_JSON}" != "${EFFECTIVE_CONFIG_OUTPUT_JSON}" ]]; then
            cp "${RUN_EFFECTIVE_CONFIG_JSON}" "${EFFECTIVE_CONFIG_OUTPUT_JSON}"
            log "vLLM effective config written to ${EFFECTIVE_CONFIG_OUTPUT_JSON}"
        fi
        rows=$(tail -n +2 "${OUTPUT_CSV}" | wc -l | tr -d ' ')
        log "Collected ${rows} FPM rows at ${OUTPUT_CSV}"
        if [[ -f "${PHASE_OUTPUT_CSV}" ]]; then
            log "Phase row counts:"
            awk -F, 'NR > 1 {count[$1]++} END {for (phase in count) print "  " phase ": " count[phase]}' \
                "${PHASE_OUTPUT_CSV}" >&2
        fi
        if [[ "${rows}" == "0" ]]; then
            log "No FPM rows were collected. Worker log tail:"
            docker logs --tail=120 "${WORKER_NAME}" >&2 || true
            exit 2
        fi
        log "First rows:"
        sed -n '1,12p' "${OUTPUT_CSV}" >&2
    else
        die "collector did not create ${COLLECTOR_OUTPUT_CSV}"
    fi
fi

if [[ "${REQUEST_SEND_RC}" != "0" ]]; then
    exit "${REQUEST_SEND_RC}"
fi

log "Done"
