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
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
GPUS="${GPUS:-all}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-0}"
FILE_DISCOVERY_TOUCH_SECONDS="${FILE_DISCOVERY_TOUCH_SECONDS:-2}"

REQUESTS="${REQUESTS:-16}"
CONCURRENCY="${CONCURRENCY:-32}"
WORKLOAD_PLAN="${WORKLOAD_PLAN:-sweep}"
MEASURED_PHASES="${MEASURED_PHASES:-context,decode,mixed}"
CONTEXT_ISL_VALUES="${CONTEXT_ISL_VALUES:-1,64,256,1024,2048,4096}"
CONTEXT_OSL="${CONTEXT_OSL:-1}"
CONTEXT_REPEATS="${CONTEXT_REPEATS:-1}"
CONTEXT_CONCURRENCY="${CONTEXT_CONCURRENCY:-1}"
DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1,2,4,8,16,32,64}"
DECODE_PAST_KV="${DECODE_PAST_KV:-1024}"
DECODE_OSL="${DECODE_OSL:-32}"
DECODE_REPEATS="${DECODE_REPEATS:-1}"
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
PROMPT_TOKEN_SEED="${PROMPT_TOKEN_SEED:-0}"
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
OUTPUT_CSV="${OUTPUT_CSV:-${RUN_DIR}/fpm_metrics.csv}"
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
  --max-model-len N             vLLM --max-model-len (default: ${MAX_MODEL_LEN})
  --max-num-seqs N              vLLM --max-num-seqs (default: ${MAX_NUM_SEQS})
  --gpu-memory-utilization X    vLLM --gpu-memory-utilization (default: ${GPU_MEMORY_UTILIZATION})
  --gpus SPEC                   Docker --gpus value for worker (default: ${GPUS})
  --enforce-eager               Force vLLM eager mode instead of standard compile/graph behavior
  --disable-prefix-caching      Pass --no-enable-prefix-caching instead of standard vLLM behavior
  --file-discovery-touch-seconds N
                                Host-side mtime refresh interval for local file discovery (default: ${FILE_DISCOVERY_TOUCH_SECONDS})
  --workload-plan sweep|legacy  Measured request plan (default: ${WORKLOAD_PLAN})
  --measured-phases CSV         Sweep phases to send: context,decode,mixed (default: ${MEASURED_PHASES})
  --context-isl-values CSV      Context sweep target ISLs (default: ${CONTEXT_ISL_VALUES})
  --context-osl N               Context sweep max_tokens (default: ${CONTEXT_OSL})
  --context-repeats N           Context sweep repetitions (default: ${CONTEXT_REPEATS})
  --context-concurrency N       Context sweep concurrency (default: ${CONTEXT_CONCURRENCY})
  --decode-batch-sizes CSV      Decode sweep batch sizes/concurrency values (default: ${DECODE_BATCH_SIZES})
  --decode-past-kv N            Decode sweep prompt length / target KV (default: ${DECODE_PAST_KV})
  --decode-osl N                Decode sweep max_tokens (default: ${DECODE_OSL})
  --decode-repeats N            Decode sweep repetitions per batch size (default: ${DECODE_REPEATS})
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
  --prompt-token-seed N         Seed for deterministic random prompt token IDs (default: ${PROMPT_TOKEN_SEED})
  --request-retries N           Retries per request for transient HTTP errors (default: ${REQUEST_RETRIES})
  --request-retry-backoff N     Base seconds between request retries (default: ${REQUEST_RETRY_BACKOFF_SECONDS})
  --request-allow-failures N    Continue if at most N requests fail after retries (default: ${REQUEST_ALLOW_FAILURES})
  --output PATH                 CSV output path (default: ${OUTPUT_CSV})
  --detail-output PATH          Full FPM detail CSV path (default: OUTPUT_detail.csv)
  --phase-output PATH           Classified step CSV path (default: OUTPUT_phase.csv)
  --workload-output PATH        Request workload CSV path (default: OUTPUT_workload.csv)
  --warmup-workload-output PATH Warmup workload CSV path (default: OUTPUT_warmup_workload.csv)
  --metadata-output PATH        Requested/effective vLLM metadata JSON (default: OUTPUT_metadata.json)
  --effective-config-output PATH
                                Effective vLLM config JSON (default: OUTPUT_effective_vllm_config.json)
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
  WORKLOAD_PLAN, MEASURED_PHASES, CONTEXT_ISL_VALUES, CONTEXT_REPEATS,
  DECODE_BATCH_SIZES, DECODE_PAST_KV, DECODE_OSL, MIX_ISL_VALUES,
  MIX_OSL_VALUES, REQUEST_RETRIES, REQUEST_RETRY_BACKOFF_SECONDS,
  REQUEST_ALLOW_FAILURES,
  VARY_ISL_OSL, REQUEST_ENDPOINT, ISL_MIN, ISL_MAX, OSL_MIN, OSL_MAX,
  PROMPT_TOKEN_SEED, ISL_VALUES, OSL_VALUES, DYN_HTTP_PORT, DYN_SYSTEM_PORT,
  DYN_FORWARDPASS_METRIC_PORT, HF_HOME.

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
  Generated requests always use deterministic random prompt token IDs through
  the completions API.

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

phase_enabled() {
    local phase="$1"
    [[ ",${MEASURED_PHASES}," == *",${phase},"* ]]
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
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --enforce-eager) ENFORCE_EAGER=1; shift ;;
        --no-enforce-eager) ENFORCE_EAGER=0; shift ;;
        --disable-prefix-caching) DISABLE_PREFIX_CACHING=1; shift ;;
        --enable-prefix-caching) DISABLE_PREFIX_CACHING=0; shift ;;
        --file-discovery-touch-seconds) FILE_DISCOVERY_TOUCH_SECONDS="$2"; shift 2 ;;
        --workload-plan) WORKLOAD_PLAN="$2"; shift 2 ;;
        --measured-phases) MEASURED_PHASES="$2"; shift 2 ;;
        --context-isl-values) CONTEXT_ISL_VALUES="$2"; shift 2 ;;
        --context-osl) CONTEXT_OSL="$2"; shift 2 ;;
        --context-repeats) CONTEXT_REPEATS="$2"; shift 2 ;;
        --context-concurrency) CONTEXT_CONCURRENCY="$2"; shift 2 ;;
        --decode-batch-sizes) DECODE_BATCH_SIZES="$2"; shift 2 ;;
        --decode-past-kv) DECODE_PAST_KV="$2"; shift 2 ;;
        --decode-osl) DECODE_OSL="$2"; shift 2 ;;
        --decode-repeats) DECODE_REPEATS="$2"; shift 2 ;;
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

if [[ "${OUTPUT_CSV}" != /* ]]; then
    OUTPUT_CSV="${PWD}/${OUTPUT_CSV}"
fi
if [[ -z "${DETAIL_OUTPUT_CSV}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        DETAIL_OUTPUT_CSV="${OUTPUT_CSV%.csv}_detail.csv"
    else
        DETAIL_OUTPUT_CSV="${OUTPUT_CSV}.detail.csv"
    fi
elif [[ "${DETAIL_OUTPUT_CSV}" != /* ]]; then
    DETAIL_OUTPUT_CSV="${PWD}/${DETAIL_OUTPUT_CSV}"
fi
if [[ -z "${PHASE_OUTPUT_CSV}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        PHASE_OUTPUT_CSV="${OUTPUT_CSV%.csv}_phase.csv"
    else
        PHASE_OUTPUT_CSV="${OUTPUT_CSV}.phase.csv"
    fi
elif [[ "${PHASE_OUTPUT_CSV}" != /* ]]; then
    PHASE_OUTPUT_CSV="${PWD}/${PHASE_OUTPUT_CSV}"
fi
if [[ -z "${WORKLOAD_OUTPUT_CSV}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        WORKLOAD_OUTPUT_CSV="${OUTPUT_CSV%.csv}_workload.csv"
    else
        WORKLOAD_OUTPUT_CSV="${OUTPUT_CSV}.workload.csv"
    fi
elif [[ "${WORKLOAD_OUTPUT_CSV}" != /* ]]; then
    WORKLOAD_OUTPUT_CSV="${PWD}/${WORKLOAD_OUTPUT_CSV}"
fi
if [[ -z "${WARMUP_WORKLOAD_OUTPUT_CSV}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        WARMUP_WORKLOAD_OUTPUT_CSV="${OUTPUT_CSV%.csv}_warmup_workload.csv"
    else
        WARMUP_WORKLOAD_OUTPUT_CSV="${OUTPUT_CSV}.warmup_workload.csv"
    fi
elif [[ "${WARMUP_WORKLOAD_OUTPUT_CSV}" != /* ]]; then
    WARMUP_WORKLOAD_OUTPUT_CSV="${PWD}/${WARMUP_WORKLOAD_OUTPUT_CSV}"
fi
if [[ -z "${METADATA_OUTPUT_JSON}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        METADATA_OUTPUT_JSON="${OUTPUT_CSV%.csv}_metadata.json"
    else
        METADATA_OUTPUT_JSON="${OUTPUT_CSV}.metadata.json"
    fi
elif [[ "${METADATA_OUTPUT_JSON}" != /* ]]; then
    METADATA_OUTPUT_JSON="${PWD}/${METADATA_OUTPUT_JSON}"
fi
if [[ -z "${EFFECTIVE_CONFIG_OUTPUT_JSON}" ]]; then
    if [[ "${OUTPUT_CSV}" == *.csv ]]; then
        EFFECTIVE_CONFIG_OUTPUT_JSON="${OUTPUT_CSV%.csv}_effective_vllm_config.json"
    else
        EFFECTIVE_CONFIG_OUTPUT_JSON="${OUTPUT_CSV}.effective_vllm_config.json"
    fi
elif [[ "${EFFECTIVE_CONFIG_OUTPUT_JSON}" != /* ]]; then
    EFFECTIVE_CONFIG_OUTPUT_JSON="${PWD}/${EFFECTIVE_CONFIG_OUTPUT_JSON}"
fi
if [[ "${RUN_DIR}" != /* ]]; then
    RUN_DIR="${PWD}/${RUN_DIR}"
fi
if [[ -z "${HF_HOME_HOST}" ]]; then
    HF_HOME_HOST="${RUN_DIR}/hf-home"
fi
if [[ "${HF_HOME_HOST}" != /* ]]; then
    HF_HOME_HOST="${PWD}/${HF_HOME_HOST}"
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
NSYS_WORKER_OUTPUT_BASE="${RUN_DIR}/nsys/fpm_worker"
NSYS_WORKER_OUTPUT_IN_CONTAINER="/work/nsys/fpm_worker"

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
NSYS_DOCKER_MOUNTS=()

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
    "${HF_HOME_HOST}"
chmod a+rwx "${RUN_DIR}" "${RUN_DIR}/discovery" "${HF_HOME_HOST}"

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

case "${WORKLOAD_PLAN}" in
    sweep|legacy) ;;
    *) die "--workload-plan must be 'sweep' or 'legacy'" ;;
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

if [[ "${WORKLOAD_PLAN}" == "sweep" ]]; then
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
        if (( context_max_isl + CONTEXT_OSL > MAX_MODEL_LEN )); then
            die "context max ISL + OSL (${context_max_isl} + ${CONTEXT_OSL}) exceeds --max-model-len ${MAX_MODEL_LEN}"
        fi
    fi
    if phase_enabled decode; then
        decode_max_batch=$(csv_max "${DECODE_BATCH_SIZES}")
        if (( decode_max_batch < 1 )); then
            die "decode batch-size list must not be empty"
        fi
        if (( decode_max_batch > MAX_NUM_SEQS )); then
            die "decode max batch size ${decode_max_batch} exceeds --max-num-seqs ${MAX_NUM_SEQS}"
        fi
        if (( DECODE_PAST_KV + DECODE_OSL > MAX_MODEL_LEN )); then
            die "decode past_kv + OSL (${DECODE_PAST_KV} + ${DECODE_OSL}) exceeds --max-model-len ${MAX_MODEL_LEN}"
        fi
    fi
    if phase_enabled mixed; then
        mixed_max_isl=$(csv_max "${MIX_ISL_VALUES}")
        mixed_max_osl=$(csv_max "${MIX_OSL_VALUES}")
        if (( mixed_max_isl < 1 || mixed_max_osl < 1 )); then
            die "mixed ISL/OSL lists must not be empty"
        fi
        if (( MIX_CONCURRENCY > MAX_NUM_SEQS )); then
            die "mixed concurrency ${MIX_CONCURRENCY} exceeds --max-num-seqs ${MAX_NUM_SEQS}"
        fi
        if (( mixed_max_isl + mixed_max_osl > MAX_MODEL_LEN )); then
            die "mixed max ISL + OSL (${mixed_max_isl} + ${mixed_max_osl}) exceeds --max-model-len ${MAX_MODEL_LEN}"
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
    if (( ISL_MAX + OSL_MAX > MAX_MODEL_LEN )); then
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
        --model "${MODEL}"
        --max-model-len "${MAX_MODEL_LEN}"
        --max-num-seqs "${MAX_NUM_SEQS}"
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    )
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
        -e "HF_HOME=/work/hf-home" \
        -e "HF_HUB_CACHE=/work/hf-home/hub" \
        -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
        -e "HF_TOKEN=${HF_TOKEN:-}" \
        "${IMAGE}" \
        python3 /work/vllm_deployment.py snapshot-effective \
            --args-json "${VLLM_DEPLOYMENT_ARGS_JSON}" \
            --output "${RUN_EFFECTIVE_CONFIG_IN_CONTAINER}" || snapshot_rc=$?
    if [[ "${snapshot_rc}" != "0" ]]; then
        log "WARNING: failed to snapshot effective vLLM config; metadata will contain requested args only"
    fi
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
    local prompt_token_seed="$((PROMPT_TOKEN_SEED + seed_offset))"
    local request_rc=0

    local request_driver_cmd=(
        python3 /work/send_requests.py
        --url "http://127.0.0.1:${HTTP_PORT}"
        --model "${MODEL}"
        --requests "${request_count}"
        --concurrency "${concurrency_count}"
        --max-tokens "${MAX_TOKENS}"
        --prompt-token-seed "${prompt_token_seed}"
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
    if [[ "${append_workload}" == "1" ]]; then
        request_driver_cmd+=(--append-workload)
    fi
    if [[ "${VARY_ISL_OSL}" == "1" || "${WORKLOAD_PLAN}" == "sweep" ]]; then
        request_driver_cmd+=(--vary-isl-osl)
        log "Sending ${request_count} ${phase} variable ISL/OSL requests at concurrency ${concurrency_count} (ISL ${ISL_MIN}..${ISL_MAX}, OSL ${OSL_MIN}..${OSL_MAX})"
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
        -e "HF_HOME=/work/hf-home" \
        -e "HF_HUB_CACHE=/work/hf-home/hub" \
        -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
        -e "HF_TOKEN=${HF_TOKEN:-}" \
        "${IMAGE}" \
        "${request_driver_cmd[@]}" || request_rc=$?

    REQUEST_INDEX_OFFSET=$((REQUEST_INDEX_OFFSET + request_count))
    return "${request_rc}"
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
log "Workload plan: ${WORKLOAD_PLAN} (${MEASURED_PHASES})"
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
    "${NSYS_DOCKER_MOUNTS[@]}" \
    "${DOCKER_ENV[@]}" \
    -e "DYN_FORWARDPASS_METRIC_PORT=${FPM_PORT}" \
    -e "DYN_SYSTEM_PORT=${SYSTEM_PORT}" \
    -e "HF_HOME=/work/hf-home" \
    -e "HF_HUB_CACHE=/work/hf-home/hub" \
    -e "TRANSFORMERS_CACHE=/work/hf-home/transformers" \
    -e "HF_TOKEN=${HF_TOKEN:-}" \
    "${IMAGE}" \
    "${WORKER_CONTAINER_CMD[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
    log "Waiting for frontend health"
    docker run --rm --network host -v "${RUN_DIR}:/work" "${IMAGE}" \
        python3 /work/wait_http.py \
            --url "http://127.0.0.1:${HTTP_PORT}/health" \
            --timeout "${START_TIMEOUT_SECONDS}" >/dev/null

    log "Waiting for model registration (${MODEL})"
    docker run --rm --network host -v "${RUN_DIR}:/work" "${IMAGE}" \
        python3 /work/wait_http.py \
            --url "http://127.0.0.1:${HTTP_PORT}/v1/models" \
            --contains "${MODEL}" \
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
        1000000000
    if [[ "${DRY_RUN}" != "1" ]]; then
        log "Waiting ${POST_WARMUP_SECONDS}s after warmup before starting FPM collector"
        sleep "${POST_WARMUP_SECONDS}"
    fi
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
        --detail-output "${COLLECTOR_DETAIL_IN_CONTAINER}"

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
            1 || sweep_rc=$?
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
                    1 || sweep_rc=$?
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
                1 || sweep_rc=$?
        done
    fi

    return "${sweep_rc}"
}

REQUEST_SEND_RC=0
if [[ "${SKIP_REQUESTS}" == "1" ]]; then
    log "Skipping sample requests. Collector is running; send traffic to http://127.0.0.1:${HTTP_PORT}."
elif [[ "${WORKLOAD_PLAN}" == "sweep" ]]; then
    send_sweep_workloads || REQUEST_SEND_RC=$?
else
    send_request_workload \
        "measured" \
        "request-driver" \
        "${REQUESTS}" \
        "${CONCURRENCY}" \
        "${REQUEST_WORKLOAD_IN_CONTAINER}" \
        "${ISL_VALUES}" \
        "${OSL_VALUES}" \
        0 || REQUEST_SEND_RC=$?
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
