#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# In-image (SLURM, inside the Dynamo vLLM 0.20.1 container) launcher for the FPM
# expert-routing capture. Extracts the CORE launch logic from
# ../collect_fpm_metrics.sh and runs the pieces as DIRECT PROCESSES on this node
# (no docker run / -v mount — we are already inside the image). Reproduces the
# golden qwen36 tp4_ep4 topology and injects the routing capture via
# `export PYTHONPATH=<inject_dir>:$PYTHONPATH`.
#
# Usage:
#   FPM_ROUTING_STAGE=A RUN_DIR=/workspace/.../runA bash run_routing_stack.sh
#   FPM_ROUTING_STAGE=B RUN_DIR=/workspace/.../runB REAL_WORKLOAD=1 bash run_routing_stack.sh
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FPM_DIR="$(cd "${HERE}/.." && pwd)"            # .../fpm_ground_truth
INJECT_DIR="${HERE}/inject"

MODEL="${MODEL:-/workspace/models/Qwen3.6-35B-A3B}"
STAGE="${FPM_ROUTING_STAGE:?set FPM_ROUTING_STAGE=A|B}"
RUN_DIR="${RUN_DIR:-/workspace/repo/aiconfigurator/routing_runs/run_${STAGE}_$(date +%Y%m%d_%H%M%S)}"

HTTP_PORT="${HTTP_PORT:-8000}"
SYSTEM_PORT="${SYSTEM_PORT:-8081}"
FPM_PORT="${FPM_PORT:-20380}"
TP_SIZE="${TP_SIZE:-4}"
GPUS="${GPUS:-0,1,2,3}"

# Workload knobs (Stage A small; Stage B full).
REQUESTS="${REQUESTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
ISL_VALUES="${ISL_VALUES:-1024,2048,4096}"
OSL_VALUES="${OSL_VALUES:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REAL_WORKLOAD="${REAL_WORKLOAD:-0}"
REAL_WORKLOAD_REQUESTS="${REAL_WORKLOAD_REQUESTS:-128}"
REAL_WORKLOAD_CONCURRENCY="${REAL_WORKLOAD_CONCURRENCY:-32}"
START_TIMEOUT="${START_TIMEOUT:-1200}"
POST_COLLECT_SECONDS="${POST_COLLECT_SECONDS:-5}"

mkdir -p "${RUN_DIR}/discovery" "${RUN_DIR}/routing" "${RUN_DIR}/logs"
ROUTING_OUT="${RUN_DIR}/routing"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

FRONTEND_PID=""; WORKER_PID=""; COLLECTOR_PID=""
cleanup() {
    local rc=$?
    log "cleanup: terminating processes (graceful SIGTERM so worker atexit flush runs)"
    [[ -n "${COLLECTOR_PID}" ]] && kill -INT "${COLLECTOR_PID}" 2>/dev/null || true
    if [[ -n "${WORKER_PID}" ]]; then
        kill -TERM "${WORKER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${WORKER_PID}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${WORKER_PID}" 2>/dev/null || true
    fi
    [[ -n "${FRONTEND_PID}" ]] && kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    wait 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT

# ----- shared Dynamo env (mirrors collect_fpm_metrics.sh DOCKER_ENV) -----
export DYN_DISCOVERY_BACKEND=file
export DYN_REQUEST_PLANE=tcp
export DYN_EVENT_PLANE=zmq
export DYN_FILE_KV="${RUN_DIR}/discovery"
export DYN_NAMESPACE=dynamo

# ----- frontend -----
log "starting frontend (http :${HTTP_PORT})"
DYN_HTTP_PORT="${HTTP_PORT}" \
    python3 -m dynamo.frontend \
        --http-port "${HTTP_PORT}" \
        --discovery-backend file --request-plane tcp --event-plane zmq \
        >"${RUN_DIR}/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

# ----- worker (reproduces golden tp4_ep4 engine_args) -----
log "starting vLLM worker (TP=${TP_SIZE}, EP via --enable-expert-parallel, stage=${STAGE})"
CUDA_VISIBLE_DEVICES="${GPUS}" \
DYN_FORWARDPASS_METRIC_PORT="${FPM_PORT}" \
DYN_SYSTEM_PORT="${SYSTEM_PORT}" \
TILELANG_CACHE_DIR="${RUN_DIR}/tilelang" \
TILELANG_TMP_DIR="${RUN_DIR}/tilelang/tmp" \
PYTHONPATH="${INJECT_DIR}:${PYTHONPATH:-}" \
FPM_ROUTING_STAGE="${STAGE}" \
FPM_ROUTING_OUT="${ROUTING_OUT}" \
FPM_ROUTING_FLUSH_EVERY="${FPM_ROUTING_FLUSH_EVERY:-512}" \
    python3 -m dynamo.vllm \
        --model "${MODEL}" \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size "${TP_SIZE}" \
        --skip-mm-profiling \
        --limit-mm-per-prompt '{"image":0,"video":0}' \
        --generation-config vllm \
        --enable-expert-parallel \
        --discovery-backend file --request-plane tcp --event-plane zmq \
        --kv-events-config '{"enable_kv_cache_events": false}' \
        >"${RUN_DIR}/logs/worker.log" 2>&1 &
WORKER_PID=$!

log "waiting for frontend /health (timeout ${START_TIMEOUT}s)"
python3 "${FPM_DIR}/wait_http.py" --url "http://127.0.0.1:${HTTP_PORT}/health" --timeout "${START_TIMEOUT}" >/dev/null
log "waiting for model registration (worker loads ${MODEL}; this can take minutes)"
# Poll /v1/models until a model actually registers (worker finished loading +
# CUDA-graph capture). A bare 200 is not enough: data[] is empty until ready.
SERVED_MODEL="$(python3 - "$HTTP_PORT" "$START_TIMEOUT" "$WORKER_PID" <<'PY'
import json, os, sys, time, urllib.request
port, timeout, wpid = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    # bail early if the worker process died
    try:
        os.kill(wpid, 0)
    except OSError:
        print("WORKER_DEAD", file=sys.stderr); sys.exit(2)
    try:
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10))
        data = d.get("data") or []
        if data:
            print(data[0]["id"]); sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
print("TIMEOUT", file=sys.stderr); sys.exit(1)
PY
)"
log "served model = ${SERVED_MODEL}"

# ----- collector (raw FPM ZMQ SUB) -----
log "starting FPM collector on :${FPM_PORT}"
python3 "${FPM_DIR}/fpm_collect.py" \
    --port "${FPM_PORT}" \
    --output "${RUN_DIR}/fpm_metrics.csv" \
    --detail-output "${RUN_DIR}/fpm_metrics_detail.csv" \
    --segment-file "${RUN_DIR}/fpm_segment.txt" \
    >"${RUN_DIR}/logs/collector.log" 2>&1 &
COLLECTOR_PID=$!
sleep 2   # avoid ZMQ slow-joiner loss on first prefill

drive() {
    local seg="$1"; shift
    printf '%s\n' "${seg}" > "${RUN_DIR}/fpm_segment.txt"
    python3 "${FPM_DIR}/send_requests.py" "$@" \
        --url "http://127.0.0.1:${HTTP_PORT}" --model "${SERVED_MODEL}" \
        --workload-output "${RUN_DIR}/request_workload.csv" --workload-label "${seg}" \
        --endpoint completions --ignore-eos --append-workload \
        --timeout 1200 --retries 5 --retry-backoff 2 --allow-failures "${ALLOW_FAILURES:-100000}"
}

log "driving workload (stage ${STAGE})"
if [[ "${REAL_WORKLOAD}" == "1" ]]; then
    drive real \
        --requests "${REAL_WORKLOAD_REQUESTS}" --concurrency "${REAL_WORKLOAD_CONCURRENCY}" \
        --max-tokens "${MAX_TOKENS}" --real-workload --real-workload-shape-source synthetic \
        --real-workload-isl-min "${RW_ISL_MIN:-128}" --real-workload-isl-max "${RW_ISL_MAX:-6144}" \
        --real-workload-isl-mean "${RW_ISL_MEAN:-3000}" \
        --real-workload-osl-min "${RW_OSL_MIN:-64}" --real-workload-osl-max "${RW_OSL_MAX:-512}" \
        --real-workload-osl-mean "${RW_OSL_MEAN:-256}"
else
    drive mixed \
        --requests "${REQUESTS}" --concurrency "${CONCURRENCY}" --max-tokens "${MAX_TOKENS}" \
        --vary-isl-osl --isl-values "${ISL_VALUES}" --osl-values "${OSL_VALUES}"
fi

log "workload done; draining collector ${POST_COLLECT_SECONDS}s"
sleep "${POST_COLLECT_SECONDS}"
log "stack run complete; RUN_DIR=${RUN_DIR}"
echo "${RUN_DIR}" > "${RUN_DIR}/RUN_DIR.txt"
