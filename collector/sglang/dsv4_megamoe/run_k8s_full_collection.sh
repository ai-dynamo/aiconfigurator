#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

NAMESPACE="${NAMESPACE:-default}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
JOB_PREFIX="${JOB_PREFIX:-aic-dsv4-megamoe}"
JOB_TAG="${JOB_PREFIX}-${RUN_ID}"
SYSTEM_NAME="${SYSTEM_NAME:-gb200}"
MODEL_CONFIG="${MODEL_CONFIG:-dsv4_pro}"
LOCAL_REPO="${LOCAL_REPO:-${AIC_ROOT}}"
PVC_NAME="${PVC_NAME:-shared-model-cache}"
PVC_MOUNT="${PVC_MOUNT:-/mnt/shared}"
SYNC_IMAGE="${SYNC_IMAGE:-ubuntu:22.04}"
REMOTE_BASE="${REMOTE_BASE:-${PVC_MOUNT}/aic_dsv4_megamoe_workspace/${JOB_TAG}}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${REMOTE_BASE}/aiconfigurator}"
REMOTE_OUTPUT_PATH="${REMOTE_OUTPUT_PATH:-${PVC_MOUNT}/aic_dsv4_megamoe_results/${JOB_TAG}}"
LOCAL_RESULT_DIR="${LOCAL_RESULT_DIR:-${LOCAL_REPO}/artifacts/${JOB_TAG}}"
PERF_FILE="${PERF_FILE:-dsv4_megamoe_module_perf.txt}"

EP_SIZES="${EP_SIZES:-4,8,16,32}"
PREFILL_EP_SIZES="${PREFILL_EP_SIZES:-${EP_SIZES}}"
DECODE_EP_SIZES="${DECODE_EP_SIZES:-${EP_SIZES}}"
PREFILL_TOKENS="${PREFILL_TOKENS:-1024,2048,4096,8192,16384,32768}"
DECODE_TOKENS="${DECODE_TOKENS:-1,2,4,8,16,32,64,128,256,512}"
PREFILL_NUM_MAX_TOKENS_PER_RANK="${PREFILL_NUM_MAX_TOKENS_PER_RANK:-32768}"
DECODE_NUM_MAX_TOKENS_PER_RANK="${DECODE_NUM_MAX_TOKENS_PER_RANK:-512}"
DISTRIBUTIONS="${DISTRIBUTIONS:-balanced,power_law_1.01,power_law_1.2,power_law_sampled_1.9}"
SOURCE_POLICY="${SOURCE_POLICY:-random}"
ROUTING_SEED="${ROUTING_SEED:-0}"
ROUTING_SEEDS="${ROUTING_SEEDS:-}"
PHASE_ORDER="${PHASE_ORDER:-context,generation}"
PRE_DISPATCH="${PRE_DISPATCH:-sglang_jit}"
INCLUDE_ROUTED_SCALE="${INCLUDE_ROUTED_SCALE:-1}"
RENORMALIZE_TOPK_WEIGHTS="${RENORMALIZE_TOPK_WEIGHTS:-1}"
NUM_WARMUP="${NUM_WARMUP:-5}"
NUM_ITERATIONS="${NUM_ITERATIONS:-20}"
CAP_POLICY="${CAP_POLICY:-case_tokens}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_SGLANG_VERSION="${TARGET_SGLANG_VERSION:-0.5.10}"
ALLOW_VERSION_MISMATCH="${ALLOW_VERSION_MISMATCH:-1}"

IMAGE="${IMAGE:-}"
IMAGE_PULL_POLICY="${IMAGE_PULL_POLICY:-IfNotPresent}"
PRIORITY_CLASS_NAME="${PRIORITY_CLASS_NAME:-}"
COMPUTE_DOMAIN="${COMPUTE_DOMAIN:-1}"
IPC_LOCK="${IPC_LOCK:-1}"
JOB_TIMEOUT_SECONDS="${JOB_TIMEOUT_SECONDS:-5400}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

if [[ -z "${NODE_SELECTOR:-}" ]]; then
  case "${SYSTEM_NAME^^}" in
    GB200|GB300)
      NODE_SELECTOR="kubernetes.io/arch=arm64,nvidia.com/gpu.product=NVIDIA-${SYSTEM_NAME^^}"
      ;;
    B200|B200_SXM|B300|B300_SXM)
      gpu_product="${SYSTEM_NAME%%_*}"
      NODE_SELECTOR="nvidia.com/gpu.product=NVIDIA-${gpu_product^^}"
      ;;
    *)
      NODE_SELECTOR=""
      ;;
  esac
fi
if [[ -n "${GPU_CLIQUE:-}" ]]; then
  NODE_SELECTOR="${NODE_SELECTOR:+${NODE_SELECTOR},}nvidia.com/gpu.clique=${GPU_CLIQUE}"
fi

mkdir -p "${LOCAL_RESULT_DIR}/jobs" "${LOCAL_RESULT_DIR}/logs" "${LOCAL_RESULT_DIR}/remote_results"

_csv_items() {
  local raw="$1"
  local item
  IFS=',' read -ra _items <<<"${raw}"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "${item}" ]] && printf '%s\n' "${item}"
  done
}

_require_pvc() {
  if [[ -z "${PVC_NAME}" ]]; then
    echo "PVC_NAME is required so the K8s jobs can see the same workspace and output path." >&2
    exit 1
  fi
}

current_job=""
cleanup_job() {
  local job="$1"
  [[ -n "${job}" ]] || return 0
  bash "${LOCAL_REPO}/collector/sglang/dsv4_megamoe/cleanup_k8s_job.sh" "${job}" "${NAMESPACE}" \
    >"${LOCAL_RESULT_DIR}/${job}_delete.log" 2>&1 || true
}

cleanup_all() {
  [[ "${DRY_RUN}" == "1" ]] && return 0
  if [[ -n "${current_job}" ]]; then
    cleanup_job "${current_job}"
  fi
  kubectl delete pod "${JOB_TAG}-sync" -n "${NAMESPACE}" --ignore-not-found=true \
    >"${LOCAL_RESULT_DIR}/sync_pod_delete.log" 2>&1 || true
}
trap cleanup_all EXIT

render_job() {
  local job="$1"
  local ep="$2"
  local phase="$3"
  local prefill_tokens="$4"
  local decode_tokens="$5"
  local cap="$6"
  local args=(
    "${LOCAL_REPO}/collector/sglang/dsv4_megamoe/render_k8s_indexed_job.py"
    --job-name "${job}"
    --namespace "${NAMESPACE}"
    --system-name "${SYSTEM_NAME}"
    --ep-size "${ep}"
    --image-pull-policy "${IMAGE_PULL_POLICY}"
    --model-config "${MODEL_CONFIG}"
    --pvc-name "${PVC_NAME}"
    --pvc-mount "${PVC_MOUNT}"
    --working-dir "${REMOTE_WORKDIR}"
    --output-path "${REMOTE_OUTPUT_PATH}"
    --perf-file "${PERF_FILE}"
    --phases "${phase}"
    --prefill-tokens "${prefill_tokens}"
    --decode-tokens "${decode_tokens}"
    --distributions "${DISTRIBUTIONS}"
    --source-policy "${SOURCE_POLICY}"
    --routing-seed "${ROUTING_SEED}"
    --routing-seeds "${ROUTING_SEEDS}"
    --pre-dispatch "${PRE_DISPATCH}"
    --include-routed-scale "${INCLUDE_ROUTED_SCALE}"
    --renormalize-topk-weights "${RENORMALIZE_TOPK_WEIGHTS}"
    --num-warmup "${NUM_WARMUP}"
    --num-iterations "${NUM_ITERATIONS}"
    --num-max-tokens-per-rank "${cap}"
    --ipc-lock "${IPC_LOCK}"
    --env AIC_WAIT_FOR_ALL_NODES=1
    --env "CAP_POLICY=${CAP_POLICY}"
    --env NCCL_DEBUG=WARN
    --env NCCL_GRAPH_MIXING_SUPPORT=0
    --env NCCL_NVLS_ENABLE=1
  )
  if [[ -n "${IMAGE}" ]]; then
    args+=(--image "${IMAGE}")
  fi
  if [[ -n "${NODE_SELECTOR}" ]]; then
    args+=(--node-selector "${NODE_SELECTOR}")
  fi
  if [[ -n "${PRIORITY_CLASS_NAME}" ]]; then
    args+=(--priority-class-name "${PRIORITY_CLASS_NAME}")
  fi
  if [[ "${COMPUTE_DOMAIN}" == "1" ]]; then
    args+=(--compute-domain)
  fi
  args+=(--toleration-key kubernetes.io/arch)
  args+=(--toleration-key nvidia.com/gpu)
  args+=(--toleration-key user-workload)
  python3 "${args[@]}" >"${LOCAL_RESULT_DIR}/jobs/${job}.yaml"
}

wait_job() {
  local job="$1"
  echo "JOB_START ${job} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
  current_job="${job}"
  kubectl apply --validate=false --server-side --dry-run=server -f "${LOCAL_RESULT_DIR}/jobs/${job}.yaml" \
    >>"${LOCAL_RESULT_DIR}/jobs_dry_run.log" 2>&1
  kubectl apply --validate=false -f "${LOCAL_RESULT_DIR}/jobs/${job}.yaml" | tee "${LOCAL_RESULT_DIR}/${job}_apply.log"
  kubectl get job,pod,svc,computedomain -n "${NAMESPACE}" -l "app=${job}" -o wide \
    >"${LOCAL_RESULT_DIR}/${job}_resources_after_apply.txt" 2>&1 || true
  kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o wide \
    >"${LOCAL_RESULT_DIR}/${job}_pods_after_apply.txt" 2>&1 || true

  local deadline=$((SECONDS + JOB_TIMEOUT_SECONDS))
  local iter=0
  while true; do
    local succeeded failed active completions
    succeeded="$(kubectl get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.status.succeeded}' 2>/dev/null || true)"
    failed="$(kubectl get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.status.failed}' 2>/dev/null || true)"
    active="$(kubectl get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.status.active}' 2>/dev/null || true)"
    completions="$(kubectl get job "${job}" -n "${NAMESPACE}" -o jsonpath='{.spec.completions}' 2>/dev/null || true)"
    succeeded="${succeeded:-0}"
    failed="${failed:-0}"
    active="${active:-0}"
    completions="${completions:-1}"
    echo "JOB_WAIT ${job} active=${active} succeeded=${succeeded}/${completions} failed=${failed} $(date '+%Y-%m-%d %H:%M:%S')" \
      | tee -a "${LOCAL_RESULT_DIR}/runner.log"
    kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o wide \
      >"${LOCAL_RESULT_DIR}/${job}_pods_wait_${iter}.txt" 2>&1 || true
    if (( succeeded >= completions )); then
      break
    fi
    if [[ "${failed}" != "0" ]]; then
      echo "JOB_FAILED ${job}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
      kubectl get job "${job}" -n "${NAMESPACE}" -o yaml >"${LOCAL_RESULT_DIR}/${job}_job_failed.yaml" 2>&1 || true
      kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o wide >"${LOCAL_RESULT_DIR}/${job}_pods_failed.txt" 2>&1 || true
      for pod in $(kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o name 2>/dev/null | sed 's#pod/##'); do
        kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers=true >"${LOCAL_RESULT_DIR}/logs/${pod}.log" 2>&1 || true
        kubectl describe pod "${pod}" -n "${NAMESPACE}" >"${LOCAL_RESULT_DIR}/logs/${pod}.describe.txt" 2>&1 || true
      done
      return 1
    fi
    if (( SECONDS > deadline )); then
      echo "JOB_TIMEOUT ${job}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
      kubectl get job "${job}" -n "${NAMESPACE}" -o yaml >"${LOCAL_RESULT_DIR}/${job}_job_timeout.yaml" 2>&1 || true
      kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o wide >"${LOCAL_RESULT_DIR}/${job}_pods_timeout.txt" 2>&1 || true
      for pod in $(kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o name 2>/dev/null | sed 's#pod/##'); do
        kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers=true >"${LOCAL_RESULT_DIR}/logs/${pod}.log" 2>&1 || true
        kubectl describe pod "${pod}" -n "${NAMESPACE}" >"${LOCAL_RESULT_DIR}/logs/${pod}.describe.txt" 2>&1 || true
      done
      return 1
    fi
    iter=$((iter + 1))
    sleep "${WAIT_POLL_SECONDS}"
  done

  kubectl get job "${job}" -n "${NAMESPACE}" -o yaml >"${LOCAL_RESULT_DIR}/${job}_job_complete.yaml" 2>&1 || true
  kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o wide >"${LOCAL_RESULT_DIR}/${job}_pods_complete.txt" 2>&1 || true
  for pod in $(kubectl get pods -n "${NAMESPACE}" -l "app=${job}" -o name 2>/dev/null | sed 's#pod/##'); do
    kubectl logs "${pod}" -n "${NAMESPACE}" --all-containers=true >"${LOCAL_RESULT_DIR}/logs/${pod}.log" 2>&1 || true
  done
  cleanup_job "${job}"
  current_job=""
  echo "JOB_END ${job} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
}

JOBS=()
for phase in $(_csv_items "${PHASE_ORDER}"); do
  case "${phase}" in
    context)
      for ep in $(_csv_items "${PREFILL_EP_SIZES}"); do
        job="${JOB_TAG}-p-e${ep}"
        render_job "${job}" "${ep}" context "${PREFILL_TOKENS}" "1" "${PREFILL_NUM_MAX_TOKENS_PER_RANK}"
        JOBS+=("${job}")
      done
      ;;
    generation)
      for ep in $(_csv_items "${DECODE_EP_SIZES}"); do
        job="${JOB_TAG}-d-e${ep}"
        render_job "${job}" "${ep}" generation "1024" "${DECODE_TOKENS}" "${DECODE_NUM_MAX_TOKENS_PER_RANK}"
        JOBS+=("${job}")
      done
      ;;
    *)
      echo "unsupported PHASE_ORDER entry: ${phase}" >&2
      exit 1
      ;;
  esac
done

cat >"${LOCAL_RESULT_DIR}/run_metadata.env" <<EOF
RUN_ID=${RUN_ID}
JOB_PREFIX=${JOB_PREFIX}
JOB_TAG=${JOB_TAG}
SYSTEM_NAME=${SYSTEM_NAME}
MODEL_CONFIG=${MODEL_CONFIG}
LOCAL_REPO=${LOCAL_REPO}
REMOTE_WORKDIR=${REMOTE_WORKDIR}
REMOTE_OUTPUT_PATH=${REMOTE_OUTPUT_PATH}
LOCAL_RESULT_DIR=${LOCAL_RESULT_DIR}
PERF_FILE=${PERF_FILE}
EP_SIZES=${EP_SIZES}
PREFILL_EP_SIZES=${PREFILL_EP_SIZES}
DECODE_EP_SIZES=${DECODE_EP_SIZES}
PREFILL_TOKENS=${PREFILL_TOKENS}
DECODE_TOKENS=${DECODE_TOKENS}
DISTRIBUTIONS=${DISTRIBUTIONS}
SOURCE_POLICY=${SOURCE_POLICY}
CAP_POLICY=${CAP_POLICY}
LOCAL_HEAD=$(git -C "${LOCAL_REPO}" rev-parse HEAD 2>/dev/null || true)
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN=1 rendered %d jobs under %s/jobs\n' "${#JOBS[@]}" "${LOCAL_RESULT_DIR}"
  exit 0
fi

_require_pvc

cat >"${LOCAL_RESULT_DIR}/sync_pod.yaml" <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${JOB_TAG}-sync
  namespace: ${NAMESPACE}
  labels:
    app: ${JOB_TAG}-sync
    aic.nvidia.com/collector: dsv4-megamoe
spec:
  restartPolicy: Never
  containers:
  - name: sync
    image: ${SYNC_IMAGE}
    command: ["/bin/bash", "-lc", "sleep 86400"]
    volumeMounts:
    - name: aic-workspace
      mountPath: ${PVC_MOUNT}
  volumes:
  - name: aic-workspace
    persistentVolumeClaim:
      claimName: ${PVC_NAME}
EOF

kubectl apply -f "${LOCAL_RESULT_DIR}/sync_pod.yaml" | tee "${LOCAL_RESULT_DIR}/sync_pod_create.log"
kubectl wait --for=condition=Ready "pod/${JOB_TAG}-sync" -n "${NAMESPACE}" --timeout=300s \
  | tee "${LOCAL_RESULT_DIR}/sync_pod_wait.log"
kubectl exec "${JOB_TAG}-sync" -n "${NAMESPACE}" -- bash -lc \
  "rm -rf '${REMOTE_BASE}' '${REMOTE_OUTPUT_PATH}' && mkdir -p '${REMOTE_WORKDIR}' '${REMOTE_OUTPUT_PATH}'"

tar \
  --exclude=.git \
  --exclude=.pytest_cache \
  --exclude='**/__pycache__' \
  --exclude=artifacts \
  -C "${LOCAL_REPO}" -cf - . \
  | kubectl exec -i "${JOB_TAG}-sync" -n "${NAMESPACE}" -- tar -C "${REMOTE_WORKDIR}" -xf -

for job in "${JOBS[@]}"; do
  wait_job "${job}"
done

kubectl exec "${JOB_TAG}-sync" -n "${NAMESPACE}" -- bash -lc \
  "cd '${REMOTE_OUTPUT_PATH}' && test -f '${PERF_FILE}' && sha256sum '${PERF_FILE}'" \
  | tee "${LOCAL_RESULT_DIR}/remote_perf_sha256.txt"
kubectl exec "${JOB_TAG}-sync" -n "${NAMESPACE}" -- tar -C "${REMOTE_OUTPUT_PATH}" -cf - . \
  | tar -C "${LOCAL_RESULT_DIR}/remote_results" -xf -

python3 "${SCRIPT_DIR}/validate_perf.py" validate \
  --perf-path "${LOCAL_RESULT_DIR}/remote_results/${PERF_FILE}" \
  --prefill-ep-sizes "${PREFILL_EP_SIZES}" \
  --decode-ep-sizes "${DECODE_EP_SIZES}" \
  --prefill-tokens "${PREFILL_TOKENS}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --distributions "${DISTRIBUTIONS}" \
  --phase-order "${PHASE_ORDER}" \
  --target-sglang-version "${TARGET_SGLANG_VERSION}" \
  --allow-version-mismatch "${ALLOW_VERSION_MISMATCH}" \
  --summary-path "${LOCAL_RESULT_DIR}/validation_summary.txt" \
  --expect-single-perf-file

echo "RUNNER_ALL_JOBS_COMPLETE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
