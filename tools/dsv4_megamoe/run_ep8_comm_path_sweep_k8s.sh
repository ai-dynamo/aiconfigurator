#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NAMESPACE=${NAMESPACE:-yuanli-dynamo}
IMAGE=${IMAGE:-lmsysorg/sglang:deepseek-v4-blackwell}
ROUTING_MODE=${ROUTING_MODE:-random}
RUN_ID=${RUN_ID:-aic-dsv4-${ROUTING_MODE}-comm-path-sweep-ep8-$(date -u +%Y%m%dt%H%M%Sz)}
POD_NAME=${POD_NAME:-${RUN_ID}}
SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT:-3600s}
GPU_PRODUCT=${GPU_PRODUCT:-NVIDIA-B200}
TARGET_HOSTNAME=${TARGET_HOSTNAME:-}
EXCLUDED_HOSTS=${EXCLUDED_HOSTS:-cluster-0967a26d-pool-14bee067-prctr-s2877,cluster-0967a26d-pool-14bee067-prctr-l9nsv}
ALLOW_DRA_NODES=${ALLOW_DRA_NODES:-0}

TOKENS=${TOKENS:-1,8,16,32,64,128,256,512,1024,2048,4096,8192}
INTERMEDIATE_HIDDENS=${INTERMEDIATE_HIDDENS:-512,1024,1536,2048,2560,3072}
REPEAT_SAMPLES=${REPEAT_SAMPLES:-5}
POWER_LAW_ALPHA=${POWER_LAW_ALPHA:-1.01}
SEED=${SEED:-0}
NUM_MAX_TOKENS_PER_RANK=${NUM_MAX_TOKENS_PER_RANK:-8192}
FLUSH_L2=${FLUSH_L2:-1}
BACKEND_VERSION=${BACKEND_VERSION:-0.5.9}
PLATEAU_TOLERANCE_PCT=${PLATEAU_TOLERANCE_PCT:-5}
REMOTE_POLL_SECONDS=${REMOTE_POLL_SECONDS:-30}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/dsv4_megamoe_comm_path_sweep/${RUN_ID}}
KEEP_POD=${KEEP_POD:-0}

cleanup() {
  if [[ "${KEEP_POD}" != "1" ]]; then
    kubectl delete pod -n "${NAMESPACE}" "${POD_NAME}" --wait=true >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

kubectl_exec_retry() {
  local attempts=${KUBECTL_EXEC_ATTEMPTS:-3}
  local sleep_seconds=${KUBECTL_EXEC_RETRY_SLEEP_SECONDS:-10}
  local attempt

  for attempt in $(seq 1 "${attempts}"); do
    if kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- "$@"; then
      return 0
    fi
    if [[ "${attempt}" -lt "${attempts}" ]]; then
      echo "kubectl exec failed for ${POD_NAME} attempt ${attempt}/${attempts}; retrying in ${sleep_seconds}s"
      sleep "${sleep_seconds}"
    fi
  done

  echo "kubectl exec failed for ${POD_NAME} after ${attempts} attempts"
  return 1
}

csv_values_yaml() {
  local csv=$1
  local value
  IFS=',' read -ra values <<< "${csv}"
  for value in "${values[@]}"; do
    value="${value//[[:space:]]/}"
    if [[ -n "${value}" ]]; then
      printf '            - %s\n' "${value}"
    fi
  done
}

HOST_AFFINITY_YAML=""
if [[ -n "${TARGET_HOSTNAME}" ]]; then
  HOST_AFFINITY_YAML=$(cat <<EOF
          - key: kubernetes.io/hostname
            operator: In
            values:
$(csv_values_yaml "${TARGET_HOSTNAME}")
EOF
)
elif [[ -n "${EXCLUDED_HOSTS}" ]]; then
  HOST_AFFINITY_YAML=$(cat <<EOF
          - key: kubernetes.io/hostname
            operator: NotIn
            values:
$(csv_values_yaml "${EXCLUDED_HOSTS}")
EOF
)
fi

DRA_TOLERATION_YAML=""
if [[ "${ALLOW_DRA_NODES}" == "1" ]]; then
  DRA_TOLERATION_YAML=$(cat <<EOF
  - key: dra
    operator: Exists
    effect: NoSchedule
EOF
)
fi

echo "Submitting ${POD_NAME}: GPU_PRODUCT=${GPU_PRODUCT}, TARGET_HOSTNAME=${TARGET_HOSTNAME:-<unset>}, "\
"EXCLUDED_HOSTS=${EXCLUDED_HOSTS:-<unset>}, ALLOW_DRA_NODES=${ALLOW_DRA_NODES}, "\
"TOKENS=${TOKENS}, INTERMEDIATE_HIDDENS=${INTERMEDIATE_HIDDENS}, "\
"REPEAT_SAMPLES=${REPEAT_SAMPLES}, ROUTING_MODE=${ROUTING_MODE}, "\
"ALPHA=${POWER_LAW_ALPHA}, SEED=${SEED}"

if [[ "${ROUTING_MODE}" != "random" && "${ROUTING_MODE}" != "power-law" ]]; then
  echo "ROUTING_MODE must be random or power-law, got ${ROUTING_MODE}" >&2
  exit 2
fi

IFS=',' read -ra requested_intermediate_hiddens <<< "${INTERMEDIATE_HIDDENS}"
for requested_intermediate_hidden in "${requested_intermediate_hiddens[@]}"; do
  requested_intermediate_hidden="${requested_intermediate_hidden//[[:space:]]/}"
  if [[ -z "${requested_intermediate_hidden}" ]]; then
    continue
  fi
  if (( requested_intermediate_hidden % 512 != 0 )); then
    echo "intermediate_hidden=${requested_intermediate_hidden} is invalid: DeepGEMM MegaMoE requires multiples of 512 for TMA-aligned SF buffers" >&2
    exit 2
  fi
done

kubectl apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: aic-dsv4-comm-path-sweep-ep8
    run-id: ${RUN_ID}
spec:
  restartPolicy: Never
  imagePullSecrets:
  - name: acr-token-secret
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: nvidia.com/gpu.product
            operator: In
            values:
            - ${GPU_PRODUCT}
${HOST_AFFINITY_YAML}
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
${DRA_TOLERATION_YAML}
  containers:
  - name: runner
    image: ${IMAGE}
    imagePullPolicy: IfNotPresent
    command: ["bash", "-lc"]
    args: ["sleep 7200"]
    env:
    - name: HF_HOME
      value: /cache
    - name: HF_HUB_OFFLINE
      value: "1"
    - name: NCCL_DEBUG
      value: WARN
    - name: PYTHONUNBUFFERED
      value: "1"
    resources:
      requests:
        cpu: "16"
        memory: 200Gi
        nvidia.com/gpu: "8"
      limits:
        cpu: "64"
        memory: 800Gi
        nvidia.com/gpu: "8"
    volumeMounts:
    - mountPath: /cache
      name: cache
    - mountPath: /dev/shm
      name: dshm
  volumes:
  - name: cache
    persistentVolumeClaim:
      claimName: shared-model-cache
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 128Gi
YAML

if ! kubectl wait -n "${NAMESPACE}" --for=condition=Ready "pod/${POD_NAME}" --timeout="${SCHEDULE_TIMEOUT}"; then
  echo "Pod ${POD_NAME} did not become Ready within ${SCHEDULE_TIMEOUT}; collecting scheduling diagnostics."
  kubectl get pod -n "${NAMESPACE}" "${POD_NAME}" -o wide || true
  kubectl describe pod -n "${NAMESPACE}" "${POD_NAME}" || true
  kubectl get events -n "${NAMESPACE}" \
    --field-selector involvedObject.kind=Pod,involvedObject.name="${POD_NAME}" \
    --sort-by=.lastTimestamp || true
  exit 1
fi
kubectl get pod -n "${NAMESPACE}" "${POD_NAME}" -o wide
kubectl_exec_retry nvidia-smi -L || echo "Warning: nvidia-smi probe failed; continuing to collector run."

kubectl cp -n "${NAMESPACE}" \
  "${REPO_ROOT}/collector/sglang/collect_dsv4_megamoe_effective_nvl_bw.py" \
  "${POD_NAME}:/tmp/collect_dsv4_megamoe_effective_nvl_bw.py"
kubectl cp -n "${NAMESPACE}" \
  "${REPO_ROOT}/collector/sglang/analyze_dsv4_megamoe_comm_sweep.py" \
  "${POD_NAME}:/tmp/analyze_dsv4_megamoe_comm_sweep.py"
kubectl cp -n "${NAMESPACE}" \
  "${REPO_ROOT}/tools/dsv4_megamoe/remote_comm_path_sweep_inner.sh" \
  "${POD_NAME}:/tmp/remote_comm_path_sweep_inner.sh"

kubectl_exec_retry bash -lc "
set -euo pipefail
chmod +x /tmp/remote_comm_path_sweep_inner.sh
rm -f /tmp/aic_comm_path_sweep_ep8.log /tmp/aic_comm_path_sweep_ep8.exit /tmp/aic_comm_path_sweep_ep8.pid
nohup env \
  RUN_ID='${RUN_ID}' \
  INTERMEDIATE_HIDDENS='${INTERMEDIATE_HIDDENS}' \
  NUM_MAX_TOKENS_PER_RANK='${NUM_MAX_TOKENS_PER_RANK}' \
  TOKENS='${TOKENS}' \
  REPEAT_SAMPLES='${REPEAT_SAMPLES}' \
  ROUTING_MODE='${ROUTING_MODE}' \
  POWER_LAW_ALPHA='${POWER_LAW_ALPHA}' \
  SEED='${SEED}' \
  FLUSH_L2='${FLUSH_L2}' \
  BACKEND_VERSION='${BACKEND_VERSION}' \
  PLATEAU_TOLERANCE_PCT='${PLATEAU_TOLERANCE_PCT}' \
  bash -lc '/tmp/remote_comm_path_sweep_inner.sh > /tmp/aic_comm_path_sweep_ep8.log 2>&1; echo \$? > /tmp/aic_comm_path_sweep_ep8.exit' \
  >/tmp/aic_comm_path_sweep_ep8.launch.log 2>&1 &
echo \$! > /tmp/aic_comm_path_sweep_ep8.pid
cat /tmp/aic_comm_path_sweep_ep8.pid
"

while true; do
  sleep "${REMOTE_POLL_SECONDS}"
  echo "---- remote log tail $(date -u +%Y-%m-%dT%H:%M:%SZ) ----"
  kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- \
    bash -lc "tail -n 60 /tmp/aic_comm_path_sweep_ep8.log 2>/dev/null || cat /tmp/aic_comm_path_sweep_ep8.launch.log 2>/dev/null || true" || true
  if kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- \
    bash -lc "test -f /tmp/aic_comm_path_sweep_ep8.exit" >/dev/null 2>&1; then
    remote_exit_code=$(kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- bash -lc "cat /tmp/aic_comm_path_sweep_ep8.exit")
    echo "Remote comm-path sweep exit code: ${remote_exit_code}"
    break
  fi
done

if [[ "${remote_exit_code}" != "0" ]]; then
  kubectl_exec_retry bash -lc "tail -n 200 /tmp/aic_comm_path_sweep_ep8.log 2>/dev/null || true" || true
  exit "${remote_exit_code}"
fi

kubectl_exec_retry bash -lc \
  "rm -f /tmp/aic_comm_path_sweep_ep8.tgz && tar -C /tmp -czf /tmp/aic_comm_path_sweep_ep8.tgz aic_comm_path_sweep_ep8"

rm -rf "${ARTIFACT_DIR}"
mkdir -p "$(dirname -- "${ARTIFACT_DIR}")"
kubectl cp -n "${NAMESPACE}" "${POD_NAME}:/tmp/aic_comm_path_sweep_ep8.tgz" "${ARTIFACT_DIR}.tgz"
extract_dir=$(mktemp -d "$(dirname -- "${ARTIFACT_DIR}")/.${RUN_ID}.extract.XXXXXX")
tar -C "${extract_dir}" -xzf "${ARTIFACT_DIR}.tgz"
mv "${extract_dir}/aic_comm_path_sweep_ep8" "${ARTIFACT_DIR}"
rm -rf "${extract_dir}"
find "${ARTIFACT_DIR}" -maxdepth 4 -type f | sort
