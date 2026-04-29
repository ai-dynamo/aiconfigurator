#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NAMESPACE=${NAMESPACE:-yuanli-dynamo}
IMAGE=${IMAGE:-lmsysorg/sglang:deepseek-v4-blackwell}
RUN_ID=${RUN_ID:-aic-dsv4-powerlaw-ep8-$(date -u +%Y%m%dt%H%M%Sz)}
POD_NAME=${POD_NAME:-${RUN_ID}}
SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT:-3600s}
TOKENS=${TOKENS:-1,2,4,8,16,32,64,128,256,384,512,1024,2048,4096,8192,16384}
REPEAT_SAMPLES=${REPEAT_SAMPLES:-5}
POWER_LAW_ALPHA=${POWER_LAW_ALPHA:-1.01}
NUM_MAX_TOKENS_PER_RANK=${NUM_MAX_TOKENS_PER_RANK:-16384}
ARTIFACT_DIR=${ARTIFACT_DIR:-${REPO_ROOT}/artifacts/deepgemm_effective_nvl_bw/${RUN_ID}}
KEEP_POD=${KEEP_POD:-0}

cleanup() {
  if [[ "${KEEP_POD}" != "1" ]]; then
    kubectl delete pod -n "${NAMESPACE}" "${POD_NAME}" --wait=true >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

kubectl apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: aic-dsv4-powerlaw-ep8
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
            - NVIDIA-B200
          - key: kubernetes.io/hostname
            operator: NotIn
            values:
            - cluster-0967a26d-pool-14bee067-prctr-s2877
            - cluster-0967a26d-pool-14bee067-prctr-l9nsv
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
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

kubectl wait -n "${NAMESPACE}" --for=condition=Ready "pod/${POD_NAME}" --timeout="${SCHEDULE_TIMEOUT}"
kubectl get pod -n "${NAMESPACE}" "${POD_NAME}" -o wide
kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- nvidia-smi -L

kubectl cp -n "${NAMESPACE}" \
  "${REPO_ROOT}/collector/sglang/collect_dsv4_megamoe_effective_nvl_bw.py" \
  "${POD_NAME}:/tmp/collect_dsv4_megamoe_effective_nvl_bw.py"

kubectl exec -n "${NAMESPACE}" "${POD_NAME}" -- bash -lc "
set -euo pipefail
rm -rf /tmp/aic_powerlaw_ep8
python3 /tmp/collect_dsv4_megamoe_effective_nvl_bw.py \
  --num-processes 8 \
  --num-max-tokens-per-rank ${NUM_MAX_TOKENS_PER_RANK} \
  --num-tokens-list ${TOKENS} \
  --repeat-samples ${REPEAT_SAMPLES} \
  --routing-mode power-law \
  --power-law-alpha ${POWER_LAW_ALPHA} \
  --output /tmp/aic_powerlaw_ep8/result.json \
  --hard-exit-after-write
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
"

rm -rf "${ARTIFACT_DIR}"
mkdir -p "$(dirname -- "${ARTIFACT_DIR}")"
kubectl cp -n "${NAMESPACE}" "${POD_NAME}:/tmp/aic_powerlaw_ep8" "${ARTIFACT_DIR}"
find "${ARTIFACT_DIR}" -maxdepth 3 -type f | sort
