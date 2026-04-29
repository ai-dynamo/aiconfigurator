#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

NAMESPACE=${NAMESPACE:-yuanli-dynamo}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-600}
SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT:-3600s}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-0}
ROUTING_MODE=${ROUTING_MODE:-random}
LOG_DIR=${LOG_DIR:-${REPO_ROOT}/artifacts/deepgemm_effective_nvl_bw}
MONITOR_LOG=${MONITOR_LOG:-${LOG_DIR}/ep8_${ROUTING_MODE}_submit_monitor_$(date -u +%Y%m%dT%H%M%SZ).log}

mkdir -p "${LOG_DIR}"

attempt=0
while true; do
  attempt=$((attempt + 1))
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "[${ts}] attempt=${attempt}: checking/submitting EP8 ${ROUTING_MODE} sweep" | tee -a "${MONITOR_LOG}"
  echo "[${ts}] config: SCHEDULE_TIMEOUT=${SCHEDULE_TIMEOUT}, TARGET_HOSTNAME=${TARGET_HOSTNAME:-<unset>}, "\
"ALLOW_DRA_NODES=${ALLOW_DRA_NODES:-0}, EXCLUDED_HOSTS=${EXCLUDED_HOSTS:-<default>}, "\
"ROUTING_MODE=${ROUTING_MODE}, FLUSH_L2=${FLUSH_L2:-1}, "\
"NUM_MAX_TOKENS_PER_RANK=${NUM_MAX_TOKENS_PER_RANK:-<default>}, TOKENS=${TOKENS:-<default>}" \
    | tee -a "${MONITOR_LOG}"

  existing=$(kubectl get pod -n "${NAMESPACE}" -l app=aic-dsv4-megamoe-effective-nvl-bw-ep8 --no-headers 2>/dev/null || true)
  if [[ -n "${existing}" ]]; then
    echo "[${ts}] existing EP8 sweep pod found; waiting ${INTERVAL_SECONDS}s" | tee -a "${MONITOR_LOG}"
    echo "${existing}" | tee -a "${MONITOR_LOG}"
    sleep "${INTERVAL_SECONDS}"
    continue
  fi

  run_id="aic-dsv4-${ROUTING_MODE}-ep8-${ts//[:]/}"
  run_id="${run_id,,}"
  set +e
  SCHEDULE_TIMEOUT="${SCHEDULE_TIMEOUT}" RUN_ID="${run_id}" ROUTING_MODE="${ROUTING_MODE}" \
    "${SCRIPT_DIR}/run_ep8_powerlaw_sweep_k8s.sh" 2>&1 | tee -a "${MONITOR_LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SUCCESS run_id=${run_id}" | tee -a "${MONITOR_LOG}"
    exit 0
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] attempt=${attempt} failed rc=${rc}; sleeping ${INTERVAL_SECONDS}s" \
    | tee -a "${MONITOR_LOG}"
  if [[ "${MAX_ATTEMPTS}" -gt 0 && "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reached MAX_ATTEMPTS=${MAX_ATTEMPTS}; exiting" | tee -a "${MONITOR_LOG}"
    exit "${rc}"
  fi
  sleep "${INTERVAL_SECONDS}"
done
