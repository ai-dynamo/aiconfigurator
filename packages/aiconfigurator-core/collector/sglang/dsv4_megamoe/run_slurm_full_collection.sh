#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SSH_TARGET="${SSH_TARGET:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d%H%M%S)}"
JOB_PREFIX="${JOB_PREFIX:-aic-dsv4-megamoe}"
JOB_TAG="${JOB_PREFIX}-${RUN_ID}"

SYSTEM_NAME="${SYSTEM_NAME:-gb300}"
MODEL_CONFIG="${MODEL_CONFIG:-dsv4_pro}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Pro}"
IMAGE="${IMAGE:-lmsysorg/sglang-staging:deepseek-v4-grace-blackwell-dev}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-coreai_tritoninference_triton3}"
SLURM_PARTITION="${SLURM_PARTITION:-gb300}"
SLURM_TIME_LIMIT="${SLURM_TIME_LIMIT:-02:00:00}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

LOCAL_REPO="${LOCAL_REPO:-${AIC_ROOT}}"
LOCAL_RESULT_DIR="${LOCAL_RESULT_DIR:-${LOCAL_REPO}/artifacts/${JOB_TAG}}"
REMOTE_ROOT_BASE="${REMOTE_ROOT_BASE:-/lustre/fsw/coreai_tritoninference_triton3/tripwire/aic-dsv4-megamoe}"
REMOTE_ROOT="${REMOTE_ROOT:-${REMOTE_ROOT_BASE}/${JOB_TAG}}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${REMOTE_ROOT}/repo}"
REMOTE_RESULTS="${REMOTE_RESULTS:-${REMOTE_ROOT}/results}"
REMOTE_LOGS="${REMOTE_LOGS:-${REMOTE_ROOT}/logs}"
REMOTE_JOBS="${REMOTE_JOBS:-${REMOTE_ROOT}/jobs}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-${REMOTE_ROOT}:${REMOTE_ROOT},/lustre:/lustre}"
CONTAINER_WRITABLE="${CONTAINER_WRITABLE:-1}"

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
TEST_ONLY="${TEST_ONLY:-0}"
WAIT="${WAIT:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"
KEEP_JOBS="${KEEP_JOBS:-0}"
COPY_VALIDATED="${COPY_VALIDATED:-0}"
TARGET_SGLANG_VERSION="${TARGET_SGLANG_VERSION:-0.5.10}"
ALLOW_VERSION_MISMATCH="${ALLOW_VERSION_MISMATCH:-1}"

mkdir -p "${LOCAL_RESULT_DIR}/jobs" "${LOCAL_RESULT_DIR}/logs" "${LOCAL_RESULT_DIR}/remote_results" \
  "${LOCAL_RESULT_DIR}/merged"
SUBMITTED_JOBS_FILE="${LOCAL_RESULT_DIR}/submitted_jobs.txt"
: >"${SUBMITTED_JOBS_FILE}"

_csv_items() {
  local raw="$1"
  local item
  IFS=',' read -ra _items <<<"${raw}"
  for item in "${_items[@]}"; do
    item="${item//[[:space:]]/}"
    [[ -n "${item}" ]] && printf '%s\n' "${item}"
  done
}

_write_cancel_script() {
  cat >"${LOCAL_RESULT_DIR}/cancel_jobs.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
SSH_TARGET=${SSH_TARGET@Q}
if [[ ! -s ${SUBMITTED_JOBS_FILE@Q} ]]; then
  echo "no submitted jobs recorded"
  exit 0
fi
jobs="\$(tr '\n' ' ' < ${SUBMITTED_JOBS_FILE@Q})"
echo "scancel \${jobs}"
ssh "\${SSH_TARGET}" "scancel \${jobs}"
EOF
  chmod +x "${LOCAL_RESULT_DIR}/cancel_jobs.sh"
}

_cleanup_on_exit() {
  local status=$?
  if [[ "${status}" != "0" && "${KEEP_JOBS}" != "1" && "${WAIT}" == "1" && -s "${SUBMITTED_JOBS_FILE}" ]]; then
    echo "runner failed; canceling submitted jobs from ${SUBMITTED_JOBS_FILE}" >&2
    bash "${LOCAL_RESULT_DIR}/cancel_jobs.sh" >>"${LOCAL_RESULT_DIR}/cleanup.log" 2>&1 || true
  fi
}
trap _cleanup_on_exit EXIT

if [[ "${DRY_RUN}" != "1" && -z "${SSH_TARGET}" ]]; then
  echo "SSH_TARGET must be set for Slurm staging/submission" >&2
  exit 1
fi

_render_job() {
  local job="$1"
  local ep="$2"
  local phase="$3"
  local prefill_tokens="$4"
  local decode_tokens="$5"
  local cap="$6"
  local remote_output="${REMOTE_RESULTS}/${job}"
  local remote_log_dir="${REMOTE_LOGS}/${job}"
  local args=(
    "${LOCAL_REPO}/collector/sglang/dsv4_megamoe/render_slurm_job.py"
    --job-name "${job}"
    --account "${SLURM_ACCOUNT}"
    --partition "${SLURM_PARTITION}"
    --time-limit "${SLURM_TIME_LIMIT}"
    --system-name "${SYSTEM_NAME}"
    --ep-size "${ep}"
    --gpus-per-node "${GPUS_PER_NODE}"
    --phase "${phase}"
    --remote-workdir "${REMOTE_WORKDIR}"
    --output-path "${remote_output}"
    --log-dir "${remote_log_dir}"
    --container-image "${IMAGE}"
    --container-mounts "${CONTAINER_MOUNTS}"
    --model-config "${MODEL_CONFIG}"
    --perf-file "${PERF_FILE}"
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
    --cap-policy "${CAP_POLICY}"
    --env "AIC_DSV4_MODEL_PATH=${MODEL_PATH}"
  )
  if [[ "${CONTAINER_WRITABLE}" == "1" ]]; then
    args+=(--container-writable)
  fi
  python3 "${args[@]}" >"${LOCAL_RESULT_DIR}/jobs/${job}.sbatch"
}

_stage_repo() {
  echo "STAGE_REMOTE ${SSH_TARGET}:${REMOTE_WORKDIR}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
  ssh "${SSH_TARGET}" "mkdir -p '${REMOTE_WORKDIR}' '${REMOTE_RESULTS}' '${REMOTE_LOGS}' '${REMOTE_JOBS}'"
  rsync -az --delete \
    --exclude=.git \
    --exclude=.pytest_cache \
    --exclude='**/__pycache__' \
    --exclude='*.pyc' \
    --exclude=artifacts \
    "${LOCAL_REPO}/" "${SSH_TARGET}:${REMOTE_WORKDIR}/"
  rsync -az "${LOCAL_RESULT_DIR}/jobs/" "${SSH_TARGET}:${REMOTE_JOBS}/"
}

_wait_job() {
  local job="$1"
  local job_id="$2"
  while true; do
    local queue_state
    queue_state="$(ssh "${SSH_TARGET}" "squeue -h -j '${job_id}' -o '%T' 2>/dev/null || true")"
    if [[ -z "${queue_state}" ]]; then
      break
    fi
    echo "JOB_WAIT ${job} id=${job_id} state=${queue_state} $(date '+%Y-%m-%d %H:%M:%S')" \
      | tee -a "${LOCAL_RESULT_DIR}/runner.log"
    sleep "${WAIT_POLL_SECONDS}"
  done

  ssh "${SSH_TARGET}" \
    "sacct -j '${job_id}' --format=JobID,JobName%60,State,ExitCode,Elapsed,NodeList -P 2>/dev/null || true" \
    >"${LOCAL_RESULT_DIR}/logs/${job}_${job_id}_sacct.txt"
  local state
  state="$(ssh "${SSH_TARGET}" "sacct -n -X -j '${job_id}' --format=State -P 2>/dev/null | head -n1 | tr -d ' ' || true")"
  if [[ "${state}" != "COMPLETED" ]]; then
    echo "JOB_NOT_COMPLETED ${job} id=${job_id} state=${state:-unknown}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
    return 1
  fi
  echo "JOB_COMPLETE ${job} id=${job_id}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
}

_submit_job() {
  local job="$1"
  local remote_script="${REMOTE_JOBS}/${job}.sbatch"
  ssh "${SSH_TARGET}" "mkdir -p '${REMOTE_LOGS}/${job}' '${REMOTE_RESULTS}/${job}'"
  if [[ "${TEST_ONLY}" == "1" ]]; then
    ssh "${SSH_TARGET}" "sbatch --test-only '${remote_script}'" | tee "${LOCAL_RESULT_DIR}/logs/${job}_test_only.log"
    return 0
  fi

  local job_id
  job_id="$(ssh "${SSH_TARGET}" "sbatch --parsable '${remote_script}'" | tee "${LOCAL_RESULT_DIR}/logs/${job}_submit.log" | tail -n1 | cut -d';' -f1)"
  if [[ -z "${job_id}" ]]; then
    echo "failed to parse sbatch job id for ${job}" >&2
    return 1
  fi
  echo "${job_id}" >>"${SUBMITTED_JOBS_FILE}"
  _write_cancel_script
  echo "JOB_SUBMITTED ${job} id=${job_id}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
  if [[ "${WAIT}" == "1" ]]; then
    _wait_job "${job}" "${job_id}"
  fi
}

JOBS=()
for phase in $(_csv_items "${PHASE_ORDER}"); do
  case "${phase}" in
    context)
      for ep in $(_csv_items "${PREFILL_EP_SIZES}"); do
        job="${JOB_TAG}-p-e${ep}"
        _render_job "${job}" "${ep}" context "${PREFILL_TOKENS}" "1" "${PREFILL_NUM_MAX_TOKENS_PER_RANK}"
        JOBS+=("${job}")
      done
      ;;
    generation)
      for ep in $(_csv_items "${DECODE_EP_SIZES}"); do
        job="${JOB_TAG}-d-e${ep}"
        _render_job "${job}" "${ep}" generation "1024" "${DECODE_TOKENS}" "${DECODE_NUM_MAX_TOKENS_PER_RANK}"
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
MODEL_PATH=${MODEL_PATH}
IMAGE=${IMAGE}
SLURM_ACCOUNT=${SLURM_ACCOUNT}
SLURM_PARTITION=${SLURM_PARTITION}
SLURM_TIME_LIMIT=${SLURM_TIME_LIMIT}
GPUS_PER_NODE=${GPUS_PER_NODE}
LOCAL_REPO=${LOCAL_REPO}
REMOTE_ROOT=${REMOTE_ROOT}
REMOTE_WORKDIR=${REMOTE_WORKDIR}
REMOTE_RESULTS=${REMOTE_RESULTS}
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

_write_cancel_script

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'DRY_RUN=1 rendered %d jobs under %s/jobs\n' "${#JOBS[@]}" "${LOCAL_RESULT_DIR}"
  exit 0
fi

_stage_repo

for job in "${JOBS[@]}"; do
  _submit_job "${job}"
done

if [[ "${TEST_ONLY}" == "1" || "${WAIT}" != "1" ]]; then
  echo "SUBMISSION_DONE wait=${WAIT} test_only=${TEST_ONLY}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
  exit 0
fi

rsync -az "${SSH_TARGET}:${REMOTE_RESULTS}/" "${LOCAL_RESULT_DIR}/remote_results/"
rsync -az "${SSH_TARGET}:${REMOTE_LOGS}/" "${LOCAL_RESULT_DIR}/logs/"

python3 "${SCRIPT_DIR}/validate_perf.py" merge \
  --input-root "${LOCAL_RESULT_DIR}/remote_results" \
  --perf-file "${PERF_FILE}" \
  --output "${LOCAL_RESULT_DIR}/merged/${PERF_FILE}"

python3 "${SCRIPT_DIR}/validate_perf.py" validate \
  --perf-path "${LOCAL_RESULT_DIR}/merged/${PERF_FILE}" \
  --prefill-ep-sizes "${PREFILL_EP_SIZES}" \
  --decode-ep-sizes "${DECODE_EP_SIZES}" \
  --prefill-tokens "${PREFILL_TOKENS}" \
  --decode-tokens "${DECODE_TOKENS}" \
  --distributions "${DISTRIBUTIONS}" \
  --phase-order "${PHASE_ORDER}" \
  --target-sglang-version "${TARGET_SGLANG_VERSION}" \
  --allow-version-mismatch "${ALLOW_VERSION_MISMATCH}" \
  --summary-path "${LOCAL_RESULT_DIR}/validation_summary.txt"

if [[ "${COPY_VALIDATED}" == "1" ]]; then
  target_system="$(printf '%s' "${SYSTEM_NAME}" | tr '[:upper:]' '[:lower:]')"
  target_dir="${LOCAL_REPO}/src/aiconfigurator_core/systems/data/${target_system}/sglang/${TARGET_SGLANG_VERSION}"
  mkdir -p "${target_dir}"
  cp "${LOCAL_RESULT_DIR}/merged/${PERF_FILE}" "${target_dir}/${PERF_FILE}"
  echo "COPIED_TO=${target_dir}/${PERF_FILE}" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
fi

echo "RUNNER_ALL_JOBS_COMPLETE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOCAL_RESULT_DIR}/runner.log"
