#!/usr/bin/env bash
set -euo pipefail

rm -rf /tmp/aic_comm_path_sweep_ep8
mkdir -p /tmp/aic_comm_path_sweep_ep8

collector_args=(
  --num-processes 8
  --num-max-tokens-per-rank "${NUM_MAX_TOKENS_PER_RANK}"
  --num-tokens-list "${TOKENS}"
  --repeat-samples "${REPEAT_SAMPLES}"
  --routing-mode "${ROUTING_MODE:-random}"
)
if [[ "${ROUTING_MODE:-random}" == "power-law" ]]; then
  collector_args+=(--power-law-alpha "${POWER_LAW_ALPHA}")
fi

IFS=',' read -ra hiddens <<< "${INTERMEDIATE_HIDDENS}"
for intermediate_hidden in "${hiddens[@]}"; do
  intermediate_hidden="${intermediate_hidden//[[:space:]]/}"
  if [[ -z "${intermediate_hidden}" ]]; then
    continue
  fi
  out_dir=/tmp/aic_comm_path_sweep_ep8/intermediate_${intermediate_hidden}
  mkdir -p "${out_dir}"
  echo "Running intermediate_hidden=${intermediate_hidden}, routing_mode=${ROUTING_MODE:-random}"
  python3 /tmp/collect_dsv4_megamoe_effective_nvl_bw.py \
    "${collector_args[@]}" \
    --intermediate-hidden "${intermediate_hidden}" \
    --seed "${SEED}" \
    --flush-l2 "${FLUSH_L2}" \
    --output "${out_dir}/samples.json" \
    --perf-output "${out_dir}/dsv4_megamoe_effective_nvl_bw_perf.txt" \
    --backend-version "${BACKEND_VERSION}" \
    --source "${RUN_ID}-intermediate-${intermediate_hidden}" \
    --hard-exit-after-write
done

device_name=$(python3 - <<'PY'
import torch
print(torch.cuda.get_device_name(0))
PY
)

python3 /tmp/analyze_dsv4_megamoe_comm_sweep.py \
  --input-dir /tmp/aic_comm_path_sweep_ep8 \
  --output-dir /tmp/aic_comm_path_sweep_ep8/analysis \
  --plateau-tolerance-pct "${PLATEAU_TOLERANCE_PCT}" \
  --perf-output /tmp/aic_comm_path_sweep_ep8/analysis/dsv4_megamoe_comm_path_perf.txt \
  --backend-version "${BACKEND_VERSION}" \
  --device-name "${device_name}" \
  --routing-mode "${ROUTING_MODE:-random}" \
  --power-law-alpha "${POWER_LAW_ALPHA}" \
  --source "${RUN_ID}"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
