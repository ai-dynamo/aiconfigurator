#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

workdir=/tmp/fpm-bench
node_rank="${FPM_NODE_RANK:-${LWS_WORKER_INDEX:-${GROVE_PCLQ_POD_INDEX:-}}}"
leader_address="${FPM_MASTER_ADDR:-${LWS_LEADER_ADDRESS:-}}"
if [[ -z "${leader_address}" \
  && -n "${GROVE_PCLQ_NAME:-}" \
  && -n "${GROVE_HEADLESS_SERVICE:-}" ]]; then
  leader_address="${GROVE_PCLQ_NAME}-0.${GROVE_HEADLESS_SERVICE}"
fi
if [[ -z "${node_rank}" \
  && -z "${leader_address}" \
  && -z "${GROVE_PCLQ_NAME:-}" \
  && -z "${GROVE_HEADLESS_SERVICE:-}" ]]; then
  node_rank=0
  leader_address=127.0.0.1
elif [[ -z "${node_rank}" || -z "${leader_address}" ]]; then
  echo "FPM runtime requires complete rank and leader discovery from FPM_NODE_*, LWS, or Grove" >&2
  exit 2
fi
if ! [[ "${node_rank}" =~ ^[0-9]+$ ]]; then
  echo "Invalid FPM node rank: ${node_rank}" >&2
  exit 2
fi
etcd_endpoint="http://${leader_address}:2379"
etcd_pid=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${etcd_pid}" ]] && kill -0 "${etcd_pid}" 2>/dev/null; then
    kill -TERM "${etcd_pid}" 2>/dev/null || true
    wait "${etcd_pid}" 2>/dev/null || true
  fi
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python3 "${workdir}/preflight.py"

if [[ "${node_rank}" == "0" ]]; then
  data_dir=/tmp/fpm-forward-etcd
  rm -rf "${data_dir}"
  etcd \
    --data-dir "${data_dir}" \
    --listen-client-urls http://0.0.0.0:2379 \
    --advertise-client-urls "${etcd_endpoint}" \
    --listen-peer-urls http://127.0.0.1:2380 \
    >/results/etcd.log 2>&1 &
  etcd_pid=$!
fi

python3 - "${leader_address}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, 2379), timeout=1):
            break
    except OSError:
        time.sleep(0.2)
else:
    raise SystemExit(f"etcd readiness timeout for {host}:2379")
PY

export ETCD_ENDPOINTS="${etcd_endpoint}"
bash "${workdir}/run.sh" \
  > >(tee /results/engine.stdout.log) \
  2> >(tee /results/engine.stderr.log >&2)
