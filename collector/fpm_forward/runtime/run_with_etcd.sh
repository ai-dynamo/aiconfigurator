#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

workdir=/tmp/fpm-bench
node_rank="${LWS_WORKER_INDEX:-0}"
leader_address="${LWS_LEADER_ADDRESS:-127.0.0.1}"
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

# etcd is a static binary with no dependency on the preflight audit, and the
# followers' readiness probe races the LEADER's preflight when etcd starts
# after it: preflight imports vLLM/torch, whose cold-vs-warm page-cache delta
# across nodes is unbounded. Start etcd first so the probe budget only has to
# cover pod-exec skew.
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

python3 "${workdir}/preflight.py"

python3 - "${leader_address}" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
deadline = time.monotonic() + 120
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
