#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail

# Collector-owned in-pod runtime for one generated FPM cell. The Generator
# renders fpm_env.sh (topology and benchmark facts as FPM_* exports) and a
# thin run.sh (engine launch only); everything that knows when collection is
# done, whether the results are valid, and which exit code the pod reports
# lives here: etcd lifecycle, preflight, the result gate, the checker, the
# DP completion barrier, follower exit classification, and engine teardown.

workdir=/tmp/fpm-bench

# Fail-closed: fpm_env.sh exits 2 on incomplete multinode rank/leader
# discovery or an invalid rank, which terminates this sourcing shell with
# the same status before any resource is started.
source "${workdir}/fpm_env.sh"

etcd_endpoint="http://${FPM_MASTER_ADDR}:2379"
engine_pid=""
etcd_pid=""

engine_shutdown_grace_seconds=30
terminate_engine() {
  local pid=$1
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local shutdown_deadline=$((SECONDS + engine_shutdown_grace_seconds))
  while kill -0 "$pid" 2>/dev/null || kill -0 -- "-$pid" 2>/dev/null; do
    if (( SECONDS >= shutdown_deadline )); then
      echo "Engine did not stop within ${engine_shutdown_grace_seconds}s; sending SIGKILL" >&2
      kill -KILL -- "-$pid" 2>/dev/null || true
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${engine_pid:-}" ]]; then
    terminate_engine "$engine_pid"
  fi
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
if [[ "${FPM_NODE_RANK}" == "0" ]]; then
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

if [[ "${FPM_NODE_RANK}" == "0" ]]; then
  # The leader owns the etcd process, so its readiness wait is etcd-aware:
  # a dead etcd (missing binary, bound port, data-dir permissions) fails the
  # cell immediately with direct evidence instead of burning the 120s probe
  # budget and reporting only "readiness timeout".
  readiness_deadline=$((SECONDS + 120))
  while ! (exec 3<>"/dev/tcp/${FPM_MASTER_ADDR}/2379") 2>/dev/null; do
    if ! kill -0 "${etcd_pid}" 2>/dev/null; then
      set +e
      wait "${etcd_pid}"
      etcd_status=$?
      set -e
      etcd_pid=""
      echo "etcd exited with status ${etcd_status} before becoming ready" >&2
      tail -n 20 /results/etcd.log >&2 || true
      exit 1
    fi
    if (( SECONDS >= readiness_deadline )); then
      echo "etcd readiness timeout for ${FPM_MASTER_ADDR}:2379" >&2
      exit 1
    fi
    sleep 1
  done
else
  python3 - "${FPM_MASTER_ADDR}" <<'PY'
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
fi

export ETCD_ENDPOINTS="${etcd_endpoint}"

benchmark_path_for_dp_rank() {
  local dp_rank=$1
  local directory=""
  local filename="$FPM_BENCHMARK_OUTPUT_PATH"
  if [[ "$FPM_BENCHMARK_OUTPUT_PATH" == */* ]]; then
    directory="${FPM_BENCHMARK_OUTPUT_PATH%/*}/"
    filename="${FPM_BENCHMARK_OUTPUT_PATH##*/}"
  fi
  if (( dp_rank == 0 )); then
    printf "%s\n" "$FPM_BENCHMARK_OUTPUT_PATH"
  elif [[ "$filename" == *.* ]]; then
    printf "%s%s_dp%s.%s\n" "$directory" "${filename%.*}" "$dp_rank" "${filename##*.}"
  else
    printf "%s%s_dp%s\n" "$directory" "$filename" "$dp_rank"
  fi
}

# Headless followers (multinode, dp=1, rank>0) never write results: their
# exit is classified against the leader's teardown instead of gated on files,
# so the overwrite check below must not run for them — a stale per-rank file
# on a follower volume must not fail the cell.
fpm_is_follower=0
if (( FPM_NODE_COUNT > 1 && FPM_DATA_PARALLEL_SIZE == 1 && FPM_NODE_RANK > 0 )); then
  fpm_is_follower=1
fi

expected_results=()
local_dp_start=$((FPM_NODE_RANK * FPM_LOCAL_DATA_PARALLEL_SIZE))
local_dp_end=$((local_dp_start + FPM_LOCAL_DATA_PARALLEL_SIZE))
if (( ! fpm_is_follower )); then
  for ((dp_rank=local_dp_start; dp_rank<local_dp_end; dp_rank++)); do
    expected_results+=("$(benchmark_path_for_dp_rank "$dp_rank")")
  done
  for path in "${expected_results[@]}"; do
    if [[ -e "$path" || -L "$path" ]]; then
      echo "Refusing to overwrite existing benchmark output: $path" >&2
      exit 1
    fi
    mkdir -p -- "$(dirname -- "$path")"
  done
fi

check_result_files() {
  python3 - "$local_dp_start" "$FPM_BENCHMARK_MODE" "$FPM_RESULT_SCHEMA_VERSION" "${expected_results[@]}" <<'PY'
import json
import pathlib
import sys

start_rank = int(sys.argv[1])
expected_mode = sys.argv[2]
expected_schema_version = int(sys.argv[3])
allowed_point_types = {"prefill", "decode"} if expected_mode == "agg" else {expected_mode}

def invalid(path, message):
    print(f"Invalid FPM benchmark result {path}: {message}", file=sys.stderr)
    raise SystemExit(20)

for offset, raw_path in enumerate(sys.argv[4:]):
    path = pathlib.Path(raw_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(10)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise SystemExit(10)
    if not isinstance(value, dict):
        invalid(path, f"top-level JSON must be an object, got {type(value).__name__}")
    expected_rank = start_rank + offset
    if (value.get("schema_version") != expected_schema_version or value.get("status") != "complete"
            or value.get("valid") is not True):
        invalid(path,
                f"schema_version={value.get('schema_version')!r} "
                f"status={value.get('status')!r} "
                f"valid={value.get('valid')!r} errors={value.get('errors')!r}")
    config = value.get("config")
    actual_mode = config.get("mode") if isinstance(config, dict) else None
    if actual_mode != expected_mode:
        invalid(path, f"benchmark mode {actual_mode!r} != {expected_mode!r}")
    coverage = value.get("coverage")
    if not isinstance(coverage, dict):
        invalid(path, "coverage must be an object")
    expected_points = coverage.get("expected_points")
    completed_points = coverage.get("completed_points")
    skipped_points = coverage.get("skipped_points")
    if (type(expected_points) is not int or expected_points <= 0
            or type(completed_points) is not int or completed_points != expected_points
            or type(skipped_points) is not int or skipped_points != 0):
        invalid(path, f"invalid coverage {coverage!r}")
    results = value.get("results")
    result_count = len(results) if isinstance(results, list) else None
    if not isinstance(results, list) or result_count != completed_points:
        invalid(path, f"results count {result_count!r} != {completed_points!r}")
    observed_ranks = set()
    for result in results:
        if not isinstance(result, dict):
            invalid(path, "result entry must be an object")
        point = result.get("point")
        point_type = point.get("point_type") if isinstance(point, dict) else None
        if point_type not in allowed_point_types:
            invalid(path, f"point type {point_type!r} is not valid for {expected_mode!r}")
        fpms = result.get("fpms")
        if not isinstance(fpms, list) or not fpms:
            invalid(path, "each result must contain at least one FPM sample")
        for fpm in fpms:
            rank = fpm.get("dp_rank") if isinstance(fpm, dict) else None
            if type(rank) is not int:
                invalid(path, f"invalid FPM dp_rank {rank!r}")
            observed_ranks.add(rank)
    if observed_ranks != {expected_rank}:
        invalid(path, f"FPM dp_ranks {sorted(observed_ranks)!r} != [{expected_rank}]")
PY
}

# The generated run.sh foreground-execs
# `python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])'`,
# so this background pid IS the engine's pid and, after setsid, its own
# process-group id: terminate_engine's group kill reaches every engine worker.
# Keep the streamed copy of the engine output quiet: tqdm-style progress
# floods (8 workers x dozens of updates/s during flashinfer autotune) add
# megabytes of bar redraws to the proxied kubectl exec channel and drown the
# failure evidence the collector quotes when a run dies. The /results logs
# keep the complete output; only the exec stream drops progress-bar lines.
# One awk per stream writes both copies so a dead stream reader cannot
# SIGPIPE the file writer.
engine_stream_noise='[0-9]+%[|]|[|] *[0-9]+/[0-9]+ [[]'
engine_stream_filter='{ print > logfile; fflush(logfile); if ($0 !~ noise) { print; fflush() } }'
bash "${workdir}/run.sh" \
  > >(awk -v logfile=/results/engine.stdout.log -v noise="${engine_stream_noise}" "${engine_stream_filter}") \
  2> >(awk -v logfile=/results/engine.stderr.log -v noise="${engine_stream_noise}" "${engine_stream_filter}" >&2) &
engine_pid=$!

if (( fpm_is_follower )); then
  # Headless followers never write results and normally exit only when the
  # leader's teardown collapses the distributed group. Treat exactly that
  # teardown (leader etcd gone) as success so a completed multinode
  # measurement is not recorded as a follower failure; an engine crash while
  # the leader is still alive stays a real failure.
  #
  # Do NOT block on engine exit: the headless engine has been observed to
  # hang forever when the DP master vanishes (leader crash and normal
  # completion alike), burning the runner's whole exec budget. Actively
  # probe the leader's etcd; once it stays gone, terminate the local engine
  # and classify by the same leader-teardown rule.
  leader_gone_probes=0
  while kill -0 "$engine_pid" 2>/dev/null; do
    if ! (exec 3<>"/dev/tcp/${FPM_MASTER_ADDR}/2379") 2>/dev/null; then
      leader_gone_probes=$((leader_gone_probes + 1))
    else
      leader_gone_probes=0
    fi
    if (( leader_gone_probes >= 3 )); then
      echo "Leader etcd is gone while the headless engine is still running; terminating engine and reporting success" >&2
      terminate_engine "$engine_pid"
      engine_pid=""
      exit 0
    fi
    sleep 5
  done
  set +e
  wait "$engine_pid"
  headless_status=$?
  set -e
  engine_pid=""
  if (( headless_status == 0 )); then
    exit 0
  fi
  teardown_deadline=$((SECONDS + 90))
  while (( SECONDS < teardown_deadline )); do
    if ! (exec 3<>"/dev/tcp/${FPM_MASTER_ADDR}/2379") 2>/dev/null; then
      echo "Headless engine exited after leader teardown; reporting success" >&2
      exit 0
    fi
    sleep 2
  done
  echo "Headless engine exited (status ${headless_status}) while the leader was still alive" >&2
  exit "$headless_status"
fi

deadline=$((SECONDS + FPM_WAIT_TIMEOUT_SECONDS))

while true; do
  set +e
  check_result_files
  result_status=$?
  set -e
  if (( result_status == 0 )); then
    break
  fi
  if (( result_status == 20 )); then
    exit 1
  fi
  if (( result_status != 10 )); then
    # 10 is the checker's only keep-waiting signal; any other status
    # (python traceback, exec failure, OOM-kill) is checker breakage
    # and must fail now instead of burning the whole wait deadline.
    echo "FPM result checker failed with unexpected status ${result_status}" >&2
    exit 1
  fi
  if ! kill -0 "$engine_pid" 2>/dev/null; then
    set +e
    wait "$engine_pid"
    engine_status=$?
    set -e
    terminate_engine "$engine_pid"
    engine_pid=""
    echo "Engine exited before writing all FPM benchmark outputs" >&2
    if (( engine_status == 0 )); then exit 1; else exit "$engine_status"; fi
  fi
  if (( SECONDS >= deadline )); then
    echo "Timed out waiting for all FPM benchmark outputs" >&2
    exit 124
  fi
  sleep 2
done

if (( FPM_NODE_COUNT > 1 && FPM_DATA_PARALLEL_SIZE > 1 )); then
  # Every DP rank writes results on its own node, but this pod's gate
  # only watched the local files. Rank 0's engine is the DP coordinator
  # and rank 0's runtime owns etcd, so tearing either down while another
  # rank is still finalizing destroys that rank's results. Rendezvous
  # before teardown; a barrier timeout proceeds with a warning so a
  # genuinely crashed follower (whose own script already failed the
  # cell) cannot hang the leader.
  barrier_port=29511
  if (( FPM_NODE_RANK == 0 )); then
    barrier_timeout="${FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS:-180}"
    python3 - "$((FPM_NODE_COUNT - 1))" "$barrier_port" "$barrier_timeout" <<'PY'
import socket
import sys
import time

expected = int(sys.argv[1])
port = int(sys.argv[2])
timeout_seconds = float(sys.argv[3])
seen = set()
try:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(max(expected, 1))
    server.settimeout(2.0)
    deadline = time.monotonic() + timeout_seconds
    while len(seen) < expected and time.monotonic() < deadline:
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        with conn:
            conn.settimeout(5.0)
            try:
                data = conn.recv(64)
            except OSError:
                continue
        if data:
            seen.add(data.decode("utf-8", "replace").strip())
except Exception as error:
    print(f"FPM completion barrier disabled by error: {error!r}", file=sys.stderr)
if len(seen) < expected:
    print(
        f"FPM completion barrier timed out: {len(seen)}/{expected} followers reported",
        file=sys.stderr,
    )
PY
  else
    barrier_timeout="${FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS:-120}"
    python3 - "$FPM_MASTER_ADDR" "$barrier_port" "$FPM_NODE_RANK" "$barrier_timeout" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
rank = sys.argv[3]
deadline = time.monotonic() + float(sys.argv[4])
while time.monotonic() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2) as conn:
            conn.sendall(rank.encode())
        break
    except OSError:
        time.sleep(1)
PY
  fi
fi

terminate_engine "$engine_pid"
engine_pid=""
# Deliberately leave the EXIT trap armed: cleanup owns the tail. It finds
# engine_pid empty, stops rank 0's etcd, and preserves this exit status.
