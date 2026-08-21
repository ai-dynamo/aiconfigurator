# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior and contract tests for the collector-owned in-pod runtime.

``fpm_exec.sh`` owns everything the thick generated run.sh used to decide:
result gating, the schema-v2 checker, refuse-to-overwrite, engine teardown,
follower exit classification, and the DP completion barrier. These tests run
the whole script against a stub ``fpm_env.sh``/``run.sh``/``preflight.py``
and a PATH-shadowed ``etcd``, reusing the baseline test input vectors.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiconfigurator.fpm_contract import (
    FPM_ENV_EXPORTED_VARS,
    FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
    fpm_expected_result_paths,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FPM_EXEC = REPO_ROOT / "collector/fpm_forward/runtime/fpm_exec.sh"

# The stubbed etcd binary keeps the pid visible for liveness assertions and
# listens on the staged per-test client port so the readiness probe passes
# immediately. The bind is fatal: every test owns a freshly allocated free
# port, so a failure is a real defect, never something to tolerate.
_ETCD_STUB = """#!/bin/bash
if [[ -n "${FPM_ETCD_TRACE:-}" ]]; then
  printf '%s\\n' "$$" "$@" > "${FPM_ETCD_TRACE}"
fi
exec python3 - <<'PY'
import socket
import time

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", 2379))
server.listen(8)
time.sleep(3600)
PY
"""

# Follower-path tests never reach the checker, so python3 can be shadowed the
# way the baseline wrapper tests shadowed it: swallow heredoc stdin, exit 0.
_PYTHON3_SHADOW = """#!/bin/bash
if [[ "${1:-}" == "-" ]]; then
  /bin/cat >/dev/null
fi
exit 0
"""


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _free_port() -> int:
    """Allocate a free localhost port whose decimal form cannot collide with
    the script's own port literals, keeping every counted text replacement
    exact. Fixed 2379/2380/29511 are intentional in production (one cell per
    pod); the tests must not depend on them being free on the host."""

    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        text = str(port)
        if not any(literal in text for literal in ("2379", "2380", "29511")):
            return port


def _benchmark_result(
    *,
    mode: str = "prefill",
    dp_rank: int = 0,
    point_types: list[str] | None = None,
    schema_version: int = FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
    status: str = "complete",
    valid: bool = True,
) -> dict:
    if point_types is None:
        point_types = ["prefill" if mode == "agg" else mode]
    results = [
        {
            "point": {"point_type": point_type},
            "fpms": [{"dp_rank": dp_rank, "wall_time": 0.001}],
        }
        for point_type in point_types
    ]
    return {
        "schema_version": schema_version,
        "status": status,
        "valid": valid,
        "coverage": {
            "expected_points": len(results),
            "completed_points": len(results),
            "skipped_points": 0,
        },
        "config": {"mode": mode},
        "results": results,
        "skipped_points": [],
        "errors": [] if valid else ["boom"],
    }


def _writer_engine_body(files: dict[str, object], *, delay: float = 0.0) -> str:
    serialized = json.dumps(dict(files))
    return f"""\
import json
import pathlib
import signal
import sys
import time

for path, payload in json.loads({serialized!r}).items():
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload))
    time.sleep({delay})
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""


def _setsid_run_script(tmp_path: Path, engine_body: str) -> str:
    """A run.sh with the generated thin script's exact launch shape."""

    engine = tmp_path / "fake_engine.py"
    engine.write_text(engine_body)
    return (
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "exec python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "
        f"python3 {shlex.quote(str(engine))}\n"
    )


def _stage(
    tmp_path: Path,
    *,
    run_script: str,
    env_overrides: dict[str, object] | None = None,
    engine_grace_seconds: int | None = None,
    teardown_seconds: int | None = None,
    shadow_python3: bool = False,
    etcd_stub: str | None = None,
) -> SimpleNamespace:
    workdir = tmp_path / "fpm-bench"
    results = tmp_path / "results"
    bin_dir = tmp_path / "bin"
    for directory in (workdir, results, bin_dir):
        directory.mkdir(exist_ok=True)

    etcd_port = _free_port()
    peer_port = _free_port()
    barrier_port = _free_port()

    script = FPM_EXEC.read_text()
    assert script.count("workdir=/tmp/fpm-bench") == 1
    script = script.replace("workdir=/tmp/fpm-bench", f"workdir={workdir}")
    assert script.count("/results/") == 4
    script = script.replace("/results/", f"{results}/")
    assert script.count("2380") == 1
    script = script.replace("2380", str(peer_port))
    assert script.count("2379") == 8
    script = script.replace("2379", str(etcd_port))
    assert script.count("barrier_port=29511") == 1
    script = script.replace("barrier_port=29511", f"barrier_port={barrier_port}")
    if engine_grace_seconds is not None:
        assert script.count("engine_shutdown_grace_seconds=30") == 1
        script = script.replace(
            "engine_shutdown_grace_seconds=30",
            f"engine_shutdown_grace_seconds={engine_grace_seconds}",
        )
    if teardown_seconds is not None:
        assert script.count("teardown_deadline=$((SECONDS + 90))") == 1
        script = script.replace(
            "teardown_deadline=$((SECONDS + 90))",
            f"teardown_deadline=$((SECONDS + {teardown_seconds}))",
        )
    script_path = tmp_path / "fpm_exec.sh"
    script_path.write_text(script)

    values: dict[str, object] = {
        "FPM_NODE_COUNT": 1,
        "FPM_DATA_PARALLEL_SIZE": 1,
        "FPM_LOCAL_DATA_PARALLEL_SIZE": 1,
        "FPM_BENCHMARK_MODE": "prefill",
        "FPM_BENCHMARK_OUTPUT_PATH": str(results / "benchmark.json"),
        "FPM_WAIT_TIMEOUT_SECONDS": 30,
        "FPM_RESULT_SCHEMA_VERSION": FPM_NATIVE_BENCHMARK_RESULT_SCHEMA_VERSION,
        "FPM_NODE_RANK": 0,
        "FPM_MASTER_ADDR": "127.0.0.1",
    }
    values.update(env_overrides or {})
    exports = "\n".join(f"export {name}={shlex.quote(str(values[name]))}" for name in FPM_ENV_EXPORTED_VARS)
    (workdir / "fpm_env.sh").write_text(f"#!/usr/bin/env bash\n{exports}\n")
    (workdir / "preflight.py").write_text("")
    (workdir / "run.sh").write_text(run_script)
    if etcd_stub is None:
        etcd_stub = _ETCD_STUB
        assert etcd_stub.count("2379") == 1
        etcd_stub = etcd_stub.replace("2379", str(etcd_port))
    _write_executable(bin_dir / "etcd", etcd_stub)
    if shadow_python3:
        _write_executable(bin_dir / "python3", _PYTHON3_SHADOW)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FPM_ETCD_TRACE"] = str(tmp_path / "etcd.trace")
    return SimpleNamespace(
        script=script_path,
        workdir=workdir,
        results=results,
        env=env,
        etcd_trace=tmp_path / "etcd.trace",
        benchmark_output=Path(str(values["FPM_BENCHMARK_OUTPUT_PATH"])),
        etcd_port=etcd_port,
        peer_port=peer_port,
        barrier_port=barrier_port,
    )


def _run(staged: SimpleNamespace, *, timeout: int = 60, extra_env: dict[str, str] | None = None):
    env = dict(staged.env)
    env.update(extra_env or {})
    return subprocess.run(
        ["bash", str(staged.script)],
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        check=False,
    )


def _process_is_running(process_pid: int) -> bool:
    """Return false for an exited process, including an unreaped Linux zombie."""
    try:
        os.kill(process_pid, 0)
    except ProcessLookupError:
        return False

    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return True

    try:
        status_lines = (proc_root / str(process_pid) / "status").read_text().splitlines()
    except FileNotFoundError:
        return False
    except OSError:
        return True

    state = next((line for line in status_lines if line.startswith("State:")), None)
    if state is None:
        return True
    state_parts = state.split()
    return len(state_parts) < 2 or state_parts[1] not in {"Z", "X", "x"}


def _assert_process_stopped(pid_path: Path) -> None:
    process_pid = int(pid_path.read_text())
    exit_deadline = time.monotonic() + 2
    while time.monotonic() < exit_deadline:
        if not _process_is_running(process_pid):
            return
        time.sleep(0.05)

    if not _process_is_running(process_pid):
        return
    try:
        os.kill(process_pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    pytest.fail(f"process {process_pid} survived fpm_exec.sh cleanup")


# ---------------------------------------------------------------------------
# schema-v2 checker matrix (baseline vectors from the thick run.sh tests)
# ---------------------------------------------------------------------------


def _run_with_static_result(tmp_path: Path, result: object, *, benchmark_mode: str = "prefill"):
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={"FPM_BENCHMARK_MODE": benchmark_mode},
    )
    body = _writer_engine_body({str(staged.benchmark_output): result})
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))
    return staged, _run(staged)


@pytest.mark.parametrize(
    ("benchmark_mode", "point_types"),
    [
        ("prefill", ["prefill"]),
        ("decode", ["decode"]),
        ("agg", ["prefill", "decode"]),
    ],
)
def test_fpm_exec_accepts_schema_v2_for_supported_benchmark_modes(tmp_path, benchmark_mode, point_types):
    result = _benchmark_result(mode=benchmark_mode, point_types=point_types)

    _, completed = _run_with_static_result(tmp_path, result, benchmark_mode=benchmark_mode)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("schema", "schema_version=3"),
        ("mode", "benchmark mode 'decode' != 'prefill'"),
        ("coverage", "invalid coverage"),
        ("result_count", "results count 0 != 1"),
        ("point_type", "point type 'decode' is not valid for 'prefill'"),
        ("dp_rank", "FPM dp_ranks [1] != [0]"),
        ("non_object", "top-level JSON must be an object"),
    ],
)
def test_fpm_exec_rejects_invalid_schema_v2_result(tmp_path, case, expected_message):
    result = _benchmark_result()
    payload: object = result
    if case == "schema":
        result["schema_version"] = 3
    elif case == "mode":
        result["config"]["mode"] = "decode"
    elif case == "coverage":
        result["coverage"]["completed_points"] = 0
    elif case == "result_count":
        result["results"] = []
    elif case == "point_type":
        result["results"][0]["point"]["point_type"] = "decode"
    elif case == "dp_rank":
        result["results"][0]["fpms"][0]["dp_rank"] = 1
    elif case == "non_object":
        payload = []
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(f"unknown test case: {case}")

    _, completed = _run_with_static_result(tmp_path, payload)

    assert completed.returncode == 1
    assert expected_message in completed.stderr


@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (_benchmark_result(schema_version=1), "schema_version=1"),
        (_benchmark_result(mode="decode"), "benchmark mode 'decode' != 'prefill'"),
        (_benchmark_result(dp_rank=1), "FPM dp_ranks [1] != [0]"),
    ],
)
def test_fpm_exec_rejects_mismatched_result_identity(tmp_path, result, expected_message):
    _, completed = _run_with_static_result(tmp_path, result)

    assert completed.returncode == 1
    assert expected_message in completed.stderr


def test_fpm_exec_rejects_terminal_failed_result(tmp_path):
    _, completed = _run_with_static_result(tmp_path, _benchmark_result(status="failed", valid=False))

    assert completed.returncode == 1
    assert "status='failed'" in completed.stderr
    assert "valid=False" in completed.stderr
    assert "boom" in completed.stderr


def test_fpm_exec_rejects_legacy_schema_v2_envelope_and_refuses_overwrite(tmp_path):
    legacy = {"schema_version": 2, "status": "passed", "config": {"dp_rank": 0}}

    staged, completed = _run_with_static_result(tmp_path, legacy)

    assert completed.returncode == 1
    assert "status='passed' valid=None" in completed.stderr
    assert json.loads(staged.benchmark_output.read_text()) == legacy

    # The refuse-to-overwrite check runs BEFORE the engine is launched.
    repeated = _run(staged)
    assert repeated.returncode == 1
    assert "Refusing to overwrite" in repeated.stderr


@pytest.mark.parametrize(
    ("written_schema_version", "expected_returncode"),
    [
        pytest.param(2, 1, id="contract-default-version-rejected"),
        pytest.param(3, 0, id="declared-version-accepted"),
    ],
)
def test_fpm_exec_checker_uses_the_declared_result_schema_version(
    tmp_path,
    written_schema_version,
    expected_returncode,
):
    """The checker must compare against FPM_RESULT_SCHEMA_VERSION from
    fpm_env.sh, not a hardcoded 2: with the env declaring version 3, a v2
    payload fails and the identical v3 payload passes."""

    staged = _stage(tmp_path, run_script="", env_overrides={"FPM_RESULT_SCHEMA_VERSION": 3})
    body = _writer_engine_body({str(staged.benchmark_output): _benchmark_result(schema_version=written_schema_version)})
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))

    completed = _run(staged)

    assert completed.returncode == expected_returncode, completed.stderr
    if expected_returncode:
        assert "schema_version" in completed.stderr


def test_fpm_exec_waits_for_every_single_node_dp_result(tmp_path):
    output = tmp_path / "results" / "benchmark.json"
    files = {}
    for rank in range(4):
        path = output if rank == 0 else output.with_name(f"benchmark_dp{rank}.json")
        payload = _benchmark_result(dp_rank=rank)
        payload["rank"] = rank
        files[str(path)] = payload
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={
            "FPM_DATA_PARALLEL_SIZE": 4,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": 4,
        },
    )
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, _writer_engine_body(files, delay=1.0)))

    completed = _run(staged)

    assert completed.returncode == 0, completed.stderr
    assert [json.loads(Path(path).read_text())["rank"] for path in files] == [0, 1, 2, 3]


def test_engine_progress_bars_are_kept_out_of_the_exec_stream(tmp_path):
    """tqdm-style progress floods (flashinfer autotune, safetensors loading)
    bloat the proxied kubectl exec channel and drown the failure evidence the
    collector quotes from it. The streamed copy of the engine output must drop
    bar lines while the /results logs keep the complete output."""

    stdout_bar = (
        "[AutoTuner]: Tuning flashinfer::trtllm_fp4_block_scale_moe:"
        "  19%|#9        | 4/21 [00:38<02:45,  9.72s/profile]"
    )
    stderr_bar = "Loading safetensors checkpoint shards:   4% Completed | 2/47 [00:21<06:43,  8.96s/it]"
    staged = _stage(tmp_path, run_script="")
    payload = json.dumps(_benchmark_result())
    engine_body = f"""\
import pathlib
import signal
import sys
import time

print({stdout_bar!r})
print("engine stdout survives the stream filter")
print({stderr_bar!r}, file=sys.stderr)
print("engine stderr survives the stream filter", file=sys.stderr)
sys.stdout.flush()
sys.stderr.flush()
pathlib.Path({str(staged.benchmark_output)!r}).write_text({payload!r})
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, engine_body))

    completed = _run(staged)

    assert completed.returncode == 0, completed.stderr
    streamed = completed.stdout + completed.stderr
    assert "engine stdout survives the stream filter" in streamed
    assert "engine stderr survives the stream filter" in streamed
    assert "AutoTuner" not in streamed
    assert "% Completed |" not in streamed
    stdout_log = (staged.benchmark_output.parent / "engine.stdout.log").read_text()
    stderr_log = (staged.benchmark_output.parent / "engine.stderr.log").read_text()
    assert stdout_bar in stdout_log
    assert "engine stdout survives the stream filter" in stdout_log
    assert stderr_bar in stderr_log
    assert "engine stderr survives the stream filter" in stderr_log


# ---------------------------------------------------------------------------
# engine lifecycle: early exit, stubborn shutdown, process-group kill, timeout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine_exit", "expected_returncode"),
    [
        (0, 1),
        (23, 23),
    ],
    ids=["clean-exit-without-results", "engine-crash-status-passthrough"],
)
def test_fpm_exec_engine_early_exit_propagates_status(tmp_path, engine_exit, expected_returncode):
    staged = _stage(tmp_path, run_script=f"#!/usr/bin/env bash\nexit {engine_exit}\n")

    completed = _run(staged)

    assert completed.returncode == expected_returncode
    assert "Engine exited before writing all FPM benchmark outputs" in completed.stderr


def test_fpm_exec_times_out_waiting_for_results(tmp_path):
    staged = _stage(tmp_path, run_script="", env_overrides={"FPM_WAIT_TIMEOUT_SECONDS": 1})
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, _writer_engine_body({})))

    completed = _run(staged)

    assert completed.returncode == 124
    assert "Timed out waiting for all FPM benchmark outputs" in completed.stderr


def test_fpm_exec_leader_fails_fast_when_etcd_dies_with_log_evidence(tmp_path):
    """etcd startup failures (missing binary flag, bound port, data-dir
    permissions) must fail the cell immediately with the etcd exit code and
    the log tail on stderr, not burn the 120s readiness budget and report
    only a timeout."""

    dying_stub = "#!/bin/bash\necho 'etcd: listen tcp: bind: address already in use' >&2\nexit 3\n"
    staged = _stage(tmp_path, run_script="", etcd_stub=dying_stub)

    started = time.monotonic()
    completed = _run(staged, timeout=60)
    elapsed = time.monotonic() - started

    assert completed.returncode == 1
    assert elapsed < 30
    assert "etcd exited with status 3 before becoming ready" in completed.stderr
    assert "address already in use" in completed.stderr
    assert "etcd readiness timeout" not in completed.stderr


@pytest.mark.parametrize(
    ("result_valid", "expected_returncode"),
    [
        (True, 0),
        (False, 1),
    ],
)
def test_fpm_exec_bounds_stubborn_engine_shutdown(tmp_path, result_valid, expected_returncode):
    result = _benchmark_result(valid=result_valid)
    pid_path = tmp_path / "engine.pid"
    child_pid_path = tmp_path / "engine-child.pid"
    staged = _stage(tmp_path, run_script="", engine_grace_seconds=1)
    engine_body = f"""\
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path(os.environ["FAKE_ENGINE_PID_PATH"]).write_text(str(os.getpid()))
child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
pathlib.Path(os.environ["FAKE_ENGINE_CHILD_PID_PATH"]).write_text(str(child.pid))
path = pathlib.Path({str(staged.benchmark_output)!r})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text({json.dumps(result)!r})
while True:
    time.sleep(0.1)
"""
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, engine_body))

    started = time.monotonic()
    completed = _run(
        staged,
        timeout=60,
        extra_env={
            "FAKE_ENGINE_PID_PATH": str(pid_path),
            "FAKE_ENGINE_CHILD_PID_PATH": str(child_pid_path),
        },
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == expected_returncode
    assert elapsed < 30
    assert "Engine did not stop within 1s; sending SIGKILL" in completed.stderr
    for process_pid_path in (pid_path, child_pid_path):
        _assert_process_stopped(process_pid_path)


def test_fpm_exec_cleans_process_group_when_engine_parent_exits(tmp_path):
    child_pid_path = tmp_path / "engine-child.pid"
    staged = _stage(tmp_path, run_script="", engine_grace_seconds=1)
    engine_body = """\
import os
import pathlib
import subprocess
import sys

child = subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(3600)",
    ],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
pathlib.Path(os.environ["FAKE_ENGINE_CHILD_PID_PATH"]).write_text(str(child.pid))
raise SystemExit(23)
"""
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, engine_body))

    try:
        completed = _run(staged, extra_env={"FAKE_ENGINE_CHILD_PID_PATH": str(child_pid_path)})
    except subprocess.TimeoutExpired:
        if child_pid_path.exists():
            os.kill(int(child_pid_path.read_text()), signal.SIGKILL)
        raise

    assert completed.returncode == 23
    assert "Engine exited before writing all FPM benchmark outputs" in completed.stderr
    assert "Engine did not stop within 1s; sending SIGKILL" in completed.stderr
    _assert_process_stopped(child_pid_path)


# ---------------------------------------------------------------------------
# follower exit classification (multinode, data_parallel_size == 1, rank > 0)
# ---------------------------------------------------------------------------

_FOLLOWER_ENV = {
    "FPM_NODE_COUNT": 2,
    "FPM_DATA_PARALLEL_SIZE": 1,
    "FPM_NODE_RANK": 1,
}


def test_fpm_exec_follower_clean_engine_exit_is_success(tmp_path):
    staged = _stage(
        tmp_path,
        run_script="#!/usr/bin/env bash\nexit 0\n",
        env_overrides=_FOLLOWER_ENV,
        shadow_python3=True,
    )

    completed = _run(staged, timeout=30)

    assert completed.returncode == 0, completed.stderr


def test_fpm_exec_follower_ignores_stale_benchmark_file(tmp_path):
    """Followers never write results, so a stale per-rank file left on a
    follower volume must not trip the refuse-to-overwrite check (baseline
    parity: the thick run.sh never gated followers on files either)."""

    staged = _stage(
        tmp_path,
        run_script="#!/usr/bin/env bash\nexit 0\n",
        env_overrides=_FOLLOWER_ENV,
        shadow_python3=True,
    )
    stale = staged.benchmark_output.with_name("benchmark_dp1.json")
    stale.write_text(json.dumps(_benchmark_result(dp_rank=1)))

    completed = _run(staged, timeout=30)

    assert completed.returncode == 0, completed.stderr
    assert "Refusing to overwrite" not in completed.stderr


def test_fpm_exec_follower_crash_after_leader_teardown_is_success(tmp_path):
    # Nothing listens on the freshly allocated staged etcd port, so the
    # leader-teardown classification is simulated deterministically.
    staged = _stage(
        tmp_path,
        run_script="#!/usr/bin/env bash\nexit 7\n",
        env_overrides=_FOLLOWER_ENV,
        shadow_python3=True,
    )

    completed = _run(staged, timeout=30)

    assert completed.returncode == 0
    assert "Headless engine exited after leader teardown; reporting success" in completed.stderr


def test_fpm_exec_follower_watchdog_terminates_a_hung_engine(tmp_path):
    """A headless engine that never exits when the leader vanishes must be
    actively terminated by the follower watchdog and classified as success —
    not left to burn the runner's exec budget. Regression guard: the watchdog
    must pass the pid to terminate_engine, or the reap crashes the script
    under set -e and the pod is misrecorded as failed."""

    # A long-lived engine that exits cleanly on SIGTERM; nothing listens on the
    # staged etcd port, so the follower watchdog sees the leader gone and must
    # actively terminate it. shadow_python3 stubs the leader-readiness probe so
    # the run reaches the follower branch deterministically.
    hung_engine = "#!/usr/bin/env bash\ntrap 'exit 0' TERM\nwhile true; do sleep 0.5; done\n"
    staged = _stage(
        tmp_path,
        run_script=hung_engine,
        env_overrides=_FOLLOWER_ENV,
        shadow_python3=True,
    )

    completed = _run(staged, timeout=60)

    assert completed.returncode == 0, completed.stderr
    assert "terminating engine and reporting success" in completed.stderr


def test_fpm_exec_follower_crash_while_leader_alive_stays_a_failure(tmp_path):
    # Staging only writes files, so it can run first to learn the etcd port
    # this test's leader stand-in must occupy.
    staged = _stage(
        tmp_path,
        run_script="#!/usr/bin/env bash\nexit 7\n",
        env_overrides=_FOLLOWER_ENV,
        shadow_python3=True,
        teardown_seconds=3,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", staged.etcd_port))
        listener.listen(8)

        completed = _run(staged, timeout=30)
    finally:
        listener.close()

    assert completed.returncode == 7
    assert "Headless engine exited (status 7) while the leader was still alive" in completed.stderr


# ---------------------------------------------------------------------------
# DP completion barrier (multinode, data_parallel_size > 1)
# ---------------------------------------------------------------------------


def test_fpm_exec_dp_leader_rendezvouses_with_follower_barrier_report(tmp_path):
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={
            "FPM_NODE_COUNT": 2,
            "FPM_DATA_PARALLEL_SIZE": 2,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": 1,
            "FPM_NODE_RANK": 0,
        },
    )
    body = _writer_engine_body({str(staged.benchmark_output): _benchmark_result(dp_rank=0)})
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))

    def report_completion() -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", staged.barrier_port), timeout=1) as conn:
                    conn.sendall(b"1")
                return
            except OSError:
                time.sleep(0.1)

    reporter = threading.Thread(target=report_completion, daemon=True)
    reporter.start()
    started = time.monotonic()
    completed = _run(staged, extra_env={"FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS": "25"})
    elapsed = time.monotonic() - started
    reporter.join(timeout=5)

    assert completed.returncode == 0, completed.stderr
    assert "FPM completion barrier timed out" not in completed.stderr
    assert elapsed < 20


def test_fpm_exec_dp_leader_barrier_timeout_warns_and_proceeds(tmp_path):
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={
            "FPM_NODE_COUNT": 2,
            "FPM_DATA_PARALLEL_SIZE": 2,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": 1,
            "FPM_NODE_RANK": 0,
        },
    )
    body = _writer_engine_body({str(staged.benchmark_output): _benchmark_result(dp_rank=0)})
    (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))

    completed = _run(staged, extra_env={"FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS": "1"})

    assert completed.returncode == 0, completed.stderr
    assert "FPM completion barrier timed out: 0/1 followers reported" in completed.stderr


def test_fpm_exec_dp_follower_reports_its_rank_to_the_leader_barrier(tmp_path):
    # The follower probes the leader's etcd port during readiness and its
    # barrier port at completion; stage first (staging only writes files) so
    # the localhost stand-ins know which ports to occupy.
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={
            "FPM_NODE_COUNT": 2,
            "FPM_DATA_PARALLEL_SIZE": 2,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": 1,
            "FPM_NODE_RANK": 1,
        },
    )
    etcd_stand_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    etcd_stand_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    barrier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    barrier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    received: list[bytes] = []

    def accept_report() -> None:
        barrier.settimeout(30)
        try:
            conn, _ = barrier.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5)
            received.append(conn.recv(64))

    try:
        etcd_stand_in.bind(("127.0.0.1", staged.etcd_port))
        etcd_stand_in.listen(8)
        barrier.bind(("127.0.0.1", staged.barrier_port))
        barrier.listen(8)
        acceptor = threading.Thread(target=accept_report, daemon=True)
        acceptor.start()

        rank_path = staged.benchmark_output.with_name("benchmark_dp1.json")
        body = _writer_engine_body({str(rank_path): _benchmark_result(dp_rank=1)})
        (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, body))

        completed = _run(staged)
        acceptor.join(timeout=10)
    finally:
        barrier.close()
        etcd_stand_in.close()

    assert completed.returncode == 0, completed.stderr
    assert received == [b"1"]


# ---------------------------------------------------------------------------
# contract pins
# ---------------------------------------------------------------------------


def test_fpm_exec_consumes_only_contract_environment():
    """Everything fpm_exec.sh needs from the Generator travels through the
    fpm_env.sh exports; the only extra FPM_* input is the operator-tunable
    barrier timeout, which is defaulted in-script and never rendered."""

    script = FPM_EXEC.read_text()
    consumed = set(re.findall(r"FPM_[A-Z0-9_]+", script))

    assert {"FPM_NODE_RANK", "FPM_MASTER_ADDR", "FPM_BENCHMARK_OUTPUT_PATH"} <= consumed
    allowed = set(FPM_ENV_EXPORTED_VARS) | {"FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS"}
    assert consumed <= allowed, sorted(consumed - allowed)


@pytest.mark.parametrize(
    ("output_name", "node_rank", "local_size", "topology_env"),
    [
        pytest.param(
            "run.v1/metrics.final.json",
            0,
            2,
            {"FPM_NODE_COUNT": 1, "FPM_DATA_PARALLEL_SIZE": 2},
            id="dotted-name-rank0",
        ),
        pytest.param(
            "plain-noext",
            1,
            2,
            {"FPM_NODE_COUNT": 2, "FPM_DATA_PARALLEL_SIZE": 4},
            id="extensionless-nonzero-node-rank",
        ),
    ],
)
def test_fpm_exec_result_naming_matches_contract_on_shared_vectors(
    tmp_path,
    output_name,
    node_rank,
    local_size,
    topology_env,
):
    """The gate only opens when the engine writes files at exactly the paths
    the shell derives; feeding it files at the CONTRACT-derived paths pins the
    two naming implementations against each other end to end."""

    output_path = tmp_path / "results" / output_name
    staged = _stage(
        tmp_path,
        run_script="",
        env_overrides={
            **topology_env,
            "FPM_LOCAL_DATA_PARALLEL_SIZE": local_size,
            "FPM_NODE_RANK": node_rank,
            "FPM_BENCHMARK_OUTPUT_PATH": str(output_path),
        },
    )
    etcd_stand_in = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    etcd_stand_in.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        if node_rank > 0:
            etcd_stand_in.bind(("127.0.0.1", staged.etcd_port))
            etcd_stand_in.listen(8)
        expected_paths = fpm_expected_result_paths(str(output_path), node_rank, local_size)
        start_rank = node_rank * local_size
        files = {path: _benchmark_result(dp_rank=start_rank + offset) for offset, path in enumerate(expected_paths)}
        (staged.workdir / "run.sh").write_text(_setsid_run_script(tmp_path, _writer_engine_body(files)))

        completed = _run(staged, extra_env={"FPM_COMPLETION_BARRIER_TIMEOUT_SECONDS": "1"})
    finally:
        etcd_stand_in.close()

    assert completed.returncode == 0, completed.stderr
    for path in expected_paths:
        assert Path(path).is_file(), path
