# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess as sp
import sys

import pytest

from aiconfigurator.sdk.perf_database import (
    get_latest_database_version,
    get_supported_databases,
)

pytestmark = [pytest.mark.e2e, pytest.mark.build]

SANITY_CHECK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../tools/sanity_check"))

# Always-validated combo: keeps one loader + query smoke path (strict
# provenance, warm_all_op_data, bf16 coverage gate) alive on PRs that touch
# no perf data at all.
SENTINEL_COMBO = ("h200_sxm", "trtllm")

# CI scopes this suite to the PR's diff via AIC_SANITY_TARGETS (build-test.yml):
#   unset -> full matrix (local/manual runs keep the historical behavior)
#   "all" -> full matrix (sanity-check tooling or the DB loader changed)
#   ""    -> sentinel combo only (no perf data changed)
#   newline-separated changed paths under .../systems/ -> sentinel + the
#     (system, backend) combos those paths belong to


def _combos_from_changed_paths(changed_paths, supported):
    combos = set()
    for path in changed_paths:
        parts = [p for p in path.strip().split("/") if p]
        if "systems" not in parts:
            continue
        rest = parts[parts.index("systems") + 1 :]
        if not rest:
            continue
        if rest[0] == "data":
            if len(rest) < 2:
                continue
            system = rest[1]
        elif rest[0].endswith(".yaml"):
            # systems/<system>.yaml spec edits feed SOL math for every
            # backend of that system.
            system = rest[0].removesuffix(".yaml")
        else:
            continue
        backends = supported.get(system)
        if not backends:
            continue
        # Any data change expands to every supported backend of the system:
        # backend-specific rows can feed other backends through manifest-gated
        # cross-backend fill (shared-tier kernel sources, e.g. gdn_perf.parquet
        # collected under sglang is consumed by trtllm/vllm too).
        combos.update((system, backend) for backend in backends)
    return combos


def _selected_combos(supported):
    targets = os.environ.get("AIC_SANITY_TARGETS")
    if targets is None or targets.strip() == "all":
        return {(system, backend) for system, backends in supported.items() for backend in backends}
    combos = _combos_from_changed_paths(targets.splitlines(), supported)
    combos.add(SENTINEL_COMBO)
    return combos


def _supported_system_backend_latest():
    """(system, backend, latest_version) for each selected system+backend combo."""
    supported = get_supported_databases()
    result = []
    for system, backend in sorted(_selected_combos(supported)):
        if backend not in supported.get(system, {}):
            continue
        fail_ok = system in ["b60"]  # xpu
        version = get_latest_database_version(system, backend)
        if version is not None:
            result.append((system, backend, version, fail_ok))
    return result


@pytest.mark.parametrize(
    "system,backend,version,fail_ok",
    _supported_system_backend_latest(),
)
def test_validate_database(system, backend, version, fail_ok):
    """
    Test that validate_database.ipynb runs successfully for the latest
    backend version of each system+backend combination that AIC supports.
    """
    env = {
        **os.environ,
        "AIC_VALIDATE_SYSTEM": system,
        "AIC_VALIDATE_BACKEND": backend,
        "AIC_VALIDATE_VERSION": version,
        "MPLBACKEND": "agg",
    }
    try:
        result = sp.run(
            [sys.executable, "-c", "import import_ipynb; import validate_database"],
            cwd=SANITY_CHECK_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except sp.TimeoutExpired:
        error_message = f"validate_database timed out (300s) for {system}/{backend}/{version}"
        if fail_ok:
            pytest.xfail(error_message)
        pytest.fail(error_message)
    success = result.returncode == 0
    error_message = (
        f"validate_database failed for {system}/{backend}/{version}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    if fail_ok and not success:
        pytest.xfail(error_message)

    assert success, error_message


# ---------------------------------------------------------------------------
# Focused selector tests: pin the diff-scoping classification so a coverage
# regression in the selector is caught by the selector suite itself.
# ---------------------------------------------------------------------------

_TRIGGERS_FILE = os.path.join(SANITY_CHECK_DIR, "sanity_full_matrix_triggers.txt")

_FAKE_SUPPORTED = {
    "gb200": {"sglang": {}, "trtllm": {}, "vllm": {}},
    "h200_sxm": {"sglang": {}, "trtllm": {}, "vllm": {}},
    "l40s": {"sglang": {}, "trtllm": {}, "vllm": {}},
}


def test_selector_cross_backend_donor_expands_to_all_system_backends():
    # gdn_perf.parquet is collected under sglang but consumed by trtllm/vllm
    # via manifest-gated cross-backend fill (shared-tier kernel sources); the
    # selector must not narrow to the collecting backend.
    combos = _combos_from_changed_paths(
        ["aic-core/src/aiconfigurator_core/systems/data/gb200/linear_attention/sglang/0.5.14/gdn_perf.parquet"],
        _FAKE_SUPPORTED,
    )
    assert combos == {("gb200", "sglang"), ("gb200", "trtllm"), ("gb200", "vllm")}


def test_selector_system_yaml_expands_to_all_system_backends():
    combos = _combos_from_changed_paths(["aic-core/src/aiconfigurator_core/systems/l40s.yaml"], _FAKE_SUPPORTED)
    assert combos == {("l40s", backend) for backend in _FAKE_SUPPORTED["l40s"]}


def test_selector_ignores_unrelated_and_unknown_paths():
    combos = _combos_from_changed_paths(
        [
            "src/aiconfigurator/cli/main.py",
            "README.md",
            "aic-core/src/aiconfigurator_core/systems/support_matrix/foo.yaml",
            "aic-core/src/aiconfigurator_core/systems/data/unknown_system/gemm/trtllm/1.0/gemm_perf.parquet",
        ],
        _FAKE_SUPPORTED,
    )
    assert combos == set()


def test_selected_combos_empty_env_is_sentinel_only(monkeypatch):
    monkeypatch.setenv("AIC_SANITY_TARGETS", "")
    assert _selected_combos(_FAKE_SUPPORTED) == {SENTINEL_COMBO}


def test_selected_combos_unset_and_all_run_full_matrix(monkeypatch):
    full = {(system, backend) for system, backends in _FAKE_SUPPORTED.items() for backend in backends}
    monkeypatch.delenv("AIC_SANITY_TARGETS", raising=False)
    assert _selected_combos(_FAKE_SUPPORTED) == full
    monkeypatch.setenv("AIC_SANITY_TARGETS", "all")
    assert _selected_combos(_FAKE_SUPPORTED) == full


FULL_MATRIX_TRIGGER_PATHS = [
    # global source manifest: governs cross-backend/cross-system fill routing
    "aic-core/src/aiconfigurator_core/systems/op_kernel_source_manifest.yaml",
    # compiled Rust database, engine, operator, and Python-binding paths
    "aic-core/rust/aiconfigurator-core/src/perf_database/interpolation.rs",
    "aic-core/rust/aiconfigurator-core/src/engine/runtime.rs",
    "aic-core/rust/aiconfigurator-core/src/operators/attention.rs",
    "aic-core/rust/aiconfigurator-core/src/common/system_spec.rs",
    "aic-core/rust/aiconfigurator-core/src/py.rs",
    # Python loader facade + shared SDK types used by the notebook
    "aic-core/src/aiconfigurator_core/sdk/perf_database.py",
    "aic-core/src/aiconfigurator_core/sdk/engine.py",
    "aic-core/src/aiconfigurator_core/sdk/common.py",
    "aic-core/src/aiconfigurator_core/sdk/system_spec.py",
    "aic-core/src/aiconfigurator_core/sdk/operations/base.py",
    # the sanity tooling and this selector itself
    "tools/sanity_check/create_charts.py",
    "tools/sanity_check/sanity_full_matrix_triggers.txt",
    "tests/e2e/tools/test_sanity_check.py",
]
DIFF_SCOPED_PATHS = [
    "aic-core/src/aiconfigurator_core/systems/data/gb200/gemm/trtllm/1.3.0rc23/gemm_perf.parquet",
    "aic-core/src/aiconfigurator_core/systems/gb200.yaml",
    "src/aiconfigurator/cli/main.py",
]


def test_full_matrix_trigger_regex_classification():
    # The workflow reads the same triggers file (single source of truth), so
    # this pin runs inside the CI container even though .github/ is not
    # shipped into the image.
    with open(_TRIGGERS_FILE) as f:
        pattern = re.compile(f.read().strip())
    for path in FULL_MATRIX_TRIGGER_PATHS:
        assert pattern.search(path), f"expected full-matrix trigger: {path}"
    for path in DIFF_SCOPED_PATHS:
        assert not pattern.search(path), f"must stay diff-scoped: {path}"
