# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for the sibling-walking shared-layer loader in PerfDatabase.

Each test sets up a minimal synthetic perf-database tree under a tmp dir
(`<systems_root>/<sys>.yaml`, `<systems_root>/data/<sys>/...`) and then
asserts on the loaded dict structure of `PerfDatabase._gemm_data`. We avoid
going through `query_gemm()` because that pulls in interpolation/SOL math
unrelated to the loader's tier-merge behavior.

The loader inherits rows from sibling `<sys>/<framework>/<version>/<op_file>`
directories (cross-version and cross-backend) when the database is loaded in
HYBRID mode. Both `tier=shared` (named) and `tier=shared_fallback`
(`kernel_source=default`, framework-implicit, low-fidelity) rows are inherited;
HYBRID already accepts coarser fallbacks, so they're not gated separately.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase, _load_op_kernel_source_manifest_entries

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEMM_HEADER = "framework,version,device,op_name,kernel_source,gemm_dtype,m,n,k,latency\n"


def _write_gemm_csv(path: Path, rows: list[tuple[str, str, int, int, int, float]]) -> None:
    """Write a GEMM perf CSV. Each row is (framework, kernel_source, m, n, k, latency)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(_GEMM_HEADER)
        for framework, kernel_source, m, n, k, latency in rows:
            f.write(f"{framework},1.0,h100,gemm,{kernel_source},bfloat16,{m},{n},{k},{latency}\n")


def _make_system_yaml(systems_root: Path, system: str, data_dir_name: str = "data") -> None:
    """Write a minimal system YAML covering only fields the constructor reads."""
    (systems_root / f"{system}.yaml").write_text(
        f"""
data_dir: {data_dir_name}/{system}
gpu:
  mem_bw: 4800000000000
  mem_bw_empirical_scaling_factor: 0.8
  mem_empirical_constant_latency: 0.000003
  mem_capacity: 151397597184
  bfloat16_tc_flops: 989000000000000
  int8_tc_flops: 1978000000000000
  fp8_tc_flops: 1978000000000000
  power: 700
  sm_version: 90
node:
  num_gpus_per_node: 8
  inter_node_bw: 50000000000
  intra_node_bw: 450000000000
  pcie_bw: 64000000000
  p2p_latency: 0.00001
misc:
  nccl_mem:
    1: 0
    2: 358612992
    4: 411041792
    8: 411041792
  other_mem: 3758096384
  nccl_version: '2.26.2'
""".lstrip()
    )


def _make_manifest(
    systems_root: Path,
    entries: list[tuple[str, str, str, list[str]]],
) -> None:
    """Write op_kernel_source_manifest.yaml. Each entry is (op_file, kernel_source, tier, frameworks)."""
    lines = ["groups:"]
    for op_file, ks, tier, frameworks in entries:
        lines.extend(
            [
                f"  - op_file: {op_file}",
                f"    kernel_source: '{ks}'",
                f"    tier: {tier}",
                f"    frameworks: [{', '.join(frameworks)}]",
            ]
        )
    (systems_root / "op_kernel_source_manifest.yaml").write_text("\n".join(lines) + "\n")


@pytest.fixture
def env(tmp_path: Path) -> Path:
    """Set up an empty systems tree and return its root.

    Each test adds the CSVs and manifest it needs.
    """
    systems_root = tmp_path / "systems"
    systems_root.mkdir()
    _make_system_yaml(systems_root, "h100_sxm")
    # Required nccl data dir (loader reads it even if we don't test nccl)
    nccl_dir = systems_root / "data" / "h100_sxm" / "nccl" / "2.26.2"
    nccl_dir.mkdir(parents=True)
    (nccl_dir / "nccl_perf.txt").write_text(
        "framework,version,device,op_name,kernel_source,nccl_dtype,num_gpus,message_size,latency\n"
    )
    # Clear the manifest LRU cache so each test sees its own manifest.
    _load_op_kernel_source_manifest_entries.cache_clear()
    return systems_root


def _backend_csv(env: Path, backend: str = "trtllm", version: str = "1.0") -> Path:
    return env / "data" / "h100_sxm" / backend / version / "gemm_perf.txt"


def _build_db(systems_root: Path, *, database_mode: str | None = "HYBRID") -> PerfDatabase:
    """Build a PerfDatabase. Defaults to HYBRID so the shared layer is on, which is
    what most tests exercise. The off-by-default behavior in non-HYBRID modes is
    covered by `test_shared_layer_off_in_silicon_mode`.
    """
    return PerfDatabase(
        system="h100_sxm",
        backend="trtllm",
        version="1.0",
        systems_root=str(systems_root),
        database_mode=database_mode,
    )


def _gemm_lookup(db: PerfDatabase, m: int, n: int, k: int) -> float | None:
    """Read latency for a single (m, n, k) triple from the loaded gemm dict."""
    qmode = common.GEMMQuantMode["bfloat16"]
    table = db._gemm_data.data
    if qmode not in table or m not in table[qmode] or n not in table[qmode][m] or k not in table[qmode][m][n]:
        return None
    return table[qmode][m][n][k]["latency"]


# ---------------------------------------------------------------------------
# Loader scenarios
# ---------------------------------------------------------------------------


def test_backend_only(env: Path) -> None:
    """Existing behavior: only the active version contains rows, no siblings."""
    _write_gemm_csv(_backend_csv(env), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.5)])
    _make_manifest(env, [])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5


def test_shared_layer_off_in_silicon_mode(env: Path) -> None:
    """In any non-HYBRID database_mode (here: unset, which defaults to SILICON
    semantics), the shared layer is OFF and sibling rows are not consulted —
    preserves bit-for-bit compatibility with `main`.
    """
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.7)])
    _make_manifest(env, [("gemm_perf.txt", "torch_flow", "shared", ["trtllm"])])

    db = _build_db(env, database_mode=None)
    assert db.enable_shared_layer is False
    assert _gemm_lookup(db, 1024, 4096, 4096) is None


def test_shared_layer_on_in_hybrid_mode_with_fallback(env: Path) -> None:
    """`database_mode='HYBRID'` enables sibling inheritance for both `tier=shared`
    (named kernel) and `tier=shared_fallback` (`kernel_source=default`) rows
    without needing any extra flag — HYBRID already accepts coarser fallbacks.
    """
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.7)])
    _write_gemm_csv(_backend_csv(env, version="0.8"), [("trtllm", "default", 2048, 4096, 4096, 0.3)])
    _make_manifest(
        env,
        [
            ("gemm_perf.txt", "torch_flow", "shared", ["trtllm"]),
            ("gemm_perf.txt", "default", "shared_fallback", ["trtllm"]),
        ],
    )

    db = _build_db(env)  # database_mode=HYBRID by helper default
    assert db.enable_shared_layer is True
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.7  # shared
    assert _gemm_lookup(db, 2048, 4096, 4096) == 0.3  # shared_fallback


def test_sibling_only_with_no_active_data(env: Path) -> None:
    """Active version's file is empty; sibling version provides the row."""
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)  # header-only

    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.7)])
    _make_manifest(env, [("gemm_perf.txt", "torch_flow", "shared", ["trtllm"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.7


def test_merged_no_conflict(env: Path) -> None:
    """Active has shape A, sibling version has shape B → both present in merged dict."""
    _write_gemm_csv(_backend_csv(env), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.5)])
    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "torch_flow", 2048, 4096, 4096, 0.9)])
    _make_manifest(env, [("gemm_perf.txt", "torch_flow", "shared", ["trtllm"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5
    assert _gemm_lookup(db, 2048, 4096, 4096) == 0.9


def test_merged_conflict_active_wins(env: Path) -> None:
    """Active and sibling both have shape A with different latencies → active wins."""
    _write_gemm_csv(_backend_csv(env), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.5)])
    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "torch_flow", 1024, 4096, 4096, 99.0)])
    _make_manifest(env, [("gemm_perf.txt", "torch_flow", "shared", ["trtllm"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5  # active wins


def test_kernel_source_filter(env: Path) -> None:
    """A sibling's rows tagged with a kernel_source not declared shared for our backend
    must NOT be loaded.

    Without filtering, the loaders strip kernel_source from dict keys, so a foreign row
    would silently clobber the active backend's row at the same (m, n, k) coordinate.
    """
    _write_gemm_csv(_backend_csv(env), [("trtllm", "torch_flow", 1024, 4096, 4096, 0.5)])

    # Sibling vLLM has its own data with kernel_source=vllm_default. It collides with
    # trtllm's row at the same (m, n, k). The manifest only lists vllm_default for
    # vllm, so trtllm must ignore it.
    _write_gemm_csv(
        _backend_csv(env, backend="vllm", version="0.5"), [("vllm", "vllm_default", 1024, 4096, 4096, 99.0)]
    )
    _make_manifest(env, [("gemm_perf.txt", "vllm_default", "shared", ["vllm"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5


def test_cross_backend_inheritance(env: Path) -> None:
    """When the manifest declares a kernel_source as shared across multiple frameworks,
    the active backend can inherit rows from a different backend's dir.
    """
    # Active trtllm has nothing for this shape.
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    # Sibling sglang has the row, tagged with a kernel_source the manifest declares
    # shared between trtllm and sglang.
    _write_gemm_csv(
        _backend_csv(env, backend="sglang", version="0.5"),
        [("sglang", "causal_conv1d_fn", 1024, 4096, 4096, 0.7)],
    )
    _make_manifest(env, [("gemm_perf.txt", "causal_conv1d_fn", "shared", ["trtllm", "sglang"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.7


def test_fallback_emits_warning(env: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Loading `tier=shared_fallback` rows in HYBRID mode emits a single WARNING per
    sibling source so the user knows latency predictions for those shapes are
    framework-implicit (kernel_source=default).
    """
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "default", 1024, 4096, 4096, 0.7)])
    _make_manifest(env, [("gemm_perf.txt", "default", "shared_fallback", ["trtllm"])])

    with caplog.at_level(logging.WARNING, logger="aiconfigurator.sdk.perf_database"):
        db = _build_db(env)  # HYBRID mode
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.7
    fallback_warnings = [r for r in caplog.records if "low-fidelity fallback" in r.getMessage()]
    assert len(fallback_warnings) == 1


def test_same_framework_outranks_other_framework(env: Path) -> None:
    """When the same shape is available from both another version of the active backend
    AND from a different backend's dir, the same-backend row wins.

    A different version of the same backend is closer to the active measurement than
    any other framework — even when the kernel_source is shared cross-framework, the
    same backend's integration is more likely to match the active runtime.
    """
    # Active trtllm has no rows for the shape.
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    # trtllm 0.9 has the shape with one latency.
    _write_gemm_csv(_backend_csv(env, version="0.9"), [("trtllm", "causal_conv1d_fn", 1024, 4096, 4096, 0.5)])
    # sglang 0.5 has the same shape with a different latency.
    _write_gemm_csv(
        _backend_csv(env, backend="sglang", version="0.5"),
        [("sglang", "causal_conv1d_fn", 1024, 4096, 4096, 99.0)],
    )
    _make_manifest(env, [("gemm_perf.txt", "causal_conv1d_fn", "shared", ["trtllm", "sglang"])])

    db = _build_db(env)
    # Pre-fix this would return 99.0 because sglang sorted before trtllm alphabetically.
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5


def test_newest_same_framework_version_wins(env: Path) -> None:
    """Among multiple sibling versions of the same backend, the newest version wins
    on shape conflict.

    Newest is closest to the active runtime. Lexicographic order is unsafe because
    `0.9.0` would beat `0.10.0` on string compare; PEP 440 sort handles it correctly.
    """
    active_csv = _backend_csv(env)
    active_csv.parent.mkdir(parents=True, exist_ok=True)
    active_csv.write_text(_GEMM_HEADER)

    # 0.10.0 is newer than 0.9.0 even though "0.10.0" < "0.9.0" lexically.
    _write_gemm_csv(_backend_csv(env, version="0.9.0"), [("trtllm", "causal_conv1d_fn", 1024, 4096, 4096, 99.0)])
    _write_gemm_csv(_backend_csv(env, version="0.10.0"), [("trtllm", "causal_conv1d_fn", 1024, 4096, 4096, 0.5)])
    _make_manifest(env, [("gemm_perf.txt", "causal_conv1d_fn", "shared", ["trtllm"])])

    db = _build_db(env)
    assert _gemm_lookup(db, 1024, 4096, 4096) == 0.5
