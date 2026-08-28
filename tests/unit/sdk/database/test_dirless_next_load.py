# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory-less `next` load (design §14; review follow-up 2026-08).

`next` is derived fleet-wide across default-governed systems: a development
drop on one system advertises the version on every sibling, whose ops are
served by channel-1 backward fill. A sibling WITHOUT its own directory for
that version must still load — the missing-directory gate applies to raw
versions and the authored current/previous, never to the advertised next.

Two-system regression per the review scenario: alpha carries the dev drop,
beta does not; beta's advertised next must resolve AND load.
"""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from aiconfigurator_core.sdk import perf_database as pdb

pytestmark = pytest.mark.unit

_ALPHA = "h200_sxm"  # real system specs so PerfDatabase can construct
_BETA = "b200_sxm"
_REAL_SYSTEMS = Path("aic-core/src/aiconfigurator_core/systems")


def _write_gemm(root: Path, system: str, version: str) -> None:
    d = root / "data" / system / "gemm" / "vllm" / version
    d.mkdir(parents=True, exist_ok=True)
    rows = {
        "framework": ["vllm"] * 2,
        "version": [version] * 2,
        "device": [system] * 2,
        "op_name": ["gemm"] * 2,
        "kernel_source": ["synthetic"] * 2,
        "gemm_dtype": ["bfloat16"] * 2,
        "m": [16, 32],
        "n": [16, 32],
        "k": [16, 32],
        "latency": [0.1, 0.2],
    }
    pq.write_table(pa.table({k: pa.array(v) for k, v in rows.items()}), d / "gemm_perf.parquet")
    # Collector V3 sidecar (§5/§7.4): strict-provenance CI rejects tables
    # without one.
    (d / "collection_meta.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "tables": {"gemm_perf": {"status": "complete"}}})
    )


@pytest.fixture
def systems_root(tmp_path):
    root = tmp_path / "systems"
    root.mkdir()
    for system in (_ALPHA, _BETA):
        shutil.copy(_REAL_SYSTEMS / f"{system}.yaml", root / f"{system}.yaml")
        spec = yaml.safe_load((root / f"{system}.yaml").read_text())
        spec["data_dir"] = f"data/{system}"
        (root / f"{system}.yaml").write_text(yaml.safe_dump(spec))
    (root / "query_versions.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "defaults": {"vllm": {"current": "1.0.0", "previous": None}}})
    )
    # both systems hold current data; only alpha has the 1.1.0 dev drop
    _write_gemm(root, _ALPHA, "1.0.0")
    _write_gemm(root, _ALPHA, "1.1.0")
    _write_gemm(root, _BETA, "1.0.0")
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()
    yield str(root)
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()


def test_beta_advertises_the_fleet_next(systems_root):
    assert pdb.get_version_slots(_BETA, "vllm", systems_root) == {"current": "1.0.0", "next": "1.1.0"}


def test_beta_dirless_next_loads_via_backward_fill(systems_root):
    for requested in ("next", "1.1.0"):
        db = pdb.get_database(_BETA, "vllm", requested, systems_paths=systems_root)
        assert db is not None, f"advertised next must load directory-less (requested {requested!r})"
        assert db.version == "1.1.0"


def test_alpha_next_still_loads_from_its_own_directory(systems_root):
    db = pdb.get_database(_ALPHA, "vllm", "next", systems_paths=systems_root)
    assert db is not None and db.version == "1.1.0"


def test_dirless_relaxation_does_not_weaken_the_raw_version_gate(systems_root, monkeypatch):
    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)
    with pytest.raises(ValueError, match="old-style raw version query"):
        pdb.get_database(_BETA, "vllm", "0.9.0", systems_paths=systems_root)


def test_dirless_relaxation_does_not_cover_authored_slots(systems_root):
    # previous is authored, not derived: absent directory stays a loud miss
    # (here previous is simply unpopulated — the alias-level error).
    with pytest.raises(ValueError, match="has no 'previous'"):
        pdb.get_database(_BETA, "vllm", "previous", systems_paths=systems_root)


@pytest.fixture
def two_roots(tmp_path):
    """Review blocker (2026-08-28): an exact next directory in a LATER root
    must win over a directory-less fallback from an earlier root."""
    base = tmp_path / "base"
    extra = tmp_path / "extra"
    for root in (base, extra):
        root.mkdir()
        shutil.copy(_REAL_SYSTEMS / f"{_BETA}.yaml", root / f"{_BETA}.yaml")
        spec = yaml.safe_load((root / f"{_BETA}.yaml").read_text())
        spec["data_dir"] = f"data/{_BETA}"
        (root / f"{_BETA}.yaml").write_text(yaml.safe_dump(spec))
    (base / "query_versions.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "defaults": {"vllm": {"current": "1.0.0", "previous": None}}})
    )
    _write_gemm(base, _BETA, "1.0.0")
    _write_gemm(extra, _BETA, "1.1.0")  # the exact next collection
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()
    yield [str(base), str(extra)]
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()


def test_exact_next_in_later_root_wins_over_dirless(two_roots):
    assert pdb.get_version_slots(_BETA, "vllm", two_roots) == {"current": "1.0.0", "next": "1.1.0"}
    db = pdb.get_database(_BETA, "vllm", "next", systems_paths=two_roots)
    assert db is not None and db.version == "1.1.0"
    assert not getattr(db, "dirless_next_load", False), (
        "exact next data in a later root must not be shadowed by the directory-less fallback"
    )


def test_probe_handle_serves_a_dirless_next_database(systems_root):
    # get_database returned the object as valid; the public model-less handle
    # (table views, ad-hoc op-list evaluation) must accept it too — the probe
    # spec carries the same dir-less tolerance as the model spec builder.
    from aiconfigurator_core.sdk.engine import EngineHandle

    db = pdb.get_database(_BETA, "vllm", "next", systems_paths=systems_root)
    assert db is not None and getattr(db, "dirless_next_load", False)
    handle = EngineHandle.for_database(db, systems_path=systems_root)
    assert handle is not None


def test_hybrid_probe_handle_serves_a_dirless_next_database(systems_root):
    """HYBRID's generic missing-data allowance must not hide the more
    specific advertised-next identity required by the Rust load gate."""
    from aiconfigurator_core.sdk.engine import EngineHandle

    db = pdb.get_database_view(
        _BETA,
        "vllm",
        "next",
        systems_paths=systems_root,
        allow_missing_data=True,
        database_mode="HYBRID",
    )
    assert db is not None and getattr(db, "dirless_next_load", False)
    handle = EngineHandle.for_database(db, systems_path=systems_root)
    assert handle is not None


def test_advertised_next_marks_a_cached_missing_data_candidate(systems_root):
    """A long-lived process may cache an unlisted missing-data identity before
    a development drop promotes that literal to the advertised next slot."""
    from aiconfigurator_core.sdk.engine import EngineHandle

    cached = pdb.get_database(
        _BETA,
        "vllm",
        "1.2.0",
        systems_paths=systems_root,
        allow_missing_data=True,
        database_mode="HYBRID",
        allow_unlisted_version=True,
    )
    assert cached is not None and not getattr(cached, "dirless_next_load", False)

    _write_gemm(Path(systems_root), _ALPHA, "1.2.0")
    pdb._load_query_slots_doc.cache_clear()
    pdb._derive_fleet_next.cache_clear()

    db = pdb.get_database_view(
        _BETA,
        "vllm",
        "next",
        systems_paths=systems_root,
        allow_missing_data=True,
        database_mode="HYBRID",
    )
    assert db is not None and db.version == "1.2.0"
    assert getattr(db, "dirless_next_load", False)
    assert EngineHandle.for_database(db, systems_path=systems_root) is not None
