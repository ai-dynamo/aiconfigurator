# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from itertools import product
from types import SimpleNamespace

import pytest
import yaml

from aiconfigurator.sdk.common import (
    BackendName,
    PerfDataFilename,
)
from aiconfigurator.sdk.perf_database import (
    LoadedOpData,
    PerfDatabase,
    _resolve_perf_data_path,
    databases_cache,
    get_all_databases,
    get_database,
    get_systems_paths,
    set_systems_paths,
)

pytestmark = pytest.mark.unit


class DummyPerfDatabase:
    def __init__(self, system, backend, version, systems_root_arg, database_mode=None):
        self.system = system
        self.backend = backend
        self.version = version
        self.systems_root = systems_root_arg
        self.database_mode = database_mode
        self.enable_shared_layer = database_mode is None or database_mode.upper() in ("SILICON", "HYBRID")
        self.strict_provenance = False


def test_perf_database_finalize_loaded_data_converts_defaultdicts():
    nested = defaultdict(lambda: defaultdict(dict))
    nested["fp8"][128]["latency"] = 1.0

    loaded = LoadedOpData(nested, PerfDataFilename.gemm, "gemm_perf.txt")
    database = object.__new__(PerfDatabase)
    database._gemm_data = loaded
    database._raw_nested_data = {"loaded": loaded, "nested": nested}

    database._finalize_loaded_data()

    assert isinstance(database._gemm_data, LoadedOpData)
    assert isinstance(database._gemm_data.data, dict)
    assert not isinstance(database._gemm_data.data, defaultdict)
    assert isinstance(database._gemm_data.data["fp8"], dict)
    assert not isinstance(database._gemm_data.data["fp8"], defaultdict)
    assert database._gemm_data["fp8"][128]["latency"] == 1.0
    assert isinstance(database._raw_nested_data, dict)
    assert isinstance(database._raw_nested_data["nested"], dict)
    assert not isinstance(database._raw_nested_data["nested"], defaultdict)
    with pytest.raises(KeyError):
        database._gemm_data["missing"]


def test_generation_dsa_load_data_binds_each_raw_attribute_independently(tmp_path):
    from aiconfigurator.sdk.operations.dsa import GenerationDSAModule

    database = SimpleNamespace(
        systems_root=str(tmp_path),
        system="test_system",
        backend="sglang",
        version="test_version",
        enable_shared_layer=True,
    )
    key = GenerationDSAModule._cache_key(database)
    cached_data, cached_raw = object(), object()
    cached_skip, cached_raw_skip = object(), object()
    data_override, skip_override = object(), object()
    database._generation_dsa_module_data = data_override
    database._generation_dsa_module_skip_data = skip_override

    GenerationDSAModule.clear_cache()
    try:
        GenerationDSAModule._data_cache[key] = cached_data
        GenerationDSAModule._raw_data_cache[key] = cached_raw
        GenerationDSAModule._skip_data_cache[key] = cached_skip
        GenerationDSAModule._raw_skip_data_cache[key] = cached_raw_skip

        GenerationDSAModule.load_data(database)

        assert database._generation_dsa_module_data is data_override
        assert database._raw_generation_dsa_module_data is cached_raw
        assert database._generation_dsa_module_skip_data is skip_override
        assert database._raw_generation_dsa_module_skip_data is cached_raw_skip
    finally:
        GenerationDSAModule.clear_cache()


def test_nccl_load_data_tolerates_null_misc(tmp_path):
    from aiconfigurator.sdk.operations.communication import NCCL

    database = SimpleNamespace(
        systems_root=str(tmp_path),
        system="test_system",
        backend="sglang",
        version="test_version",
        enable_shared_layer=True,
        system_spec={"data_dir": "data", "misc": None},
    )

    NCCL.clear_cache()
    try:
        NCCL.load_data(database)
        assert database._nccl_data.loaded is False
        assert database._oneccl_data is None
    finally:
        NCCL.clear_cache()


def test_get_database_with_yaml_and_data_path(tmp_path, monkeypatch):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    system = "testsys"
    backend = "cuda"
    version = "v1"

    systems_dir = tmp_path / "systems_dir"
    systems_dir.mkdir()

    yaml_path = systems_dir / f"{system}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"data_dir": "data"}, f)

    data_subdir = systems_dir / "data" / backend / version
    data_subdir.mkdir(parents=True)

    databases_cache.clear()

    db1 = get_database(system, backend, version, systems_paths=str(systems_dir))

    assert isinstance(db1, DummyPerfDatabase), "Expected a DummyPerfDatabase"

    assert db1.system == system
    assert db1.backend == backend
    assert db1.version == version
    assert db1.systems_root == str(systems_dir)

    db2 = get_database(system, backend, version, systems_paths=str(systems_dir))
    assert db2 is db1, "Repeated calls with identical args should return the same database object"


def test_get_all_databases(tmp_path, monkeypatch):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    systems_dir = tmp_path / "systems_dir"
    systems_dir.mkdir()

    versions = ["v1", "v2", "v3"]
    system_yamls = ["testsys_0", "testsys_1", "testsys_2"]
    data_dirs = ["data0", "data1", "data2"]
    # Set up dummy yamls
    for idx, yaml_file in enumerate(system_yamls):
        with open(systems_dir / f"{yaml_file}.yaml", "w") as f:
            yaml.dump({"data_dir": f"data{idx}"}, f)
    for data, backend, version in product(data_dirs, BackendName, versions):
        data_subdir = systems_dir / data / backend.value / version
        data_subdir.mkdir(parents=True)

    # max_workers=1 keeps loading in-parent so the DummyPerfDatabase monkeypatch
    # is honored. The ProcessPoolExecutor path re-imports the module in workers
    # and would bypass the patch.
    database_dict = get_all_databases(systems_paths=str(systems_dir), max_workers=1)

    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v3"], DummyPerfDatabase)

    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v3"], DummyPerfDatabase)

    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v3"], DummyPerfDatabase)


def test_get_all_databases_does_not_seed_formula_only_cache_with_shared_database(tmp_path, monkeypatch):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    systems_dir = tmp_path / "systems_dir"
    systems_dir.mkdir()
    (systems_dir / "testsys.yaml").write_text("data_dir: data\n")
    (systems_dir / "data" / "sglang" / "v1").mkdir(parents=True)
    databases_cache.clear()

    try:
        all_databases = get_all_databases(systems_paths=str(systems_dir), max_workers=1)
        shared_db = all_databases["testsys"]["sglang"]["v1"]
        empirical_db = get_database(
            "testsys",
            "sglang",
            "v1",
            systems_paths=str(systems_dir),
            database_mode="EMPIRICAL",
        )

        assert shared_db.enable_shared_layer is True
        assert empirical_db.enable_shared_layer is False
        assert empirical_db is not shared_db
    finally:
        databases_cache.clear()


def test_get_database_uses_default_systems_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    system = "testsys"
    backend = "cuda"
    version = "v1"

    systems_root = tmp_path / "systems_root"
    systems_root.mkdir()

    yaml_path = systems_root / f"{system}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump({"data_dir": "data"}, f)

    data_subdir = systems_root / "data" / backend / version
    data_subdir.mkdir(parents=True)

    databases_cache.clear()
    previous_paths = get_systems_paths()
    try:
        set_systems_paths(str(systems_root))
        db = get_database(system, backend, version)
        assert isinstance(db, DummyPerfDatabase)
        assert db.systems_root == str(systems_root)
    finally:
        set_systems_paths(previous_paths)


def test_get_database_conflict_returns_first(tmp_path, monkeypatch):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    system = "h100"
    backend = "trtllm"
    version = "v1"

    systems_root_a = tmp_path / "systems_a"
    systems_root_b = tmp_path / "systems_b"
    systems_root_a.mkdir()
    systems_root_b.mkdir()

    (systems_root_a / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_a"}))
    (systems_root_b / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_b"}))

    (systems_root_a / "data_a" / backend / version).mkdir(parents=True)
    (systems_root_b / "data_b" / backend / version).mkdir(parents=True)

    databases_cache.clear()
    db = get_database(system, backend, version, systems_paths=[str(systems_root_a), str(systems_root_b)])
    assert isinstance(db, DummyPerfDatabase)
    assert db.systems_root == str(systems_root_a)


def test_get_all_databases_system_config_conflict(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    caplog.set_level("WARNING")
    system = "h100"

    systems_root_a = tmp_path / "systems_a"
    systems_root_b = tmp_path / "systems_b"
    systems_root_a.mkdir()
    systems_root_b.mkdir()

    (systems_root_a / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_a"}))
    (systems_root_b / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_b"}))

    (systems_root_a / "data_a" / "trtllm" / "v1").mkdir(parents=True)
    (systems_root_b / "data_b" / "vllm" / "v0").mkdir(parents=True)

    databases_cache.clear()
    # max_workers=1 keeps loading in-parent so the DummyPerfDatabase monkeypatch
    # is honored. The ProcessPoolExecutor path re-imports the module in workers
    # and would bypass the patch.
    database_dict = get_all_databases(systems_paths=[str(systems_root_a), str(systems_root_b)], max_workers=1)

    assert "trtllm" in database_dict[system]
    assert "vllm" in database_dict[system]
    assert any("System config 'h100' already loaded from" in record.message for record in caplog.records)


def test_get_all_databases_conflicting_backend_version_keeps_first(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("aiconfigurator.sdk.perf_database.PerfDatabase", DummyPerfDatabase)
    caplog.set_level("WARNING")
    system = "h100"

    systems_root_a = tmp_path / "systems_a"
    systems_root_b = tmp_path / "systems_b"
    systems_root_a.mkdir()
    systems_root_b.mkdir()

    (systems_root_a / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_a"}))
    (systems_root_b / f"{system}.yaml").write_text(yaml.safe_dump({"data_dir": "data_b"}))

    (systems_root_a / "data_a" / "trtllm" / "v1").mkdir(parents=True)
    (systems_root_b / "data_b" / "trtllm" / "v1").mkdir(parents=True)

    databases_cache.clear()
    # max_workers=1 keeps loading in-parent so the DummyPerfDatabase monkeypatch
    # is honored. The ProcessPoolExecutor path re-imports the module in workers
    # and would bypass the patch.
    database_dict = get_all_databases(systems_paths=[str(systems_root_a), str(systems_root_b)], max_workers=1)
    db = database_dict[system]["trtllm"]["v1"]
    assert db.systems_root == str(systems_root_a)
    assert any("Database 'h100/trtllm/v1' already loaded from" in record.message for record in caplog.records)


def test_resolve_perf_data_path_falls_back_to_legacy_txt(tmp_path):
    legacy_file = tmp_path / "gemm_perf.txt"
    legacy_file.write_text("framework,version,device,op_name,gemm_dtype,m,n,k,latency\n")

    assert _resolve_perf_data_path(str(tmp_path / "gemm_perf.parquet")) == str(legacy_file)


def _dsv4_megamoe_row(
    *,
    phase: str = "context",
    num_tokens: int = 1024,
    latency: float = 1.25,
    power: float | None = None,
    distribution: str = "balanced",
) -> dict[str, str]:
    row = {
        "framework": "SGLang",
        "version": "unknown",
        "device": "NVIDIA GB200",
        "op_name": "dsv4_megamoe_module",
        "kernel_source": "deepgemm_megamoe",
        "phase": phase,
        "moe_dtype": "w4a8_mxfp4_mxfp8",
        "kernel_dtype": "fp8_fp4",
        "num_tokens": str(num_tokens),
        "global_num_tokens": "8192",
        "hidden_size": "7168",
        "inter_size": "3072",
        "topk": "6",
        "num_experts": "384",
        "num_fused_shared_experts": "0",
        "moe_tp_size": "1",
        "moe_ep_size": "8",
        "distribution": distribution,
        "source_policy": "random",
        "pre_dispatch": "sglang_jit",
        "num_max_tokens_per_rank": "16384",
        "effective_num_max_tokens_per_rank": "16448",
        "routed_scaling_factor": "2.5",
        "includes_routed_scale": "true",
        "includes_gate_topk": "false",
        "buffer_policy": "cached_sglang",
        "includes_buffer_init": "false",
        "used_cuda_graph": "true",
        "latency": str(latency),
    }
    if power is not None:
        row["power"] = str(power)
    return row


def _write_dsv4_megamoe_perf(csv_file, *rows: dict[str, str], include_phase: bool = True) -> None:
    fields = list(_dsv4_megamoe_row().keys())
    for row in rows:
        fields.extend(field for field in row if field not in fields)
    if not include_phase:
        fields = [field for field in fields if field != "phase"]
    csv_file.write_text(
        ",".join(fields) + "\n" + "".join(",".join(row.get(field, "") for field in fields) + "\n" for row in rows)
    )


def test_dsv4_megamoe_module_support_matrix_empty_without_data(tmp_path):
    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "sglang" / "0.5.10"
    data_dir.mkdir(parents=True)
    # The engine table view resolves through the probe handle, whose Rust
    # SystemSpec load needs the full gpu/node shape (the loader-stub era got
    # away with a 1-key gpu dict).
    (systems_root / "gb200.yaml").write_text(
        yaml.safe_dump(
            {
                "data_dir": "data",
                "gpu": {
                    "sm_version": 100,
                    "mem_bw": 8_000_000_000_000.0,
                    "mem_bw_empirical_scaling_factor": 0.8,
                    "mem_empirical_constant_latency": 0.000003,
                    "bfloat16_tc_flops": 2_500_000_000_000_000.0,
                    "fp8_tc_flops": 5_000_000_000_000_000.0,
                },
                "node": {
                    "num_gpus_per_node": 4,
                    "inter_node_bw": 100_000_000_000.0,
                    "intra_node_bw": 900_000_000_000.0,
                    "p2p_latency": 0.000001,
                },
                "misc": {"nccl_version": "test"},
            }
        )
    )

    db = PerfDatabase("gb200", "sglang", "0.5.10", str(systems_root))

    assert db.supported_quant_mode["dsv4_megamoe_module"] == []


def test_comprehensive_router_survives_scoped_stub_patch(mutable_comprehensive_perf_db, stub_perf_db, monkeypatch):
    """Fixture-order regression: a scoped ``stub_perf_db`` fetch patch active
    while the comprehensive singleton is (re)used must not be captured as the
    router's pass-through — after cache clears, ``test_system`` reloads must
    still resolve to the synthetic tables (the router is module-level and
    scoped patches layer on top of it). Fixture order matters: the
    comprehensive singleton must be built BEFORE the scoped stub patch is
    active, or the singleton itself is constructed through the stubbed fetch
    and every later same-worker test reads a bf16-only singleton."""
    from aiconfigurator.sdk.operations import warm_all_op_data
    from aiconfigurator.sdk.operations.base import clear_all_op_caches
    from aiconfigurator.sdk.operations.gemm import GEMM

    db = mutable_comprehensive_perf_db
    clear_all_op_caches()
    try:
        db.__dict__.pop("_gemm_data", None)
        warm_all_op_data(db)
        assert db._gemm_data.loaded, "synthetic gemm table lost after a cache clear under a scoped stub patch"
        assert len(db._gemm_data) > 0
    finally:
        clear_all_op_caches()
        GEMM.load_data(db)
