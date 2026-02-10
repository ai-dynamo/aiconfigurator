# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import pytest
import yaml

from aiconfigurator.sdk.common import (
    BackendName,
    CommQuantMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from aiconfigurator.sdk.perf_database import (
    databases_cache,
    get_all_databases,
    get_database,
    get_systems_paths,
    load_context_attention_data,
    load_context_mla_data,
    load_custom_allreduce_data,
    load_gemm_data,
    load_generation_attention_data,
    load_generation_mla_data,
    load_mla_bmm_data,
    load_moe_data,
    load_nccl_data,
    load_wideep_moe_compute_data,
    set_systems_paths,
)

pytestmark = pytest.mark.unit


class DummyPerfDatabase:
    def __init__(self, system, backend, version, systems_root_arg):
        self.system = system
        self.backend = backend
        self.version = version
        self.systems_root = systems_root_arg


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

    database_dict = get_all_databases(systems_paths=str(systems_dir))

    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_0"][BackendName.trtllm.value]["v3"], DummyPerfDatabase)

    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_1"][BackendName.sglang.value]["v3"], DummyPerfDatabase)

    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v1"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v2"], DummyPerfDatabase)
    assert isinstance(database_dict["testsys_2"][BackendName.vllm.value]["v3"], DummyPerfDatabase)


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
    database_dict = get_all_databases(systems_paths=[str(systems_root_a), str(systems_root_b)])

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
    database_dict = get_all_databases(systems_paths=[str(systems_root_a), str(systems_root_b)])
    db = database_dict[system]["trtllm"]["v1"]
    assert db.systems_root == str(systems_root_a)
    assert any("Database 'h100/trtllm/v1' already loaded from" in record.message for record in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# 1) load_custom_allreduce_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_custom_allreduce_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_custom_allreduce_data should return None.
    """
    fake_path = tmp_path / "does_not_exist.csv"
    result = load_custom_allreduce_data(str(fake_path))
    assert result is None


def test_load_custom_allreduce_data_basic(tmp_path):
    """
    Create a tiny CSV with two lines of:
        (dtype,tp_size,message_size,allreduce_strategy,layer_name,latency)
    The loader ignores the dtype from the file and always uses CommQuantMode.half as the key.
    We verify that:
      - data[CommQuantMode.half][tp_size][strategy][message_size] == latency_float
    """
    # 1) Prepare a minimal CSV file
    csv_file = tmp_path / "custom_ar.csv"
    lines = [
        "framework,version,device,op_name,kernel_source,allreduce_dtype,num_gpus,message_size,latency\n",
        "TRTLLM,1.0.0rc6,NVIDIA B200,all_reduce,TRTLLM,float16,2,128,0.0038\n",
        "TRTLLM,1.0.0rc6,NVIDIA B200,all_reduce,TRTLLM,float16,2,8192,0.0045\n",
    ]
    csv_file.write_text("".join(lines))

    # 2) Call the loader
    data = load_custom_allreduce_data(str(csv_file))

    # 3) Verify structure and values
    key_dtype = CommQuantMode.half  # loader always forces dtype→CommQuantMode.half
    assert key_dtype in data

    # Expected format:
    # data[dtype][tp_size][allreduce_strategy][message_size] = {"latency": float, "power": float}
    assert 2 in data[key_dtype]
    assert "AUTO" in data[key_dtype][2]
    assert data[key_dtype][2]["AUTO"][128]["latency"] == pytest.approx(0.0038)
    assert data[key_dtype][2]["AUTO"][8192]["latency"] == pytest.approx(0.0045)


# ─────────────────────────────────────────────────────────────────────────────
# 2) load_nccl_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_nccl_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_nccl_data should return None.
    """
    fake_path = tmp_path / "no_nccl.csv"
    result = load_nccl_data(str(fake_path))
    assert result is None


def test_load_nccl_data_basic(tmp_path):
    """
    Create a tiny CSV with two lines of (dtype,num_gpus,message_size,library,operation,latency).
    We use dtype strings that match enum names in CommQuantMode, e.g. "half" and "int8".
    The loader does:
      dtype_enum = CommQuantMode[dtype_str]
      latency = float(latency)
      nccl_data[dtype_enum][operation][num_gpus][message_size] = latency
    """
    csv_file = tmp_path / "nccl.csv"
    lines = [
        "nccl_dtype,num_gpus,message_size,kernel_source,op_name,latency\n",
        # half, 2 GPUs, 512 bytes, library=NCCL, operation="allgather", latency=1.0
        "half,2,512,NCCL,allgather,1.0\n",
        # int8, 4 GPUs, 1024 bytes, library=NCCL, operation="allreduce", latency=2.5
        "int8,4,1024,NCCL,allreduce,2.5\n",
    ]
    csv_file.write_text("".join(lines))

    data = load_nccl_data(str(csv_file))

    # Check existence of keys
    half_key = CommQuantMode.half
    int8_key = CommQuantMode.int8

    # "allgather" entry:
    assert half_key in data
    assert "allgather" in data[half_key]
    assert 2 in data[half_key]["allgather"]
    assert 512 in data[half_key]["allgather"][2]
    assert data[half_key]["allgather"][2][512]["latency"] == pytest.approx(1.0)

    # "allreduce" entry:
    assert int8_key in data
    assert "allreduce" in data[int8_key]
    assert 4 in data[int8_key]["allreduce"]
    assert 1024 in data[int8_key]["allreduce"][4]
    assert data[int8_key]["allreduce"][4][1024]["latency"] == pytest.approx(2.5)


# ─────────────────────────────────────────────────────────────────────────────
# 3) load_gemm_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_gemm_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_gemm_data should return None.
    """
    fake_path = tmp_path / "no_gemm.csv"
    result = load_gemm_data(str(fake_path))
    assert result is None


def test_load_gemm_data_basic(tmp_path):
    """
    Create a CSV with lines of:
        (backend_name,version,hardware,op_name,quant_mode,m,n,k,layer_name,latency)
    We pick quant_mode="float16" (which exists as a key in GEMMQuantMode).
    The loader does:
      quant_enum = common.GEMMQuantMode[quant_mode_str]
      m,n,k → int
      latency → float
      gemm_data[quant_enum][m][n][k] = latency
    """
    csv_file = tmp_path / "gemm.csv"
    lines = [
        "framework,version,device,op_name,gemm_dtype,m,n,k,latency\n",
        "trt,1.0,hwX,opX,float16,128,256,512,0.789\n",
    ]
    csv_file.write_text("".join(lines))

    data = load_gemm_data(str(csv_file))

    key_mode = GEMMQuantMode.float16
    assert key_mode in data
    assert 128 in data[key_mode]
    assert 256 in data[key_mode][128]
    assert 512 in data[key_mode][128][256]
    assert data[key_mode][128][256][512]["latency"] == pytest.approx(0.789)


# ─────────────────────────────────────────────────────────────────────────────
# 4) load_moe_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_moe_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_moe_data should return None.
    """
    fake_path = tmp_path / "no_moe.csv"
    result = load_moe_data(str(fake_path))
    assert result == (None, None)


def test_load_moe_data_basic(tmp_path):
    """
    Create a CSV with one line of:
        (backend_name,version,hardware,op_name,quant_mode,num_tokens,hidden_size,inter_size,topk,
         num_experts,moe_tp_size,moe_ep_size,workload_distribution,layer_name,latency)
    We pick:
      quant_mode="float16", num_tokens=1, hidden_size=16, inter_size=32, topk=2,
      num_experts=4, moe_tp_size=2, moe_ep_size=2, workload_distribution="uniform", latency=1.23
    The loader converts quant_mode→common.MoEQuantMode[quant_mode_str], then:
      moe_data[quant_mode_enum][workload_distribution][topk][num_experts][hidden_size][inter_size]
        [moe_tp_size][moe_ep_size][num_tokens] = latency
    """
    csv_file = tmp_path / "moe.csv"
    headers = (
        "framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,"
        "inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n"
    )
    line = (
        ",".join(
            [
                "trt",  # backend_name
                "1.0",  # version
                "hwX",  # hardware
                "opX",  # op_name
                "moe_torch_flow",  # kernel_source
                "float16",  # quant_mode
                "1",  # num_tokens
                "16",  # hidden_size
                "32",  # inter_size
                "2",  # topk
                "4",  # num_experts
                "2",  # moe_tp_size
                "2",  # moe_ep_size
                "uniform",  # workload_distribution
                "1.23",  # latency
            ]
        )
        + "\n"
    )
    csv_file.write_text(headers + line)

    data, _ = load_moe_data(str(csv_file))

    qm = MoEQuantMode.float16
    assert qm in data
    # The exact order of nested dict is:
    #   [qm][workload_distribution][topk][num_experts][hidden_size][inter_size][moe_tp_size][moe_ep_size][num_tokens]  # noqa: E501
    assert "uniform" in data[qm]
    assert 2 in data[qm]["uniform"]  # topk
    assert 4 in data[qm]["uniform"][2]  # num_experts
    assert 16 in data[qm]["uniform"][2][4]  # hidden_size
    assert 32 in data[qm]["uniform"][2][4][16]  # inter_size
    assert 2 in data[qm]["uniform"][2][4][16][32]  # moe_tp_size
    assert 2 in data[qm]["uniform"][2][4][16][32][2]  # moe_ep_size
    assert 1 in data[qm]["uniform"][2][4][16][32][2][2]  # num_tokens
    assert data[qm]["uniform"][2][4][16][32][2][2][1]["latency"] == pytest.approx(1.23)


# ─────────────────────────────────────────────────────────────────────────────
# 5) load_context_attention_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_context_attention_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_context_attention_data should return None.
    """
    fake_path = tmp_path / "no_ctx_attn.csv"
    result = load_context_attention_data(str(fake_path))
    assert result is None


def test_load_context_attention_data_basic(tmp_path):
    """
    Create a CSV with one line of:
        (backend_name,version,hardware,op_name,b,s,n,kv_n,d,beam,quant_mode,kv_cache_dtype,step,latency)
    - b=1, s=2, n=4, kv_n=4 (so internally kv_n becomes 0 because kv_n==n),
      d=16, beam=8 (ignored after parsing), quant_mode="float16", kv_cache_dtype="float16",
      step=1, latency=0.321.
    The loader does:
      kv_n = 0 if n == kv_n else kv_n
      quant_mode = common.FMHAQuantMode[quant_mode_str]
      kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype_str]
      context_attention_data[quant_mode_enum][kv_cache_dtype_enum][kv_n][n][s][b] = latency
    So we expect data[FMHAQuantMode.float16][KVCacheQuantMode.float16][0][4][2][1] == 0.321.
    """
    csv_file = tmp_path / "ctx_attn.csv"
    headers = (
        "framework,version,device,op_name,batch_size,isl,num_heads,num_key_value_heads,head_dim,"
        "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
    )
    fields = [
        "trt",  # backend_name
        "1.0",  # version
        "hwX",  # hardware
        "context_attention",  # op_name
        "1",  # b
        "2",  # s
        "4",  # n
        "4",  # kv_n  → becomes 0 internally
        "16",  # d  (ignored after parsing)
        "8",  # beam (ignored after parsing)
        "float16",  # quant_mode → FMHAQuantMode.float16
        "float16",  # kv_cache_dtype → KVCacheQuantMode.float16
        "1",  # step
        "0.321",  # latency
    ]
    csv_file.write_text(headers + ",".join(fields) + "\n")

    data = load_context_attention_data(str(csv_file))

    qm = FMHAQuantMode.float16
    kcd = KVCacheQuantMode.float16

    # kv_n became 0 because n == kv_n in the code
    assert qm in data
    assert kcd in data[qm]
    assert 0 in data[qm][kcd]  # kv_n == 0
    assert 4 in data[qm][kcd][0][16][0]  # n == 4
    assert 2 in data[qm][kcd][0][16][0][4]  # s == 2
    assert 1 in data[qm][kcd][0][16][0][4][2]  # b == 1
    assert data[qm][kcd][0][16][0][4][2][1]["latency"] == pytest.approx(0.321)


# ─────────────────────────────────────────────────────────────────────────────
# 6) load_generation_attention_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_generation_attention_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_generation_attention_data should return None.
    """
    fake_path = tmp_path / "no_gen_attn.csv"
    result = load_generation_attention_data(str(fake_path))
    assert result is None


def test_load_generation_attention_data_basic(tmp_path):
    """
    Create a CSV with:
        (backend_name,version,hardware,op_name,b,s,n,kv_n,d,beam,quant_mode,kv_cache_dtype,step,latency)
    - b=1, s=2, n=4, kv_n=4 → becomes 0 internally
      d=16, beam=8 (ignored), quant_mode="ignored" (not used), kv_cache_dtype="float16",
      step=1, so stored s = original s + step = 2 + 1 = 3, latency=0.987.
    The loader does:
      kv_n = 0 if n == kv_n else kv_n
      s = s + step
      kv_cache_dtype = common.KVCacheQuantMode[kv_cache_dtype_str]
      generation_attention_data[kv_cache_dtype_enum][kv_n][n][b][s] = latency
    So we expect data[KVCacheQuantMode.float16][0][4][1][3] == 0.987.
    """
    csv_file = tmp_path / "gen_attn.csv"
    headers = (
        "framework,version,device,op_name,batch_size,isl,num_heads,num_key_value_heads,head_dim,"
        "beam_width,attn_dtype,kv_cache_dtype,step,latency\n"
    )
    fields = [
        "trt",  # backend_name
        "1.0",  # version
        "hwX",  # hardware
        "generation_attention",  # op_name
        "1",  # b
        "2",  # s=2
        "4",  # n
        "4",  # kv_n→0
        "16",  # d (ignored)
        "8",  # beam (ignored)
        "dummy",  # quant_mode (not actually used downstream)
        "float16",  # kv_cache_dtype→KVCacheQuantMode.float16
        "1",  # step
        "0.987",  # latency
    ]
    csv_file.write_text(headers + ",".join(fields) + "\n")

    data = load_generation_attention_data(str(csv_file))

    kcd = KVCacheQuantMode.float16
    assert kcd in data
    assert 0 in data[kcd]  # kv_n turned into 0
    assert 4 in data[kcd][0][16][0]  # n == 4
    assert 1 in data[kcd][0][16][0][4]  # b == 1
    assert 3 in data[kcd][0][16][0][4][1]  # s = original 2 + step 1 = 3
    assert data[kcd][0][16][0][4][1][3]["latency"] == pytest.approx(0.987)


# ─────────────────────────────────────────────────────────────────────────────
# 7) load_context_mla_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_context_mla_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_context_mla_data should return None.
    """
    fake_path = tmp_path / "no_ctx_mla.csv"
    result = load_context_mla_data(str(fake_path))
    assert result is None


def test_load_context_mla_data_basic(tmp_path):
    """
    Create a CSV line of (backend_name,version,hardware,op_name,quant_mode,kv_cache_dtype,
    b,s,tp_size,step,latency). We pick:
      quant_mode="float16", kv_cache_dtype="float16",
      b=1, s=2, tp_size=4, step=1, latency=1.111.
    The loader does:
      quant_mode_enum = common.FMHAQuantMode[quant_mode_str]
      kv_cache_dtype_enum = common.KVCacheQuantMode[kv_cache_dtype_str]
      Then:
        context_mla_data[quant_mode_enum][kv_cache_dtype_enum][tp_size][s][b] = latency
    """
    csv_file = tmp_path / "ctx_mla.csv"
    headers = "framework,version,device,op_name,mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,step,latency\n"
    fields = [
        "trt",  # backend_name (ignored)
        "1.0",  # version (ignored)
        "hwX",  # hardware (ignored)
        "opZ",  # op_name (ignored as key)
        "float16",  # quant_mode → common.FMHAQuantMode.float16
        "float16",  # kv_cache_dtype → common.KVCacheQuantMode.float16
        "1",  # b
        "2",  # s
        "4",  # tp_size
        "1",  # step (ignored downstream)
        "1.111",  # latency
    ]
    csv_file.write_text(headers + ",".join(fields) + "\n")

    data = load_context_mla_data(str(csv_file))

    qm = FMHAQuantMode.float16
    kcd = KVCacheQuantMode.float16

    num_heads = 128 // 4  # tp_size == 4 -> num_heads == 128 // 4 == 32

    assert qm in data
    assert kcd in data[qm]
    assert num_heads in data[qm][kcd]
    assert 2 in data[qm][kcd][num_heads]  # s == 2
    assert 1 in data[qm][kcd][num_heads][2]  # b == 1
    assert data[qm][kcd][num_heads][2][1]["latency"] == pytest.approx(1.111)


# ─────────────────────────────────────────────────────────────────────────────
# 8) load_generation_mla_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_generation_mla_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_generation_mla_data should return None.
    """
    fake_path = tmp_path / "no_gen_mla.csv"
    result = load_generation_mla_data(str(fake_path))
    assert result is None


def test_load_generation_mla_data_basic(tmp_path):
    """
    Create a CSV with (backend_name,version,hardware,op_name,quant_mode,kv_cache_dtype,
    b,s,tp_size,step,latency). We pick:
      b=1, s=2, tp_size=4, step=1 → stored s=2+1=3, quant_mode unused, kv_cache_dtype="float16",
      latency=2.222.
    The loader does:
      s = s + step
      kv_cache_dtype_enum = common.KVCacheQuantMode[kv_cache_dtype_str]
      generation_mla_data[kv_cache_dtype_enum][tp_size][b][new_s] = latency
    """
    csv_file = tmp_path / "gen_mla.csv"
    headers = "framework,version,device,op_name,mla_dtype,kv_cache_dtype,batch_size,isl,tp_size,step,latency\n"
    fields = [
        "trt",  # backend_name (ignored)
        "1.0",  # version (ignored)
        "hwY",  # hardware (ignored)
        "opW",  # op_name (ignored)
        "ignored",  # quant_mode (not used downstream)
        "float16",  # kv_cache_dtype → common.KVCacheQuantMode.float16
        "1",  # b
        "2",  # s=2
        "4",  # tp_size
        "1",  # step → new_s=3
        "2.222",  # latency
    ]
    csv_file.write_text(headers + ",".join(fields) + "\n")

    data = load_generation_mla_data(str(csv_file))

    kcd = KVCacheQuantMode.float16
    num_heads = 128 // 4  # tp_size == 4 -> num_heads == 128 // 4 == 32

    assert kcd in data
    assert num_heads in data[kcd]
    assert 1 in data[kcd][num_heads]  # b == 1
    assert 3 in data[kcd][num_heads][1]  # s = original 2 + step 1 = 3
    assert data[kcd][num_heads][1][3]["latency"] == pytest.approx(2.222)


# ─────────────────────────────────────────────────────────────────────────────
# 9) load_mla_bmm_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_mla_bmm_data_nonexistent(tmp_path):
    """
    If the file does not exist, load_mla_bmm_data should return None.
    """
    fake_path = tmp_path / "no_mla_bmm.csv"
    result = load_mla_bmm_data(str(fake_path))
    assert result is None


def test_load_mla_bmm_data_basic(tmp_path):
    """
    Create a CSV with one line of:
        (backend_name,version,hardware,op_name,quant_mode,num_tokens,num_heads,latency)
    We pick:
      quant_mode="half", num_tokens=8, num_heads=2, latency=3.333
    The loader does:
      quant_enum = common.GEMMQuantMode[quant_mode_str]
      mla_bmm_data[quant_enum][op_name][num_heads][num_tokens] = latency
    """
    csv_file = tmp_path / "mla_bmm.csv"
    headers = "framework,version,device,op_name,bmm_dtype,num_tokens,num_heads,latency\n"
    fields = [
        "trt",  # backend_name (ignored)
        "1.0",  # version (ignored)
        "hwZ",  # hardware (ignored)
        "bmm_op",  # op_name → used as a key in the nested dict
        "float16",  # quant_mode → common.GEMMQuantMode.float16
        "8",  # num_tokens
        "2",  # num_heads
        "3.333",  # latency
    ]
    csv_file.write_text(headers + ",".join(fields) + "\n")

    data = load_mla_bmm_data(str(csv_file))

    qg = GEMMQuantMode.float16  # Using 'half' as string in CSV should map to float16
    assert qg in data
    assert "bmm_op" in data[qg]
    assert 2 in data[qg]["bmm_op"]
    assert 8 in data[qg]["bmm_op"][2]
    assert data[qg]["bmm_op"][2][8]["latency"] == pytest.approx(3.333)


# ─────────────────────────────────────────────────────────────────────────────
# 10) load_wideep_moe_compute_data
# ─────────────────────────────────────────────────────────────────────────────


def test_load_wideep_moe_compute_data(tmp_path):
    """
    Test loading wideep MoE compute data with the format from:
    aiconfigurator/src/aiconfigurator/systems/data/gb200_sxm/trtllm/1.2.0rc6/wideep_moe_perf.txt

    CSV columns:
        framework,version,device,op_name,kernel_source,moe_dtype,moe_kernel,num_tokens,
        dp_num_tokens,rank0_num_tokens,hidden_size,inter_size,topk,num_experts,num_slots,
        moe_tp_size,moe_ep_size,distribution,simulation_mode,latency

    Structure: [kernel_source][quant_mode][distribution][topk][num_experts][hidden_size]
               [inter_size][num_slots][moe_tp_size][moe_ep_size][num_tokens] -> {latency, power, energy}
    """
    csv_file = tmp_path / "wideep_moe_perf.txt"
    headers = (
        "framework,version,device,op_name,kernel_source,moe_dtype,moe_kernel,num_tokens,"
        "dp_num_tokens,rank0_num_tokens,hidden_size,inter_size,topk,num_experts,num_slots,"
        "moe_tp_size,moe_ep_size,distribution,simulation_mode,latency\n"
    )
    # wideep_moe (no EPLB)
    line1 = (
        "TRTLLM,1.2.0rc6,NVIDIA GB200,wideep_moe,wideep_compute_cutlass,nvfp4,cutlass,"
        "1,1,1,7168,2048,8,256,256,1,4,power_law_1.01,accurate,0.08142079710960388\n"
    )
    # wideep_moe_eplb (with EPLB)
    line2 = (
        "TRTLLM,1.2.0rc6,NVIDIA GB200,wideep_moe_eplb,wideep_compute_cutlass,nvfp4,cutlass,"
        "1,1,1,7168,2048,8,256,288,1,4,power_law_1.01_eplb,accurate,0.07909759879112244\n"
    )

    csv_file.write_text(headers + line1 + line2)

    data = load_wideep_moe_compute_data(str(csv_file))

    assert data is not None

    qm = MoEQuantMode.nvfp4
    kernel_source = "wideep_compute_cutlass"

    # Verify non-EPLB entry: power_law_1.01, num_slots=256
    result = data[kernel_source][qm]["power_law_1.01"][8][256][7168][2048][256][1][4][1]
    assert result["latency"] == pytest.approx(0.08142079710960388)

    # Verify EPLB entry: power_law_1.01_eplb, num_slots=288
    result_eplb = data[kernel_source][qm]["power_law_1.01_eplb"][8][256][7168][2048][288][1][4][1]
    assert result_eplb["latency"] == pytest.approx(0.07909759879112244)
