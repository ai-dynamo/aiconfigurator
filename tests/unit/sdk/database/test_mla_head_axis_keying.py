# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""#1458 head-axis keying: MLA module tables key ``[native][local]`` (native
from the model pin — sweeps have tp_size==1, the product cannot derive it);
kernel tables stay local-only with the ``128 // tp_size`` backfill retired.
Guardrails scan every shipped parquet. Rationale:
docs/perf_database/head-axis-keying.md.
"""

from pathlib import Path

import pytest

from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.operations.mla import (
    _MLA_MODULE_NATIVE_HEADS,
    _require_native_bucket,
    _resolve_mla_module_native_key,
    load_context_mla_data,
    load_context_mla_module_data,
    load_generation_mla_data,
    load_generation_mla_module_data,
    load_wideep_context_mla_data,
    load_wideep_generation_mla_data,
)

_MODULE_HEADER = (
    "framework,version,device,op_name,kernel_source,model,architecture,"
    "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,"
    "step,latency"
)
_DSV3 = "deepseek-ai/DeepSeek-V3"


def _module_row(
    *,
    model: str = _DSV3,
    num_heads: int,
    bs: int = 1,
    isl: int = 1024,
    tp: int = 1,
    step: int = 0,
    lat: float = 1.0,
    op_name: str = "mla_context_module",
) -> str:
    return (
        f"vllm,test,NVIDIA B200,{op_name},default,{model},DeepseekV3ForCausalLM,"
        f"bfloat16,bfloat16,bfloat16,{num_heads},{bs},{isl},{tp},{step},{lat}"
    )


def _write_csv(path, header: str, rows: list[str]) -> str:
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return str(path)


# ───────────────────────────────────────────────────────────────────────
# Module loaders: [native][local] nesting and the model pin
# ───────────────────────────────────────────────────────────────────────


def test_load_context_mla_module_keys_by_native_then_local(tmp_path):
    rows = [
        _module_row(num_heads=128, bs=1, isl=1024, lat=2.0),
        _module_row(num_heads=16, bs=1, isl=1024, lat=0.4),
        _module_row(num_heads=16, bs=4, isl=2048, lat=1.6),
    ]
    path = _write_csv(tmp_path / "ctx_mod.txt", _MODULE_HEADER, rows)
    data = load_context_mla_module_data(path)
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    by_native = data[fmha][kv][gemm]
    assert set(by_native.keys()) == {128}
    assert set(by_native[128].keys()) == {16, 128}
    assert by_native[128][16][2048][4]["latency"] == pytest.approx(1.6)


def test_load_generation_mla_module_keys_by_native_then_local(tmp_path):
    rows = [
        _module_row(num_heads=128, bs=8, isl=4096, step=1, lat=0.09, op_name="mla_generation_module"),
        _module_row(num_heads=16, bs=8, isl=4096, step=1, lat=0.02, op_name="mla_generation_module"),
    ]
    path = _write_csv(tmp_path / "gen_mod.txt", _MODULE_HEADER, rows)
    data = load_generation_mla_module_data(path)
    kv = next(iter(data))
    gemm = next(iter(data[kv]))
    by_native = data[kv][gemm]
    assert set(by_native.keys()) == {128}
    assert by_native[128][16][8][4097]["latency"] == pytest.approx(0.02)


def test_module_aliases_collapse_into_one_native_bucket(tmp_path):
    """The vllm 0.22.0 alias trio (V3 / R1 / V3.1-NVFP4) shares the 128-native
    geometry; under [native][local] they land in one bucket, first row wins."""
    rows = [
        _module_row(model="deepseek-ai/DeepSeek-V3", num_heads=16, lat=0.4),
        _module_row(model="deepseek-ai/DeepSeek-R1", num_heads=16, lat=0.5),
        _module_row(model="nvidia/DeepSeek-V3.1-NVFP4", num_heads=16, lat=0.6),
    ]
    path = _write_csv(tmp_path / "ctx_alias.txt", _MODULE_HEADER, rows)
    data = load_context_mla_module_data(path)
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    by_native = data[fmha][kv][gemm]
    assert set(by_native.keys()) == {128}
    assert by_native[128][16][1024][1]["latency"] == pytest.approx(0.4)


def test_load_mla_module_rejects_unpinned_model(tmp_path):
    rows = [_module_row(model="unknown/NewModel", num_heads=16)]
    path = _write_csv(tmp_path / "ctx_unknown.txt", _MODULE_HEADER, rows)
    with pytest.raises(ValueError, match="unpinned model"):
        load_context_mla_module_data(path)


def test_load_mla_module_rejects_missing_model_column(tmp_path):
    header_no_model = (
        "framework,version,device,op_name,kernel_source,architecture,"
        "mla_dtype,kv_cache_dtype,gemm_type,num_heads,batch_size,isl,tp_size,step,latency"
    )
    row = (
        "vllm,test,NVIDIA B200,mla_context_module,default,DeepseekV3ForCausalLM,"
        "bfloat16,bfloat16,bfloat16,16,1,1024,1,0,0.4"
    )
    path = _write_csv(tmp_path / "ctx_no_model.txt", header_no_model, [row])
    with pytest.raises(ValueError, match="no model column"):
        load_context_mla_module_data(path)


def test_load_mla_module_tp_rows_must_be_rank_local(tmp_path):
    """tp > 1 with num_heads * tp != native is the #1429 stale fingerprint;
    a consistent chain row (64 * 2 == 128) loads into the native bucket."""
    stale = _write_csv(tmp_path / "ctx_stale.txt", _MODULE_HEADER, [_module_row(num_heads=128, tp=2)])
    with pytest.raises(ValueError, match="rank-local"):
        load_context_mla_module_data(stale)

    ok = _write_csv(tmp_path / "ctx_tp_ok.txt", _MODULE_HEADER, [_module_row(num_heads=64, tp=2)])
    data = load_context_mla_module_data(ok)
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    assert set(data[fmha][kv][gemm].keys()) == {128}


# ───────────────────────────────────────────────────────────────────────
# Native resolution ladder (query side)
# ───────────────────────────────────────────────────────────────────────


def test_resolve_native_key_ladder():
    two = {64: "a", 128: "b"}
    assert _resolve_mla_module_native_key(two, 128) == 128  # exact
    assert _resolve_mla_module_native_key(two, 96) == 64  # nearest <=
    assert _resolve_mla_module_native_key(two, 32) == 64  # below all -> smallest
    assert _resolve_mla_module_native_key({128: "b"}, 64) == 128  # sole bucket
    assert _resolve_mla_module_native_key({128: "b"}, None) == 128  # legacy caller, one bucket
    assert _resolve_mla_module_native_key(two, None) is None  # legacy caller, ambiguous
    assert _resolve_mla_module_native_key({}, 128) is None
    # Query-side wrapper turns the ambiguous-legacy miss into a typed error.
    with pytest.raises(PerfDataNotAvailableError, match="native"):
        _require_native_bucket({64: {}, 128: {}}, None, "context")


# ───────────────────────────────────────────────────────────────────────
# Kernel loaders: retired 128 // tp_size backfill is a hard error
# ───────────────────────────────────────────────────────────────────────

_KERNEL_HEADER_NO_HEADS = "mla_dtype,kv_cache_dtype,batch_size,isl,step,tp_size,latency,kernel_source"
_KERNEL_ROW_NO_HEADS = "bfloat16,bfloat16,1,1024,1,2,0.5,flashinfer"


@pytest.mark.parametrize(
    "loader",
    [
        load_context_mla_data,
        load_generation_mla_data,
        load_wideep_context_mla_data,
        load_wideep_generation_mla_data,
    ],
)
def test_kernel_loaders_reject_rows_without_num_heads(tmp_path, loader):
    path = _write_csv(tmp_path / f"{loader.__name__}.txt", _KERNEL_HEADER_NO_HEADS, [_KERNEL_ROW_NO_HEADS])
    with pytest.raises(ValueError, match="num_heads"):
        loader(path)


# ───────────────────────────────────────────────────────────────────────
# Shipped-data guardrails (#1458)
# ───────────────────────────────────────────────────────────────────────


def _data_root() -> Path:
    import aiconfigurator_core

    return Path(aiconfigurator_core.__file__).parent / "systems" / "data"


def test_shipped_mla_module_models_are_pinned():
    """Every shipped MLA module parquet must name only pinned models, and any
    genuine tp-sweep row must be rank-local against the pinned native. A new
    module-data PR extends ``_MLA_MODULE_NATIVE_HEADS`` (both languages) or
    fails here."""
    pq = pytest.importorskip("pyarrow.parquet")
    # rglob by filename: family dirs are discovered structurally by the layout
    # resolver (any first-level dir), so path-shape globs would miss op-centric
    # placements like the mla_bmm/ dir introduced by #1435.
    files = sorted(_data_root().rglob("mla_*_module_perf.parquet"))
    assert files, f"no shipped MLA module tables found under {_data_root()}"

    offenders = []
    for path in files:
        table = pq.read_table(path, columns=["model", "num_heads", "tp_size"])
        for model, heads, tp in zip(
            table["model"].to_pylist(), table["num_heads"].to_pylist(), table["tp_size"].to_pylist(), strict=True
        ):
            native = _MLA_MODULE_NATIVE_HEADS.get(str(model))
            if native is None:
                offenders.append(f"{path.relative_to(_data_root())}: unpinned model {model!r}")
                break
            tp = max(1, int(tp))
            if tp > 1 and int(heads) * tp != native:
                offenders.append(f"{path.relative_to(_data_root())}: {model} heads={heads} tp={tp} vs native {native}")
                break
    assert not offenders, "MLA module pin violations shipped:\n" + "\n".join(offenders)


def test_shipped_mla_kernel_tables_carry_num_heads():
    """The retired ``128 // tp_size`` backfill is gone; every shipped kernel
    and WideEP row must carry the rank-local ``num_heads`` column."""
    pq = pytest.importorskip("pyarrow.parquet")
    patterns = (
        "context_mla_perf.parquet",
        "generation_mla_perf.parquet",
        "wideep_context_mla_perf.parquet",
        "wideep_generation_mla_perf.parquet",
    )
    files = [p for pattern in patterns for p in sorted(_data_root().rglob(pattern))]
    assert files, f"no shipped MLA kernel tables found under {_data_root()}"
    offenders = [
        str(p.relative_to(_data_root())) for p in files if "num_heads" not in {f.name for f in pq.read_schema(p)}
    ]
    assert not offenders, "MLA kernel tables without num_heads shipped:\n" + "\n".join(offenders)


# DSA keeps its architecture level as the model-identity key (no structural
# change in #1458); this pin turns the "one native per architecture" assumption
# from luck into a loud contract. The moment a second native ships under one
# architecture, this fails and that data PR must migrate the DSA loaders to
# [native][local] (same recipe as the MLA module tables above).
_DSA_MODEL_NATIVE_HEADS = {
    "deepseek-ai/DeepSeek-V3.2": 128,
    "zai-org/GLM-5": 64,
    "zai-org/GLM-5-FP8": 64,
    "nvidia/GLM-5-NVFP4": 64,
    "nvidia/GLM-5.2-NVFP4": 64,
    # SM90 skip-indexer probe collection (pipelines 62700025/62872230,
    # 2026-08-14..15) ships rows for the zai GLM-5.2 artifacts; same
    # 64-head geometry as the NVFP4 sibling above.
    "zai-org/GLM-5.2": 64,
    "zai-org/GLM-5.2-FP8": 64,
}


def test_shipped_dsa_module_tables_keep_one_native_per_architecture():
    pq = pytest.importorskip("pyarrow.parquet")
    files = sorted(_data_root().rglob("dsa_*_module_perf.parquet"))
    assert files, f"no shipped DSA module tables found under {_data_root()}"

    offenders = []
    for path in files:
        table = pq.read_table(path, columns=["model", "architecture", "num_heads", "tp_size"])
        natives_by_arch: dict[str, set[int]] = {}
        for model, arch, heads, tp in zip(
            table["model"].to_pylist(),
            table["architecture"].to_pylist(),
            table["num_heads"].to_pylist(),
            table["tp_size"].to_pylist(),
            strict=True,
        ):
            native = _DSA_MODEL_NATIVE_HEADS.get(str(model))
            if native is None:
                offenders.append(f"{path.relative_to(_data_root())}: unpinned model {model!r}")
                break
            natives_by_arch.setdefault(str(arch), set()).add(native)
            tp = max(1, int(tp))
            if tp > 1 and int(heads) * tp != native:
                offenders.append(f"{path.relative_to(_data_root())}: {model} heads={heads} tp={tp} vs native {native}")
                break
        else:
            for arch, natives in natives_by_arch.items():
                if len(natives) > 1:
                    offenders.append(
                        f"{path.relative_to(_data_root())}: architecture {arch} mixes natives {sorted(natives)}"
                    )
    assert not offenders, "DSA one-native-per-architecture pin violated:\n" + "\n".join(offenders)
