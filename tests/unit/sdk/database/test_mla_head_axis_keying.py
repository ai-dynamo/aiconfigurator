# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""#1458 head-axis keying: MLA module tables key ``[native][local]`` (native
from the model pin — sweeps have tp_size==1, the product cannot derive it);
kernel tables stay local-only with the ``128 // tp_size`` backfill retired.
Guardrails scan every shipped parquet. Rationale:
docs/perf_database/head-axis-keying.md.

The keying/pin behavior lives in the engine since PR-6
(``table_view.rs::view_context_mla_module`` + ``MLA_MODULE_NATIVE_HEADS`` +
``mla_module_native_heads``); these tests exercise it through the engine
table view over synthetic parquet trees. The shipped-data scans hold their
own copy of the pin table — extending it means updating the Rust table AND
this test's pin copy.
"""

from pathlib import Path

import pytest

from aiconfigurator_core.sdk.errors import PerfDataNotAvailableError
from aiconfigurator_core.sdk.perf_database import PerfDatabase

# Test-held copy of the model pin (single source: the Rust
# ``perf_database/table_view.rs::MLA_MODULE_NATIVE_HEADS``).
_MLA_MODULE_NATIVE_HEADS = {
    "deepseek-ai/DeepSeek-V3": 128,
    "deepseek-ai/DeepSeek-R1": 128,
    "nvidia/DeepSeek-V3.1-NVFP4": 128,
    "moonshotai/Kimi-K3": 96,
}

_DSV3 = "deepseek-ai/DeepSeek-V3"

_MODULE_COLUMNS = (
    "framework",
    "version",
    "device",
    "op_name",
    "kernel_source",
    "model",
    "architecture",
    "mla_dtype",
    "kv_cache_dtype",
    "gemm_type",
    "num_heads",
    "batch_size",
    "isl",
    "tp_size",
    "step",
    "latency",
)
_MODULE_INT_COLUMNS = {"num_heads", "batch_size", "isl", "tp_size", "step"}

_SYSTEM_YAML = """\
data_dir: data
gpu:
  sm_version: 90
  mem_bw: 4800000000000.0
  mem_bw_empirical_scaling_factor: 0.8
  mem_empirical_constant_latency: 0.000003
  bfloat16_tc_flops: 989000000000000.0
  fp8_tc_flops: 1978000000000000.0
node:
  num_gpus_per_node: 8
  inter_node_bw: 50000000000.0
  intra_node_bw: 450000000000.0
  p2p_latency: 0.00001
misc:
  nccl_version: '2.26.2'
"""


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
) -> dict:
    return {
        "framework": "vllm",
        "version": "test",
        "device": "NVIDIA B200",
        "op_name": op_name,
        "kernel_source": "default",
        "model": model,
        "architecture": "DeepseekV3ForCausalLM",
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "gemm_type": "bfloat16",
        "num_heads": num_heads,
        "batch_size": bs,
        "isl": isl,
        "tp_size": tp,
        "step": step,
        "latency": lat,
    }


def _write_parquet(path: Path, rows: list[dict], columns: tuple[str, ...], int_columns: set[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = {}
    for name in columns:
        values = [row[name] for row in rows]
        if name in int_columns:
            table[name] = pa.array([int(v) for v in values], type=pa.int64())
        elif name == "latency":
            table[name] = pa.array([float(v) for v in values], type=pa.float64())
        else:
            table[name] = pa.array([str(v) for v in values], type=pa.string())
    pq.write_table(pa.table(table), path)


def _view_over_parquet(tmp_path: Path, basename: str, attribute: str, write_rows) -> dict:
    """Build a minimal legacy-layout system tree holding ONE parquet, load a
    real PerfDatabase over it, and fetch the engine table view — the PR-6
    replacement for calling the retired Python loader on a bare file path.
    ``write_rows`` is a callable(path) so callers control schema deviations
    (e.g. a missing column)."""
    from aiconfigurator_core.sdk.engine_table_view import fetch_table_view

    systems_root = tmp_path / "systems"
    data_dir = systems_root / "data" / "vllm" / "test"
    data_dir.mkdir(parents=True)
    (systems_root / "testsys.yaml").write_text(_SYSTEM_YAML, encoding="utf-8")
    write_rows(data_dir / basename)
    db = PerfDatabase("testsys", "vllm", "test", str(systems_root), database_mode="SILICON")
    return fetch_table_view(db, attribute)


def _module_view(tmp_path: Path, basename: str, attribute: str, rows: list[dict]) -> dict:
    return _view_over_parquet(
        tmp_path,
        basename,
        attribute,
        lambda path: _write_parquet(path, rows, _MODULE_COLUMNS, _MODULE_INT_COLUMNS),
    )


# ───────────────────────────────────────────────────────────────────────
# Module tables: [native][local] nesting and the model pin
# ───────────────────────────────────────────────────────────────────────


def test_context_mla_module_view_keys_by_native_then_local(tmp_path):
    rows = [
        _module_row(num_heads=128, bs=1, isl=1024, lat=2.0),
        _module_row(num_heads=16, bs=1, isl=1024, lat=0.4),
        _module_row(num_heads=16, bs=4, isl=2048, lat=1.6),
    ]
    data = _module_view(tmp_path, "mla_context_module_perf.parquet", "_context_mla_module_data", rows)
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    by_native = data[fmha][kv][gemm]
    assert set(by_native.keys()) == {128}
    assert set(by_native[128].keys()) == {16, 128}
    assert by_native[128][16][2048][4]["latency"] == pytest.approx(1.6)


def test_generation_mla_module_view_keys_by_native_then_local(tmp_path):
    rows = [
        _module_row(num_heads=128, bs=8, isl=4096, step=1, lat=0.09, op_name="mla_generation_module"),
        _module_row(num_heads=16, bs=8, isl=4096, step=1, lat=0.02, op_name="mla_generation_module"),
    ]
    data = _module_view(tmp_path, "mla_generation_module_perf.parquet", "_generation_mla_module_data", rows)
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
    data = _module_view(tmp_path, "mla_context_module_perf.parquet", "_context_mla_module_data", rows)
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    by_native = data[fmha][kv][gemm]
    assert set(by_native.keys()) == {128}
    assert by_native[128][16][1024][1]["latency"] == pytest.approx(0.4)


def test_mla_module_view_rejects_unpinned_model(tmp_path):
    rows = [_module_row(model="unknown/NewModel", num_heads=16)]
    with pytest.raises(PerfDataNotAvailableError, match="unpinned model"):
        _module_view(tmp_path, "mla_context_module_perf.parquet", "_context_mla_module_data", rows)


def test_mla_module_view_rejects_missing_model_column(tmp_path):
    columns = tuple(c for c in _MODULE_COLUMNS if c != "model")
    rows = [_module_row(num_heads=16, lat=0.4)]
    with pytest.raises(PerfDataNotAvailableError, match="no model column"):
        _view_over_parquet(
            tmp_path,
            "mla_context_module_perf.parquet",
            "_context_mla_module_data",
            lambda path: _write_parquet(path, rows, columns, _MODULE_INT_COLUMNS),
        )


def test_mla_module_view_tp_rows_must_be_rank_local(tmp_path):
    """tp > 1 with num_heads * tp != native is the #1429 stale fingerprint;
    a consistent chain row (64 * 2 == 128) loads into the native bucket."""
    with pytest.raises(PerfDataNotAvailableError, match="rank-local"):
        _module_view(
            tmp_path / "stale",
            "mla_context_module_perf.parquet",
            "_context_mla_module_data",
            [_module_row(num_heads=128, tp=2)],
        )

    data = _module_view(
        tmp_path / "ok",
        "mla_context_module_perf.parquet",
        "_context_mla_module_data",
        [_module_row(num_heads=64, tp=2)],
    )
    fmha = next(iter(data))
    kv = next(iter(data[fmha]))
    gemm = next(iter(data[fmha][kv]))
    assert set(data[fmha][kv][gemm].keys()) == {128}


# Native resolution ladder (query side): retired to the compiled engine with
# #1357 PR-5 (see aic-core/rust operators; anchored by the parity goldens).


# ───────────────────────────────────────────────────────────────────────
# Kernel tables: retired 128 // tp_size backfill is a hard error
# ───────────────────────────────────────────────────────────────────────

_KERNEL_COLUMNS = (
    "mla_dtype",
    "kv_cache_dtype",
    "batch_size",
    "isl",
    "step",
    "tp_size",
    "latency",
    "kernel_source",
)
_KERNEL_INT_COLUMNS = {"batch_size", "isl", "step", "tp_size"}
_KERNEL_ROW_NO_HEADS = {
    "mla_dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "batch_size": 1,
    "isl": 1024,
    "step": 1,
    "tp_size": 2,
    "latency": 0.5,
    "kernel_source": "flashinfer",
}


@pytest.mark.parametrize(
    ("basename", "attribute"),
    [
        ("context_mla_perf.parquet", "_context_mla_data"),
        ("generation_mla_perf.parquet", "_generation_mla_data"),
        ("wideep_context_mla_perf.parquet", "_wideep_context_mla_data"),
        ("wideep_generation_mla_perf.parquet", "_wideep_generation_mla_data"),
    ],
)
def test_kernel_views_reject_rows_without_num_heads(tmp_path, basename, attribute):
    with pytest.raises(PerfDataNotAvailableError, match="num_heads"):
        _view_over_parquet(
            tmp_path,
            basename,
            attribute,
            lambda path: _write_parquet(path, [_KERNEL_ROW_NO_HEADS], _KERNEL_COLUMNS, _KERNEL_INT_COLUMNS),
        )


# ───────────────────────────────────────────────────────────────────────
# Shipped-data guardrails (#1458)
# ───────────────────────────────────────────────────────────────────────


def _data_root() -> Path:
    import aiconfigurator_core

    return Path(aiconfigurator_core.__file__).parent / "systems" / "data"


def test_shipped_mla_module_models_are_pinned():
    """Every shipped MLA module parquet must name only pinned models, and any
    genuine tp-sweep row must be rank-local against the pinned native. A new
    module-data PR extends the Rust ``MLA_MODULE_NATIVE_HEADS``
    (perf_database/table_view.rs) AND this test's pin copy, or fails here."""
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
# architecture, this fails and that data PR must migrate the DSA table views to
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


_MSA_MODEL_NATIVE_HEADS = {
    "MiniMaxAI/MiniMax-M3": 64,
}


def test_shipped_msa_module_tables_keep_one_native_per_architecture():
    """MSA twin of the DSA guardrail above: the MSA module tables reuse the
    DSA-module schema and `[architecture][local]` keying, so the same
    one-native-per-architecture invariant must hold for shipped rows."""
    pq = pytest.importorskip("pyarrow.parquet")
    files = sorted(_data_root().rglob("msa_*_module_perf.parquet"))
    assert files, f"no shipped MSA module tables found under {_data_root()}"

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
            native = _MSA_MODEL_NATIVE_HEADS.get(str(model))
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
    assert not offenders, "MSA one-native-per-architecture pin violated:\n" + "\n".join(offenders)
