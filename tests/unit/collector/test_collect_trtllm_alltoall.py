# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contract tests for the retargeted trtllm alltoall collector.

``collector/network/slurm/collect_trtllm_alltoall.py`` now emits the unified
``moe_a2a`` schema. Torch / tensorrt_llm stay behind function-level imports so
the case plan, the row mapping, the world-layout derivation and the provenance
sidecar are exercised as real code; the benchmark internals are covered by
source-contract assertions.
"""

import json
from pathlib import Path

import pytest
import yaml

from collector import provenance
from collector.network.slurm import collect_trtllm_alltoall as ata

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "network" / "slurm" / "collect_trtllm_alltoall.py"
SOURCE_TEXT = SOURCE_PATH.read_text()

# The frozen moe_a2a CSV header — the SAME literal Task 4's
# tests/unit/collector/test_collect_moe_a2a.py::MOE_A2A_HEADER pins for the
# sglang DeepEP writer: both collectors share _build_moe_a2a_row and both
# tables are read by load_moe_a2a_data (aic-core .../sdk/operations/
# moe_comm.py ~:378-395). The SDK-side twin is
# tests/unit/sdk/database/test_collector_schema_contract.py::MOE_A2A_HEADER.
MOE_A2A_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "comm_backend,phase,comm_dtype,ep_size,node_num,hidden_size,topk,num_experts,"
    "num_tokens,sms,transmit_us,notify_us,latency"
)


def _case(ep_size=8, num_tokens=4096, moe_dtype=ata.MoEDtype.NVFP4) -> ata.AlltoallTestCase:
    return ata.AlltoallTestCase(
        num_tokens=num_tokens,
        hidden_size=7168,
        num_experts=256,
        top_k=8,
        ep_size=ep_size,
        moe_dtype=moe_dtype,
    )


def _two_sided_result() -> ata.AlltoallBenchmarkResult:
    return ata.AlltoallBenchmarkResult(
        prepare_latency_ms=0.05,
        dispatch_latency_ms=0.85,
        combine_latency_ms=1.2,
        combine_low_precision_latency_ms=0.6,
    )


# ---------------------------------------------------------------------------
# Mapping contract — must mirror the SDK legacy adapter EXACTLY
# ---------------------------------------------------------------------------


def test_mappings_mirror_the_sdk_legacy_trtllm_adapter():
    # The oracle: aic-core's _adapt_legacy_trtllm_alltoall keys legacy rows
    # into the unified store with exactly these two maps. A drift here would
    # put new-schema rows on different keys than adapted legacy rows.
    from aiconfigurator_core.sdk.operations import moe_comm

    assert ata.KERNEL_SOURCE_TO_COMM_BACKEND == moe_comm._LEGACY_TRTLLM_KERNEL_TO_BACKEND
    assert ata.OP_TO_PHASE_DTYPE == moe_comm._LEGACY_TRTLLM_OP_TO_PHASE_DTYPE


def test_four_way_op_name_to_phase_mapping_including_fp4():
    rows = ata.build_unified_rows(_case(), _two_sided_result(), kernel_source="NVLinkTwoSided", node_num=2)
    assert [(row["phase"], row["comm_dtype"]) for row in rows] == [
        # D5 within a case: ascending (phase, comm_dtype); the low-precision
        # combine keys as "fp4", every other row as the run's moe_dtype.
        ("combine", "fp4"),
        ("combine", "nvfp4"),
        ("dispatch", "nvfp4"),
        ("prepare", "nvfp4"),
    ]
    assert all(row["comm_backend"] == "nvlink_two_sided" for row in rows)
    assert all(row["sms"] == 0 for row in rows)  # legacy alltoall rows carry no SM budget


def test_one_sided_emits_no_prepare_row_and_maps_to_nvlink_one_sided():
    result = ata.AlltoallBenchmarkResult(dispatch_latency_ms=0.4, combine_latency_ms=0.7)
    rows = ata.build_unified_rows(
        _case(moe_dtype=ata.MoEDtype.BFLOAT16), result, kernel_source="NVLinkOneSided", node_num=1
    )
    assert [(row["phase"], row["comm_dtype"]) for row in rows] == [
        ("combine", "bfloat16"),
        ("dispatch", "bfloat16"),
    ]
    assert all(row["comm_backend"] == "nvlink_one_sided" for row in rows)


def test_latency_is_emitted_in_microseconds_as_transmit_with_zero_notify():
    # UNITS: load_moe_a2a_data divides the latency column by 1000
    # (moe_comm.py:393 "collector records us") while the adapted LEGACY trtllm
    # ms rows are stored raw, so a 0.85 ms measurement must be written as
    # 850.0 for both paths to reach the same ms leaf. The trtllm measurement
    # has no transmit/notify split (the DeepEP LL precedent,
    # moe_comm.py:232-234): the whole latency rides transmit_us.
    [row] = [
        row
        for row in ata.build_unified_rows(_case(), _two_sided_result(), kernel_source="NVLinkTwoSided", node_num=2)
        if row["phase"] == "dispatch"
    ]
    assert row["latency"] == pytest.approx(850.0)
    assert row["transmit_us"] == pytest.approx(850.0)
    assert row["notify_us"] == 0.0


# ---------------------------------------------------------------------------
# Writer contract — the frozen unified header
# ---------------------------------------------------------------------------


def test_writer_emits_the_frozen_moe_a2a_header(tmp_path):
    from collector.helper import finalize_perf_files, log_perf

    perf_file = tmp_path / "moe_a2a_perf.txt"
    for row in ata.build_unified_rows(_case(), _two_sided_result(), kernel_source="NVLinkTwoSided", node_num=2):
        assert log_perf(
            item_list=[row],
            framework=ata.FRAMEWORK,
            version="1.3.0rc10",
            device_name="NVIDIA GB200",
            op_name=ata.OP_NAME,
            kernel_source="NVLinkTwoSided",
            perf_filename=str(perf_file),
        )
    assert perf_file.read_text().splitlines()[0] == MOE_A2A_HEADER

    [parquet_path] = finalize_perf_files([perf_file])
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    assert table.column_names == MOE_A2A_HEADER.split(",")
    assert "power" not in table.column_names  # Task 4 ruling: no power column on this table
    dispatch = [row for row in table.to_pylist() if row["phase"] == "dispatch"][0]
    assert dispatch == {
        "framework": "TRTLLM",
        "version": "1.3.0rc10",
        "device": "NVIDIA GB200",
        "op_name": "moe_a2a",
        # kernel_source stays the declared ground-truth label
        # (collector/kernel_source_backends.yaml:162-163); comm_backend
        # carries the unified consumer key.
        "kernel_source": "NVLinkTwoSided",
        "comm_backend": "nvlink_two_sided",
        "phase": "dispatch",
        "comm_dtype": "nvfp4",
        "ep_size": 8,
        "node_num": 2,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 4096,
        "sms": 0,
        "transmit_us": pytest.approx(850.0),
        "notify_us": pytest.approx(0.0),
        "latency": pytest.approx(850.0),
    }


def test_comparability_new_writer_leaf_equals_adapted_legacy_leaf(tmp_path):
    # THE comparability pin: one 0.85 ms dispatch measurement (nvfp4, ep=8 on
    # 4-GPU nodes) written by the new writer and read through the new-schema
    # loader must land on the same key with the same ms leaf as the same
    # measurement in a legacy trtllm_alltoall row through the adapter.
    from aiconfigurator_core.sdk.operations import moe_comm
    from collector.helper import finalize_perf_files, log_perf

    legacy_rows = [
        {
            "kernel_source": "NVLinkTwoSided",
            "op_name": op_name,
            "moe_dtype": "nvfp4",
            "num_tokens": "4096",
            "hidden_size": "7168",
            "topk": "8",
            "num_experts": "256",
            "moe_ep_size": "8",  # legacy files carry no num_nodes: adapter derives 8 // 4 = 2
            "distribution": "balanced",
            "latency": str(latency_ms),
        }
        for op_name, latency_ms in [
            ("alltoall_prepare", 0.05),
            ("alltoall_dispatch", 0.85),
            ("alltoall_combine", 1.2),
            ("alltoall_combine_low_precision", 0.6),
        ]
    ]
    legacy_store = moe_comm._moe_a2a_store()
    moe_comm._adapt_legacy_trtllm_alltoall(legacy_store, legacy_rows)

    perf_file = tmp_path / "moe_a2a_perf.txt"
    for row in ata.build_unified_rows(_case(), _two_sided_result(), kernel_source="NVLinkTwoSided", node_num=2):
        log_perf(
            item_list=[row],
            framework=ata.FRAMEWORK,
            version="1.3.0rc10",
            device_name="NVIDIA GB200",
            op_name=ata.OP_NAME,
            kernel_source="NVLinkTwoSided",
            perf_filename=str(perf_file),
        )
    [parquet_path] = finalize_perf_files([perf_file])
    loaded = moe_comm.load_moe_a2a_data(str(parquet_path))

    for phase, comm_dtype, latency_ms in [
        ("prepare", "nvfp4", 0.05),
        ("dispatch", "nvfp4", 0.85),
        ("combine", "nvfp4", 1.2),
        ("combine", "fp4", 0.6),
    ]:
        legacy_leaf = legacy_store["nvlink_two_sided"][phase][comm_dtype][8][2][7168][8][256][0][4096]
        new_leaf = loaded["nvlink_two_sided"][phase][comm_dtype][8][2][7168][8][256][0][4096]
        assert legacy_leaf["latency"] == pytest.approx(latency_ms), (phase, comm_dtype)
        assert new_leaf["latency"] == pytest.approx(legacy_leaf["latency"]), (phase, comm_dtype)


# ---------------------------------------------------------------------------
# World layout — node_num from the environment, never fabricated
# ---------------------------------------------------------------------------


def test_gpus_per_node_explicit_arg_wins():
    assert ata.resolve_gpus_per_node(4, {"SLURM_NTASKS_PER_NODE": "8"}) == 4


def test_gpus_per_node_falls_back_to_slurm_env():
    assert ata.resolve_gpus_per_node(None, {"SLURM_NTASKS_PER_NODE": "4"}) == 4


def test_gpus_per_node_raises_without_a_source():
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="gpus-per-node"):
        ata.resolve_gpus_per_node(None, {})
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="positive"):
        ata.resolve_gpus_per_node(0, {})


def test_node_num_is_world_size_over_gpus_per_node():
    assert ata.derive_node_num(8, 4) == 2
    assert ata.derive_node_num(72, 4) == 18
    assert ata.derive_node_num(2, 2) == 1  # sub-node job: launcher packs 2 tasks on one node


def test_node_num_raises_on_non_integral_worlds():
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="not an integral number of nodes"):
        ata.derive_node_num(6, 4)


# ---------------------------------------------------------------------------
# Case plan — D5 order, counted drops, zero-case raise
# ---------------------------------------------------------------------------


def test_case_plan_is_emitted_in_d5_sort_order():
    cases = ata.get_default_test_cases(8)
    assert len(cases) == 30  # 1 shape x 30 token counts x 1 dtype
    keys = [case.sort_key() for case in cases]
    assert keys == sorted(keys)
    tokens = [case.num_tokens for case in cases]
    assert tokens == sorted(tokens)


def test_case_plan_drops_are_counted_and_zero_cases_raise(capsys):
    ata.get_default_test_cases(8)
    assert "dropped: 0 shapes" in capsys.readouterr().out
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="zero cases"):
        ata.get_default_test_cases(384)  # 256 experts cannot shard over 384 ranks


def test_case_plan_ids_are_world_and_kernel_source_specific():
    cases = ata.get_default_test_cases(8)
    ids_two_sided = ata.case_plan_ids(cases, kernel_source="NVLinkTwoSided", node_num=2)
    ids_one_sided = ata.case_plan_ids(cases, kernel_source="NVLinkOneSided", node_num=2)
    assert len(ids_two_sided) == len(cases)
    assert all(entry.startswith(f"{ata.MODULE_NAME}:run_case:") for entry in ids_two_sided)
    assert provenance.case_plan_hash(ids_two_sided) != provenance.case_plan_hash(ids_one_sided)


# ---------------------------------------------------------------------------
# Provenance — membership, closure, sidecar
# ---------------------------------------------------------------------------


def test_module_is_an_enrolled_standalone_collector_with_a_real_closure():
    assert ata.MODULE_NAME in provenance.STANDALONE_COLLECTOR_MODULES
    closures = provenance.load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    assert ata.MODULE_NAME in closures
    digest = provenance.collector_hash(ata.MODULE_NAME, REPO_ROOT, closures)
    assert digest.startswith("sha256:")


def test_sidecar_is_written_with_the_trtllm_module_identity(tmp_path):
    from collector.helper import finalize_perf_files, log_perf
    from collector.wideep.sglang.collect_moe_a2a import write_moe_a2a_sidecar

    perf_file = tmp_path / "moe_a2a_perf.txt"
    rows = ata.build_unified_rows(_case(), _two_sided_result(), kernel_source="NVLinkTwoSided", node_num=2)
    for row in rows:
        log_perf(
            item_list=[row],
            framework=ata.FRAMEWORK,
            version="1.3.0rc10",
            device_name="NVIDIA GB200",
            op_name=ata.OP_NAME,
            kernel_source="NVLinkTwoSided",
            perf_filename=str(perf_file),
        )
    [parquet_path] = finalize_perf_files([perf_file])

    case_ids = ata.case_plan_ids([_case()], kernel_source="NVLinkTwoSided", node_num=2)
    runtime_meta = {"framework": "trtllm", "version": "1.3.0rc10", "image": "img", "image_digest": "sha256:0"}
    meta_path = write_moe_a2a_sidecar(
        tmp_path,
        runtime_meta=runtime_meta,
        case_ids=case_ids,
        parquet_path=parquet_path,
        failure_count=0,
        module_name=ata.MODULE_NAME,
    )
    doc = yaml.safe_load(meta_path.read_text())
    table = doc["tables"]["moe_a2a_perf"]
    assert doc["runtime"]["framework"] == "trtllm"
    assert table["rows"] == len(rows)
    assert table["status"] == "complete"
    assert table["case_plan_hash"] == provenance.case_plan_hash(case_ids)
    closures = provenance.load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    assert table["collector_hash"] == provenance.collector_hash(ata.MODULE_NAME, REPO_ROOT, closures)


def test_runtime_meta_gates_the_installed_version_against_the_manifest_pin():
    manifest = yaml.safe_load((REPO_ROOT / "collector" / "framework_manifest.yaml").read_text())
    spec = manifest["frameworks"]["trtllm"]["default"]
    pinned = spec["version"]
    meta = ata.resolve_runtime_meta(pinned, spec["images"]["default"])
    assert meta["framework"] == "trtllm"
    assert meta["version"] == pinned
    assert "@" not in meta["image"]
    assert meta["image_variant"] == "default"
    assert meta["image_digest"].startswith("sha256:")
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="manifest trtllm pin"):
        ata.resolve_runtime_meta("1.2.0rc5", spec["images"]["default"])


def test_runtime_meta_attests_the_launched_image_only(tmp_path):
    # F17: CONTAINER_IMAGE is operator-overridable in the launcher, so the
    # sidecar must attest the ref that actually ran — a ref outside the
    # manifest pins is refused, and a missing ref cannot silently fall back
    # to the manifest default.
    manifest = yaml.safe_load((REPO_ROOT / "collector" / "framework_manifest.yaml").read_text())
    spec = manifest["frameworks"]["trtllm"]["default"]
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="--image-ref"):
        ata.resolve_runtime_meta(spec["version"], None)
    with pytest.raises(ata.TrtllmAlltoallDeclarationError, match="not a manifest trtllm image variant"):
        ata.resolve_runtime_meta(spec["version"], "someone/rebuilt-trtllm:latest")


def test_stale_staging_artifacts_are_refused_before_any_write():
    # F19: log_perf appends; a resubmission into a directory with a previous
    # attempt's CSV would finalize duplicates under this run's attestation.
    assert "stale_output_artifacts(output_dir" in SOURCE_TEXT
    assert "refuses to run into" in SOURCE_TEXT


def test_rank0_persistence_result_is_agreed_before_the_next_collective():
    # F18: a rank-0-only write failure must become a classified case failure
    # every rank agrees on, not an exception peers never see (they would hang
    # in the next case's barrier).
    assert "persist_failed" in SOURCE_TEXT
    assert 'op_name="alltoall_persistence"' in SOURCE_TEXT


def test_classified_failure_records_are_rank_scoped(tmp_path):
    case = _case()
    ata.record_failure(tmp_path, case, RuntimeError("boom"), rank=3, kernel_source="NVLinkTwoSided", node_num=2)
    ata.record_failure(
        tmp_path,
        case,
        RuntimeError("lp boom"),
        rank=3,
        kernel_source="NVLinkTwoSided",
        node_num=2,
        op_name="alltoall_combine_low_precision",
    )
    records = json.loads((tmp_path / "errors_trtllm_alltoall.rank3.json").read_text())
    assert [record["op"] for record in records] == ["moe_a2a", "alltoall_combine_low_precision"]
    assert all(record["classification"] == "unexpected" for record in records)
    assert records[0]["case"]["node_num"] == 2


# ---------------------------------------------------------------------------
# Source contract — no silent skips, no power, no legacy writer
# ---------------------------------------------------------------------------


def test_no_power_column_is_ever_emitted():
    # Task 4 ruling (collect_moe_a2a._power_columns): the moe_a2a table emits
    # no power column, per TABLE — one shared file, one header, and no timed
    # region here that could be re-scoped per emitted row family-wide.
    assert "power_stats" not in SOURCE_TEXT
    assert "measure_power=False" in SOURCE_TEXT


def test_legacy_writer_and_silent_skips_are_gone():
    assert "log_alltoall_perf" not in SOURCE_TEXT  # D4: legacy-format emission removed
    assert "traceback.print_exc" not in SOURCE_TEXT  # failures are classified records now
    assert "record_failure(" in SOURCE_TEXT
    assert "all_reduce" in SOURCE_TEXT  # ranks agree whether a case produced data
    # MNNVL-unsupported is execute-or-raise, not a silent early return.
    assert "def require_mnnvl_support" in SOURCE_TEXT
    assert "def check_mnnvl_support" not in SOURCE_TEXT
    # The low-precision combine probe failure rides out to a classified record.
    assert "combine_low_precision_error" in SOURCE_TEXT


def test_no_fabricated_node_num_derivation_remains():
    # The legacy loader fabricated node_num = max(1, ep_size // 4); the
    # collector must derive it from the launcher's declared layout instead
    # (the docstrings cite the legacy expression, so match its code shape).
    assert "node_num = max(" not in SOURCE_TEXT
    assert "world_size // gpus_per_node" in SOURCE_TEXT
