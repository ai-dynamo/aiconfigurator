# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free contract tests for the standalone ``moe_a2a`` collector.

``collector/wideep/sglang/collect_moe_a2a.py`` keeps torch / deep_ep behind
function-level imports precisely so the case plan, the row builder, the
distributed-identity derivation and the provenance sidecar can be exercised
without a GPU (DeepEP and torch.distributed cannot run in this environment).
The benchmark internals are covered by source-contract assertions instead.
"""

import json
from pathlib import Path

import pytest
import yaml

from collector.wideep.sglang import collect_moe_a2a as a2a

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "wideep" / "sglang" / "collect_moe_a2a.py"
SOURCE_TEXT = SOURCE_PATH.read_text()

# The frozen moe_a2a CSV header: helper.log_perf's five prefix columns plus
# this collector's payload, in the order load_moe_a2a_data keys them
# (aic-core .../sdk/operations/moe_comm.py::load_moe_a2a_data ~:378-395).
# The SDK-side twin pinning the same literal against the real loader is
# tests/unit/sdk/database/test_collector_schema_contract.py::MOE_A2A_HEADER.
MOE_A2A_HEADER = (
    "framework,version,device,op_name,kernel_source,"
    "comm_backend,phase,comm_dtype,ep_size,node_num,hidden_size,topk,num_experts,"
    "num_tokens,sms,transmit_us,notify_us,latency"
)

SHAPE = a2a.MoeA2AShape(hidden_size=7168, topk=8, num_experts=256)


# ---------------------------------------------------------------------------
# Base-op yaml expansion
# ---------------------------------------------------------------------------


def test_base_op_yaml_declares_the_three_workload_axes():
    grid = a2a.get_moe_a2a_workload_grid()
    assert grid["sms"] == [16, 20, 24]
    assert grid["ht_token_counts"][0] == 16
    assert grid["ht_token_counts"][-1] == 65536
    assert grid["ll_token_counts"] == [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 256]
    # Every axis ascending — the declared grid is the D5 sort order's source.
    for axis in grid.values():
        assert axis == sorted(axis)


def test_declared_shapes_come_from_the_wideep_model_rows(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    shapes = a2a.get_moe_a2a_shapes()
    # The persisted comm key has no model column, so the declared wideep
    # model rows collapse onto 7 physical (hidden, topk, experts) tuples.
    assert shapes == [
        a2a.MoeA2AShape(3072, 8, 256),
        a2a.MoeA2AShape(3584, 16, 896),
        a2a.MoeA2AShape(4096, 6, 256),
        a2a.MoeA2AShape(6144, 8, 256),
        a2a.MoeA2AShape(7168, 6, 384),
        a2a.MoeA2AShape(7168, 8, 256),
        a2a.MoeA2AShape(7168, 8, 384),
    ]
    assert shapes == sorted(shapes)


def test_shapes_carry_the_declared_routing(monkeypatch):
    # F16: the workload's routing is a declared fact riding on the shape —
    # DeepSeek-V3-style rows declare group-limited routing (8 groups, top-4),
    # Kimi/GLM rows declare global routing (1/1). The routing fields stay out
    # of the shape's identity (compare=False), so the persisted key is
    # unchanged.
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    routing = {
        (shape.hidden_size, shape.topk, shape.num_experts): (shape.num_expert_group, shape.topk_group)
        for shape in a2a.get_moe_a2a_shapes()
    }
    assert routing[(7168, 8, 256)] == (8, 4)  # DeepSeek-V3/R1
    assert routing[(7168, 8, 384)] == (1, 1)  # Kimi-K2.5
    assert routing[(3584, 16, 896)] == (1, 1)  # Kimi-K3


def _fake_moe_recipe(model_name, hidden_size, topk, num_experts, *, num_expert_group=1, topk_group=1):
    from types import SimpleNamespace

    return SimpleNamespace(
        model_name=model_name,
        hidden_size=hidden_size,
        topk=topk,
        num_experts=num_experts,
        sglang_moe_num_expert_group=num_expert_group,
        sglang_moe_topk_group=topk_group,
    )


def test_conflicting_routing_declarations_raise(monkeypatch):
    # Two models sharing a (hidden, topk, experts) key but disagreeing on
    # routing would write indistinguishable rows measured under different
    # traffic patterns — an unresolvable declaration, so it fails loudly.
    import collector.case_generator as case_generator

    monkeypatch.setattr(
        case_generator,
        "get_common_moe_test_cases",
        lambda backend: [
            _fake_moe_recipe("model-a", 7168, 8, 256, num_expert_group=8, topk_group=4),
            _fake_moe_recipe("model-b", 7168, 8, 256, num_expert_group=1, topk_group=1),
        ],
    )
    monkeypatch.setattr(case_generator, "is_wideep_moe_model", lambda name: True)
    with pytest.raises(a2a.MoeA2ADeclarationError, match="conflicting routing"):
        a2a.get_moe_a2a_shapes()


def test_ht_topk_group_budget_derives_from_the_declaration():
    grouped = a2a.MoeA2AShape(7168, 8, 256, num_expert_group=8, topk_group=4)
    global_routing = a2a.MoeA2AShape(7168, 8, 384, num_expert_group=1, topk_group=1)
    # DeepSeek-style: the deepep test's min(nodes, 4), with 4 now sourced
    # from the declared topk_group instead of a hardcoded constant.
    assert a2a.ht_num_topk_groups(grouped, num_nodes=2) == 2
    assert a2a.ht_num_topk_groups(grouped, num_nodes=8) == 4
    assert a2a.ht_num_topk_groups(grouped, num_nodes=18) == 4
    # Global routing: every node group stays selectable, masking degenerates
    # to plain top-k at any world.
    assert a2a.ht_num_topk_groups(global_routing, num_nodes=8) == 8
    assert a2a.ht_num_topk_groups(global_routing, num_nodes=18) == 18


def test_case_plan_ids_carry_the_routing_identity():
    shape = a2a.MoeA2AShape(7168, 8, 256, num_expert_group=8, topk_group=4)
    case = a2a.MoeA2ACase("deepep_ht", shape, num_tokens=1024, sms=20)
    [case_id] = a2a.case_plan_ids([case], ep_size=16, node_num=4)
    payload = json.loads(case_id.split(":run_case:", 1)[1])
    assert payload["num_expert_group"] == 8
    assert payload["topk_group"] == 4


def test_shapes_stay_correlated_never_crossed(monkeypatch):
    # topk/num_experts travel with their hidden_size: 4096 is DeepSeek-V4-Flash
    # (topk 6, 256 experts) and never appears with another model's topk.
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    by_hidden = {}
    for shape in a2a.get_moe_a2a_shapes():
        by_hidden.setdefault(shape.hidden_size, set()).add((shape.topk, shape.num_experts))
    assert by_hidden[4096] == {(6, 256)}
    assert by_hidden[3072] == {(8, 256)}
    assert by_hidden[6144] == {(8, 256)}


def test_case_plan_is_the_grid_times_the_shapes(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    shapes = a2a.get_moe_a2a_shapes()
    grid = a2a.get_moe_a2a_workload_grid()
    cases = a2a.build_case_plan(shapes=shapes, grid=grid, ep_size=16, node_num=2)

    ht = [case for case in cases if case.comm_backend == "deepep_ht"]
    ll = [case for case in cases if case.comm_backend == "deepep_ll"]
    assert len(ht) == 7 * 3 * 13 == 273
    assert len(ll) == 7 * 14 == 98
    assert len(cases) == 371
    # LL rows carry no SM budget — the value the SDK's legacy adapter assigns.
    assert {case.sms for case in ll} == {0}


def test_case_plan_is_emitted_in_d5_sort_order(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    cases = a2a.build_case_plan(
        shapes=a2a.get_moe_a2a_shapes(), grid=a2a.get_moe_a2a_workload_grid(), ep_size=16, node_num=2
    )
    keys = [case.sort_key() for case in cases]
    assert keys == sorted(keys)
    # Ascending on every non-token key axis at each nesting level of the
    # consumer's store: for a fixed shape, sms is first seen ascending.
    for shape in {case.shape for case in cases}:
        sms_order = []
        for case in cases:
            if case.comm_backend == "deepep_ht" and case.shape == shape and case.sms not in sms_order:
                sms_order.append(case.sms)
        assert sms_order == sorted(sms_order)


def test_shapes_not_divisible_by_the_world_are_dropped_with_a_count(capsys):
    shapes = [a2a.MoeA2AShape(7168, 8, 256), a2a.MoeA2AShape(7168, 6, 384)]
    grid = {"ht_token_counts": [128], "ll_token_counts": [8], "sms": [24]}
    cases = a2a.build_case_plan(shapes=shapes, grid=grid, ep_size=256, node_num=32)
    # 384 % 256 != 0 -> that shape cannot be sharded over this world.
    assert {case.shape for case in cases} == {a2a.MoeA2AShape(7168, 8, 256)}
    assert "1 shapes with num_experts % ep_size != 0" in capsys.readouterr().out


def test_zero_case_expansion_raises_with_the_logged_reasons():
    shapes = [a2a.MoeA2AShape(7168, 6, 384)]
    grid = {"ht_token_counts": [128], "ll_token_counts": [8], "sms": [24]}
    with pytest.raises(a2a.MoeA2ADeclarationError, match="expanded to zero cases"):
        a2a.build_case_plan(shapes=shapes, grid=grid, ep_size=256, node_num=32)


def test_case_plan_hash_is_world_specific(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    shapes = a2a.get_moe_a2a_shapes()
    grid = a2a.get_moe_a2a_workload_grid()
    from collector import provenance

    def plan_hash(ep_size, node_num):
        cases = a2a.build_case_plan(shapes=shapes, grid=grid, ep_size=ep_size, node_num=node_num)
        return provenance.case_plan_hash(a2a.case_plan_ids(cases, ep_size=ep_size, node_num=node_num))

    # Same grid, different world = a different attested plan.
    assert plan_hash(16, 2) != plan_hash(32, 4)
    assert plan_hash(16, 2) == plan_hash(16, 2)


def test_base_op_yaml_is_a_valid_collector_case_file():
    doc = yaml.safe_load((REPO_ROOT / "collector" / "cases" / "base_ops" / "moe_a2a.yaml").read_text())
    assert doc["schema_version"] == 1
    assert doc["op"] == "moe_a2a"
    assert set(doc["common_case_values"]["moe_a2a"]) == {"ht_token_counts", "ll_token_counts", "sms"}


# ---------------------------------------------------------------------------
# Distributed identity
# ---------------------------------------------------------------------------


def test_node_num_is_derived_from_world_size_and_gpus_per_node():
    identity = a2a.derive_dist_identity(
        {"RANK": "9", "WORLD_SIZE": "32", "LOCAL_RANK": "1", "MASTER_ADDR": "10.0.0.1"},
        gpus_per_node=8,
    )
    assert (identity.rank, identity.world_size, identity.local_rank) == (9, 32, 1)
    assert identity.node_num == 4
    assert identity.ep_size == 32
    assert identity.master_addr == "10.0.0.1"


def test_slurm_environment_variables_are_honoured():
    identity = a2a.derive_dist_identity(
        {"SLURM_PROCID": "5", "SLURM_NTASKS": "16", "SLURM_LOCALID": "5"}, gpus_per_node=8
    )
    assert (identity.rank, identity.world_size, identity.local_rank, identity.node_num) == (5, 16, 5, 2)


def test_non_integral_node_count_raises():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="not an integral number of nodes"):
        a2a.derive_dist_identity({"RANK": "0", "WORLD_SIZE": "20"}, gpus_per_node=8)


def test_gpus_per_node_is_cross_checked_against_the_visible_devices():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="does not match the 4 CUDA device"):
        a2a.derive_dist_identity({"RANK": "0", "WORLD_SIZE": "16"}, gpus_per_node=8, visible_device_count=4)


def test_non_positive_gpus_per_node_raises():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="must be positive"):
        a2a.derive_dist_identity({"RANK": "0", "WORLD_SIZE": "8"}, gpus_per_node=0)


def test_node_num_never_comes_from_a_filename():
    # The retired pipeline parsed node_num out of a log FILENAME
    # (deepep/extract_data.py:_extract_node_num_from_filename).
    assert "_extract_node_num_from_filename" not in SOURCE_TEXT
    assert "world_size // gpus_per_node" in SOURCE_TEXT


# ---------------------------------------------------------------------------
# Writer contract
# ---------------------------------------------------------------------------


def test_row_builder_emits_the_frozen_moe_a2a_payload(tmp_path):
    from collector.helper import finalize_perf_files, log_perf

    row = a2a._build_moe_a2a_row(
        comm_backend="deepep_ht",
        phase="dispatch",
        ep_size=32,
        node_num=4,
        shape=SHAPE,
        num_tokens=4096,
        sms=24,
        transmit_us=812.5,
        notify_us=37.5,
    )

    perf_file = tmp_path / "moe_a2a_perf.txt"
    assert log_perf(
        item_list=[row],
        framework="SGLang",
        version="0.5.10",
        device_name="NVIDIA B200",
        op_name=a2a.OP_NAME,
        kernel_source=a2a.KERNEL_SOURCE,
        perf_filename=str(perf_file),
    )
    assert perf_file.read_text().splitlines()[0] == MOE_A2A_HEADER

    [parquet_path] = finalize_perf_files([perf_file])
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    assert table.column_names == MOE_A2A_HEADER.split(",")
    assert table.to_pylist()[0] == {
        "framework": "SGLang",
        "version": "0.5.10",
        "device": "NVIDIA B200",
        "op_name": "moe_a2a",
        "kernel_source": "deepep",
        "comm_backend": "deepep_ht",
        "phase": "dispatch",
        # The DeepEP legs' dtype key: the SDK's legacy adapters store DeepEP
        # rows under "default" (moe_comm.py:191), so new-schema rows must key
        # identically to overwrite/extend those leaves.
        "comm_dtype": "default",
        "ep_size": 32,
        "node_num": 4,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 4096,
        "sms": 24,
        "transmit_us": pytest.approx(812.5),
        "notify_us": pytest.approx(37.5),
        # MICROSECONDS: load_moe_a2a_data divides this column by 1000
        # (moe_comm.py:393 "collector records us"), matching the legacy
        # adapter's us-summed transmit+notify (moe_comm.py:200-204).
        "latency": pytest.approx(850.0),
    }


def test_latency_is_transmit_plus_notify():
    row = a2a._build_moe_a2a_row(
        comm_backend="deepep_ht",
        phase="combine",
        ep_size=16,
        node_num=2,
        shape=SHAPE,
        num_tokens=512,
        sms=16,
        transmit_us=101.25,
        notify_us=9.75,
    )
    assert row["latency"] == pytest.approx(row["transmit_us"] + row["notify_us"]) == pytest.approx(111.0)


def test_ll_rows_carry_no_notify_split_and_no_sm_budget():
    row = a2a._build_moe_a2a_row(
        comm_backend="deepep_ll",
        phase="dispatch",
        ep_size=16,
        node_num=2,
        shape=SHAPE,
        num_tokens=128,
        sms=a2a.LL_SMS,
        transmit_us=64.0,
        notify_us=0.0,
    )
    # The legacy LL table had no transmit/notify split either — its adapter
    # sums a single per-phase average (moe_comm.py:232-234) — and LL rows key
    # on sms=0 (moe_comm.py:191).
    assert row["sms"] == 0
    assert row["latency"] == pytest.approx(64.0)


def test_both_phases_share_one_table_and_one_header(tmp_path):
    import csv

    from collector.helper import log_perf

    perf_file = tmp_path / "moe_a2a_perf.txt"
    for phase in ("combine", "dispatch"):
        log_perf(
            item_list=[
                a2a._build_moe_a2a_row(
                    comm_backend="deepep_ht",
                    phase=phase,
                    ep_size=16,
                    node_num=2,
                    shape=SHAPE,
                    num_tokens=64,
                    sms=16,
                    transmit_us=10.0,
                    notify_us=1.0,
                )
            ],
            framework="SGLang",
            version="0.5.10",
            device_name="NVIDIA B200",
            op_name=a2a.OP_NAME,
            kernel_source=a2a.KERNEL_SOURCE,
            perf_filename=str(perf_file),
        )
    with open(perf_file, newline="") as handle:
        rows = list(csv.DictReader(handle))
    # D5: combine before dispatch keeps the store's phase level ascending.
    assert [row["phase"] for row in rows] == ["combine", "dispatch"]
    assert perf_file.read_text().count("comm_backend") == 1


def test_no_power_column_is_emitted(tmp_path, monkeypatch):
    # D7: absent is the loader's supported case (has_power = "power" in
    # rows[0]); a fabricated 0.0 or a present-but-null column is not. This
    # table emits no power at all because no sampled region corresponds to a
    # single row's workload -- see _power_columns' docstring.
    from collector.helper import log_perf

    assert a2a._power_columns() is None
    monkeypatch.setenv("COLLECTOR_MEASURE_POWER", "1")
    assert a2a._power_columns() is None

    perf_file = tmp_path / "moe_a2a_perf.txt"
    log_perf(
        item_list=[
            a2a._build_moe_a2a_row(
                comm_backend="deepep_ht",
                phase="dispatch",
                ep_size=16,
                node_num=2,
                shape=SHAPE,
                num_tokens=64,
                sms=16,
                transmit_us=10.0,
                notify_us=1.0,
            )
        ],
        framework="SGLang",
        version="0.5.10",
        device_name="NVIDIA B200",
        op_name=a2a.OP_NAME,
        kernel_source=a2a.KERNEL_SOURCE,
        perf_filename=str(perf_file),
        power_stats=a2a._power_columns(),
    )
    header = perf_file.read_text().splitlines()[0]
    assert "power" not in header
    assert header == MOE_A2A_HEADER


def test_power_is_never_sampled_across_the_tuning_sweep():
    # The sampler must not wrap run_ht_case/run_ll_case: an HT row's latency
    # is one winning config out of a 116-configuration sweep, so power
    # averaged over the sweep would give the loader energy = power x latency
    # from two different workloads.
    assert "power_monitoring_only" not in SOURCE_TEXT
    assert "stop_sampling" not in SOURCE_TEXT


def test_ll_buffer_is_allocated_only_after_the_ht_buffers_are_torn_down():
    # DeepEP low_latency_mode=True and =False Buffers are allocated in
    # mutually exclusive branches in the source scripts and would otherwise
    # double the resident RDMA/NVL allocation for the whole run.
    main_body = SOURCE_TEXT.split("def main(", 1)[1]
    pre_loop, run_loop = main_body.split("for case in cases:", 1)
    assert "create_ll_buffer(" not in pre_loop
    assert run_loop.index("ht_buffer = None") < run_loop.index("ll_buffer = create_ll_buffer(")


def test_measurement_sync_and_warmup_are_ported():
    # test_internode.py:123-124 settles the ranks before the first profiled
    # dispatch; test_low_latency.py reaches its reported numbers only after
    # utils.bench's warmup passes.
    assert "group.barrier()\n    time.sleep(1)" in SOURCE_TEXT
    assert a2a.LL_WARMUP_ITERS == 50
    assert "for _ in range(LL_WARMUP_ITERS):" in SOURCE_TEXT


def test_per_case_reseeding_is_documented_as_a_deviation():
    assert "Per-case reseeding." in SOURCE_TEXT


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------


def _fabricate_parquet(tmp_path, row_count: int) -> Path:
    from collector.helper import finalize_perf_files, log_perf

    perf_file = tmp_path / "moe_a2a_perf.txt"
    for index in range(row_count):
        log_perf(
            item_list=[
                a2a._build_moe_a2a_row(
                    comm_backend="deepep_ht",
                    phase="dispatch",
                    ep_size=16,
                    node_num=2,
                    shape=SHAPE,
                    num_tokens=64 * (index + 1),
                    sms=16,
                    transmit_us=10.0,
                    notify_us=1.0,
                )
            ],
            framework="SGLang",
            version="0.5.10",
            device_name="NVIDIA B200",
            op_name=a2a.OP_NAME,
            kernel_source=a2a.KERNEL_SOURCE,
            perf_filename=str(perf_file),
        )
    [parquet_path] = finalize_perf_files([perf_file])
    return parquet_path


RUNTIME_META = {
    "framework": "wideep_sglang",
    "version": "0.5.10",
    "image": "deepseek-v4-blackwell",
}


def test_sidecar_carries_the_design_5_table_fields(tmp_path):
    parquet_path = _fabricate_parquet(tmp_path, 3)
    meta_path = a2a.write_moe_a2a_sidecar(
        tmp_path,
        runtime_meta=RUNTIME_META,
        case_ids=["case-a", "case-b"],
        parquet_path=parquet_path,
        failure_count=0,
    )
    doc = yaml.safe_load(meta_path.read_text())

    assert meta_path.name == "collection_meta.yaml"
    assert doc["schema_version"] == 1
    assert doc["runtime"] == RUNTIME_META
    table = doc["tables"]["moe_a2a_perf"]
    # provenance.py:59 _TABLE_FIELD_ORDER
    assert list(table) == [
        "collector_ref",
        "collector_hash",
        "case_plan_hash",
        "collected_at",
        "rows",
        "status",
    ]
    from collector import provenance

    assert table["case_plan_hash"] == provenance.case_plan_hash(["case-a", "case-b"])
    assert table["collector_hash"].startswith("sha256:")
    assert table["rows"] == 3
    assert table["status"] == provenance.STATUS_COMPLETE


def test_sidecar_status_is_partial_when_cases_failed(tmp_path):
    from collector import provenance

    parquet_path = _fabricate_parquet(tmp_path, 1)
    meta_path = a2a.write_moe_a2a_sidecar(
        tmp_path,
        runtime_meta=RUNTIME_META,
        case_ids=["case-a"],
        parquet_path=parquet_path,
        failure_count=2,
    )
    table = yaml.safe_load(meta_path.read_text())["tables"]["moe_a2a_perf"]
    assert table["status"] == provenance.STATUS_PARTIAL


def test_sidecar_refuses_an_empty_case_plan(tmp_path):
    parquet_path = _fabricate_parquet(tmp_path, 1)
    with pytest.raises(a2a.MoeA2ADeclarationError, match="empty case plan"):
        a2a.write_moe_a2a_sidecar(
            tmp_path,
            runtime_meta=RUNTIME_META,
            case_ids=[],
            parquet_path=parquet_path,
            failure_count=0,
        )


def test_collector_hash_resolves_through_the_real_closures():
    from collector import provenance

    closures = provenance.load_closures(REPO_ROOT / "collector" / "hash_closures.yaml")
    entry = closures[a2a.MODULE_NAME]
    assert "collector/cases/base_ops/moe_a2a.yaml" in entry
    assert "__model_cases__" in entry
    assert provenance.collector_hash(a2a.MODULE_NAME, REPO_ROOT, closures).startswith("sha256:")


def test_module_is_registered_as_a_standalone_provenance_producer():
    from collector import provenance

    assert a2a.MODULE_NAME in provenance.STANDALONE_COLLECTOR_MODULES
    assert a2a.MODULE_NAME in provenance.enumerate_provenance_modules()
    # Standalone, NOT an OpEntry: collect.py's executor is single-host.
    assert a2a.MODULE_NAME not in provenance.enumerate_registry_modules()


def test_runtime_meta_rejects_a_version_that_is_not_the_manifest_pin():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="manifest wideep_sglang pin"):
        a2a.resolve_runtime_meta("0.5.14", None)


def test_runtime_meta_records_the_launched_image_variant():
    # F17: the sidecar attests the image the launcher actually passed to
    # srun (the GB200 launcher runs the grace_blackwell variant, not
    # `default`), including which manifest variant it is.
    from collector.framework_manifest import get_collector_runtime

    pinned = get_collector_runtime("sglang", workload="wideep")
    meta = a2a.resolve_runtime_meta(pinned.version, pinned.image("grace_blackwell"))
    assert meta["framework"] == "wideep_sglang"
    assert meta["version"] == pinned.version
    assert meta["image"] == pinned.image("grace_blackwell")
    assert meta["image_variant"] == "grace_blackwell"

    default_meta = a2a.resolve_runtime_meta(pinned.version, pinned.image())
    assert default_meta["image_variant"] == "default"


def test_runtime_meta_requires_the_launched_image():
    from collector.framework_manifest import get_collector_runtime

    pinned = get_collector_runtime("sglang", workload="wideep")
    with pytest.raises(a2a.MoeA2ADeclarationError, match="--image-ref"):
        a2a.resolve_runtime_meta(pinned.version, None)


def test_runtime_meta_rejects_an_image_the_manifest_does_not_pin():
    from collector.framework_manifest import get_collector_runtime

    pinned = get_collector_runtime("sglang", workload="wideep")
    with pytest.raises(a2a.MoeA2ADeclarationError, match="not a manifest wideep_sglang image variant"):
        a2a.resolve_runtime_meta(pinned.version, "someone/rebuilt-image:latest")


def test_launcher_passes_the_image_it_launches():
    # The launcher-to-sidecar identity: whatever ${CONTAINER_IMAGE} srun
    # receives is exactly what the collector is told to attest.
    launcher = REPO_ROOT / "collector" / "network" / "slurm" / "submit_moe_a2a.sh"
    text = launcher.read_text(encoding="utf-8")
    assert '--image-ref \\"${CONTAINER_IMAGE}\\"' in text


def test_stale_output_artifacts_fail_closed(tmp_path):
    # F19: log_perf appends, so a rerun into a directory holding a previous
    # attempt's staging CSV would finalize stale rows under this run's
    # attestation. The collector refuses such a directory outright.
    from collector.helper import stale_output_artifacts
    from collector.registry_types import PerfFile

    assert stale_output_artifacts(tmp_path, PerfFile.MOE_A2A.value) == []
    (tmp_path / PerfFile.MOE_A2A.value).write_text("header\nrow\n")
    (tmp_path / "errors_moe_a2a.rank0.json").write_text("[]")
    stale = stale_output_artifacts(tmp_path, PerfFile.MOE_A2A.value)
    assert PerfFile.MOE_A2A.value in stale
    assert "errors_moe_a2a.rank0.json" in stale
    # main() consults it before any benchmark or write.
    assert "stale_output_artifacts(output_dir" in SOURCE_TEXT
    assert "refuses to run into" in SOURCE_TEXT


def test_alternate_ll_transports_refuse_to_finalize():
    # F21: --allow-mnnvl / --disable-nvlink change the LL Buffer construction
    # but no persisted identity records them, so such runs are diagnostic:
    # staged rows only, no parquet, no sidecar.
    assert a2a.transport_is_default(allow_mnnvl=False, disable_nvlink=False)
    assert not a2a.transport_is_default(allow_mnnvl=True, disable_nvlink=False)
    assert not a2a.transport_is_default(allow_mnnvl=False, disable_nvlink=True)
    finalize_at = SOURCE_TEXT.index("finalize_perf_files([perf_path])")
    guard_at = SOURCE_TEXT.index("if diagnostic_transport:")
    assert guard_at < finalize_at, "the diagnostic-transport refusal must gate finalization"


# ---------------------------------------------------------------------------
# Dormant vLLM leg (D3) and source contract
# ---------------------------------------------------------------------------


def test_vllm_framework_is_declared_but_dormant():
    parser_args = a2a.parse_args(["--gpus-per-node", "8", "--framework", "vllm"])
    assert parser_args.framework == "vllm"
    with pytest.raises(NotImplementedError, match="wideep_vllm"):
        a2a.require_supported_framework("vllm")
    a2a.require_supported_framework("sglang")  # the live leg


def test_unknown_mode_raises():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="unsupported --modes"):
        a2a.resolve_modes("deepep_ht,deepep_xl")


def test_no_silent_case_skipping():
    # failure_handling.md: a queued case is executed or raises a classified
    # error which is then RECORDED, never silently dropped.
    assert "skipping" not in SOURCE_TEXT
    assert "MoeA2ABenchmarkError" in SOURCE_TEXT
    assert "record_failure" in SOURCE_TEXT


def test_identity_columns_are_live_not_fabricated_constants():
    # The retired pipeline hardcoded them (extract_data.py:14-21).
    assert 'DEVICE = "NVIDIA' not in SOURCE_TEXT
    assert "get_device_name" in SOURCE_TEXT
    assert "NODE_NUM_DEFAULT" not in SOURCE_TEXT


def test_rank_zero_owns_every_write():
    assert "if identity.rank == 0:" in SOURCE_TEXT
    assert SOURCE_TEXT.count("_emit_case_rows(") == 2  # the def plus the single rank-0 call site


def test_source_path_is_the_registered_standalone_module():
    assert SOURCE_PATH.exists()
    assert a2a.MODULE_NAME.replace(".", "/") + ".py" == "collector/wideep/sglang/collect_moe_a2a.py"


def test_failures_are_recorded_as_classified_data_per_rank(tmp_path):
    # failure_handling.md: a failed case is DATA. The record is rank-scoped
    # because the output dir is shared storage across nodes.
    identity = a2a.derive_dist_identity({"RANK": "3", "WORLD_SIZE": "16"}, gpus_per_node=8)
    case = a2a.MoeA2ACase("deepep_ht", SHAPE, num_tokens=8192, sms=20)
    a2a.record_failure(tmp_path, case, a2a.MoeA2ABenchmarkError("boom"), identity)

    path = tmp_path / "errors_moe_a2a.rank3.json"
    [record] = json.loads(path.read_text())
    assert record["classification"] == "unexpected"
    assert record["error_type"] == "MoeA2ABenchmarkError"
    assert record["module"] == a2a.MODULE_NAME
    assert record["rank"] == 3
    assert record["case"] == {
        "comm_backend": "deepep_ht",
        "ep_size": 16,
        "node_num": 2,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 8192,
        "sms": 20,
    }
