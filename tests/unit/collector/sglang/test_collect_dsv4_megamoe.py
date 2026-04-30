# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mod():
    return importlib.import_module("collector.sglang.collect_dsv4_megamoe")


@pytest.fixture()
def compute_mod():
    return importlib.import_module("collector.sglang.collect_dsv4_megamoe_compute")


def _import_real_torch_or_skip():
    saved_mock = None
    if isinstance(sys.modules.get("torch"), MagicMock):
        saved_mock = sys.modules.pop("torch")
    try:
        return pytest.importorskip("torch")
    except pytest.skip.Exception:
        if saved_mock is not None:
            sys.modules["torch"] = saved_mock
        raise


@pytest.mark.unit
def test_get_dsv4_megamoe_probe_test_cases_shape(mod):
    cases = mod.get_dsv4_megamoe_test_cases()

    assert cases
    assert all(len(case) == 12 for case in cases)
    assert {case[0] for case in cases} == {"mxfp4"}
    assert {case[8] for case in cases} == {"deepseek-ai/DeepSeek-V4"}
    assert {case[9] for case in cases} == {"dsv4_megamoe_probe.json"}
    assert {case[10] for case in cases} == {"model_router"}
    assert {case[11] for case in cases} == {None}
    assert all(case[1] <= mod.DEFAULT_MEGAMOE_TOKEN_CAP for case in cases)
    assert all(case[2:8] == [4096, 2048, 6, 256, 1, 1] for case in cases)


@pytest.mark.unit
def test_get_dsv4_megamoe_compute_test_cases_shape(compute_mod):
    cases = compute_mod.get_dsv4_megamoe_compute_test_cases()

    assert cases
    assert all(len(case) == 12 for case in cases)
    assert {case[0] for case in cases} == {"mxfp4"}
    assert {case[8] for case in cases} == {"deepseek-ai/DeepSeek-V4"}
    assert {case[9] for case in cases} == {"dsv4_megamoe_compute_perf.txt"}
    assert {case[10] for case in cases} == {"uniform", "power_law"}
    assert {case[11] for case in cases} == {None, compute_mod.DEFAULT_POWER_LAW_ALPHA}
    assert all(case[1] <= compute_mod.DEFAULT_MEGAMOE_TOKEN_CAP for case in cases)
    assert all(case[2:7] == [4096, 2048, 6, 256, 1] for case in cases)
    assert {case[7] for case in cases} == set(compute_mod.DEFAULT_TARGET_EP_LIST)


@pytest.mark.unit
def test_env_defaults_enable_megamoe(mod):
    defaults = mod.dsv4_megamoe_env_defaults()

    assert defaults["SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE"] == "1"
    assert defaults["SGLANG_OPT_FIX_HASH_MEGA_MOE"] == "1"
    assert defaults["SGLANG_DSV4_MODE"] == "2604"
    assert defaults["SGLANG_DSV4_2604_SUBMODE"] == "2604B"
    assert defaults["SGLANG_DSV4_FP4_EXPERTS"] == "1"


@pytest.mark.unit
def test_default_model_config_is_layer_indexable(mod):
    task = mod.Dsv4MegaMoETask(*mod.get_dsv4_megamoe_test_cases()[0])
    config = mod._default_model_config(task)

    assert config["architectures"] == ["DeepseekV4ForCausalLM"]
    assert config["model_type"] == "deepseek_ref"
    assert config["num_hash_layers"] == mod.DEFAULT_DSV4_NUM_HASH_LAYERS
    assert config["n_hash_layers"] == mod.DEFAULT_DSV4_NUM_HASH_LAYERS
    assert len(config["compress_ratios"]) == config["num_hidden_layers"]
    assert all(ratio == 0 for ratio in config["compress_ratios"])


@pytest.mark.unit
def test_validate_megamoe_probe_requires_observed_deepgemm_call(mod):
    report = {
        "schema": mod.REPORT_SCHEMA_VERSION,
        "status": "ok",
        "observed": {
            "should_use_mega_moe": True,
            "forward_mega_moe_calls": 1,
            "run_mega_routed_calls": 1,
            "deep_gemm_fp8_fp4_mega_moe_calls": 0,
            "mega_l1_weights_present": True,
            "mega_l2_weights_present": True,
            "mega_moe_weights_built": True,
        },
    }

    assert "deep_gemm_mega_moe_not_called" in mod.validate_megamoe_probe(report)

    report["observed"]["deep_gemm_fp8_fp4_mega_moe_calls"] = 1
    assert mod.validate_megamoe_probe(report) == []


@pytest.mark.unit
def test_verify_megamoe_probe_file(mod, tmp_path):
    report = {
        "schema": mod.REPORT_SCHEMA_VERSION,
        "status": "ok",
        "observed": {
            "should_use_mega_moe": True,
            "forward_mega_moe_calls": 1,
            "run_mega_routed_calls": 1,
            "deep_gemm_fp8_fp4_mega_moe_calls": 1,
            "mega_l1_weights_present": True,
            "mega_l2_weights_present": True,
            "mega_moe_weights_built": True,
        },
    }
    report_path = tmp_path / "probe.json"
    report_path.write_text(json.dumps(report))

    assert mod.verify_megamoe_probe_file(report_path) == report


@pytest.mark.unit
def test_dry_run_writes_report_without_subprocess(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_DSV4_MEGAMOE_DRY_RUN", "1")
    monkeypatch.setenv("AIC_DSV4_MEGAMOE_REPORT_DIR", str(tmp_path))

    case = mod.get_dsv4_megamoe_test_cases()[0]
    mod.run_dsv4_megamoe(*case, device="cuda:7")

    reports = sorted(Path(tmp_path).glob("dsv4_megamoe_dry_run_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["schema"] == mod.REPORT_SCHEMA_VERSION
    assert payload["status"] == "dry_run"
    assert payload["device_id"] == 7
    assert payload["task"]["moe_type"] == "mxfp4"
    assert payload["task"]["model_name"] == "deepseek-ai/DeepSeek-V4"
    assert payload["env"]["SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE"] == "1"


@pytest.mark.unit
def test_compute_coerce_rejects_multi_rank(compute_mod):
    case = compute_mod.get_dsv4_megamoe_compute_test_cases()[0]
    case[6] = 2

    with pytest.raises(ValueError, match="local-compute collector requires moe_tp_size == 1"):
        compute_mod._coerce_compute_task(*case)


@pytest.mark.unit
def test_compute_coerce_accepts_target_ep_size(compute_mod):
    case = compute_mod.get_dsv4_megamoe_compute_test_cases()[0]
    case[7] = 8

    task = compute_mod._coerce_compute_task(*case)

    assert task.moe_tp_size == 1
    assert task.moe_ep_size == 8


@pytest.mark.unit
def test_compute_coerce_rejects_model_router_distribution(compute_mod):
    case = compute_mod.get_dsv4_megamoe_compute_test_cases()[0]
    case[10] = "model_router"

    with pytest.raises(ValueError, match="Unsupported DSv4 MegaMoE compute distribution"):
        compute_mod._coerce_compute_task(*case)


@pytest.mark.unit
def test_compute_dry_run_writes_report_without_subprocess(compute_mod, monkeypatch, tmp_path):
    monkeypatch.setenv("AIC_DSV4_MEGAMOE_DRY_RUN", "1")
    monkeypatch.setenv("AIC_DSV4_MEGAMOE_REPORT_DIR", str(tmp_path))

    case = compute_mod.get_dsv4_megamoe_compute_test_cases()[0]
    compute_mod.run_dsv4_megamoe_compute(*case, device="cuda:3")

    reports = sorted(Path(tmp_path).glob("dsv4_megamoe_compute_dry_run_*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload["schema"] == compute_mod.REPORT_SCHEMA_VERSION
    assert payload["status"] == "dry_run"
    assert payload["device_id"] == 3
    assert payload["task"]["moe_type"] == "mxfp4"
    assert payload["task"]["distribution"] == "uniform"
    assert payload["env"]["SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE"] == "1"


@pytest.mark.unit
def test_compute_local_expert_indices_include_local_routed_and_shared(compute_mod):
    assert compute_mod._local_expert_indices(
        routed_num_experts=256,
        target_moe_ep_size=8,
        num_fused_shared_experts=0,
    ) == tuple(range(32))

    assert compute_mod._local_expert_indices(
        routed_num_experts=8,
        target_moe_ep_size=4,
        num_fused_shared_experts=1,
    ) == (0, 1, 8)


@pytest.mark.unit
def test_compute_build_target_ep_workload_uses_bottleneck_rank(compute_mod):
    case = compute_mod.get_dsv4_megamoe_compute_test_cases()[0]
    case[1] = 16
    case[7] = 4
    case[10] = "power_law"
    case[11] = compute_mod.DEFAULT_POWER_LAW_ALPHA
    task = compute_mod._coerce_compute_task(*case)

    workload = compute_mod._build_target_ep_workload(
        task,
        routed_topk=6,
        routed_num_experts=256,
        num_fused_shared_experts=0,
    )

    assert workload.moe_ep_size == 4
    assert workload.experts_per_rank == 64
    assert workload.routed_rank_loads[0] == max(workload.routed_rank_loads)
    assert len(workload.rank0_masked_m) == 64
    assert sum(workload.rank0_masked_m) > 0


@pytest.mark.unit
def test_compute_local_topk_tensors_mask_remote_experts(compute_mod):
    torch = _import_real_torch_or_skip()

    task = compute_mod.Dsv4MegaMoEComputeTask(
        "mxfp4",
        8,
        4096,
        2048,
        6,
        256,
        1,
        4,
        "deepseek-ai/DeepSeek-V4",
        compute_mod.PERF_FILENAME,
        "uniform",
        None,
    )

    workload = compute_mod._build_target_ep_workload(
        task,
        routed_topk=6,
        routed_num_experts=256,
        num_fused_shared_experts=1,
    )
    topk_ids, topk_weights, masked_m = compute_mod._make_local_topk_tensors(workload, device="cpu")

    assert topk_ids.shape[1] == 7
    assert topk_weights.shape == topk_ids.shape
    assert topk_ids.dtype == torch.int32
    assert masked_m.shape == (65,)
    assert int(masked_m.sum().item()) == int((topk_ids >= 0).sum().item())
    assert topk_ids[topk_ids >= 0].max() < 65
    assert torch.all(topk_weights[topk_ids == -1] == 0)


@pytest.mark.unit
def test_compute_source_predispatch_topk_uses_source_local_rows(compute_mod):
    torch = _import_real_torch_or_skip()

    task = compute_mod.Dsv4MegaMoEComputeTask(
        "mxfp4",
        17,
        4096,
        2048,
        2,
        8,
        1,
        4,
        "deepseek-ai/DeepSeek-V4",
        compute_mod.PERF_FILENAME,
        "uniform",
        None,
    )
    workload = compute_mod._build_target_ep_workload(
        task,
        routed_topk=2,
        routed_num_experts=8,
        num_fused_shared_experts=0,
    )

    source_rank, topk_ids, topk_weights = compute_mod._make_source_predispatch_topk_tensors(
        workload,
        device="cpu",
    )

    assert source_rank == 0
    assert topk_ids.shape == (workload.num_tokens_per_rank, workload.mega_topk)
    assert topk_weights.shape == topk_ids.shape
    assert topk_ids.dtype == torch.int32
    assert topk_ids.max().item() >= workload.experts_per_rank
    assert len(workload.rank0_local_token_indices) > topk_ids.shape[0]
    assert torch.all(topk_weights == 1.0 / float(workload.mega_topk))


@pytest.mark.unit
def test_compute_detects_deep_gemm_recv_stats_signature(compute_mod):
    def with_stats(_y, _l1, _l2, _buf, cumulative_local_expert_recv_stats=None):
        return cumulative_local_expert_recv_stats

    def without_stats(_y, _l1, _l2, _buf):
        return None

    assert compute_mod._deep_gemm_supports_recv_stats(SimpleNamespace(fp8_fp4_mega_moe=with_stats))
    assert not compute_mod._deep_gemm_supports_recv_stats(SimpleNamespace(fp8_fp4_mega_moe=without_stats))


@pytest.mark.unit
def test_probe_module_does_not_write_perf_rows(mod):
    source = Path(mod.__file__).read_text()

    assert "log_perf(" not in source
    assert "from helper import benchmark_with_power, log_perf" not in source


@pytest.mark.unit
def test_dsv4_megamoe_diagnostic_is_not_default_registered():
    from collector.sglang.registry import REGISTRY

    ops = {entry.op for entry in REGISTRY}
    assert "dsv4_megamoe" not in ops
    assert "dsv4_megamoe_compute" not in ops
