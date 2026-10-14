# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for TensorRT-LLM's standalone serving-parity DeepEP collector."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest

from collector.wideep.trtllm import collect_moe_a2a as a2a

pytestmark = pytest.mark.unit

SHAPE = a2a.MoeA2AShape(hidden_size=7168, topk=8, num_experts=256)
GRID = {"ht_token_counts": [16, 32], "ll_token_counts": [1, 2]}


def _case(
    backend: str = a2a.COMM_BACKEND_HT,
    dtype: str = "bfloat16",
    *,
    tokens: int = 16,
) -> a2a.MoeA2ACase:
    quant = next(spec for spec in a2a.QUANT_SPECS[backend] if spec.comm_dtype == dtype)
    return a2a.MoeA2ACase(
        comm_backend=backend,
        inference_phase=a2a.INFERENCE_PHASES[backend],
        quant=quant,
        shape=SHAPE,
        num_tokens=tokens,
        ep_size=8,
        node_num=1,
    )


def _result(dtype: str = "bfloat16") -> a2a.BenchmarkResult:
    return a2a.BenchmarkResult(
        (
            a2a.PhaseMeasurement("combine", 31.0, dtype),
            a2a.PhaseMeasurement("dispatch", 17.0, dtype),
        )
    )


def test_case_plan_uses_distinct_trtllm_backend_and_dtype_keys():
    cases = a2a.build_case_plan(shapes=[SHAPE], token_grid=GRID, ep_size=8, node_num=1)
    assert {case.comm_backend for case in cases} == {
        "trtllm_deepep_ht",
        "trtllm_deepep_ll",
    }
    assert {case.quant.comm_dtype for case in cases if case.comm_backend == a2a.COMM_BACKEND_HT} == {
        "bfloat16",
        "nvfp4",
    }
    assert {case.quant.comm_dtype for case in cases if case.comm_backend == a2a.COMM_BACKEND_LL} == {
        "bfloat16",
        "fp8",
        "nvfp4",
    }
    assert all(case.sms == 0 for case in cases)
    assert len({case.invocation_key() for case in cases}) == len(cases)
    assert len({case.physical_key(phase) for case in cases for phase in ("combine", "dispatch")}) == len(cases) * 2


def test_canary_selects_every_backend_and_truthful_dtype():
    unsupported_ll_shape = a2a.MoeA2AShape(hidden_size=3072, topk=8, num_experts=256)
    cases = a2a.build_case_plan(
        shapes=[unsupported_ll_shape, SHAPE],
        token_grid=GRID,
        ep_size=8,
        node_num=1,
    )
    selected = a2a.select_canary_cases(cases)
    assert {(case.comm_backend, case.quant.comm_dtype) for case in selected} == {
        ("trtllm_deepep_ht", "bfloat16"),
        ("trtllm_deepep_ht", "nvfp4"),
        ("trtllm_deepep_ll", "bfloat16"),
        ("trtllm_deepep_ll", "fp8"),
        ("trtllm_deepep_ll", "nvfp4"),
    }
    assert len(selected) == 5
    assert {case.shape.hidden_size for case in selected if case.comm_backend == a2a.COMM_BACKEND_HT} == {3072}
    assert {case.shape.hidden_size for case in selected if case.comm_backend == a2a.COMM_BACKEND_LL} == {7168}


@pytest.mark.parametrize("system", ["h100_sxm", "h200_sxm"])
def test_hopper_low_latency_excludes_unsupported_nvfp4_and_w4(system: str):
    cases = a2a.build_case_plan(
        shapes=[SHAPE],
        token_grid=GRID,
        ep_size=8,
        node_num=1,
        modes=(a2a.COMM_BACKEND_LL,),
        system=system,
    )
    assert {case.quant.comm_dtype for case in cases} == {"bfloat16", "fp8"}


@pytest.mark.parametrize("system", ["gb200", "gb300", "b200_sxm", "b300_sxm"])
def test_blackwell_low_latency_keeps_nvfp4_but_excludes_unsupported_w4(system: str):
    cases = a2a.build_case_plan(
        shapes=[SHAPE],
        token_grid=GRID,
        ep_size=8,
        node_num=1,
        modes=(a2a.COMM_BACKEND_LL,),
        system=system,
    )
    assert {case.quant.comm_dtype for case in cases} == {"bfloat16", "fp8", "nvfp4"}


@pytest.mark.parametrize(
    ("system", "ep_size", "hidden_3072_cases", "hidden_6144_fp8_cases"),
    [
        ("gb200", 4, 42, 14),
        ("gb300", 4, 42, 14),
        ("b200_sxm", 8, 42, 14),
        ("b300_sxm", 8, 42, 14),
        ("h100_sxm", 8, 28, 14),
        ("h200_sxm", 8, 28, 14),
    ],
)
def test_pinned_ll_hidden_limits_remain_observable_failures(
    system: str,
    ep_size: int,
    hidden_3072_cases: int,
    hidden_6144_fp8_cases: int,
):
    cases = a2a.build_case_plan(
        shapes=a2a.get_moe_a2a_shapes(required_expert_parallel_size=ep_size),
        token_grid=a2a.get_moe_a2a_token_grid(),
        ep_size=ep_size,
        node_num=1,
        modes=(a2a.COMM_BACKEND_LL,),
        system=system,
    )

    assert sum(case.shape.hidden_size == 3072 for case in cases) == hidden_3072_cases
    assert sum(case.shape.hidden_size == 6144 and case.quant.comm_dtype == "fp8" for case in cases) == (
        hidden_6144_fp8_cases
    )


@pytest.mark.parametrize(
    ("system", "backend", "quantization", "phase", "expected"),
    [
        ("h100_sxm", a2a.COMM_BACKEND_HT, "fp8_block", "dispatch", "bfloat16"),
        ("h200_sxm", a2a.COMM_BACKEND_HT, "fp8", "combine", "bfloat16"),
        ("h100_sxm", a2a.COMM_BACKEND_LL, "fp8_block", "dispatch", "fp8"),
        ("gb300", a2a.COMM_BACKEND_LL, "nvfp4", "dispatch", "nvfp4"),
        ("gb300", a2a.COMM_BACKEND_LL, "nvfp4", "combine", "fp4"),
    ],
)
def test_phase_quant_system_communication_dtype_mapping(system, backend, quantization, phase, expected):
    assert (
        a2a.communication_dtype_for(
            system=system,
            backend=backend,
            model_quantization=quantization,
            communication_phase=phase,
        )
        == expected
    )


def test_hopper_rejects_unimplemented_ll_nvfp4_mapping():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="no pinned"):
        a2a.communication_dtype_for(
            system="h200_sxm",
            backend=a2a.COMM_BACKEND_LL,
            model_quantization="nvfp4",
            communication_phase="combine",
        )


def test_routing_is_part_of_invocation_identity_and_collision_is_rejected():
    grouped = replace(
        SHAPE,
        routing=a2a.RoutingSpec("DeepSeekV3", 8, 4, 2.5),
    )
    global_routing = replace(
        SHAPE,
        routing=a2a.RoutingSpec("DeepSeekV3", 1, 1, 2.5),
    )
    with pytest.raises(a2a.MoeA2ADeclarationError, match="physical key"):
        a2a.build_case_plan(
            shapes=[grouped, global_routing],
            token_grid={"ht_token_counts": [16], "ll_token_counts": [1]},
            ep_size=8,
            node_num=1,
            modes=(a2a.COMM_BACKEND_HT,),
        )


def test_declared_shapes_retain_grouped_and_global_routing():
    shapes = a2a.get_moe_a2a_shapes(required_expert_parallel_size=8)
    by_geometry = {(shape.hidden_size, shape.topk, shape.num_experts): shape.routing for shape in shapes}
    assert by_geometry[(7168, 8, 256)] == a2a.RoutingSpec("DeepSeekV3", 8, 4, 2.5)
    assert by_geometry[(3072, 8, 256)] == a2a.RoutingSpec()
    assert (3584, 16, 896) not in by_geometry


def test_kimi_k3_is_excluded_by_the_shared_operation_declaration():
    from collector.case_generator import get_common_moe_test_cases, is_wideep_moe_model

    recipes = get_common_moe_test_cases(backend="trtllm", required_expert_parallel_size=8)
    kimi = next(recipe for recipe in recipes if recipe.model_name == "moonshotai/Kimi-K3")

    assert not is_wideep_moe_model(kimi.model_name)


def test_declared_duplicate_shapes_deduplicate_on_physical_identity(capsys):
    cases = a2a.build_case_plan(
        shapes=[SHAPE, SHAPE],
        token_grid={"ht_token_counts": [16], "ll_token_counts": [1]},
        ep_size=8,
        node_num=1,
        modes=(a2a.COMM_BACKEND_HT,),
    )
    assert len(cases) == 2
    assert "deduplicated: 2" in capsys.readouterr().out


def test_no_cross_framework_backend_label_is_accepted():
    with pytest.raises(ValueError, match="belongs to vllm, not trtllm"):
        a2a.build_case_plan(
            shapes=[SHAPE],
            token_grid=GRID,
            ep_size=8,
            node_num=1,
            modes=("deepep_ht",),
        )
    rows = a2a.build_unified_rows(_case(), _result())
    assert {row["comm_backend"] for row in rows} == {"trtllm_deepep_ht"}
    assert all(not row["comm_backend"].startswith("deepep_") for row in rows)


def test_modes_map_to_forced_deepep_selectors():
    assert a2a.FORCED_METHODS == {
        "trtllm_deepep_ht": "DEEPEP",
        "trtllm_deepep_ll": "DEEPEPLOWLATENCY",
    }
    assert a2a.resolve_modes("trtllm_deepep_ll,trtllm_deepep_ht") == (
        "trtllm_deepep_ll",
        "trtllm_deepep_ht",
    )
    with pytest.raises(a2a.MoeA2ADeclarationError, match="unsupported --modes"):
        a2a.resolve_modes("NVLINK_ONE_SIDED")


def test_factory_call_contract_forces_method_without_auto_selector():
    calls = []

    class Factory:
        @staticmethod
        def _create_forced_method(*args, **kwargs):
            calls.append((args, kwargs))
            return object()

        @staticmethod
        def create_strategy(*args, **kwargs):
            raise AssertionError("auto-selection must never be called")

    case = _case(a2a.COMM_BACKEND_LL)
    model_config = object()
    result = a2a.create_forced_communication(Factory, case=case, model_config=model_config, experts_per_rank=32)
    assert result is not None
    [(args, kwargs)] = calls
    assert args == (
        "DEEPEPLOWLATENCY",
        model_config,
        256,
        256,
        8,
        32,
    )
    assert kwargs == {
        "payload_in_workspace": False,
        "alltoall_result_do_sum": True,
        "use_flashinfer": False,
    }


def test_factory_none_is_an_error_not_a_fallback():
    class Factory:
        @staticmethod
        def _create_forced_method(*args, **kwargs):
            return None

    with pytest.raises(a2a.MoeA2ABenchmarkError, match="refusing to fall back"):
        a2a.create_forced_communication(
            Factory,
            case=_case(),
            model_config=object(),
            experts_per_rank=32,
        )


def test_rc11_adapter_reuses_resources_only_across_the_token_axis():
    adapter = a2a.TensorRTLLMBenchmarkAdapter(max_num_tokens_per_rank=256)
    base = _case(tokens=16)
    same_resources = _case(tokens=32)
    different_dtype = _case(dtype="nvfp4", tokens=16)
    different_shape = a2a.MoeA2ACase(
        comm_backend=base.comm_backend,
        inference_phase=base.inference_phase,
        quant=base.quant,
        shape=a2a.MoeA2AShape(hidden_size=4096, topk=base.shape.topk, num_experts=base.shape.num_experts),
        num_tokens=base.num_tokens,
        ep_size=base.ep_size,
        node_num=base.node_num,
    )

    assert adapter._resource_key(base) == adapter._resource_key(same_resources)
    assert adapter._resource_key(base) != adapter._resource_key(different_dtype)
    assert adapter._resource_key(base) != adapter._resource_key(different_shape)


def test_rc11_adapter_failure_reset_destroys_active_resource_once():
    class Backend:
        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    backend = Backend()
    adapter = a2a.TensorRTLLMBenchmarkAdapter()
    adapter._active_resource_key = ("old",)
    adapter._active_backend = backend
    adapter._active_moe = object()

    adapter.reset_after_failure()

    assert backend.destroy_calls == 1
    assert adapter._active_resource_key is None
    assert adapter._active_backend is None
    assert adapter._active_moe is None


def test_adapter_has_no_internal_mpi_collective_or_barrier():
    source = inspect.getsource(a2a.TensorRTLLMBenchmarkAdapter.run)
    assert "mpi_allgather" not in source
    assert "mpi_barrier" not in source
    assert "all_rank_num_tokens" in source


def test_adapter_uses_pinned_router_instead_of_with_replacement_ids():
    source = inspect.getsource(a2a.TensorRTLLMBenchmarkAdapter.run)
    assert "torch.randint" not in source
    assert "routing_method.apply(router_logits)" in source
    assert "_validate_unique_experts" in source


def test_rows_have_no_prepare_phase_and_preserve_truthful_dtype():
    rows = a2a.build_unified_rows(_case(a2a.COMM_BACKEND_LL, "fp8", tokens=2), _result("fp8"))
    assert [(row["phase"], row["comm_dtype"]) for row in rows] == [
        ("combine", "fp8"),
        ("dispatch", "fp8"),
    ]
    assert all(row["sms"] == 0 and row["notify_us"] == 0 for row in rows)
    assert [row["latency"] for row in rows] == [31.0, 17.0]


def test_nvfp4_low_latency_uses_phase_specific_consumer_dtypes():
    rows = a2a.build_unified_rows(
        _case(a2a.COMM_BACKEND_LL, "nvfp4", tokens=2),
        a2a.BenchmarkResult(
            (
                a2a.PhaseMeasurement("combine", 31.0, "fp4"),
                a2a.PhaseMeasurement("dispatch", 17.0, "nvfp4"),
            )
        ),
    )
    assert [(row["phase"], row["comm_dtype"]) for row in rows] == [
        ("combine", "fp4"),
        ("dispatch", "nvfp4"),
    ]


def test_prepare_or_unsupported_dtype_never_gets_relabelled():
    with pytest.raises(a2a.MoeA2ABenchmarkError, match=r"exactly combine\+dispatch"):
        a2a.build_unified_rows(
            _case(),
            a2a.BenchmarkResult(
                (
                    a2a.PhaseMeasurement("prepare", 2.0, "bfloat16"),
                    a2a.PhaseMeasurement("dispatch", 3.0, "bfloat16"),
                )
            ),
        )
    with pytest.raises(a2a.MoeA2ABenchmarkError, match="not a truthful dtype"):
        a2a.build_unified_rows(_case(), _result("fp8"))


def test_zero_case_paths_fail_closed():
    with pytest.raises(a2a.MoeA2ADeclarationError, match="zero shapes"):
        a2a.build_case_plan(shapes=[], token_grid=GRID, ep_size=8, node_num=1)
    with pytest.raises(a2a.MoeA2ADeclarationError, match="not divisible by ep_size=384"):
        a2a.build_case_plan(shapes=[SHAPE], token_grid=GRID, ep_size=384, node_num=1)
    with pytest.raises(a2a.MoeA2ADeclarationError, match="zero-case"):
        a2a.run_collection(
            cases=[],
            adapter=object(),
            output_dir=Path("."),
            rank=0,
            version="1.3.0rc11",
            device_name="none",
            runtime_meta={},
        )


def test_runtime_source_and_abi_attestation_hooks_fail_closed():
    runtime = a2a.get_collector_runtime("trtllm_a2a")
    image_digest = runtime.image().split("@", 1)[1]
    meta = a2a.resolve_runtime_meta(
        runtime.version,
        source_commit=a2a.TARGET_SOURCE_COMMIT,
        observed_abi=runtime.abi,
        observed_image_digest=image_digest,
    )
    assert meta["source_commit"] == a2a.TARGET_SOURCE_COMMIT
    assert meta["abi"] == runtime.abi
    with pytest.raises(a2a.MoeA2ADeclarationError, match="source commit"):
        a2a.resolve_runtime_meta(
            runtime.version,
            source_commit="0" * 40,
            observed_abi=runtime.abi,
            observed_image_digest=image_digest,
        )
    with pytest.raises(a2a.MoeA2ADeclarationError, match="ABI mismatch"):
        a2a.resolve_runtime_meta(
            runtime.version,
            source_commit=a2a.TARGET_SOURCE_COMMIT,
            observed_abi={},
            observed_image_digest=image_digest,
        )
    with pytest.raises(a2a.MoeA2ADeclarationError, match="image digest mismatch"):
        a2a.resolve_runtime_meta(
            runtime.version,
            source_commit=a2a.TARGET_SOURCE_COMMIT,
            observed_abi=runtime.abi,
            observed_image_digest="sha256:" + "0" * 64,
        )


class _GoodAdapter:
    def run(self, case, all_rank_num_tokens):
        assert all_rank_num_tokens == [case.num_tokens] * case.ep_size
        return _result(case.quant.comm_dtype)


class _PartialAdapter:
    def __init__(self):
        self.calls = 0

    def run(self, case, all_rank_num_tokens):
        assert all_rank_num_tokens == [case.num_tokens] * case.ep_size
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("rank-local benchmark failure")
        return _result(case.quant.comm_dtype)


class _ClosingAdapter(_GoodAdapter):
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_adapter_closes_before_final_ready_and_barrier(tmp_path, monkeypatch):
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: True)
    monkeypatch.setattr(a2a, "write_moe_a2a_sidecar", lambda output_dir, **kwargs: output_dir / "meta")
    adapter = _ClosingAdapter()
    stages = []

    a2a.run_collection(
        cases=[_case()],
        adapter=adapter,
        output_dir=tmp_path,
        rank=0,
        version="1.3.0rc11",
        device_name="NVIDIA B200",
        runtime_meta={},
        finalize=lambda paths, **kwargs: [tmp_path / "moe_a2a_perf.parquet"],
        stage_agreement=lambda stage, failed: stages.append((stage, adapter.close_calls)) or failed,
        final_barrier=lambda: stages.append(("barrier", adapter.close_calls)),
    )

    assert adapter.close_calls == 1
    assert stages[-3:] == [("adapter_close", 1), ("final_ready", 1), ("barrier", 1)]


def test_adapter_closes_on_preflight_failure(tmp_path):
    (tmp_path / "moe_a2a_perf.parquet").write_text("stale")
    adapter = _ClosingAdapter()

    with pytest.raises(a2a.MoeA2AWriteError, match="stale output artifacts"):
        a2a.run_collection(
            cases=[_case()],
            adapter=adapter,
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
        )

    assert adapter.close_calls == 1


def test_write_failure_is_recorded_and_cannot_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: False)
    with pytest.raises(a2a.MoeA2AWriteError, match="log_perf rejected"):
        a2a.run_collection(
            cases=[_case()],
            adapter=_GoodAdapter(),
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
        )
    [record] = json.loads((tmp_path / "errors_moe_a2a_trtllm.rank0.json").read_text())
    assert record["error_type"] == "MoeA2AWriteError"
    assert record["classification"] == "unexpected"


def test_stale_staging_is_rejected_before_append(tmp_path):
    path = tmp_path / "moe_a2a_perf.txt"
    path.write_text("stale\n")
    with pytest.raises(a2a.MoeA2AWriteError, match="stale output artifacts"):
        a2a.run_collection(
            cases=[_case()],
            adapter=_GoodAdapter(),
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
        )
    assert path.read_text() == "stale\n"


def test_stale_failure_staging_is_rejected_before_benchmark(tmp_path):
    path = tmp_path / "errors_moe_a2a_trtllm.rank0.json"
    path.write_text('[{"error": "old run"}]\n')
    adapter = _PartialAdapter()

    with pytest.raises(a2a.MoeA2AWriteError, match="stale output artifacts"):
        a2a.run_collection(
            cases=[_case()],
            adapter=adapter,
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
        )

    assert adapter.calls == 0
    assert path.read_text() == '[{"error": "old run"}]\n'


def test_partial_collection_writes_partial_attestation_then_raises(tmp_path, monkeypatch):
    cases = [_case(tokens=16), _case(tokens=32)]
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: True)
    observed = {}

    def fake_sidecar(output_dir, **kwargs):
        observed.update(kwargs)
        return output_dir / "collection_meta.yaml"

    monkeypatch.setattr(a2a, "write_moe_a2a_sidecar", fake_sidecar)
    with pytest.raises(a2a.MoeA2ABenchmarkError, match="partial"):
        a2a.run_collection(
            cases=cases,
            adapter=_PartialAdapter(),
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={"framework": "wideep_trtllm"},
            finalize=lambda paths, **kwargs: [tmp_path / "moe_a2a_perf.parquet"],
        )
    assert observed["failure_count"] == 1
    assert observed["module_name"] == a2a.MODULE_NAME
    assert len(observed["case_ids"]) == 2


def test_peer_failure_prevents_rank_zero_from_persisting_rows(tmp_path, monkeypatch):
    writes = []
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: writes.append(kwargs) or True)

    with pytest.raises(a2a.MoeA2ABenchmarkError, match="zero rows"):
        a2a.run_collection(
            cases=[_case()],
            adapter=_GoodAdapter(),
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
            failure_agreement=lambda local_failed: True,
        )

    assert writes == []
    [record] = json.loads((tmp_path / "errors_moe_a2a_trtllm.rank0.json").read_text())
    assert record["error_type"] == "MoeA2APeerError"


def test_classified_error_record_carries_physical_identity(tmp_path):
    case = _case(a2a.COMM_BACKEND_LL, "nvfp4", tokens=2)
    a2a.record_failure(tmp_path, case, a2a.MoeA2ABenchmarkError("boom"), rank=3)
    [record] = json.loads((tmp_path / "errors_moe_a2a_trtllm.rank3.json").read_text())
    assert record["classification"] == "unexpected"
    assert record["case"] == {
        "comm_backend": "trtllm_deepep_ll",
        "comm_dtype": "nvfp4",
        "inference_phase": "generation",
        "ep_size": 8,
        "node_num": 1,
        "hidden_size": 7168,
        "topk": 8,
        "num_experts": 256,
        "num_tokens": 2,
        "sms": 0,
    }


@pytest.mark.parametrize(
    "name",
    [
        "moe_a2a_perf.parquet",
        "moe_a2a_perf.parquet.sha256",
        "collection_meta.yaml",
        "errors_moe_a2a_trtllm.rank7.json",
    ],
)
def test_every_owned_stale_artifact_is_rejected_before_benchmark(tmp_path, name):
    (tmp_path / name).write_text("stale\n")
    adapter = _PartialAdapter()
    with pytest.raises(a2a.MoeA2AWriteError, match="stale output artifacts"):
        a2a.run_collection(
            cases=[_case()],
            adapter=adapter,
            output_dir=tmp_path,
            rank=0,
            version="1.3.0rc11",
            device_name="NVIDIA B200",
            runtime_meta={},
        )
    assert adapter.calls == 0


def test_finalize_disables_existing_parquet_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: True)
    monkeypatch.setattr(a2a, "write_moe_a2a_sidecar", lambda output_dir, **kwargs: output_dir / "collection_meta.yaml")
    observed = []

    def finalize(paths, **kwargs):
        observed.append((paths, kwargs))
        return [tmp_path / "moe_a2a_perf.parquet"]

    a2a.run_collection(
        cases=[_case()],
        adapter=_GoodAdapter(),
        output_dir=tmp_path,
        rank=0,
        version="1.3.0rc11",
        device_name="NVIDIA B200",
        runtime_meta={},
        finalize=finalize,
    )
    assert observed == [([str(tmp_path / "moe_a2a_perf.txt")], {"merge_existing": False})]
