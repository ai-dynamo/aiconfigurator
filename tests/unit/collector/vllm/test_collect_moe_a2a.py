# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the vLLM standalone DeepEP collector."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

from collector.wideep.sglang.collect_moe_a2a import MoeA2AShape, PhaseTiming
from collector.wideep.vllm import collect_moe_a2a as a2a

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "wideep" / "vllm" / "collect_moe_a2a.py"
SOURCE = SOURCE_PATH.read_text()
SHAPE = MoeA2AShape(7168, 8, 256)
WORLD_SIZE = 8
NODE_NUM = 2
GRID = {
    "ht_token_counts": [16, 512],
    "ll_token_counts": [1, 16, 256],
    "sms": [16, 20, 24],
}


class FakeAdapter:
    def __init__(self, *, fail_tokens=(), bad_sms=None, bad_capacity=False, close_error: BaseException | None = None):
        self.fail_tokens = set(fail_tokens)
        self.bad_sms = bad_sms
        self.bad_capacity = bad_capacity
        self.close_error = close_error
        self.closed = False
        self.seen = []

    def benchmark(self, case):
        self.seen.append(case)
        if case.num_tokens in self.fail_tokens:
            raise RuntimeError(f"synthetic failure for {case.num_tokens}")
        sms = {
            "deepep_ht": a2a.HT_SMS,
            "deepep_ll": a2a.LL_SMS,
            "deepep_v2": 17,
        }[case.comm_backend]
        if self.bad_sms is not None:
            sms = self.bad_sms
        return a2a.BenchmarkResult(
            timings={
                "dispatch": PhaseTiming(11.0, 0.0),
                "combine": PhaseTiming(7.0, 0.0),
            },
            sms=sms,
            capacity=case.capacity + int(self.bad_capacity),
        )

    def close(self):
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


def _plan(backends=a2a.BACKENDS):
    return a2a.build_case_plan(
        shapes=[SHAPE],
        grid=GRID,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
        backends=backends,
    )


def test_import_does_not_import_vllm_or_deep_ep():
    code = (
        "import json,sys;"
        "import collector.wideep.vllm.collect_moe_a2a;"
        "print(json.dumps([n for n in sys.modules if n == 'vllm' or n.startswith('vllm.') "
        "or n == 'deep_ep' or n.startswith('deep_ep.')]))"
    )
    output = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
    )
    assert json.loads(output) == []


def test_declared_shapes_use_vllm_population(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    shapes = a2a.get_vllm_moe_a2a_shapes()
    assert shapes
    assert shapes == sorted(set(shapes))
    assert SHAPE in shapes


def test_ll_shape_population_declares_the_pinned_kernel_hidden_sizes(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    baseline = a2a.get_vllm_moe_a2a_shapes(required_expert_parallel_size=WORLD_SIZE)
    ll_shapes = a2a.get_vllm_moe_a2a_shapes(
        required_expert_parallel_size=WORLD_SIZE,
        supported_hidden_sizes=a2a.DEEPEP_LL_SUPPORTED_HIDDEN_SIZES,
    )

    assert ll_shapes
    assert set(ll_shapes) < set(baseline)
    assert all(shape.hidden_size in a2a.DEEPEP_LL_SUPPORTED_HIDDEN_SIZES for shape in ll_shapes)
    assert any(shape.hidden_size == 3584 for shape in baseline)
    assert all(shape.hidden_size != 3584 for shape in ll_shapes)


def test_transport_defaults_are_publishable_and_alternates_are_diagnostic():
    args = a2a.parse_args(["--gpus-per-node", "4"])

    assert args.allow_mnnvl is True
    assert args.disable_nvlink is False
    assert a2a.transport_is_default(
        allow_mnnvl=args.allow_mnnvl,
        disable_nvlink=args.disable_nvlink,
    )
    assert not a2a.transport_is_default(allow_mnnvl=False, disable_nvlink=False)
    assert not a2a.transport_is_default(allow_mnnvl=True, disable_nvlink=True)
    assert "diagnostic transport staged rows" in SOURCE


def test_rank_lifecycle_agreement_is_cpu_gloo_not_benchmark_nccl():
    assert 'agreement_group = dist.new_group(ranks, backend="gloo")' in SOURCE
    assert 'device="cpu"' in SOURCE
    assert "group=agreement_group" in SOURCE
    assert '"failure_agreement": "gloo_cpu"' in SOURCE


def test_case_failures_are_persisted_before_runtime_capability_gate():
    main_source = SOURCE[SOURCE.index("def main(") :]
    assert main_source.index("_write_failures(") < main_source.index(
        "DeepEP did not report a runtime topology capability"
    )


def test_plan_maps_backend_phase_dtype_sms_and_capacity():
    cases = _plan()
    by_backend = {backend: [case for case in cases if case.comm_backend == backend] for backend in a2a.BACKENDS}

    assert {(case.inference_phase, case.sms, case.capacity) for case in by_backend["deepep_ht"]} == {
        ("context", 20, a2a.HT_BUFFER_SIZE_BYTES)
    }
    assert {(case.inference_phase, case.sms) for case in by_backend["deepep_ll"]} == {("generation", 0)}
    assert {case.capacity for case in by_backend["deepep_ll"]} == {1, 16, 256}
    assert {(case.num_tokens, case.inference_phase, case.sms, case.capacity) for case in by_backend["deepep_v2"]} == {
        (1, "generation", None, 1),
        (16, "generation", None, 16),
        (256, "generation", None, 256),
        (512, "context", None, 512),
    }


def test_plan_ignores_sglang_sms_sweep_and_ht_is_20_only():
    cases = _plan(("deepep_ht",))
    assert {case.sms for case in cases} == {20}
    assert len(cases) == len(GRID["ht_token_counts"])


def test_plan_is_deterministic_and_static_keys_are_unique():
    left = _plan()
    right = _plan()
    assert left == right
    keys = [
        case.persisted_key(ep_size=WORLD_SIZE, node_num=NODE_NUM) for case in left if case.comm_backend != "deepep_v2"
    ]
    assert len(keys) == len(set(keys))
    assert [case.sort_key() for case in left] == sorted(case.sort_key() for case in left)
    assert a2a.case_plan_ids(left, world_size=WORLD_SIZE, node_num=NODE_NUM) == a2a.case_plan_ids(
        right,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    )


@pytest.mark.parametrize(("world_size", "node_num"), [(8, 2), (16, 2), (16, 4), (32, 4)])
def test_supported_multi_node_world_sizes(world_size, node_num):
    assert a2a.build_case_plan(
        shapes=[SHAPE],
        grid=GRID,
        world_size=world_size,
        node_num=node_num,
        backends=("deepep_ll",),
    )


def test_unsupported_world_size_and_zero_case_fail_closed():
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="world sizes"):
        a2a.build_case_plan(shapes=[SHAPE], grid=GRID, world_size=3, node_num=2)

    indivisible = MoeA2AShape(4096, 6, 10)
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="not divisible"):
        a2a.build_case_plan(
            shapes=[indivisible],
            grid=GRID,
            world_size=8,
            node_num=2,
            backends=("deepep_ll",),
        )

    unsupported_ll = MoeA2AShape(3584, 8, 256)
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="pinned kernel capability"):
        a2a.build_case_plan(
            shapes=[unsupported_ll],
            grid=GRID,
            world_size=WORLD_SIZE,
            node_num=NODE_NUM,
            backends=("deepep_ll",),
        )
    assert a2a.build_case_plan(
        shapes=[unsupported_ll],
        grid=GRID,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
        backends=("deepep_ht",),
    )


def test_pure_adapter_builds_unified_rows_and_full_unique_keys():
    cases = _plan()
    adapter = FakeAdapter()
    result = a2a.collect_with_adapter(
        cases,
        adapter=adapter,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    )

    assert adapter.closed
    assert not result.failures
    assert len(result.rows) == 2 * len(cases)
    assert {row["comm_backend"] for row in result.rows} == set(a2a.BACKENDS)
    assert {row["phase"] for row in result.rows} == {"dispatch", "combine"}
    assert {row["comm_dtype"] for row in result.rows} == {"default"}
    assert {row["sms"] for row in result.rows if row["comm_backend"] == "deepep_ht"} == {20}
    assert {row["sms"] for row in result.rows if row["comm_backend"] == "deepep_ll"} == {0}
    assert {row["sms"] for row in result.rows if row["comm_backend"] == "deepep_v2"} == {17}
    keys = [a2a._row_key(row) for row in result.rows]
    assert len(keys) == len(set(keys))
    assert all(row["latency"] == row["transmit_us"] + row["notify_us"] for row in result.rows)


def test_collection_prepares_adapter_before_first_case_and_closes_after_last():
    events = []

    class PreparingAdapter(FakeAdapter):
        def prepare(self, cases):
            events.append(("prepare", tuple(cases)))

        def benchmark(self, case):
            assert events and events[0][0] == "prepare"
            events.append(("benchmark", case))
            return super().benchmark(case)

        def close(self):
            events.append(("close",))
            super().close()

    cases = _plan(("deepep_ll",))
    adapter = PreparingAdapter()
    a2a.collect_with_adapter(
        cases,
        adapter=adapter,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    )

    assert events[0] == ("prepare", tuple(cases))
    assert [event[0] for event in events].count("benchmark") == len(cases)
    assert events[-1] == ("close",)


def test_real_adapter_sizes_one_ll_buffer_for_the_whole_plan(monkeypatch):
    hints = []

    class FakeBuffer:
        @staticmethod
        def get_low_latency_rdma_size_hint(**kwargs):
            hints.append(kwargs)
            return kwargs["num_max_dispatch_tokens_per_rank"] + kwargs["hidden"] + kwargs["num_experts"]

    deep_ep = ModuleType("deep_ep")
    deep_ep.Buffer = FakeBuffer
    monkeypatch.setitem(sys.modules, "deep_ep", deep_ep)
    identity = a2a.DistIdentity(
        rank=0,
        world_size=WORLD_SIZE,
        local_rank=0,
        gpus_per_node=4,
        node_num=NODE_NUM,
        master_addr="127.0.0.1",
        master_port="29500",
    )
    cases = _plan(("deepep_ll",))
    adapter = a2a.VllmBenchmarkAdapter(group=None, identity=identity)

    adapter.prepare(cases)

    assert len(hints) == len(cases)
    assert adapter._ll_rdma_bytes == max(
        case.capacity + case.shape.hidden_size + case.shape.num_experts for case in cases
    )
    assert adapter._ll_num_qps_per_rank == max(case.shape.num_experts // WORLD_SIZE for case in cases)
    assert {adapter._runtime_buffer_key(case) for case in cases} == {("deepep_ll",)}


def test_real_adapter_rejects_mixed_backend_buffer_plan():
    identity = a2a.DistIdentity(
        rank=0,
        world_size=WORLD_SIZE,
        local_rank=0,
        gpus_per_node=4,
        node_num=NODE_NUM,
        master_addr="127.0.0.1",
        master_port="29500",
    )
    adapter = a2a.VllmBenchmarkAdapter(group=None, identity=identity)

    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="exactly one DeepEP backend"):
        adapter.prepare(_plan(("deepep_ht", "deepep_ll")))


def test_canary_selects_every_backend_and_v2_inference_phase():
    selected = a2a.select_canary_cases(_plan())
    assert {(case.comm_backend, case.inference_phase) for case in selected} == {
        ("deepep_ht", "context"),
        ("deepep_ll", "generation"),
        ("deepep_v2", "context"),
        ("deepep_v2", "generation"),
    }
    assert len(selected) == 4


def test_adapter_forward_context_uses_pinned_serving_api(monkeypatch):
    calls = []

    class FakeVllmConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeParallelConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTorch(ModuleType):
        int32 = "int32"

        @staticmethod
        def full(shape, value, **kwargs):
            return {"shape": shape, "value": value, **kwargs}

    @contextmanager
    def fake_set_forward_context(attn_metadata, config, *, num_tokens, num_tokens_across_dp):
        calls.append(("enter", attn_metadata, config, num_tokens, num_tokens_across_dp))
        yield
        calls.append(("exit",))

    vllm = ModuleType("vllm")
    vllm.__path__ = []
    config = ModuleType("vllm.config")
    config.VllmConfig = FakeVllmConfig
    config.ParallelConfig = FakeParallelConfig
    forward_context = ModuleType("vllm.forward_context")
    forward_context.set_forward_context = fake_set_forward_context
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.config", config)
    monkeypatch.setitem(sys.modules, "vllm.forward_context", forward_context)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch("torch"))

    case = next(case for case in _plan(("deepep_v2",)) if case.inference_phase == "generation")
    identity = a2a.DistIdentity(
        rank=3,
        world_size=WORLD_SIZE,
        local_rank=3,
        gpus_per_node=4,
        node_num=NODE_NUM,
        master_addr="127.0.0.1",
        master_port="29500",
    )
    adapter = a2a.VllmBenchmarkAdapter(group=None, identity=identity)
    with adapter._forward_context(case):
        calls.append(("body",))

    assert calls[0][0:2] == ("enter", None)
    assert isinstance(calls[0][2], FakeVllmConfig)
    assert calls[0][2].parallel_config.data_parallel_size == WORLD_SIZE
    assert calls[0][2].parallel_config.data_parallel_rank == 3
    assert calls[0][3] == case.num_tokens
    assert calls[1:] == [("body",), ("exit",)]


def test_case_failure_is_classified_data_and_adapter_still_closes():
    cases = _plan(("deepep_ll",))
    adapter = FakeAdapter(fail_tokens={16})
    result = a2a.collect_with_adapter(
        cases,
        adapter=adapter,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    )

    assert adapter.closed
    assert len(result.failures) == 1
    assert result.failures[0].error_type == "RuntimeError"
    assert "synthetic failure" in result.failures[0].error
    assert len(result.rows) == 2 * (len(cases) - 1)


def test_peer_failure_discards_rank_local_success_rows():
    case = _plan(("deepep_ht",))[0]
    result = a2a.collect_with_adapter(
        [case],
        adapter=FakeAdapter(),
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
        failure_agreement=lambda local_failed: True,
    )

    assert result.rows == []
    assert len(result.failures) == 1
    assert result.failures[0].error_type == "VllmMoeA2APeerError"


def test_adapter_close_failure_aborts_before_returning_rows():
    case = _plan(("deepep_ht",))[0]
    adapter = FakeAdapter(close_error=RuntimeError("cudaErrorUnknown during destroy"))

    with pytest.raises(RuntimeError, match="cudaErrorUnknown"):
        a2a.collect_with_adapter(
            [case],
            adapter=adapter,
            world_size=WORLD_SIZE,
            node_num=NODE_NUM,
        )

    assert adapter.closed


def test_rank_error_records_non_case_failure(tmp_path):
    identity = a2a.DistIdentity(
        rank=2,
        world_size=4,
        local_rank=2,
        gpus_per_node=4,
        node_num=1,
        master_addr="127.0.0.1",
        master_port="12345",
    )

    path = a2a._write_rank_error(
        tmp_path,
        RuntimeError("CUDA runtime exception: cudaErrorUnknown"),
        identity,
        stage="fatal_runtime",
    )

    [record] = json.loads(path.read_text())
    assert record["classification"] == "unexpected"
    assert record["stage"] == "fatal_runtime"
    assert record["case"] is None
    assert record["rank"] == 2
    assert "cudaErrorUnknown" in record["error"]


@pytest.mark.parametrize(
    ("adapter", "match"),
    [
        (FakeAdapter(bad_sms=99), "expected"),
        (FakeAdapter(bad_capacity=True), "capacity"),
    ],
)
def test_adapter_contract_mismatch_becomes_visible_failure(adapter, match):
    [case] = a2a.build_case_plan(
        shapes=[SHAPE],
        grid={"ht_token_counts": [16], "ll_token_counts": [1], "sms": [16]},
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
        backends=("deepep_ht",),
    )
    result = a2a.collect_with_adapter(
        [case],
        adapter=adapter,
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    )
    assert len(result.failures) == 1
    assert match in result.failures[0].error
    assert result.rows == []


def test_write_loss_is_checked(monkeypatch, tmp_path):
    row = a2a.collect_with_adapter(
        _plan(("deepep_ht",)),
        adapter=FakeAdapter(),
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    ).rows[0]
    monkeypatch.setattr(a2a, "log_perf", lambda **kwargs: False)
    with pytest.raises(a2a.VllmMoeA2ABenchmarkError, match="write loss"):
        a2a._write_rows(
            [row],
            perf_path=tmp_path / "moe_a2a_perf.txt",
            runtime_meta={"version": "0.test"},
            device_name="fake",
        )


def test_stale_staging_is_rejected_before_append(tmp_path):
    path = tmp_path / "moe_a2a_perf.txt"
    path.write_text("stale\n")
    with pytest.raises(a2a.VllmMoeA2ABenchmarkError, match="stale staging"):
        a2a._write_rows(
            [],
            perf_path=path,
            runtime_meta={"version": "0.test"},
            device_name="fake",
        )
    assert path.read_text() == "stale\n"


def test_writer_uses_sglang_unified_schema(tmp_path):
    rows = a2a.collect_with_adapter(
        _plan(("deepep_ht",)),
        adapter=FakeAdapter(),
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    ).rows
    path = tmp_path / "moe_a2a_perf.txt"
    a2a._write_rows(
        rows,
        perf_path=path,
        runtime_meta={"version": "0.test"},
        device_name="NVIDIA Test GPU",
    )
    with path.open(newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert len(persisted) == len(rows)
    assert list(persisted[0]) == [
        "framework",
        "version",
        "device",
        "op_name",
        "kernel_source",
        "comm_backend",
        "phase",
        "comm_dtype",
        "ep_size",
        "node_num",
        "hidden_size",
        "topk",
        "num_experts",
        "num_tokens",
        "sms",
        "transmit_us",
        "notify_us",
        "latency",
    ]
    assert {row["framework"] for row in persisted} == {"vLLM"}
    assert {row["kernel_source"] for row in persisted} == {"deepep"}


def test_runtime_attestation_uses_live_hooks_and_rejects_wrong_commit(tmp_path):
    runtime = a2a.get_collector_runtime("vllm", workload="wideep")
    image_digest = runtime.image().split("@", 1)[1]
    observed_abi = runtime.abi_for_backend("deepep_ht") | {
        "system": "h100_sxm",
        "deep_ep_scaleup_ranks": "8",
    }
    live_abi = {
        "torch": "2.11.0",
        "deep_ep_api": "Buffer",
        "deep_ep_distribution": "1.2.1+73b6ea4",
    }
    meta = a2a.attest_vllm_runtime(
        source_root=tmp_path,
        backend="deepep_ht",
        observed_abi=observed_abi,
        observed_image_digest=image_digest,
        installed_version_getter=lambda name: runtime.version,
        source_commit_getter=lambda path: a2a.TARGET_VLLM_SOURCE_COMMIT,
        live_abi_getter=lambda backend: live_abi,
    )
    assert meta["framework"] == "wideep_vllm"
    assert meta["version"] == runtime.version
    assert meta["source_commit"] == a2a.TARGET_VLLM_SOURCE_COMMIT
    assert meta["abi"] == observed_abi
    assert meta["live_abi"] == live_abi
    assert meta["image_digest"] == image_digest
    grace_image, grace_image_digest = runtime.image("grace_blackwell").split("@", 1)
    grace_meta = a2a.attest_vllm_runtime(
        source_root=tmp_path,
        backend="deepep_ht",
        observed_abi=observed_abi,
        observed_image_digest=grace_image_digest,
        installed_version_getter=lambda name: runtime.version,
        source_commit_getter=lambda path: a2a.TARGET_VLLM_SOURCE_COMMIT,
        live_abi_getter=lambda backend: live_abi,
    )
    assert grace_meta["image"] == grace_image
    assert grace_meta["image_digest"] == grace_image_digest
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="source must be"):
        a2a.attest_vllm_runtime(
            source_root=tmp_path,
            backend="deepep_ht",
            observed_abi=observed_abi,
            observed_image_digest=image_digest,
            installed_version_getter=lambda name: runtime.version,
            source_commit_getter=lambda path: "wrong",
            live_abi_getter=lambda backend: live_abi,
        )
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="image digest mismatch"):
        a2a.attest_vllm_runtime(
            source_root=tmp_path,
            backend="deepep_ht",
            observed_abi=observed_abi,
            observed_image_digest="sha256:" + "0" * 64,
            installed_version_getter=lambda name: runtime.version,
            source_commit_getter=lambda path: a2a.TARGET_VLLM_SOURCE_COMMIT,
            live_abi_getter=lambda backend: live_abi,
        )


def test_runtime_attestation_splits_v2_and_legacy_nvl4_abis(tmp_path):
    runtime = a2a.get_collector_runtime("vllm", workload="wideep")
    image_digest = runtime.image().split("@", 1)[1]
    wheel_sha = "a" * 64
    v2_abi = runtime.abi_for_backend("deepep_v2") | {
        "system": "gb200",
        "deep_ep_topology_source": "nccl_lsa",
        "deep_ep_overlay_wheel_sha256": wheel_sha,
    }
    meta = a2a.attest_vllm_runtime(
        source_root=tmp_path,
        backend="deepep_v2",
        observed_abi=v2_abi,
        observed_image_digest=image_digest,
        installed_version_getter=lambda name: runtime.version,
        source_commit_getter=lambda path: a2a.TARGET_VLLM_SOURCE_COMMIT,
        live_abi_getter=lambda backend: {
            "torch": "2.11.0",
            "deep_ep_api": "ElasticBuffer",
            "deep_ep_distribution": "1.2.1+b306af0",
            "nccl": "2.30.4",
        },
    )
    assert meta["abi"]["deep_ep"] == a2a.V2_DEEPEP_COMMIT

    patch_sha = a2a._file_sha256(a2a.LEGACY_NVL4_PATCH)
    legacy_abi = runtime.abi_for_backend("deepep_ht") | {
        "system": "gb200",
        "deep_ep_scaleup_ranks": "4",
        "deep_ep_patch_sha256": patch_sha,
        "deep_ep_overlay_wheel_sha256": wheel_sha,
    }
    a2a.attest_vllm_runtime(
        source_root=tmp_path,
        backend="deepep_ht",
        observed_abi=legacy_abi,
        observed_image_digest=image_digest,
        installed_version_getter=lambda name: runtime.version,
        source_commit_getter=lambda path: a2a.TARGET_VLLM_SOURCE_COMMIT,
        live_abi_getter=lambda backend: {
            "torch": "2.11.0",
            "deep_ep_api": "Buffer",
            "deep_ep_distribution": "1.2.1+local",
        },
    )


def test_git_head_reads_checkout_metadata_when_git_is_unavailable(tmp_path, monkeypatch):
    git_dir = tmp_path / ".git"
    ref = git_dir / "refs/heads/main"
    ref.parent.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    ref.write_text(a2a.TARGET_VLLM_SOURCE_COMMIT + "\n")
    monkeypatch.setattr(a2a.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))

    assert a2a._git_head(tmp_path) == a2a.TARGET_VLLM_SOURCE_COMMIT


def test_git_head_reads_checkout_metadata_when_git_exits_non_zero(tmp_path, monkeypatch):
    """A staged checkout has `.git/HEAD` but no object store, so `git` exits 2."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text(a2a.TARGET_VLLM_SOURCE_COMMIT + "\n")

    def _exit_two(*args, **kwargs):
        raise a2a.subprocess.CalledProcessError(2, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(a2a.subprocess, "run", _exit_two)

    assert a2a._git_head(tmp_path) == a2a.TARGET_VLLM_SOURCE_COMMIT


def test_git_head_raises_when_neither_git_nor_metadata_resolves(tmp_path, monkeypatch):
    def _exit_two(*args, **kwargs):
        raise a2a.subprocess.CalledProcessError(2, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(a2a.subprocess, "run", _exit_two)

    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="cannot attest vLLM source commit"):
        a2a._git_head(tmp_path)


def test_collector_ref_degrades_to_unknown_outside_a_repo(tmp_path, monkeypatch):
    """`collector_hash` pins the code that ran; a missing repo ref never voids a run."""

    def _exit_two(*args, **kwargs):
        raise a2a.subprocess.CalledProcessError(2, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(a2a.subprocess, "run", _exit_two)

    assert a2a._git_collector_ref(tmp_path) == "unknown"


def test_collector_ref_uses_host_attestation_when_git_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_COLLECTOR_REF", a2a.TARGET_VLLM_SOURCE_COMMIT)
    monkeypatch.setattr(a2a, "_unattested_git_collector_ref", lambda _path: "unknown")

    assert a2a._git_collector_ref(tmp_path) == a2a.TARGET_VLLM_SOURCE_COMMIT


def test_collector_ref_rejects_invalid_or_mismatched_host_attestation(tmp_path, monkeypatch):
    monkeypatch.setenv("AIC_COLLECTOR_REF", "not-a-sha")
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="invalid AIC_COLLECTOR_REF"):
        a2a._git_collector_ref(tmp_path)

    monkeypatch.setenv("AIC_COLLECTOR_REF", "a" * 40)
    monkeypatch.setattr(a2a, "_unattested_git_collector_ref", lambda _path: "b" * 40)
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="does not match mounted checkout"):
        a2a._git_collector_ref(tmp_path)


def test_classified_case_failures_do_not_demote_finalized_table(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    del pyarrow
    rows = a2a.collect_with_adapter(
        _plan(("deepep_ht",)),
        adapter=FakeAdapter(),
        world_size=WORLD_SIZE,
        node_num=NODE_NUM,
    ).rows
    perf = tmp_path / "moe_a2a_perf.txt"
    a2a._write_rows(
        rows,
        perf_path=perf,
        runtime_meta={"version": "0.test"},
        device_name="fake",
    )
    [parquet] = a2a.finalize_perf_files([perf], merge_existing=False)
    runtime = {
        "framework": "vllm",
        "version": "0.test",
        "source_commit": a2a.TARGET_VLLM_SOURCE_COMMIT,
    }
    complete = a2a._write_sidecar(
        tmp_path,
        runtime_meta=runtime,
        case_ids=["a"],
        parquet_path=parquet,
        failure_count=0,
    )
    import yaml

    assert yaml.safe_load(complete.read_text())["tables"]["moe_a2a_perf"]["status"] == "complete"
    with_failures = a2a._write_sidecar(
        tmp_path,
        runtime_meta=runtime,
        case_ids=["a"],
        parquet_path=parquet,
        failure_count=1,
    )
    assert yaml.safe_load(with_failures.read_text())["tables"]["moe_a2a_perf"]["status"] == "complete"
    with pytest.raises(a2a.VllmMoeA2ADeclarationError, match="empty"):
        a2a._write_sidecar(
            tmp_path,
            runtime_meta=runtime,
            case_ids=[],
            parquet_path=parquet,
            failure_count=0,
        )


def test_exact_vllm_prepare_finalize_classes_and_calls_are_present():
    for class_name in (
        "DeepEPHTPrepareAndFinalize",
        "DeepEPLLPrepareAndFinalize",
        "DeepEPV2PrepareAndFinalize",
    ):
        assert class_name in SOURCE
    assert "prepare_finalize.prepare(" in SOURCE
    assert "prepare_finalize.finalize(" in SOURCE
    assert "deep_ep.ElasticBuffer(" in SOURCE
    assert "allow_hybrid_mode=True" in SOURCE
    assert "get_theoretical_num_sms(" in SOURCE
    assert 'backend="nccl"' in SOURCE
    assert SOURCE.index('elif case.comm_backend == "deepep_v2":') < SOURCE.index(
        "from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import"
    )


def test_v2_is_never_silently_skipped():
    v2_body = SOURCE.split('elif case.comm_backend == "deepep_v2":', 1)[1].split("else:  # defensive", 1)[0]
    assert "deep_ep.ElasticBuffer(" in v2_body
    assert "get_logical_domain_size()" in v2_body
    assert "get_physical_domain_size()" in v2_body
    assert "_record_capability(" in v2_body
    assert "continue" not in v2_body
    assert "query_nccl_gin_type" not in SOURCE
