# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD heterogeneous pools: prefill / A / F each on their own hardware.

Covers the three guarantees the feature rests on:

1. **Inheritance** — a pool that is not given its own system/backend resolves
   to the top-level one, and the homogeneous call path is untouched (no new
   kwargs reach the ops, same perf DB object).
2. **Routing** — A ops are priced against the A pool's DB and F ops against
   the F pool's DB; F-node grouping uses the F pool's ``gpus_per_node``.
3. **Bottleneck pricing** — cross-pool A2F/F2A transfers are charged at the
   slower endpoint (bandwidth = min of both sides), symmetrically.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

from aiconfigurator.sdk.config import AFDConfig
from aiconfigurator.sdk.inference_session import AFDInferenceSession
from aiconfigurator.sdk.operations import AFDFAllGather, AFDFReduceScatter, AFDTransfer
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit


class _StubDatabase:
    """Perf DB stub whose p2p latency scales with ``ns_per_byte``.

    A larger ``ns_per_byte`` is a *slower* link, i.e. lower bandwidth.
    """

    system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": 80 * (1 << 30)}}

    def __init__(self, name: str, ns_per_byte: float = 1.0) -> None:
        self.system = name
        self.version = f"{name}-version"
        self._ns_per_byte = ns_per_byte
        self.p2p_calls: list[int] = []

    # No ``query_p2p``: the analytic P2P formula moved into the compiled engine
    # and the per-call lookup was retired. Pricing now comes from
    # ``_hetero_engine`` below, which keys off this object's ``_ns_per_byte``.


class _HeteroEngineStub:
    """Stands in for ``_engine_comm_query``, pricing per *database*.

    The engine rejects duck-typed databases (``_require_real_database``), and
    bottleneck pricing is by definition about two endpoints disagreeing -- so
    the seam to intercept is the one function that reaches the engine. Latency
    is the probe volume times the database's own ``_ns_per_byte``.
    """

    def __init__(self) -> None:
        self.calls: list = []

    def __call__(self, database, op):
        volume = int(op._h) * 2
        self.calls.append((getattr(database, "system", database), volume))
        rate = getattr(database, "_ns_per_byte", 1.0)
        if hasattr(database, "p2p_calls"):
            database.p2p_calls.append(volume)
        return PerformanceResult(latency=float(volume) * rate, energy=0.0)


@pytest.fixture()
def hetero_engine(monkeypatch) -> _HeteroEngineStub:
    from aiconfigurator_core.sdk.operations import afd_transfer as afd_transfer_module

    stub = _HeteroEngineStub()
    monkeypatch.setattr(afd_transfer_module, "_engine_comm_query", stub)
    return stub


def _backend(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=SimpleNamespace(value=name),
        get_default_free_gpu_memory_fraction=lambda *_a, **_k: 0.9,
        get_kv_cache_memory_check_params=lambda: (0.0, 0.0),
        memory_fraction_of_free=lambda: 1.0,
    )


def _session(*, f_database=None, f_backend=None, afd_config=None) -> AFDInferenceSession:
    return AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(nextn=0),
        f_model_config=SimpleNamespace(nextn=0),
        database=_StubDatabase("a-system"),
        backend=_backend("a-backend"),
        afd_config=afd_config or AFDConfig(n_a_nodes=1, n_f_nodes=1, gpus_per_node=8, tp_a=2, a_batch_size=4),
        f_database=f_database,
        f_backend=f_backend,
    )


# ---------------------------------------------------------------------------
# 1. Inheritance: an unspecified pool must stay identical to the A pool
# ---------------------------------------------------------------------------


class TestPoolInheritance:
    def test_session_defaults_f_pool_to_a_pool(self):
        session = _session()
        assert session._f_database is session._a_database
        assert session._f_backend is session._a_backend
        assert session._database is session._a_database  # legacy A-side alias
        assert session.is_hetero is False

    def test_session_reports_hetero_on_distinct_database(self):
        session = _session(f_database=_StubDatabase("f-system"))
        assert session.is_hetero is True
        assert session._f_database is not session._a_database

    def test_session_reports_hetero_on_distinct_backend(self):
        session = _session(f_backend=_backend("f-backend"))
        assert session.is_hetero is True

    def test_afd_config_gpus_per_node_defaults_to_shared(self):
        cfg = AFDConfig(n_a_nodes=1, n_f_nodes=1, gpus_per_node=8, tp_a=2, a_batch_size=4)
        assert cfg.effective_a_gpus_per_node == 8
        assert cfg.effective_f_gpus_per_node == 8

    def test_transfer_without_peer_is_priced_on_one_side(self, hetero_engine):
        """No ``peer_database`` kwarg == pre-hetero behavior, byte for byte."""
        a_db = _StubDatabase("a", ns_per_byte=1.0)
        op = AFDTransfer(
            name="a2f",
            scale_factor=1.0,
            direction="a2f",
            hidden_size=128,
            n_a_workers=4,
            n_f_workers=8,
            gpus_per_node=8,
        )
        result = op.query(a_db, x=16)
        assert float(result) == pytest.approx(16 * 128 * 2 * 1.0)


# ---------------------------------------------------------------------------
# 2. Routing: per-pool database and per-pool F-node grouping
# ---------------------------------------------------------------------------


class TestPoolRouting:
    def test_sum_latency_uses_requested_database(self):
        f_db = _StubDatabase("f")
        session = _session(f_database=f_db)
        seen = []

        class _Op:
            _name = "op"

            # ``_sum_latency`` calls the internal shim entry, not the
            # deprecated public ``query``; a double that only defines the
            # latter is no longer faithful to what the session invokes.
            def _engine_query(self, database, **_kwargs):
                seen.append(database.system)
                return 1.0

        def _run(**kwargs):
            session._sum_latency(
                [_Op()],
                batch_size=1,
                seq_len=1,
                model=SimpleNamespace(model_name="m"),
                runtime_config=SimpleNamespace(prefix=0, gen_seq_imbalance_correction_scale=1.0),
                is_context=False,
                **kwargs,
            )

        _run(database=session._f_database)
        _run(database=session._a_database)
        _run()  # omitted == A pool
        assert seen == ["f", "a-system", "a-system"]

    def test_check_memory_dict_uses_pool_capacity(self):
        f_db = _StubDatabase("f")
        f_db.system_spec = {"gpu": {"mem_capacity": 24 * (1 << 30)}}
        session = _session(f_database=f_db, f_backend=_backend("f-backend"))
        runtime_config = SimpleNamespace()
        memory = {"total": 40.0, "weights": 40.0, "activations": 0.0, "kvcache": 0.0, "nccl": 0.0, "others": 0.0}

        seen: list[float] = []

        class _Summary:
            def set_memory_and_check_oom(self, _memory, capacity, **_kwargs):
                seen.append(capacity)

        import aiconfigurator.sdk.inference_session as sess_mod

        original = sess_mod.InferenceSummary
        sess_mod.InferenceSummary = lambda _rc: _Summary()
        try:
            session._check_memory_dict(memory, runtime_config, None, pool="a")
            session._check_memory_dict(memory, runtime_config, None, pool="f")
        finally:
            sess_mod.InferenceSummary = original

        # A pool keeps the 80 GiB device, F pool is checked against its own 24 GiB.
        assert seen == [80 * (1 << 30), 24 * (1 << 30)]

    def test_check_memory_dict_rejects_unknown_pool(self):
        session = _session()
        with pytest.raises(ValueError, match="pool must be 'a' or 'f'"):
            session._check_memory_dict({}, SimpleNamespace(), None, pool="x")

    @pytest.mark.parametrize("op_cls", [AFDFAllGather, AFDFReduceScatter])
    def test_f_collectives_group_by_f_gpus_per_node(self, op_cls):
        """F-node grouping is an F-pool fact: 8 F GPUs over 4-GPU nodes = 2 nodes."""
        op = op_cls(
            name="f_op",
            scale_factor=1.0,
            hidden_size=128,
            n_a_workers=4,
            n_f_workers=8,
            gpus_per_node=8,
            f_gpus_per_node=4,
        )
        assert op.num_f_nodes == 2
        assert op.f_gpus_in_node == 4

    def test_transfer_groups_by_f_gpus_per_node(self):
        op = AFDTransfer(
            name="a2f",
            scale_factor=1.0,
            direction="a2f",
            hidden_size=128,
            n_a_workers=4,
            n_f_workers=8,
            gpus_per_node=8,
            f_gpus_per_node=4,
        )
        assert op.num_f_nodes == 2


# ---------------------------------------------------------------------------
# 3. Bottleneck pricing for the cross-pool link
# ---------------------------------------------------------------------------


def _a2f(**kwargs) -> AFDTransfer:
    return AFDTransfer(
        name="a2f",
        scale_factor=1.0,
        direction="a2f",
        hidden_size=128,
        n_a_workers=4,
        n_f_workers=8,
        gpus_per_node=8,
        **kwargs,
    )


class TestBottleneckPricing:
    def test_slow_peer_wins(self, hetero_engine):
        fast = _StubDatabase("fast", ns_per_byte=1.0)
        slow = _StubDatabase("slow", ns_per_byte=4.0)
        op = _a2f()
        priced = float(op.query(fast, x=16, peer_database=slow))
        assert priced == pytest.approx(float(op.query(slow, x=16)))

    def test_fast_peer_does_not_speed_up_slow_side(self, hetero_engine):
        fast = _StubDatabase("fast", ns_per_byte=1.0)
        slow = _StubDatabase("slow", ns_per_byte=4.0)
        op = _a2f()
        priced = float(op.query(slow, x=16, peer_database=fast))
        assert priced == pytest.approx(float(op.query(slow, x=16)))

    def test_pricing_is_symmetric(self, hetero_engine):
        """min-bandwidth is commutative: swapping the endpoints changes nothing."""
        fast = _StubDatabase("fast", ns_per_byte=1.0)
        slow = _StubDatabase("slow", ns_per_byte=4.0)
        op = _a2f()
        assert float(op.query(fast, x=16, peer_database=slow)) == pytest.approx(
            float(op.query(slow, x=16, peer_database=fast))
        )

    def test_identical_peer_is_a_noop(self, hetero_engine):
        db = _StubDatabase("same", ns_per_byte=2.0)
        op = _a2f()
        assert float(op.query(db, x=16, peer_database=db)) == pytest.approx(float(op.query(db, x=16)))

    def test_comm_overhead_factor_still_applies(self, hetero_engine):
        fast = _StubDatabase("fast", ns_per_byte=1.0)
        slow = _StubDatabase("slow", ns_per_byte=4.0)
        op = _a2f(comm_overhead_factor=2.0)
        assert float(op.query(fast, x=16, peer_database=slow)) == pytest.approx(16 * 128 * 2 * 4.0 * 2.0)


# ---------------------------------------------------------------------------
# 4. AFDConfig topology under per-pool node widths
# ---------------------------------------------------------------------------


class TestAfdConfigHeteroTopology:
    def test_workers_derive_from_respective_side(self):
        """A on 8-GPU nodes, F on 4-GPU nodes: each side counts its own GPUs."""
        cfg = AFDConfig(
            n_a_nodes=2,
            n_f_nodes=3,
            gpus_per_node=8,
            a_gpus_per_node=8,
            f_gpus_per_node=4,
            tp_a=2,
            a_batch_size=4,
        )
        assert cfg.n_a_workers == 2 * 8 // 2
        assert cfg.n_f_workers == 3 * 4
        assert cfg.tp_f == 12
        assert cfg.effective_a_gpus_per_node == 8
        assert cfg.effective_f_gpus_per_node == 4

    def test_tp_a_must_divide_a_side_node_width(self):
        with pytest.raises(ValueError, match="divisor of the A-pool"):
            AFDConfig(
                n_a_nodes=1,
                n_f_nodes=1,
                gpus_per_node=8,
                a_gpus_per_node=4,
                f_gpus_per_node=8,
                tp_a=8,
                a_batch_size=4,
            )

    def test_rejects_non_positive_pool_width(self):
        with pytest.raises(ValueError, match="f_gpus_per_node"):
            AFDConfig(n_a_nodes=1, n_f_nodes=1, gpus_per_node=8, f_gpus_per_node=0, tp_a=2, a_batch_size=4)


# ---------------------------------------------------------------------------
# 5. Task-level pool resolution and validation
# ---------------------------------------------------------------------------


def _afd_task(**overrides) -> Task:
    base = {
        "serving_mode": "afd",
        "model_path": "Qwen/Qwen3-32B",
        "system_name": "h200_sxm",
        "backend_name": "trtllm",
        "total_gpus": 32,
    }
    base.update(overrides)
    return Task(**base)


class TestTaskPoolResolution:
    def test_unset_pools_inherit_top_level(self):
        task = _afd_task()
        for pool in Task.AFD_POOLS:
            assert task.afd_pool_attr(pool, "system_name") == "h200_sxm"
            assert task.afd_pool_attr(pool, "backend_name") == "trtllm"
        assert task.afd_pools_are_homogeneous is True
        assert task.afd_pool_overrides() == {}

    def test_f_pool_override_is_detected(self):
        task = _afd_task(afd_f_system_name="b200_sxm")
        assert task.afd_pool_attr("afd_f", "system_name") == "b200_sxm"
        assert task.afd_pool_attr("afd_a", "system_name") == "h200_sxm"
        assert task.afd_pools_are_homogeneous is False
        assert set(task.afd_pool_overrides()) == {"afd_f"}

    def test_prefill_pool_override_is_detected(self):
        task = _afd_task(afd_prefill_system_name="b200_sxm")
        assert set(task.afd_pool_overrides()) == {"afd_prefill"}

    def test_per_pool_gpus_per_node_resolved_from_that_system(self):
        """h200_sxm has 8 GPUs per node, gb200 has 4."""
        task = _afd_task(afd_f_system_name="gb200")
        assert task._afd_a_gpus_per_node == 8
        assert task._afd_f_gpus_per_node == 4

    def test_pool_backend_version_resolved_independently(self):
        """A pool that moves hardware cannot inherit the top-level DB version."""
        task = _afd_task(afd_f_system_name="b200_sxm")
        assert task.afd_pool_attr("afd_f", "backend_version")

    def test_unknown_pool_name_rejected(self):
        task = _afd_task()
        with pytest.raises(ValueError, match="pool must be one of"):
            task.afd_pool_attr("afd_z", "system_name")

    def test_unknown_pool_system_rejected(self):
        """Construction resolves each pool's node width, so a bad system fails there."""
        with pytest.raises(ValueError, match="valid system yaml spec for every pool"):
            _afd_task(afd_f_system_name="no_such_system_xyz")

    def test_unknown_pool_backend_rejected(self):
        """Backend names are policed by validate(), which run() calls first."""
        task = _afd_task(afd_a_backend_name="not_a_backend")
        with pytest.raises(ValueError, match="is not a known backend"):
            task.validate()

    def test_prefill_pool_without_combined_with_pd_rejected(self):
        task = _afd_task(afd_combined_with_pd=False, afd_prefill_system_name="b200_sxm")
        with pytest.raises(ValueError, match="no static prefill pool"):
            task.validate()

    def test_homogeneous_afd_task_still_validates(self):
        """Regression guard: the new pool checks must not reject a plain AFD task."""
        _afd_task().validate()


# ---------------------------------------------------------------------------
# 6. afd_pareto level: homogeneous parity, per-pool routing, cross-vendor
# ---------------------------------------------------------------------------


class _PoolDatabase:
    """Perf DB stub identified by name, with its own node width."""

    def __init__(self, name: str, num_gpus_per_node: int = 8, version: str = "v1") -> None:
        self.system = name
        self.version = version
        self.system_spec = {"node": {"num_gpus_per_node": num_gpus_per_node}}


def _patch_pareto(monkeypatch, *, max_batch_size: int = 1024):
    """Stub afd_pareto's heavy dependencies and record per-pool routing.

    Mirrors ``test_afd_pareto_review_actions._patch_afd_pareto_fixed_batch_dependencies``
    but additionally captures which database/backend each pool was evaluated
    against, which is what the hetero wiring has to get right.
    """
    import copy

    import pandas as pd

    from aiconfigurator.sdk import pareto_analysis as pa

    captured = {
        "afd_configs": [],
        "session_kwargs": [],
        "a_batch_dbs": [],
        "f_batch_dbs": [],
        "prefill_dbs": [],
        "backends": [],
    }

    class FakeBackend:
        def __init__(self, name: str = "trtllm") -> None:
            self.name = SimpleNamespace(value=name)

        def get_partition_memory_usage(self, *_args, **_kwargs):
            return {"total": 1.0, "kvcache": 0.1}

        def get_kv_cache_memory_check_params(self):
            return 0.0, 0.0

    class FakeSummary:
        def __init__(self, runtime_config, afd_config, a_system, f_system):
            self._rc = copy.deepcopy(runtime_config)
            self._cfg = copy.deepcopy(afd_config)
            self._a_system = a_system
            self._f_system = f_system

        def check_oom(self):
            return False

        def get_result_dict(self):
            label = self._a_system if self._a_system == self._f_system else f"{self._a_system}+{self._f_system}"
            return {
                "model": "Qwen/Qwen3-32B",
                "phase": "decode",
                "isl": self._rc.isl,
                "osl": self._rc.osl,
                "(a)nodes": self._cfg.n_a_nodes,
                "(a)tp": self._cfg.tp_a,
                "(a)bs": self._cfg.a_batch_size,
                "(a)workers": self._cfg.n_a_workers,
                "(f)nodes": self._cfg.n_f_nodes,
                "(f)tp": self._cfg.tp_f,
                "(f)ep": self._cfg.f_moe_ep_size,
                "(f)workers": self._cfg.n_f_workers,
                "ttft": 0.0,
                "tpot": 10.0,
                "request_latency": 10.0 * max(self._rc.osl - 1, 1),
                "seq/s": 1.0,
                "request_rate": 1.0,
                "tokens/s": float(self._rc.osl),
                "tokens/s/gpu": 1.0,
                "tokens/s/user": float(self._rc.osl),
                "concurrency": self._rc.batch_size,
                "parallel": f"a{self._cfg.n_a_nodes}n-tp{self._cfg.tp_a}+f{self._cfg.n_f_nodes}n",
                "num_total_gpus": 16,
                "memory": 1.0,
                "power_w": 0.0,
                "backend": "trtllm",
                "version": "test-version",
                "system": label,
            }

        def get_summary_df(self):
            return pd.DataFrame([{"tokens/s/gpu": 1.0}])

    class FakeAFDInferenceSession:
        def __init__(self, *, afd_config, **kwargs):
            captured["afd_configs"].append(copy.deepcopy(afd_config))
            captured["session_kwargs"].append(kwargs)
            self._cfg = afd_config
            self._a_system = str(kwargs.get("a_system_name") or "")
            self._f_system = str(kwargs.get("f_system_name") or "")

        def run_afd(self, runtime_config, **_kwargs):
            return FakeSummary(runtime_config, self._cfg, self._a_system, self._f_system)

    def fake_analytical_max_batch_size(backend, _model, database, _ops, *, include_kvcache, **_kwargs):
        # include_kvcache marks the A pool (it owns the KV cache); F is the other.
        bucket = "a_batch_dbs" if include_kvcache else "f_batch_dbs"
        captured[bucket].append(getattr(database, "system", None))
        captured["backends"].append((bucket, backend.name.value))
        return max_batch_size

    def fake_derive_a_batch_size(_model_path, _model_config, backend, database, **_kwargs):
        captured["a_batch_dbs"].append(getattr(database, "system", None))
        captured["backends"].append(("a_batch_dbs", backend.name.value))
        return max_batch_size, object(), SimpleNamespace(attn_ops=[])

    def fake_enumerate_prefill(*, database, prefill_database=None, **_kwargs):
        effective = prefill_database or database
        captured["prefill_dbs"].append(getattr(effective, "system", None))
        return [
            {
                "tp": 1,
                "pp": 1,
                "dp": 1,
                "moe_tp": 1,
                "moe_ep": 1,
                "batch_size": 1,
                "num_gpus": 1,
                "workers": 1,
                "ttft": 1.0,
                "seq_s": 1000.0,
                "memory": 1.0,
                "power": 0.0,
                "system": getattr(effective, "system", None),
            }
        ]

    monkeypatch.setattr(pa, "get_backend", lambda name: FakeBackend(name))
    monkeypatch.setattr(pa, "get_model", lambda *_a, **_k: object())
    monkeypatch.setattr(pa, "AFDInferenceSession", FakeAFDInferenceSession)
    monkeypatch.setattr(pa, "_analytical_max_batch_size", fake_analytical_max_batch_size)
    monkeypatch.setattr(pa, "_derive_a_batch_size", fake_derive_a_batch_size)
    monkeypatch.setattr(pa, "_quick_balance_ratio", lambda *_a, **_k: 1.0)
    monkeypatch.setattr(pa, "_enumerate_afd_prefill_options", fake_enumerate_prefill)
    monkeypatch.setattr(
        "aiconfigurator.sdk.afd_partition.build_afd_ops_partition",
        lambda *_a, **_k: SimpleNamespace(attn_ops=[], ffn_ops=[]),
    )
    return captured, pa


_AFD_CANDIDATES = [(1, 1, 2, 1, 3, "optimistic")]


def _pareto_kwargs(**overrides):
    from aiconfigurator.sdk.config import RuntimeConfig

    kwargs = dict(
        model_path="Qwen/Qwen3-32B",
        runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
        database=_PoolDatabase("h200_sxm"),
        backend_name="trtllm",
        afd_parallel_config_list=list(_AFD_CANDIDATES),
        gpus_per_node=8,
        combined_with_pd=False,
        total_batch_size=256,
    )
    kwargs.update(overrides)
    return kwargs


class TestAfdParetoHeteroWiring:
    def test_homogeneous_output_is_identical_column_by_column(self, monkeypatch):
        """Spec guarantee: passing no per-pool argument reproduces the old rows.

        The baseline call omits every ``a_*`` / ``f_*`` argument; the second one
        passes them explicitly set to the shared values. Both DataFrames must
        match column by column (NaN-aware, so unset columns count as equal).
        """
        import pandas as pd

        captured, pa = _patch_pareto(monkeypatch)
        baseline = pa.afd_pareto(**_pareto_kwargs())

        shared_db = _PoolDatabase("h200_sxm")
        explicit = pa.afd_pareto(
            **_pareto_kwargs(
                database=shared_db,
                a_database=shared_db,
                a_backend_name="trtllm",
                a_system_name="h200_sxm",
                a_gpus_per_node=8,
                f_database=shared_db,
                f_backend_name="trtllm",
                f_system_name="h200_sxm",
                f_gpus_per_node=8,
            )
        )

        assert not baseline.empty
        assert list(baseline.columns) == list(explicit.columns)
        pd.testing.assert_frame_equal(baseline, explicit)

    def test_homogeneous_row_keeps_bare_system_label(self, monkeypatch):
        _captured, pa = _patch_pareto(monkeypatch)
        df = pa.afd_pareto(**_pareto_kwargs())
        assert df["system"].tolist() == ["h200_sxm"]

    def test_a_and_f_ops_use_their_own_database(self, monkeypatch):
        captured, pa = _patch_pareto(monkeypatch)
        pa.afd_pareto(
            **_pareto_kwargs(
                a_database=_PoolDatabase("h200_sxm"),
                a_system_name="h200_sxm",
                f_database=_PoolDatabase("b200_sxm"),
                f_system_name="b200_sxm",
            )
        )
        assert set(captured["a_batch_dbs"]) == {"h200_sxm"}
        assert set(captured["f_batch_dbs"]) == {"b200_sxm"}

    def test_session_receives_both_pools(self, monkeypatch):
        captured, pa = _patch_pareto(monkeypatch)
        a_db = _PoolDatabase("h200_sxm")
        f_db = _PoolDatabase("b200_sxm")
        pa.afd_pareto(
            **_pareto_kwargs(
                a_database=a_db,
                a_system_name="h200_sxm",
                f_database=f_db,
                f_system_name="b200_sxm",
            )
        )
        kwargs = captured["session_kwargs"][0]
        assert kwargs["database"] is a_db
        assert kwargs["f_database"] is f_db
        assert kwargs["a_system_name"] == "h200_sxm"
        assert kwargs["f_system_name"] == "b200_sxm"

    def test_hetero_row_carries_a_plus_f_system_label(self, monkeypatch):
        _captured, pa = _patch_pareto(monkeypatch)
        df = pa.afd_pareto(
            **_pareto_kwargs(
                a_system_name="h200_sxm",
                f_database=_PoolDatabase("b200_sxm"),
                f_system_name="b200_sxm",
            )
        )
        assert df["system"].tolist() == ["h200_sxm+b200_sxm"]

    def test_per_pool_node_width_shapes_each_side(self, monkeypatch):
        """A on 8-GPU nodes, F on GB200's 4-GPU nodes."""
        captured, pa = _patch_pareto(monkeypatch)
        pa.afd_pareto(
            **_pareto_kwargs(
                afd_parallel_config_list=[(2, 3, 2, 1, 3, "optimistic")],
                total_batch_size=None,
                a_gpus_per_node=8,
                f_database=_PoolDatabase("gb200", num_gpus_per_node=4),
                f_gpus_per_node=4,
            )
        )
        cfg = captured["afd_configs"][0]
        assert cfg.n_a_workers == 2 * 8 // 2
        assert cfg.n_f_workers == 3 * 4
        assert cfg.tp_f == 12

    def test_tp_a_exceeding_a_node_width_is_skipped(self, monkeypatch):
        """tp_a must fit inside one A node; such candidates are pruned."""
        captured, pa = _patch_pareto(monkeypatch)
        with pytest.raises(Exception):  # noqa: B017 - NoFeasibleConfigError or empty-result error
            pa.afd_pareto(
                **_pareto_kwargs(
                    afd_parallel_config_list=[(1, 1, 8, 1, 3, "optimistic")],
                    total_batch_size=None,
                    a_gpus_per_node=4,
                    f_gpus_per_node=8,
                )
            )
        assert captured["afd_configs"] == []

    def test_cross_vendor_nvidia_a_with_intel_b60_f(self, monkeypatch):
        """A on H200 (NCCL) + F on B60 (Intel, oneCCL): two perf DBs coexist."""
        captured, pa = _patch_pareto(monkeypatch)
        df = pa.afd_pareto(
            **_pareto_kwargs(
                a_database=_PoolDatabase("h200_sxm"),
                a_backend_name="trtllm",
                a_system_name="h200_sxm",
                f_database=_PoolDatabase("b60"),
                f_backend_name="vllm",
                f_system_name="b60",
                f_gpus_per_node=8,
            )
        )
        assert not df.empty
        assert set(captured["a_batch_dbs"]) == {"h200_sxm"}
        assert set(captured["f_batch_dbs"]) == {"b60"}
        # Each pool is evaluated on its own framework.
        assert ("a_batch_dbs", "trtllm") in captured["backends"]
        assert ("f_batch_dbs", "vllm") in captured["backends"]
        assert df["system"].tolist() == ["h200_sxm+b60"]

    def test_prefill_pool_uses_its_own_database(self, monkeypatch):
        """combined_with_pd: the static prefill pool can sit on another device."""
        captured, pa = _patch_pareto(monkeypatch)
        pa.afd_pareto(
            **_pareto_kwargs(
                combined_with_pd=True,
                prefill_database=_PoolDatabase("b200_sxm"),
                prefill_system_name="b200_sxm",
                prefill_gpus_per_node=8,
            )
        )
        assert captured["prefill_dbs"] == ["b200_sxm"]

    def test_prefill_pool_defaults_to_shared_database(self, monkeypatch):
        captured, pa = _patch_pareto(monkeypatch)
        pa.afd_pareto(**_pareto_kwargs(combined_with_pd=True))
        assert captured["prefill_dbs"] == ["h200_sxm"]


# ---------------------------------------------------------------------------
# 7. sweep_afd passes every per-pool argument straight through to afd_pareto
# ---------------------------------------------------------------------------


class TestSweepAfdPassthrough:
    _POOL_ARGS = (
        "a_database",
        "a_backend_name",
        "a_system_name",
        "a_gpus_per_node",
        "f_database",
        "f_backend_name",
        "f_system_name",
        "f_gpus_per_node",
        "prefill_gpus_per_node",
    )

    def test_every_pool_argument_reaches_afd_pareto(self, monkeypatch):
        import pandas as pd

        from aiconfigurator.sdk import pareto_analysis as pa
        from aiconfigurator.sdk import sweep as sweep_mod
        from aiconfigurator.sdk.config import RuntimeConfig

        seen: dict = {}

        def fake_afd_pareto(**kwargs):
            seen.update(kwargs)
            return pd.DataFrame([{"tokens/s/gpu": 1.0}])

        monkeypatch.setattr(pa, "afd_pareto", fake_afd_pareto)

        a_db, f_db = _PoolDatabase("h200_sxm"), _PoolDatabase("b60")
        sweep_mod.sweep_afd(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
            database=a_db,
            backend_name="trtllm",
            model_config=None,
            afd_parallel_config_list=list(_AFD_CANDIDATES),
            gpus_per_node=8,
            a_database=a_db,
            a_backend_name="trtllm",
            a_system_name="h200_sxm",
            a_gpus_per_node=8,
            f_database=f_db,
            f_backend_name="vllm",
            f_system_name="b60",
            f_gpus_per_node=8,
            prefill_gpus_per_node=4,
        )

        for name in self._POOL_ARGS:
            assert name in seen, f"sweep_afd dropped {name!r} on the way to afd_pareto"
        assert seen["a_database"] is a_db
        assert seen["f_database"] is f_db
        assert seen["a_system_name"] == "h200_sxm"
        assert seen["f_system_name"] == "b60"
        assert seen["f_backend_name"] == "vllm"
        assert seen["prefill_gpus_per_node"] == 4

    def test_omitting_pool_arguments_forwards_none(self, monkeypatch):
        """Defaults must arrive as None so afd_pareto applies its own fallback."""
        import pandas as pd

        from aiconfigurator.sdk import pareto_analysis as pa
        from aiconfigurator.sdk import sweep as sweep_mod
        from aiconfigurator.sdk.config import RuntimeConfig

        seen: dict = {}

        def fake_afd_pareto(**kwargs):
            seen.update(kwargs)
            return pd.DataFrame([{"tokens/s/gpu": 1.0}])

        monkeypatch.setattr(pa, "afd_pareto", fake_afd_pareto)

        sweep_mod.sweep_afd(
            model_path="Qwen/Qwen3-32B",
            runtime_config=RuntimeConfig(isl=128, osl=32, tpot=25.0),
            database=_PoolDatabase("h200_sxm"),
            backend_name="trtllm",
            model_config=None,
            afd_parallel_config_list=list(_AFD_CANDIDATES),
            gpus_per_node=8,
        )

        for name in self._POOL_ARGS:
            assert seen[name] is None, f"{name!r} should default to None, got {seen[name]!r}"
