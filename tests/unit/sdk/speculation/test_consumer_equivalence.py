# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-contract tests for the compile-time materialization design.

The compiled engine is the only step executor, so a scheme reaches the
engine exclusively through what ``materialize_spec_scheme`` folds into the
model at ``get_model`` time. Three guarantees are pinned here:

1. Golden equivalence — legacy ``nextn`` construction and an explicit mtp
   SpeculationConfig produce models with identical engine-relevant state
   (same ``_nextn``, same engine identity), so MTP predictions stay
   bit-identical.
2. Materialization — a scheme with its own verify width / draft ops / bytes
   lands in the op lists (``draft_`` prefix, ``scale_num_tokens`` width
   folding, ``_nextn`` width channel) without mutating scheme-owned ops,
   and idempotently.
3. Memory accounting + engine routing — draft weights/KV enter
   ``_get_memory_usage`` through the scheme hooks, and the engine identity
   is keyed by the speculation config so same-width schemes never share a
   cached engine handle.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from aiconfigurator_core.sdk import common
from aiconfigurator_core.sdk.backends.base_backend import BaseBackend
from aiconfigurator_core.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator_core.sdk.rust_engine_step import _engine_config_json, should_use_rust_engine_step
from aiconfigurator_core.sdk.speculation import DraftOpSpec, NullScheme, SpeculationConfig
from aiconfigurator_core.sdk.speculation.materialize import materialize_spec_scheme
from aiconfigurator_core.sdk.speculation.mtp import MTPScheme

pytestmark = pytest.mark.unit


class _RecordingOp:
    """Fake op carrying the attributes materialization touches."""

    def __init__(self, name: str, latency_ms: float = 1.0) -> None:
        self._name = name
        self._latency_ms = latency_ms
        self._scale_num_tokens = 1
        self._scale_factor = 1.0

    def query(self, *args, **kwargs) -> float:
        return self._latency_ms

    def get_weights(self) -> float:
        return 0.0


class _MemoryBackend(BaseBackend):
    """Memory-path backend: uses the REAL BaseBackend._get_memory_usage."""

    def find_best_agg_result_under_constraints(self, model, database, runtime_config, **kwargs):
        raise NotImplementedError


def _model_config(speculation=None) -> ModelConfig:
    return ModelConfig(
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        moe_quant_mode=common.MoEQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        comm_quant_mode=common.CommQuantMode.half,
        speculation=speculation,
    )


def _fake_model(nextn: int = 0, spec_scheme=None, speculation=None):
    model = SimpleNamespace()
    model.model_path = "test-model"
    model._nextn = nextn
    model.encoder_ops = []
    model.context_ops = [_RecordingOp("context_attention", 11.0)]
    model.generation_ops = [_RecordingOp("generation_attention", 2.0)]
    model.config = _model_config(speculation=speculation)
    model.config.nextn = nextn
    if spec_scheme is None:
        spec_scheme = NullScheme() if nextn == 0 else MTPScheme(depth=nextn)
    model.spec_scheme = spec_scheme
    return model


def _database():
    return SimpleNamespace(
        backend="test-backend",
        version="test-version",
        system="test-system",
        system_spec={"gpu": {"mem_capacity": 80 * (1 << 30)}},
    )


class _FakeDraftScheme(NullScheme):
    """Verify width 6; one gen draft op at 5 tokens/request (non-divisible —
    priced at full width) and one at 3 (divisible — folds x2); one context
    draft op; nonzero weight/KV bytes."""

    kind = "fake_draft"

    def __init__(self) -> None:
        self.gen_op = _RecordingOp("dspark_backbone", 3.0)
        self.gen_op_divisible = _RecordingOp("dspark_head", 1.0)
        self.ctx_op = _RecordingOp("dspark_precompute", 5.0)

    def verify_width(self) -> int:
        return 6

    def build_draft_generation_ops(self, model):
        return [
            DraftOpSpec(op=self.gen_op, tokens_per_request=5, query_overrides={"s": 133}),
            DraftOpSpec(op=self.gen_op_divisible, tokens_per_request=3),
        ]

    def build_draft_context_ops(self, model):
        return [DraftOpSpec(op=self.ctx_op, tokens_per_request=1)]

    def draft_weights_bytes(self, model) -> float:
        return 4.0 * (1 << 30)

    def draft_kv_bytes_per_sequence(self, model, seq_len: int) -> float:
        return 1024.0

    def verify_attention_sequence_basis(self) -> bool:
        return True  # block verify shares one KV pass (all real draft schemes)


class TestGoldenEquivalence:
    def test_legacy_and_explicit_mtp_share_engine_state(self):
        legacy = _fake_model(nextn=2)  # spec_scheme derived: MTPScheme(2)
        explicit = _fake_model(
            nextn=2,
            spec_scheme=MTPScheme(depth=2),
            speculation=SpeculationConfig(kind="mtp", params={"depth": 2}),
        )

        # Materialization must be a no-op for MTP: the legacy nextn contract
        # (model families + engine nextn scalar) stays authoritative.
        for model in (legacy, explicit):
            materialize_spec_scheme(model)
            assert model._nextn == 2
            assert [op._name for op in model.generation_ops] == ["generation_attention"]
            assert not getattr(model, "_spec_scheme_materialized", False)

        # Same engine identity: an explicit mtp SpeculationConfig rides the
        # nextn key, never the speculation content key.
        db = _database()
        assert _engine_config_json(legacy, db) == _engine_config_json(explicit, db)


class TestMaterialization:
    def test_width_channel_and_draft_ops_land_in_op_lists(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        # verify width 6 -> engine decode-batch multiplier (_nextn + 1) = 6.
        assert model._nextn == 5

        gen_names = [op._name for op in model.generation_ops]
        assert gen_names == ["generation_attention", "draft_dspark_backbone", "draft_dspark_head"]
        ctx_names = [op._name for op in model.context_ops]
        assert ctx_names == ["context_attention", "draft_dspark_precompute"]

        by_name = {op._name: op for op in model.generation_ops}
        # 6 % 5 != 0: result-scaled by 5/6 (token-linear identity), query
        # stays at the full phase width.
        assert by_name["draft_dspark_backbone"]._scale_num_tokens == 1
        assert by_name["draft_dspark_backbone"]._scale_factor == pytest.approx(5 / 6)
        # 6 // 3 == 2: the divisor folds the op back to its drafted token count.
        assert by_name["draft_dspark_head"]._scale_num_tokens == 2

    def test_scheme_owned_ops_are_not_mutated(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        # The scheme's cached ops keep their identity: schemes that cache a
        # built draft model stay re-inspectable after materialization.
        assert scheme.gen_op._name == "dspark_backbone"
        assert scheme.gen_op_divisible._scale_num_tokens == 1
        assert scheme.ctx_op._name == "dspark_precompute"
        assert model.generation_ops[1] is not scheme.gen_op

    def test_materialization_is_idempotent(self):
        model = _fake_model(nextn=0, spec_scheme=_FakeDraftScheme())
        materialize_spec_scheme(model)
        once = [op._name for op in model.generation_ops]
        materialize_spec_scheme(model)
        assert [op._name for op in model.generation_ops] == once


class TestAttentionWidthChannel:
    """Sequence-basis fold + roofline guard on dense decode attention."""

    @staticmethod
    def _gen_attention(name="generation_attention"):
        from aiconfigurator_core.sdk.operations.attention import GenerationAttention

        return GenerationAttention(name, 1.0, n=32, n_kv=8, kv_cache_dtype=common.KVCacheQuantMode.bfloat16)

    def test_target_verify_attention_folds_to_sequence_basis(self):
        scheme = _FakeDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        target_attn = self._gen_attention()
        model.generation_ops = [target_attn]
        materialize_spec_scheme(model)

        # verify width 6: batch divisor 6, real query width 6 for the guard.
        assert target_attn._scale_num_tokens == 6
        assert target_attn._verify_query_tokens == 6

    def test_draft_block_attention_folds_full_width(self):
        class _BlockDraftScheme(_FakeDraftScheme):
            def __init__(self) -> None:
                super().__init__()
                self.attn_op = TestAttentionWidthChannel._gen_attention("dspark_attention")

            def build_draft_generation_ops(self, model):
                # 5 drafted tokens per request inside a width-6 phase: gemms
                # would refuse the non-integer fold, but attention folds by
                # the FULL width (one KV read per request) and carries the
                # real query width for the guard.
                return [DraftOpSpec(op=self.attn_op, tokens_per_request=5)]

        scheme = _BlockDraftScheme()
        model = _fake_model(nextn=0, spec_scheme=scheme)
        materialize_spec_scheme(model)

        draft_attn = next(op for op in model.generation_ops if op._name == "draft_dspark_attention")
        assert draft_attn._scale_num_tokens == 6
        assert draft_attn._verify_query_tokens == 5
        # scheme-owned op untouched (copy semantics)
        assert scheme.attn_op._scale_num_tokens == 1

    # Query-time behavior (sequence-basis batch fold, roofline guard) is
    # priced only by the Rust oracle — anchored by the
    # `generation_attention_*` tests in `operators/attention.rs` (single-oracle
    # rule: Python carries the fields onto the wire; it never prices them).

    def test_engine_spec_carries_the_width_fields(self):
        import json

        op = self._gen_attention()
        wire = json.loads(op._spec_json())["GenerationAttention"]
        assert wire["scale_num_tokens"] == 1
        assert wire["verify_query_tokens"] == 0
        op._scale_num_tokens = 6
        op._verify_query_tokens = 6
        wire = json.loads(op._spec_json())["GenerationAttention"]
        assert wire["scale_num_tokens"] == 6
        assert wire["verify_query_tokens"] == 6

    def test_copy_preserves_the_width_fields(self):
        # materialize copy.copy()s scheme-owned ops before folding; the
        # pyo3 pickle protocol must round-trip the width channel.
        import copy

        op = self._gen_attention()
        op._scale_num_tokens = 6
        op._verify_query_tokens = 5
        dup = copy.copy(op)
        assert dup._scale_num_tokens == 6
        assert dup._verify_query_tokens == 5


class TestMemoryAndRouting:
    def test_memory_accounting_includes_draft_bytes(self):
        base = _fake_model(nextn=0)
        drafted = _fake_model(nextn=0, spec_scheme=_FakeDraftScheme())
        materialize_spec_scheme(drafted)
        database = _database()
        database.system_spec["misc"] = {"nccl_mem": {1: 0.0}, "other_mem": 0.0}

        backend = _MemoryBackend()
        kwargs = dict(batch_size=4, beam_width=1, isl=512, osl=64)
        for model in (base, drafted):
            model._num_heads = 8
            model._head_size = 128
            model._num_experts = 0
            model.model_family = "GPT"
            model.get_kvcache_bytes_per_sequence = lambda seq_len: 2048.0
            model._cp_kv_memory_divisor = lambda: 1

        m_base = backend._get_memory_usage(base, database, **kwargs)
        m_draft = backend._get_memory_usage(drafted, database, **kwargs)

        one_gib = 1 << 30
        # weights: +4 GiB from the scheme (materialized draft_ ops excluded
        # from the op-list sum; the scheme hook is the single source of truth)
        assert m_draft["weights"] - m_base["weights"] == pytest.approx(4.0, rel=1e-6)
        # kv: +batch * 1024 bytes
        assert (m_draft["kvcache"] - m_base["kvcache"]) * one_gib == pytest.approx(4 * 1024.0, rel=1e-6)
        # activations: the materialized _nextn drives the (nextn+1) width factor
        assert m_draft["activations"] == pytest.approx(m_base["activations"] * 6, rel=1e-6)

    def test_engine_identity_keyed_by_speculation_config(self):
        db = _database()
        plain = _fake_model(nextn=0)
        eagle = _fake_model(
            nextn=0,
            spec_scheme=NullScheme(),  # identity comes from the config, not the scheme object
            speculation=SpeculationConfig(kind="eagle3", params={"num_speculative_tokens": 5}),
        )
        # Same widths, different speculation content -> different engines.
        assert _engine_config_json(plain, db) != _engine_config_json(eagle, db)

        # The key is the config's content hash, so equal configs share.
        eagle2 = _fake_model(
            nextn=0,
            spec_scheme=NullScheme(),
            speculation=SpeculationConfig(kind="eagle3", params={"num_speculative_tokens": 5}),
        )
        assert _engine_config_json(eagle, db) == _engine_config_json(eagle2, db)
        payload = json.loads(_engine_config_json(eagle, db))
        assert payload["speculation"] is not None
        assert json.loads(_engine_config_json(plain, db))["speculation"] is None

    def test_rust_engine_routing_has_no_scheme_bypass(self):
        # Explicit "rust" routes to the compiled engine regardless of scheme;
        # the one remaining delegation is the synthetic-database default.
        explicit = RuntimeConfig(engine_step_backend="rust")
        default = RuntimeConfig()
        synthetic_db = _database()
        assert should_use_rust_engine_step(explicit, synthetic_db) is True
        assert should_use_rust_engine_step(default, synthetic_db) is False
