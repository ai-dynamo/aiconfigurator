# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity for the ``compile_engine`` + ``EngineHandle`` path.

Two surfaces:

1. **Op-transfer round-trip fidelity.** ``compile_engine`` produces bincode
   bytes; ``EngineHandle`` consumes them and computes. We also inspect the
   intermediate ``EngineSpec`` JSON to confirm the op count and variant tags
   match the model's actual ``context_ops`` / ``generation_ops``. This is where
   a misnamed field or a wrong phase-pair tag would surface loudly.

2. **Integration parity.** For a smoke subset of the existing
   ``EngineStepParityCase``s — spanning all three backends (vllm + sglang +
   trtllm) so backend-specific op-transfer divergences (the ``MoEDispatch``
   flavor split, trtllm comm quant, the SGLang/TRT-LLM Fallback-MLA chain) are
   covered — compare the compiled-engine path against Python's ``BaseBackend``
   reference for static_ctx, static_gen, mixed-step, and decode-step within
   ``PARITY_RTOL``, exercising ``compile_engine`` / ``EngineHandle`` directly.

These tests require the maturin-built ``aiconfigurator_core`` extension.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json

import pytest

# Reuse the existing harness's case definitions and constants so this test
# tracks the same smoke matrix.
from test_engine_step_parity import PARITY_RTOL, SMOKE_CASES, EngineStepParityCase

from aiconfigurator.sdk import config, engine, perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.models import get_model

pytestmark = pytest.mark.integration


# The subset deliberately spans ALL THREE backends. The full
# `test_engine_step_parity.py` harness runs over vllm + sglang + trtllm;
# validating only vllm here would leave backend-specific op-transfer fidelity
# bugs (e.g. the `MoEDispatch` flavor split — trtllm emits `TrtllmAlltoall`,
# sglang/vllm emit `CustomAllReduce` — trtllm comm quant, and the
# SGLang/TRT-LLM-only Fallback-MLA chain) uncovered here.
#
# Cases drawn straight from `SMOKE_CASES` so this tracks the same matrix. All
# compute (no error-symmetry cases), so every surface yields a real number.
#
#   vllm   : the original 5 b200_sxm/vllm/0.19.0 cases.
#   sglang : Kimi-K2.5 (Fallback-MLA + MoE) and MiniMax-M2.5 (MoE), both
#            b200_sxm/sglang/0.5.10. SGLang's MoEDispatch flavor is the same
#            `CustomAllReduce` else-branch as vllm; its distinct value is the
#            Fallback-MLA path and the sglang perf tables.
#   trtllm : gpt-oss-20b (MoE -> exercises the `TrtllmAlltoall` flavor +
#            trtllm comm quant + trtllm MoE) and Nemotron-Super-49B (dense,
#            CustomAllReduce-heavy), both b200_sxm/trtllm/1.3.0rc10. The MoE
#            case is the load-bearing one: it is the only subset member that
#            hits the trtllm dispatch-flavor branch.
_SUBSET_IDS_BY_BACKEND = {
    "vllm": [
        "minimax-m25-b200-vllm-019-isl1024-osl2",
        "kimi-k25-b200-vllm-019-isl1024-osl2",
        "minimax-m25-b200-vllm-019-sampled-prefix",
        "minimax-m27-b200-vllm-019-isl1024-osl2",
        "qwen3-30b-a3b-b200-vllm-019-isl1024-osl2",
    ],
    "sglang": [
        "kimi-k25-b200-sglang-0510-isl1024-osl2",
        "minimax-m25-b200-sglang-0510-isl1024-osl2",
    ],
    "trtllm": [
        "gpt-oss-20b-b200-trtllm-130rc10-isl1024-osl2",
        "nemotron-nas-b200-trtllm-130rc10-isl1024-osl2",
    ],
}

# Preserve the per-backend ordering (vllm, then sglang, then trtllm) so the
# parametrize ids group readably and the determinism sweep covers vllm first.
_SUBSET_BY_ID = {p.id: p for p in SMOKE_CASES}
_declared_ids = [cid for ids in _SUBSET_IDS_BY_BACKEND.values() for cid in ids]
_missing_ids = [cid for cid in _declared_ids if cid not in _SUBSET_BY_ID]
if _missing_ids:
    raise AssertionError(f"subset declares case ids absent from SMOKE_CASES: {_missing_ids}")
_SUBSET_CASES = [_SUBSET_BY_ID[cid] for cid in _declared_ids]


# --------------------------------------------------------------------------- #
# Per-backend max-rtol collector. `_assert_within` only emits numbers on
# failure; reporting the observed worst-case drift per backend is useful for
# tracking parity. Each surface check records its observed rtol here;
# a session-scoped fixture prints the per-backend maxima at teardown (visible
# under `pytest -s` / `-rP`).
# --------------------------------------------------------------------------- #

_OBSERVED_RTOL: dict[str, float] = collections.defaultdict(float)


def _record_rtol(backend: str, observed_rtol: float) -> None:
    if observed_rtol > _OBSERVED_RTOL[backend]:
        _OBSERVED_RTOL[backend] = observed_rtol


@pytest.fixture(scope="session", autouse=True)
def _report_max_rtol():
    yield
    if not _OBSERVED_RTOL:
        return
    lines = ["", f"compile_engine pre-validation: max observed rtol per backend (tol={PARITY_RTOL * 100:.2f}%)"]
    for backend in sorted(_OBSERVED_RTOL):
        lines.append(f"  {backend:8s} max_rtol={_OBSERVED_RTOL[backend] * 100:.4f}%")
    print("\n".join(lines))


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return func(*args, **kwargs)


def _build_python_model(case: EngineStepParityCase):
    database = _quiet(perf_database.get_database, case.system_name, case.backend_name, case.backend_version)
    if database is None:
        pytest.skip(f"no perf database for {case.system_name}/{case.backend_name}/{case.backend_version}")
    model_config = config.ModelConfig(
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
    )
    model = _quiet(get_model, case.model_path, model_config, case.backend_name)
    backend = get_backend(case.backend_name)
    return model, backend, database


def _compile_handle(case: EngineStepParityCase) -> engine.EngineHandle:
    spec_bytes = _quiet(
        engine.compile_engine,
        case.model_path,
        case.system_name,
        case.backend_name,
        backend_version=case.backend_version,
        tp_size=case.tp_size,
        pp_size=case.pp_size,
        attention_dp_size=case.attention_dp_size,
        moe_tp_size=case.moe_tp_size,
        moe_ep_size=case.moe_ep_size,
    )
    return engine.EngineHandle(spec_bytes)


# --------------------------------------------------------------------------- #
# 1. Op-transfer round-trip fidelity.
# --------------------------------------------------------------------------- #


class TestOpTransferRoundTrip:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_op_count_and_tags_match_model(self, case: EngineStepParityCase) -> None:
        model, _backend, database = _build_python_model(case)

        spec_json = _quiet(
            engine.build_engine_spec_json,
            model,
            model_path=case.model_path,
            system=case.system_name,
            backend=case.backend_name,
            backend_version=case.backend_version,
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            nextn_accept_rates=None,
            database=database,
        )
        spec = json.loads(spec_json)

        # Vision is decomposed into encoder child ops; for these text-only
        # models `encoder_ops` is empty, so context_ops count is exact.
        encoder_ops = list(getattr(model, "encoder_ops", []) or [])
        expected_ctx = len(encoder_ops) + len(model.context_ops)
        expected_gen = len(model.generation_ops)

        assert len(spec["context_ops"]) == expected_ctx, (
            f"context op count mismatch: spec={len(spec['context_ops'])} model={expected_ctx}"
        )
        assert len(spec["generation_ops"]) == expected_gen, (
            f"generation op count mismatch: spec={len(spec['generation_ops'])} model={expected_gen}"
        )

        # Every emitted op is a single-key externally-tagged dict; no Vision tag
        # ever appears in a compiled spec.
        for op_dict in spec["context_ops"] + spec["generation_ops"]:
            assert isinstance(op_dict, dict) and len(op_dict) == 1, f"bad op shape: {op_dict}"
            (tag,) = op_dict.keys()
            assert tag != "Vision", "compiled spec must never contain a Vision op"

        # The op names round-trip through the wire in order: spec op name ==
        # the Python op `_name` for each list (after the encoder prefix).
        spec_ctx_names = [next(iter(d.values()))["name"] for d in spec["context_ops"]]
        py_ctx_names = [op._name for op in encoder_ops] + [op._name for op in model.context_ops]
        assert spec_ctx_names == py_ctx_names, "context op names/order drifted"

        spec_gen_names = [next(iter(d.values()))["name"] for d in spec["generation_ops"]]
        py_gen_names = [op._name for op in model.generation_ops]
        assert spec_gen_names == py_gen_names, "generation op names/order drifted"

    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_bincode_round_trip_runs(self, case: EngineStepParityCase) -> None:
        # compile -> bincode bytes -> AicEngine builds and computes a positive
        # static total. Proves the bytes decode in Rust (from_bincode) and the
        # op list is queryable end to end.
        handle = _compile_handle(case)
        ctx, gen, total = handle.run_static(batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2))
        assert total > 0.0 and ctx > 0.0 and gen > 0.0


# --------------------------------------------------------------------------- #
# 2. Integration pre-validation against Python BaseBackend.
# --------------------------------------------------------------------------- #


def _python_static(case: EngineStepParityCase, mode: str, stride: int) -> float:
    model, backend, database = _build_python_model(case)
    rc = config.RuntimeConfig(
        batch_size=case.batch_size,
        beam_width=1,
        isl=case.isl,
        osl=max(case.osl, 2),
        prefix=case.prefix,
    )
    (
        ctx_lat,
        _ctx_e,
        gen_lat,
        _gen_e,
        _ctx_s,
        _gen_s,
    ) = _quiet(backend._run_static_breakdown, model, database, rc, mode, stride)
    if mode == "static_ctx":
        return float(sum(ctx_lat.values()))
    if mode == "static_gen":
        return float(sum(gen_lat.values()))
    return float(sum(ctx_lat.values()) + sum(gen_lat.values()))


def _python_mixed(case: EngineStepParityCase) -> float:
    model, backend, database = _build_python_model(case)
    rc = config.RuntimeConfig(
        batch_size=case.batch_size, beam_width=1, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix
    )
    latency_ms, _, _, _ = _quiet(
        backend._get_mix_step_latency,
        model,
        database,
        rc,
        case.isl,  # ctx_tokens
        case.batch_size,  # gen_tokens
        case.isl,
        max(case.osl, 2),
        case.prefix,
    )
    return float(latency_ms)


def _python_decode(case: EngineStepParityCase) -> float:
    model, backend, database = _build_python_model(case)
    rc = config.RuntimeConfig(
        batch_size=case.batch_size, beam_width=1, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix
    )
    latency_ms, _, _, _ = _quiet(
        backend._get_genonly_step_latency,
        model,
        database,
        rc,
        case.batch_size,  # gen_tokens
        case.isl,
        max(case.osl, 2),
    )
    return float(latency_ms)


def _assert_within(name: str, python_value: float, new_value: float, *, backend: str) -> None:
    allowed = max(abs(python_value) * PARITY_RTOL, 1e-9)
    delta = new_value - python_value
    if python_value:
        observed_rtol = abs(delta) / abs(python_value)
    else:
        # Both sides zero is exact parity; only a nonzero delta against a zero
        # reference is undefined (treated as infinite drift).
        observed_rtol = 0.0 if delta == 0 else float("inf")
    _record_rtol(backend, observed_rtol)
    pct = observed_rtol * 100
    assert abs(delta) <= allowed, (
        f"[{backend}] {name} drift: python={python_value:.4f} new={new_value:.4f} "
        f"delta={delta:.4f} ({pct:.2f}%) tol={PARITY_RTOL * 100:.2f}%"
    )


class TestCompileEngineStaticParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_static_ctx_and_gen(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        osl = max(case.osl, 2)
        # stride=1 matches the existing harness's static comparison granularity.
        new_ctx, new_gen, new_total = handle.run_static(
            batch_size=case.batch_size, isl=case.isl, osl=osl, prefix=case.prefix, stride=1
        )
        py_ctx = _python_static(case, "static_ctx", 1)
        py_gen = _python_static(case, "static_gen", 1)
        _assert_within("static_ctx", py_ctx, new_ctx, backend=case.backend_name)
        _assert_within("static_gen", py_gen, new_gen, backend=case.backend_name)
        _assert_within("static_total", py_ctx + py_gen, new_total, backend=case.backend_name)


class TestCompileEngineMixedStepParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_mixed_step(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        new_val = handle.mixed_step_latency(case.isl, case.batch_size, case.isl, max(case.osl, 2), case.prefix)
        py_val = _python_mixed(case)
        _assert_within("mixed_step", py_val, new_val, backend=case.backend_name)

    @pytest.mark.parametrize(
        "ctx_tokens,gen_tokens,isl,osl,prefix",
        [
            (512, 4, 4096, 128, 0),  # chunked prefill: ctx_tokens < isl
            (512, 4, 4096, 128, 256),  # chunked + cached prefix
            (300, 7, 1000, 64, 100),  # ragged chunk + prefix + decode overlap
        ],
    )
    def test_mixed_step_chunked_prefill(self, ctx_tokens, gen_tokens, isl, osl, prefix) -> None:
        """Chunked prefill (ctx_tokens < isl) was the largest pre-rewrite
        composition gap: Python queries context attention at the FULL per-req
        isl then divides by ceil(isl/ctx), the old Rust queried the chunk
        directly. The rewritten three-pass mirror must match exactly."""
        case = _SUBSET_BY_ID["minimax-m25-b200-vllm-019-isl1024-osl2"].values[0]
        model, backend, database = _build_python_model(case)
        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=isl, osl=osl, prefix=prefix)
        py_val, _, _, _ = _quiet(
            backend._get_mix_step_latency, model, database, rc, ctx_tokens, gen_tokens, isl, osl, prefix
        )
        handle = _compile_handle(case)
        new_val = handle.mixed_step_latency(ctx_tokens, gen_tokens, isl, osl, prefix)
        _assert_within("mixed_step_chunked", float(py_val), new_val, backend=case.backend_name)


class TestCompileEngineDecodeStepParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_decode_step(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        new_val = handle.decode_step_latency(case.batch_size, case.isl, max(case.osl, 2))
        py_val = _python_decode(case)
        _assert_within("decode_step", py_val, new_val, backend=case.backend_name)


# --------------------------------------------------------------------------- #
# 2b. Imbalance-correction scale threading (session.rs used to hardcode 1.0).
# --------------------------------------------------------------------------- #


class TestImbalanceScaleParity:
    """Non-1.0 seq/gen imbalance-correction scales must produce identical
    Python and Rust numbers. Regression for the session.rs hardcode: the wire
    accepted the scales but every RuntimeContext pinned them to 1.0, so any
    task setting them diverged silently on the rust path."""

    _CASE_ID = "minimax-m25-b200-vllm-019-isl1024-osl2"
    _CTX_SCALE = 1.3
    _GEN_SCALE = 0.85

    def _case(self) -> EngineStepParityCase:
        return _SUBSET_BY_ID[self._CASE_ID].values[0]

    def test_static_scales_thread_through(self) -> None:
        case = self._case()
        model, backend, database = _build_python_model(case)
        rc = config.RuntimeConfig(
            batch_size=case.batch_size,
            beam_width=1,
            isl=case.isl,
            osl=max(case.osl, 2),
            prefix=case.prefix,
            seq_imbalance_correction_scale=self._CTX_SCALE,
            gen_seq_imbalance_correction_scale=self._GEN_SCALE,
        )
        ctx_lat, _, gen_lat, _, _, _ = _quiet(backend._run_static_breakdown, model, database, rc, "static", 1)
        py_ctx = float(sum(ctx_lat.values()))
        py_gen = float(sum(gen_lat.values()))

        handle = _compile_handle(case)
        new_ctx, new_gen, _ = handle.run_static(
            batch_size=case.batch_size,
            isl=case.isl,
            osl=max(case.osl, 2),
            prefix=case.prefix,
            seq_imbalance_correction_scale=self._CTX_SCALE,
            gen_seq_imbalance_correction_scale=self._GEN_SCALE,
            stride=1,
        )
        _assert_within("static_ctx@scale", py_ctx, new_ctx, backend=case.backend_name)
        _assert_within("static_gen@scale", py_gen, new_gen, backend=case.backend_name)

        # The scales must actually bite: a scaled run differs from unscaled.
        base_ctx, base_gen, _ = handle.run_static(
            batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix, stride=1
        )
        assert new_ctx != base_ctx, "ctx scale did not affect the rust static path"
        assert new_gen != base_gen, "gen scale did not affect the rust static path"

    def test_mixed_and_decode_scales_thread_through(self) -> None:
        case = self._case()
        model, backend, database = _build_python_model(case)
        rc = config.RuntimeConfig(
            batch_size=case.batch_size,
            beam_width=1,
            isl=case.isl,
            osl=max(case.osl, 2),
            prefix=case.prefix,
            seq_imbalance_correction_scale=self._CTX_SCALE,
            gen_seq_imbalance_correction_scale=self._GEN_SCALE,
        )
        py_mixed, _, _, _ = _quiet(
            backend._get_mix_step_latency,
            model,
            database,
            rc,
            case.isl,
            case.batch_size,
            case.isl,
            max(case.osl, 2),
            case.prefix,
        )
        py_decode, _, _, _ = _quiet(
            backend._get_genonly_step_latency,
            model,
            database,
            rc,
            case.batch_size,
            case.isl,
            max(case.osl, 2),
        )

        handle = _compile_handle(case)
        new_mixed = handle.mixed_step_latency(
            case.isl,
            case.batch_size,
            case.isl,
            max(case.osl, 2),
            case.prefix,
            seq_imbalance_correction_scale=self._CTX_SCALE,
            gen_seq_imbalance_correction_scale=self._GEN_SCALE,
        )
        new_decode = handle.decode_step_latency(
            case.batch_size,
            case.isl,
            max(case.osl, 2),
            gen_seq_imbalance_correction_scale=self._GEN_SCALE,
        )
        _assert_within("mixed_step@scale", float(py_mixed), new_mixed, backend=case.backend_name)
        _assert_within("decode_step@scale", float(py_decode), new_decode, backend=case.backend_name)


# --------------------------------------------------------------------------- #
# 2c. SGLang WideEP (deepep_moe) — MLA + MoE + DeepEP dispatch routing.
# --------------------------------------------------------------------------- #


class TestWideEpDeepEpParity:
    """SGLang WideEP DeepSeek (moe_backend=deepep_moe) end-to-end parity.

    Covers three previously-divergent surfaces at once: the WideEP MLA
    per-rank-heads table coordinate (tp=8 -> heads=16; the bridge used to emit
    raw tp), the deepep MoE compute routing (Rust used to read `moe_perf`
    where Python reads the wideep context/generation tables), and the DeepEP
    dispatch flavor emission (the emitter used to map every sglang dispatch to
    CustomAllReduce). Data lives on h200_sxm/sglang/0.5.6.post2 (the only
    shipped version with the deepep dispatch parquets)."""

    _MODEL = "deepseek-ai/DeepSeek-V3"
    _SYSTEM = "h200_sxm"
    _VERSION = "0.5.6.post2"

    def _build(self):
        from aiconfigurator.sdk import common

        database = _quiet(perf_database.get_database, self._SYSTEM, "sglang", self._VERSION)
        if database is None:
            pytest.skip(f"no perf database for {self._SYSTEM}/sglang/{self._VERSION}")
        model_config = config.ModelConfig(
            tp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            moe_backend="deepep_moe",
            attention_backend="flashinfer",
            gemm_quant_mode=common.GEMMQuantMode.fp8_block,
            moe_quant_mode=common.MoEQuantMode.fp8_block,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.fp8_block,
        )
        model = _quiet(get_model, self._MODEL, model_config, "sglang")
        backend = get_backend("sglang")
        spec_json = _quiet(
            engine.build_engine_spec_json,
            model,
            model_path=self._MODEL,
            system=self._SYSTEM,
            backend="sglang",
            backend_version=self._VERSION,
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            nextn_accept_rates=None,
            database=database,
        )
        import aiconfigurator_core

        handle = engine.EngineHandle(bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json)))
        return model, backend, database, handle

    def test_wideep_static_parity(self) -> None:
        model, backend, database, handle = self._build()
        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=1024, osl=4, prefix=0)
        ctx_lat, _, gen_lat, _, _, _ = _quiet(backend._run_static_breakdown, model, database, rc, "static", 1)
        py_ctx, py_gen = float(sum(ctx_lat.values())), float(sum(gen_lat.values()))
        new_ctx, new_gen, _ = handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
        _assert_within("wideep_static_ctx", py_ctx, new_ctx, backend="sglang")
        _assert_within("wideep_static_gen", py_gen, new_gen, backend="sglang")

    def test_wideep_mixed_and_decode_parity(self) -> None:
        model, backend, database, handle = self._build()
        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=1024, osl=4, prefix=0)
        py_mixed, _, _, _ = _quiet(backend._get_mix_step_latency, model, database, rc, 1024, 2, 1024, 4, 0)
        py_decode, _, _, _ = _quiet(backend._get_genonly_step_latency, model, database, rc, 2, 1024, 4)
        new_mixed = handle.mixed_step_latency(1024, 2, 1024, 4, 0)
        new_decode = handle.decode_step_latency(2, 1024, 4)
        _assert_within("wideep_mixed", float(py_mixed), new_mixed, backend="sglang")
        _assert_within("wideep_decode", float(py_decode), new_decode, backend="sglang")


# --------------------------------------------------------------------------- #
# 2d. TRT-LLM WideEP (NVLink Two-Sided alltoall) — gb200.
# --------------------------------------------------------------------------- #


class TestTrtllmWideEpParity:
    """TRT-LLM WideEP DeepSeek (enable_wideep, attention_dp=8) on gb200.

    Covers the `TrtLLMWideEPMoEDispatch` port (prepare+dispatch pre /
    combine post through the trtllm_alltoall table, kernel auto-selected as
    NVLinkTwoSided via moe_backend="wideep") and the alltoall loader keying
    (kernel_source/op_name/num_nodes — the pre-fix loader collapsed 1,556 of
    2,096 gb200 rows). This path used to fail opspec conversion entirely
    (`TrtLLMWideEPMoEDispatch` had no `_to_opspec` branch)."""

    def _build(self):
        from aiconfigurator.sdk import common

        database = _quiet(perf_database.get_database, "gb200", "trtllm", "1.3.0rc10")
        if database is None:
            pytest.skip("no perf database for gb200/trtllm/1.3.0rc10")
        model_config = config.ModelConfig(
            tp_size=1,
            attention_dp_size=8,
            moe_tp_size=1,
            moe_ep_size=8,
            enable_wideep=True,
            gemm_quant_mode=common.GEMMQuantMode.nvfp4,
            moe_quant_mode=common.MoEQuantMode.nvfp4,
            kvcache_quant_mode=common.KVCacheQuantMode.fp8,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
        )
        model = _quiet(get_model, "deepseek-ai/DeepSeek-V3", model_config, "trtllm")
        backend = get_backend("trtllm")
        spec_json = _quiet(
            engine.build_engine_spec_json,
            model,
            model_path="deepseek-ai/DeepSeek-V3",
            system="gb200",
            backend="trtllm",
            backend_version="1.3.0rc10",
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            nextn_accept_rates=None,
            database=database,
        )
        import aiconfigurator_core

        handle = engine.EngineHandle(bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json)))
        return model, backend, database, handle

    def test_trtllm_wideep_static_parity(self) -> None:
        model, backend, database, handle = self._build()
        rc = config.RuntimeConfig(batch_size=1, beam_width=1, isl=1024, osl=4, prefix=0)
        ctx_lat, _, gen_lat, _, _, _ = _quiet(backend._run_static_breakdown, model, database, rc, "static", 1)
        py_ctx, py_gen = float(sum(ctx_lat.values())), float(sum(gen_lat.values()))
        new_ctx, new_gen, _ = handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
        _assert_within("trtllm_wideep_static_ctx", py_ctx, new_ctx, backend="trtllm")
        _assert_within("trtllm_wideep_static_gen", py_gen, new_gen, backend="trtllm")


# --------------------------------------------------------------------------- #
# 3. Determinism across rayon thread counts.
# --------------------------------------------------------------------------- #


class TestDeterminism:
    @pytest.mark.parametrize("case", _SUBSET_CASES[:2])
    def test_run_static_deterministic(self, case: EngineStepParityCase) -> None:
        # The actual RAYON_NUM_THREADS sweep is driven by the test runner (run
        # this file with =1 and =8). Within a single process we still assert
        # repeated calls are bit-identical (pure per-call execution, no
        # cross-call state).
        handle = _compile_handle(case)
        a = handle.run_static(batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), stride=1)
        b = handle.run_static(batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), stride=1)
        assert a == b
