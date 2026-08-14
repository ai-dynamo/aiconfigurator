# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity for the ``compile_engine`` + ``EngineHandle`` path.

Three surfaces:

1. **Op-transfer round-trip fidelity.** ``compile_engine`` produces bincode
   bytes; ``EngineHandle`` consumes them and computes. We also inspect the
   intermediate ``EngineSpec`` JSON to confirm the op count and variant tags
   match the model's actual ``context_ops`` / ``generation_ops``. This is where
   a misnamed field or a wrong phase-pair tag would surface loudly.

2. **Integration parity vs golden fixtures.** For a smoke subset of the
   existing ``EngineStepParityCase``s — spanning all three backends (vllm +
   sglang + trtllm) so backend-specific op-transfer divergences (the
   ``MoEDispatch`` flavor split, trtllm comm quant, the SGLang/TRT-LLM
   Fallback-MLA chain) are covered — compare the compiled-engine path against
   the FROZEN Python ``BaseBackend`` reference in
   ``goldens/compile_engine.json`` (captured by the retired
   ``regenerate_goldens.py`` while the Python step path was alive; new
   records are pinned from the live rust engine by ``pin_goldens.py``) for
   static_ctx, static_gen, mixed-step, and decode-step within
   ``PARITY_RTOL``.

3. **Per-op FFI anchor.** ``EngineHandle.run_static_per_op`` folded by name
   must reproduce the frozen Python summary per-op dicts (latency + energy +
   source) from ``goldens/per_op.json`` — the Gate-3 precondition that per-op
   values cross the FFI with real op names.

These tests require the maturin-built ``aiconfigurator_core`` extension.
"""

from __future__ import annotations

import collections
import contextlib
import io
import json

import pytest

# Reuse the existing harness's case definitions, constants, and golden loader
# so this test tracks the same smoke matrix and fixture workflow.
from test_engine_step_parity import (
    _REGENERATE_HINT,
    PARITY_RTOL,
    POWER_CASES,
    SMOKE_CASES,
    EngineStepParityCase,
    load_parity_golden,
)

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
# Cases drawn straight from `SMOKE_CASES` (plus one `POWER_CASES` member) so
# this tracks the same matrix. All compute (no error-symmetry cases), so every
# surface yields a real number.
#
#   vllm   : the original 5 b200_sxm/vllm/0.19.0 cases, plus the
#            Qwen3-30B-A3B b200_sxm/vllm/0.22.0 POWER_CASES member — every
#            0.19.0/0.5.x/1.3.0rc10 identity is latency-only (energy_wms == 0
#            in every per-op golden), so this is the one subset case whose
#            per-op energy comparison actually executes.
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
        "qwen3-30b-a3b-b200-vllm-022-power",
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

# Subset members on power-carrying database identities: their per-op goldens
# must carry nonzero energy_wms, so the energy comparison branch is proven to
# execute (see the anti-vacuous guard in TestCompileEnginePerOpParity).
_POWER_SUBSET_IDS = {"qwen3-30b-a3b-b200-vllm-022-power"}

# Preserve the per-backend ordering (vllm, then sglang, then trtllm) so the
# parametrize ids group readably and the determinism sweep covers vllm first.
_SUBSET_BY_ID = {p.id: p for p in [*SMOKE_CASES, *POWER_CASES]}
_declared_ids = [cid for ids in _SUBSET_IDS_BY_BACKEND.values() for cid in ids]
_missing_ids = [cid for cid in _declared_ids if cid not in _SUBSET_BY_ID]
if _missing_ids:
    raise AssertionError(f"subset declares case ids absent from SMOKE_CASES: {_missing_ids}")
_SUBSET_CASES = [_SUBSET_BY_ID[cid] for cid in _declared_ids]

# Golden records key on the pytest param id; recover it from the (frozen,
# hashable) case value the tests are parametrized with.
_SUBSET_CASE_IDS = {p.values[0]: p.id for p in _SUBSET_CASES}


def _golden_reference(key: str) -> float:
    """One frozen Python reference value from ``goldens/compile_engine.json``."""
    references = load_parity_golden("compile_engine.json")["references"]
    value = references.get(key)
    if value is None:
        pytest.fail(f"no compile-engine golden reference '{key}'; {_REGENERATE_HINT}", pytrace=False)
    return float(value)


def _case_reference(case: EngineStepParityCase, metric: str) -> float:
    return _golden_reference(f"{_SUBSET_CASE_IDS[case]}::{metric}")


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
# 2. Integration parity against the frozen Python BaseBackend reference.
#
# The Python side of every reference is a FROZEN golden value in
# `goldens/compile_engine.json`, captured by the retired
# `regenerate_goldens.py` while the Python step path was still alive. Only
# the compiled-engine side runs live; `pin_goldens.py` appends records for
# new cases (pinned from the live rust engine, provenance-marked).
# --------------------------------------------------------------------------- #


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


# Chunked-prefill shapes: shared by the parametrized test and the golden
# pin path (pin_goldens.py) so the fixture keys track the test matrix.
_CHUNKED_PREFILL_CASE_ID = "minimax-m25-b200-vllm-019-isl1024-osl2"
_CHUNKED_PREFILL_SHAPES = [
    (512, 4, 4096, 128, 0),  # chunked prefill: ctx_tokens < isl
    (512, 4, 4096, 128, 256),  # chunked + cached prefix
    (300, 7, 1000, 64, 100),  # ragged chunk + prefix + decode overlap
]


def _chunked_prefill_key(ctx_tokens: int, gen_tokens: int, isl: int, osl: int, prefix: int) -> str:
    return f"chunked_prefill::ctx{ctx_tokens}_gen{gen_tokens}_isl{isl}_osl{osl}_prefix{prefix}::mixed_step"


class TestCompileEngineStaticParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_static_ctx_and_gen(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        osl = max(case.osl, 2)
        # stride=1 matches the existing harness's static comparison granularity.
        new_ctx, new_gen, new_total = handle.run_static(
            batch_size=case.batch_size, isl=case.isl, osl=osl, prefix=case.prefix, stride=1
        )
        py_ctx = _case_reference(case, "static_ctx")
        py_gen = _case_reference(case, "static_gen")
        _assert_within("static_ctx", py_ctx, new_ctx, backend=case.backend_name)
        _assert_within("static_gen", py_gen, new_gen, backend=case.backend_name)
        _assert_within("static_total", py_ctx + py_gen, new_total, backend=case.backend_name)


class TestCompileEngineMixedStepParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_mixed_step(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        new_val = handle.mixed_step_latency(case.isl, case.batch_size, case.isl, max(case.osl, 2), case.prefix)
        py_val = _case_reference(case, "mixed_step")
        _assert_within("mixed_step", py_val, new_val, backend=case.backend_name)

    @pytest.mark.parametrize("ctx_tokens,gen_tokens,isl,osl,prefix", _CHUNKED_PREFILL_SHAPES)
    def test_mixed_step_chunked_prefill(self, ctx_tokens, gen_tokens, isl, osl, prefix) -> None:
        """Chunked prefill (ctx_tokens < isl) was the largest pre-rewrite
        composition gap: Python queries context attention at the FULL per-req
        isl then divides by ceil(isl/ctx), the old Rust queried the chunk
        directly. The rewritten three-pass mirror must match exactly."""
        case = _SUBSET_BY_ID[_CHUNKED_PREFILL_CASE_ID].values[0]
        handle = _compile_handle(case)
        new_val = handle.mixed_step_latency(ctx_tokens, gen_tokens, isl, osl, prefix)
        py_val = _golden_reference(_chunked_prefill_key(ctx_tokens, gen_tokens, isl, osl, prefix))
        _assert_within("mixed_step_chunked", py_val, new_val, backend=case.backend_name)


class TestCompileEngineDecodeStepParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_decode_step(self, case: EngineStepParityCase) -> None:
        handle = _compile_handle(case)
        new_val = handle.decode_step_latency(case.batch_size, case.isl, max(case.osl, 2))
        py_val = _case_reference(case, "decode_step")
        _assert_within("decode_step", py_val, new_val, backend=case.backend_name)


# --------------------------------------------------------------------------- #
# 2a. Per-op FFI anchor (Gate-3 precondition: per-op values cross the op-list
# FFI with real op names, latencies, energies, and source tags — not the
# synthetic `rust_engine_step_*` collapse).
# --------------------------------------------------------------------------- #


def _fold_per_op_entries(entries) -> dict[str, tuple[float, float, str]]:
    """Fold the FFI's ``(name, latency_ms, energy_wms, source)`` tuples by
    name (``+=``), mirroring ``rust_engine_step._fold_per_op``: duplicate
    names accumulate, sources merge to ``"mixed"`` on mismatch."""
    folded: dict[str, list] = {}
    for name, latency_ms, energy_wms, source in entries:
        record = folded.get(name)
        if record is None:
            folded[name] = [float(latency_ms), float(energy_wms), str(source)]
        else:
            record[0] += float(latency_ms)
            record[1] += float(energy_wms)
            if record[2] != source:
                record[2] = "mixed"
    return {name: (record[0], record[1], record[2]) for name, record in folded.items()}


# Known per-op source-TAG divergences between the frozen Python summary dicts
# and the compiled engine's per-op leaves. Latency and energy match bit-exact
# on every one of these (verified across the full 10-case subset); only the
# provenance label differs, and each pattern is pinned as an exact
# (op_name, golden_tag, rust_tag) triple so any NEW divergence — or a change
# to these — still fails:
#  - `*_p2p` at pp=1: Python's `P2P.query()` answers the zero-transfer case
#    through the comm formula path and tags "empirical"; the compiled engine
#    emits the zero-latency leaf with its default "silicon" tag.
#  - `context_attention` (non-MLA families): the Python module query merges
#    multi-component provenance internally and returns source="mixed"; the
#    compiled engine reports the fused attention leaf's own "silicon" tag.
_ACCEPTED_SOURCE_TAG_DIVERGENCES = {
    ("context_p2p", "empirical", "silicon"),
    ("generation_p2p", "empirical", "silicon"),
}


class TestCompileEnginePerOpParity:
    @pytest.mark.parametrize("case", _SUBSET_CASES)
    def test_static_per_op_matches_golden(self, case: EngineStepParityCase) -> None:
        case_id = _SUBSET_CASE_IDS[case]
        golden = load_parity_golden("per_op.json")["cases"].get(case_id)
        if golden is None:
            pytest.fail(f"no per-op golden for case '{case_id}'; {_REGENERATE_HINT}", pytrace=False)

        handle = _compile_handle(case)
        ctx_entries, gen_entries = handle.run_static_per_op(
            batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix, stride=1
        )
        energy_compared = False
        for phase, entries in (("context", ctx_entries), ("generation", gen_entries)):
            folded = _fold_per_op_entries(entries)
            expected = golden[phase]
            # The keysets must match EXACTLY — a golden op the rust fold lacks
            # (or vice versa) is a lost/renamed op, not a tolerable gap.
            assert set(folded) == set(expected), (
                f"[{case.backend_name}] {phase} per-op keyset drift: "
                f"golden_only={sorted(set(expected) - set(folded))} "
                f"rust_only={sorted(set(folded) - set(expected))}"
            )
            for name in sorted(expected):
                exp = expected[name]
                latency, energy, source = folded[name]
                _assert_within(
                    f"{phase}::{name}::latency", float(exp["latency_ms"]), latency, backend=case.backend_name
                )
                golden_energy = float(exp["energy_wms"])
                if golden_energy > 0.0:
                    _assert_within(f"{phase}::{name}::energy", golden_energy, energy, backend=case.backend_name)
                    energy_compared = True
                else:
                    assert energy == 0.0, (
                        f"[{case.backend_name}] {phase}::{name} golden energy is 0 but the rust fold produced {energy}"
                    )
                golden_source = str(exp["source"])
                if source != golden_source and (name, golden_source, source) not in _ACCEPTED_SOURCE_TAG_DIVERGENCES:
                    pytest.fail(
                        f"[{case.backend_name}] {phase}::{name} source tag drift: "
                        f"golden={golden_source!r} rust={source!r} "
                        f"(not in _ACCEPTED_SOURCE_TAG_DIVERGENCES)"
                    )
        # Anti-vacuous guard: the power-identity subset member exists so the
        # energy comparison above actually executes (every other subset case
        # sits on a latency-only identity where all golden energy_wms are 0).
        if case_id in _POWER_SUBSET_IDS:
            assert energy_compared, (
                f"{case_id} sits on a power-carrying identity but no golden op carried "
                "energy_wms > 0 — the per-op energy comparison never executed"
            )


# --------------------------------------------------------------------------- #
# 2b. Imbalance-correction scale threading (session.rs used to hardcode 1.0).
# --------------------------------------------------------------------------- #


# Shared by the imbalance tests and the golden pin path.
_IMBALANCE_CASE_ID = "minimax-m25-b200-vllm-019-isl1024-osl2"
_IMBALANCE_CTX_SCALE = 1.3
_IMBALANCE_GEN_SCALE = 0.85


class TestImbalanceScaleParity:
    """Non-1.0 seq/gen imbalance-correction scales must reproduce the frozen
    Python numbers. Regression for the session.rs hardcode: the wire accepted
    the scales but every RuntimeContext pinned them to 1.0, so any task
    setting them diverged silently on the rust path."""

    def _case(self) -> EngineStepParityCase:
        return _SUBSET_BY_ID[_IMBALANCE_CASE_ID].values[0]

    def test_static_scales_thread_through(self) -> None:
        case = self._case()
        handle = _compile_handle(case)
        new_ctx, new_gen, _ = handle.run_static(
            batch_size=case.batch_size,
            isl=case.isl,
            osl=max(case.osl, 2),
            prefix=case.prefix,
            seq_imbalance_correction_scale=_IMBALANCE_CTX_SCALE,
            gen_seq_imbalance_correction_scale=_IMBALANCE_GEN_SCALE,
            stride=1,
        )
        _assert_within(
            "static_ctx@scale", _golden_reference("imbalance_scale::static_ctx"), new_ctx, backend=case.backend_name
        )
        _assert_within(
            "static_gen@scale", _golden_reference("imbalance_scale::static_gen"), new_gen, backend=case.backend_name
        )

        # The scales must actually bite: a scaled run differs from unscaled.
        base_ctx, base_gen, _ = handle.run_static(
            batch_size=case.batch_size, isl=case.isl, osl=max(case.osl, 2), prefix=case.prefix, stride=1
        )
        assert new_ctx != base_ctx, "ctx scale did not affect the rust static path"
        assert new_gen != base_gen, "gen scale did not affect the rust static path"

    def test_mixed_and_decode_scales_thread_through(self) -> None:
        case = self._case()
        handle = _compile_handle(case)
        new_mixed = handle.mixed_step_latency(
            case.isl,
            case.batch_size,
            case.isl,
            max(case.osl, 2),
            case.prefix,
            seq_imbalance_correction_scale=_IMBALANCE_CTX_SCALE,
            gen_seq_imbalance_correction_scale=_IMBALANCE_GEN_SCALE,
        )
        new_decode = handle.decode_step_latency(
            case.batch_size,
            case.isl,
            max(case.osl, 2),
            gen_seq_imbalance_correction_scale=_IMBALANCE_GEN_SCALE,
        )
        _assert_within(
            "mixed_step@scale", _golden_reference("imbalance_scale::mixed_step"), new_mixed, backend=case.backend_name
        )
        _assert_within(
            "decode_step@scale",
            _golden_reference("imbalance_scale::decode_step"),
            new_decode,
            backend=case.backend_name,
        )


# --------------------------------------------------------------------------- #
# 2c. Large-EP (ex-WideEP) — MLA + EP MoE + all-to-all dispatch routing.
#
# The large-EP ops compile natively as of AIC-1601 (emission gated by
# `test_large_ep_op_graph_compiles_natively` in test_rust_engine_step.py).
# Both classes below exercise the post-deprecation internal contract
# (`ModelConfig.moe_comm_backend` per phase + the system's `num_gpus_per_node`)
# end-to-end through a native EngineHandle.
# --------------------------------------------------------------------------- #


_WIDEEP_SGLANG_MODEL = "deepseek-ai/DeepSeek-V3"
_WIDEEP_SGLANG_SYSTEM = "h200_sxm"
_WIDEEP_SGLANG_VERSION = "0.5.6.post2"


def _build_wideep_sglang():
    """(model, backend, database, spec_json) for the SGLang WideEP config;
    shared by the parity tests (handle side) and the golden capture (python
    references)."""
    from aiconfigurator.sdk import common

    database = _quiet(perf_database.get_database, _WIDEEP_SGLANG_SYSTEM, "sglang", _WIDEEP_SGLANG_VERSION)
    if database is None:
        pytest.skip(f"no perf database for {_WIDEEP_SGLANG_SYSTEM}/sglang/{_WIDEEP_SGLANG_VERSION}")
    model_config = config.ModelConfig(
        tp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        moe_comm_backend={"context": "deepep_ht", "generation": "deepep_ll"},
        num_gpus_per_node=8,
        attention_backend="flashinfer",
        gemm_quant_mode=common.GEMMQuantMode.fp8_block,
        moe_quant_mode=common.MoEQuantMode.fp8_block,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8_block,
    )
    model = _quiet(get_model, _WIDEEP_SGLANG_MODEL, model_config, "sglang")
    backend = get_backend("sglang")
    spec_json = _quiet(
        engine.build_engine_spec_json,
        model,
        model_path=_WIDEEP_SGLANG_MODEL,
        system=_WIDEEP_SGLANG_SYSTEM,
        backend="sglang",
        backend_version=_WIDEEP_SGLANG_VERSION,
        kv_block_size=None,
        systems_path=None,
        nextn=0,
        database=database,
    )
    return model, backend, database, spec_json


def _handle_from_spec_json(spec_json: str) -> engine.EngineHandle:
    import aiconfigurator_core

    return engine.EngineHandle(bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json)))


class TestWideEpDeepEpParity:
    """SGLang large-EP DeepSeek (deepep_ht/deepep_ll) end-to-end parity.

    Covers three previously-divergent surfaces at once: the WideEP MLA
    per-rank-heads table coordinate (tp=8 -> heads=16; the bridge used to emit
    raw tp), the modeled large-EP MoE compute routing over stock `moe_perf`,
    and the DeepEP dispatch flavor emission (the emitter used to map every
    sglang dispatch to CustomAllReduce). Data lives on h200_sxm/sglang/0.5.6.post2
    (the only shipped version with the deepep dispatch parquets)."""

    def _skip_when_wideep_sglang_golden_absent(self) -> None:
        """Skip if the golden capture dropped this surface at regeneration.

        The wideep_sglang capture skips on ``OpConversionError`` (large-EP
        native compile pending AIC-1601) and on ``PerfDataNotAvailableError``
        (the shipped h200_sxm/sglang moe_perf data has no ``moe_ep_size=8``
        slice after the 2b046af3 A2A-only large-EP model). Either drop yields
        an empty ``wideep_sglang::*`` block in the goldens — the tests are
        no-ops on those runs rather than treating a missing key as a
        regression.
        """
        references = load_parity_golden("compile_engine.json")["references"]
        if "wideep_sglang::static_ctx" not in references:
            pytest.skip("wideep_sglang golden absent (surface skipped at capture time)")

    def test_wideep_static_parity(self) -> None:
        self._skip_when_wideep_sglang_golden_absent()
        _model, _backend, _database, spec_json = _build_wideep_sglang()
        handle = _handle_from_spec_json(spec_json)
        new_ctx, new_gen, _ = handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
        _assert_within("wideep_static_ctx", _golden_reference("wideep_sglang::static_ctx"), new_ctx, backend="sglang")
        _assert_within("wideep_static_gen", _golden_reference("wideep_sglang::static_gen"), new_gen, backend="sglang")

    def test_wideep_mixed_and_decode_parity(self) -> None:
        self._skip_when_wideep_sglang_golden_absent()
        _model, _backend, _database, spec_json = _build_wideep_sglang()
        handle = _handle_from_spec_json(spec_json)
        new_mixed = handle.mixed_step_latency(1024, 2, 1024, 4, 0)
        new_decode = handle.decode_step_latency(2, 1024, 4)
        _assert_within("wideep_mixed", _golden_reference("wideep_sglang::mixed_step"), new_mixed, backend="sglang")
        _assert_within("wideep_decode", _golden_reference("wideep_sglang::decode_step"), new_decode, backend="sglang")


# --------------------------------------------------------------------------- #
# 2d. TRT-LLM large-EP (NVLink Two-Sided alltoall) — gb200.
# --------------------------------------------------------------------------- #


def _build_wideep_trtllm():
    """(model, backend, database, spec_json) for the TRT-LLM WideEP config;
    shared by the parity test (handle side) and the golden capture."""
    from aiconfigurator.sdk import common

    database = _quiet(perf_database.get_database, "gb200", "trtllm", "1.3.0rc10")
    if database is None:
        pytest.skip("no perf database for gb200/trtllm/1.3.0rc10")
    model_config = config.ModelConfig(
        tp_size=1,
        attention_dp_size=8,
        moe_tp_size=1,
        moe_ep_size=8,
        moe_comm_backend={"context": "nvlink_two_sided", "generation": "nvlink_two_sided"},
        num_gpus_per_node=4,
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
        database=database,
    )
    return model, backend, database, spec_json


class TestTrtllmWideEpParity:
    """TRT-LLM large-EP DeepSeek (nvlink_two_sided, attention_dp=8) on gb200.

    Covers the trtllm all-to-all port (prepare+dispatch pre / combine post
    through the trtllm_alltoall table, kernel NVLinkTwoSided) and the
    alltoall loader keying (kernel_source/op_name/num_nodes — the pre-fix
    loader collapsed 1,556 of 2,096 gb200 rows)."""

    def test_trtllm_wideep_static_parity(self) -> None:
        _model, _backend, _database, spec_json = _build_wideep_trtllm()
        handle = _handle_from_spec_json(spec_json)
        new_ctx, new_gen, _ = handle.run_static(batch_size=1, isl=1024, osl=4, prefix=0, stride=1)
        _assert_within(
            "trtllm_wideep_static_ctx", _golden_reference("wideep_trtllm::static_ctx"), new_ctx, backend="trtllm"
        )
        _assert_within(
            "trtllm_wideep_static_gen", _golden_reference("wideep_trtllm::static_gen"), new_gen, backend="trtllm"
        )


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
