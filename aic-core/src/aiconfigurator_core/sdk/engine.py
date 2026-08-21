# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compiled-engine builder.

This module is the Python half of the "Python builds, Rust executes"
architecture. ``compile_engine`` reuses the existing, unmodified Python model
layer (``sdk/models/*.py``) to build a model, walks its
``context_ops`` / ``generation_ops`` lists, converts each ``Operation`` to the
plain-data ``OpSpec`` wire form, and ships the whole thing across the boundary
as a bincode-serialised ``EngineSpec``.

Since the pyo3 op unification the ``Operation`` objects the model layer
builds ARE engine ops (Rust pyclasses; ``operations/*.py`` keeps thin
data-plane shells). Serialization is therefore Rust-to-Rust:

1. Each op serializes itself to the externally-tagged ``OpSpec`` wire form
   (``{"Gemm": {<fields>}}``) via the Rust ``_spec_json`` /
   ``ops_json_from_ops`` surfaces — there is no Python field-by-field
   serializer left to drift. The one Python-side op (``FPMForwardOp``, a
   whole-model orchestration wrapper) converts through a small adapter dict
   + ``op_from_spec_json``.
2. The Rust ``engine_spec_bincode_from_json`` ``#[pyfunction]`` decodes the
   assembled ``EngineSpec`` JSON and re-encodes it as bincode bytes (JSON is
   the debuggable wire; serde_json round-trips ``EngineConfig``'s flattened
   layout where bincode can't). Those bytes are what ``AicEngine.from_spec``
   and the Rust ``AicEngineBuilder`` consume.

``EngineHandle`` wraps the compiled bytes plus an ``AicEngine`` and exposes the
per-call surface (``run_static`` / ``predict_*_latency`` / ``mixed_step_latency``
/ ``decode_step_latency``). The agg sweep is orchestrated in Python, so there is
no ``run_agg`` here.

The live ``rust_engine_step.py`` helpers build on this path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiconfigurator_core
from aiconfigurator_core.sdk.config_builders import apply_nextn, build_model_config
from aiconfigurator_core.sdk.models import get_model
from aiconfigurator_core.sdk.operations import FPMForwardOp
from aiconfigurator_core.sdk.operations.base import Operation

# Reuse the exact quant-mode -> Rust ``DataType`` serde-string mappers the live
# ctypes bridge uses, so the compiled ``EngineConfig`` decodes the same way.
# The Python quant enum names (``int8_wo`` / ``int4_wo`` / ``sq`` / ``fp8_ootb``)
# are NOT valid ``DataType`` variants; these mappers collapse them to the
# accepted strings (``int8`` / ``int4`` / ...).
from aiconfigurator_core.sdk.rust_engine_step import (
    _moe_quant_to_dtype as _rust_moe_quant_to_dtype,
)
from aiconfigurator_core.sdk.rust_engine_step import (
    _quant_to_dtype as _rust_quant_to_dtype,
)

# Schema versions must match the Rust crate constants
# (`ENGINE_SPEC_SCHEMA_VERSION` / `ENGINE_CONFIG_SCHEMA_VERSION` in `lib.rs`).
# bincode op payloads are positional, so a producer/consumer skew is only
# distinguishable by this version; the Rust consumer gates on it before
# decoding the op lists. ENGINE_SPEC history:
#
# - 2 (v0.10.0): op-payload layout change — the CP + perf-DB refactor added
#   serialized `OpSpec` fields such as `seq_split` / `cp_size`.
# - 3 (PR #1405): MTP acceptance moved above aic-core — `nextn_accept_rates`
#   removed from the spec payload.
# - 4 (PR #1355): `Msa{Context,Generation}` op variants inserted (bincode
#   enum indices after `DsaGeneration` shifted). The MSA insertion and #1405
#   each claimed version 3 on their own branch, so their merge needed a
#   fresh number.
# - 5 (PR #1460): `MlaModule{Context,Generation}` payloads gained
#   `native_num_heads` (always serialized — bincode decodes positionally).
# - 6: `Kda` op variant appended (Kimi-K3 KDA kernels; draft_tokens field).
#   Claimed version 5 concurrently with #1460; renumbered at the merge.
# - 7: `MoEDispatchOp` gained `attn_ar_modeled` (always serialized — bincode
#   decodes positionally).
# - 8: `GemmOp` gained `below_grid_sol` (always serialized — bincode decodes
#   positionally).
# - 9: the Rust `Op::FpmForward` whole-model variant (forward_model="fpm").
#   Claimed 5, 7 and 8 concurrently with other landings; renumbered at each
#   merge (same precedent as the v3/v4 collision).
# - 10 (issue #1498): `Mhc` payload gained `seq_split` (CP per-rank token
#   division) — a bincode op-layout change. Claimed 7 concurrently with
#   `attn_ar_modeled`; renumbered at the rebase.
# - 11 (AIC-1601): the wideEP MoE op variants (`WideEpMoe` /
#   `WideEpMoeDispatch`) were removed mid-enum, shifting every later bincode
#   enum index; large-EP is now modeled natively by the `MoeAllToAll` /
#   `MoeExpertCompute` variants appended after `FpmForward`, and
#   `MoeExpertComputeOp` carries the
#   `enable_eplb` legacy-fidelity field.
# - 12 (PR-6): `DsaModuleOp` gained `attn_projection_quant_modes` — a
#   bincode op-layout change (same class as v5/v7/v8/v10; serde(default)
#   only covers the JSON wire, bincode is positional).
# - 13 (deprecation-cleanup PR): the engine owns shared-layer source
#   resolution. `EngineConfig` dropped the Python-resolved `perf_db_sources`
#   map (a bincode config-layout change) and gained `enable_shared_layer` /
#   `strict_provenance` policy flags; the engine re-derives every table's
#   source list from the perf-data tree
#   (`perf_database/source_resolution.rs`).
# Single owner: the Rust crate constant. Python re-exports it for
# diagnostics/tests instead of declaring a twin to keep in sync.
ENGINE_SPEC_SCHEMA_VERSION = aiconfigurator_core.engine_spec_schema_version()
ENGINE_CONFIG_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)


class OpConversionError(RuntimeError):
    """Raised when an ``Operation`` cannot be converted to an ``OpSpec``."""


# --------------------------------------------------------------------------- #
# Op -> wire. Ops are Rust objects; conversion is delegated to the engine.
# --------------------------------------------------------------------------- #


def _fpm_spec_dict(op: FPMForwardOp) -> dict:
    """The ``FpmForward`` OpSpec dict for the one Python-side op class.

    ``FPMForwardOp`` is a whole-model orchestration wrapper (its per-op
    roofline sources live in ``sol_ops``); it stays Python because its fields
    are compile products, not silicon tables. The identity strings were
    normalized by ``_norm_identity`` at op construction; Rust compares them
    verbatim."""
    return {
        "FpmForward": {
            "name": op._name,
            "phase": op._phase,
            "model_path": op._model_path,
            "match_identity": list(op._match_identity),
            "weight_bytes": op._weight_bytes,
            # Speculative verify width for the equivalent-AR decode mapping
            # (1 = plain AR). Set by the fpm hybrid rewrite in models when a
            # draft scheme is materialized.
            "verify_width": int(getattr(op, "_verify_width", 1) or 1),
            "sol_ops": [json.loads(_as_engine_op(c)._spec_json()) for c in op._sol_ops],
        }
    }


def _as_engine_op(op: Any) -> Operation:
    """Return the engine-backed form of ``op``.

    Engine ops (Rust ``Operation`` subclasses, i.e. every family shell) pass
    through; ``FPMForwardOp`` converts via its adapter dict +
    ``op_from_spec_json``. Anything else — the AFD orchestration ops, ad-hoc
    stand-ins — raises ``OpConversionError``, the established contract for
    graphs the native engine cannot represent."""
    if isinstance(op, Operation):
        return op
    if isinstance(op, FPMForwardOp):
        return aiconfigurator_core.op_from_spec_json(json.dumps(_fpm_spec_dict(op)))
    raise OpConversionError(
        f"no OpSpec conversion for {type(op).__module__}.{type(op).__name__} (op name={getattr(op, '_name', '?')!r})"
    )


def _ops_json(ops: Any) -> str:
    """OpSpec JSON array via the Rust ``ops_json_from_ops`` gate (which
    refuses retired tombstone ops recursively — the ``deepep_moe`` dispatch
    flavor — exactly where the retired Python ``_to_opspec`` used to raise)."""
    engine_ops = [_as_engine_op(op) for op in ops]
    try:
        return aiconfigurator_core.ops_json_from_ops(engine_ops)
    except ValueError as exc:
        if "no native variant" in str(exc):
            raise OpConversionError(str(exc)) from None
        raise


# --------------------------------------------------------------------------- #
# EngineConfig assembly.
# --------------------------------------------------------------------------- #


def _shared_layer_flag(database: Any) -> bool | None:
    """The database view's shared-layer flag for the wire, ``None`` when no
    database is bound (the engine then derives it from ``database_mode``,
    mirroring ``_shared_layer_enabled``). The engine resolves per-op sources
    ITSELF (schema v13, ``perf_database/source_resolution.rs``); Python only
    ships the policy bit — including explicit ``shared_layer=`` overrides
    regression harnesses use to pin per-version behavior."""
    if database is None:
        return None
    flag = getattr(database, "enable_shared_layer", None)
    return None if flag is None else bool(flag)


def _strict_provenance_flag(database: Any) -> bool:
    """The database view's fail-closed provenance mode for the wire (absent
    database -> False, matching a bare load)."""
    return bool(getattr(database, "strict_provenance", False)) if database is not None else False


def _engine_config_dict(
    *,
    model: Any,
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    kv_block_size: int | None,
    systems_path: str | None,
    nextn: int,
    database: Any = None,
) -> dict:
    """Build the ``EngineConfig`` JSON (matches the Rust modularised struct).

    The quant/parallel fields are read off the resolved ``model.config`` so the
    compiled engine identity reflects the quant inference that happened inside
    ``get_model``. The Rust ``EngineConfig`` only uses ``speculative.nextn``
    for latency (decode-batch scaling) and the model/system/backend/parallel
    fields to locate the perf database; the rest are carried for completeness.
    """
    cfg = model.config
    model_nextn = getattr(model, "_nextn", None)
    effective_nextn = int(model_nextn) if model_nextn is not None else int(nextn)
    speculative = None
    if effective_nextn:
        speculative = {"nextn": effective_nextn}
    # The Rust ``EngineConfig`` flattens ``parallel`` / ``quantization`` /
    # ``speculative`` via ``#[serde(flatten)]``, so their fields live at the
    # TOP LEVEL of this dict (NOT nested under sub-keys).
    engine: dict[str, Any] = {
        "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
        "model_name": model_path,
        "system_name": system,
        "systems_path": systems_path,
        "backend": backend,
        "backend_version": backend_version,
        "kv_block_size": kv_block_size,
        # ParallelMapping (flattened)
        "tp_size": int(cfg.tp_size or 1),
        "pp_size": int(cfg.pp_size or 1),
        "attention_dp_size": _opt_int(getattr(cfg, "attention_dp_size", None)),
        "moe_tp_size": _opt_int(getattr(cfg, "moe_tp_size", None)),
        "moe_ep_size": _opt_int(getattr(cfg, "moe_ep_size", None)),
        "cp_size": _opt_int(getattr(cfg, "cp_size", None)),
        # QuantizationConfig (flattened)
        "weight_dtype": _rust_quant_to_dtype(getattr(cfg, "gemm_quant_mode", None)),
        "moe_dtype": _rust_moe_quant_to_dtype(getattr(cfg, "moe_quant_mode", None)),
        "activation_dtype": _rust_quant_to_dtype(getattr(cfg, "fmha_quant_mode", None)),
        "kv_cache_dtype": _rust_quant_to_dtype(getattr(cfg, "kvcache_quant_mode", None)),
        # Shared-layer policy bits only (schema v13): the engine resolves
        # per-op sources itself (`perf_database/source_resolution.rs`), so the
        # wire carries the flag, not the resolved map.
        "enable_shared_layer": _shared_layer_flag(database),
        "strict_provenance": _strict_provenance_flag(database),
        # Perf-database query mode + enabled empirical transfer kinds, read off
        # the live database view so the compiled engine answers HYBRID/EMPIRICAL
        # queries the same way the Python step does. Presets are resolved here
        # (single source of truth in ``common.TRANSFER_PRESETS``); the wire form
        # is always explicit kind tokens, ``None`` = the default ALL policy.
        "database_mode": _database_mode_name(database),
        "transfer_policy": _transfer_policy_tokens(database),
        "extra": {},
    }
    # SpeculativeConfig (flattened, Option<>): emit nextn at the top level
    # when MTP is active. When inactive, omit it so the
    # flattened Option deserializes to None.
    if speculative is not None:
        engine["nextn"] = speculative["nextn"]
    return engine


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _database_mode_name(database: Any) -> str:
    """The database view's query mode as the wire token (default SILICON)."""
    if database is None:
        return "SILICON"
    mode = getattr(database, "get_default_database_mode", lambda: None)()
    return getattr(mode, "name", str(mode)) if mode is not None else "SILICON"


def _transfer_policy_tokens(database: Any) -> list[str] | None:
    """The view's enabled transfer kinds as explicit wire tokens.

    ``None`` = the default ALL-transfers policy (backward-compatible absent
    key). A non-default policy serialises as a sorted list of kind values so
    the Rust side never needs the preset vocabulary.
    """
    if database is None:
        return None
    policy = getattr(database, "transfer_policy", None)
    if policy is None:
        return None
    from aiconfigurator_core.sdk.common import ALL_TRANSFERS

    if frozenset(policy) == ALL_TRANSFERS:
        return None
    return sorted(kind.value for kind in policy)


# --------------------------------------------------------------------------- #
# Public entry points.
# --------------------------------------------------------------------------- #


def compile_engine(
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None = None,
    *,
    tp_size: int = 1,
    pp_size: int = 1,
    attention_dp_size: int = 1,
    moe_tp_size: int | None = None,
    moe_ep_size: int | None = None,
    gemm_quant_mode: str | None = None,
    moe_quant_mode: str | None = None,
    kvcache_quant_mode: str | None = None,
    fmha_quant_mode: str | None = None,
    comm_quant_mode: str | None = None,
    nextn: int = 0,
    kv_block_size: int | None = None,
    systems_path: str | None = None,
    forward_model: str | None = None,
) -> bytes:
    """Compile a model into bincoded ``EngineSpec`` bytes.

    Signature matches the kwargs the Rust ``AicEngineBuilder`` passes. Reuses
    ``cli/api._build_model_config`` + ``sdk/models.get_model`` (quant inferred
    inside ``get_model``) to build the model, then walks ``encoder_ops`` (vision
    decomposed), ``context_ops`` and ``generation_ops`` into OpSpecs and returns
    the bytes produced by the Rust ``engine_spec_bincode_from_json`` pyfunction.
    """
    # `_build_model_config` resolves MoE parallelism defaults internally and
    # does not take a model_path (quant inference is done inside `get_model`).
    resolved_moe_tp = moe_tp_size if moe_tp_size is not None else 1
    resolved_moe_ep = moe_ep_size if moe_ep_size is not None else 1
    model_config = build_model_config(
        tp_size=tp_size,
        pp_size=pp_size,
        attention_dp_size=attention_dp_size,
        moe_tp_size=resolved_moe_tp,
        moe_ep_size=resolved_moe_ep,
        gemm_quant_mode=gemm_quant_mode,
        kvcache_quant_mode=kvcache_quant_mode,
        fmha_quant_mode=fmha_quant_mode,
        moe_quant_mode=moe_quant_mode,
        comm_quant_mode=comm_quant_mode,
        forward_model=forward_model,
    )
    # Apply MTP BEFORE get_model so the walked op lists carry the
    # (L+nextn)/L compute scale; accepted-token progress is applied above core.
    apply_nextn(model_config, nextn)
    model = get_model(model_path, model_config, backend)

    # The database supplies the shared-layer perf sources, the query mode and
    # the transfer policy stamped into the compiled `EngineConfig`. Load lazily
    # and tolerate failure; the Rust core falls back to its own defaults.
    database = _maybe_load_database(system, backend, backend_version, systems_path)

    spec_json = build_engine_spec_json(
        model,
        model_path=model_path,
        system=system,
        backend=backend,
        backend_version=backend_version,
        kv_block_size=kv_block_size,
        systems_path=systems_path,
        nextn=model_config.nextn,
        database=database,
    )

    return bytes(aiconfigurator_core.engine_spec_bincode_from_json(spec_json))


def build_ops_json(ops: Any) -> str:
    """Serialize an op list to the OpSpec JSON array the ad-hoc op-list
    evaluation FFI (``AicEngine.evaluate_ops_json``) consumes.

    Serves op lists deliberately NOT emitted into the compiled ``EngineSpec``
    (the VL encoder phase) and the single-op plumbing. Raises
    ``OpConversionError`` for ops the spec cannot express, exactly like the
    spec builder."""
    return _ops_json(ops)


def build_engine_spec_json(
    model: Any,
    *,
    model_path: str,
    system: str,
    backend: str,
    backend_version: str | None,
    kv_block_size: int | None,
    systems_path: str | None,
    nextn: int,
    database: Any = None,
) -> str:
    """Walk a built model's op lists into an ``EngineSpec`` JSON string.

    Separated from ``compile_engine`` so the op-transfer round-trip test can
    inspect the JSON (and the decoded ops) without going through bincode.
    """
    # Vision encoder ops are intentionally NOT emitted into the spec.
    #
    # The compile path threads no image configuration (num_images_per_request,
    # image_height/width, num_image_tokens), so the compiled engine cannot
    # reproduce `BaseBackend._run_encoder`'s token-count math (eff_batch, eff_s,
    # pre/post-merge counts) needed to query the vision ops with correct shapes.
    # Python's `run_static` already treats any request without image dimensions
    # as text-only and skips the encoder entirely (base_backend `_run_encoder`
    # early-return). Emitting the encoder ops here would make the compiled engine
    # query them unconditionally (with wrong shapes), diverging from the Python
    # reference for VL models. Vision modeling in the compiled path is deferred
    # until runtime image config is threaded through compile_engine (#1567).
    context_ops = json.loads(_ops_json(model.context_ops))
    generation_ops = json.loads(_ops_json(model.generation_ops))

    spec = {
        "schema_version": ENGINE_SPEC_SCHEMA_VERSION,
        "engine": _engine_config_dict(
            model=model,
            model_path=model_path,
            system=system,
            backend=backend,
            backend_version=backend_version,
            kv_block_size=kv_block_size,
            systems_path=systems_path,
            nextn=nextn,
            database=database,
        ),
        "context_ops": context_ops,
        "generation_ops": generation_ops,
    }
    return json.dumps(spec)


def build_database_probe_spec_json(
    database: Any, *, systems_path: str | None = None, database_mode: str | None = None
) -> str:
    """``EngineSpec`` JSON with EMPTY op lists: an engine bound to
    ``database``'s perf tables only (same shared-layer sources, query mode
    and transfer policy as the live Python view). Compiled by tools that
    evaluate ad-hoc op lists (``evaluate_ops_json`` /
    ``evaluate_ops_sol_json``) without a model — the sanity-check notebook
    sources its per-op reference values through this.

    ``database_mode`` overrides the view's own query mode (wire token, e.g.
    ``"SOL"``): the table-view path probes estimate-only databases (a spec
    yaml with no collected data) under the SOL mode token so their analytic
    views stay servable."""
    engine: dict[str, Any] = {
        "schema_version": ENGINE_CONFIG_SCHEMA_VERSION,
        "model_name": "__database_probe__",
        "system_name": database.system,
        "systems_path": systems_path,
        "backend": database.backend,
        "backend_version": database.version,
        "kv_block_size": None,
        "tp_size": 1,
        "pp_size": 1,
        "attention_dp_size": None,
        "moe_tp_size": None,
        "moe_ep_size": None,
        "cp_size": None,
        "weight_dtype": None,
        "moe_dtype": None,
        "activation_dtype": None,
        "kv_cache_dtype": None,
        "enable_shared_layer": _shared_layer_flag(database),
        "strict_provenance": _strict_provenance_flag(database),
        "database_mode": database_mode or _database_mode_name(database),
        "transfer_policy": _transfer_policy_tokens(database),
        "extra": {},
    }
    spec = {
        "schema_version": ENGINE_SPEC_SCHEMA_VERSION,
        "engine": engine,
        "context_ops": [],
        "generation_ops": [],
    }
    return json.dumps(spec)


# --------------------------------------------------------------------------- #
# Model-less probe plumbing.
#
# A small LRU of probe engines keyed by database identity, serving the two
# PERMANENT consumers that need engine values without a compiled model: the
# table-view bindings (``engine_table_view.fetch_table_view``) and the
# single-op evaluation helper behind the Python-side orchestration surfaces
# (the AFD comm ops, the ``_sum_latency`` fallback loop). It is NOT a public
# per-op query surface — the deprecated ``query_*``/``Operation.query`` shims
# that once rode this plumbing were removed after their one-release window;
# long-term callers use ``EngineHandle.evaluate_ops_json`` /
# ``evaluate_ops_sol_json`` (op-list FFI) or the phase/run entry points.
# --------------------------------------------------------------------------- #

_PROBE_HANDLE_CACHE: dict[str, EngineHandle] = {}
_PROBE_HANDLE_CACHE_MAX = 8
# Bumped by _clear_probe_handle_cache. Consumers that memoize a probe-spec
# KEY per database (engine_table_view.fetch_table_view) tag the memo with
# this generation and re-resolve after any documented eviction lever fires,
# so a cleared LRU can never be resurrected through a stale memoized key
# (stale source map, stale SOL-mode decision).
_PROBE_CACHE_GENERATION = 0


def _require_real_database(database: Any) -> None:
    """The probe engine loads perf tables from DISK by (system, backend,
    version, sources): only a real ``PerfDatabase`` can be served. Synthetic
    or duck-typed stand-ins must fail loudly here — silently answering from
    disk while the caller believes its in-memory tables are being read would
    be an incorrect-value bug. (A real instance whose loaded tables were
    monkeypatched afterwards is undetectable; the engine answers from the
    on-disk rows.)"""
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    if not isinstance(database, PerfDatabase):
        raise TypeError(
            "the engine table view and the single-op plumbing route through the compiled "
            f"engine, which loads perf tables from disk — got {type(database)!r} instead of "
            "PerfDatabase. Evaluate ad-hoc data via EngineHandle.evaluate_ops_json on a real "
            "database, or walk the raw loaded tables (database._<family>_data) directly."
        )


def _probe_spec_key(database: Any, mode_token: str | None) -> tuple[str, str | None]:
    """The ``(probe-spec JSON, systems_path)`` pair for ``database`` — the LRU
    key. The spec carries the load identity + data-resolution policy bits
    (schema v13: the engine resolves per-op sources itself), so building the
    key is cheap; callers that fetch many views per database
    (``engine_table_view``) still memoize it to skip the JSON assembly."""
    _require_real_database(database)
    systems_path = getattr(database, "systems_root", None) or os.environ.get("AICONFIGURATOR_SYSTEMS_PATH")
    key = build_database_probe_spec_json(database, systems_path=systems_path, database_mode=mode_token)
    return key, systems_path


def _probe_handle_from_key(key: str, systems_path: str | None) -> EngineHandle:
    """LRU half of ``_probe_handle_for``: every probe handle in the process
    lives in ``_PROBE_HANDLE_CACHE`` (cap ``_PROBE_HANDLE_CACHE_MAX``), so the
    eviction levers govern every pinned Rust perf-DB load."""
    handle = _PROBE_HANDLE_CACHE.get(key)
    if handle is None:
        handle = EngineHandle(aiconfigurator_core.engine_spec_bincode_from_json(key), systems_path=systems_path)
        while len(_PROBE_HANDLE_CACHE) >= _PROBE_HANDLE_CACHE_MAX:
            _PROBE_HANDLE_CACHE.pop(next(iter(_PROBE_HANDLE_CACHE)))
        _PROBE_HANDLE_CACHE[key] = handle
    return handle


def _probe_handle_for(database: Any, mode_token: str | None) -> EngineHandle:
    """Cached model-less probe engine over ``database``'s live view, optionally
    re-moded per call (``mode_token`` is the wire token, e.g. ``"SILICON"``).
    The cache key is the probe spec JSON itself: it captures system, backend,
    version, resolved per-op sources, query mode and transfer policy, so any
    view change produces a distinct entry."""
    key, systems_path = _probe_spec_key(database, mode_token)
    return _probe_handle_from_key(key, systems_path)


def _clear_probe_handle_cache() -> None:
    """Same eviction contract as the engine-step handle LRU (each handle pins
    a Rust-side perf-DB load); called from ``clear_all_op_caches``. Advancing
    the generation invalidates per-database probe-spec memos too, so the next
    fetch re-resolves sources (and the SOL-mode decision) from disk."""
    global _PROBE_CACHE_GENERATION
    _PROBE_CACHE_GENERATION += 1
    _PROBE_HANDLE_CACHE.clear()


def _evaluate_single_op(
    database: Any,
    op: Any,
    *,
    is_context: bool,
    batch_size: int,
    s: int,
    prefix: int = 0,
    x: int | None = None,
    imbalance_correction_scale: float = 1.0,
):
    """Evaluate ONE Python ``Operation`` through the compiled engine, under
    the database's live mode.

    The permanent single-op plumbing behind the Python-side ORCHESTRATION
    surfaces (the AFD comm ops' twin evaluation, the ``_sum_latency``
    fallback loop). The retired per-call query shims used to ride this too,
    with a per-call ``database_mode`` re-moding dimension (including the
    ``SOL_FULL`` decomposition triple) — that dimension left with the shims;
    per-call SOL decomposition is served by
    ``EngineHandle.evaluate_ops_sol_json`` directly."""
    from aiconfigurator_core.sdk.performance_result import PerformanceResult

    ops_json = build_ops_json([op])
    eval_kwargs = dict(
        is_context=bool(is_context),
        batch_size=int(batch_size),
        s=int(s),
        prefix=int(prefix),
        imbalance_correction_scale=float(imbalance_correction_scale),
        x=None if x is None else int(x),
    )
    handle = _probe_handle_for(database, None)
    (_, latency, energy, source) = handle.evaluate_ops_json(ops_json, **eval_kwargs)[0]
    return PerformanceResult(latency, energy=energy, source=source)


def _maybe_load_database(system: str, backend: str, backend_version: str | None, systems_path: str | None) -> Any:
    try:
        from aiconfigurator_core.sdk import perf_database

        return perf_database.get_database(system, backend, backend_version, systems_paths=systems_path)
    except Exception:
        return None


class EngineHandle:
    """Python wrapper over compiled ``EngineSpec`` bytes + a Rust ``AicEngine``.

    Exposes the per-call surface that shells through the PyO3 ``AicEngine``
    methods. The agg sweep is orchestrated in Python (mix/genonly step counting
    lives in ``base_backend``), so there is no ``run_agg`` here.
    """

    def __init__(self, spec_bytes: bytes, *, systems_path: str | None = None) -> None:
        self._bytes = bytes(spec_bytes)
        self._systems_path = systems_path
        self._engine = aiconfigurator_core.AicEngine.from_spec(self._bytes, systems_path)

    @classmethod
    def compile(cls, model_path: str, system: str, backend: str, **kwargs: Any) -> EngineHandle:
        """Compile + wrap in one call. ``systems_path`` is forwarded to both."""
        systems_path = kwargs.get("systems_path")
        spec_bytes = compile_engine(model_path, system, backend, **kwargs)
        return cls(spec_bytes, systems_path=systems_path)

    @classmethod
    def for_database(cls, database: Any, *, systems_path: str | None = None) -> EngineHandle:
        """Model-less handle bound to ``database``'s perf tables (empty op
        lists — see :func:`build_database_probe_spec_json`). Serves ad-hoc
        op-list evaluation only (``evaluate_ops_json`` /
        ``evaluate_ops_sol_json``); the whole-run methods return zeros."""
        spec_json = build_database_probe_spec_json(database, systems_path=systems_path)
        spec_bytes = aiconfigurator_core.engine_spec_bincode_from_json(spec_json)
        return cls(spec_bytes, systems_path=systems_path)

    @property
    def spec_bytes(self) -> bytes:
        return self._bytes

    def run_static(
        self,
        *,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        beam_width: int = 1,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
        mode: str = "static",
        stride: int = 32,
    ) -> tuple[float, float, float]:
        """Return ``(context_ms, generation_ms, total_ms)``."""
        return self._engine.run_static(
            int(batch_size),
            int(beam_width),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
            mode,
            int(stride),
        )

    def predict_prefill_latency(self, bs: int, isl: int, prefix: int = 0) -> float:
        return self._engine.predict_prefill_latency(int(bs), int(isl), int(prefix))

    def predict_decode_latency(self, bs: int, isl: int, osl: int = 2) -> float:
        return self._engine.predict_decode_latency(int(bs), int(isl), int(osl))

    def mixed_step_latency(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> float:
        return self._engine.mixed_step_latency(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
        )

    def mixed_step_breakdown(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> tuple[float, float, float, float]:
        """Return total/shared-non-attn/context-attn/decode-attn latency."""
        return self._engine.mixed_step_breakdown(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
        )

    def decode_step_latency(
        self,
        gen_tokens: int,
        isl: int,
        osl: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> float:
        return self._engine.decode_step_latency(
            int(gen_tokens), int(isl), int(osl), float(gen_seq_imbalance_correction_scale)
        )

    def run_static_per_op(
        self,
        *,
        batch_size: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        beam_width: int = 1,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
        mode: str = "static",
        stride: int = 32,
    ) -> tuple[list[tuple[str, float, float, str]], list[tuple[str, float, float, str]]]:
        """``run_static`` with the per-op values kept: ``(context, generation)``
        lists of ``(name, latency_ms, energy_wms, source)``, name-folded (each
        name appears once, accumulated with Python's phase-dict semantics;
        generation values are per-step-folded, then weighted by
        ``repeat_count``)."""
        return self._engine.run_static_per_op(
            int(batch_size),
            int(beam_width),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
            mode,
            int(stride),
        )

    def mixed_step_breakdown_per_op(
        self,
        ctx_tokens: int,
        gen_tokens: int,
        isl: int,
        osl: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> tuple[
        list[tuple[str, float, float, str]],
        list[tuple[str, float, float, str]],
        list[tuple[str, float, float, str]],
    ]:
        """``mixed_step_breakdown`` with the per-op values kept:
        ``(shared_non_attention, context_attention, decode_attention)`` lists;
        context-attention entries arrive already divided by ``ceil(isl/ctx)``."""
        return self._engine.mixed_step_breakdown_per_op(
            int(ctx_tokens),
            int(gen_tokens),
            int(isl),
            int(osl),
            int(prefix),
            float(seq_imbalance_correction_scale),
            float(gen_seq_imbalance_correction_scale),
        )

    def decode_step_per_op(
        self,
        gen_tokens: int,
        isl: int,
        osl: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
    ) -> list[tuple[str, float, float, str]]:
        """``decode_step_latency`` with the per-op values kept."""
        return self._engine.decode_step_per_op(
            int(gen_tokens), int(isl), int(osl), float(gen_seq_imbalance_correction_scale)
        )

    def evaluate_context_ops(
        self,
        indices: list[int],
        *,
        batch_size: int,
        s: int,
        prefix: int = 0,
        seq_imbalance_correction_scale: float = 1.0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]:
        """Thin op-list evaluation over the compiled CONTEXT op list: evaluate
        the ops at ``indices`` (positions in the spec's ``context_ops``, which
        mirror ``model.context_ops`` order) at the context-phase shape.
        ``x`` overrides the per-op token count verbatim (callers with their
        own x policy, e.g. AFD's uniform ``batch * s``); ``None`` keeps the
        base-phase rule (``batch * s``, logits-GEMM exception)."""
        return self._engine.evaluate_context_ops(
            [int(i) for i in indices],
            int(batch_size),
            int(s),
            int(prefix),
            float(seq_imbalance_correction_scale),
            int(x) if x is not None else None,
        )

    def evaluate_generation_ops(
        self,
        indices: list[int],
        *,
        batch_size: int,
        s: int,
        gen_seq_imbalance_correction_scale: float = 1.0,
        prefix: int = 0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]:
        """Thin op-list evaluation over the compiled GENERATION op list at the
        decode-step shape (see :meth:`evaluate_context_ops`). The base decode
        walk carries no prefix; ``prefix`` exists for orchestrations that
        thread it (AFD)."""
        return self._engine.evaluate_generation_ops(
            [int(i) for i in indices],
            int(batch_size),
            int(s),
            float(gen_seq_imbalance_correction_scale),
            int(prefix),
            int(x) if x is not None else None,
        )

    def evaluate_ops_json(
        self,
        ops_json: str,
        *,
        is_context: bool,
        batch_size: int,
        s: int,
        prefix: int = 0,
        imbalance_correction_scale: float = 1.0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, str]]:
        """Evaluate an ad-hoc op list (JSON array of OpSpec objects) against
        this engine's database — serves op lists deliberately NOT in the
        compiled spec (the VL encoder phase); the caller keeps the shape math."""
        return self._engine.evaluate_ops_json(
            ops_json,
            bool(is_context),
            int(batch_size),
            int(s),
            int(prefix),
            float(imbalance_correction_scale),
            int(x) if x is not None else None,
        )

    def evaluate_ops_sol_json(
        self,
        ops_json: str,
        *,
        is_context: bool,
        batch_size: int,
        s: int,
        prefix: int = 0,
        imbalance_correction_scale: float = 1.0,
        x: int | None = None,
    ) -> list[tuple[str, float, float, float]]:
        """:meth:`evaluate_ops_json` under the SOL_FULL view: every op is
        forced onto its analytic SOL branch and the roofline decomposition is
        kept. Returns ``(name, sol_time_ms, sol_math_ms, sol_mem_ms)`` tuples
        (name-folded, ``+=`` on all three) — the compiled-engine replacement
        for per-call ``query_*(..., database_mode=SOL_FULL)`` triples. Raises
        for op families whose SOL path does not export its decomposition."""
        return self._engine.evaluate_ops_sol_json(
            ops_json,
            bool(is_context),
            int(batch_size),
            int(s),
            int(prefix),
            float(imbalance_correction_scale),
            int(x) if x is not None else None,
        )

    def last_provenance(self) -> str | None:
        """Empirical provenance tier fired during the most recent engine call
        on this handle (worst tier across ops, per Python's
        ``util_empirical.PROVENANCE_ORDER``), or ``None`` for a pure-silicon
        answer. Per-call state: every compute method resets the accumulator on
        entry. The rust engine-step bridge forwards non-silicon tiers into
        ``util_empirical.note_provenance`` so ``capture_provenance()`` /
        support-matrix HYBRID labelling behave identically on both engines."""
        return self._engine.last_provenance()
