# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The EngineSpec must carry a LITERAL version, never a slot alias.

The Rust side reloads the perf database from the spec's ``backend_version``
string verbatim (``AicEngine.from_spec`` and the native ``AicEngineBuilder``
path the Dynamo Mocker uses) and resolves no slot aliases — slot semantics
live in the python layer only. These tests are version-agnostic: they assert
the alias is REPLACED by whatever the slots currently resolve to, pinning no
particular version.
"""

import json

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import get_model
from aiconfigurator_core.sdk import perf_database
from aiconfigurator_core.sdk.config import ModelConfig
from aiconfigurator_core.sdk.engine import build_engine_spec_json

pytestmark = pytest.mark.unit

_MODEL = "Qwen/Qwen3-1.7B"
_SYSTEM = "b200_sxm"
_BACKEND = "vllm"


def _spec_backend_version(backend_version, database):
    cfg = ModelConfig(
        tp_size=1,
        pp_size=1,
        gemm_quant_mode=common.GEMMQuantMode.bfloat16,
        kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
        fmha_quant_mode=common.FMHAQuantMode.bfloat16,
    )
    model = get_model(_MODEL, cfg, _BACKEND)
    spec = json.loads(
        build_engine_spec_json(
            model,
            model_path=_MODEL,
            system=_SYSTEM,
            backend=_BACKEND,
            backend_version=backend_version,
            kv_block_size=None,
            systems_path=None,
            nextn=0,
            database=database,
        )
    )
    return spec["engine"]["backend_version"]


def test_spec_resolves_current_alias_to_the_loaded_database_version():
    db = perf_database.get_database(_SYSTEM, _BACKEND, "current")
    embedded = _spec_backend_version("current", db)
    assert embedded == db.version
    assert embedded not in ("current", "previous", "next")


def test_spec_resolves_alias_even_without_a_loaded_database():
    # The compile path tolerates a failed database load; the spec must still
    # carry the literal the slots resolve the alias to.
    expected = perf_database.resolve_query_version(_SYSTEM, _BACKEND, "current")
    embedded = _spec_backend_version("current", None)
    assert embedded == expected
    assert embedded not in ("current", "previous", "next")


def test_spec_keeps_explicit_literal_versions_verbatim():
    db = perf_database.get_database(_SYSTEM, _BACKEND, "current")
    assert _spec_backend_version(db.version, db) == db.version


def test_spec_maps_omitted_version_to_the_current_literal():
    # An omitted version means the current slot; the wire must carry its
    # literal, never null (the Rust reload path resolves no defaults).
    expected = perf_database.resolve_query_version(_SYSTEM, _BACKEND, "current")
    assert _spec_backend_version(None, None) == expected


def test_spec_build_rejects_unlisted_versions_without_the_escape(monkeypatch):
    # Review blocker (2026-08): the shim used to fall back to the raw input
    # when resolution failed, smuggling ungated coordinates onto the wire.
    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)
    with pytest.raises(ValueError, match="old-style raw version query"):
        _spec_backend_version("0.22.0", None)


def test_spec_build_rejects_unpopulated_alias(monkeypatch):
    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)
    with pytest.raises(ValueError, match="has no 'previous'"):
        _spec_backend_version("previous", None)


def test_engine_handle_reload_path_resolves_the_alias():
    # The actual reload path the review blocker names: compile with an ALIAS,
    # construct an EngineHandle from the bytes (the native engine reloads the
    # database from the embedded string verbatim), and get a real answer —
    # possible only if the wire carried the literal.
    from aiconfigurator_core.sdk.engine import EngineHandle, compile_engine

    blob = compile_engine(_MODEL, _SYSTEM, _BACKEND, "current")
    handle = EngineHandle(blob)
    result = handle.run_static(batch_size=1, isl=64, osl=2, mode="static")
    assert result is not None


def test_engine_handle_compile_rejects_unlisted_versions(monkeypatch):
    from aiconfigurator_core.sdk.engine import compile_engine

    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)
    with pytest.raises(ValueError, match="old-style raw version query"):
        compile_engine(_MODEL, _SYSTEM, _BACKEND, "0.22.0")
