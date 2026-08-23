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
