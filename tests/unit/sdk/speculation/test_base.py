# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk.speculation import (
    NullScheme,
    SpecSchemeBase,
    SpeculationConfig,
    build_spec_scheme,
    get_spec_scheme_cls,
    register_spec_scheme,
)

pytestmark = pytest.mark.unit


def test_registry_round_trip():
    from aiconfigurator_core.sdk.speculation.base import _SPEC_SCHEME_REGISTRY

    try:

        @register_spec_scheme("_test_scheme")
        class _TestScheme(NullScheme):
            kind = "_test_scheme"

        assert get_spec_scheme_cls("_test_scheme") is _TestScheme
    finally:
        # The registry is process-global and has no public unregister:
        # leaking '_test_scheme' would make later tests order-dependent.
        _SPEC_SCHEME_REGISTRY.pop("_test_scheme", None)


def test_unknown_kind_raises_with_known_kinds():
    with pytest.raises(ValueError) as excinfo:
        get_spec_scheme_cls("definitely_not_registered")
    # The error must help the caller: list what IS registered.
    assert "none" in str(excinfo.value)


def test_null_scheme_contract():
    scheme = NullScheme()
    assert scheme.kind == "none"
    assert scheme.verify_width() == 1
    assert scheme.build_draft_generation_ops(model=None) == []
    assert scheme.build_draft_context_ops(model=None) == []
    assert scheme.draft_weights_bytes(model=None) == 0.0
    assert scheme.draft_kv_bytes_per_sequence(model=None, seq_len=4096) == 0.0
    assert scheme.expected_progress(0.85) == pytest.approx(1.85)
    scheme.validate(model=None, backend_name="trtllm")  # no-op, must not raise


def test_build_spec_scheme_defaults_to_null():
    assert isinstance(build_spec_scheme(None, None), NullScheme)
    assert isinstance(build_spec_scheme(None, SpeculationConfig()), NullScheme)


def test_identity_hash_distinguishes_draft_configs():
    base = SpeculationConfig(kind="dspark", params={"num_draft_tokens": 7})
    a = SpeculationConfig(kind="dspark", params={"num_draft_tokens": 7}, draft_config={"x": 1, "y": 2})
    b = SpeculationConfig(kind="dspark", params={"num_draft_tokens": 7}, draft_config={"x": 1, "y": 3})
    assert a.identity_hash() != b.identity_hash()
    assert a.identity_hash() != base.identity_hash()


def test_identity_hash_distinguishes_draft_model_paths():
    # Path is what resolves the draft op graph when draft_config is not
    # injected: same kind/params, different artifact -> different engine.
    a = SpeculationConfig(kind="draft_model", params={"num_speculative_tokens": 3}, draft_model_path="org/draft-a")
    b = SpeculationConfig(kind="draft_model", params={"num_speculative_tokens": 3}, draft_model_path="org/draft-b")
    assert a.identity_hash() != b.identity_hash()


def test_identity_hash_is_key_order_insensitive():
    a = SpeculationConfig(kind="dspark", params={"a": 1, "b": 2}, draft_config={"x": 1, "y": 2})
    b = SpeculationConfig(kind="dspark", params={"b": 2, "a": 1}, draft_config={"y": 2, "x": 1})
    assert a.identity_hash() == b.identity_hash()


def test_spec_scheme_base_is_abstract():
    with pytest.raises(TypeError):
        SpecSchemeBase()  # type: ignore[abstract]
