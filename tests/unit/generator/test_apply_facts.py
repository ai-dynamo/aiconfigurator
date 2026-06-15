# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for facts-default cli-arg application (Phase 4a).

Token convention (verified against ``engine.py``): the in-memory
``{role}_cli_args_list`` is the ``shlex.split`` of the rendered cli string, i.e.
a flat list of single tokens. A value flag is two tokens (``--block-size`` then
``256``); a bool flag is one token (``--trust-remote-code``); a False/None bool
is omitted. Appended defaults must match this exactly.
"""

from aiconfigurator.generator.facts.apply import _entry_matches, apply_model_default_args


def _model():
    return {"defaults": [
        {"match": {"backend": "vllm"}, "roles": ["*"],
         "backend_args": {"block-size": 256, "trust-remote-code": True}},
        {"match": {"backend": "vllm", "system": "gb200"}, "roles": ["decode"],
         "backend_args": {"moe-backend": "deep_gemm_mega_moe"}},
    ]}


def test_entry_matches():
    assert _entry_matches({"backend": "vllm"}, backend="vllm", system="h200", variant=None)
    assert not _entry_matches({"backend": "vllm", "system": "gb200"}, backend="vllm", system="h200", variant=None)
    assert _entry_matches({}, backend="vllm", system="h200", variant=None)


def test_defaults_appended_when_absent():
    tokens = ["--tensor-parallel-size", "8"]
    apply_model_default_args(tokens, _model(), backend="vllm", system="h200", role="agg", variant=None)
    # value flag -> flag token + stringified value token
    assert "--block-size" in tokens and "256" in tokens
    # bool True -> flag-only token
    assert "--trust-remote-code" in tokens
    # gb200-scoped entry must NOT apply on h200
    assert "--moe-backend" not in tokens


def test_user_value_wins():
    tokens = ["--block-size", "128"]
    apply_model_default_args(tokens, _model(), backend="vllm", system="h200", role="agg", variant=None)
    assert tokens.count("--block-size") == 1 and "128" in tokens and "256" not in tokens
    # the *-scoped trust-remote-code is still absent -> appended
    assert "--trust-remote-code" in tokens


def test_bool_false_omitted():
    model = {"defaults": [
        {"match": {}, "roles": ["*"], "backend_args": {"no-such-flag": False}},
    ]}
    tokens = ["--tensor-parallel-size", "8"]
    apply_model_default_args(tokens, model, backend="vllm", system="h200", role="agg", variant=None)
    assert "--no-such-flag" not in tokens


def test_role_filter_skips_non_matching_role():
    model = {"defaults": [
        {"match": {"backend": "sglang"}, "roles": ["decode"],
         "backend_args": {"speculative-num-steps": 3}},
    ]}
    tokens: list[str] = []
    apply_model_default_args(tokens, model, backend="sglang", system="h200", role="agg", variant=None)
    assert "--speculative-num-steps" not in tokens
    tokens2: list[str] = []
    apply_model_default_args(tokens2, model, backend="sglang", system="h200", role="decode", variant=None)
    assert "--speculative-num-steps" in tokens2 and "3" in tokens2
