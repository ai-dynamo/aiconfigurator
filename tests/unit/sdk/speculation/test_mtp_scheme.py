# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiconfigurator_core.sdk import common, models
from aiconfigurator_core.sdk import config as sdk_config
from aiconfigurator_core.sdk.config_builders import resolve_speculation
from aiconfigurator_core.sdk.speculation import NullScheme, SpeculationConfig
from aiconfigurator_core.sdk.speculation.mtp import MTPScheme

pytestmark = pytest.mark.unit


def _model_config(**kwargs) -> sdk_config.ModelConfig:
    return sdk_config.ModelConfig(
        tp_size=8,
        pp_size=1,
        moe_tp_size=1,
        moe_ep_size=8,
        gemm_quant_mode=common.GEMMQuantMode.fp8,
        moe_quant_mode=common.MoEQuantMode.fp8,
        kvcache_quant_mode=common.KVCacheQuantMode.fp8,
        fmha_quant_mode=common.FMHAQuantMode.fp8,
        **kwargs,
    )


class TestResolveSpeculation:
    def test_legacy_nextn_desugars_to_mtp(self):
        cfg = _model_config(nextn=2)
        spec = resolve_speculation(cfg)
        assert spec.kind == "mtp"
        assert spec.params["depth"] == 2
        assert cfg.speculation is spec

    def test_nextn_zero_resolves_to_none(self):
        cfg = _model_config(nextn=0)
        spec = resolve_speculation(cfg)
        assert spec.kind == "none"

    def test_mtp_speculation_writes_back_nextn(self):
        cfg = _model_config(speculation=SpeculationConfig(kind="mtp", params={"depth": 3}))
        spec = resolve_speculation(cfg)
        assert spec.kind == "mtp"
        assert cfg.nextn == 3

    def test_conflicting_nextn_and_mtp_depth_raises(self):
        cfg = _model_config(nextn=2, speculation=SpeculationConfig(kind="mtp", params={"depth": 3}))
        with pytest.raises(ValueError):
            resolve_speculation(cfg)

    def test_non_mtp_scheme_with_nextn_raises(self):
        cfg = _model_config(nextn=2, speculation=SpeculationConfig(kind="dspark"))
        with pytest.raises(ValueError):
            resolve_speculation(cfg)

    def test_idempotent(self):
        cfg = _model_config(nextn=2)
        first = resolve_speculation(cfg)
        second = resolve_speculation(cfg)
        assert first == second


class TestMTPScheme:
    @pytest.mark.parametrize("depth", [1, 2, 3, 5])
    def test_verify_width_is_nextn_plus_one(self, depth):
        scheme = MTPScheme(depth=depth)
        assert scheme.verify_width() == depth + 1

    def test_cost_identities(self):
        scheme = MTPScheme(depth=2)
        assert scheme.build_draft_generation_ops(model=None) == []
        assert scheme.build_draft_context_ops(model=None) == []
        assert scheme.draft_weights_bytes(model=None) == 0.0
        assert scheme.draft_kv_bytes_per_sequence(model=None, seq_len=8192) == 0.0

    def test_depth_must_be_positive(self):
        with pytest.raises(ValueError):
            MTPScheme(depth=0)


class TestGetModelAttachesScheme:
    def test_null_scheme_attached_when_disabled(self):
        cfg = _model_config(nextn=0)
        model = models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "sglang")
        assert isinstance(model.spec_scheme, NullScheme)

    def test_mtp_scheme_attached_and_consistent_with_model_nextn(self):
        cfg = _model_config(nextn=2)
        model = models.get_model("deepseek-ai/DeepSeek-V4-Flash", cfg, "sglang")
        assert isinstance(model.spec_scheme, MTPScheme)
        assert model.spec_scheme.verify_width() == model._nextn + 1 == 3
