# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS Blackwell MoE-quant promotion for the v2 Task.

On the TRT-LLM backend, GPT-OSS-120b/20b deployed on a Blackwell-class system
(SM >= 100: b200_sxm / b300_sxm / gb200 / gb300) default ``moe_quant_mode`` to
``w4a8_mxfp4_mxfp8`` for higher tensor-core throughput, unless the user set it
explicitly.  In disagg mode each role is promoted independently based on its own
system.  (Port of the legacy V1 ``TaskConfigFactory`` gpt-oss-blackwell promotion.)

``backend_version`` is passed explicitly so __post_init__ skips the DB lookup.
"""

import pytest

from aiconfigurator.sdk import common
from aiconfigurator.sdk.task_v2 import Task

pytestmark = pytest.mark.unit

_W4A8 = common.MoEQuantMode.w4a8_mxfp4_mxfp8
_GPTOSS = "openai/gpt-oss-120b"


def _agg(system, *, model=_GPTOSS, backend="trtllm", **kwargs):
    return Task(
        serving_mode="agg",
        model_path=model,
        system_name=system,
        backend_name=backend,
        backend_version="dummy",
        total_gpus=8,
        **kwargs,
    )


def _disagg(prefill_system, decode_system, *, model=_GPTOSS, backend="trtllm", **kwargs):
    return Task(
        serving_mode="disagg",
        prefill_model_path=model,
        decode_model_path=model,
        prefill_system_name=prefill_system,
        decode_system_name=decode_system,
        prefill_backend_name=backend,
        decode_backend_name=backend,
        prefill_backend_version="dummy",
        decode_backend_version="dummy",
        total_gpus=16,
        **kwargs,
    )


class TestAggPromotion:
    @pytest.mark.parametrize("system", ["b200_sxm", "b300_sxm", "gb200", "gb300"])
    def test_promotes_on_every_blackwell_system(self, system):
        assert _agg(system).moe_quant_mode == _W4A8

    def test_no_promotion_on_non_blackwell(self):
        assert _agg("h200_sxm").moe_quant_mode != _W4A8

    def test_no_promotion_for_non_gptoss_model(self):
        assert _agg("b200_sxm", model="Qwen/Qwen3-32B").moe_quant_mode != _W4A8

    def test_no_promotion_on_non_trtllm_backend(self):
        # gpt-oss on Blackwell but not TRT-LLM -> no promotion.
        assert _agg("b200_sxm", backend="vllm").moe_quant_mode != _W4A8

    def test_explicit_moe_quant_is_respected(self):
        # User-set moe_quant_mode wins over the promotion.
        task = _agg("b200_sxm", moe_quant_mode=common.MoEQuantMode.fp8)
        assert task.moe_quant_mode == common.MoEQuantMode.fp8


class TestDisaggPromotion:
    def test_both_roles_blackwell_promotes_both(self):
        task = _disagg("b200_sxm", "b200_sxm")
        assert task.prefill_moe_quant_mode == _W4A8
        assert task.decode_moe_quant_mode == _W4A8

    def test_mixed_systems_promotes_only_blackwell_role(self):
        # prefill on Hopper, decode on Blackwell -> only decode promoted.
        task = _disagg("h200_sxm", "b200_sxm")
        assert task.prefill_moe_quant_mode != _W4A8
        assert task.decode_moe_quant_mode == _W4A8
