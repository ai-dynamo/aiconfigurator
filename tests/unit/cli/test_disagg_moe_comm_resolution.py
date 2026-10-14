# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for disaggregated MoE communication resolution."""

import pytest

pytestmark = pytest.mark.unit


def test_disagg_resolves_each_role_moe_comm_before_session_run(monkeypatch):
    import aiconfigurator.cli.api as api
    import aiconfigurator.sdk.inference_session as inference_session

    databases = {"prefill": object(), "decode": object()}
    resolver_calls = []
    captured = {}

    def fake_resolve(model_config, **kwargs):
        resolver_calls.append((model_config, kwargs))
        phase = kwargs["required_phases"][0]
        model_config.moe_comm_backend = {phase: "deepep_ht" if phase == "context" else "deepep_ll"}
        model_config.num_gpus_per_node = 4

    class FakeSummary:
        def check_oom(self):
            return False

        def get_result_dict(self):
            return {"ttft": 1.0, "tpot": 1.0}

        def get_power_data_coverage(self):
            return 1.0

        def get_per_ops_data(self):
            return {}

        def get_per_ops_source(self):
            return {}

        def get_moe_comm_fallbacks(self):
            return ()

    class FakeSession:
        def __init__(self, *, prefill_database, prefill_backend, decode_database, decode_backend):
            assert prefill_database is databases["prefill"]
            assert decode_database is databases["decode"]

        def set_latency_correction_scales(self, *_args):
            pass

        def run_disagg(self, **kwargs):
            captured.update(kwargs)
            return FakeSummary()

    monkeypatch.setattr(api, "check_is_moe", lambda _model_path: True)
    monkeypatch.setattr(api, "_resolve_moe_parallelism", lambda *args, **kwargs: (1, 64))
    monkeypatch.setattr(api, "resolve_model_config_moe_comm", fake_resolve)
    monkeypatch.setattr(
        api,
        "resolve_context_fmha_by_data",
        lambda *args, **kwargs: pytest.fail("resolved WideEP config must bypass generic FMHA fallback"),
    )
    monkeypatch.setattr(api, "resolve_dsv4_moe_arch", lambda *args, **kwargs: None)
    monkeypatch.setattr(api, "resolve_nvfp4_for_system", lambda *args, **kwargs: None)
    monkeypatch.setattr(inference_session, "DisaggInferenceSession", FakeSession)

    api._run_disagg_estimate(
        model_path="deepseek-ai/DeepSeek-V3",
        system_name="prefill",
        decode_system_name="decode",
        backend_name="vllm",
        resolved_version="0.24.0",
        isl=256,
        osl=256,
        image_height=0,
        image_width=0,
        num_images=1,
        enable_encoder_dp=False,
        prefill_tp_size=8,
        prefill_pp_size=1,
        prefill_attention_dp_size=1,
        prefill_moe_tp_size=8,
        prefill_moe_ep_size=1,
        prefill_batch_size=32,
        prefill_num_workers=1,
        decode_tp_size=1,
        decode_pp_size=1,
        decode_attention_dp_size=64,
        decode_moe_tp_size=1,
        decode_moe_ep_size=64,
        decode_batch_size=512,
        decode_num_workers=1,
        gemm_quant_mode=None,
        kvcache_quant_mode=None,
        fmha_quant_mode=None,
        moe_quant_mode=None,
        comm_quant_mode=None,
        load_database=lambda system: databases[system],
        get_backend=lambda _backend: object(),
        get_model=lambda *_args: object(),
    )

    assert [call[1]["required_phases"] for call in resolver_calls] == [("context",), ("generation",)]
    assert resolver_calls[0][1]["database"] is databases["prefill"]
    assert resolver_calls[1][1]["database"] is databases["decode"]
    assert captured["prefill_model_config"].moe_comm_backend == {"context": "deepep_ht"}
    assert captured["decode_model_config"].moe_comm_backend == {"generation": "deepep_ll"}
