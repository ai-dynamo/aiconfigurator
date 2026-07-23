# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the V1 -> V2 task-config compatibility shim."""

import pytest

from aiconfigurator.sdk import task_v2
from aiconfigurator.sdk.task_v1_compat import convert_v1_to_v2, is_v1_config

pytestmark = pytest.mark.unit


class TestIsV1Config:
    def test_detects_structural_markers(self):
        assert is_v1_config({"mode": "patch", "serving_mode": "agg"})
        assert is_v1_config({"config": {"worker_config": {}}})
        assert is_v1_config({"profiles": []})

    def test_attention_backend_alone_is_not_v1(self):
        # attention_backend / wideep_num_slots are now first-class flat V2 fields,
        # so they must NOT trigger V1 auto-conversion on their own.
        assert not is_v1_config({"serving_mode": "agg", "model_path": "x", "attention_backend": "fa3"})
        assert not is_v1_config({"serving_mode": "agg", "model_path": "x", "wideep_num_slots": 288})

    def test_bare_disagg_top_level_not_treated_as_v1(self):
        # No structural marker -> left to V2 prefix-discipline (rejected there),
        # not silently treated as V1 and mirrored to roles.
        assert not is_v1_config({"serving_mode": "disagg", "model_path": "x"})
        # A real V1 disagg carries mode/config and IS detected.
        assert is_v1_config({"serving_mode": "disagg", "model_path": "x", "mode": "patch"})

    def test_rejects_v2_flat(self):
        assert not is_v1_config({"serving_mode": "agg", "model_path": "x", "system_name": "h200_sxm", "total_gpus": 8})
        assert not is_v1_config({"serving_mode": "disagg", "prefill_model_path": "x", "decode_model_path": "y"})


class TestConvertAgg:
    def test_top_level_and_worker_config(self):
        v1 = {
            "mode": "patch",
            "serving_mode": "agg",
            "model_path": "Qwen/Qwen3-32B-FP8",
            "system_name": "h200_sxm",
            "backend_name": "trtllm",
            "total_gpus": 16,
            "isl": 4000,
            "osl": 500,
            "ttft": 600.0,
            "tpot": 16,
            "profiles": ["fp8"],
            "config": {
                "worker_config": {
                    "num_gpu_per_worker": [1, 2, 4],
                    "tp_list": [1, 2, 4],
                    "moe_ep_list": [1, 2],
                    "gemm_quant_mode": "fp8",
                    "enable_wideep": True,
                },
            },
        }
        out = convert_v1_to_v2(v1)
        # top-level passthrough
        assert out["serving_mode"] == "agg"
        assert out["model_path"] == "Qwen/Qwen3-32B-FP8"
        assert out["system_name"] == "h200_sxm"
        assert out["total_gpus"] == 16
        assert out["isl"] == 4000 and out["osl"] == 500
        # list -> agg_*_candidates
        assert out["agg_num_gpu_candidates"] == [1, 2, 4]
        assert out["agg_tp_candidates"] == [1, 2, 4]
        assert out["agg_moe_ep_candidates"] == [1, 2]
        # scalar quant/flag stay top-level (no prefix) in agg
        assert out["gemm_quant_mode"] == "fp8"
        assert out["enable_wideep"] is True
        # profiles -> explicit quant fields
        assert out["gemm_quant_mode"] == "fp8"
        # structural keys dropped
        assert "mode" not in out
        assert "profiles" not in out
        assert "config" not in out


class TestConvertDisagg:
    def test_fan_out_and_renames(self):
        v1 = {
            "mode": "patch",
            "serving_mode": "disagg",
            "model_path": "Qwen/Qwen3-32B-FP8",
            "system_name": "h200_sxm",
            "decode_system_name": "h100_sxm",
            "backend_name": "trtllm",
            "backend_version": "1.2.0rc5",
            "total_gpus": 16,
            "isl": 4000,
            "osl": 500,
            "ttft": 600.0,
            "tpot": 16,
            "config": {
                "prefill_worker_config": {
                    "tp_list": [1, 2],
                    "gemm_quant_mode": "fp8",
                    "enable_wideep": True,
                },
                "decode_worker_config": {
                    "tp_list": [4, 8],
                    "moe_ep_list": [1, 2],
                },
                "replica_config": {"max_gpu_per_replica": 128, "max_prefill_worker": 4, "max_decode_worker": 8},
                "advanced_tuning_config": {
                    "prefill_max_batch_size": 1,
                    "decode_latency_correction_scale": 1.08,
                },
            },
        }
        out = convert_v1_to_v2(v1)
        # shared top-level worker spec fanned out to both roles
        assert out["prefill_model_path"] == "Qwen/Qwen3-32B-FP8"
        assert out["decode_model_path"] == "Qwen/Qwen3-32B-FP8"
        assert out["prefill_backend_name"] == "trtllm"
        assert out["decode_backend_name"] == "trtllm"
        assert out["prefill_backend_version"] == "1.2.0rc5"
        # system names: prefill from system_name, decode from decode_system_name
        assert out["prefill_system_name"] == "h200_sxm"
        assert out["decode_system_name"] == "h100_sxm"
        # per-role worker_config
        assert out["prefill_tp_candidates"] == [1, 2]
        assert out["prefill_gemm_quant_mode"] == "fp8"
        assert out["prefill_enable_wideep"] is True
        assert out["decode_tp_candidates"] == [4, 8]
        assert out["decode_moe_ep_candidates"] == [1, 2]
        # replica: worker -> workers rename
        assert out["max_gpu_per_replica"] == 128
        assert out["max_prefill_workers"] == 4
        assert out["max_decode_workers"] == 8
        # advanced tuning: _scale rename
        assert out["prefill_max_batch_size"] == 1
        assert out["decode_latency_correction"] == 1.08
        # top-level shared fields must NOT leak (V2 disagg forbids them)
        assert "model_path" not in out
        assert "system_name" not in out

    def test_decode_system_falls_back_to_prefill(self):
        out = convert_v1_to_v2({"serving_mode": "disagg", "model_path": "x", "system_name": "h200_sxm"})
        assert out["prefill_system_name"] == "h200_sxm"
        assert out["decode_system_name"] == "h200_sxm"


class TestUnmappable:
    def test_unmappable_fields_raise(self):
        """V1 fields with no V2 equivalent hard-fail conversion -- no silent drop."""
        v1 = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"worker_config": {"foo_bar": [1]}},
        }
        with pytest.raises(ValueError) as exc:
            convert_v1_to_v2(v1)
        msg = str(exc.value)
        assert "no V2 equivalent" in msg
        assert "foo_bar" in msg

    def test_nextn_accept_rates_folds_to_accepted(self):
        """V1 nextn_accept_rates lists fold into the scalar V2 ``nextn_accepted``
        (chain acceptance expectation), preserving V1 results."""
        v1 = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"nextn": 2, "nextn_accept_rates": [0.85, 0.3, 0.0, 0.0, 0.0]},
        }
        out = convert_v1_to_v2(v1)
        assert "nextn_accept_rates" not in out
        assert out["nextn"] == 2
        assert out["nextn_accepted"] == pytest.approx(0.85 + 0.85 * 0.3)

        # V1 defaulted the rates when absent; the fold must preserve that.
        v1_no_rates = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"nextn": 1},
        }
        out = convert_v1_to_v2(v1_no_rates)
        assert out["nextn_accepted"] == pytest.approx(0.85)

        # nextn absent -> rates dropped, no nextn_accepted emitted.
        v1_off = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"nextn_accept_rates": [0.85, 0.3, 0.0, 0.0, 0.0]},
        }
        out = convert_v1_to_v2(v1_off)
        assert "nextn_accepted" not in out
        assert "nextn_accept_rates" not in out

        # Explicit nextn: 0 (disabled) behaves the same as absent.
        v1_zero = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"nextn": 0, "nextn_accept_rates": [0.85, 0.3, 0.0, 0.0, 0.0]},
        }
        out = convert_v1_to_v2(v1_zero)
        assert out["nextn"] == 0
        assert "nextn_accepted" not in out
        assert "nextn_accept_rates" not in out

        # nextn beyond the historic 5-element list stays defined (missing
        # positions fold as zero acceptance).
        v1_long = {
            "serving_mode": "agg",
            "model_path": "x",
            "config": {"nextn": 6},
        }
        out = convert_v1_to_v2(v1_long)
        assert out["nextn_accepted"] == pytest.approx(0.85 + 0.85 * 0.3)

    def test_attention_backend_and_wideep_num_slots_map(self):
        """attention_backend (config-level) and wideep_num_slots (top-level) now map to
        shared V2 fields instead of being dropped."""
        out = convert_v1_to_v2(
            {
                "serving_mode": "agg",
                "model_path": "x",
                "wideep_num_slots": 288,
                "config": {"attention_backend": "fa3"},
            }
        )
        assert out["attention_backend"] == "fa3"
        assert out["wideep_num_slots"] == 288

    def test_multiple_profiles_raise(self):
        """Multiple V1 profiles drop all but the first -- a silent semantic change, so reject."""
        with pytest.raises(ValueError) as exc:
            convert_v1_to_v2({"serving_mode": "agg", "model_path": "x", "profiles": ["fp8", "nvfp4"]})
        assert "profiles" in str(exc.value)


class TestFromYamlAutoConvert:
    def test_v1_yaml_detected_warns_and_converts(self, monkeypatch):
        # Skip heavy __post_init__ (model/DB resolution) — we test the convert seam only.
        monkeypatch.setattr(task_v2.Task, "__post_init__", lambda self: None)
        v1 = {
            "mode": "patch",
            "serving_mode": "agg",
            "model_path": "Qwen/Qwen3-32B-FP8",
            "system_name": "h200_sxm",
            "total_gpus": 8,
            "config": {"worker_config": {"tp_list": [1, 2]}},
        }
        with pytest.warns(DeprecationWarning):
            task = task_v2.Task.from_yaml(v1)
        assert task.serving_mode == "agg"
        assert task.model_path == "Qwen/Qwen3-32B-FP8"
        assert task.agg_tp_candidates == [1, 2]

    def test_v2_yaml_not_converted_no_warning(self, monkeypatch, recwarn):
        monkeypatch.setattr(task_v2.Task, "__post_init__", lambda self: None)
        v2 = {"serving_mode": "agg", "model_path": "Qwen/Qwen3-32B-FP8", "system_name": "h200_sxm", "total_gpus": 8}
        task = task_v2.Task.from_yaml(v2)
        assert task.model_path == "Qwen/Qwen3-32B-FP8"
        assert not any(issubclass(w.category, DeprecationWarning) for w in recwarn.list)
