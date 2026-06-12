import argparse
import csv
import dataclasses
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))

import parallel_config_patch as pcp
from random_prompt_tokens import load_random_prompt_token_config

from collector.layerwise.vllm import collect as public_collect
from collector.layerwise.vllm import datapoint_generator as dpg
from collector.layerwise.vllm import engine, nsys, results, scheduler, worker
from collector.layerwise.vllm.data import DataPoint, RepresentativeLayer, WorkUnit
from collector.layerwise.vllm.registry import LayerwiseModel

cl = SimpleNamespace(
    Attempt=scheduler.Attempt,
    CSV_COLUMNS=results.CSV_COLUMNS,
    DataPoint=DataPoint,
    RandomPromptTokenConfig=worker.RandomPromptTokenConfig,
    RepresentativeLayer=RepresentativeLayer,
    Scheduler=scheduler.Scheduler,
    StatusIndex=scheduler.StatusIndex,
    StatusStore=scheduler.StatusStore,
    WorkUnit=WorkUnit,
    _aggregate_step_rows=nsys._aggregate_step_rows,
    _build_arg_parser=public_collect._build_arg_parser,
    _ctx_cache_salt_prefix=worker._ctx_cache_salt_prefix,
    _ctx_marker_iteration=worker._ctx_marker_iteration,
    _ctx_prefix_stream_key=worker._ctx_prefix_stream_key,
    _dummy_prompts=worker._dummy_prompts,
    _engine_tokens=engine._engine_tokens,
    _effective_latency_source=scheduler._effective_latency_source,
    _filter_datapoints_for_model_max_len=dpg._filter_datapoints_for_model_max_len,
    _gen_cache_salt_prefix=worker._gen_cache_salt_prefix,
    _gen_prefix_stream_key=worker._gen_prefix_stream_key,
    _live_decode_past_lengths=worker._live_decode_past_lengths,
    _latency_us_from_agg=nsys._latency_us_from_agg,
    _lookup_aggs=nsys._lookup_aggs,
    _lookup_scheduler_timing_aggs=scheduler._lookup_scheduler_timing_aggs,
    _lookup_worker_wall_aggs=scheduler._lookup_worker_wall_aggs,
    _prefix_suffix_prompts=worker._prefix_suffix_prompts,
    _reduce_agg_latency=nsys._reduce_agg_latency,
    _use_live_step_driver=worker._use_live_step_driver,
    build_public_work_units=dpg.build_public_work_units,
    build_work_units=dpg.build_work_units,
    oom_dominates=scheduler.oom_dominates,
    PromptTokenFactory=worker.PromptTokenFactory,
    _attempt_signature=scheduler._attempt_signature,
)


def _unit(tmp_path, datapoints):
    row_base = {
        "framework": "vLLM",
        "framework_version": "test",
        "system": "gpu",
        "model": "model",
        "attn_tp": 1,
        "moe_tp": 1,
        "ep": 1,
        "num_slots": "",
        "gemm_quant": "bf16",
        "moe_quant": "bf16",
        "attn_quant": "bf16",
        "kv_quant": "bf16",
    }
    return cl.WorkUnit(
        work_unit_id="wu_test",
        model_dir=str(tmp_path / "model"),
        row_base=row_base,
        representative=cl.RepresentativeLayer(
            layer_index=0,
            layer_type="dense",
            measured_layer_count=1,
            layer_multiplier=1,
            target_layers=(0,),
        ),
        target_layers=[0],
        datapoints=datapoints,
    )


def _args(tmp_path):
    return argparse.Namespace(
        work_dir=str(tmp_path / "work"),
        output=str(tmp_path / "out.csv"),
        gpus="0",
        max_workers=None,
        timeout=30,
        nsys_capture="full",
        extra_vllm_arg=[],
        latency_source="span",
        moe_decode_gpu_batch_threshold=8,
        ctx_warmup_runs=0,
        ctx_measured_runs=1,
        ctx_repeat_aggregation="median",
        gen_warmup_runs=0,
        gen_measured_runs=1,
        gen_repeat_aggregation="median",
        prompt_seed=None,
        rollup=r"layers\.(\d+)\.(self_attn|mlp)",
        rank_reduce="sum",
    )


def _build_args(tmp_path, **overrides):
    args = argparse.Namespace(
        model="model",
        work_dir=str(tmp_path / "work"),
        config_cache_dir=None,
        no_config_cache=True,
        system="gpu",
        framework_version="test",
        tp_sizes="1,2,4",
        moe_tp=1,
        num_slots=None,
        moe_noop=False,
        target_layer_count=1,
        target_layers=None,
        target_layer_config_depth=None,
        phases="both",
        ctx_new_tokens="1",
        ctx_past_kv="0",
        no_filter_model_max_len=False,
        gen_batch_sizes="1",
        gen_past_kv="1",
        max_num_seqs=16,
        gemm_quant="bf16",
        moe_quant="bf16",
        attn_quant="bf16",
        kv_quant="bf16",
        max_num_batched_tokens=None,
        physical_tp_real_weights=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class VllmCollectLayerwiseTests(unittest.TestCase):
    def test_repeat_defaults_use_six_run_trimmed_mean(self):
        args = cl._build_arg_parser().parse_args(["--model", "model"])

        self.assertEqual(args.ctx_measured_runs, 6)
        self.assertEqual(args.ctx_repeat_aggregation, "trimmed_mean")
        self.assertEqual(args.gen_measured_runs, 6)
        self.assertEqual(args.gen_repeat_aggregation, "trimmed_mean")
        self.assertEqual(args.latency_source, "auto")
        self.assertEqual(args.max_decode_batch_size, "512")
        self.assertEqual(args.ctx_target_layer_count, 0)
        self.assertEqual(args.gen_target_layer_count, 1)
        self.assertFalse(hasattr(args, "gen_driver"))
        self.assertFalse(hasattr(args, "ctx_driver"))
        self.assertEqual(args.nsys_capture, "full")

    def test_public_both_sweep_splits_context_and_decode_target_depth(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "both",
                "--run-preset",
                "smoke",
                "--max-decode-batch-size",
                "8",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(1,))
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 2)
            ctx_unit = next(unit for unit in units if unit.datapoints[0].phase == "ctx")
            gen_unit = next(unit for unit in units if unit.datapoints[0].phase == "gen")
            self.assertEqual(ctx_unit.representative.measured_layer_count, 4)
            self.assertEqual(ctx_unit.representative.layer_multiplier, 4)
            self.assertEqual(ctx_unit.target_layers, [0, 1, 2, 3])
            self.assertEqual(ctx_unit.model_layer_count, 4)
            self.assertTrue(ctx_unit.uses_full_layer_depth())
            self.assertIsNone(ctx_unit.max_model_len)
            self.assertEqual(gen_unit.representative.measured_layer_count, 1)
            self.assertEqual(gen_unit.representative.layer_multiplier, 4)
            self.assertEqual(gen_unit.target_layers, [0])
            self.assertEqual(gen_unit.model_layer_count, 1)
            self.assertTrue(gen_unit.uses_full_layer_depth())
            self.assertIsNone(gen_unit.max_model_len)

    def test_public_tp1_moe_defaults_to_dummy_context_and_noop_decode_without_real_weights(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "both",
                "--run-preset",
                "smoke",
                "--max-decode-batch-size",
                "8",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(1,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(
                    dpg,
                    "_resolve_real_weight_model_dir",
                    return_value=str(tmp_path / "snapshot"),
                ) as resolver,
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 2)
            self.assertEqual({unit.datapoints[0].phase for unit in units}, {"ctx", "gen"})
            self.assertTrue(all(unit.model_dir == str(tmp_path / "patched") for unit in units))
            self.assertTrue(all(unit.extra_vllm_args == () for unit in units))
            ctx_unit = next(unit for unit in units if unit.datapoints[0].phase == "ctx")
            gen_unit = next(unit for unit in units if unit.datapoints[0].phase == "gen")
            self.assertFalse(ctx_unit.moe_noop)
            self.assertTrue(gen_unit.moe_noop)
            self.assertEqual(ctx_unit.moe_weight_mode, "dummy")
            self.assertEqual(gen_unit.moe_weight_mode, "noop")
            self.assertTrue(all(unit.router_weight_model is None for unit in units))
            self.assertEqual(gen_unit.gen_driver, "prefix_cache")
            resolver.assert_not_called()
            patcher.assert_called()

    def test_public_patched_moe_uses_router_weight_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--moe-real-router",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].model_dir, str(tmp_path / "patched"))
            self.assertEqual(units[0].router_weight_model, "model")
            self.assertEqual(units[0].extra_vllm_args, ())
            patcher.assert_called_once()

    def test_public_physical_tp_uses_depth_only_patch_and_gpu_group(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism") as parallel_patcher,
                mock.patch.object(dpg, "patch_model_path", return_value=str(tmp_path / "depth_only")) as depth_patcher,
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].model_dir, str(tmp_path / "depth_only"))
            self.assertEqual(
                units[0].extra_vllm_args,
                (
                    "--tensor-parallel-size",
                    "2",
                    "--worker-extension-cls",
                    "vllm_worker_extension.LayerwiseWorkerExtension",
                ),
            )
            self.assertEqual(units[0].physical_gpus, 2)
            self.assertEqual(units[0].router_weight_model, None)
            parallel_patcher.assert_not_called()
            depth_patcher.assert_called_once()

    def test_public_physical_tp_requires_diagnostic_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))

            with self.assertRaisesRegex(ValueError, "diagnostic-only"):
                cl.build_public_work_units(args, [model])

    def test_public_physical_tp_ep_enables_vllm_expert_parallel(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "2",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1, 2))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism") as parallel_patcher,
                mock.patch.object(dpg, "patch_model_path", return_value=str(tmp_path / "depth_only")) as depth_patcher,
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].model_dir, str(tmp_path / "depth_only"))
            self.assertEqual(units[0].row_base["moe_tp"], 1)
            self.assertEqual(units[0].row_base["ep"], 2)
            self.assertEqual(
                units[0].extra_vllm_args,
                (
                    "--tensor-parallel-size",
                    "2",
                    "--worker-extension-cls",
                    "vllm_worker_extension.LayerwiseWorkerExtension",
                    "--enable-expert-parallel",
                ),
            )
            self.assertEqual(units[0].physical_gpus, 2)
            parallel_patcher.assert_not_called()
            depth_patcher.assert_called_once()

    def test_public_physical_tp_real_weights_uses_original_snapshot(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
                "--physical-tp-real-weights",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "_resolve_real_weight_model_dir", return_value=str(tmp_path / "snapshot")),
                mock.patch.object(dpg, "patch_for_parallelism") as parallel_patcher,
                mock.patch.object(dpg, "patch_model_path") as depth_patcher,
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].model_dir, str(tmp_path / "snapshot"))
            self.assertIn("--load-format", units[0].extra_vllm_args)
            self.assertEqual(units[0].router_weight_model, None)
            parallel_patcher.assert_not_called()
            depth_patcher.assert_not_called()

    def test_context_schedule_envelope_does_not_add_fpm_scheduler(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
                "--physical-tp-real-weights",
                "--latency-source",
                "schedule_to_update",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(2,), ep_sizes=(1,))
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "_resolve_real_weight_model_dir", return_value=str(tmp_path / "snapshot")),
            ):
                units = cl.build_public_work_units(args, [model])

            extra = units[0].extra_vllm_args
            self.assertNotIn("--scheduler-cls", extra)

    def test_non_context_schedule_envelope_does_not_add_fpm_scheduler(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "gen",
                "--run-preset",
                "smoke",
                "--latency-source",
                "span",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(1,), ep_sizes=(1,))
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertNotIn("--scheduler-cls", units[0].extra_vllm_args)

    def test_context_worker_wall_keeps_full_depth_for_typed_moe_layers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--latency-source",
                "worker_wall",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
                "--physical-tp-real-weights",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "_resolve_real_weight_model_dir", return_value=str(tmp_path / "snapshot")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].target_layers, [0, 1, 2, 3])
            self.assertEqual(units[0].model_layer_count, 4)
            self.assertEqual(units[0].representative.measured_layer_count, 4)
            self.assertEqual(units[0].representative.layer_multiplier, 4)

    def test_public_physical_tp_real_weights_requires_full_depth(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "2",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--physical-tp",
                "--allow-multi-gpu-diagnostic",
                "--physical-tp-real-weights",
                "--target-layer-count",
                "1",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="moe", tp_sizes=(2,), ep_sizes=(1,))
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "moe_intermediate_size": 512,
                "num_experts": 16,
                "num_hidden_layers": 4,
            }

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                self.assertRaisesRegex(ValueError, "full-depth"),
            ):
                cl.build_public_work_units(args, [model])

    def test_scheduler_max_workers_does_not_truncate_physical_gpu_group(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.gpus = "0,1"
            args.max_workers = 1
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                physical_gpus=2,
            )
            scheduler = cl.Scheduler(args, [unit])

            self.assertEqual(scheduler._acquire_gpu_group({}, 2), "0,1")

    def test_scheduler_includes_work_unit_extra_vllm_args(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("gen", 1, 1, 4096)]),
                extra_vllm_args=("--load-format", "auto"),
            )
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            extra = spec["extra_vllm_args"]
            self.assertEqual(extra[extra.index("--load-format") + 1], "auto")

    def test_scheduler_spec_includes_work_unit_scheduler_limits(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)]),
                max_num_seqs=16,
                max_num_batched_tokens=4096,
            )
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertEqual(spec["max_num_seqs"], 16)
            self.assertEqual(spec["max_num_batched_tokens"], 4096)

    def test_datapoint_ids_and_parse_keys_are_stable(self):
        ctx = cl.DataPoint("ctx", 1, 128, 0)
        gen = cl.DataPoint("gen", 4, 1, 16)

        self.assertEqual(ctx.datapoint_id("wu"), "wu:ctx:bs1:new128:past0")
        self.assertEqual(gen.datapoint_id("wu"), "wu:gen:bs4:new1:past16")
        self.assertEqual(ctx.parse_key(), (128, 1, 0))
        self.assertEqual(gen.parse_key(), (17, 4, 16))

    def test_status_index_tracks_terminal_and_active_started(self):
        events = [
            {"event": "started", "work_unit_id": "wu", "datapoint_id": "a"},
            {"event": "success", "work_unit_id": "wu", "datapoint_id": "a"},
            {"event": "started", "work_unit_id": "wu", "datapoint_id": "b"},
            {"event": "started", "work_unit_id": "wu", "datapoint_id": "c"},
            {"event": "failed_error", "work_unit_id": "wu", "datapoint_id": "c"},
        ]
        index = cl.StatusIndex(events)

        self.assertTrue(index.is_terminal("a"))
        self.assertTrue(index.is_terminal("c"))
        self.assertEqual(index.active_started("wu", {"a", "b", "c"}), "b")

    def test_aggregate_step_rows_uses_span_for_latency_source(self):
        parsed = cl._aggregate_step_rows([
            {
                "step": 17,
                "batch_size": 4,
                "past_kv": 16,
                "measure_run": 0,
                "gpu_us": 100.0,
                "rms_us": 5.0,
                "start_ns": 1_000,
                "end_ns": 101_000,
                "kernel_count": 2,
                "rms_kernel_count": 1,
            },
            {
                "step": 17,
                "batch_size": 4,
                "past_kv": 16,
                "measure_run": 0,
                "gpu_us": 50.0,
                "rms_us": 3.0,
                "start_ns": 80_000,
                "end_ns": 130_000,
                "kernel_count": 1,
                "rms_kernel_count": 1,
            },
        ])

        agg = parsed[(17, 4, 16, 0)]
        self.assertEqual(agg["gpu_us"], 150.0)
        self.assertEqual(agg["rms_us"], 8.0)
        self.assertEqual(agg["span_us"], 129.0)
        self.assertEqual(agg["kernel_count"], 3)
        self.assertEqual(agg["rms_kernel_count"], 2)

    def test_latency_source_can_select_span_or_gpu_sum(self):
        agg = {"gpu_us": 150.0, "span_us": 129.0}

        self.assertEqual(cl._latency_us_from_agg(agg, "span"), 129.0)
        self.assertEqual(cl._latency_us_from_agg(agg, "gpu"), 150.0)
        self.assertEqual(cl._latency_us_from_agg(agg, "gpu_capped"), 129.0)
        self.assertEqual(
            cl._latency_us_from_agg({"gpu_us": 100.0, "span_us": 129.0}, "gpu_capped"),
            100.0,
        )

    def test_parser_accepts_scheduler_timing_latency_source(self):
        parser = cl._build_arg_parser()

        args = parser.parse_args(["--latency-source", "schedule_to_update"])

        self.assertEqual(args.latency_source, "schedule_to_update")

    def test_scheduler_timing_latency_source_is_context_only(self):
        self.assertEqual(
            cl._effective_latency_source("schedule_to_update", cl.DataPoint("ctx", 1, 128, 0)),
            "schedule_to_update",
        )
        self.assertEqual(
            cl._effective_latency_source("schedule_to_update", cl.DataPoint("gen", 8, 1, 4096)),
            "span",
        )

    def test_auto_latency_source_selects_span_and_moe_high_batch_gpu(self):
        self.assertEqual(
            cl._effective_latency_source("auto", cl.DataPoint("ctx", 1, 128, 0)),
            "span",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", cl.DataPoint("ctx", 1, 1024, 0), includes_moe=True),
            "span",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", cl.DataPoint("gen", 4, 1, 4096), includes_moe=True),
            "span",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", cl.DataPoint("gen", 8, 1, 4096), includes_moe=True),
            "gpu",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", cl.DataPoint("gen", 8, 1, 4096), includes_moe=False),
            "span",
        )

    def test_scheduler_timing_lookup_matches_datapoint_shape(self):
        events = [
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 128,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 0,
                "schedule_to_update_ms": 10.0,
                "fpm_wall_time_ms": 8.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 128,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 1,
                "schedule_to_update_ms": 14.0,
                "fpm_wall_time_ms": 12.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 128,
                "control_bs": 1,
                "control_past": 0,
                "control_run": None,
                "schedule_to_update_ms": 100.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 256,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 0,
                "schedule_to_update_ms": 99.0,
            },
        ]

        aggs = cl._lookup_scheduler_timing_aggs(events, "wu", cl.DataPoint("ctx", 1, 128, 0))
        latency_us, _rms_us, kernel_count, _rms_kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="span",
            aggregation="mean",
        )

        self.assertEqual(measure_count, 2)
        self.assertEqual(kernel_count, 0)
        self.assertEqual(latency_us, 10_000.0)

    def test_scheduler_timing_lookup_sums_chunked_context_by_run(self):
        events = [
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 4096,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 0,
                "fpm_wall_time_ms": 220.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 4096,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 0,
                "fpm_wall_time_ms": 87.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 4096,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 1,
                "fpm_wall_time_ms": 221.0,
            },
            {
                "event": "scheduler_update_wall_time",
                "work_unit_id": "wu",
                "control_phase": "ctx",
                "control_step": 4096,
                "control_bs": 1,
                "control_past": 0,
                "control_run": 1,
                "fpm_wall_time_ms": 88.0,
            },
        ]

        aggs = cl._lookup_scheduler_timing_aggs(events, "wu", cl.DataPoint("ctx", 1, 4096, 0))
        latency_us, _rms_us, _kernel_count, _rms_kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="span",
            aggregation="mean",
        )

        self.assertEqual(measure_count, 2)
        self.assertEqual(latency_us, 308_000.0)

    def test_repeat_aggregation_selects_median_latency(self):
        parsed = cl._aggregate_step_rows([
            {
                "step": 64,
                "batch_size": 1,
                "past_kv": 0,
                "measure_run": 0,
                "gpu_us": 100.0,
                "start_ns": 0,
                "end_ns": 100_000,
                "kernel_count": 1,
            },
            {
                "step": 64,
                "batch_size": 1,
                "past_kv": 0,
                "measure_run": 1,
                "gpu_us": 200.0,
                "start_ns": 0,
                "end_ns": 200_000,
                "kernel_count": 2,
            },
            {
                "step": 64,
                "batch_size": 1,
                "past_kv": 0,
                "measure_run": 2,
                "gpu_us": 300.0,
                "start_ns": 0,
                "end_ns": 300_000,
                "kernel_count": 3,
            },
        ])

        aggs = cl._lookup_aggs(parsed, (64, 1, 0))
        latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="span",
            aggregation="median",
        )

        self.assertEqual(latency_us, 200.0)
        self.assertEqual(rms_us, 0.0)
        self.assertEqual(kernel_count, 2)
        self.assertEqual(rms_kernel_count, 0)
        self.assertEqual(measure_count, 3)

    def test_repeat_aggregation_can_drop_min_and_max_then_average(self):
        aggs = [
            {"gpu_us": value, "span_us": value, "kernel_count": 1}
            for value in [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0]
        ]

        latency_us, rms_us, kernel_count, rms_kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="gpu_capped",
            aggregation="trimmed_mean",
        )

        self.assertEqual(latency_us, 35.0)
        self.assertEqual(rms_us, 0.0)
        self.assertEqual(kernel_count, 1)
        self.assertEqual(rms_kernel_count, 0)
        self.assertEqual(measure_count, 6)

    def test_trimmed_mean_falls_back_to_mean_for_partial_parse(self):
        aggs = [
            {"gpu_us": 10.0, "span_us": 10.0, "rms_us": 1.0, "kernel_count": 1},
            {"gpu_us": 30.0, "span_us": 30.0, "rms_us": 3.0, "kernel_count": 1},
        ]

        latency_us, rms_us, _kernel_count, _rms_kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="gpu_capped",
            aggregation="trimmed_mean",
        )

        self.assertEqual(latency_us, 20.0)
        self.assertEqual(rms_us, 2.0)
        self.assertEqual(measure_count, 2)

    def test_dummy_prompts_are_random_token_ids(self):
        token_config = cl.RandomPromptTokenConfig(
            vocab_size=128,
            excluded_token_ids=frozenset({0, 1, 2, 127}),
        )
        prompts = cl._dummy_prompts(1, 64, token_config)

        ids = prompts[0]["prompt_token_ids"]
        self.assertEqual(len(ids), 64)
        self.assertGreater(len(set(ids)), 1)
        self.assertTrue(all(0 <= token_id < 128 for token_id in ids))
        self.assertTrue(all(token_id not in token_config.excluded_token_ids for token_id in ids))

    def test_prompt_token_config_excludes_special_ids(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            (model_dir / "config.json").write_text(json.dumps({
                "vocab_size": 16,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }))
            (model_dir / "tokenizer_config.json").write_text(json.dumps({
                "added_tokens_decoder": {
                    "3": {"special": True},
                    "4": {"special": False},
                },
            }))

            token_config = load_random_prompt_token_config(str(model_dir))

            self.assertEqual(token_config.vocab_size, 16)
            self.assertEqual(token_config.excluded_token_ids, frozenset({1, 2, 3}))

    def test_prefix_suffix_prompts_reuse_prefix_and_vary_suffix(self):
        token_config = cl.RandomPromptTokenConfig(
            vocab_size=128,
            excluded_token_ids=frozenset({0, 1, 2, 127}),
        )
        prompt_factory = cl.PromptTokenFactory(seed=123)

        first = cl._prefix_suffix_prompts(
            1,
            prefix_len=16,
            suffix_len=4,
            token_config=token_config,
            prompt_factory=prompt_factory,
            prefix_stream_key_prefix=("ctx-prefix", "wu", 1, 16),
            cache_salt_prefix="ctx-prefix",
        )[0]
        second = cl._prefix_suffix_prompts(
            1,
            prefix_len=16,
            suffix_len=4,
            token_config=token_config,
            prompt_factory=prompt_factory,
            prefix_stream_key_prefix=("ctx-prefix", "wu", 1, 16),
            cache_salt_prefix="ctx-prefix",
        )[0]

        self.assertEqual(first["prompt_token_ids"][:16], second["prompt_token_ids"][:16])
        self.assertNotEqual(first["prompt_token_ids"][16:], second["prompt_token_ids"][16:])
        self.assertEqual(first["cache_salt"], second["cache_salt"])

    def test_prompt_factory_seed_controls_reproducibility(self):
        token_config = cl.RandomPromptTokenConfig(
            vocab_size=128,
            excluded_token_ids=frozenset({0, 1, 2, 127}),
        )

        first = cl.PromptTokenFactory(seed=123).sample(32, token_config)
        second = cl.PromptTokenFactory(seed=123).sample(32, token_config)
        third = cl.PromptTokenFactory(seed=456).sample(32, token_config)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_ctx_prefix_cache_key_is_shared_across_new_tokens_and_runs(self):
        first = cl.DataPoint("ctx", 1, 16, 4096)
        second = cl.DataPoint("ctx", 1, 128, 4096)

        self.assertEqual(
            cl._ctx_prefix_stream_key("wu", first),
            cl._ctx_prefix_stream_key("wu", second),
        )
        self.assertEqual(
            cl._ctx_cache_salt_prefix("wu", first),
            cl._ctx_cache_salt_prefix("wu", second),
        )

    def test_gen_prefix_cache_key_is_shared_across_batch_and_past(self):
        first = cl.DataPoint("gen", 1, 1, 128)
        second = cl.DataPoint("gen", 4, 1, 4096)

        self.assertEqual(
            cl._gen_prefix_stream_key("wu"),
            cl._gen_prefix_stream_key("wu"),
        )
        self.assertEqual(
            cl._gen_cache_salt_prefix("wu", first),
            cl._gen_cache_salt_prefix("wu", second),
        )

    def test_live_decode_past_lengths_support_staggered_diagnostic(self):
        dp = cl.DataPoint("gen", 4, 1, 4097)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LAYERWISE_LIVE_DECODE_PAST_OFFSETS", None)
            self.assertEqual(cl._live_decode_past_lengths(dp), [4097, 4097, 4097, 4097])

        with mock.patch.dict(os.environ, {"LAYERWISE_LIVE_DECODE_PAST_OFFSETS": "-1,0,0,1"}):
            self.assertEqual(cl._live_decode_past_lengths(dp), [4096, 4097, 4097, 4098])

        with (
            mock.patch.dict(os.environ, {"LAYERWISE_LIVE_DECODE_PAST_OFFSETS": "0,1"}),
            self.assertRaises(ValueError),
        ):
            cl._live_decode_past_lengths(dp)

    def test_live_step_driver_is_opt_in(self):
        dp = cl.DataPoint("gen", 1, 1, 4096)

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(cl._use_live_step_driver(dp))

        with mock.patch.dict(os.environ, {"LAYERWISE_USE_LIVE_STEP_DRIVER": "1"}, clear=True):
            self.assertTrue(cl._use_live_step_driver(dp))

    def test_oom_pruning_is_phase_local_and_monotonic(self):
        failed_ctx = cl.DataPoint("ctx", 1, 128, 0)
        self.assertTrue(cl.oom_dominates(failed_ctx, cl.DataPoint("ctx", 1, 256, 0)))
        self.assertFalse(cl.oom_dominates(failed_ctx, cl.DataPoint("ctx", 1, 64, 0)))
        self.assertFalse(cl.oom_dominates(failed_ctx, cl.DataPoint("gen", 1, 1, 128)))

        failed_gen = cl.DataPoint("gen", 4, 1, 128)
        self.assertTrue(cl.oom_dominates(failed_gen, cl.DataPoint("gen", 8, 1, 128)))
        self.assertTrue(cl.oom_dominates(failed_gen, cl.DataPoint("gen", 4, 1, 256)))
        self.assertFalse(cl.oom_dominates(failed_gen, cl.DataPoint("gen", 2, 1, 256)))

    def test_status_store_manifest_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 1, 0)])
            store = cl.StatusStore(tmp_path / "work")

            store.write_missing_manifest([unit])
            store.write_missing_manifest([unit])

            rows = [line for line in store.manifest_path.read_text().splitlines() if line]
            self.assertEqual(len(rows), 1)
            self.assertIn("wu_test:ctx:bs1:new1:past0", rows[0])

    def test_scheduler_pending_and_oom_skip_status(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dps = [
                cl.DataPoint("gen", 1, 1, 16),
                cl.DataPoint("gen", 2, 1, 16),
                cl.DataPoint("gen", 2, 1, 32),
            ]
            unit = _unit(tmp_path, dps)
            scheduler = cl.Scheduler(_args(tmp_path), [unit])
            scheduler.store.write_missing_manifest([unit])

            failed_id = dps[1].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event(
                "failed_oom",
                work_unit_id=unit.work_unit_id,
                datapoint_id=failed_id,
            )
            scheduler._mark_oom_dominated(unit, dps[1], failed_id)

            pending_ids = {
                dp.datapoint_id(unit.work_unit_id)
                for dp in scheduler._pending_datapoints(unit)
            }
            self.assertIn(dps[0].datapoint_id(unit.work_unit_id), pending_ids)
            self.assertNotIn(failed_id, pending_ids)
            self.assertNotIn(dps[2].datapoint_id(unit.work_unit_id), pending_ids)

    def test_generic_crash_marks_failed_error_not_fatal_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dps = [cl.DataPoint("gen", 1, 1, 16)]
            unit = _unit(tmp_path, dps)
            scheduler = cl.Scheduler(_args(tmp_path), [unit])
            dpid = dps[0].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event("started", work_unit_id=unit.work_unit_id, datapoint_id=dpid)
            attempt = cl.Attempt(
                work_unit=unit,
                gpu="0",
                attempt_id=1,
                spec_path=tmp_path / "spec.json",
                report_base=tmp_path / "report",
                stdout_path=tmp_path / "out",
                stderr_path=tmp_path / "err",
                process=None,
                stdout_handle=None,
                stderr_handle=None,
                pending_ids={dpid},
            )

            scheduler._mark_crashed_attempt(
                attempt,
                1,
                "RuntimeError: Check failed: args->intermediate_size % 128 == 0",
                successes=0,
            )

            events = scheduler.store.load_events()
            self.assertTrue(any(event["event"] == "failed_error" for event in events))
            self.assertFalse(any(event["event"] == "failed_fatal_cuda" for event in events))

    def test_repeated_identical_crash_omits_work_unit_with_error_log(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dps = [
                cl.DataPoint("gen", 1, 1, 16),
                cl.DataPoint("gen", 2, 1, 16),
            ]
            unit = _unit(tmp_path, dps)
            scheduler = cl.Scheduler(_args(tmp_path), [unit])
            pending_ids = {dp.datapoint_id(unit.work_unit_id) for dp in dps}
            stderr_tail = "RuntimeError: Check failed: args->intermediate_size % 128 == 0"
            for attempt_id in range(1, 4):
                attempt = cl.Attempt(
                    work_unit=unit,
                    gpu="0",
                    attempt_id=attempt_id,
                    spec_path=tmp_path / f"spec{attempt_id}.json",
                    report_base=tmp_path / f"report{attempt_id}",
                    stdout_path=tmp_path / f"out{attempt_id}",
                    stderr_path=tmp_path / f"err{attempt_id}",
                    process=None,
                    stdout_handle=None,
                    stderr_handle=None,
                    pending_ids=pending_ids,
                )
                scheduler._mark_crashed_attempt(attempt, 1, stderr_tail, successes=0)

            events = scheduler.store.load_events()
            omitted = [event for event in events if event["event"] == "work_unit_omitted"]
            skipped = [event for event in events if event["event"] == "skipped_same_error"]
            self.assertEqual(len(omitted), 1)
            self.assertIn("intermediate_size", omitted[0]["stderr_tail"])
            self.assertEqual({event["datapoint_id"] for event in skipped}, pending_ids)

    def test_clean_exit_distinguishes_unparsed_from_unstarted_datapoints(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dps = [
                cl.DataPoint("gen", 1, 1, 16),
                cl.DataPoint("gen", 1, 1, 32),
            ]
            unit = _unit(tmp_path, dps)
            scheduler = cl.Scheduler(_args(tmp_path), [unit])
            pending_ids = {dp.datapoint_id(unit.work_unit_id) for dp in dps}
            completed_id = dps[0].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event(
                "completed_execution",
                work_unit_id=unit.work_unit_id,
                datapoint_id=completed_id,
            )
            attempt = cl.Attempt(
                work_unit=unit,
                gpu="0",
                attempt_id=1,
                spec_path=tmp_path / "spec.json",
                report_base=tmp_path / "report",
                stdout_path=tmp_path / "out",
                stderr_path=tmp_path / "err",
                process=None,
                stdout_handle=None,
                stderr_handle=None,
                pending_ids=pending_ids,
            )

            scheduler._mark_clean_parse_failures(attempt)

            terminal = {
                event["datapoint_id"]: event["event"]
                for event in scheduler.store.load_events()
                if event.get("datapoint_id") in pending_ids
                and event.get("event") in {"failed_parse", "skipped_not_started"}
            }
            self.assertEqual(terminal[completed_id], "failed_parse")
            self.assertEqual(
                terminal[dps[1].datapoint_id(unit.work_unit_id)],
                "skipped_not_started",
            )

    def test_scheduler_adds_gpt_oss_runtime_defaults_for_decode_only(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)])
            unit.row_base["model"] = "openai/gpt-oss-120b"
            unit.row_base["system"] = "b300_sxm"
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)
            args = spec["extra_vllm_args"]

            self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
            self.assertEqual(args[args.index("--max-cudagraph-capture-size") + 1], "2048")
            self.assertEqual(args[args.index("--stream-interval") + 1], "20")
            self.assertIn("--enable-prefix-caching", args)
            self.assertNotIn("--no-enable-prefix-caching", args)
            self.assertNotIn("--tensor-parallel-size", args)
            self.assertNotIn("--enable-expert-parallel", args)

    def test_scheduler_enables_prefix_cache_for_gen(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)])
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertIn("--enable-prefix-caching", spec["extra_vllm_args"])
            self.assertNotIn("--no-enable-prefix-caching", spec["extra_vllm_args"])

    def test_scheduler_spec_keeps_simulated_tp_out_of_runtime_engine(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            unit.row_base["attn_tp"] = 2
            unit.row_base["moe_tp"] = 2
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertNotIn("tensor_parallel_size", spec)
            self.assertNotIn("--tensor-parallel-size", spec["extra_vllm_args"])

    def test_worker_cmd_defaults_to_full_capture(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            scheduler = cl.Scheduler(_args(tmp_path), [_unit(tmp_path, [])])

            cmd = scheduler._worker_cmd(tmp_path / "spec.json", tmp_path / "report")

            self.assertNotIn("--capture-range=cudaProfilerApi", cmd)
            self.assertNotIn("--capture-range-end=stop", cmd)

    def test_worker_cmd_can_use_cuda_profiler_capture(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.nsys_capture = "cuda_profiler_api"
            scheduler = cl.Scheduler(args, [_unit(tmp_path, [])])

            cmd = scheduler._worker_cmd(tmp_path / "spec.json", tmp_path / "report")

            self.assertIn("--capture-range=cudaProfilerApi", cmd)
            self.assertIn("--capture-range-end=stop", cmd)

    def test_cuda_profiler_capture_rejects_per_layer_sources(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.nsys_capture = "cuda_profiler_api"
            unit = _unit(tmp_path, [cl.DataPoint("gen", 1, 1, 4096)])

            with self.assertRaisesRegex(ValueError, "per-layer latency sources"):
                cl.Scheduler(args, [unit])

    def test_scheduler_adds_default_vllm_extra_args(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)])
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)
            extra = spec["extra_vllm_args"]

            self.assertIn("--skip-mm-profiling", extra)
            self.assertEqual(extra[extra.index("--limit-mm-per-prompt") + 1], '{"image":0,"video":0}')
            self.assertEqual(extra[extra.index("--generation-config") + 1], "vllm")

    def test_scheduler_preserves_explicit_default_vllm_extra_arg_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)])
            args = _args(tmp_path)
            args.extra_vllm_arg = [
                "--generation-config=auto",
                "--limit-mm-per-prompt={\"image\":1}",
                "--no-skip-mm-profiling",
            ]
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)
            extra = spec["extra_vllm_args"]

            self.assertNotIn("--skip-mm-profiling", extra)
            self.assertIn("--no-skip-mm-profiling", extra)
            self.assertNotIn("--generation-config", extra)
            self.assertIn("--generation-config=auto", extra)
            self.assertNotIn("--limit-mm-per-prompt", extra)
            self.assertIn("--limit-mm-per-prompt={\"image\":1}", extra)

    def test_scheduler_keeps_prefix_cache_for_gpt_oss_prefix_context(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 4096)])
            unit.row_base["model"] = "openai/gpt-oss-120b"
            unit.row_base["system"] = "b300_sxm"
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertNotIn("--no-enable-prefix-caching", spec["extra_vllm_args"])
            self.assertIn("--enable-prefix-caching", spec["extra_vllm_args"])

    def test_scheduler_does_not_enable_prefix_cache_for_zero_past_context(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertNotIn("--enable-prefix-caching", spec["extra_vllm_args"])
            self.assertNotIn("--no-enable-prefix-caching", spec["extra_vllm_args"])

    def test_scheduler_enables_prefix_cache_for_context_with_past_kv(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 4096)])
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertIn("--enable-prefix-caching", spec["extra_vllm_args"])
            self.assertNotIn("--no-enable-prefix-caching", spec["extra_vllm_args"])

    def test_scheduler_live_decode_keeps_prefix_cache_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 8, 1, 4096)])
            unit = dataclasses.replace(unit, gen_driver="live_decode")
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertEqual(spec["gen_driver"], "live_decode")
            self.assertNotIn("--enable-prefix-caching", spec["extra_vllm_args"])
            self.assertIn("--no-enable-prefix-caching", spec["extra_vllm_args"])

    def test_scheduler_preserves_explicit_gpt_oss_vllm_overrides(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("gen", 4, 1, 4096)])
            unit.row_base["model"] = "openai/gpt-oss-120b"
            unit.row_base["system"] = "b300_sxm"
            args = _args(tmp_path)
            args.extra_vllm_arg = [
                "--kv-cache-dtype=bf16",
                "--stream-interval=7",
                "--max-cudagraph-capture-size=1024",
                "--enable-prefix-caching",
            ]
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)
            extra = spec["extra_vllm_args"]

            self.assertNotIn("--kv-cache-dtype", extra)
            self.assertIn("--kv-cache-dtype=bf16", extra)
            self.assertNotIn("--stream-interval", extra)
            self.assertIn("--stream-interval=7", extra)
            self.assertNotIn("--max-cudagraph-capture-size", extra)
            self.assertIn("--max-cudagraph-capture-size=1024", extra)
            self.assertNotIn("--no-enable-prefix-caching", extra)

    def test_engine_tokens_let_vllm_pick_deployment_defaults(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[
                cl.DataPoint("ctx", 1, 128, 0),
                cl.DataPoint("gen", 4, 1, 16),
                cl.DataPoint("gen", 2, 1, 32),
            ],
            extra_vllm_args=[],
        )

        self.assertNotIn("--compilation-config", tokens)
        self.assertNotIn("--max-model-len", tokens)
        self.assertNotIn("--max-num-batched-tokens", tokens)
        self.assertNotIn("--max-num-seqs", tokens)

    def test_engine_tokens_do_not_enable_runtime_tensor_parallel(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 128, 0)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--tensor-parallel-size", tokens)
        self.assertNotIn("--worker-extension-cls", tokens)

    def test_engine_tokens_context_does_not_inject_token_budget(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 3, 0)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--max-num-batched-tokens", tokens)

    def test_engine_tokens_prefix_cache_does_not_inject_deployment_limits(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 32, 1, 65536)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--max-model-len", tokens)
        self.assertNotIn("--max-num-seqs", tokens)
        self.assertNotIn("--max-num-batched-tokens", tokens)

    def test_engine_tokens_live_decode_preserves_explicit_max_num_seqs_only(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 8, 1, 4096)],
            extra_vllm_args=[],
            max_num_seqs=16,
            gen_driver="live_decode",
        )

        self.assertEqual(tokens[tokens.index("--max-num-seqs") + 1], "16")
        self.assertNotIn("--max-model-len", tokens)
        self.assertNotIn("--max-num-batched-tokens", tokens)

    def test_engine_tokens_honor_explicit_scheduler_token_budget(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 8, 1, 4096)],
            extra_vllm_args=[],
            max_num_seqs=16,
            max_num_batched_tokens=4096,
            gen_driver="live_decode",
        )

        self.assertEqual(tokens[tokens.index("--max-num-seqs") + 1], "16")
        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "4096")

    def test_engine_tokens_prefix_cache_does_not_inject_block_budget(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 1, 1, 128)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--max-num-batched-tokens", tokens)

    def test_engine_tokens_context_past_kv_does_not_inject_model_len(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 8192, 8192)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--max-model-len", tokens)
        self.assertNotIn("--max-num-batched-tokens", tokens)

    def test_context_marker_iteration_uses_prefix_cache_measure_step(self):
        self.assertEqual(
            cl._ctx_marker_iteration(
                cl.DataPoint("ctx", 1, 8192, 0),
                8192,
            ),
            1,
        )
        self.assertEqual(
            cl._ctx_marker_iteration(
                cl.DataPoint("ctx", 1, 8192, 8192),
                8192,
            ),
            1,
        )

    def test_model_max_len_filter_accounts_for_generated_token(self):
        datapoints = [
            cl.DataPoint("ctx", 1, 8, 31),
            cl.DataPoint("ctx", 1, 8, 32),
            cl.DataPoint("gen", 1, 1, 38),
            cl.DataPoint("gen", 1, 1, 39),
        ]

        filtered, skipped = cl._filter_datapoints_for_model_max_len(datapoints, 40)

        self.assertEqual(skipped, 2)
        self.assertEqual(
            [dp.shape_key for dp in filtered],
            ["ctx:bs1:new8:past31", "gen:bs1:new1:past38"],
        )

    def test_build_work_units_filters_context_grid_by_model_max_len(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "max_position_embeddings": 10,
            }
            args = _build_args(
                tmp_path,
                phases="ctx",
                tp_sizes="1",
                ctx_new_tokens="1,2,8",
                ctx_past_kv="0,8",
            )

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_work_units(args)

            self.assertEqual(
                [dp.shape_key for dp in units[0].datapoints],
                [
                    "ctx:bs1:new1:past0",
                    "ctx:bs1:new2:past0",
                    "ctx:bs1:new8:past0",
                    "ctx:bs1:new1:past8",
                ],
            )

    def test_dense_build_work_units_do_not_emit_moe_ep_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
            }
            args = _build_args(tmp_path, tp_sizes="1,2,4", moe_tp=2, num_slots=288)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual([u.row_base["attn_tp"] for u in units], [1, 2, 4])
            self.assertEqual([u.row_base["moe_tp"] for u in units], [1, 1, 1])
            self.assertEqual([u.row_base["ep"] for u in units], [1, 1, 1])
            self.assertEqual([u.row_base["num_slots"] for u in units], ["", "", ""])
            self.assertEqual(
                [call.kwargs["ep"] for call in patcher.mock_calls],
                [1, 1, 1],
            )
            self.assertEqual(
                [call.kwargs["num_slots"] for call in patcher.mock_calls],
                [None, None, None],
            )

    def test_build_work_units_preserves_resolved_max_num_seqs_for_filtered_decode(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
            }
            args = _build_args(
                tmp_path,
                phases="gen",
                tp_sizes="1",
                gen_batch_sizes="8",
                gen_past_kv="4096",
                max_num_seqs=16,
            )

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].max_num_seqs, 16)

    def test_build_work_units_preserves_explicit_max_num_batched_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
            }
            args = _build_args(
                tmp_path,
                phases="gen",
                tp_sizes="1",
                gen_batch_sizes="8",
                gen_past_kv="4096",
                max_num_batched_tokens=4096,
            )

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].max_num_batched_tokens, 4096)

    def test_public_split_phase_context_leaves_max_model_len_to_vllm_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
            }
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--ctx-new-tokens",
                "128,1024",
                "--gen-past-kv",
                "4096",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(1,))

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertIsNone(units[0].max_model_len)

    def test_public_max_model_len_override_wins_over_deployment_length(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
            }
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "smoke",
                "--ctx-new-tokens",
                "128,1024",
                "--max-model-len",
                "4104",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(1,))

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].max_model_len, 4104)

    def test_public_default_max_model_len_is_not_injected_after_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 4,
                "max_position_embeddings": 40960,
            }
            args = cl._build_arg_parser().parse_args([
                "--models",
                "model",
                "--tp-sizes",
                "1",
                "--ep-sizes",
                "1",
                "--phases",
                "ctx",
                "--run-preset",
                "full",
            ])
            args.run_dir = tmp_path
            args.work_dir = str(tmp_path / "profiles")
            args.config_cache_dir = str(tmp_path / "config_cache")
            model = LayerwiseModel(model="model", kind="dense", tp_sizes=(1,))

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_public_work_units(args, [model])

            self.assertEqual(len(units), 1)
            self.assertIsNone(units[0].max_model_len)
            self.assertTrue(
                all(
                    dp.past_kv + dp.new_tokens + 1 <= dense_config["max_position_embeddings"]
                    for dp in units[0].datapoints
                )
            )

    def test_dense_build_work_units_can_keep_multiple_initial_layers(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
            }
            args = _build_args(tmp_path, phases="ctx", target_layer_count=4)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].target_layers, [0, 1, 2, 3])
            self.assertEqual(units[0].representative.layer_type, "dense")
            self.assertEqual(units[0].representative.measured_layer_count, 4)
            self.assertEqual(units[0].representative.layer_multiplier, 4)
            self.assertEqual(units[0].model_layer_count, 4)
            self.assertTrue(units[0].uses_full_layer_depth())
            self.assertEqual(patcher.mock_calls[0].kwargs["num_hidden_layers"], 4)

    def test_dense_representative_layer_scales_to_model_layer_count(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 64,
            }
            args = _build_args(tmp_path, phases="ctx", target_layer_count=1)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].representative.layer_type, "dense")
            self.assertEqual(units[0].representative.measured_layer_count, 1)
            self.assertEqual(units[0].representative.layer_multiplier, 64)
            self.assertEqual(units[0].model_layer_count, 1)
            self.assertTrue(units[0].uses_full_layer_depth())

    def test_dense_build_work_units_can_keep_explicit_target_layer(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 64,
            }
            args = _build_args(tmp_path, phases="ctx", target_layers="1")

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].target_layers, [1])
            self.assertEqual(units[0].representative.layer_index, 1)
            self.assertEqual(units[0].representative.measured_layer_count, 1)
            self.assertEqual(units[0].representative.layer_multiplier, 1)
            self.assertEqual(units[0].model_layer_count, 2)
            self.assertFalse(units[0].uses_full_layer_depth())
            self.assertEqual(patcher.mock_calls[0].kwargs["num_hidden_layers"], 2)

    def test_dense_build_work_units_can_keep_explicit_layer_with_deeper_config(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            dense_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_hidden_layers": 64,
            }
            args = _build_args(
                tmp_path,
                phases="ctx",
                target_layers="1",
                target_layer_config_depth=64,
            )

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=dense_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].target_layers, [1])
            self.assertEqual(units[0].model_layer_count, 64)
            self.assertFalse(units[0].uses_full_layer_depth())
            self.assertEqual(patcher.mock_calls[0].kwargs["num_hidden_layers"], 64)

    def test_moe_build_work_units_keep_ep_and_num_slots_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_experts": 8,
                "moe_intermediate_size": 512,
            }
            args = _build_args(tmp_path, tp_sizes="2,4", moe_tp=2, num_slots=8)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual([u.row_base["attn_tp"] for u in units], [2, 4])
            self.assertEqual([u.row_base["moe_tp"] for u in units], [2, 2])
            self.assertEqual([u.row_base["ep"] for u in units], [1, 2])
            self.assertEqual([u.row_base["num_slots"] for u in units], [8, 8])
            self.assertEqual(
                [call.kwargs["ep"] for call in patcher.mock_calls],
                [1, 2],
            )
            self.assertEqual(
                [call.kwargs["num_slots"] for call in patcher.mock_calls],
                [8, 8],
            )

    def test_num_local_experts_build_work_units_as_moe(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_local_experts": 8,
                "moe_intermediate_size": 512,
            }
            args = _build_args(tmp_path, tp_sizes="2,4", moe_tp=2, num_slots=8)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual([u.row_base["attn_tp"] for u in units], [2, 4])
            self.assertEqual([u.row_base["moe_tp"] for u in units], [2, 2])
            self.assertEqual([u.row_base["ep"] for u in units], [1, 2])
            self.assertEqual([u.row_base["num_slots"] for u in units], [8, 8])
            self.assertEqual(
                [call.kwargs["ep"] for call in patcher.mock_calls],
                [1, 2],
            )

    def test_moe_noop_preserves_layer_type_representatives(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            moe_config = {
                "architectures": ["GptOssForCausalLM"],
                "model_type": "gpt_oss",
                "layer_types": ["sliding_attention", "full_attention"],
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_local_experts": 8,
                "num_experts_per_tok": 4,
            }
            args = _build_args(tmp_path, tp_sizes="2,4", moe_tp=1, moe_noop=True)

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual([u.target_layers for u in units], [[0], [1], [0], [1]])
            self.assertEqual(
                [u.representative.layer_type for u in units],
                ["sliding_attention_moe", "full_attention_moe", "sliding_attention_moe", "full_attention_moe"],
            )
            self.assertEqual([u.representative.layer_multiplier for u in units], [1, 1, 1, 1])
            self.assertEqual([u.moe_noop for u in units], [True, True, True, True])
            self.assertEqual(
                [call.kwargs["extra_overrides"] for call in patcher.mock_calls],
                [
                    {"layer_types": ["sliding_attention", "full_attention"]},
                    {"layer_types": ["sliding_attention", "full_attention"]},
                ],
            )
            self.assertEqual(
                [call.kwargs["num_hidden_layers"] for call in patcher.mock_calls],
                [2, 2],
            )

    def test_explicit_context_envelope_uses_one_full_depth_unit(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            moe_config = {
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_experts": 8,
                "moe_intermediate_size": 512,
                "num_hidden_layers": 4,
                "layer_types": ["linear_attention", "linear_attention", "full_attention", "full_attention"],
            }
            args = _build_args(
                tmp_path,
                phases="ctx",
                tp_sizes="1",
                target_layer_count=4,
                latency_source="schedule_to_update",
            )

            with (
                mock.patch.object(dpg, "_load_original_config", return_value=moe_config),
                mock.patch.object(dpg, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(len(units), 1)
            self.assertEqual(units[0].target_layers, [0, 1, 2, 3])
            self.assertEqual(units[0].representative.measured_layer_count, 4)
            self.assertEqual(units[0].representative.layer_multiplier, 4)
            self.assertEqual(units[0].model_layer_count, 4)
            self.assertTrue(units[0].uses_full_layer_depth())
            self.assertEqual(patcher.mock_calls[0].kwargs["num_hidden_layers"], 4)

    def test_scheduler_spec_includes_moe_noop(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = cl.WorkUnit(
                work_unit_id="wu_moe",
                model_dir=str(tmp_path / "model"),
                row_base=_unit(tmp_path, []).row_base,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=1,
                    layer_multiplier=1,
                    target_layers=(0,),
                ),
                target_layers=[0],
                datapoints=[cl.DataPoint("gen", 1, 1, 16)],
                model_layer_count=1,
                moe_noop=True,
            )
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["moe_noop"])
            self.assertEqual(spec["moe_weight_mode"], "noop")
            self.assertTrue(spec["enable_layer_patch"])

    def test_layerwise_csv_schema_includes_moe_weight_mode(self):
        self.assertIn("moe_weight_mode", cl.CSV_COLUMNS)
        self.assertLess(
            cl.CSV_COLUMNS.index("includes_moe"),
            cl.CSV_COLUMNS.index("moe_weight_mode"),
        )

    def test_scheduler_disables_layer_patch_for_full_depth_schedule_timing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "schedule_to_update"
            args.nsys_capture = "none"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                target_layers=[0, 1, 2, 3],
                model_layer_count=4,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=4,
                    layer_multiplier=4,
                    target_layers=(0, 1, 2, 3),
                ),
                includes_moe=True,
            )
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertFalse(spec["enable_layerwise_nvtx_tracing"])
            self.assertFalse(spec["enable_layer_patch"])
            self.assertFalse(spec["enable_step_marker"])

    def test_scheduler_keeps_layer_patch_for_router_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "schedule_to_update"
            args.nsys_capture = "none"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                target_layers=[0, 1, 2, 3],
                model_layer_count=4,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=4,
                    layer_multiplier=4,
                    target_layers=(0, 1, 2, 3),
                ),
                includes_moe=True,
                router_weight_model="model",
                physical_gpus=2,
            )
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["enable_layer_patch"])
            self.assertFalse(spec["enable_step_marker"])

    def test_scheduler_keeps_layer_patch_for_sampled_layer(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                target_layers=[0],
                model_layer_count=4,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=1,
                    layer_multiplier=4,
                    target_layers=(0,),
                ),
                includes_moe=True,
            )
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["enable_layer_patch"])
            self.assertTrue(spec["enable_step_marker"])

    def test_auto_context_uses_span_latency_source(self):
        small_ctx = cl.DataPoint("ctx", 1, 128, 0)
        large_ctx = cl.DataPoint("ctx", 1, 1024, 0)

        self.assertEqual(
            cl._effective_latency_source("auto", small_ctx, includes_moe=True),
            "span",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", small_ctx, includes_moe=False),
            "span",
        )
        self.assertEqual(
            cl._effective_latency_source("auto", large_ctx, includes_moe=False),
            "span",
        )

    def test_auto_context_keeps_step_marker_for_nsys_span_timing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            args.nsys_capture = "none"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                target_layers=[0, 1, 2, 3],
                model_layer_count=4,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="dense",
                    measured_layer_count=4,
                    layer_multiplier=4,
                    target_layers=(0, 1, 2, 3),
                ),
            )
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertEqual(
                cl._effective_latency_source("auto", unit.datapoints[0], includes_moe=False),
                "span",
            )
            self.assertTrue(spec["enable_step_marker"])

    def test_worker_wall_success_rows_are_not_representative_scaled(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "worker_wall"
            args.nsys_capture = "none"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=4,
                    layer_multiplier=40,
                    target_layers=(0, 1, 2, 3),
                ),
                target_layers=[0, 1, 2, 3],
                model_layer_count=4,
                includes_moe=True,
            )
            scheduler = cl.Scheduler(args, [unit])
            results._write_csv_header_if_needed(Path(args.output))
            dpid = unit.datapoints[0].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event(
                "measurement_wall_time",
                work_unit_id=unit.work_unit_id,
                datapoint_id=dpid,
                phase="ctx",
                batch_size=1,
                new_tokens=128,
                past_kv=0,
                run=0,
                wall_latency_ms=9.0,
            )
            attempt = cl.Attempt(
                work_unit=unit,
                gpu="0",
                attempt_id=1,
                spec_path=tmp_path / "spec.json",
                report_base=tmp_path / "report",
                stdout_path=tmp_path / "out",
                stderr_path=tmp_path / "err",
                process=None,
                stdout_handle=None,
                stderr_handle=None,
                pending_ids={dpid},
            )

            successes = scheduler._parse_scheduler_timing_only(attempt)

            self.assertEqual(successes, 1)
            with Path(args.output).open() as f:
                row = list(csv.DictReader(f))[0]
            self.assertEqual(row["latency_source"], "worker_wall")
            self.assertEqual(float(row["latency_ms"]), 9.0)
            self.assertEqual(row["layer_multiplier"], "4")

    def test_explicit_worker_wall_source_uses_worker_wall_events(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "worker_wall"
            args.nsys_capture = "none"
            unit = dataclasses.replace(
                _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)]),
                target_layers=[0, 1, 2, 3],
                model_layer_count=4,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="dense",
                    measured_layer_count=4,
                    layer_multiplier=4,
                    target_layers=(0, 1, 2, 3),
                ),
            )
            scheduler = cl.Scheduler(args, [unit])
            results._write_csv_header_if_needed(Path(args.output))
            dpid = unit.datapoints[0].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event(
                "measurement_wall_time",
                work_unit_id=unit.work_unit_id,
                datapoint_id=dpid,
                phase="ctx",
                batch_size=1,
                new_tokens=128,
                past_kv=0,
                run=0,
                wall_latency_ms=11.0,
            )
            attempt = cl.Attempt(
                work_unit=unit,
                gpu="0",
                attempt_id=1,
                spec_path=tmp_path / "spec.json",
                report_base=tmp_path / "report",
                stdout_path=tmp_path / "out",
                stderr_path=tmp_path / "err",
                process=None,
                stdout_handle=None,
                stderr_handle=None,
                pending_ids={dpid},
            )

            successes = scheduler._parse_scheduler_timing_only(attempt)

            self.assertEqual(successes, 1)
            with Path(args.output).open() as f:
                row = list(csv.DictReader(f))[0]
            self.assertEqual(row["latency_source"], "worker_wall")
            self.assertEqual(float(row["latency_ms"]), 11.0)

    def test_scheduler_resumes_after_existing_attempt_ids(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            store = cl.StatusStore(Path(args.work_dir))
            store.append_event("attempt_started", work_unit_id="wu_old", attempt_id=7)

            scheduler = cl.Scheduler(args, [_unit(tmp_path, [cl.DataPoint("gen", 1, 1, 16)])])

            self.assertEqual(scheduler.attempt_counter, 7)

    def test_auto_context_launch_does_not_enable_scheduler_timing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            scheduler = cl.Scheduler(args, [unit])
            fake_process = mock.Mock()

            with mock.patch.object(scheduler, "_worker_cmd", return_value=["worker"]), mock.patch(
                "collector.layerwise.vllm.scheduler.subprocess.Popen",
                return_value=fake_process,
            ) as popen:
                attempt = scheduler._launch_attempt(unit, "0", unit.datapoints)

            attempt.stdout_handle.close()
            attempt.stderr_handle.close()
            self.assertNotIn("LAYERWISE_SCHEDULER_TIMING", popen.call_args.kwargs["env"])
            self.assertEqual(popen.call_args.kwargs["env"]["DYN_FORWARDPASS_METRIC_PORT"], "20381")

    def test_auto_context_scheduler_spec_keeps_module_nvtx(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["enable_layerwise_nvtx_tracing"])

    def test_auto_context_attempt_uses_nsys_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            scheduler = cl.Scheduler(args, [unit], worker_entrypoint=tmp_path / "collect.py")

            cmd = scheduler._worker_cmd(
                tmp_path / "spec.json",
                tmp_path / "report",
                capture_nsys=scheduler._attempt_needs_nsys(unit, unit.datapoints),
            )

            self.assertEqual(cmd[:2], ["nsys", "profile"])

    def test_decode_span_scheduler_spec_keeps_module_nvtx(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("gen", 1, 1, 128)])
            scheduler = cl.Scheduler(args, [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["enable_layerwise_nvtx_tracing"])

    def test_auto_decode_attempt_uses_nsys_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("gen", 1, 1, 128)])
            scheduler = cl.Scheduler(args, [unit], worker_entrypoint=tmp_path / "collect.py")

            cmd = scheduler._worker_cmd(
                tmp_path / "spec.json",
                tmp_path / "report",
                capture_nsys=scheduler._attempt_needs_nsys(unit, unit.datapoints),
            )

            self.assertEqual(cmd[:2], ["nsys", "profile"])

    def test_auto_decode_launch_does_not_enable_scheduler_timing(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "auto"
            unit = _unit(tmp_path, [cl.DataPoint("gen", 1, 1, 4096)])
            scheduler = cl.Scheduler(args, [unit])
            fake_process = mock.Mock()

            with mock.patch.object(scheduler, "_worker_cmd", return_value=["worker"]), mock.patch(
                "collector.layerwise.vllm.scheduler.subprocess.Popen",
                return_value=fake_process,
            ) as popen:
                attempt = scheduler._launch_attempt(unit, "0", unit.datapoints)

            attempt.stdout_handle.close()
            attempt.stderr_handle.close()
            self.assertNotIn("LAYERWISE_SCHEDULER_TIMING", popen.call_args.kwargs["env"])
            self.assertEqual(popen.call_args.kwargs["env"]["DYN_FORWARDPASS_METRIC_PORT"], "20381")

    def test_schedule_to_update_success_rows_are_not_representative_scaled(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.latency_source = "schedule_to_update"
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            unit = dataclasses.replace(
                unit,
                representative=cl.RepresentativeLayer(
                    layer_index=0,
                    layer_type="moe",
                    measured_layer_count=1,
                    layer_multiplier=30,
                    target_layers=(0,),
                ),
            )
            scheduler = cl.Scheduler(args, [unit])
            results._write_csv_header_if_needed(Path(args.output))
            dpid = unit.datapoints[0].datapoint_id(unit.work_unit_id)
            scheduler.store.append_event(
                "scheduler_update_wall_time",
                work_unit_id=unit.work_unit_id,
                control_phase="ctx",
                control_step=128,
                control_bs=1,
                control_past=0,
                control_run=0,
                schedule_to_update_ms=7.0,
            )
            attempt = cl.Attempt(
                work_unit=unit,
                gpu="0",
                attempt_id=1,
                spec_path=tmp_path / "spec.json",
                report_base=tmp_path / "report",
                stdout_path=tmp_path / "out",
                stderr_path=tmp_path / "err",
                process=None,
                stdout_handle=None,
                stderr_handle=None,
                pending_ids={dpid},
            )
            attempt.report_base.with_suffix(".nsys-rep").write_text("rep")

            with mock.patch("collector.layerwise.vllm.scheduler.subprocess.run") as run, mock.patch(
                "collector.layerwise.vllm.scheduler.parse_step_sweep",
                return_value=([], {}),
            ):
                run.return_value = mock.Mock(returncode=0, stderr="")
                successes = scheduler._parse_attempt_report(attempt)

            self.assertEqual(successes, 1)
            with Path(args.output).open() as f:
                rows = list(__import__("csv").DictReader(f))
            self.assertEqual(rows[0]["latency_ms"], "7.0")
            self.assertEqual(rows[0]["measured_layer_count"], "1")
            self.assertEqual(rows[0]["layer_multiplier"], "1")

    def test_parallel_config_patch_treats_num_local_experts_as_moe(self):
        config = {
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 1024,
            "num_local_experts": 8,
            "moe_intermediate_size": 512,
        }

        with self.assertRaisesRegex(ValueError, "Constraint violated"):
            pcp.patch_for_parallelism(
                "model",
                attn_tp=4,
                moe_tp=1,
                ep=2,
                original_config=config,
            )

    def test_parallel_config_patch_divides_num_local_experts_for_ep(self):
        config = {
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 1024,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "experts_per_token": 4,
        }

        with mock.patch.object(pcp, "patch_model_path", return_value="patched") as patcher:
            result = pcp.patch_for_parallelism(
                "model",
                attn_tp=4,
                moe_tp=1,
                ep=4,
                original_config=config,
            )

        self.assertEqual(result, "patched")
        overrides = patcher.call_args.kwargs["overrides"]
        self.assertEqual(overrides["num_attention_heads"], 2)
        self.assertEqual(overrides["num_key_value_heads"], 2)
        self.assertEqual(overrides["intermediate_size"], 256)
        self.assertEqual(overrides["num_local_experts"], 32)
        self.assertNotIn("num_experts_per_tok", overrides)
        self.assertNotIn("experts_per_token", overrides)


if __name__ == "__main__":
    unittest.main()
