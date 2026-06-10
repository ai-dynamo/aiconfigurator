import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))

import parallel_config_patch as pcp
from collector.layerwise.vllm import collect as public_collect
from collector.layerwise.vllm import datapoint_generator as dpg
from collector.layerwise.vllm import engine
from collector.layerwise.vllm import nsys
from collector.layerwise.vllm import scheduler
from collector.layerwise.vllm import worker
from collector.layerwise.vllm.data import DataPoint, RepresentativeLayer, WorkUnit
from random_prompt_tokens import load_random_prompt_token_config

cl = SimpleNamespace(
    CTX_MAX_NUM_BATCHED_TOKENS_FLOOR=dpg.CTX_MAX_NUM_BATCHED_TOKENS_FLOOR,
    DataPoint=DataPoint,
    RandomPromptTokenConfig=worker.RandomPromptTokenConfig,
    RepresentativeLayer=RepresentativeLayer,
    Scheduler=scheduler.Scheduler,
    StatusIndex=scheduler.StatusIndex,
    StatusStore=scheduler.StatusStore,
    VLLM_DEFAULT_BLOCK_SIZE=dpg.VLLM_DEFAULT_BLOCK_SIZE,
    WorkUnit=WorkUnit,
    _aggregate_step_rows=nsys._aggregate_step_rows,
    _build_arg_parser=public_collect._build_arg_parser,
    _ctx_cache_salt_prefix=worker._ctx_cache_salt_prefix,
    _ctx_marker_iteration=worker._ctx_marker_iteration,
    _ctx_prefix_request_index_offset=worker._ctx_prefix_request_index_offset,
    _ctx_suffix_request_index_offset=worker._ctx_suffix_request_index_offset,
    _deterministic_prefix_suffix_prompts=worker._deterministic_prefix_suffix_prompts,
    _dummy_prompts=worker._dummy_prompts,
    _engine_tokens=engine._engine_tokens,
    _filter_datapoints_for_model_max_len=dpg._filter_datapoints_for_model_max_len,
    _gen_cache_salt_prefix=worker._gen_cache_salt_prefix,
    _gen_prompt_request_index_offset=worker._gen_prompt_request_index_offset,
    _latency_us_from_agg=nsys._latency_us_from_agg,
    _lookup_aggs=nsys._lookup_aggs,
    _reduce_agg_latency=nsys._reduce_agg_latency,
    build_work_units=dpg.build_work_units,
    oom_dominates=scheduler.oom_dominates,
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
        nsys_capture="cuda_profiler_api",
        extra_vllm_arg=[],
        latency_source="span",
        ctx_warmup_runs=0,
        ctx_measured_runs=1,
        ctx_repeat_aggregation="median",
        gen_warmup_runs=0,
        gen_measured_runs=1,
        gen_repeat_aggregation="median",
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
        gemm_quant="bf16",
        moe_quant="bf16",
        attn_quant="bf16",
        kv_quant="bf16",
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
        self.assertFalse(hasattr(args, "gen_driver"))
        self.assertFalse(hasattr(args, "ctx_driver"))
        self.assertEqual(args.nsys_capture, "cuda_profiler_api")

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

        first = cl._deterministic_prefix_suffix_prompts(
            1,
            prefix_len=16,
            suffix_len=4,
            token_config=token_config,
            prefix_request_index_offset=10,
            suffix_request_index_offset=100,
            cache_salt_prefix="ctx-prefix",
        )[0]
        second = cl._deterministic_prefix_suffix_prompts(
            1,
            prefix_len=16,
            suffix_len=4,
            token_config=token_config,
            prefix_request_index_offset=10,
            suffix_request_index_offset=200,
            cache_salt_prefix="ctx-prefix",
        )[0]

        self.assertEqual(first["prompt_token_ids"][:16], second["prompt_token_ids"][:16])
        self.assertNotEqual(first["prompt_token_ids"][16:], second["prompt_token_ids"][16:])
        self.assertEqual(first["cache_salt"], second["cache_salt"])

    def test_ctx_prefix_cache_key_is_shared_across_new_tokens_and_runs(self):
        first = cl.DataPoint("ctx", 1, 16, 4096)
        second = cl.DataPoint("ctx", 1, 128, 4096)

        self.assertEqual(
            cl._ctx_prefix_request_index_offset(first),
            cl._ctx_prefix_request_index_offset(second),
        )
        self.assertEqual(
            cl._ctx_cache_salt_prefix("wu", first),
            cl._ctx_cache_salt_prefix("wu", second),
        )
        self.assertNotEqual(
            cl._ctx_suffix_request_index_offset(first, 0, warmup=False),
            cl._ctx_suffix_request_index_offset(second, 0, warmup=False),
        )
        self.assertNotEqual(
            cl._ctx_suffix_request_index_offset(first, 0, warmup=False),
            cl._ctx_suffix_request_index_offset(first, 1, warmup=False),
        )

    def test_gen_prefix_cache_key_is_shared_across_batch_and_past(self):
        first = cl.DataPoint("gen", 1, 1, 128)
        second = cl.DataPoint("gen", 4, 1, 4096)

        self.assertEqual(
            cl._gen_prompt_request_index_offset(first),
            cl._gen_prompt_request_index_offset(second),
        )
        self.assertEqual(
            cl._gen_cache_salt_prefix("wu", first),
            cl._gen_cache_salt_prefix("wu", second),
        )

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

    def test_worker_cmd_defaults_to_cuda_profiler_capture(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            scheduler = cl.Scheduler(_args(tmp_path), [_unit(tmp_path, [])])

            cmd = scheduler._worker_cmd(tmp_path / "spec.json", tmp_path / "report")

            self.assertIn("--capture-range=cudaProfilerApi", cmd)
            self.assertIn("--capture-range-end=stop", cmd)

    def test_worker_cmd_can_trace_full_process(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            args = _args(tmp_path)
            args.nsys_capture = "full"
            scheduler = cl.Scheduler(args, [_unit(tmp_path, [])])

            cmd = scheduler._worker_cmd(tmp_path / "spec.json", tmp_path / "report")

            self.assertNotIn("--capture-range=cudaProfilerApi", cmd)
            self.assertNotIn("--capture-range-end=stop", cmd)

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

    def test_scheduler_enables_prefix_cache_for_prefix_context(self):
        with tempfile.TemporaryDirectory() as td:
            tmp_path = Path(td)
            unit = _unit(tmp_path, [cl.DataPoint("ctx", 1, 128, 0)])
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=0)

            self.assertIn("--enable-prefix-caching", spec["extra_vllm_args"])

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

    def test_engine_tokens_default_to_vllm_deployment_parity(self):
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
        self.assertEqual(tokens[tokens.index("--max-model-len") + 1], "129")
        self.assertEqual(
            tokens[tokens.index("--max-num-batched-tokens") + 1],
            str(cl.CTX_MAX_NUM_BATCHED_TOKENS_FLOOR),
        )
        self.assertEqual(tokens[tokens.index("--max-num-seqs") + 1], "4")

    def test_engine_tokens_do_not_enable_runtime_tensor_parallel(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 128, 0)],
            extra_vllm_args=[],
        )

        self.assertNotIn("--tensor-parallel-size", tokens)
        self.assertNotIn("--worker-extension-cls", tokens)

    def test_engine_tokens_context_uses_automatic_token_budget_floor(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 3, 0)],
            extra_vllm_args=[],
        )

        self.assertEqual(
            tokens[tokens.index("--max-num-batched-tokens") + 1],
            str(cl.CTX_MAX_NUM_BATCHED_TOKENS_FLOOR),
        )

    def test_engine_tokens_prefix_cache_keeps_decode_budget_small(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 32, 1, 65536)],
            extra_vllm_args=[],
        )

        self.assertEqual(tokens[tokens.index("--max-model-len") + 1], "65538")
        self.assertEqual(tokens[tokens.index("--max-num-seqs") + 1], "32")
        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "32")

    def test_engine_tokens_prefix_cache_uses_at_least_one_block(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("gen", 1, 1, 128)],
            extra_vllm_args=[],
        )

        self.assertEqual(
            tokens[tokens.index("--max-num-batched-tokens") + 1],
            str(cl.VLLM_DEFAULT_BLOCK_SIZE),
        )

    def test_engine_tokens_include_context_past_kv_in_model_len(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 8192, 8192)],
            extra_vllm_args=[],
        )

        self.assertEqual(tokens[tokens.index("--max-model-len") + 1], "16385")
        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "8192")

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

    def test_gpt_oss_build_work_units_targets_first_moe_layer_for_noop(self):
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

            self.assertEqual([u.target_layers for u in units], [[0], [0]])
            self.assertEqual([u.representative.layer_type for u in units], ["moe", "moe"])
            self.assertEqual([u.representative.layer_multiplier for u in units], [1, 1])
            self.assertEqual([u.moe_noop for u in units], [True, True])
            self.assertEqual(
                [call.kwargs["extra_overrides"] for call in patcher.mock_calls],
                [{"layer_types": ["sliding_attention"]}, {"layer_types": ["sliding_attention"]}],
            )
            self.assertEqual(
                [call.kwargs["num_hidden_layers"] for call in patcher.mock_calls],
                [1, 1],
            )

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
                moe_noop=True,
            )
            scheduler = cl.Scheduler(_args(tmp_path), [unit])

            spec = scheduler._make_spec(unit, unit.datapoints, attempt_id=1)

            self.assertTrue(spec["moe_noop"])

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
