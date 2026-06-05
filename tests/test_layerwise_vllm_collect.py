import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "common"))
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "vllm"))

import collect_layerwise as cl
from random_prompt_tokens import load_random_prompt_token_config


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
        no_restrict_cudagraph_sizes=False,
        extra_vllm_args="",
        extra_vllm_arg=[],
        latency_source="span",
        min_max_num_batched_tokens=1,
        ctx_warmup_runs=0,
        ctx_measured_runs=1,
        ctx_repeat_aggregation="median",
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
        include_moe_layer=False,
        target_layer_count=1,
        target_layers=None,
        target_layer_config_depth=None,
        phases="both",
        ctx_new_tokens="1",
        ctx_past_kv="0",
        gen_batch_sizes="1",
        gen_past_kv="1",
        gemm_quant="bf16",
        moe_quant="bf16",
        attn_quant="bf16",
        kv_quant="bf16",
        min_max_num_batched_tokens=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class VllmCollectLayerwiseTests(unittest.TestCase):
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
                "start_ns": 1_000,
                "end_ns": 101_000,
                "kernel_count": 2,
            },
            {
                "step": 17,
                "batch_size": 4,
                "past_kv": 16,
                "measure_run": 0,
                "gpu_us": 50.0,
                "start_ns": 80_000,
                "end_ns": 130_000,
                "kernel_count": 1,
            },
        ])

        agg = parsed[(17, 4, 16, 0)]
        self.assertEqual(agg["gpu_us"], 150.0)
        self.assertEqual(agg["span_us"], 129.0)
        self.assertEqual(agg["kernel_count"], 3)

    def test_latency_source_can_select_span_or_gpu_sum(self):
        agg = {"gpu_us": 150.0, "span_us": 129.0}

        self.assertEqual(cl._latency_us_from_agg(agg, "span"), 129.0)
        self.assertEqual(cl._latency_us_from_agg(agg, "gpu"), 150.0)
        self.assertEqual(cl._latency_us_from_agg(agg, "gpu_capped"), 129.0)
        self.assertEqual(
            cl._latency_us_from_agg({"gpu_us": 100.0, "span_us": 129.0}, "gpu_capped"),
            100.0,
        )

    def test_filter_rows_to_target_layers_excludes_skipped_layer_kernels(self):
        rows = [
            {"rollup_parts": ("0", "mlp"), "gpu_us": 100.0},
            {"rollup_parts": ("1", "mlp"), "gpu_us": 200.0},
            {"rollup_parts": ("2", "mlp"), "gpu_us": 300.0},
        ]

        filtered = cl._filter_rows_to_target_layers(rows, [1])

        self.assertEqual(filtered, [rows[1]])

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
        latency_us, kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="span",
            aggregation="median",
        )

        self.assertEqual(latency_us, 200.0)
        self.assertEqual(kernel_count, 2)
        self.assertEqual(measure_count, 3)

    def test_repeat_aggregation_can_drop_min_and_max_then_average(self):
        aggs = [
            {"gpu_us": value, "span_us": value, "kernel_count": 1}
            for value in [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0]
        ]

        latency_us, kernel_count, measure_count = cl._reduce_agg_latency(
            aggs,
            latency_source="gpu_capped",
            aggregation="trimmed_mean",
        )

        self.assertEqual(latency_us, 35.0)
        self.assertEqual(kernel_count, 1)
        self.assertEqual(measure_count, 6)

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

    def test_engine_tokens_use_one_decode_graph_engine_for_mixed_points(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[
                cl.DataPoint("ctx", 1, 128, 0),
                cl.DataPoint("gen", 4, 1, 16),
                cl.DataPoint("gen", 2, 1, 32),
            ],
            restrict_cudagraph_sizes=True,
            extra_vllm_args=[],
        )

        config = json.loads(tokens[tokens.index("--compilation-config") + 1])
        self.assertEqual(config["mode"], 0)
        self.assertEqual(config["cudagraph_mode"], "FULL_DECODE_ONLY")
        self.assertEqual(config["cudagraph_capture_sizes"], [2, 4])
        self.assertEqual(tokens[tokens.index("--max-model-len") + 1], "129")
        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "128")
        self.assertEqual(tokens[tokens.index("--max-num-seqs") + 1], "4")

    def test_engine_tokens_can_floor_max_num_batched_tokens(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 3, 0)],
            restrict_cudagraph_sizes=True,
            extra_vllm_args=[],
            min_max_num_batched_tokens=4096,
        )

        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "4096")

    def test_engine_tokens_include_context_past_kv_in_model_len(self):
        tokens = cl._engine_tokens(
            model_dir="/model",
            datapoints=[cl.DataPoint("ctx", 1, 8192, 8192)],
            restrict_cudagraph_sizes=True,
            extra_vllm_args=[],
            min_max_num_batched_tokens=1,
        )

        self.assertEqual(tokens[tokens.index("--max-model-len") + 1], "16385")
        self.assertEqual(tokens[tokens.index("--max-num-batched-tokens") + 1], "8192")

    def test_context_marker_milestone_selects_later_prefill_chunk(self):
        self.assertEqual(cl._ctx_marker_milestone(cl.DataPoint("ctx", 1, 8192, 0), 8192), 1)
        self.assertEqual(cl._ctx_marker_milestone(cl.DataPoint("ctx", 1, 8192, 8192), 8192), 2)

    def test_context_past_kv_must_start_on_chunk_boundary(self):
        with self.assertRaisesRegex(ValueError, "chunk boundary"):
            cl._validate_ctx_past_kv([cl.DataPoint("ctx", 1, 512, 1024)], 8192)

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
                mock.patch.object(cl, "_load_original_config", return_value=dense_config),
                mock.patch.object(cl, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
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
                mock.patch.object(cl, "_load_original_config", return_value=dense_config),
                mock.patch.object(cl, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].target_layers, [0, 1, 2, 3])
            self.assertEqual(patcher.mock_calls[0].kwargs["num_hidden_layers"], 4)

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
                mock.patch.object(cl, "_load_original_config", return_value=dense_config),
                mock.patch.object(cl, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
            ):
                units = cl.build_work_units(args)

            self.assertEqual(units[0].target_layers, [1])
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
                mock.patch.object(cl, "_load_original_config", return_value=dense_config),
                mock.patch.object(cl, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
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
                mock.patch.object(cl, "_load_original_config", return_value=moe_config),
                mock.patch.object(cl, "patch_for_parallelism", return_value=str(tmp_path / "patched")) as patcher,
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


if __name__ == "__main__":
    unittest.main()
