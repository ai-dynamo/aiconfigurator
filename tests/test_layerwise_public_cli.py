import argparse
from pathlib import Path
from unittest import mock

import pytest

from collector.layerwise.common.paths import default_run_dir, slugify
from collector.layerwise.fpm import collect as fpm_collect
from collector.layerwise.fpm.datapoint_generator import default_shapes, generate_fpm_cases
from collector.layerwise.fpm.docker import build_collect_command
from collector.layerwise.vllm import collect as vllm_collect
from collector.layerwise.vllm import datapoint_generator as vllm_datapoints
from collector.layerwise.vllm.registry import select_models


def test_default_run_dir_includes_prefix_and_model_slug():
    path = default_run_dir("fpm_vllm", model="Qwen/Qwen3-32B")

    assert path.parent == Path(".tmp/layerwise-artifacts/runs")
    assert "fpm_vllm_Qwen_Qwen3-32B_" in path.name
    assert slugify("Qwen/Qwen3-32B") == "Qwen_Qwen3-32B"


def test_layerwise_public_cli_defaults_to_all_registry_models(tmp_path):
    args = vllm_collect._build_arg_parser().parse_args([
        "--run-dir",
        str(tmp_path),
        "--run-preset",
        "smoke",
        "--tp-sizes",
        "1,2",
        "--phases",
        "gen",
    ])
    args.run_dir = tmp_path
    args.work_dir = str(tmp_path / "profiles")
    args.output = str(tmp_path / "layerwise.csv")
    args.config_cache_dir = str(tmp_path / "config_cache")

    def fake_config(model):
        if "35B-A3B" in model:
            return {
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "intermediate_size": 1024,
                "num_experts": 8,
                "moe_intermediate_size": 512,
            }
        return {
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "intermediate_size": 1024,
        }

    with (
        mock.patch.object(vllm_datapoints, "_load_original_config", side_effect=fake_config),
        mock.patch.object(vllm_datapoints, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
        mock.patch.object(vllm_datapoints, "resolve_max_decode_batch_size", return_value=64),
    ):
        units = vllm_datapoints.build_public_work_units(args, select_models(args.models))

    assert {unit.row_base["model"] for unit in units} == {
        "Qwen/Qwen3-32B",
        "Qwen/Qwen3.6-35B-A3B",
    }
    assert any(unit.row_base["ep"] == 2 for unit in units if unit.row_base["model"].endswith("35B-A3B"))
    assert all(unit.row_base["ep"] == 1 for unit in units if unit.row_base["model"] == "Qwen/Qwen3-32B")


def test_layerwise_auto_ep_sizes_skip_intermediate_moe_tp(tmp_path):
    args = vllm_collect._build_arg_parser().parse_args([
        "--run-dir",
        str(tmp_path),
        "--run-preset",
        "smoke",
        "--models",
        "Qwen/Qwen3.6-35B-A3B",
        "--tp-sizes",
        "4,8",
        "--ep-sizes",
        "1,2,4,8",
        "--phases",
        "gen",
    ])
    args.run_dir = tmp_path
    args.work_dir = str(tmp_path / "profiles")
    args.output = str(tmp_path / "layerwise.csv")
    args.config_cache_dir = str(tmp_path / "config_cache")

    fake_config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 1024,
        "num_experts": 8,
        "moe_intermediate_size": 512,
    }

    with (
        mock.patch.object(vllm_datapoints, "_load_original_config", return_value=fake_config),
        mock.patch.object(vllm_datapoints, "patch_for_parallelism", return_value=str(tmp_path / "patched")),
        mock.patch.object(vllm_datapoints, "resolve_max_decode_batch_size", return_value=64),
    ):
        units = vllm_datapoints.build_public_work_units(args, select_models(args.models))

    assert [(unit.row_base["attn_tp"], unit.row_base["ep"]) for unit in units] == [
        (4, 1),
        (4, 4),
        (8, 1),
        (8, 8),
    ]


def test_layerwise_full_decode_default_uses_requested_max_batch_size():
    values = vllm_datapoints.values_for_preset("full", max_decode_batch_size=1024)

    assert values["ctx_new_tokens"] == "1,16,128,256,512,1024,2048,4096,8192"
    assert values["ctx_past_kv"] == "0,16,128,256,512,1024,2048,4096,8192,16384,32768"
    assert values["gen_batch_sizes"] == "1,2,4,8,16,32,64,128,256,512,1024"
    assert values["gen_past_kv"] == "1,16,128,256,512,1024,2048,4096,8192,16384,32768"


def test_layerwise_full_decode_default_can_follow_smaller_vllm_default():
    values = vllm_datapoints.values_for_preset("full", max_decode_batch_size=256)

    assert values["gen_batch_sizes"] == "1,2,4,8,16,32,64,128,256"


def test_layerwise_public_cli_supports_single_ad_hoc_model():
    args = vllm_collect._build_arg_parser().parse_args([
        "--model",
        "local/model",
        "--model-kind",
        "moe",
        "--ep-sizes",
        "1,2",
        "--num-slots",
        "8",
        "--gemm-quant",
        "fp8",
    ])

    models = vllm_collect._selected_models(args)

    assert len(models) == 1
    assert models[0].model == "local/model"
    assert models[0].kind == "moe"
    assert models[0].ep_sizes == (1, 2, 4, 8)
    assert models[0].num_slots == 8
    assert models[0].gemm_quant == "fp8"


def test_fpm_public_cli_requires_model():
    with pytest.raises(SystemExit):
        fpm_collect._build_arg_parser().parse_args([])


def test_fpm_public_cli_has_no_smoke_preset():
    with pytest.raises(SystemExit):
        fpm_collect._build_arg_parser().parse_args([
            "--model",
            "Qwen/Qwen3-32B",
            "--run-preset",
            "smoke",
        ])


def test_fpm_default_shapes_are_full_sweep():
    assert default_shapes() == {
        "contexts": "128,1024,4096",
        "context_repeats": "6",
        "decode_batches": "1,4,16",
        "decode_past_kv": "4096",
        "decode_osl": "8",
        "decode_repeats": "6",
        "real_workload_requests": "128",
        "real_workload_concurrency": "32",
        "real_workload_dataset": "OpenAssistant/oasst1",
        "real_workload_shape_source": "scaled_dataset",
        "real_workload_isl_min": "100",
        "real_workload_isl_max": "16384",
        "real_workload_isl_mean": "4096",
        "real_workload_osl_min": "100",
        "real_workload_osl_max": "4096",
        "real_workload_osl_mean": "1024",
    }


def test_fpm_public_cli_defaults_to_real_workload():
    args = fpm_collect._build_arg_parser().parse_args([
        "--model",
        "Qwen/Qwen3-32B",
    ])
    fpm_collect._apply_shape_defaults(args)

    assert args.real_workload is True
    assert args.real_workload_requests == 128
    assert args.real_workload_concurrency == 32
    assert args.real_workload_dataset == "OpenAssistant/oasst1"
    assert args.real_workload_shape_source == "scaled_dataset"
    assert args.real_workload_isl_min == 100
    assert args.real_workload_isl_max == 16384
    assert args.real_workload_isl_mean == 4096
    assert args.real_workload_osl_min == 100
    assert args.real_workload_osl_max == 4096
    assert args.real_workload_osl_mean == 1024


def test_fpm_case_generation_and_shell_command(tmp_path):
    args = argparse.Namespace(
        model="Qwen/Qwen3-32B",
        phases="context,decode",
        contexts="128",
        context_repeats="6",
        decode_batches="1,4",
        decode_osl="8",
        decode_repeats=6,
        real_workload=True,
        real_workload_requests=128,
        real_workload_concurrency=32,
        real_workload_dataset="OpenAssistant/oasst1",
        real_workload_shape_source="scaled_dataset",
        real_workload_isl_min=100,
        real_workload_isl_max=16384,
        real_workload_isl_mean=4096,
        real_workload_osl_min=100,
        real_workload_osl_max=4096,
        real_workload_osl_mean=1024,
        image="image",
        warmup_requests=None,
        gpus=None,
        keep_running=False,
        dry_run=True,
        extra_vllm_arg=["--foo", "bar"],
    )

    cases = generate_fpm_cases("1,2", "1,2", "1024")
    assert [(case.tp_size, case.ep_size, case.decode_past_kv) for case in cases] == [
        (1, 1, 1024),
        (2, 1, 1024),
        (2, 2, 1024),
    ]

    cmd = build_collect_command(args, cases[-1], tmp_path)
    assert cmd.argv[:2] == ["bash", "collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh"]
    assert ["--tp-size", "2"] == cmd.argv[cmd.argv.index("--tp-size"): cmd.argv.index("--tp-size") + 2]
    assert ["--ep-size", "2"] == cmd.argv[cmd.argv.index("--ep-size"): cmd.argv.index("--ep-size") + 2]
    assert "--real-workload" in cmd.argv
    assert "--max-model-len" not in cmd.argv
    assert ["--real-workload-requests", "128"] == cmd.argv[
        cmd.argv.index("--real-workload-requests"): cmd.argv.index("--real-workload-requests") + 2
    ]
    assert ["--real-workload-isl-max", "16384"] == cmd.argv[
        cmd.argv.index("--real-workload-isl-max"): cmd.argv.index("--real-workload-isl-max") + 2
    ]
    assert ["--real-workload-osl-mean", "1024"] == cmd.argv[
        cmd.argv.index("--real-workload-osl-mean"): cmd.argv.index("--real-workload-osl-mean") + 2
    ]
    assert "--dry-run" in cmd.argv
    assert cmd.argv[-3:] == ["--", "--foo", "bar"]


def test_fpm_case_generation_skips_intermediate_ep_sizes():
    cases = generate_fpm_cases("4,8", "1,2,4,8", "1024")

    assert [(case.tp_size, case.ep_size, case.decode_past_kv) for case in cases] == [
        (4, 1, 1024),
        (4, 4, 1024),
        (8, 1, 1024),
        (8, 8, 1024),
    ]
