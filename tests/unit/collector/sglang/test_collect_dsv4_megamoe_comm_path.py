import argparse

from collector.sglang.collect_dsv4_megamoe_comm_path import run_comm_path_collection


def _args(tmp_path, effective_collector):
    return argparse.Namespace(
        num_processes=8,
        num_max_tokens_per_rank=8192,
        num_tokens_list="1,8",
        intermediate_hiddens="512,3072",
        target_intermediate_hidden=3072,
        repeat_samples=1,
        routing_mode="random",
        power_law_alpha=1.01,
        power_law_remap_hot_rank_to_zero=False,
        hidden_size=7168,
        num_experts=384,
        topk=6,
        masked_ratio=0.0,
        activation_clamp=10.0,
        fast_math=1,
        flush_l2=1,
        seed=0,
        plateau_tolerance_pct=5.0,
        output_dir=tmp_path / "out",
        analysis_output_dir=None,
        perf_output=None,
        framework="SGLang",
        backend_version="0.5.9",
        device_name="",
        kernel_source="DeepGEMM_fp8_fp4_mega_moe",
        source="unit",
        effective_collector=effective_collector,
        python_executable="/usr/bin/python3",
        dry_run=True,
    )


def test_comm_path_collector_dry_run_builds_sweep_commands(tmp_path):
    effective_collector = tmp_path / "collect_dsv4_megamoe_effective_nvl_bw.py"
    effective_collector.write_text("# fake collector\n", encoding="utf-8")

    summary = run_comm_path_collection(_args(tmp_path, effective_collector))

    assert summary["dry_run"] is True
    assert summary["perf_output"].endswith("analysis/dsv4_megamoe_comm_path_perf.txt")
    assert len(summary["commands"]) == 2
    assert "--intermediate-hidden 512" in summary["commands"][0]
    assert "--intermediate-hidden 3072" in summary["commands"][1]
    assert "--source unit-intermediate-3072" in summary["commands"][1]
    assert (tmp_path / "out" / "collection_summary.json").exists()
