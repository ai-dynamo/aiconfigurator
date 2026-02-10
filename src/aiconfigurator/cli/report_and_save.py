# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import traceback

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from prettytable import PrettyTable

from aiconfigurator.generator.api import (
    generate_backend_artifacts,
    load_generator_overrides_from_args,
)
from aiconfigurator.generator.module_bridge import task_config_to_generator_config
from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk.pareto_analysis import draw_pareto_to_string
from aiconfigurator.sdk.task import TaskConfig
from aiconfigurator.sdk.utils import safe_mkdir

logger = logging.getLogger(__name__)


def _check_power_data_available(best_configs: dict[str, pd.DataFrame], threshold: float = 0.9) -> bool:
    """
    Check if power data is available and meaningful across configurations.

    Args:
        best_configs: Dictionary of experiment name to best configurations DataFrame
        threshold: Minimum ratio of configs with meaningful power data (default 0.9)

    Returns:
        True if power data should be displayed (>= threshold of configs have power >= 1W)
    """
    total_count = 0
    power_count = 0

    for exp_name, config_df in best_configs.items():
        if config_df is not None and not config_df.empty and "power_w" in config_df.columns:
            power_values = config_df["power_w"].values
            total_count += len(power_values)
            # Count how many configs have meaningful power data (>= 1W)
            power_count += sum(1 for p in power_values if p >= 1.0)

    if total_count == 0:
        return False

    # Show power column if >= threshold of configs have meaningful power data
    power_ratio = power_count / total_count
    return power_ratio >= threshold


def _plot_worker_setup_table(
    exp_name: str,
    config_df: pd.DataFrame,
    total_gpus: int,
    tpot_target: float,
    top: int,
    is_moe: bool,
    request_latency_target: float | None,
    show_power: bool = True,
) -> str:
    """Plot worker setup table for a single experiment."""
    buf = []

    if config_df is None or config_df.empty:
        return ""

    config_df["tokens/s/gpu_cluster"] = (
        config_df["tokens/s/gpu"]
        * (total_gpus // config_df["num_total_gpus"])
        * config_df["num_total_gpus"]
        / total_gpus
        if total_gpus > 0
        else 0
    )
    constraint_col = "tpot"
    constraint_target = tpot_target
    constraint_label = "TPOT"
    if request_latency_target is not None and request_latency_target > 0:
        constraint_col = "request_latency"
        constraint_target = request_latency_target
        constraint_label = "request latency"
    top_configs = (
        config_df[config_df[constraint_col] <= constraint_target]
        .sort_values(by="tokens/s/gpu_cluster", ascending=False)
        .head(top)
        .copy()
    )

    if top_configs.empty:
        return f"\nNo configurations for {exp_name} met the {constraint_label} constraint."

    top_configs["replicas"] = total_gpus // top_configs["num_total_gpus"]
    top_configs["total_gpus_used"] = top_configs["num_total_gpus"] * top_configs["replicas"]

    buf.append(f"\n{exp_name} Top Configurations: (Sorted by tokens/s/gpu)")
    table = PrettyTable()

    # Check if it is disagg config by checking for prefill/decode specific columns
    is_disagg = "(p)tp" in top_configs.columns

    if is_disagg:
        field_names = [
            "Rank",
            "backend",
            "\033[1mtokens/s/gpu\033[0m",
            "tokens/s/user",
            "TTFT",
            "request_latency",
            "concurrency",
            "total_gpus (used)",
            "replicas",
            "gpus/replica",
            "(p)workers",
            "(p)gpus/worker",
            "(p)parallel",
            "(p)bs",
            "(d)workers",
            "(d)gpus/worker",
            "(d)parallel",
            "(d)bs",
        ]
        if show_power:
            field_names.append("power_w")
        table.field_names = field_names
        for i, row in enumerate(top_configs.to_dict("records")):
            if is_moe:
                p_parallel = (
                    f"tp\033[4m{row['(p)tp']}\033[0mpp\033[4m{row['(p)pp']}\033[0m"
                    f"dp\033[4m{row['(p)dp']}\033[0metp{row['(p)moe_tp']}ep{row['(p)moe_ep']}"
                )
                d_parallel = (
                    f"tp\033[4m{row['(d)tp']}\033[0mpp\033[4m{row['(d)pp']}\033[0m"
                    f"dp\033[4m{row['(d)dp']}\033[0metp{row['(d)moe_tp']}ep{row['(d)moe_ep']}"
                )
                p_gpus_worker = (
                    f"{row['(p)pp'] * row['(p)tp'] * row['(p)dp']} "
                    f"(=\033[4m{row['(p)tp']}\033[0mx\033[4m{row['(p)pp']}\033[0mx\033[4m{row['(p)dp']}\033[0m)"
                )
                d_gpus_worker = (
                    f"{row['(d)pp'] * row['(d)tp'] * row['(d)dp']} "
                    f"(=\033[4m{row['(d)tp']}\033[0mx\033[4m{row['(d)pp']}\033[0mx\033[4m{row['(d)dp']}\033[0m)"
                )
            else:
                p_parallel = f"tp\033[4m{row['(p)tp']}\033[0mpp\033[4m{row['(p)pp']}\033[0m"
                d_parallel = f"tp\033[4m{row['(d)tp']}\033[0mpp\033[4m{row['(d)pp']}\033[0m"
                p_gpus_worker = (
                    f"{row['(p)pp'] * row['(p)tp']} (=\033[4m{row['(p)tp']}\033[0mx\033[4m{row['(p)pp']}\033[0m)"
                )
                d_gpus_worker = (
                    f"{row['(d)pp'] * row['(d)tp']} (=\033[4m{row['(d)tp']}\033[0mx\033[4m{row['(d)pp']}\033[0m)"
                )
            row_data = [
                i + 1,
                row["backend"],
            ]
            row_data.extend(
                [
                    f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m",
                    f"{row['tokens/s/user']:.2f}",
                    f"{row['ttft']:.2f}",
                    f"{row['request_latency']:.2f}",
                    f"{row['concurrency'] * row['replicas']} (={row['concurrency']}x{row['replicas']})",
                    f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                    row["replicas"],
                    (
                        f"{row['num_total_gpus']} "
                        f"(={row['(p)workers']}x{row['(p)pp'] * row['(p)tp'] * row['(p)dp']}"
                        f"+{row['(d)workers']}x{row['(d)pp'] * row['(d)tp'] * row['(d)dp']})"
                    ),
                    row["(p)workers"],
                    p_gpus_worker,
                    p_parallel,
                    row["(p)bs"],
                    row["(d)workers"],
                    d_gpus_worker,
                    d_parallel,
                    row["(d)bs"],
                ]
            )
            if show_power:
                row_data.append(f"{row['power_w']:.1f}W")
            table.add_row(row_data)
    else:  # agg
        field_names = [
            "Rank",
            "backend",
            "\033[1mtokens/s/gpu\033[0m",
            "tokens/s/user",
            "TTFT",
            "request_latency",
            "concurrency",
            "total_gpus (used)",
            "replicas",
            "gpus/replica",
            "gpus/worker",
            "parallel",
            "bs",
        ]
        if show_power:
            field_names.append("power_w")
        table.field_names = field_names
        for i, row in enumerate(top_configs.to_dict("records")):
            if is_moe:
                parallel = (
                    f"tp\033[4m{row['tp']}\033[0mpp\033[4m{row['pp']}\033[0m"
                    f"dp\033[4m{row['dp']}\033[0metp{row['moe_tp']}ep{row['moe_ep']}"
                )
                gpus_worker = (
                    f"{row['pp'] * row['tp'] * row['dp']} "
                    f"(=\033[4m{row['tp']}\033[0mx\033[4m{row['pp']}\033[0mx\033[4m{row['dp']}\033[0m)"
                )
            else:
                parallel = f"tp\033[4m{row['tp']}\033[0mpp\033[4m{row['pp']}\033[0m"
                gpus_worker = (
                    f"{row['pp'] * row['tp']} (=\033[4m{row['tp']}\033[0mx\033[4m{row['pp']}"
                    f"\033[0mx\033[4m{row['dp']}\033[0m)"
                )
            row_data = [
                i + 1,
                row["backend"],
            ]
            row_data.extend(
                [
                    f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m",
                    f"{row['tokens/s/user']:.2f}",
                    f"{row['ttft']:.2f}",
                    f"{row['request_latency']:.2f}",
                    f"{row['concurrency'] * row['replicas']} (={row['concurrency']}x{row['replicas']})",
                    f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                    row["replicas"],
                    row["num_total_gpus"],
                    gpus_worker,
                    parallel,
                    row["bs"],
                ]
            )
            if show_power:
                row_data.append(f"{row['power_w']:.1f}W")
            table.add_row(row_data)

    buf.append(table.get_string())
    return "\n".join(buf)


def log_final_summary(
    chosen_exp: str,
    best_throughputs: dict[str, float],
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    task_configs: dict[str, TaskConfig],
    mode: str,
    pareto_x_axis: dict[str, str] | None = None,
    top_n: int = 5,
):
    """Log final summary of configuration results"""

    # Consolidate and format results into a summary box for clear presentation
    summary_box = []
    summary_box.append("*" * 80)
    summary_box.append("*{:^78}*".format(" Dynamo aiconfigurator Final Results "))
    summary_box.append("*" * 80)

    summary_box.append("  " + "-" * 76)
    summary_box.append("  Input Configuration & SLA Target:")

    # For multi-backend mode, get task_config using the backend from best_configs
    chosen_best_config = best_configs.get(chosen_exp)
    if chosen_best_config is not None and "backend" in chosen_best_config.columns and not chosen_best_config.empty:
        chosen_backend = chosen_best_config["backend"].iloc[0]
        task_config_key = f"{chosen_exp}_{chosen_backend}"
        # Verify the key exists (for multi-backend mode)
        if task_config_key in task_configs:
            chosen_task_config = task_configs[task_config_key]
        else:
            chosen_task_config = task_configs[chosen_exp]
    else:
        chosen_task_config = task_configs[chosen_exp]

    summary_box.append(
        f"    Model: {chosen_task_config.config.model_path} (is_moe: {chosen_task_config.config.is_moe})"
    )
    summary_box.append(f"    Total GPUs: {chosen_task_config.total_gpus}")
    if mode == "default":
        agg_value = best_throughputs.get("agg", 0.0)
        disagg_value = best_throughputs.get("disagg", 0.0)
        if agg_value > 0 and disagg_value > 0:
            benefit_ratio = disagg_value / agg_value
        elif agg_value == 0 and disagg_value > 0:
            benefit_ratio = float("inf")
        elif agg_value > 0 and disagg_value == 0:
            benefit_ratio = 0.0
        else:
            benefit_ratio = 0.0  # handle case where both are 0
        summary_box.append(
            f"    Best Experiment Chosen: \033[1m{chosen_exp} at "
            f"{best_throughputs[chosen_exp]:.2f} tokens/s/gpu "
            f"(disagg {benefit_ratio:.2f}x better)\033[0m"
        )
    else:
        summary_box.append(
            f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu\033[0m"
        )

    summary_box.append("  " + "-" * 76)

    # ============================= overall summary
    summary_box.append("  Overall Best Configuration:")
    best_config_df = best_configs[chosen_exp]
    best_throughput = best_throughputs[chosen_exp]

    summary_box.append(f"    - Best Throughput: {best_throughput * chosen_task_config.total_gpus:,.2f} tokens/s")
    summary_box.append(f"    - Per-GPU Throughput: {best_throughput:.2f} tokens/s/gpu")
    if not best_config_df.empty:
        best_conf_details = best_config_df.iloc[0]
        summary_box.append(f"    - Per-User Throughput: {best_conf_details['tokens/s/user']:.2f} tokens/s/user")
        summary_box.append(f"    - TTFT: {best_conf_details['ttft']:.2f}ms")
        summary_box.append(f"    - TPOT: {best_conf_details['tpot']:.2f}ms")
        summary_box.append(f"    - Request Latency: {best_conf_details['request_latency']:.2f}ms")
    summary_box.append("  " + "-" * 76)

    # ============================= pareto frontier
    pareto_plot_buf = ""
    if len(pareto_fronts) <= 10:  # avoid overly crowded plots
        summary_box.append("  Pareto Frontier:")
        target_x_axis = "tokens/s/user"
        if pareto_x_axis:
            target_x_axis = pareto_x_axis.get(chosen_exp, target_x_axis)
        series_payload = []
        for name, df in pareto_fronts.items():
            if df is None or df.empty:
                continue
            series_axis = pareto_x_axis.get(name, target_x_axis) if pareto_x_axis else target_x_axis
            if series_axis != target_x_axis:
                continue
            series_payload.append({"df": df, "label": name})
        highlight_series = None
        if not best_config_df.empty:
            highlight_series = {
                "df": best_config_df.head(1),
                "label": f"{chosen_exp} best",
            }
        pareto_plot_buf = draw_pareto_to_string(
            f"{chosen_task_config.config.model_path} Pareto Frontier",
            series_payload,
            highlight=highlight_series,
            x_label=target_x_axis,
            y_label="tokens/s/gpu_cluster",
        )
        summary_box.append(pareto_plot_buf)
    summary_box.append("  " + "-" * 76)

    # ============================= deployment details
    summary_box.append("  Deployment Details:")
    summary_box.append(
        "    (p) stands for prefill, (d) stands for decode, bs stands for batch size, "
        "a replica stands for the smallest scalable unit xPyD of the disagg system"
    )
    summary_box.append("    Some math: total gpus used = replicas * gpus/replica")
    summary_box.append(
        "               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; "
        "for Agg, gpus/replica = gpus/worker"
    )
    summary_box.append(
        "               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; "
        "tp * pp for dense models (underlined \033[4mnumbers\033[0m are the actual values in math)"
    )

    # Check if power data is available before plotting tables
    show_power = _check_power_data_available(best_configs)

    # Plot worker setup tables for all experiments
    for exp_name, config_df in best_configs.items():
        # For multi-backend mode, use the first backend's config for table display
        # (total_gpus, is_moe, etc. should be the same across backends)
        if "backend" in config_df.columns and not config_df.empty:
            first_backend = config_df["backend"].iloc[0]
            task_config_key = f"{exp_name}_{first_backend}"
            # Verify the key exists (for multi-backend mode)
            if task_config_key not in task_configs:
                task_config_key = exp_name
        else:
            task_config_key = exp_name

        if not config_df.empty and "backend" not in config_df.columns:
            config_df = config_df.copy()
            config_df["backend"] = task_configs[task_config_key].backend_name

        exp_task_config = task_configs[task_config_key].config
        total_gpus = getattr(task_configs[task_config_key], "total_gpus", None) or 0
        table_buf = _plot_worker_setup_table(
            exp_name,
            config_df,
            total_gpus,
            exp_task_config.runtime_config.tpot,
            top_n,
            exp_task_config.is_moe,
            exp_task_config.runtime_config.request_latency,
            show_power,
        )
        summary_box.append(table_buf)

    summary_box.append("*" * 80)
    logger.info("\n" + "\n".join(summary_box))


def save_results(
    args,
    best_configs: dict[str, pd.DataFrame],
    pareto_fronts: dict[str, pd.DataFrame],
    task_configs: dict[str, TaskConfig],
    save_dir: str,
    generated_backend_version: str | None = None,
    backend: str | None = None,
):
    """Save the results to a directory."""

    first_exp_name = list(task_configs.keys())[0]
    first_task = task_configs[first_exp_name]
    first_task_config = first_task.config

    backend_str = backend or first_task.backend_name

    result_prefix = (
        f"{first_task_config.model_path}_{first_task.system_name}_{backend_str}_"
        f"isl{first_task_config.runtime_config.isl}_osl{first_task_config.runtime_config.osl}_"
        f"ttft{int(first_task_config.runtime_config.ttft)}_tpot{int(first_task_config.runtime_config.tpot)}"
    )
    result_dir_path = os.path.join(save_dir, f"{result_prefix}_{random.randint(0, 1000000)}")

    logger.info(f"Saving results to {result_dir_path}")
    try:
        safe_result_dir = safe_mkdir(result_dir_path, exist_ok=True)
        generator_overrides = load_generator_overrides_from_args(args)

        # Save overall pareto plots in the root directory
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        pareto_axis = {}
        for exp_name, cfg in task_configs.items():
            runtime_cfg = cfg.config.runtime_config
            if runtime_cfg.request_latency is not None and runtime_cfg.request_latency > 0:
                pareto_axis[exp_name] = "request_latency"
            else:
                pareto_axis[exp_name] = "tokens/s/user"
        all_request_latency = bool(pareto_axis) and all(axis == "request_latency" for axis in pareto_axis.values())
        global_x_axis = "request_latency" if all_request_latency else "tokens/s/user"
        maximize_x = not all_request_latency
        plt.title(f"{first_task_config.model_path} tokens/s/gpu vs {global_x_axis}")

        # Define markers for backends and colors for serving modes
        backend_markers = {
            "trtllm": "o",  # circle
            "vllm": "s",  # square
            "sglang": "^",  # triangle
        }
        serving_mode_colors = {
            "agg": "#1f77b4",  # blue
            "disagg": "#ff7f0e",  # orange
        }
        # Fallback colors for non-standard experiment names
        exp_colors = [
            "blue",
            "red",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
        ]
        color_idx = 0

        for exp_name, pareto_df in pareto_fronts.items():
            if pareto_axis.get(exp_name, global_x_axis) != global_x_axis:
                continue
            if pareto_df.empty:
                continue

            # Check if this is multi-backend mode (pareto_df has "backend" column)
            if "backend" in pareto_df.columns:
                # Plot each backend with different marker, color by serving mode (agg/disagg)
                # Note: pareto_df is already the combined Pareto frontier, so we just plot points
                # grouped by backend, without recomputing Pareto per backend
                color = serving_mode_colors.get(exp_name, exp_colors[color_idx % len(exp_colors)])
                for backend_name in pareto_df["backend"].unique():
                    backend_df = pareto_df[pareto_df["backend"] == backend_name].sort_values(by=global_x_axis)
                    marker = backend_markers.get(backend_name, "o")
                    label = f"{exp_name} ({backend_name})"
                    # Plot directly without recomputing Pareto
                    ax.plot(
                        backend_df[global_x_axis],
                        backend_df["tokens/s/gpu"],
                        color=color,
                        marker=marker,
                        label=label,
                        linestyle="-",
                        markersize=8,
                    )
                color_idx += 1
            else:
                # Single backend mode
                pareto_analysis.draw_pareto(
                    pareto_df,
                    global_x_axis,
                    "tokens/s/gpu",
                    ax,
                    exp_colors[color_idx % len(exp_colors)],
                    exp_name,
                    maximize_x=maximize_x,
                )
                color_idx += 1

        # Add axis labels and legend
        ax.set_xlabel(global_x_axis)
        ax.set_ylabel("tokens/s/gpu")
        ax.legend()

        plt.savefig(os.path.join(safe_result_dir, "pareto_frontier.png"))
        plt.close()

        # Save each experiment's results in its own subdirectory
        for exp_name, pareto_df in pareto_fronts.items():
            exp_dir = os.path.join(safe_result_dir, exp_name)
            safe_mkdir(exp_dir, exist_ok=True)

            # 1. Save best config dataframe
            best_config_df = best_configs.get(exp_name)  # top n configs
            if best_config_df is not None:
                best_config_df.to_csv(os.path.join(exp_dir, "best_config_topn.csv"), index=False)

            # 2. Save all pareto dataframe
            if pareto_df is not None:
                pareto_df.to_csv(os.path.join(exp_dir, "pareto.csv"), index=False)

            # 3. Save the config for this experiment
            # For multi-backend mode, we'll get task_config per row later
            # For single-backend mode, get it once here
            is_multi_backend = best_config_df is not None and "backend" in best_config_df.columns
            if not is_multi_backend:
                exp_task_config = task_configs[exp_name]
                effective_generated_version = generated_backend_version or exp_task_config.backend_version
            else:
                # For multi-backend, we'll use the first backend's version for warnings
                # but actual generation will use per-row backend
                first_backend = best_config_df["backend"].iloc[0]
                task_config_key = f"{exp_name}_{first_backend}"
                # Verify the key exists (for multi-backend mode)
                if task_config_key not in task_configs:
                    task_config_key = exp_name
                    is_multi_backend = False  # Treat as single backend
                exp_task_config = task_configs[task_config_key]
                effective_generated_version = generated_backend_version or exp_task_config.backend_version

            if generated_backend_version:
                logger.warning(
                    "\n" + "=" * 80 + "\n"
                    "  ⚠️  IMPORTANT: Config Generation Version\n" + "=" * 80 + "\n"
                    "  Experiment: %s\n"
                    "  Using generated_config_version: %s\n"
                    "\n"
                    "  Config formats differ across backend releases. Please ensure you pass\n"
                    "  the correct --generated_config_version to match your deployment target!\n" + "=" * 80,
                    exp_name,
                    generated_backend_version,
                )
            else:
                logger.warning(
                    "\n" + "=" * 80 + "\n"
                    "  ⚠️  IMPORTANT: Config Generation Version Not Specified\n" + "=" * 80 + "\n"
                    "  Experiment: %s\n"
                    "  --generated_config_version NOT provided\n"
                    "  Defaulting to backend_version: %s\n"
                    "\n"
                    "  Config formats differ across backend releases. If you are targeting\n"
                    "  a different version, please pass --generated_config_version explicitly!\n" + "=" * 80,
                    exp_name,
                    exp_task_config.backend_version,
                )

            with open(os.path.join(exp_dir, "config.yaml"), "w") as f:  # for future aic repro
                yaml.safe_dump(json.loads(exp_task_config.pretty()), f, sort_keys=False)

            # 4. Save the generated config for this experiment, sub-directory for each best config
            if best_config_df is not None:
                for i, (idx, result_df) in enumerate(best_config_df.iterrows()):
                    # For multi-backend mode, get the task_config for this row's backend
                    if is_multi_backend and "backend" in result_df:
                        row_backend = result_df["backend"]
                        row_task_config_key = f"{exp_name}_{row_backend}"
                        # Verify the key exists (for multi-backend mode)
                        if row_task_config_key in task_configs:
                            row_task_config = task_configs[row_task_config_key]
                            row_backend_version = generated_backend_version or row_task_config.backend_version
                        else:
                            row_task_config = exp_task_config
                            row_backend_version = effective_generated_version
                    else:
                        row_task_config = exp_task_config
                        row_backend_version = effective_generated_version

                    cfg = task_config_to_generator_config(
                        task_config=row_task_config,
                        result_df=result_df,
                        generator_overrides=generator_overrides,
                    )

                    top_config_dir = os.path.join(exp_dir, f"top{i + 1}")
                    safe_mkdir(top_config_dir, exist_ok=True)
                    with open(os.path.join(top_config_dir, "generator_config.yaml"), "w") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)

                    try:
                        generate_backend_artifacts(
                            params=cfg,
                            backend=row_task_config.backend_name,
                            backend_version=row_backend_version,
                            output_dir=top_config_dir,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Failed to generate backend config from aic generator: %s, %s",
                            exc,
                            traceback.format_exc(),
                        )

    except Exception:
        logger.exception("Failed to save results")
