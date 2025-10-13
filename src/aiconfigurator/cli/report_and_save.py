# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import random
import traceback
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from prettytable import PrettyTable

from aiconfigurator.generator.api import generate_backend_config
from aiconfigurator.generator.cli_args import build_dynamo_config
from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk.pareto_analysis import draw_pareto_to_string
from aiconfigurator.sdk.task import TaskConfig, task_config_to_generator_config
from aiconfigurator.sdk.utils import safe_mkdir

logger = logging.getLogger(__name__)


def _plot_worker_setup_table(exp_name: str, config_df: pd.DataFrame, total_gpus: int, tpot_target: float, top: int, is_moe: bool) -> str:
    """Plot worker setup table for a single experiment."""
    buf = []
    
    if config_df is None or config_df.empty:
        return ""

    config_df['tokens/s/gpu_cluster'] = config_df['tokens/s/gpu'] * (total_gpus // config_df['num_total_gpus']) \
        * config_df['num_total_gpus'] / total_gpus if total_gpus > 0 else 0
    top_configs = config_df[config_df['tpot'] <= tpot_target].sort_values(by='tokens/s/gpu_cluster', ascending=False).head(top).copy()
    
    if top_configs.empty:
        return f"\nNo configurations for {exp_name} met the TPOT constraint."

    top_configs['replicas'] = total_gpus // top_configs['num_total_gpus']
    top_configs['total_gpus_used'] = top_configs['num_total_gpus'] * top_configs['replicas']
    
    buf.append(f"\n{exp_name} Top Configurations: (Sorted by tokens/s/gpu)")
    table = PrettyTable()
    
    # Check if it is disagg config by checking for prefill/decode specific columns
    is_disagg = '(p)tp' in top_configs.columns

    if is_disagg:
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "TTFT", "concurrency", "total_gpus(used)", "replicas", "gpus/replica", 
                             "(p)workers", "(p)gpus/worker", "(p)parallel", "(p)bs",
                             "(d)workers", "(d)gpus/worker", "(d)parallel", "(d)bs"]
        for i, row in enumerate(top_configs.to_dict('records')):
            if is_moe:
                p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0mdp\033[4m{row["(p)dp"]}\033[0metp{row["(p)moe_tp"]}ep{row["(p)moe_ep"]}'
                d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0mdp\033[4m{row["(d)dp"]}\033[0metp{row["(d)moe_tp"]}ep{row["(d)moe_ep"]}'
                p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]*row["(p)dp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0mx\033[4m{row["(p)dp"]}\033[0m)'
                d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]*row["(d)dp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0mx\033[4m{row["(d)dp"]}\033[0m)'
            else:
                p_parallel = f'tp\033[4m{row["(p)tp"]}\033[0mpp\033[4m{row["(p)pp"]}\033[0m'
                d_parallel = f'tp\033[4m{row["(d)tp"]}\033[0mpp\033[4m{row["(d)pp"]}\033[0m'
                p_gpus_worker = f'{row["(p)pp"]*row["(p)tp"]} (=\033[4m{row["(p)tp"]}\033[0mx\033[4m{row["(p)pp"]}\033[0m)'
                d_gpus_worker = f'{row["(d)pp"]*row["(d)tp"]} (=\033[4m{row["(d)tp"]}\033[0mx\033[4m{row["(d)pp"]}\033[0m)'
            table.add_row([
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", f"{row['ttft']:.2f}",
                f"{row['concurrency']*row['replicas']}(={row['concurrency']}x{row['replicas']})",
                f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})", row['replicas'],
                f"{row['num_total_gpus']} (={row['(p)workers']}x{row['(p)pp']*row['(p)tp']*row['(p)dp']}+{row['(d)workers']}x{row['(d)pp']*row['(d)tp']*row['(d)dp']})",
                row['(p)workers'], p_gpus_worker, p_parallel, row['(p)bs'],
                row['(d)workers'], d_gpus_worker, d_parallel, row['(d)bs'],
            ])
    else: # agg
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "TTFT", "concurrency", "total_gpus(used)", 
                             "replicas", "gpus/replica", "gpus/worker", "parallel", "bs"]
        for i, row in enumerate(top_configs.to_dict('records')):
            if is_moe:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0mdp\033[4m{row["dp"]}\033[0metp{row["moe_tp"]}ep{row["moe_ep"]}'
                gpus_worker = f'{row["pp"]*row["tp"]*row["dp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0mx\033[4m{row["dp"]}\033[0m)'
            else:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0m'
                gpus_worker = f'{row["pp"]*row["tp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0m)'
            table.add_row([
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", f"{row['ttft']:.2f}",
                f"{row['concurrency']*row['replicas']}(={row['concurrency']}x{row['replicas']})", f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                row['replicas'], row['num_total_gpus'],
                gpus_worker, parallel, row['bs']
            ])
            
    buf.append(table.get_string())
    return "\n".join(buf)
    
def log_final_summary(
        chosen_exp: str, 
        best_throughputs: Dict[str, float], 
        best_configs: Dict[str, pd.DataFrame], 
        pareto_fronts: Dict[str, pd.DataFrame], 
        task_configs: Dict[str, TaskConfig],
        mode: str,
):
    """Log final summary of configuration results"""
    
    # Consolidate and format results into a summary box for clear presentation
    summary_box = []
    summary_box.append("*" * 80)
    summary_box.append("*{:^78}*".format(" Dynamo aiconfigurator Final Results "))
    summary_box.append("*" * 80)

    summary_box.append("  " + "-" * 76)
    summary_box.append("  Input Configuration & SLA Target:")
    summary_box.append(f"    Model: {task_configs[chosen_exp].config.model_name} (is_moe: {task_configs[chosen_exp].config.is_moe})")
    summary_box.append(f"    Total GPUs: {task_configs[chosen_exp].total_gpus}")
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
            benefit_ratio = 0.0 # handle case where both are 0
        summary_box.append(f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu (disagg {benefit_ratio:.2f}x better)\033[0m")        
    else:
        summary_box.append(f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu\033[0m")
        
    summary_box.append("  " + "-" * 76)


    # ============================= overall summary
    summary_box.append("  Overall Best Configuration:")
    best_config_df = best_configs[chosen_exp]
    best_throughput = best_throughputs[chosen_exp]
    
    summary_box.append(f"    - Best Throughput: {best_throughput:.2f} tokens/s/gpu")
    if not best_config_df.empty:
        best_conf_details = best_config_df.iloc[0]
        summary_box.append(f"    - User Throughput: {best_conf_details['tokens/s/user']:.2f} tokens/s/user")
        summary_box.append(f"    - TTFT: {best_conf_details['ttft']:.2f}ms")
        summary_box.append(f"    - TPOT: {best_conf_details['tpot']:.2f}ms")
    summary_box.append("  " + "-" * 76)

    # ============================= pareto frontier
    pareto_plot_buf = ""
    if len(pareto_fronts) <= 10:  # avoid overly crowded plots
        summary_box.append("  Pareto Frontier:")
        series_payload = [
            {"df": df, "label": name}
            for name, df in pareto_fronts.items()
            if df is not None and not df.empty
        ]
        highlight_series = None
        if not best_config_df.empty:
            highlight_series = {
                "df": best_config_df.head(1),
                "label": f"{chosen_exp} best",
            }
        pareto_plot_buf = draw_pareto_to_string(
            f"{task_configs[chosen_exp].config.model_name} Pareto Frontier",
            series_payload,
            highlight=highlight_series,
        )
        summary_box.append(pareto_plot_buf)
    summary_box.append("  " + "-" * 76)

    # ============================= deployment details
    summary_box.append("  Deployment Details:")
    summary_box.append(f"    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system")
    summary_box.append(f"    Some math: total gpus used = replicas * gpus/replica")
    summary_box.append(f"               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker")
    summary_box.append(f"               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined \033[4mnumbers\033[0m are the actual values in math)")
    
    # Plot worker setup tables for all experiments
    for exp_name, config_df in best_configs.items():
        exp_task_config = task_configs[exp_name].config
        total_gpus = getattr(task_configs[exp_name], "total_gpus", None) or 0
        table_buf = _plot_worker_setup_table(exp_name, config_df, total_gpus, exp_task_config.runtime_config.tpot, 5, exp_task_config.is_moe)
        summary_box.append(table_buf)

    summary_box.append("*" * 80)
    logger.info("\n" + "\n".join(summary_box))

def save_results(
    args,
    best_configs: Dict[str, pd.DataFrame], 
    pareto_fronts: Dict[str, pd.DataFrame], 
    task_configs: Dict[str, TaskConfig], 
    save_dir: str,
    generated_backend_version: Optional[str] = None,
):
    """Save the results to a directory."""
    
    first_exp_name = list(task_configs.keys())[0]
    first_task_config = task_configs[first_exp_name].config
    
    result_prefix = f"{first_task_config.model_name}_isl{first_task_config.runtime_config.isl}_osl{first_task_config.runtime_config.osl}_ttft{int(first_task_config.runtime_config.ttft)}_tpot{int(first_task_config.runtime_config.tpot)}"
    result_dir_path = os.path.join(save_dir, f'{result_prefix}_{random.randint(0,1000000)}')
    
    logger.info(f'Saving results to {result_dir_path}')
    try:
        safe_result_dir = safe_mkdir(result_dir_path, exist_ok=True)

        # Save overall pareto plots in the root directory
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        plt.title(f"{first_task_config.model_name} tokens/s/gpu vs tokens/s/user")
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, (exp_name, pareto_df) in enumerate(pareto_fronts.items()):
            if not pareto_df.empty:
                pareto_analysis.draw_pareto(
                    pareto_df, 'tokens/s/user', 'tokens/s/gpu', ax, colors[i % len(colors)], exp_name
                )
        plt.savefig(os.path.join(safe_result_dir, 'pareto_frontier.png'))
        plt.close()

        # Save each experiment's results in its own subdirectory
        for exp_name, pareto_df in pareto_fronts.items():
            exp_dir = os.path.join(safe_result_dir, exp_name)
            safe_mkdir(exp_dir, exist_ok=True)

            # 1. Save best config dataframe
            best_config_df = best_configs.get(exp_name) # top n configs
            if best_config_df is not None:
                best_config_df.to_csv(os.path.join(exp_dir, 'best_config_topn.csv'), index=False)

            # 2. Save all pareto dataframe
            if pareto_df is not None:
                pareto_df.to_csv(os.path.join(exp_dir, 'pareto.csv'), index=False)

            # 3. Save the config for this experiment
            exp_task_config = task_configs[exp_name]

            with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f: # for future aic repro
                yaml.safe_dump(json.loads(exp_task_config.pretty()), f, sort_keys=False)
            
            # 4. Save the generated config for this experiment, sub-directory for each best config
            if best_config_df is not None:
                dynamo_overrides = build_dynamo_config(args)
                for i, (idx, result_df) in enumerate(best_config_df.iterrows()):
                    cfg = task_config_to_generator_config(task_config=exp_task_config, result_df=result_df)

                    top_config_dir = os.path.join(exp_dir, f'top{i+1}')
                    safe_mkdir(top_config_dir, exist_ok=True)
                    with open(os.path.join(top_config_dir, 'generator_config.yaml'), 'w') as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    
                    try:
                        artifacts = generate_backend_config.from_runtime(
                            cfg=cfg,
                            backend=exp_task_config.backend_name,
                            version=generated_backend_version or exp_task_config.backend_version,
                            overrides=dynamo_overrides,                    
                            save_dir=top_config_dir,
                        )
                    except Exception as exc:
                        logger.warning("Failed to generate backend config from aic generator: %s, %s", exc, traceback.format_exc())

    except Exception as exc:
        logger.error("Failed to save results: %s, %s", exc, traceback.format_exc())
