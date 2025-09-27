# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from aiconfigurator.sdk import pareto_analysis
from aiconfigurator.sdk import config
from aiconfigurator.sdk.perf_database import get_database
from aiconfigurator.sdk import common
from aiconfigurator.sdk.models import get_model_family, check_is_moe
from aiconfigurator.cli.helpers import DynamoConfig, add_dynamo_cli, build_dynamo_config, _dump_backend_file
from aiconfigurator.cli.backends import get_config_generator
from aiconfigurator.sdk.pareto_analysis import draw_pareto_to_string, get_pareto_front, interpolate_throughput_at_tpot, get_best_config_under_tpot_constraint
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.interpolate import interp1d
import yaml
import argparse
import plotext
import time
import os
import random
import matplotlib.pyplot as plt
import json
from prettytable import PrettyTable
from aiconfigurator import __version__


logger = logging.getLogger(__name__)


def configure_parser(parser):
    """
    Configures the argument parser for the CLI.
    """
    # Core arguments
    parser.add_argument("--total_gpus", type=int, required=True, help="Total GPUs for deployment.")
    parser.add_argument("--model", choices=common.SupportedModels.keys(), type=str, required=True, help="Model name.")
    parser.add_argument("--system", type=str, required=True, help="Default system name.")
    parser.add_argument("--decode_system", type=str, default=None, help="System name for disagg decode workers. Default is None, which means using the same system name as the default system name.")
    parser.add_argument("--backend", choices=[backend.value for backend in common.BackendName], type=str, default=common.BackendName.trtllm.value, help="Backend name.")
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    parser.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    parser.add_argument("--ttft", type=float, default=1000, help="Time to first token in ms.")
    parser.add_argument("--tpot", type=float, default=20, help="Time per output token in ms.")
    
    # General options
    parser.add_argument("--yaml_path", type=str, default=None, help="Path to a YAML file with experiment definitions. CLI arguments below can override YAML values.")    
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the results.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    # TODO: These are for backend config generation, need to see how to fit them in the new flow.
    # chosen_backend = add_dynamo_cli(parser, common.BackendName.trtllm.value)

    # parser.epilog = (
    #     "\nNOTE:\n"
    #     "  • The extra dynamo_config.* parameters shown here are for "
    #     f"backend '{chosen_backend}'.\n"
    #     "  • To see configuration for another backend run:\n"
    #     "        aiconfigurator cli --backend <backend_name> --help\n"
    # )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

def main(args):
    """
    Main function for the CLI.
    """
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, 
                        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')
    
    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")

    if args.decode_system is None:
        args.decode_system = args.system

    task_configs = {}
    
    if args.yaml_path:
        # YAML mode
        logger.info(f"Loading experiments from {args.yaml_path}")
        try:
            with open(args.yaml_path, 'r') as f:
                experiment_configs = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading experiment YAML file: {e}")
            exit(1)

        # Collect non-None CLI args to pass to TaskConfig constructor
        cli_args = {
            'model_name': args.model,
            'system_name': args.system,
            'decode_system_name': args.decode_system,
            'backend_name': args.backend,
            'isl': args.isl,
            'osl': args.osl,
            'ttft': args.ttft,
            'tpot': args.tpot,
        }
        cli_args = {k: v for k, v in cli_args.items() if v is not None}

        for exp_name, exp_config in experiment_configs.items():
            if not isinstance(exp_config, dict) or 'serving_mode' not in exp_config:
                logger.warning(f"Skipping invalid experiment configuration: {exp_name}")
                continue
            
            # The constructor will use its defaults, then CLI args, then yaml_config will overwrite all.
            task_init_args = {
                'serving_mode': exp_config['serving_mode'],
                'model_name': exp_config.get('model_name'),
                **cli_args, # CLI args take precedence over internal defaults
                'yaml_config': exp_config
            }
            if not task_init_args['model_name']:
                logger.error(f"Experiment '{exp_name}' is missing required 'model_name'.")
                continue
            
            task_config = TaskConfig(**task_init_args)
            if 
            task_configs[exp_name] = task_config
    else:
        # Default mode
        logger.info("No YAML path provided, running with default settings (agg vs. disagg).")
        
        # Create Agg Task
        task_configs['agg'] = TaskConfig(
            serving_mode='agg',
            model_name=args.model,
            system_name=args.system,
            backend_name=args.backend,
            isl=args.isl,
            osl=args.osl,
            ttft=args.ttft,
            tpot=args.tpot,
            max_gpu_per_replica=args.total_gpus,
        )

        # Create Disagg Task
        task_configs['disagg'] = TaskConfig(
            serving_mode='disagg',
            model_name=args.model,
            system_name=args.system,
            decode_system_name=args.decode_system,
            backend_name=args.backend,
            isl=args.isl,
            osl=args.osl,
            ttft=args.ttft,
            tpot=args.tpot,
            max_gpu_per_replica=args.total_gpus,
        )
        
    results = {}
    start_time = time.time()
    
    # Run each experiment
    for exp_name, task_config in task_configs.items():
        try:
            logger.info(f"Starting experiment: {exp_name}")
            task_runner = TaskRunner()
            result_df = task_runner.run(task_config)
            
            if result_df is not None and not result_df.empty:
                results[exp_name] = result_df
                logger.info(f"Experiment {exp_name} completed with {len(result_df)} results.")
            else:
                logger.warning(f"Experiment {exp_name} returned no results.")

        except Exception as e:
            logger.error(f"Error running experiment {exp_name}: {e}")
            logger.exception("Full traceback")

    # After all experiments are run, proceed with comparison and summary
    if len(results) < 1:
        logger.error("No successful experiment runs to compare.")
        exit(1)

    # Step 2: Get Pareto frontiers (tps/gpu vs. tps/user) for all experiments
    pareto_fronts = {name: get_pareto_front(df, 'tokens/s/user', 'tokens/s/gpu') for name, df in results.items()}

    # Step 3: Get top1 actual config under TPOT constraint for each experiment
    best_configs = {}
    best_throughputs = {}
    for name, pareto_df in pareto_fronts.items():
        tpot_target = task_configs[name].config.runtime_config.tpot
        best_config_df = get_best_config_under_tpot_constraint(args.total_gpus, pareto_df, tpot_target)
        best_configs[name] = best_config_df
        if not best_config_df.empty:
            best_throughputs[name] = best_config_df['tokens/s/gpu_cluster'].values[0]
        else:
            best_throughputs[name] = 0.0

    # Step 4: Compare performance and select overall best experiment
    chosen_exp = "none"
    if best_throughputs:
        chosen_exp = max(best_throughputs, key=best_throughputs.get)

    log_final_summary(chosen_exp, best_throughputs, best_configs, pareto_fronts, task_configs, args.total_gpus)

    if args.save_dir:
        save_results(chosen_exp, best_configs, pareto_fronts, task_configs, args.save_dir)

    end_time = time.time()
    logger.info(f"All experiments completed in {end_time - start_time:.2f} seconds")

def _plot_worker_setup_table(exp_name: str, pareto_df: pd.DataFrame, total_gpus: int, tpot_target: float, top: int, is_moe: bool) -> str:
    """Plot worker setup table for a single experiment."""
    buf = []
    
    if pareto_df is None or pareto_df.empty:
        return ""

    pareto_df['tokens/s/gpu_cluster'] = pareto_df['tokens/s/gpu'] * (total_gpus // pareto_df['num_total_gpus']) \
        * pareto_df['num_total_gpus'] / total_gpus if total_gpus > 0 else 0
    top_configs = pareto_df[pareto_df['tpot'] <= tpot_target].sort_values(by='tokens/s/gpu_cluster', ascending=False).head(top).copy()
    
    if top_configs.empty:
        return f"\nNo configurations for {exp_name} met the TPOT constraint."

    top_configs['replicas'] = total_gpus // top_configs['num_total_gpus']
    top_configs['total_gpus_used'] = top_configs['num_total_gpus'] * top_configs['replicas']
    
    buf.append(f"\n{exp_name} Top Configurations: (Sorted by tokens/s/gpu)")
    table = PrettyTable()
    
    # Check if it is disagg config by checking for prefill/decode specific columns
    is_disagg = '(p)tp' in top_configs.columns

    if is_disagg:
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "concurrency", "total_gpus(used)", "replicas", "gpus/replica", 
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
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}", row['concurrency'],
                f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})", row['replicas'],
                f"{row['num_total_gpus']} (={row['(p)workers']}x{row['(p)pp']*row['(p)tp']*row['(p)dp']}+{row['(d)workers']}x{row['(d)pp']*row['(d)tp']*row['(d)dp']})",
                row['(p)workers'], p_gpus_worker, p_parallel, row['(p)bs'],
                row['(d)workers'], d_gpus_worker, d_parallel, row['(d)bs'],
            ])
    else: # agg
        table.field_names = ["Rank", f"\033[1mtokens/s/gpu\033[0m", "tokens/s/user", "concurrency", "total_gpus(used)", 
                             "replicas", "gpus/replica", "gpus/worker", "parallel", "bs"]
        for i, row in enumerate(top_configs.to_dict('records')):
            if is_moe:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0mdp\033[4m{row["dp"]}\033[0metp{row["moe_tp"]}ep{row["moe_ep"]}'
                gpus_worker = f'{row["pp"]*row["tp"]*row["dp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0mx\033[4m{row["dp"]}\033[0m)'
            else:
                parallel = f'tp\033[4m{row["tp"]}\033[0mpp\033[4m{row["pp"]}\033[0m'
                gpus_worker = f'{row["pp"]*row["tp"]} (=\033[4m{row["tp"]}\033[0mx\033[4m{row["pp"]}\033[0m)'
            table.add_row([
                i + 1, f"\033[1m{row['tokens/s/gpu_cluster']:.2f}\033[0m", f"{row['tokens/s/user']:.2f}",
                row['concurrency'], f"{total_gpus} ({row['total_gpus_used']}={row['replicas']}x{row['num_total_gpus']})",
                row['replicas'], row['num_total_gpus'],
                gpus_worker, parallel, row['bs']
            ])
            
    buf.append(table.get_string())
    return "\n".join(buf)
    
def log_final_summary(chosen_exp: str, best_throughputs: Dict[str, float], best_configs: Dict[str, pd.DataFrame], pareto_fronts: Dict[str, pd.DataFrame], task_configs: Dict[str, TaskConfig], total_gpus: int):
    """Log final summary of configuration results"""
    
    # Consolidate and format results into a summary box for clear presentation
    summary_box = []
    summary_box.append("*" * 80)
    summary_box.append("*{:^78}*".format(" Dynamo aiconfigurator Final Results "))
    summary_box.append("*" * 80)

    summary_box.append("  " + "-" * 76)
    summary_box.append("  Input Configuration & SLA Target:")
    # Find the first experiment to get model and is_moe
    first_exp_name = list(task_configs.keys())[0]
    first_task_config = task_configs[first_exp_name].config
    summary_box.append(f"    Model: {first_task_config.model_name} (is_moe: {first_task_config.is_moe})")
    summary_box.append(f"    Total GPUs: {total_gpus}")
    summary_box.append(f"    Best Experiment Chosen: \033[1m{chosen_exp} at {best_throughputs[chosen_exp]:.2f} tokens/s/gpu\033[0m")
    summary_box.append("  " + "-" * 76)


    # ============================= overall summary
    summary_box.append("  Overall Best Configuration:")
    best_config_df = best_configs[chosen_exp]
    best_throughput = best_throughputs[chosen_exp]
    
    summary_box.append(f"    - Best Throughput: {best_throughput:.2f} tokens/s/gpu")
    if not best_config_df.empty:
        best_conf_details = best_config_df.iloc[0]
        summary_box.append(f"      - User Throughput: {best_conf_details['tokens/s/user']:.2f} tokens/s/user")
        summary_box.append(f"      - TTFT: {best_conf_details['ttft']:.2f}ms")
        summary_box.append(f"      - TPOT: {best_conf_details['tpot']:.2f}ms")
    summary_box.append("  " + "-" * 76)

    # ============================= pareto frontier
    if len(pareto_fronts) < 3:  # Only show command line plot for small number of experiments
        summary_box.append("  Pareto Frontier:")
        
        # Prepare data for plotting - use first two experiments or all if less than 2
        exp_names = list(pareto_fronts.keys())
        pareto1 = pareto_fronts[exp_names[0]] if len(exp_names) > 0 else pd.DataFrame()
        pareto2 = pareto_fronts[exp_names[1]] if len(exp_names) > 1 else pd.DataFrame()
        
        pareto_plot_buf = draw_pareto_to_string(f"{first_task_config.model_name} Pareto Frontier", 
                                                        best_config_df,
                                                        pareto1, pareto2)
        summary_box.append(pareto_plot_buf)
    summary_box.append("  " + "-" * 76)

    # ============================= deployment details
    summary_box.append("  Deployment Details:")
    summary_box.append(f"    (p) stands for prefill, (d) stands for decode, bs stands for batch size, a replica stands for the smallest scalable unit xPyD of the disagg system")
    summary_box.append(f"    Some math: total gpus used = replicas * gpus/replica")
    summary_box.append(f"               gpus/replica = (p)gpus/worker * (p)workers + (d)gpus/worker * (d)workers; for Agg, gpus/replica = gpus/worker")
    summary_box.append(f"               gpus/worker = tp * pp * dp = etp * ep * pp for MoE models; tp * pp for dense models (underlined \033[4mnumbers\033[0m are the actual values in math)")
    
    # Plot worker setup tables for all experiments
    for exp_name, pareto_df in pareto_fronts.items():
        exp_task_config = task_configs[exp_name].config
        table_buf = _plot_worker_setup_table(exp_name, pareto_df, total_gpus, exp_task_config.runtime_config.tpot, 5, exp_task_config.is_moe)
        summary_box.append(table_buf)

    summary_box.append("*" * 80)
    logger.info("\n" + "\n".join(summary_box))

def save_results(chosen_exp: str, best_configs: Dict[str, pd.DataFrame], pareto_fronts: Dict[str, pd.DataFrame], task_configs: Dict[str, TaskConfig], save_dir: str):
    """Save the results to a directory."""
    
    first_exp_name = list(task_configs.keys())[0]
    first_task_config = task_configs[first_exp_name].config
    
    result_prefix = f"{first_task_config.model_name}_isl{first_task_config.runtime_config.isl}_osl{first_task_config.runtime_config.osl}_ttft{int(first_task_config.runtime_config.ttft)}_tpot{int(first_task_config.runtime_config.tpot)}"
    result_dir_path = os.path.join(save_dir, f'{result_prefix}_{random.randint(0,1000000)}')
    
    logger.info(f'Saving results to {result_dir_path}')
    os.makedirs(result_dir_path, exist_ok=True)

    for exp_name, pareto_df in pareto_fronts.items():
        pareto_df.to_csv(os.path.join(result_dir_path, f'{exp_name}_pareto.csv'), index=False)
        best_configs[exp_name].to_csv(os.path.join(result_dir_path, f'{exp_name}_best_config.csv'), index=False)

    # Save plot with all pareto fronts
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    plt.title(f"{first_task_config.model_name} tokens/s/gpu vs tokens/s/user")
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    for i, (exp_name, pareto_df) in enumerate(pareto_fronts.items()):
        if not pareto_df.empty:
            pareto_analysis.draw_pareto(pareto_df, 'tokens/s/user', 'tokens/s/gpu', ax, colors[i % len(colors)], exp_name)
        
    plt.savefig(os.path.join(result_dir_path, 'pareto_frontier.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo AIConfigurator for Disaggregated Serving Deployment")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
    