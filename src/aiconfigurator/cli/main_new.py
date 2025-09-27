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
from typing import Dict, List, Tuple, Optional, Any, Union
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
import copy


logger = logging.getLogger(__name__)


def _add_default_mode_arguments(parser):
    parser.add_argument("--total_gpus", type=int, required=True, help="Total GPUs for deployment.")
    parser.add_argument("--model", choices=common.SupportedModels.keys(), type=str, required=True, help="Model name.")
    parser.add_argument("--system", type=str, required=True, help="Default system name.")
    parser.add_argument("--decode_system", type=str, default=None, help="System name for disagg decode workers. Defaults to --system if omitted.")
    parser.add_argument("--backend", choices=[backend.value for backend in common.BackendName], type=str, default=common.BackendName.trtllm.value, help="Backend name.")
    parser.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    parser.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    parser.add_argument("--ttft", type=float, default=1000.0, help="Time to first token in ms.")
    parser.add_argument("--tpot", type=float, default=20.0, help="Time per output token in ms.")


def _add_experiments_mode_arguments(parser):
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to a YAML file containing experiment definitions.")


def configure_parser(parser):
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save the results.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    default_parser = subparsers.add_parser("default", help="Run the default agg vs disagg comparison.")
    _add_default_mode_arguments(default_parser)

    experiments_parser = subparsers.add_parser("exp", help="Run one or more experiments defined in a YAML file.")
    _add_experiments_mode_arguments(experiments_parser)
    


def _build_default_task_configs(args) -> Dict[str, TaskConfig]:
    decode_system = args.decode_system or args.system
    common_kwargs: Dict[str, Any] = {
        "model_name": args.model,
        "system_name": args.system,
        "backend_name": args.backend,
        "total_gpus": args.total_gpus,
        "isl": args.isl,
        "osl": args.osl,
        "ttft": args.ttft,
        "tpot": args.tpot,
    }

    agg_task = TaskConfig(serving_mode="agg", **common_kwargs)

    disagg_kwargs = dict(common_kwargs)
    disagg_kwargs["decode_system_name"] = decode_system
    disagg_task = TaskConfig(serving_mode="disagg", **disagg_kwargs)

    return {"agg": agg_task, "disagg": disagg_task}


_EXPERIMENT_RESERVED_KEYS = {
    "mode",
    "serving_mode",
    "model_name",
    "system_name",
    "decode_system_name",
    "backend_name",
    "backend_version",
    "profiles",
    "isl",
    "osl",
    "ttft",
    "tpot",
    "enable_wide_ep",
    "total_gpus",
    "use_specific_quant_mode",
}


def _build_yaml_config(exp_config: dict, config_section: dict) -> Optional[dict]:
    if not config_section:
        config_section = {
            key: copy.deepcopy(value)
            for key, value in exp_config.items()
            if key not in _EXPERIMENT_RESERVED_KEYS
        }
    if not config_section:
        return None

    yaml_config = {
        "mode": exp_config.get("mode", "patch"),
        "config": config_section,
    }
    if "profiles" in exp_config:
        yaml_config["profiles"] = exp_config["profiles"]
    return yaml_config


def _build_experiment_task_configs(args) -> Dict[str, TaskConfig]:
    try:
        with open(args.yaml_path, "r", encoding="utf-8") as fh:
            experiment_data = yaml.safe_load(fh) or {}
    except Exception as exc:
        logger.error("Error loading experiment YAML file '%s': %s", args.yaml_path, exc)
        raise SystemExit(1) from exc

    if not isinstance(experiment_data, dict):
        logger.error("Experiment YAML root must be a mapping.")
        raise SystemExit(1)

    order = experiment_data.get("exps")
    if isinstance(order, list):
        experiment_names = [name for name in order if name in experiment_data]
    else:
        experiment_names = [name for name in experiment_data.keys() if name != "exps"]

    task_configs: Dict[str, TaskConfig] = {}

    for exp_name in experiment_names:
        exp_config = experiment_data.get(exp_name)
        if not isinstance(exp_config, dict):
            logger.warning("Skipping experiment '%s': configuration is not a mapping.", exp_name)
            continue

        config_section = exp_config.get("config")
        if not isinstance(config_section, dict):
            config_section = {}
        else:
            config_section = copy.deepcopy(config_section)

        serving_mode = exp_config.get("serving_mode")
        model_name = exp_config.get("model_name")
        if serving_mode not in {"agg", "disagg"} or not model_name:
            logger.warning("Skipping experiment '%s': missing serving_mode or model_name.", exp_name)
            continue

        # system
        if serving_mode == "agg":
            inferred_system = exp_config.get("system_name")
            inferred_decode_system = None
        else:
            inferred_system = exp_config.get("system_name")
            inferred_decode_system = exp_config.get("decode_system_name") or inferred_system
        system_name = inferred_system
        if not system_name:
            logger.warning(
                "Skipping experiment '%s': no system name provided (provide system_name at the top level or inside worker config).",
                exp_name,
            )
            continue

        # backend, default to trtllm
        if serving_mode == "agg":
            backend_name = exp_config.get("backend_name") or common.BackendName.trtllm.value
            backend_version = exp_config.get("backend_version")
        else:
            backend_name = exp_config.get("backend_name") or common.BackendName.trtllm.value
            backend_version = exp_config.get("backend_version")

        total_gpus = exp_config.get("total_gpus")
        if total_gpus is None:
            logger.warning("Skipping experiment '%s': total_gpus not provided in YAML.", exp_name)
            continue

        task_kwargs: Dict[str, Any] = {
            "serving_mode": serving_mode,
            "model_name": model_name,
            "system_name": system_name,
            "backend_name": backend_name,
            "total_gpus": total_gpus,
        }

        if backend_version is not None:
            task_kwargs["backend_version"] = backend_version

        if serving_mode == "disagg":
            task_kwargs["decode_system_name"] = inferred_decode_system or system_name

        # Per-experiment overrides for runtime numeric parameters if provided at top level
        for numeric_key in ("isl", "osl", "ttft", "tpot"):
            if numeric_key in exp_config:
                task_kwargs[numeric_key] = exp_config[numeric_key]

        if "enable_wide_ep" in exp_config:
            task_kwargs["enable_wide_ep"] = exp_config["enable_wide_ep"]
        if "use_specific_quant_mode" in exp_config:
            task_kwargs["use_specific_quant_mode"] = exp_config["use_specific_quant_mode"]

        yaml_config = _build_yaml_config(exp_config, config_section)
        if yaml_config:
            task_kwargs["yaml_config"] = yaml_config

        try:
            task_configs[exp_name] = TaskConfig(**task_kwargs)
        except Exception as exc:
            logger.error("Failed to build TaskConfig for experiment '%s': %s", exp_name, exc)
            logger.exception("Full traceback")

    if not task_configs:
        logger.error("No valid experiments found in '%s'.", args.yaml_path)
        raise SystemExit(1)

    return task_configs


def _execute_task_configs(task_configs: Dict[str, TaskConfig], save_dir: Optional[str]) -> None:
    results: Dict[str, pd.DataFrame] = {}
    start_time = time.time()
    runner = TaskRunner()

    for exp_name, task_config in task_configs.items():
        try:
            logger.info("Starting experiment: %s", exp_name)
            logger.info("Task config: %s", task_config.pretty())
            result_df = runner.run(task_config)
            if result_df is not None and not result_df.empty:
                results[exp_name] = result_df
                logger.info("Experiment %s completed with %d results.", exp_name, len(result_df))
            else:
                logger.warning("Experiment %s returned no results.", exp_name)
        except Exception as exc:
            logger.error("Error running experiment %s: %s", exp_name, exc)
            logger.exception("Full traceback")

    if len(results) < 1:
        logger.error("No successful experiment runs to compare.")
        raise SystemExit(1)

    pareto_fronts = {name: get_pareto_front(df, 'tokens/s/user', 'tokens/s/gpu') for name, df in results.items()}

    best_configs: Dict[str, pd.DataFrame] = {}
    best_throughputs: Dict[str, float] = {}
    for name, pareto_df in pareto_fronts.items():
        tpot_target = task_configs[name].config.runtime_config.tpot
        per_total = getattr(task_configs[name], "total_gpus", None) or 0
        best_config_df = get_best_config_under_tpot_constraint(per_total, pareto_df, tpot_target)
        best_configs[name] = best_config_df
        if not best_config_df.empty:
            best_throughputs[name] = best_config_df['tokens/s/gpu_cluster'].values[0]
        else:
            best_throughputs[name] = 0.0

    chosen_exp = max(best_throughputs, key=best_throughputs.get) if best_throughputs else "none"

    log_final_summary(chosen_exp, best_throughputs, best_configs, pareto_fronts, task_configs)

    if save_dir:
        save_results(chosen_exp, best_configs, pareto_fronts, task_configs, save_dir)

    end_time = time.time()
    logger.info("All experiments completed in %.2f seconds", end_time - start_time)


def main(args):
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s')

    logger.info(f"Loading Dynamo AIConfigurator version: {__version__}")

    if args.mode == "default":
        task_configs = _build_default_task_configs(args)
    elif args.mode == "exp":
        task_configs = _build_experiment_task_configs(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

    _execute_task_configs(task_configs, args.save_dir)

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
    
def log_final_summary(chosen_exp: str, best_throughputs: Dict[str, float], best_configs: Dict[str, pd.DataFrame], pareto_fronts: Dict[str, pd.DataFrame], task_configs: Dict[str, TaskConfig]):
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
    summary_box.append(f"    Total GPUs: {task_configs[first_exp_name].total_gpus}")
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
        per_total = getattr(task_configs[exp_name], "total_gpus", None) or 0
        table_buf = _plot_worker_setup_table(exp_name, pareto_df, per_total, exp_task_config.runtime_config.tpot, 5, exp_task_config.is_moe)
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
    