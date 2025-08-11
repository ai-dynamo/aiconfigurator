# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr
from aiconfigurator.webapp.components.static_tab import create_static_tab
from aiconfigurator.webapp.components.ifb_tab import create_ifb_tab
from aiconfigurator.webapp.components.ifb_pareto_tab import create_ifb_pareto_tab
from aiconfigurator.webapp.components.disagg_pareto_tab import create_disagg_pareto_tab
from aiconfigurator.webapp.components.pareto_comparison_tab import create_pareto_comparison_tab
from aiconfigurator.webapp.components.disagg_pd_ratio_tab import create_disagg_pd_ratio_tab
from aiconfigurator.webapp.components.readme_tab import create_readme_tab
from aiconfigurator.webapp.events.event_handler import EventHandler
from collections import defaultdict
import argparse
import logging
import aiconfigurator
import sys
from typing import List

def configure_parser(parser):
    """
    Configures the argument parser for the WebApp.
    """
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--enable_ifb", action="store_true", help="Enable IFB tab")
    parser.add_argument("--enable_disagg_pd_ratio", action="store_true", help="Enable Disagg PD Ratio tab")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--experimental", help="enable experimental features", action="store_true")

def main(args):
    """
    Main function for the WebApp.
    """
    app_config = {
        'enable_ifb': args.enable_ifb,
        'enable_disagg_pd_ratio': args.enable_disagg_pd_ratio,
        'experimental': args.experimental,
        'debug': args.debug,
    }

    if app_config['debug']:
        logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
                    datefmt="%m-%d %H:%M:%S")
    else:
        logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(asctime)s] %(message)s",
                    datefmt="%m-%d %H:%M:%S")
    
    with gr.Blocks(css="""
        .config-column {
            border-right: 5px solid #e0e0e0;
            padding-right: 20px;
        }
        .config-column:last-child {
            border-right: none;
        }
    """) as demo:
        pareto_results_state = gr.State(defaultdict())

        # title
        with gr.Row():
            gr.Markdown(
                f"""
                <div style="text-align: center;">
                    <h1>Dynamo aiconfigurator for Disaggregated Serving Deployment</h1>
                    <p style="font-size: 14px; margin-top: -10px;">Version {aiconfigurator.__version__}</p>
                </div>
                """
            )
        
        # create tabs
        with gr.Tabs() as tabs:
            readme_components = create_readme_tab(app_config)            
            static_components = create_static_tab(app_config)
            if app_config['enable_ifb']:
                ifb_components = create_ifb_tab(app_config)
            ifb_pareto_components = create_ifb_pareto_tab(app_config)
            disagg_pareto_components = create_disagg_pareto_tab(app_config)
            if app_config['enable_disagg_pd_ratio']:
                disagg_pd_ratio_components = create_disagg_pd_ratio_tab(app_config)
            pareto_comparison_components = create_pareto_comparison_tab(app_config)
        
        # setup events
        EventHandler.setup_static_events(static_components)
        if app_config['enable_ifb']:
            EventHandler.setup_ifb_events(ifb_components)
        EventHandler.setup_ifb_pareto_events(ifb_pareto_components)
        EventHandler.setup_disagg_pareto_events(disagg_pareto_components)
        EventHandler.setup_save_events(ifb_pareto_components['result_name'], ifb_pareto_components['save_btn'], ifb_pareto_components['result_df'], pareto_comparison_components['candidates_dropdown'], pareto_results_state)
        EventHandler.setup_save_events(disagg_pareto_components['result_name'], disagg_pareto_components['save_btn'], disagg_pareto_components['result_df'], pareto_comparison_components['candidates_dropdown'], pareto_results_state)
        if app_config['enable_disagg_pd_ratio']:
            EventHandler.setup_disagg_pd_ratio_events(disagg_pd_ratio_components)
        EventHandler.setup_pareto_comparison_events(pareto_comparison_components, pareto_results_state)

        demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo aiconfigurator Web App")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
