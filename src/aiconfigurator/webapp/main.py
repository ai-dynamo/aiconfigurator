# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging
from collections import defaultdict

import gradio as gr

import aiconfigurator
from aiconfigurator.webapp.components.agg_pareto_tab import create_agg_pareto_tab
from aiconfigurator.webapp.components.agg_tab import create_agg_tab
from aiconfigurator.webapp.components.disagg_pareto_tab import create_disagg_pareto_tab
from aiconfigurator.webapp.components.disagg_pd_ratio_tab import create_disagg_pd_ratio_tab
from aiconfigurator.webapp.components.pareto_comparison_tab import create_pareto_comparison_tab
from aiconfigurator.webapp.components.profiling.create_profiling_tab import create_profiling_tab
from aiconfigurator.webapp.components.readme_tab import create_readme_tab
from aiconfigurator.webapp.components.static_tab import create_static_tab
from aiconfigurator.webapp.events.event_handler import EventHandler


def configure_parser(parser):
    """
    Configures the argument parser for the WebApp.
    """
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server name")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--enable_agg", action="store_true", help="Enable Agg tab")
    parser.add_argument("--enable_disagg_pd_ratio", action="store_true", help="Enable Disagg PD Ratio tab")
    parser.add_argument("--enable_profiling", action="store_true", help="Enable Profiling tab")
    parser.add_argument("--debug", help="Debug mode", action="store_true")
    parser.add_argument("--experimental", help="enable experimental features", action="store_true")


def main(args):
    """
    Main function for the WebApp.
    """
    app_config = {
        "enable_agg": args.enable_agg,
        "enable_disagg_pd_ratio": args.enable_disagg_pd_ratio,
        "enable_profiling": args.enable_profiling,
        "experimental": args.experimental,
        "debug": args.debug,
    }

    if app_config["debug"]:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(asctime)s] %(message)s",
            datefmt="%m-%d %H:%M:%S",
        )

    with gr.Blocks(
        title="Dynamo aiconfigurator for Disaggregated Serving Deployment",
        css="""
        .config-column {
            border-right: 5px solid #e0e0e0;
            padding-right: 20px;
        }
        .config-column:last-child {
            border-right: none;
        }
    """,
    ) as demo:
        pareto_results_state = gr.State(defaultdict())

        # title
        with gr.Row():
            gr.Markdown(
                f"""
                <div style="text-align: center;">
                    <h1>Dynamo aiconfigurator for Disaggregated Serving Deployment</h1>
                    <p style="font-size: 14px; margin-top: -10px;">
                        Version {aiconfigurator.__version__}
                    </p>
                    <p style="font-size: 12px; margin-top: -10px; color: #666;">
                        Use of this service is for test and evaluation purposes only.
                        Results are estimates and may be inaccurate.
                        The AI Configurator software available at
                        <a href="https://github.com/ai-dynamo/aiconfigurator/" target="_blank">
                        https://github.com/ai-dynamo/aiconfigurator/</a>
                        is governed by the Apache 2.0 License.
                    </p>
                </div>
                """
            )

        # create tabs
        with gr.Tabs():
            readme_components = create_readme_tab(app_config)  # noqa: F841
            static_components = create_static_tab(app_config)
            if app_config["enable_agg"]:
                agg_components = create_agg_tab(app_config)
            agg_pareto_components = create_agg_pareto_tab(app_config)
            disagg_pareto_components = create_disagg_pareto_tab(app_config)
            if app_config["enable_disagg_pd_ratio"]:
                disagg_pd_ratio_components = create_disagg_pd_ratio_tab(app_config)
            pareto_comparison_components = create_pareto_comparison_tab(app_config)
            if app_config["enable_profiling"]:
                profiling_components = create_profiling_tab(app_config)

        # setup events
        EventHandler.setup_static_events(static_components)
        if app_config["enable_agg"]:
            EventHandler.setup_agg_events(agg_components)
        EventHandler.setup_agg_pareto_events(agg_pareto_components)
        EventHandler.setup_disagg_pareto_events(disagg_pareto_components)
        EventHandler.setup_save_events(
            agg_pareto_components["result_name"],
            agg_pareto_components["save_btn"],
            agg_pareto_components["result_df"],
            pareto_comparison_components["candidates_dropdown"],
            pareto_results_state,
        )
        EventHandler.setup_save_events(
            disagg_pareto_components["result_name"],
            disagg_pareto_components["save_btn"],
            disagg_pareto_components["result_df"],
            pareto_comparison_components["candidates_dropdown"],
            pareto_results_state,
        )
        if app_config["enable_disagg_pd_ratio"]:
            EventHandler.setup_disagg_pd_ratio_events(disagg_pd_ratio_components)
        EventHandler.setup_pareto_comparison_events(pareto_comparison_components, pareto_results_state)
        if app_config["enable_profiling"]:
            EventHandler.setup_profiling_events(profiling_components)

        demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamo aiconfigurator Web App")
    configure_parser(parser)
    args = parser.parse_args()
    main(args)
