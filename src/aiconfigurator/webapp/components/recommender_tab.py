# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr

from aiconfigurator.webapp.components.base import (
    create_model_misc_config,
    create_model_path_config,
    create_model_quant_config,
    create_runtime_config,
    create_system_config,
)


def create_recommender_tab(app_config):
    with gr.Tab("Sizing Recommender"):
        with gr.Accordion("Introduction"):
            gr.Markdown(
                label="introduction",
                value=r"""
                    **Sizing Recommender** finds the minimum GPU count and optimal deployment
                    configuration (agg vs disagg, TP/PP/DP) needed to meet a performance target.
                    Specify a model, system, backend, workload (ISL/OSL), SLA targets (TTFT/TPOT),
                    and either a target request rate (req/s) or target concurrency (concurrent users).
                    The recommender sweeps all valid parallelism configurations and computes the
                    minimum GPU count via replica scaling. Designed as a procurement sizing tool —
                    the output is unconstrained.
                """,
            )

        model_path_components = create_model_path_config(app_config)
        runtime_config_components = create_runtime_config(app_config, with_sla=True, with_request_latency=True)
        model_misc_config_components = create_model_misc_config(app_config)
        model_system_components = create_system_config(app_config)
        model_quant_components = create_model_quant_config(app_config)

        with gr.Row():
            target_request_rate = gr.Number(
                value=10,
                label="Target request rate (req/s)",
                info="Target system throughput. Fill this OR target concurrency, not both.",
            )
            target_concurrency = gr.Number(
                value=None,
                label="Target concurrency (concurrent users)",
                info="Target concurrent requests. Fill this OR target request rate, not both.",
            )

        recommend_btn = gr.Button("Recommend Sizing", visible=True)

        result_df = gr.Dataframe(
            label="GPU Recommendations",
            interactive=False,
            visible=True,
        )
        debugging_box = gr.Textbox(label="Debugging", lines=5, required=False, elem_classes=["debug-output"])

        download_btn = gr.Button("Download")
        output_file = gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        "model_path_components": model_path_components,
        "runtime_config_components": runtime_config_components,
        "model_system_components": model_system_components,
        "model_quant_components": model_quant_components,
        "model_misc_config_components": model_misc_config_components,
        "target_request_rate": target_request_rate,
        "target_concurrency": target_concurrency,
        "recommend_btn": recommend_btn,
        "result_df": result_df,
        "debugging_box": debugging_box,
        "download_btn": download_btn,
        "output_file": output_file,
    }
