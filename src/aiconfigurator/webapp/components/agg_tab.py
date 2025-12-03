# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr

from aiconfigurator.sdk.common import ColumnsAgg
from aiconfigurator.webapp.components.base import (
    create_model_misc_config,
    create_model_name_config,
    create_model_parallel_config,
    create_model_quant_config,
    create_runtime_config,
    create_system_config,
)


def create_agg_tab(app_config):
    with gr.Tab("Agg Estimation"):
        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r"""
                    **Please read the readme tab before using this tab.**  
                    This mode is used to estimate the in-flight(continous) batching performance of the model and provide configuration suggestion.  
                    Please click the 'Estimate' button to run the estimation. It will takes **a few minutes** to complete.
                """,
            )

        model_name_components = create_model_name_config(app_config)
        runtime_config_components = create_runtime_config(app_config, with_sla=True)
        model_system_components = create_system_config(app_config)
        model_quant_components = create_model_quant_config(app_config)
        model_parallel_components = create_model_parallel_config(app_config, single_select=True)
        model_misc_config_components = create_model_misc_config(app_config)
        # agg section, by default, they are invisible
        estimate_btn = gr.Button("Estimate Agg Inference", visible=True)
        result_df = gr.Dataframe(label="Suggested Config List", headers=ColumnsAgg, interactive=False, visible=True)
        debugging_box = gr.Textbox(label="Debugging", lines=5, required=False)
        download_btn = gr.Button("Download")
        output_file = gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        "model_name_components": model_name_components,
        "runtime_config_components": runtime_config_components,
        "model_system_components": model_system_components,
        "model_quant_components": model_quant_components,
        "model_parallel_components": model_parallel_components,
        "model_misc_config_components": model_misc_config_components,
        "estimate_btn": estimate_btn,
        "result_df": result_df,
        "debugging_box": debugging_box,
        "download_btn": download_btn,
        "output_file": output_file,
        "introduction": introduction,
    }
