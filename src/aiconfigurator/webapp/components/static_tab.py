# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.components.base import create_model_name_config, create_system_config, create_model_quant_config, create_model_parallel_config, create_runtime_config, create_model_misc_config
import gradio as gr
from aiconfigurator.sdk.common import ColumnsStatic

def create_static_tab(app_config):
    with gr.Tab("Static Estimation"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**  
                    This mode is used to estimate the static batching performance of the model.
                '''
            )
    
        model_name_components = create_model_name_config(app_config)
        runtime_config_components = create_runtime_config(app_config)
        model_system_components = create_system_config(app_config)
        model_quant_components = create_model_quant_config(app_config)
        model_parallel_components = create_model_parallel_config(app_config, single_select=True)
        model_misc_config_components = create_model_misc_config(app_config)
        mode = gr.Dropdown(choices=['static', 'static_ctx', 'static_gen'], 
                                value='static', label='inference mode', interactive=True)
        estimate_btn = gr.Button("Estimate Static Inference")
        with gr.Row():
            summary_box = gr.Markdown( 
                label="Summary", 
            )
            context_breakdown_box = gr.Markdown(
                label="Context Breakdown", 
            )
            generation_breakdown_box = gr.Markdown(
                label="Generation Breakdown", 
            )
        record_df = gr.Dataframe(
            label="Records:",
            headers=ColumnsStatic,
            interactive=False
        )
        debugging_box = gr.Textbox(label="Debugging", lines=5)
        with gr.Row():
            clear_btn = gr.Button("Clear")
            download_btn = gr.Button("Download")
        output_file=gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        'model_name_components': model_name_components,
        'runtime_config_components': runtime_config_components,
        'model_system_components': model_system_components,
        'model_quant_components': model_quant_components,
        'model_parallel_components': model_parallel_components,
        'model_misc_config_components': model_misc_config_components,
        'mode': mode,
        'estimate_btn': estimate_btn,
        'summary_box': summary_box,
        'context_breakdown_box': context_breakdown_box,
        'generation_breakdown_box': generation_breakdown_box,
        'record_df': record_df,
        'debugging_box': debugging_box,
        'clear_btn': clear_btn,
        'download_btn': download_btn,
        'output_file': output_file,
        'introduction': introduction
    }