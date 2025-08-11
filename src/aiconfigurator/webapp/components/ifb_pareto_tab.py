# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.components.base import create_model_name_config, create_system_config, create_model_quant_config, create_model_parallel_config, create_runtime_config, create_model_misc_config
import gradio as gr
from aiconfigurator.sdk.common import ColumnsIFB

def create_ifb_pareto_tab(app_config):
    with gr.Tab("Agg(IFB) Pareto Estimation"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**  
                    This mode is used to estimate the Pareto frontier for the in-flight(continous) batching of the model.  
                    The Pareto frontier is the set of points that represent the best trade-off between the tokens/s/user and the tokens/s/gpu.  
                    Please click the 'Estimate' button to run the estimation. It will takes **a few minutes** to complete.  
                    If you would like to compare this result with other runs, you can click 'Save for comparison' button below the estimation buttion. And switch to Pareto Comparison tab for more info.
                '''
            )
    
        model_name_components = create_model_name_config(app_config)
        runtime_config_components = create_runtime_config(app_config, with_sla=True)
        model_misc_config_components = create_model_misc_config(app_config)
        model_system_components = create_system_config(app_config)
        model_quant_components = create_model_quant_config(app_config)
        model_parallel_components = create_model_parallel_config(app_config, single_select=False)

        runtime_config_components['tpot'].visible = False
        
        estimate_btn = gr.Button('Estimate IFB Pareto', visible=True)
        with gr.Row():
            result_name = gr.Textbox(value="", label="Result name")
            save_btn = gr.Button("Save for comparison", interactive=False)

        pareto_html = gr.HTML(value='')
        result_df = gr.Dataframe(
            label="IFB pareto datapoints",
            headers=['index']+ColumnsIFB,
            interactive=False,
            visible=True
        )
        debugging_box = gr.Textbox(label="Debugging", lines=5)

        download_btn = gr.Button("Download")
        output_file=gr.File(label="When you click the download button, the downloaded form will be displayed here.")
        
    return {
        'model_name_components': model_name_components,
        'runtime_config_components': runtime_config_components,
        'model_system_components': model_system_components,
        'model_quant_components': model_quant_components,
        'model_parallel_components': model_parallel_components,
        'model_misc_config_components': model_misc_config_components,
        'estimate_btn': estimate_btn,
        'result_df': result_df,
        'debugging_box': debugging_box,
        'pareto_html': pareto_html,
        'result_name': result_name,
        'save_btn': save_btn,
        'download_btn': download_btn,
        'output_file': output_file,
        'introduction': introduction
    }