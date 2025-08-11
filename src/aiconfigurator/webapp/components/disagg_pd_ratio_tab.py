# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.components.base import create_model_name_config,create_system_config, create_model_quant_config, create_model_parallel_config, create_runtime_config, create_model_misc_config
import gradio as gr
from aiconfigurator.sdk.common import ColumnsStatic

def create_disagg_pd_ratio_tab(app_config):
    with gr.Tab("Disaggregation PD Ratio Analysis"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**  
                    This mode is used to analyze p/d ratioin disaggregated system.  
                    Please click the 'Estimate' button to run the estimation. It will takes **a few seconds** to complete.  
                    The throughput curve as well as the result dataframe will be shown below.  
                    A dotted vertical line will be shown on the throughput curve, which indicates the ttft limit and tpot limit for prefill and decode respectively.  
                    This mode is used to get a matched p/d ratio for the disaggregated system. Based on the throughput curve, you can select a point which is close to the ttft limit and tpot limit.  
                    You will then have the seq/s of prefill and decode respectively. Setting num_workers of p and d to make seq/s of p_total and d_total close to each other.  
                    E.g., if seq/s of p is 10, seq/s of d is 20, you can set num_workers of p (provider) to 2 and num_workers of d (consumer) to 1. You will get a system throughput of 20.
                '''
            )

        model_name_components = create_model_name_config(app_config)
        runtime_config_components = create_runtime_config(app_config, with_sla=True)        
        model_misc_config_components = create_model_misc_config(app_config)
        with gr.Row():       
            with gr.Column(elem_classes="config-column"):
                gr.Markdown("### Prefill config")
                prefill_model_system_components = create_system_config(app_config)
                prefill_model_quant_components = create_model_quant_config(app_config)
                prefill_model_parallel_components = create_model_parallel_config(app_config, single_select=True, disagg=False)
            with gr.Column(elem_classes="config-column"):
                gr.Markdown("### Decode config")
                decode_model_system_components = create_system_config(app_config)
                decode_model_quant_components = create_model_quant_config(app_config)                
                decode_model_parallel_components = create_model_parallel_config(app_config, single_select=True, disagg=False)
        
        estimate_btn = gr.Button("Estimate Prefill/Decode Throughput")

        with gr.Row():
            prefill_throughput_html = gr.HTML(value='')
            decode_throughput_html = gr.HTML(value='')

        with gr.Row():
            prefill_result_df = gr.Dataframe(
                label="Prefill datapoints",
                headers=['index']+ColumnsStatic,
                interactive=False,
                visible=True,
                height=300,
            )
            decode_result_df = gr.Dataframe(
                label="Decode datapoints",
                headers=['index']+ColumnsStatic,
                interactive=False,
                visible=True,
                height=300,
            )
        debugging_box = gr.Textbox(label="Debugging", lines=5)

        download_btn = gr.Button("Download")
        output_file=gr.File(label="When you click the download button, the downloaded form will be displayed here.")

    return {
        'model_name_components': model_name_components,
        'runtime_config_components': runtime_config_components,
        'model_misc_config_components': model_misc_config_components,
        'prefill_model_system_components': prefill_model_system_components,
        'prefill_model_quant_components': prefill_model_quant_components,
        'prefill_model_parallel_components': prefill_model_parallel_components,
        'decode_model_system_components': decode_model_system_components,
        'decode_model_quant_components': decode_model_quant_components,
        'decode_model_parallel_components': decode_model_parallel_components,
        'estimate_btn': estimate_btn,
        'prefill_throughput_html': prefill_throughput_html,
        'decode_throughput_html': decode_throughput_html,
        'prefill_result_df': prefill_result_df,
        'decode_result_df': decode_result_df,
        'debugging_box': debugging_box,
        'download_btn': download_btn,
        'output_file': output_file,
        'introduction': introduction
    }