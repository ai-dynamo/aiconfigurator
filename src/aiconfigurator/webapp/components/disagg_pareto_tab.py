# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiconfigurator.webapp.components.base import create_model_name_config,create_system_config, create_model_quant_config, create_model_parallel_config, create_runtime_config, create_model_misc_config
import gradio as gr
from aiconfigurator.sdk.common import ColumnsDisagg

def create_disagg_pareto_tab(app_config):
    with gr.Tab("Disaggregation Pareto Estimation"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**  
                    This mode is used to estimate the Pareto frontier for the disaggregation of the model.  
                    The Pareto frontier is the set of points that represent the best trade-off between the tokens/s/user and the tokens/s/gpu.  
                    Please click the 'Estimate' button to run the estimation. It will takes **a few minutes** to complete.  
                    If you would like to compare this result with other runs, you can click 'Save for comparison' button below the estimation buttion. And switch to Pareto Comparison tab for more info.
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
                prefill_model_parallel_components = create_model_parallel_config(app_config, single_select=False, disagg=True)
                prefill_correction_scale = gr.Number(value=1.0, label="Prefill correction scale", interactive=True)
            with gr.Column(elem_classes="config-column"):
                gr.Markdown("### Decode config")
                decode_model_system_components = create_system_config(app_config)
                decode_model_quant_components = create_model_quant_config(app_config)                
                decode_model_parallel_components = create_model_parallel_config(app_config, single_select=False, disagg=True)
                decode_correction_scale = gr.Number(value=1.0, label="Decode correction scale", interactive=True)
        runtime_config_components['tpot'].visible = False

        # constraint section
        with gr.Accordion("Advanced settings"):
            gr.Markdown(r'''
                        **refer to readme tab for more details.**  
                        Constraints on searching the disagg system:  
                        1. num_gpu_list controls the exact target num gpus of the disagg system, say, 8,16 will help to find
                            the disagg system with 8 or 16 gpus. You can specify it by a list separated by comma.
                            If it's not specified, it will ignore this setting.  
                        2. max_num_gpu will further filter out those exceed the limit. It can also work without num_gpu_list. Which is also
                            a common use case to limit the total gpus in a disagg system.  
                        3. prefill and decode max num workers is controling the max number of the prefill/decode workers in a disagg system.  
                            By Default, it's 32. If you don't want a large system like this, you can set it to a smaller value such as 8
                            which means it will only search for the disagg system with up to 8 workers of prefill or decode.
                        4. prefill and decode max batch size is controling the max batch size of the prefill/decode workers in a disagg system.
                            By default, it's 1 for prefill and 512 for decode. Why 1 for prefill: in concurrency-based bench method, it's difficult to saturate prefill batch size slot.
                        ''')
            with gr.Row():
                num_gpu_list = gr.Textbox(value=None, label="num total gpu list of the disagg system, say,: 8,16", interactive=True)
                max_num_gpu = gr.Number(value=None, label="max gpus used in the disagg system", interactive=True)
            with gr.Row():
                prefill_max_num_worker = gr.Number(value=32, label="Prefill max num worker", interactive=True)
                decode_max_num_worker = gr.Number(value=32, label="Decode max num worker", interactive=True)
            with gr.Row():
                prefill_max_batch_size = gr.Number(value=1, label="Prefill max batch size", interactive=True)
                decode_max_batch_size = gr.Number(value=512, label="Decode max batch size", interactive=True)
        
        estimate_btn = gr.Button("Estimate Disaggregation Pareto")
        with gr.Row():
            result_name = gr.Textbox(value="", label="Result name")
            save_btn = gr.Button("Save for comparison", interactive=False)

        pareto_html = gr.HTML(value='')
        result_df = gr.Dataframe(
            label="Disaggregation pareto datapoints",
            headers=['index']+ColumnsDisagg,
            interactive=False,
            visible=True,
            height=300
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
        'prefill_correction_scale': prefill_correction_scale,
        'decode_model_system_components': decode_model_system_components,
        'decode_model_quant_components': decode_model_quant_components,
        'decode_model_parallel_components': decode_model_parallel_components,
        'decode_correction_scale': decode_correction_scale,
        'num_gpu_list': num_gpu_list,
        'max_num_gpu': max_num_gpu,
        'prefill_max_num_worker': prefill_max_num_worker,
        'decode_max_num_worker': decode_max_num_worker,
        'prefill_max_batch_size': prefill_max_batch_size,
        'decode_max_batch_size': decode_max_batch_size,
        'estimate_btn': estimate_btn,
        'pareto_html': pareto_html,
        'result_df': result_df,
        'debugging_box': debugging_box,
        'result_name': result_name,
        'save_btn': save_btn,
        'download_btn': download_btn,
        'output_file': output_file,
        'introduction': introduction
    }