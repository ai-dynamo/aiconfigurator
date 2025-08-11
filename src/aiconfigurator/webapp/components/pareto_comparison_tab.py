# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gradio as gr

def create_pareto_comparison_tab(app_config):
    with gr.Tab("Pareto Comparison"):

        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r'''
                    **Please read the readme tab before using this tab.**  
                    This mode is used to compare the Pareto frontier of different runs from ifb or disaggregation.  
                    Please select the results you want to compare, it allows multiple selections.  
                    The system will draw the Pareto frontier for each selected result.  
                '''
            )
        
        candidates_dropdown = gr.Dropdown(
            choices=[],
            label="Select results for comparison (multiple allowed)",
            multiselect=True
        )
        with gr.Row():
            compare_btn = gr.Button("Compare selected results")
            clear_btn = gr.Button("Clear all results")

        pareto_html = gr.HTML(value='')
        download_btn = gr.Button("Download Pareto HTML")
        output_file = gr.File(label="Pareto HTML")

    return {
        "candidates_dropdown": candidates_dropdown,
        "compare_btn": compare_btn,
        "clear_btn": clear_btn,
        "pareto_html": pareto_html,
        "download_btn": download_btn,
        "output_file": output_file,
        "introduction": introduction
    }