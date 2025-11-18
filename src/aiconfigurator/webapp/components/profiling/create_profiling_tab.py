import gradio as gr

from aiconfigurator.webapp.components.base import create_model_name_config, create_runtime_config, create_system_config
from aiconfigurator.webapp.components.profiling.constants import (
    CONFIG_MODAL_HTML,
    PLOT_INTERACTION_INSTRUCTIONS,
    TABLE_CSS,
)
from aiconfigurator.webapp.components.profiling.create_results_tabs import create_results_tabs, get_empty_tables
from aiconfigurator.webapp.events.event_profiler import setup_profiling_events


def create_profiling_tab(app_config):
    with gr.Tab("Profiling"):
        # Inject CSS for table styling
        gr.HTML(TABLE_CSS)
        # Inject modal HTML for config display
        gr.HTML(CONFIG_MODAL_HTML)
        with gr.Accordion("Introduction"):
            introduction = gr.Markdown(
                label="introduction",
                value=r"""Generates profiling data for the model.""",
            )
        with gr.Accordion("Setup Your Profiling Job"):
            model_name_components = create_model_name_config(app_config)
            model_system_components = create_system_config(app_config, gpu_config=True)
            runtime_config_components = create_runtime_config(
                app_config, with_sla=True, max_context_length=True, prefix_length=False
            )
            generate_btn = gr.Button("Generate Profiling Job", variant="primary")
            status = gr.Textbox(
                label="Status",
                value="Ready to generate profiling plots",
                interactive=False,
                show_label=False,
                lines=5,
            )
            empty_prefill_html, empty_decode_html, empty_cost_html = get_empty_tables()

    with gr.Accordion("Performance Results"):
        gr.Markdown(PLOT_INTERACTION_INSTRUCTIONS)
        results_tabs = create_results_tabs(empty_prefill_html, empty_decode_html, empty_cost_html)

    components = {
        "introduction": introduction,
        "model_name_components": model_name_components,
        "model_system_components": model_system_components,
        "runtime_config_components": runtime_config_components,
        "generate_btn": generate_btn,
        "status": status,
        "results_tabs": results_tabs,
    }
    setup_profiling_events(components)
    return components
