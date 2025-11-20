from pathlib import Path

import gradio as gr

from aiconfigurator.webapp.components.base import create_model_name_config, create_runtime_config, create_system_config
from aiconfigurator.webapp.components.profiling.constants import (
    CONFIG_MODAL_HTML,
    PLOT_INTERACTION_INSTRUCTIONS,
)
from aiconfigurator.webapp.events.event_profiler import setup_profiling_events


def _load_profiling_javascript():
    """
    Load all JavaScript modules for Chart.js + DataTables visualization.

    Returns:
        str: Combined JavaScript code wrapped in an IIFE for Gradio's js parameter
    """
    profiling_dir = Path(__file__).parent
    js_dir = profiling_dir / "js"

    # Load all JS files IN THEIR ORDER
    js_files = [
        "cdn_loader.js",
        "chart_renderer.js",
        "table_renderer.js",
        "sync_interactions.js",
        "config_modal.js",
        "main.js",
    ]

    combined_js = []
    for js_file in js_files:
        js_path = js_dir / js_file
        with open(js_path) as f:
            combined_js.append(f"// ===== {js_file} =====")
            combined_js.append(f.read())

    js_code = "\n\n".join(combined_js)
    return f"() => {{ {js_code} }}"


def _load_profiling_css():
    """Load CSS for profiling visualization."""
    profiling_dir = Path(__file__).parent
    css_path = profiling_dir / "styles.css"

    with open(css_path) as f:
        return f"<style>\n{f.read()}\n</style>"


def create_profiling_tab(app_config):
    with gr.Tab("Profiling") as profiling_tab:
        # Hidden input and button for selection callback (visible=True but hidden via CSS)
        selection_input = gr.Textbox(value="", visible=True, elem_id="profiling_selection_input", container=False)
        selection_button = gr.Button("Submit Selection", visible=True, elem_id="profiling_selection_button")

        # Hidden component for JSON data
        json_data = gr.Textbox(value="", visible=False, elem_id="profiling_json_data")

        # Inject CSS and modal
        gr.HTML(_load_profiling_css())
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

    with gr.Accordion("Performance Results"):
        gr.Markdown(PLOT_INTERACTION_INSTRUCTIONS)

        with gr.Tab("Prefill Performance"):
            gr.HTML('<div class="chart-container"><canvas id="prefill_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="prefill_table_wrapper"></div>')

        with gr.Tab("Decode Performance"):
            gr.HTML('<div class="chart-container"><canvas id="decode_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="decode_table_wrapper"></div>')

        with gr.Tab("Cost vs SLA"):
            gr.HTML('<div class="chart-container"><canvas id="cost_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="cost_table_wrapper"></div>')

    # Load JavaScript when profiling tab is selected
    profiling_tab.select(fn=None, js=_load_profiling_javascript())

    components = {
        "introduction": introduction,
        "model_name_components": model_name_components,
        "model_system_components": model_system_components,
        "runtime_config_components": runtime_config_components,
        "generate_btn": generate_btn,
        "status": status,
        "json_data": json_data,
        "selection_input": selection_input,
        "selection_button": selection_button,
    }
    setup_profiling_events(components)
    return components
