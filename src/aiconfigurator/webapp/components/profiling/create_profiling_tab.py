# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import gradio as gr

from aiconfigurator.webapp.components.base import create_model_name_config, create_runtime_config, create_system_config
from aiconfigurator.webapp.components.profiling.constants import (
    COST_TAB_DESCRIPTION,
    DECODE_TAB_DESCRIPTION,
    PLOT_INTERACTION_INSTRUCTIONS,
    PREFILL_TAB_DESCRIPTION,
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
        "gpu_cost_toggle.js",
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


def create_profiling_ui_components():
    """
    Create hidden UI components for profiling (selection input/button, JSON data).

    Returns:
        dict: Dictionary with selection_input, selection_button, and json_data components
    """
    selection_input = gr.Textbox(value="", visible=True, elem_id="profiling_selection_input", container=False)
    selection_button = gr.Button("Submit Selection", visible=True, elem_id="profiling_selection_button")
    json_data = gr.Textbox(value="", visible=False, elem_id="profiling_json_data")

    return {
        "selection_input": selection_input,
        "selection_button": selection_button,
        "json_data": json_data,
    }


def create_performance_results_section():
    """
    Create the Performance Results section with all tabs (Prefill, Decode, Cost).

    This section contains the interactive charts and tables for profiling results.
    """
    with gr.Accordion("Performance Results"):
        gr.Markdown(PLOT_INTERACTION_INSTRUCTIONS)

        with gr.Tab("Cost vs SLA"):
            with gr.Accordion("Description", open=True):
                gr.Markdown(COST_TAB_DESCRIPTION)
            show_gpu_cost = gr.Checkbox(  # noqa: F841 - accessed via JS
                label="Show GPU cost",
                value=False,
                elem_id="show_gpu_cost_checkbox",
            )
            gpu_cost_per_hr = gr.Number(  # noqa: F841 - accessed via JS
                label="GPU cost / hr ($)",
                value=2.00,
                minimum=0,
                elem_id="gpu_cost_per_hr_input",
                interactive=True,
                optional=True,
            )
            gr.HTML('<div class="chart-container"><canvas id="cost_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="cost_table_wrapper"></div>')

        with gr.Tab("Prefill Performance"):
            with gr.Accordion("Description", open=True):
                gr.Markdown(PREFILL_TAB_DESCRIPTION)
            gr.HTML('<div class="chart-container"><canvas id="prefill_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="prefill_table_wrapper"></div>')

        with gr.Tab("Decode Performance"):
            with gr.Accordion("Description", open=True):
                gr.Markdown(DECODE_TAB_DESCRIPTION)
            gr.HTML('<div class="chart-container"><canvas id="decode_chart"></canvas></div>')
            gr.Markdown("#### Data Points")
            gr.HTML('<div id="decode_table_wrapper"></div>')


def _setup_button_validation(generate_btn, model_name_components, model_system_components, runtime_config_components):
    """
    Setup validation to enable the generate button only when all required fields are filled.

    Args:
        generate_btn: The generate button component
        model_name_components: Model name components dictionary
        model_system_components: System components dictionary
        runtime_config_components: Runtime config components dictionary
    """

    def validate_fields(model_name, system, backend, version, min_gpu, max_gpu, gpus_per_node, isl, osl):
        """Check if all required fields are filled."""
        # Check dropdowns - all must be selected
        dropdowns_filled = all([model_name, system, backend, version])

        # Check number fields - must have valid numeric values (not None, not empty string)
        numbers_filled = all(
            [
                min_gpu is not None and min_gpu != "",
                max_gpu is not None and max_gpu != "",
                gpus_per_node is not None and gpus_per_node != "",
                isl is not None and isl != "",
                osl is not None and osl != "",
            ]
        )

        return gr.update(interactive=dropdowns_filled and numbers_filled)

    # Get all required components
    required_inputs = [
        model_name_components["model_name"],
        model_system_components["system"],
        model_system_components["backend"],
        model_system_components["version"],
        model_system_components["min_gpu_per_engine"],
        model_system_components["max_gpu_per_engine"],
        model_system_components["gpus_per_node"],
        runtime_config_components["isl"],
        runtime_config_components["osl"],
    ]

    # Attach change and blur listeners to all required fields
    # - change: fires when value changes
    # - blur: fires when field loses focus (catches clearing fields reliably)
    for component in required_inputs:
        component.change(
            fn=validate_fields,
            inputs=required_inputs,
            outputs=generate_btn,
        )
        # Also listen to blur event to catch field clearing
        if hasattr(component, "blur"):
            component.blur(
                fn=validate_fields,
                inputs=required_inputs,
                outputs=generate_btn,
            )


def create_setup_section(app_config):
    """
    Create the Setup Your Profiling Job section.

    Args:
        app_config: Application configuration

    Returns:
        dict: Dictionary with all setup components
    """
    with gr.Accordion("Introduction"):
        introduction = gr.Markdown(
            label="introduction",
            value=r"""Generates profiling data for the model.""",
        )

    with gr.Accordion("Setup Your Profiling Job"):
        model_name_components = create_model_name_config(app_config)
        model_system_components = create_system_config(app_config, gpu_config=True)
        runtime_config_components = create_runtime_config(app_config, with_sla=True, prefix_length=False)
        generate_btn = gr.Button("Generate Profiling Job", variant="primary", interactive=False)
        status = gr.Textbox(
            label="Status",
            value="Ready to generate profiling plots",
            interactive=False,
            show_label=False,
            lines=5,
        )

    # Setup validation to enable button when all required fields are filled
    _setup_button_validation(
        generate_btn,
        model_name_components,
        model_system_components,
        runtime_config_components,
    )

    return {
        "introduction": introduction,
        "model_name_components": model_name_components,
        "model_system_components": model_system_components,
        "runtime_config_components": runtime_config_components,
        "generate_btn": generate_btn,
        "status": status,
    }


def inject_profiling_assets():
    """Inject CSS for profiling visualization. Modal is injected via JS to stay outside Gradio."""
    gr.HTML(_load_profiling_css())


def create_profiling_tab(app_config):
    """
    Create the full profiling tab with setup and results sections.

    Args:
        app_config: Application configuration

    Returns:
        dict: Dictionary with all profiling components
    """
    with gr.Tab("Profiling") as profiling_tab:
        # Create hidden UI components
        ui_components = create_profiling_ui_components()

        # Create setup section
        setup_components = create_setup_section(app_config)

        # Create performance results section
        create_performance_results_section()

    # Load JavaScript when profiling tab is selected
    profiling_tab.select(fn=None, js=_load_profiling_javascript())

    # Combine all components
    components = {**setup_components, **ui_components}
    setup_profiling_events(components)

    # Inject CSS and modal
    inject_profiling_assets()

    return components
