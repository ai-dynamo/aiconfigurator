"""
Event handler for profiling tab.

This module sets up event handlers for the profiling functionality,
connecting UI components to the profiling orchestration logic.
"""

from aiconfigurator.webapp.components.profiling.profiler import generate_profiling_plots


def setup_profiling_events(components):
    """
    Set up event handlers for profiling tab interactions.

    Args:
        components: Dictionary of all UI components from create_profiling_tab
    """
    # Extract nested component dictionaries
    model_name_components = components["model_name_components"]
    model_system_components = components["model_system_components"]
    runtime_config_components = components["runtime_config_components"]
    results_tabs = components["results_tabs"]

    # Extract individual components
    model_name = model_name_components["model_name"]
    system = model_system_components["system"]
    backend = model_system_components["backend"]
    version = model_system_components["version"]
    min_gpu_per_engine = model_system_components["min_gpu_per_engine"]
    max_gpu_per_engine = model_system_components["max_gpu_per_engine"]
    gpu_cost_per_hour = model_system_components["gpu_cost_per_hour"]

    isl = runtime_config_components["isl"]
    osl = runtime_config_components["osl"]
    ttft = runtime_config_components["ttft"]
    tpot = runtime_config_components["tpot"]

    # Result components
    prefill_plot = results_tabs["prefill_plot"]
    decode_plot = results_tabs["decode_plot"]
    cost_plot = results_tabs["cost_plot"]
    prefill_table = results_tabs["prefill_table"]
    decode_table = results_tabs["decode_table"]
    cost_table = results_tabs["cost_table"]

    status = components["status"]
    generate_btn = components["generate_btn"]

    # Prepare inputs for the generate function
    inputs = [
        model_name,
        system,
        backend,
        version,
        min_gpu_per_engine,
        max_gpu_per_engine,
        gpu_cost_per_hour,
        isl,
        osl,
        ttft,
        tpot,
    ]

    # Prepare outputs
    outputs = [
        prefill_plot,
        decode_plot,
        cost_plot,
        status,
        prefill_table,
        decode_table,
        cost_table,
    ]

    # Wire up the button click event
    generate_btn.click(
        fn=generate_profiling_plots,
        inputs=inputs,
        outputs=outputs,
    )
