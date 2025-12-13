# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Event handler for profiling tab.

This module sets up event handlers for the profiling functionality,
connecting UI components to the profiling orchestration logic.
"""

import gradio as gr

from aiconfigurator.webapp.components.profiling.core.orchestrator import generate_profiling_plots


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

    # Extract individual components
    model_name = model_name_components["model_name"]
    system = model_system_components["system"]
    backend = model_system_components["backend"]
    version = model_system_components["version"]
    min_gpu_per_engine = model_system_components["min_gpu_per_engine"]
    max_gpu_per_engine = model_system_components["max_gpu_per_engine"]

    isl = runtime_config_components["isl"]
    osl = runtime_config_components["osl"]
    ttft = runtime_config_components["ttft"]
    tpot = runtime_config_components["tpot"]

    status = components["status"]
    json_data = components["json_data"]
    selection_input = components["selection_input"]
    selection_button = components["selection_button"]
    generate_btn = components["generate_btn"]

    # Add checkbox for Select button control
    allow_confirm = gr.Checkbox(value=False, visible=False)

    # Prepare inputs for the generate function
    inputs = [
        model_name,
        system,
        backend,
        version,
        min_gpu_per_engine,
        max_gpu_per_engine,
        isl,
        osl,
        ttft,
        tpot,
        allow_confirm,
    ]

    # Prepare outputs (now just JSON data and status)
    outputs = [json_data, status]

    # Wire up the button click event
    generate_btn.click(
        fn=generate_profiling_plots,
        inputs=inputs,
        outputs=outputs,
    )

    # Wire up JSON data change to trigger visualization
    json_data.change(
        fn=None,
        inputs=[json_data],
        outputs=[],
        js=(
            "(data) => { if (data && data.trim() && window.initializeVisualizations) "
            "window.initializeVisualizations(data); }"
        ),
    )

    # Wire up selection button - when clicked, read the input and process
    def on_selection_button_click(selection_json):
        """Handle datapoint selection when button is clicked."""
        print(f"[PROFILING DEBUG] Button clicked! Received: {selection_json[:100] if selection_json else 'None'}")

        if not selection_json or selection_json.strip() == "":
            print("[PROFILING] Empty selection received")
            return

        import json

        try:
            selection = json.loads(selection_json)
            print("\n" + "=" * 60)
            print("[PROFILING] Datapoint Selected!")
            print("=" * 60)
            print(f"Plot Type: {selection['plotType']}")
            print(f"Row Index: {selection['rowIndex']}")
            print(f"Timestamp: {selection['timestamp']}")
            print("\nRow Data:")
            for idx, value in enumerate(selection["rowData"]):
                print(f"  Column {idx}: {value}")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"[PROFILING ERROR] Failed to parse selection: {e!s}")

    selection_button.click(
        fn=on_selection_button_click,
        inputs=[selection_input],
        outputs=[],
    )
