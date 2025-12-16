# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone Profiling Viewer with Hardcoded Data

This script demonstrates the Performance Results visualization
with hardcoded profiling data and a working "Select" button.

This script reuses components from the main profiling module without
duplicating code or embedding JavaScript strings.
"""

import json
from pathlib import Path

import gradio as gr

from aiconfigurator.webapp.components.profiling import (
    create_performance_results_section,
    create_profiling_ui_components,
    inject_profiling_assets,
    load_profiling_javascript,
)


def get_hardcoded_profiling_data():
    """
    Load profiling data from JSON file.

    This data matches the structure produced by the profiling orchestrator.
    To customize the data, edit the sample_profiling_data.json file.
    """
    data_file = Path(__file__).parent / "sample_profiling_data.json"
    with open(data_file) as f:
        data = json.load(f)
    return json.dumps(data)


def on_selection_button_click(selection_json):
    """Handle datapoint selection when button is clicked."""
    print(f"\n[STANDALONE VIEWER] Button clicked! Received: {selection_json[:100] if selection_json else 'None'}")

    if not selection_json or selection_json.strip() == "":
        print("[STANDALONE VIEWER] Empty selection received")
        return

    try:
        selection = json.loads(selection_json)
        print("\n" + "=" * 80)
        print("🎯 DATAPOINT SELECTED!")
        print("=" * 80)
        print(f"📊 Plot Type: {selection['plotType']}")
        print(f"📍 Row Index: {selection['rowIndex']}")
        print(f"🕐 Timestamp: {selection['timestamp']}")
        print("\n📋 Row Data:")
        for idx, value in enumerate(selection["rowData"]):
            print(f"  Column {idx}: {value}")
        print("=" * 80 + "\n")
    except Exception as e:
        print(f"[STANDALONE VIEWER ERROR] Failed to parse selection: {e!s}")


def create_standalone_viewer():
    """Create the standalone profiling viewer interface using refactored components."""
    with gr.Blocks(title="Standalone Profiling Viewer") as demo:
        # Create hidden UI components (reused from profiling module)
        ui_components = create_profiling_ui_components()
        selection_input = ui_components["selection_input"]
        selection_button = ui_components["selection_button"]
        json_data = ui_components["json_data"]

        # Inject CSS and modal (reused from profiling module)
        inject_profiling_assets()

        # Title
        gr.Markdown("# 📊 Standalone Profiling Viewer")
        gr.Markdown(
            """
            This is a standalone viewer with **profiling data loaded automatically** from JSON.

            ✅ Data loads automatically on page load

            ✅ **"Select" button** works - click it to print datapoint to console

            ❌ **"Show Config" button** can be hidden by requirements
            """
        )

        # Performance Results Section (reused from profiling module)
        create_performance_results_section()

        # Trigger visualization when JSON data changes
        json_data.change(
            fn=None,
            inputs=[json_data],
            outputs=[],
            js=(
                "(data) => { if (data && data.trim() && window.initializeVisualizations) "
                "window.initializeVisualizations(data); }"
            ),
        )

        # Handle selection button
        selection_button.click(
            fn=on_selection_button_click,
            inputs=[selection_input],
            outputs=[],
        )

        # Load JavaScript and data automatically on page load
        def load_all():
            """Load JavaScript and data automatically."""
            return get_hardcoded_profiling_data()

        demo.load(fn=load_all, inputs=[], outputs=[json_data], js=load_profiling_javascript())

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Starting Standalone Profiling Viewer")
    print("=" * 80)
    print("\n📝 Instructions:")
    print("  1. Data will load automatically when the page opens")
    print("  2. Interact with the charts and tables")
    print("  3. Click 'Select' button in any table row to see the datapoint printed here")
    print("\n" + "=" * 80 + "\n")

    demo = create_standalone_viewer()
    demo.launch(server_name="0.0.0.0", share=False)
