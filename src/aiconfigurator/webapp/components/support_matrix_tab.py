# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from aiconfigurator.sdk import common


def parse_version(ver_str):
    """
    Parse version string into comparable tuple.
    Handles formats like '1.0.0', '1.0.0rc6', '0.5.6.post2', etc.
    Returns a tuple for comparison.
    """
    if not ver_str or not isinstance(ver_str, str):
        return (0, 0, 0, -1, 0)

    version_str = ver_str.lower()

    # Extract numeric version pattern (e.g., "1.2.3" from "v1.2.3rc4" or "1.2.3_suffix")
    version_match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if version_match:
        major, minor, patch = map(int, version_match.groups())
        version_parts = [major, minor, patch]

        # Handle post releases (e.g., "0.5.6.post2")
        if ".post" in version_str:
            post_match = re.search(r"\.post(\d+)", version_str)
            if post_match:
                post_num = int(post_match.group(1))
                version_parts.append(2)  # Post release indicator (higher than stable)
                version_parts.append(post_num)
            else:
                version_parts.extend([2, 0])
        # Handle release candidates (lower priority than stable releases)
        elif "rc" in version_str:
            rc_match = re.search(r"rc(\d+)", version_str)
            if rc_match:
                rc_num = int(rc_match.group(1))
                version_parts.append(0)  # RC indicator
                version_parts.append(rc_num)  # RC number
            else:
                version_parts.extend([0, 0])
        else:
            version_parts.append(1)  # Stable release (higher priority than RC)
            version_parts.append(0)  # No RC number

        return tuple(version_parts)

    # Try to extract version from other patterns (e.g., "v0.20_fix0719")
    version_match = re.search(r"v?(\d+)\.(\d+)", version_str)
    if version_match:
        major, minor = map(int, version_match.groups())
        version_parts = [major, minor, 0, 1, 0]  # Assume stable release
        return tuple(version_parts)

    # For completely non-standard versions, try to extract any numbers
    numbers = re.findall(r"\d+", version_str)
    if numbers:
        # Use first few numbers found, pad with zeros
        version_parts = [int(x) for x in numbers[:3]]
        while len(version_parts) < 3:
            version_parts.append(0)
        version_parts.extend([0, 0])  # Add RC indicators
        return tuple(version_parts)

    # If no numbers found, return a very low priority tuple
    return (0, 0, 0, -1, 0)


# Hard-coded latest versions for each backend
LATEST_VERSIONS = {
    "vllm": ["0.16.0"],
    "sglang": ["0.5.9"],
    "trtllm": ["1.2.0rc6.post3", "1.2.0rc6"],  # Check both variants, post3 is newer
}


def get_latest_supported_version(df, huggingface_id, system, backend):
    """
    Get the latest version status for a given HuggingFace ID and system combination.
    Uses hard-coded latest versions. If the latest version fails, returns "FAIL".

    Returns:
        tuple: (version, is_latest, error_msg) where:
            - version: Latest version string if latest version passes, "FAIL" if latest version fails,
                       None if no data exists
            - is_latest: True if the returned version is the hard-coded latest backend version, False otherwise
            - error_msg: Error message if version fails, None otherwise
    """
    # Filter for this specific combination (both PASS and FAIL)
    subset = df[(df["HuggingFaceID"] == huggingface_id) & (df["System"] == system) & (df["Backend"] == backend)]

    if subset.empty:
        return (None, False, None)

    # Get all versions with their statuses and error messages
    # For each version, track if it has any FAIL entries and collect error messages
    version_has_fail = {}
    version_has_pass = {}
    version_error_msgs = {}  # Store error messages for failed versions

    for _, row in subset.iterrows():
        version = row["Version"]
        status = row["Status"]
        # Get error message - check if ErrMsg column exists
        error_msg = None
        if "ErrMsg" in row.index:
            error_msg = row["ErrMsg"]
            # Handle NaN/None values
            if pd.isna(error_msg):
                error_msg = None
            else:
                error_msg = str(error_msg).strip()
                if not error_msg:
                    error_msg = None

        if status == "FAIL":
            version_has_fail[version] = True
            # Collect error messages, combine if multiple modes have errors
            if error_msg:
                if version in version_error_msgs:
                    # Combine error messages from different modes
                    existing = version_error_msgs[version]
                    if existing and existing != error_msg:
                        version_error_msgs[version] = f"{existing} | {error_msg}"
                    # If same message, keep it
                else:
                    version_error_msgs[version] = error_msg
        elif status == "PASS":
            version_has_pass[version] = True

    if len(version_has_fail) == 0 and len(version_has_pass) == 0:
        return (None, False, None)

    # Get the hard-coded latest versions for this backend (may be a list for trtllm)
    latest_versions = LATEST_VERSIONS.get(backend, [])

    # Check each latest version (in order, checking newer versions first)
    for latest_version in latest_versions:
        if latest_version in version_has_pass:
            # Latest version passes
            return (latest_version, True, None)
        elif latest_version in version_has_fail:
            # Latest version fails - return FAIL with error message
            error_msg = version_error_msgs.get(latest_version, "No error message available")
            return ("FAIL", False, error_msg)

    # None of the hard-coded latest versions exist in data - find the latest passing version
    passing_versions = sorted(version_has_pass.keys(), key=parse_version, reverse=True)
    if passing_versions:
        # Return the latest passing version (not the hard-coded latest)
        return (passing_versions[0], False, None)
    else:
        # No passing version exists - collect error messages from all failed versions
        all_error_msgs = []
        for version in sorted(version_has_fail.keys(), key=parse_version, reverse=True):
            if version in version_error_msgs:
                all_error_msgs.append(f"{version}: {version_error_msgs[version]}")
        error_msg = " | ".join(all_error_msgs) if all_error_msgs else "No error message available"
        return ("FAIL", False, error_msg)


def create_system_matrix(df, system_name, mode_filter="all"):
    """
    Create a 2D matrix for a specific system showing the latest supported
    backend version for each (HuggingFaceID, Backend) combination.

    Args:
        df: DataFrame with support matrix data
        system_name: Name of the system to create matrix for
        mode_filter: Filter by mode ('agg', 'disagg', or 'all')

    Returns:
        Tuple of (DataFrame for display, error messages dict, is_latest dict)
    """
    # Filter by system and mode
    system_df = df[df["System"] == system_name].copy()

    if mode_filter != "all":
        system_df = system_df[system_df["Mode"] == mode_filter]

    if system_df.empty:
        return pd.DataFrame(), {}, {}

    # Get unique HuggingFace IDs and Backends
    huggingface_ids = sorted(system_df["HuggingFaceID"].unique())
    backends = sorted(system_df["Backend"].unique())

    # Build the matrix
    matrix_data = []
    matrix_is_latest = {}  # Track if each cell is the latest version
    matrix_error_msgs = {}  # Track error messages for FAIL cells - keyed by (row_idx, col_idx)
    for row_idx, hf_id in enumerate(huggingface_ids):
        row = [hf_id]  # First column is HuggingFace ID
        for col_idx, backend in enumerate(backends):
            latest_version, is_latest, error_msg = get_latest_supported_version(system_df, hf_id, system_name, backend)
            if latest_version is None:
                row.append("FAIL")
                matrix_is_latest[(row_idx, col_idx)] = False
                matrix_error_msgs[(row_idx, col_idx)] = "No data available"
            else:
                row.append(latest_version)
                matrix_is_latest[(row_idx, col_idx)] = is_latest
                if latest_version == "FAIL":
                    matrix_error_msgs[(row_idx, col_idx)] = error_msg
                else:
                    matrix_error_msgs[(row_idx, col_idx)] = None
        matrix_data.append(row)

    # Create DataFrame with HuggingFace ID as first column, then backends
    columns = ["HuggingFace ID"] + backends
    matrix_df = pd.DataFrame(matrix_data, columns=columns)

    # Apply styling to cells based on their values using pandas Styler
    def apply_row_styling(row):
        """Apply styling to each row."""
        styles = [""] * len(row)  # First column (HuggingFace ID) has no styling
        for col_idx in range(1, len(row)):
            cell_value = row.iloc[col_idx]
            row_idx = row.name
            if cell_value == "FAIL":
                styles[col_idx] = "background-color: #ffcccc; color: #cc0000; font-weight: bold;"
            elif (row_idx, col_idx - 1) in matrix_is_latest:
                is_latest = matrix_is_latest.get((row_idx, col_idx - 1), True)
                if not is_latest:
                    styles[col_idx] = "background-color: #d4ed9f;"
                else:
                    styles[col_idx] = "background-color: #ccffcc;"
        return styles

    # Apply styling using pandas Styler
    styled_df = matrix_df.style.apply(apply_row_styling, axis=1)

    return styled_df, matrix_error_msgs, matrix_is_latest


def create_system_heatmap(df, system_name, mode_filter="all"):
    """
    Create a heatmap visualization for a specific system.

    Args:
        df: DataFrame with support matrix data
        system_name: Name of the system
        mode_filter: Filter by mode ('agg', 'disagg', or 'all')

    Returns:
        HTML string with the plotly figure
    """
    # Filter by system and mode
    system_df = df[df["System"] == system_name].copy()

    if mode_filter != "all":
        system_df = system_df[system_df["Mode"] == mode_filter]

    if system_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this system",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(title=f"Support Matrix for {system_name}", height=400, width=800)
        return fig.to_html(include_plotlyjs="cdn")

    # Get unique HuggingFace IDs and Backends
    huggingface_ids = sorted(system_df["HuggingFaceID"].unique())
    backends = sorted(system_df["Backend"].unique())

    # Build the matrix data
    matrix_data = []
    matrix_text = []
    matrix_hover = []  # Store hover text with error messages
    for hf_id in huggingface_ids:
        row = []
        text_row = []
        hover_row = []
        for backend in backends:
            latest_version, is_latest, error_msg = get_latest_supported_version(system_df, hf_id, system_name, backend)
            if latest_version is None or latest_version == "FAIL":
                row.append(0)  # FAIL = 0 (red)
                text_row.append("FAIL")
                # Format error message for hover
                if error_msg:
                    error_display = error_msg.replace("\\n", "<br>")
                    hover_row.append(
                        f"<b>Model:</b> {hf_id}<br><b>Backend:</b> {backend}<br><b>Status:</b> FAIL<br><b>Error:</b> {error_display}"
                    )
                else:
                    hover_row.append(f"<b>Model:</b> {hf_id}<br><b>Backend:</b> {backend}<br><b>Status:</b> FAIL")
            else:
                # Use 1 for latest version (green), 0.5 for supported but not latest (yellow-green)
                row.append(1 if is_latest else 0.5)
                text_row.append(latest_version)
                hover_row.append(
                    f"<b>Model:</b> {hf_id}<br><b>Backend:</b> {backend}<br><b>Version:</b> {latest_version}"
                )
        matrix_data.append(row)
        matrix_text.append(text_row)
        matrix_hover.append(hover_row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_data,
            x=backends,
            y=huggingface_ids,
            colorscale=[[0, "red"], [0.5, "#d4ed9f"], [1, "green"]],
            colorbar=dict(title="Status", tickvals=[0, 0.5, 1], ticktext=["FAIL", "Supported (not latest)", "Latest"]),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=matrix_hover,
            text=matrix_text,
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(
        title=f"Support Matrix for {system_name} ({mode_filter if mode_filter != 'all' else 'All Modes'})",
        xaxis_title="Backend",
        yaxis_title="HuggingFace Model ID",
        height=max(600, len(huggingface_ids) * 20),
        width=max(800, len(backends) * 100),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=8)),
    )

    return fig.to_html(include_plotlyjs="cdn")


def load_support_matrix_data():
    """Load and return the support matrix as a DataFrame."""
    matrix_data = common.get_support_matrix()
    df = pd.DataFrame(matrix_data)
    return df


def create_support_matrix_tab(app_config):
    """Create the support matrix visualization tab."""
    with gr.Tab("Support Matrix"):
        with gr.Accordion("Introduction", open=False):
            introduction = gr.Markdown(
                label="introduction",
                value=r"""
                This tab visualizes the aiconfigurator support matrix, showing the latest supported backend version
                for each (HuggingFace Model ID, System, Backend) combination.
                
                **Features:**
                - One subtab per system
                - 2D matrix view: rows are HuggingFace IDs, columns are backends
                - Cell values show the latest backend version that is supported (PASS)
                - Red cells with "FAIL" indicate no supported version exists
                - Filter by mode (agg/disagg/all) to see mode-specific support
                
                **Usage:**
                1. Select a system from the tabs below
                2. Use the mode filter to view aggregated, disaggregated, or all modes
                3. Green cells show the latest supported version
                4. Red cells indicate no support for that combination
                """,
            )

        # Load data
        initial_df = load_support_matrix_data()
        unique_systems = sorted(initial_df["System"].unique())

        # Mode filter
        mode_filter = gr.Dropdown(
            choices=["all", "agg", "disagg"],
            value="all",
            label="Mode Filter",
            interactive=True,
        )

        # Create subtabs for each system
        with gr.Tabs() as system_tabs:
            system_components = {}

            for system_name in unique_systems:
                with gr.Tab(system_name):
                    # Create matrix for this system
                    initial_matrix_df, initial_error_msgs, initial_is_latest = create_system_matrix(
                        initial_df, system_name, "all"
                    )
                    initial_heatmap = create_system_heatmap(initial_df, system_name, "all")

                    # Matrix table view using Gradio Dataframe
                    matrix_dataframe = gr.Dataframe(
                        value=initial_matrix_df,
                        label="Support Matrix",
                        interactive=False,
                        wrap=True,
                        max_height="100vh",  # Use viewport height to remove scrollbar
                        elem_classes=["support-matrix-table"],
                    )

                    # Function to show error when a cell is clicked
                    def make_show_error(system_name):
                        """Create a closure to capture system_name and access stored data."""

                        def show_error(evt: gr.SelectData):
                            """Show error message when a FAIL cell is clicked."""
                            if not evt or not hasattr(evt, "index"):
                                return

                            # Get current error messages and matrix from stored components
                            if system_name not in system_components:
                                return

                            error_msgs_dict = system_components[system_name]["error_msgs"]
                            matrix_df = system_components[system_name]["matrix_df"]

                            # evt.index is a tuple (row, col) for Dataframe
                            if isinstance(evt.index, (list, tuple)) and len(evt.index) == 2:
                                row_idx, col_idx = evt.index
                                # col_idx 0 is "HuggingFace ID", so backend columns start at 1
                                if col_idx > 0:
                                    error_msg = error_msgs_dict.get((row_idx, col_idx - 1))
                                    if error_msg and str(error_msg).strip() and str(error_msg).strip() != "None":
                                        # Get model from row value (first column)
                                        if hasattr(evt, "row_value") and evt.row_value and len(evt.row_value) > 0:
                                            model = str(evt.row_value[0])
                                        else:
                                            model = ""

                                        # Get backend from column name
                                        if matrix_df is not None and col_idx < len(matrix_df.columns):
                                            backend = matrix_df.columns[col_idx]
                                        else:
                                            backend = ""

                                        # Format error message - convert escaped newlines to actual newlines
                                        formatted_msg = str(error_msg)
                                        # Replace double-escaped newlines first, then single-escaped
                                        formatted_msg = formatted_msg.replace("\\\\n", "\n").replace("\\n", "\n")

                                        title = f"Error Details - {model} / {backend}"
                                        # Use gr.Info() to show the error - format as markdown code block for better readability
                                        # Markdown code blocks preserve newlines and formatting
                                        gr.Info(f"**{title}**\n\n```\n{formatted_msg}\n```")

                        return show_error

                    # Connect select event on Dataframe
                    matrix_dataframe.select(
                        fn=make_show_error(system_name),
                        inputs=None,
                        outputs=None,
                    )

                    # Heatmap view
                    heatmap_html = gr.HTML(value=initial_heatmap, label="Heatmap View")

                    # Store components and data
                    system_components[system_name] = {
                        "matrix_dataframe": matrix_dataframe,
                        "heatmap_html": heatmap_html,
                        "error_msgs": initial_error_msgs,
                        "is_latest": initial_is_latest,
                        "matrix_df": initial_matrix_df,
                    }

        # Connect mode filter to all system tabs
        def update_mode_filter(mode):
            """Update all system visualizations when mode changes."""
            updates = []
            for system_name in unique_systems:
                matrix_df, error_msgs, is_latest = create_system_matrix(initial_df, system_name, mode)
                heatmap = create_system_heatmap(initial_df, system_name, mode)
                # Update stored error messages, is_latest, and matrix_df
                system_components[system_name]["error_msgs"] = error_msgs
                system_components[system_name]["is_latest"] = is_latest
                system_components[system_name]["matrix_df"] = matrix_df
                updates.append(matrix_df)
                updates.append(heatmap)
            return updates

        # Get all output components in order
        all_outputs = []
        for system_name in unique_systems:
            all_outputs.append(system_components[system_name]["matrix_dataframe"])
            all_outputs.append(system_components[system_name]["heatmap_html"])

        # Connect mode filter
        mode_filter.change(
            fn=update_mode_filter,
            inputs=[mode_filter],
            outputs=all_outputs,
        )

    return {
        "introduction": introduction,
        "mode_filter": mode_filter,
    }
