# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import html as html_module
import re

import gradio as gr
import pandas as pd
from packaging.version import InvalidVersion, Version

from aiconfigurator.sdk import common


def parse_version(ver_str):
    """
    Parse version string into comparable Version (PEP 440).
    Invalid or empty strings return Version("0.0.0") so they sort first.
    """
    if not ver_str:
        return Version("0.0.0")
    try:
        return Version(ver_str)
    except InvalidVersion:
        return Version("0.0.0")


# Cell colors for support matrix (used in styling and legend)
COLOR_FAIL_BG = "#ffcccc"
COLOR_FAIL_TEXT = "#cc0000"
COLOR_AT_OR_ABOVE_MIN = "#80ff80"  # Green: version at or above target
COLOR_BELOW_MIN = "#ccffcc"  # Light green: version below target

SUPPORT_MATRIX_LEGEND = (
    f'<div style="margin-bottom: 10px; font-size: 0.95em;">'
    f"<strong>Legend:</strong>"
    f'<span style="margin-left: 8px; margin-right: 24px;">'
    f'<span style="background: {COLOR_AT_OR_ABOVE_MIN}; padding: 2px 10px; border-radius: 4px; margin-right: 8px;">Green</span>'
    f"at or above target version</span>"
    f'<span style="margin-right: 24px;">'
    f'<span style="background: {COLOR_BELOW_MIN}; padding: 2px 10px; border-radius: 4px; margin-right: 8px;">Light green</span>'
    f"below target version</span>"
    f'<span style="margin-right: 24px;">'
    f'<span style="background: {COLOR_FAIL_BG}; color: {COLOR_FAIL_TEXT}; padding: 2px 10px; border-radius: 4px; font-weight: bold;">FAIL</span>'
    f" test failed (click cell to see error message)</span>"
    f"</div>"
)

# Hard-coded target version per backend (used for "target" check and as minimum for green)
TARGET_VERSIONS = {
    "vllm": "0.14.0",
    "sglang": "0.5.9",
    "trtllm": "1.2.0rc6",
}


def get_latest_supported_version(df, huggingface_id, system, backend):
    """
    Get the latest version status for a given HuggingFace ID and system combination.
    Uses hard-coded latest versions. If the latest version fails, returns "FAIL".

    Returns:
        tuple: (version, is_latest, error_msg) where:
            - version: Latest version string if latest version passes, "FAIL" if latest version fails,
                       None if no data exists
            - is_latest: True if the returned version is at or above the hard-coded minimum (green),
                         False if below (light green). Ignored when version is "FAIL".
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

    min_ver = TARGET_VERSIONS.get(backend)

    def at_or_above_min(ver_str):
        if min_ver is None:
            return True
        try:
            return parse_version(ver_str) >= parse_version(min_ver)
        except Exception:
            return False

    latest_version = min_ver
    if latest_version is not None:
        if latest_version in version_has_pass:
            return (latest_version, at_or_above_min(latest_version), None)
        if latest_version in version_has_fail:
            error_msg = version_error_msgs.get(latest_version, "No error message available")
            return ("FAIL", False, error_msg)

    # Hard-coded latest not in data or no min set - find the latest passing version
    passing_versions = sorted(version_has_pass.keys(), key=parse_version, reverse=True)
    if passing_versions:
        v = passing_versions[0]
        return (v, at_or_above_min(v), None)
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
                styles[col_idx] = f"background-color: {COLOR_FAIL_BG}; color: {COLOR_FAIL_TEXT}; font-weight: bold;"
            elif (row_idx, col_idx - 1) in matrix_is_latest:
                is_latest = matrix_is_latest.get((row_idx, col_idx - 1), True)
                if not is_latest:
                    styles[col_idx] = f"background-color: {COLOR_BELOW_MIN};"
                else:
                    styles[col_idx] = f"background-color: {COLOR_AT_OR_ABOVE_MIN};"
        return styles

    # Apply styling using pandas Styler
    styled_df = matrix_df.style.apply(apply_row_styling, axis=1)

    return styled_df, matrix_error_msgs, matrix_is_latest


def _extract_error_signature(err_msg):
    """
    Extract a short error signature from a full traceback/ErrMsg for grouping.
    Prefers the last exception line (e.g. "RuntimeError: message"), then falls back to first line or truncated.
    """
    if not err_msg or pd.isna(err_msg):
        return "No error message"
    text = str(err_msg).strip()
    if not text:
        return "No error message"
    lines = [ln.strip() for ln in text.replace("\\n", "\n").split("\n") if ln.strip()]
    # Find lines that look like "ExceptionType: message" (Python exception format)
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if re.match(r"^[A-Za-z][A-Za-z0-9_.]*Error[^:]*:", line) or re.match(
            r"^[A-Za-z][A-Za-z0-9_.]*Exception[^:]*:", line
        ):
            return line[:500]  # Cap length for display
    if lines:
        return lines[-1][:500] if lines else "No error message"
    return text[:500] if len(text) > 500 else text


def get_top_errors_for_system(df, system_name, mode_filter="all", top_n=10):
    """
    Return the top N most common error signatures for a system (FAIL rows only).

    Args:
        df: Full support matrix DataFrame
        system_name: System to filter by
        mode_filter: 'all', 'agg', or 'disagg'
        top_n: Maximum number of errors to return

    Returns:
        List of (error_signature, count) sorted by count descending.
    """
    subset = df[(df["System"] == system_name) & (df["Status"] == "FAIL")].copy()
    if mode_filter != "all":
        subset = subset[subset["Mode"] == mode_filter]
    if subset.empty:
        return []
    if "ErrMsg" not in subset.columns:
        return []
    subset["_signature"] = subset["ErrMsg"].apply(_extract_error_signature)
    counts = subset["_signature"].value_counts()
    return list(counts.head(top_n).items())


def _format_top_errors_markdown(top_errors):
    """Format top errors as Markdown for display."""
    if not top_errors:
        return "_No failures recorded for this system with the current filter._"
    lines = ["#### Most common errors (top 10)\n", "| # | Count | Error |", "|---|-------|-------|"]
    for i, (sig, count) in enumerate(top_errors, 1):
        # Escape pipe and newline for table cells
        cell = sig.replace("|", "\\|").replace("\n", " ")
        if len(cell) > 200:
            cell = cell[:197] + "..."
        lines.append(f"| {i} | {count} | {cell} |")
    return "\n".join(lines)


def load_support_matrix_data():
    """Load and return the support matrix as a DataFrame."""
    matrix_data = common.get_support_matrix()
    df = pd.DataFrame(matrix_data)
    return df


def create_support_matrix_tab(app_config):
    """Create the support matrix visualization tab."""
    with gr.Tab("Support Matrix"):
        with gr.Accordion("Introduction", open=True):
            introduction = gr.Markdown(
                label="introduction",
                value=r"""
                This tab visualizes the aiconfigurator support matrix, showing the latest supported backend version
                for each (HuggingFace Model ID, System, Backend) combination.
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
        with gr.Tabs():
            system_components = {}

            for system_name in unique_systems:
                with gr.Tab(system_name):
                    # Create matrix for this system
                    initial_matrix_df, initial_error_msgs, initial_is_latest = create_system_matrix(
                        initial_df, system_name, "all"
                    )

                    # Legend at top of table
                    gr.HTML(SUPPORT_MATRIX_LEGEND, elem_classes=["support-matrix-legend"])

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
                                        # gr.Info renders HTML. Use scrollable container so long messages stay on screen.
                                        escaped_msg = html_module.escape(formatted_msg)
                                        html_message = (
                                            f"<b>{html_module.escape(title)}</b><br><br>"
                                            f'<div style="max-height: 85vh; overflow: auto; border: 1px solid #ccc; border-radius: 4px; padding: 8px;">'
                                            f'<pre style="white-space: pre-wrap; margin: 0;">{escaped_msg}</pre>'
                                            f"</div>"
                                        )
                                        gr.Info(html_message, duration=60)

                        return show_error

                    # Connect select event on Dataframe
                    matrix_dataframe.select(
                        fn=make_show_error(system_name),
                        inputs=None,
                        outputs=None,
                    )

                    # Top 10 most common errors for this system (below the table)
                    initial_top_errors = get_top_errors_for_system(initial_df, system_name, "all", top_n=10)
                    initial_errors_md = _format_top_errors_markdown(initial_top_errors)
                    top_errors_markdown = gr.Markdown(
                        value=initial_errors_md,
                        label="Most common errors for this system",
                        elem_classes=["support-matrix-top-errors"],
                    )

                    # Store components and data
                    system_components[system_name] = {
                        "matrix_dataframe": matrix_dataframe,
                        "top_errors_markdown": top_errors_markdown,
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
                top_errors = get_top_errors_for_system(initial_df, system_name, mode, top_n=10)
                errors_md = _format_top_errors_markdown(top_errors)
                # Update stored error messages, is_latest, and matrix_df
                system_components[system_name]["error_msgs"] = error_msgs
                system_components[system_name]["is_latest"] = is_latest
                system_components[system_name]["matrix_df"] = matrix_df
                updates.append(matrix_df)
                updates.append(errors_md)
            return updates

        # Get all output components in order
        all_outputs = []
        for system_name in unique_systems:
            all_outputs.append(system_components[system_name]["matrix_dataframe"])
            all_outputs.append(system_components[system_name]["top_errors_markdown"])

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
