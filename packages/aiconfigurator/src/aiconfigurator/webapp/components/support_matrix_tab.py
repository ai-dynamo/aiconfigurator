# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import html as html_module
import re
import shlex

import gradio as gr
import pandas as pd
from packaging.version import Version

from aiconfigurator.sdk import common


def parse_version(ver_str):
    """
    Parse version string into comparable Version (PEP 440).
    Invalid or empty strings return Version("0.0.0") so they sort first.
    """
    return common.parse_support_matrix_version(ver_str) or Version("0.0.0")


# Cell colors for support matrix (used in styling and legend)
COLOR_FAIL_BG = "#ffcccc"
COLOR_FAIL_TEXT = "#cc0000"
COLOR_HW_INCOMPATIBLE_BG = "#e5e7eb"
COLOR_HW_INCOMPATIBLE_TEXT = "#475569"
COLOR_LATEST_PASS = "#80ff80"  # Green: latest tested backend version passes
COLOR_OLDER_PASS = "#ccffcc"  # Light green: older tested backend version passes

SUPPORT_MATRIX_LEGEND = (
    f'<div style="margin-bottom: 10px; font-size: 0.95em;">'
    f"<strong>Legend:</strong>"
    f'<span style="margin-left: 8px; margin-right: 24px;">'
    f'<span style="background: {COLOR_LATEST_PASS}; padding: 2px 10px; border-radius: 4px; margin-right: 8px;">Green</span>'
    f"latest tested version passes (click cell for command)</span>"
    f'<span style="margin-right: 24px;">'
    f'<span style="background: {COLOR_OLDER_PASS}; padding: 2px 10px; border-radius: 4px; margin-right: 8px;">Light green</span>'
    f"older tested version passes (click cell for command)</span>"
    f'<span style="margin-right: 24px;">'
    f'<span style="background: {COLOR_FAIL_BG}; color: {COLOR_FAIL_TEXT}; padding: 2px 10px; border-radius: 4px; font-weight: bold;">FAIL</span>'
    f" test failed (click cell to see command and details)</span>"
    f'<span style="margin-right: 24px;">'
    f'<span style="background: {COLOR_HW_INCOMPATIBLE_BG}; color: {COLOR_HW_INCOMPATIBLE_TEXT}; padding: 2px 10px; border-radius: 4px; font-weight: bold;">HW</span>'
    f" GPU does not support model datatype (click cell for command)</span>"
    f"</div>"
)


def _clean_cell_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _support_matrix_cli_constraints(model: str) -> dict[str, str]:
    lower_model = model.lower()
    large_model_markers = (
        "deepseek",
        "glm-5",
        "kimi",
        "llama-4-maverick",
        "llama-4-scout",
        "minimax",
        "mimo-v2",
        "nemotron-ultra",
    )
    sizes = [float(match) for match in re.findall(r"(\d+(?:\.\d+)?)b", lower_model)]
    max_size = max(sizes, default=None)
    if max_size is not None and max_size < 10:
        return {"total_gpus": "4", "ttft": "1500.0", "tpot": "50.0"}
    if max_size is not None and max_size < 100 and not any(marker in lower_model for marker in large_model_markers):
        return {"total_gpus": "32", "ttft": "2000.0", "tpot": "50.0"}
    return {"total_gpus": "128", "ttft": "2000000.0", "tpot": "50000.0"}


def _support_matrix_command(model: str, system: str, backend: str, version: str, mode: str) -> str:
    constraints = _support_matrix_cli_constraints(model)
    parts = [
        "uv",
        "run",
        "aiconfigurator",
        "cli",
        "default",
        "--model-path",
        model,
        "--total-gpus",
        constraints["total_gpus"],
        "--system",
        system,
        "--backend",
        backend,
        "--backend-version",
        version,
        "--database-mode",
        "SILICON",
        "--isl",
        "256",
        "--osl",
        "256",
        "--prefix",
        "128",
        "--ttft",
        constraints["ttft"],
        "--tpot",
        constraints["tpot"],
        "--top-n",
        "1",
        "--no-color",
    ]
    _ = mode
    return " ".join(shlex.quote(str(part)) for part in parts)


def _fallback_support_matrix_command(row) -> str:
    return _support_matrix_command(
        str(row["HuggingFaceID"]),
        str(row["System"]),
        str(row["Backend"]),
        str(row["Version"]),
        str(row["Mode"]),
    )


def _format_row_details(row) -> str:
    command = _clean_cell_value(row["Command"]) if "Command" in row.index else None
    if (
        command is None
        or command.startswith("aiconfigurator cli default")
        or "tools/support_matrix/generate_support_matrix.py" in command
    ):
        command = _fallback_support_matrix_command(row)
    error_msg = _clean_cell_value(row["ErrMsg"]) if "ErrMsg" in row.index else None

    lines = [
        f"Mode: {row['Mode']}",
        f"Status: {row['Status']}",
        f"Version: {row['Version']}",
        "",
        "Command:",
        command,
    ]
    if error_msg:
        lines.extend(["", "Details:", error_msg])
    return "\n".join(lines)


def _format_missing_row_details(
    *,
    model: str,
    system: str,
    backend: str,
    version: str,
    mode: str,
) -> str:
    command = _support_matrix_command(model, system, backend, version, mode)
    return "\n".join(
        [
            f"Mode: {mode}",
            "Status: FAIL",
            f"Version: {version}",
            "",
            "Command:",
            command,
            "",
            "Details:",
            "No support-matrix row exists for this model/system/backend cell. "
            "Run the command above to check or collect the latest backend version.",
        ]
    )


def get_latest_supported_version(df, huggingface_id, system, backend):
    """
    Get the latest version status for a given HuggingFace ID and system combination.
    Uses the latest backend version present in the filtered matrix data. If that
    latest version fails for the model/system/backend combination, returns "FAIL".

    Returns:
        tuple: (version, is_latest, detail_msg) where:
            - version: Latest version string if latest version passes, "FAIL" if latest version fails,
                       None if no data exists
            - is_latest: True if the returned version is the latest tested version (green),
                         False if an older passing version is shown (light green). Ignored when version is "FAIL".
            - detail_msg: Command plus error details for the displayed cell
    """
    # Filter for this specific combination (both PASS and FAIL)
    subset = df[(df["HuggingFaceID"] == huggingface_id) & (df["System"] == system) & (df["Backend"] == backend)]

    if subset.empty:
        return (None, False, None)

    # Get all versions with their statuses and error messages
    # For each version, track PASS/FAIL/HW_INCOMPATIBLE entries and collect messages
    version_has_fail = {}
    version_has_pass = {}
    version_has_hw_incompatible = {}
    version_error_msgs = {}  # Store error messages for failed versions
    version_details = {}

    for _, row in subset.iterrows():
        version = row["Version"]
        status = row["Status"]
        detail_msg = _format_row_details(row)
        if version in version_details:
            existing = version_details[version]
            if detail_msg and detail_msg not in existing:
                version_details[version] = f"{existing}\n\n---\n\n{detail_msg}"
        else:
            version_details[version] = detail_msg
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
        elif status == "HW_INCOMPATIBLE":
            version_has_hw_incompatible[version] = True
            if error_msg:
                if version in version_error_msgs:
                    existing = version_error_msgs[version]
                    if existing and existing != error_msg:
                        version_error_msgs[version] = f"{existing} | {error_msg}"
                else:
                    version_error_msgs[version] = error_msg

    if len(version_has_fail) == 0 and len(version_has_pass) == 0 and len(version_has_hw_incompatible) == 0:
        return (None, False, None)

    backend_versions = df[(df["System"] == system) & (df["Backend"] == backend)]["Version"].dropna().unique()
    latest_version = max((str(version) for version in backend_versions), key=parse_version, default=None)
    if latest_version in version_has_pass:
        return (latest_version, True, version_details.get(latest_version))
    if latest_version in version_has_hw_incompatible and latest_version not in version_has_fail:
        error_msg = version_error_msgs.get(latest_version, "Hardware is incompatible with model datatype")
        return ("HW_INCOMPATIBLE", False, version_details.get(latest_version, error_msg))
    if latest_version in version_has_fail:
        error_msg = version_error_msgs.get(latest_version, "No error message available")
        return ("FAIL", False, version_details.get(latest_version, error_msg))

    # Latest backend version is not tested for this model; show the newest passing version.
    passing_versions = sorted(version_has_pass.keys(), key=parse_version, reverse=True)
    if passing_versions:
        v = passing_versions[0]
        return (v, False, version_details.get(v))
    # No passing version exists - prefer real failures over hardware-incompatible rows.
    all_error_msgs = []
    all_failure_details = []
    for version in sorted(version_has_fail.keys(), key=parse_version, reverse=True):
        if version in version_error_msgs:
            all_error_msgs.append(f"{version}: {version_error_msgs[version]}")
        if version in version_details:
            all_failure_details.append(version_details[version])
    if all_error_msgs or version_has_fail:
        error_msg = " | ".join(all_error_msgs) if all_error_msgs else "No error message available"
        detail_msg = "\n\n---\n\n".join(all_failure_details) if all_failure_details else error_msg
        return ("FAIL", False, detail_msg)

    hw_error_msgs = []
    hw_details = []
    for version in sorted(version_has_hw_incompatible.keys(), key=parse_version, reverse=True):
        if version in version_error_msgs:
            error_msg = version_error_msgs[version]
            if error_msg not in hw_error_msgs:
                hw_error_msgs.append(error_msg)
        if version in version_details:
            hw_details.append(version_details[version])
    error_msg = " | ".join(hw_error_msgs) if hw_error_msgs else "Hardware is incompatible with model datatype"
    detail_msg = "\n\n---\n\n".join(hw_details) if hw_details else error_msg
    return ("HW_INCOMPATIBLE", False, detail_msg)


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
    matrix_error_msgs = {}  # Track command/details messages keyed by (row_idx, col_idx)
    for row_idx, hf_id in enumerate(huggingface_ids):
        row = [hf_id]  # First column is HuggingFace ID
        for col_idx, backend in enumerate(backends):
            latest_version, is_latest, detail_msg = get_latest_supported_version(system_df, hf_id, system_name, backend)
            if latest_version is None:
                backend_versions = system_df[system_df["Backend"] == backend]["Version"].dropna().unique()
                latest_backend_version = max(
                    (str(version) for version in backend_versions),
                    key=parse_version,
                    default="unknown",
                )
                command_mode = mode_filter if mode_filter in {"agg", "disagg"} else "all"
                row.append("FAIL")
                matrix_is_latest[(row_idx, col_idx)] = False
                matrix_error_msgs[(row_idx, col_idx)] = _format_missing_row_details(
                    model=hf_id,
                    system=system_name,
                    backend=backend,
                    version=latest_backend_version,
                    mode=command_mode,
                )
            else:
                row.append(latest_version)
                matrix_is_latest[(row_idx, col_idx)] = is_latest
                if latest_version in ("FAIL", "HW_INCOMPATIBLE"):
                    matrix_error_msgs[(row_idx, col_idx)] = detail_msg
                else:
                    matrix_error_msgs[(row_idx, col_idx)] = detail_msg
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
            elif cell_value == "HW_INCOMPATIBLE":
                styles[col_idx] = (
                    f"background-color: {COLOR_HW_INCOMPATIBLE_BG}; "
                    f"color: {COLOR_HW_INCOMPATIBLE_TEXT}; font-weight: bold;"
                )
            elif (row_idx, col_idx - 1) in matrix_is_latest:
                is_latest = matrix_is_latest.get((row_idx, col_idx - 1), True)
                if not is_latest:
                    styles[col_idx] = f"background-color: {COLOR_OLDER_PASS};"
                else:
                    styles[col_idx] = f"background-color: {COLOR_LATEST_PASS};"
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
        unique_systems = common.sort_support_matrix_systems(initial_df["System"].unique())

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

                    # Function to show support details when a cell is clicked
                    def make_show_error(system_name):
                        """Create a closure to capture system_name and access stored data."""

                        def show_error(evt: gr.SelectData):
                            """Show the support-matrix command and status details when a cell is clicked."""
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
                                    detail_msg = error_msgs_dict.get((row_idx, col_idx - 1))
                                    if detail_msg and str(detail_msg).strip() and str(detail_msg).strip() != "None":
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

                                        cell_value = ""
                                        if hasattr(evt, "row_value") and evt.row_value and len(evt.row_value) > col_idx:
                                            cell_value = str(evt.row_value[col_idx])

                                        # Format details - convert escaped newlines to actual newlines
                                        formatted_msg = str(detail_msg)
                                        # Replace double-escaped newlines first, then single-escaped
                                        formatted_msg = formatted_msg.replace("\\\\n", "\n").replace("\\n", "\n")

                                        title_prefix = (
                                            "Hardware Incompatibility"
                                            if cell_value == "HW_INCOMPATIBLE"
                                            else "Error Details"
                                            if cell_value == "FAIL"
                                            else "Support Details"
                                        )
                                        title = f"{title_prefix} - {model} / {backend}"
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
