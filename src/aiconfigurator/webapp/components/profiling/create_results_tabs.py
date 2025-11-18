"""
UI components for results display in the Dynamo SLA Profiler webapp.

This module provides functions to build the results tabs with plots and tables.
"""

from numbers import Real

import gradio as gr

from aiconfigurator.webapp.components.profiling.constants import (
    COST_TAB_DESCRIPTION,
    COST_TABLE_HEADERS,
    DECODE_TAB_DESCRIPTION,
    DECODE_TABLE_HEADERS,
    PREFILL_TAB_DESCRIPTION,
    PREFILL_TABLE_HEADERS,
)


def get_empty_tables():
    """Get empty table HTML for all three table types."""
    return (
        build_table_html(PREFILL_TABLE_HEADERS, []),
        build_table_html(DECODE_TABLE_HEADERS, []),
        build_table_html(COST_TABLE_HEADERS, []),
    )


def _format_cell(value, is_config=False):
    """Format a cell value for display in HTML table."""
    if is_config:
        # For config cells, return a button with the config stored in a data attribute
        import html

        config_escaped = html.escape(value)
        return (
            f'<button class="config-btn" data-config="{config_escaped}" onclick="showConfig(this)">Show Config</button>'
        )
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, Real):
        if isinstance(value, int):
            return f"{value}"
        return f"{value:.3f}"
    return str(value)


def build_table_html(headers, rows):
    """
    Build an HTML table from headers and rows.

    Args:
        headers: List of header strings
        rows: List of row data (each row is a list of values)

    Returns:
        HTML string containing the table
    """
    header_html = "".join(f"<th>{header}</th>" for header in headers)

    if not rows:
        empty_row = (
            f"<tr><td class='dynamo-table-empty' colspan='{len(headers)}'>"
            "No data selected yet. Click points on the plot to populate this table."
            "</td></tr>"
        )
        body_html = empty_row
    else:
        rows_html = []
        for row in rows:
            cells = []
            for idx, cell in enumerate(row):
                # Check if this is the last column (Config column)
                is_config = idx == len(row) - 1 and headers[idx] == "Config"
                cells.append(f"<td>{_format_cell(cell, is_config=is_config)}</td>")
            rows_html.append("<tr>" + "".join(cells) + "</tr>")
        body_html = "".join(rows_html)

    return (
        "<div class='dynamo-table-wrapper'>"
        "<table class='dynamo-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )


def create_results_tabs(empty_prefill_html, empty_decode_html, empty_cost_html):
    """
    Create the results tabs with plots and tables.

    Args:
        empty_prefill_html: Empty prefill table HTML
        empty_decode_html: Empty decode table HTML
        empty_cost_html: Empty cost table HTML

    Returns:
        Dictionary of Gradio components
    """
    with gr.Tab("Prefill Performance"):
        prefill_plot = gr.Plot(
            label="Prefill Performance",
            show_label=False,
            elem_id="prefill_plot",
        )
        gr.Markdown(PREFILL_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        prefill_table = gr.HTML(
            value=empty_prefill_html,
            elem_id="prefill_table",
        )

    with gr.Tab("Decode Performance"):
        decode_plot = gr.Plot(
            label="Decode Performance",
            show_label=False,
            elem_id="decode_plot",
        )
        gr.Markdown(DECODE_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        decode_table = gr.HTML(
            value=empty_decode_html,
            elem_id="decode_table",
        )

    with gr.Tab("Cost vs SLA"):
        cost_plot = gr.Plot(
            label="Cost vs SLA",
            show_label=False,
            elem_id="cost_plot",
        )
        gr.Markdown(COST_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        cost_table = gr.HTML(
            value=empty_cost_html,
            elem_id="cost_table",
        )

    return {
        "prefill_plot": prefill_plot,
        "decode_plot": decode_plot,
        "cost_plot": cost_plot,
        "prefill_table": prefill_table,
        "decode_table": decode_table,
        "cost_table": cost_table,
    }
