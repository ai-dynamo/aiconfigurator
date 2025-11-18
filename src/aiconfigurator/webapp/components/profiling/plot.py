"""
Interactive plotting functions for Gradio webapp using Plotly.

This module provides interactive versions of the profiler plots using Plotly,
which integrates seamlessly with Gradio's gr.Plot component.
"""

import numpy as np
import plotly.graph_objects as go

from aiconfigurator.webapp.components.profiling.constants import PLOTLY_COLORS


def _add_target_line(fig, target_value, label, max_y):
    """
    Add a target reference line to a plot.

    Args:
        fig: Plotly Figure object
        target_value: X-coordinate of the vertical line
        label: Label for the target line
        max_y: Maximum Y value for the line
    """
    fig.add_trace(
        go.Scatter(
            x=[target_value, target_value],
            y=[0, max_y * 1.1],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=label,
            hovertemplate=f"{label}<extra></extra>",
        )
    )


def _compute_parato(x, y):
    """
    compute the pareto front (top-left is better) for the given x and y values
    return sorted lists of the x and y values for the pareto front
    """
    # Validate inputs
    if x is None or y is None:
        return [], []

    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    if len(x) == 0:
        return [], []

    # Build point list and sort by x asc, then y desc so we prefer smaller x and larger y.
    points = list(zip(x, y))
    points.sort(key=lambda p: (p[0], -p[1]))

    # Single pass to keep only non-dominated points (minimize x, maximize y).
    pareto = []
    max_y = float("-inf")
    for px, py in points:
        if py > max_y:
            pareto.append((px, py))
            max_y = py

    # Return sorted by x ascending for convenience
    pareto.sort(key=lambda p: (p[0], p[1]))
    xs = [px for px, _ in pareto]
    ys = [py for _, py in pareto]
    return xs, ys


def _configure_selection_style(fig, mode, selected_color="red", selected_size=16):
    """
    Configure selection appearance for interactive plots.

    Args:
        fig: Plotly Figure object
        mode: Trace mode (e.g., "markers+text", "lines+markers")
        selected_color: Color for selected markers
        selected_size: Size for selected markers
    """
    fig.update_traces(
        selected=dict(marker=dict(color=selected_color, size=selected_size)),
        unselected=dict(marker=dict(opacity=0.4 if "text" in mode else 0.5)),
        selector=dict(mode=mode),
    )


def plot_prefill_performance_interactive(
    prefill_results: tuple,
    target_ttft: float,
    prefill_table_data: list,
) -> go.Figure:
    """
    Create interactive Plotly plot for prefill performance.

    Args:
        prefill_results: Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)
        target_ttft: Target TTFT in milliseconds (for reference line)
        prefill_table_data: List of table rows including config data

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    num_gpus_list, ttft_list, thpt_per_gpu_list = prefill_results

    fig = go.Figure()

    # Add scatter plot for data points with custom data
    fig.add_trace(
        go.Scatter(
            x=ttft_list,
            y=thpt_per_gpu_list,
            mode="markers+text",
            marker=dict(size=12, color="blue", line=dict(width=2, color="darkblue")),
            text=[f"{n} GPU(s)" for n in num_gpus_list],
            textposition="top center",
            textfont=dict(size=10),
            name="GPU Configurations",
            hovertemplate="<b>%{text}</b><br>"
            + "TTFT: %{x:.2f} ms<br>"
            + "Throughput: %{y:.2f} tokens/s/GPU<br>"
            + "<extra></extra>",
            customdata=prefill_table_data,
        )
    )

    # Add target TTFT line
    max_thpt = max(thpt_per_gpu_list) if thpt_per_gpu_list else 1000
    _add_target_line(fig, target_ttft, f"Target TTFT: {target_ttft} ms", max_thpt)

    # Configure layout
    fig.update_layout(
        title={
            "text": "Prefill Performance",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Time to First Token (ms)",
        yaxis_title="Prefill Throughput per GPU (tokens/s/GPU)",
        hovermode="closest",
        showlegend=True,
        autosize=True,
        clickmode="event+select",
    )

    # Configure selection appearance
    _configure_selection_style(fig, "markers+text", selected_color="red", selected_size=16)

    return fig


def plot_decode_performance_interactive(
    decode_results: list,
    target_itl: float,
    decode_table_data: list,
) -> go.Figure:
    """
    Create interactive Plotly plot for decode performance.

    Args:
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list, batch_size_list)
        target_itl: Target ITL in milliseconds (for reference line)
        decode_table_data: List of table rows including config data

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    fig = go.Figure()

    # Calculate the starting index for each GPU configuration in table data
    data_idx = 0

    # Plot each GPU configuration
    for idx, decode_result in enumerate(decode_results):
        num_gpus = decode_result[0]
        itl_list = decode_result[1]
        thpt_per_gpu_list = decode_result[2]
        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]

        # Extract rows for this GPU configuration from table data
        num_points = len(itl_list)
        customdata = decode_table_data[data_idx : data_idx + num_points]
        data_idx += num_points

        fig.add_trace(
            go.Scatter(
                x=itl_list,
                y=thpt_per_gpu_list,
                mode="lines+markers",
                marker=dict(size=8, color=color),
                line=dict(color=color, width=2),
                name=f"{num_gpus} GPU(s)",
                hovertemplate=f"<b>{num_gpus} GPU(s)</b><br>"
                + "ITL: %{x:.2f} ms<br>"
                + "Throughput: %{y:.2f} tokens/s/GPU<br>"
                + "<extra></extra>",
                customdata=customdata,
            )
        )

    # Add target ITL line
    all_thpt = [thpt for result in decode_results for thpt in result[2] if result[2]]
    max_thpt = max(all_thpt) if all_thpt else 1000
    _add_target_line(fig, target_itl, f"Target ITL: {target_itl} ms", max_thpt)

    # Configure layout
    fig.update_layout(
        title={
            "text": "Decode Performance",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Inter Token Latency (ms)",
        yaxis_title="Decode Throughput per GPU (tokens/s/GPU)",
        hovermode="closest",
        showlegend=True,
        autosize=True,
        clickmode="event+select",
    )

    # Configure selection appearance for markers
    _configure_selection_style(fig, "lines+markers", selected_color="yellow", selected_size=12)

    return fig


def plot_cost_sla_interactive(
    isl: int,
    osl: int,
    prefill_results: tuple,
    decode_results: list,
    gpu_cost_per_hour: float,
    cost_table_data: list,
) -> go.Figure:
    """
    Create interactive Plotly plot for cost vs SLA analysis.

    Args:
        isl: Input sequence length
        osl: Output sequence length
        prefill_results: Tuple of (num_gpus, ttft, thpt_per_gpu) for prefill
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list, batch_size_list) for decode
        gpu_cost_per_hour: Cost per GPU per hour in dollars. If None or 0, shows GPU hours instead.
        cost_table_data: List of table rows including config data

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    # Determine if we should use cost or GPU hours
    # Handle empty strings, None, or 0 values
    if gpu_cost_per_hour is None or gpu_cost_per_hour == "" or gpu_cost_per_hour == 0:
        use_gpu_hours = True
    else:
        use_gpu_hours = False

    # Compute Pareto fronts
    p_ttft, p_thpt = _compute_parato(prefill_results[1], prefill_results[2])

    _d_itl, _d_thpt = [], []
    for _d_result in decode_results:
        _d_itl.extend(_d_result[1])
        _d_thpt.extend(_d_result[2])
        # Note: _d_result[3] contains batch_sizes but not needed for pareto computation
    d_itl, d_thpt = _compute_parato(_d_itl, _d_thpt)

    # Convert to numpy arrays for element-wise operations
    p_ttft = np.array(p_ttft)
    p_thpt = np.array(p_thpt)
    d_itl = np.array(d_itl)
    d_thpt = np.array(d_thpt)

    # Calculate cost or GPU hours metrics
    fig = go.Figure()

    # Track data index for cost_table_data
    data_idx = 0

    for idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
        if use_gpu_hours:
            # Calculate GPU hours for this TTFT curve
            prefill_gpu_hours = isl * 1000 / _p_thpt / 3600

            # Calculate tokens per user and GPU hours arrays (element-wise operations)
            tokens_per_user_array = 1000 / d_itl  # Element-wise division with numpy array
            y_array = osl * 1000 / d_thpt / 3600 + prefill_gpu_hours
            y_label = "GPU Hours"
            hover_y_label = "GPU Hours"
        else:
            # Calculate costs for this TTFT curve
            prefill_cost = isl * 1000 / _p_thpt * gpu_cost_per_hour / 3600

            # Calculate tokens per user and cost arrays (element-wise operations)
            tokens_per_user_array = 1000 / d_itl  # Element-wise division with numpy array
            y_array = osl * 1000 / d_thpt * gpu_cost_per_hour / 3600 + prefill_cost
            y_label = "Cost ($)"
            hover_y_label = "Cost"

        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]

        # Extract rows for this curve from table data
        num_points = len(d_itl)
        customdata = cost_table_data[data_idx : data_idx + num_points]
        data_idx += num_points

        # Add line plot for this TTFT curve
        hover_template = (
            f"<b>TTFT: {_p_ttft:.2f}ms</b><br>"
            + "Tokens/User: %{x:.2f}<br>"
            + f"{hover_y_label}: %{{y:.3f}}<br>"
            + "<extra></extra>"
        )

        fig.add_trace(
            go.Scatter(
                x=tokens_per_user_array,
                y=y_array,
                mode="lines+markers",
                marker=dict(size=10, symbol="x", color=color, line=dict(width=2)),
                line=dict(color=color, width=2),
                name=f"TTFT: {_p_ttft:.2f}ms",
                hovertemplate=hover_template,
                customdata=customdata,
            )
        )

    # Configure layout
    if use_gpu_hours:
        title = f"GPU Hours Per 1000 i{isl}o{osl} requests Under Different SLA"
    else:
        title = f"Cost Per 1000 i{isl}o{osl} requests (GPU/hour = ${gpu_cost_per_hour:.2f}) Under Different SLA"

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Tokens per User",
        yaxis_title=y_label,
        hovermode="closest",
        showlegend=True,
        autosize=True,
        clickmode="event+select",
    )

    # Configure selection appearance for markers
    _configure_selection_style(fig, "lines+markers", selected_color="yellow", selected_size=14)

    return fig
