# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Color palette for charts
PLOTLY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Plot interaction instructions
PLOT_INTERACTION_INSTRUCTIONS = """
How to interact with plots:

Hover over chart points to see detailed information and highlight the corresponding table row

Click chart points to jump to the corresponding row in the table

Hover over table rows to highlight the corresponding chart point

Use the "Show Config" button in the table to view configuration details if presents
"""

# Tab descriptions
PREFILL_TAB_DESCRIPTION = """
**Prefill Performance**: Interactive plot showing the relationship between Time to First Token (TTFT)
and throughput per GPU for different GPU counts. Hover over points to see details and highlight table rows.
Click points to jump to the corresponding row in the table.
"""

DECODE_TAB_DESCRIPTION = """
**Decode Performance**: Interactive plot showing the relationship between Inter Token Latency (ITL)
and throughput per GPU for different GPU counts. Hover over points to see details and highlight table rows.
Click points to jump to the corresponding row in the table.
"""

COST_TAB_DESCRIPTION = """
**Cost Analysis**: Interactive plot showing the cost per 1000 requests under different SLA configurations.
Lower curves represent better cost efficiency for the same throughput. Hover over points to see details and highlight table rows.
Click points to jump to the corresponding row in the table.
"""

# Table headers for different performance metrics
PREFILL_TABLE_HEADERS = [
    "GPUs",
    "TTFT (ms)",
    "Throughput (tokens/s/GPU)",
    "Action",
]

DECODE_TABLE_HEADERS = [
    "GPUs",
    "ITL (ms)",
    "Throughput (tokens/s/GPU)",
    "Action",
]

COST_TABLE_HEADERS = [
    "TTFT (ms)",
    "Prefill Thpt (tokens/s/GPU)",
    "ITL (ms)",
    "Decode Thpt (tokens/s/GPU)",
    "Tokens/User",
    "Cost ($)",
    "Action",
]
