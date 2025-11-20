# Color palette for charts
PLOTLY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Plot interaction instructions
PLOT_INTERACTION_INSTRUCTIONS = """
**How to interact with plots:**
- **Hover** over points to see detailed information
- **Click** points to select them (click again to deselect)
- **Multiple selection**: Click multiple points with shift key or select tools from the top right corner to compare specific configurations
- The table below each plot will filter to show only selected points, or all points if none are selected
"""

# Tab descriptions
PREFILL_TAB_DESCRIPTION = """
**Prefill Performance**: Interactive plot showing the relationship between Time to First Token (TTFT)
and throughput per GPU for different GPU counts. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
"""

DECODE_TAB_DESCRIPTION = """
**Decode Performance**: Interactive plot showing the relationship between Inter Token Latency (ITL)
and throughput per GPU for different GPU counts. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
"""

COST_TAB_DESCRIPTION = """
**Cost Analysis**: Interactive plot showing the cost per 1000 requests under different SLA configurations.
Lower curves represent better cost efficiency for the same throughput. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
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

# Modal HTML for config display
CONFIG_MODAL_HTML = """
<div id="configModal" class="config-modal">
    <div class="config-modal-content">
        <div class="config-modal-header">
            <h3>Configuration YAML</h3>
            <button class="config-modal-close" onclick="closeConfigModal()">&times;</button>
        </div>
        <div class="config-modal-body">
            <pre id="configContent"></pre>
        </div>
        <div class="config-modal-footer">
            <button class="config-copy-btn" onclick="copyConfig()">Copy to Clipboard</button>
        </div>
    </div>
</div>
"""
