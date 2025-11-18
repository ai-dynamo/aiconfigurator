# Plotly color palette
PLOTLY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# CSS styles for custom table rendering
TABLE_CSS = """
<style>
    .dynamo-table-wrapper {
        overflow-x: auto;
        margin-top: 0.5rem;
    }
    .dynamo-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
    }
    .dynamo-table thead {
        background: rgba(0, 0, 0, 0.05);
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .dynamo-table th,
    .dynamo-table td {
        padding: 0.55rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    }
    .dynamo-table tbody tr:hover {
        background: rgba(0, 0, 0, 0.05);
    }
    .dynamo-table-empty {
        text-align: center;
        padding: 0.85rem 0;
        opacity: 0.7;
    }
    .config-btn {
        padding: 0.6rem 1.2rem;
        background: #1f77b4;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: background 0.2s;
    }
    .config-btn:hover {
        background: #1557a0;
    }
    .config-modal {
        display: none;
        position: fixed;
        z-index: 10000;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgba(0, 0, 0, 0.5);
    }
    .config-modal-content {
        background-color: #fefefe;
        margin: 5% auto;
        padding: 0;
        border: 1px solid #888;
        border-radius: 8px;
        width: 80%;
        max-width: 800px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .config-modal-header {
        padding: 1rem 1.5rem;
        background: #f7f7f7;
        border-bottom: 1px solid #ddd;
        border-radius: 8px 8px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .config-modal-header h3 {
        margin: 0;
        font-size: 1.2rem;
    }
    .config-modal-close {
        color: #aaa;
        font-size: 2rem;
        font-weight: bold;
        cursor: pointer;
        border: none;
        background: none;
        padding: 0;
        line-height: 1;
    }
    .config-modal-close:hover,
    .config-modal-close:focus {
        color: #000;
    }
    .config-modal-body {
        padding: 1.5rem;
        max-height: 70vh;
        overflow-y: auto;
    }
    .config-modal-body pre {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 4px;
        overflow-x: auto;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .config-modal-footer {
        padding: 1rem 1.5rem;
        background: #f7f7f7;
        border-top: 1px solid #ddd;
        border-radius: 0 0 8px 8px;
        text-align: right;
    }
    .config-copy-btn {
        padding: 0.7rem 1.5rem;
        background: #2ca02c;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background 0.2s;
    }
    .config-copy-btn:hover {
        background: #228a22;
    }
    .config-copy-btn.copied {
        background: #666;
    }
</style>
"""

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
    "Config",
]

DECODE_TABLE_HEADERS = [
    "GPUs",
    "ITL (ms)",
    "Throughput (tokens/s/GPU)",
    "Config",
]

COST_TABLE_HEADERS = [
    "TTFT (ms)",
    "Prefill Thpt (tokens/s/GPU)",
    "ITL (ms)",
    "Decode Thpt (tokens/s/GPU)",
    "Tokens/User",
    "Cost ($)",
    "Config",
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
