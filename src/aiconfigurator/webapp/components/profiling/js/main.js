/**
 * Main visualization orchestrator.
 * Initializes charts and tables from JSON data with synchronized interactions.
 */

// Storage for chart and table instances
const charts = {
    prefill: null,
    decode: null,
    cost: null
};

const tables = {
    prefill: null,
    decode: null,
    cost: null
};

/**
 * Inject config modal directly into document.body (outside Gradio container)
 * This prevents Gradio's .prose styles from affecting highlight.js
 */
function injectConfigModal() {
    if (document.getElementById('configModal')) {
        return; // Already injected
    }
    
    const modalHTML = `
<div id="configModal" class="config-modal">
    <div class="config-modal-content">
        <div class="config-modal-header">
            <h3>Configuration YAML</h3>
            <button class="config-modal-close" onclick="closeConfigModal()">&times;</button>
        </div>
        <div class="config-modal-body">
            <pre><code id="configContent" class="language-yaml"></code></pre>
        </div>
        <div class="config-modal-footer">
            <button class="config-action-btn" onclick="copyConfig()">Copy to Clipboard</button>
            <button class="config-action-btn" onclick="downloadConfig()">Download</button>
        </div>
    </div>
</div>`;
    
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

/**
 * Initialize all visualizations from JSON data
 */
function initializeVisualizations(jsonData) {
    waitForLibraries(() => {
        // Inject modal outside Gradio container
        injectConfigModal();
        
        const data = typeof jsonData === 'string' ? JSON.parse(jsonData) : jsonData;
        _initializeVisualizationsInternal(data);
    });
}

/**
 * Internal initialization after libraries are confirmed loaded
 */
function _initializeVisualizationsInternal(data) {
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            charts[key] = null;
        }
    });
    
    if (data.prefill) {
        const maxY = Math.max(...data.prefill.chart.datasets[0].data.map(p => p.y)) * 1.1;
        
        charts.prefill = createChart('prefill_chart', {
            data: { datasets: data.prefill.chart.datasets },
            xAxisLabel: data.prefill.chart.axes.x.title,
            yAxisLabel: data.prefill.chart.axes.y.title,
            xMin: data.prefill.chart.axes.x.min,
            yMin: data.prefill.chart.axes.y.min,
            yMax: maxY,
            targetLine: data.prefill.chart.target_line
        }, 'prefill');
        
        tables.prefill = createTable(
            'prefill_table_wrapper',
            data.prefill.table.columns,
            data.prefill.table.data,
            'prefill',
            data.settings
        );
    }
    
    if (data.decode) {
        const allYValues = data.decode.chart.datasets.flatMap(ds => ds.data.map(p => p.y));
        const maxY = Math.max(...allYValues) * 1.1;
        
        charts.decode = createChart('decode_chart', {
            data: { datasets: data.decode.chart.datasets },
            xAxisLabel: data.decode.chart.axes.x.title,
            yAxisLabel: data.decode.chart.axes.y.title,
            xMin: data.decode.chart.axes.x.min,
            yMin: data.decode.chart.axes.y.min,
            yMax: maxY,
            targetLine: data.decode.chart.target_line
        }, 'decode');
        
        tables.decode = createTable(
            'decode_table_wrapper',
            data.decode.table.columns,
            data.decode.table.data,
            'decode',
            data.settings
        );
    }
    
    if (data.cost) {
        charts.cost = createChart('cost_chart', {
            data: { datasets: data.cost.chart.datasets },
            title: data.cost.chart.title,
            xAxisLabel: data.cost.chart.axes.x.title,
            yAxisLabel: data.cost.chart.axes.y.title,
            xMin: data.cost.chart.axes.x.min,
            yMin: data.cost.chart.axes.y.min
        }, 'cost');
        
        tables.cost = createTable(
            'cost_table_wrapper',
            data.cost.table.columns,
            data.cost.table.data,
            'cost',
            data.settings
        );
    }
}

// Export for use in Gradio
window.initializeVisualizations = initializeVisualizations;

