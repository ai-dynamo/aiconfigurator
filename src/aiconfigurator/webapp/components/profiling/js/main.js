// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Main visualization orchestrator.
 * Initializes charts and tables from JSON data with synchronized interactions.
 */

// Storage for chart and table instances
const charts = {
    prefill: null,
    decode: null,
    cost: null
}

const tables = {
    prefill: null,
    decode: null,
    cost: null
}

/**
 * Inject config modal directly into document.body (outside Gradio container)
 * This prevents Gradio's .prose styles from affecting highlight.js
 */
function injectConfigModal() {
    if (document.getElementById("configModal")) {
        return // Already injected
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
</div>`
    
    document.body.insertAdjacentHTML("beforeend", modalHTML)
}

/**
 * Initialize all visualizations from JSON data
 */
function initializeVisualizations(jsonData) {
    waitForLibraries(() => {
        // Inject modal outside Gradio container
        injectConfigModal()
        
        const data = typeof jsonData === "string" ? JSON.parse(jsonData) : jsonData
        _initializeVisualizationsInternal(data)
    })
}

/**
 * Find reference points for a dataset (prefill/decode plots)
 */
function findReferencePoints(datasets, targetValue) {
    if (!targetValue) return { maxUnderSLA: null, maxOverall: null, minLatencyUnderSLA: null }
    
    // Flatten all points from all datasets
    const allPoints = []
    datasets.forEach((dataset, dsIdx) => {
        dataset.data.forEach((point, ptIdx) => {
            allPoints.push({
                ...point,
                datasetIndex: dsIdx,
                pointIndex: ptIdx
            })
        })
    })
    
    // Find max throughput overall (highest y value)
    const maxOverall = allPoints.reduce((max, point) => 
        !max || point.y > max.y ? point : max
    , null)
    
    // Find max throughput under SLA (highest y where x <= target)
    const pointsUnderSLA = allPoints.filter(p => p.x <= targetValue)
    const maxUnderSLA = pointsUnderSLA.reduce((max, point) => 
        !max || point.y > max.y ? point : max
    , null)
    
    // Find latency-optimized: lowest latency (x) among points under SLA
    const minLatencyUnderSLA = pointsUnderSLA.reduce((min, point) => 
        !min || point.x < min.x ? point : min
    , null)
    
    return { maxUnderSLA, maxOverall, minLatencyUnderSLA }
}

/**
 * Find reference points for cost plot (uses table data for throughput)
 */
function findCostReferencePoints(datasets, tableData, targetTTFT, targetITL) {
    if (!tableData || tableData.length === 0) {
        return { maxUnderSLA: null, maxOverall: null, minLatencyUnderSLA: null }
    }
    
    // Flatten all points from all datasets
    const allPoints = []
    datasets.forEach((dataset, dsIdx) => {
        dataset.data.forEach((point, ptIdx) => {
            // Get corresponding table row
            const tableRow = tableData[point.tableIdx]
            if (tableRow) {
                // Table structure: [TTFT, Prefill Thpt, ITL, Decode Thpt, Tokens/User, Cost, Config]
                allPoints.push({
                    ...point,
                    datasetIndex: dsIdx,
                    pointIndex: ptIdx,
                    ttft: tableRow[0],
                    prefillThpt: tableRow[1],
                    itl: tableRow[2],
                    decodeThpt: tableRow[3]  // Use decode throughput per GPU as the objective
                })
            }
        })
    })
    
    // Find max decode throughput overall (highest decodeThpt)
    const maxOverall = allPoints.reduce((max, point) => 
        !max || point.decodeThpt > max.decodeThpt ? point : max
    , null)
    
    // Find max throughput under SLA (highest decodeThpt where TTFT <= targetTTFT AND ITL <= targetITL)
    let pointsUnderSLA = allPoints
    
    // Apply TTFT constraint if provided
    if (targetTTFT !== null && targetTTFT !== undefined && !isNaN(targetTTFT)) {
        pointsUnderSLA = pointsUnderSLA.filter(p => p.ttft <= targetTTFT)
    }
    
    // Apply ITL constraint if provided
    if (targetITL !== null && targetITL !== undefined && !isNaN(targetITL)) {
        pointsUnderSLA = pointsUnderSLA.filter(p => p.itl <= targetITL)
    }
    
    const maxUnderSLA = pointsUnderSLA.reduce((max, point) => 
        !max || point.decodeThpt > max.decodeThpt ? point : max
    , null)
    
    // Find latency-optimized: lowest combined latency (TTFT + ITL) among points under SLA
    const minLatencyUnderSLA = pointsUnderSLA.reduce((min, point) => {
        const latencyScore = point.ttft + point.itl
        const minScore = min ? min.ttft + min.itl : Infinity
        return latencyScore < minScore ? point : min
    }, null)
    
    return { maxUnderSLA, maxOverall, minLatencyUnderSLA }
}

/**
 * Internal initialization after libraries are confirmed loaded
 */
function _initializeVisualizationsInternal(data) {
    Object.keys(charts).forEach(key => {
        if (charts[key]) {
            charts[key].destroy()
            charts[key] = null
        }
    })
    
    if (data.prefill) {
        const maxY = Math.max(...data.prefill.chart.datasets[0].data.map(p => p.y)) * 1.1
        const targetValue = data.prefill.chart.target_line?.value
        const refPoints = findReferencePoints(data.prefill.chart.datasets, targetValue)
        
        charts.prefill = createChart("prefill_chart", {
            data: { datasets: data.prefill.chart.datasets },
            xAxisLabel: data.prefill.chart.axes.x.title,
            yAxisLabel: data.prefill.chart.axes.y.title,
            xMin: data.prefill.chart.axes.x.min,
            yMin: data.prefill.chart.axes.y.min,
            yMax: maxY,
            targetLine: data.prefill.chart.target_line,
            referencePoints: refPoints
        }, "prefill")
        
        tables.prefill = createTable(
            "prefill_table_wrapper",
            data.prefill.table.columns,
            data.prefill.table.data,
            "prefill",
            data.settings,
            refPoints
        )
    }
    
    if (data.decode) {
        const allYValues = data.decode.chart.datasets.flatMap(ds => ds.data.map(p => p.y))
        const maxY = Math.max(...allYValues) * 1.1
        const targetValue = data.decode.chart.target_line?.value
        const refPoints = findReferencePoints(data.decode.chart.datasets, targetValue)
        
        charts.decode = createChart("decode_chart", {
            data: { datasets: data.decode.chart.datasets },
            xAxisLabel: data.decode.chart.axes.x.title,
            yAxisLabel: data.decode.chart.axes.y.title,
            xMin: data.decode.chart.axes.x.min,
            yMin: data.decode.chart.axes.y.min,
            yMax: maxY,
            targetLine: data.decode.chart.target_line,
            referencePoints: refPoints
        }, "decode")
        
        tables.decode = createTable(
            "decode_table_wrapper",
            data.decode.table.columns,
            data.decode.table.data,
            "decode",
            data.settings,
            refPoints
        )
    }
    
    if (data.cost) {
        // For cost plot, we need both TTFT and ITL targets, and use table data for throughput
        const targetTTFT = data.prefill?.chart?.target_line?.value
        const targetITL = data.decode?.chart?.target_line?.value
        const refPoints = findCostReferencePoints(
            data.cost.chart.datasets, 
            data.cost.table.data,
            targetTTFT,
            targetITL
        )
        
        charts.cost = createChart("cost_chart", {
            data: { datasets: data.cost.chart.datasets },
            title: data.cost.chart.title,
            xAxisLabel: data.cost.chart.axes.x.title,
            yAxisLabel: data.cost.chart.axes.y.title,
            xMin: data.cost.chart.axes.x.min,
            yMin: data.cost.chart.axes.y.min,
            referencePoints: refPoints,
            tableData: data.cost.table.data  // Pass table data for tooltip enrichment
        }, "cost")
        
        tables.cost = createTable(
            "cost_table_wrapper",
            data.cost.table.columns,
            data.cost.table.data,
            "cost",
            data.settings,
            refPoints
        )
    }
}

// Export for use in Gradio
window.initializeVisualizations = initializeVisualizations

