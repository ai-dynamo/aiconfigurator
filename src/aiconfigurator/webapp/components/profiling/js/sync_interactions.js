// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Synchronized interactions between charts and tables.
 */

// Track highlighted rows
const highlightedRows = {
    prefill: null,
    decode: null,
    cost: null
}

/**
 * Highlight table row from chart hover
 */
function highlightTableRow(plotType, rowIndex) {
    const table = tables[plotType]
    if (!table) return
    
    const tableId = `${plotType}_table`
    
    // Clear previous highlights
    $(`#${tableId} tbody tr`).removeClass("table-hover-highlight")
    
    // Highlight the row
    const row = table.row(rowIndex).node()
    if (row) {
        $(row).addClass("table-hover-highlight")
        highlightedRows[plotType] = rowIndex
    }
}

/**
 * Clear table highlight
 */
function clearTableHighlight(plotType) {
    const tableId = `${plotType}_table`
    $(`#${tableId} tbody tr`).removeClass("table-hover-highlight")
    highlightedRows[plotType] = null
}

/**
 * Highlight chart point from table hover
 */
function highlightChartPoint(plotType, rowIndex) {
    const chart = charts[plotType]
    if (!chart) return
    
    // Find the point in the chart data
    chart.data.datasets.forEach((dataset, datasetIdx) => {
        dataset.data.forEach((point, pointIdx) => {
            if (point.tableIdx === rowIndex) {
                chart.setActiveElements([{datasetIndex: datasetIdx, index: pointIdx}])
                chart.tooltip.setActiveElements([{datasetIndex: datasetIdx, index: pointIdx}])
                chart.update("none")
            }
        })
    })
}

/**
 * Clear chart highlight
 */
function clearChartHighlight(plotType) {
    const chart = charts[plotType]
    if (!chart) return
    
    chart.setActiveElements([])
    chart.tooltip.setActiveElements([])
    chart.update("none")
}

/**
 * Scroll to table row
 */
function scrollToTableRow(plotType, rowIndex) {
    const table = tables[plotType]
    if (!table) return
    
    // Go to the page containing this row
    const page = Math.floor(rowIndex / table.page.len())
    table.page(page).draw("page")
    
    // Scroll to the row
    const row = table.row(rowIndex).node()
    if (row) {
        row.scrollIntoView({ behavior: "smooth", block: "center" })
        // Flash highlight
        $(row).addClass("table-flash-highlight")
        setTimeout(() => $(row).removeClass("table-flash-highlight"), 1000)
    }
}

