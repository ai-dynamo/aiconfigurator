// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * DataTables rendering and configuration.
 */

// Global settings from JSON data
let allowConfirmDatapoint = false

/**
 * Initialize DataTables table with hover synchronization
 */
function createTable(wrapperId, columns, data, plotType, settings, referencePoints) {
    const tableId = `${plotType}_table`
    
    if (settings && settings.allow_confirm_datapoint !== undefined) {
        allowConfirmDatapoint = settings.allow_confirm_datapoint
    }
    
    if ($.fn.DataTable.isDataTable(`#${tableId}`)) {
        $(`#${tableId}`).DataTable().destroy()
        $(`#${tableId}`).remove()
    }
    
    const wrapper = document.getElementById(wrapperId)
    if (!wrapper) return null
    
    wrapper.innerHTML = `<table id="${tableId}" class="display" style="width:100%"></table>`
    
    const refPoints = referencePoints || {}
    
    const table = $(`#${tableId}`).DataTable({
        data: data,
        columns: columns.map((col, idx) => ({
            title: col,
            data: idx,
            render: idx === columns.length - 1 ? 
                (data, type, row, meta) => renderActionButtons(data, type, row, meta, plotType) 
                : null
        })),
        paging: true,
        pageLength: 25,
        searching: true,
        ordering: true,
        info: true,
        select: {
            style: "single"
        },
        createdRow: function(row, data, dataIndex) {
            // Add reference point styling
            if (refPoints.minLatencyUnderSLA && dataIndex === refPoints.minLatencyUnderSLA.tableIdx) {
                $(row).addClass("ref-point-min-latency")
            }
            if (refPoints.maxUnderSLA && dataIndex === refPoints.maxUnderSLA.tableIdx) {
                $(row).addClass("ref-point-max-under-sla")
            }
            if (refPoints.maxOverall && dataIndex === refPoints.maxOverall.tableIdx) {
                $(row).addClass("ref-point-max-overall")
            }
            
            // Add hover handler
            $(row).on("mouseenter", function() {
                highlightChartPoint(plotType, dataIndex)
                highlightedRows[plotType] = dataIndex
                $(this).addClass("table-hover-highlight")
            })
            
            $(row).on("mouseleave", function() {
                clearChartHighlight(plotType)
                highlightedRows[plotType] = null
                $(this).removeClass("table-hover-highlight")
            })
        }
    })
    
    return table
}

/**
 * Render action buttons (Show Config + Select) for table cells
 */
function renderActionButtons(data, type, row, meta, plotType) {
    if (type === "display") {
        const buttons = []
        
        // Show Config button (only if config exists and not empty)
        if (data && data.trim() !== "") {
            const escaped = String(data).replace(/"/g, "&quot;").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            buttons.push(`<button data-config="${escaped}" onclick="showConfig(this)">Show Config</button>`)
        }
        
        // Select button (only if allowed)
        if (allowConfirmDatapoint) {
            // Store row data in a global object instead of inline JSON
            const rowKey = `${plotType}_row_${meta.row}`
            window.profilingRowData = window.profilingRowData || {}
            window.profilingRowData[rowKey] = row
            
            buttons.push(`<button onclick="selectDatapoint(${meta.row}, '${plotType}')">Select</button>`)
        }
        
        return buttons.join(" ")
    }
    return data
}

/**
 * Handle datapoint selection
 */
window.selectDatapoint = function(rowIndex, plotType) {
    // Get row data from global storage
    const rowKey = `${plotType}_row_${rowIndex}`
    const rowData = window.profilingRowData ? window.profilingRowData[rowKey] : null
    
    if (!rowData) {
        console.error(`[Profiling] Row data not found for ${rowKey}`)
        return
    }
    
    // Send to Python: Fill hidden input and click hidden button
    const selectionData = {
        plotType: plotType,
        rowIndex: rowIndex,
        rowData: rowData,
        timestamp: new Date().toISOString()
    }
    
    // Find the hidden input and button
    const input = document.querySelector("#profiling_selection_input textarea")
    
    button = document.getElementById("profiling_selection_button")
    
    if (!button) {
        // Try finding any button inside the container
        const container = document.querySelector('[id*="profiling_selection_button"]')
        if (container) {
            button = container.querySelector("button")
        }
    }
    
    console.log("[Profiling] Found components:", {
        input: !!input,
        button: !!button,
        buttonElement: button
    })
    
    if (input && button) {
        // Ask user to confirm selection
        const confirmed = confirm("Do you want to select this datapoint?")
        
        if (confirmed) {
            // Fill the input with JSON data
            input.value = JSON.stringify(selectionData)
            
            // Trigger input change event
            input.dispatchEvent(new Event("input", { bubbles: true }))
            input.dispatchEvent(new Event("change", { bubbles: true }))
            
            // Try multiple ways to click the button
            button.click()  // Standard click
            
            // Also try dispatching click event
            button.dispatchEvent(new MouseEvent("click", {
                bubbles: true,
                cancelable: true,
                view: window
            }))
        }
    } else {
        console.error("[Profiling] Hidden input or button not found")
    }
}

