// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * GPU Cost toggle functionality.
 * Handles switching between GPU hours and GPU cost display in the cost chart.
 */

// Store original cost data for restoration
let originalCostData = null

/**
 * Initialize GPU cost toggle listeners
 */
function initializeGpuCostToggle() {
    // Wait for Gradio to render the components
    const checkInterval = setInterval(() => {
        const checkbox = document.querySelector("#show_gpu_cost_checkbox input[type='checkbox']")
        const input = document.querySelector("#gpu_cost_per_hr_input input")
        
        if (checkbox && input) {
            clearInterval(checkInterval)
            
            // Add event listeners
            checkbox.addEventListener("change", () => updateCostDisplay())
            input.addEventListener("change", () => updateCostDisplay())
            input.addEventListener("input", () => updateCostDisplay())
        }
    }, 100)
    
    // Stop checking after 10 seconds
    setTimeout(() => clearInterval(checkInterval), 10000)
}

/**
 * Update cost chart display based on checkbox and input values
 */
function updateCostDisplay() {
    const checkbox = document.querySelector("#show_gpu_cost_checkbox input[type='checkbox']")
    const input = document.querySelector("#gpu_cost_per_hr_input input")
    
    if (!checkbox || !input || !charts.cost) return
    
    const showGpuCost = checkbox.checked
    const gpuCostPerHr = parseFloat(input.value)
    
    // Store original data on first access
    if (!originalCostData) {
        originalCostData = {
            datasets: charts.cost.data.datasets.map(ds => ({
                ...ds,
                data: ds.data.map(p => ({...p}))
            })),
            yAxisLabel: charts.cost.options.scales.y.title.text
        }
    }
    
    if (showGpuCost && gpuCostPerHr > 0 && !isNaN(gpuCostPerHr)) {
        // Transform data: multiply GPU hours by cost per hour
        charts.cost.data.datasets.forEach((dataset, dsIdx) => {
            const origDataset = originalCostData.datasets[dsIdx]
            dataset.data.forEach((point, ptIdx) => {
                const origPoint = origDataset.data[ptIdx]
                // Assuming y-axis is GPU hours, convert to cost
                point.y = origPoint.y * gpuCostPerHr
            })
        })
        
        // Update y-axis label
        charts.cost.options.scales.y.title.text = "Cost ($)"
    } else {
        // Restore original data
        charts.cost.data.datasets.forEach((dataset, dsIdx) => {
            const origDataset = originalCostData.datasets[dsIdx]
            dataset.data.forEach((point, ptIdx) => {
                const origPoint = origDataset.data[ptIdx]
                point.y = origPoint.y
            })
        })
        
        // Restore original y-axis label
        charts.cost.options.scales.y.title.text = originalCostData.yAxisLabel
    }
    
    charts.cost.update()
}

/**
 * Reset stored original data (call when new data is loaded)
 */
function resetGpuCostData() {
    originalCostData = null
}

// Initialize toggle when visualizations are loaded
const originalInitializeVisualizations = window.initializeVisualizations
window.initializeVisualizations = function(jsonData) {
    resetGpuCostData()
    originalInitializeVisualizations(jsonData)
    initializeGpuCostToggle()
}

