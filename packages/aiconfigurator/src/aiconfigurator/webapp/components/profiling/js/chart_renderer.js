// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Chart.js rendering and configuration.
 */

/**
 * Create Chart.js chart with hover synchronization
 */
function createChart(canvasId, config, plotType) {
    const ctx = document.getElementById(canvasId)
    if (!ctx) return null

    // Determine chart type based on plot
    const chartType = plotType === "prefill" ? "scatter" : "line"
    const showLine = plotType !== "prefill"

    // Mark reference points in datasets
    const refPoints = config.referencePoints || {}

    // For cost plot, enrich all points with table data
    if (plotType === "cost" && config.tableData) {
        config.data.datasets.forEach((dataset) => {
            dataset.data.forEach((point) => {
                const tableRow = config.tableData[point.tableIdx]
                if (tableRow) {
                    // Table structure: [TTFT, Prefill Thpt, ITL, Decode Thpt, Tokens/User, Cost, Config]
                    point.ttft = tableRow[0]
                    point.prefillThpt = tableRow[1]
                    point.itl = tableRow[2]
                    point.decodeThpt = tableRow[3]
                }
            })
        })
    }

    // Configure datasets
    config.data.datasets.forEach((dataset, dsIdx) => {
        if (showLine) {
            dataset.showLine = true
            dataset.borderWidth = 2
            dataset.pointRadius = 5
            dataset.pointHoverRadius = 7
        } else {
            dataset.pointRadius = 8
            dataset.pointHoverRadius = 12
        }

        // Mark reference points with special styling
        dataset.data.forEach((point, ptIdx) => {
            // Check if this is max throughput under SLA (red)
            if (refPoints.maxUnderSLA &&
                point.tableIdx === refPoints.maxUnderSLA.tableIdx &&
                dsIdx === refPoints.maxUnderSLA.datasetIndex &&
                ptIdx === refPoints.maxUnderSLA.pointIndex) {
                point.isMaxUnderSLA = true
            }
            // Check if this is max throughput overall (yellow)
            if (refPoints.maxOverall &&
                point.tableIdx === refPoints.maxOverall.tableIdx &&
                dsIdx === refPoints.maxOverall.datasetIndex &&
                ptIdx === refPoints.maxOverall.pointIndex) {
                point.isMaxOverall = true
            }
            // Check if this is latency-optimized (green)
            if (refPoints.minLatencyUnderSLA &&
                point.tableIdx === refPoints.minLatencyUnderSLA.tableIdx &&
                dsIdx === refPoints.minLatencyUnderSLA.datasetIndex &&
                ptIdx === refPoints.minLatencyUnderSLA.pointIndex) {
                point.isMinLatency = true
            }
        })
    })

    // Add target line as a dataset if provided
    if (config.targetLine) {
        const targetDataset = {
            label: config.targetLine.label,
            data: [
                { x: config.targetLine.value, y: config.yMin || 0 },
                { x: config.targetLine.value, y: config.yMax || 1000 }
            ],
            showLine: true,
            borderColor: "red",
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
            order: 999
        }
        config.data.datasets.push(targetDataset)
    }

    const chart = new Chart(ctx, {
        type: chartType,
        data: config.data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: !!config.title,
                    text: config.title,
                    font: { size: 16 }
                },
                tooltip: {
                    callbacks: {
                        title: function (context) {
                            const point = context[0].raw
                            if (point.gpuLabel) {
                                return point.gpuLabel
                            }
                            return context[0].dataset.label || ""
                        },
                        label: function (context) {
                            const point = context.raw
                            const dataset = context.dataset

                            if (dataset.label && dataset.label.startsWith("Target")) {
                                return null
                            }

                            const xLabel = config.xAxisLabel || "X"
                            // Read y-axis label dynamically from chart options (allows GPU cost toggle to update it)
                            const yLabel = context.chart.options.scales.y.title.text || config.yAxisLabel || "Y"

                            const labels = [`${xLabel}: ${point.x.toFixed(2)}`, `${yLabel}: ${point.y.toFixed(6)}`]

                            // For cost plot, always show TTFT, ITL, and decode throughput
                            if (plotType === "cost" && point.ttft !== undefined) {
                                labels.push(`TTFT: ${point.ttft.toFixed(2)} ms`)
                                labels.push(`ITL: ${point.itl.toFixed(2)} ms`)
                                labels.push(`Decode Thpt: ${point.decodeThpt.toFixed(2)} tokens/s/GPU`)
                            }

                            // Add reference point labels at the top
                            if (point.isMinLatency) {
                                const labelText = plotType === "cost"
                                    ? "🟢 Latency-Optimized (Lowest TTFT+ITL Under SLA)"
                                    : "🟢 Latency-Optimized (Lowest Under SLA)"
                                labels.unshift(labelText)
                            }
                            if (point.isMaxUnderSLA) {
                                const labelText = plotType === "cost"
                                    ? "🔴 Max Decode Throughput/GPU Under SLA"
                                    : "🔴 Max Throughput Under SLA"
                                labels.unshift(labelText)
                            }
                            if (point.isMaxOverall) {
                                const labelText = plotType === "cost"
                                    ? "🟡 Max Decode Throughput/GPU"
                                    : "🟡 Max Throughput"
                                labels.unshift(labelText)
                            }

                            return labels
                        }
                    }
                },
                legend: {
                    display: true,
                    position: "top"
                }
            },
            scales: {
                x: {
                    type: "linear",
                    title: {
                        display: true,
                        text: config.xAxisLabel
                    },
                    min: config.xMin
                },
                y: {
                    type: "linear",
                    title: {
                        display: true,
                        text: config.yAxisLabel
                    },
                    min: config.yMin
                }
            },
            onHover: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const element = activeElements[0]
                    const datasetIndex = element.datasetIndex
                    const dataIndex = element.index
                    const point = chart.data.datasets[datasetIndex].data[dataIndex]

                    if (point.tableIdx !== undefined) {
                        highlightTableRow(plotType, point.tableIdx)
                    }
                } else {
                    clearTableHighlight(plotType)
                }
            },
            onClick: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const element = activeElements[0]
                    const datasetIndex = element.datasetIndex
                    const dataIndex = element.index
                    const point = chart.data.datasets[datasetIndex].data[dataIndex]

                    if (point.tableIdx !== undefined) {
                        scrollToTableRow(plotType, point.tableIdx)
                    }
                }
            }
        },
        plugins: [{
            id: "referencePointsPlugin",
            afterDatasetsDraw: function (chart) {
                const ctx = chart.ctx

                chart.data.datasets.forEach((dataset, dsIdx) => {
                    const meta = chart.getDatasetMeta(dsIdx)

                    dataset.data.forEach((point, ptIdx) => {
                        if (point.isMaxUnderSLA || point.isMaxOverall || point.isMinLatency) {
                            const element = meta.data[ptIdx]
                            if (!element) return

                            const x = element.x
                            const y = element.y
                            const radius = element.options.radius + 8

                            ctx.save()
                            // Priority: maxUnderSLA (red) > maxOverall (yellow) > minLatency (green)
                            if (point.isMaxUnderSLA) {
                                ctx.strokeStyle = "rgba(255, 0, 0, 0.8)"    // Red
                            } else if (point.isMaxOverall) {
                                ctx.strokeStyle = "rgba(255, 215, 0, 0.8)" // Yellow
                            } else {
                                ctx.strokeStyle = "rgba(34, 197, 94, 0.9)"  // Green
                            }
                            ctx.lineWidth = 3
                            ctx.setLineDash([5, 5])

                            ctx.beginPath()
                            ctx.arc(x, y, radius, 0, 2 * Math.PI)
                            ctx.stroke()
                            ctx.restore()
                        }
                    })
                })
            }
        }]
    })

    return chart
}

