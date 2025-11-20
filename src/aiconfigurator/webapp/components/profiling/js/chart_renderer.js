/**
 * Chart.js rendering and configuration.
 */

/**
 * Create Chart.js chart with hover synchronization
 */
function createChart(canvasId, config, plotType) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Determine chart type based on plot
    const chartType = plotType === 'prefill' ? 'scatter' : 'line';
    const showLine = plotType !== 'prefill';
    
    // Configure datasets
    config.data.datasets.forEach(dataset => {
        if (showLine) {
            dataset.showLine = true;
            dataset.borderWidth = 2;
            dataset.pointRadius = 5;
            dataset.pointHoverRadius = 7;
        } else {
            dataset.pointRadius = 8;
            dataset.pointHoverRadius = 12;
        }
    });
    
    // Add target line as a dataset if provided
    if (config.targetLine) {
        const targetDataset = {
            label: config.targetLine.label,
            data: [
                {x: config.targetLine.value, y: config.yMin || 0},
                {x: config.targetLine.value, y: config.yMax || 1000}
            ],
            showLine: true,
            borderColor: 'red',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
            order: 999
        };
        config.data.datasets.push(targetDataset);
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
                        title: function(context) {
                            const point = context[0].raw;
                            if (point.gpuLabel) {
                                return point.gpuLabel;
                            }
                            return context[0].dataset.label || '';
                        },
                        label: function(context) {
                            const point = context.raw;
                            const dataset = context.dataset;
                            
                            if (dataset.label && dataset.label.startsWith('Target')) {
                                return null;
                            }
                            
                            const xLabel = config.xAxisLabel || 'X';
                            const yLabel = config.yAxisLabel || 'Y';
                            
                            return `${xLabel}: ${point.x.toFixed(2)}\n ${yLabel}: ${point.y.toFixed(2)}`;
                        }
                    }
                },
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: config.xAxisLabel
                    },
                    min: config.xMin
                },
                y: {
                    type: 'linear',
                    title: {
                        display: true,
                        text: config.yAxisLabel
                    },
                    min: config.yMin
                }
            },
            onHover: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const element = activeElements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataIndex = element.index;
                    const point = chart.data.datasets[datasetIndex].data[dataIndex];
                    
                    if (point.tableIdx !== undefined) {
                        highlightTableRow(plotType, point.tableIdx);
                    }
                } else {
                    clearTableHighlight(plotType);
                }
            },
            onClick: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const element = activeElements[0];
                    const datasetIndex = element.datasetIndex;
                    const dataIndex = element.index;
                    const point = chart.data.datasets[datasetIndex].data[dataIndex];
                    
                    if (point.tableIdx !== undefined) {
                        scrollToTableRow(plotType, point.tableIdx);
                    }
                }
            }
        }
    });
    
    return chart;
}

