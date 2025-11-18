const DEBUG_MODE = false; // Turn on to see debug logs

function logDebug(message) {
    if (DEBUG_MODE) {
        console.log(`[DEBUG] ${message}`);
    }
}
// Storage for selected points (multi-selection)
const selectedPointKeys = {
    prefill: [],
    decode: [],
    cost: []
};

// Storage for all data points
const allDataPoints = {
    prefill: [],
    decode: [],
    cost: []
};

// Lookup from point key to row values
const pointDataLookup = {
    prefill: {},
    decode: {},
    cost: {}
};

const tableHeaders = {
    prefill: ["GPUs", "TTFT (ms)", "Throughput (tokens/s/GPU)", "Config"],
    decode: ["GPUs", "ITL (ms)", "Throughput (tokens/s/GPU)", "Config"],
    cost: [
        "TTFT (ms)",
        "Prefill Thpt (tokens/s/GPU)",
        "ITL (ms)",
        "Decode Thpt (tokens/s/GPU)",
        "Tokens/User",
        "Cost ($)",
        "Config"
    ]
};

function getTraceUid(trace, fallbackIndex) {
    if (!trace) {
        return `trace-${fallbackIndex}`;
    }
    return trace.uid || `trace-${fallbackIndex}`;
}

function makePointKey(traceUid, pointIndex) {
    return `${traceUid}:${pointIndex}`;
}

function getDisplayRows(plotType) {
    if (!selectedPointKeys[plotType] || selectedPointKeys[plotType].length === 0) {
        return allDataPoints[plotType].map((row) => row.values);
    }

    const lookup = pointDataLookup[plotType] || {};
    return selectedPointKeys[plotType]
        .map((key) => lookup[key])
        .filter(Boolean)
        .map((row) => row.values);
}

function computeSelectedKeys(plotDiv, lookup) {
    const keys = [];
    if (!plotDiv || !plotDiv.data) {
        return keys;
    }

    plotDiv.data.forEach((trace, traceIdx) => {
        if (!trace) {
            return;
        }

        const traceUid = getTraceUid(trace, traceIdx);
        const selectedPoints = trace.selectedpoints;

        if (!Array.isArray(selectedPoints) || selectedPoints.length === 0) {
            return;
        }

        selectedPoints.forEach((pointIndex) => {
            const key = makePointKey(traceUid, pointIndex);
            if (!lookup || lookup[key]) {
                keys.push(key);
            }
        });
    });

    return keys;
}

function normalizeRow(row) {
    if (row == null) {
        return [];
    }
    if (Array.isArray(row)) {
        return row.slice();
    }
    if (typeof row === "object") {
        if (typeof row[Symbol.iterator] === "function") {
            return Array.from(row);
        }
        return Object.values(row);
    }
    return [row];
}

function formatCell(value, isConfig = false) {
    if (value == null) {
        return "";
    }
    if (isConfig) {
        // For config cells, return a button with the config stored in a data attribute
        const configEscaped = String(value).replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
        return `<button class="config-btn" data-config="${configEscaped}" onclick="showConfig(this)">Show Config</button>`;
    }
    if (typeof value === "number" && Number.isFinite(value)) {
        if (Number.isInteger(value)) {
            return value.toString();
        }
        return value.toFixed(3);
    }
    return `${value}`;
}

function renderTableHTML(headers, rows) {
    const safeHeaders = headers || [];
    const headerCells = safeHeaders.map((header) => `<th>${header}</th>`).join("");

    let bodyHtml = "";
    if (!rows || rows.length === 0) {
        bodyHtml = `<tr><td class="dynamo-table-empty" colspan="${safeHeaders.length || 1}">No data selected yet. Click points on the plot to populate this table.</td></tr>`;
    } else {
        bodyHtml = rows
            .map((row) => {
                const normalized = normalizeRow(row);
                const length = safeHeaders.length > 0 ? safeHeaders.length : normalized.length;
                const cells = Array.from({ length }, (_, idx) => {
                    const value = normalized[idx];
                    // Check if this is the last column (Config column)
                    const isConfig = idx === length - 1 && safeHeaders[idx] === "Config";
                    return `<td>${formatCell(value, isConfig)}</td>`;
                });
                return `<tr>${cells.join("")}</tr>`;
            })
            .join("");
    }

    return `
        <div class="dynamo-table-wrapper">
            <table class="dynamo-table">
                <thead><tr>${headerCells}</tr></thead>
                <tbody>${bodyHtml}</tbody>
            </table>
        </div>
    `;
}

function updateDataTable(tableId, data, plotType) {
    const container = document.getElementById(tableId);
    if (!container) {
        logDebug(`Table container ${tableId} not found`);
        return;
    }

    const headers = tableHeaders[plotType] || [];
    container.innerHTML = renderTableHTML(headers, data);
    logDebug(`Updated table ${tableId} with ${data ? data.length : 0} rows`);
}

function resizePlotlyGraphs() {
    const plots = document.querySelectorAll('.js-plotly-plot');
    logDebug(`Found ${plots.length} Plotly graphs`);
    for (let i = 0; i < plots.length; i++) {
        if (window.Plotly && plots[i]) {
            window.Plotly.relayout(plots[i], {autosize: true});
            logDebug(`Resized plot ${i}`);
        }
    }
}

function setupPlotClickHandler(plotId, tableId, plotType) {
    const attemptSetup = () => {
        const plotContainer = document.querySelector(`#${plotId}`);
        if (!plotContainer) {
            logDebug(`Plot ${plotId} not found, retrying...`);
            setTimeout(attemptSetup, 500);
            return;
        }

        const plotDiv = plotContainer.querySelector('.js-plotly-plot');
        if (!plotDiv) {
            logDebug(`Plotly div not found in ${plotId}, retrying...`);
            setTimeout(attemptSetup, 500);
            return;
        }

        logDebug(`Setting up handlers for ${plotId}`);

        const headers = tableHeaders[plotType] || [];

        const syncSelection = (source) => {
            const lookup = pointDataLookup[plotType] || {};
            const keys = computeSelectedKeys(plotDiv, lookup);
            selectedPointKeys[plotType] = keys;
            updateDataTable(tableId, getDisplayRows(plotType), plotType);
            logDebug(`Selection synced for ${plotType} (${source || 'update'}): ${keys.length} point(s)`);
        };

        const refreshAllDataPoints = () => {
            if (!plotDiv || !plotDiv.data) {
                return;
            }

            const rows = [];
            const lookup = {};
            plotDiv.data.forEach((trace, traceIdx) => {
                if (!trace || !trace.customdata) {
                    return;
                }

                const traceUid = getTraceUid(trace, traceIdx);

                trace.customdata.forEach((item, pointIndex) => {
                    const normalized = normalizeRow(item);
                    if (normalized.length === 0) {
                        return;
                    }

                    const alignedRow = headers.length
                        ? headers.map((_, idx) => normalized[idx])
                        : normalized;

                    const key = makePointKey(traceUid, pointIndex);
                    const rowObj = { key, values: alignedRow };
                    rows.push(rowObj);
                    lookup[key] = rowObj;
                });
            });

            const newHash = JSON.stringify(rows.map((row) => [row.key, row.values]));
            if (plotDiv.__dynamo_data_hash !== newHash) {
                plotDiv.__dynamo_data_hash = newHash;
                allDataPoints[plotType] = rows;
                pointDataLookup[plotType] = lookup;
                syncSelection('data-refresh');
                logDebug(`Stored ${rows.length} data points for ${plotType}`);
            }
        };

        refreshAllDataPoints();

        if (plotDiv.on) {
            plotDiv.on('plotly_afterplot', refreshAllDataPoints);
            plotDiv.on('plotly_restyle', refreshAllDataPoints);
            plotDiv.on('plotly_relayout', refreshAllDataPoints);
        }

        plotDiv.on('plotly_click', function(data) {
            logDebug(`Click detected on ${plotId}`, data);
            if (data.points && data.points.length > 0) {
                setTimeout(() => syncSelection('click'), 0);
            }
        });

        if (plotDiv.on) {
            plotDiv.on('plotly_selected', function(eventData) {
                if (!eventData || !eventData.points) {
                    return;
                }

                syncSelection('selection-tool');
            });

            plotDiv.on('plotly_deselect', function() {
                syncSelection('deselect');
            });
        }

        logDebug(`Handlers configured for ${plotId}`);
    };

    setTimeout(attemptSetup, 500);
}

// Wait for DOM to be ready and set up observers
setTimeout(() => {
    // Find all tab buttons and add click listeners
    const tabButtons = document.querySelectorAll('button[role="tab"]');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            setTimeout(resizePlotlyGraphs, 150);
        });
    });

    // Use MutationObserver to detect tab visibility changes
    const observer = new MutationObserver(() => {
        resizePlotlyGraphs();
    });

    // Observe changes to elements with tab content
    const tabPanels = document.querySelectorAll('[role="tabpanel"]');
    tabPanels.forEach(panel => {
        observer.observe(panel, {
            attributes: true,
            attributeFilter: ['style', 'class', 'hidden']
        });
    });

    // Initial resize
    resizePlotlyGraphs();

    // Setup click handlers for all plots
    setupPlotClickHandler('prefill_plot', 'prefill_table', 'prefill');
    setupPlotClickHandler('decode_plot', 'decode_table', 'decode');
    setupPlotClickHandler('cost_plot', 'cost_table', 'cost');
}, 1000);

// Also resize on window resize
window.addEventListener('resize', resizePlotlyGraphs);

// Modal interaction functions for config display
// Attach to window object so inline onclick handlers can access them
window.showConfig = function(button) {
    const configYaml = button.getAttribute('data-config');
    if (!configYaml) {
        console.error('No config data found');
        return;
    }
    
    // Unescape HTML entities
    const textarea = document.createElement('textarea');
    textarea.innerHTML = configYaml;
    const decodedConfig = textarea.value;
    
    // Display in modal
    const modal = document.getElementById('configModal');
    const content = document.getElementById('configContent');
    if (modal && content) {
        content.textContent = decodedConfig;
        modal.style.display = 'block';
    }
};

window.closeConfigModal = function() {
    const modal = document.getElementById('configModal');
    if (modal) {
        modal.style.display = 'none';
    }
};

window.copyConfig = function() {
    const content = document.getElementById('configContent');
    const copyBtn = document.querySelector('.config-copy-btn');
    
    if (!content) {
        return;
    }
    
    const text = content.textContent;
    
    // Try to use the Clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            // Success feedback
            if (copyBtn) {
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                    copyBtn.classList.remove('copied');
                }, 2000);
            }
        }).catch(err => {
            console.error('Failed to copy:', err);
            fallbackCopy(text, copyBtn);
        });
    } else {
        fallbackCopy(text, copyBtn);
    }
};

function fallbackCopy(text, copyBtn) {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        document.execCommand('copy');
        if (copyBtn) {
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.classList.remove('copied');
            }, 2000);
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
    }
    
    document.body.removeChild(textarea);
}

// Close modal when clicking outside of it
window.addEventListener('click', function(event) {
    const modal = document.getElementById('configModal');
    if (event.target === modal) {
        closeConfigModal();
    }
});

