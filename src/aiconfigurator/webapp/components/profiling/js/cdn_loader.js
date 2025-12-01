// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * CDN library loader for Chart.js and DataTables.
 * Loads libraries dynamically and sets global flag when ready.
 */

(function loadLibraries() {
    function loadScript(src, name) {
        return new Promise((resolve, reject) => {
            if (name === "jQuery" && typeof jQuery !== "undefined") {
                resolve()
                return
            }
            if (name === "Chart.js" && typeof Chart !== "undefined") {
                resolve()
                return
            }
            if (name === "highlight.js" && typeof hljs !== "undefined") {
                resolve()
                return
            }
            
            const script = document.createElement("script")
            script.src = src
            script.onload = resolve
            script.onerror = () => reject(new Error(`Failed to load ${name}`))
            document.head.appendChild(script)
        })
    }
    
    function loadStylesheet(href) {
        if (!document.querySelector(`link[href="${href}"]`)) {
            const link = document.createElement("link")
            link.rel = "stylesheet"
            link.href = href
            document.head.appendChild(link)
        }
    }
    
    // Load stylesheets (independent, can load anytime)
    loadStylesheet("https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css")
    
    // Load both highlight.js themes
    const hljsLightTheme = document.createElement("link")
    hljsLightTheme.rel = "stylesheet"
    hljsLightTheme.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/atom-one-light.min.css"
    hljsLightTheme.media = "(prefers-color-scheme: light)"
    document.head.appendChild(hljsLightTheme)
    
    const hljsDarkTheme = document.createElement("link")
    hljsDarkTheme.rel = "stylesheet"
    hljsDarkTheme.href = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/atom-one-dark.min.css"
    hljsDarkTheme.media = "(prefers-color-scheme: dark)"
    document.head.appendChild(hljsDarkTheme)
    
    // Load independent libraries concurrently
    Promise.all([
        loadScript("https://code.jquery.com/jquery-3.7.1.min.js", "jQuery"),
        loadScript("https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js", "Chart.js"),
        loadScript("https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js", "highlight.js")
    ])
    .then(() => {
        // Configure highlight.js
        hljs.configure({
            languages: ["yaml"],
            ignoreUnescapedHTML: true // allow unescaped HTML in YAML (as we are free from XSS attacks)
        })
        // Load dependent libraries concurrently
        return Promise.all([
            loadScript("https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js", "DataTables"),
            loadScript("https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/languages/yaml.min.js", "highlight.js-yaml")
        ])
    })
    .then(() => { 
        window.profilingLibrariesLoaded = true 
    })
    .catch((err) => {
        window.profilingLibrariesLoaded = false
        alert("⚠️ Failed to load visualization libraries.\n\nInternet access is required to load JS libraries from CDN.\n\nPlease check your internet connection and refresh the page.")
    })
})()

function checkLibrariesLoaded() {
    return window.profilingLibrariesLoaded === true || 
           (typeof Chart !== "undefined" && typeof jQuery !== "undefined" && typeof jQuery.fn.DataTable !== "undefined" && typeof hljs !== "undefined")
}

function waitForLibraries(callback, retries = 40) {
    if (retries <= 0) return
    
    if (checkLibrariesLoaded()) {
        callback()
    } else {
        setTimeout(() => waitForLibraries(callback, retries - 1), 500)
    }
}

