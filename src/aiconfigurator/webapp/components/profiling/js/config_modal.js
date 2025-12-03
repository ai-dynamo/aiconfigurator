// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Configuration modal display and copy functionality.
 * Modal is injected outside Gradio container to avoid .prose style conflicts.
 */

/**
 * Show config modal with YAML content
 */
window.showConfig = function(button) {
    const configYaml = button.getAttribute("data-config")
    if (!configYaml) {
        console.error("[Profiling] No config data found")
        return
    }
    
    // Unescape HTML entities
    const textarea = document.createElement("textarea")
    textarea.innerHTML = configYaml
    const decodedConfig = textarea.value
    
    // Display in modal
    const modal = document.getElementById("configModal")
    const content = document.getElementById("configContent")
    if (modal && content) {
        content.textContent = decodedConfig
        
        // Apply highlight.js YAML syntax highlighting
        if (typeof hljs !== "undefined" && hljs.highlightAll) {
            // clear previous highlights
            $("code").each((idx, element) => {
                $(element).removeAttr("data-highlighted")
            })
            // apply new highlights
            hljs.highlightAll()
        } else {
            console.error("[Profiling] Highlight.js not found")
        }
        
        modal.style.display = "block"
    }
}

/**
 * Close config modal
 */
window.closeConfigModal = function() {
    const modal = document.getElementById("configModal")
    if (modal) {
        modal.style.display = "none"
    }
}

/**
 * Copy config to clipboard
 */
window.copyConfig = function() {
    const content = document.getElementById("configContent")
    if (!content) {
        console.error("[Profiling] Config content not found")
        return
    }
    
    const text = content.textContent
    const copyBtn = event.target
    
    // Use modern Clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log("[Profiling] Config copied to clipboard")
                if (copyBtn) {
                    const originalText = copyBtn.textContent
                    copyBtn.textContent = "Copied!"
                    copyBtn.classList.add("active")
                    setTimeout(() => {
                        copyBtn.textContent = originalText
                        copyBtn.classList.remove("active")
                    }, 2000)
                }
            })
            .catch(err => {
                console.error("[Profiling] Copy failed:", err)
                fallbackCopy(text, copyBtn)
            })
    } else {
        fallbackCopy(text, copyBtn)
    }
}

/**
 * Download config as YAML file
 */
window.downloadConfig = function() {
    const content = document.getElementById("configContent")
    if (!content) {
        console.error("[Profiling] Config content not found")
        return
    }
    
    const text = content.textContent
    const downloadBtn = event.target
    
    try {
        // Create a Blob with the YAML content
        const blob = new Blob([text], { type: "text/yaml" })
        const url = URL.createObjectURL(blob)
        
        // Create a temporary anchor element and trigger download
        const a = document.createElement("a")
        a.href = url
        a.download = "config.yaml"
        document.body.appendChild(a)
        a.click()
        
        // Cleanup
        document.body.removeChild(a)
        URL.revokeObjectURL(url)
        
        console.log("[Profiling] Config downloaded as config.yaml")
        
        // Visual feedback
        if (downloadBtn) {
            const originalText = downloadBtn.textContent
            downloadBtn.textContent = "Downloaded!"
            downloadBtn.classList.add("active")
            setTimeout(() => {
                downloadBtn.textContent = originalText
                downloadBtn.classList.remove("active")
            }, 2000)
        }
    } catch (err) {
        console.error("[Profiling] Download failed:", err)
    }
}

/**
 * Fallback copy method for older browsers
 */
function fallbackCopy(text, copyBtn) {
    const textarea = document.createElement("textarea")
    textarea.value = text
    textarea.style.position = "fixed"
    textarea.style.opacity = "0"
    document.body.appendChild(textarea)
    textarea.select()
    
    try {
        const success = document.execCommand("copy")
        if (success && copyBtn) {
            const originalText = copyBtn.textContent
            copyBtn.textContent = "Copied!"
            copyBtn.classList.add("active")
            setTimeout(() => {
                copyBtn.textContent = originalText
                copyBtn.classList.remove("active")
            }, 2000)
        }
    } catch (err) {
        console.error("[Profiling] Fallback copy failed:", err)
    }
    
    document.body.removeChild(textarea)
}

/**
 * Close modal when clicking outside
 */
window.addEventListener("click", function(event) {
    const modal = document.getElementById("configModal")
    if (event.target === modal) {
        window.closeConfigModal()
    }
})
