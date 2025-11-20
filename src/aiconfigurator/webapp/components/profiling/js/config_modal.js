/**
 * Configuration modal display and copy functionality.
 */

/**
 * Show config modal with YAML content
 */
window.showConfig = function(button) {
    const configYaml = button.getAttribute('data-config');
    if (!configYaml) {
        console.error('[Profiling] No config data found');
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

/**
 * Close config modal
 */
window.closeConfigModal = function() {
    const modal = document.getElementById('configModal');
    if (modal) {
        modal.style.display = 'none';
    }
};

/**
 * Copy config to clipboard
 */
window.copyConfig = function() {
    const content = document.getElementById('configContent');
    if (!content) {
        console.error('[Profiling] Config content not found');
        return;
    }
    
    const text = content.textContent;
    const copyBtn = document.querySelector('.config-copy-btn');
    
    // Use modern Clipboard API
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text)
            .then(() => {
                console.log('[Profiling] Config copied to clipboard');
                if (copyBtn) {
                    const originalText = copyBtn.textContent;
                    copyBtn.textContent = 'Copied!';
                    copyBtn.classList.add('copied');
                    setTimeout(() => {
                        copyBtn.textContent = originalText;
                        copyBtn.classList.remove('copied');
                    }, 2000);
                }
            })
            .catch(err => {
                console.error('[Profiling] Copy failed:', err);
                fallbackCopy(text, copyBtn);
            });
    } else {
        fallbackCopy(text, copyBtn);
    }
};

/**
 * Fallback copy method for older browsers
 */
function fallbackCopy(text, copyBtn) {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        const success = document.execCommand('copy');
        if (success && copyBtn) {
            const originalText = copyBtn.textContent;
            copyBtn.textContent = 'Copied!';
            copyBtn.classList.add('copied');
            setTimeout(() => {
                copyBtn.textContent = originalText;
                copyBtn.classList.remove('copied');
            }, 2000);
        }
    } catch (err) {
        console.error('[Profiling] Fallback copy failed:', err);
    }
    
    document.body.removeChild(textarea);
}

/**
 * Close modal when clicking outside
 */
window.addEventListener('click', function(event) {
    const modal = document.getElementById('configModal');
    if (event.target === modal) {
        window.closeConfigModal();
    }
});

