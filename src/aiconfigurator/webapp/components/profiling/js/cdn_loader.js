/**
 * CDN library loader for Chart.js and DataTables.
 * Loads libraries dynamically and sets global flag when ready.
 */

(function loadLibraries() {
    function loadScript(src, name) {
        return new Promise((resolve, reject) => {
            if (name === 'jQuery' && typeof jQuery !== 'undefined') {
                resolve();
                return;
            }
            if (name === 'Chart.js' && typeof Chart !== 'undefined') {
                resolve();
                return;
            }
            
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed to load ${name}`));
            document.head.appendChild(script);
        });
    }
    
    if (!document.querySelector('link[href*="datatables"]')) {
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.href = 'https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css';
        document.head.appendChild(link);
    }
    
    loadScript('https://code.jquery.com/jquery-3.7.1.min.js', 'jQuery')
        .then(() => loadScript('https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js', 'DataTables'))
        .then(() => loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js', 'Chart.js'))
        .then(() => { window.profilingLibrariesLoaded = true; })
        .catch((err) => {
            window.profilingLibrariesLoaded = false;
            alert('⚠️ Failed to load visualization libraries.\n\nInternet access is required to load JS libraries from CDN.\n\nPlease check your internet connection and refresh the page.');
        });
})();

function checkLibrariesLoaded() {
    return window.profilingLibrariesLoaded === true || 
           (typeof Chart !== 'undefined' && typeof jQuery !== 'undefined' && typeof jQuery.fn.DataTable !== 'undefined');
}

function waitForLibraries(callback, retries = 40) {
    if (retries <= 0) return;
    
    if (checkLibrariesLoaded()) {
        callback();
    } else {
        setTimeout(() => waitForLibraries(callback, retries - 1), 500);
    }
}

