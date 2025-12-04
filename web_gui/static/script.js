// Global variables
let currentOperation = null;
let operationInterval = null;
let eventSource = null;

// Configuration - Adjust these values to control monitoring frequency and reduce CLI clutter
const CONFIG = {
    SYSTEM_MONITORING_INTERVAL: 30000,    // 30 seconds (was 5000ms = 5 seconds)
    OPERATION_MONITORING_INTERVAL: 1000,  // 1 second for real-time operation updates
    DISABLE_SYSTEM_MONITORING: false,     // Set to true to disable system monitoring entirely
    CLI_STREAM_DELAY: 100                 // 100ms delay for CLI output streaming
};

// To reduce CLI clutter:
// - Increase SYSTEM_MONITORING_INTERVAL to 60000 (1 minute) or 120000 (2 minutes)
// - Set DISABLE_SYSTEM_MONITORING to true to stop system monitoring entirely
// - OPERATION_MONITORING_INTERVAL should stay at 1000ms for real-time updates during operations
//
// Quick settings for different levels of verbosity:
// CONFIG.SYSTEM_MONITORING_INTERVAL = 30000;   // 30 seconds (current - balanced)
// CONFIG.SYSTEM_MONITORING_INTERVAL = 60000;   // 1 minute (less verbose)
// CONFIG.SYSTEM_MONITORING_INTERVAL = 120000;  // 2 minutes (minimal)
// CONFIG.DISABLE_SYSTEM_MONITORING = true;     // Disable completely (quietest)

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    try {
        // Load categories
        await loadCategories();
        await loadGPUInfo();
        await loadSystemInfo();
        
        // Start system monitoring
        startSystemMonitoring();
        
        // Start CLI output streaming
        startCLIOutputStreaming();
        
        // Load initial CLI output
        await loadInitialCLIOutput();
        
        logMessage('Application initialized successfully!', 'success');
    } catch (error) {
        logMessage(`Failed to initialize: ${error.message}`, 'error');
    }
}

async function loadCategories() {
    try {
        const response = await fetch('/api/categories');
        const categories = await response.json();
        
        const select = document.getElementById('category');
        select.innerHTML = '<option value="">Select Category</option>';
        
        categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
            select.appendChild(option);
        });
        
        select.value = 'chest'; // Default selection
        
        // Load data availability information
        await loadDataAvailability();
        
        // Add event listener for category selection
        select.addEventListener('change', updateCategoryInfo);
        
        // Show initial category info
        await updateCategoryInfo();
        
    } catch (error) {
        logMessage(`Failed to load categories: ${error.message}`, 'error');
    }
}

async function updateCategoryInfo() {
    const category = document.getElementById('category').value;
    const categoryInfo = document.getElementById('categoryInfo');
    
    if (!category) {
        categoryInfo.textContent = 'Please select a category';
        return;
    }
    
    try {
        const response = await fetch('/api/data-info');
        const dataInfo = await response.json();
        
        if (dataInfo.categories[category] && dataInfo.categories[category].available) {
            const size = dataInfo.categories[category].size_mb;
            categoryInfo.textContent = `✅ Data available: ${size} MB`;
            categoryInfo.style.color = 'var(--success)';
        } else {
            categoryInfo.textContent = '❌ No data file available for this category';
            categoryInfo.style.color = 'var(--error)';
        }
        
        // Refresh outputs when category changes
        await refreshOutputs();
    } catch (error) {
        categoryInfo.textContent = 'Error checking data availability';
        categoryInfo.style.color = 'var(--error)';
    }
}

async function loadGPUInfo() {
    try {
        const response = await fetch('/api/gpu-info');
        const gpuInfo = await response.json();
        
        if (gpuInfo.available) {
            document.getElementById('gpuInfo').textContent = `GPU 1: NVIDIA RTX 3050 (${gpuInfo.gpus.length} GPU(s) Available)`;
        } else {
            document.getElementById('gpuInfo').textContent = 'GPU 1: NVIDIA RTX 3050 (CUDA Not Available)';
        }
    } catch (error) {
        logMessage(`Failed to load GPU info: ${error.message}`, 'error');
    }
}

async function loadSystemInfo() {
    try {
        const response = await fetch('/api/system-info');
        const systemInfo = await response.json();
        
        document.getElementById('cpuInfo').textContent = `${systemInfo.cpu_count} cores`;
        document.getElementById('memoryInfo').textContent = `${systemInfo.memory_total.toFixed(1)} GB`;
        document.getElementById('storageInfo').textContent = `${systemInfo.disk_total.toFixed(1)} GB`;
    } catch (error) {
        logMessage(`Failed to load system info: ${error.message}`, 'error');
    }
}

async function loadInitialCLIOutput() {
    try {
        const response = await fetch('/api/cli-output');
        const cliOutput = await response.json();
        
        const console = document.getElementById('consoleOutput');
        console.innerHTML = ''; // Clear existing content
        
        cliOutput.forEach(entry => {
            addCLIOutputToConsole(entry);
        });
        
        // Scroll to bottom
        console.scrollTop = console.scrollHeight;
    } catch (error) {
        logMessage(`Failed to load initial CLI output: ${error.message}`, 'error');
    }
}

async function loadDataAvailability() {
    try {
        const response = await fetch('/api/data-info');
        const dataInfo = await response.json();
        
        const dataAvailability = document.getElementById('dataAvailability');
        const dataInfoDiv = document.getElementById('dataInfo');
        
        if (dataInfo.total_available > 0) {
            dataAvailability.style.display = 'block';
            
            let html = `<div class="data-summary">📊 ${dataInfo.total_available} categories available (${dataInfo.total_size_mb} MB total)</div>`;
            
            // Show only categories that have data
            Object.entries(dataInfo.categories).forEach(([category, info]) => {
                if (info.available) {
                    html += `
                        <div class="data-category available">
                            <span>✅ ${category.charAt(0).toUpperCase() + category.slice(1)}</span>
                            <span class="size">${info.size_mb} MB</span>
                        </div>
                    `;
                }
            });
            
            dataInfoDiv.innerHTML = html;
        } else {
            dataAvailability.style.display = 'none';
        }
        
    } catch (error) {
        logMessage(`Failed to load data availability: ${error.message}`, 'error');
    }
}

function startCLIOutputStreaming() {
    try {
        // Close existing connection if any
        if (eventSource) {
            eventSource.close();
        }
        
        // Create new EventSource for Server-Sent Events
        eventSource = new EventSource('/api/cli-output-stream');
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                addCLIOutputToConsole(data);
            } catch (error) {
                console.error('Failed to parse CLI output:', error);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('CLI output stream error:', error);
            // Try to reconnect after a delay
            setTimeout(() => {
                if (eventSource.readyState === EventSource.CLOSED) {
                    startCLIOutputStreaming();
                }
            }, 5000);
        };
        
        logMessage('CLI output streaming started', 'success');
    } catch (error) {
        logMessage(`Failed to start CLI output streaming: ${error.message}`, 'error');
    }
}

function addCLIOutputToConsole(entry) {
    const console = document.getElementById('consoleOutput');
    
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry log-cli';
    
    const timestampSpan = document.createElement('span');
    timestampSpan.className = 'log-timestamp';
    timestampSpan.textContent = `[${entry.timestamp}]`;
    
    const messageSpan = document.createElement('span');
    messageSpan.className = 'log-message log-cli';
    messageSpan.textContent = entry.message;
    
    // Add operation ID if available
    if (entry.operation_id) {
        const operationSpan = document.createElement('span');
        operationSpan.className = 'log-operation';
        operationSpan.textContent = `[${entry.operation_id}]`;
        logEntry.appendChild(operationSpan);
    }
    
    logEntry.appendChild(timestampSpan);
    logEntry.appendChild(messageSpan);
    
    console.appendChild(logEntry);
    
    // Auto-scroll to bottom
    console.scrollTop = console.scrollHeight;
    
    // Limit log entries to prevent memory issues
    if (console.children.length > 1000) {
        console.removeChild(console.firstChild);
    }
}

async function clearCLIOutput() {
    try {
        const response = await fetch('/api/clear-cli-output', { method: 'POST' });
        const result = await response.json();
        
        if (result.success) {
            const console = document.getElementById('consoleOutput');
            console.innerHTML = '';
            logMessage('CLI output cleared', 'success');
        } else {
            logMessage(`Failed to clear CLI output: ${result.message}`, 'error');
        }
    } catch (error) {
        logMessage(`Failed to clear CLI output: ${error.message}`, 'error');
    }
}

function startSystemMonitoring() {
    // System monitoring interval - reduced from 5 seconds to 30 seconds to reduce CLI clutter
    // You can adjust CONFIG.SYSTEM_MONITORING_INTERVAL or set CONFIG.DISABLE_SYSTEM_MONITORING to true
    if (CONFIG.DISABLE_SYSTEM_MONITORING) {
        return; // Disable system monitoring
    }
    
    setInterval(async () => {
        try {
            await loadSystemInfo();
        } catch (error) {
            // Silent fail for monitoring
        }
    }, CONFIG.SYSTEM_MONITORING_INTERVAL);
}

// Operation functions
async function runLineformerTest() {
    const category = document.getElementById('category').value;
    
    if (!category) {
        showToast('Please select a category', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/run-lineformer-test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category, output_path: 'output' })
        });
        
        const result = await response.json();
        if (result.status === 'started') {
            startOperationMonitoring(result.operation_id, 'lineformer_test');
            showToast('Lineformer test started successfully!', 'success');
        }
    } catch (error) {
        logMessage(`Failed to start Lineformer test: ${error.message}`, 'error');
        showToast('Failed to start test', 'error');
    }
}

async function run3DStatic() {
    const category = document.getElementById('category').value;
    
    if (!category) {
        showToast('Please select a category', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/run-3d-static', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            // Handle HTTP error responses
            const errorMsg = result.error || `Server error: ${response.status}`;
            logMessage(`Failed to start 3D static visualization: ${errorMsg}`, 'error');
            showToast(`Failed to start visualization: ${errorMsg}`, 'error');
            return;
        }
        
        if (result.status === 'started') {
            startOperationMonitoring(result.operation_id, '3d_static');
            showToast(`3D static visualization started for ${category}!`, 'success');
        } else if (result.status === 'error') {
            const errorMsg = result.error || 'Unknown error occurred';
            logMessage(`Failed to start 3D static visualization: ${errorMsg}`, 'error');
            showToast(`Failed to start visualization: ${errorMsg}`, 'error');
        }
    } catch (error) {
        logMessage(`Failed to start 3D static visualization: ${error.message}`, 'error');
        showToast(`Failed to start visualization: ${error.message}`, 'error');
    }
}

async function run3DGif() {
    const category = document.getElementById('category').value;
    
    if (!category) {
        showToast('Please select a category', 'warning');
        return;
    }
    
    try {
        const response = await fetch('/api/run-3d-gif', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ category })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            // Handle HTTP error responses
            const errorMsg = result.error || `Server error: ${response.status}`;
            logMessage(`Failed to start 3D GIF visualization: ${errorMsg}`, 'error');
            showToast(`Failed to start visualization: ${errorMsg}`, 'error');
            return;
        }
        
        if (result.status === 'started') {
            startOperationMonitoring(result.operation_id, '3d_gif');
            showToast(`3D GIF visualization started for ${category}!`, 'success');
        } else if (result.status === 'error') {
            const errorMsg = result.error || 'Unknown error occurred';
            logMessage(`Failed to start 3D GIF visualization: ${errorMsg}`, 'error');
            showToast(`Failed to start visualization: ${errorMsg}`, 'error');
        }
    } catch (error) {
        logMessage(`Failed to start 3D GIF visualization: ${error.message}`, 'error');
        showToast(`Failed to start visualization: ${error.message}`, 'error');
    }
}

async function runBatchLineformer() {
    try {
        const response = await fetch('/api/run-batch-lineformer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ output_path: 'output' })
        });
        
        const result = await response.json();
        if (result.status === 'started') {
            startOperationMonitoring(result.operation_id, 'batch_lineformer');
            showToast('Batch Lineformer test started!', 'success');
        }
    } catch (error) {
        logMessage(`Failed to start batch Lineformer test: ${error.message}`, 'error');
        showToast('Failed to start batch test', 'error');
    }
}

async function runBatch3DVisualization() {
    try {
        const response = await fetch('/api/run-batch-3d-visualization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        
        const result = await response.json();
        if (result.status === 'started') {
            startOperationMonitoring(result.operation_id, 'batch_3d_visualization');
            showToast('Batch 3D visualization started!', 'success');
        }
    } catch (error) {
        logMessage(`Failed to start batch 3D visualization: ${error.message}`, 'error');
        showToast('Failed to start batch visualization', 'error');
    }
}

function startOperationMonitoring(operationId, operationType) {
    if (operationInterval) {
        clearInterval(operationInterval);
    }
    
    currentOperation = operationId;
    disableButtons(true);
    
    operationInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/operation-status/${operationId}`);
            const operation = await response.json();
            
            if (operation.error) {
                clearInterval(operationInterval);
                enableButtons();
                return;
            }
            
            // Update progress
            if (operation.progress !== undefined) {
                document.getElementById('progressBar').style.width = `${operation.progress}%`;
            }
            
            // Update status - show progress percentage and last log message
            let statusText = operation.status || 'Running...';
            if (operation.progress !== undefined && operation.progress > 0) {
                statusText = `Progress: ${operation.progress}%`;
            }
            
            // Show last log message if available and meaningful
            if (operation.logs && operation.logs.length > 0) {
                const lastLog = operation.logs[operation.logs.length - 1];
                // Filter out debug messages and show meaningful progress
                if (!lastLog.includes('🔍') && !lastLog.includes('📁') && !lastLog.includes('✅') && !lastLog.includes('⚠️')) {
                    // Extract progress info from log if it contains progress
                    if (lastLog.match(/\d+%/)) {
                        statusText = lastLog.substring(0, 60) + (lastLog.length > 60 ? '...' : '');
                    } else if (lastLog.toLowerCase().includes('rendering') || lastLog.toLowerCase().includes('frame') || lastLog.toLowerCase().includes('processing')) {
                        statusText = lastLog.substring(0, 60) + (lastLog.length > 60 ? '...' : '');
                    }
                }
            }
            
            document.getElementById('statusText').textContent = statusText;
            
            // Check if operation completed
            if (operation.status === 'completed' || operation.status === 'failed') {
                clearInterval(operationInterval);
                enableButtons();
                
                if (operation.status === 'completed') {
                    showToast('Operation completed successfully!', 'success');
                    // Show output viewer if it's a visualization operation
                    if (operationType.includes('3d') || operationType.includes('visualization')) {
                        showOutputViewer();
                        refreshOutputs();
                    }
                } else {
                    showToast('Operation failed', 'error');
                }
            }
        } catch (error) {
            logMessage(`Failed to get operation status: ${error.message}`, 'error');
        }
    }, CONFIG.OPERATION_MONITORING_INTERVAL);
}

function disableButtons(disabled) {
    const buttons = ['run3DStatic', 'run3DGif'];
    buttons.forEach(id => {
        const button = document.getElementById(id);
        if (button) {
            button.disabled = disabled;
            if (disabled) {
                button.innerHTML = '<div class="loading"></div> Processing...';
            } else {
                // Restore original button content
                restoreButtonContent(id);
            }
        }
    });
}

function restoreButtonContent(buttonId) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    
    switch (buttonId) {
        case 'run3DStatic':
            button.innerHTML = '<i class="fas fa-cube"></i> Run Static Reconstruction (PNG)';
            break;
        case 'run3DGif':
            button.innerHTML = '<i class="fas fa-film"></i> Run Dynamic Reconstruction (GIF)';
            break;
    }
}

function enableButtons() {
    disableButtons(false);
}

// Output viewing functions
function showOutputViewer() {
    const outputPanel = document.getElementById('outputPanel');
    outputPanel.style.display = 'block';
    outputPanel.scrollIntoView({ behavior: 'smooth' });
}

async function refreshOutputs() {
    const category = document.getElementById('category').value;
    const outputType = document.getElementById('viewOutputType').value;
    
    if (!category) {
        showToast('Please select a category first', 'warning');
        return;
    }
    
    try {
        const response = await fetch(`/api/get-output-files?category=${category}&type=${outputType}`);
        const result = await response.json();
        
        if (result.error) {
            showToast(`Error loading outputs: ${result.error}`, 'error');
            return;
        }
        
        displayOutputs(result.files, category);
        
    } catch (error) {
        logMessage(`Failed to refresh outputs: ${error.message}`, 'error');
        showToast('Failed to refresh outputs', 'error');
    }
}

function displayOutputs(files, category) {
    const outputFiles = document.getElementById('outputFiles');
    
    if (files.length === 0) {
        outputFiles.innerHTML = '<div class="no-outputs">No outputs available for this category. Run a visualization first.</div>';
        return;
    }
    
    // Limit to show only the 20 most recent files to avoid overwhelming the UI
    const displayFiles = files.slice(0, 20);
    
    let html = '<div class="output-grid">';
    
    displayFiles.forEach((file, index) => {
        const fileSize = (file.size / 1024 / 1024).toFixed(2); // Convert to MB
        const fileType = file.type === 'gif' ? 'GIF Animation' : 'Static Image';
        const locationText = file.location === category ? `${category}/` : file.location;
        
        // Use lazy loading for images - only load when visible
        const imageSrc = index < 6 ? `/api/view-file/${file.path}` : ''; // Load first 6 immediately
        const dataSrc = `/api/view-file/${file.path}`;
        
        html += `
            <div class="output-item ${file.type}">
                <div class="output-preview">
                    <img ${imageSrc ? `src="${imageSrc}"` : ''} data-src="${dataSrc}" alt="${file.name}" class="preview-image" loading="lazy">
                </div>
                <div class="output-info">
                    <h4>${file.name}</h4>
                    <p class="file-type">${fileType}</p>
                    <p class="file-size">${fileSize} MB</p>
                    <p class="file-location">📁 ${locationText}</p>
                    <div class="output-actions">
                        <button class="btn btn-sm btn-primary" onclick="viewInFolder('${file.path}', '${category}')">
                            <i class="fas fa-folder-open"></i> View in Folder
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    
    if (files.length > 20) {
        html += `<div class="no-outputs" style="grid-column: 1 / -1; padding: 20px; text-align: center;">
            Showing 20 most recent files. Total: ${files.length} files available.
        </div>`;
    }
    
    html += '</div>';
    outputFiles.innerHTML = html;
    
    // Lazy load remaining images when they come into view
    const lazyImages = document.querySelectorAll('img[data-src]');
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    observer.unobserve(img);
                }
            });
        });
        
        lazyImages.forEach(img => imageObserver.observe(img));
    } else {
        // Fallback for browsers without IntersectionObserver
        lazyImages.forEach(img => {
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
        });
    }
}

async function viewInFolder(filePath, category) {
    try {
        // Determine the folder path from the file path
        // File paths are like: "3D_vis/chest/3d_chest_0.55_0.3_0.01.png" or "chest/elevation_20_sigma_0.55_alpha_0.3_axisoff/angle_357.0.png"
        const pathParts = filePath.split('/');
        
        // If file is in 3D_vis/{category}/, open that folder
        if (pathParts[0] === '3D_vis' && pathParts[1] === category) {
            const folderPath = `3D_vis/${category}`;
            const response = await fetch(`/api/open-output-folder?category=${category}&path=${encodeURIComponent(folderPath)}`);
            const result = await response.json();
            
            if (result.success) {
                showToast(`Opened folder: ${folderPath}`, 'success');
            } else {
                showToast(`Failed to open folder: ${result.message}`, 'error');
            }
        } else if (pathParts[0] === category) {
            // If file is in category folder, determine if it's in a subdirectory
            if (pathParts.length > 2) {
                // File is in a subdirectory, open that subdirectory
                const folderPath = pathParts.slice(0, -1).join('/');
                const response = await fetch(`/api/open-output-folder?category=${category}&path=${encodeURIComponent(folderPath)}`);
                const result = await response.json();
                
                if (result.success) {
                    showToast(`Opened folder: ${folderPath}`, 'success');
                } else {
                    showToast(`Failed to open folder: ${result.message}`, 'error');
                }
            } else {
                // File is directly in category folder
                const response = await fetch(`/api/open-output-folder?category=${category}&path=${category}`);
                const result = await response.json();
                
                if (result.success) {
                    showToast(`Opened folder: ${category}`, 'success');
                } else {
                    showToast(`Failed to open folder: ${result.message}`, 'error');
                }
            }
        } else {
            // Fallback to general category folder
            const response = await fetch(`/api/open-output-folder?category=${category}`);
            const result = await response.json();
            
            if (result.success) {
                showToast(result.message, 'success');
            } else {
                showToast(`Failed to open folder: ${result.message}`, 'error');
            }
        }
    } catch (error) {
        logMessage(`Failed to open folder: ${error.message}`, 'error');
        showToast('Failed to open folder', 'error');
    }
}

async function openOutputFolder() {
    try {
        const category = document.getElementById('category').value;
        const response = await fetch(`/api/open-output-folder?category=${category}`);
        const result = await response.json();
        
        if (result.success) {
            showToast(result.message, 'success');
        } else {
            showToast(`Failed to open folder: ${result.message}`, 'error');
        }
    } catch (error) {
        logMessage(`Failed to open output folder: ${error.message}`, 'error');
        showToast('Failed to open output folder', 'error');
    }
}

function logMessage(message, type = 'info') {
    const console = document.getElementById('consoleOutput');
    const timestamp = new Date().toLocaleTimeString();
    
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    
    const timestampSpan = document.createElement('span');
    timestampSpan.className = 'log-timestamp';
    timestampSpan.textContent = `[${timestamp}]`;
    
    const messageSpan = document.createElement('span');
    messageSpan.className = `log-message log-${type}`;
    messageSpan.textContent = message;
    
    logEntry.appendChild(timestampSpan);
    logEntry.appendChild(messageSpan);
    
    console.appendChild(logEntry);
    console.scrollTop = console.scrollHeight;
    
    // Limit log entries to prevent memory issues
    if (console.children.length > 1000) {
        console.removeChild(console.firstChild);
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const title = type === 'error' ? 'Error' : 
                 type === 'warning' ? 'Warning' : 
                 type === 'success' ? 'Success' : 'Info';
    
    toast.innerHTML = `
        <h4>${title}</h4>
        <p>${message}</p>
    `;
    
    document.getElementById('toastContainer').appendChild(toast);
    
    // Show toast
    setTimeout(() => toast.classList.add('show'), 100);
    
    // Hide and remove toast
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Add some interactive effects
document.addEventListener('mousemove', function(e) {
    const cards = document.querySelectorAll('.info-card');
    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        card.style.transform = `perspective(1000px) rotateX(${(y - rect.height/2) * 0.01}deg) rotateY(${(x - rect.width/2) * 0.01}deg) translateZ(10px)`;
    });
});

// Reset card transforms on mouse leave
document.addEventListener('mouseleave', function(e) {
    const cards = document.querySelectorAll('.info-card');
    cards.forEach(card => {
        card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)';
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});