// API Base URL - use relative path for production compatibility
const API_BASE = window.location.origin;

// Global state
let currentPage = 'home';
let selectedFiles = [];
let uploadedFile = null;

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    initializeNavigation();
    checkModelStatus();
    setupFileInputs();
    loadUptime();
    setInterval(loadUptime, 60000); // Update every minute
});

// Navigation
function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const page = btn.getAttribute('data-page');
            switchPage(page);
        });
    });
}

function switchPage(page) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    
    // Show selected page
    document.getElementById(page).classList.add('active');
    
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-page') === page) {
            btn.classList.add('active');
        }
    });
    
    currentPage = page;
    
    // Load page-specific data
    if (page === 'visualizations') {
        loadVisualizations();
    } else if (page === 'retrain') {
        loadRetrainStats();
        loadRecentSessions();
    } else if (page === 'uptime') {
        loadUptime();
    }
}

// Model Status
async function checkModelStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        const statusIndicator = document.getElementById('statusIndicator');
        if (data.model_loaded) {
            statusIndicator.textContent = '✅ Model Loaded';
            statusIndicator.className = 'status-indicator success';
        } else {
            statusIndicator.textContent = '❌ Model Not Loaded';
            statusIndicator.className = 'status-indicator error';
        }
    } catch (error) {
        const statusIndicator = document.getElementById('statusIndicator');
        statusIndicator.textContent = '❌ Connection Error';
        statusIndicator.className = 'status-indicator error';
    }
}

// File Inputs
function setupFileInputs() {
    // Single image input
    const imageInput = document.getElementById('imageInput');
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadedFile = file;
            document.getElementById('fileName').textContent = file.name;
            displayImagePreview(file);
            document.getElementById('predictBtn').disabled = false;
        }
    });
    
    // Multiple image input
    const multiImageInput = document.getElementById('multiImageInput');
    multiImageInput.addEventListener('change', function(e) {
        selectedFiles = Array.from(e.target.files);
        document.getElementById('fileCount').textContent = `${selectedFiles.length} file(s) selected`;
        displayUploadedFiles(selectedFiles);
        document.getElementById('saveFilesBtn').disabled = selectedFiles.length === 0;
    });
    
    // Sliders
    const epochsSlider = document.getElementById('epochs');
    const fineTuneSlider = document.getElementById('fineTuneEpochs');
    
    epochsSlider.addEventListener('input', function() {
        document.getElementById('epochsValue').textContent = this.value;
    });
    
    fineTuneSlider.addEventListener('input', function() {
        document.getElementById('fineTuneEpochsValue').textContent = this.value;
    });
}

// Image Preview
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
    };
    reader.readAsDataURL(file);
}

// Display Uploaded Files
function displayUploadedFiles(files) {
    const container = document.getElementById('uploadedFiles');
    if (files.length === 0) {
        container.innerHTML = '';
        return;
    }
    
    let html = '<h3>Selected Files:</h3><div class="uploaded-files">';
    files.forEach(file => {
        html += `<div class="uploaded-file-item">- ${file.name} (${formatFileSize(file.size)})</div>`;
    });
    html += '</div>';
    container.innerHTML = html;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// Prediction
async function makePrediction() {
    if (!uploadedFile) {
        showMessage('predictionOutput', 'Please select an image first.', 'error');
        return;
    }
    
    const output = document.getElementById('predictionOutput');
    output.innerHTML = '<div class="loading">Predicting...</div>';
    
    try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const result = await response.json();
        displayPredictionResult(result);
        
    } catch (error) {
        output.innerHTML = `<div class="result-message error">Prediction error: ${error.message}</div>`;
    }
}

function displayPredictionResult(result) {
    const output = document.getElementById('predictionOutput');
    
    let html = `
        <div class="metric-card">
            <div class="metric-label">Predicted Class</div>
            <div class="metric-value">${result.predicted_class.toUpperCase()}</div>
            <div style="margin-top: 0.5rem; color: #666;">${(result.confidence * 100).toFixed(2)}% confidence</div>
        </div>
        
        <h3 style="margin-top: 1.5rem;">Class Probabilities</h3>
    `;
    
    // Sort probabilities
    const sortedProbs = Object.entries(result.probabilities)
        .sort((a, b) => b[1] - a[1]);
    
    // Create bar chart data
    const chartData = [{
        x: sortedProbs.map(p => p[0]),
        y: sortedProbs.map(p => p[1] * 100),
        type: 'bar',
        marker: {
            color: sortedProbs.map(p => `rgba(31, 119, 180, ${0.5 + p[1] * 0.5})`)
        }
    }];
    
    const chartLayout = {
        title: 'Prediction Probabilities',
        xaxis: { title: 'Class' },
        yaxis: { title: 'Probability (%)' },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    };
    
    html += '<div id="probChart" style="margin: 1rem 0;"></div>';
    html += '<table class="probability-table"><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>';
    
    sortedProbs.forEach(([class_name, prob]) => {
        html += `<tr>
            <td>${class_name}</td>
            <td>
                ${(prob * 100).toFixed(2)}%
                <div class="probability-bar" style="width: ${prob * 100}%"></div>
            </td>
        </tr>`;
    });
    
    html += '</tbody></table>';
    output.innerHTML = html;
    
    // Render Plotly chart
    Plotly.newPlot('probChart', chartData, chartLayout, {responsive: true});
}

// Visualizations
async function loadVisualizations() {
    const container = document.getElementById('visualizationsContent');
    container.innerHTML = '<div class="loading">Loading visualizations...</div>';
    
    try {
        // Try to load feature data
        const response = await fetch(`${API_BASE}/visualizations/data`);
        
        if (!response.ok) {
            throw new Error('Failed to load visualization data');
        }
        
        const data = await response.json();
        
        if (!data.features || data.features.length === 0) {
            container.innerHTML = `
                <div class="result-message info">
                    Feature CSV file not found. Please run feature extraction first.
                </div>
            `;
            return;
        }
        
        displayVisualizations(data);
        
    } catch (error) {
        container.innerHTML = `
            <div class="result-message error">
                Error loading visualizations: ${error.message}
            </div>
        `;
    }
}

function displayVisualizations(data) {
    const container = document.getElementById('visualizationsContent');
    let html = `
        <div class="result-message success">
            Loaded features from ${data.source}
        </div>
        <div class="result-message info">
            Total samples: ${data.total_samples}
        </div>
    `;
    
    // Feature 1: Mean Intensity
    if (data.has_mean_intensity) {
        html += `
            <div class="visualization-section">
                <h3>Feature 1: Mean Intensity Distribution by Tumor Class</h3>
                <p><strong>Interpretation</strong>: This visualization shows the average pixel intensity across different tumor types.</p>
                <div id="chart1"></div>
            </div>
        `;
    }
    
    // Feature 2: Standard Deviation
    if (data.has_std_intensity) {
        html += `
            <div class="visualization-section">
                <h3>Feature 2: Intensity Variability (Standard Deviation) by Class</h3>
                <p><strong>Interpretation</strong>: Standard deviation measures texture variability in the images.</p>
                <div id="chart2"></div>
            </div>
        `;
    }
    
    // Feature 3: Gradient Mean
    if (data.has_gradient_mean) {
        html += `
            <div class="visualization-section">
                <h3>Feature 3: Edge Strength (Gradient Mean) by Class</h3>
                <p><strong>Interpretation</strong>: Gradient magnitude indicates edge strength and boundaries in images.</p>
                <div id="chart3"></div>
            </div>
        `;
    }
    
    // Class Distribution
    if (data.class_distribution) {
        html += `
            <div class="visualization-section">
                <h3>Class Distribution</h3>
                <div id="chart4"></div>
            </div>
        `;
    }
    
    container.innerHTML = html;
    
    // Render charts
    if (data.has_mean_intensity) {
        renderBoxPlot('chart1', data.mean_intensity_data, 'Mean Intensity by Tumor Class', 'class', 'mean_intensity');
    }
    
    if (data.has_std_intensity) {
        renderViolinPlot('chart2', data.std_intensity_data, 'Intensity Standard Deviation by Tumor Class', 'class', 'std_intensity');
    }
    
    if (data.has_gradient_mean) {
        renderScatterPlot('chart3', data.gradient_data, 'Mean Intensity vs Gradient Mean', 'mean_intensity', 'gradient_mean');
    }
    
    if (data.class_distribution) {
        renderPieChart('chart4', data.class_distribution, 'Class Distribution');
    }
}

function renderBoxPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d[xCol]))];
    
    classes.forEach(cls => {
        const values = data.filter(d => d[xCol] === cls).map(d => d[yCol]);
        traces.push({
            y: values,
            type: 'box',
            name: cls,
            boxpoints: 'outliers'
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderViolinPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d[xCol]))];
    
    classes.forEach(cls => {
        const values = data.filter(d => d[xCol] === cls).map(d => d[yCol]);
        traces.push({
            y: values,
            type: 'violin',
            name: cls,
            box: { visible: true }
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderScatterPlot(containerId, data, title, xCol, yCol) {
    const traces = [];
    const classes = [...new Set(data.map(d => d.class))];
    
    classes.forEach(cls => {
        const filtered = data.filter(d => d.class === cls);
        traces.push({
            x: filtered.map(d => d[xCol]),
            y: filtered.map(d => d[yCol]),
            mode: 'markers',
            type: 'scatter',
            name: cls
        });
    });
    
    Plotly.newPlot(containerId, traces, {
        title: title,
        xaxis: { title: xCol },
        yaxis: { title: yCol },
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

function renderPieChart(containerId, distribution, title) {
    const trace = {
        labels: Object.keys(distribution),
        values: Object.values(distribution),
        type: 'pie'
    };
    
    Plotly.newPlot(containerId, [trace], {
        title: title,
        margin: { l: 50, r: 50, t: 50, b: 50 }
    }, {responsive: true});
}

// Save Files
async function saveFiles() {
    if (selectedFiles.length === 0) {
        showMessage('uploadResult', 'Please select files first.', 'error');
        return;
    }
    
    const classSelect = document.getElementById('classSelect').value;
    const resultDiv = document.getElementById('uploadResult');
    resultDiv.innerHTML = '<div class="loading">Saving files...</div>';
    
    try {
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        formData.append('class_name', classSelect);
        
        const response = await fetch(`${API_BASE}/retrain`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Failed to save files');
        }
        
        const result = await response.json();
        resultDiv.innerHTML = `
            <div class="result-message success">
                Successfully saved ${result.files_uploaded} file(s) to database.
            </div>
            <div class="result-message info">
                Files are ready for retraining. Go to 'Retrain Model' to trigger retraining.
            </div>
        `;
        
        // Clear selection
        selectedFiles = [];
        document.getElementById('multiImageInput').value = '';
        document.getElementById('fileCount').textContent = '';
        document.getElementById('uploadedFiles').innerHTML = '';
        document.getElementById('saveFilesBtn').disabled = true;
        
    } catch (error) {
        resultDiv.innerHTML = `<div class="result-message error">Error saving files: ${error.message}</div>`;
    }
}

// Retrain Stats
async function loadRetrainStats() {
    try {
        const response = await fetch(`${API_BASE}/database/stats`);
        const stats = await response.json();
        
        document.getElementById('totalUploaded').textContent = stats.total_uploaded_images || 0;
        document.getElementById('processed').textContent = stats.processed_images || 0;
        document.getElementById('trainingSessions').textContent = stats.total_training_sessions || 0;
        document.getElementById('completed').textContent = stats.completed_sessions || 0;
        
    } catch (error) {
        console.error('Error loading retrain stats:', error);
    }
}

async function loadRecentSessions() {
    try {
        const response = await fetch(`${API_BASE}/retrain/sessions`);
        const sessions = await response.json();
        
        const container = document.getElementById('recentSessions');
        
        if (!sessions || sessions.length === 0) {
            container.innerHTML = '<div class="result-message info">No training sessions yet.</div>';
            return;
        }
        
        let html = '<h3>Recent Training Sessions</h3><table class="sessions-table"><thead><tr>';
        html += '<th>ID</th><th>Timestamp</th><th>Status</th><th>Epochs</th><th>Accuracy</th><th>Images Used</th>';
        html += '</tr></thead><tbody>';
        
        sessions.forEach(session => {
            html += `<tr>
                <td>${session.id}</td>
                <td>${new Date(session.session_timestamp).toLocaleString()}</td>
                <td>${session.status}</td>
                <td>${session.epochs || '-'}</td>
                <td>${session.final_accuracy ? (session.final_accuracy * 100).toFixed(2) + '%' : '-'}</td>
                <td>${session.images_used || '-'}</td>
            </tr>`;
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
        
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

// Trigger Retraining
async function triggerRetraining() {
    const epochs = document.getElementById('epochs').value;
    const fineTuneEpochs = document.getElementById('fineTuneEpochs').value;
    const resultDiv = document.getElementById('retrainResult');
    
    resultDiv.innerHTML = '<div class="loading">Retraining model... This may take a while.</div>';
    
    try {
        const response = await fetch(`${API_BASE}/retrain/trigger`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                epochs: parseInt(epochs),
                fine_tune_epochs: parseInt(fineTuneEpochs)
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Retraining failed');
        }
        
        const result = await response.json();
        resultDiv.innerHTML = `
            <div class="result-message success">
                ✅ Model retraining completed successfully!
            </div>
            <div class="result-message info">
                The model has been updated. New predictions will use the retrained model.
            </div>
        `;
        
        // Reload stats
        loadRetrainStats();
        loadRecentSessions();
        checkModelStatus();
        
    } catch (error) {
        resultDiv.innerHTML = `<div class="result-message error">Retraining error: ${error.message}</div>`;
    }
}

// Uptime
async function loadUptime() {
    try {
        const response = await fetch(`${API_BASE}/uptime`);
        const data = await response.json();
        
        document.getElementById('uptimeValue').textContent = data.uptime_formatted || '-';
        document.getElementById('totalRequests').textContent = data.total_requests || 0;
        document.getElementById('modelStatusValue').textContent = data.model_loaded ? '🟢 Online' : '🔴 Offline';
        
        // Update model info
        const modelInfo = document.getElementById('modelInfo');
        if (data.model_loaded) {
            modelInfo.innerHTML = `
                <div class="info-item">✅ Model is loaded and ready</div>
                <div class="info-item">Model Path: models/brain_tumor_model.h5</div>
            `;
        } else {
            modelInfo.innerHTML = '<div class="info-item">❌ Model is not loaded</div>';
        }
        
        // Update performance info
        const performanceInfo = document.getElementById('performanceInfo');
        performanceInfo.innerHTML = `
            <div class="info-item">Request handling: Active</div>
            <div class="info-item">Session started: ${new Date(data.timestamp).toLocaleString()}</div>
        `;
        
    } catch (error) {
        console.error('Error loading uptime:', error);
    }
}

// Utility
function showMessage(containerId, message, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = `<div class="result-message ${type}">${message}</div>`;
}

