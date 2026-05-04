const classMappings = {
    "imran_khan": "Imran Khan",
    "nawaz_sharif": "Nawaz Sharif",
    "shahbaz_sharif": "Shahbaz Sharif",
    "bilawal_bhutto": "Bilawal Bhutto Zardari",
    "asif_zardari": "Asif Ali Zardari",
    "mariyam_nawaz": "Maryam Nawaz",
    "maulana_fazlur_rehman": "Maulana Fazlur Rehman",
    "pervez_musharraf": "Pervez Musharraf",
    "siraj_ul_haq": "Siraj-ul-Haq",
    "fawad_chaudhry": "Fawad Chaudhry",
    "shah_mehmood_qureshi": "Shah Mehmood Qureshi",
    "sheikh_rasheed": "Sheikh Rasheed Ahmad",
    "jahangir_tareen": "Jahangir Tareen",
    "pervez_elahi": "Chaudhry Pervaiz Elahi",
    "asad_umar": "Asad Umar",
    "shaukat_tarin": "Shaukat Tarin"
};

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const classifyBtn = document.getElementById('classify-btn');
const btnText = document.getElementById('btn-text');
const spinner = document.getElementById('spinner');
const errorMsg = document.getElementById('error-message');
const errorText = document.getElementById('error-text');

const resultsPanel = document.getElementById('results-panel');
const primaryName = document.getElementById('primary-name');
const primaryConfidence = document.getElementById('primary-confidence');
const primaryBar = document.getElementById('primary-bar');
const top3Container = document.getElementById('top3-container');

let selectedFile = null;

// Format class name mapping
function formatClassName(id) {
    if (classMappings[id]) {
        return classMappings[id];
    }
    // Fallback: capitalize words
    return id.split('_')
             .map(word => word.charAt(0).toUpperCase() + word.slice(1))
             .join(' ');
}

// Handle file selection
function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
        showError('Please select a valid image file. (JPEG, PNG, etc)');
        return;
    }
    
    selectedFile = file;
    hideError();
    resultsPanel.classList.add('hidden');
    
    // Process image preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadPrompt.classList.add('hidden');
        previewContainer.classList.remove('hidden');
        dropZone.classList.add('has-image');
        classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Drag & Drop Events
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFile(e.target.files[0]);
    }
});

// UI states
function setButtonLoading(isLoading) {
    if (isLoading) {
        classifyBtn.disabled = true;
        btnText.textContent = "Analyzing...";
        spinner.classList.remove('hidden');
    } else {
        classifyBtn.disabled = false;
        btnText.textContent = "Classify Image";
        spinner.classList.add('hidden');
    }
}

function showError(msg) {
    errorText.textContent = msg;
    errorMsg.classList.remove('hidden');
}

function hideError() {
    errorMsg.classList.add('hidden');
}

// Render Results
function displayResults(data) {
    if (!data || !data.predicted_class) {
        showError("Invalid response from server.");
        return;
    }

    // Set Primary result
    primaryName.textContent = formatClassName(data.predicted_class);
    primaryConfidence.textContent = (data.confidence * 100).toFixed(1) + "%";
    
    // Reset bar width to 0 for animation, then apply real width
    primaryBar.style.width = '0%';
    
    // Clear old top 3
    top3Container.innerHTML = '';
    
    // Generate Top 3 HTML
    if (data.top3 && data.top3.length > 0) {
        data.top3.forEach((item, idx) => {
            const row = document.createElement('div');
            row.className = 'top3-item';
            
            row.innerHTML = `
                <div class="top3-header">
                    <span class="name">${formatClassName(item.class)}</span>
                    <span class="conf">${(item.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="top3-bar-container">
                    <div class="top3-fill" id="top3-bar-${idx}" style="width: 0%"></div>
                </div>
            `;
            top3Container.appendChild(row);
        });
    }

    resultsPanel.classList.remove('hidden');

    // Trigger animations after a tiny delay so the DOM updates
    setTimeout(() => {
        primaryBar.style.width = (data.confidence * 100) + "%";
        
        if (data.top3 && data.top3.length > 0) {
            data.top3.forEach((item, idx) => {
                const bar = document.getElementById(`top3-bar-${idx}`);
                if (bar) {
                    // Stagger animation slightly
                    setTimeout(() => {
                        bar.style.width = (item.confidence * 100) + "%";
                    }, idx * 100);
                }
            });
        }
    }, 50);
}

// API Call
classifyBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    hideError();
    setButtonLoading(true);
    resultsPanel.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }

        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
        
    } catch (err) {
        console.error(err);
        showError("Failed to connect to backend server. Make sure it is running on http://localhost:5000 and CORS is enabled.");
    } finally {
        setButtonLoading(false);
    }
});
