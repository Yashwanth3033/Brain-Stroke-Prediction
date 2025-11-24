// Algorithm selection
const algorithmCards = document.querySelectorAll('.algorithm-card');
let selectedAlgorithm = 'random-forest';

algorithmCards.forEach(card => {
    card.addEventListener('click', () => {
        algorithmCards.forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        selectedAlgorithm = card.dataset.algorithm;
    });
});

// Form handling
const form = document.getElementById('strokeForm');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Show loading
    loading.style.display = 'block';
    results.innerHTML = '<p style="text-align: center; color: #666;">Processing...</p>';

    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Get form data
    const formData = new FormData(form);
    const data = Object.fromEntries(formData);

    // Simulate predictions (replace with actual API calls)
    const predictions = simulatePredictions(data);

    // Hide loading
    loading.style.display = 'none';

    // Display results
    displayResults(predictions);
});

// Enhanced keyboard navigation
const inputs = document.querySelectorAll('#strokeForm input, #strokeForm select');
inputs.forEach((input, index) => {
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            event.preventDefault();
            const nextInput = inputs[index + 1];
            if (nextInput) {
                nextInput.focus();
            } else {
                document.querySelector('.predict-btn').focus();
            }
        }
    });
});

function simulatePredictions(data) {
    // Simulate different algorithm results
    const age = parseInt(data.age);
    const hasHypertension = data.hypertension === '1';
    const hasHeartDisease = data.heart_disease === '1';
    const glucose = parseFloat(data.avg_glucose_level);
    const bmi = parseFloat(data.bmi);

    // Base risk calculation
    let baseRisk = 0;
    if (age > 65) baseRisk += 30;
    else if (age > 45) baseRisk += 15;
    
    if (hasHypertension) baseRisk += 20;
    if (hasHeartDisease) baseRisk += 25;
    if (glucose > 140) baseRisk += 15;
    if (bmi > 30) baseRisk += 10;
    if (data.smoking_status === 'smokes') baseRisk += 15;

    return {
        'random-forest': {
            risk: Math.min(baseRisk + Math.random() * 10, 95),
            confidence: 85 + Math.random() * 10,
            algorithm: 'Random Forest Classifier'
        },
        'neural-network': {
            risk: Math.min(baseRisk + Math.random() * 15 - 5, 95),
            confidence: 88 + Math.random() * 8,
            algorithm: 'Deep Neural Network'
        },
        'svm': {
            risk: Math.min(baseRisk + Math.random() * 12 - 3, 95),
            confidence: 82 + Math.random() * 12,
            algorithm: 'Support Vector Machine'
        }
    };
}

function displayResults(predictions) {
    let html = '';

    Object.entries(predictions).forEach(([key, pred]) => {
        const riskLevel = pred.risk < 30 ? 'low-risk' : pred.risk < 70 ? 'medium-risk' : 'high-risk';
        const riskText = pred.risk < 30 ? 'LOW RISK' : pred.risk < 70 ? 'MODERATE RISK' : 'HIGH RISK';

        html += `
            <div class="result-card ${riskLevel}">
                <div class="algorithm-info">${pred.algorithm}</div>
                <div class="risk-indicator">
                    <div class="risk-level">${riskText}</div>
                    <div class="risk-percentage">${pred.risk.toFixed(1)}%</div>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pred.confidence}%"></div>
                </div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 5px;">
                    Confidence: ${pred.confidence.toFixed(1)}%
                </div>
            </div>
        `;
    });

    results.innerHTML = html;
}