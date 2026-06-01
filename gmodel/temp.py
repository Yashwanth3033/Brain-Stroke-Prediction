import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. SETUP: Define feature names
# ---------------------------------------------------------
feature_names = [
    "avg_glucose_level", "diabetes", "cholesterol", "age",
    "work_type", "whitebloodcellcount", "bpdastolic", "bpsystolic",
    "bmi", "alcohol", "smoking_status", "redbloodcellcount",
    "residence_type", "gender", "family_history", "hypertension",
    "ever_married", "heart_disease"
]
n_features = len(feature_names)
n_samples = 100
np.random.seed(123) # New seed for different random patterns

# ---------------------------------------------------------
# 2. CREATE MORE UNEVEN SYNTHETIC DATA
# ---------------------------------------------------------
# A) Create Feature Data (X) - Same as before
X_test_synthetic = pd.DataFrame(np.zeros((n_samples, n_features)), columns=feature_names)
for col in feature_names:
    if col in ['diabetes', 'gender', 'hypertension', 'heart_disease', 'ever_married']:
        X_test_synthetic[col] = np.random.randint(0, 2, n_samples) # Binary
    else:
        # Mix of skewed distributions for continuous features
        dist_type = np.random.choice(['exponential', 'lognormal', 'uniform'])
        if dist_type == 'exponential':
            X_test_synthetic[col] = np.random.exponential(scale=1.0, size=n_samples)
        elif dist_type == 'lognormal':
            X_test_synthetic[col] = np.random.lognormal(mean=0, sigma=0.5, size=n_samples)
        else:
            X_test_synthetic[col] = np.random.rand(n_samples)

# B) Create Highly Uneven & Correlated SHAP values
X_norm = (X_test_synthetic - X_test_synthetic.min()) / (X_test_synthetic.max() - X_test_synthetic.min() + 1e-5)

# Decide direction of impact (positive or negative) for each feature
signs = np.random.choice([-1, 1], size=n_features)

# 1. Introduce Non-linearity: Use power to make tails longer and bunches tighter
power_factors = np.random.choice([1.5, 2.0, 3.0], size=n_features)
base_shap = (X_norm.values ** power_factors) * signs

# 2. Add Asymmetric Noise: Use exponential noise that follows the feature's sign
# This creates a "fanning out" effect in the direction of impact
noise_scale = np.random.rand(n_features) * 0.5 + 0.1
asymmetric_noise = np.random.exponential(scale=noise_scale, size=(n_samples, n_features)) * signs
shap_values_synthetic = base_shap + asymmetric_noise

# 3. Add a random bias to each feature so they aren't all centered on zero
# Some features will now be mostly positive or mostly negative
feature_bias = np.random.randn(n_features) * 0.5
shap_values_synthetic += feature_bias

# 4. Apply importance scaling (first features get wider spread)
importance_weights = np.linspace(3.0, 0.1, n_features)
shap_values_synthetic = shap_values_synthetic * importance_weights

# 5. Add a few random outliers for extra irregularity
num_outliers = int(n_samples * n_features * 0.01) # 1% of points
outlier_indices = (np.random.randint(0, n_samples, num_outliers), np.random.randint(0, n_features, num_outliers))
shap_values_synthetic[outlier_indices] *= np.random.choice([-2.5, 2.5], size=num_outliers)

# ---------------------------------------------------------
# 3. GENERATE THE PLOT
# ---------------------------------------------------------
print("Generating highly uneven SHAP Beeswarm plot...")
plt.figure(figsize=(10, 8))

shap.summary_plot(
    shap_values_synthetic,
    X_test_synthetic,
    plot_type="dot",
    max_display=n_features,
    show=False
)

plt.title("SHAP Summary Plot - Synthetic Data with High Irregularity", fontsize=14, pad=20)
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()
plt.show()