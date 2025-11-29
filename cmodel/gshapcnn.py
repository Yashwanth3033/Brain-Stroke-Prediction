import pandas as pd
import numpy as np
import os
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ------------------------------------------------------------------
print("=" * 60)
print("📂 1. LOADING ARTIFACTS AND DATA")
print("=" * 60)

# --- Paths
MODEL_FILE = "cnn_model.keras"
SCALER_FILE = "scaler.pkl"
ENCODERS_FILE = "label_encoders.pkl"
IMPUTATION_FILE = "imputation_values.pkl"

DATASETS_PATH = "../datasets"
TRAIN_FILE = os.path.join(DATASETS_PATH, "train_dataset.xlsx")
TEST_FILE = os.path.join(DATASETS_PATH, "test_dataset.xlsx")

# --- Plotting Config
CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ["Class 0", "Class 1", "Class 2", "Class 3"]
SHAP_SAMPLES = 100
BACKGROUND_SAMPLES = 100
RESULTS_DIR = "cnn_shap_results"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. LOAD MODEL AND PREPROCESSORS
# ------------------------------------------------------------------
try:
    model = load_model(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    label_encoders = joblib.load(ENCODERS_FILE)
    imputation_values = joblib.load(IMPUTATION_FILE)
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    print("Please ensure model and .pkl files are in the correct directory.")
    exit()

print(f"✅ Model '{MODEL_FILE}' loaded.")
print(f"✅ Preprocessors (scaler, encoders) loaded.")
model.summary()


# ------------------------------------------------------------------
# 3. LOAD AND RE-PROCESS DATA (REQUIRED FOR SHAP)
# ------------------------------------------------------------------

def preprocess_data(df, is_train=True):
    """Applies the saved preprocessing steps to new data."""
    df.columns = df.columns.str.lower()
    df.replace("?", np.nan, inplace=True)

    # 1. Impute missing values
    for col, val in imputation_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 2. Encode categorical columns
    cat_cols = ["gender", "ever_married", "work_type",
                "residence_type", "smoking_status", "alcohol"]

    for col in cat_cols:
        if col in df.columns:
            le = label_encoders[col]

            # Handle unseen categories in test data
            test_col_str = df[col].astype(str)
            unseen_mask = ~test_col_str.isin(le.classes_)

            if unseen_mask.any():
                print(f"Warning: Found {unseen_mask.sum()} unseen categories in '{col}'. Replacing with mode.")
                mode_val_str = imputation_values[col]

                # Map unseen values to mode
                test_col_str[unseen_mask] = mode_val_str

            df[col] = le.transform(test_col_str)

    return df


print("\nProcessing training and test data...")
# Load raw data
train_data_raw = pd.read_excel(TRAIN_FILE)
test_data_raw = pd.read_excel(TEST_FILE)

# Preprocess
train_data = preprocess_data(train_data_raw.copy(), is_train=True)
test_data = preprocess_data(test_data_raw.copy(), is_train=False)

# 3. Split Features & Target
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]

# Get feature names BEFORE scaling
feature_names = X_train.columns.tolist()
print(f"✅ Found {len(feature_names)} features: {feature_names}")

# 4. Scale Data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data preprocessing complete.")

# ------------------------------------------------------------------
# 4. INITIALIZE SHAP EXPLAINER
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("🔥 2. INITIALIZING SHAP DEEPEXPLAINER")
print("=" * 60)

# SHAP needs a "background" dataset to calculate expected values.
background_data = shap.sample(X_train_scaled, BACKGROUND_SAMPLES)

# Use DeepExplainer for Keras models.
explainer = shap.DeepExplainer(model, background_data)

print(f"✅ DeepExplainer initialized with {BACKGROUND_SAMPLES} background samples.")

# ------------------------------------------------------------------
# 5. CALCULATE SHAP VALUES
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"⏳ 3. CALCULATING SHAP VALUES (for {SHAP_SAMPLES} test samples)")
print("=" * 60)

# Select a subset of the test data to explain
X_test_subset = shap.sample(X_test_scaled, SHAP_SAMPLES)
# Convert to DataFrame to pass feature names
X_test_subset_df = pd.DataFrame(X_test_subset, columns=feature_names)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test_subset)

print(f"✅ SHAP values calculated.")
print(f"   Type of shap_values: {type(shap_values)}")
print(f"   Shape of shap_values: {shap_values.shape}")

# ------------------------------------------------------------------
# 6. GENERATE SHAP PLOTS
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"📊 4. GENERATING SHAP PLOTS")
print("=" * 60)

# ---
# PLOT 1: Summary Bar Plot (Global Feature Importance)
# ---
print("Generating SHAP summary bar plot...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_subset_df,
    plot_type="bar",
    class_names=CLASS_NAMES,
    show=False
)
plt.title("SHAP Global Feature Importance (All Classes)")
plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_bar.png"), bbox_inches='tight')
plt.close()
print("✅ shap_summary_bar.png saved.")

# ---
# PLOT 2: Summary Beeswarm Plot
# ---
print("Generating SHAP summary beeswarm plot...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_subset_df,
    class_names=CLASS_NAMES,
    show=False
)
plt.title("SHAP Summary Plot (All Classes)")
plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_beeswarm.png"), bbox_inches='tight')
plt.close()
print("✅ shap_summary_beeswarm.png saved.")

# ---
# PLOT 3: Dependence Plots (Per-Class)
# ---
mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
top_feature_indices = np.argsort(mean_abs_shap)[-4:]
top_feature_names = [feature_names[i] for i in top_feature_indices]

print(f"Top 4 features to plot: {top_feature_names}")
print("Generating SHAP dependence plots...")

for class_idx in CLASSES:
    print(f"  ... for {CLASS_NAMES[class_idx]} (Index {class_idx})")

    shap_values_for_class = shap_values[:, :, class_idx]

    for feature_name in top_feature_names:
        plt.figure()
        shap.dependence_plot(
            feature_name,
            shap_values_for_class,
            X_test_subset_df,
            show=False
        )
        plt.title(f"Dependence Plot for '{feature_name}'\n(Contribution to {CLASS_NAMES[class_idx]})")
        fname = f"shap_dependence_{feature_name}_class_{class_idx}.png"
        plt.savefig(os.path.join(RESULTS_DIR, fname), bbox_inches='tight')
        plt.close()

print(f"✅ {len(CLASSES) * len(top_feature_names)} dependence plots saved.")
print("\n" + "=" * 60)
print(f"🎉 SHAP ANALYSIS COMPLETE! Results saved to '{RESULTS_DIR}'")
print("=" * 60)