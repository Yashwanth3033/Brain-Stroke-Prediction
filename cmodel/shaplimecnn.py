# Convolutional Neural Network Model - Test & shaplimecnn.py
import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
import shap
from lime import lime_tabular
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# -----------------------------
# 1️. Paths & Load Data
# -----------------------------
datasets_path = "../datasets"
test_file = os.path.join(datasets_path, "test_dataset.xlsx")

test_data = pd.read_excel(test_file)

# Normalize column names
test_data.columns = test_data.columns.str.lower()

# Replace "?" with NaN
test_data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️. Handle Missing Values
# -----------------------------
for col in test_data.columns:
    if test_data[col].dtype in [np.float64, np.int64]:
        median_val = test_data[col].median()
        test_data[col] = test_data[col].fillna(median_val)
    else:
        mode_val = test_data[col].mode()[0]
        test_data[col] = test_data[col].fillna(mode_val)

# -----------------------------
# 3️. Encode Categorical Columns
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]

for col in cat_cols:
    if col in test_data.columns:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col].astype(str))

# -----------------------------
# 4️. Split Features & Target
# -----------------------------
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]

# Store feature names for SHAP and LIME
feature_names = X_test.columns.tolist()

# -----------------------------
# 5.Load existing scaler & scale test data
# -----------------------------
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6️. Load trained CNN model
# -----------------------------
cnn_model = load_model("cnn_model.keras")
num_classes = cnn_model.output_shape[-1]

# -----------------------------
# 7️. Predict and evaluate
# -----------------------------
y_pred_cnn = np.argmax(cnn_model.predict(X_test_scaled), axis=1)
report_cnn_text = classification_report(y_test, y_pred_cnn)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)
accuracy = accuracy_score(y_test, y_pred_cnn)

# -----------------------------
# 8️. Create Results Folder
# -----------------------------
results_folder = "cnn_results"
os.makedirs(results_folder, exist_ok=True)
# -----------------------------
# 9.  SHAP Analysis
# -----------------------------
print("\n📊 Generating SHAP explanations...")


def model_predict(X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return cnn_model.predict(X, verbose=0)


# Use a smaller subset
shap_sample_size = min(100, X_test_scaled.shape[0])
X_shap_sample = X_test_scaled[:shap_sample_size]

print(f"Initializing SHAP explainer with {shap_sample_size} samples...")
background = shap.kmeans(X_test_scaled, 50).data
explainer = shap.KernelExplainer(model_predict, background)

print("Calculating SHAP values (this may take a few minutes)...")
shap_values = explainer.shap_values(X_shap_sample)

# Debug shape information
print(f"SHAP values shape: {[np.array(sv).shape for sv in shap_values]}")
print(f"Features shape: {X_shap_sample.shape}")

# SHAP Summary Plot (bar)
print("Generating SHAP summary bar plot...")
try:
    all_shap_values = np.array(shap_values)
    mean_abs_shap = np.abs(all_shap_values).mean(axis=(0, 2))

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance')

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance (Mean Absolute SHAP Values)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
    print("✓ SHAP summary bar plot saved")
except Exception as e:
    print(f"⚠️ Could not generate summary bar plot: {str(e)}")
finally:
    plt.close()

# SHAP Summary Plot (beeswarm)
print("Generating SHAP summary beeswarm plot...")
try:
    plt.figure(figsize=(10, 8))
    class_idx = 0
    class_shap_values = shap_values[class_idx]

    shap_values_reshaped = np.zeros((X_shap_sample.shape[0], X_shap_sample.shape[1]))
    for i in range(X_shap_sample.shape[1]):
        shap_values_reshaped[:, i] = class_shap_values[i].mean()

    shap.summary_plot(
        shap_values_reshaped,
        X_shap_sample,
        feature_names=feature_names,
        show=False,
        plot_type="dot"
    )
    plt.title(f"SHAP Impact Distribution (Class {class_idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
    print("✓ SHAP summary beeswarm plot saved")
except Exception as e:
    print(f"⚠️ Could not generate beeswarm plot: {str(e)}")
finally:
    plt.close()

# SHAP Waterfall Plot
print("Generating SHAP waterfall plot...")
try:
    plt.figure(figsize=(10, 8))
    pred_probs = model_predict(X_shap_sample[0:1])
    pred_class = np.argmax(pred_probs[0])

    waterfall_exp = shap.Explanation(
        values=shap_values[pred_class][:, 0],
        base_values=explainer.expected_value[pred_class],
        data=X_shap_sample[0],
        feature_names=feature_names
    )

    shap.plots.waterfall(waterfall_exp, show=False)
    plt.title(f"SHAP Waterfall Plot (Sample 0, Predicted Class: {pred_class})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "shap_waterfall_sample_0.png"), dpi=300, bbox_inches='tight')
    print("✓ SHAP waterfall plot saved")
except Exception as e:
    print(f"⚠️ Could not generate waterfall plot: {str(e)}")
finally:
    plt.close()

# SHAP Force Plot for first prediction
print("Generating SHAP force plot...")
try:
    plt.figure(figsize=(20, 3))
    pred_probs = model_predict(X_shap_sample[0:1])
    pred_class = np.argmax(pred_probs[0])

    shap.plots.force(
        explainer.expected_value[pred_class],
        shap_values[pred_class][:, 0],
        X_shap_sample[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot (Sample 0, Predicted Class: {pred_class})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "shap_force_sample_0.png"), dpi=300, bbox_inches='tight')
    print("✓ SHAP force plot saved")
except Exception as e:
    print(f"⚠️ Could not generate force plot: {str(e)}")
finally:
    plt.close()

# SHAP Dependence plots for top 3 features
print("Generating SHAP dependence plots...")
try:
    all_shap_values = np.array(shap_values)
    feature_importance_values = np.abs(all_shap_values).mean(axis=(0, 2))
    top_features_idx = np.argsort(feature_importance_values)[-3:][::-1]

    CLASS_TO_PLOT = 1
    print(f"Generating SHAP dependence plots for Class {CLASS_TO_PLOT}...")

    for idx in top_features_idx:
        feature_name = feature_names[idx]
        plt.figure(figsize=(8, 6))

        class_shap_transposed = shap_values[CLASS_TO_PLOT].T

        shap.dependence_plot(
            ind=feature_name,
            shap_values=class_shap_transposed,
            features=X_shap_sample,
            feature_names=feature_names,
            show=False
        )
        plt.title(f"SHAP Dependence for '{feature_name}' (Class {CLASS_TO_PLOT})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"shap_dependence_{feature_name}_class_{CLASS_TO_PLOT}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
    print(f"✓ SHAP dependence plots saved for top 3 features")
except Exception as e:
    print(f"⚠️ Could not generate dependence plots: {str(e)}")

# -----------------------------
# 10.LIME Analysis
# -----------------------------
print("\n🔍 Generating LIME explanations...")


def predict_fn(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return model_predict(x)


# Create LIME explainer with proper training data
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_test_scaled,
    feature_names=feature_names,
    class_names=[f'Class {i}' for i in range(num_classes)],
    mode='classification',
    random_state=42
)

# Explain first 5 predictions
num_samples_to_explain = min(5, len(X_test_scaled))

for i in range(num_samples_to_explain):
    try:
        exp = lime_explainer.explain_instance(
            X_test_scaled[i],
            predict_fn,
            num_features=10,
            top_labels=1,
            num_samples=500
        )

        pred_class = np.argmax(predict_fn(X_test_scaled[i]))

        plt.figure(figsize=(12, 6))
        exp.as_pyplot_figure(label=pred_class)
        plt.title(f"LIME Explanation - Sample {i} (Predicted: Class {pred_class})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_folder, f"lime_explanation_sample_{i}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ LIME explanation saved for sample {i}")
    except Exception as e:
        print(f"⚠️ Could not generate LIME explanation for sample {i}: {str(e)}")

print(f"✓ LIME explanations saved for {num_samples_to_explain} samples")

# -----------------------------
# 11. Summary
# -----------------------------
print(f"\n{'=' * 60}")
print(f"✅ All results saved in '{results_folder}/' folder")
print(f"{'=' * 60}")
print("\nGenerated files:")
print("  1. classification_report.txt")
print("  2. confusion_matrix.png")
print("  3. shap_summary_bar.png")
print("  4. shap_summary_beeswarm.png")
print("  5. shap_waterfall_sample_0.png")
print("  6. shap_force_sample_0.png")
print(f"  7-9. shap_dependence_[feature]_class_1.png (3 plots)")
print(f"  10-14. lime_explanation_sample_[0-4].png (5 plots)")
print(f"\nTotal: ~14 files")
print(f"\n{'=' * 60}")
print("🎯 CNN Explainability Analysis Complete!")
print(f"{'=' * 60}")