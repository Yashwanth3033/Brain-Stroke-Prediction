# Gradient Boosting Model - Test & gbtest_model.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

# -----------------------------
# 1️⃣ Paths & Load Data
# -----------------------------
datasets_path = "../datasets"
test_file = os.path.join(datasets_path, "test_dataset.xlsx")

test_data = pd.read_excel(test_file)
test_data.columns = test_data.columns.str.lower()
test_data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️⃣ Handle Missing Values
# -----------------------------
for col in test_data.columns:
    if test_data[col].dtype in [np.float64, np.int64]:
        median_val = test_data[col].median()
        test_data[col] = test_data[col].fillna(median_val)
    else:
        mode_val = test_data[col].mode()[0]
        test_data[col] = test_data[col].fillna(mode_val)

# -----------------------------
# 3️⃣ Encode Categorical Columns
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]

for col in cat_cols:
    if col in test_data.columns:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col].astype(str))

# -----------------------------
# 4️⃣ Split Features & Target
# -----------------------------
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]

# Store feature names for later use
feature_names = X_test.columns.tolist()

# -----------------------------
# 5️⃣ Load scaler & scale test data
# -----------------------------
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 6️⃣ Load GB model
# -----------------------------
gb_model = joblib.load("gb_model.pkl")

# -----------------------------
# 7️⃣ Predict & Evaluate
# -----------------------------
y_pred_gb = gb_model.predict(X_test_scaled)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb)
cm_gb = confusion_matrix(y_test, y_pred_gb)

# -----------------------------
# 8️⃣ Create Results Folder
# -----------------------------
results_folder = "gb_results"
os.makedirs(results_folder, exist_ok=True)

# -----------------------------
# 9️⃣ Show & Save Basic Results
# -----------------------------
print(f"Tested on {X_test_scaled.shape[0]} samples and {X_test_scaled.shape[1]} features.\n")
print(f"Gradient Boosting Test Accuracy: {accuracy_gb*100:.2f}%\n")
print("===== Gradient Boosting Classification Report =====")
print(report_gb)

# Save classification report to text file
with open(os.path.join(results_folder, "classification_report.txt"), "w") as f:
    f.write(f"Tested on {X_test_scaled.shape[0]} samples and {X_test_scaled.shape[1]} features.\n\n")
    f.write(f"Gradient Boosting Test Accuracy: {accuracy_gb*100:.2f}%\n\n")
    f.write("===== Gradient Boosting Classification Report =====\n")
    f.write(report_gb)

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Blues")
plt.title("Gradient Boosting - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Confusion matrix saved")

# -----------------------------
# 🔟 SHAP Analysis
# -----------------------------
print("\n📊 Generating SHAP explanations...")

# Create a SHAP explainer by passing the model's predict_proba method and the scaled test data
explainer = shap.Explainer(gb_model.predict_proba, X_test_scaled, feature_names=feature_names)
shap_values = explainer(X_test_scaled) # This returns a shap.Explanation object

# SHAP Summary Plot (bar) - this works directly with the new shap_values object
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, feature_names=feature_names, 
                  plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP summary bar plot saved")

# SHAP Summary Plot (beeswarm) - this also works directly
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP summary beeswarm plot saved")

# SHAP Waterfall plot for the first prediction's predicted class
# Get the predicted class for the first sample
predicted_class_for_sample_0 = y_pred_gb[0]
plt.figure(figsize=(10, 6))
# Select the SHAP values for the specific class that was predicted
shap.plots.waterfall(shap_values[0, :, predicted_class_for_sample_0], show=False)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "shap_waterfall_sample_0.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP waterfall plot saved")

# SHAP Force plot for first prediction (as matplotlib)
# This requires selecting the expected value and shap values for the predicted class
plt.figure(figsize=(20, 3))
shap.plots.force(shap_values.base_values[0][predicted_class_for_sample_0], 
                 shap_values.values[0, :, predicted_class_for_sample_0], 
                 shap_values.data[0],
                 feature_names=feature_names, matplotlib=True, show=False)
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "shap_force_sample_0.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ SHAP force plot saved")

# SHAP Dependence plots for top 3 features
# To get global feature importance, we take the mean absolute SHAP value across all classes
feature_importance = np.abs(shap_values.values).mean(axis=(0, 2))
top_features_idx = np.argsort(feature_importance)[-3:][::-1]

# Define which class to plot the dependence for (e.g., Class 1)
CLASS_TO_PLOT = 1 
print(f"\nGenerating SHAP dependence plots for Class {CLASS_TO_PLOT}...")

for idx in top_features_idx:
    feature_name = feature_names[idx]
    plt.figure(figsize=(8, 6))
    
    # We must pass the 2D SHAP values for a specific class, not the 3D array.
    # shap_values.values[:, :, CLASS_TO_PLOT] gives SHAP values for all samples and features for Class 1.
    shap.dependence_plot(
        ind=feature_name, 
        shap_values=shap_values.values[:, :, CLASS_TO_PLOT], 
        features=X_test_scaled, 
        feature_names=feature_names, 
        show=False
    )
    plt.title(f"SHAP Dependence for '{feature_name}' (for Class {CLASS_TO_PLOT})")
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"shap_dependence_{feature_name}_class_{CLASS_TO_PLOT}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
print(f"✓ SHAP dependence plots saved for top 3 features")

# -----------------------------
# 1️⃣1️⃣ LIME Analysis
# -----------------------------
print("\n🔍 Generating LIME explanations...")

# Create LIME explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_test_scaled,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Explain first 5 predictions
num_samples_to_explain = min(5, len(X_test_scaled))

for i in range(num_samples_to_explain):
    exp = lime_explainer.explain_instance(
        X_test_scaled[i],
        gb_model.predict_proba,
        num_features=len(feature_names)
    )
    
    # Save as figure
    fig = exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"lime_explanation_sample_{i}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
print(f"✓ LIME explanations saved for {num_samples_to_explain} samples")

# -----------------------------
# 1️⃣2️⃣ Feature Importance Plot
# -----------------------------
print("\n📈 Generating feature importance plot...")

feature_importance_gb = gb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance_gb
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Gradient Boosting - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(results_folder, "feature_importance.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved")

# -----------------------------
# 1️⃣3️⃣ Summary
# -----------------------------
print(f"\n{'='*60}")
print(f"✅ All results saved in '{results_folder}/' folder")
print(f"{'='*60}")
print("\nGenerated files:")
print("  1. classification_report.txt")
print("  2. confusion_matrix.png")
print("  3. shap_summary_bar.png")
print("  4. shap_summary_beeswarm.png")
print("  5. shap_waterfall_sample_0.png")
print("  6. shap_force_sample_0.png")
print(f"  7-9. shap_dependence_[feature].png (3 plots)")
print(f"  10-14. lime_explanation_sample_[0-4].png (5 plots)")
print("  15. feature_importance.png")
print(f"\nTotal: ~15 files")