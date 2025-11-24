# =========================================================
# 🔬 COMPREHENSIVE MODEL EVALUATION (gb_model)
# =========================================================
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
import lime
import lime.lime_tabular
import shap
import warnings
import shutil

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =========================================================
# 📁 PATH CONFIGURATION
# =========================================================
DATASET_DIR = "../datasets"
MODEL_DIR = "." # Assuming models are in root dir as per last script
RESULTS_DIR = "gb_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# add evaluation subfolder (all results will be moved there at the end)
EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

TEST_DATA_PATH = os.path.join(DATASET_DIR, "test_dataset.xlsx")
MODEL_PATH = os.path.join(MODEL_DIR, "gb_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# =========================================================
# 📊 LOAD DATA AND MODEL
# =========================================================
print("=" * 70)
print("📂 LOADING DATA AND MODEL ARTIFACTS")
print("=" * 70)

try:
    # Load test data
    test_data = pd.read_excel(TEST_DATA_PATH)
    test_data.columns = test_data.columns.str.lower()
    test_data.replace("?", np.nan, inplace=True)
    print(f"✅ Test data loaded from: {TEST_DATA_PATH} ({test_data.shape})")

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ Scaler loaded from: {SCALER_PATH}")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("👉 Please ensure test_dataset.xlsx, gb_model.pkl, and scaler.pkl are in the correct paths!")
    exit()

# -----------------------------
# Handle Missing Values (from previous script)
# -----------------------------
for col in test_data.columns:
    if test_data[col].dtype in [np.float64, np.int64]:
        median_val = test_data[col].median()
        test_data[col] = test_data[col].fillna(median_val)
    else:
        # Check if mode() returns an empty series
        if not test_data[col].mode().empty:
            mode_val = test_data[col].mode()[0]
            test_data[col] = test_data[col].fillna(mode_val)
        else:
            # Fallback if mode is empty (e.g., all NaN)
            test_data[col] = test_data[col].fillna("Unknown")

# -----------------------------
# Encode Categorical Columns (from previous script)
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]
encoders = {} # Store encoders if needed, though not strictly required for test-only

for col in cat_cols:
    if col in test_data.columns:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col].astype(str))
        encoders[col] = le

# -----------------------------
# Split Features & Target
# -----------------------------
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]
print(f"✅ Data split: X_test ({X_test.shape}), y_test ({y_test.shape})")

# -----------------------------
# Define Metadata
# -----------------------------
feature_names = X_test.columns.tolist()
class_labels = np.sort(y_test.unique())
n_classes = len(class_labels)

# Define class names (assuming 0=No Stroke, 1=Stroke based on project context)
if n_classes == 2:
    class_names = ['No Stroke', 'Stroke']
else:
    class_names = [f'Class {i}' for i in class_labels]

print(f"\n📊 Model Information:")
print(f"   Model Type: Gradient Boosting Classifier")
print(f"   Features: {len(feature_names)}")
print(f"   Classes: {n_classes} ({', '.join(class_names)})")

# -----------------------------
# Scale Test Data
# -----------------------------
X_test_scaled = scaler.transform(X_test)
# Create a DataFrame for easier sampling later
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
print(f"✅ X_test data scaled successfully.")


# =========================================================
# 🎯 GENERATE PREDICTIONS
# =========================================================
print("\n" + "=" * 70)
print("🎯 GENERATING PREDICTIONS (using scaled data)")
print("=" * 70)

# *** IMPORTANT: Use X_test_scaled for the gb_model ***
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

print(f"✅ Predictions generated for {len(y_pred)} samples")

# =========================================================
# 📊 PERFORMANCE METRICS
# =========================================================
print("\n" + "=" * 70)
print("📊 PERFORMANCE METRICS")
print("=" * 70)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

print("\n" + "=" * 40)
print("WEIGHTED METRICS (Accounts for Class Imbalance)")
print("=" * 40)
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision_weighted:.4f} ({precision_weighted * 100:.2f}%)")
print(f"Recall:    {recall_weighted:.4f} ({recall_weighted * 100:.2f}%)")
print(f"F1-Score:  {f1_weighted:.4f} ({f1_weighted * 100:.2f}%)")

print("\n" + "=" * 40)
print("MACRO METRICS (Equal Weight Per Class)")
print("=" * 40)
print(f"Precision: {precision_macro:.4f} ({precision_macro * 100:.2f}%)")
print(f"Recall:    {recall_macro:.4f} ({recall_macro * 100:.2f}%)")
print(f"F1-Score:  {f1_macro:.4f} ({f1_macro * 100:.2f}%)")

# Save metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)',
               'F1-Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    'Value': [accuracy, precision_weighted, recall_weighted, f1_weighted,
              precision_macro, recall_macro, f1_macro]
})
metrics_df.to_csv(os.path.join(EVAL_DIR, 'performance_metrics.csv'), index=False)

# =========================================================
# 📋 DETAILED CLASSIFICATION REPORT
# =========================================================
print("\n" + "=" * 70)
print("📋 DETAILED CLASSIFICATION REPORT")
print("=" * 70)

# Get labels that actually appear in test set
present_labels = sorted(y_test.unique())
present_class_names = [class_names[i] for i in present_labels]

report = classification_report(
    y_test,
    y_pred,
    labels=present_labels,
    target_names=present_class_names,
    digits=4
)
print("\n", report)

# Save report
report_dict = classification_report(
    y_test,
    y_pred,
    labels=present_labels,
    target_names=present_class_names,
    output_dict=True,
    digits=4
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(EVAL_DIR, 'classification_report.csv'))
print(f"\n💾 Classification report saved: {EVAL_DIR}/classification_report.csv")


# =========================================================
# 📈 CONFUSION MATRIX
# =========================================================
print("\n" + "=" * 70)
print("📈 CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(y_test, y_pred, labels=present_labels)

print("\nRaw Confusion Matrix:")
print(cm)

# Calculate per-class metrics from confusion matrix
print("\nPer-Class Analysis:")
for idx, (label, name) in enumerate(zip(present_labels, present_class_names)):
    tp = cm[idx, idx]
    fp = cm[:, idx].sum() - tp
    fn = cm[idx, :].sum() - tp
    tn = cm.sum() - (tp + fp + fn)

    print(f"\n{name} (Class {label}):")
    print(f"   True Positives (TP):  {tp}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Negatives (TN):  {tn}")

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=present_class_names,
    yticklabels=present_class_names,    # <-- fixed: was 'ytickslabels'
    cbar_kws={'label': 'Count'},
    square=True
)
plt.title('Confusion Matrix - Gradient Boosting', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"\n💾 Confusion matrix saved: {EVAL_DIR}/confusion_matrix.png")
plt.close()

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=present_class_names,
    yticklabels=present_class_names,    # <-- fixed: was 'ytickslabels'
    cbar_kws={'label': 'Percentage'},
    square=True
)
plt.title('Normalized Confusion Matrix - Gradient Boosting', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
plt.close()

# =========================================================
# 📊 ROC CURVE AND AUC
# =========================================================
print("\n" + "=" * 70)
print("📊 ROC CURVE AND AUC SCORE")
print("=" * 70)

if n_classes == 2:
    # Binary classification
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✅ ROC AUC Score: {roc_auc:.4f}")
    plt.close()
else:
    # Multiclass classification
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted', labels=present_labels)
        print(f"✅ ROC AUC Score (Weighted): {roc_auc:.4f}")
    except ValueError as e:
        print(f"⚠️  Could not calculate AUC: {e}")

print(f"💾 ROC curve saved: {EVAL_DIR}/roc_curve.png")

# =========================================================
# 🔍 LIME INTERPRETATION
# =========================================================
print("\n" + "=" * 70)
print("🔍 LIME - LOCAL INTERPRETABLE MODEL-AGNOSTIC EXPLANATIONS")
print("=" * 70)

# Initialize LIME
# *** IMPORTANT: Use X_test_scaled for the training_data background ***
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_test_scaled, # Use scaled data as background
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    random_state=42
)

# Generate explanations for sample instances
print("\n📊 Generating LIME explanations...")
num_samples = min(5, len(X_test_scaled))

for i in range(num_samples):
    print(f"\n--- Instance {i + 1} ---")
    print(f"   True Label: {y_test.iloc[i]} ({class_names[y_test.iloc[i]]})")
    print(f"   Predicted: {y_pred[i]} ({class_names[y_pred[i]]})")
    print(f"   Probabilities: {y_pred_proba[i]}")

    # Generate explanation
    # *** IMPORTANT: Explain the scaled instance ***
    exp = explainer_lime.explain_instance(
        X_test_scaled[i], # Use scaled instance
        model.predict_proba,
        num_features=10,
        top_labels=n_classes
    )

    # Save as HTML
    exp.save_to_file(os.path.join(EVAL_DIR, f'lime_explanation_{i + 1}.html'))
    
    # Save as figure
    fig = exp.as_pyplot_figure(label=y_pred[i])
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"lime_explanation_plot_{i+1}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print top features
    print(f"\n   Top 10 Feature Contributions (for predicted class {y_pred[i]}):")
    for feature, weight in exp.as_list(label=y_pred[i])[:10]:
        print(f"         {feature}: {weight:+.4f}")

print(f"\n💾 LIME explanations saved: {EVAL_DIR}/lime_explanation_*.html/.png")

# =========================================================
# 🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS (fixed for multi-class)
# =========================================================
print("\n" + "=" * 70)
print("🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS")
print("=" * 70)

print("\n📊 Computing SHAP values (this may take a moment)...")

# use a sample for speed & stability
sample_size = min(100, len(X_test_scaled_df))
X_test_sample = X_test_scaled_df.iloc[:sample_size]
y_test_sample = y_test.iloc[:sample_size]

# Try an explainer that accepts a callable (predict_proba) to support multi-class
background = shap.sample(X_test_scaled_df, min(200, len(X_test_scaled_df)), random_state=0)

try:
    # Explainer will choose a fast tree-aware explainer if appropriate, otherwise a suitable fallback.
    explainer_shap = shap.Explainer(model.predict_proba, background, feature_names=feature_names)
    shap_expl = explainer_shap(X_test_sample)  # this returns a shap.Explanation object
    shap_is_explanation = True
    print("✓ shap.Explainer computed (Explanation object).")
except Exception as e:
    print("⚠️ shap.Explainer failed, falling back to KernelExplainer:", e)
    ke_bg = shap.sample(X_test_scaled_df, min(50, len(X_test_scaled_df)), random_state=0)
    explainer_shap = shap.KernelExplainer(model.predict_proba, ke_bg)
    # KernelExplainer returns a legacy list-of-arrays via .shap_values(...)
    shap_vals_legacy = explainer_shap.shap_values(X_test_sample.values)
    shap_is_explanation = False
    print("✓ KernelExplainer computed (legacy shap values).")

# Ensure a consistent variable name for later use
if shap_is_explanation:
    shap_values = shap_expl          # Explanation object
else:
    shap_values = shap_vals_legacy   # legacy list

print(f"✅ SHAP values ready for {sample_size} samples")

# -----------------------------
# Plotting SHAP visualizations (handle both Explanation and legacy list)
# -----------------------------
print("\n📊 Generating SHAP visualizations...")

plt.figure(figsize=(12, 10))
if shap_is_explanation:
    # shap_expl.values shape: (samples, features) for single-output or (samples, features, classes) for multi-class
    # Use the Explanation object directly in summary_plot
    shap.summary_plot(shap_expl, X_test_sample, feature_names=feature_names, show=False, max_display=20)
else:
    # legacy: shap_vals_legacy is list[class][samples, features]; choose class index to visualize (e.g., positive class if available)
    cls_idx = 1 if len(shap_vals_legacy) > 1 else 0
    shap.summary_plot(shap_vals_legacy[cls_idx], X_test_sample, feature_names=feature_names, show=False, max_display=20)

plt.title('SHAP Summary Plot - Feature Impact', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'shap_summary.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Summary plot (beeswarm) saved")

plt.figure(figsize=(12, 10))
if shap_is_explanation:
    shap.summary_plot(shap_expl, X_test_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
else:
    cls_idx = 1 if len(shap_vals_legacy) > 1 else 0
    shap.summary_plot(shap_vals_legacy[cls_idx], X_test_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)

plt.title('SHAP Bar Plot - Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'shap_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Bar plot saved")

# Waterfall plots (per-instance) - create robust fallback if shapes differ
print("\n📊 Generating individual SHAP waterfall plots...")
num_waterfalls = min(3, len(X_test_sample))
for i in range(num_waterfalls):
    try:
        plt.figure(figsize=(12, 8))
        if shap_is_explanation:
            # If Explanation has per-class values, select predicted class index if present
            vals = shap_expl.values
            if vals.ndim == 3:
                pred_class = y_pred[i]
                vals_i = vals[i, :, pred_class]
                base_val = shap_expl.base_values[i][pred_class] if np.shape(shap_expl.base_values) != () else shap_expl.base_values[pred_class]
            else:
                vals_i = vals[i]
                base_val = shap_expl.base_values[i] if np.shape(shap_expl.base_values) != () else shap_expl.base_values
            exp_obj = shap.Explanation(values=vals_i, base_values=base_val, data=X_test_sample.iloc[i].values, feature_names=feature_names)
            shap.waterfall_plot(exp_obj, show=False, max_display=15)
        else:
            pred_class = y_pred[i]
            vals_i = shap_vals_legacy[pred_class][i]
            base_val = explainer_shap.expected_value[pred_class] if hasattr(explainer_shap, "expected_value") else None
            exp_obj = shap.Explanation(values=vals_i, base_values=base_val, data=X_test_sample.iloc[i].values, feature_names=feature_names)
            shap.waterfall_plot(exp_obj, show=False, max_display=15)

        plt.title(f'SHAP Waterfall - Instance {i + 1}\nTrue: {class_names[y_test_sample.iloc[i]]} | Pred: {class_names[y_pred[i]]}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'shap_waterfall_{i + 1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Waterfall plot {i + 1} saved")
    except Exception as e:
        print(f"   ⚠️  Could not generate waterfall plot {i + 1}: {e}")
        plt.close()
        continue

# prepare shap_values variable for later importance computation
if shap_is_explanation:
    shap_values_for_import = shap_expl
else:
    shap_values_for_import = shap_vals_legacy

print(f"\n💾 SHAP visualizations saved: {EVAL_DIR}/shap_*.png")

# =========================================================
# 📊 FEATURE IMPORTANCE COMPARISON
# =========================================================
print("\n" + "=" * 70)
print("📊 FEATURE IMPORTANCE COMPARISON")
print("=" * 70)

# Model's built-in feature importance
model_importance = pd.DataFrame({
    'Feature': feature_names,
    'Model_Importance': model.feature_importances_
}).sort_values('Model_Importance', ascending=False)

# Calculate Mean absolute SHAP values (robust for both Explanation and legacy)
if shap_is_explanation:
    vals = shap_expl.values  # shape: (samples, features) or (samples, features, classes)
    if vals.ndim == 3:
        # average over samples and classes -> (features,)
        mean_shap = np.mean(np.abs(vals), axis=(0, 2))
    else:
        mean_shap = np.mean(np.abs(vals), axis=0)
else:
    # shap_vals_legacy: list[class][samples, features] OR array
    if isinstance(shap_vals_legacy, list):
        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals_legacy], axis=0)
    else:
        mean_shap = np.abs(shap_vals_legacy).mean(axis=0)

shap_importance = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Importance': mean_shap
}).sort_values('SHAP_Importance', ascending=False)


# Merge both importance measures
importance_comparison = model_importance.merge(shap_importance, on='Feature')
importance_comparison['Model_Rank'] = importance_comparison['Model_Importance'].rank(ascending=False)
importance_comparison['SHAP_Rank'] = importance_comparison['SHAP_Importance'].rank(ascending=False)

# Save comparison
importance_comparison.to_csv(
    os.path.join(EVAL_DIR, 'feature_importance_comparison.csv'),
    index=False
)

print("\n📊 Top 15 Features (Comparison):")
print(importance_comparison.head(15).to_string(index=False))

print(f"\n💾 Feature importance saved: {EVAL_DIR}/feature_importance_comparison.csv")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Model importance
top_model = model_importance.head(15).sort_values('Model_Importance')
axes[0].barh(range(len(top_model)), top_model['Model_Importance'])
axes[0].set_yticks(range(len(top_model)))
axes[0].set_yticklabels(top_model['Feature'])
axes[0].set_xlabel('Importance', fontsize=11)
axes[0].set_title('Gradient Boosting Feature Importance', fontsize=13, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)

# SHAP importance
top_shap = shap_importance.head(15).sort_values('SHAP_Importance')
axes[1].barh(range(len(top_shap)), top_shap['SHAP_Importance'], color='coral')
axes[1].set_yticks(range(len(top_shap)))
axes[1].set_yticklabels(top_shap['Feature'])
axes[1].set_xlabel('Mean |SHAP Value|', fontsize=11)
axes[1].set_title('SHAP Feature Importance', fontsize=13, fontweight='bold')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'feature_importance_comparison.png'),
            dpi=300, bbox_inches='tight')
print(f"💾 Comparison plot saved: {EVAL_DIR}/feature_importance_comparison.png")
plt.close()

# =========================================================
# 📝 GENERATE EVALUATION REPORT
# =========================================================
print("\n" + "=" * 70)
print("📝 GENERATING EVALUATION REPORT")
print("=" * 70)

report_content = f"""
{'=' * 70}
MODEL EVALUATION REPORT
{'=' * 70}

📅 Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Model Type: Gradient Boosting Classifier
🎯 Task: {'Binary' if n_classes == 2 else 'Multi-class'} Classification

{'=' * 70}
DATASET INFORMATION
{'=' * 70}

Test Samples: {len(X_test)}
Number of Features: {len(feature_names)}
Number of Classes: {n_classes}

Class Distribution (Test Set):
{y_test.value_counts().sort_index().to_string()}

{'=' * 70}
PERFORMANCE METRICS
{'=' * 70}

OVERALL ACCURACY: {accuracy:.4f} ({accuracy * 100:.2f}%)

WEIGHTED METRICS (Accounts for Class Imbalance):
  - Precision: {precision_weighted:.4f}
  - Recall:    {recall_weighted:.4f}
  - F1-Score:  {f1_weighted:.4f}

MACRO METRICS (Equal Weight Per Class):
  - Precision: {precision_macro:.4f}
  - Recall:    {recall_macro:.4f}
  - F1-Score:  {f1_macro:.4f}

{'=' * 70}
PER-CLASS PERFORMANCE
{'=' * 70}

{classification_report(y_test, y_pred, labels=present_labels,
                       target_names=present_class_names, digits=4)}

{'=' * 70}
CONFUSION MATRIX (Raw Count)
{'=' * 70}

{cm}

{'=' * 70}
TOP 10 MOST IMPORTANT FEATURES
{'=' * 70}

Gradient Boosting Importance (model.feature_importances_):
{model_importance.head(10)[['Feature', 'Model_Importance']].to_string(index=False)}

SHAP Importance (Mean |SHAP Value|):
{shap_importance.head(10)[['Feature', 'SHAP_Importance']].to_string(index=False)}

{'=' * 70}
INTERPRETABILITY METHODS
{'=' * 70}

✅ LIME (Local Interpretable Model-agnostic Explanations)
   - Generated explanations for {num_samples} instances
   - Files: lime_explanation_*.html / .png

✅ SHAP (SHapley Additive exPlanations)
   - Computed SHAP values for {sample_size} instances
   - Generated summary, bar, and waterfall plots
   - Files: shap_*.png

{'=' * 70}
SAVED ARTIFACTS
{'=' * 70}

📁 {RESULTS_DIR}/
   ├── performance_metrics.csv
   ├── classification_report.csv
   ├── confusion_matrix.png
   ├── confusion_matrix_normalized.png
   ├── roc_curve.png
   ├── lime_explanation_*.html ({num_samples} files)
   ├── lime_explanation_plot_*.png ({num_samples} files)
   ├── shap_summary.png
   ├── shap_bar.png
   ├── shap_waterfall_*.png (3 files)
   ├── feature_importance_comparison.csv
   ├── feature_importance_comparison.png
   └── evaluation_report.txt (this file)

{'=' * 70}
CONCLUSIONS
{'=' * 70}

Model achieves {accuracy * 100:.2f}% accuracy on test set.
Weighted F1-Score: {f1_weighted:.4f}
Macro F1-Score: {f1_macro:.4f}

The model's predictions can be explained using:
- LIME for instance-level interpretability
- SHAP for both global and local feature importance

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

# Save report
with open(os.path.join(EVAL_DIR, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"💾 Evaluation report saved: {EVAL_DIR}/evaluation_report.txt")

# =========================================================
# 📊 FINAL SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("✅ COMPREHENSIVE EVALUATION COMPLETE")
print("=" * 70)

print(f"""
🎯 Test Set Performance:
   • Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)
   • Precision:       {precision_weighted:.4f}
   • Recall:          {recall_weighted:.4f}
   • F1-Score:        {f1_weighted:.4f}

📊 Evaluation Metrics:
   • Confusion Matrix:      ✅
   • Classification Report: ✅
   • ROC Curve:             ✅
   • Feature Importance:    ✅

🔍 Interpretability:
   • LIME Explanations:     ✅ ({num_samples} instances)
   • SHAP Analysis:         ✅ (Global + Local)
   • Waterfall Plots:       ✅ (3 instances)

📁 All Results Saved To: {RESULTS_DIR}/
   Total Files Generated: {len(os.listdir(RESULTS_DIR))}

💡 Next Steps:
   1. Review evaluation_report.txt for detailed analysis
   2. Open lime_explanation_*.html in browser for interactive explanations
   3. Examine SHAP plots for feature importance insights
   4. Compare model importance vs SHAP importance
   5. Analyze confusion matrix for error patterns
""")

print("=" * 70)
print("🎉 EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)

# after all files have been generated (before final summary print), move results into evaluation folder
print(f"\nMoving all generated result files into: {EVAL_DIR}")
for fname in os.listdir(RESULTS_DIR):
    src = os.path.join(RESULTS_DIR, fname)
    dst = os.path.join(EVAL_DIR, fname)
    # skip the evaluation folder itself
    if os.path.abspath(src) == os.path.abspath(EVAL_DIR):
        continue
    try:
        # move files only (skip dirs)
        if os.path.isfile(src):
            shutil.move(src, dst)
    except Exception as e:
        print(f"  ⚠️ Could not move {src} -> {dst}: {e}")

print(f"All results available in: {EVAL_DIR}")

# =========================================================
# 🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS (fixed for multi-class)
# =========================================================
print("\n" + "=" * 70)
print("🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS")
print("=" * 70)

print("\n📊 Computing SHAP values (this may take a moment)...")

# use a sample for speed & stability
sample_size = min(100, len(X_test_scaled_df))
X_test_sample = X_test_scaled_df.iloc[:sample_size]
y_test_sample = y_test.iloc[:sample_size]

# Try an explainer that accepts a callable (predict_proba) to support multi-class
background = shap.sample(X_test_scaled_df, min(200, len(X_test_scaled_df)), random_state=0)

try:
    # Explainer will choose a fast tree-aware explainer if appropriate, otherwise a suitable fallback.
    explainer_shap = shap.Explainer(model.predict_proba, background, feature_names=feature_names)
    shap_expl = explainer_shap(X_test_sample)  # this returns a shap.Explanation object
    shap_is_explanation = True
    print("✓ shap.Explainer computed (Explanation object).")
except Exception as e:
    print("⚠️ shap.Explainer failed, falling back to KernelExplainer:", e)
    ke_bg = shap.sample(X_test_scaled_df, min(50, len(X_test_scaled_df)), random_state=0)
    explainer_shap = shap.KernelExplainer(model.predict_proba, ke_bg)
    # KernelExplainer returns a legacy list-of-arrays via .shap_values(...)
    shap_vals_legacy = explainer_shap.shap_values(X_test_sample.values)
    shap_is_explanation = False
    print("✓ KernelExplainer computed (legacy shap values).")

# Ensure a consistent variable name for later use
if shap_is_explanation:
    shap_values = shap_expl          # Explanation object
else:
    shap_values = shap_vals_legacy   # legacy list

print(f"✅ SHAP values ready for {sample_size} samples")

# =========================================================
# 🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS (Upgraded)
# =========================================================
print("\n" + "=" * 70)
print("🎨 SHAP - SHAPLEY ADDITIVE EXPLANATIONS")
print("=" * 70)

print("\n📊 Computing SHAP values (this may take a moment)...")

# use a sample for speed & stability
sample_size = min(100, len(X_test_scaled_df))
X_test_sample = X_test_scaled_df.iloc[:sample_size]
y_test_sample = y_test.iloc[:sample_size]

# Use a background dataset for the explainer
background = shap.sample(X_test_scaled_df, min(200, len(X_test_scaled_df)), random_state=0)

shap_interaction_values = None # Initialize
shap_is_explanation = False

try:
    # Use shap.Explainer, which will auto-select the best method (e.g., TreeExplainer for XGBoost)
    explainer_shap = shap.Explainer(model, background, feature_names=feature_names)
    shap_expl = explainer_shap(X_test_sample)  # this returns a shap.Explanation object
    shap_is_explanation = True
    print("✓ shap.Explainer computed (Explanation object).")
    
    # --- (ADDITION) COMPUTE INTERACTION VALUES ---
    try:
        print("📊 Computing SHAP interaction values (this may be slow)...")
        shap_interaction_values = explainer_shap.shap_interaction_values(X_test_sample)
        print("✅ SHAP interaction values computed.")
    except Exception as ie:
        print(f"⚠️ Could not compute SHAP interaction values: {ie}. Interaction plot will be skipped.")

except Exception as e:
    # Fallback for models not supported by shap.Explainer (like scikit-learn's GradientBoostingClassifier)
    print(f"⚠️ shap.Explainer failed ({e}). Falling back to KernelExplainer (slower).")
    ke_bg = shap.sample(X_test_scaled_df, min(50, len(X_test_scaled_df)), random_state=0)
    explainer_shap = shap.KernelExplainer(model.predict_proba, ke_bg)
    shap_vals_legacy = explainer_shap.shap_values(X_test_sample.values)
    shap_is_explanation = False
    print("✓ KernelExplainer computed (legacy shap values).")
    print("⚠️ SHAP interaction plot skipped: KernelExplainer does not support interaction values.")

# Ensure a consistent variable name for later use
if shap_is_explanation:
    shap_values = shap_expl          # Explanation object
else:
    shap_values = shap_vals_legacy   # legacy list

print(f"✅ SHAP values ready for {sample_size} samples")

# -----------------------------
# Plotting SHAP visualizations
# -----------------------------
print("\n📊 Generating SHAP visualizations...")

# ---
# PLOT 1: Summary Beeswarm Plot (All Classes) - As requested
# ---
print("Generating SHAP summary beeswarm plot...")
plt.figure()
if shap_is_explanation:
    # The modern Explanation object handles multi-class plotting automatically
    shap.summary_plot(
        shap_expl, 
        X_test_sample, 
        class_names=class_names, # Add class names for multi-class view
        show=False, 
        max_display=20
    )
else:
    # The legacy `shap_values` is a list of arrays. Passing the full list plots all classes.
    shap.summary_plot(
        shap_vals_legacy, # Pass the full list
        X_test_sample, 
        class_names=class_names, 
        show=False, 
        max_display=20
    )
plt.title("SHAP Summary Plot (All Classes)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'shap_summary_beeswarm.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ shap_summary_beeswarm.png saved")

# ---
# PLOT 2: Summary Bar Plot (Mean Absolute)
# ---
print("Generating SHAP summary bar plot...")
plt.figure(figsize=(12, 10))
if shap_is_explanation:
    # plot_type="bar" averages over all classes by default
    shap.summary_plot(shap_expl, X_test_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
else:
    # Passing the list averages over all classes
    shap.summary_plot(shap_vals_legacy, X_test_sample, feature_names=feature_names, plot_type="bar", show=False, max_display=20)

plt.title('SHAP Bar Plot - Mean Absolute Feature Importance', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, 'shap_bar.png'), dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Bar plot saved")

# ---
# PLOT 3: Waterfall plots (per-instance)
# ---
print("\n📊 Generating individual SHAP waterfall plots...")
num_waterfalls = min(3, len(X_test_sample))
for i in range(num_waterfalls):
    try:
        plt.figure(figsize=(12, 8))
        if shap_is_explanation:
            # If Explanation has per-class values, select predicted class index
            vals = shap_expl.values
            if vals.ndim == 3:
                pred_class = y_pred[i]
                # Slicing the Explanation object is the modern way
                exp_obj = shap_expl[i, :, pred_class]
            else:
                exp_obj = shap_expl[i]
            shap.waterfall_plot(exp_obj, show=False, max_display=15)
        else:
            # Legacy method
            pred_class = y_pred[i]
            vals_i = shap_vals_legacy[pred_class][i]
            base_val = explainer_shap.expected_value[pred_class]
            exp_obj = shap.Explanation(values=vals_i, base_values=base_val, data=X_test_sample.iloc[i].values, feature_names=feature_names)
            shap.waterfall_plot(exp_obj, show=False, max_display=15)

        plt.title(f'SHAP Waterfall - Instance {i + 1}\nTrue: {class_names[y_test_sample.iloc[i]]} | Pred: {class_names[y_pred[i]]}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, f'shap_waterfall_{i + 1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Waterfall plot {i + 1} saved")
    except Exception as e:
        print(f"   ⚠️  Could not generate waterfall plot {i + 1}: {e}")
        plt.close()
        continue

# ---
# PLOT 4: (ADDITION) SHAP INTERACTION PLOT
# ---
if shap_interaction_values is not None:
    print("\n📊 Generating SHAP interaction plot...")
    try:
        plt.figure()
        
        # For multi-class (like yours), shap_interaction_values is a list of arrays.
        # We'll plot the interactions for the *first* class (e.g., Class 0)
        # or change index to [1] for Class 1, etc.
        if isinstance(shap_interaction_values, list) and len(shap_interaction_values) > 0:
            plot_values = shap_interaction_values[0] # Using Class 0
            title = f'SHAP Interaction Values Summary (Class: {class_names[0]})'
        else:
            plot_values = shap_interaction_values # For binary or single-output
            title = 'SHAP Interaction Values Summary'

        shap.summary_plot(plot_values, X_test_sample, feature_names=feature_names, show=False, max_display=10)
        
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(EVAL_DIR, 'shap_interaction_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ SHAP interaction summary plot saved: shap_interaction_summary.png")
    
    except Exception as e:
        print(f"⚠️ Could not plot SHAP interaction values: {e}")
        plt.close()
else:
    print("⚠️ SHAP interaction plot skipped (values not computed).")


# prepare shap_values variable for later importance computation
if shap_is_explanation:
    shap_values_for_import = shap_expl
else:
    shap_values_for_import = shap_vals_legacy

print(f"\n💾 SHAP visualizations saved: {EVAL_DIR}/shap_*.png")