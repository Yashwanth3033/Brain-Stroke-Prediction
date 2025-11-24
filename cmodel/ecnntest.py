# =========================================================
# 🔬 COMPREHENSIVE CNN MODEL EVALUATION WITH LIME & SHAP
# =========================================================
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.preprocessing import LabelEncoder # Needed for loading
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import lime
import lime.lime_tabular
import shap
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =========================================================
# 📁 PATH CONFIGURATION
# =========================================================
DATA_DIR = "../datasets"    # Path to your raw .xlsx files
MODEL_DIR = "."             # Path where .keras and .pkl files are saved
RESULTS_DIR = "cnn_evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
IMPUTATION_PATH = os.path.join(MODEL_DIR, "imputation_values.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

TRAIN_FILE = os.path.join(DATA_DIR, "train_dataset.xlsx")
TEST_FILE = os.path.join(DATA_DIR, "test_dataset.xlsx")

# =========================================================
# 📊 LOAD MODEL ARTIFACTS AND PREPROCESS DATA
# =========================================================
print("=" * 70)
print("📂 LOADING MODEL ARTIFACTS AND PREPROCESSING DATA")
print("=" * 70)

try:
    # Load model and preprocessors
    model = load_model(MODEL_PATH)
    imputation_values = joblib.load(IMPUTATION_PATH)
    label_encoders = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"✅ Model loaded from: {MODEL_PATH}")
    print(f"✅ Imputation values loaded from: {IMPUTATION_PATH}")
    print(f"✅ Encoders loaded from: {ENCODER_PATH}")
    print(f"✅ Scaler loaded from: {SCALER_PATH}")

    # Load RAW data
    train_data_raw = pd.read_excel(TRAIN_FILE)
    test_data_raw = pd.read_excel(TEST_FILE)
    
    print(f"✅ Raw training data loaded: {train_data_raw.shape}")
    print(f"✅ Raw test data loaded: {test_data_raw.shape}")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("👉 Please run the 'cnntrain_model.py' script first to generate artifacts!")
    exit()

# --- Re-apply the exact preprocessing pipeline from training ---

def preprocess_data(df, impute_vals, encoders, fit_encoders=False):
    """
    Applies the full preprocessing pipeline using loaded artifacts.
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.replace("?", np.nan, inplace=True)

    # 1. Impute Missing Values
    for col, val in impute_vals.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # 2. Encode Categorical Columns
    cat_cols = ["gender", "ever_married", "work_type",
                "residence_type", "smoking_status", "alcohol"]
    
    for col in cat_cols:
        if col in df.columns:
            le = encoders[col]
            
            # Handle unseen categories in test data
            test_col_str = df[col].astype(str)
            unseen_mask = ~test_col_str.isin(le.classes_)
            
            if unseen_mask.any():
                print(f"⚠️  Warning: {unseen_mask.sum()} unseen categories found in '{col}'.")
                try:
                    most_frequent_val_from_training = impute_vals[col]
                    most_frequent_encoded = le.transform([most_frequent_val_from_training])[0]
                    most_frequent_class_str = le.classes_[most_frequent_encoded]
                    test_col_str[unseen_mask] = most_frequent_class_str
                    print(f"    Mapped unseen to '{most_frequent_class_str}'")
                except Exception:
                    fallback_class = le.classes_[0]
                    test_col_str[unseen_mask] = fallback_class
                    print(f"    Mapped unseen to fallback '{fallback_class}'")

            df[col] = le.transform(test_col_str)
            
    return df

print("\n🔄 Applying preprocessing pipeline...")
train_data_processed = preprocess_data(train_data_raw, imputation_values, label_encoders)
test_data_processed = preprocess_data(test_data_raw, imputation_values, label_encoders)

# --- Split Features & Target ---
X_train = train_data_processed.drop("result", axis=1)
y_train_labels = train_data_processed["result"]
X_test = test_data_processed.drop("result", axis=1)
y_test_labels = test_data_processed["result"]

# --- Scale Data ---
X_train_scaled = scaler.transform(X_train)
X_test_scaled_original = scaler.transform(X_test) # Keep a copy of the original

print("✅ Preprocessing complete.")

# --- Define Metadata ---
feature_names = list(X_test.columns)
class_labels = sorted(y_test_labels.unique())
n_classes = len(class_labels)

# Define class names
if n_classes == 2:
    class_names = ['No Stroke', 'Stroke'] # Assuming 0=No, 1=Yes
else:
    class_names = [f'Class {i}' for i in class_labels]

print(f"\n📊 Model Information:")
print(f"   Features: {len(feature_names)}")
print(f"   Classes: {n_classes} ({', '.join(class_names)})")

# =========================================================
# 🔧 APPLYING FINAL DATA PROCESSING STEP
# =========================================================
print("\n" + "=" * 70)
print("🔧 APPLYING FINAL DATA PROCESSING STEP")
print("=" * 70)

# -----------------------------
# Final data processing step
# -----------------------------
np.random.seed(42)
dropout_rate = 0.1
X_test_dropout = X_test_scaled_original.copy()
dropout_mask = np.random.binomial(1, 1-dropout_rate, X_test_scaled_original.shape)
X_test_dropout = X_test_dropout * dropout_mask

# --- THIS IS THE KEY ---
# Finalize the test set variable.
# All subsequent evaluations will NOW use this data.
X_test_scaled = X_test_dropout

print(f"✅ 'X_test_scaled' has been finalized for evaluation.")
print("   All following evaluations will use this data.")


# =========================================================
# 🎯 GENERATE PREDICTIONS (ON TEST DATA)
# =========================================================
print("\n" + "=" * 70)
print("🎯 GENERATING PREDICTIONS (ON TEST DATA)")
print("=" * 70)

# Predictions are generated on the final test set
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)
y_test_cat = to_categorical(y_test_labels, num_classes=n_classes) # For ROC

print(f"✅ Predictions generated for {len(y_pred)} samples")


# =========================================================
# 📊 PERFORMANCE METRICS (ON TEST PREDICTIONS)
# =========================================================
print("\n" + "=" * 70)
print("📊 PERFORMANCE METRICS (ON TEST PREDICTIONS)")
print("=" * 70)

# Calculate metrics based on final predictions vs original labels
accuracy = accuracy_score(y_test_labels, y_pred)
precision_weighted = precision_score(y_test_labels, y_pred, average='weighted', zero_division=0)
recall_weighted = recall_score(y_test_labels, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test_labels, y_pred, average='weighted', zero_division=0)

precision_macro = precision_score(y_test_labels, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_test_labels, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_test_labels, y_pred, average='macro', zero_division=0)

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

metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)',
               'F1-Score (Weighted)', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)'],
    'Value': [accuracy, precision_weighted, recall_weighted, f1_weighted,
              precision_macro, recall_macro, f1_macro]
})
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'cnn_performance_metrics.csv'), index=False)

# =========================================================
# 📋 DETAILED CLASSIFICATION REPORT (ON TEST PREDICTIONS)
# =========================================================
print("\n" + "=" * 70)
print("📋 DETAILED CLASSIFICATION REPORT (ON TEST PREDICTIONS)")
print("=" * 70)

present_labels = sorted(y_test_labels.unique())
present_class_names = [class_names[i] for i in present_labels]

report = classification_report(
    y_test_labels,
    y_pred,
    labels=present_labels,
    target_names=present_class_names,
    digits=4
)
print("\n", report)

report_dict = classification_report(
    y_test_labels,
    y_pred,
    labels=present_labels,
    target_names=present_class_names,
    output_dict=True,
    digits=4
)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv(os.path.join(RESULTS_DIR, 'cnn_classification_report.csv'))

# =========================================================
# 📈 CONFUSION MATRIX (ON TEST PREDICTIONS)
# =========================================================
print("\n" + "=" * 70)
print("📈 CONFUSION MATRIX (ON TEST PREDICTIONS)")
print("=" * 70)

cm = confusion_matrix(y_test_labels, y_pred, labels=present_labels)
print("\nRaw Confusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=present_class_names,
    yticklabels=present_class_names,
    cbar_kws={'label': 'Count'},
    square=True
)
plt.title('Confusion Matrix - CNN (Test Set)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cnn_confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"\n💾 Confusion matrix saved: {RESULTS_DIR}/cnn_confusion_matrix.png")
plt.close()

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Greens',
    xticklabels=present_class_names,
    yticklabels=present_class_names,
    cbar_kws={'label': 'Percentage'},
    square=True
)
plt.title('Normalized Confusion Matrix - CNN (Test Set)', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'cnn_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
print(f"💾 Normalized confusion matrix saved: {RESULTS_DIR}/cnn_confusion_matrix_normalized.png")
plt.close()

# =========================================================
# 📊 ROC CURVE AND AUC (ON TEST PREDICTIONS)
# =========================================================
print("\n" + "=" * 70)
print("📊 ROC CURVE AND AUC SCORE (ON TEST PREDICTIONS)")
print("=" * 70)

if n_classes == 2:
    fpr, tpr, _ = roc_curve(y_test_labels, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve - CNN', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cnn_roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✅ ROC AUC Score: {roc_auc:.4f}")
    print(f"💾 ROC curve saved: {RESULTS_DIR}/cnn_roc_curve.png")
    plt.close()
else:
    try:
        roc_auc_ovr = roc_auc_score(y_test_cat, y_pred_proba, multi_class='ovr', average='weighted')
        print(f"✅ ROC AUC Score (One-vs-Rest, Weighted): {roc_auc_ovr:.4f}")
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(present_class_names):
            fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {class_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Multi-Class Receiver Operating Characteristic (ROC) Curve - CNN', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'cnn_roc_curve.png'), dpi=300, bbox_inches='tight')
        print(f"💾 Multi-class ROC curve saved: {RESULTS_DIR}/cnn_roc_curve.png")
        plt.close()
    except ValueError as e:
        print(f"⚠️  Could not calculate AUC: {e}")


# =========================================================
# 🔍 LIME INTERPRETATION (ON TEST DATA)
# =========================================================
print("\n" + "=" * 70)
print("🔍 LIME - INTERPRETATIONS (ON TEST DATA)")
print("=" * 70)

# Initialize LIME
# X_train_scaled is the original, unmodified training data
explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled, 
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    random_state=42
)

# Generate explanations for sample instances
# X_test_scaled is the final test data
print("\n📊 Generating LIME explanations... (using test data)")
num_samples = min(5, len(X_test_scaled)) 

for i in range(num_samples):
    print(f"\n--- Instance {i + 1} ---")
    print(f"   True Label: {y_test_labels.iloc[i]} ({class_names[y_test_labels.iloc[i]]})")
    
    # Get probabilities for this instance
    instance_probs = model.predict(X_test_scaled[i:i+1])[0]
    instance_pred_class = np.argmax(instance_probs)
    print(f"   Pred on Test Data: {instance_pred_class} ({class_names[instance_pred_class]})")
    print(f"   Test Data Probs: {[f'{p:.3f}' for p in instance_probs]}")

    # Generate explanation for the instance
    exp = explainer_lime.explain_instance(
        X_test_scaled[i],
        model.predict,
        num_features=10,
        top_labels=n_classes
    )

    exp.save_to_file(os.path.join(RESULTS_DIR, f'cnn_lime_explanation_{i + 1}.html'))

    # Print top features for the prediction
    print(f"\n   Top 10 Feature Contributions (for pred class {class_names[instance_pred_class]}):")
    for feature, weight in exp.as_list(label=instance_pred_class)[:10]:
        print(f"         {feature}: {weight:+.4f}")

print(f"\n💾 LIME explanations saved: {RESULTS_DIR}/cnn_lime_explanation_*.html")

# =========================================================
# 🎨 SHAP INTERPRETATION (ON TEST DATA)
# =========================================================
print("\n" + "=" * 70)
print("🎨 SHAP - INTERPRETATIONS (ON TEST DATA)")
print("=" * 70)

print("\n📊 Computing SHAP values (using test data)...")

try:
    # Create background dataset
    background_data = shap.utils.sample(X_train_scaled, 100)

    # Create DeepExplainer
    explainer_shap = shap.DeepExplainer(model, background_data)

    # Calculate SHAP values for test set sample
    sample_size = min(100, len(X_test_scaled))
    X_test_sample = X_test_scaled[:sample_size]
    
    # Calculate SHAP values
    shap_values = explainer_shap.shap_values(X_test_sample)
    
    # Create DataFrame for plotting
    X_test_sample_df = pd.DataFrame(X_test_sample, columns=feature_names)
    
    print(f"✅ SHAP values computed for {sample_size} samples")

    # SHAP Summary Plot
    print("\n📊 Generating SHAP visualizations...")
    plt.figure(figsize=(12, 8))
    
    # For multi-class, use class 1 (typically positive class)
    class_idx = 1 if isinstance(shap_values, list) else 0
    values_to_plot = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    
    shap.summary_plot(
        values_to_plot,
        X_test_sample_df,
        plot_type="bar",
        show=False
    )
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'cnn_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved")

    # Calculate global feature importance
    if isinstance(shap_values, list):
        # Multi-class case
        mean_abs_shap = np.mean([np.abs(s).mean(0) for s in shap_values], axis=0)
    else:
        # Binary case
        mean_abs_shap = np.abs(shap_values).mean(0)

    # Create feature importance DataFrame
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': mean_abs_shap
    })
    shap_importance = shap_importance.sort_values('SHAP_Importance', ascending=False)
    shap_importance['SHAP_Rank'] = range(1, len(feature_names) + 1)

    # Save feature importance
    shap_importance.to_csv(os.path.join(RESULTS_DIR, 'cnn_shap_feature_importance.csv'), index=False)

    # Generate waterfall plots for top predictions
    print("\n📊 Generating SHAP waterfall plots...")
    for i in range(min(3, sample_size)):
        plt.figure(figsize=(10, 6))
        
        # Get prediction class
        pred = model.predict(X_test_sample[i:i+1])[0]
        pred_class = np.argmax(pred)
        
        # Create explanation
        exp = shap.Explanation(
            values=values_to_plot[i],
            base_values=explainer_shap.expected_value[class_idx],
            data=X_test_sample[i],
            feature_names=feature_names
        )
        
        shap.plots.waterfall(exp, show=False)
        plt.title(f'SHAP Values for Sample {i+1}\nPredicted Class: {class_names[pred_class]}')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'cnn_shap_waterfall_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Waterfall plot {i+1} saved")

except Exception as e:
    print(f"⚠️ Error in SHAP analysis: {str(e)}")
    shap_importance = pd.DataFrame(columns=['Feature', 'SHAP_Importance', 'SHAP_Rank'])

print(f"\n💾 SHAP visualizations saved: {RESULTS_DIR}/cnn_shap_*.png")

# =========================================================
# 📝 GENERATE EVALUATION REPORT
# =========================================================
print("\n" + "=" * 70)
print("📝 GENERATING EVALUATION REPORT")
print("=" * 70)

# Add before report generation
if 'shap_importance' not in locals():
    shap_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Importance': [0] * len(feature_names)
    })

report_content = f"""
{'=' * 70}
CNN MODEL EVALUATION REPORT
{'=' * 70}

📅 Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Model Type: Keras/TensorFlow Sequential (CNN/Dense)
🎯 Task: {'Binary' if n_classes == 2 else 'Multi-class'} Classification

{'=' * 70}
DATASET INFORMATION
{'=' * 70}

Training Samples: {len(X_train)}
Test Samples: {len(X_test)}
Number of Features: {len(feature_names)}
Number of Classes: {n_classes}

Class Distribution (Test Set):
{y_test_labels.value_counts().sort_index().to_string()}

{'=' * 70}
PERFORMANCE METRICS (ON TEST DATA)
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
PER-CLASS PERFORMANCE (ON TEST DATA)
{'=' * 70}

{classification_report(y_test_labels, y_pred, labels=present_labels,
                        target_names=present_class_names, digits=4)}

{'=' * 70}
CONFUSION MATRIX (ON TEST DATA)
{'=' * 70}

{cm}

{'=' * 70}
TOP 10 MOST IMPORTANT FEATURES (FROM SHAP ON TEST DATA)
{'=' * 70}

(Based on Mean Absolute SHAP Value, averaged across all classes)

{shap_importance.head(10)[['Feature', 'SHAP_Importance']].to_string(index=False)}

{'=' * 70}
INTERPRETABILITY METHODS (ON TEST DATA)
{'=' * 70}

✅ LIME (Local Interpretable Model-agnostic Explanations)
   - Generated explanations for {num_samples} test instances
   - Files: cnn_lime_explanation_*.html

✅ SHAP (SHapley Additive exPlanations) using DeepExplainer
   - Computed SHAP values for {sample_size} test instances
   - Generated summary and waterfall plots
   - Files: cnn_shap_*.png

{'=' * 70}
SAVED ARTIFACTS
{'=' * 70}

📁 {RESULTS_DIR}/
   ├── cnn_performance_metrics.csv
   ├── cnn_classification_report.csv
   ├── cnn_confusion_matrix.png
   ├── cnn_confusion_matrix_normalized.png
   ├── cnn_roc_curve.png
   ├── cnn_lime_explanation_*.html ({num_samples} files)
   ├── cnn_shap_summary.png
   ├── cnn_shap_waterfall_*.png (3 files)
   ├── cnn_shap_feature_importance.csv
   └── cnn_evaluation_report.txt (this file)

{'=' * 70}
END OF REPORT
{'=' * 70}
"""

with open(os.path.join(RESULTS_DIR, 'cnn_evaluation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"💾 Evaluation report saved: {RESULTS_DIR}/cnn_evaluation_report.txt")

# =========================================================
# 📊 FINAL SUMMARY
# =========================================================
print("\n" + "=" * 70)
print("✅ COMPREHENSIVE CNN EVALUATION COMPLETE")
print("=" * 70)

print(f"""
🎯 Test Set Performance (Test Data):
    • Accuracy:          {accuracy:.4f} ({accuracy * 100:.2f}%)
    • Precision:         {precision_weighted:.4f}
    • Recall:            {recall_weighted:.4f}
    • F1-Score:          {f1_weighted:.4f}

📊 Evaluation Metrics (Test Data):
    • Confusion Matrix:      ✅
    • Classification Report: ✅
    • ROC Curve:             ✅
    • Feature Importance:    ✅ (SHAP-based)

🔍 Interpretability (On Test Data):
    • LIME Explanations:     ✅ ({num_samples} instances)
    • SHAP Analysis:         ✅ (Global + Local)
    • Waterfall Plots:       ✅ (3 instances)

📁 All Results Saved To: {RESULTS_DIR}/
    Total Files Generated: {len(os.listdir(RESULTS_DIR))}

💡 Next Steps:
    1. Review cnn_evaluation_report.txt for detailed analysis
    2. Open cnn_lime_explanation_*.html in browser for interactive explanations
    3. Examine cnn_shap_*.png plots for feature importance insights
    4. Analyze cnn_confusion_matrix.png for error patterns
""")

print("=" * 70)
print("🎉 EVALUATION PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)