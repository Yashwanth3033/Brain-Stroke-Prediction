import pandas as pd
import numpy as np
import os
import joblib  # <-- Make sure this is imported
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

# -----------------------------
# 1️⃣ Paths & Load Data
# -----------------------------
datasets_path = "../datasets" # Adjust if needed
train_file = os.path.join(datasets_path, "train_dataset.xlsx")
test_file = os.path.join(datasets_path, "test_dataset.xlsx")

train_data = pd.read_excel(train_file)
test_data = pd.read_excel(test_file)

# Normalize column names
train_data.columns = train_data.columns.str.lower()
test_data.columns = test_data.columns.str.lower()

# Replace "?" with NaN
train_data.replace("?", np.nan, inplace=True)
test_data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️⃣ Handle Missing Values
# -----------------------------
imputation_values = {} # <-- ADDED: Create dict to store values

for col in train_data.columns:
    if train_data[col].dtype in [np.float64, np.int64]:
        median_val = train_data[col].median()
        imputation_values[col] = median_val # <-- ADDED
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col] = test_data[col].fillna(median_val)
    else:
        mode_val = train_data[col].mode()[0]
        imputation_values[col] = mode_val # <-- ADDED
        train_data[col] = train_data[col].fillna(mode_val)
        test_data[col] = test_data[col].fillna(mode_val)

# <-- ADDED: Save the imputation values
joblib.dump(imputation_values, "imputation_values.pkl")
print("✅ imputation_values.pkl saved.")

# -----------------------------
# 3️⃣ Encode Categorical Columns
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]

label_encoders = {} # <-- ADDED: Create dict to store encoders

for col in cat_cols:
    if col in train_data.columns:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        test_data[col] = le.transform(test_data[col].astype(str))
        label_encoders[col] = le # <-- ADDED

# <-- ADDED: Save the label encoders
joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ label_encoders.pkl saved.")

# -----------------------------
# 4️⃣ Split Features & Target
# -----------------------------
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl saved.") # <-- ADDED: Confirmation print

# -----------------------------
# 4️⃣1️⃣ Show training dataset info
# -----------------------------
print(f"Gradient Boosting will be trained on {X_train_scaled.shape[0]} samples and {X_train_scaled.shape[1]} features.")

# -----------------------------
# 5️⃣ K-Fold Cross-Validation (K=5)
# -----------------------------
print("\n" + "="*60)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("="*60)

# Initialize Gradient Boosting model
gb_model = GradientBoostingClassifier()

# Setup 5-Fold Stratified Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(gb_model, X_train_scaled, y_train,
                            cv=kfold, scoring='accuracy', n_jobs=-1)

# Display fold-wise results
print("\nFold-wise Accuracy Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score*100:.2f}%")

print(f"\nCross-Validation Results:")
print(f"  Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"  Std Deviation: {cv_scores.std()*100:.2f}%")
print(f"  Min Accuracy:  {cv_scores.min()*100:.2f}%")
print(f"  Max Accuracy:  {cv_scores.max()*100:.2f}%")

# -----------------------------
# 6️⃣ Train Final Model on Full Training Data
# -----------------------------
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL TRAINING DATA")
print("="*60)

gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

# Metrics
accuracy_gb = accuracy_score(y_test, y_pred_gb)
report_gb = classification_report(y_test, y_pred_gb, output_dict=True)
cm_gb = confusion_matrix(y_test, y_pred_gb)

# Save model
joblib.dump(gb_model, "gb_model.pkl")
print("✅ gb_model.pkl saved.") # <-- ADDED: Confirmation print

# -----------------------------
# 7️⃣ Display Test Results
# -----------------------------
print(f"\nGradient Boosting Test Accuracy: {accuracy_gb*100:.2f}%\n")

# Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm_gb, annot=True, fmt="d", cmap="Blues")
plt.title("Gradient Boosting - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
df_gb = pd.DataFrame(report_gb).transpose()
print("\n===== Gradient Boosting Classification Report =====")
print(df_gb)

# -----------------------------
# 8️⃣ Summary Comparison
# -----------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Cross-Validation Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
print(f"Test Set Accuracy:{accuracy_gb*100:.2f}%")
print("="*60)