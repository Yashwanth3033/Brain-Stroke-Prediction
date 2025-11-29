import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# -----------------------------
# Paths & Load Test Data
# -----------------------------
datasets_path = "../datasets"
test_file = os.path.join(datasets_path, "test_dataset.xlsx")
test_data = pd.read_excel(test_file)

test_data.columns = test_data.columns.str.lower()
test_data.replace("?", np.nan, inplace=True)

# -----------------------------
# Load Imputation & Fill Missing
# -----------------------------
imputation_values = joblib.load("imputation_values.pkl")
for col in test_data.columns:
    if col in imputation_values:
        test_data[col].fillna(imputation_values[col], inplace=True)

# -----------------------------
# Encode Categorical Columns
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type", "residence_type", "smoking_status", "alcohol"]
label_encoders = joblib.load("label_encoders.pkl")

for col in cat_cols:
    if col in test_data.columns and col in label_encoders:
        le = label_encoders[col]
        test_col_str = test_data[col].astype(str)
        unseen_mask = ~test_col_str.isin(le.classes_)
        if unseen_mask.any():
            most_freq_class = le.classes_[0]
            test_data.loc[unseen_mask, col] = most_freq_class
        test_data[col] = le.transform(test_col_str)

# -----------------------------
# Split Features
# -----------------------------
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"].astype(int)

# -----------------------------
# Scale & Reshape
# -----------------------------
scaler = joblib.load("scaler.pkl")
X_test_scaled = scaler.transform(X_test)
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# -----------------------------
# Load Model & Predict
# -----------------------------
cnn_model = load_model("cnn_model.keras")
predictions = cnn_model.predict(X_test_scaled)

y_pred = np.argmax(predictions, axis=1)

# ==========================================================
# 🧪 NEW: PRINT CLASSIFICATION REPORT + ACCURACY + CONFUSION
# ==========================================================
print("\n===== TEST SET CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Optional visual confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
