import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

print("="*60)
print("🚀 STARTING MODEL TRAINING (XGBOOST)")
print("="*60)

# -----------------------------
# 1️⃣ Paths & Load Data
# -----------------------------
datasets_path = "../datasets"
train_file = os.path.join(datasets_path, "train_dataset.xlsx")
test_file  = os.path.join(datasets_path, "test_dataset.xlsx")

train_data = pd.read_excel(train_file)
test_data  = pd.read_excel(test_file)

# Normalize column names
train_data.columns = train_data.columns.str.lower()
test_data.columns = test_data.columns.str.lower()

# Replace "?" with NaN
train_data.replace("?", np.nan, inplace=True)
test_data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️⃣ Handle Missing Values & Save imputation_values.pkl
# -----------------------------
print("Handling missing values and saving imputation file...")
imputation_values = {} 

for col in train_data.columns:
    if train_data[col].dtype in [np.float64, np.int64]:
        median_val = train_data[col].median()
        imputation_values[col] = median_val 
        train_data[col] = train_data[col].fillna(median_val)
        test_data[col]  = test_data[col].fillna(median_val)
    else:
        mode_val = train_data[col].mode()[0]
        imputation_values[col] = mode_val 
        train_data[col] = train_data[col].fillna(mode_val)
        test_data[col]  = test_data[col].fillna(mode_val)

joblib.dump(imputation_values, "imputation_values.pkl")
print("✅ imputation_values.pkl saved.")

# -----------------------------
# 3️⃣ Encode Categorical Columns & Save label_encoders.pkl
# -----------------------------
print("Encoding categorical features and saving encoders...")
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]

label_encoders = {} 

for col in cat_cols:
    if col in train_data.columns:
        le = LabelEncoder()
        # Combine train and test to ensure all categories are seen by encoder
        all_values = pd.concat([train_data[col], test_data[col]]).astype(str).unique()
        le.fit(all_values)
        
        # Transform train and test
        train_data[col] = le.transform(train_data[col].astype(str))
        test_data[col]  = le.transform(test_data[col].astype(str))
        label_encoders[col] = le 

joblib.dump(label_encoders, "label_encoders.pkl")
print("✅ label_encoders.pkl saved.")

# -----------------------------
# 4️⃣ Split Features & Target & Save scaler.pkl
# -----------------------------
print("Scaling data and saving scaler...")
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]
X_test  = test_data.drop("result", axis=1)
y_test  = test_data["result"]

# Ensure column order is the same
X_test = X_test[X_train.columns]

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl saved.") 

# -----------------------------
# 4️⃣1️⃣ Show training dataset info
# -----------------------------
print(f"XGBoost will be trained on {X_train_scaled.shape[0]} samples and {X_train_scaled.shape[1]} features.")

# -----------------------------
# 5️⃣ K-Fold Cross-Validation (K=5)
# -----------------------------
print("\n" + "="*60)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("="*60)

# v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v v
# *** CHANGED: Use XGBClassifier ***
xgb_model = XGBClassifier(
    objective='multi:softmax',  # Specify multi-class
    num_class=4,                # Specify number of classes
    use_label_encoder=False, 
    eval_metric='mlogloss'
)
# ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

# Setup 5-Fold Stratified Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train,
                            cv=kfold, scoring='accuracy', n_jobs=-1)

# Display fold-wise results
print("\nFold-wise Accuracy Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f" Fold {i}: {score*100:.2f}%")

print(f"\nCross-Validation Results:")
print(f" Mean Accuracy: {cv_scores.mean()*100:.2f}%")
print(f" Std Deviation: {cv_scores.std()*100:.2f}%")

# -----------------------------
# 6️⃣ Train Final Model on Full Training Data & Save gb_model.pkl
# -----------------------------
print("\n" + "="*60)
print("TRAINING FINAL MODEL ON FULL TRAINING DATA")
print("="*60)

# Re-initialize the model to train on all data
xgb_model_final = XGBClassifier(
    objective='multi:softmax', 
    num_class=4, 
    use_label_encoder=False, 
    eval_metric='mlogloss'
)
xgb_model_final.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model_final.predict(X_test_scaled)

# Metrics
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
cm_xgb = confusion_matrix(y_test, y_pred_xgb)

# Save model (still using the same filename your SHAP script expects)
joblib.dump(xgb_model_final, "gb_model.pkl")
print("✅ gb_model.pkl (XGBoost model) saved.")

# -----------------------------
# 7️⃣ Display Test Results
# -----------------------------
print(f"\nXGBoost Test Accuracy: {accuracy_xgb*100:.2f}%\n")

# Confusion Matrix Heatmap
plt.figure(figsize=(6,5))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
df_xgb = pd.DataFrame(report_xgb).transpose()
print("\n===== XGBoost Classification Report =====")
print(df_xgb)

# -----------------------------
# 8️⃣ Summary Comparison
# -----------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Cross-Validation Mean Accuracy: {cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)")
print(f"Test Set Accuracy: {accuracy_xgb*100:.2f}%")
print("="*60)
print("🎉 TRAINING COMPLETE")
print("="*60)