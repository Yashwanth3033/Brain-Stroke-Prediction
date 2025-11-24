import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1️⃣ Paths & Load Data
# -----------------------------
datasets_path = "../datasets"  # Adjust if needed
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
# 2️⃣ Handle Missing Values (NO LEAKAGE)
# -----------------------------
# Store imputation values from training data only
imputation_values = {}

for col in train_data.columns:
    if col == 'result':  # Skip target variable
        continue

    if train_data[col].dtype in [np.float64, np.int64]:
        # Use training data median only
        imputation_values[col] = train_data[col].median()
        train_data[col] = train_data[col].fillna(imputation_values[col])
        test_data[col] = test_data[col].fillna(imputation_values[col])
    else:
        # Use training data mode only
        imputation_values[col] = train_data[col].mode()[0]
        train_data[col] = train_data[col].fillna(imputation_values[col])
        test_data[col] = test_data[col].fillna(imputation_values[col])

# Save imputation values for future use
joblib.dump(imputation_values, "imputation_values.pkl")

# -----------------------------
# 3️⃣ Encode Categorical Columns (NO LEAKAGE)
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]

label_encoders = {}

for col in cat_cols:
    if col in train_data.columns:
        le = LabelEncoder()
        # Fit only on training data
        train_data[col] = le.fit_transform(train_data[col].astype(str))

        # Handle unseen categories in test data
        test_col_str = test_data[col].astype(str)
        # Map unseen categories to a default value (e.g., most frequent class)
        unseen_mask = ~test_col_str.isin(le.classes_)
        if unseen_mask.any():
            print(f"Warning: {unseen_mask.sum()} unseen categories in {col} for test data")
            # Replace unseen categories with most frequent training category
            most_frequent = train_data[col].mode()[0]
            test_col_str[unseen_mask] = le.classes_[most_frequent]

        test_data[col] = le.transform(test_col_str)

        # Save encoder for future use
        label_encoders[col] = le

# Save all encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# -----------------------------
# 4️⃣ Split Features & Target
# -----------------------------
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]
X_test = test_data.drop("result", axis=1)
y_test = test_data["result"]

# -----------------------------
# 5️⃣ Scale Data (NO LEAKAGE)
# -----------------------------
# Create new scaler and fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't refit

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")

# Convert target to categorical for multi-class
y_test_cat = to_categorical(y_test)

# -----------------------------
# 6️⃣ Show training dataset info
# -----------------------------
num_classes = len(np.unique(y_train))
print(f"CNN will be trained on {X_train_scaled.shape[0]} samples and {X_train_scaled.shape[1]} features.")
print(f"Number of classes: {num_classes}")


# -----------------------------
# 7️⃣ Function to Create CNN Model
# -----------------------------
def create_cnn_model(input_dim, num_classes):
    """Create and compile a CNN model"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# -----------------------------
# 8️⃣ K-Fold Cross-Validation (K=5)
# -----------------------------
print("\n" + "=" * 60)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Setup 5-Fold Stratified Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store results
fold_accuracies = []
fold_losses = []
fold_histories = []

# Perform K-Fold Cross-Validation
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train), 1):
    print(f"\n--- Training Fold {fold}/5 ---")

    # Split data for this fold
    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = to_categorical(y_train.iloc[train_idx], num_classes=num_classes)
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = to_categorical(y_train.iloc[val_idx], num_classes=num_classes)

    # Create and train model for this fold
    model_fold = create_cnn_model(X_train_scaled.shape[1], num_classes)

    history_fold = model_fold.fit(
        X_train_fold, y_train_fold,
        epochs=20,
        batch_size=32,
        validation_data=(X_val_fold, y_val_fold),
        verbose=0  # Set to 1 to see training progress for each fold
    )

    # Evaluate on validation fold
    loss_fold, acc_fold = model_fold.evaluate(X_val_fold, y_val_fold, verbose=0)
    fold_accuracies.append(acc_fold)
    fold_losses.append(loss_fold)
    fold_histories.append(history_fold)

    print(f"Fold {fold} - Validation Accuracy: {acc_fold * 100:.2f}%, Loss: {loss_fold:.4f}")

# Display K-Fold results
print("\n" + "=" * 60)
print("K-FOLD CROSS-VALIDATION RESULTS")
print("=" * 60)
print("\nFold-wise Accuracy:")
for i, acc in enumerate(fold_accuracies, 1):
    print(f"  Fold {i}: {acc * 100:.2f}%")

print(f"\nCross-Validation Summary:")
print(f"  Mean Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"  Std Deviation: {np.std(fold_accuracies) * 100:.2f}%")
print(f"  Min Accuracy:  {np.min(fold_accuracies) * 100:.2f}%")
print(f"  Max Accuracy:  {np.max(fold_accuracies) * 100:.2f}%")

# -----------------------------
# 9️⃣ Train Final Model on Full Training Data
# -----------------------------
print("\n" + "=" * 60)
print("TRAINING FINAL MODEL ON FULL TRAINING DATA")
print("=" * 60)

y_train_cat = to_categorical(y_train, num_classes=num_classes)

# Create final model
cnn_model = create_cnn_model(X_train_scaled.shape[1], num_classes)

# Display model summary
print("\nModel Architecture:")
cnn_model.summary()

# Train final model
history = cnn_model.fit(
    X_train_scaled, y_train_cat,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_scaled, y_test_cat),
    verbose=1
)

# -----------------------------
# 🔟 Evaluate Final Model on Test Set
# -----------------------------
loss, acc = cnn_model.evaluate(X_test_scaled, y_test_cat, verbose=0)

# Predict and generate classification report
y_pred_cnn = np.argmax(cnn_model.predict(X_test_scaled), axis=1)
report_cnn = classification_report(y_test, y_pred_cnn, output_dict=True)
cm_cnn = confusion_matrix(y_test, y_pred_cnn)

# Save final CNN model
cnn_model.save("cnn_model.keras")

# -----------------------------
# 1️⃣1️⃣ Visualization & Results
# -----------------------------
# Plot K-Fold training curves
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('K-Fold Cross-Validation Training History', fontsize=16)

for fold, hist in enumerate(fold_histories, 1):
    row = (fold - 1) // 3
    col = (fold - 1) % 3

    ax = axes[row, col]
    ax.plot(hist.history['accuracy'], label='Train Acc', alpha=0.8)
    ax.plot(hist.history['val_accuracy'], label='Val Acc', alpha=0.8)
    ax.set_title(f'Fold {fold}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Remove empty subplot
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()

# Plot final model training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Final Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Final Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Greens")
plt.title("CNN - Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
df_cnn = pd.DataFrame(report_cnn).transpose()
print("\n===== CNN Classification Report (Test Set) =====")
print(df_cnn)

# -----------------------------
# 1️⃣2️⃣ Final Summary
# -----------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Cross-Validation Mean Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (±{np.std(fold_accuracies) * 100:.2f}%)")
print(f"Final Model Test Accuracy:      {acc * 100:.2f}%")
print("=" * 60)