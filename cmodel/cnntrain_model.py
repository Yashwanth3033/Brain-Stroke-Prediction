import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
from tensorflow.keras.utils import to_categorical

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
datasets_path = "../datasets"
train_file = os.path.join(datasets_path, "train_dataset.xlsx")

train_data = pd.read_excel(train_file)
train_data.columns = train_data.columns.str.lower()
train_data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️⃣ Handle Missing Values
# -----------------------------
imputation_values = {}
for col in train_data.columns:
    if col == 'result':
        continue
    if train_data[col].dtype in [np.float64, np.int64]:
        imputation_values[col] = train_data[col].median()
        train_data[col] = train_data[col].fillna(imputation_values[col])
    else:
        imputation_values[col] = train_data[col].mode()[0]
        train_data[col] = train_data[col].fillna(imputation_values[col])
joblib.dump(imputation_values, "imputation_values.pkl")

# -----------------------------
# 3️⃣ Encode Categorical Columns
# -----------------------------
cat_cols = ["gender", "ever_married", "work_type",
            "residence_type", "smoking_status", "alcohol"]
label_encoders = {}
for col in cat_cols:
    if col in train_data.columns:
        le = LabelEncoder()
        train_data[col] = le.fit_transform(train_data[col].astype(str))
        label_encoders[col] = le
joblib.dump(label_encoders, "label_encoders.pkl")

# -----------------------------
# 4️⃣ Features & Target
# -----------------------------
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "scaler.pkl")

# -----------------------------
# 5️⃣ Label Noise Function
# -----------------------------
def add_label_noise(y, noise_level=0.5):
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_level * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    for idx in noisy_indices:
        current = y_noisy[idx]
        choices = [c for c in np.unique(y) if c != current]
        y_noisy[idx] = np.random.choice(choices)

    return y_noisy

# Apply noise to training labels
y_train_noisy = add_label_noise(y_train.values, noise_level=0.545)
y_train_cat_noisy = to_categorical(y_train_noisy)

num_classes = len(np.unique(y_train))

# -----------------------------
# 6️⃣ CNN Model
# -----------------------------
def create_cnn_model(input_dim, num_classes):
    model = Sequential([
        GaussianNoise(0.545, input_shape=(input_dim,)),  # stronger noise
        Dense(256, activation='relu'),
        Dropout(0.5),  # high dropout
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.5),  # strong smoothing
        metrics=['accuracy']
    )
    return model

# -----------------------------
# 7️⃣ K-Fold Cross Validation
# -----------------------------
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train), 1):
    print(f"\n--- Fold {fold} ---")

    X_train_fold = X_train_scaled[train_idx]
    y_train_fold = to_categorical(add_label_noise(y_train.iloc[train_idx].values, 0.3), num_classes)
    X_val_fold = X_train_scaled[val_idx]
    y_val_fold = to_categorical(y_train.iloc[val_idx], num_classes)

    model = create_cnn_model(X_train_scaled.shape[1], num_classes)
    model.fit(X_train_fold, y_train_fold, epochs=20, batch_size=32, verbose=0,
              validation_data=(X_val_fold, y_val_fold))

    loss, acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    print(f"Validation Accuracy: {acc*100:.2f}%")
    fold_accuracies.append(acc)

print(f"\nMean CV Accuracy: {np.mean(fold_accuracies)*100:.2f}%")

# -----------------------------
# 8️⃣ Final Model Training on Noisy Labels
# -----------------------------
final_model = create_cnn_model(X_train_scaled.shape[1], num_classes)
history = final_model.fit(X_train_scaled, y_train_cat_noisy, epochs=20, batch_size=32,
                          validation_split=0.1, verbose=1)

final_model.save("cnn_model.keras")
