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
#
# # cleaned_cnn_pipeline.py
# import os
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
# from tensorflow.keras.utils import to_categorical
# from pandas.api.types import is_numeric_dtype
#
# # -----------------------------
# # Config
# # -----------------------------
# DATASETS_PATH = "../datasets"
# TRAIN_FILE = os.path.join(DATASETS_PATH, "train_dataset.xlsx")
# TEST_FILE = os.path.join(DATASETS_PATH, "test_dataset.xlsx")  # optional
#
# ARTIFACTS_DIR = "artifacts"
# os.makedirs(ARTIFACTS_DIR, exist_ok=True)
#
# CAT_COLS = ["gender", "ever_married", "work_type", "residence_type", "smoking_status", "alcohol"]
# TARGET_COL = "result"
# RANDOM_STATE = 42
# EPOCHS = 20
# BATCH_SIZE = 32
#
# # -----------------------------
# # Utility functions
# # -----------------------------
# def safe_mode(series):
#     m = series.mode()
#     return m.iloc[0] if not m.empty else np.nan
#
# def load_excel_safe(path):
#     df = pd.read_excel(path)
#     df.columns = df.columns.str.lower()
#     df.replace("?", np.nan, inplace=True)
#     return df
#
# # -----------------------------
# # 1. Load data
# # -----------------------------
# train_df = load_excel_safe(TRAIN_FILE)
# has_test = os.path.exists(TEST_FILE)
# test_df = load_excel_safe(TEST_FILE) if has_test else None
#
# # -----------------------------
# # 2. Imputation (fit on train only)
# # -----------------------------
# imputation_values = {}
# for col in train_df.columns:
#     if col == TARGET_COL:
#         continue
#     if is_numeric_dtype(train_df[col]):
#         imputation_values[col] = float(train_df[col].median())
#     else:
#         imputation_values[col] = safe_mode(train_df[col].astype(str))
#     # apply to train
#     train_df[col] = train_df[col].fillna(imputation_values[col])
#
# # apply to test if present (use train values only)
# if has_test:
#     for col, val in imputation_values.items():
#         if col in test_df.columns:
#             test_df[col] = test_df[col].fillna(val)
#
# joblib.dump(imputation_values, os.path.join(ARTIFACTS_DIR, "imputation_values.pkl"))
#
# # -----------------------------
# # 3. Encoding categorical columns (fit on train only)
# #    - handle unseen categories in test by mapping them to the most frequent training category (string form)
# # -----------------------------
# label_encoders = {}
# most_freq_category = {}
#
# for col in CAT_COLS:
#     if col not in train_df.columns:
#         continue
#     # convert to string before encoding to avoid dtype issues
#     train_df[col] = train_df[col].astype(str)
#     le = LabelEncoder()
#     le.fit(train_df[col])
#     label_encoders[col] = le
#
#     # store the most frequent training category (string)
#     most_freq_category[col] = safe_mode(train_df[col])
#
#     # transform train
#     train_df[col] = le.transform(train_df[col])
#
#     # transform test safely if present
#     if has_test and col in test_df.columns:
#         test_df[col] = test_df[col].astype(str)
#         # replace unseen categories (strings) with most frequent training category (string),
#         # because LabelEncoder.transform will complain on unseen labels
#         mask_unseen = ~test_df[col].isin(le.classes_)
#         if mask_unseen.any():
#             # replace unseen with most frequent training string
#             test_df.loc[mask_unseen, col] = most_freq_category[col]
#         test_df[col] = le.transform(test_df[col])
#
# joblib.dump(label_encoders, os.path.join(ARTIFACTS_DIR, "label_encoders.pkl"))
# joblib.dump(most_freq_category, os.path.join(ARTIFACTS_DIR, "most_freq_category.pkl"))
#
# # -----------------------------
# # 4. Prepare features and target
# # -----------------------------
# if TARGET_COL not in train_df.columns:
#     raise KeyError(f"Target column '{TARGET_COL}' not found in training data")
#
# X_train = train_df.drop(TARGET_COL, axis=1)
# y_train = train_df[TARGET_COL].astype(int)
#
# if has_test:
#     if TARGET_COL not in test_df.columns:
#         raise KeyError(f"Target column '{TARGET_COL}' not found in test data")
#     X_test = test_df.drop(TARGET_COL, axis=1)
#     y_test = test_df[TARGET_COL].astype(int)
# else:
#     X_test = None
#     y_test = None
#
# # -----------------------------
# # 5. Scaling (fit scaler on train only)
# # -----------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
#
# if has_test:
#     X_test_scaled = scaler.transform(X_test)
#
# # -----------------------------
# # 6. Helper: model builder
# # -----------------------------
# def create_cnn_model(input_dim, num_classes, gaussian_noise=None, dropout_rates=(0.3, 0.3)):
#     layers = []
#     if gaussian_noise is not None:
#         layers.append(GaussianNoise(gaussian_noise, input_shape=(input_dim,)))
#         layers.append(Dense(256, activation="relu"))
#     else:
#         layers.append(Dense(128, activation="relu", input_shape=(input_dim,)))
#     layers.append(Dropout(dropout_rates[0]))
#     layers.append(Dense(64, activation="relu"))
#     layers.append(Dropout(dropout_rates[1]))
#     layers.append(Dense(32, activation="relu"))
#     layers.append(Dense(num_classes, activation="softmax"))
#
#     model = Sequential(layers)
#     model.compile(optimizer="adam",
#                   loss=tf.keras.losses.CategoricalCrossentropy(),
#                   metrics=["accuracy"])
#     return model
#
# # -----------------------------
# # 7. K-Fold Cross-Validation (Stratified)
# # -----------------------------
# num_classes = len(np.unique(y_train))
# print(f"Training samples: {X_train_scaled.shape[0]}, features: {X_train_scaled.shape[1]}, classes: {num_classes}")
#
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
# fold_accuracies = []
# fold_histories = []
#
# for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_scaled, y_train), start=1):
#     print(f"\n--- Fold {fold} ---")
#     X_tr = X_train_scaled[train_idx]
#     X_val = X_train_scaled[val_idx]
#     y_tr = to_categorical(y_train.iloc[train_idx], num_classes=num_classes)
#     y_val = to_categorical(y_train.iloc[val_idx], num_classes=num_classes)
#
#     model = create_cnn_model(X_train_scaled.shape[1], num_classes)
#     hist = model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), verbose=0)
#
#     loss, acc = model.evaluate(X_val, y_val, verbose=0)
#     print(f"Fold {fold} - val acc: {acc*100:.2f}% loss: {loss:.4f}")
#     fold_accuracies.append(acc)
#     fold_histories.append(hist)
#
# print("\nCV results:")
# print(f"Mean acc: {np.mean(fold_accuracies)*100:.2f}%, std: {np.std(fold_accuracies)*100:.2f}%")
#
# # -----------------------------
# # 8. Final training on full train set (standard labels)
# # -----------------------------
# y_train_cat = to_categorical(y_train, num_classes=num_classes)
# final_model = create_cnn_model(X_train_scaled.shape[1], num_classes)
# history = final_model.fit(X_train_scaled, y_train_cat, epochs=EPOCHS, batch_size=BATCH_SIZE,
#                           validation_split=0.1, verbose=1)
#
# # -----------------------------
# # 9. Evaluate on test if provided
# # -----------------------------
# if has_test:
#     y_test_cat = to_categorical(y_test, num_classes=num_classes)
#     loss, acc = final_model.evaluate(X_test_scaled, y_test_cat, verbose=0)
#     y_pred = np.argmax(final_model.predict(X_test_scaled), axis=1)
#     print(f"\nTest loss: {loss:.4f}, test acc: {acc*100:.2f}%")
#
#     print("\nClassification report (test):")
#     print(classification_report(y_test, y_pred))
#
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title("Confusion Matrix (Test)")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.show()
#
# # -----------------------------
# # 10. Save final model
# # -----------------------------
# final_model.save(os.path.join(ARTIFACTS_DIR, "cnn_model.keras"))
# print(f"Saved model to {ARTIFACTS_DIR}/cnn_model.keras")
