import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --------------------------
# 1. Load Train & Test Data
# --------------------------
train_data = pd.read_excel("train_dataset.xlsx")
test_data  = pd.read_excel("test_dataset.xlsx")

print("Before cleaning:")
print("Train:", train_data.shape)
print("Test:", test_data.shape)

# Replace "?" with NaN
train_data = train_data.replace("?", np.nan).infer_objects(copy=False)
test_data  = test_data.replace("?", np.nan).infer_objects(copy=False)

# --------------------------
# 2. Features & Target Split
# --------------------------
X_train = train_data.drop("result", axis=1)
y_train = train_data["result"]

X_test  = test_data.drop("result", axis=1)
y_test  = test_data["result"]

# --------------------------
# 3. Identify column types
# --------------------------
num_cols = X_train.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# --------------------------
# 4. Pipelines
# --------------------------
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# --------------------------
# 5. Full Pipeline (Preprocessing + Model)
# --------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --------------------------
# 6. Save the pmodel
# --------------------------
joblib.dump(model, "stroke_model.pkl")
