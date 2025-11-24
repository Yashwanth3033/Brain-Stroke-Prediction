import pandas as pd
from sklearn.model_selection import train_test_split

# 1️⃣ Load dataset
data = pd.read_excel("stroke_dataset.xlsx")

# Optional: remove any leading/trailing spaces in column names
data.columns = data.columns.str.strip()

# 2️⃣ Features & target
X = data.drop("result", axis=1)  # All columns except target
y = data["result"]               # Target column

# 3️⃣ Split dataset
# Train = 10,800; Test = rest (~2,686)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=10800, random_state=42, stratify=y
)

# 4️⃣ Combine features + target to save in Excel
train_data = X_train.copy()
train_data["result"] = y_train

test_data = X_test.copy()
test_data["result"] = y_test

# 5️⃣ Save as Excel
train_data.to_excel("train_dataset.xlsx", index=False)
test_data.to_excel("test_dataset.xlsx", index=False)

print("Train dataset saved:", train_data.shape)
print("Test dataset saved:", test_data.shape)
