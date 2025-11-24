import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------------
# 1️⃣ Load Dataset
# -----------------------------
dataset_path = "stroke_dataset.xlsx"  # replace with your path
data = pd.read_excel(dataset_path)

# Normalize column names
data.columns = data.columns.str.lower()

# Replace "?" with NaN
data.replace("?", np.nan, inplace=True)

# -----------------------------
# 2️⃣ Handle Missing Values
# -----------------------------
for col in data.columns:
    if data[col].dtype in [np.float64, np.int64]:
        data[col] = data[col].fillna(data[col].median())
    else:
        data[col] = data[col].fillna(data[col].mode()[0])

# -----------------------------
# 3️⃣ Plot Bar Graphs for Each Column
# -----------------------------
for col in data.columns:
    plt.figure(figsize=(8, 5))

    if data[col].dtype == 'object' or len(data[col].unique()) <= 20:
        # Categorical or low cardinality column
        sns.countplot(x=col, data=data, palette="Set2", order=data[col].value_counts().index)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        for p in plt.gca().patches:
            height = p.get_height()
            plt.gca().annotate(f'{height}', (p.get_x() + p.get_width() / 2., height),
                               ha='center', va='bottom', fontsize=10)
    else:
        # Numeric column with many unique values -> create bins
        sns.histplot(data[col], bins=20, kde=False, color='skyblue')
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()
