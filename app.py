from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder="web")  # set template folder to web

# Paths to pmodel and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pmodel", "gb_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "pmodel", "scaler.pkl")

# Load trained pmodel and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Class labels
stroke_types = {
    0: "No Stroke",
    1: "Ischemic Stroke",
    2: "Hemorrhagic Stroke",
    3: "TIA"
}

# Feature names (same as in your CLI app)
features = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "family_history", "work_type", "residence_type", "avg_glucose_level",
    "bmi", "smoking_status", "alcohol", "diabetes", "BP_systolic",
    "BP_diastolic", "cholesterol", "whitebloodcellcount", "redbloodcellcount"
]

@app.route("/")
def home():
    return render_template("index.html", features=features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input from form
        user_input = []
        for feature in features:
            value = request.form.get(feature, "0")
            try:
                value = float(value)
            except ValueError:
                value = 0  # default if blank or invalid
            user_input.append(value)

        # Scale input
        user_input_scaled = scaler.transform([user_input])

        # Predict
        prediction = model.predict(user_input_scaled)[0]
        predicted_class = stroke_types.get(prediction, "Unknown")

        return render_template("index.html", features=features, result=f"Predicted: {predicted_class}")

    except Exception as e:
        return render_template("index.html", features=features, result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
