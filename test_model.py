import joblib
import numpy as np
from pathlib import Path

# Define the paths to the saved model and scaler files.
BEST_MODEL_PATH = Path("models") / "best_model.pkl"
SCALER_PATH = Path("models") / "scaler.pkl"

# Load the trained best model from disk.
best_model = joblib.load(BEST_MODEL_PATH)

# Load the standard scaler used during training.
scaler = joblib.load(SCALER_PATH)

# Construct a sample patient record using the requested feature values.
# This matches the exact patient data from the requirement.
sample_patient = {
    "age_years": 25,
    "gender": 1,
    "height": 170,
    "weight": 65,
    "BMI": 22.5,
    "ap_hi": 110,
    "ap_lo": 70,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

# Convert the sample patient record into a 2D NumPy array for scaler/model input.
feature_order = [
    "age_years",
    "gender",
    "height",
    "weight",
    "BMI",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
]

patient_array = np.array([[sample_patient[feature] for feature in feature_order]], dtype=float)

# Scale the input data using the loaded scaler.
scaled_patient = scaler.transform(patient_array)

# Make a prediction using the loaded best model.
prediction = best_model.predict(scaled_patient)[0]

# Use predict_proba() to get the probability for each class.
# proba[0] corresponds to class 0 (no heart disease), and proba[1] corresponds to class 1 (heart disease).
disease_probability = None
class_0_probability = None
class_1_probability = None
if hasattr(best_model, "predict_proba"):
    proba = best_model.predict_proba(scaled_patient)[0]
    class_0_probability = proba[0]
    class_1_probability = proba[1]
    disease_probability = class_1_probability

# Determine risk level based on the probability of heart disease (class 1).
# Risk is low when disease probability is under 0.30, medium between 0.30 and 0.70, and high when 0.70 or above.
if disease_probability is not None:
    if disease_probability >= 0.70:
        risk_level = "High Risk"
    elif disease_probability >= 0.30:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
else:
    # Fallback when probability is unavailable: use the predicted class.
    risk_level = "High Risk" if prediction == 1 else "Low Risk"

# Print results using the required output order.
print("Disease Probability", round(disease_probability, 4) if disease_probability is not None else "N/A")
print("Probability Class 0", round(class_0_probability, 4) if class_0_probability is not None else "N/A")
print("Probability Class 1", round(class_1_probability, 4) if class_1_probability is not None else "N/A")
print("Prediction", prediction)
print("Risk Level", risk_level)
