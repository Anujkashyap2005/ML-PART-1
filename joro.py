import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "KNN_heart.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
expected_columns = joblib.load(BASE_DIR / "columns.pkl")

expected_columns = list(expected_columns)
scaler_columns = list(scaler.feature_names_in_)

if len(expected_columns) != model.n_features_in_:
    st.error(
        f"columns.pkl contains {len(expected_columns)} columns, "
        f"but the model expects {model.n_features_in_}."
    )
    st.stop()

st.title("Heart Disease Prediction BY ANUJ KASHYAP")
st.write("Enter the patient details:")

age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    values = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        f"Sex_{sex}": 1,
        f"ChestPainType_{chest_pain}": 1,
        f"RestingECG_{resting_ecg}": 1,
        f"ExerciseAngina_{exercise_angina}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    # Create all columns expected by the model
    input_df = pd.DataFrame([values])
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    try:
        # Scale only the five columns used to fit scaler.pkl
        numeric_df = input_df[scaler_columns]
        numeric_df = input_df[scaler_columns]
        input_df = input_df.astype(float)
        input_df.loc[:, scaler_columns] = scaler.transform(numeric_df)

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

    except Exception as error:
        st.error(f"Prediction error: {error}")