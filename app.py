import streamlit as st
import pandas as pd
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")


st.title("HEART STROKE PREDICTION BY ANUJ KASHYAP ")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("Sex",["M",'F'])
chest_pain = st.selectbox("Chest pain type",["ATA",'NAP','TA','ASY'])
resting_BP = st.number_input("Resting Blood Presior(mm Hg)", 80,200,120)
ol = st.number_input("Cholestrol (mg / dl)", 100,600,200)
fasting_ds = st.selectbox("Fasting Blood suger > 120 mg/dl",[0,1])
resting_ecg = st.selectbox("Resting ECG", ["Normal","ST","LVH"])
max_hr = st.slider("max heart rate", 60,220,150)
excersice_enginia = st.selectbox("Excersice-Indused Angina",["Y","N"])
old_peak = st.slider("OLD peak (ST deprission)", 0.0,6.0,1.0)
st_slope = st.selectbox("ST Slope", ["UP","FLAT","DOWN"])

# When Predict is clicked
if st.button("Predict"):

    # Create a raw input dictionary
    raw_input = {
        'Age': age,
        'RestingBP': resting_BP,
        'Cholesterol':ol,
        'FastingBS': fasting_ds,
        'MaxHR': max_hr,
        'Oldpeak': old_peak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + excersice_enginia: 1,
        'ST_Slope_' + st_slope: 1
    }

 # Create input dataframe
    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
