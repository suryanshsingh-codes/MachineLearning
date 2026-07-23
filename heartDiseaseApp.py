import streamlit as st   # Streamlit for building interactive UI
import pandas as pd      # Pandas for handling tabular data
import joblib            # Joblib for loading saved ML models and objects

# --- Load trained model and preprocessing assets ---


model = joblib.load(r"C:\Users\Suryansh Singh\Desktop\Machine Learning\HeartDiseaseFinalApp\knn_heart_model.pkl")
scaler = joblib.load(r"C:\Users\Suryansh Singh\Desktop\Machine Learning\HeartDiseaseFinalApp\heart_scaler.pkl")
expected_columns = joblib.load(r"C:\Users\Suryansh Singh\Desktop\Machine Learning\HeartDiseaseFinalApp\heart_columns.pkl")


# --- Frontend UI setup ---
st.title("Heart Stroke Prediction")  
st.markdown("Provide the following details to check your heart stroke risk:")

# --- Collect user input via Streamlit widgets ---
age = st.slider("Age", 18, 100, 40)                     # Slider: age range 18–100, default 40
sex = st.selectbox("Sex", ["M", "F"])                   # Dropdown: Male/Female
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# --- Prediction workflow ---
if st.button("Predict"):

    # Step 1: Build raw input dictionary (binary encoding for categorical features)
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    # Step 2: Convert dictionary to DataFrame
    input_df = pd.DataFrame([raw_input])

    # Step 3: Ensure all expected columns exist (fill missing with 0)
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Step 4: Reorder columns to match training schema
    input_df = input_df[expected_columns]

    # Step 5: Scale input features
    scaled_input = scaler.transform(input_df)

    # Step 6: Generate prediction
    prediction = model.predict(scaled_input)[0]

    # Step 7: Display result with appropriate UI feedback
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
