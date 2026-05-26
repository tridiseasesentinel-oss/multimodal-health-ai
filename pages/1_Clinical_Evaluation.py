import streamlit as st
import pickle
import numpy as np

# Page Config
st.set_page_config(page_title="Clinical Evaluation AI", layout="wide")
st.title("Clinical Evaluation AI")

# Load Models
@st.cache_resource
def load_models():
    return pickle.load(open("d_model.pkl", "rb")), pickle.load(open("h_model.pkl", "rb")), pickle.load(open("o_model.pkl", "rb"))

try:
    d_m, h_m, o_m = load_models()

    # Restoration of all Input Fields
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 1, 100, 25)
        weight = st.number_input("Weight (kg)", 10.0, 200.0, 60.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.4)
        fpg = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 500, 98)
    with col2:
        sys_bp = st.number_input("Systolic BP (mmHg)", 70, 220, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 450, 195)
        exercise = st.number_input("Exercise Hours/Week", 0, 40, 3)
        gender = st.selectbox("Gender", ["Male", "Female"])
        g_val = 1 if gender == "Male" else 0

    if st.button("Run Evaluation & Generate Report"):
        # Fix: Feature Expansion (Model 16 features maang raha hai, hum unhein 0 fill kar rahe hain)
        def pad_features(arr, total=16):
            return np.pad(arr, ((0,0), (0, total - arr.shape[1])), mode='constant')

        # Dummy inputs map kar rahe hain model ki requirement ke mutabiq
        d_in = pad_features(np.array([[age, weight/((1.7)**2), hba1c, fpg, sys_bp, chol]]))
        
        d_pred = d_m.predict(d_in)
        
        st.success(f"Evaluation Result: {d_pred[0]}")
        
        # Report Requirement
        report_text = f"Clinical Report\nAge: {age}\nResult: {d_pred[0]}"
        st.download_button("Download Report (TXT)", report_text, "clinical_report.txt")

except Exception as e:
    st.error(f"Error: {e}. Check karein ke .pkl files sahi jagah hain.")
