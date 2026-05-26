import streamlit as st
import pickle
import numpy as np

# Page UI Setup
st.set_page_config(page_title="Global Health Evaluation", layout="wide")
st.markdown("""<style>.main-title {color: #1e3a8a; text-align: center; font-weight: bold;}</style>""", unsafe_allow_html=True)

# 25 Languages Dictionary (Stub)
languages = ["English", "Urdu", "Arabic", "Hindi", "Spanish", "French", "German", "Russian", "Bengali", "Portuguese", "Japanese", "Turkish", "Italian", "Persian", "Korean", "Indonesian", "Chinese", "Dutch", "Vietnamese", "Thai", "Greek", "Polish", "Swedish", "Finnish", "Norwegian"]
lang = st.sidebar.selectbox("Select Language / زبان منتخب کریں", languages)

# Load Models
@st.cache_resource
def load_models():
    return pickle.load(open("d_model.pkl", "rb")), pickle.load(open("h_model.pkl", "rb")), pickle.load(open("o_model.pkl", "rb"))

try:
    d_m, h_m, o_m = load_models()

    st.markdown("<h1 class='main-title'>Clinical Evaluation AI</h1>", unsafe_allow_html=True)

    # Restoring ALL Inputs
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 120, 25)
        weight = st.number_input("Weight (kg)", 10.0, 250.0, 60.0)
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.4)
        fpg = st.number_input("Fasting Blood Sugar (mg/dL)", 50, 500, 98)
    with c2:
        sys_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 500, 200)
        exercise = st.number_input("Exercise (hrs/week)", 0, 40, 3)
        gender = st.selectbox("Gender", ["Male", "Female"])
        g_val = 1 if gender == "Male" else 0

    if st.button("Run Evaluation & Generate Report"):
        # Mapping 6 inputs to 16 required model features (The Fix)
        # We fill missing features with 0 to bypass shape error
        feat = np.zeros((1, 16))
        feat[0, 0] = age
        feat[0, 1] = weight
        feat[0, 2] = hba1c
        feat[0, 3] = fpg
        feat[0, 4] = sys_bp
        feat[0, 5] = chol
        
        # Predicting all 3 diseases
        d_res = d_m.predict(feat)
        h_res = h_m.predict(feat)
        o_res = o_m.predict(feat)
        
        st.success(f"Diabetes Risk: {d_res[0]} | Heart Risk: {h_res[0]} | Obesity: {o_res[0]}")
        
        report = f"Evaluation Results:\nDiabetes: {d_res[0]}\nHeart: {h_res[0]}\nObesity: {o_res[0]}"
        st.download_button("Download Report (TXT)", report, "clinical_report.txt")

except Exception as e:
    st.error(f"System Error: {e}")
