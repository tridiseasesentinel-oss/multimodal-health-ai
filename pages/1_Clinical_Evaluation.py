import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(layout="wide")

# 1. MODELS SAFE LOAD
def load_models():
    models = {}
    model_files = {"d_m": "d_model.pkl", "h_m": "h_model.pkl", "o_m": "o_model.pkl"}
    for key, filename in model_files.items():
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    models[key] = pickle.load(f)
            except:
                models[key] = None
        else:
            models[key] = None
    return models

models = load_models()

st.title("Clinical Evaluation AI")

# Check agar files nahi milin
if any(m is None for m in models.values()):
    st.error("❌ Error: Model files (.pkl) missing in the root directory. Files check karein.")
else:
    # INPUTS
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 1, 100, 25)
        weight = st.number_input("Weight (kg)", 10.0, 200.0, 60.0)
    with c2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        g_val = 1 if gender == "Male" else 0

    if st.button("Run Evaluation"):
        # Bypass feature name mismatch
        for m in models.values():
            if hasattr(m, 'get_booster'):
                m.get_booster().feature_names = None
        
        # Prediction logic
        try:
            d_res = models['d_m'].predict(np.array([[float(age), 22.0, 5.4, 98.0, 120.0, 195.0]]))
            h_res = models['h_m'].predict(np.array([[float(age), float(g_val), 120.0, 195.0, 72.0, 3.0]]))
            o_res = models['o_m'].predict(np.array([[float(age), float(g_val), 22.0]]))
            
            st.write(f"### Results:")
            st.success(f"Diabetes Risk: {d_res[0]}")
            st.success(f"Heart Risk: {h_res[0]}")
            st.success(f"Obesity Level: {o_res[0]}")
            
            # DOWNLOAD REQUIREMENT
            rep = f"Clinical Report\nDiabetes: {d_res[0]}\nHeart: {h_res[0]}\nObesity: {o_res[0]}"
            st.download_button("Download Report (TXT)", rep, "report.txt")
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
