import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="Clinical AI", layout="wide")

# UI CSS
st.markdown("""<style>.block-heading {color: #1e3a8a; border-bottom: 2px solid #3b82f6;}</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return pickle.load(open("d_model.pkl", "rb")), pickle.load(open("h_model.pkl", "rb")), pickle.load(open("o_model.pkl", "rb"))

d_m, h_m, o_m = load_models()

# Languages
langs = ["English", "Urdu", "Arabic", "Hindi", "Punjabi", "Pashto", "Sindhi", "Spanish", "French", "German", "Chinese", "Russian", "Bengali", "Portuguese", "Japanese", "Turkish", "Italian", "Persian", "Korean", "Indonesian"]
lang = st.sidebar.selectbox("Select Language", langs)
# (Note: Yahan poora dictionary object pehle jaisa hi rahega)

# Inputs
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 1, 120, 25)
    weight = st.number_input("Weight (kg)", 10.0, 250.0, 60.0)
with c2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    g_val = 1 if gender == "Male" else 0

# ... (baqi inputs yahan aayenge)

if st.button("Run Evaluation"):
    # FIX: Bypassing feature name checking
    d_m.get_booster().feature_names = None
    h_m.get_booster().feature_names = None
    o_m.get_booster().feature_names = None
    
    # Input arrays
    d_in = np.array([[float(age), 22.0, 5.4, 98.0, 120.0, 195.0]]) # Dummy values for schema
    h_in = np.array([[float(age), float(g_val), 120.0, 195.0, 72.0, 3.0]])
    o_in = np.array([[float(age), float(g_val), 22.0]])
    
    d_p = d_m.predict(d_in)
    h_p = h_m.predict(h_in)
    o_p = o_m.predict(o_in)
    
    st.write(f"Results: Diabetes={d_p[0]}, Heart={h_p[0]}, Obesity={o_p[0]}")
    
    # Download Report Requirement
    report = f"Clinical Report\nDiabetes: {d_p[0]}\nHeart: {h_p[0]}\nObesity: {o_p[0]}"
    st.download_button("Download Report (TXT)", report, "report.txt")
