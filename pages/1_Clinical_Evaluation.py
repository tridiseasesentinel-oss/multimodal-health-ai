import streamlit as st
import pickle
import numpy as np

# Page UI Setup
st.set_page_config(page_title="Global Health Evaluation", layout="wide")
st.markdown("""<style>
    .main-title {color: #1e3a8a; text-align: center; font-weight: bold;}
    .report-box {background-color: #f0f9ff; padding: 20px; border-radius: 10px; border: 1px solid #bae6fd;}
</style>""", unsafe_allow_html=True)

# Language Settings
langs = ["English", "Urdu", "Arabic", "Hindi", "French"] # Add remaining
lang = st.sidebar.selectbox("Select Language / زبان منتخب کریں", langs)

st.markdown("<h1 class='main-title'>Clinical Evaluation AI</h1>", unsafe_allow_html=True)

# Models Load
@st.cache_resource
def load_models():
    return pickle.load(open("d_model.pkl", "rb")), pickle.load(open("h_model.pkl", "rb")), pickle.load(open("o_model.pkl", "rb"))

try:
    d_m, h_m, o_m = load_models()
    
    # Patient Info & Inputs
    c1, c2 = st.columns(2)
    with c1:
        p_name = st.text_input("Patient Name")
        p_id = st.text_input("Patient ID")
        age = st.number_input("Age", 1, 120, 25)
        weight = st.number_input("Weight (kg)", 10.0, 250.0, 60.0)
    with c2:
        sys_bp = st.number_input("Systolic BP (mmHg)", 80, 250, 120)
        chol = st.number_input("Cholesterol (mg/dL)", 100, 500, 200)
        gender = st.selectbox("Gender", ["Male", "Female"])
        g_val = 1 if gender == "Male" else 0

    if st.button("Run Evaluation & Generate Report"):
        # Fix: Using exact training features sequence (16 features)
        # Sequence: [Age, BMI, HbA1c, FPG, SBP, Chol, Gender, ...others]
        features = np.zeros((1, 16))
        features[0, 0] = age
        features[0, 1] = weight / 3.0 # Simplified BMI
        features[0, 4] = sys_bp
        features[0, 5] = chol
        features[0, 6] = g_val
        
        d_res = d_m.predict(features)[0]
        h_res = h_m.predict(features)[0]
        
        # Display Results with Decoration
        st.markdown("<div class='report-box'>", unsafe_allow_html=True)
        st.subheader(f"Report for: {p_name} (ID: {p_id})")
        st.metric("Diabetes Risk", "High" if d_res == 1 else "Low")
        st.metric("Heart Risk", "High" if h_res == 1 else "Low")
        
        # Precautionary Measures
        st.write("### 🩺 Precautionary Measures")
        if d_res == 1:
            st.warning("- Maintain a balanced diet.\n- Regular blood sugar monitoring.\n- Avoid sugary beverages.")
        if h_res == 1:
            st.error("- Reduce sodium intake.\n- Daily 30 min exercise.\n- Avoid stress.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Download
        st.download_button("Download Report (TXT)", f"Name: {p_name}\nID: {p_id}\nDiabetes: {d_res}", "report.txt")

except Exception as e:
    st.error(f"Error: {e}")
