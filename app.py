import streamlit as st
import pickle
import numpy as np
import os

# 1. Advanced Structural Page Configurations
st.set_page_config(
    page_title="Multimodal Health AI Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Premium Professional UI Theme Customization (CSS Injection)
st.markdown("""
    <style>
    /* Main Layout Background Styling */
    .main { background-color: #f7f9fc; }
    
    /* Elegant Clean Metrics Panels */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        border-top: 5px solid #1e3d59;
        margin-bottom: 20px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 6px 18px rgba(0, 0, 0, 0.08);
    }
    
    /* Typography Overrides */
    h1 { color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-weight: 700; text-align: center; }
    h3 { color: #17b978; font-weight: 600; margin-bottom: 25px; }
    h4 { color: #1e3d59; font-weight: 600; margin-top: 0; margin-bottom: 10px; text-transform: uppercase; font-size: 14px; letter-spacing: 0.5px; }
    
    /* Button Optimization */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0px 4px 10px rgba(23, 185, 120, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Banner Title
st.markdown("<h1>🏥 Integrated Multimodal Clinical Decision Support System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6c757d; font-size: 15px;'>Diagnostic Analytics & Predictive Expert Framework Engine</p>", unsafe_allow_html=True)
st.markdown("---")

# 3. Dynamic Patient Demographic Data Inputs
st.markdown("### 📋 Patient Clinical Demographics & Physiological Metrics")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=45, step=1)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.4, step=0.1)
    gender = st.selectbox("Biological Sex", options=["Female", "Male"])
    high_bp = st.selectbox("Diagnosed High BP History ?", options=["No", "Yes"])

with col2:
    high_chol = st.selectbox("Diagnosed High Cholesterol History ?", options=["No", "Yes"])
    smoker = st.selectbox("Tobacco Smoking Profile Status?", options=["No", "Yes"])
    exercise = st.selectbox("Regular Structured Physical Activity?", options=["Yes", "No"])

# Explicit Matrix Encodings for Model Pipelines
gender_num = 1 if gender == "Male" else 0
bp_num = 1 if high_bp == "Yes" else 0
chol_num = 1 if high_chol == "Yes" else 0
smoke_num = 1 if smoker == "Yes" else 0
exe_num = 1 if exercise == "Yes" else 0

st.markdown("<br>", unsafe_allow_html=True)

# 4. Evaluation and Diagnostics Execution Pipeline
if st.button("Generate Diagnostic Multimodal Evaluation"):
    
    # EXACT FILENAMES MAPPED ACCORDING TO YOUR WORKSPACE DIRECTORY
    diabetes_file = 'd_model.pkl'
    heart_file = 'h_model.pkl'
    obesity_file = 'o_model.pkl'
    
    # Check if files exist before trying to load them
    if not (os.path.exists(diabetes_file) and os.path.exists(heart_file) and os.path.exists(obesity_file)):
        st.error("❌ Execution Error: Verified model artifacts are missing from the current system context. Ensure d_model.pkl, h_model.pkl, and o_model.pkl are available.")
    else:
        try:
            # Safe binary reading block
            m_d = pickle.load(open(diabetes_file, 'rb'))
            m_h = pickle.load(open(heart_file, 'rb'))
            m_o = pickle.load(open(obesity_file, 'rb'))
            
            # Pad features array matrices exactly matching training features dimensions
            diabetes_features = np.array([bp_num, chol_num, 1, bmi, smoke_num, 0, 0, exe_num, 0, 1, 0, 3, 0, gender_num, age, 6]).reshape(1, -1)
            heart_features = np.array([age, gender_num, 130, 220, exe_num, smoke_num, 0, 0, bmi, bp_num, 0, 0, 1, 1, 7, 1, 200, 100, 5, 12]).reshape(1, -1)
            obesity_features = np.array([gender_num, age, 1.75, 75.0, 1, 1, 2, 3, 1, smoke_num, 2, 0, 1, 1, 1, 2]).reshape(1, -1)
            
            # Predict
            d_pred = int(m_d.predict(diabetes_features)[0])
            h_pred = int(m_h.predict(heart_features)[0])
            o_pred = int(m_o.predict(obesity_features)[0])
            
            # Professional Medical Stratification Mapping
            d_map = {0: "Healthy (No Diabetes Indication)", 1: "Pre-Diabetes Diagnostic Stratum", 2: "Diabetes Clinical Stage II"}
            h_map = {0: "Healthy Cardiovascular Stratum", 1: "Cardiovascular Disease Risk Detected"}
            o_map = {0: "Underweight Stratum", 1: "Normal Physiological Weight", 2: "Obesity Type I", 3: "Obesity Type II", 4: "Obesity Type III", 5: "Overweight Class I", 6: "Overweight Class II"}
            
            # Render Modern Assessment Dashboard Grid Panel Layout
            st.markdown("### 📊 Multimodal Clinical Assessment Summary")
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                st.markdown(f"""
                    <div class='metric-card' style='border-top-color: #17b978;'>
                        <h4>Metabolic Profile</h4>
                        <p style='font-size:18px; color:#1e3d59; margin:0;'><b>{d_map.get(d_pred, 'Unclassified')}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                st.markdown(f"""
                    <div class='metric-card' style='border-top-color: #ff4b4b;'>
                        <h4>Cardiovascular Profile</h4>
                        <p style='font-size:18px; color:#1e3d59; margin:0;'><b>{h_map.get(h_pred, 'Unclassified')}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            with res_col3:
                st.markdown(f"""
                    <div class='metric-card' style='border-top-color: #f1c40f;'>
                        <h4>Adiposity Stratification</h4>
                        <p style='font-size:18px; color:#1e3d59; margin:0;'><b>{o_map.get(o_pred, 'Unclassified')}</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
            # 5. Dynamic Evidence-Based Precaution System
            st.markdown("---")
            st.markdown("### 📋 Personalized Preventive Framework Directive")
            
            flags_triggered = False
            if d_pred > 0:
                st.warning("⚠️ **Endocrine Management Protocol:** Restrict refined carbohydrate intake, optimize soluble dietary fiber matrices, and establish glycemic threshold tracking charts.")
                flags_triggered = True
            if h_pred == 1:
                st.error("⚠️ **Cardiovascular Preservation Protocol:** Enforce standard sodium reduction therapies (<1,500mg/day) and schedule routine diagnostic echocardiograms.")
                flags_triggered = True
            if o_pred >= 2:
                st.info("⚠️ **Metabolic Intervention Protocol:** Initiate structured caloric-deficit target adjustments and evaluate baseline serum lipid profiling parameters.")
                flags_triggered = True
                
            if not flags_triggered:
                st.success("✅ **General Preventative Maintenance Protocol:** Patient biological markers demonstrate optimal equilibrium profile. Maintain routine longitudinal health evaluations.")

        except Exception as e:
            st.error(f"❌ Internal Pipeline Exception Error: {e}")
