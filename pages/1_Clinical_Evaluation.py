import streamlit as st
import pickle
import numpy as np
import os

st.set_page_config(page_title="Health Check Evaluation", layout="wide")

# Custom UI CSS Styling for Buttons & Alerts
st.markdown("""
    <style>
    .block-heading {
        color: #1a365d;
        font-size: 22px;
        font-weight: 600;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    div.stButton > button:first-child {
        background-color: #1a365d !important; /* Navy Blue Theme Button */
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 12px !important;
        border-radius: 6px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Loading ML Models safely
@st.cache_resource
def load_diagnostic_models():
    d = pickle.load(open("d_model.pkl", "rb"))
    h = pickle.load(open("h_model.pkl", "rb"))
    o = pickle.load(open("o_model.pkl", "rb"))
    return d, h, o

try:
    d_model, h_model, o_model = load_diagnostic_models()
except Exception as e:
    st.error("Model configurations or pickle pathways are currently initializing. Please verify workspace files.")

st.title("Patient Health Information Form")
st.markdown("Please fill out the general vitals below to evaluate current health status metrics.")

# ----------------- SECTION 1: USER DETAILS -----------------
st.markdown("<div class='block-heading'>1. Patient Identification Records</div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    p_name = st.text_input("Patient Full Name (Mareez Ka Naam)", "Shazeen Amjad")
with c2:
    p_id = st.text_input("Case ID / File Number", "CR-2026-0892")
with c3:
    p_age = st.number_input("Age (Umar) - Years", min_value=1, max_value=120, value=20, step=1)

# ----------------- SECTION 2: HEALTH METRICS -----------------
st.markdown("<br><div class='block-heading'>2. General Health Vitals & Measurements</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input("Weight in Kilograms (Wazan - kg)", min_value=10.0, max_value=250.0, value=60.0, step=0.5)
    height_cm = st.number_input("Height in Centimeters (Kadd - cm)", min_value=50.0, max_value=250.0, value=165.0, step=0.5)
    
    # Automatic BMI calculator so the user doesn't have to compute it manually
    height_m = height_cm / 100.0
    bmi_calc = round(weight / (height_m ** 2), 2)
    st.info(f"Calculated Body Mass Index (BMI) is: **{bmi_calc}**")
    
    hba1c = st.number_input("HbA1c Percentage / Average 3-Month Sugar (Sugar Test %)", min_value=3.0, max_value=15.0, value=5.4, step=0.1)
    fpg = st.number_input("Fasting Blood Sugar / Glucose Level (mg/dL)", min_value=50, max_value=500, value=98, step=1)

with col2:
    systolic_bp = st.number_input("Systolic Blood Pressure / Upper BP Reading (mmHg)", min_value=70, max_value=220, value=120, step=1)
    heart_rate = st.number_input("Resting Heart Rate / Pulse (Dil ki Dharkan - BPM)", min_value=40, max_value=180, value=72, step=1)
    cholesterol = st.number_input("Total Serum Cholesterol Level (mg/dL)", min_value=100, max_value=450, value=195, step=1)
    
    exercise_hours = st.number_input("Weekly Physical Activity / Exercise Time (Hours)", min_value=0, max_value=40, value=3, step=1)
    
    gender_selection = st.selectbox("Biological Sex (Jins)", ["Male (Mard)", "Female (Aurat)"])
    gender_num = 1 if gender_selection == "Male (Mard)" else 0

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- INFERENCE PIPELINE -----------------
if st.button("Run Diagnostic Evaluation & Generate Report"):
    
    # Formulating precise feature vectors matching model structures
    # Diabetes Input Features
    d_input = np.array([[p_age, bmi_calc, hba1c, fpg, systolic_bp, cholesterol]])
    
    # Heart Condition Input Features
    h_input = np.array([[p_age, gender_num, systolic_bp, cholesterol, heart_rate, exercise_hours]])
    
    # Obesity Category Input Features
    o_input = np.array([[p_age, gender_num, bmi_calc]])
    
    try:
        # Running ML inference matrix
        d_pred = d_model.predict(d_input)[0]
        h_pred = h_model.predict(h_input)[0]
        o_pred = o_model.predict(o_input)[0]
        
        st.markdown("<br><div class='block-heading'>3. Diagnostic Evaluation Outputs</div>", unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(label="Diabetes Evaluation", value="Positive / Risk Detected" if d_pred == 1 else "Normal / Clear")
        with res2:
            st.metric(label="Cardiovascular Risk Status", value="High Risk Condition" if h_pred == 1 else "Stable / Low Risk")
        with res3:
            # Map index vectors back to categorical parameters for clarity
            obesity_map = {0: "Underweight", 1: "Normal Weight", 2: "Overweight", 3: "Obese Class"}
            st.metric(label="Adiposity Status Category", value=obesity_map.get(o_pred, "Evaluated"))
            
        st.success("Analysis complete! You can now compile the standardized clinical report metrics.")
        
    except Exception as error:
        st.error(f"Inference process halted. Technical details: {str(error)}")
