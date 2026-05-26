import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import time

st.set_page_config(page_title="Global Health Evaluation", layout="wide")

# Premium Dynamic Tech UI Theme Styling
st.markdown("""
    <style>
    .block-heading {
        color: #1e3a8a; /* Deep Royal Blue */
        font-size: 24px;
        font-weight: 700;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 6px;
        margin-bottom: 20px;
    }
    .precaution-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 15px;
        border-radius: 6px;
        margin-top: 12px;
        font-size: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        width: 100% !important;
        padding: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# Cache loading for models safely
@st.cache_resource
def load_diagnostic_models():
    d = pickle.load(open("d_model.pkl", "rb"))
    h = pickle.load(open("h_model.pkl", "rb"))
    o = pickle.load(open("o_model.pkl", "rb"))
    return d, h, o

try:
    d_model, h_model, o_model = load_diagnostic_models()
except Exception as e:
    st.error("System configuration files are initializing. Please hold on.")

# ----------------- GLOBAL MULTILINGUAL TRANSLATION DICTIONARY -----------------
with st.sidebar:
    st.markdown("### 🌐 System Settings")
    lang = st.selectbox(
        "Select Portal Language / زبان منتخب کریں", 
        ["English", "Urdu"]
    )

# Comprehensive Multi-Language Pack
text_pack = {
    "English": {
        "title": "Personal Health Evaluation Form",
        "subtitle": "Please enter your measurements below to calculate your health metrics.",
        "h1": "1. Basic Information", "h2": "2. Health Vitals & Measurements", "h3": "3. Evaluation Results",
        "h4": "📋 Recommended Precautionary Measures",
        "name": "Full Name", "name_placeholder": "Enter full name here", "ref_id": "Reference ID / Case ID", "age": "Age (Years)",
        "weight": "Weight (kg)", "height_ft": "Height (Feet)", "height_in": "Height (Inches)", "bmi_msg": "Your calculated Body Mass Index (BMI) is",
        "hba1c": "Sugar Percentage - HbA1c (%)", "fpg": "Fasting Blood Sugar Level (mg/dL)",
        "bp": "Blood Pressure - Upper Reading (mmHg)", "hr": "Heart Rate / Pulse (BPM)", "chol": "Cholesterol Level (mg/dL)",
        "exercise": "Exercise Time per Week (Hours)", "gender": "Gender", "btn": "Run Evaluation & Generate Report",
        "db_lbl": "Diabetes Risk", "ht_lbl": "Heart Condition Risk", "ob_lbl": "Weight Classification",
        "pos": "Risk Detected (Positive)", "neg": "Normal / Clear", "high_r": "High Risk Detected", "low_r": "Normal / Low Risk",
        "success": "Analysis complete! Your digital clinical report is ready."
    },
    "Urdu": {
        "title": "شخصی صحت کی جانچ کا فارم",
        "subtitle": "صحت کے پیرامیٹرز کا حساب لگانے کے لیے براہ کرم نیچے اپنی پیمائش درج کریں۔",
        "h1": "1. بنیادی معلومات", "h2": "2. صحت کے اہم وائٹلز اور پیمائش", "h3": "3. تشخیص کے نتائج",
        "h4": "📋 تجویز کردہ احتیاطی تدابیر",
        "name": "پورا نام", "name_placeholder": "یہاں مریض کا نام درج کریں", "ref_id": "کیس آئی ڈی / حوالہ نمبر", "age": "عمر (سال)",
        "weight": "وزن (کلوگرام)", "height_ft": "قد (فٹ)", "height_in": "قد (انچ)", "bmi_msg": "آپ کا حساب کردہ باڈی ماس انڈیکس (BMI) یہ ہے",
        "hba1c": "شوگر کا تناسب - HbA1c (%)", "fpg": "فاسٹنگ بلڈ شوگر لیول (mg/dL)",
        "bp": "بلڈ پریشر - اوپر کی ریڈنگ (mmHg)", "hr": "دل کی دھڑکن کی رفتار (BPM)", "chol": "کولیسٹرول کی سطح (mg/dL)",
        "exercise": "ہر ہفتے ورزش کا وقت (گھنٹوں میں)", "gender": "جنس", "btn": "تشخیص چلائیں اور رپورٹ تیار کریں",
        "db_lbl": "ذیابیطس کا خطرہ (Diabetes)", "ht_lbl": "دل کی بیماری کا خطرہ (Heart)", "ob_lbl": "وزن کی درجہ بندی (Obesity)",
        "pos": "خطرہ موجود ہے (Positive)", "neg": "نارمل / محفوظ", "high_r": "زیادہ خطرہ موجود ہے", "low_r": "کم خطرہ / نارمل",
        "success": "تجزیہ مکمل ہو گیا ہے! آپ کی ڈیجیٹل طبی رپورٹ تیار ہے۔"
    }
}

t = text_pack[lang]

st.title(t["title"])
st.markdown(t["subtitle"])

# ----------------- SECTION 1: REGISTRATION -----------------
st.markdown(f"<div class='block-heading'>{t['h1']}</div>", unsafe_allow_html=True)

if 'auto_case_id' not in st.session_state:
    st.session_state.auto_case_id = f"CR-{time.strftime('%m%d')}-{str(int(time.time()))[-4:]}"

c1, c2, c3 = st.columns(3)
with c1:
    p_name = st.text_input(t["name"], value="", placeholder=t["name_placeholder"])
with c2:
    p_id = st.text_input(t["ref_id"], value=st.session_state.auto_case_id, disabled=True)
with c3:
    p_age = st.number_input(t["age"], min_value=1, max_value=120, value=25, step=1)

# ----------------- SECTION 2: VITALS & PHYSICAL METRICS -----------------
st.markdown(f"<br><div class='block-heading'>{t['h2']}</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input(t["weight"], min_value=10.0, max_value=250.0, value=60.0, step=0.5)
    
    # Feet & Inches Selection Components
    ht_c1, ht_c2 = st.columns(2)
    with ht_c1:
        ft_val = st.number_input(t["height_ft"], min_value=1, max_value=8, value=5, step=1)
    with ht_c2:
        in_val = st.number_input(t["height_in"], min_value=0, max_value=11, value=6, step=1)
        
    # Math metric conversion logic
    total_inches = (ft_val * 12) + in_val
    height_cm = total_inches * 2.54
    height_m = height_cm / 100.0
    bmi_calc = round(weight / (height_m ** 2), 2)
    st.info(f"{t['bmi_msg']}: **{bmi_calc}**")
    
    hba1c = st.number_input(t["hba1c"], min_value=3.0, max_value=15.0, value=5.4, step=0.1)
    fpg = st.number_input(t["fpg"], min_value=50, max_value=500, value=98, step=1)

with col2:
    systolic_bp = st.number_input(t["bp"], min_value=70, max_value=220, value=120, step=1)
    heart_rate = st.number_input(t["hr"], min_value=40, max_value=180, value=72, step=1)
    cholesterol = st.number_input(t["chol"], min_value=100, max_value=450, value=195, step=1)
    exercise_hours = st.number_input(t["exercise"], min_value=0, max_value=40, value=3, step=1)
    
    if lang == "Urdu":
        gender_options = ["مرد (Male)", "عورت (Female)"]
    else:
        gender_options = ["Male", "Female"]
        
    gender_selection = st.selectbox(t["gender"], gender_options)
    gender_num = 1 if gender_selection in ["Male", "مرد (Male)"] else 0

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- ACCURATE PREDICTION ENGINE -----------------
if st.button(t["btn"]):
    
    # DataFrame conversion matching the exact training columns for XGBoost structure alignment
    d_df = pd.DataFrame([[p_age, bmi_calc, hba1c, fpg, systolic_bp, cholesterol]], 
                        columns=['Age', 'BMI', 'HbA1c', 'FastingBloodSugar', 'SystolicBP', 'Cholesterol'])
                        
    h_df = pd.DataFrame([[p_age, gender_num, systolic_bp, cholesterol, heart_rate, exercise_hours]], 
                        columns=['Age', 'Gender', 'SystolicBP', 'Cholesterol', 'HeartRate', 'ExerciseHours'])
                        
    o_df = pd.DataFrame([[p_age, gender_num, bmi_calc]], 
                        columns=['Age', 'Gender', 'BMI'])
    
    try:
        # Step-by-step evaluation
        d_pred = d_model.predict(d_df)[0]
        h_pred = h_model.predict(h_df)[0]
        o_pred = o_model.predict(o_df)[0]
        
        # Safe extraction for prediction array types
        d_res = int(d_pred[0]) if hasattr(d_pred, '__len__') else int(d_pred)
        h_res = int(h_pred[0]) if hasattr(h_pred, '__len__') else int(h_pred)
        o_res = int(o_pred[0]) if hasattr(o_pred, '__len__') else int(o_pred)
        
        st.markdown(f"<br><div class='block-heading'>{t['h3']}</div>", unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(label=t["db_lbl"], value=t["pos"] if d_res == 1 else t["neg"])
        with res2:
            st.metric(label=t["ht_lbl"], value=t["high_r"] if h_res == 1 else t["low_r"])
        with res3:
            obesity_map = {0: "Underweight", 1: "Normal Weight", 2: "Overweight", 3: "Obese Class"}
            obesity_map_ur = {0: "کم وزن", 1: "نارمل وزن", 2: "زیادہ وزن", 3: "موٹاپا"}
            
            final_ob_val = obesity_map_ur.get(o_res, "نارمل وزن") if lang == "Urdu" else obesity_map.get(o_res, "Normal Weight")
            st.metric(label=t["ob_lbl"], value=final_ob_val)
            
        st.success(t["success"])
        
        # ----------------- PRECAUTIONARY MEASURES SECTION -----------------
        st.markdown(f"<br><div class='block-heading'>{t['h4']}</div>", unsafe_allow_html=True)
        
        # 1. Diabetes Precautions
        if d_res == 1:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>⚠️ <b>ذیابیطس کے لیے:</b> میٹھی اشیاء اور زیادہ کاربوہائیڈریٹس والی غذاؤں سے پرہیز کریں۔ روزانہ باقاعدگی سے 30 منٹ چہل قدمی کریں اور شوگر لیمل مانیٹر کریں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>⚠️ <b>For Diabetes:</b> Strictly limit sugar and high-carbohydrate foods. Engage in 30 minutes of daily physical walks and track blood glucose regularly.</div>", unsafe_allow_html=True)
        else:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>✅ <b>ذیابیطس کے لیے:</b> آپ کا بلڈ شوگر لیول محفوظ رینج میں ہے۔ متوازن اور فائبر سے بھرپور غذا کا استعمال جاری رکھیں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>✅ <b>For Diabetes:</b> Your glucose markers look normal. Continue maintaining a balanced, high-fiber dietary routine.</div>", unsafe_allow_html=True)

        # 2. Heart Precautions
        if h_res == 1:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>⚠️ <b>دل کی صحت کے لیے:</b> کھانے میں نمک اور چکنائی (Oily/Fried items) کا استعمال فوری کم کریں۔ بلڈ پریشر باقاعدگی سے چیک کریں اور ہلکی ورزش کریں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>⚠️ <b>For Heart Condition:</b> Reduce salt consumption and avoid fried/saturated fats. Regularly monitor blood pressure and practice light cardiovascular routines.</div>", unsafe_allow_html=True)
        else:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>✅ <b>دل کی صحت کے لیے:</b> دل کی دھڑکن اور بلڈ پریشر کے اشارے بہترین ہیں۔ فعال طرزِ زندگی برقرار رکھیں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>✅ <b>For Heart Condition:</b> Excellent cardiovascular vitals. Continue your low-sodium habits and regular exercise.</div>", unsafe_allow_html=True)

        # 3. Obesity Precautions
        if o_res >= 2:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>⚠️ <b>وزن کے انتظام کے لیے:</b> فاسٹ فوڈ اور سافٹ ڈرنکس سے مکمل پرہیز کریں۔ پورشن کنٹرول (Portion Control) فارمولے پر عمل کریں اور روزانہ ورزش کریں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>⚠️ <b>For Weight Management:</b> Avoid processed junk foods and high-calorie soft drinks. Apply strict portion control and stay active daily.</div>", unsafe_allow_html=True)
        else:
            if lang == "Urdu":
                st.markdown("<div class='precaution-box'>✅ <b>وزن کے انتظام کے لیے:</b> آپ کا باڈی ماس انڈیکس (BMI) بالکل یاسر زون میں ہے۔ صحت بخش عادات جاری رکھیں۔</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='precaution-box'>✅ <b>For Weight Management:</b> Your Body Mass Index is within the ideal healthy range. Maintain this active physical lifestyle.</div>", unsafe_allow_html=True)

    except Exception as error:
        st.error(f"Inference error details: {str(error)}")
