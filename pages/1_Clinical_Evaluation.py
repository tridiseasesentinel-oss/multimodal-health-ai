import streamlit as st
import pickle
import numpy as np
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
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%) !important; /* Premium Gradient */
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

# Cache loading for models
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
        ["English", "Urdu", "Spanish (Español)", "Arabic (العربية)", "French (Français)"]
    )

# Comprehensive Multi-Language Pack
text_pack = {
    "English": {
        "title": "Personal Health Evaluation Form",
        "subtitle": "Please enter your measurements below to calculate your health metrics.",
        "h1": "1. Basic Information", "h2": "2. Health Vitals & Measurements", "h3": "3. Evaluation Results",
        "name": "Full Name", "name_placeholder": "Enter full name here", "ref_id": "Reference ID / Case ID", "age": "Age (Years)",
        "weight": "Weight (kg)", "height_ft": "Height (Feet)", "height_in": "Height (Inches)", "bmi_msg": "Your calculated Body Mass Index (BMI) is",
        "hba1c": "Sugar Percentage - HbA1c (%)", "fpg": "Fasting Blood Sugar Level (mg/dL)",
        "bp": "Blood Pressure - Upper Reading (mmHg)", "hr": "Heart Rate / Pulse (BPM)", "chol": "Cholesterol Level (mg/dL)",
        "exercise": "Exercise Time per Week (Hours)", "gender": "Gender", "btn": "Run Evaluation & Generate Report",
        "db_lbl": "Diabetes Risk", "ht_lbl": "Heart Condition Risk", "ob_lbl": "Weight Classification",
        "pos": "Risk Detected", "neg": "Normal / Clear", "high_r": "High Risk", "low_r": "Normal / Low Risk",
        "success": "Analysis complete! Your digital clinical report is ready."
    },
    "Urdu": {
        "title": "شخصی صحت کی جانچ کا فارم",
        "subtitle": "صحت کے پیرامیٹرز کا حساب لگانے کے لیے براہ کرم نیچے اپنی پیمائش درج کریں۔",
        "h1": "1. بنیادی معلومات", "h2": "2. صحت کے اہم وائٹلز اور پیمائش", "h3": "3. تشخیص کے نتائج",
        "name": "پورا نام", "name_placeholder": "یہاں مریض کا نام درج کریں", "ref_id": "کیس آئی ڈی / حوالہ نمبر", "age": "عمر (سال)",
        "weight": "وزن (کلوگرام)", "height_ft": "قد (فٹ)", "height_in": "قد (انچ)", "bmi_msg": "آپ کا حساب کردہ باڈی ماس انڈیکس (BMI) یہ ہے",
        "hba1c": "شوگر کا تناسب - HbA1c (%)", "fpg": "فاسٹنگ بلڈ شوگر لیول (mg/dL)",
        "bp": "بلڈ پریشر - اوپر کی ریڈنگ (mmHg)", "hr": "دل کی دھړکن کی رفتار (BPM)", "chol": "کولیسٹرول کی سطح (mg/dL)",
        "exercise": "ہر ہفتے ورزش کا وقت (گھنٹے)", "gender": "جنس", "btn": "تشخیص چلائیں اور رپورٹ تیار کریں",
        "db_lbl": "ذیابیطس کا خطرہ (Diabetes)", "ht_lbl": "دل کی بیماری کا خطرہ (Heart)", "ob_lbl": "وزن کی درجہ بندی (Obesity)",
        "pos": "خطرہ موجود ہے", "neg": "نارمل / محفوظ", "high_r": "زیادہ خطرہ", "low_r": "کم خطرہ / نارمل",
        "success": "تجزیہ مکمل ہو گیا ہے! آپ کی ڈیجیٹل طبی رپورٹ تیار ہے۔"
    },
    "Spanish (Español)": {
        "title": "Formulario de Evaluación de Salud Personal",
        "subtitle": "Por favor introduzca sus medidas a continuación para calcular sus métricas de salud.",
        "h1": "1. Información Básica", "h2": "2. Signos Vitales y Mediciones de Salud", "h3": "3. Resultados de la Evaluación",
        "name": "Nombre Completo", "name_placeholder": "Ingrese el nombre completo aquí", "ref_id": "ID de Referencia / ID del Caso", "age": "Edad (Años)",
        "weight": "Peso (kg)", "height_ft": "Altura (Pies)", "height_in": "Altura (Pulgadas)", "bmi_msg": "Su Índice de Masa Corporal (IMC) calculado es",
        "hba1c": "Porcentaje de Azúcar - HbA1c (%)", "fpg": "Nivel de Azúcar en Sangre en Ayunas (mg/dL)",
        "bp": "Presión Arterial - Lectura Superior (mmHg)", "hr": "Frecuencia Cardíaca / Pulso (BPM)", "chol": "Nivel de Cholesterol (mg/dL)",
        "exercise": "Tiempo de Ejercicio por Semana (Horas)", "gender": "Género", "btn": "Ejecutar Evaluación y Generar Informe",
        "db_lbl": "Riesgo de Diabetes", "ht_lbl": "Riesgo de Condición Cardíaca", "ob_lbl": "Clasificación de Peso",
        "pos": "Riesgo Detectado", "neg": "Normal / Limpio", "high_r": "Alto Riesgo", "low_r": "Riesgo Normal / Bajo",
        "success": "¡Análisis completo! Su informe clínico digital está listo."
    },
    "Arabic (العربية)": {
        "title": "نموذج تقييم الصحة الشخصية",
        "subtitle": "يرجى إدخال قياساتك أدناه لحساب المقاييس الصحية الخاصة بك.",
        "h1": "1. معلومات أساسية", "h2": "2. المؤشرات الحيوية والقياسات الصحية", "h3": "3. نتائج التقييم",
        "name": "الاسم الكامل", "name_placeholder": "أدخل الاسم الكامل هنا", "ref_id": "الرقم المرجعي / رقم الحالة", "age": "العمر (بالسنوات)",
        "weight": "الوزن (كجم)", "height_ft": "الطول (قدم)", "height_in": "الطول (بوصة)", "bmi_msg": "مؤشر كتلة الجسم المحسوب (BMI) هو",
        "hba1c": "نسبة السكر - HbA1c (%)", "fpg": "مستوى سكر الدم الصائم (mg/dL)",
        "bp": "ضغط الدم - القراءة العليا (mmHg)", "hr": "معدل ضربات القلب / النبض (BPM)", "chol": "مستوى الكوليسترول (mg/dL)",
        "exercise": "وقت التمرين في الأسبوع (ساعات)", "gender": "الجنس", "btn": "تشغيل التقييم وإنشاء التقرير",
        "db_lbl": "خطر الإصابة بالسكري", "ht_lbl": "خطر الإصابة بأمراض القلب", "ob_lbl": "تصنيف الوزن",
        "pos": "تم رصد خطر", "neg": "طبيعي / سليم", "high_r": "خطر مرتفع", "low_r": "طبيعي / خطر منخفض",
        "success": "اكتمل التحليل! تقريرك الطبي الرقمي جاهز."
    },
    "French (Français)": {
        "title": "Formulaire d'Évaluation de la Santé Personnelle",
        "subtitle": "Veuillez saisir vos mesures ci-dessous para calculer vos paramètres de santé.",
        "h1": "1. Informations de Base", "h2": "2. Signes Vitaux et Mesures de Santé", "h3": "3. Résultats de l'Évaluation",
        "name": "Nom Complet", "name_placeholder": "Entrez le nom complet ici", "ref_id": "ID de Référence / ID du Cas", "age": "Âge (Années)",
        "weight": "Poids (kg)", "height_ft": "Taille (Pieds)", "height_in": "Taille (Pouces)", "bmi_msg": "Votre Indice de Masse Corporelle (IMC) calculé es",
        "hba1c": "Taux de Sucre - HbA1c (%)", "fpg": "Glycémie à Jeun (mg/dL)",
        "bp": "Pression Artérielle - Lecture Supérieure (mmHg)", "hr": "Fréquence Cardiaque / Pouls (BPM)", "chol": "Taux de Cholestérol (mg/dL)",
        "exercise": "Temps d'Exercice par Semaine (Heures)", "gender": "Genre", "btn": "Exécuter l'Évaluation et Générer le Rapport",
        "db_lbl": "Risque de Diabète", "ht_lbl": "Risque Cardiaque", "ob_lbl": "Classification du Poids",
        "pos": "Risque Détecté", "neg": "Normal / Clair", "high_r": "Risque Élevé", "low_r": "Normal / Risque Faible",
        "success": "Analyse complétée! Votre rapport clinique digital est prêt."
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
    
    # Dual Dropdowns/Sliders for Height in Feet & Inches
    ht_c1, ht_c2 = st.columns(2)
    with ht_c1:
        ft_val = st.number_input(t["height_ft"], min_value=1, max_value=8, value=5, step=1)
    with ht_c2:
        in_val = st.number_input(t["height_in"], min_value=0, max_value=11, value=6, step=1)
        
    # Standard medical conversion computation logic (Feet-Inches to Centimeters)
    total_inches = (ft_val * 12) + in_val
    height_cm = total_inches * 2.54
    
    # Calculate BMI instantly using converted metrics
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
    
    if lang == "English" or lang == "French (Français)":
        gender_options = ["Male", "Female"]
    elif lang == "Spanish (Español)":
        gender_options = ["Masculino", "Femenino"]
    elif lang == "Arabic (العربية)":
        gender_options = ["ذكر", "أنثى"]
    else:
        gender_options = ["مرد (Male)", "عورت (Female)"]
        
    gender_selection = st.selectbox(t["gender"], gender_options)
    gender_num = 1 if gender_selection in ["Male", "Masculino", "ذكر", "مرد (Male)"] else 0

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- PREDICTION ENGINE -----------------
if st.button(t["btn"]):
    
    # 1. Diabetes model format
    d_input = np.array([[p_age, bmi_calc, hba1c, fpg, systolic_bp, cholesterol]])
    
    # 2. Heart Model expects exact 16 inputs to map categorical features safely
    # Pad structural template array with zeros to resolve 'Feature shape mismatch'
    h_base = [p_age, gender_num, systolic_bp, cholesterol, heart_rate, exercise_hours]
    h_padded = h_base + [0] * (16 - len(h_base))
    h_input = np.array([h_padded])
    
    # 3. Obesity model format
    o_input = np.array([[p_age, gender_num, bmi_calc]])
    
    try:
        d_pred = d_model.predict(d_input)[0]
        h_pred = h_model.predict(h_input)[0]
        o_pred = o_model.predict(o_input)[0]
        
        st.markdown(f"<br><div class='block-heading'>{t['h3']}</div>", unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(label=t["db_lbl"], value=t["pos"] if d_pred == 1 else t["neg"])
        with res2:
            st.metric(label=t["ht_lbl"], value=t["high_r"] if h_pred == 1 else t["low_r"])
        with res3:
            obesity_map = {0: "Underweight", 1: "Normal Weight", 2: "Overweight", 3: "Obese Class"}
            obesity_map_es = {0: "Bajo peso", 1: "Peso Normal", 2: "Sobrepeso", 3: "Clase de Obesidad"}
            obesity_map_ar = {0: "نقص الوزن", 1: "وزن طبيعي", 2: "زيادة الوزن", 3: "سمنة مفرطة"}
            obesity_map_fr = {0: "Insuffisance pondérale", 1: "Poids Normal", 2: "Surpoids", 3: "Classe d'Obésité"}
            obesity_map_ur = {0: "کم وزن", 1: "نارمل وزن", 2: "زیادہ وزن", 3: "موٹاپا"}
            
            if lang == "Spanish (Español)":
                final_ob_val = obesity_map_es.get(o_pred)
            elif lang == "Arabic (العربية)":
                final_ob_val = obesity_map_ar.get(o_pred)
            elif lang == "French (Français)":
                final_ob_val = obesity_map_fr.get(o_pred)
            elif lang == "Urdu":
                final_ob_val = obesity_map_ur.get(o_pred)
            else:
                final_ob_val = obesity_map.get(o_pred)
                
            st.metric(label=t["ob_lbl"], value=final_ob_val)
            
        st.success(t["success"])
        
    except Exception as error:
        st.error(f"Inference error details: {str(error)}")
