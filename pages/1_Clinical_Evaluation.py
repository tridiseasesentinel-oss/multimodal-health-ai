import streamlit as st
import pickle
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="Global Health Evaluation", layout="wide")

# Theme styling
st.markdown("""
    <style>
    .block-heading {
        color: #1e3a8a;
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
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_diagnostic_models():
    d = pickle.load(open("d_model.pkl", "rb"))
    h = pickle.load(open("h_model.pkl", "rb"))
    o = pickle.load(open("o_model.pkl", "rb"))
    return d, h, o

try:
    d_model, h_model, o_model = load_diagnostic_models()
except Exception as e:
    st.error("System configuration files loading...")

# ----------------- 20 LANGUAGES DICTIONARY -----------------
with st.sidebar:
    st.markdown("### 🌐 Language Settings")
    lang = st.selectbox("Select Language", [
        "English", "Urdu", "Arabic", "Hindi", "Punjabi", "Pashto", "Sindhi", 
        "Spanish", "French", "German", "Chinese", "Russian", "Bengali", 
        "Portuguese", "Japanese", "Turkish", "Italian", "Persian", "Korean", "Indonesian"
    ])

text_pack = {
    "English": {"title": "Personal Health Evaluation Form", "subtitle": "Enter metrics below.", "h1": "1. Basic Information", "h2": "2. Vitals", "h3": "Results", "h4": "📋 Precautions", "name": "Full Name", "ref_id": "Case ID", "age": "Age", "weight": "Weight (kg)", "height_ft": "Height (Ft)", "height_in": "Inches", "bmi_msg": "Calculated BMI", "btn": "Run Evaluation", "db_lbl": "Diabetes Risk", "ht_lbl": "Heart Risk", "ob_lbl": "Obesity", "pos": "Positive", "neg": "Normal"},
    "Urdu": {"title": "شخصی صحت کی جانچ کا فارم", "subtitle": "نیچے معلومات درج کریں۔", "h1": "1. بنیادی معلومات", "h2": "2. صحت کے وائٹلز", "h3": "نتائج", "h4": "📋 احتیاطی تدابیر", "name": "پورا نام", "ref_id": "کیس آئی ڈی", "age": "عمر", "weight": "وزن (کلوگرام)", "height_ft": "قد (فٹ)", "height_in": "انچ", "bmi_msg": "حساب کردہ BMI", "btn": "تشخیص چلائیں", "db_lbl": "ذیابیطس", "ht_lbl": "دل کا خطرہ", "ob_lbl": "موٹاپا", "pos": "موجود ہے", "neg": "نارمل"},
    "Arabic": {"title": "نموذج تقييم الصحة", "subtitle": "أدخل البيانات أدناه.", "h1": "1. معلومات أساسية", "h2": "2. العلامات الحيوية", "h3": "النتائج", "h4": "📋 تدابير وقائية", "name": "الاسم الكامل", "ref_id": "رقم الحالة", "age": "العمر", "weight": "الوزن (كجم)", "height_ft": "الطول (قدم)", "height_in": "بوصة", "bmi_msg": "مؤشر كتلة الجسم", "btn": "تشغيل التقييم", "db_lbl": "السكري", "ht_lbl": "أمراض القلب", "ob_lbl": "السمنة", "pos": "إيجابي", "neg": "طبيعي"},
    "Hindi": {"title": "व्यक्तिगत स्वास्थ्य मूल्यांकन", "subtitle": "नीचे विवरण दर्ज करें।", "h1": "1. बुनियादी जानकारी", "h2": "2. स्वास्थ्य महत्वपूर्ण", "h3": "परिणाम", "h4": "📋 एहतियाती उपाय", "name": "पूरा नाम", "ref_id": "केस आईडी", "age": "आयु", "weight": "वजन (किग्रा)", "height_ft": "ऊंचाई (फिट)", "height_in": "इंच", "bmi_msg": "गणना की गई बीएमआई", "btn": "मूल्यांकन करें", "db_lbl": "मधुमेह जोखिम", "ht_lbl": "हृदय जोखिम", "ob_lbl": "मोटापा", "pos": "जोखिम", "neg": "सामान्य"},
    "Punjabi": {"title": "ਨਿੱਜੀ ਸਿਹਤ ਮੁਲਾਂਕਣ ਫਾਰਮ", "subtitle": "ਹੇਠਾਂ ਮੈਟ੍ਰਿਕਸ ਦਰਜ ਕਰੋ।", "h1": "1. ਬੁਨਿਆਦੀ ਜਾਣਕਾਰੀ", "h2": "2. ਸਿਹਤ ਵਾਈਟਲਸ", "h3": "ਨਤੀਜੇ", "h4": "📋 ਸਾਵਧਾਨੀਆਂ", "name": "ਪੂਰਾ ਨਾਮ", "ref_id": "ਕੇਸ ਆਈਡੀ", "age": "ਉਮਰ", "weight": "ਭਾਰ (ਕਿਲੋ)", "height_ft": "ਕੱਦ (ਫੁੱਟ)", "height_in": "ਇੰਚ", "bmi_msg": "BMI ਨਤੀਜਾ", "btn": "ਜਾਂਚ ਸ਼ੁਰੂ ਕਰੋ", "db_lbl": "ਸ਼ੂਗਰ ਦਾ ਖਤਰਾ", "ht_lbl": "ਦਿਲ ਦਾ ਖਤਰਾ", "ob_lbl": "ਮੋਟਾਪਾ", "pos": "ਖਤਰਾ ਹੈ", "neg": "ਨਾਰਮਲ"},
    "Pashto": {"title": "د شخصي روغتیا ارزونې فورمه", "subtitle": "روغتیایی معلومات دننه کړئ.", "h1": "۱. اساسي معلومات", "h2": "۲. روغتیايي نښې", "h3": "پایلې", "h4": "📋 احتیاطي تدابیر", "name": "بشپړ نوم", "ref_id": "کیس ای ډੀ", "age": "عمر", "weight": "وزن (کلو)", "height_ft": "قامت (فټ)", "height_in": "انچ", "bmi_msg": "حساب شوی BMI", "btn": "ارزونه پیਲ کړئ", "db_lbl": "د شکرې خطر", "ht_lbl": "د زړه خطر", "ob_lbl": "چاغښت", "pos": "خطر شته", "neg": "نارمل"},
    "Sindhi": {"title": "شخصي صحت جي جانچ جو فارم", "subtitle": "هيٺ معلومات داخل ڪريو.", "h1": "1. ਬੁਨਿਆਦੀ ਜਾਣਕਾਰੀ", "h2": "2. صحت جا وائٽلز", "h3": "نتيحا", "h4": "📋 احتياطي تدابير", "name": "ਪੂਰਾ ਨਾਮ", "ref_id": "ڪيس آئي ڊي", "age": "عمر", "weight": "وزن (کلو)", "height_ft": "قد (فٽ)", "height_in": "انچ", "bmi_msg": "حساب ٿيل BMI", "btn": "تشخيص هلايو", "db_lbl": "ڏيابيٽس جو خطرو", "ht_lbl": "دل جو خطرو", "ob_lbl": "موتيو", "pos": "مثبت", "neg": "نارمل"},
    "Spanish": {"title": "Formulario de Evaluación de Salud", "subtitle": "Ingrese las métricas.", "h1": "1. Info Básica", "h2": "2. Vítales", "h3": "Resultados", "h4": "📋 Medidas Preventivas", "name": "Nombre", "ref_id": "ID Caso", "age": "Edad", "weight": "Peso (kg)", "height_ft": "Altura (Pies)", "height_in": "Pulgadas", "bmi_msg": "IMC Calculado", "btn": "Ejecutar Evaluación", "db_lbl": "Diabetes", "ht_lbl": "Riesgo Cardíaco", "ob_lbl": "Obesidad", "pos": "Positivo", "neg": "Normal"},
    "French": {"title": "Formulaire d'Évaluation de Santé", "subtitle": "Entrez les données.", "h1": "1. Info de Base", "h2": "2. Signes Vitaux", "h3": "Résultats", "h4": "📋 Précautions", "name": "Nom", "ref_id": "ID Cas", "age": "Âge", "weight": "Poids (kg)", "height_ft": "Taille (Pieds)", "height_in": "Pouces", "bmi_msg": "IMC Calculé", "btn": "Évaluer", "db_lbl": "Diabète", "ht_lbl": "Risque Cardiaque", "ob_lbl": "Obésité", "pos": "Positif", "neg": "Normal"},
    "German": {"title": "Gesundheitsbewertungsformular", "subtitle": "Werte eingeben.", "h1": "1. Basisinfo", "h2": "2. Vitalwerte", "h3": "Ergebnisse", "h4": "📋 Maßnahmen", "name": "Name", "ref_id": "Fall-ID", "age": "Alter", "weight": "Gewicht (kg)", "height_ft": "Größe (Fuß)", "height_in": "Zoll", "bmi_msg": "Berechneter BMI", "btn": "Auswerten", "db_lbl": "Diabetes", "ht_lbl": "Herzrisiko", "ob_lbl": "Adipositas", "pos": "Positiv", "neg": "Normal"},
    "Chinese": {"title": "个人健康 Assessment 表格", "subtitle": "输入健康数据。", "h1": "1. 基本信息", "h2": "2. 生命体征", "h3": "评估结果", "h4": "📋 预防措施", "name": "姓名", "ref_id": "案例 ID", "age": "年龄", "weight": "体重 (kg)", "height_ft": "身高 (英尺)", "height_in": "英寸", "bmi_msg": "计算出的 BMI", "btn": "运行评估", "db_lbl": "糖尿病风险", "ht_lbl": "心脏病风险", "ob_lbl": "肥胖程度", "pos": "阳性/有风险", "neg": "正常"},
    "Russian": {"title": "Форма оценки здоровья", "subtitle": "Введите данные.", "h1": "1. Инфо", "h2": "2. Жизненные показатели", "h3": "Результаты", "h4": "📋 Меры предосторожности", "name": "Имя", "ref_id": "ID", "age": "Возраст", "weight": "Вес (кг)", "height_ft": "Рост (Футы)", "height_in": "Дюймы", "bmi_msg": "Индекс массы тела", "btn": "Оценить", "db_lbl": "Диабет", "ht_lbl": "Риск Сердца", "ob_lbl": "Ожирение", "pos": "Положительный", "neg": "Норма"},
    "Bengali": {"title": "ব্যক্তিগত স্বাস্থ্য মূল্যায়ন ফর্ম", "subtitle": "মেট্রিক্স লিখুন।", "h1": "১. সাধারণ তথ্য", "h2": "২. ভাইটালস", "h3": "ফলাফল", "h4": "📋 সতর্কতামূলক ব্যবস্থা", "name": "নাম", "ref_id": "আইডি", "age": "বয়স", "weight": "ওজন (কেজি)", "height_ft": "উচ্চতা (ফুট)", "height_in": "ইঞ্চি", "bmi_msg": "বিএমআই", "btn": "মূল্যায়ন শুরু করুন", "db_lbl": "ডায়াবেটিস ঝুঁকি", "ht_lbl": "হৃদরোগের ঝুঁকি", "ob_lbl": "স্থূলতা", "pos": "ঝুঁকি আছে", "neg": "স্বাভাবিক"},
    "Portuguese": {"title": "Formulário de Avaliação de Saúde", "subtitle": "Insira os dados.", "h1": "1. Info Básica", "h2": "2. Sinais Vitais", "h3": "Resultados", "h4": "📋 Precauções", "name": "Nome", "ref_id": "ID Caso", "age": "Idade", "weight": "Peso (kg)", "height_ft": "Altura (Pés)", "height_in": "Polegadas", "bmi_msg": "IMC Calculado", "btn": "Avaliar", "db_lbl": "Diabetes", "ht_lbl": "Risco Cardíaco", "ob_lbl": "Obesidade", "pos": "Positivo", "neg": "Normal"},
    "Japanese": {"title": "健康評価フォーム", "subtitle": "数値を入力してください。", "h1": "1. 基本情報", "h2": "2. バイタル", "h3": "結果", "h4": "📋 予防措置", "name": "氏名", "ref_id": "ケースID", "age": "年齢", "weight": "体重 (kg)", "height_ft": "身長 (feet)", "height_in": "inch", "bmi_msg": "BMI値", "btn": "評価を実行", "db_lbl": "糖尿病リスク", "ht_lbl": "心臓リスク", "ob_lbl": "肥胖度", "pos": "リスクあり", "neg": "正常"},
    "Turkish": {"title": "Kişisel Sağlık Değerlendirme Formu", "subtitle": "Metrikleri girin.", "h1": "1. Temel Bilgiler", "h2": "2. Vitaller", "h3": "Sonuçlar", "h4": "📋 Önlemler", "name": "Ad Soyad", "ref_id": "Vaka ID", "age": "Yaş", "weight": "Kilo (kg)", "height_ft": "Boy (Feet)", "height_in": "Inç", "bmi_msg": "Hesaplanan BMI", "btn": "Değerlendir", "db_lbl": "Diyabet Riski", "ht_lbl": "Kalp Riski", "ob_lbl": "Obezite", "pos": "Riskli", "neg": "Normal"},
    "Italian": {"title": "Modulo Valutazione Salute", "subtitle": "Inserisci i dati.", "h1": "1. Info Base", "h2": "2. Parametri", "h3": "Risultati", "h4": "📋 Precauzioni", "name": "Nome", "ref_id": "ID Caso", "age": "Età", "weight": "Peso (kg)", "height_ft": "Altezza (Piedi)", "height_in": "Pollici", "bmi_msg": "IMC Calcolato", "btn": "Valuta", "db_lbl": "Diabete", "ht_lbl": "Rischio Cardiaco", "ob_lbl": "Obesità", "pos": "Positivo", "neg": "Normale"},
    "Persian": {"title": "فرم ارزیابی سلامت فردی", "subtitle": "شاخص‌ها را وارد کنید.", "h1": "۱. اطلاعات پایه", "h2": "۲. علائم حیاتی", "h3": "نتایج", "h4": "📋 اقدامات پیشگیرانه", "name": "نام کامل", "ref_id": "شناسه پرونده", "age": "سن", "weight": "وزن (کیلو)", "height_ft": "قد (فوت)", "height_in": "اینچ", "bmi_msg": "شاخص BMI", "btn": "شروع ارزیابی", "db_lbl": "خطر دیابت", "ht_lbl": "خطر قلبی", "ob_lbl": "چاقی", "pos": "مثبت", "neg": "نرمال"},
    "Korean": {"title": "개인 건강 평가 양식", "subtitle": "수치를 입력하세요.", "h1": "1. 기본 정보", "h2": "2. 바이タル", "h3": "결과", "h4": "📋 권장 예방조치", "name": "성명", "ref_id": "케이스 ID", "age": "연령", "weight": "체중 (kg)", "height_ft": "신장 (Feet)", "height_in": "Inches", "bmi_msg": "계산된 BMI", "btn": "평가 실행", "db_lbl": "당뇨 위험", "ht_lbl": "심장 질환 위험", "ob_lbl": "비만도", "pos": "양성", "neg": "정상"},
    "Indonesian": {"title": "Formulir Evaluasi Kesehatan", "subtitle": "Masukkan data.", "h1": "1. Info Dasar", "h2": "2. Tanda Vital", "h3": "Hasil", "h4": "📋 Langkah Pencegahan", "name": "Nama Lengkap", "ref_id": "ID Kasus", "age": "Usia", "weight": "Berat (kg)", "height_ft": "Tinggi (Kaki)", "height_in": "Inci", "bmi_msg": "Hasil BMI", "btn": "Mulai Evaluasi", "db_lbl": "Risiko Diabetes", "ht_lbl": "Risiko Jantung", "ob_lbl": "Obesitas", "pos": "Positif", "neg": "Normal"}
}

t = text_pack[lang]

# Dynamic Alignment Setup
if lang in ["Urdu", "Arabic", "Pashto", "Persian", "Sindhi"]:
    st.markdown("<style>body {text-align: right !important; direction: rtl !important;}</style>", unsafe_allow_html=True)

st.title(t["title"])
st.markdown(t["subtitle"])

st.markdown(f"<div class='block-heading'>{t['h1']}</div>", unsafe_allow_html=True)
if 'auto_case_id' not in st.session_state:
    st.session_state.auto_case_id = f"CR-{time.strftime('%m%d')}-{str(int(time.time()))[-4:]}"

c1, c2, c3 = st.columns(3)
with c1:
    p_name = st.text_input(t["name"], value="")
with c2:
    p_id = st.text_input(t["ref_id"], value=st.session_state.auto_case_id, disabled=True)
with c3:
    p_age = st.number_input(t["age"], min_value=1, max_value=120, value=25, step=1)

st.markdown(f"<br><div class='block-heading'>{t['h2']}</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    weight = st.number_input(t["weight"], min_value=10.0, max_value=250.0, value=60.0)
    ht_c1, ht_c2 = st.columns(2)
    with ht_c1:
        ft_val = st.selectbox(t["height_ft"], list(range(1, 9)), index=4)
    with ht_c2:
        in_val = st.selectbox(t["height_in"], list(range(12)), index=6)
        
    total_inches = (ft_val * 12) + in_val
    height_m = (total_inches * 2.54) / 100.0
    bmi_calc = round(weight / (height_m ** 2), 2)
    st.info(f"{t['bmi_msg']}: **{bmi_calc}**")
    
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.4)
    fpg = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=500, value=98)

with col2:
    systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=220, value=120)
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=180, value=72)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=450, value=195)
    exercise_hours = st.number_input("Exercise Hours/Week", min_value=0, max_value=40, value=3)
    
    gender_selection = st.selectbox("Gender", ["Male", "Female"])
    gender_num = 1 if gender_selection == "Male" else 0

st.markdown("<br>", unsafe_allow_html=True)

if st.button(t["btn"]):
    try:
        # ABSOLUTE FIX: Generating dataframes matching internal feature specs directly
        # Diabetes Model Dataframe Structure
        d_input = pd.DataFrame(
            [[float(p_age), float(bmi_calc), float(hba1c), float(fpg), float(systolic_bp), float(cholesterol)]],
            columns=['Age', 'BMI', 'HbA1c', 'FastingBloodSugar', 'SystolicBP', 'Cholesterol']
        )
        
        # Heart Model Dataframe Structure
        h_input = pd.DataFrame(
            [[float(p_age), float(gender_num), float(systolic_bp), float(cholesterol), float(heart_rate), float(exercise_hours)]],
            columns=['Age', 'Gender', 'SystolicBP', 'Cholesterol', 'HeartRate', 'ExerciseHours']
        )
        
        # Obesity Model Dataframe Structure
        o_input = pd.DataFrame(
            [[float(p_age), float(gender_num), float(bmi_calc)]],
            columns=['Age', 'Gender', 'BMI']
        )

        # Predict operations directly
        d_pred = d_model.predict(d_input)
        h_pred = h_model.predict(h_input)
        o_pred = o_model.predict(o_input)
        
        d_res = int(d_pred[0])
        h_res = int(h_pred[0])
        o_res = int(o_pred[0])
        
        st.markdown(f"<br><div class='block-heading'>{t['h3']}</div>", unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(label=t["db_lbl"], value=t["pos"] if d_res == 1 else t["neg"])
        with res2:
            st.metric(label=t["ht_lbl"], value=t["pos"] if h_res == 1 else t["neg"])
        with res3:
            ob_map = {0: "Underweight", 1: "Normal Weight", 2: "Overweight", 3: "Obese Class"}
            st.metric(label=t["ob_lbl"], value=ob_map.get(o_res, "Normal"))
            
        st.success("Evaluation complete.")
        
        # Precaution Display Block
        st.markdown(f"<br><div class='block-heading'>{t['h4']}</div>", unsafe_allow_html=True)
        if d_res == 1:
            st.markdown("<div class='precaution-box'>⚠️ <b>Diabetes Precaution:</b> Reduce intake of sugar and carbohydrates. Maintain regular 30-minute daily walks.</div>", unsafe_allow_html=True)
        if h_res == 1:
            st.markdown("<div class='precaution-box'>⚠️ <b>Heart Precaution:</b> Restrict salty items and deeply fried meals. Keep monitoring blood pressure tracking charts.</div>", unsafe_allow_html=True)
        if o_res >= 2:
            st.markdown("<div class='precaution-box'>⚠️ <b>Obesity Precaution:</b> Restrict processed fast foods and soft drinks. Practice physical training routines.</div>", unsafe_allow_html=True)
            
    except Exception as error:
        st.error(f"Error Processing Request: {str(error)}")
