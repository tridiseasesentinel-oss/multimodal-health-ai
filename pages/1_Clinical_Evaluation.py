import streamlit as st
import numpy as np
import pickle
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Configuration Setup
st.set_page_config(page_title="Clinical Evaluation Matrix", layout="wide", initial_sidebar_state="expanded")

# Inject Clean Professional CSS Layout for Split Counter Alignments
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; }
    h2, h3 { color: #1e3d59; font-family: 'Segoe UI', sans-serif; }
    
    /* Layout styling for splitting number inputs (+ and - styling structure) */
    div.stNumberInput div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #ced4da !important;
        border-radius: 6px !important;
    }
    div.stNumberInput button {
        background-color: #f1f3f5 !important;
        color: #1e3d59 !important;
        border: none !important;
    }
    
    /* Global Action Buttons Styling */
    .stButton>button {
        background-color: #17b978 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 24px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Defensive Loading Mechanism for Machine Learning Framework Binary Files
def load_clinical_models():
    try:
        with open("d_model.pkl", "rb") as f:
            d_model = pickle.load(f)
        with open("h_model.pkl", "rb") as f:
            h_model = pickle.load(f)
        with open("o_model.pkl", "rb") as f:
            o_model = pickle.load(f)
        return d_model, h_model, o_model
    except FileNotFoundError:
        st.error("Critical Error: Model binary files (.pkl) missing from directory. Please check if files are uploaded or named correctly.")
        return None, None, None

d_model, h_model, o_model = load_clinical_models()

st.title("Patient Diagnostic Evaluation Interface")
st.markdown("---")

# 1. Patient Demographics Sub-Section
st.subheader("Patient Identification Records")
c1, c2, c3 = st.columns(3)
with c1:
    patient_name = st.text_input("Patient Full Name", value="John Doe")
with c2:
    patient_id = st.text_input("Clinical Case ID / File Number", value="CR-2026-0892")
with c3:
    patient_age = st.number_input("Biological Age (Years)", min_value=1, max_value=120, value=45, step=1)

# 2. Advanced Diagnostic Feature Inputs Panel Split Layout
st.subheader("Physiological and Metabolic Metrics Matrix")

col_a, col_b = st.columns(2)

with col_a:
    bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
    blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1)
    cholesterol = st.number_input("Total Serum Cholesterol (mg/dL)", min_value=100, max_value=400, value=195, step=1)
    glucose = st.number_input("Fasting Plasma Glucose (mg/dL)", min_value=50, max_value=300, value=98, step=1)

with col_b:
    hbA1c = st.number_input("Hemoglobin A1c (%)", min_value=4.0, max_value=15.0, value=5.4, step=0.1)
    heart_rate = st.number_input("Resting Heart Rate (BPM)", min_value=40, max_value=180, value=72, step=1)
    physical_activity = st.number_input("Weekly Physical Activity (Hours)", min_value=0, max_value=42, value=3, step=1)
    gender_select = st.selectbox("Biological Sex Registration", options=["Male", "Female", "Other"])

# Mapping choices back to structural evaluation positions
gender_numeric = 1 if gender_select == "Male" else 0

st.markdown("<br>", unsafe_allow_html=True)

# Professional ReportLab Clinical PDF Generator Implementation Function
def generate_pdf_report(name, pid, age, d_res, h_res, o_res):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=24, leading=28, textColor=colors.HexColor("#1e3d59"), spaceAfter=15)
    section_style = ParagraphStyle('SecTitle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.HexColor("#17b978"), spaceBefore=12, spaceAfter=8)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['BodyText'], fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor("#333333"))
    
    # Document Header Section
    story.append(Paragraph("TRI-DISEASE SENTINEL CLINICAL REPORT", title_style))
    story.append(Paragraph("Automated Diagnostic Predictive Engine Summary Output", body_style))
    story.append(Spacer(1, 15))
    
    # Demographics Block Setup
    story.append(Paragraph("Patient Administrative Metadata", section_style))
    demo_data = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(name, body_style), Paragraph("<b>Case File ID:</b>", body_style), Paragraph(pid, body_style)],
        [Paragraph("<b>Age:</b>", body_style), Paragraph(str(age), body_style), Paragraph("<b>Evaluation Date:</b>", body_style), Paragraph("2026-05-26", body_style)]
    ]
    t_demo = Table(demo_data, colWidths=[100, 160, 100, 160])
    t_demo.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8f9fa")),
        ('PADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#e9ecef")),
    ]))
    story.append(t_demo)
    story.append(Spacer(1, 20))
    
    # Model Findings Analytics Section
    story.append(Paragraph("Machine Learning Evaluation Insights", section_style))
    findings_data = [
        [Paragraph("<b>Diagnostic Module</b>", body_style), Paragraph("<b>Status Assessment Finding</b>", body_style)],
        [Paragraph("Metabolic Profiling (Diabetes)", body_style), Paragraph(d_res, body_style)],
        [Paragraph("Cardiovascular Stratification (Heart)", body_style), Paragraph(h_res, body_style)],
        [Paragraph("Adiposity Classification (Obesity)", body_style), Paragraph(o_res, body_style)]
    ]
    t_findings = Table(findings_data, colWidths=[260, 260])
    t_findings.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor("#1e3d59")),
        ('TEXTCOLOR', (0,0), (1,0), colors.white),
        ('PADDING', (0,0), (-1,-1), 10),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#ced4da")),
    ]))
    # Quick color adjustment for header visibility row mapping
    for i in range(2):
        t_findings.setStyle(TableStyle([('TEXTCOLOR', (i,0), (i,0), colors.white)]))
        
    story.append(t_findings)
    story.append(Spacer(1, 30))
    
    # Disclaimer Standard Signoff
    story.append(Paragraph("<i>Notice: This computation is generated via ensemble predictive pipelines. Final diagnostic validation must be evaluated by a certified clinical practitioner.</i>", body_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# 3. Execution Infrastructure Trigger Pipeline
if st.button("Generate Diagnostic Multimodal Evaluation", use_container_width=True):
    if d_model and h_model and o_model:
        # Array matching shapes based on standard clinical dataset input columns
        d_input = np.array([[glucose, blood_pressure, bmi, hbA1c, patient_age]])
        h_input = np.array([[patient_age, gender_numeric, cholesterol, heart_rate, blood_pressure]])
        o_input = np.array([[patient_age, bmi, physical_activity, gender_numeric]])
        
        # Computing individual predictions
        d_pred = d_model.predict(d_input)[0]
        h_pred = h_model.predict(h_input)[0]
        o_pred = o_model.predict(o_input)[0]
        
        # Status parsing assignments
        d_status = "Positive Diagnosis Identified" if d_pred == 1 else "Normal / No Anomalies Detected"
        h_status = "Elevated Risk Stratification Detected" if h_pred == 1 else "Optimal Stable State"
        o_status = "Adiposity Risk / High BMI Classification" if o_pred == 1 else "Normal Weight Index Range"
        
        st.markdown("---")
        st.subheader("System Diagnostic Findings")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Metabolic Diabetes Module", value=d_status)
        with col2:
            st.metric(label="Cardiovascular Heart Module", value=h_status)
        with col3:
            st.metric(label="Adiposity Obesity Module", value=o_status)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Trigger Document Output Stream
        pdf_data = generate_pdf_report(patient_name, patient_id, patient_age, d_status, h_status, o_status)
        
        st.download_button(
            label="Download Clinical Diagnostic PDF Report",
            data=pdf_data,
            file_name=f"Clinical_Report_{patient_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
