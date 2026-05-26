import streamlit as st

st.set_page_config(
    page_title="Health Analytics Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern Tech UI Styling
st.markdown("""
    <style>
    .main-hero {
        text-align: center;
        padding: 60px 40px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); /* Deep Tech Midnight Gradient */
        color: white;
        border-radius: 16px;
        margin-top: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    .main-title {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }
    .main-desc {
        font-size: 17px;
        max-width: 750px;
        margin: 0 auto 30px auto;
        line-height: 1.6;
        color: #94a3b8; /* Soft Slate Grey */
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #06b6d4 0%, #3b82f6 100%) !important; /* Glowing Teal to Blue Premium Button */
        color: white !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 12px 40px !important;
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='main-hero'>
        <div class='main-title'>Smart Health Check & Risk Evaluation Portal</div>
        <div class='main-desc'>
            Analyze vital health metrics and generate smart risk assessments instantly. 
            This automated workspace provides real-time insights into metabolic parameters, 
            cardiovascular vitals, and physical weight classifications using an intuitive diagnostic engine.
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Get Started", use_container_width=True):
        st.switch_page("pages/1_Clinical_Evaluation.py")
