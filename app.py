import streamlit as st

# Layout Setup
st.set_page_config(
    page_title="Tri-Disease Sentinel AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Clinical Corporate Styling (Zero Emojis for Research look)
st.markdown("""
    <style>
    .hero-container {
        text-align: center;
        padding: 60px 30px;
        background: linear-gradient(135deg, #1e3d59 0%, #17b978 100%);
        color: white;
        border-radius: 12px;
        margin-top: 20px;
    }
    .logo-text {
        font-size: 64px;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }
    .hero-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .hero-subtitle {
        font-size: 16px;
        max-width: 650px;
        margin: 0 auto 30px auto;
        line-height: 1.5;
        opacity: 0.95;
    }
    .stButton>button {
        background-color: #ffffff !important;
        color: #1e3d59 !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 12px 40px !important;
        border-radius: 25px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Main Hero Container Section
st.markdown("""
    <div class='hero-container'>
        <div class='logo-text'>TDS // CLINICAL AI</div>
        <div class='hero-title'>Integrated Multimodal Clinical Decision Support System</div>
        <div class='hero-subtitle'>
            An enterprise-grade deep predictive analytics platform leveraging machine learning architectures 
            for automated metabolic profiling, cardiovascular stratification, and adiposity classification.
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Centered Action Button to move to the form
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Get Started", use_container_width=True):
        st.switch_page("pages/1_Clinical_Evaluation.py")
