import streamlit as st

st.set_page_config(
    page_title="Health Diagnostic Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple & Clean Clinical Palette
st.markdown("""
    <style>
    .main-hero {
        text-align: center;
        padding: 50px 20px;
        background-color: #1a365d; /* Dark Professional Navy */
        color: white;
        border-radius: 10px;
        margin-top: 15px;
    }
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .main-desc {
        font-size: 16px;
        max-width: 700px;
        margin: 0 auto 25px auto;
        line-height: 1.6;
        color: #e2e8f0;
    }
    div.stButton > button:first-child {
        background-color: #10b981 !important; /* Premium Mint Green Action Button */
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 10px 35px !important;
        border-radius: 6px !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='main-hero'>
        <div class='main-title'>Automated Health Testing & Risk Prediction Portal</div>
        <div class='main-desc'>
            Welcome to the digital diagnostic assistant. This portal helps you calculate health risk metrics 
            for Diabetes, Heart Conditions, and Body Mass Category using your general health vitals. 
            Fill out the simple automated form on the next page to download your certified diagnostic summary.
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Open Diagnostic Form", use_container_width=True):
        st.switch_page("pages/1_Clinical_Evaluation.py")
