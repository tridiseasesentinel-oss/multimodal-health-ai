import streamlit as st

st.set_page_config(page_title="Support Channels", layout="wide")

st.markdown("""
    <style>
    h2, h3 { color: #1e3d59; font-family: 'Segoe UI', sans-serif; }
    .support-card {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Technical Support & Assistance Node")
st.markdown("---")

st.markdown("""
    <div class='support-card'>
        <h3>Enterprise System Support</h3>
        <p>For deployment configuration inquiries, runtime issues, or multimodal pipeline troubleshooting, reach out to the diagnostic administration channel.</p>
        <p><b>Communication Endpoint:</b> support@tridiseasesentinel.org</p>
    </div>
    <div class='support-card'>
        <h3>Methodological Verification</h3>
        <p>Inquiries regarding deep predictive accuracy bounds, ensemble meta-classifier profiling, or clinical training dataset optimization protocols can be channeled to the research desk.</p>
        <p><b>Research Endpoint:</b> clinical-research@numl-faisalabad.edu.pk</p>
    </div>
""", unsafe_allow_html=True)
