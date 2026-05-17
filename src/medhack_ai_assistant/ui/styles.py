import streamlit as st


APP_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    max-width: 1280px;
}

.profile-icon {
    width: 58px;
    height: 58px;
    border-radius: 50%;
    background: #eef2ff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
}

.patient-stat {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 10px 0;
}

.patient-stat-icon {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}

.patient-stat-ok .patient-stat-icon {
    background: #dcfce7;
}

.patient-stat-warn .patient-stat-icon {
    background: #fef3c7;
}

.patient-stat-bad .patient-stat-icon {
    background: #ffedd5;
}

.patient-stat-label {
    color: #64748b;
    font-size: 0.9rem;
}

.patient-stat-value {
    color: #0f172a;
    font-weight: 700;
    font-size: 1rem;
}
</style>
"""

def render_global_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
