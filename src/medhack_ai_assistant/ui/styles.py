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
    padding: 12px 10px;
    min-width: 0;
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
    font-size: 1.05rem;
    line-height: 1.25;
    word-break: break-word;
}

.factor-card {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.factor-card-meta {
    color: #64748b;
    font-size: 0.9rem;
}

.factor-chip-list {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.factor-chip {
    display: inline-flex;
    align-items: center;
    min-height: 32px;
    padding: 6px 10px;
    border: 1px solid #cbd5e1;
    border-radius: 999px;
    background: #f8fafc;
    color: #0f172a;
    font-size: 0.92rem;
    font-weight: 600;
    line-height: 1.2;
}

.empty-state {
    padding: 14px 16px;
    border: 1px dashed #cbd5e1;
    border-radius: 8px;
    background: #f8fafc;
}

.empty-state-title {
    color: #0f172a;
    font-weight: 700;
}

.empty-state-text {
    margin-top: 4px;
    color: #64748b;
    font-size: 0.92rem;
}
</style>
"""

def render_global_styles() -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
