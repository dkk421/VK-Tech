import streamlit as st


def render_global_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1180px; }
        .small-muted { color: #667085; font-size: 0.92rem; }
        .result-ok {
            border-left: 4px solid #16a34a;
            background: #f0fdf4;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .result-warn {
            border-left: 4px solid #f59e0b;
            background: #fffbeb;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .result-bad {
            border-left: 4px solid #dc2626;
            background: #fef2f2;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
