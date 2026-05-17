import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhack_ai_assistant.streamlit_app import (
    load_selected_dataset,
    render_batch_analysis_page,
    render_ai_section,
    render_detailed_data,
    render_findings,
    render_header,
    render_patient_overview,
    render_specialist_conclusions,
    select_exam,
    setup_page,
)


def main() -> None:
    setup_page()
    render_header()

    if st.session_state.get("batch_analysis_page"):
        render_batch_analysis_page()
        return

    dataframe = load_selected_dataset()
    exam = select_exam(dataframe)

    top_patient, top_factors = st.columns([1.5, 1], gap="large")
    with top_patient:
        render_patient_overview(exam)
    with top_factors:
        render_findings(exam)

    bottom_ai, bottom_conclusions = st.columns([1, 1.4], gap="large")
    with bottom_ai:
        render_ai_section(exam)
    with bottom_conclusions:
        render_specialist_conclusions(exam)

    render_detailed_data(dataframe)


if __name__ == "__main__":
    main()
