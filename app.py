import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhack_ai_assistant.streamlit_app import (
    load_selected_dataset,
    render_ai_section,
    render_findings,
    render_header,
    render_patient_summary,
    render_specialist_conclusions,
    select_exam,
    setup_page,
)


def main() -> None:
    setup_page()

    dataframe = load_selected_dataset()
    exam = select_exam(dataframe)

    render_header()
    render_patient_summary(exam)
    render_ai_section(exam)
    render_findings(exam)
    render_specialist_conclusions(exam)


if __name__ == "__main__":
    main()
