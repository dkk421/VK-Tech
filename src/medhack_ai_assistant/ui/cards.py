import pandas as pd
import streamlit as st

from medhack_ai_assistant.config import TARGET_COLUMN
from medhack_ai_assistant.domain.models import PatientExam
from medhack_ai_assistant.ui.formatters import display_value, shorten


def render_patient_summary(exam: PatientExam) -> None:
    st.subheader("Карточка пациента")

    col_patient, col_exam, col_date, col_factors = st.columns(4)
    col_patient.metric("Пациент", exam.patient_id)
    col_exam.metric("Осмотр", exam.exam_row_id)
    col_date.metric("Дата", display_value(exam.consultation_date) or "не указана")
    col_factors.metric("Факторов", len(exam.assigned_harmful_factors))

    row = st.session_state.get("selected_row")
    if isinstance(row, pd.Series):
        _render_known_target(row)


def render_findings(exam: PatientExam) -> None:
    st.subheader("Вредные факторы")
    if exam.assigned_harmful_factors:
        st.write(", ".join(exam.assigned_harmful_factors))
    else:
        st.info("Вредные факторы не указаны.")


def render_specialist_conclusions(exam: PatientExam) -> None:
    st.subheader("Заключения специалистов")
    if not exam.specialist_conclusions:
        st.info("Заключения специалистов отсутствуют.")
        return

    rows = [
        {
            "Специалист": display_value(item.specialist),
            "Дата": display_value(item.consultation_date),
            "Группа здоровья": display_value(item.health_group),
            "МКБ": display_value(item.mkb_code),
            "Описание МКБ": display_value(item.mkb_description),
            "Заключение": shorten(display_value(item.conclusion), 220),
        }
        for item in exam.specialist_conclusions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_known_target(row: pd.Series) -> None:
    if TARGET_COLUMN not in row.index or pd.isna(row[TARGET_COLUMN]):
        return

    st.markdown("**Разметка из train.csv**")
    has_contraindications = bool(row[TARGET_COLUMN])
    if has_contraindications:
        factors = display_value(row.get("contraindicated_factors", "")) or "факторы не указаны"
        st.markdown(
            f"<div class='result-bad'><b>Есть противопоказания.</b><br>Факторы: {factors}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='result-ok'><b>Противопоказания в разметке не указаны.</b></div>",
            unsafe_allow_html=True,
        )
