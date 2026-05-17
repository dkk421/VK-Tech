import pandas as pd
import streamlit as st

from medhack_ai_assistant.config import TARGET_COLUMN
from medhack_ai_assistant.domain.models import PatientExam
from medhack_ai_assistant.ui.formatters import display_value, shorten


def render_patient_summary_card(
    exam: PatientExam,
    *,
    risks_count: int = 0,
    data_status: str = "готово к AI-анализу",
) -> None:
    conclusions_count = len(exam.specialist_conclusions)

    with st.container(border=True):
        st.markdown("### Сводка по пациенту")

        left, right = st.columns([1.2, 1])

        with left:
            st.markdown(f"#### Пациент #{exam.patient_id}")
            st.caption(f"Осмотр #{exam.exam_row_id}")
            st.write(f"Дата консультации: {display_value(exam.consultation_date) or 'не указана'}")
            st.write(f"Статус данных: **{data_status}**")

        with right:
            render_patient_stat(
                icon="📋",
                label="Факторов в направлении",
                value=str(risks_count),
                status="ok",
            )

            render_patient_stat(
                icon="🩺",
                label="Заключений специалистов",
                value=str(conclusions_count),
                status="warn",
            )


def render_patient_stat(
    *,
    icon: str,
    label: str,
    value: str,
    status: str,
) -> None:
    st.markdown(
        f"""
        <div class="patient-stat patient-stat-{status}">
            <div class="patient-stat-icon">{icon}</div>
            <div>
                <div class="patient-stat-label">{label}</div>
                <div class="patient-stat-value">{value}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_patient_summary(exam: PatientExam) -> None:
    render_patient_summary_card(
        exam,
        risks_count=len(exam.assigned_harmful_factors),
    )

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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


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
