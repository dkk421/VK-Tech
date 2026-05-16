import asyncio

import pandas as pd
import streamlit as st

from medhack_ai_assistant.config import ID_COLUMN, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from medhack_ai_assistant.data import load_data
from medhack_ai_assistant.domain.models import PatientExam
from medhack_ai_assistant.pipeline import analyze_patient_exam, run_quality_gate
from medhack_ai_assistant.services.dashboard import build_patient_exam
from medhack_ai_assistant.ui.ai_rendering import render_analysis_result
from medhack_ai_assistant.ui.formatters import readable_dataframe, translate_quality_reason
from medhack_ai_assistant.ui.styles import render_global_styles


def setup_page() -> None:
    st.set_page_config(
        page_title="MedHack AI Assistant",
        page_icon="🏥",
        layout="wide",
    )
    render_global_styles()


def render_header() -> None:
    st.title("MedHack AI Assistant")
    st.caption(
        "Минимальная Streamlit-версия для просмотра медосмотров и проверки противопоказаний."
    )


def load_selected_dataset() -> pd.DataFrame:
    with st.sidebar:
        st.header("Данные")
        source = st.radio(
            "Источник",
            ("test.csv", "train.csv", "Загрузить CSV"),
        )
        st.markdown(
            "<p class='small-muted'>LLM-анализ запускается только по отдельной кнопке. "
            "Карточка пациента работает локально.</p>",
            unsafe_allow_html=True,
        )

    try:
        dataframe, dataset_name = _get_dataset(source)
    except Exception as exc:
        st.error(f"Не удалось загрузить данные: {exc}")
        st.stop()

    if dataframe.empty:
        st.warning("В выбранном наборе данных нет строк.")
        st.stop()

    st.session_state["dataset_name"] = dataset_name
    return dataframe


def select_exam(dataframe: pd.DataFrame) -> PatientExam:
    render_dataset_summary(dataframe)
    row = _select_exam_row(dataframe)
    if row is None:
        st.stop()

    try:
        exam = build_patient_exam(row)
    except Exception as exc:
        st.error(f"Не удалось разобрать карточку пациента: {exc}")
        with st.expander("Показать исходную строку"):
            st.json(row.to_dict() if hasattr(row, "to_dict") else dict(row))
        st.stop()

    st.session_state["selected_row"] = row
    return exam


def render_dataset_summary(dataframe: pd.DataFrame) -> None:
    st.subheader("Набор данных")
    dataset_name = st.session_state.get("dataset_name", "CSV")

    col_count, col_rows, col_target = st.columns(3)
    col_count.metric("Файл", dataset_name)
    col_rows.metric("Строк", f"{len(dataframe):,}".replace(",", " "))
    if TARGET_COLUMN in dataframe.columns:
        positives = int(dataframe[TARGET_COLUMN].fillna(False).astype(bool).sum())
        col_target.metric("С противопоказаниями", positives)
    else:
        col_target.metric("Целевая метка", "нет")

    with st.expander("Показать первые строки"):
        st.dataframe(readable_dataframe(dataframe.head(20)), use_container_width=True)


def render_ai_section(exam: PatientExam) -> None:
    st.subheader("AI-анализ")
    quality = run_quality_gate(exam)

    if quality.can_analyze:
        st.success("Данных достаточно для анализа.")
    else:
        st.warning("Данных недостаточно для надежного анализа.")
        for reason in quality.reasons:
            st.write(f"- {translate_quality_reason(reason)}")

    st.info(
        "Кнопка обращается к vLLM endpoint из переменных окружения. "
        "Если модель не запущена, приложение покажет понятную ошибку."
    )

    if st.button("Запустить AI-анализ", type="primary"):
        with st.spinner("Анализирую осмотр..."):
            try:
                result = asyncio.run(analyze_patient_exam(exam))
            except Exception as exc:
                st.error(f"AI-анализ не выполнен: {exc}")
                return
        render_analysis_result(result)


@st.cache_data(show_spinner=False)
def _load_builtin_dataset(use_train: bool) -> pd.DataFrame:
    return load_data(TRAIN_PATH if use_train else TEST_PATH)


def _get_dataset(source: str) -> tuple[pd.DataFrame, str]:
    if source == "train.csv":
        return _load_builtin_dataset(True), "train.csv"
    if source == "test.csv":
        return _load_builtin_dataset(False), "test.csv"

    uploaded_file = st.file_uploader("CSV-файл с осмотрами", type=["csv"])
    if uploaded_file is None:
        st.info("Загрузите CSV-файл, чтобы продолжить.")
        st.stop()
    return pd.read_csv(uploaded_file), uploaded_file.name


def _select_exam_row(dataframe: pd.DataFrame) -> pd.Series | None:
    st.subheader("Выбор осмотра")

    if ID_COLUMN not in dataframe.columns:
        st.error(f"В данных нет обязательного столбца `{ID_COLUMN}`.")
        return None

    ids = dataframe[ID_COLUMN].dropna().astype(int).tolist()
    selected_id = st.selectbox(
        "ID осмотра",
        ids,
        format_func=str,
    )

    matched = dataframe.loc[dataframe[ID_COLUMN] == selected_id]
    if matched.empty:
        st.error("Осмотр с таким ID не найден.")
        return None
    return matched.iloc[0]
