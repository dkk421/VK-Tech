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
from medhack_ai_assistant.ui.cards import render_patient_summary_card


def setup_page() -> None:
    st.set_page_config(
        page_title="MedHack AI Assistant",
        page_icon="🏥",
        layout="wide",
    )

    render_global_styles()


def render_header() -> None:
    st.markdown(
        """
        <div class="app-header">
            <div>
                <div class="hero-title">Медицинское заключение</div>
                <div class="muted">Сводка по осмотру, вредным факторам и заключениям специалистов</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_selected_dataset() -> pd.DataFrame:
    with st.sidebar:
        st.header("Источник данных")
        source = st.radio(
            "Источник",
            ("test.csv", "train.csv", "Загрузить CSV"),
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
    st.subheader("Метрики")
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
        st.dataframe(readable_dataframe(dataframe.head(20)), width="stretch")


def render_detailed_data(dataframe: pd.DataFrame) -> None:
    row = st.session_state.get("selected_row")

    with st.expander("Детальные данные"):
        st.caption(
            f"Источник: {st.session_state.get('dataset_name', 'CSV')}; "
            f"строк в наборе: {len(dataframe):,}".replace(",", " ")
        )
        if isinstance(row, pd.Series):
            st.markdown("**Исходная строка выбранного осмотра**")
            st.json(row.to_dict())

        st.markdown("**Первые строки набора**")
        st.dataframe(readable_dataframe(dataframe.head(20)), width="stretch")


def render_ai_action(exam: PatientExam) -> None:
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

    if st.button("Запустить AI-анализ", type="primary", width="stretch"):
        with st.spinner("Анализирую осмотр..."):
            try:
                result = asyncio.run(analyze_patient_exam(exam))
            except Exception as exc:
                st.error(f"AI-анализ не выполнен: {exc}")
                return
        st.session_state["ai_result"] = result
        st.session_state["ai_result_exam_id"] = exam.exam_row_id


def render_ai_section(exam: PatientExam) -> None:
    st.subheader("AI-анализ")
    render_ai_action(exam)

    result = st.session_state.get("ai_result")
    result_exam_id = st.session_state.get("ai_result_exam_id")

    if result is None or result_exam_id != exam.exam_row_id:
        st.info("Запустите AI-анализ, чтобы увидеть результат по выбранному осмотру.")
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
    if ID_COLUMN not in dataframe.columns:
        st.error(f"В данных нет обязательного столбца `{ID_COLUMN}`.")
        return None

    ids = dataframe[ID_COLUMN].dropna().astype(int).tolist()
    with st.sidebar:
        st.header("Осмотр")
        selected_id = st.selectbox(
            "ID осмотра",
            ids,
            format_func=str,
        )

        with st.expander("Показать первые строки"):
            st.dataframe(readable_dataframe(dataframe.head(20)), width="stretch")

    matched = dataframe.loc[dataframe[ID_COLUMN] == selected_id]
    if matched.empty:
        st.error("Осмотр с таким ID не найден.")
        return None
    return matched.iloc[0]


def render_patient_overview(exam: PatientExam) -> None:
    quality = run_quality_gate(exam)
    data_status = "готово к AI-анализу" if quality.can_analyze else "нужны дополнительные данные"

    render_patient_summary_card(
        exam,
        risks_count=len(exam.assigned_harmful_factors),
        data_status=data_status,
    )
