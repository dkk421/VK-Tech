import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from medhack_ai_assistant.config import ID_COLUMN, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from medhack_ai_assistant.data import load_data
from medhack_ai_assistant.domain.models import AnalysisResult, PatientExam
from medhack_ai_assistant.pipeline import analyze_patient_exam, run_quality_gate
from medhack_ai_assistant.services.dashboard import build_patient_exam


def setup_page() -> None:
    st.set_page_config(
        page_title="MedHack AI Assistant",
        page_icon="🏥",
        layout="wide",
    )
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


def render_header() -> None:
    st.title("MedHack AI Assistant")
    st.caption("Минимальная Streamlit-версия для просмотра медосмотров и проверки противопоказаний.")


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
        st.dataframe(_readable_dataframe(dataframe.head(20)), use_container_width=True)


def render_patient_summary(exam: PatientExam) -> None:
    st.subheader("Карточка пациента")

    col_patient, col_exam, col_date, col_factors = st.columns(4)
    col_patient.metric("Пациент", exam.patient_id)
    col_exam.metric("Осмотр", exam.exam_row_id)
    col_date.metric("Дата", _display_value(exam.consultation_date) or "не указана")
    col_factors.metric("Факторов", len(exam.assigned_harmful_factors))

    row = st.session_state.get("selected_row")
    if isinstance(row, pd.Series):
        _render_known_target(row)


def render_ai_section(exam: PatientExam) -> None:
    st.subheader("AI-анализ")
    quality = run_quality_gate(exam)

    if quality.can_analyze:
        st.success("Данных достаточно для анализа.")
    else:
        st.warning("Данных недостаточно для надежного анализа.")
        for reason in quality.reasons:
            st.write(f"- {_translate_quality_reason(reason)}")

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
        _render_analysis_result(result)


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
            "Специалист": _display_value(item.specialist),
            "Дата": _display_value(item.consultation_date),
            "Группа здоровья": _display_value(item.health_group),
            "МКБ": _display_value(item.mkb_code),
            "Описание МКБ": _display_value(item.mkb_description),
            "Заключение": _shorten(_display_value(item.conclusion), 220),
        }
        for item in exam.specialist_conclusions
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


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


def _render_known_target(row: pd.Series) -> None:
    if TARGET_COLUMN not in row.index or pd.isna(row[TARGET_COLUMN]):
        return

    st.markdown("**Разметка из train.csv**")
    has_contraindications = bool(row[TARGET_COLUMN])
    if has_contraindications:
        factors = _display_value(row.get("contraindicated_factors", "")) or "факторы не указаны"
        st.markdown(
            f"<div class='result-bad'><b>Есть противопоказания.</b><br>Факторы: {factors}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='result-ok'><b>Противопоказания в разметке не указаны.</b></div>",
            unsafe_allow_html=True,
        )


def _render_analysis_result(result: AnalysisResult) -> None:
    verdict_class = {
        "FIT": "result-ok",
        "UNFIT": "result-bad",
        "NEEDS_MORE_INFO": "result-warn",
    }.get(result.verdict, "result-warn")
    verdict_label = {
        "FIT": "Противопоказания не найдены",
        "UNFIT": "Есть противопоказания",
        "NEEDS_MORE_INFO": "Нужно больше данных",
    }.get(result.verdict, result.verdict)

    confidence = result.raw_llm_response.get("confidence")
    confidence_text = ""
    if isinstance(confidence, int | float) and confidence > 0:
        confidence_text = f"<br>Уверенность: {confidence:.0%}"

    st.markdown(
        f"<div class='{verdict_class}'><b>{verdict_label}</b>{confidence_text}"
        f"<br>{_display_value(result.summary)}</div>",
        unsafe_allow_html=True,
    )

    if result.factors:
        st.markdown("**Факторы с противопоказаниями**")
        st.write(", ".join(result.factors))

    if result.evidence:
        st.markdown("**Обоснование**")
        evidence = [
            {
                "Фактор": _display_value(item.get("factor", "")),
                "МКБ": _display_value(item.get("mkb_code", "")),
                "Источник": _display_value(item.get("source_specialist", "")),
                "Причина": _display_value(item.get("reason", "")),
            }
            for item in result.evidence
        ]
        st.dataframe(pd.DataFrame(evidence), use_container_width=True, hide_index=True)

    follow_up = result.follow_up_draft or {}
    specialists = follow_up.get("specialists") or []
    tests = follow_up.get("tests") or []
    if specialists or tests:
        st.markdown("**Рекомендации**")
        if specialists:
            st.write("Специалисты: " + ", ".join(map(_display_value, specialists)))
        if tests:
            st.write("Обследования: " + ", ".join(map(_display_value, tests)))


def _readable_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = result[column].map(_display_value)
    return result


def _display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    try:
        fixed = text.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if _mojibake_score(fixed) < _mojibake_score(text) else text


def _mojibake_score(text: str) -> int:
    markers = ("Р", "С", "Ð", "Ñ", "Ѓ", "Џ", "Ќ", "Ђ")
    return sum(text.count(marker) for marker in markers)


def _shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def _translate_quality_reason(reason: str) -> str:
    translations = {
        "Missing assigned harmful factors.": "не указаны вредные факторы",
        "Missing specialist conclusions.": "нет заключений специалистов",
    }
    return translations.get(reason, _display_value(reason))
