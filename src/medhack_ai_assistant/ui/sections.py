import pandas as pd
import streamlit as st

from medhack_ai_assistant.config import ID_COLUMN, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from medhack_ai_assistant.data import load_data
from medhack_ai_assistant.domain.models import PatientExam
from medhack_ai_assistant.pipeline import run_quality_gate
from medhack_ai_assistant.services.dashboard import build_patient_exam
from medhack_ai_assistant.services.document_export import (
    MedicalConclusionData,
    generate_medical_docx,
)
from medhack_ai_assistant.ui.ai_rendering import render_analysis_result
from medhack_ai_assistant.ui.formatters import readable_dataframe, translate_quality_reason
from medhack_ai_assistant.ui.styles import render_global_styles
from medhack_ai_assistant.ui.cards import render_patient_summary_card
from solution import analyze_row, analyze_row_as_result, format_factors


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
            key="dataset_source",
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

    if st.button("Запустить AI-анализ", type="primary", width="stretch"):
        with st.spinner("Анализирую осмотр..."):
            try:
                row = st.session_state.get("selected_row")
                if row is None:
                    raise ValueError("Selected source row is missing.")
                result = analyze_row_as_result(row, exam=exam)
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
    _render_single_patient_document_downloads(exam, result)


def render_batch_analysis_launcher() -> None:
    train_rows = len(_load_builtin_dataset(True))

    with st.container(border=True):
        st.markdown("### Анализ пациентов")
        st.caption("Локальный анализ по train.csv через правила из solution.py")
        count = st.number_input(
            "Количество пациентов",
            min_value=1,
            max_value=max(train_rows, 1),
            value=min(10, train_rows),
            step=1,
        )

        if st.button("Сделать анализ", type="primary", width="stretch"):
            st.session_state["batch_analysis_page"] = True
            st.session_state["batch_analysis_limit"] = int(count)
            st.rerun()


def _render_single_patient_document_downloads(exam: PatientExam, result) -> None:
    factors = format_factors(list(result.factors))
    is_unfit = result.verdict == "UNFIT" and factors != "0"
    mkb_codes = tuple(result.raw_llm_response.get("mkb_codes") or ())
    conclusion_data = MedicalConclusionData(
        exam=exam,
        factors=factors,
        is_unfit=is_unfit,
        summary=result.summary,
        mkb_codes=mkb_codes,
    )

    st.download_button(
        "Скачать заключение DOCX",
        data=generate_medical_docx(conclusion_data),
        file_name=f"medical_conclusion_{exam.exam_row_id}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"single_docx_{exam.exam_row_id}",
        width="stretch",
    )


def render_batch_analysis_page() -> None:
    limit = int(st.session_state.get("batch_analysis_limit", 10))
    train = _load_builtin_dataset(True).head(limit)

    left, right = st.columns([1, 4])
    with left:
        if st.button("Назад", width="stretch"):
            st.session_state["batch_analysis_page"] = False
            st.rerun()
    with right:
        st.subheader(f"Анализ {len(train)} пациентов из train.csv")
        st.caption("Для каждого пациента используется тот же локальный контур, что и в кнопке AI-анализа.")

    with st.spinner("Формирую список заключений..."):
        results = _build_batch_analysis_results(train)

    unfit_count = sum(1 for item in results if item["is_unfit"])
    fit_count = len(results) - unfit_count
    metric_fit, metric_unfit, metric_total = st.columns(3)
    metric_total.metric("Всего", len(results))
    metric_fit.metric("Без противопоказаний", fit_count)
    metric_unfit.metric("С противопоказаниями", unfit_count)

    st.markdown("### Список пациентов")
    for item in results:
        _render_batch_patient_row(item)


def _build_batch_analysis_results(dataframe: pd.DataFrame) -> list[dict]:
    results = []
    for _, row in dataframe.iterrows():
        try:
            exam = build_patient_exam(row)
            analysis = analyze_row(row)
            factors = format_factors(analysis["factors"])
            is_unfit = factors != "0"
            docx = generate_medical_docx(
                MedicalConclusionData(
                    exam=exam,
                    factors=factors,
                    is_unfit=is_unfit,
                    summary=analysis["summary"],
                    mkb_codes=tuple(analysis["mkb_codes"]),
                )
            )
            results.append(
                {
                    "exam": exam,
                    "analysis": analysis,
                    "factors": factors,
                    "is_unfit": is_unfit,
                    "docx": docx,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "exam": None,
                    "analysis": {"summary": f"Не удалось разобрать строку: {exc}", "mkb_codes": []},
                    "factors": "0",
                    "is_unfit": False,
                    "docx": b"",
                }
            )
    return results


def _render_batch_patient_row(item: dict) -> None:
    exam = item["exam"]
    analysis = item["analysis"]
    factors = item["factors"]
    is_unfit = item["is_unfit"]

    with st.container(border=True):
        id_col, info_col, factor_col, status_col, doc_col = st.columns([1, 2, 2, 1.4, 1.2])

        if exam is None:
            id_col.write("—")
            info_col.write(analysis["summary"])
            factor_col.write("0")
            status_col.warning("Ошибка данных")
            return

        if id_col.button(str(exam.exam_row_id), key=f"open_exam_{exam.exam_row_id}", width="stretch"):
            st.session_state["batch_analysis_page"] = False
            st.session_state["dataset_source"] = "train.csv"
            st.session_state["selected_exam_id"] = int(exam.exam_row_id)
            st.rerun()
        id_col.caption(f"Пациент #{exam.patient_id}")
        info_col.write(f"Дата: {exam.consultation_date or 'не указана'}")
        info_col.caption(f"Заключений специалистов: {len(exam.specialist_conclusions)}")
        factor_col.write(";".join(exam.assigned_harmful_factors) if exam.assigned_harmful_factors else "0")
        factor_col.caption(analysis["summary"])

        if is_unfit:
            status_col.error(f"Есть противопоказания: {factors}")
        else:
            status_col.success("Противопоказаний нет")

        doc_col.download_button(
            "Скачать DOCX",
            data=item["docx"],
            file_name=f"medical_conclusion_{exam.exam_row_id}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"docx_{exam.exam_row_id}",
            width="stretch",
        )


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
            index=_selected_exam_index(ids),
            format_func=str,
            key="selected_exam_id",
        )

        with st.expander("Показать первые строки"):
            st.dataframe(readable_dataframe(dataframe.head(20)), width="stretch")

        render_batch_analysis_launcher()

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


def _selected_exam_index(ids: list[int]) -> int:
    selected_id = st.session_state.get("selected_exam_id")
    if selected_id in ids:
        return ids.index(selected_id)
    return 0
