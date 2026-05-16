from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.medhack_ai_assistant.config import ID_COLUMN, TEXT_COLUMNS
from src.medhack_ai_assistant.data import build_binary_submission, validate_columns
from src.medhack_ai_assistant.expert_pipeline import (
    infer_dataframe,
    infer_single_row,
    run_expert_pipeline,
)
from src.medhack_ai_assistant.hybrid import run_hybrid_pipeline


st.set_page_config(page_title="MedHack Expert 29н", layout="wide")
st.title("Экспертная система допуска к работе (приказ 29н)")

mode = st.sidebar.selectbox(
    "Режим",
    ("Анализ загруженного CSV", "Сгенерировать submission (expert)", "Сгенерировать submission (hybrid)"),
)

uploaded = st.file_uploader("CSV (train или test)", type=["csv"])

if mode.startswith("Сгенерировать"):
    if st.button("Запустить пайплайн"):
        with st.spinner("Обработка..."):
            if mode.endswith("(hybrid)"):
                result = run_hybrid_pipeline(evaluate_train=False)
            else:
                result = run_expert_pipeline(evaluate_train=False)
        st.success(f"Файл сохранён: {result.submission_path}")
        st.dataframe(result.submission.head(20))
        buf = io.BytesIO()
        result.submission.to_csv(buf, index=False)
        st.download_button(
            "Скачать submission",
            buf.getvalue(),
            file_name=result.submission_path.name,
            mime="text/csv",
        )
    st.stop()

if uploaded is None:
    st.info("Загрузите CSV с колонками assigned_harmful_factors и specialist_conclusions.")
    st.stop()

df = pd.read_csv(uploaded)
validate_columns(df, (*TEXT_COLUMNS, ID_COLUMN))

st.subheader("Данные")
st.dataframe(df[[ID_COLUMN, "assigned_harmful_factors"]].head(50))

exam_ids = df[ID_COLUMN].astype(int).tolist()
selected_id = st.selectbox("exam_row_id", exam_ids)

row = df[df[ID_COLUMN] == selected_id].iloc[0]
inference, report = infer_single_row(row)

col1, col2 = st.columns(2)
with col1:
    st.metric("Противопоказания", "Да" if inference.has_contraindications else "Нет")
    st.write("**Факторы с противопоказаниями:**")
    st.code(report.contraindicated_factors or "(пусто)")
    st.write(f"Уверенность: {inference.confidence:.0%}")

with col2:
    st.write("**Назначенные факторы:**")
    st.code(row["assigned_harmful_factors"])

st.markdown(report.markdown)

st.subheader("Выгрузка результата")
st.caption("CSV с бинарным решением по всем строкам загруженного файла.")


@st.cache_data(show_spinner="Анализ всех строк…")
def analyze_uploaded_csv(dataframe: pd.DataFrame) -> pd.DataFrame:
    predictions, _, _ = infer_dataframe(dataframe)
    return predictions


if st.button("Проанализировать весь файл", type="primary"):
    st.session_state["analysis_ready"] = True

if st.session_state.get("analysis_ready"):
    preds = analyze_uploaded_csv(df)
    export_df = build_binary_submission(preds[ID_COLUMN], preds["has_contraindications"])

    st.dataframe(export_df.head(20))
    st.caption(f"Всего строк: {len(export_df)}")

    csv_buffer = io.BytesIO()
    export_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    st.download_button(
        label="Скачать analyzed.csv",
        data=csv_buffer.getvalue(),
        file_name="analyzed.csv",
        mime="text/csv",
    )

    with st.expander("Подробный результат (с факторами)"):
        st.dataframe(
            preds[
                [
                    ID_COLUMN,
                    "contraindicated_factors",
                    "has_contraindications",
                    "expert_confidence",
                ]
            ].head(100)
        )