import pandas as pd
import streamlit as st

from medhack_ai_assistant.domain.models import AnalysisResult
from medhack_ai_assistant.ui.formatters import display_value


def render_analysis_result(result: AnalysisResult) -> None:
    render_ai_verdict(result)
    render_ai_summary(result)
    render_ai_evidence(result)
    render_follow_up(result)


def render_ai_verdict(result: AnalysisResult) -> None:
    verdict = result.verdict

    if verdict == "FIT":
        st.success("Явных противопоказаний не обнаружено")
    elif verdict == "UNFIT":
        st.error("Возможны противопоказания")
    else:
        st.warning("Требуется дополнительная проверка")


def render_ai_summary(result: AnalysisResult) -> None:
    summary = display_value(result.summary)
    if summary:
        st.markdown("**Краткий вывод**")
        st.write(summary)

    if result.factors:
        st.markdown("**Факторы с противопоказаниями**")
        st.write(", ".join(map(display_value, result.factors)))


def render_ai_evidence(result: AnalysisResult) -> None:
    if not result.evidence:
        return

    st.markdown("**Обоснование**")
    evidence = [
        {
            "Фактор": display_value(item.get("factor", "")),
            "МКБ": display_value(item.get("mkb_code", "")),
            "Источник": display_value(item.get("source_specialist", "")),
            "Причина": display_value(item.get("reason", "")),
        }
        for item in result.evidence
    ]
    st.dataframe(pd.DataFrame(evidence), width="stretch", hide_index=True)


def render_follow_up(result: AnalysisResult) -> None:
    follow_up = result.follow_up_draft or {}
    specialists = follow_up.get("specialists") or []
    tests = follow_up.get("tests") or []
    if not specialists and not tests:
        return

    st.markdown("**Рекомендации**")
    if specialists:
        st.write("Специалисты: " + ", ".join(map(display_value, specialists)))
    if tests:
        st.write("Обследования: " + ", ".join(map(display_value, tests)))
