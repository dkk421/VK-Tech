"""Формирование развёрнутого отчёта экспертной системы."""

from __future__ import annotations

from dataclasses import dataclass

from src.medhack_ai_assistant.case_model import ExamCase
from src.medhack_ai_assistant.kb_loader import factor_title_map
from src.medhack_ai_assistant.rule_engine import ExpertInferenceResult, RuleTrace


@dataclass
class FactorAssessment:
    factor_code: str
    factor_title: str
    status: str
    traces: list[RuleTrace]


@dataclass
class ExpertReport:
    exam_row_id: int
    summary: str
    has_contraindications: bool
    contraindicated_factors: str
    factor_assessments: list[FactorAssessment]
    unmatched_pathologies: list[str]
    disclaimer: str
    markdown: str


def _status_for_factor(factor: str, result: ExpertInferenceResult) -> tuple[str, list[RuleTrace]]:
    traces = [trace for trace in result.traces if trace.factor_code == factor]
    if traces:
        return "противопоказание", traces
    partial = [trace for trace in result.partial_matches if trace.factor_code == factor]
    if partial:
        return "недостаточно данных (возможное противопоказание)", partial
    return "допуск", []


def build_expert_report(case: ExamCase, result: ExpertInferenceResult) -> ExpertReport:
    titles = factor_title_map()
    assessments: list[FactorAssessment] = []

    for factor in sorted(case.assigned_factors, key=lambda value: (len(value.split(".")), value)):
        status, traces = _status_for_factor(factor, result)
        assessments.append(
            FactorAssessment(
                factor_code=factor,
                factor_title=titles.get(factor, factor),
                status=status,
                traces=traces,
            )
        )

    triggered_mkb = {
        (trace.finding.mkb_code, trace.finding.conclusion[:80])
        for trace in result.traces
    }
    unmatched: list[str] = []
    for finding in case.pathology_findings:
        key = (finding.mkb_code, finding.conclusion[:80])
        if key not in triggered_mkb:
            label = finding.specialist
            if finding.mkb_code:
                label += f", МКБ {finding.mkb_code}"
            if finding.conclusion:
                label += f": {finding.conclusion[:120]}"
            unmatched.append(label)

    n_contra = len(result.contraindicated_factors)
    n_assigned = len(case.assigned_factors)
    if n_contra:
        summary = (
            f"Выявлены противопоказания к {n_contra} из {n_assigned} "
            f"назначенных факторов: {result.contraindicated_factors_str}."
        )
    else:
        summary = (
            f"По данным профосмотра противопоказаний к {n_assigned} "
            "назначенным факторам не выявлено."
        )

    lines = [
        f"## Заключение экспертной системы (приказ 29н)",
        f"**ID осмотра:** {case.exam_row_id}",
        "",
        f"### Резюме",
        summary,
        "",
        "### Оценка по каждому фактору",
    ]

    for assessment in assessments:
        lines.append(f"\n#### П. {assessment.factor_code} — {assessment.factor_title}")
        lines.append(f"- **Статус:** {assessment.status}")
        if assessment.traces:
            for trace in assessment.traces:
                finding = trace.finding
                lines.append(
                    f"- **Правило {trace.rule_id}** ({trace.appendix_ref or 'Прил.2'}): "
                    f"{trace.disease_name}"
                )
                lines.append(
                    f"  - Основание: {finding.specialist}, МКБ {finding.mkb_code or '—'}, "
                    f"«{finding.conclusion[:200]}»"
                )
                lines.append(f"  - Тип совпадения: {trace.match_type}, уверенность {trace.confidence:.0%}")

    if unmatched:
        lines.append("\n### Патология без срабатывания правил по назначенным факторам")
        for item in unmatched[:10]:
            lines.append(f"- {item}")

    disclaimer = (
        "Решение носит рекомендательный характер и основано на автоматическом "
        "сопоставлении с Перечнем медицинских противопоказаний (приказ 29н). "
        "Окончательное заключение формирует врач-профпатолог."
    )
    lines.extend(["", "### Ограничения", disclaimer])

    return ExpertReport(
        exam_row_id=case.exam_row_id,
        summary=summary,
        has_contraindications=result.has_contraindications,
        contraindicated_factors=result.contraindicated_factors_str,
        factor_assessments=assessments,
        unmatched_pathologies=unmatched,
        disclaimer=disclaimer,
        markdown="\n".join(lines),
    )
