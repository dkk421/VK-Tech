from typing import Any

import pandas as pd

from medhack_ai_assistant.domain.models import PatientExam, QualityGateResult
from medhack_ai_assistant.services.dashboard import build_patient_exam


def run_quality_gate(row_or_exam: pd.Series | dict[str, Any] | PatientExam) -> QualityGateResult:
    """Check whether a patient package has enough data for local analysis."""
    reasons: list[str] = []

    try:
        exam = _ensure_exam(row_or_exam)
    except Exception as exc:
        return QualityGateResult(
            status="NEEDS_MORE_INFO",
            reasons=(f"Cannot parse patient package: {exc}",),
            can_analyze=False,
        )

    if not exam.assigned_harmful_factors:
        reasons.append("Missing assigned harmful factors.")
    if not exam.specialist_conclusions:
        reasons.append("Missing specialist conclusions.")

    if reasons:
        return QualityGateResult(
            status="NEEDS_MORE_INFO",
            reasons=tuple(reasons),
            can_analyze=False,
        )

    return QualityGateResult(status="READY", reasons=(), can_analyze=True)


def _ensure_exam(row_or_exam: pd.Series | dict[str, Any] | PatientExam) -> PatientExam:
    if isinstance(row_or_exam, PatientExam):
        return row_or_exam
    return build_patient_exam(row_or_exam)
