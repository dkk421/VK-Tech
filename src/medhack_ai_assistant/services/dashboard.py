from typing import Any
from pathlib import Path

import pandas as pd

from medhack_ai_assistant.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from medhack_ai_assistant.data import load_data
from medhack_ai_assistant.domain.models import (
    DashboardFinding,
    DashboardResult,
    PatientExam,
    SpecialistConclusion,
)
from medhack_ai_assistant.parsing.conclusions import (
    parse_specialist_conclusions,
    split_factor_codes,
)


def load_backend_dataset(path: str | None = None, *, use_train: bool = False) -> pd.DataFrame:
    dataset_path = TRAIN_PATH if use_train else TEST_PATH
    if path is not None:
        dataset_path = Path(path)
    return load_data(dataset_path)


def get_dashboard_by_exam_id(
    exam_row_id: int,
    data: pd.DataFrame | None = None,
    *,
    use_train: bool = False,
) -> DashboardResult:
    dataset = data if data is not None else load_backend_dataset(use_train=use_train)
    matched = dataset.loc[dataset[ID_COLUMN] == exam_row_id]

    if matched.empty:
        raise ValueError(f"Exam row not found: {exam_row_id}")

    return build_dashboard(matched.iloc[0])


def build_dashboard(row: pd.Series | dict[str, Any]) -> DashboardResult:
    exam = build_patient_exam(row)
    diagnoses = _build_diagnosis_findings(exam.specialist_conclusions)
    normal_items = _build_normal_findings(exam.specialist_conclusions)
    decision_label, decision_reason = _build_decision(exam, diagnoses)

    return DashboardResult(
        exam=exam,
        diagnoses=diagnoses,
        normal_items=normal_items,
        decision_label=decision_label,
        decision_reason=decision_reason,
    )


def build_patient_exam(row: pd.Series | dict[str, Any]) -> PatientExam:
    value = row.get if isinstance(row, dict) else row.get
    has_target = value(TARGET_COLUMN, None)

    return PatientExam(
        exam_row_id=int(value("exam_row_id")),
        patient_id=int(value("patient_id")),
        consultation_date=str(value("consultation_date", "")),
        assigned_harmful_factors=split_factor_codes(value("assigned_harmful_factors", "")),
        specialist_conclusions=parse_specialist_conclusions(value("specialist_conclusions", "")),
        has_contraindications=_to_optional_bool(has_target),
        contraindicated_factors=split_factor_codes(value("contraindicated_factors", "")),
    )


def _build_diagnosis_findings(
    conclusions: tuple[SpecialistConclusion, ...],
) -> tuple[DashboardFinding, ...]:
    return tuple(
        DashboardFinding(
            title=conclusion.mkb_description or conclusion.conclusion or "Диагноз без описания",
            status="diagnosis",
            details=conclusion.conclusion,
            source_specialist=conclusion.specialist,
            source_date=conclusion.consultation_date,
            mkb_code=conclusion.mkb_code,
        )
        for conclusion in conclusions
        if conclusion.has_diagnosis
    )



def _build_normal_findings(
    conclusions: tuple[SpecialistConclusion, ...],
) -> tuple[DashboardFinding, ...]:
    return tuple(
        DashboardFinding(
            title=conclusion.specialist or "Осмотр",
            status="normal",
            details=conclusion.conclusion or conclusion.mkb_description or "Без значимых отклонений",
            source_specialist=conclusion.specialist,
            source_date=conclusion.consultation_date,
            mkb_code=conclusion.mkb_code,
        )
        for conclusion in conclusions
        if conclusion.has_diagnosis
    )


def _build_decision(
    exam: PatientExam,
    diagnoses: tuple[DashboardFinding, ...],
) -> tuple[str, str]:
    if exam.has_contraindications is True:
        factors = ", ".join(exam.contraindicated_factors) or "не указаны"
        return "Есть противопоказания", f"В обучающих данных указаны противопоказанные факторы: {factors}."

    if exam.has_contraindications is False:
        return "Противопоказания не указаны", "В обучающих данных для этого осмотра нет противопоказаний."

    if diagnoses:
        return "??????? ????????", "? ????????? ????????????????? ?????? ???? ????????, ??????? ????? ?????? AI-????????."

    return "Явных отклонений не найдено", "В доступных структурированных данных нет диагнозов или маркеров внимания."


def _to_optional_bool(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    return bool(value)
