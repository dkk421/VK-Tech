"""Пайплайн экспертной системы: инференс, CV, submission."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from src.medhack_ai_assistant.case_model import build_exam_case
from src.medhack_ai_assistant.config import (
    EXPERT_SUBMISSION_PATH,
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from src.medhack_ai_assistant.data import load_data, save_expert_submission
from src.medhack_ai_assistant.explanation import ExpertReport, build_expert_report
from src.medhack_ai_assistant.kb_loader import load_rules_catalog, rules_by_factor
from src.medhack_ai_assistant.metrics import calculate_autonomous_score
from src.medhack_ai_assistant.rule_engine import ExpertInferenceResult, infer_case


@dataclass(frozen=True)
class ExpertPipelineResult:
    submission: pd.DataFrame
    submission_path: Path
    metrics: dict[str, float] | None
    sample_report: ExpertReport | None


def infer_dataframe(
    df: pd.DataFrame,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, list[ExpertInferenceResult], list[ExpertReport]]:
    rules_index = rules_by_factor(load_rules_catalog().rules)
    rows: list[dict] = []
    results: list[ExpertInferenceResult] = []
    reports: list[ExpertReport] = []

    for _, row in df.iterrows():
        case = build_exam_case(row)
        inference = infer_case(case, strict=strict, rules_index=rules_index)
        report = build_expert_report(case, inference)
        results.append(inference)
        reports.append(report)
        rows.append({
            ID_COLUMN: case.exam_row_id,
            "contraindicated_factors": inference.contraindicated_factors_str,
            "has_contraindications": inference.has_contraindications,
            "expert_confidence": inference.confidence,
        })

    return pd.DataFrame(rows), results, reports


def evaluate_expert_cv(
    train: pd.DataFrame,
    *,
    n_splits: int = 5,
    strict: bool = True,
) -> dict[str, float]:
    if "patient_id" not in train.columns:
        preds, _, _ = infer_dataframe(train, strict=strict)
        return calculate_autonomous_score(
            train["contraindicated_factors"],
            preds["contraindicated_factors"],
        )

    oof = pd.Series(index=train.index, dtype=object)
    groups = train["patient_id"]

    for fold, (_, valid_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(train, groups=groups)
    ):
        valid = train.iloc[valid_idx]
        preds, _, _ = infer_dataframe(valid, strict=strict)
        oof.iloc[valid_idx] = preds["contraindicated_factors"].values

    oof = oof.fillna("")
    return calculate_autonomous_score(train["contraindicated_factors"], oof)


def run_expert_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    submission_path: Path = EXPERT_SUBMISSION_PATH,
    *,
    evaluate_train: bool = True,
    strict: bool = True,
    sample_exam_row_id: int | None = 1015330919,
) -> ExpertPipelineResult:
    train = load_data(train_path)
    test = load_data(test_path)

    metrics = None
    if evaluate_train and TARGET_COLUMN in train.columns:
        metrics = evaluate_expert_cv(train, strict=strict)

    test_preds, _, test_reports = infer_dataframe(test, strict=strict)
    submission = save_expert_submission(
        exam_row_id=test[ID_COLUMN],
        contraindicated_factors=test_preds["contraindicated_factors"],
        has_contraindications=test_preds["has_contraindications"],
        output_path=submission_path,
    )

    sample_report = None
    if sample_exam_row_id is not None:
        match = test[test[ID_COLUMN] == sample_exam_row_id]
        if not match.empty:
            case = build_exam_case(match.iloc[0])
            inference = infer_case(case, strict=strict)
            sample_report = build_expert_report(case, inference)
        elif sample_exam_row_id and (train[ID_COLUMN] == sample_exam_row_id).any():
            case = build_exam_case(train[train[ID_COLUMN] == sample_exam_row_id].iloc[0])
            inference = infer_case(case, strict=strict)
            sample_report = build_expert_report(case, inference)

    return ExpertPipelineResult(
        submission=submission,
        submission_path=submission_path,
        metrics=metrics,
        sample_report=sample_report,
    )


def infer_single_row(row: pd.Series, *, strict: bool = True) -> tuple[ExpertInferenceResult, ExpertReport]:
    case = build_exam_case(row)
    inference = infer_case(case, strict=strict)
    report = build_expert_report(case, inference)
    return inference, report
