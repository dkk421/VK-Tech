"""Гибрид экспертной системы и ML-модели."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.medhack_ai_assistant.case_model import build_exam_case
from src.medhack_ai_assistant.config import (
    HYBRID_SUBMISSION_PATH,
    ID_COLUMN,
    ModelConfig,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
)
from src.medhack_ai_assistant.data import load_data, save_expert_submission
from src.medhack_ai_assistant.expert_pipeline import infer_dataframe
from src.medhack_ai_assistant.metrics import calculate_autonomous_score
from src.medhack_ai_assistant.ml import (
    predict_with_threshold,
    train_and_validate,
    train_final_model,
)
from src.medhack_ai_assistant.preprocessing import build_text_feature
from src.medhack_ai_assistant.config import TEXT_COLUMNS
from src.medhack_ai_assistant.conclusions_parser import format_factors
from src.medhack_ai_assistant.rule_engine import infer_case


@dataclass(frozen=True)
class HybridPipelineResult:
    submission: pd.DataFrame
    submission_path: Path
    metrics: dict[str, float] | None
    ml_threshold: float


def _factors_from_partial_matches(case, ml_unfit: bool) -> set[str]:
    """Если ML сигнализирует о негодности, добавить факторы с частичным совпадением правил."""
    if not ml_unfit:
        return set()
    loose = infer_case(case, strict=False)
    return loose.contraindicated_factors & case.assigned_factors


def run_hybrid_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    submission_path: Path = HYBRID_SUBMISSION_PATH,
    config: ModelConfig | None = None,
    *,
    evaluate_train: bool = True,
) -> HybridPipelineResult:
    config = config or ModelConfig()
    train = load_data(train_path)
    test = load_data(test_path)

    x_train = build_text_feature(train, TEXT_COLUMNS)
    y_train = train[TARGET_COLUMN].astype(bool)
    x_test = build_text_feature(test, TEXT_COLUMNS)

    validation = train_and_validate(x_train, y_train, config)
    model = train_final_model(x_train, y_train, config)
    ml_pred = predict_with_threshold(model, x_test, validation.threshold)

    expert_preds, _, _ = infer_dataframe(test, strict=True)

    hybrid_factors: list[str] = []
    hybrid_flags: list[bool] = []

    for idx, row in test.iterrows():
        case = build_exam_case(row)
        expert_row = expert_preds.loc[expert_preds[ID_COLUMN] == case.exam_row_id].iloc[0]
        factors = set(
            part.strip()
            for part in str(expert_row["contraindicated_factors"]).split(";")
            if part.strip()
        )
        ml_unfit = bool(ml_pred.loc[idx])
        if not factors and ml_unfit:
            factors = _factors_from_partial_matches(case, ml_unfit=True)
        if not factors and ml_unfit:
            partial = infer_case(case, strict=False)
            if partial.partial_matches:
                factors = {trace.factor_code for trace in partial.partial_matches}
                factors &= case.assigned_factors

        factor_str = format_factors(factors)
        hybrid_factors.append(factor_str)
        hybrid_flags.append(len(factors) > 0)

    submission = save_expert_submission(
        exam_row_id=test[ID_COLUMN],
        contraindicated_factors=pd.Series(hybrid_factors, index=test.index),
        has_contraindications=pd.Series(hybrid_flags, index=test.index),
        output_path=submission_path,
    )

    metrics = None
    if evaluate_train and "contraindicated_factors" in train.columns:
        train_expert, _, _ = infer_dataframe(train, strict=True)
        metrics = calculate_autonomous_score(
            train["contraindicated_factors"],
            train_expert["contraindicated_factors"],
        )

    return HybridPipelineResult(
        submission=submission,
        submission_path=submission_path,
        metrics=metrics,
        ml_threshold=validation.threshold,
    )
