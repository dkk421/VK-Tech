from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_PATH,
    TEXT_COLUMNS,
    TRAIN_PATH,
    SUBMISSION_PATH,
    ModelConfig,
)
from .data import load_data, save_submission, validate_columns
from .ml import (
    ValidationResult,
    predict_with_threshold,
    train_and_validate,
    train_final_model,
)
from .preprocessing import build_text_feature


@dataclass(frozen=True)
class PipelineResult:
    target_distribution: pd.Series
    target_distribution_normalized: pd.Series
    validation: ValidationResult
    submission: pd.DataFrame
    submission_path: Path


def run_training_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    submission_path: Path = SUBMISSION_PATH,
    config: ModelConfig | None = None,
) -> PipelineResult:
    config = config or ModelConfig()

    train = load_data(train_path)
    test = load_data(test_path)

    validate_columns(train, (*TEXT_COLUMNS, TARGET_COLUMN))
    validate_columns(test, (*TEXT_COLUMNS, ID_COLUMN))

    x = build_text_feature(train, TEXT_COLUMNS)
    y = train[TARGET_COLUMN].astype(bool)
    x_test = build_text_feature(test, TEXT_COLUMNS)

    validation = train_and_validate(x, y, config)

    model = train_final_model(x, y, config)
    test_pred = predict_with_threshold(model, x_test, validation.threshold)

    submission = save_submission(
        exam_row_id=test[ID_COLUMN],
        predictions=test_pred,
        output_path=submission_path,
    )

    return PipelineResult(
        target_distribution=y.value_counts(),
        target_distribution_normalized=y.value_counts(normalize=True),
        validation=validation,
        submission=submission,
        submission_path=submission_path,
    )
