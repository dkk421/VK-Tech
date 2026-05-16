from pathlib import Path

import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")


def build_binary_submission(
    exam_row_id: pd.Series,
    has_contraindications: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame({
        "exam_row_id": exam_row_id,
        "has_contraindications": has_contraindications.astype(bool),
    })


def save_submission(
    exam_row_id: pd.Series,
    predictions: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    submission = build_binary_submission(exam_row_id, predictions)
    submission.to_csv(output_path, index=False)
    return submission


def save_expert_submission(
    exam_row_id: pd.Series,
    contraindicated_factors: pd.Series,
    has_contraindications: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    submission = pd.DataFrame({
        "exam_row_id": exam_row_id,
        "contraindicated_factors": contraindicated_factors.fillna("").astype(str),
        "has_contraindications": has_contraindications.astype(bool),
    })
    submission.to_csv(output_path, index=False)
    return submission
