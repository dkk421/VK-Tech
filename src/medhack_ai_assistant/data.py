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


def save_submission(
    exam_row_id: pd.Series,
    predictions: pd.Series,
    output_path: Path,
) -> pd.DataFrame:
    submission = pd.DataFrame({
        "exam_row_id": exam_row_id,
        "has_contraindications": predictions,
    })
    submission.to_csv(output_path, index=False)
    return submission
