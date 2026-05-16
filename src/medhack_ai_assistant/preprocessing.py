import pandas as pd

from .data import validate_columns


def build_text_feature(
    df: pd.DataFrame,
    text_columns: tuple[str, ...],
) -> pd.Series:
    validate_columns(df, text_columns)

    text_parts = df.loc[:, text_columns].fillna("").astype(str)
    return text_parts.agg(" ".join, axis=1)
