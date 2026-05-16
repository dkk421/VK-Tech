from typing import Any

import pandas as pd


def readable_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    for column in result.columns:
        if result[column].dtype == "object":
            result[column] = result[column].map(display_value)
    return result


def display_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""

    text = str(value)
    try:
        fixed = text.encode("cp1251").decode("utf-8")
    except UnicodeError:
        return text
    return fixed if _mojibake_score(fixed) < _mojibake_score(text) else text


def shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "..."


def translate_quality_reason(reason: str) -> str:
    translations = {
        "Missing assigned harmful factors.": "не указаны вредные факторы",
        "Missing specialist conclusions.": "нет заключений специалистов",
    }
    return translations.get(reason, display_value(reason))


def _mojibake_score(text: str) -> int:
    markers = ("Р ", "РЎ", "Гђ", "Г‘", "Рѓ", "РЏ", "РЊ", "Р‚")
    return sum(text.count(marker) for marker in markers)
