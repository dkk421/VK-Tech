import json
from collections.abc import Iterable
from typing import Any

import pandas as pd

from medhack_ai_assistant.domain.models import SpecialistConclusion


def split_factor_codes(raw_value: Any) -> tuple[str, ...]:
    if pd.isna(raw_value):
        return ()
    return tuple(
        part.strip()
        for part in str(raw_value).split(";")
        if part.strip()
    )


def parse_specialist_conclusions(raw_value: Any) -> tuple[SpecialistConclusion, ...]:
    if pd.isna(raw_value) or raw_value == "":
        return ()

    payload = _load_json_array(raw_value)
    conclusions: list[SpecialistConclusion] = []

    for item in payload:
        if not isinstance(item, dict):
            continue

        conclusions.append(
            SpecialistConclusion(
                specialist=_clean(item.get("specialist")),
                consultation_date=_clean(item.get("consultation_date")),
                conclusion=_clean(item.get("conclusion")),
                health_group=_clean(item.get("health_group")),
                mkb_code=_clean(item.get("mkb_code")),
                mkb_description=_clean(item.get("mkb_description")),
            )
        )

    return tuple(conclusions)


def _load_json_array(raw_value: Any) -> Iterable[Any]:
    if isinstance(raw_value, list):
        return raw_value

    try:
        payload = json.loads(str(raw_value))
    except json.JSONDecodeError as exc:
        raise ValueError("specialist_conclusions contains invalid JSON") from exc

    if not isinstance(payload, list):
        raise ValueError("specialist_conclusions must be a JSON array")

    return payload


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()

