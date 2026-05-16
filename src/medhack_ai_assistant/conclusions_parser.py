"""Парсинг assigned_harmful_factors и specialist_conclusions."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from src.medhack_ai_assistant.metrics import parse_factors_to_set

EXCLUSION_PHRASES = (
    "не выявлен",
    "не обнаружен",
    "без патолог",
    "практически здоров",
    "здоров",
    "санирована",
    "без нарушения функции",
    "вне обострения",
    "ремиссия",
    "достижнуто целевое",
    "контролируем",
)

NORMAL_MKB_PREFIXES = ("Z00", "Z01", "Z02", "Z03", "Z04", "Z08", "Z09", "Z10", "Z11", "Z12")


def parse_factors(text: str | float | None) -> set[str]:
    return parse_factors_to_set(text)


def safe_parse_conclusions(raw: str | float | None) -> list[dict[str, Any]]:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return []
    text = str(raw).strip()
    if not text or text == "[]":
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def parse_health_group_tier(health_group: str) -> int | None:
    if not health_group:
        return None
    normalized = health_group.lower().replace(" ", "")
    if "3б" in normalized or "3b" in normalized:
        return 4
    if "3а" in normalized or "3a" in normalized:
        return 3
    if "3" in normalized:
        return 3
    if "2" in normalized:
        return 2
    if "1" in normalized:
        return 1
    return None


def extract_hearing_stage(text: str) -> int | None:
    lowered = text.lower()
    match = re.search(r"тугоухост[ьи]\s*(\d+)\s*ст", lowered)
    if match:
        return int(match.group(1))
    if "iii" in lowered or "3 ст" in lowered:
        return 3
    if "ii" in lowered or "2 ст" in lowered:
        return 2
    if "i ст" in lowered or "1 ст" in lowered:
        return 1
    return None


def extract_hypertension_stage(text: str) -> int | None:
    lowered = text.lower()
    for pattern in (
        r"(?:гб|аг|гипертенз)\w*\s*(\d)\s*ст",
        r"(\d)\s*ст(?:епени)?",
        r"iii\s*ст",
        r"ii\s*ст",
    ):
        match = re.search(pattern, lowered)
        if match:
            if match.lastindex:
                return int(match.group(1))
    if "3ст" in lowered or "3 ст" in lowered or "iii" in lowered:
        return 3
    if "2ст" in lowered or "2 ст" in lowered or "ii" in lowered:
        return 2
    if "1ст" in lowered or "1 ст" in lowered:
        return 1
    return None


def extract_myopia_degree(text: str) -> int | None:
    lowered = text.lower()
    if "выс" in lowered or "высок" in lowered:
        return 3
    if "ср" in lowered or "средн" in lowered:
        return 2
    if "слаб" in lowered:
        return 1
    return None


def mkb_prefix(code: str) -> str:
    code = (code or "").strip().upper()
    if not code:
        return ""
    if len(code) >= 3 and code[0].isalpha():
        return code[:3]
    return code


def is_normal_finding(mkb_code: str, conclusion: str) -> bool:
    code = (mkb_code or "").strip().upper()
    text = (conclusion or "").lower().strip()
    if code and any(code.startswith(prefix) for prefix in NORMAL_MKB_PREFIXES):
        if not text or any(phrase in text for phrase in EXCLUSION_PHRASES):
            return True
        if len(text) < 40 and "осмотр" in text:
            return True
    if not code and text:
        if any(phrase in text for phrase in ("здоров", "z00", "не выявлен", "без патолог")):
            return True
    return False


def has_exclusion_context(text: str) -> bool:
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in EXCLUSION_PHRASES)


def format_factors(codes: set[str]) -> str:
    if not codes:
        return ""
    return "; ".join(sorted(codes, key=lambda value: (len(value.split(".")), value)))
