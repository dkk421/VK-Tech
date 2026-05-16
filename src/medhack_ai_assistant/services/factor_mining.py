from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from medhack_ai_assistant.config import MINING_STATS_PATH, TARGET_COLUMN, TRAIN_PATH
from medhack_ai_assistant.data import load_data, validate_columns
from medhack_ai_assistant.domain.models import MiningContext, PatientExam
from medhack_ai_assistant.parsing.conclusions import (
    parse_specialist_conclusions,
    split_factor_codes,
)


TOP_LINKS_PER_FACTOR = 12
SIGNIFICANT_HEALTH_GROUPS = {"3а группа", "3б группа", "3a группа", "3b группа"}


def build_mining_stats(train_path: Path = TRAIN_PATH) -> dict[str, Any]:
    train = load_data(train_path)
    validate_columns(
        train,
        (
            "assigned_harmful_factors",
            "specialist_conclusions",
            "contraindicated_factors",
            TARGET_COLUMN,
        ),
    )

    assigned_counts: Counter[str] = Counter()
    positive_assigned_counts: Counter[str] = Counter()
    contraindicated_counts: Counter[str] = Counter()
    factor_mkb_counts: dict[str, Counter[str]] = defaultdict(Counter)
    factor_specialist_counts: dict[str, Counter[str]] = defaultdict(Counter)
    total_rows = len(train)
    positive_rows = 0
    bad_json_rows = 0
    empty_log_rows = 0

    for _, row in train.iterrows():
        factors = split_factor_codes(row.get("assigned_harmful_factors", ""))
        assigned_counts.update(factors)
        is_positive = bool(row.get(TARGET_COLUMN))

        try:
            conclusions = parse_specialist_conclusions(row.get("specialist_conclusions", ""))
        except ValueError:
            bad_json_rows += 1
            conclusions = ()

        if not conclusions:
            empty_log_rows += 1

        if not is_positive:
            continue

        positive_rows += 1
        positive_assigned_counts.update(factors)
        contraindicated_factors = split_factor_codes(row.get("contraindicated_factors", ""))
        contraindicated_counts.update(contraindicated_factors)

        target_factors = contraindicated_factors or factors
        significant_conclusions = [
            conclusion
            for conclusion in conclusions
            if _is_significant_conclusion(conclusion.mkb_code, conclusion.health_group)
        ]

        for factor in target_factors:
            for conclusion in significant_conclusions:
                if conclusion.mkb_code and conclusion.mkb_code != "Z00.0":
                    factor_mkb_counts[factor][conclusion.mkb_code] += 1
                if conclusion.specialist:
                    factor_specialist_counts[factor][conclusion.specialist] += 1

    factor_risks = {
        factor: {
            "assigned_count": count,
            "positive_count": positive_assigned_counts[factor],
            "risk": _safe_ratio(positive_assigned_counts[factor], count),
        }
        for factor, count in assigned_counts.items()
    }

    stats = {
        "meta": {
            "total_rows": total_rows,
            "positive_rows": positive_rows,
            "bad_json_rows": bad_json_rows,
            "empty_log_rows": empty_log_rows,
        },
        "top_assigned_factors": _counter_to_records(assigned_counts),
        "top_contraindicated_factors": _counter_to_records(contraindicated_counts),
        "factor_risks": factor_risks,
        "factor_mkb_links": {
            factor: _counter_to_records(counter, limit=TOP_LINKS_PER_FACTOR)
            for factor, counter in factor_mkb_counts.items()
        },
        "factor_specialist_links": {
            factor: _counter_to_records(counter, limit=TOP_LINKS_PER_FACTOR)
            for factor, counter in factor_specialist_counts.items()
        },
    }
    return stats


def load_or_build_mining_stats(
    path: Path = MINING_STATS_PATH,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"Mining stats must be a JSON object: {path}")
        return payload

    stats = build_mining_stats()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
    return stats


def get_mining_context(
    exam: PatientExam,
    stats: dict[str, Any] | None = None,
) -> MiningContext:
    stats = stats or load_or_build_mining_stats()
    factor_risks = stats.get("factor_risks", {})
    factor_mkb_links = stats.get("factor_mkb_links", {})
    factor_specialist_links = stats.get("factor_specialist_links", {})

    return MiningContext(
        factor_risks={
            factor: dict(factor_risks.get(factor, {}))
            for factor in exam.assigned_harmful_factors
        },
        factor_mkb_links={
            factor: list(factor_mkb_links.get(factor, []))
            for factor in exam.assigned_harmful_factors
        },
        factor_specialist_links={
            factor: list(factor_specialist_links.get(factor, []))
            for factor in exam.assigned_harmful_factors
        },
    )


def empty_mining_context() -> MiningContext:
    return MiningContext(
        factor_risks={},
        factor_mkb_links={},
        factor_specialist_links={},
    )


def _is_significant_conclusion(mkb_code: str, health_group: str) -> bool:
    return bool(
        (mkb_code and mkb_code != "Z00.0")
        or health_group in SIGNIFICANT_HEALTH_GROUPS
    )


def _counter_to_records(
    counter: Counter[str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    items = counter.most_common(limit)
    total = sum(counter.values())
    return [
        {
            "value": value,
            "count": count,
            "share": _safe_ratio(count, total),
        }
        for value, count in items
    ]


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
