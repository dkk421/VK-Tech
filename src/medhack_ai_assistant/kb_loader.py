"""Загрузка базы знаний приказа 29н."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.medhack_ai_assistant.kb_models import (
    ContraindicationRule,
    FactorsCatalog,
    RulesCatalog,
    SynonymsCatalog,
)
from src.medhack_ai_assistant.config import (
    FACTORS_PATH,
    RULES_PATH,
    SYNONYMS_PATH,
)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_factors_catalog(path: Path = FACTORS_PATH) -> FactorsCatalog:
    return FactorsCatalog.model_validate(_load_json(path))


@lru_cache(maxsize=1)
def load_rules_catalog(path: Path = RULES_PATH) -> RulesCatalog:
    return RulesCatalog.model_validate(_load_json(path))


@lru_cache(maxsize=1)
def load_synonyms(path: Path = SYNONYMS_PATH) -> SynonymsCatalog:
    return SynonymsCatalog.model_validate(_load_json(path))


def factor_title_map(catalog: FactorsCatalog | None = None) -> dict[str, str]:
    catalog = catalog or load_factors_catalog()
    return {entry.code: entry.title for entry in catalog.factors}


def rules_by_factor(rules: list[ContraindicationRule]) -> dict[str, list[ContraindicationRule]]:
    index: dict[str, list[ContraindicationRule]] = {}
    for rule in rules:
        for code in rule.factor_codes:
            index.setdefault(code, []).append(rule)
    return index
