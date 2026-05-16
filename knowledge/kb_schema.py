"""Pydantic-схемы базы знаний приказа 29н."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FactorEntry(BaseModel):
    code: str
    title: str
    parent_code: str | None = None
    required_specialists: list[str] = Field(default_factory=list)


class SeveritySpec(BaseModel):
    hearing_stage_min: int | None = None
    hypertension_stage_min: int | None = None
    health_group_min: int | None = None
    myopia_degree_min: int | None = None


class ContraindicationRule(BaseModel):
    rule_id: str
    disease_name: str
    mkb_patterns: list[str] = Field(default_factory=list)
    text_patterns: list[str] = Field(default_factory=list)
    severity: SeveritySpec | None = None
    factor_codes: list[str] = Field(default_factory=list)
    appendix_ref: str = ""
    notes: str = ""
    validated: bool = False
    require_mkb_or_text: bool = True


class FactorsCatalog(BaseModel):
    version: str = "1.0"
    factors: list[FactorEntry]


class RulesCatalog(BaseModel):
    version: str = "1.0"
    rules: list[ContraindicationRule]


class SynonymsCatalog(BaseModel):
    version: str = "1.0"
    text_to_mkb_prefix: dict[str, str] = Field(default_factory=dict)
    abbreviations: dict[str, list[str]] = Field(default_factory=dict)
