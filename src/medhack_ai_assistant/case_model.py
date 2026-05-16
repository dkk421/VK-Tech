"""Модель кейса профосмотра для экспертной системы."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.medhack_ai_assistant.conclusions_parser import (
    extract_hearing_stage,
    extract_hypertension_stage,
    extract_myopia_degree,
    is_normal_finding,
    mkb_prefix,
    parse_factors,
    parse_health_group_tier,
    safe_parse_conclusions,
)


@dataclass
class SpecialistFinding:
    specialist: str
    consultation_date: str
    mkb_code: str
    mkb_prefix: str
    mkb_description: str
    conclusion: str
    health_group: str
    health_group_tier: int | None
    hearing_stage: int | None = None
    hypertension_stage: int | None = None
    myopia_degree: int | None = None

    @property
    def combined_text(self) -> str:
        return " ".join(
            part for part in (self.conclusion, self.mkb_description, self.mkb_code) if part
        ).lower()


@dataclass
class ExamCase:
    exam_row_id: int
    patient_id: int | None
    assigned_factors: set[str]
    findings: list[SpecialistFinding] = field(default_factory=list)

    @property
    def pathology_findings(self) -> list[SpecialistFinding]:
        return [
            finding
            for finding in self.findings
            if not is_normal_finding(finding.mkb_code, finding.conclusion)
        ]

    @property
    def all_mkb_codes(self) -> set[str]:
        return {finding.mkb_code for finding in self.findings if finding.mkb_code}


def finding_from_dict(item: dict[str, Any]) -> SpecialistFinding:
    conclusion = str(item.get("conclusion") or "")
    mkb_code = str(item.get("mkb_code") or "").strip().upper()
    return SpecialistFinding(
        specialist=str(item.get("specialist") or ""),
        consultation_date=str(item.get("consultation_date") or ""),
        mkb_code=mkb_code,
        mkb_prefix=mkb_prefix(mkb_code),
        mkb_description=str(item.get("mkb_description") or ""),
        conclusion=conclusion,
        health_group=str(item.get("health_group") or ""),
        health_group_tier=parse_health_group_tier(str(item.get("health_group") or "")),
        hearing_stage=extract_hearing_stage(conclusion),
        hypertension_stage=extract_hypertension_stage(conclusion),
        myopia_degree=extract_myopia_degree(conclusion),
    )


def build_exam_case(row: pd.Series) -> ExamCase:
    exam_row_id = int(row["exam_row_id"])
    patient_id = int(row["patient_id"]) if "patient_id" in row and pd.notna(row["patient_id"]) else None
    assigned = parse_factors(row.get("assigned_harmful_factors"))
    conclusions = safe_parse_conclusions(row.get("specialist_conclusions"))
    findings = [finding_from_dict(item) for item in conclusions]
    return ExamCase(
        exam_row_id=exam_row_id,
        patient_id=patient_id,
        assigned_factors=assigned,
        findings=findings,
    )
