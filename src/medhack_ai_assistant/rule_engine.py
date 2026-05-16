"""Движок правил противопоказаний."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.medhack_ai_assistant.kb_models import ContraindicationRule, SeveritySpec
from src.medhack_ai_assistant.case_model import ExamCase, SpecialistFinding
from src.medhack_ai_assistant.conclusions_parser import (
    format_factors,
    has_exclusion_context,
    is_normal_finding,
)
from src.medhack_ai_assistant.kb_loader import load_rules_catalog, rules_by_factor


@dataclass
class RuleTrace:
    rule_id: str
    factor_code: str
    disease_name: str
    appendix_ref: str
    match_type: str
    finding: SpecialistFinding
    confidence: float


@dataclass
class ExpertInferenceResult:
    exam_row_id: int
    contraindicated_factors: set[str]
    has_contraindications: bool
    confidence: float
    traces: list[RuleTrace] = field(default_factory=list)
    partial_matches: list[RuleTrace] = field(default_factory=list)

    @property
    def contraindicated_factors_str(self) -> str:
        return format_factors(self.contraindicated_factors)


def _mkb_matches(patterns: list[str], finding: SpecialistFinding) -> bool:
    code = finding.mkb_code
    if not code or not patterns:
        return False
    for pattern in patterns:
        pattern = pattern.strip().upper()
        if not pattern:
            continue
        if len(pattern) <= 3 and pattern.isalpha():
            if finding.mkb_prefix == pattern or code.startswith(pattern):
                return True
        elif code == pattern or code.startswith(pattern):
            return True
    return False


def _text_matches(patterns: list[str], finding: SpecialistFinding) -> bool:
    if not patterns:
        return False
    text = finding.combined_text
    return any(pattern.lower() in text for pattern in patterns if pattern)


def _severity_matches(severity: SeveritySpec | None, finding: SpecialistFinding) -> bool:
    if severity is None:
        return True
    if severity.hearing_stage_min is not None:
        stage = finding.hearing_stage
        if stage is None or stage < severity.hearing_stage_min:
            return False
    if severity.hypertension_stage_min is not None:
        stage = finding.hypertension_stage
        tier_ok = (
            finding.health_group_tier is not None
            and finding.health_group_tier >= severity.hypertension_stage_min
        )
        if stage is None or stage < severity.hypertension_stage_min:
            if not tier_ok:
                return False
    if severity.myopia_degree_min is not None:
        degree = finding.myopia_degree
        tier_ok = (
            finding.health_group_tier is not None
            and finding.health_group_tier >= 3
        )
        if degree is None or degree < severity.myopia_degree_min:
            if not tier_ok:
                return False
    if severity.health_group_min is not None:
        if finding.health_group_tier is None or finding.health_group_tier < severity.health_group_min:
            return False
    return True


def match_rule(
    rule: ContraindicationRule,
    finding: SpecialistFinding,
    *,
    strict: bool = True,
) -> tuple[bool, str, float]:
    if is_normal_finding(finding.mkb_code, finding.conclusion):
        return False, "none", 0.0

    text = finding.combined_text
    if has_exclusion_context(text) and not _mkb_matches(
        [p for p in rule.mkb_patterns if len(p) > 3],
        finding,
    ):
        if strict:
            return False, "excluded", 0.0

    mkb_ok = _mkb_matches(rule.mkb_patterns, finding)
    text_ok = _text_matches(rule.text_patterns, finding)

    if rule.require_mkb_or_text and not mkb_ok and not text_ok:
        return False, "none", 0.0

    if not _severity_matches(rule.severity, finding):
        return False, "severity", 0.0

    if mkb_ok and text_ok:
        return True, "mkb+text", 0.95
    if mkb_ok:
        return True, "mkb", 0.85
    if text_ok:
        if strict:
            return False, "text_only_strict", 0.0
        return True, "text", 0.65

    return False, "none", 0.0


def infer_case(
    case: ExamCase,
    *,
    strict: bool = True,
    rules_index: dict[str, list[ContraindicationRule]] | None = None,
) -> ExpertInferenceResult:
    catalog = load_rules_catalog()
    rules_index = rules_index or rules_by_factor(catalog.rules)

    contraindicated: set[str] = set()
    traces: list[RuleTrace] = []
    partial: list[RuleTrace] = []
    confidences: list[float] = []

    findings = case.pathology_findings or case.findings

    for factor in sorted(case.assigned_factors):
        applicable = rules_index.get(factor, [])
        factor_triggered = False
        for rule in applicable:
            for finding in findings:
                matched, match_type, confidence = match_rule(rule, finding, strict=strict)
                if not matched:
                    loose_matched, loose_type, loose_conf = match_rule(
                        rule, finding, strict=False
                    )
                    if loose_matched and loose_conf >= 0.5:
                        partial.append(
                            RuleTrace(
                                rule_id=rule.rule_id,
                                factor_code=factor,
                                disease_name=rule.disease_name,
                                appendix_ref=rule.appendix_ref,
                                match_type=loose_type,
                                finding=finding,
                                confidence=loose_conf,
                            )
                        )
                    continue
                contraindicated.add(factor)
                confidences.append(confidence)
                traces.append(
                    RuleTrace(
                        rule_id=rule.rule_id,
                        factor_code=factor,
                        disease_name=rule.disease_name,
                        appendix_ref=rule.appendix_ref,
                        match_type=match_type,
                        finding=finding,
                        confidence=confidence,
                    )
                )
                factor_triggered = True
                break
            if factor_triggered:
                break

    contraindicated &= case.assigned_factors
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return ExpertInferenceResult(
        exam_row_id=case.exam_row_id,
        contraindicated_factors=contraindicated,
        has_contraindications=len(contraindicated) > 0,
        confidence=overall_confidence,
        traces=traces,
        partial_matches=partial,
    )
