from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SpecialistConclusion:
    specialist: str
    consultation_date: str
    conclusion: str
    health_group: str
    mkb_code: str
    mkb_description: str

    @property
    def has_diagnosis(self) -> bool:
        return bool(self.mkb_code and self.mkb_code != "Z00.0")

    @property
    def has_attention_marker(self) -> bool:
        normal_health_group = self.health_group.strip().lower()
        return self.has_diagnosis or bool(self.conclusion) or normal_health_group not in {"", "1 группа"}


@dataclass(frozen=True)
class PatientExam:
    exam_row_id: int
    patient_id: int
    consultation_date: str
    assigned_harmful_factors: tuple[str, ...]
    specialist_conclusions: tuple[SpecialistConclusion, ...]
    has_contraindications: bool | None = None
    contraindicated_factors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DashboardFinding:
    title: str
    status: str
    details: str
    source_specialist: str
    source_date: str
    mkb_code: str = ""


@dataclass(frozen=True)
class DashboardResult:
    exam: PatientExam
    diagnoses: tuple[DashboardFinding, ...]
    attention_items: tuple[DashboardFinding, ...]
    normal_items: tuple[DashboardFinding, ...]
    decision_label: str
    decision_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class QualityGateResult:
    status: str
    reasons: tuple[str, ...]
    can_analyze: bool


@dataclass(frozen=True)
class MiningContext:
    factor_risks: dict[str, dict[str, Any]]
    factor_mkb_links: dict[str, list[dict[str, Any]]]
    factor_specialist_links: dict[str, list[dict[str, Any]]]


@dataclass(frozen=True)
class RAGContext:
    chunks: tuple[dict[str, Any], ...]
    text: str
    total_chars: int


@dataclass(frozen=True)
class AnalysisResult:
    status: str
    verdict: str
    summary: str
    factors: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    follow_up_draft: dict[str, Any]
    quality_gate: QualityGateResult
    rag_context: RAGContext
    mining_context: MiningContext
    raw_llm_response: dict[str, Any]
    exam: PatientExam | None = None

    def to_dict(self) -> dict:
        return asdict(self)

