from dataclasses import asdict, dataclass


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
        return self.has_diagnosis or bool(self.conclusion) or self.health_group not in {"", "1 группа"}


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

