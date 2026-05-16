from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TEST_PATH,
    TEXT_COLUMNS,
    TRAIN_PATH,
    SUBMISSION_PATH,
    ModelConfig,
)
from .data import load_data, save_submission, validate_columns
from .domain.models import (
    AnalysisResult,
    MiningContext,
    PatientExam,
    QualityGateResult,
    RAGContext,
)
from .ml import (
    AsyncVLLMClient,
    LLMMessage,
    ValidationResult,
    VLLMClientError,
    VLLMResponseFormatError,
    predict_with_threshold,
    train_and_validate,
    train_final_model,
)
from .parsing.order_29n import build_order_context_for_prompt
from .preprocessing import build_text_feature
from .services.dashboard import build_patient_exam
from .services.factor_mining import empty_mining_context, get_mining_context


@dataclass(frozen=True)
class PipelineResult:
    target_distribution: pd.Series
    target_distribution_normalized: pd.Series
    validation: ValidationResult
    submission: pd.DataFrame
    submission_path: Path


SYSTEM_PROMPT = """Ты — медицинский ассистент врача-профпатолога.
Твоя задача: сопоставить вредные факторы работника, заключения врачей,
контекст Приказа Минздрава 29н и историческую статистику решений.
Верни только валидный JSON без markdown.
Не добавляй противопоказанные факторы, которых нет во входном списке факторов пациента.
Если данных недостаточно или качество пакета плохое, используй verdict NEEDS_MORE_INFO."""


EMPTY_RAG_CONTEXT = RAGContext(chunks=(), text="", total_chars=0)


def run_training_pipeline(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    submission_path: Path = SUBMISSION_PATH,
    config: ModelConfig | None = None,
) -> PipelineResult:
    config = config or ModelConfig()

    train = load_data(train_path)
    test = load_data(test_path)

    validate_columns(train, (*TEXT_COLUMNS, TARGET_COLUMN))
    validate_columns(test, (*TEXT_COLUMNS, ID_COLUMN))

    x = build_text_feature(train, TEXT_COLUMNS)
    y = train[TARGET_COLUMN].astype(bool)
    x_test = build_text_feature(test, TEXT_COLUMNS)

    validation = train_and_validate(x, y, config)

    model = train_final_model(x, y, config)
    test_pred = predict_with_threshold(model, x_test, validation.threshold)

    submission = save_submission(
        exam_row_id=test[ID_COLUMN],
        predictions=test_pred,
        output_path=submission_path,
    )

    return PipelineResult(
        target_distribution=y.value_counts(),
        target_distribution_normalized=y.value_counts(normalize=True),
        validation=validation,
        submission=submission,
        submission_path=submission_path,
    )


def run_quality_gate(row_or_exam: pd.Series | dict[str, Any] | PatientExam) -> QualityGateResult:
    reasons: list[str] = []

    try:
        exam = _ensure_exam(row_or_exam)
    except Exception as exc:
        return QualityGateResult(
            status="NEEDS_MORE_INFO",
            reasons=(f"Cannot parse patient package: {exc}",),
            can_analyze=False,
        )

    if not exam.assigned_harmful_factors:
        reasons.append("Missing assigned harmful factors.")
    if not exam.specialist_conclusions:
        reasons.append("Missing specialist conclusions.")

    if reasons:
        return QualityGateResult(
            status="NEEDS_MORE_INFO",
            reasons=tuple(reasons),
            can_analyze=False,
        )

    return QualityGateResult(status="READY", reasons=(), can_analyze=True)


def build_patient_prompt(
    exam: PatientExam,
    rag_context: RAGContext,
    mining_context: MiningContext,
) -> list[LLMMessage]:
    payload = {
        "patient": {
            "exam_row_id": exam.exam_row_id,
            "patient_id": exam.patient_id,
            "consultation_date": exam.consultation_date,
            "assigned_harmful_factors": exam.assigned_harmful_factors,
        },
        "specialist_conclusions": [
            {
                "specialist": conclusion.specialist,
                "consultation_date": conclusion.consultation_date,
                "conclusion": conclusion.conclusion,
                "health_group": conclusion.health_group,
                "mkb_code": conclusion.mkb_code,
                "mkb_description": conclusion.mkb_description,
            }
            for conclusion in exam.specialist_conclusions
        ],
        "mining_context": {
            "factor_risks": mining_context.factor_risks,
            "factor_mkb_links": mining_context.factor_mkb_links,
            "factor_specialist_links": mining_context.factor_specialist_links,
        },
        "order_29n_rag_context": {
            "total_chars": rag_context.total_chars,
            "chunks": rag_context.chunks,
            "text": rag_context.text,
        },
        "required_json_schema": {
            "verdict": "FIT | UNFIT | NEEDS_MORE_INFO",
            "summary": "short doctor-facing summary",
            "contraindication_factors": ["factor codes from assigned_harmful_factors only"],
            "evidence": [
                {
                    "factor": "factor code",
                    "mkb_code": "ICD-10 code",
                    "source_specialist": "doctor or exam name",
                    "reason": "why this is relevant",
                }
            ],
            "follow_up_draft": {
                "specialists": ["recommended specialists"],
                "tests": ["recommended tests"],
            },
            "confidence": 0.0,
        },
    }

    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=json.dumps(payload, ensure_ascii=False),
        ),
    ]


async def analyze_patient_exam(
    row_or_exam: pd.Series | dict[str, Any] | PatientExam,
    *,
    vllm_client: AsyncVLLMClient | None = None,
) -> AnalysisResult:
    quality_gate = run_quality_gate(row_or_exam)

    try:
        exam = _ensure_exam(row_or_exam)
    except Exception:
        return _analysis_result(
            status="NEEDS_MORE_INFO",
            verdict="NEEDS_MORE_INFO",
            summary="Patient package cannot be parsed and needs a data quality follow-up.",
            quality_gate=quality_gate,
        )

    if not quality_gate.can_analyze:
        return _analysis_result(
            status="NEEDS_MORE_INFO",
            verdict="NEEDS_MORE_INFO",
            summary="Patient package did not pass Quality Gate.",
            quality_gate=quality_gate,
            exam=exam,
        )

    mining_context = get_mining_context(exam)
    rag_context = build_order_context_for_prompt(exam.assigned_harmful_factors)
    messages = build_patient_prompt(exam, rag_context, mining_context)

    owns_client = vllm_client is None
    client = vllm_client or AsyncVLLMClient()

    try:
        raw_response = await client.chat_json(messages)
    except (VLLMClientError, VLLMResponseFormatError) as exc:
        return _analysis_result(
            status="MODEL_UNAVAILABLE",
            verdict="NEEDS_MORE_INFO",
            summary=f"Model inference is unavailable or invalid: {exc}",
            quality_gate=quality_gate,
            rag_context=rag_context,
            mining_context=mining_context,
            raw_llm_response={},
            exam=exam,
        )
    finally:
        if owns_client:
            await client.aclose()

    return _build_analysis_from_llm(
        exam=exam,
        quality_gate=quality_gate,
        rag_context=rag_context,
        mining_context=mining_context,
        raw_response=raw_response,
    )


def _ensure_exam(row_or_exam: pd.Series | dict[str, Any] | PatientExam) -> PatientExam:
    if isinstance(row_or_exam, PatientExam):
        return row_or_exam
    return build_patient_exam(row_or_exam)


def _build_analysis_from_llm(
    *,
    exam: PatientExam,
    quality_gate: QualityGateResult,
    rag_context: RAGContext,
    mining_context: MiningContext,
    raw_response: dict[str, Any],
) -> AnalysisResult:
    assigned_factors = set(exam.assigned_harmful_factors)
    requested_verdict = str(raw_response.get("verdict", "NEEDS_MORE_INFO")).strip().upper()
    if requested_verdict not in {"FIT", "UNFIT", "NEEDS_MORE_INFO"}:
        requested_verdict = "NEEDS_MORE_INFO"

    factors = tuple(
        factor
        for factor in _as_string_tuple(raw_response.get("contraindication_factors"))
        if factor in assigned_factors
    )
    verdict = requested_verdict
    status = "OK"

    if requested_verdict == "UNFIT" and not factors:
        verdict = "NEEDS_MORE_INFO"
        status = "REVIEW_REQUIRED"

    confidence = _clamp_float(raw_response.get("confidence"), default=0.0)
    raw_response = {**raw_response, "confidence": confidence, "verdict": verdict}

    return _analysis_result(
        status=status,
        verdict=verdict,
        summary=str(raw_response.get("summary", "")).strip(),
        factors=factors,
        evidence=_as_dict_tuple(raw_response.get("evidence")),
        follow_up_draft=_as_dict(raw_response.get("follow_up_draft")),
        quality_gate=quality_gate,
        rag_context=rag_context,
        mining_context=mining_context,
        raw_llm_response=raw_response,
        exam=exam,
    )


def _analysis_result(
    *,
    status: str,
    verdict: str,
    summary: str,
    quality_gate: QualityGateResult,
    factors: tuple[str, ...] = (),
    evidence: tuple[dict[str, Any], ...] = (),
    follow_up_draft: dict[str, Any] | None = None,
    rag_context: RAGContext = EMPTY_RAG_CONTEXT,
    mining_context: MiningContext | None = None,
    raw_llm_response: dict[str, Any] | None = None,
    exam: PatientExam | None = None,
) -> AnalysisResult:
    return AnalysisResult(
        status=status,
        verdict=verdict,
        summary=summary,
        factors=factors,
        evidence=evidence,
        follow_up_draft=follow_up_draft or {},
        quality_gate=quality_gate,
        rag_context=rag_context,
        mining_context=mining_context or empty_mining_context(),
        raw_llm_response=raw_llm_response or {},
        exam=exam,
    )


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _as_dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, dict))


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
