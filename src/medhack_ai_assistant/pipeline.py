from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import joblib

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


# ==========================================
# ЗАГРУЗКА ЛОКАЛЬНЫХ МОДЕЛЕЙ
# ==========================================

_MODELS_CACHE = {
    "models": None,
    "thresholds": None,
    "factors": None,
    "tfidf": None,
}


def _get_models_dir() -> Path:
    """Получить папку с моделями."""
    return Path(__file__).parent.parent / "models"


def _load_local_models() -> dict[str, Any]:
    """Загрузить обученные XGBoost модели (с кэшированием)."""
    if _MODELS_CACHE["models"] is not None:
        return _MODELS_CACHE
    
    models_dir = _get_models_dir()
    
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Папка с моделями не найдена: {models_dir}\n"
            "Пожалуйста, сначала обучите модель: python src/medhack_ai_assistant/services/local_ai.py"
        )
    
    try:
        _MODELS_CACHE["models"] = joblib.load(models_dir / "xgboost_models.pkl")
        _MODELS_CACHE["thresholds"] = joblib.load(models_dir / "best_thresholds.pkl")
        _MODELS_CACHE["factors"] = joblib.load(models_dir / "all_factors.pkl")
        _MODELS_CACHE["tfidf"] = joblib.load(models_dir / "model_config.pkl")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Не удалось загрузить модели из {models_dir}: {e}\n"
            "Убедитесь, что вы запустили: python src/medhack_ai_assistant/services/local_ai.py"
        )
    
    return _MODELS_CACHE


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


def analyze_patient_exam(
    row_or_exam: pd.Series | dict[str, Any] | PatientExam,
) -> AnalysisResult:
    """Анализ осмотра с использованием локальной XGBoost модели."""
    quality_gate = run_quality_gate(row_or_exam)

    try:
        exam = _ensure_exam(row_or_exam)
    except Exception as e:
        return _analysis_result(
            status="PARSE_ERROR",
            verdict="NEEDS_MORE_INFO",
            summary=f"Не удалось распарсить данные осмотра: {e}",
            quality_gate=quality_gate,
        )

    if not quality_gate.can_analyze:
        return _analysis_result(
            status="INSUFFICIENT_DATA",
            verdict="NEEDS_MORE_INFO",
            summary="Данные осмотра не прошли проверку качества. Требуются дополнительные сведения.",
            quality_gate=quality_gate,
            exam=exam,
        )

    try:
        models_cache = _load_local_models()
        final_models = models_cache["models"]
        best_thresholds = models_cache["thresholds"]
        all_factors = models_cache["factors"]
    except FileNotFoundError as e:
        return _analysis_result(
            status="MODEL_ERROR",
            verdict="NEEDS_MORE_INFO",
            summary=f"Не удалось загрузить модели: {e}",
            quality_gate=quality_gate,
            exam=exam,
        )

    # Подготовка текста пациента для предсказания
    try:
        text_features = _prepare_exam_text(exam)
        
        # Предсказания для каждого фактора
        predicted_factors = {}
        confidences = {}
        
        for factor in all_factors:
            if factor not in final_models:
                continue
            
            model = final_models[factor]
            threshold = best_thresholds.get(factor, 0.5)
            
            # Используем pipeline для TF-IDF + XGBoost
            proba = model.predict_proba([text_features])[0, 1]
            is_unfit = proba >= threshold
            
            predicted_factors[factor] = is_unfit
            confidences[factor] = float(proba)
        
        # Фильтруем по assigned_harmful_factors (бизнес-правило)
        assigned_set = set(exam.assigned_harmful_factors)
        final_factors = tuple(
            f for f, is_unfit in predicted_factors.items()
            if is_unfit and f in assigned_set
        )
        
        # Определяем вердикт
        verdict = "UNFIT" if final_factors else "FIT"
        status = "OK"
        
        # Подготовка объяснения
        summary = _build_explanation(
            verdict, final_factors, exam, confidences, predicted_factors
        )
        
        # Топ-слова (доказательства)
        evidence = _extract_top_words_evidence(
            model=final_models.get(final_factors[0]) if final_factors else None,
            text=text_features,
            top_n=3
        )
        
        return _analysis_result(
            status=status,
            verdict=verdict,
            summary=summary,
            factors=final_factors,
            evidence=evidence,
            quality_gate=quality_gate,
            raw_llm_response={
                "predicted_factors": predicted_factors,
                "confidences": confidences,
            },
            exam=exam,
        )
    
    except Exception as e:
        return _analysis_result(
            status="PREDICTION_ERROR",
            verdict="NEEDS_MORE_INFO",
            summary=f"Ошибка при анализе: {e}",
            quality_gate=quality_gate,
            exam=exam,
        )


def _ensure_exam(row_or_exam: pd.Series | dict[str, Any] | PatientExam) -> PatientExam:
    if isinstance(row_or_exam, PatientExam):
        return row_or_exam
    return build_patient_exam(row_or_exam)


def _prepare_exam_text(exam: PatientExam) -> str:
    """Подготовка текста осмотра для TF-IDF векторизации."""
    parts = []
    
    if exam.specialist_conclusions:
        parts.append(" ".join(exam.specialist_conclusions))
    
    if exam.assigned_harmful_factors:
        parts.append(" ".join(exam.assigned_harmful_factors))
    
    if exam.diagnoses:
        parts.append(" ".join(exam.diagnoses))
    
    if exam.icd_codes:
        parts.append(" ".join(exam.icd_codes))
    
    return " ".join(parts)


def _build_explanation(
    verdict: str,
    factors: tuple[str, ...],
    exam: PatientExam,
    confidences: dict[str, float],
    all_predictions: dict[str, bool],
) -> str:
    """Формирование текстового объяснения."""
    if verdict == "FIT":
        return (
            f"Осмотр пациента #{exam.exam_row_id}: явных противопоказаний не выявлено. "
            f"Пациент годен к работе с назначенными вредными факторами."
        )
    
    # UNFIT
    summary_parts = [f"Осмотр пациента #{exam.exam_row_id}: выявлены противопоказания."]
    
    for factor in factors:
        confidence = confidences.get(factor, 0)
        summary_parts.append(
            f"• {factor} (доверие: {confidence:.1%})"
        )
    
    summary_parts.append(
        f"\nДиагнозы: {'; '.join(exam.diagnoses) if exam.diagnoses else 'не указаны'}. "
        f"Требуется повторная оценка врачом-профпатологом."
    )
    
    return "\n".join(summary_parts)


def _extract_top_words_evidence(
    model,
    text: str,
    top_n: int = 3,
) -> tuple[dict[str, Any], ...]:
    """Извлечение топ-слов которые повлияли на решение."""
    if model is None:
        return ()
    
    try:
        # Получаем TF-IDF векторизатор из pipeline
        vectorizer = model.named_steps.get("tfidf")
        if vectorizer is None:
            return ()
        
        # Преобразуем текст
        vector = vectorizer.transform([text])
        
        # Получаем названия признаков (слова)
        feature_names = vectorizer.get_feature_names_out()
        
        # Получаем веса (TF-IDF scores)
        scores = vector.toarray()[0]
        
        # Находим топ-слова
        top_indices = np.argsort(scores)[-top_n:][::-1]
        
        evidence = []
        for idx in top_indices:
            if scores[idx] > 0:
                evidence.append({
                    "reasoning": f"Найден признак: {feature_names[idx]}",
                    "score": float(scores[idx]),
                })
        
        return tuple(evidence)
    
    except Exception:
        return ()



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
