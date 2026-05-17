import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from medhack_ai_assistant.config import PROJECT_ROOT

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **_: Any):
        return iterable


# =========================================================================
# ОБЯЗАТЕЛЬНО: Объявляем класс, чтобы joblib смог распаковать модель
# =========================================================================
class RobustMultiOutputClassifier:
    def __init__(self, base_estimator, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.single_class_models = {}

    def fit(self, X, Y):
        pass

    def predict_proba(self, X):
        num_samples = X.shape[0]
        num_outputs = len(self.estimators_)
        probas = np.zeros((num_samples, num_outputs))
        
        for i in range(num_outputs):
            clf = self.estimators_[i]
            if clf is not None:
                probas[:, i] = clf.predict_proba(X)[:, 1]
            else:
                probas[:, i] = float(self.single_class_models[i])
        return probas


def register_joblib_compat_classes() -> None:
    import sys
    import types

    main_module = sys.modules.get("main")
    if main_module is None:
        main_module = types.ModuleType("main")
        sys.modules["main"] = main_module

    setattr(main_module, "RobustMultiOutputClassifier", RobustMultiOutputClassifier)
    setattr(sys.modules["__main__"], "RobustMultiOutputClassifier", RobustMultiOutputClassifier)


def _find_project_root() -> Path:
    configured = os.getenv("SOLUTION_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    if (PROJECT_ROOT / "data").exists():
        return PROJECT_ROOT

    current = Path(__file__).resolve().parent
    if (current / "data").exists():
        return current
    return Path.cwd()


BASE_DIR = _find_project_root()
MODEL_PATH = BASE_DIR / "data" / "models_weights" / "final_solution.joblib"

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
LEGACY_TRAIN_PATH = BASE_DIR / "train.csv"
LEGACY_TEST_PATH = BASE_DIR / "test.csv"

VALIDATION_MODE = os.getenv("SOLUTION_VALIDATION_MODE", "0") == "1"

if VALIDATION_MODE:
    DATA_PATH = TRAIN_PATH if TRAIN_PATH.exists() else LEGACY_TRAIN_PATH
    LIMIT_ROWS = int(os.getenv("SOLUTION_LIMIT_ROWS", "300"))
    OUTPUT_CSV_PATH = BASE_DIR / "val_submission.csv"
else:
    DATA_PATH = TEST_PATH if TEST_PATH.exists() else LEGACY_TEST_PATH
    LIMIT_ROWS = None
    OUTPUT_CSV_PATH = BASE_DIR / "submission.csv"


# =========================================================================
# КЭШ ДЛЯ КАРТИРОВАНИЯ ML-МОДЕЛИ
# =========================================================================
_MODEL_ARTIFACTS_CACHE: dict[str, Any] | None = None

def get_model_artifacts() -> dict[str, Any]:
    global _MODEL_ARTIFACTS_CACHE
    if _MODEL_ARTIFACTS_CACHE is None:
        if not MODEL_PATH.exists():
            print(f"Error: Model file {MODEL_PATH} not found.")
            raise FileNotFoundError(f"Missing weights file at {MODEL_PATH}")
        register_joblib_compat_classes()
        _MODEL_ARTIFACTS_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_ARTIFACTS_CACHE


def normalize_factor_code(value) -> str:
    if pd.isna(value) or value is None:
        return ""
    code = str(value).strip().replace(",", ".")
    code = re.sub(r"\s+", "", code)
    code = re.sub(r"\.+", ".", code).strip(".")
    if not re.fullmatch(r"\d{1,2}(?:\.\d{1,3}){0,3}", code):
        return ""
    return code


def calculate_jaccard(preds: list, targets: list) -> float:
    set_p = set(preds) if (preds and preds != ["0"]) else {"0"}
    set_t = set(targets) if (targets and targets != ["0"]) else {"0"}
    if set_p == {"0"} and set_t == {"0"}:
        return 1.0
    if set_p == {"0"} or set_t == {"0"}:
        return 0.0

    intersection_count = 0
    matched_targets = set()
    for p in set_p:
        for t in set_t:
            if t not in matched_targets:
                if p == t or p.startswith(t + ".") or t.startswith(p + "."):
                    intersection_count += 1
                    matched_targets.add(t)
                    break
    union_count = len(set_p) + len(set_t) - intersection_count
    return intersection_count / union_count if union_count > 0 else 0.0


def extract_patient_features(row: pd.Series | dict[str, Any]) -> tuple[list[str], set[str], str]:
    raw_factors_str = str(row.get("assigned_harmful_factors", ""))
    assigned_factors = [
        normalize_factor_code(factor)
        for factor in raw_factors_str.split(";")
        if normalize_factor_code(factor)
    ]

    patient_mkb_codes = set()
    full_conclusion_text = ""
    raw_conclusions = str(row.get("specialist_conclusions", ""))

    if raw_conclusions and raw_conclusions != "nan":
        full_conclusion_text = raw_conclusions.lower()
        found_mkb = re.findall(r"\b[A-Z]\d{2}(?:\.\d)?\b", raw_conclusions.upper())
        for match in found_mkb:
            patient_mkb_codes.add(match)

        if raw_conclusions.strip().startswith("["):
            try:
                conclusions_json = json.loads(raw_conclusions)
                for obj in conclusions_json:
                    mkb = str(obj.get("mkb_code", "")).upper().strip()
                    if mkb:
                        patient_mkb_codes.add(mkb)
                    full_conclusion_text += " " + str(obj.get("conclusion", "")).lower()
                    full_conclusion_text += " " + str(obj.get("mkb_description", "")).lower()
            except Exception:
                pass

    return assigned_factors, patient_mkb_codes, full_conclusion_text


def extract_mkb_details(row: pd.Series | dict[str, Any]) -> dict[str, str]:
    details: dict[str, str] = {}
    raw_conclusions = str(row.get("specialist_conclusions", ""))

    if not raw_conclusions or raw_conclusions == "nan":
        return details

    if not raw_conclusions.strip().startswith("["):
        for match in re.findall(r"\b[A-Z]\d{2}(?:\.\d)?\b", raw_conclusions.upper()):
            details.setdefault(match, "")
        return details

    try:
        conclusions_json = json.loads(raw_conclusions)
    except Exception:
        return details

    for obj in conclusions_json:
        mkb = str(obj.get("mkb_code", "")).upper().strip()
        if not mkb:
            continue
        description = str(obj.get("mkb_description", "")).strip()
        conclusion = str(obj.get("conclusion", "")).strip()
        details[mkb] = description or conclusion

    return details


def build_human_summary(
    *,
    predicted_factors: list[str],
    mkb_details: dict[str, str],
) -> str:
    if not predicted_factors:
        return "По данным заключений специалистов и правилам Приказа 29н явные противопоказания не выявлены."

    factors_text = format_factors(predicted_factors)
    if not mkb_details:
        return (
            f"Выявлены возможные противопоказания по факторам: {factors_text}. "
            "Основание: математический анализ текста медицинских заключений моделью ML."
        )

    diagnoses = []
    for code, description in sorted(mkb_details.items()):
        if description:
            diagnoses.append(f"{code} — {description}")
        else:
            diagnoses.append(code)

    diagnoses_text = "; ".join(diagnoses[:6])
    if len(diagnoses) > 6:
        diagnoses_text += f"; ещё {len(diagnoses) - 6}"

    return (
        f"Выявлены возможные противопоказания по факторам: {factors_text}. "
        f"В заключениях присутствуют связанные диагнозы/коды МКБ: {diagnoses_text}."
    )


# =========================================================================
# ОБНОВЛЕННАЯ ФУНКЦИЯ ПРЕДСКАЗАНИЯ НА ОСНОВЕ LOGISTIC REGRESSION
# =========================================================================
def predict_row_factors(
    row: pd.Series | dict[str, Any],
    *args, **kwargs
) -> list[str]:
    # Получаем артефакты обученного пайплайна
    artifacts = get_model_artifacts()
    tfidf = artifacts["tfidf"]
    robust_clf = artifacts["clf"]
    best_threshold = artifacts["best_threshold"]
    factors_columns = artifacts["factors"]

    # 1. Извлекаем базовые свойства (для фильтрации по назначенным факторам)
    assigned_factors, _, _ = extract_patient_features(row)
    if not assigned_factors:
        return []

    # 2. Формируем текстовый признак ровно так же, как при обучении
    specialist_text = str(row.get('specialist_conclusions', "")).replace("nan", "")
    assigned_text = str(row.get('assigned_harmful_factors', "")).replace("nan", "")
    combined_text = f"{specialist_text} {assigned_text}"

    # 3. Инференс ML модели
    X_transformed = tfidf.transform([combined_text])
    probas = robust_clf.predict_proba(X_transformed)[0]  # Берем вероятности для одной строки

    # Применяем порог
    predicted_indices = np.where(probas >= best_threshold)[0]
    
    validated_contraindications = []
    for idx in predicted_indices:
        pred_factor = factors_columns[idx]
        
        # Фильтрация: проверяем, пересекается ли предсказанный фактор с тем, что реально назначен сотруднику
        # Это предотвращает ложные срабатывания по факторам, которых у человека в принципе нет на работе.
        for assigned_factor in assigned_factors:
            if (
                assigned_factor == pred_factor
                or assigned_factor.startswith(pred_factor + ".")
                or pred_factor.startswith(assigned_factor + ".")
            ):
                validated_contraindications.append(assigned_factor)

    return sorted(set(validated_contraindications))


def format_factors(factors: list[str]) -> str:
    return ";".join(factors).strip() if factors else "0"


def analyze_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    exam_id = int(row.get("exam_row_id", 0))
    assigned_factors, patient_mkb_codes, _ = extract_patient_features(row)
    predicted_factors = predict_row_factors(row)
    mkb_details = extract_mkb_details(row)

    verdict = "UNFIT" if predicted_factors else "FIT"
    summary = build_human_summary(
        predicted_factors=predicted_factors,
        mkb_details=mkb_details,
    )

    evidence = [
        {
            "factor": factor,
            "mkb_code": ";".join(sorted(patient_mkb_codes)) or "",
            "source_specialist": "specialist_conclusions",
            "reason": build_human_summary(
                predicted_factors=[factor],
                mkb_details=mkb_details,
            ),
        }
        for factor in predicted_factors
    ]

    return {
        "exam_row_id": exam_id,
        "verdict": verdict,
        "summary": summary,
        "factors": predicted_factors,
        "factors_submission": format_factors(predicted_factors),
        "assigned_factors": assigned_factors,
        "mkb_codes": sorted(patient_mkb_codes),
        "evidence": evidence,
        "rules_count": len(get_model_artifacts()["factors"]),
    }


def analyze_row_as_result(row: pd.Series | dict[str, Any], exam=None):
    from medhack_ai_assistant.domain.models import (
        AnalysisResult,
        MiningContext,
        QualityGateResult,
        RAGContext,
    )

    data = analyze_row(row)
    quality = QualityGateResult(status="READY", reasons=(), can_analyze=True)
    return AnalysisResult(
        status="OK",
        verdict=data["verdict"],
        summary=data["summary"],
        factors=tuple(data["factors"]),
        evidence=tuple(data["evidence"]),
        follow_up_draft={"specialists": [], "tests": []},
        quality_gate=quality,
        rag_context=RAGContext(chunks=(), text="", total_chars=0),
        mining_context=MiningContext(
            factor_risks={},
            factor_mkb_links={},
            factor_specialist_links={},
        ),
        raw_llm_response={
            "confidence": 1.0 if data["factors"] else 0.75,
            "rules_count": data["rules_count"],
            "factors_submission": data["factors_submission"],
            "mkb_codes": data["mkb_codes"],
        },
        exam=exam,
    )


def run_solution(
    data_path: Path = DATA_PATH,
    output_csv_path: Path = OUTPUT_CSV_PATH,
    *,
    validation_mode: bool = VALIDATION_MODE,
    limit_rows: int | None = LIMIT_ROWS,
) -> None:
    print(f"Mode: {'validation' if validation_mode else 'test inference'}")

    artifacts = get_model_artifacts()
    print(f"ML Model loaded successfully. Supported target factors: {len(artifacts['factors'])}")

    if not data_path.exists():
        print(f"Error: file {data_path.name} was not found.")
        raise SystemExit(1)

    df_data = pd.read_csv(data_path)
    if limit_rows:
        df_data = df_data.head(limit_rows)
    print(f"Loaded rows: {len(df_data)}")

    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write("exam_row_id,factors\n")

    jaccard_scores = []
    stats_zeros = 0
    stats_contra = 0

    print("Starting ML Logistic Regression inference analysis...")

    for _, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Analysis"):
        exam_id = int(row["exam_row_id"])

        true_factors = []
        if validation_mode:
            raw_target = str(row.get("factors", "0"))
            if raw_target and raw_target != "0" and raw_target != "nan":
                true_factors = [
                    normalize_factor_code(factor)
                    for factor in raw_target.split(";")
                    if normalize_factor_code(factor)
                ]

        predicted_factors = predict_row_factors(row)
        final_str = format_factors(predicted_factors)

        if predicted_factors:
            stats_contra += 1
        else:
            stats_zeros += 1

        with open(output_csv_path, "a", encoding="utf-8") as f:
            f.write(f"{exam_id},{final_str}\n")

        if validation_mode:
            preds = final_str.split(";") if final_str != "0" else []
            jaccard_scores.append(calculate_jaccard(preds, true_factors))

    print(f"\nProcessed rows: {len(df_data)} | fit: {stats_zeros} | contraindications: {stats_contra}")
    print(f"Done. File saved: {output_csv_path}")

    if validation_mode:
        print(f"Mean Jaccard Score: {np.mean(jaccard_scores):.5f}")


if __name__ == "__main__":
    run_solution()
