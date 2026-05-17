import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from medhack_ai_assistant.services.inference import (
    analyze_row,
    analyze_row_as_result,
    build_human_summary,
    calculate_jaccard,
    extract_mkb_details,
    extract_patient_features,
    format_factors,
    get_model_artifacts,
    normalize_factor_code,
    predict_row_factors,
    register_joblib_compat_classes,
    run_solution,
)


__all__ = [
    "analyze_row",
    "analyze_row_as_result",
    "build_human_summary",
    "calculate_jaccard",
    "extract_mkb_details",
    "extract_patient_features",
    "format_factors",
    "get_model_artifacts",
    "normalize_factor_code",
    "predict_row_factors",
    "register_joblib_compat_classes",
    "run_solution",
]


if __name__ == "__main__":
    run_solution()
