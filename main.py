import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medhack_ai_assistant.pipeline import run_training_pipeline
from medhack_ai_assistant.services.dashboard import get_dashboard_by_exam_id


def main() -> None:
    result = run_training_pipeline()
    dashboard = get_dashboard_by_exam_id(1015884464)
    print(dashboard.decision_label)
    print(dashboard.diagnoses)

if __name__ == "__main__":
    main()
