import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.medhack_ai_assistant.expert_pipeline import run_expert_pipeline
from src.medhack_ai_assistant.hybrid import run_hybrid_pipeline
from src.medhack_ai_assistant.pipeline import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="MedHack AI Assistant")
    parser.add_argument(
        "--mode",
        choices=("ml", "expert", "hybrid"),
        default="expert",
        help="Режим: ml (бинарный), expert (правила 29н), hybrid",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Не считать метрики на train",
    )
    args = parser.parse_args()

    if args.mode == "ml":
        result = run_training_pipeline()
        print("Target distribution:")
        print(result.target_distribution)
        print(f"Best threshold: {result.validation.threshold:.2f}")
        print(f"Validation F1: {result.validation.f1:.4f}")
        print(f"Submission saved to: {result.submission_path}")
        return

    if args.mode == "expert":
        result = run_expert_pipeline(evaluate_train=not args.no_eval)
        if result.metrics:
            print("Expert CV metrics (train):")
            for key, value in result.metrics.items():
                print(f"  {key}: {value:.4f}")
        print(f"Submission saved to: {result.submission_path}")
        if result.sample_report:
            print("\n--- Sanity report (sample) ---\n")
            print(result.sample_report.markdown[:2000])
        return

    result = run_hybrid_pipeline(evaluate_train=not args.no_eval)
    if result.metrics:
        print("Hybrid expert-component metrics (train):")
        for key, value in result.metrics.items():
            print(f"  {key}: {value:.4f}")
    print(f"ML threshold: {result.ml_threshold:.2f}")
    print(f"Submission saved to: {result.submission_path}")


if __name__ == "__main__":
    main()
