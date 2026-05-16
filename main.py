from src.medhack_ai_assistant.pipeline import run_training_pipeline


def main() -> None:
    result = run_training_pipeline()

    print("Target distribution:")
    print(result.target_distribution)
    print(result.target_distribution_normalized)
    print(f"Best threshold: {result.validation.threshold:.2f}")
    print(f"Validation F1: {result.validation.f1:.4f}")
    print(f"Submission saved to: {result.submission_path}")
    print(result.submission["has_contraindications"].value_counts())


if __name__ == "__main__":
    main()
