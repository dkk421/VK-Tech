#!/usr/bin/env python3
"""
CLI entrypoint for model training and evaluation.

This script trains the ML model on the training dataset and creates
a submission file with predictions on the test dataset.

Usage:
    python train.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from medhack_ai_assistant.pipeline import run_training_pipeline


def main():
    """Run the complete training pipeline."""
    print("=" * 70)
    print("🏥 MedHack AI Assistant - Model Training")
    print("=" * 70)
    print()

    try:
        result = run_training_pipeline()

        print("✅ Training completed successfully!")
        print()
        print("📊 Results Summary")
        print("-" * 70)
        print(f"Submission saved to: {result.submission_path}")
        print()

        print("Target Distribution (Training Data):")
        print(result.target_distribution)
        print()

        print("Target Distribution (Normalized):")
        print(result.target_distribution_normalized)
        print()

        print("Validation Results:")
        print(f"  - Accuracy:        {result.validation.accuracy:.4f}")
        print(f"  - Precision:       {result.validation.precision:.4f}")
        print(f"  - Recall:          {result.validation.recall:.4f}")
        print(f"  - F1 Score:        {result.validation.f1_score:.4f}")
        print(f"  - ROC AUC:         {result.validation.roc_auc:.4f}")
        print(f"  - Optimal Threshold: {result.validation.threshold:.4f}")
        print()

        print("Submission Preview:")
        print(result.submission.head(10))
        print(f"  ... ({len(result.submission)} rows total)")
        print()

        print("=" * 70)
        print("✨ Training pipeline completed successfully!")
        print("=" * 70)

    except Exception as e:
        print("❌ Error during training:")
        print(f"  {type(e).__name__}: {str(e)}")
        print()
        raise


if __name__ == "__main__":
    main()
