"""Быстрая проверка экспертной системы."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.medhack_ai_assistant.config import TRAIN_PATH
from src.medhack_ai_assistant.expert_pipeline import infer_single_row, run_expert_pipeline

train = pd.read_csv(TRAIN_PATH)
row = train[train["exam_row_id"] == 1015330919].iloc[0]
inf, report = infer_single_row(row)
assert inf.contraindicated_factors == {"4.4"}, inf.contraindicated_factors
print("Sanity OK:", report.contraindicated_factors)
result = run_expert_pipeline(evaluate_train=True)
print("Metrics:", result.metrics)
