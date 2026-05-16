"""
Отчёт о покрытии базы знаний по train.csv.

  python scripts/validate_kb_coverage.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.medhack_ai_assistant.case_model import build_exam_case
from src.medhack_ai_assistant.config import RULES_PATH, TRAIN_PATH
from src.medhack_ai_assistant.conclusions_parser import parse_factors
from src.medhack_ai_assistant.kb_loader import load_rules_catalog, rules_by_factor
from src.medhack_ai_assistant.rule_engine import infer_case


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    catalog = load_rules_catalog(RULES_PATH)
    index = rules_by_factor(catalog.rules)

    assigned_counter: Counter = Counter()
    covered_counter: Counter = Counter()

    triggered_rows = 0
    for _, row in train.iterrows():
        case = build_exam_case(row)
        for factor in case.assigned_factors:
            assigned_counter[factor] += 1
            if factor in index:
                covered_counter[factor] += 1
        result = infer_case(case, rules_index=index)
        if result.has_contraindications:
            triggered_rows += 1

    all_factors = set(assigned_counter)
    covered = {factor for factor in all_factors if factor in index}
    print(f"Строк train: {len(train)}")
    print(f"Уникальных факторов в направлениях: {len(all_factors)}")
    print(f"Факторов с хотя бы одним правилом в KB: {len(covered)}")
    print(f"Строк с срабатыванием экспертной системы: {triggered_rows}")
    print(f"Правил в KB: {len(catalog.rules)}")

    uncovered = sorted(all_factors - covered, key=lambda value: -assigned_counter[value])[:20]
    if uncovered:
        print("\nТоп непокрытых факторов (по частоте):")
        for factor in uncovered:
            print(f"  {factor}: {assigned_counter[factor]} назначений")


if __name__ == "__main__":
    main()
