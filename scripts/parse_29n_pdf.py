"""
Полуавтоматическое извлечение таблиц Приложения 2 из PDF приказа 29н.

Использование (из корня VK-Tech-test-another-models):
  python scripts/parse_29n_pdf.py
  python scripts/parse_29n_pdf.py --pdf data/29N.pdf --output knowledge/rules_raw.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.medhack_ai_assistant.config import PDF_29N_PATH, RULES_RAW_PATH
from src.medhack_ai_assistant.kb_models import ContraindicationRule, RulesCatalog


def expand_factor_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    for part in re.split(r"[,;]", raw):
        part = part.strip()
        if not part:
            continue
        range_match = re.match(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", part)
        if range_match:
            start, end = range_match.groups()
            if "." not in start and "." not in end:
                for value in range(int(start), int(end) + 1):
                    tokens.append(str(value))
            else:
                tokens.extend([start, end])
            continue
        tokens.append(part)
    return tokens


def parse_mkb_cell(raw: str) -> list[str]:
    patterns: list[str] = []
    for chunk in re.split(r"[,;]", raw):
        chunk = chunk.strip().upper()
        if not chunk:
            continue
        range_match = re.match(r"([A-Z]\d{2})\s*[-–]\s*([A-Z]\d{2})", chunk)
        if range_match:
            patterns.append(range_match.group(1))
            patterns.append(range_match.group(2))
        else:
            patterns.append(chunk.replace(" ", ""))
    return patterns


def extract_rules_from_pdf(pdf_path: Path) -> list[dict]:
    import pdfplumber

    rules: list[dict] = []
    rule_counter = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
            for table in tables:
                for row in table:
                    if not row or len(row) < 4:
                        continue
                    cells = [str(cell or "").strip() for cell in row]
                    mkb_cell = cells[2] if len(cells) > 2 else ""
                    factor_cell = cells[-1] if cells else ""
                    if not re.search(r"[A-Z]\d{2}", mkb_cell, re.I):
                        continue
                    factors = expand_factor_tokens(factor_cell)
                    if not factors:
                        continue
                    rule_counter += 1
                    disease = cells[1] if len(cells) > 1 else ""
                    rules.append(
                        ContraindicationRule(
                            rule_id=f"pdf_{rule_counter:04d}",
                            disease_name=disease[:500],
                            mkb_patterns=parse_mkb_cell(mkb_cell),
                            text_patterns=[],
                            factor_codes=factors,
                            appendix_ref="Прил.2 (из PDF)",
                            validated=False,
                        ).model_dump()
                    )
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсинг приказа 29н (Приложение 2)")
    parser.add_argument("--pdf", type=Path, default=PDF_29N_PATH)
    parser.add_argument("--output", type=Path, default=RULES_RAW_PATH)
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF не найден: {args.pdf}")
        print("Положите 29N.pdf в data/ (см. data/README_29N.txt)")
        sys.exit(1)

    raw_rules = extract_rules_from_pdf(args.pdf)
    catalog = RulesCatalog(version="1.0", rules=raw_rules)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {"version": catalog.version, "source": str(args.pdf), "rules": raw_rules},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Извлечено правил: {len(raw_rules)} → {args.output}")


if __name__ == "__main__":
    main()
