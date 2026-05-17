#!/usr/bin/env python3
"""
Примеры использования AI-анализа для осмотров.

Используйте эти примеры для интеграции в свои приложения.
"""

import asyncio
import json
from pathlib import Path

import pandas as pd

# Добавляем src в путь для импортов
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from medhack_ai_assistant.pipeline import analyze_patient_exam
from medhack_ai_assistant.services.dashboard import build_patient_exam
from medhack_ai_assistant.data import load_data
from medhack_ai_assistant.config import TRAIN_PATH, ID_COLUMN


async def example_1_single_exam():
    """Пример 1: Анализ одного осмотра по ID."""
    print("\n" + "=" * 70)
    print("ПРИМЕР 1: Анализ одного осмотра по ID")
    print("=" * 70)

    # Загружаем данные
    train = load_data(TRAIN_PATH)
    exam_id = train[ID_COLUMN].iloc[0]

    # Находим строку по ID
    row = train[train[ID_COLUMN] == exam_id].iloc[0]

    # Запускаем анализ
    print(f"\nАнализирую осмотр #{exam_id}...")
    result = await analyze_patient_exam(row)

    # Выводим результаты
    print(f"\n✅ Результат анализа:")
    print(f"   Вердикт: {result.verdict}")
    print(f"   Статус: {result.status}")
    print(f"   Факторы: {', '.join(result.factors) or 'нет'}")
    print(f"   Резюме: {result.summary[:100]}...")
    print(f"   Доверие: {result.raw_llm_response.get('confidence', 0):.1%}")


async def example_2_batch_analysis():
    """Пример 2: Анализ нескольких осмотров."""
    print("\n" + "=" * 70)
    print("ПРИМЕР 2: Батч-анализ нескольких осмотров")
    print("=" * 70)

    train = load_data(TRAIN_PATH)
    rows = train.head(3)

    results = []
    for idx, (_, row) in enumerate(rows.iterrows(), 1):
        exam_id = row[ID_COLUMN]
        print(f"\n[{idx}/3] Анализирую осмотр #{exam_id}...")

        try:
            result = await analyze_patient_exam(row)
            results.append({
                'exam_id': exam_id,
                'verdict': result.verdict,
                'factors': ', '.join(result.factors) or 'нет',
                'status': result.status,
            })
            print(f"      ✅ {result.verdict}")
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
            results.append({
                'exam_id': exam_id,
                'verdict': 'ERROR',
                'factors': str(e),
                'status': 'FAILED',
            })

    # Выводим сводку
    print("\n" + "-" * 70)
    print("📊 Сводка результатов:")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


async def example_3_detailed_result():
    """Пример 3: Детальный анализ результата."""
    print("\n" + "=" * 70)
    print("ПРИМЕР 3: Детальный анализ с объяснениями")
    print("=" * 70)

    train = load_data(TRAIN_PATH)
    row = train[train[ID_COLUMN] == train[ID_COLUMN].iloc[0]].iloc[0]

    print(f"\nАнализирую осмотр...")
    result = await analyze_patient_exam(row)

    print(f"\n📋 Вердикт: {result.verdict}")
    print(f"📌 Статус: {result.status}")
    print(f"💪 Доверие: {result.raw_llm_response.get('confidence', 0):.1%}")

    if result.factors:
        print(f"\n⚠️  Противопоказанные факторы:")
        for factor in result.factors:
            print(f"   • {factor}")

    if result.evidence:
        print(f"\n📌 Доказательства:")
        for i, evidence in enumerate(result.evidence, 1):
            print(f"\n   {i}. {evidence.get('diagnosis', 'N/A')}")
            if 'icd_code' in evidence:
                print(f"      МКБ-код: {evidence.get('icd_code')}")
            if 'specialist' in evidence:
                print(f"      Специалист: {evidence.get('specialist')}")
            if 'reasoning' in evidence:
                print(f"      Обоснование: {evidence.get('reasoning')}")

    print(f"\n📝 Резюме:")
    print(f"   {result.summary}")

    if result.follow_up_draft:
        print(f"\n📄 Черновик документа:")
        print(json.dumps(result.follow_up_draft, ensure_ascii=False, indent=2))

    # Quality Gate
    if result.quality_gate:
        print(f"\n✅ Quality Gate:")
        print(f"   Можно анализировать: {result.quality_gate.can_analyze}")
        if result.quality_gate.reasons:
            print(f"   Причины ограничений:")
            for reason in result.quality_gate.reasons:
                print(f"   • {reason}")


async def example_4_exam_object():
    """Пример 4: Использование PatientExam объекта напрямую."""
    print("\n" + "=" * 70)
    print("ПРИМЕР 4: Использование PatientExam объекта")
    print("=" * 70)

    train = load_data(TRAIN_PATH)
    row = train[train[ID_COLUMN] == train[ID_COLUMN].iloc[0]].iloc[0]

    # Преобразуем в PatientExam объект
    exam = build_patient_exam(row)

    print(f"\nОсмотр: ID={exam.exam_row_id}")
    print(f"Факторы: {exam.assigned_harmful_factors}")
    print(f"Диагнозы: {exam.diagnoses}")

    print(f"\nЗапускаю анализ...")
    result = await analyze_patient_exam(exam)

    print(f"✅ Результат: {result.verdict}")


async def main():
    """Запуск всех примеров."""
    print("\n🏥 Примеры использования MedHack AI Assistant")

    try:
        await example_1_single_exam()
        await example_2_batch_analysis()
        await example_3_detailed_result()
        await example_4_exam_object()

        print("\n" + "=" * 70)
        print("✅ Все примеры выполнены успешно!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("\n💡 Убедитесь, что:")
        print("   1. vLLM сервер запущен: python run_vllm_server.py")
        print("   2. Все зависимости установлены: pip install -e .")
        print("   3. Данные загружены: data/train.csv существует")


if __name__ == "__main__":
    asyncio.run(main())
