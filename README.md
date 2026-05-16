# MedHack AI Assistant

Проект решает задачу бинарной классификации медицинского текста: по данным
профосмотра модель определяет, есть ли у пациента противопоказания к вредным
или опасным производственным факторам из направления.

## Экспертная система (приказ 29н)

- База знаний: [`knowledge/contraindication_rules.json`](knowledge/contraindication_rules.json), [`knowledge/factors.json`](knowledge/factors.json).
- Сопоставление `assigned_harmful_factors` + `specialist_conclusions` с правилами Приложения 2.
- Развёрнутый отчёт с трассировкой правил (Streamlit или `infer_single_row`).
- Положите PDF в `data/29N.pdf` и запустите `python scripts/parse_29n_pdf.py` для расширения KB.

```bash
python main.py --mode expert          # submission_expert.csv + метрики CV
python main.py --mode hybrid          # submission_hybrid.csv (правила + XGBoost)
streamlit run app.py                  # UI для врача
python scripts/validate_kb_coverage.py
```

## Текущий ML-подход

- Используется NLP-подход на основе TF-IDF.
- В один текстовый признак объединяются поля:
  - `assigned_harmful_factors`
  - `specialist_conclusions`
- Классификатор: `SGDClassifier` с логистической регрессией
  (`loss="log_loss"`).
- Для дисбаланса классов используется `class_weight="balanced"`.
- Оптимальный порог предсказания подбирается на валидационной выборке с точки
  зрения F1-меры.
- После подбора порога модель переобучается на всей обучающей выборке и
  формирует файл `submission.csv`.

## Планируемый сценарий использования

В будущем планируется UI для врача: специалист загружает файл с данными
профосмотра, а ИИ-ассистент анализирует противопоказания и формирует черновик
документа для профпатолога.

Целевая логика продукта: построить модель, которая по данным профосмотра
определяет, к каким факторам из направления у пациента есть противопоказания.

## Запуск модели

```bash
python main.py
```

Ожидаемые входные файлы:

- `data/train.csv`
- `data/test.csv`

Результат сохраняется в `submission.csv`.

## Архитектура проекта

```text
.
├── app.py                         # Streamlit UI (экспертная система)
├── main.py                        # CLI: --mode ml|expert|hybrid
├── knowledge/                     # база знаний 29н (JSON)
├── scripts/                       # parse_29n_pdf.py, validate_kb_coverage.py
├── notebooks/kb_validation.ipynb
├── data/                          # train/test, 29N.pdf (добавить вручную)
├── src/medhack_ai_assistant/
│   ├── case_model.py              # нормализация кейса
│   ├── conclusions_parser.py      # парсинг JSON заключений
│   ├── rule_engine.py             # движок правил
│   ├── explanation.py             # развёрнутый отчёт
│   ├── expert_pipeline.py         # batch + CV
│   ├── hybrid.py                  # правила + XGBoost
│   └── ...
├── submission_expert.csv
└── submission_hybrid.csv
```

Такое разделение позволяет переиспользовать один и тот же ML-код и в CLI, и в
будущем интерфейсе врача.
