# MedHack AI Assistant

Интеллектуальный ассистент врача-профпатолога для анализа профосмотров и выявления противопоказанных вредных факторов по данным `train.csv` / `test.csv`.

Проект сейчас работает в локальном режиме: Streamlit-интерфейс вызывает готовую ML-модель из `data/models_weights/final_solution.joblib`, показывает заключение по выбранному пациенту и формирует DOCX-документ врача.

## Основные сценарии

### Запуск UI

```powershell
uv run streamlit run app.py
```

В интерфейсе можно:

- выбрать пациента по `exam_row_id`;
- запустить AI-анализ по одному пациенту;
- скачать заключение врача в формате DOCX;
- запустить пакетный анализ нескольких пациентов из `train.csv`;
- перейти из пакетного списка в карточку конкретного пациента.

### Генерация submission

```powershell
uv run python solution.py
```

Результат сохраняется в корень проекта:

```text
submission.csv
```

Формат:

```csv
exam_row_id,factors
10158844,0
10158845,18.1;12.2
10158854,6.3
```

### Обучение модели

```powershell
uv run python s.py
```

Скрипт обучает TF-IDF + Logistic Regression multi-output модель и сохраняет основной артефакт:

```text
data/models_weights/final_solution.joblib
```

### Альтернативная генерация submission

```powershell
uv run python generate_submission.py
```

Этот скрипт напрямую загружает сохраненную модель и формирует `submission.csv`.

## Роли ключевых файлов

```text
app.py                                      Streamlit entrypoint
solution.py                                 тонкий entrypoint для submission и совместимых импортов
s.py                                        обучение модели и сохранение final_solution.joblib
generate_submission.py                      отдельный скрипт генерации submission из сохраненной модели

data/train.csv                              обучающий датасет
data/test.csv                               тестовый датасет
data/models_weights/final_solution.joblib   основная сохраненная ML-модель

src/medhack_ai_assistant/ui/                Streamlit UI-компоненты
src/medhack_ai_assistant/services/          inference, dashboard, mining, export DOCX/PDF
src/medhack_ai_assistant/parsing/           парсинг заключений и Приказа 29н
src/medhack_ai_assistant/domain/            dataclass-модели
```

## Важные замечания

- `solution.py`, `s.py` и `generate_submission.py` оставлены на местах специально, чтобы не менять рабочие команды перед дедлайном.
- Основная inference-логика находится в `src/medhack_ai_assistant/services/inference.py`.
- `solution1.py` считается локальным черновиком и добавлен в `.gitignore`.
- `submission.csv` также игнорируется git, потому что это генерируемый результат.
- Для загрузки старого `joblib`-артефакта в `solution.py` есть compatibility alias для `main.RobustMultiOutputClassifier`.
