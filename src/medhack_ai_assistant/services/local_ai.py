import ast
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from tqdm import tqdm

# ==========================================
# КОНФИГУРАЦИЯ (С расширением под XGBoost)
# ==========================================
@dataclass
class ModelConfig:
    max_features: int = 15000       # Текст заключений сложный, увеличиваем размер словаря
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    random_state: int = 42
    threshold_min: float = 0.05     
    threshold_max: float = 0.85
    threshold_count: int = 81
    validation_size: float = 0.2
    
    # Гиперпараметры XGBoost
    n_estimators: int = 180       
    max_depth: int = 5            
    learning_rate: float = 0.07   
    
    # Параметры задачи
    alpha: float = 0.5             # Весовой коэффициент для Autonomous Score


@dataclass(frozen=True)
class ValidationResult:
    threshold: float
    f1: float


# ==========================================
# ДВИЖОК РАСЧЕТА ОФИЦИАЛЬНОЙ МЕТРИКИ
# ==========================================
def parse_factors_to_set(value):
    """
    Безопасно переводит строку факторов или число (0) в множество (set).
    Учитывает, что '0' или пустые значения — это признак "Годен".
    """
    val_str = str(value).strip()
    if pd.isna(value) or val_str in ("", "0", "0.0", "nan", "None"):
        return set()
    # Разбиваем по ';' и очищаем от пробелов
    return {f.strip() for f in val_str.split(';') if f.strip()}


def calculate_autonomous_score(y_true_strings, y_pred_strings, alpha=0.5):
    """
    Вычисляет Autonomous Score строго по формуле из ТЗ хакатона.
    """
    true_sets = [parse_factors_to_set(x) for x in y_true_strings]
    pred_sets = [parse_factors_to_set(x) for x in y_pred_strings]
    
    # Часть 1: F1 для факта «не годен» (1 - если есть хотя бы одна причина, иначе 0)
    true_binary = np.array([1 if len(s) > 0 else 0 for s in true_sets])
    pred_binary = np.array([1 if len(s) > 0 else 0 for s in pred_sets])
    
    f1_component = f1_score(true_binary, pred_binary, pos_label=1, zero_division=0)
    
    # Часть 2: Jaccard для причин (только для РЕАЛЬНО «не годных» по y_true)
    jaccard_scores = []
    for t_set, p_set, is_unfit in zip(true_sets, pred_sets, true_binary):
        if is_unfit == 1:
            intersection = len(t_set.intersection(p_set))
            union = len(t_set.union(p_set))
            j_index = intersection / union if union > 0 else 0.0
            jaccard_scores.append(j_index)
            
    jaccard_component = np.mean(jaccard_scores) if jaccard_scores else 0.0
    
    # Итоговый взвешенный скор
    autonomous_score = alpha * f1_component + (1 - alpha) * jaccard_component
    
    return {
        'autonomous_score': autonomous_score,
        'f1_binary_unfit': f1_component,
        'jaccard_factors': jaccard_component
    }


# ==========================================
# СБОРКА КОНВЕЙЕРА И МОДЕЛЕЙ
# ==========================================
def build_model(config: ModelConfig, scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=config.max_features,
                ngram_range=(config.ngram_min, config.ngram_max),
                min_df=config.min_df,
                lowercase=True,
            ),
        ),
        (
            "clf",
            XGBClassifier(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=config.random_state,
                n_jobs=-1
            ),
        ),
    ])


def find_best_threshold(y_true: pd.Series, positive_proba: np.ndarray, config: ModelConfig) -> ValidationResult:
    thresholds = np.linspace(config.threshold_min, config.threshold_max, config.threshold_count)
    scores = [f1_score(y_true, positive_proba >= threshold, zero_division=0) for threshold in thresholds]
    best_idx = int(np.argmax(scores))
    return ValidationResult(threshold=float(thresholds[best_idx]), f1=float(scores[best_idx]))


def prepare_text_features(df: pd.DataFrame) -> pd.Series:
    """Извлечение текстовых признаков из медицинских записей."""
    specialist_text = df['specialist_conclusions'].fillna("").astype(str)
    assigned_text = df['assigned_harmful_factors'].fillna("").astype(str)
    return specialist_text + " " + assigned_text


def parse_factors_from_clean_text(text):
    """Вспомогательный парсер факторов для построения списков обучения"""
    val_str = str(text).strip()
    if pd.isna(text) or val_str in ("", "0", "0.0", "nan", "None"):
        return []
    return [f.strip() for f in val_str.split(';') if f.strip()]


# ==========================================
# ОСНОВНОЙ СТАРТ СКРИПТА
# ==========================================
if __name__ == "__main__":
    config = ModelConfig()

    print("Загрузка данных...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Таргет в обучающей выборке (используем имя из исходного датасета для чтения)
    target_col_in_train = 'contraindicated_factors' if 'contraindicated_factors' in train.columns else 'factors'
    
    # Предобработка меток
    train = train.dropna(subset=['has_contraindications'])
    train['clean_target_str'] = train[target_col_in_train].fillna("0").astype(str)
    train['target_list'] = train['clean_target_str'].apply(parse_factors_from_clean_text)
    
    # Находим все уникальные факторы противопоказаний
    all_unique_factors = sorted(list(set([factor for sublist in train['target_list'] for factor in sublist])))
    print(f"Обнаружено уникальных факторов противопоказаний: {len(all_unique_factors)}")

    print("Генерация текстовых фичей через TF-IDF...")
    train['text_features'] = prepare_text_features(train)
    X_test_text = prepare_text_features(test)

    # ----------------------------------------------------------------
    # ЭТАП 1: Разделение на Train/Valid и запуск расчета локальной метрики
    # ----------------------------------------------------------------
    print(f"\nВыделение {config.validation_size * 100}% данных на валидацию...")
    train_idx, valid_idx = train_test_split(
        train.index, 
        test_size=config.validation_size, 
        random_state=config.random_state
    )
    
    df_train_fold = train.loc[train_idx]
    df_valid_fold = train.loc[valid_idx]

    best_thresholds = {}
    valid_preds_dict = {}

    print("Обучение пула моделей XGBoost и оптимизация порогов...")
    for factor in tqdm(all_unique_factors, desc="Валидация факторов"):
        y_train_bi = df_train_fold['target_list'].apply(lambda x: 1 if factor in x else 0)
        y_valid_bi = df_valid_fold['target_list'].apply(lambda x: 1 if factor in x else 0)
        
        if y_train_bi.sum() == 0:
            best_thresholds[factor] = 0.5
            valid_preds_dict[factor] = np.zeros(len(df_valid_fold), dtype=bool)
            continue
            
        # Расчет scale_pos_weight
        pos_c = y_train_bi.sum()
        neg_c = len(y_train_bi) - pos_c
        scale_pos_weight = neg_c / pos_c if pos_c > 0 else 1.0
        
        # Обучение
        model_fold = build_model(config, scale_pos_weight=scale_pos_weight)
        model_fold.fit(df_train_fold['text_features'], y_train_bi)
        
        # Поиск лучшего порога конкретно под данный фактор
        valid_proba = model_fold.predict_proba(df_valid_fold['text_features'])[:, 1]
        val_res = find_best_threshold(y_valid_bi, valid_proba, config)
        best_thresholds[factor] = val_res.threshold
        
        valid_preds_dict[factor] = (valid_proba >= val_res.threshold)

    # Применение бизнес-правил к валидационным предсказаниям
    df_valid_preds = pd.DataFrame(valid_preds_dict, index=df_valid_fold.index)
    df_valid_fold['assigned_list'] = df_valid_fold['assigned_harmful_factors'].fillna("").apply(parse_factors_from_clean_text)
    
    valid_predicted_strings = []
    for idx, row in df_valid_preds.iterrows():
        pred_fact = row[row == True].index.tolist()
        assigned = df_valid_fold.loc[idx, 'assigned_list']
        filtered = [f for f in pred_fact if f in assigned]
        
        # Записываем в формате ТЗ хакатона (без пробелов, если пусто - 0)
        valid_predicted_strings.append(";".join(sorted([str(f) for f in filtered])) if filtered else "0")

    # Расчет финального скора на валидации
    metrics = calculate_autonomous_score(
        y_true_strings=df_valid_fold['clean_target_str'],
        y_pred_strings=valid_predicted_strings,
        alpha=config.alpha
    )
    
    print("\n" + "="*55)
    print(f"РЕЗУЛЬТАТЫ ЛОКАЛЬНОЙ ВАЛИДАЦИИ:")
    print("="*55)
    print(f"AUTONOMOUS SCORE:  {metrics['autonomous_score']:.5f}")
    print(f"├── F1 (Факт 'Не годен'): {metrics['f1_binary_unfit']:.5f}")
    print(f"└── Jaccard (Причины):    {metrics['jaccard_factors']:.5f}")
    print("="*55 + "\n")

    # ----------------------------------------------------------------
    # ЭТАП 2: Обучение моделей на 100% данных и сборка сабмита по ТЗ
    # ----------------------------------------------------------------
    print("Обучение финальных моделей на полной выборке...")
    final_models = {}
    for factor in tqdm(all_unique_factors, desc="Финальное обучение"):
        y_full_bi = train['target_list'].apply(lambda x: 1 if factor in x else 0)
        if y_full_bi.sum() == 0: continue
            
        pos_c = y_full_bi.sum()
        neg_c = len(y_full_bi) - pos_c
        scale_pos_weight = neg_c / pos_c if pos_c > 0 else 1.0
        
        model = build_model(config, scale_pos_weight=scale_pos_weight)
        model.fit(train['text_features'], y_full_bi)
        final_models[factor] = model

    print("Генерация предсказаний для тестового датасета...")
    test_preds_dict = {}
    for factor, model in final_models.items():
        thresh = best_thresholds[factor]
        positive_proba = model.predict_proba(X_test_text)[:, 1]
        test_preds_dict[factor] = (positive_proba >= thresh)
        
    df_test_preds = pd.DataFrame(test_preds_dict, index=test.index)

    print("Применение бизнес-ограничений и сохранение по формату ТЗ...")
    test['assigned_list'] = test['assigned_harmful_factors'].fillna("").apply(parse_factors_from_clean_text)
    
    final_test_rows = []
    for idx, row in df_test_preds.iterrows():
        predicted_factors = row[row == True].index.tolist()
        assigned_factors = test.loc[idx, 'assigned_list']
        filtered_factors = [f for f in predicted_factors if f in assigned_factors]
        
        # СТРОГО ПО ТЗ: разделение через ";" без пробелов, если список пуст - пишем "0"
        if filtered_factors:
            final_test_rows.append(";".join(sorted([str(f) for f in filtered_factors])))
        else:
            final_test_rows.append("0")

    # СТРОГО ПО ТЗ: Колонки 'exam_row_id' и 'factors'
    submission = pd.DataFrame({
        'exam_row_id': test['exam_row_id'],
        'factors': final_test_rows
    })

    submission.to_csv("submission.csv", index=False)
    print("\nФайл 'submission.csv' успешно сгенерирован и полностью готов к отправке!")