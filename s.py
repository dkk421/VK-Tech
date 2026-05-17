import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

@dataclass
class ModelConfig:
    max_features: int = 35000       # Немного увеличим словарь для точности
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    random_state: int = 42
    validation_size: float = 0.2
    C: float = 2.0                  # Чуть ослабим регуляризацию для лучшей подгонки текста
    max_iter: int = 200
    alpha: float = 0.5              # Вес альфа для Autonomous Score (F1 vs Jaccard)

# --- Кастомный класс для отказоустойчивого обучения ---
class RobustMultiOutputClassifier:
    def __init__(self, base_estimator, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.single_class_models = {}

    def fit(self, X, Y):
        from joblib import Parallel, delayed
        self.estimators_ = []
        self.single_class_models = {}
        num_outputs = Y.shape[1]
        
        tasks = []
        for i in range(num_outputs):
            y_sub = Y[:, i]
            unique_classes = np.unique(y_sub)
            if len(unique_classes) > 1:
                from sklearn.base import clone
                clf = clone(self.base_estimator)
                tasks.append((i, clf, X, y_sub))
            else:
                self.single_class_models[i] = unique_classes[0]
                
        if tasks:
            def _fit_single(index, clf, x_data, y_data):
                clf.fit(x_data, y_data)
                return index, clf
            trained_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single)(idx, clf, X, y_sub) for idx, clf, X, y_sub in tasks
            )
            trained_dict = {idx: clf for idx, clf in trained_results}
        else:
            trained_dict = {}

        for i in range(num_outputs):
            if i in trained_dict:
                self.estimators_.append(trained_dict[i])
            else:
                self.estimators_.append(None)
        return self

    def predict_proba(self, X):
        num_samples = X.shape[0]
        num_outputs = len(self.estimators_)
        probas = np.zeros((num_samples, num_outputs))
        
        for i in range(num_outputs):
            clf = self.estimators_[i]
            if clf is not None:
                # Берем вероятность класса 1
                probas[:, i] = clf.predict_proba(X)[:, 1]
            else:
                probas[:, i] = float(self.single_class_models[i])
        return probas

# --- МЕТРИКА ХАКАТОНА (AUTONOMOUS SCORE) ---
def calculate_autonomous_score(y_true, y_pred, alpha=0.5):
    # 1. Часть первая: F1 для факта "не годен"
    # Человек "не годен" (1), если у него есть хоть один маркер противопоказания
    has_contra_true = (y_true.sum(axis=1) > 0).astype(int)
    has_contra_pred = (y_pred.sum(axis=1) > 0).astype(int)
    
    f1_not_fit = f1_score(has_contra_true, has_contra_pred, zero_division=0)
    
    # 2. Часть вторая: Jaccard для реально не годных
    actual_not_fit_mask = (has_contra_true == 1)
    
    if idx_count := np.sum(actual_not_fit_mask):
        y_true_sub = y_true[actual_not_fit_mask]
        y_pred_sub = y_pred[actual_not_fit_mask]
        
        jaccard_list = []
        for i in range(idx_count):
            intersection = np.logical_and(y_true_sub[i], y_pred_sub[i]).sum()
            union = np.logical_or(y_true_sub[i], y_pred_sub[i]).sum()
            
            jaccard = intersection / union if union > 0 else 0.0
            jaccard_list.append(jaccard)
            
        mean_jaccard = np.mean(jaccard_list)
    else:
        mean_jaccard = 0.0
        
    # Итоговый скор
    score = alpha * f1_not_fit + (1 - alpha) * mean_jaccard
    return score, f1_not_fit, mean_jaccard

# --- Парсеры данных ---
def _find_project_root() -> Path:
    configured = os.getenv("SOLUTION_BASE_DIR")
    if configured: return Path(configured).expanduser().resolve()
    current = Path(__file__).resolve().parent
    return current if (current / "data").exists() else Path.cwd()

BASE_DIR = _find_project_root()
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
LEGACY_TRAIN_PATH = BASE_DIR / "train.csv"
MODELS_DIR = BASE_DIR / "data" / "models_weights"

def normalize_factor_code(value) -> str:
    if pd.isna(value) or value is None: return ""
    code = str(value).strip().replace(",", ".")
    code = re.sub(r"\s+", "", code)
    code = re.sub(r"\.+", ".", code).strip(".")
    return code if re.fullmatch(r"\d{1,2}(?:\.\d{1,3}){0,3}", code) else ""

def parse_factors_to_set(value) -> set[str]:
    val_str = str(value).strip()
    if pd.isna(value) or val_str in ("", "0", "0.0", "nan", "None"): return set()
    return {normalize_factor_code(f) for f in val_str.split(';') if normalize_factor_code(f)}

def prepare_text_features(df: pd.DataFrame) -> pd.Series:
    return df['specialist_conclusions'].fillna("").astype(str) + " " + df['assigned_harmful_factors'].fillna("").astype(str)

def main():
    config = ModelConfig()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_file = TRAIN_PATH if TRAIN_PATH.exists() else LEGACY_TRAIN_PATH
    if not train_file.exists():
        print(f"Ошибка: файл {train_file} не найден!")
        return

    print("Загрузка датасета...")
    train = pd.read_csv(train_file).dropna(subset=['has_contraindications'])
    
    target_col = 'contraindicated_factors' if 'contraindicated_factors' in train.columns else 'factors'
    train['target_list'] = train[target_col].fillna("0").apply(lambda x: list(parse_factors_to_set(x)))
    
    all_unique_factors = sorted(list(set([f for sublist in train['target_list'] for f in sublist])))
    print(f"Уникальных факторов в таргете: {len(all_unique_factors)}")

    y_matrix = np.zeros((len(train), len(all_unique_factors)), dtype=int)
    for i, target_list in enumerate(train['target_list']):
        for factor in target_list:
            if factor in all_unique_factors:
                y_matrix[i, all_unique_factors.index(factor)] = 1
                
    train['text_features'] = prepare_text_features(train)

    print("Векторизация текстов (TF-IDF)...")
    tfidf = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        lowercase=True
    )
    X_transformed = tfidf.fit_transform(train['text_features'])

    X_train, X_val, y_train, y_val = train_test_split(
        X_transformed, y_matrix, 
        test_size=config.validation_size, 
        random_state=config.random_state
    )

    print("Обучение базовых линейных моделей на CPU...")
    base_lr = LogisticRegression(C=config.C, max_iter=config.max_iter, random_state=config.random_state)
    robust_clf = RobustMultiOutputClassifier(base_estimator=base_lr, n_jobs=-1)
    robust_clf.fit(X_train, y_train)
    
    # Получаем вероятности вместо жестких меток 0/1
    val_probas = robust_clf.predict_proba(X_val)

    # --- СВЕРХВАЖНО: ПОДБОР ПОРОГА ПОД METРИКУ ХАКАТОНА ---
    print("\nОптимизация порога под Autonomous Score...")
    best_threshold = 0.5
    best_score = -1.0
    best_f1 = -1.0
    best_jac = -1.0
    
    # Перебираем пороги вероятностей
    for th in np.linspace(0.05, 0.50, 46):
        preds_th = (val_probas >= th).astype(int)
        score, f1_not, jac_mean = calculate_autonomous_score(y_val, preds_th, alpha=config.alpha)
        
        if score > best_score:
            best_score = score
            best_threshold = th
            best_f1 = f1_not
            best_jac = jac_mean

    print(f" Порог зафиксирован: {best_threshold:.3f}")
    print(f"-> Валидационный Autonomous Score: {best_score:.5f}")
    print(f"   └─ F1 (факт негодности): {best_f1:.5f}")
    print(f"   └─ Jaccard (совпадение причин): {best_jac:.5f}")

    print("\nПереобучение финальной модели на 100% данных...")
    robust_clf.fit(X_transformed, y_matrix)

    # Сохраняем кастомный мета-пайплайн
    model_data = {
        "tfidf": tfidf,
        "clf": robust_clf,
        "best_threshold": best_threshold,
        "factors": all_unique_factors
    }
    
    model_path = MODELS_DIR / "final_solution.joblib"
    joblib.dump(model_data, model_path, compress=3)
    
    print("\n" + "="*50)
    print("ВСЁ ГОТОВО К ОТПРАВКЕ НА ЛИДЕРБОРД!")
    print(f"Файл сохранен: {model_path.resolve()}")
    print("="*50)

if __name__ == "__main__":
    main()