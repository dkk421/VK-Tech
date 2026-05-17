import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# =========================================================================
# ОБЯЗАТЕЛЬНО: Объявляем тот же класс, чтобы joblib смог распаковать модель
# =========================================================================
class RobustMultiOutputClassifier:
    def __init__(self, base_estimator, n_jobs=-1):
        self.base_estimator = base_estimator
        self.n_jobs = n_jobs
        self.estimators_ = []
        self.single_class_models = {}

    def fit(self, X, Y):
        pass # Для инференса fit не нужен, но структура класса должна совпадать

    def predict_proba(self, X):
        num_samples = X.shape[0]
        num_outputs = len(self.estimators_)
        probas = np.zeros((num_samples, num_outputs))
        
        for i in range(num_outputs):
            clf = self.estimators_[i]
            if clf is not None:
                probas[:, i] = clf.predict_proba(X)[:, 1]
            else:
                probas[:, i] = float(self.single_class_models[i])
        return probas


def _find_project_root() -> Path:
    configured = os.getenv("SOLUTION_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path(__file__).resolve().parent
    if (current / "data").exists():
        return current
    return Path.cwd()

BASE_DIR = _find_project_root()
TEST_PATH = BASE_DIR / "data" / "test.csv" 
LEGACY_TEST_PATH = BASE_DIR / "test.csv"
MODEL_PATH = BASE_DIR / "data" / "models_weights" / "final_solution.joblib"
SUBMISSION_PATH = BASE_DIR / "submission.csv"

def prepare_text_features(df: pd.DataFrame) -> pd.Series:
    specialist_text = df['specialist_conclusions'].fillna("").astype(str)
    assigned_text = df['assigned_harmful_factors'].fillna("").astype(str)
    return specialist_text + " " + assigned_text

def main():
    if not MODEL_PATH.exists():
        print(f"Ошибка: Файл модели {MODEL_PATH.name} не найден!")
        return

    test_file = TEST_PATH if TEST_PATH.exists() else LEGACY_TEST_PATH
    if not test_file.exists():
        print(f"Ошибка: Тестовый файл {test_file} не найден!")
        return

    print("Загрузка обученной модели и метаданных...")
    # Теперь распаковка пройдет успешно, так как класс RobustMultiOutputClassifier объявлен выше
    artifacts = joblib.load(MODEL_PATH)
    tfidf = artifacts["tfidf"]
    robust_clf = artifacts["clf"]
    best_threshold = artifacts["best_threshold"]
    factors_columns = artifacts["factors"]

    print("Загрузка тестовых данных...")
    test_df = pd.read_csv(test_file)
    
    if 'exam_row_id' not in test_df.columns:
        print("Ошибка: В тестовом датасете нет колонки 'exam_row_id'!")
        return

    print("Подготовка признаков...")
    text_features = prepare_text_features(test_df)

    print("Векторизация и расчет вероятностей...")
    X_test = tfidf.transform(text_features)
    probas = robust_clf.predict_proba(X_test)

    print(f"Применение оптимального порога ({best_threshold:.3f})...")
    binary_preds = (probas >= best_threshold).astype(int)

    print("Формирование финальных строк факторов...")
    final_factors = []
    
    for i in range(len(test_df)):
        active_indices = np.where(binary_preds[i] == 1)[0]
        
        if len(active_indices) > 0:
            row_factors = [factors_columns[idx] for idx in active_indices]
            row_factors = sorted(list(set(row_factors)))
            factors_str = ";".join(row_factors)
        else:
            factors_str = "0"
            
        final_factors.append(factors_str)

    submission = pd.DataFrame({
        "exam_row_id": test_df["exam_row_id"],
        "factors": final_factors
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    
    print("\n" + "="*50)
    print("САБМИШН УСПЕШНО СГЕНЕРИРОВАН!")
    print(f"Путь к файлу: {SUBMISSION_PATH.resolve()}")
    print(f"Размерность: {submission.shape[0]} строк")
    print(f"Доля не годных (с факторами): {(submission['factors'] != '0').mean():.2%}")
    print("="*50)

if __name__ == "__main__":
    main()