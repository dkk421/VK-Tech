from dataclasses import dataclass

# Этот файл больше не содержит локального XGBoost-тренировочного скрипта.
# Для обучения/валидации используйте современные модули в src/medhack_ai_assistant/ml.py и src/medhack_ai_assistant/metrics.py.

@dataclass
class ModelConfig:
    """Legacy configuration stub.

    Оставлен для совместимости, если где-то используется импорт.
    """
    max_features: int = 15000
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 2
    random_state: int = 42
    threshold_min: float = 0.05
    threshold_max: float = 0.85
    threshold_count: int = 81
    validation_size: float = 0.2
    n_estimators: int = 180
    max_depth: int = 5
    learning_rate: float = 0.07
    alpha: float = 0.5

__all__ = ["ModelConfig"]
