import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score


def parse_factors_to_set(text):

    if pd.isna(text) or str(text).strip() == "" or str(text).strip().lower() == "nan":
        return set()
    # Разбиваем по разделяющему знаку и очищаем от пробелов
    return {f.strip() for f in str(text).split(';') if f.strip()}


def calculate_autonomous_score(y_true, y_pred, alpha=0.5):

    # 1. Парсим строки факторов в множества (sets)
    true_sets = [parse_factors_to_set(x) for x in y_true]
    pred_sets = [parse_factors_to_set(x) for x in y_pred]
    
    # 2. Вычисляем бинарный флаг "не годен" (1 - если список факторов не пуст, иначе 0)
    true_binary = np.array([1 if len(s) > 0 else 0 for s in true_sets])
    pred_binary = np.array([1 if len(s) > 0 else 0 for s in pred_sets])
    
    # Компонента 1: Обычный бинарный F1-score
    # Используем pos_label=1, так как нас интересует именно класс «не годен»
    f1_component = f1_score(true_binary, pred_binary, pos_label=1, zero_division=0)
    
    # Компонента 2: Jaccard-сходство для причин (только для РЕАЛЬНО «не годных»)
    jaccard_scores = []
    
    for t_set, p_set, is_unfit in zip(true_sets, pred_sets, true_binary):
        if is_unfit == 1:  
            intersection = len(t_set.intersection(p_set))
            union = len(t_set.union(p_set))
            
           
            j_index = intersection / union if union > 0 else 0.0
            jaccard_scores.append(j_index)
            
    # Среднее значение по всем реально не годным сотрудникам
    jaccard_component = np.mean(jaccard_scores) if jaccard_scores else 0.0
    
    # Итоговый Autonomous Score
    autonomous_score = alpha * f1_component + (1 - alpha) * jaccard_component
    
    return {
        'autonomous_score': autonomous_score,
        'f1_binary_unfit': f1_component,
        'jaccard_factors': jaccard_component
    }


if __name__ == "__main__":
    # Добавляем интерфейс командной строки для удобства валидации на хакатоне
    parser = argparse.ArgumentParser(description="Расчет метрики Autonomous Score")
    parser.add_argument("--true_path", type=str, required=True, help="Путь к файлу с истинными ответами (CSV)")
    parser.add_argument("--pred_path", type=str, required=True, help="Путь к файлу submission (CSV)")
    parser.add_argument("--id_col", type=str, default="exam_row_id", help="Название колонки с ID")
    parser.add_argument("--target_col", type=str, default="contraindicated_factors", help="Название колонки с факторами")
    parser.add_argument("--alpha", type=float, default=0.5, help="Коэффициент alpha для взвешивания")
    
    args = parser.parse_args()
    
    # Загрузка файлов
    df_true = pd.read_csv(args.true_path)
    df_pred = pd.read_csv(args.pred_path)
    
    # Выравниваем предсказания по ID истинных значений, чтобы гарантировать правильный порядок строк
    df_pred = df_pred.set_index(args.id_col).reindex(df_true[args.id_col]).reset_index()
    
    # Расчет
    metrics = calculate_autonomous_score(
        y_true=df_true[args.target_col],
        y_pred=df_pred[args.target_col],
        alpha=args.alpha
    )
    
    # Вывод результатов в консоль
    print("\n" + "="*40)
    print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (Alpha = {args.alpha}):")
    print("="*40)
    print(f"Autonomous Score:  {metrics['autonomous_score']:.5f}")
    print(f"├── F1 (Fact Unfit): {metrics['f1_binary_unfit']:.5f}")
    print(f"└── Jaccard (Causes): {metrics['jaccard_factors']:.5f}")
    print("="*40 + "\n")
