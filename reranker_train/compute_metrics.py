import os
import numpy as np
import pandas as pd  # для удобной таблицы, можно и без него

def conv(x):
    if x == b'True' or x == 'True':   # для байтов и строк
        return 1
    if x == b'False' or x == 'False':
        return 0
    return float(x) if '.' in str(x) else int(x)

def compute_metrics(df):
    # df должен содержать колонки 'idx' и 'rel', отсортированные по idx
    df = df.sort_values('idx').reset_index(drop=True)
    rel = df['rel'].values
    positions = df['idx'].values  # предполагаем, что позиции начинаются с 1
    # убедимся, что idx идут подряд (если нет, то берём порядковый номер)
    # но для надёжности будем использовать порядок строк (индекс + 1)
    k = 10
    # количество релевантных во всём списке
    total_relevant = int(rel.sum())
    
    # reciprocal rank
    first_rel_idx = np.where(rel == 1)[0]
    reciprocal_rank = 1.0 / (first_rel_idx[0] + 1) if len(first_rel_idx) > 0 else 0.0
    
    # precision@10, recall@10
    top10_rel = rel[:k].sum()
    precision_at_10 = top10_rel / k
    recall_at_10 = top10_rel / total_relevant if total_relevant > 0 else 0.0
    
    # f1@10
    if precision_at_10 + recall_at_10 > 0:
        f1_at_10 = 2 * precision_at_10 * recall_at_10 / (precision_at_10 + recall_at_10)
    else:
        f1_at_10 = 0.0
    
    # average precision @10
    if total_relevant == 0:
        ap_at_10 = 0.0
    else:
        precisions = []
        rel_cum = 0
        for i in range(min(k, len(rel))):
            if rel[i] == 1:
                rel_cum += 1
                p = rel_cum / (i + 1)
                precisions.append(p)
        # число релевантных, учитываемых в AP@10, обычно min(10, total_relevant)
        r = min(k, total_relevant)
        ap_at_10 = sum(precisions) / r if r > 0 else 0.0
    
    return {
        'total_relevant': total_relevant,
        'total_predicted': len(df),   # общее количество документов в выдаче
        'reciprocal_rank': reciprocal_rank,
        'precision@10': precision_at_10,
        'recall@10': recall_at_10,
        'f1_score@10': f1_at_10,
        'average_precision@10': ap_at_10
    }

# Собираем все CSV-файлы (кроме скриптов)
files = [f for f in os.listdir('.') if not f.endswith('.py') and os.path.isfile(f)]

all_results = []
for file in files:
    print(f"Обработка {file}...")
    try:
        # Читаем CSV; предполагаем, что колонки: idx,max,mean,std,perc_90,num_passed,density,range,rel
        data = np.loadtxt(file, delimiter=',', skiprows=1, converters=conv)
        # Если файл пустой или содержит только заголовок – пропускаем
        if data.ndim == 1:
            # только одна строка: превращаем в 2D массив
            data = data.reshape(1, -1)
        # Извлекаем idx и rel
        idx = data[:, 0].astype(int)
        rel = data[:, -1].astype(int)   # последняя колонка – rel
        df = pd.DataFrame({'idx': idx, 'rel': rel})
        metrics = compute_metrics(df)
        metrics['query'] = file
        all_results.append(metrics)
    except Exception as e:
        print(f"Ошибка при чтении {file}: {e}")

# Формируем итоговую таблицу
df_results = pd.DataFrame(all_results)
# Переупорядочим колонки согласно заданию
columns_order = ['query', 'total_relevant', 'total_predicted', 'reciprocal_rank',
                 'precision@10', 'recall@10', 'f1_score@10', 'average_precision@10']
df_results = df_results[columns_order]

# Выводим таблицу
print("\nРезультаты:")
print(df_results.to_string(index=False))
# Сохраняем в CSV-файл (опционально)
df_results.to_csv('metrics_summary.csv', index=False)
print("\nТаблица сохранена в metrics_summary.csv")