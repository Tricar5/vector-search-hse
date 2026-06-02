import os

import numpy as np
import pandas as pd


def compute_metrics(df: pd.DataFrame, k: int = 10) -> dict:
    """Принимает DataFrame с колонками 'idx' и 'rel', отсортированный по idx."""
    df = df.sort_values('idx').reset_index(drop=True)
    rel = df['rel'].values
    total_relevant = int(rel.sum())

    first_rel = np.where(rel == 1)[0]
    reciprocal_rank = 1.0 / (first_rel[0] + 1) if len(first_rel) > 0 else 0.0

    top_k = rel[:k]
    p_at_k = top_k.sum() / k
    r_at_k = top_k.sum() / total_relevant if total_relevant > 0 else 0.0
    f1_at_k = (2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0

    rel_cum, precisions = 0, []
    for i in range(min(k, len(rel))):
        if rel[i] == 1:
            rel_cum += 1
            precisions.append(rel_cum / (i + 1))
    r = min(k, total_relevant)
    ap_at_k = sum(precisions) / r if r > 0 else 0.0

    return {
        'total_relevant': total_relevant,
        'total_predicted': len(df),
        'accuracy': rel[:25].sum() / 25.0,
        'reciprocal_rank': reciprocal_rank,
        f'precision@{k}': p_at_k,
        f'recall@{k}': r_at_k,
        f'f1_score@{k}': f1_at_k,
        f'average_precision@{k}': ap_at_k,
    }


def evaluate_directory(directory: str = '.', output: str = 'metrics_summary.csv') -> pd.DataFrame:
    files = [f for f in os.listdir(directory) if not f.endswith('.py') and os.path.isfile(os.path.join(directory, f))]
    results = []
    for file in files:
        path = os.path.join(directory, file)
        print(f'Обработка {file}...')
        try:
            df = pd.read_csv(path, header=0, usecols=[0, -1], names=['idx', 'rel'])
            metrics = compute_metrics(df)
            metrics['query'] = file
            results.append(metrics)
        except Exception as e:
            print(f'Ошибка при чтении {file}: {e}')

    columns = ['query', 'total_relevant', 'total_predicted', 'accuracy', 'reciprocal_rank',
               'precision@10', 'recall@10', 'f1_score@10', 'average_precision@10']
    df_results = pd.DataFrame(results)[columns]
    df_results.to_csv(output, index=False)
    print(f'\nРезультаты:\n{df_results.to_string(index=False)}')
    print(f'\nТаблица сохранена в {output}')
    return df_results


if __name__ == '__main__':
    evaluate_directory()
