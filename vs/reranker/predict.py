"""Reranker inference script — loads PRD model from MLflow and runs predictions.

Usage:
    python -m vs.reranker.predict --input data/sample.csv
    python -m vs.reranker.predict --input data/sample.csv --output results.csv
    python -m vs.reranker.predict  # uses synthetic demo data
"""
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import typer
from sklearn.metrics import roc_auc_score

from vs.reranker._env import load_mlflow_env


FEATURES = [
    'max',
    'mean',
    'std',
    'perc_90',
    'num_passed',
    'range',
]

app = typer.Typer(help='Запуск инференса PRD-реранкера из MLflow')


def _load_prd_model(
    tracking_uri: str,
    experiment_name: str,
) -> tuple:
    """Возвращает (loaded_model, threshold, run_id)."""
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        typer.echo(f'Эксперимент "{experiment_name}" не найден.', err=True)
        raise typer.Exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.deployment = 'PRD'",
        order_by=['start_time DESC'],
        max_results=1,
    )
    if not runs:
        typer.echo('PRD-run не найден. Сначала выполните обучение: vs reranker-train', err=True)
        raise typer.Exit(1)

    prd_run = runs[0]
    run_id = prd_run.info.run_id
    threshold = float(prd_run.data.params.get('decision_threshold', 0.5))

    typer.echo(f'PRD Run ID:  {run_id}')
    typer.echo(f'Run name:    {prd_run.info.run_name}')
    typer.echo(f'Threshold:   {threshold}')
    typer.echo('Метрики:')
    for k, v in sorted(prd_run.data.metrics.items()):
        if any(k.startswith(p) for p in ('train_', 'val_', 'test_')):
            typer.echo(f'  {k}: {v:.4f}')

    model_uri = f'runs:/{run_id}/model'
    typer.echo(f'\nЗагрузка модели из {model_uri} ...')
    model = mlflow.lightgbm.load_model(model_uri)
    typer.echo('Модель загружена.')
    return model, threshold, run_id


def _make_demo_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [0.92, 0.85, 0.05, 0.90, 5, 0.10],
            [0.45, 0.38, 0.08, 0.43, 1, 0.25],
            [0.78, 0.70, 0.07, 0.76, 3, 0.15],
            [0.30, 0.25, 0.03, 0.29, 0, 0.08],
            [0.95, 0.91, 0.03, 0.94, 6, 0.07],
        ],
        columns=FEATURES,
    )


@app.command()
def run(
    input_path: str = typer.Option(
        None,
        '--input', '-i',
        help='CSV с признаками (колонки: max,mean,std,perc_90,num_passed,range). '
             'Если не задан — используются синтетические демо-данные.',
    ),
    output_path: str = typer.Option(
        None,
        '--output', '-o',
        help='Куда сохранить результаты (CSV). По умолчанию — stdout.'
    ),
    experiment: str = typer.Option(
        'reranker-lgbm',
        '--experiment',
        help='Название эксперимента в MLflow'
    ),
) -> None:
    tracking_uri = load_mlflow_env()
    model, threshold, run_id = _load_prd_model(tracking_uri, experiment)

    df = pd.read_csv(input_path)
    missing = set(FEATURES) - set(df.columns)
    if missing:
        typer.echo(f'Отсутствуют признаки: {missing}', err=True)
        raise typer.Exit(1)
    X = df[FEATURES]

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    results = df.copy()
    results['pred_proba'] = np.round(proba, 4)
    results['pred_label'] = preds
    results['verdict'] = np.where(preds == 1, 'РЕЛЕВАНТЕН', 'нерелевантен')

    typer.echo(f'Порог: {threshold}')
    typer.echo(f'Предсказаний: {len(results)}  |  релевантных: {int(preds.sum())}')
    typer.echo('\n' + results[['pred_proba', 'pred_label', 'verdict']].head(10).to_string())

    if 'rel' in results.columns:
        auc = roc_auc_score(results['rel'], proba)
        typer.echo(f'\nAUC на переданных данных: {auc:.4f}')

    if output_path:
        results.to_csv(output_path, index=False)
        typer.echo(f'\nРезультаты сохранены в {output_path}')


if __name__ == '__main__':
    app()
