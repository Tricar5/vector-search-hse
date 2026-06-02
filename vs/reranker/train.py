"""Reranker training script with MLflow logging.

Usage:
    python -m vs.reranker.train
    python -m vs.reranker.train model.num_leaves=15 training.threshold=0.3
    python -m vs.reranker.train --config-name=default
"""
from pathlib import Path

import hydra
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from omegaconf import DictConfig
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from vs.reranker._env import load_mlflow_env

_CONF_PATH = str(Path(__file__).parents[2] / 'conf' / 'reranker')


def _find_best_threshold(model: lgb.LGBMClassifier, X: pd.DataFrame,
                         y: pd.Series) -> float:
    proba = model.predict_proba(X)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (proba >= thr)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(round(thr, 2))
    return best_thr


def _log_split_metrics(
    run: mlflow.ActiveRun,
    model: lgb.LGBMClassifier,
    splits: list,
    threshold: float,
) -> None:
    for name, X, y in splits:
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= threshold).astype(int)
        mlflow.log_metric(f'{name}_auc', roc_auc_score(y, proba))
        mlflow.log_metric(f'{name}_f1', f1_score(y, preds, zero_division=0))
        mlflow.log_metric(f'{name}_accuracy', accuracy_score(y, preds))
        mlflow.log_metric(f'{name}_precision', precision_score(y, preds, zero_division=0))
        mlflow.log_metric(f'{name}_recall', recall_score(y, preds, zero_division=0))


def _set_prd_tag(client: mlflow.tracking.MlflowClient, run_id: str, val_auc: float,
                 test_auc: float, overfitting: float) -> None:
    experiment = client.get_experiment_by_name(mlflow.get_experiment(
        mlflow.get_run(run_id).info.experiment_id
    ).name)
    all_prd = client.search_runs(
        experiment_ids=[mlflow.get_run(run_id).info.experiment_id],
        filter_string="tags.deployment = 'PRD'",
    )
    for old in all_prd:
        client.delete_tag(old.info.run_id, 'deployment')
    client.set_tag(run_id, 'deployment', 'PRD')
    client.set_tag(
        run_id,
        'prd_reason',
        f'val_auc={val_auc:.4f} test_auc={test_auc:.4f} overfit={overfitting:.4f}',
    )


@hydra.main(config_path=_CONF_PATH, config_name='default', version_base='1.3')
def train(cfg: DictConfig) -> None:
    tracking_uri = load_mlflow_env()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    seed = cfg.data.seed
    np.random.seed(seed)

    df = pd.read_csv(cfg.data.path)
    features = list(cfg.data.features)
    X, y = df[features], df[cfg.data.target]

    neg, pos = int((y == 0).sum()), int((y == 1).sum())
    scale_pos_weight = neg / pos

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=cfg.data.test_size + cfg.data.val_size,
        random_state=seed, stratify=y,
    )
    rel_val = cfg.data.val_size / (cfg.data.test_size + cfg.data.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - rel_val, random_state=seed, stratify=y_temp,
    )
    params = {
        'objective': cfg.model.objective,
        'metric': cfg.model.metric,
        'boosting_type': cfg.model.boosting_type,
        'num_leaves': cfg.model.num_leaves,
        'min_child_samples': cfg.model.min_child_samples,
        'subsample': cfg.model.subsample,
        'colsample_bytree': cfg.model.colsample_bytree,
        'learning_rate': cfg.model.learning_rate,
        'n_estimators': cfg.model.n_estimators,
        'scale_pos_weight': scale_pos_weight,
        'random_state': seed,
        'verbose': cfg.model.verbose,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],
        callbacks=[
            lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )

    if cfg.training.threshold is None:
        threshold = _find_best_threshold(model, X_val, y_val)
        print(f'Подобранный порог (val F1): {threshold}')
    else:
        threshold = float(cfg.training.threshold)
        print(f'Фиксированный порог: {threshold}')

    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)

    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, test_proba)
    test_f1 = f1_score(y_test, test_preds, zero_division=0)
    overfitting = train_auc - val_auc
    with mlflow.start_run(run_name='reranker_lgbm') as run:
        run_id = run.info.run_id

        mlflow.set_tag('model_type', 'LGBMClassifier')
        mlflow.set_tag('dataset', cfg.data.path)
        mlflow.set_tag('features', ','.join(features))
        mlflow.set_tag('stage', 'final')

        mlflow.log_param('seed', seed)
        mlflow.log_param('test_size', cfg.data.test_size)
        mlflow.log_param('val_size', cfg.data.val_size)
        mlflow.log_param('n_train', len(X_train))
        mlflow.log_param('n_val', len(X_val))
        mlflow.log_param('n_test', len(X_test))
        mlflow.log_param('class_imbalance_ratio', round(scale_pos_weight, 4))
        mlflow.log_param('decision_threshold', threshold)
        for k, v in params.items():
            mlflow.log_param(k, v)

        _log_split_metrics(
            run, model,
            [('train', X_train, y_train), ('val', X_val, y_val),
             ('test', X_test, y_test)],
            threshold,
        )

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.lightgbm.log_model(
            model,
            name='model',
            signature=signature,
            input_example=X_train.head(3),
        )

        sample = X_test.copy()
        sample['y_true'] = y_test.values
        sample['y_pred'] = test_preds
        sample['y_proba'] = test_proba
        sample.to_csv('/tmp/sample_predictions.csv', index=False)
        mlflow.log_artifact('/tmp/sample_predictions.csv', artifact_path='artifacts')

        fn = int(((y_test.values == 1) & (test_preds == 0)).sum())
        fp = int(((y_test.values == 0) & (test_preds == 1)).sum())
        mlflow.log_metric('test_false_negatives', fn)
        mlflow.log_metric('test_false_positives', fp)
        print(f'FN={fn}  FP={fp}')

        client = mlflow.tracking.MlflowClient()
        _set_prd_tag(client, run_id, val_auc, test_auc, overfitting)

        print(f'\nRun ID: {run_id}')
        print(f'MLflow UI: {tracking_uri}/#/experiments')
        print(f'Тег deployment=PRD установлен.')


if __name__ == '__main__':
    train()
