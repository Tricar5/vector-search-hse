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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from vs.reranker._env import load_mlflow_env

_CONF_PATH = str(Path(__file__).parents[2] / 'conf' / 'reranker')


def _find_best_threshold(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    proba = model.predict_proba(X)[:, 1]
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y, proba >= thr, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, float(round(thr, 2))
    return best_thr


def _prepare_data(cfg: DictConfig, seed: int) -> tuple:
    df = pd.read_csv(cfg.data.path)
    features = list(cfg.data.features)
    X, y = df[features], df[cfg.data.target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=cfg.data.test_size + cfg.data.val_size,
        random_state=seed, stratify=y,
    )
    rel_val = cfg.data.val_size / (cfg.data.test_size + cfg.data.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - rel_val,
        random_state=seed, stratify=y_temp,
    )
    return features, X_train, X_val, X_test, y_train, y_val, y_test


def _build_model(
    cfg: DictConfig,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    scale_pos_weight: float,
    seed: int,
) -> lgb.LGBMClassifier:
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
        eval_set=[(X_train, y_train), (X_val, y_val)],
        callbacks=[
            lgb.early_stopping(cfg.training.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model, params


def _log_split_metrics(
    model: lgb.LGBMClassifier,
    splits: list[tuple],
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


def _set_prd_tag(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    val_auc: float,
    test_auc: float,
    overfitting: float,
) -> None:
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    for old in client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.deployment = 'PRD'",
    ):
        client.delete_tag(old.info.run_id, 'deployment')
    client.set_tag(run_id, 'deployment', 'PRD')
    client.set_tag(
        run_id,
        'prd_reason',
        f'val_auc={val_auc:.4f} test_auc={test_auc:.4f} overfit={overfitting:.4f}',
    )


def _log_run(
    cfg: DictConfig,
    model: lgb.LGBMClassifier,
    params: dict,
    features: list[str],
    splits: tuple,
    threshold: float,
    scale_pos_weight: float,
    seed: int,
    tracking_uri: str,
) -> str:
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    test_proba = model.predict_proba(X_test)[:, 1]
    test_preds = (test_proba >= threshold).astype(int)
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, test_proba)
    overfitting = train_auc - val_auc

    with mlflow.start_run(run_name='reranker_lgbm') as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            'model_type': 'LGBMClassifier',
            'dataset': cfg.data.path,
            'features': ','.join(features),
            'stage': 'final',
        })
        mlflow.log_params({
            'seed': seed,
            'test_size': cfg.data.test_size,
            'val_size': cfg.data.val_size,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'class_imbalance_ratio': round(scale_pos_weight, 4),
            'decision_threshold': threshold,
            **params,
        })

        _log_split_metrics(
            model,
            [('train', X_train, y_train), ('val', X_val, y_val), ('test', X_test, y_test)],
            threshold,
        )

        fn = int(((y_test.values == 1) & (test_preds == 0)).sum())
        fp = int(((y_test.values == 0) & (test_preds == 1)).sum())
        mlflow.log_metrics({'test_false_negatives': fn, 'test_false_positives': fp})

        mlflow.lightgbm.log_model(
            model,
            name='model',
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
        )

        sample = X_test.assign(y_true=y_test.values, y_pred=test_preds, y_proba=test_proba)
        sample.to_csv('/tmp/sample_predictions.csv', index=False)
        mlflow.log_artifact('/tmp/sample_predictions.csv', artifact_path='artifacts')

        _set_prd_tag(mlflow.tracking.MlflowClient(), run_id, val_auc, test_auc, overfitting)

        print(f'FN={fn}  FP={fp}')
        print(f'\nRun ID: {run_id}')
        print(f'MLflow UI: {tracking_uri}/#/experiments')
        print('Тег deployment=PRD установлен.')

    return run_id


@hydra.main(config_path=_CONF_PATH, config_name='default', version_base='1.3')
def train(cfg: DictConfig) -> None:
    tracking_uri = load_mlflow_env()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    seed = cfg.data.seed
    np.random.seed(seed)

    features, X_train, X_val, X_test, y_train, y_val, y_test = _prepare_data(cfg, seed)
    scale_pos_weight = int((y_train == 0).sum()) / int((y_train == 1).sum())

    model, params = _build_model(cfg, X_train, X_val, y_train, y_val, scale_pos_weight, seed)

    if cfg.training.threshold is None:
        threshold = _find_best_threshold(model, X_val, y_val)
        print(f'Подобранный порог (val F1): {threshold}')
    else:
        threshold = float(cfg.training.threshold)
        print(f'Фиксированный порог: {threshold}')

    _log_run(
        cfg, model, params, features,
        (X_train, X_val, X_test, y_train, y_val, y_test),
        threshold, scale_pos_weight, seed, tracking_uri,
    )


if __name__ == '__main__':
    train()
