# ML Reranker — трекинг экспериментов с MLflow

Реранкер — бинарный классификатор LightGBM, обученный поверх результатов векторного поиска. Принимает 6 агрегированных признаков сходства и предсказывает релевантность документа.

---

## Структура компонентов

```
vs/reranker/
├── train.py            # обучение + логирование в MLflow
├── predict.py          # инференс PRD-модели из MLflow (typer CLI)
├── compute_metrics.py  # метрики ранжирования по CSV-предсказаниям
├── concatenate_tables.py # сборка единого датасета из CSV-запросов
└── _env.py             # загрузка .env и URI MLflow-сервера

conf/reranker/
└── default.yaml        # гиперпараметры, пути данных, настройки обучения

research/mlflow/
├── train_reranker_mlflow.ipynb  # полный EDA + обучение + отчёт
└── load_prd_model.ipynb         # загрузка PRD-модели и инференс
```

---

## Признаки модели

| Признак | Описание |
|---|---|
| `max` | максимальное косинусное сходство в топ-N |
| `mean` | среднее сходство |
| `std` | стандартное отклонение сходства |
| `perc_90` | 90-й перцентиль сходства |
| `num_passed` | число кандидатов выше порога |
| `range` | разброс (max − min) |

Целевая переменная — `rel` (0/1). Дисбаланс классов ~4.8:1 (746 neg / 154 pos), компенсируется `scale_pos_weight`.

---

## Конфигурация (`conf/reranker/default.yaml`)

```yaml
data:
  path: model/merged_for_boosting.csv
  features: [max, mean, std, perc_90, num_passed, range]
  target: rel
  test_size: 0.15
  val_size: 0.15
  seed: 42

model:
  objective: binary
  metric: auc
  boosting_type: gbdt
  num_leaves: 31
  min_child_samples: 10
  learning_rate: 0.03
  n_estimators: 500

training:
  early_stopping_rounds: 20
  threshold: null   # null = автоподбор по F1 на val

mlflow:
  experiment: reranker-lgbm
```

Параметры переопределяются через аргументы командной строки (Hydra).

---

## Подготовка данных

Перед обучением нужно собрать отдельные CSV-файлы запросов в единый датасет.

```bash
# Запустить из папки с CSV-файлами запросов
python -m vs.reranker.concatenate_tables
# → создаёт merged_for_boosting.csv
```

Каждый CSV ожидает колонки: `idx, max, mean, std, perc_90, num_passed, density, range, rel`.

---

## Обучение

```bash
# Обучение с параметрами из default.yaml
python -m vs.reranker.train

# Переопределение гиперпараметров
python -m vs.reranker.train model.num_leaves=15 training.threshold=0.3

# Через общий CLI пакета
python -m vs.cli reranker-train
python -m vs.cli reranker-train model.learning_rate=0.05 training.early_stopping_rounds=30
```

По окончании обучения скрипт автоматически:
1. Подбирает порог по F1 на val-выборке (если `threshold: null`)
2. Логирует метрики train / val / test (AUC, F1, Precision, Recall, Accuracy)
3. Сохраняет модель в MLflow с сигнатурой и примером входных данных
4. Сохраняет артефакт `sample_predictions.csv` с предсказаниями на тесте
5. Снимает тег `deployment=PRD` со всех предыдущих ранов и ставит на текущий

---

## Инференс

```bash
# Через модуль
python -m vs.reranker.predict --input data/candidates.csv --output results.csv

# Через общий CLI
python -m vs.cli reranker-predict --input data/candidates.csv

# Сменить эксперимент MLflow
python -m vs.cli reranker-predict --input data/candidates.csv --experiment my-experiment
```

Скрипт находит последний PRD-ран в MLflow, вытаскивает порог из параметров (`decision_threshold`) и добавляет к результатам колонки `pred_proba`, `pred_label`, `verdict`. Если во входном CSV есть колонка `rel` — дополнительно выводит AUC.

---

## Метрики ранжирования

```bash
# Запустить из папки с CSV-предсказаниями
python -m vs.reranker.compute_metrics
# → выводит таблицу и сохраняет metrics_summary.csv
```

Вычисляемые метрики: `accuracy@25`, `reciprocal_rank`, `precision@10`, `recall@10`, `f1_score@10`, `average_precision@10`.

---

## Ноутбуки

### `research/mlflow/train_reranker_mlflow.ipynb`

Полный цикл исследования:
- EDA датасета (900 строк, 17% позитивного класса)
- Сравнение конфигураций LightGBM: conservative / balanced / dart
- Анализ ошибок: 8 FN, 11 FP
- Тест робастности (шум Гаусса)
- Артефакты: confusion matrix, ROC, feature importance, error analysis CSV

### `research/mlflow/load_prd_model.ipynb`

Демонстрация загрузки PRD-модели:
- Поиск рана по тегу `deployment=PRD`
- Загрузка модели из S3 через MLflow
- Инференс на 10 случайных сэмплах с выводом метрик

---

## Переменные окружения

Создайте `.env` в корне проекта (перекрывает `.env.default`):

```dotenv
MLFLOW_URI=http://localhost:5050
```

MLflow-клиент автоматически читает `MLFLOW_TRACKING_URI`, который выставляется при импорте `load_mlflow_env()`.
