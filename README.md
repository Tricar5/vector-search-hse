# Контекстный векторный поиск по мультимедийной базе знаний

## Состав команды

1. **Казачинский Андрей** (@tricar5)
2. **Харитонов Борис** (@Floating_N)

**Куратор - Кирилл Малюшитский (@malyushitsky)**

# План работ

| Этап                                                                                                      | Описание                                                                                                                                        | Срок              |
|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Сбор данных и EDA                                                                                         | Сбор данных и их анализ (различные статистики по данным - длительность видео, распределение на тематики)                                        | ноябрь 2025       |
| Бейзлайн                                                                                                  | Реализовать поиск по паре текстовый запрос - top-k релевантных видео (используем предобученный clip на базе ViT)                                | декабрь 2026      |
| Улучшение бейзлайна                                                                                       | Дообучение clip на наших данных/добавление доп модели для классификации тематики видео для сужения пространства поиска релевантных поиску видео | январь 2026       |
| Сервис                                                                                                    | Бэкенд Сервис + пайплайн + пользовательский интерфейс (Тг бот, Gradio, Streamlit)                                                               | январь 2026       |
| Эксперименты с DL                                                                                         | Добавление доп текстовых эмбеддингов для поиска (аудио переводим в текст с помощью whisper)                                                     | февраль-март 2026 |
| Доработка задачи: улучшение сервисной части по обратной связи от команды курса по промышленной разработке | Реализация улучшений на основе фидбека от команды промышленной разработки                                                                       | февраль-март 2026 |

## Описание данных

Корпус коротких видео для поиска. Сбор самостоятельный

## Описание моделей

Базовая модель для построения эмбеддингов - CLIP/ruCLIP

## Архитектура

## Работа с проектом

Установка и работа с пакетами осуществляется через [poetry]()

```shell
make dev.install
```

После установки

```shell
poetry env activate
```

## Инфраструктура

### PG для логов

```shell
make dc.up
```

### Копирование .env с дефолтными значения

```shell
make env
```

## Запуск приложения

Для работы приложения на локале требуется PG и .env. Более подробно [тут](docs/service.md)

### Запуск приложения fastapi

```shell
make run
```

Для работы приложения требуются файлы локальных индексов.

### Миграции на базу данных

1. Накат миграций (Пример: всех)

```shell
make db.migrate head
```

2. Откат миграций (по одной)

```shell
make db.rollback
```

3. Создание новой миграции после обновления моделей

```shell
make db.revision 'new_migration'
```

### Метрики приложения

Prometheus-метрики доступны на `/metrics`. Сервис пишет:
- RPS и статус-коды по каждой ручке
- Latency (p50 / p95 / p99)
- Время работы поискового движка по типу запроса (text / image / audio)
- Количество результатов и длину текстовых запросов

Grafana-дашборд поднимается вместе с Prometheus:

```shell
docker compose up prometheus grafana
```

Grafana: [http://localhost:3000](http://localhost:3000) (admin / admin), Prometheus: [http://localhost:9090](http://localhost:9090)

## CLI

Все команды доступны через `python -m vs.cli`. Список:

```shell
python -m vs.cli --help
```

---

### download-deps — скачивание зависимостей

Модели и словари хранятся в папке `model/`. Источники:
- **CLIP ViT-B/32** — OpenAI CDN (с SHA256-проверкой)
- **AudioCLIP-Full-Training.pt** — [GitHub Releases AndreyGuzhov/AudioCLIP v0.1](https://github.com/AndreyGuzhov/AudioCLIP/releases/tag/v0.1)
- **bpe_simple_vocab_16e6.txt.gz** — тот же GitHub Releases

**Проверить наличие всех файлов:**

```shell
python -m vs.cli download-deps check
# [OK   ]  CLIP ViT-B/32              model/ViT-B-32.pt  (354 MB)
# [OK   ]  BPE vocab                  model/bpe_simple_vocab_16e6.txt.gz  (1 MB)
# [OK   ]  AudioCLIP weights          model/AudioCLIP-Full-Training.pt  (537 MB)
```

**Скачать всё одной командой (рекомендуется):**

```shell
python -m vs.cli download-deps all

# кастомная папка
python -m vs.cli download-deps all --model-dir /data/models
```

**Скачать по отдельности:**

```shell
python -m vs.cli download-deps clip       # CLIP ViT-B/32
python -m vs.cli download-deps vocab      # BPE-словарь токенизатора
python -m vs.cli download-deps audioclip  # AudioCLIP (только при model_type: audioclip)

# AudioCLIP с прямой ссылкой вместо GitHub Releases
python -m vs.cli download-deps audioclip --url https://example.com/AudioCLIP-Full-Training.pt
```

---

### local_index_pipe — построение векторного индекса

Индексирует видеофайлы: извлекает кадры, строит CLIP-эмбеддинги, сохраняет индекс.

```shell
python -m vs.cli local_index_pipe \
  --video-dir data/video \
  --frame-rate 1.0 \
  --batch-size 128 \
  --index-path model/index.pkl \
  --metadata-path model/metadata.pkl
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--video-dir` / `-v` | `data/video` | Папка с видеофайлами |
| `--extensions` | `mp4,mov` | Расширения через запятую |
| `--frame-rate` | `1.0` | Кадров в секунду для индексации |
| `--batch-size` | `128` | Размер батча при энкодинге |
| `--index-path` | `model/index.pkl` | Куда сохранить индекс |
| `--metadata-path` | `model/metadata.pkl` | Куда сохранить метаданные |

---

### local_thumbnail — генерация превью

Извлекает кадры-превью для каждого видео. Запускать после `local_index_pipe`.

```shell
python -m vs.cli local_thumbnail \
  --metadata-path model/metadata.pkl \
  --thumbnail-path model/thumbnails.pkl
```

---

### reranker-train — обучение реранкера

Обучает LightGBM-реранкер на собранной разметке, логирует эксперимент в MLflow.
Параметры передаются через [Hydra](https://hydra.cc/) в формате `key=value`.

```shell
python -m vs.cli reranker-train data_path=result.csv model_output=model/model.pkl
```

---

### reranker-predict — инференс реранкера

Прогоняет модель реранкера из MLflow на новых данных.

```shell
python -m vs.cli reranker-predict [HYDRA OPTIONS]
```

---

### Типовой сценарий: первичная настройка с нуля

```shell
# 1. Скачать все веса моделей
python -m vs.cli download-deps all

# 2. Построить индекс по видео
python -m vs.cli local_index_pipe --video-dir data/video

# 3. Сгенерировать превью
python -m vs.cli local_thumbnail

# 4. Запустить инфраструктуру и сервис
make dc.up
make run
```
