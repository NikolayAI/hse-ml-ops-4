# hse-ml-ops-4

MLOps-домашка №4: обучение ResNet18 для классификации AI vs Human Generated Images
с трекингом в TensorBoard, хранением моделей в S3 (MinIO) и DVC-пайплайном для
дообучения.

## Структура

```
hse-ml-ops-4/
├── src/                     # код пайплайна
│   ├── dataset.py           # ImageDataset (CSV + папка с картинками)
│   ├── training.py          # общие функции: transforms, train/eval, метрики
│   ├── train.py             # стадия train_base
│   ├── finetune.py          # стадия finetune (скачивает базовую модель из S3)
│   └── s3_utils.py          # boto3-клиент MinIO
├── params.yaml              # все параметры пайплайна (читает DVC и скрипты)
├── dvc.yaml                 # описание стадий: train_base → finetune
├── dvc.lock                 # зафиксированные хеши после последнего dvc repro
├── docker-compose.yml       # MinIO (S3)
├── requirements.txt         # python-зависимости
├── metrics/                 # JSON с финальными метриками каждой стадии
├── my_logs/                 # TensorBoard event-файлы
├── checkpoints/             # сохранённые веса моделей
└── train_model.ipynb        # итоговый ноутбук с выводами и графиками
```

## Требования

- Python 3.10+ (проверено на 3.13)
- Docker + docker compose
- macOS / Linux. На Apple Silicon обучение использует MPS, иначе CPU/CUDA.
- Датасет `ai-vs-human-generated-dataset-hw/` — должен лежать рядом с этим
  репозиторием (или быть симлинком сюда). Пути прописаны в `params.yaml`.

## Воспроизведение результатов

### 1. Поставить зависимости

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Поднять MinIO (S3)

```bash
docker compose up -d
# UI:    http://localhost:9001  (логин/пароль: minioadmin / minioadmin)
# API:   http://localhost:9000
```

Бакет `models` создаётся автоматически при первой выкладке.

### 3. Положить датасет рядом

`params.yaml` ожидает датасет по относительным путям
`ai-vs-human-generated-dataset-hw/Train_1/...`, `Test_1/...`, `Train_2/...`,
`Test_2/...`. Скопируйте папку или сделайте симлинк:

```bash
ln -s /путь/к/ai-vs-human-generated-dataset-hw .
```

### 4. Запустить пайплайн

```bash
dvc repro          # выполнит train_base → finetune
dvc dag            # покажет граф зависимостей
dvc metrics show   # таблица метрик из metrics/*.json
```

`dvc repro` сам:
1. обучит базовую модель на `Train_1`, оценит на `Test_1`,
2. сохранит `checkpoints/resnet18_base.pt` и выложит его в `s3://models/`,
3. скачает базовую модель из S3, дообучит на `Train_2`, оценит на `Test_2`,
4. выложит `resnet18_finetuned.pt` в S3,
5. запишет TensorBoard-логи в `my_logs/{train_base,finetune}/` и финальные
   метрики в `metrics/{train_base,finetune}.json`.

### 5. Посмотреть метрики в TensorBoard

```bash
tensorboard --logdir=my_logs
# открыть http://localhost:6006
```

В TensorBoard логируются: `train/{loss,accuracy,f1,precision,recall}`, `lr`,
а также `test/*` по той же схеме. Гиперпараметры пишутся text-тегом `hparams`.

### 6. Посмотреть ноутбук

```bash
jupyter notebook train_model.ipynb
```

# остановить MinIO
docker compose down
```
