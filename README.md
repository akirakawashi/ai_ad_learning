# AI Ad Learning

Инструменты для подготовки данных, проверки разметки и обучения моделей AI Ad.
В репозитории есть две рабочие группы команд: `detect` для детектора рекламных
щитов и `cls` для классификации брендов.

Соседние репозитории:

- `../ai_ad_ml` — рабочий пайплайн и боевые веса;
- `../ai_ad_backend` — API и бизнес-логика продукта.

## Структура

```text
src/adlearn/
├── paths.py              общие пути к данным и моделям
├── cli.py                точка входа adlearn <задача> <команда>
├── core/                 изображения, группировка и контактные листы
├── detection/            разметка, проверка, наборы и обучение детектора
└── classification/       признаки, классификатор и эксперименты с VLM

data/
├── raw/                  исходные фотографии для детектора
├── dashcam/              кадры с регистратора
├── negatives/            внешние отрицательные примеры
├── detection/
│   ├── prelabel/         псевдоразметка и отчёт детектора
│   ├── review/           проверка стоковых кадров
│   ├── review_dashcam/   проверка кадров регистратора
│   ├── negatives/        подготовленные кадры без рекламы
│   ├── dataset/          готовый набор YOLO
│   ├── export/           пакеты для CVAT
│   └── preview/          контактные листы
├── classification/
│   ├── raw/              кадры по брендам
│   ├── features/         кэш признаков
│   ├── model.joblib      обученная голова классификатора
│   └── runs/             результаты экспериментов
└── runs/                 прогоны обучения Ultralytics

models/
├── detection/best.pt     локальная копия боевых весов
├── pretrained/           исходные веса для обучения
└── vlm/                  GGUF-модель и mmproj для llama-server
```

Каталоги `data/` и `models/` не хранятся в Git. В репозитории остаются только
`.gitkeep`, поэтому данные и модели нужно подготовить локально.

## Настройка

Нужны Python 3.12 и `uv`:

```bash
uv sync
cp ../ai_ad_ml/models/detection/best_pt_v4.pt models/detection/best.pt
```

Пайплайн `ai_ad_ml` использует `best_pt_v4.pt`. В этом репозитории команды
детектора ожидают те же веса под локальным именем `models/detection/best.pt`.

Для обучения с нуля положите исходные веса в
`models/pretrained/yolo11m.pt`. Для локальной VLM нужны файлы:

```text
models/vlm/Qwen3VL-8B-Instruct-Q4_K_M.gguf
models/vlm/mmproj-Qwen3VL-8B-Instruct-F16.gguf
```

Команды, которые запускают YOLO, по умолчанию выбирают GPU `0`, а извлечение
признаков классификатора — `cuda`. Автоматического перехода на процессор нет.
На машине без CUDA явно передавайте `--device cpu`:

```bash
uv run adlearn detect prelabel --device cpu
uv run adlearn detect review --stage scan --device cpu
uv run adlearn detect train --device cpu
uv run adlearn detect eval --weights models/detection/best.pt --device cpu
uv run adlearn cls features --device cpu
```

Обработка на процессоре заметно медленнее.

## Команды детектора

Полная справка: `uv run adlearn detect --help` и
`uv run adlearn detect <команда> --help`.

| Команда | Что делает |
|---|---|
| `prelabel` | Прогоняет фотографии через детектор, пишет YOLO-разметку, CSV-отчёт и архив для CVAT |
| `bundle` | Собирает спорные кадры из псевдоразметки в пакеты для CVAT |
| `build` | Объединяет ZIP-выгрузку CVAT из `--export` с уверенной псевдоразметкой |
| `build3` | Собирает набор из архива ручной разметки, стока, регистратора и негативов |
| `handmade` | Собирает набор только из распакованных изображений и меток CVAT |
| `check` | Проверяет изображения, метки и состав частей набора |
| `preview` | Рисует рамки и собирает контактные листы |
| `review` | Проводит рамки через VLM-судью и готовит спорные кадры для CVAT |
| `negatives` | Создаёт пустую разметку для фотографий без рекламы и добавляет преобразованные копии |
| `train` | Обучает детектор; по умолчанию начинает с `models/pretrained/yolo11m.pt` |
| `eval` | Сравнивает один или несколько файлов весов на `test_dashcam`, `test` или `val` |

Обычный цикл подготовки и обучения:

```bash
uv run adlearn detect prelabel
uv run adlearn detect bundle
# Исправить разметку в CVAT.
uv run adlearn detect build --export /путь/к/выгрузке-cvat.zip
uv run adlearn detect check
uv run adlearn detect preview --split train
uv run adlearn detect eval --weights models/detection/best.pt
uv run adlearn detect train
uv run adlearn detect eval --weights \
    models/detection/best.pt \
    data/runs/ad_object_v2/weights/best.pt
```

`build3` нужен для набора из четырёх уже подготовленных источников. Значение
`--archive` по умолчанию содержит путь с машины автора, поэтому на другом
компьютере его передают явно:

```bash
uv run adlearn detect build3 --archive /путь/к/архиву/yolo
```

Подробности о наборах и разметке: [docs/detection.md](docs/detection.md) и
[docs/cvat.md](docs/cvat.md).

### Проверка псевдоразметки

`review` выполняется по стадиям:

```bash
uv run adlearn detect review --stage scan
uv run adlearn detect review --stage judge
uv run adlearn detect review --stage sheets
# Просмотреть листы и заполнить review/verdicts.csv.
uv run adlearn detect review --stage apply
uv run adlearn detect review --stage cvat
uv run adlearn detect review --stage import --export /путь/к/выгрузке-cvat.zip
```

| Стадия | Результат |
|---|---|
| `scan` | Рамки, уверенности и JPEG-вырезки |
| `judge` | Ответ VLM для каждой вырезки |
| `sheets` | Контактные листы по категориям ответа |
| `apply` | Чистая разметка после ручных вердиктов |
| `cvat` | Архив спорных кадров для CVAT |
| `import` | Исправленные метки из выгрузки CVAT |

Значение `--stage all` запускает только `scan`, `judge` и `sheets`.
Стадии `apply`, `cvat` и `import` требуют решения человека и запускаются
отдельно.

Для `judge` поднимите локальный сервер:

```bash
~/llama.cpp/build/bin/llama-server \
    -m models/vlm/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
    --mmproj models/vlm/mmproj-Qwen3VL-8B-Instruct-F16.gguf \
    -ngl 99 -c 8192 --host 127.0.0.1 --port 8080
```

Другой адрес задаётся через `--vlm-url`. Для общего OpenAI-совместимого сервера
доступны `--model` и `--api-key`; без этих ключей команда использует
`PIPELINE_VLM_MODEL` и `PIPELINE_VLM_API_KEY` из окружения. Локальный
`llama-server` имя модели и ключ обычно не требует.

Отрицательные примеры готовятся отдельно:

```bash
uv run adlearn detect negatives --source /путь/к/фотографиям
```

Результат появится в `data/detection/negatives/`.

## Команды классификации

Классификация реализована как рабочий набор из восьми команд:

| Команда | Что делает |
|---|---|
| `features` | Рассчитывает визуальные, цветовые и диагностические признаки |
| `train` | Обучает и сохраняет голову классификатора |
| `ablate` | Сравнивает группы признаков и контрольные варианты |
| `predict` | Определяет бренд для папки изображений |
| `vlm` | Проверяет изображения через llama-server или общий OpenAI-совместимый сервер |
| `compare` | Сравнивает результаты двух прогонов VLM |
| `probe` | Собирает проверочную выборку из брендов, двойников и уличных кадров |
| `blind` | Заменяет имена файлов номерами и сохраняет ключ отдельно |

Полная справка: `uv run adlearn cls --help` и
`uv run adlearn cls <команда> --help`.

Пример проверки VLM на размеченной выборке:

```bash
uv run adlearn cls vlm \
    --source data/classification/probe/r200 \
    --labels data/classification/probe/r200/labels.csv \
    --output data/classification/vlm_runs/r1.csv
```

Для общего сервера доступны `--model` и `--api-key`. Локальный
`llama-server` имя модели игнорирует и обычно не требует ключ.

Классификатор с обучаемой головой различает `beeline`, `megafon`, `tele2`
и `other`. VLM-проверка охватывает более широкий список операторов. Рабочая
копия промпта для пайплайна находится в
`../ai_ad_ml/ml/pipeline/scripts/vlm.py`; изменения между репозиториями
переносятся вручную.

## Проверки

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```
